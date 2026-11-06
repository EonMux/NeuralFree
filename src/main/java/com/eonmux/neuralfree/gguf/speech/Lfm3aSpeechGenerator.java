package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.function.Consumer;

/**
 * An LFM3-Audio backbone answering with speech through the MaskGIT depthformer and liquid-codec decoder of a
 * {@link Lfm3aGenClip}, the pipeline of llama.cpp's {@code lfm3a_gen_audio_pipeline}
 * (tools/mtmd/mtmd-helper-gen.cpp:1273-1720). The pipeline talks in three modes. In AUDIO mode every step samples
 * one speech-code frame with the depthformer, decodes it with the liquid codec and feeds the frame's feedback
 * embedding back into the backbone. In TEXT mode every step samples an ordinary token. MIXED mode alternates
 * blocks of {@value #N_TEXT_BLOCK} text slots and {@value #N_AUDIO_BLOCK} audio frames until the model emits
 * text_end (which hands the turn to audio until the end-of-audio frame) or im_end (which ends it). A fresh
 * request must open with audio_start, which is the plain text-to-speech path; a continued response opens with the
 * asked mode's switch token, or in TEXT mode with the logit bias that bans both switch tokens
 * (mtmd-helper-gen.cpp:1336-1352).
 *
 * <p>The prompt is rendered by the caller: the speech routes render the model's {@code tokenizer.chat_template.tts}
 * template with the request's {@code language} and {@code voice}, the chat route the model's conversation
 * template plus the asked mode's switch token, and both call
 * {@link #generate(int[], Mode, boolean, SpeechOptions, OmniSink)}. {@link #synthesize(String, SpeechOptions)}
 * renders the speech template itself when the model carries one, and otherwise speaks the bare text with an
 * audio_start token appended (a voice then refuses by name, because nothing would condition on it). Text tokens
 * sample with the request's sampler over the options' {@link SpeechOptions#sampler()} or its seed, top-k and
 * top-p; the speech codes sample with the reference's own sampler over the generator file's model-card defaults
 * (mtmd-helper-gen.cpp:1307-1312, 1600-1628) — the caller's top-k/top-p do not reach it. The waveform streams
 * out as the frames decode, exactly the samples the one-shot form concatenates. Bound to one backbone and one
 * opened generator file; drives the backbone's forward backend and must run on its engine thread. The generator
 * owns the clip (its {@link #close} closes it). Not thread-safe.
 *
 * <p>The backbone may run over any engine backend. Over a {@link GGUFModelGPU} the MaskGIT depthformer and the
 * liquid-codec decoder also run on that engine's device, through {@link GpuLfm3aDepthformer} and
 * {@link GpuLfm3aCodec} over weights uploaded once by {@link Lfm3aGenClip#bindDevice}; the codebook sampling and
 * the ISTFT stay on the host. Over the CPU engine the same graphs run on the host through
 * {@link Lfm3aDepthformer} and {@link Lfm3aCodecSession}.
 */
public final class Lfm3aSpeechGenerator implements SpeechGenerator {

    /** The generator family name. */
    public static final String NAME = "lfm3a";

    /** The switching pieces the pipeline resolves from the vocabulary (mtmd-helper-gen.cpp:1645-1648). */
    public static final String AUDIO_START_PIECE = "<|audio_start|>";
    public static final String MIXED_START_PIECE = "<|mixed_start|>";
    public static final String TEXT_END_PIECE = "<|text_end|>";
    public static final String IM_END_PIECE = "<|im_end|>";

    /** Interleaved block sizes, fixed by the model card (mtmd-helper-gen.cpp:1682-1683). */
    public static final int N_TEXT_BLOCK = 6;
    public static final int N_AUDIO_BLOCK = 12;

    /** The MaskGIT codebook count and the codebooks revealed per depthformer pass. */
    private static final int N_CODEBOOKS = 12;
    private static final int[] SCHEDULE = {4, 4, 4};

    /** The generation mode a request or the model's last switch token picked. */
    public enum Mode {
        /** Text only: the logit bias bans both switch tokens. */
        TEXT,
        /** Speech until the end-of-audio frame. */
        AUDIO,
        /** Interleaved blocks of text and audio until im_end or the end of audio. */
        MIXED
    }

    /** One turn's outputs: the emitted text, the whole waveform, the prompt it prefilled and the steps it sampled. */
    public static final class OmniTurn {
        private final String text;
        private final AudioClip audio;
        private final int promptTokens;
        private final int steps;

        OmniTurn(String text, AudioClip audio, int promptTokens, int steps) {
            this.text = text;
            this.audio = audio;
            this.promptTokens = promptTokens;
            this.steps = steps;
        }

        /** The text the model spoke around its audio, control pieces excluded. */
        public String text() {
            return text;
        }

        /** The whole waveform of the turn, its audio chunks concatenated. */
        public AudioClip audio() {
            return audio;
        }

        /** The prompt rows the turn prefilled, the switch token of a continued turn included. */
        public int promptTokens() {
            return promptTokens;
        }

        /** The steps the turn sampled: speech frames and text tokens alike (llama.cpp's {@code n_decoded}). */
        public int steps() {
            return steps;
        }
    }

    /** Receives a turn as it happens: text pieces when sampled, audio chunks as their frames decode. */
    public interface OmniSink {
        /** A sampled text piece; control pieces never arrive. */
        void text(String piece);

        /** Consecutive samples of the turn's waveform; concatenating them gives the whole audio. */
        void audio(AudioClip chunk);
    }

    private final EngineModel model;
    private final Lfm3aGenClip clip;
    private final SpeechDeviceWeights device;
    private final GGUFTokenizer tokenizer;
    private final ForwardBackend backend;
    private final float[] outputNorm;
    private final float normEps;
    private final int width;
    private final int nLayer;
    private final int audioStart;
    private final int mixedStart;
    private final int textEnd;
    private final int imEnd;
    private final Capture capture;
    private boolean closed;

    private Lfm3aSpeechGenerator(EngineModel model, Lfm3aGenClip clip, SpeechDeviceWeights device)
            throws java.io.IOException {
        this.model = model;
        this.clip = clip;
        this.device = device;
        this.tokenizer = model.tokenizer();
        this.backend = model.forwardBackend();
        ModelSpec spec = model.spec();
        HeadSpec head = spec.head();
        if (head.outputNorm() == null) {
            throw new GgufFormatException("lfm3a: the backbone has no output norm");
        }
        if (spec.embedding().tokenEmbd() == null) {
            throw new GgufFormatException("lfm3a: the backbone has no token embedding table");
        }
        if (spec.embedding().embdScale() != 1f || spec.embedding().tokenEmbdNorm() != null
                || spec.embedding().inputRmsNorm()) {
            throw new UnsupportedOperationException("lfm3a: the backbone applies an input normalization the"
                    + " feedback embedding rows do not go through (llama.cpp feeds raw rows)");
        }
        this.width = spec.embedding().nEmbd();
        if (width != clip.depthformer().talkerWidth()) {
            throw new IllegalArgumentException("lfm3a: the backbone hidden width " + width + " does not match the"
                    + " generator's projection dim " + clip.depthformer().talkerWidth() + "; the depthformer"
                    + " takes the backbone's final hidden state");
        }
        this.outputNorm = model.cpuWeights().floats(head.outputNorm());
        if (outputNorm.length != width) {
            throw new GgufFormatException("lfm3a: output_norm holds " + outputNorm.length + " values, expected "
                    + width);
        }
        this.normEps = spec.hparams().normEps();
        if (clip.depthformer().nCodebooks() != N_CODEBOOKS) {
            throw new GgufFormatException("lfm3a: the depthformer runs " + clip.depthformer().nCodebooks()
                    + " codebooks, the reference pipeline schedules three passes of four over " + N_CODEBOOKS);
        }
        this.nLayer = spec.nLayer();
        this.audioStart = requireToken(AUDIO_START_PIECE);
        this.mixedStart = requireToken(MIXED_START_PIECE);
        this.textEnd = requireToken(TEXT_END_PIECE);
        this.imEnd = requireToken(IM_END_PIECE);
        this.capture = new Capture(nLayer, width);
    }

    /**
     * Binds a generator to {@code model} and {@code clip}.
     *
     * @throws IllegalArgumentException when the model is not an lfm3a backbone or its hidden width differs from
     *                                  the generator's talker width
     * @throws GgufFormatException      when the backbone or the clip lacks a tensor the pipeline reads
     * @throws java.io.IOException      when the backbone's output norm cannot be read, or the generator file
     *                                  cannot be bound to the backbone's device
     * @throws UnsupportedOperationException when the backbone needs an input normalization the feedback rows do
     *                                       not go through, or its vocabulary lacks a control token (named)
     */
    public static Lfm3aSpeechGenerator open(EngineModel model, Lfm3aGenClip clip) throws java.io.IOException {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(clip, "clip");
        if (model.spec().arch() != Arch.LFM3) {
            throw new IllegalArgumentException("speech generation through the lfm3a pipeline needs the "
                    + Arch.LFM3.ggufName() + " backbone, not " + model.spec().arch().ggufName());
        }
        SpeechDeviceWeights device = model instanceof GGUFModelGPU
                ? clip.bindDevice(((GGUFModelGPU) model).forwardContext())
                : null;
        return new Lfm3aSpeechGenerator(model, clip, device);
    }

    private int requireToken(String piece) {
        int id = tokenizer.getTokenId(piece);
        if (id < 0) {
            throw new UnsupportedOperationException("lfm3a: the backbone vocabulary has no '" + piece
                    + "' token; the pipeline needs it (mtmd-helper-gen.cpp:1645-1647)");
        }
        return id;
    }

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public int sampleRate() {
        return clip.codec().sampleRate();
    }

    @Override
    public AudioClip synthesize(String text, SpeechOptions options) {
        Collecting sink = new Collecting();
        runFresh(text, options, sink);
        return sink.turn(0, 0).audio();
    }

    @Override
    public void synthesize(String text, SpeechOptions options, Consumer<AudioClip> sink) {
        Objects.requireNonNull(sink, "sink");
        runFresh(text, options, audioOnly(sink));
    }

    /**
     * The frames of one speech render, for parity tests: one row of {@value #N_CODEBOOKS} codes per frame. The
     * text renders through the model's speech template, which this path requires.
     *
     * @throws UnsupportedOperationException when the model carries no {@code tokenizer.chat_template.tts}
     */
    int[][] generateCodes(String text, SpeechOptions options) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(options, "options");
        checkOpen();
        if (text.trim().isEmpty()) {
            throw new IllegalArgumentException("the speech text must not be blank");
        }
        List<int[]> frames = new ArrayList<>();
        run(ttsPrompt(text, options), Mode.AUDIO, false, options, null, frames);
        return frames.toArray(new int[0][]);
    }

    /**
     * Runs one turn: the prompt tokens go in, the model answers in its mode until im_end, the end of audio, or
     * the step cap. A fresh request ({@code continuing = false}) must open in AUDIO mode, which is the plain
     * text-to-speech path; a continued response opens with the asked mode's token, or in TEXT mode with the logit
     * bias that bans both switch tokens.
     *
     * @param promptTokens the rendered prompt, ending in the asked mode's switch token when continuing
     * @param asked        the mode to open a continued response in; ignored when fresh (the prompt decides)
     * @param continuing   true when {@code promptTokens} extends an earlier turn on the same backend context
     * @param options      the text-step sampler (or its seed, top-k and top-p) and the step cap; the speech codes
     *                     sample with the file's defaults
     * @param sink         receives the text pieces and audio chunks as they happen, or null for none
     * @return the turn's whole text, audio and prompt length
     * @throws IllegalArgumentException on an empty prompt, or a fresh prompt whose last token is not
     *                                  audio_start (mtmd-helper-gen.cpp:1336)
     */
    public OmniTurn generate(int[] promptTokens, Mode asked, boolean continuing, SpeechOptions options,
                             OmniSink sink) {
        checkOpen();
        Collecting collecting = new Collecting();
        RunStats stats = run(promptTokens, asked, continuing, options,
                sink == null ? collecting : fanOut(sink, collecting), null);
        return collecting.turn(stats.promptTokens, stats.steps);
    }

    /** The fresh text-to-speech path of {@link #synthesize}. */
    private void runFresh(String text, SpeechOptions options, OmniSink sink) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(options, "options");
        checkOpen();
        if (text.trim().isEmpty()) {
            throw new IllegalArgumentException("the speech text must not be blank");
        }
        if (options.speakerReference() != null) {
            throw new UnsupportedOperationException("lfm3a has no speaker encoder: \"speaker_ref\" cannot clone a"
                    + " voice on this generator");
        }
        run(freshPrompt(text, options), Mode.AUDIO, false, options, sink, null);
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("the speech generator is closed");
        }
    }

    /**
     * The speech prompt of {@link #synthesize}: the model's {@code tts} template rendered with the utterance,
     * language, voice and BOS text when it carries one, else the bare text plus an audio_start token. A voice on
     * the raw-text path refuses by name, because nothing would condition on it.
     */
    private int[] freshPrompt(String text, SpeechOptions options) {
        if (tokenizer.ttsTemplate() != null) {
            return ttsPrompt(text, options);
        }
        if (options.hasVoice()) {
            throw new UnsupportedOperationException("lfm3a has no voice presets: \"" + options.voice()
                    + "\" names none (the model conditions on nothing but the prompt; a voice reaches it only"
                    + " through the model's speech chat template, which this checkpoint does not carry)");
        }
        int[] ids = tokenizer.encode(text, false, true);
        int[] prompt = new int[ids.length + 1];
        System.arraycopy(ids, 0, prompt, 0, ids.length);
        prompt[ids.length] = audioStart;
        return prompt;
    }

    /**
     * Renders the model's {@code tts} template with the utterance, the language, the voice and the BOS text and
     * tokenizes it; the last token must be {@code <|audio_start|>}.
     */
    private int[] ttsPrompt(String text, SpeechOptions options) {
        String template = tokenizer.ttsTemplate();
        if (template == null) {
            throw new UnsupportedOperationException("lfm3a: the model has no '" + GGUFTokenizer.KEY_TTS_TEMPLATE
                    + "' metadata; the tts prompt cannot be rendered");
        }
        List<Map<String, Object>> messages = new ArrayList<>();
        Map<String, Object> utterance = new LinkedHashMap<>();
        utterance.put("role", "user");
        utterance.put("content", text);
        messages.add(utterance);
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("messages", messages);
        context.put("bos_token", bosText());
        String language = options.language();
        if (language != null && !language.trim().isEmpty()) {
            context.put("language", language);
        }
        String voice = options.voice();
        if (voice != null && !voice.trim().isEmpty()) {
            context.put("voice", voice);
        }
        String rendered = MiniJinja.render(template, context);
        int[] ids = tokenizer.encode(rendered, false, true);
        if (ids.length == 0) {
            throw new IllegalArgumentException("the rendered tts prompt tokenizes to no tokens");
        }
        int last = ids[ids.length - 1];
        if (last == mixedStart) {
            throw new IllegalArgumentException("the tts prompt ends in '" + MIXED_START_PIECE + "'; plain speech"
                    + " synthesis needs it to end in '" + AUDIO_START_PIECE + "'");
        }
        if (last != audioStart) {
            throw new IllegalArgumentException("the tts prompt must come from the \"tts\" chat template of the"
                    + " model and end in '" + AUDIO_START_PIECE + "', its last token is '"
                    + tokenizer.decode(last) + "'");
        }
        return ids;
    }

    private String bosText() {
        int bos = tokenizer.bosId();
        return bos >= 0 ? tokenizer.decode(bos) : "";
    }

    /** One run's prompt length and sampled steps. */
    private static final class RunStats {
        final int promptTokens;
        final int steps;

        RunStats(int promptTokens, int steps) {
            this.promptTokens = promptTokens;
            this.steps = steps;
        }
    }

    /** The turn of {@link #generate}, without the caller's sink: the loop over the backend. */
    private RunStats run(int[] promptTokens, Mode asked, boolean continuing, SpeechOptions options, OmniSink sink,
                         List<int[]> frames) {
        Objects.requireNonNull(promptTokens, "promptTokens");
        Objects.requireNonNull(asked, "mode");
        Objects.requireNonNull(options, "options");
        checkOpen();
        if (promptTokens.length == 0) {
            throw new IllegalArgumentException("the prompt holds no tokens");
        }

        List<Integer> prompt = new ArrayList<>(promptTokens.length + 1);
        for (int t : promptTokens) {
            prompt.add(t);
        }
        // only a continued text-only turn bans the switch tokens (mtmd-helper-gen.cpp:1327); im_end and text_end
        // must stay sampleable everywhere else, they are what ends the turn
        int[] banned = null;
        if (continuing) {
            if (asked == Mode.AUDIO) {
                prompt.add(audioStart);
            } else if (asked == Mode.MIXED) {
                prompt.add(mixedStart);
            } else {
                banned = new int[]{audioStart, mixedStart};
            }
        }
        int last = prompt.get(prompt.size() - 1);
        Mode mode = last == audioStart ? Mode.AUDIO : last == mixedStart ? Mode.MIXED : Mode.TEXT;
        if (!continuing && mode != Mode.AUDIO) {
            throw new IllegalArgumentException("the prompt must come from the \"tts\" chat template of the model:"
                    + " a fresh response opens with " + AUDIO_START_PIECE + " (mtmd-helper-gen.cpp:1336)");
        }
        SamplerConfig textConfig = options.sampler() != null ? options.sampler()
                : SamplerConfig.builder()
                        .topK(options.topK())
                        .topP(options.topP())
                        .seed(options.seed())
                        .build();
        Sampler textSampler = banned == null ? Sampler.create(textConfig) : Sampler.create(textConfig, banned);
        int[] tokens = new int[prompt.size()];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = prompt.get(i);
        }
        Loop loop = new Loop(mode, continuing, options, textSampler, sink, tokens.length, frames);
        backend.setHiddenTap(capture);
        int steps;
        try {
            backend.reset();
            capture.clear();
            loop.lastLogits = backend.prefill(tokens, 0);
            loop.hidden = normalizedLastRow();
            steps = loop.execute();
        } finally {
            backend.setHiddenTap(HiddenTap.NONE);
        }
        return new RunStats(tokens.length, steps);
    }

    private static OmniSink fanOut(OmniSink first, OmniSink second) {
        return new OmniSink() {
            @Override
            public void text(String piece) {
                first.text(piece);
                second.text(piece);
            }

            @Override
            public void audio(AudioClip chunk) {
                first.audio(chunk);
                second.audio(chunk);
            }
        };
    }

    private static OmniSink audioOnly(Consumer<AudioClip> sink) {
        return new OmniSink() {
            @Override
            public void text(String piece) {
                // the plain synthesis exposes audio only
            }

            @Override
            public void audio(AudioClip chunk) {
                sink.accept(chunk);
            }
        };
    }

    /**
     * One generation loop, the pipeline state of mtmd-helper-gen.cpp:1673-1700: the mode, the interleaved block
     * countdown, the speech-code history the repetition penalty reads, and the codec session. Lives only inside
     * one {@code run} call; the outer generator seeds its {@code lastLogits} and {@code hidden} from the prompt
     * prefill before {@link #execute}.
     */
    private final class Loop {
        private final boolean continuing;
        private final SpeechOptions options;
        private final Sampler textSampler;
        private final OmniSink sink;
        private final Random rng;
        private final List<int[]> frames;
        private final int chunk;
        private int[] hist = new int[0];

        private Mode mode;
        private boolean speaking;
        private boolean textDone;
        private int slotsLeft = N_TEXT_BLOCK;
        private int pos;
        private float[] lastLogits;
        private float[] hidden;
        private boolean stop;
        private Lfm3aCodec.Decoder wav;
        private Lfm3aDepthformer.Engine depthformer;

        Loop(Mode mode, boolean continuing, SpeechOptions options, Sampler textSampler, OmniSink sink,
             int promptLength, List<int[]> frames) {
            this.mode = mode;
            this.continuing = continuing;
            this.options = options;
            this.textSampler = textSampler;
            this.sink = sink;
            this.frames = frames;
            this.rng = new Random(options.seed());
            this.pos = promptLength;
            long chunkSamples = (long) options.chunkFrames() * clip.codec().hop();
            this.chunk = chunkSamples > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) chunkSamples;
        }

        /**
         * Steps until im_end, the end of audio, or the step cap; then flushes the codec tail (lines 1676-1697).
         * The cap counts every sampled step, text and audio alike, as llama.cpp's {@code n_predict} does
         * (server-context.cpp:3705, tools/tts/tts.cpp:158-181).
         *
         * @return the steps sampled
         */
        int execute() {
            int steps = 0;
            wav = clip.codec().newDecoder(device);
            try {
                depthformer = clip.depthformer().newEngine(device);
                try {
                    for (; steps < options.maxFrames() && !stop; steps++) {
                        boolean interleaving = mode == Mode.MIXED;
                        if (nextIsText()) {
                            stepText(interleaving);
                        } else {
                            stepAudio();
                        }
                        if (interleaving && !textDone && !stop && --slotsLeft == 0) {
                            speaking = !speaking;
                            slotsLeft = speaking ? N_AUDIO_BLOCK : N_TEXT_BLOCK;
                        }
                    }
                    emitAudio(wav.flushClip());
                    return steps;
                } finally {
                    depthformer.close();
                }
            } finally {
                wav.close();
            }
        }

        /** next_is_text: TEXT always; MIXED between its blocks (mtmd-helper-gen.cpp:1686-1688). */
        private boolean nextIsText() {
            return mode == Mode.TEXT || (mode == Mode.MIXED && !speaking);
        }

        /** The text branch (mtmd-helper-gen.cpp:1392-1416): stop, switch, emit, decode. */
        private void stepText(boolean interleaving) {
            int sampled = textSampler.sample(lastLogits);
            textSampler.accept(sampled);
            if (sampled == imEnd || (!interleaving && sampled == textEnd)) {
                stop = true;
                return;
            }
            if (!interleaving) {
                if (sampled == audioStart) {
                    mode = Mode.AUDIO;
                } else if (sampled == mixedStart) {
                    mode = Mode.MIXED;
                    slotsLeft = N_TEXT_BLOCK;
                    speaking = false;
                    textDone = false;
                }
            } else if (sampled == textEnd) {
                textDone = true;
                speaking = true;
            }
            if (!isControl(sampled)) {
                sink.text(tokenizer.getToken(sampled));
            }
            capture.clear();
            lastLogits = backend.decode(sampled, pos);
            hidden = normalizedLastRow();
            pos++;
        }

        /**
         * The audio branch (mtmd-helper-gen.cpp:1417-1450): sample the frame, decode it into the stream, and feed
         * its feedback embedding; the end-of-audio frame flushes the decoder and hands the turn back to text.
         */
        private void stepAudio() {
            Frame frame = sampleFrame(hidden);
            if (frame.isEos && !continuing) {
                stop = true;
                return;
            }
            if (frame.isEos) {
                emitAudio(wav.flushClip());
                mode = Mode.TEXT;
            } else {
                if (frames != null) {
                    frames.add(frame.codes);
                }
                hist = concat(hist, frame.codes);
                emitAudio(wav.decodeClip(frame.codes));
            }
            capture.clear();
            lastLogits = backend.prefillEmbeddings(new float[][]{frame.feedback}, pos);
            hidden = normalizedLastRow();
            pos++;
        }

        private void emitAudio(AudioClip audio) {
            if (sink != null && audio.length() > 0) {
                emit(audio, chunk, sink);
            }
        }

        private boolean isControl(int token) {
            return token == audioStart || token == mixedStart || token == textEnd || token == imEnd
                    || tokenizer.specialTokens().isEog(token);
        }

        /**
         * One speech-code frame (mtmd-helper-gen.cpp:1530-1580): the shuffled codebooks revealed four per
         * depthformer pass over {@code state}, then the end-code check, then the all-known pass that builds the
         * feedback embedding. A fresh end frame builds none — the turn stops without a waveform.
         */
        Frame sampleFrame(float[] state) {
            int[] order = Lfm3aMaskgitSampler.shuffledOrder(N_CODEBOOKS, rng);
            int[] codes = new int[N_CODEBOOKS];
            Arrays.fill(codes, -1);
            int vocab = clip.vocab();
            int k = 0;
            for (int pass = 0; pass < SCHEDULE.length; pass++) {
                Lfm3aDepthformer.Frame p = depthformer.pass(state, codes);
                float[] all = p.logits();
                for (int j = 0; j < SCHEDULE[pass]; j++, k++) {
                    int cb = order[k];
                    float[] l = new float[vocab];
                    System.arraycopy(all, cb * vocab, l, 0, vocab);
                    Lfm3aMaskgitSampler.applyRepPenalty(l, cb, hist, N_CODEBOOKS, clip.penaltyLastN(),
                            clip.penaltyRepeat());
                    codes[cb] = Lfm3aMaskgitSampler.sample(l, clip.temp(), clip.topK(), rng);
                }
            }
            boolean isEos = false;
            for (int c : codes) {
                isEos = isEos || c == vocab - 1;
            }
            float[] feedback;
            if (isEos) {
                Arrays.fill(codes, vocab - 1);
                feedback = continuing ? clip.depthformer().feedback(codes) : null;
            } else {
                feedback = clip.depthformer().feedback(codes);
            }
            return new Frame(codes, feedback, isEos);
        }
    }

    /** One frame's outcome: the sampled codes, the feedback embedding and whether the audio ends here. */
    private static final class Frame {
        final int[] codes;
        final float[] feedback;
        final boolean isEos;

        Frame(int[] codes, float[] feedback, boolean isEos) {
            this.codes = codes;
            this.feedback = feedback;
            this.isEos = isEos;
        }
    }

    private static int[] concat(int[] a, int[] b) {
        int[] out = new int[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /** Delivers {@code audio} in chunks of {@code chunk} samples, or whole when the chunk is not positive. */
    private static void emit(AudioClip audio, int chunk, OmniSink sink) {
        if (chunk <= 0 || audio.length() <= chunk) {
            sink.audio(audio);
            return;
        }
        float[] samples = audio.samples();
        for (int off = 0; off < samples.length; off += chunk) {
            sink.audio(new AudioClip(java.util.Arrays.copyOfRange(samples, off,
                    Math.min(samples.length, off + chunk)), audio.sampleRate()));
        }
    }

    /** The final hidden state: the output norm over the residual row captured at tap {@code nLayer}. */
    private float[] normalizedLastRow() {
        if (!capture.saw) {
            throw new IllegalStateException("the lfm3a backbone delivered no hidden row at tap " + nLayer);
        }
        float[] row = new float[width];
        System.arraycopy(capture.last, (capture.lastRows - 1) * width, row, 0, width);
        CpuOps.rmsNorm(row, 0, row, 0, 1, width, outputNorm, normEps);
        return row;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        clip.close();
    }

    /** Collects the streamed pieces and chunks into one turn or clip. */
    private static final class Collecting implements OmniSink {
        private final StringBuilder text = new StringBuilder();
        private final List<AudioClip> chunks = new ArrayList<>();
        private int length;

        @Override
        public void text(String piece) {
            text.append(piece);
        }

        @Override
        public void audio(AudioClip chunk) {
            chunks.add(chunk);
            length += chunk.length();
        }

        OmniTurn turn(int promptTokens, int steps) {
            float[] samples = new float[length];
            int at = 0;
            for (AudioClip chunk : chunks) {
                System.arraycopy(chunk.samples(), 0, samples, at, chunk.length());
                at += chunk.length();
            }
            int rate = chunks.isEmpty() ? Lfm3aCodec.SAMPLE_RATE : chunks.get(0).sampleRate();
            return new OmniTurn(text.toString(), new AudioClip(samples, rate), promptTokens, steps);
        }
    }

    /** Copies the residual rows entering the output norm, the tap {@code nLayer} of every forward call. */
    private static final class Capture implements HiddenTap {
        private final int layer;
        private final int width;
        private float[] last = new float[0];
        private int lastRows;
        private boolean saw;

        Capture(int layer, int width) {
            this.layer = layer;
            this.width = width;
        }

        void clear() {
            lastRows = 0;
            saw = false;
        }

        @Override
        public boolean wants(int l) {
            return l == layer;
        }

        @Override
        public void onHidden(int l, float[] rows, int nRows, int w) {
            if (l != layer) {
                return;
            }
            if (w != width) {
                throw new IllegalStateException("hidden tap " + l + " delivered rows of " + w + " floats, expected "
                        + width);
            }
            if (last.length < nRows * width) {
                last = new float[nRows * width];
            }
            System.arraycopy(rows, 0, last, 0, nRows * width);
            lastRows = nRows;
            saw = true;
        }
    }

    @Override
    public String toString() {
        return "Lfm3aSpeechGenerator{" + model.spec().arch().ggufName() + " + " + clip.path() + '}';
    }
}
