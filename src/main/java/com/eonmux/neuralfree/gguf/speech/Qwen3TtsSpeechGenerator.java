package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;

import java.util.Objects;
import java.util.Random;
import java.util.function.Consumer;

/**
 * Text-to-speech through a qwen3tts backbone and the code predictor plus code2wav decoder of a
 * {@link Qwen3TtsClip}, the generation pipeline of llama.cpp's {@code qwen3tts_gen_audio_pipeline}
 * (tools/mtmd/mtmd-helper-gen.cpp:138-510).
 *
 * <p>The prompt is built the way the reference builds it: the tokenized {@code
 * "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n"} wrapper contributes its first three
 * tokens and then its utterance body; between them sit the talker's control rows, each the sum of a {@code <tts_*>}
 * token row and a codec control row ({@code <|codec_think|>}, the language token, {@code <|codec_bos|>}). The
 * rows go to the backbone as embedding rows (the projection MLP is folded into {@code token_embd} at conversion
 * time), position by position. A speaker reference adds one more row after the control rows: the file's
 * {@link Qwen3TtsSpeakerEncoder} x-vector summed with {@code <tts_pad>}, so the talker speaks in the reference's
 * voice.
 *
 * <p>Each frame: the backbone's sampled codec token becomes codec 0, the {@link Qwen3TtsCodePredictor} produces
 * the other 15 codes and the feedback embedding, the backbone decodes the codec-0 embedding plus the feedback and
 * its {@code <tts_pad>} overlay, and samples the next codec token. Sampled {@code <|codec_eos_token|>} (or any
 * end-of-generation token) ends the speech. The codes flush through the clip's code2wav decoder whenever a whole
 * decoder window is buffered: each window is decoded and handed to the sink as soon as its frames exist, while the
 * backbone keeps generating the next frames; the one-shot result concatenates the streamed clips.
 *
 * <p>Bound to one backbone and one opened generator file; drives the backbone's forward backend and must run on
 * its engine thread. The generator owns the clip (its {@link #close} closes it).
 *
 * <p>The backbone may run over any engine backend: its token embedding rows and output norm are read through
 * {@link EngineModel#embeddingRow} and {@link EngineModel#outputNormWeights}, so a device engine serves them from
 * its host view (or a device readback it adds), and the sampled codes and PCM match the CPU engine. Over a
 * {@link GGUFModelGPU} the generator's own graphs also run on that engine's device: the code predictor through
 * {@link GpuQwen3TtsCodePredictor} and the code2wav decoder through {@link GpuQwen3TtsCode2Wav}, over weights
 * uploaded once by {@link Qwen3TtsClip#bindDevice}. Over the CPU engine the same graphs run on the host through
 * {@link Qwen3TtsCodePredictor.Session} and {@link Qwen3TtsCode2WavSession}.
 */
public final class Qwen3TtsSpeechGenerator implements SpeechGenerator {

    /** The speech family name. */
    public static final String NAME = "qwen3tts";

    private static final String FULL_PREFIX = "<|im_start|>assistant\n";
    private static final String FULL_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n";

    /** The special token pieces the pipeline requires (mtmd-helper-gen.cpp:404-411). */
    private static final String CODEC_0 = "<|codec_0|>";
    private static final String CODEC_BOS = "<|codec_bos|>";
    private static final String CODEC_EOS = "<|codec_eos_token|>";
    private static final String CODEC_PAD = "<|codec_pad|>";
    private static final String CODEC_THINK = "<|codec_think|>";
    private static final String CODEC_THINK_BOS = "<|codec_think_bos|>";
    private static final String CODEC_THINK_EOS = "<|codec_think_eos|>";
    private static final String TTS_PAD = "<tts_pad>";
    private static final String TTS_BOS = "<tts_text_bos>";
    private static final String TTS_EOS = "<tts_text_eod>";

    private final EngineModel model;
    private final Qwen3TtsClip clip;
    private final SpeechDeviceWeights device;
    private final GGUFTokenizer tokenizer;
    private final ForwardBackend backend;
    private final float[] outputNorm;
    private final float normEps;
    private final int width;
    private final int nLayer;
    private final int codec0;
    private final int codecBos;
    private final int codecEos;
    private final int codecPad;
    private final int codecThink;
    private final int codecThinkBos;
    private final int codecThinkEos;
    private final int ttsPad;
    private final int ttsBos;
    private final int ttsEos;
    private final Capture capture;
    private boolean closed;

    private Qwen3TtsSpeechGenerator(EngineModel model, Qwen3TtsClip clip, SpeechDeviceWeights device)
            throws java.io.IOException {
        this.model = model;
        this.clip = clip;
        this.device = device;
        this.tokenizer = model.tokenizer();
        this.backend = model.forwardBackend();
        ModelHparams hp = model.spec().hparams();
        HeadSpec head = model.spec().head();
        if (head.outputNorm() == null) {
            throw new GgufFormatException("qwen3tts: the backbone has no output norm");
        }
        if (model.spec().embedding().tokenEmbd() == null) {
            throw new GgufFormatException("qwen3tts: the backbone has no token embedding table");
        }
        if (model.spec().embedding().embdScale() != 1f || model.spec().embedding().tokenEmbdNorm() != null
                || model.spec().embedding().inputRmsNorm()) {
            throw new UnsupportedOperationException("qwen3tts: the backbone applies an input normalization the"
                    + " prompt embedding rows do not go through (llama.cpp feeds raw rows)");
        }
        this.width = model.spec().embedding().nEmbd();
        int talkerWidth = clip.predictor().talkerWidth();
        if (width != talkerWidth) {
            throw new IllegalArgumentException("qwen3tts: the backbone hidden width " + width + " does not match the"
                    + " generator's talker width " + talkerWidth + "; the code predictor takes the talker's hidden"
                    + " state");
        }
        this.outputNorm = model.outputNormWeights();
        if (outputNorm.length != width) {
            throw new GgufFormatException("qwen3tts: output_norm holds " + outputNorm.length + " values, expected "
                    + width);
        }
        this.normEps = hp.normEps();
        this.nLayer = model.spec().nLayer();
        this.codec0 = requireToken(CODEC_0);
        this.codecBos = requireToken(CODEC_BOS);
        this.codecEos = requireToken(CODEC_EOS);
        this.codecPad = requireToken(CODEC_PAD);
        this.codecThink = requireToken(CODEC_THINK);
        this.codecThinkBos = requireToken(CODEC_THINK_BOS);
        this.codecThinkEos = requireToken(CODEC_THINK_EOS);
        this.ttsPad = requireToken(TTS_PAD);
        this.ttsBos = requireToken(TTS_BOS);
        this.ttsEos = requireToken(TTS_EOS);
        this.capture = new Capture(nLayer, width);
    }

    /**
     * Binds a generator to {@code model} and {@code clip}.
     *
     * @throws IllegalArgumentException when the model is not a qwen3tts backbone
     * @throws GgufFormatException      when the backbone or the clip lacks a tensor the pipeline reads
     * @throws UnsupportedOperationException when the backbone needs a CPU engine only it cannot provide, or its
     *                                       vocabulary lacks a control token (named)
     */
    public static Qwen3TtsSpeechGenerator open(EngineModel model, Qwen3TtsClip clip) throws java.io.IOException {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(clip, "clip");
        if (model.spec().arch() != Arch.QWEN3TTS) {
            throw new IllegalArgumentException("speech generation through the qwen3tts pipeline needs the "
                    + Arch.QWEN3TTS.ggufName() + " backbone, not " + model.spec().arch().ggufName());
        }
        SpeechDeviceWeights device = model instanceof GGUFModelGPU
                ? clip.bindDevice(((GGUFModelGPU) model).forwardContext())
                : null;
        return new Qwen3TtsSpeechGenerator(model, clip, device);
    }

    private int requireToken(String piece) {
        int id = tokenizer.getTokenId(piece);
        if (id < 0) {
            throw new UnsupportedOperationException("qwen3tts: the backbone vocabulary has no '" + piece
                    + "' token; the pipeline needs it (mtmd-helper-gen.cpp:404-411)");
        }
        return id;
    }

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public int sampleRate() {
        return clip.code2Wav().sampleRate();
    }

    @Override
    public AudioClip synthesize(String text, SpeechOptions options) {
        CollectingSink sink = new CollectingSink(clip.code2Wav().sampleRate());
        run(text, options, sink);
        return sink.clip();
    }

    @Override
    public void synthesize(String text, SpeechOptions options, Consumer<AudioClip> sink) {
        Objects.requireNonNull(sink, "sink");
        run(text, options, sink);
    }

    /** The codes of one generation, for parity tests: one row of {@code codesPerFrame()} codes per frame. */
    int[][] generateCodes(String text, SpeechOptions options) {
        java.util.ArrayList<int[]> frames = new java.util.ArrayList<>();
        generateFrames(text, options, frames::add);
        return frames.toArray(new int[0][]);
    }

    /**
     * Runs the backbone and the code predictor, handing every {@code codesPerFrame()}-wide frame to
     * {@code frameSink} as it is sampled; the sink may decode it while the next frame generates. Returns the
     * number of frames sampled.
     */
    private int generateFrames(String text, SpeechOptions options, java.util.function.Consumer<int[]> frameSink) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(frameSink, "frameSink");
        if (closed) {
            throw new IllegalStateException("the speech generator is closed");
        }
        String language = SpeechOptions.resolveLanguage(options.language());
        int languageToken = requireLanguage(language);
        int[] promptTokens = tokenizePrompt(text);
        float[] speaker = speakerEmbedding(options);
        float[][] prompt = promptEmbeddings(promptTokens, languageToken, speaker);
        Qwen3TtsCodePredictor predictorGraph = clip.predictor();
        int codesPerFrame = predictorGraph.codesPerFrame();
        // the reference falls back to the pipeline defaults when a request leaves top-k/top-p unset
        int topK = options.topK() > 0 ? options.topK() : SpeechOptions.DEFAULT_TOP_K;
        float topP = options.topP() > 0f ? options.topP() : SpeechOptions.DEFAULT_TOP_P;
        Random rng = new Random(options.seed());
        Sampler sampler = Sampler.create(SamplerConfig.builder()
                .topK(topK)
                .topP(topP)
                .seed(options.seed())
                .build(), tokenizer.specialTokens().suppressTokens());
        backend.setHiddenTap(capture);
        Qwen3TtsCodePredictor.Engine predictor = device != null
                ? new GpuQwen3TtsCodePredictor(predictorGraph, device, topK, topP)
                : predictorGraph.newSession(topK, topP);
        try (Qwen3TtsCodePredictor.Engine session = predictor) {
            backend.reset();
            capture.clear();
            float[] logits = backend.prefillEmbeddings(prompt, 0);
            float[] hidden = normalizedLastRow();
            int pos = prompt.length;
            int token = sampleBackbone(logits, sampler);
            int frame = 0;
            while (!isEnd(token) && frame < options.maxFrames()) {
                Qwen3TtsCodePredictor.Frame predicted = session.process(hidden, token - codec0, rng);
                int[] frameCodes = predicted.codes();
                if (frameCodes.length != codesPerFrame) {
                    throw new IllegalStateException("code predictor frame " + frame + " holds " + frameCodes.length
                            + " codes, expected " + codesPerFrame);
                }
                frameSink.accept(frameCodes);
                float[] feedback = predicted.feedback().clone();
                float[] overlay = tokenRow(ttsPad);
                for (int i = 0; i < feedback.length; i++) {
                    feedback[i] += overlay[i];
                }
                capture.clear();
                logits = backend.prefillEmbeddings(new float[][]{feedback}, pos);
                hidden = normalizedLastRow();
                pos++;
                token = sampleBackbone(logits, sampler);
                frame++;
            }
            return frame;
        } finally {
            backend.setHiddenTap(HiddenTap.NONE);
        }
    }

    private void run(String text, SpeechOptions options, Consumer<AudioClip> sink) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(options, "options");
        if (text.trim().isEmpty()) {
            throw new IllegalArgumentException("the speech text must not be blank");
        }
        if (options.hasVoice()) {
            throw new UnsupportedOperationException("qwen3tts has no voice presets: \"" + options.voice()
                    + "\" names none (use \"speaker_ref\" with a reference waveform to clone a voice)");
        }
        // the decoder always runs a whole window (the reference pads a short batch); a chunk subdivides the
        // decoded samples a streaming sink receives, it never changes the decoder's batch
        long chunkSamples = (long) options.chunkFrames() * clip.code2Wav().hop();
        int chunk = chunkSamples > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) chunkSamples;
        int sampleRate = clip.code2Wav().sampleRate();
        try (Qwen3TtsCode2Wav.Decoder decoder = device != null
                ? new GpuQwen3TtsCode2Wav(clip.code2Wav(), device)
                : clip.code2Wav().newSession()) {
            WindowSink windows = new WindowSink(decoder, clip.code2Wav().window(), chunk, sampleRate, sink);
            // each frame decodes with the window once the window fills, so the first window's audio leaves
            // before the backbone generates the following frames
            int produced = generateFrames(text, options, windows::add);
            windows.finish();
            if (produced == 0) {
                sink.accept(new AudioClip(new float[0], sampleRate));
            }
        }
    }

    /**
     * Buffers the generated frames into decoder windows and emits each decoded window as it fills: the samples of
     * consecutive windows concatenate, exactly as decoding the whole sequence at the end does because the same
     * session carries the decoder state across its calls.
     */
    private static final class WindowSink {

        private final Qwen3TtsCode2Wav.Decoder decoder;
        private final int[][] batch;
        private final int chunk;
        private final int sampleRate;
        private final Consumer<AudioClip> sink;
        private int count;

        WindowSink(Qwen3TtsCode2Wav.Decoder decoder, int window, int chunk, int sampleRate,
                   Consumer<AudioClip> sink) {
            this.decoder = decoder;
            this.batch = new int[window][];
            this.chunk = chunk;
            this.sampleRate = sampleRate;
            this.sink = sink;
        }

        void add(int[] frame) {
            batch[count++] = frame;
            if (count == batch.length) {
                decode();
            }
        }

        /** Decodes the frames buffered since the last window; a short final batch is what the one-shot path ends
         * with too. */
        void finish() {
            decode();
        }

        private void decode() {
            if (count == 0) {
                return;
            }
            int[][] frames = new int[count][];
            System.arraycopy(batch, 0, frames, 0, count);
            count = 0;
            AudioClip audio = decoder.decodeClip(frames);
            if (chunk <= 0 || audio.length() <= chunk) {
                sink.accept(audio);
                return;
            }
            float[] samples = audio.samples();
            for (int off = 0; off < samples.length; off += chunk) {
                sink.accept(new AudioClip(java.util.Arrays.copyOfRange(samples, off,
                        Math.min(samples.length, off + chunk)), sampleRate));
            }
        }
    }

    /** Samples the backbone's next codec token over the padded logits row. */
    private int sampleBackbone(float[] paddedLogits, Sampler sampler) {
        int token = sampler.sample(paddedLogits);
        if (!isEnd(token)) {
            int code = token - codec0;
            if (code < 0 || code >= clip.predictor().vocab0()) {
                throw new IllegalStateException("the qwen3tts backbone sampled token " + token + ", outside the"
                        + " codec vocabulary starting at " + codec0);
            }
        }
        return token;
    }

    private boolean isEnd(int token) {
        return token == codecEos || tokenizer.specialTokens().isEog(token);
    }

    private int requireLanguage(String language) {
        String piece = "<|codec_language_" + language + "|>";
        int id = tokenizer.getTokenId(piece);
        if (id < 0) {
            throw new UnsupportedOperationException("qwen3tts: the vocabulary has no language token '" + piece
                    + "' for \"" + language + "\"");
        }
        return id;
    }

    // ------------------------------------------------------------------ prompt construction

    /** Tokenizes the chat-wrapped speech text and checks the wrapper's fixed slices. */
    private int[] tokenizePrompt(String text) {
        int[] ids = tokenizer.encode(FULL_PREFIX + text + FULL_SUFFIX, false, true);
        if (ids.length < 8) {
            throw new IllegalArgumentException("the tokenized speech prompt holds " + ids.length + " tokens; the"
                    + " qwen3tts pipeline needs the chat wrapper's three leading and five trailing tokens");
        }
        return ids;
    }

    /** Prompt rows: the wrapper's three leading token rows, the control rows, the speaker row, the utterance, the
     * closing rows. */
    private float[][] promptEmbeddings(int[] ids, int languageToken, float[] speaker) {
        int n = ids.length;
        java.util.ArrayList<float[]> rows = new java.util.ArrayList<>();
        rows.add(tokenRow(ids[0]));
        rows.add(tokenRow(ids[1]));
        rows.add(tokenRow(ids[2]));
        rows.add(sumRows(ttsPad, codecThink));
        rows.add(sumRows(ttsPad, codecThinkBos));
        rows.add(sumRows(ttsPad, languageToken));
        rows.add(sumRows(ttsPad, codecThinkEos));
        if (speaker != null) {
            rows.add(sumVec(ttsPad, speaker));
        }
        rows.add(sumRows(ttsBos, codecPad));
        for (int i = 3; i < n - 5; i++) {
            rows.add(sumRows(ids[i], codecPad));
        }
        rows.add(sumRows(ttsEos, codecPad));
        rows.add(sumRows(ttsPad, codecBos));
        return rows.toArray(new float[0][]);
    }

    private float[] tokenRow(int token) {
        float[] row;
        try {
            row = model.embeddingRow(token);
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException("qwen3tts: reading token " + token + " from the backbone's"
                    + " embedding table failed", e);
        }
        if (row.length != width) {
            throw new IllegalStateException("the backbone embedding row is " + row.length + " wide, the talker"
                    + " hidden width is " + width);
        }
        return row;
    }

    private float[] sumRows(int a, int b) {
        float[] row = tokenRow(a);
        float[] other = tokenRow(b);
        CpuOps.add(row, 0, other, 0, row, 0, width);
        return row;
    }

    private float[] sumVec(int token, float[] vector) {
        float[] row = tokenRow(token);
        CpuOps.add(row, 0, vector, 0, row, 0, width);
        return row;
    }

    /**
     * The speaker encoder's x-vector of the request's reference, or null without one.
     *
     * @throws UnsupportedOperationException when a reference is given but the generator file carries no speaker
     *                                       encoder
     */
    private float[] speakerEmbedding(SpeechOptions options) {
        AudioClip reference = options.speakerReference();
        if (reference == null) {
            return null;
        }
        Qwen3TtsSpeakerEncoder encoder = clip.speakerEncoder();
        if (encoder == null) {
            throw new UnsupportedOperationException("qwen3tts: \"speaker_ref\" needs the model's speaker encoder,"
                    + " which the generator file " + clip.path() + " does not carry");
        }
        float[][] rows = encoder.encode(reference);
        if (rows.length != 1 || rows[0].length != width) {
            throw new IllegalStateException("the qwen3tts speaker encoder returned " + rows.length + " rows of "
                    + (rows.length == 0 ? 0 : rows[0].length) + " values, expected one row of " + width);
        }
        return rows[0];
    }

    /** The final hidden state: the output norm over the residual row captured at tap {@code nLayer}. */
    private float[] normalizedLastRow() {
        if (!capture.saw) {
            throw new IllegalStateException("the qwen3tts backbone delivered no hidden row at tap " + nLayer);
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

    /** Collects the streamed chunks into one clip. */
    private static final class CollectingSink implements Consumer<AudioClip> {
        private final int sampleRate;
        private float[] samples = new float[0];
        private int length;

        CollectingSink(int sampleRate) {
            this.sampleRate = sampleRate;
        }

        @Override
        public void accept(AudioClip clip) {
            int need = length + clip.length();
            if (need > samples.length) {
                samples = java.util.Arrays.copyOf(samples, Math.max(need, Math.max(64, samples.length * 2)));
            }
            System.arraycopy(clip.samples(), 0, samples, length, clip.length());
            length = need;
        }

        AudioClip clip() {
            return new AudioClip(java.util.Arrays.copyOf(samples, length), sampleRate);
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
        return "Qwen3TtsSpeechGenerator{" + model.spec().arch().ggufName() + " + " + clip.path() + '}';
    }
}
