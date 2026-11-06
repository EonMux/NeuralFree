package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.PocketTtsSession;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Consumer;

/**
 * Text-to-speech through a pockettts backbone and the flow-matching plus mimi decoder graphs of a
 * {@link PocketTtsClip}, the generation pipeline of llama.cpp's {@code pockettts_gen_audio_pipeline}
 * (tools/mtmd/mtmd-helper-gen.cpp:522-1082).
 *
 * <p>The prompt is the prepared text's tokens followed by the {@code <|audio_bos|>} row. A speaker reference puts
 * the file's {@link PocketTtsSpeakerEncoder} rows first, behind an optional {@code <|bos_before_voice|>} row, so the
 * backbone conditions every chunk on the reference's voice. Each frame the backbone's
 * hidden state drives two graphs of the generator file: {@link PocketTtsFlow} turns it into a continuous latent
 * (flow matching from Gaussian noise of standard deviation {@code sqrt(temperature)} and the end-of-speech probe),
 * and the latent's projection through {@code a.gen.input_linear} becomes the backbone's next input row. There are
 * no sampled tokens, so the pack's temperature is the only draw control; {@code top-k} and {@code top-p} of a
 * speech request do not apply. A probe score above -4 latches the end of speech, after which the pack's
 * {@code frames_after_eos} frames (or the reference's prompt-length guess) are kept and the stopping frame is
 * discarded.
 *
 * <p>Long prompts are split into chunks of at most 50 tokens on sentence punctuation, then on weaker punctuation,
 * and each chunk restarts the backbone and the decoder; a chunk never shares state with the next one. Latents are
 * decoded in windows of {@value PocketTtsMimiDecoder#WINDOW_FRAMES} frames by {@link PocketTtsMimiSession}, whose
 * convolution contexts and attention window carry across the windows of one chunk.
 *
 * <p>Bound to one backbone and one opened generator file; drives the backbone's forward backend and must run on its
 * engine thread. The generator owns the clip (its {@link #close} closes it).
 *
 * <p>The backbone may run over any engine backend. Over a {@link GGUFModelGPU} the flow network and the mimi
 * decoder also run on that engine's device, through {@link GpuPocketTtsFlow} and
 * {@link GpuPocketTtsMimiDecoder} over weights uploaded once by {@link PocketTtsClip#bindDevice}; the Gaussian
 * draws and the end-of-speech ordering stay on the host. Over the CPU engine the same graphs run on the host
 * through {@link PocketTtsFlow} and {@link PocketTtsMimiSession}.
 */
public final class PocketTtsSpeechGenerator implements SpeechGenerator {

    /** The speech family name. */
    public static final String NAME = "pockettts";

    /** The token that starts generation (mtmd-helper-gen.cpp:404-411). */
    private static final String AUDIO_BOS = "<|audio_bos|>";
    /** The optional token before a voice reference (mtmd-helper-gen.cpp:790-791). */
    private static final String BOS_BEFORE_VOICE = "<|bos_before_voice|>";

    /** {@code MAX_TOKEN_PER_CHUNK} of the reference (mtmd-helper-gen.cpp:1067). */
    static final int MAX_CHUNK_TOKENS = 50;
    /** The reference's frame rate, its generation-budget unit (mtmd-helper-gen.cpp:1068). */
    static final double FRAME_RATE = 12.5;

    private final EngineModel model;
    private final PocketTtsClip clip;
    private final SpeechDeviceWeights device;
    private final GGUFTokenizer tokenizer;
    private final int width;
    private final int audioBos;
    private final int bosBeforeVoice;
    private boolean closed;

    private PocketTtsSpeechGenerator(EngineModel model, PocketTtsClip clip, SpeechDeviceWeights device) {
        this.model = model;
        this.clip = clip;
        this.device = device;
        this.tokenizer = model.tokenizer();
        this.width = model.spec().hparams().nEmbd();
        if (model.spec().embedding().tokenEmbd() == null) {
            throw new GgufFormatException("pockettts: the backbone has no token embedding table");
        }
        if (model.spec().embedding().embdScale() != 1f || model.spec().embedding().tokenEmbdNorm() != null
                || model.spec().embedding().inputRmsNorm()) {
            throw new UnsupportedOperationException("pockettts: the backbone applies an input normalization the"
                    + " prompt embedding rows do not go through (llama.cpp feeds raw rows)");
        }
        if (clip.flow().condWidth() != width) {
            throw new IllegalArgumentException("pockettts: the backbone hidden width " + width + " does not match"
                    + " the flow network's conditioning width " + clip.flow().condWidth());
        }
        this.audioBos = tokenizer.getTokenId(AUDIO_BOS);
        if (audioBos < 0) {
            throw new UnsupportedOperationException("pockettts: the backbone vocabulary has no '" + AUDIO_BOS
                    + "' token; the pipeline needs it (mtmd-helper-gen.cpp:404-411)");
        }
        this.bosBeforeVoice = tokenizer.getTokenId(BOS_BEFORE_VOICE);
    }

    /**
     * Binds a generator to {@code model} and {@code clip}.
     *
     * @throws IllegalArgumentException when the model is not a pockettts backbone
     * @throws GgufFormatException      when the backbone or the clip lacks a tensor the pipeline reads
     * @throws UnsupportedOperationException when the backbone needs an input normalization the reference does not
     *                                       apply, or its vocabulary lacks {@code <|audio_bos|>}
     * @throws java.io.IOException          when the generator file cannot be bound to the backbone's device
     */
    public static PocketTtsSpeechGenerator open(EngineModel model, PocketTtsClip clip) throws java.io.IOException {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(clip, "clip");
        if (model.spec().arch() != Arch.POCKETTTS) {
            throw new IllegalArgumentException("speech generation through the pockettts pipeline needs the "
                    + Arch.POCKETTTS.ggufName() + " backbone, not " + model.spec().arch().ggufName());
        }
        HeadSpec head = model.spec().head();
        if (head.outputNorm() == null || head.outputNormBias() == null) {
            throw new GgufFormatException("pockettts: the backbone has no biased output norm");
        }
        SpeechDeviceWeights device = model instanceof GGUFModelGPU
                ? clip.bindDevice(((GGUFModelGPU) model).forwardContext())
                : null;
        return new PocketTtsSpeechGenerator(model, clip, device);
    }

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public int sampleRate() {
        return clip.decoder().sampleRate();
    }

    @Override
    public AudioClip synthesize(String text, SpeechOptions options) {
        CollectingSink sink = new CollectingSink(clip.decoder().sampleRate());
        run(text, options, sink);
        return sink.clip();
    }

    @Override
    public void synthesize(String text, SpeechOptions options, Consumer<AudioClip> sink) {
        Objects.requireNonNull(sink, "sink");
        run(text, options, sink);
    }

    /** The latents of one synthesis, for parity tests: one {@link PocketTtsFlow#latent()}-wide row per frame. */
    float[][] generateLatents(String text, SpeechOptions options) {
        ensureOpen();
        List<float[]> frames = new ArrayList<>();
        generate(text, options, frames, clip -> { });
        return frames.toArray(new float[0][]);
    }

    private void run(String text, SpeechOptions options, Consumer<AudioClip> sink) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(options, "options");
        ensureOpen();
        if (options.hasVoice()) {
            throw new UnsupportedOperationException("pockettts has no voice presets: \"voice\": \""
                    + options.voice() + "\" names none (use \"speaker_ref\" with a reference waveform to clone a"
                    + " voice)");
        }
        requireLanguage(options);
        generate(text, options, null, sink);
    }

    /**
     * Generates the frames and decodes them as they fill the decoder's window. When {@code latents} is non-null the
     * frames are collected there and nothing is decoded; the streaming sink is unused then.
     */
    private void generate(String text, SpeechOptions options, List<float[]> latents, Consumer<AudioClip> sink) {
        String prepared = prepareText(text, clip.pack().padShortText());
        if (prepared.isEmpty()) {
            throw new IllegalArgumentException("the speech text must not be blank");
        }
        int[] ids = tokenizer.encode(prepared, false, false);
        if (ids.length == 0) {
            throw new IllegalArgumentException("the speech text tokenizes to no token");
        }
        List<int[]> chunks = splitChunks(ids);
        PocketTtsPack pack = clip.pack();
        Random rng = new Random(options.seed());
        int maxFrames = options.maxFrames();
        int total = 0;
        float[][] voice = speakerRows(options);
        try (PocketTtsSession session = PocketTtsSession.open(model);
             PocketTtsFlow.Engine flow = clip.flow().newEngine(device)) {
            for (int ci = 0; ci < chunks.size() && total < maxFrames; ci++) {
                int[] chunk = chunks.get(ci);
                total += generateChunk(session, flow, chunk, voice, pack, rng, Math.min(maxFrames - total,
                        chunkBudget(chunk.length)), latents, sink, options);
            }
        } catch (IOException e) {
            throw new IllegalStateException("the pockettts backbone could not be bound: " + e.getMessage(), e);
        }
        if (latents == null && total == 0) {
            sink.accept(new AudioClip(new float[0], clip.decoder().sampleRate()));
        }
    }

    /** Runs one chunk from its prompt to its last kept frame; returns the number of kept frames. */
    private int generateChunk(PocketTtsSession session, PocketTtsFlow.Engine flow, int[] chunk,
                              float[][] voice, PocketTtsPack pack,
                              Random rng, int budget, List<float[]> latents, Consumer<AudioClip> sink,
                              SpeechOptions options) {
        float[][] prompt = promptRows(chunk, voice);
        session.reset();
        float[] hidden = session.prefillEmbeddings(prompt, 0);
        int pos = prompt.length;
        int framesAfterEos = pack.framesAfterEos(chunk.length);
        WindowSink windows = latents == null ? new WindowSink(sink, options, clip) : null;
        List<float[]> pending = new ArrayList<>();
        int step = 0;
        int eosStep = -1;
        int kept = 0;
        try (PocketTtsMimiDecoder.Decoder decoder = clip.decoder().newDecoder(device)) {
            while (true) {
                if (flow.isEndOfSpeech(hidden) && eosStep < 0) {
                    eosStep = step;
                }
                // the frame of the stopping step is still drawn, then discarded, matching step_gen()
                float[] latent = flow.decode(hidden, pack.temperature(), rng);
                if ((eosStep >= 0 && step >= eosStep + framesAfterEos) || step >= budget) {
                    break;
                }
                if (latents != null) {
                    latents.add(latent);
                } else {
                    pending.add(latent);
                    if (pending.size() == PocketTtsMimiDecoder.WINDOW_FRAMES) {
                        decode(decoder, pending, windows);
                        pending.clear();
                    }
                }
                kept++;
                step++;
                float[] feedback = flow.project(latent);
                hidden = session.decodeEmbedding(feedback, pos);
                pos++;
            }
            if (windows != null && !pending.isEmpty()) {
                decode(decoder, pending, windows);
            }
        }
        return kept;
    }

    private void decode(PocketTtsMimiDecoder.Decoder decoder, List<float[]> frames, WindowSink sink) {
        float[] samples = decoder.decode(frames.toArray(new float[0][]));
        sink.accept(new AudioClip(samples, clip.decoder().sampleRate()));
    }

    /** The per-chunk frame budget of mtmd-helper-gen.cpp:898-900: {@code ceil((n/3 + 2) * 12.5)}. */
    static int chunkBudget(int tokens) {
        return (int) Math.ceil((tokens / 3.0 + 2.0) * FRAME_RATE);
    }

    // ------------------------------------------------------------------ text preparation

    /**
     * The reference's {@code prepare_text} normalization (mtmd-helper-gen.cpp:954-983): newlines and carriage
     * returns become spaces, semicolons commas, the ends are trimmed, a leading lower-case ASCII letter is
     * capitalized, a trailing alphanumeric gets a period, and a short prompt optionally gets eight leading spaces.
     */
    static String prepareText(String text, boolean padShort) {
        StringBuilder b = new StringBuilder(text.length() + 1);
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '\n' || c == '\r') {
                b.append(' ');
            } else if (c == ';') {
                b.append(',');
            } else {
                b.append(c);
            }
        }
        int start = 0;
        while (start < b.length() && b.charAt(start) == ' ') {
            start++;
        }
        int end = b.length();
        while (end > start && b.charAt(end - 1) == ' ') {
            end--;
        }
        if (start == end) {
            return "";
        }
        String s = b.substring(start, end);
        char first = s.charAt(0);
        if (first >= 'a' && first <= 'z') {
            s = (char) (first - 'a' + 'A') + s.substring(1);
        }
        char last = s.charAt(s.length() - 1);
        if ((last >= 'a' && last <= 'z') || (last >= 'A' && last <= 'Z') || (last >= '0' && last <= '9')) {
            s = s + '.';
        }
        if (padShort && countWords(s) < 5) {
            s = "        " + s;
        }
        return s;
    }

    /** Words separated by spaces, as mtmd-helper-gen.cpp:985-998 counts them. */
    static int countWords(String s) {
        int n = 0;
        boolean inWord = false;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ' ') {
                inWord = false;
            } else if (!inWord) {
                inWord = true;
                n++;
            }
        }
        return n;
    }

    // ------------------------------------------------------------------ chunking

    /** Splits a token sequence into chunks of at most {@value #MAX_CHUNK_TOKENS} tokens
     * (mtmd-helper-gen.cpp:849-896). */
    List<int[]> splitChunks(int[] ids) {
        List<int[]> out = new ArrayList<>();
        if (ids.length <= MAX_CHUNK_TOKENS) {
            out.add(ids);
            return out;
        }
        int[] eosPunct = punctIds(".!...?");
        int[] midPunct = punctIds(",;:");
        List<int[]> segments = new ArrayList<>();
        for (int[] segment : splitOn(ids, eosPunct)) {
            if (segment.length <= MAX_CHUNK_TOKENS) {
                segments.add(segment);
                continue;
            }
            List<int[]> sub = splitOn(segment, midPunct);
            if (sub.size() > 1) {
                segments.addAll(sub);
            } else {
                segments.add(segment);
            }
        }
        for (int[] segment : segments) {
            if (segment.length == 0) {
                continue;
            }
            int[] last = out.isEmpty() ? null : out.get(out.size() - 1);
            if (last != null && last.length + segment.length <= MAX_CHUNK_TOKENS) {
                int[] merged = Arrays.copyOf(last, last.length + segment.length);
                System.arraycopy(segment, 0, merged, last.length, segment.length);
                out.set(out.size() - 1, merged);
            } else {
                out.add(segment);
            }
        }
        if (out.isEmpty()) {
            out.add(ids);
        }
        return out;
    }

    /**
     * The token ids of the pieces the reference splits on; the leading token is dropped, the tokenizer's dummy
     * prefix (mtmd-helper-gen.cpp:820-830).
     */
    private int[] punctIds(String s) {
        int[] ids = tokenizer.encode(s, false, false);
        return ids.length <= 1 ? new int[0] : Arrays.copyOfRange(ids, 1, ids.length);
    }

    /** Cuts after runs of boundary tokens, so punctuation stays with the sentence it ends. */
    private static List<int[]> splitOn(int[] ids, int[] boundary) {
        List<int[]> out = new ArrayList<>();
        int start = 0;
        boolean previousBoundary = false;
        for (int i = 0; i < ids.length; i++) {
            boolean isBoundary = contains(boundary, ids[i]);
            if (!isBoundary && previousBoundary) {
                out.add(Arrays.copyOfRange(ids, start, i));
                start = i;
            }
            previousBoundary = isBoundary;
        }
        out.add(Arrays.copyOfRange(ids, start, ids.length));
        return out;
    }

    private static boolean contains(int[] values, int value) {
        for (int v : values) {
            if (v == value) {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ prompt construction

    /**
     * The embedding rows of one chunk: the optional {@code <|bos_before_voice|>} row, the voice conditioning rows,
     * the chunk's token rows and the {@code <|audio_bos|>} row (mtmd-helper-gen.cpp:578-633). Package-private so the
     * voice-prompt test can check that order.
     */
    float[][] promptRows(int[] chunk, float[][] voice) {
        int extra = voice == null ? 0 : voice.length + (bosBeforeVoice >= 0 ? 1 : 0);
        float[][] rows = new float[extra + chunk.length + 1][];
        int at = 0;
        if (voice != null) {
            if (bosBeforeVoice >= 0) {
                rows[at++] = tokenRow(bosBeforeVoice);
            }
            for (float[] row : voice) {
                rows[at++] = row.clone();
            }
        }
        for (int token : chunk) {
            rows[at++] = tokenRow(token);
        }
        rows[at] = tokenRow(audioBos);
        return rows;
    }

    /**
     * The speaker encoder's rows of the request's reference, or null without one.
     *
     * @throws UnsupportedOperationException when a reference is given but the generator file carries no speaker
     *                                       encoder
     */
    private float[][] speakerRows(SpeechOptions options) {
        AudioClip reference = options.speakerReference();
        if (reference == null) {
            return null;
        }
        PocketTtsSpeakerEncoder encoder = clip.speakerEncoder();
        if (encoder == null) {
            throw new UnsupportedOperationException("pockettts: \"speaker_ref\" needs the model's speaker encoder,"
                    + " which the generator file " + clip.path() + " does not carry");
        }
        if (encoder.width() != width) {
            throw new IllegalArgumentException("pockettts: the speaker encoder writes " + encoder.width()
                    + "-value rows, the backbone takes " + width);
        }
        return encoder.encode(reference);
    }

    private float[] tokenRow(int token) {
        float[] row;
        try {
            row = model.embeddingRow(token);
        } catch (IOException e) {
            throw new java.io.UncheckedIOException("pockettts: reading token " + token + " from the backbone's"
                    + " embedding table failed", e);
        }
        if (row.length != width) {
            throw new IllegalStateException("the backbone embedding row is " + row.length + " wide, expected "
                    + width);
        }
        return row;
    }

    // ------------------------------------------------------------------ refusals

    /** A pack speaks one language; a request naming another is refused by name rather than ignored. */
    private void requireLanguage(SpeechOptions options) {
        String requested = options.language();
        if (requested == null || requested.trim().isEmpty()) {
            return;
        }
        String language = SpeechOptions.resolveLanguage(requested);
        String speaks = clip.pack().language();
        if (!language.equals(speaks)) {
            throw new UnsupportedOperationException("the pockettts pack \"" + clip.pack().variant() + "\" speaks "
                    + speaks + "; \"lang\": \"" + requested + "\" is not supported by this model");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the pockettts speech generator is closed");
        }
    }

    /** The generator owns the clip and releases it here; the model stays open. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        clip.close();
    }

    @Override
    public String toString() {
        return "PocketTtsSpeechGenerator{" + model.spec().arch().ggufName() + " + " + clip.path() + '}';
    }

    /** Emits each decoded window, subdivided into the caller's requested chunk of samples. */
    private static final class WindowSink {
        private final Consumer<AudioClip> sink;
        private final int chunkSamples;

        WindowSink(Consumer<AudioClip> sink, SpeechOptions options, PocketTtsClip clip) {
            this.sink = sink;
            long samples = (long) options.chunkFrames() * clip.decoder().hop();
            this.chunkSamples = samples > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) samples;
        }

        void accept(AudioClip audio) {
            if (chunkSamples <= 0 || audio.length() <= chunkSamples) {
                sink.accept(audio);
                return;
            }
            float[] samples = audio.samples();
            for (int off = 0; off < samples.length; off += chunkSamples) {
                sink.accept(new AudioClip(Arrays.copyOfRange(samples, off,
                        Math.min(samples.length, off + chunkSamples)), audio.sampleRate()));
            }
        }
    }

    /** Collects the streamed windows into one clip. */
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
                samples = Arrays.copyOf(samples, Math.max(need, Math.max(64, samples.length * 2)));
            }
            System.arraycopy(clip.samples(), 0, samples, length, clip.length());
            length = need;
        }

        AudioClip clip() {
            return new AudioClip(Arrays.copyOf(samples, length), sampleRate);
        }
    }
}
