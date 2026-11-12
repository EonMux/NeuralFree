package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.SpeechFormat;
import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.PagedEvaluations;
import com.eonmux.neuralfree.gguf.engine.PoolingOperator;
import com.eonmux.neuralfree.gguf.ops.audio.GpuWavTokenizer;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionScorer;
import com.eonmux.neuralfree.gguf.ops.heads.EncodeGate;
import com.eonmux.neuralfree.gguf.ops.heads.PoolingHead;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * The host-head tasks of one open {@link GGUFModelGPU}: pooled embeddings, per-token embeddings, classification
 * (sequence, token and decision heads), reranking, next-token logits and the audio decoder. Each task runs the
 * model's forward backend and reads the host-side head weights once ({@link #poolingHead}), so the model's
 * {@link EngineModel} entry points stay thin. The model owns the tasks and runs them on its engine thread.
 */
final class GpuModelTasks implements AutoCloseable {

    private final GGUFModelGPU model;
    private PoolingHead poolingHead;
    private DecisionScorer decisionScorer;
    private PoolingOperator poolingOp;
    private GpuWavTokenizer wavTokenizer;
    /** Speech decoding geometry of the optional audio keys; null when the model declares none. */
    private SpeechFormat speechFormat;

    GpuModelTasks(GGUFModelGPU model) {
        this.model = Objects.requireNonNull(model, "model");
    }

    /**
     * The pooled embedding of {@code tokens} (MEAN / CLS / LAST), the RANK scores, the sparse embedding (the
     * SPLADE reduction), or the per-token rows (NONE).
     */
    float[] encode(int[] tokens, int[] tokenTypes) throws IOException {
        requireTokens(tokens);
        EncodeGate.checkEncode(model.spec());
        PoolingHead head = poolingHead();
        float[][] rows = hidden(tokens, tokenTypes);
        if (head.pooling() != PoolingKind.NONE) {
            return head.pool(rows);
        }
        if (head.sparsePooling() != null) {
            return head.sparseEmbed(rows, tokens);
        }
        return flatten(head.poolTokens(rows, tokens));
    }

    /**
     * {@link #encode(int[], int[])} over a media prompt: the prompt's text and media rows pool as one sequence, the
     * media rows where the projector placed them.
     *
     * @throws UnsupportedOperationException for a sparse embedding
     */
    float[] encodeMedia(MediaPrompt prompt) throws IOException {
        model.ensureOpen();
        Objects.requireNonNull(prompt, "prompt");
        EncodeGate.checkEncode(model.spec());
        PoolingHead head = poolingHead();
        if (head.sparsePooling() != null) {
            throw new UnsupportedOperationException("a sparse embedding encodes token inputs, not a media prompt");
        }
        float[][] rows = hidden(prompt);
        return head.pooling() == PoolingKind.NONE ? flatten(head.poolTokens(rows)) : head.pool(rows);
    }

    /** Per-token embeddings of a model with NONE pooling through the dense projections. */
    float[][] encodeTokens(int[] tokens) throws IOException {
        requireTokens(tokens);
        EncodeGate.checkEncodeTokens(model.spec());
        return poolingHead().poolTokens(hidden(tokens, null), tokens);
    }

    /**
     * The pooled output row of every token sequence of one batch (MEAN / CLS / LAST pooling, the RANK scores): a
     * paged memory runs the sequences as multi-sequence forwards from position 0, an encoder planned without memory
     * packs them into memoryless batches ({@link PagedEvaluations#memorylessSequences}); any other form embeds them
     * one after another.
     */
    float[][] encodeBatch(int[][] sequences) throws IOException {
        model.ensureOpen();
        PagedEvaluations.checkSequences(sequences);
        EncodeGate.checkEncodeBatch(model.spec());
        ForwardBackend f = model.forwardBackend();
        boolean packs = PagedEvaluations.packsMemoryless(f);
        if (!PagedEvaluations.paged(f) && !packs) {
            float[][] out = new float[sequences.length][];
            for (int i = 0; i < sequences.length; i++) {
                out[i] = encode(sequences[i], null);
            }
            return out;
        }
        PoolingHead head = poolingHead();
        float[][][] hidden;
        if (packs) {
            hidden = PagedEvaluations.memorylessSequences(f, sequences);
        } else {
            f.reset();
            hidden = PagedEvaluations.hiddenSequences(f, sequences);
        }
        float[][] out = new float[sequences.length][];
        for (int s = 0; s < sequences.length; s++) {
            out[s] = head.pool(hidden[s]);
        }
        return out;
    }

    /** Classifier output of one input: decision-head scores, or the sequence classifier scores. */
    float[] classify(int[] tokens, int[] decisionOrder) throws IOException {
        requireTokens(tokens);
        DecisionScorer scorer = decisionScorer();
        if (scorer != null) {
            scorer.checkOrder(tokens.length, decisionOrder);
            return scorer.score(hidden(tokens, null), tokens, decisionOrder);
        }
        if (decisionOrder != null) {
            throw new IllegalArgumentException(model.spec().arch().ggufName() + " reads no decision order");
        }
        if (!EncodeGate.hasSequenceClassifier(model.spec())) {
            throw new UnsupportedOperationException("classification needs a RANK classifier, another sequence"
                    + " classifier or a decision head; "
                    + model.spec().arch().ggufName() + " pools by " + model.spec().head().pooling()
                    + (EncodeGate.hasTokenClassifier(model.spec()) ? " and classifies per token (use tokenClassify)"
                    : ""));
        }
        return poolingHead().classify(hidden(tokens, null)).scores();
    }

    /** Per-token classifier scores of one input, {@code [tokens.length][outputs]}. */
    float[][] tokenClassify(int[] tokens) throws IOException {
        requireTokens(tokens);
        EncodeGate.checkTokenClassify(model.spec());
        return poolingHead().classifyTokens(hidden(tokens, null), tokens);
    }

    /** The classifier labels of the model's sequence or token head; empty for a decision or head-less model. */
    String[] classLabels() {
        model.ensureOpen();
        if (DecisionScorer.present(model.spec())
                || (!EncodeGate.hasSequenceClassifier(model.spec())
                && !EncodeGate.hasTokenClassifier(model.spec()))) {
            return new String[0];
        }
        try {
            return poolingHead().classifier().labels();
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read the classifier tensors of "
                    + model.spec().arch().ggufName(), e);
        }
    }

    /** The sequence classifier's output for one input, with the final activation applied or skipped. */
    ClassifierOutput classifyOutput(int[] tokens, boolean activation) throws IOException {
        requireTokens(tokens);
        if (!EncodeGate.hasDecisionClassifier(model.spec())) {
            if (DecisionScorer.present(model.spec())) {
                throw new UnsupportedOperationException(model.spec().arch().ggufName() + " is a decision model; its"
                        + " decision scores come from classify, not a classifier output");
            }
            if (!EncodeGate.hasSequenceClassifier(model.spec())) {
                throw new UnsupportedOperationException("a classifier output needs a RANK classifier or another"
                        + " sequence classifier; " + model.spec().arch().ggufName() + " pools by "
                        + model.spec().head().pooling());
            }
        }
        return poolingHead().classify(hidden(tokens, null), activation);
    }

    /** {@link #classifyOutput(int[], boolean)} over a media prompt (a paged memory is required). */
    ClassifierOutput classifyOutputMedia(MediaPrompt prompt, boolean activation) throws IOException {
        model.ensureOpen();
        Objects.requireNonNull(prompt, "prompt");
        if (!EncodeGate.hasDecisionClassifier(model.spec())) {
            if (DecisionScorer.present(model.spec())) {
                throw new UnsupportedOperationException(model.spec().arch().ggufName() + " is a decision model; its"
                        + " decision scores come from classify, not a classifier output");
            }
            if (!EncodeGate.hasSequenceClassifier(model.spec())) {
                throw new UnsupportedOperationException("a classifier output needs a RANK classifier or another"
                        + " sequence classifier; " + model.spec().arch().ggufName() + " pools by "
                        + model.spec().head().pooling());
            }
        }
        return poolingHead().classify(hidden(prompt), activation);
    }

    /** The next-token logits over the whole vocabulary after {@code tokens}, from position 0. */
    float[] nextLogits(int[] tokens) throws IOException {
        requireTokens(tokens);
        if (model.spec().supportClass() != SupportClass.GEN) {
            throw new UnsupportedOperationException("next-token logits need a generative model; "
                    + model.spec().arch().ggufName() + " is a " + model.spec().supportClass() + " model");
        }
        ForwardBackend f = model.forwardBackend();
        if (f.memory() == null) {
            // a non-causal generative model (an lfm3 d1 checkpoint) evaluates the whole prompt in one step
            return f.forwardAll(tokens, new int[]{tokens.length - 1})[0].clone();
        }
        f.reset();
        if (PagedEvaluations.paged(f)) {
            return PagedEvaluations.nextLogits(f, tokens);
        }
        return f.prefill(tokens, 0).clone();
    }

    /** The next-token logits after the last row of a media prompt. */
    float[] nextLogits(MediaPrompt prompt) throws IOException {
        model.ensureOpen();
        Objects.requireNonNull(prompt, "prompt");
        if (model.spec().supportClass() != SupportClass.GEN) {
            throw new UnsupportedOperationException("next-token logits need a generative model; "
                    + model.spec().arch().ggufName() + " is a " + model.spec().supportClass() + " model");
        }
        ForwardBackend f = model.forwardBackend();
        if (f.memory() != null) {
            f.reset();
        }
        if (PagedEvaluations.paged(f)) {
            return PagedEvaluations.nextLogits(f, prompt);
        }
        if (f.memory() == null) {
            return PagedEvaluations.memorylessNextLogits(f, prompt);
        }
        throw new UnsupportedOperationException("a media prompt needs paged memory (neuralfree.parallel > 1"
                + " or neuralfree.kv.paged=true); " + model.spec().arch().ggufName()
                + " holds a contiguous memory");
    }

    /** Decision scores of one media prompt, read by the model's decision head. */
    float[] classifyMedia(MediaPrompt prompt, int[] decisionOrder) throws IOException {
        model.ensureOpen();
        Objects.requireNonNull(prompt, "prompt");
        DecisionScorer scorer = decisionScorer();
        if (scorer == null) {
            throw new UnsupportedOperationException("classification needs a decision head; "
                    + model.spec().arch().ggufName() + " has none");
        }
        ForwardBackend f = model.forwardBackend();
        if (!PagedEvaluations.paged(f) && f.memory() != null) {
            throw new UnsupportedOperationException("a media prompt needs paged memory (neuralfree.parallel > 1"
                    + " or neuralfree.kv.paged=true); " + model.spec().arch().ggufName()
                    + " holds a contiguous memory");
        }
        int[] tokens = f.memory() == null ? PagedEvaluations.tokens(prompt) : PagedEvaluations.tokens(f, prompt);
        scorer.checkOrder(tokens.length, decisionOrder);
        return scorer.score(hidden(prompt), tokens, decisionOrder);
    }

    /** Relevance of a tokenized {@code document} for a tokenized {@code query}. */
    float rerank(int[] query, int[] document) throws IOException {
        model.ensureOpen();
        EncodeGate.checkRerank(model.spec());
        return poolingHead().rerank(hidden(ClassifierHead.rerankInput(model.tokenizer(), query, document), null));
    }

    /** Relevance of {@code document} for {@code query} through the model's rerank prompt. */
    float rerank(String query, String document) throws IOException {
        model.ensureOpen();
        EncodeGate.checkRerank(model.spec());
        return poolingHead().rerank(hidden(ClassifierHead.rerankInput(model.tokenizer(), query, document), null));
    }

    /** WavTokenizer features of audio codes through the decoder bound at load. */
    float[][] decodeAudioCodes(int[] codes) {
        model.ensureOpen();
        if (model.spec().arch() != Arch.WAVTOKENIZER_DEC) {
            throw new UnsupportedOperationException("audio code decoding not supported by "
                    + model.spec().arch().ggufName());
        }
        return wavTokenizer.decode(codes);
    }

    /** The speech decoding geometry the model declares, or null. */
    SpeechFormat speechFormat() {
        return speechFormat;
    }

    /** The speech decoding geometry, refusing like {@link #synthesizeSpeech(int[])} when there is none. */
    SpeechFormat requireSpeechFormat() {
        model.ensureOpen();
        if (model.spec().arch() != Arch.WAVTOKENIZER_DEC) {
            throw new UnsupportedOperationException("speech synthesis not supported by "
                    + model.spec().arch().ggufName());
        }
        if (speechFormat == null) {
            throw new UnsupportedOperationException("the model declares no speech decoding geometry: give the GGUF"
                    + " keys " + SpeechFormat.SAMPLE_RATE_KEY + ", "
                    + SpeechFormat.N_FFT_KEY + " and "
                    + SpeechFormat.HOP_LENGTH_KEY + " (all three) to synthesize"
                    + " waveforms");
        }
        return speechFormat;
    }

    /** The waveform of audio codec codes through the loaded model's speech decoder. */
    AudioClip synthesizeSpeech(int[] codes) {
        return requireSpeechFormat().decode(wavTokenizer.decode(codes));
    }

    /**
     * {@link #synthesizeSpeech(int[])} delivered frame by frame; see
     * {@link SpeechFormat#decode(float[][], Consumer)}.
     */
    void synthesizeSpeech(int[] codes, Consumer<AudioClip> sink) {
        requireSpeechFormat().decode(wavTokenizer.decode(codes), sink);
    }

    /** The pooling operator over the host heads (their first touch may read a tensor, so it can fail). */
    PoolingOperator poolingOperator() {
        if (poolingOp == null) {
            poolingOp = PoolingOperator.of(model.spec(), this::uncheckedHead, this::uncheckedScorer);
        }
        return poolingOp;
    }

    /** Records the audio decoder the load bound, closing the previous one when replaced. */
    void bindWavTokenizer(GpuWavTokenizer decoder) {
        this.wavTokenizer = decoder;
    }

    /** Records the speech decoding geometry read at load. */
    void setSpeechFormat(SpeechFormat format) {
        this.speechFormat = format;
    }

    /** Releases the audio decoder, if the load bound one. */
    @Override
    public void close() throws IOException {
        GpuWavTokenizer decoder = wavTokenizer;
        wavTokenizer = null;
        if (decoder != null) {
            decoder.close();
        }
    }

    /** Checks that the model is open and {@code tokens} is a non-empty input. */
    private void requireTokens(int[] tokens) {
        model.ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("tokens must not be empty");
        }
    }

    /**
     * Final hidden rows from position 0, emptying the memory first (the generation state is discarded); a paged
     * memory runs them as a one-sequence batch.
     */
    private float[][] hidden(int[] tokens, int[] tokenTypes) {
        requireTokens(tokens);
        ForwardBackend f = model.forwardBackend();
        if (f.memory() != null) {
            f.reset();
        }
        if (PagedEvaluations.paged(f)) {
            return PagedEvaluations.hiddenRows(f, PagedEvaluations.untyped(tokens, tokenTypes));
        }
        return f.hiddenRows(tokens, tokenTypes);
    }

    /**
     * {@link #hidden(int[], int[])} over a media prompt: the paged memory runs the chunks as consecutive batches; a
     * model planned without memory runs the whole prompt as one non-causal batch; any other memory form refuses by
     * name.
     */
    private float[][] hidden(MediaPrompt prompt) {
        ForwardBackend f = model.forwardBackend();
        if (f.memory() != null) {
            f.reset();
        }
        if (PagedEvaluations.paged(f)) {
            return PagedEvaluations.hiddenRows(f, prompt);
        }
        if (f.memory() == null) {
            return PagedEvaluations.memorylessRows(f, prompt);
        }
        throw new UnsupportedOperationException("a media prompt needs paged memory (neuralfree.parallel > 1"
                + " or neuralfree.kv.paged=true); " + model.spec().arch().ggufName()
                + " holds a contiguous memory");
    }

    private PoolingHead poolingHead() throws IOException {
        if (poolingHead == null) {
            poolingHead = PoolingHead.of(model.spec(), model.cpuWeights());
        }
        return poolingHead;
    }

    /** The decision head of clef, vllm-decision2, kev and the modern-bert / lfm2 decision models, or null. */
    private DecisionScorer decisionScorer() throws IOException {
        if (decisionScorer == null && DecisionScorer.present(model.spec())) {
            decisionScorer = DecisionScorer.forModel(model.spec(), model.cpuWeights(), model.tokenizer());
        }
        return decisionScorer;
    }

    private PoolingHead uncheckedHead() {
        try {
            return poolingHead();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private DecisionScorer uncheckedScorer() {
        try {
            return decisionScorer();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Concatenates {@code rows} in order. */
    private static float[] flatten(float[][] rows) {
        int total = 0;
        for (float[] r : rows) {
            total = Math.addExact(total, r.length);
        }
        float[] out = new float[total];
        int off = 0;
        for (float[] r : rows) {
            System.arraycopy(r, 0, out, off, r.length);
            off += r.length;
        }
        return out;
    }
}
