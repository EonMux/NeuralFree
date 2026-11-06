package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;
import java.util.Random;

/**
 * The qwen3tts code predictor of llama.cpp's {@code clip_graph_qwen3tts_gen} GEN_CODE graph
 * (tools/mtmd/models/qwen3tts-gen.cpp:150-400): the talker hidden state and the backbone's codec-0 token become
 * the frame's 16 codebook codes.
 *
 * <ol>
 *   <li>the {@code small_to_mtp_projection} maps both inputs into the predictor width when the talker is wider;</li>
 *   <li>position 0 carries the talker hidden state (it only seeds the key/value cache), position 1 the codec-0
 *       embedding; each further step embeds the previous code with that step's codebook table;</li>
 *   <li>every position runs the predictor transformer: RMS norms, per-head Q/K RMS norms, NEOX RoPE at the step
 *       position, causal attention over the 16 cache rows, SwiGLU FFN;</li>
 *   <li>the final RMS norm feeds the step's page of the stacked per-codebook LM head, sampled with top-k/top-p
 *       ({@link SpeechSampling}, the graph's {@code do_sampling});</li>
 *   <li>the frame's feedback row is the codec-0 embedding plus the sum of the 15 acoustic codebook embeddings,
 *       which the backbone adds to its {@code <tts_pad>} overlay.</li>
 * </ol>
 *
 * <p>Tensors use the clip generator names of tools/mtmd/clip-impl.h:248-252 (ggml dimension order, {@code ne[0]}
 * first); the codebook tables are 3-D {@code [n_embd, vocab, n_codebooks]} pages. Immutable geometry over shared
 * host weights; a {@link Session} holds the per-frame key/value cache and is not thread-safe.
 */
public final class Qwen3TtsCodePredictor {

    /** The {@code clip.gen.audio.projector_type} of the code predictor. */
    public static final String PROJECTOR_TYPE = "qwen3tts_gen";
    /** The {@code general.architecture} of a clip generator file. */
    public static final String ARCH = "clip";

    /** Tensor names of tools/mtmd/clip-impl.h:248-252. */
    public static final String PROJ_IN = "a.gen.code.proj_in";
    public static final String EMBD = "a.gen.code.embd";
    public static final String HEAD = "a.gen.code.head";
    public static final String OUT_EMBD = "a.gen.code.out_embd";
    public static final String OUTPUT_NORM = "a.gen.code.output_norm";

    /** The NEOX RoPE base llama.cpp hardcodes for the predictor (clip.cpp:2191). */
    public static final float ROPE_THETA = 1000000.0f;

    /** Layer tensor suffixes (gguf names {@code a.gen.code.blk.<il>.<part>}). */
    static final String ATTN_NORM = "ln1.weight";
    static final String ATTN_Q = "attn_q.weight";
    static final String ATTN_Q_NORM = "attn_q_norm.weight";
    static final String ATTN_K = "attn_k.weight";
    static final String ATTN_K_NORM = "attn_k_norm.weight";
    static final String ATTN_V = "attn_v.weight";
    static final String ATTN_OUT = "attn_out.weight";
    static final String FFN_NORM = "ln2.weight";
    static final String FFN_GATE = "ffn_gate.weight";
    static final String FFN_UP = "ffn_up.weight";
    static final String FFN_DOWN = "ffn_down.weight";

    /**
     * What a code predictor session is to the speech pipeline: one frame's codes and feedback row from the talker
     * hidden state and the backbone's codec-0 code. Implemented by the host {@link Session} and by the device
     * executor of a GPU engine.
     */
    public interface Engine extends AutoCloseable {
        /** {@link Session#process(float[], int, Random)}. */
        Frame process(float[] hiddenState, int code0, Random rng);

        /** Releases any per-session device state; the host session holds none. */
        @Override
        default void close() {
        }
    }

    /** Resolved tensors of one transformer layer. */
    private static final class Layer {
        final float[] attnNorm;
        final CpuTensorRef q;
        final float[] qNorm;
        final CpuTensorRef k;
        final float[] kNorm;
        final CpuTensorRef v;
        final CpuTensorRef out;
        final float[] ffnNorm;
        final CpuTensorRef gate;
        final CpuTensorRef up;
        final CpuTensorRef down;

        Layer(float[] attnNorm, CpuTensorRef q, float[] qNorm, CpuTensorRef k, float[] kNorm, CpuTensorRef v,
              CpuTensorRef out, float[] ffnNorm, CpuTensorRef gate, CpuTensorRef up, CpuTensorRef down) {
            this.attnNorm = attnNorm;
            this.q = q;
            this.qNorm = qNorm;
            this.k = k;
            this.kNorm = kNorm;
            this.v = v;
            this.out = out;
            this.ffnNorm = ffnNorm;
            this.gate = gate;
            this.up = up;
            this.down = down;
        }
    }

    private final int nEmbd;
    private final int talkerWidth;
    private final int nLayer;
    private final int nHead;
    private final int nHeadKv;
    private final int dHead;
    private final int nFf;
    private final int nCodebooks;
    private final int vocab;
    private final int vocab0;
    private final float eps;
    private final CpuMatVec matVec;
    private final CpuTensorRef projIn;
    private final float[] projInBias;
    private final CpuTensorRef embd;
    private final CpuTensorRef head;
    private final CpuTensorRef outEmbd;
    private final float[] outputNorm;
    private final Layer[] layers;
    private final CpuWeights weights;

    private Qwen3TtsCodePredictor(int nEmbd, int talkerWidth, int nLayer, int nHead, int nHeadKv, int dHead, int nFf,
                                  int nCodebooks, int vocab, int vocab0, float eps, CpuWeights weights,
                                  CpuTensorRef projIn, float[] projInBias, CpuTensorRef embd, CpuTensorRef head,
                                  CpuTensorRef outEmbd, float[] outputNorm, Layer[] layers) {
        this.nEmbd = nEmbd;
        this.talkerWidth = talkerWidth;
        this.nLayer = nLayer;
        this.nHead = nHead;
        this.nHeadKv = nHeadKv;
        this.dHead = dHead;
        this.nFf = nFf;
        this.nCodebooks = nCodebooks;
        this.vocab = vocab;
        this.vocab0 = vocab0;
        this.eps = eps;
        this.weights = weights;
        this.matVec = weights.matVec();
        this.projIn = projIn;
        this.projInBias = projInBias;
        this.embd = embd;
        this.head = head;
        this.outEmbd = outEmbd;
        this.outputNorm = outputNorm;
        this.layers = layers;
    }

    /**
     * Resolves the predictor of an opened clip generator file.
     *
     * @throws IllegalArgumentException when the file is not a qwen3tts code predictor
     * @throws GgufFormatException      when a key or a tensor shape differs from the loader's
     */
    public static Qwen3TtsCodePredictor open(GgufKv kv, TensorCatalog tc, CpuWeights weights) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(weights, "weights");
        String type = kv.str(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, null);
        if (!PROJECTOR_TYPE.equals(type)) {
            throw new IllegalArgumentException(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE + " is "
                    + (type == null ? "absent" : "'" + type + "'") + ", a qwen3tts code predictor declares '"
                    + PROJECTOR_TYPE + "'");
        }
        int nEmbd = positive(kv, GgufKeys.CLIP_GEN_AUDIO_EMBEDDING_LENGTH, "embedding_length");
        int nLayer = positive(kv, GgufKeys.CLIP_GEN_AUDIO_BLOCK_COUNT, "block_count");
        int nHead = positive(kv, GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT, "attention.head_count");
        int nHeadKv = kv.u32(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT_KV, nHead);
        int nFf = positive(kv, GgufKeys.CLIP_GEN_AUDIO_FEED_FORWARD_LENGTH, "feed_forward_length");
        float eps = kv.f32Required(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_LAYER_NORM_EPSILON);
        if (nEmbd % nHead != 0 || nHeadKv <= 0 || nHead % nHeadKv != 0) {
            throw new GgufFormatException("qwen3tts code predictor: heads " + nHead + "/" + nHeadKv
                    + " must divide the embedding width " + nEmbd + " and each other");
        }
        int dHead = nEmbd / nHead;
        if (dHead % 2 != 0) {
            throw new GgufFormatException("qwen3tts code predictor: head dim " + dHead + " must be even for RoPE");
        }
        if (!(eps > 0f) || Float.isInfinite(eps)) {
            throw new GgufFormatException("qwen3tts code predictor: layer norm epsilon must be positive, got " + eps);
        }

        TensorRef embdRef = tc.require(EMBD + ".weight");
        TensorRef headRef = tc.require(HEAD + ".weight");
        TensorRef outEmbdRef = tc.require(OUT_EMBD + ".weight");
        int nCodebooks = dim(embdRef, 2);
        if (nCodebooks <= 0 || dim(headRef, 2) != nCodebooks) {
            throw new GgufFormatException("qwen3tts code predictor: the codebook tables disagree on the codebook"
                    + " count (" + nCodebooks + " embeddings, " + dim(headRef, 2) + " heads)");
        }
        int vocab = dim(embdRef, 1);
        int vocab0 = dim(outEmbdRef, 1);
        int talkerWidth = dim(embdRef, 0);
        shape(outEmbdRef, talkerWidth, vocab0);
        shape(headRef, nEmbd, vocab, nCodebooks);
        int projWidth = -1;
        CpuTensorRef projIn = weights.getOrNull(PROJ_IN + ".weight");
        float[] projBias = weights.floatsOrNull(PROJ_IN + ".bias");
        if (projIn != null) {
            checkShape(projIn, talkerWidth, nEmbd);
            if (projBias == null || projBias.length != nEmbd) {
                throw new GgufFormatException("qwen3tts code predictor: " + PROJ_IN + ".bias must hold " + nEmbd
                        + " floats");
            }
            projWidth = nEmbd;
        } else if (talkerWidth != nEmbd) {
            throw new GgufFormatException("qwen3tts code predictor: no " + PROJ_IN + ".weight maps the "
                    + talkerWidth + "-wide talker state onto the " + nEmbd + "-wide predictor");
        }
        if (projBias != null && projIn == null) {
            throw new GgufFormatException("qwen3tts code predictor: " + PROJ_IN + ".bias without .weight");
        }
        float[] outputNorm = weights.floats(OUTPUT_NORM + ".weight");
        if (outputNorm.length != nEmbd) {
            throw new GgufFormatException("qwen3tts code predictor: " + OUTPUT_NORM + ".weight holds "
                    + outputNorm.length + " floats, expected " + nEmbd);
        }
        Layer[] layers = new Layer[nLayer];
        for (int il = 0; il < nLayer; il++) {
            layers[il] = layer(weights, il, nEmbd, nFf, nHead, nHeadKv, dHead);
        }
        return new Qwen3TtsCodePredictor(nEmbd, talkerWidth, nLayer, nHead, nHeadKv, dHead, nFf, nCodebooks, vocab,
                vocab0, eps, weights, projIn, projBias, weights.get(EMBD + ".weight"), weights.get(HEAD + ".weight"),
                weights.get(OUT_EMBD + ".weight"), outputNorm, layers);
    }

    private static Layer layer(CpuWeights weights, int il, int nEmbd, int nFf, int nHead, int nHeadKv, int dHead) {
        String prefix = "a.gen.code.blk." + il + ".";
        float[] attnNorm = weights.floats(prefix + ATTN_NORM);
        CpuTensorRef q = weights.get(prefix + ATTN_Q);
        float[] qNorm = weights.floats(prefix + ATTN_Q_NORM);
        CpuTensorRef k = weights.get(prefix + ATTN_K);
        float[] kNorm = weights.floats(prefix + ATTN_K_NORM);
        CpuTensorRef v = weights.get(prefix + ATTN_V);
        CpuTensorRef out = weights.get(prefix + ATTN_OUT);
        float[] ffnNorm = weights.floats(prefix + FFN_NORM);
        CpuTensorRef gate = weights.get(prefix + FFN_GATE);
        CpuTensorRef up = weights.get(prefix + FFN_UP);
        CpuTensorRef down = weights.get(prefix + FFN_DOWN);
        vector(attnNorm, nEmbd, prefix + ATTN_NORM);
        checkShape(q, nEmbd, nHead * dHead);
        vector(qNorm, dHead, prefix + ATTN_Q_NORM);
        checkShape(k, nEmbd, nHeadKv * dHead);
        vector(kNorm, dHead, prefix + ATTN_K_NORM);
        checkShape(v, nEmbd, nHeadKv * dHead);
        checkShape(out, nHead * dHead, nEmbd);
        vector(ffnNorm, nEmbd, prefix + FFN_NORM);
        checkShape(gate, nEmbd, nFf);
        checkShape(up, nEmbd, nFf);
        checkShape(down, nFf, nEmbd);
        return new Layer(attnNorm, q, qNorm, k, kNorm, v, out, ffnNorm, gate, up, down);
    }

    /** Predictor width ({@code clip.gen.audio.embedding_length}). */
    public int nEmbd() {
        return nEmbd;
    }

    /** Talker hidden width, the input of the projection and the codebook tables' row width. */
    public int talkerWidth() {
        return talkerWidth;
    }

    /** Predictor block count. */
    public int nLayer() {
        return nLayer;
    }

    /** Codebook pages (15 in the reference). */
    public int nCodebooks() {
        return nCodebooks;
    }

    /** Codebook vocabulary rows of the predictor tables. */
    public int vocab() {
        return vocab;
    }

    /** Codec-0 vocabulary rows of the feedback table ({@code a.gen.code.out_embd}). */
    public int vocab0() {
        return vocab0;
    }

    /** Codes per frame: codec 0 plus every acoustic codebook. */
    public int codesPerFrame() {
        return nCodebooks + 1;
    }

    /** Attention heads per query. */
    int nHead() {
        return nHead;
    }

    /** Attention key/value heads. */
    int nHeadKv() {
        return nHeadKv;
    }

    /** Head width of the predictor attention. */
    int dHead() {
        return dHead;
    }

    /** FFN hidden width. */
    int nFf() {
        return nFf;
    }

    /** Layer norm epsilon. */
    float eps() {
        return eps;
    }

    /** The attention-norm weights of layer {@code il}. */
    float[] attnNorm(int il) {
        return layers[il].attnNorm;
    }

    /** The per-head query-norm weights of layer {@code il}. */
    float[] qNorm(int il) {
        return layers[il].qNorm;
    }

    /** The per-head key-norm weights of layer {@code il}. */
    float[] kNorm(int il) {
        return layers[il].kNorm;
    }

    /** The FFN-norm weights of layer {@code il}. */
    float[] ffnNorm(int il) {
        return layers[il].ffnNorm;
    }

    /** The output-norm weights. */
    float[] outputNormWeights() {
        return outputNorm;
    }

    /** The projection bias, or null when the predictor takes the talker width directly. */
    float[] projInBias() {
        return projInBias;
    }

    /** The {@code a.gen.code.blk.<il>.<part>} tensor name. */
    static String layerTensor(int il, String part) {
        return "a.gen.code.blk." + il + "." + part;
    }

    /** Whether the predictor projects the talker width onto its own width. */
    boolean hasProjIn() {
        return projIn != null;
    }

    /** A fresh per-frame key/value cache over {@code topK}/{@code topP} truncated sampling; not thread-safe. */
    public Session newSession(int topK, float topP) {
        return new Session(topK, topP);
    }

    /**
     * A session with the reference's default sampling ({@value SpeechOptions#DEFAULT_TOP_K} /
     * {@value SpeechOptions#DEFAULT_TOP_P}); not thread-safe.
     */
    public Session newSession() {
        return new Session(SpeechOptions.DEFAULT_TOP_K, SpeechOptions.DEFAULT_TOP_P);
    }

    /**
     * One frame's code predictor run over the 16-row key/value cache of the reference graph.
     *
     * <p>The cache is allocated once and every position overwrites its row, so the same session serves every frame
     * of a synthesis; the rows beyond the current position are never read.
     */
    public final class Session implements Engine {
        private final float[][] keys = new float[nLayer][];
        private final float[][] values = new float[nLayer][];
        private final int kvCh = dHead * nHeadKv;
        private final int nKv = codesPerFrame();
        private final float[] work = new float[nEmbd];
        private final float[] attn = new float[nHead * dHead];
        private final float[] scratch = new float[Math.max(kvCh, nHead * dHead)];
        private final float[] scores = new float[nKv];
        private final int topK;
        private final float topP;

        private Session(int topK, float topP) {
            if (topK < 0) {
                throw new IllegalArgumentException("top-k must not be negative, got " + topK);
            }
            if (!(topP > 0f) || topP > 1f || Float.isNaN(topP)) {
                throw new IllegalArgumentException("top-p must be in (0, 1], got " + topP);
            }
            this.topK = topK;
            this.topP = topP;
            for (int il = 0; il < nLayer; il++) {
                keys[il] = new float[kvCh * nKv];
                values[il] = new float[kvCh * nKv];
            }
        }

        /**
         * One frame: the backbone hidden state and its sampled codec-0 code become the 16 codes and the feedback
         * row for the backbone.
         *
         * @param hiddenState the talker's final hidden state ({@code talkerWidth()} floats)
         * @param code0       the backbone's codec-0 index in {@code [0, vocab0())}
         * @param rng         the draw source, one uniform per acoustic codebook
         * @throws IllegalArgumentException on a wrong hidden width or an out-of-range code
         */
        public Frame process(float[] hiddenState, int code0, Random rng) {
            Objects.requireNonNull(hiddenState, "hiddenState");
            Objects.requireNonNull(rng, "rng");
            if (hiddenState.length != talkerWidth) {
                throw new IllegalArgumentException("the talker hidden state is " + hiddenState.length + " wide, the"
                        + " code predictor takes " + talkerWidth);
            }
            if (code0 < 0 || code0 >= vocab0) {
                throw new IllegalArgumentException("codec-0 code " + code0 + " is outside [0, " + vocab0 + ")");
            }
            int[] codes = new int[nCodes()];
            codes[0] = code0;
            step(project(hiddenState), 0);
            // position 1: the codec-0 embedding, sampled with head page 0
            step(project(embed(outEmbd, code0)), 1);
            codes[1] = samplePage(0, rng);
            // steps 1..nCodebooks-1: embed the previous code with that step's codebook, sample its page
            for (int g = 1; g < nCodebooks; g++) {
                step(project(embed(embd, (long) (g - 1) * vocab + codes[g])), g + 1);
                codes[g + 1] = samplePage(g, rng);
            }
            return new Frame(codes, feedback(codes));
        }

        /** The step's LM head page over the normalized row in {@link #work}, sampled by the graph's draw. */
        private int samplePage(int page, Random rng) {
            float[] logits = new float[vocab];
            matVec.matvecExpert(head, page, nCodebooks, work, 0, logits, 0);
            return SpeechSampling.draw(logits, topK, topP, rng.nextFloat());
        }

        /** The feedback row: codec 0 plus every acoustic codebook embedding. */
        private float[] feedback(int[] codes) {
            float[] sum = embed(outEmbd, codes[0]);
            for (int g = 0; g < nCodebooks; g++) {
                addEmbedding(embd, (long) g * vocab + codes[g + 1], sum);
            }
            return sum;
        }

        private float[] project(float[] x) {
            if (projIn == null) {
                return x.clone();
            }
            float[] y = new float[nEmbd];
            matVec.matmul(projIn, x, 0, talkerWidth, y, 0, nEmbd, 1);
            CpuOps.addRowBias(y, 0, 1, nEmbd, projInBias, 0);
            return y;
        }

        private float[] embed(CpuTensorRef table, long row) {
            float[] out = new float[(int) table.cols()];
            weights.embedRow(table, row, out, 0);
            return out;
        }

        private void addEmbedding(CpuTensorRef table, long row, float[] dst) {
            float[] rowValues = new float[(int) table.cols()];
            weights.embedRow(table, row, rowValues, 0);
            for (int i = 0; i < rowValues.length; i++) {
                dst[i] += rowValues[i];
            }
        }

        /** Runs one position through every layer, writing its key/value cache row, leaving the normed row in work. */
        private void step(float[] x, int pos) {
            float[] cur = x;
            float[] residual = new float[nEmbd];
            float[] projected = new float[nEmbd];
            for (int il = 0; il < nLayer; il++) {
                Layer l = layers[il];
                System.arraycopy(cur, 0, residual, 0, nEmbd);
                rmsNorm(cur, l.attnNorm, work);
                float[] q = new float[nHead * dHead];
                float[] k = new float[kvCh];
                float[] v = new float[kvCh];
                matVec.matmul(l.q, work, 0, nEmbd, q, 0, nHead * dHead, 1);
                matVec.matmul(l.k, work, 0, nEmbd, k, 0, kvCh, 1);
                matVec.matmul(l.v, work, 0, nEmbd, v, 0, kvCh, 1);
                for (int h = 0; h < nHead; h++) {
                    int headOff = h * dHead;
                    rmsNormRange(q, headOff, l.qNorm, scratch, 0, dHead);
                    rope(scratch, 0, dHead, pos);
                    System.arraycopy(scratch, 0, q, headOff, dHead);
                }
                for (int h = 0; h < nHeadKv; h++) {
                    int headOff = h * dHead;
                    rmsNormRange(k, headOff, l.kNorm, scratch, 0, dHead);
                    rope(scratch, 0, dHead, pos);
                    System.arraycopy(scratch, 0, k, headOff, dHead);
                    System.arraycopy(k, headOff, keys[il], pos * kvCh + headOff, dHead);
                    System.arraycopy(v, headOff, values[il], pos * kvCh + headOff, dHead);
                }
                attention(q, keys[il], values[il], pos, kvCh, scores, attn);
                matVec.matmul(l.out, attn, 0, nHead * dHead, projected, 0, nEmbd, 1);
                CpuOps.add(residual, 0, projected, 0, cur, 0, nEmbd);

                System.arraycopy(cur, 0, residual, 0, nEmbd);
                rmsNorm(cur, l.ffnNorm, work);
                float[] gate = new float[nFf];
                float[] up = new float[nFf];
                matVec.matmul(l.gate, work, 0, nEmbd, gate, 0, nFf, 1);
                matVec.matmul(l.up, work, 0, nEmbd, up, 0, nFf, 1);
                for (int i = 0; i < nFf; i++) {
                    gate[i] = silu(gate[i]) * up[i];
                }
                matVec.matmul(l.down, gate, 0, nFf, projected, 0, nEmbd, 1);
                CpuOps.add(residual, 0, projected, 0, cur, 0, nEmbd);
            }
            rmsNorm(cur, outputNorm, work);
        }
    }

    /** The three parts of one frame: its 16 codes and the feedback row. */
    public static final class Frame {
        private final int[] codes;
        private final float[] feedback;

        Frame(int[] codes, float[] feedback) {
            this.codes = codes;
            this.feedback = feedback;
        }

        /** This frame's codes: codec 0 then the 15 acoustic codebooks (a copy). */
        public int[] codes() {
            return codes.clone();
        }

        /** The talker-width embedding sum the backbone adds to its {@code <tts_pad>} overlay (a copy). */
        public float[] feedback() {
            return feedback.clone();
        }
    }

    // ------------------------------------------------------------------ math

    private void rmsNorm(float[] x, float[] weight, float[] out) {
        double sum = 0.0;
        for (int i = 0; i < nEmbd; i++) {
            sum += (double) x[i] * x[i];
        }
        float scale = (float) (1.0 / Math.sqrt(sum / nEmbd + eps));
        for (int i = 0; i < nEmbd; i++) {
            out[i] = x[i] * scale * weight[i];
        }
    }

    private void rmsNormRange(float[] x, int off, float[] weight, float[] out, int outOff, int width) {
        double sum = 0.0;
        for (int i = 0; i < width; i++) {
            sum += (double) x[off + i] * x[off + i];
        }
        float scale = (float) (1.0 / Math.sqrt(sum / width + eps));
        for (int i = 0; i < width; i++) {
            out[outOff + i] = x[off + i] * scale * weight[i];
        }
    }

    /** NEOX RoPE of one head: pairs {@code (i, i + dHead/2)} rotated by {@code pos * theta^(-2i/dHead)}. */
    private void rope(float[] x, int off, int width, int pos) {
        int half = width / 2;
        for (int i = 0; i < half; i++) {
            double angle = pos * Math.pow(ROPE_THETA, -2.0 * i / width);
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            float x0 = x[off + i];
            float x1 = x[off + i + half];
            x[off + i] = x0 * cos - x1 * sin;
            x[off + i + half] = x0 * sin + x1 * cos;
        }
    }

    /** Causal scaled dot-product attention over the cache rows {@code 0..pos}. */
    private void attention(float[] q, float[] keys, float[] values, int pos, int kvCh, float[] scores,
                           float[] out) {
        int headsPerKv = nHead / nHeadKv;
        float scale = (float) (1.0 / Math.sqrt(dHead));
        for (int h = 0; h < nHead; h++) {
            int kvHead = h / headsPerKv;
            int qOff = h * dHead;
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j <= pos; j++) {
                float dot = 0f;
                int kOff = j * kvCh + kvHead * dHead;
                for (int d = 0; d < dHead; d++) {
                    dot += q[qOff + d] * keys[kOff + d];
                }
                scores[j] = dot * scale;
                if (scores[j] > max) {
                    max = scores[j];
                }
            }
            double sum = 0.0;
            for (int j = 0; j <= pos; j++) {
                double e = Math.exp(scores[j] - max);
                scores[j] = (float) e;
                sum += e;
            }
            float inv = (float) (1.0 / sum);
            for (int d = 0; d < dHead; d++) {
                float acc = 0f;
                for (int j = 0; j <= pos; j++) {
                    acc += scores[j] * values[j * kvCh + kvHead * dHead + d];
                }
                out[qOff + d] = acc * inv;
            }
        }
    }

    private static float silu(float x) {
        return (float) (x / (1.0 + Math.exp(-x)));
    }

    // ------------------------------------------------------------------ validation

    private int nCodes() {
        return nCodebooks + 1;
    }

    private static int positive(GgufKv kv, String key, String what) {
        int v = kv.u32Required(key);
        if (v <= 0) {
            throw new GgufFormatException("qwen3tts code predictor: " + what + " must be positive, got " + v);
        }
        return v;
    }

    private static int dim(TensorRef ref, int i) {
        long v = ref.ne(i);
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new GgufFormatException("qwen3tts code predictor: tensor " + ref.name() + " dimension " + i
                    + " is " + v);
        }
        return (int) v;
    }

    private static void shape(TensorRef ref, long... ne) {
        boolean ok = ref.rank() >= ne.length;
        for (int i = 0; i < ne.length && ok; i++) {
            ok = ref.ne(i) == ne[i];
        }
        for (int i = ne.length; i < ref.rank() && ok; i++) {
            ok = ref.ne(i) == 1;
        }
        if (!ok) {
            throw new GgufFormatException("qwen3tts code predictor: tensor " + ref.name() + " has shape " + ref
                    + ", expected " + java.util.Arrays.toString(ne) + " (ggml order)");
        }
    }

    private static void checkShape(CpuTensorRef ref, long cols, long rows) {
        if (ref.cols() != cols || ref.rows() != rows) {
            throw new GgufFormatException("qwen3tts code predictor: tensor " + ref + " is " + ref.rows() + " x "
                    + ref.cols() + ", expected " + rows + " x " + cols);
        }
    }

    private static void vector(float[] v, int n, String name) {
        if (v == null || v.length != n) {
            throw new GgufFormatException("qwen3tts code predictor: " + name + " must hold " + n + " floats, got "
                    + (v == null ? "none" : v.length));
        }
    }

    @Override
    public String toString() {
        return "Qwen3TtsCodePredictor{embd=" + nEmbd + ", talker=" + talkerWidth + ", layers=" + nLayer
                + ", heads=" + nHead + "/" + nHeadKv + ", codebooks=" + nCodebooks + ", vocab=" + vocab + '}';
    }
}
