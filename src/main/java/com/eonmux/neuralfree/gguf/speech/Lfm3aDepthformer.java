package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * The lfm3a MaskGIT depthformer of llama.cpp's {@code clip_graph_lfm3a_gen::build_gen_code}
 * (tools/mtmd/models/lfm3a-gen.cpp:113-170): one pass turns the backbone's hidden state and the frame's known
 * codes into logits for every codebook at once, plus the feedback embedding the backbone consumes next.
 *
 * <ol>
 *   <li>every codebook gets one position: its own slice of the projected hidden state (the projection emits one
 *       {@code n_embd} row per codebook) plus the code's embedding when the codebook is revealed, else the mask
 *       embedding;</li>
 *   <li>the positions run a bidirectional LFM2 stack (no mask, no cache): RMS norms, per-head Q/K RMS norms,
 *       full-width NEOX RoPE at the codebook index, the per-head sigmoid attention gate and a SwiGLU FFN, with
 *       the LFM3 sandwich norms when a layer carries them;</li>
 *   <li>the output RMS norm and the per-codebook head norm (each position scaled by its codebook's norm page)
 *       feed the embedding table tied to the output head, one page per codebook;</li>
 *   <li>the feedback row sums the output-embedding rows of the frame's codes, one row per codebook.</li>
 * </ol>
 *
 * <p>Tensors use the clip generator names of tools/mtmd/clip-impl.h (ggml dimension order, {@code ne[0]} first;
 * {@code a.gen.code.mask_embd} carries no {@code .weight} suffix, as the converter writes it). Immutable geometry
 * over shared host weights; the graph is stateless, so one instance serves every frame and every caller.
 */
public final class Lfm3aDepthformer {

    /** The {@code clip.gen.audio.projector_type} of the depthformer. */
    public static final String PROJECTOR_TYPE = "lfm3a_gen";

    /** Codebooks per frame, fixed by the Liquid codec (llama.cpp's {@code n_cb}). */
    public static final int N_CODEBOOKS = 12;

    /**
     * What the speech pipeline runs through: one MaskGIT depthformer pass. Implemented by the stateless host graph
     * itself and by the device executor of a GPU engine.
     */
    public interface Engine extends AutoCloseable {
        /** {@link Lfm3aDepthformer#pass(float[], int[])}. */
        Frame pass(float[] hiddenState, int[] codes);

        /** Releases any per-engine device state; the host graph holds none. */
        @Override
        default void close() {
        }
    }

    /** The NEOX RoPE base llama.cpp hardcodes for the depthformer (clip.cpp:2165). */
    public static final float ROPE_THETA = 1000000.0f;

    /** Tensor base names of tools/mtmd/clip-impl.h:248-256. */
    public static final String PROJ_IN = "a.gen.code.proj_in";
    public static final String EMBD = "a.gen.code.embd";
    public static final String OUT_EMBD = "a.gen.code.out_embd";
    public static final String OUTPUT_NORM = "a.gen.code.output_norm";
    public static final String HEAD_NORM = "a.gen.code.head_norm";
    public static final String MASK_EMBD = "a.gen.code.mask_embd";

    /** Layer tensor suffixes ({@code a.gen.code.blk.<il>.<part>}). */
    static final String LN1 = "ln1.weight";
    static final String ATTN_Q = "attn_q.weight";
    static final String ATTN_Q_NORM = "attn_q_norm.weight";
    static final String ATTN_K = "attn_k.weight";
    static final String ATTN_K_NORM = "attn_k_norm.weight";
    static final String ATTN_V = "attn_v.weight";
    static final String ATTN_OUT = "attn_out.weight";
    static final String ATTN_GATE = "attn_gate.weight";
    static final String ATTN_POST_NORM = "attn_post_norm.weight";
    static final String LN2 = "ln2.weight";
    static final String FFN_GATE = "ffn_gate.weight";
    static final String FFN_UP = "ffn_up.weight";
    static final String FFN_DOWN = "ffn_down.weight";
    static final String FFN_POST_NORM = "ffn_post_norm.weight";

    /** Resolved tensors of one depthformer layer. */
    static final class Layer {
        final float[] ln1;
        final CpuTensorRef q;
        final float[] qNorm;
        final CpuTensorRef k;
        final float[] kNorm;
        final CpuTensorRef v;
        final CpuTensorRef out;
        final CpuTensorRef attnGate;
        final float[] attnPostNorm;
        final float[] ln2;
        final CpuTensorRef gate;
        final CpuTensorRef up;
        final CpuTensorRef down;
        final float[] ffPostNorm;

        Layer(float[] ln1, CpuTensorRef q, float[] qNorm, CpuTensorRef k, float[] kNorm, CpuTensorRef v,
              CpuTensorRef out, CpuTensorRef attnGate, float[] attnPostNorm, float[] ln2, CpuTensorRef gate,
              CpuTensorRef up, CpuTensorRef down, float[] ffPostNorm) {
            this.ln1 = ln1;
            this.q = q;
            this.qNorm = qNorm;
            this.k = k;
            this.kNorm = kNorm;
            this.v = v;
            this.out = out;
            this.attnGate = attnGate;
            this.attnPostNorm = attnPostNorm;
            this.ln2 = ln2;
            this.gate = gate;
            this.up = up;
            this.down = down;
            this.ffPostNorm = ffPostNorm;
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
    private final float eps;
    private final CpuMatVec matVec;
    private final CpuWeights weights;
    private final CpuTensorRef projIn;
    private final float[] projInBias;
    private final CpuTensorRef embd;
    private final CpuTensorRef outEmbd;
    private final float[] outputNorm;
    private final float[] headNorm;
    private final float[] maskEmbd;
    private final Layer[] layers;

    private Lfm3aDepthformer(int nEmbd, int talkerWidth, int nLayer, int nHead, int nHeadKv, int dHead, int nFf,
                             int nCodebooks, int vocab, float eps, CpuWeights weights, CpuTensorRef projIn,
                             float[] projInBias, CpuTensorRef embd, CpuTensorRef outEmbd, float[] outputNorm,
                             float[] headNorm, float[] maskEmbd, Layer[] layers) {
        this.nEmbd = nEmbd;
        this.talkerWidth = talkerWidth;
        this.nLayer = nLayer;
        this.nHead = nHead;
        this.nHeadKv = nHeadKv;
        this.dHead = dHead;
        this.nFf = nFf;
        this.nCodebooks = nCodebooks;
        this.vocab = vocab;
        this.eps = eps;
        this.weights = weights;
        this.matVec = weights.matVec();
        this.projIn = projIn;
        this.projInBias = projInBias;
        this.embd = embd;
        this.outEmbd = outEmbd;
        this.outputNorm = outputNorm;
        this.headNorm = headNorm;
        this.maskEmbd = maskEmbd;
        this.layers = layers;
    }

    /**
     * Resolves the depthformer of an opened clip generator file.
     *
     * @throws IllegalArgumentException when the file is not an lfm3a depthformer
     * @throws GgufFormatException      when a key or a tensor shape differs from the loader's
     */
    public static Lfm3aDepthformer open(GgufKv kv, TensorCatalog tc, CpuWeights weights) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(weights, "weights");
        String type = kv.str(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, null);
        if (!PROJECTOR_TYPE.equals(type)) {
            throw new IllegalArgumentException(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE + " is "
                    + (type == null ? "absent" : "'" + type + "'") + ", an lfm3a depthformer declares '"
                    + PROJECTOR_TYPE + "'");
        }
        int nEmbd = positive(kv, GgufKeys.CLIP_GEN_AUDIO_EMBEDDING_LENGTH, "embedding_length");
        int nLayer = positive(kv, GgufKeys.CLIP_GEN_AUDIO_BLOCK_COUNT, "block_count");
        int nHead = positive(kv, GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT, "attention.head_count");
        int nHeadKv = kv.u32(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT_KV, nHead);
        int nFf = positive(kv, GgufKeys.CLIP_GEN_AUDIO_FEED_FORWARD_LENGTH, "feed_forward_length");
        float eps = kv.f32Required(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_LAYER_NORM_EPSILON);
        if (nEmbd % nHead != 0 || nHeadKv <= 0 || nHead % nHeadKv != 0) {
            throw new GgufFormatException("lfm3a depthformer: heads " + nHead + "/" + nHeadKv
                    + " must divide the embedding width " + nEmbd + " and each other");
        }
        int dHead = nEmbd / nHead;
        if (!(eps > 0f) || Float.isInfinite(eps)) {
            throw new GgufFormatException("lfm3a depthformer: layer norm epsilon must be positive, got " + eps);
        }

        TensorRef embdRef = tc.require(EMBD + ".weight");
        TensorRef outEmbdRef = tc.require(OUT_EMBD + ".weight");
        int nCodebooks = dim(embdRef, 2);
        int vocab = dim(embdRef, 1);
        shape(embdRef, nEmbd, vocab, nCodebooks);
        if (nCodebooks <= 0) {
            throw new GgufFormatException("lfm3a depthformer: " + EMBD + ".weight holds no codebook page");
        }
        int talkerWidth = dim(outEmbdRef, 0);
        // the merged per-codebook table: row code + codebook * vocab, as the converter writes it (lfm3.py:550)
        shape(outEmbdRef, talkerWidth, (long) vocab * nCodebooks);
        CpuTensorRef projIn = weights.get(PROJ_IN + ".weight");
        float[] projInBias = weights.floats(PROJ_IN + ".bias");
        // one projection row per codebook position: [n_mmproj_embd, n_embd * n_codebooks] (lfm3a-gen.cpp:141-143)
        checkShape(projIn, talkerWidth, (long) nEmbd * nCodebooks);
        vector(projInBias, PROJ_IN + ".bias", nEmbd * nCodebooks);
        float[] outputNorm = weights.floats(OUTPUT_NORM + ".weight");
        vector(outputNorm, OUTPUT_NORM + ".weight", nEmbd);
        TensorRef headNormRef = tc.require(HEAD_NORM + ".weight");
        headNormShape(headNormRef, nEmbd, nCodebooks);
        float[] headNorm = weights.floats(HEAD_NORM + ".weight");
        float[] maskEmbd = weights.floats(MASK_EMBD);
        vector(maskEmbd, MASK_EMBD, nEmbd);

        Layer[] layers = new Layer[nLayer];
        for (int il = 0; il < nLayer; il++) {
            layers[il] = layer(weights, il, nEmbd, nFf, nHead, nHeadKv, dHead);
        }
        return new Lfm3aDepthformer(nEmbd, talkerWidth, nLayer, nHead, nHeadKv, dHead, nFf, nCodebooks, vocab,
                eps, weights, projIn, projInBias, weights.get(EMBD + ".weight"), weights.get(OUT_EMBD + ".weight"),
                outputNorm, headNorm, maskEmbd, layers);
    }

    private static Layer layer(CpuWeights weights, int il, int nEmbd, int nFf, int nHead, int nHeadKv, int dHead) {
        String prefix = "a.gen.code.blk." + il + ".";
        float[] ln1 = weights.floats(prefix + LN1);
        CpuTensorRef q = weights.get(prefix + ATTN_Q);
        float[] qNorm = weights.floats(prefix + ATTN_Q_NORM);
        CpuTensorRef k = weights.get(prefix + ATTN_K);
        float[] kNorm = weights.floats(prefix + ATTN_K_NORM);
        CpuTensorRef v = weights.get(prefix + ATTN_V);
        CpuTensorRef out = weights.get(prefix + ATTN_OUT);
        CpuTensorRef attnGate = weights.get(prefix + ATTN_GATE);
        float[] attnPostNorm = weights.floatsOrNull(prefix + ATTN_POST_NORM);
        float[] ln2 = weights.floats(prefix + LN2);
        CpuTensorRef gate = weights.get(prefix + FFN_GATE);
        CpuTensorRef up = weights.get(prefix + FFN_UP);
        CpuTensorRef down = weights.get(prefix + FFN_DOWN);
        float[] ffPostNorm = weights.floatsOrNull(prefix + FFN_POST_NORM);
        vector(ln1, prefix + LN1, nEmbd);
        checkShape(q, nEmbd, nHead * dHead);
        vector(qNorm, prefix + ATTN_Q_NORM, dHead);
        checkShape(k, nEmbd, nHeadKv * dHead);
        vector(kNorm, prefix + ATTN_K_NORM, dHead);
        checkShape(v, nEmbd, nHeadKv * dHead);
        checkShape(out, nHead * dHead, nEmbd);
        checkShape(attnGate, nEmbd, nHead);
        if (attnPostNorm != null) {
            vector(attnPostNorm, prefix + ATTN_POST_NORM, nEmbd);
        }
        vector(ln2, prefix + LN2, nEmbd);
        checkShape(gate, nEmbd, nFf);
        checkShape(up, nEmbd, nFf);
        checkShape(down, nFf, nEmbd);
        if (ffPostNorm != null) {
            vector(ffPostNorm, prefix + FFN_POST_NORM, nEmbd);
        }
        return new Layer(ln1, q, qNorm, k, kNorm, v, out, attnGate, attnPostNorm, ln2, gate, up, down, ffPostNorm);
    }

    // ------------------------------------------------------------------ accessors

    /** Depthformer width ({@code clip.gen.audio.embedding_length}). */
    public int nEmbd() {
        return nEmbd;
    }

    /** The hidden-state width the projection takes, i.e. the backbone's embedding width. */
    public int talkerWidth() {
        return talkerWidth;
    }

    /** Depthformer block count. */
    public int nLayer() {
        return nLayer;
    }

    /** Codebook pages ({@code a.gen.code.embd} page count, 12 in the reference). */
    public int nCodebooks() {
        return nCodebooks;
    }

    /** Codebook vocabulary rows; the last code of every codebook is the end-of-audio code. */
    public int vocab() {
        return vocab;
    }

    /** A fresh execution engine over {@code device}, or the host graph when {@code device} is null. */
    public Engine newEngine(SpeechDeviceWeights device) {
        if (device != null) {
            return new GpuLfm3aDepthformer(this, device);
        }
        return new Engine() {
            @Override
            public Frame pass(float[] hiddenState, int[] codes) {
                return Lfm3aDepthformer.this.pass(hiddenState, codes);
            }
        };
    }

    /** The GGUF name of one tensor of depthformer layer {@code index}. */
    static String layerTensor(int index, String part) {
        return "a.gen.code.blk." + index + "." + part;
    }

    CpuTensorRef projInWeight() {
        return projIn;
    }

    float[] projInBiasValues() {
        return projInBias;
    }

    CpuTensorRef embdWeight() {
        return embd;
    }

    CpuTensorRef outEmbdWeight() {
        return outEmbd;
    }

    float[] outputNormValues() {
        return outputNorm;
    }

    float[] headNormValues() {
        return headNorm;
    }

    float[] maskEmbdValues() {
        return maskEmbd;
    }

    Layer[] layerTensors() {
        return layers;
    }

    float epsValue() {
        return eps;
    }

    int nHeadValue() {
        return nHead;
    }

    int nHeadKvValue() {
        return nHeadKv;
    }

    int dHeadValue() {
        return dHead;
    }

    int nFfValue() {
        return nFf;
    }

    // ------------------------------------------------------------------ one pass

    /**
     * One MaskGIT pass: the hidden state and the frame's codes (a negative code masks its codebook) become the
     * logits of every codebook and the feedback embedding of the revealed codes.
     *
     * @param hiddenState the backbone's raw hidden state ({@code talkerWidth()} floats)
     * @param codes       one code per codebook in {@code [-1, vocab())}; {@code -1} masks the codebook
     * @return the codebook-major logits ({@code nCodebooks()} pages of {@code vocab()} each) and the feedback row
     * @throws IllegalArgumentException on a wrong hidden width, a wrong code count or an out-of-range code
     */
    public Frame pass(float[] hiddenState, int[] codes) {
        Objects.requireNonNull(hiddenState, "hiddenState");
        Objects.requireNonNull(codes, "codes");
        if (hiddenState.length != talkerWidth) {
            throw new IllegalArgumentException("the hidden state is " + hiddenState.length + " wide, the"
                    + " depthformer projection takes " + talkerWidth);
        }
        if (codes.length != nCodebooks) {
            throw new IllegalArgumentException("the frame holds " + codes.length + " codes, the depthformer runs "
                    + nCodebooks + " codebooks");
        }
        for (int p = 0; p < nCodebooks; p++) {
            if (codes[p] < -1 || codes[p] >= vocab) {
                throw new IllegalArgumentException("codebook " + p + " code " + codes[p] + " is outside [-1, "
                        + vocab + ")");
            }
        }
        float[] feedback = feedback(codes);
        if (allKnown(codes)) {
            // the reference returns before building the transformer when nothing is masked
            return new Frame(null, feedback);
        }

        // one position per codebook: the codebook's projected hidden state + code embedding or mask embedding
        float[] cur = new float[nCodebooks * nEmbd];
        float[] base = new float[nCodebooks * nEmbd];
        matVec.matmul(projIn, hiddenState, 0, talkerWidth, base, 0, nCodebooks * nEmbd, 1);
        for (int d = 0; d < base.length; d++) {
            base[d] += projInBias[d];
        }
        float[] row = new float[nEmbd];
        for (int p = 0; p < nCodebooks; p++) {
            int off = p * nEmbd;
            System.arraycopy(base, off, cur, off, nEmbd);
            if (codes[p] >= 0) {
                weights.embedRow(embd, (long) codes[p] + (long) p * vocab, row, 0);
            } else {
                System.arraycopy(maskEmbd, 0, row, 0, nEmbd);
            }
            for (int d = 0; d < nEmbd; d++) {
                cur[off + d] += row[d];
            }
        }

        // bidirectional LFM2 stack over the codebook positions; as build_lfm_layer does, each block reads the
        // normed stream and its result adds back into the stream it was normed from
        float[] residual = new float[nCodebooks * nEmbd];
        float[] projected = new float[nCodebooks * nEmbd];
        for (int il = 0; il < nLayer; il++) {
            Layer l = layers[il];
            System.arraycopy(cur, 0, residual, 0, cur.length);
            rmsNormRows(cur, l.ln1, cur);
            lfmAttention(l, cur, projected);
            if (l.attnPostNorm != null) {
                rmsNormRows(projected, l.attnPostNorm, projected);
            }
            addInto(cur, residual, projected);
            System.arraycopy(cur, 0, residual, 0, cur.length);
            rmsNormRows(cur, l.ln2, cur);
            ffn(l, cur, projected);
            if (l.ffPostNorm != null) {
                rmsNormRows(projected, l.ffPostNorm, projected);
            }
            addInto(cur, residual, projected);
        }

        // output norm, then the per-codebook head norm over each position
        rmsNormRows(cur, outputNorm, cur);
        float[] logits = new float[nCodebooks * vocab];
        for (int p = 0; p < nCodebooks; p++) {
            rmsNormRow(cur, p * nEmbd, headNorm, p * nEmbd, row, 0);
            matVec.matvecExpert(embd, p, nCodebooks, row, 0, logits, p * vocab);
        }
        return new Frame(logits, feedback);
    }

    /**
     * The feedback embedding of fully known codes: the sum over codebooks of their output-embedding rows, the
     * backbone-width row the reference returns when no codebook is masked. Row {@code p} of the merged table is
     * {@code code + p * vocab()}.
     *
     * @throws IllegalArgumentException when a code is negative or out of range
     */
    public float[] feedback(int[] codes) {
        Objects.requireNonNull(codes, "codes");
        if (codes.length != nCodebooks) {
            throw new IllegalArgumentException("the frame holds " + codes.length + " codes, the depthformer runs "
                    + nCodebooks + " codebooks");
        }
        float[] feedback = new float[talkerWidth];
        float[] row = new float[talkerWidth];
        for (int p = 0; p < nCodebooks; p++) {
            int c = codes[p];
            if (c < 0) {
                continue; // the early passes feed masked codes whose rows the graph never consumes
            }
            if (c >= vocab) {
                throw new IllegalArgumentException("codebook " + p + " code " + c + " is outside [0, " + vocab + ")");
            }
            // the merged per-codebook table indexes code + codebook * n_vocab (lfm3a-gen.cpp:123-125)
            weights.embedRow(outEmbd, (long) p * vocab + c, row, 0);
            for (int d = 0; d < talkerWidth; d++) {
                feedback[d] += row[d];
            }
        }
        return feedback;
    }

    private boolean allKnown(int[] codes) {
        for (int c : codes) {
            if (c < 0) {
                return false;
            }
        }
        return true;
    }

    // ------------------------------------------------------------------ the LFM2 operators

    /** Bidirectional GQA with Q/K RMS norms, full-width NEOX RoPE at the codebook index and the sigmoid gate. */
    private void lfmAttention(Layer l, float[] cur, float[] out) {
        int nPos = nCodebooks;
        int qWidth = nHead * dHead;
        int kvWidth = nHeadKv * dHead;
        float[] q = new float[nPos * qWidth];
        float[] k = new float[nPos * kvWidth];
        float[] v = new float[nPos * kvWidth];
        float[] scratch = new float[Math.max(qWidth, kvWidth)];
        float[] scores = new float[nPos];
        for (int p = 0; p < nPos; p++) {
            int off = p * nEmbd;
            matVec.matmul(l.q, cur, off, nEmbd, q, p * qWidth, qWidth, 1);
            matVec.matmul(l.k, cur, off, nEmbd, k, p * kvWidth, kvWidth, 1);
            matVec.matmul(l.v, cur, off, nEmbd, v, p * kvWidth, kvWidth, 1);
            for (int h = 0; h < nHead; h++) {
                rmsNormRange(q, p * qWidth + h * dHead, l.qNorm, scratch, 0, dHead);
                ropeNeox(scratch, 0, dHead, p);
                System.arraycopy(scratch, 0, q, p * qWidth + h * dHead, dHead);
            }
            for (int h = 0; h < nHeadKv; h++) {
                rmsNormRange(k, p * kvWidth + h * dHead, l.kNorm, scratch, 0, dHead);
                ropeNeox(scratch, 0, dHead, p);
                System.arraycopy(scratch, 0, k, p * kvWidth + h * dHead, dHead);
            }
        }
        int headsPerKv = nHead / nHeadKv;
        float scale = (float) (1.0 / Math.sqrt(dHead));
        float[] attn = new float[qWidth];
        float[] gate = new float[nHead];
        for (int p = 0; p < nPos; p++) {
            for (int h = 0; h < nHead; h++) {
                int kvHead = h / headsPerKv;
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < nPos; j++) {
                    float dot = 0f;
                    int kOff = j * kvWidth + kvHead * dHead;
                    for (int d = 0; d < dHead; d++) {
                        dot += q[p * qWidth + h * dHead + d] * k[kOff + d];
                    }
                    scores[j] = dot * scale;
                    if (scores[j] > max) {
                        max = scores[j];
                    }
                }
                double sum = 0.0;
                for (int j = 0; j < nPos; j++) {
                    double e = Math.exp(scores[j] - max);
                    scores[j] = (float) e;
                    sum += e;
                }
                float inv = (float) (1.0 / sum);
                for (int d = 0; d < dHead; d++) {
                    float acc = 0f;
                    for (int j = 0; j < nPos; j++) {
                        acc += scores[j] * v[j * kvWidth + kvHead * dHead + d];
                    }
                    attn[h * dHead + d] = acc * inv;
                }
            }
            // the per-head sigmoid gate reads the layer input, as the reference's build_lfm_attn does
            matVec.matmul(l.attnGate, cur, p * nEmbd, nEmbd, gate, 0, nHead, 1);
            for (int h = 0; h < nHead; h++) {
                float g = sigmoid(gate[h]);
                for (int d = 0; d < dHead; d++) {
                    attn[h * dHead + d] *= g;
                }
            }
            matVec.matmul(l.out, attn, 0, qWidth, out, p * nEmbd, nEmbd, 1);
        }
    }

    /** The SwiGLU FFN over every position. */
    private void ffn(Layer l, float[] cur, float[] out) {
        float[] gate = new float[nFf];
        float[] up = new float[nFf];
        for (int p = 0; p < nCodebooks; p++) {
            int off = p * nEmbd;
            matVec.matmul(l.gate, cur, off, nEmbd, gate, 0, nFf, 1);
            matVec.matmul(l.up, cur, off, nEmbd, up, 0, nFf, 1);
            for (int i = 0; i < nFf; i++) {
                gate[i] = silu(gate[i]) * up[i];
            }
            matVec.matmul(l.down, gate, 0, nFf, out, off, nEmbd, 1);
        }
    }

    // ------------------------------------------------------------------ math

    /** RMS norm over every {@code nCodebooks()} row of {@code x}, scaled by one shared weight. */
    private void rmsNormRows(float[] x, float[] weight, float[] out) {
        for (int p = 0; p < nCodebooks; p++) {
            rmsNormRow(x, p * nEmbd, weight, 0, out, p * nEmbd);
        }
    }

    /** RMS norm of one row: {@code out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]}. */
    private void rmsNormRow(float[] x, int xOff, float[] weight, int weightOff, float[] out, int outOff) {
        double sum = 0.0;
        for (int i = 0; i < nEmbd; i++) {
            sum += (double) x[xOff + i] * x[xOff + i];
        }
        float inv = (float) (1.0 / Math.sqrt(sum / nEmbd + eps));
        for (int i = 0; i < nEmbd; i++) {
            out[outOff + i] = x[xOff + i] * inv * weight[weightOff + i];
        }
    }

    /** RMS norm of a head-sized range of one row. */
    private void rmsNormRange(float[] x, int off, float[] weight, float[] out, int outOff, int width) {
        double sum = 0.0;
        for (int i = 0; i < width; i++) {
            sum += (double) x[off + i] * x[off + i];
        }
        float inv = (float) (1.0 / Math.sqrt(sum / width + eps));
        for (int i = 0; i < width; i++) {
            out[outOff + i] = x[off + i] * inv * weight[i];
        }
    }

    /** NEOX RoPE of one head: pairs {@code (i, i + width/2)} rotated by {@code pos * theta^(-2i/width)}. */
    private static void ropeNeox(float[] x, int off, int width, int pos) {
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

    private static void addInto(float[] target, float[] residual, float[] sum) {
        for (int i = 0; i < target.length; i++) {
            target[i] = residual[i] + sum[i];
        }
    }

    private static float silu(float x) {
        return (float) (x / (1.0 + Math.exp(-x)));
    }

    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    // ------------------------------------------------------------------ validation

    private static int positive(GgufKv kv, String key, String what) {
        int v = kv.u32Required(key);
        if (v <= 0) {
            throw new GgufFormatException("lfm3a depthformer: " + what + " must be positive, got " + v);
        }
        return v;
    }

    private static int dim(TensorRef ref, int i) {
        long v = ref.ne(i);
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new GgufFormatException("lfm3a depthformer: tensor " + ref.name() + " dimension " + i + " is " + v);
        }
        return (int) v;
    }

    /**
     * The head norm holds one {@code n_embd} vector per codebook: the converter stacks the checkpoint's pages, so
     * the ggml shape is {@code [n_embd, n_codebooks]} for one-dimensional pages and
     * {@code [n_embd, 1, n_codebooks]} for {@code [1, n_embd]} pages.
     */
    private static void headNormShape(TensorRef ref, long nEmbd, long nCodebooks) {
        boolean ok = ref.rank() >= 2 && ref.ne(0) == nEmbd;
        if (ref.rank() == 2) {
            ok &= ref.ne(1) == nCodebooks;
        } else if (ref.rank() >= 3) {
            ok &= ref.ne(1) == 1 && ref.ne(2) == nCodebooks;
            for (int i = 3; i < ref.rank() && ok; i++) {
                ok = ref.ne(i) == 1;
            }
        } else {
            ok = false;
        }
        if (!ok) {
            throw new GgufFormatException("lfm3a depthformer: tensor " + ref.name() + " has shape " + ref
                    + ", expected [" + nEmbd + ", " + nCodebooks + "] or [" + nEmbd + ", 1, " + nCodebooks
                    + "] (ggml order)");
        }
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
            throw new GgufFormatException("lfm3a depthformer: tensor " + ref.name() + " has shape " + ref
                    + ", expected " + java.util.Arrays.toString(ne) + " (ggml order)");
        }
    }

    private static void checkShape(CpuTensorRef ref, long cols, long rows) {
        if (ref.cols() != cols || ref.rows() != rows) {
            throw new GgufFormatException("lfm3a depthformer: tensor " + ref + " is " + ref.rows() + " x "
                    + ref.cols() + ", expected " + rows + " x " + cols);
        }
    }

    private static void vector(float[] v, String name, int n) {
        if (v == null || v.length != n) {
            throw new GgufFormatException("lfm3a depthformer: " + name + " must hold " + n + " floats, got "
                    + (v == null ? "none" : v.length));
        }
    }

    /** One pass's outputs: the codebook-major logits (null when every code was known) and the feedback row. */
    public static final class Frame {
        private final float[] logits;
        private final float[] feedback;

        Frame(float[] logits, float[] feedback) {
            this.logits = logits;
            this.feedback = feedback;
        }

        /** The codebook-major logits, null when the pass ran with every code revealed. */
        public float[] logits() {
            return logits;
        }

        /** The backbone-width feedback row (a copy). */
        public float[] feedback() {
            return feedback.clone();
        }
    }

    @Override
    public String toString() {
        return "Lfm3aDepthformer{embd=" + nEmbd + ", talker=" + talkerWidth + ", layers=" + nLayer + ", heads="
                + nHead + "/" + nHeadKv + ", codebooks=" + nCodebooks + ", vocab=" + vocab + '}';
    }
}
