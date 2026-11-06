package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkVector;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * The Granite Speech conformer encoder and its QFormer projector (llama.cpp tools/mtmd/models/granite-speech.cpp):
 * the input projection of every mel row, then per block a half-step SiLU FFN, the attention over blocks of
 * {@code clip.audio.chunk_size} frames with the Shaw relative-position term ({@code q . attn_rel_pos_emb[q - k +
 * max_pos_emb]}, a partial last block attending only its real frames), the conv module (pointwise conv, sigmoid
 * GLU, depthwise conv with symmetric padding, the folded batch norm, SiLU, pointwise conv), a second half-step FFN
 * and the block layer norm. After block {@code n_layer / 2} the CTC branch adds
 * {@code mid(softmax(out(x)))}; the blocks of {@code clip.audio.feature_layer} (0 = the projected input) are
 * concatenated before the final output. The QFormer runs {@code window / downsample_rate} learned queries per
 * window of {@code clip.audio.projector.window_size} frames (zero frames pad the last window) through
 * {@value #PROJ_LAYERS} post-norm blocks of self-attention within the window, cross-attention into the window and
 * the GELU FFN, then the linear projection. Package-private executor support.
 */
final class GraniteSpeechTower implements FamilyEncoder {

    /** Depth of the QFormer (granite-speech.cpp {@code n_proj_layers}). */
    static final int PROJ_LAYERS = MmprojTensorTable.GRANITE_QFORMER_BLOCKS;
    /** Epsilon of the QFormer layer norms (granite-speech.cpp {@code proj_eps}). */
    static final float PROJ_EPS = 1e-12f;

    private final int nMel;
    private final int width;
    private final int ctcLayer;
    private final boolean[] tapped;
    private final int nTaps;
    private final CpuMatVec matVec;
    private final CpuTensorRef inpProj;
    private final float[] inpProjB;
    private final Block[] blocks;
    private final CpuTensorRef ctcOut;
    private final float[] ctcOutB;
    private final CpuTensorRef ctcMid;
    private final float[] ctcMidB;
    private final QFormer qformer;

    private GraniteSpeechTower(ClipHparams hp, TensorCatalog tc, CpuMatVec matVec, Function<String, float[]> floats,
                               Function<String, CpuTensorRef> refs) {
        this.nMel = hp.audio().nMelBins();
        this.width = hp.nEmbd();
        this.ctcLayer = hp.nLayer() / 2;
        this.tapped = new boolean[hp.nLayer() + 1];
        for (int f : hp.featureLayers()) {
            tapped[f] = true;
        }
        this.nTaps = hp.featureLayers().length;
        this.matVec = matVec;
        this.inpProj = refs.apply("a.input_projection.weight");
        this.inpProjB = floats.apply("a.input_projection.bias");
        this.blocks = new Block[hp.nLayer()];
        for (int il = 0; il < blocks.length; il++) {
            blocks[il] = new Block(tc, hp, il, floats, refs);
        }
        this.ctcOut = refs.apply("a.enc_ctc_out.weight");
        this.ctcOutB = floats.apply("a.enc_ctc_out.bias");
        this.ctcMid = refs.apply("a.enc_ctc_out_mid.weight");
        this.ctcMidB = floats.apply("a.enc_ctc_out_mid.bias");
        this.qformer = new QFormer(hp, matVec, floats, refs);
    }

    /** Output rows of a chunk of {@code nFrames} frames: the windows times the queries per window. */
    static int tokens(int nFrames, int window, int downsample) {
        return (nFrames + window - 1) / window * (window / downsample);
    }

    /** Shape-checks the input projection, the blocks, the CTC branch, the feature taps and the QFormer. */
    static void validate(ClipHparams hp, TensorCatalog tc) {
        int e = hp.nEmbd();
        if (e % hp.nHead() != 0) {
            throw new IllegalArgumentException("granite_speech needs head_count to divide embedding_length, got "
                    + hp.nHead() + " / " + e);
        }
        shape(tc, tc.require("a.input_projection.weight"), hp.audio().nMelBins(), e);
        shape(tc, tc.require("a.input_projection.bias"), e);
        for (int il = 0; il < hp.nLayer(); il++) {
            Block.validate(tc, hp, il);
        }
        TensorRef ctc = tc.require("a.enc_ctc_out.weight");
        long classes = ctc.ne(1);
        shape(tc, ctc, e, classes);
        shape(tc, tc.require("a.enc_ctc_out.bias"), classes);
        shape(tc, tc.require("a.enc_ctc_out_mid.weight"), classes, e);
        shape(tc, tc.require("a.enc_ctc_out_mid.bias"), e);
        boolean[] seen = new boolean[hp.nLayer() + 1];
        for (int f : hp.featureLayers()) {
            if (f < 0 || seen[f]) {
                throw new IllegalArgumentException("clip.audio.feature_layer must list distinct blocks 0.."
                        + hp.nLayer() + " for granite_speech, got " + f);
            }
            seen[f] = true;
        }
        QFormer.validate(hp, tc);
    }

    /** Binds the executor over {@code model} with the weights of {@code floats} / {@code refs}. */
    static GraniteSpeechTower bind(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                                   Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        return new GraniteSpeechTower(model.hparams(), model.catalog(), Objects.requireNonNull(matVec, "matVec"),
                floats, refs);
    }

    /** The input projection, the blocks with the CTC branch and the feature taps, then the QFormer. */
    @Override
    public float[][] encode(MelChunk chunk) {
        if (chunk.nMel() != nMel) {
            throw new IllegalArgumentException("mel chunk has " + chunk.nMel() + " bins, the encoder needs " + nMel);
        }
        int n = chunk.nLen();
        float[] x = AudioOps.linear(matVec, inpProj, inpProjB, AudioOps.transpose(chunk.data(), nMel, n), n);
        int encWidth = width * (nTaps + 1);
        float[] enc = new float[n * encWidth];
        int slot = 0;
        if (tapped[0]) {
            place(x, n, enc, encWidth, slot++);
        }
        for (int il = 0; il < blocks.length; il++) {
            blocks[il].forward(matVec, x, n);
            if (tapped[il + 1]) {
                place(x, n, enc, encWidth, slot++);
            }
            if (il + 1 == ctcLayer) {
                float[] mid = AudioOps.linear(matVec, ctcOut, ctcOutB, x, n);
                softmaxRows(mid, (int) ctcOut.rows());
                AudioOps.addScaled(x, AudioOps.linear(matVec, ctcMid, ctcMidB, mid, n), 1f);
            }
        }
        place(x, n, enc, encWidth, slot);
        return qformer.forward(enc, n, encWidth);
    }

    /** Copies the {@code [n][width]} rows into column slot {@code slot} of the {@code [n][encWidth]} rows. */
    private void place(float[] x, int n, float[] enc, int encWidth, int slot) {
        for (int t = 0; t < n; t++) {
            System.arraycopy(x, t * width, enc, t * encWidth + slot * width, width);
        }
    }

    /** {@code ggml_soft_max} over every {@code width} row of {@code x}, in place. */
    private static void softmaxRows(float[] x, int width) {
        for (int base = 0; base < x.length; base += width) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < width; i++) {
                max = Math.max(max, x[base + i]);
            }
            double sum = 0;
            for (int i = 0; i < width; i++) {
                float w = (float) Math.exp(x[base + i] - max);
                x[base + i] = w;
                sum += w;
            }
            float inv = (float) (1.0 / sum);
            for (int i = 0; i < width; i++) {
                x[base + i] *= inv;
            }
        }
    }

    /** One conformer block {@code a.blk.N}. */
    private static final class Block {
        private final int width;
        private final int heads;
        private final int dh;
        private final int context;
        private final int maxPosEmb;
        private final float eps;
        private final float[] ffNormW;
        private final float[] ffNormB;
        private final CpuTensorRef ffUp;
        private final float[] ffUpB;
        private final CpuTensorRef ffDown;
        private final float[] ffDownB;
        private final float[] ln1W;
        private final float[] ln1B;
        private final CpuTensorRef q;
        private final CpuTensorRef k;
        private final CpuTensorRef v;
        private final CpuTensorRef o;
        private final float[] oB;
        private final float[] relPos;
        private final float[] normConvW;
        private final float[] normConvB;
        private final CpuTensorRef pw1;
        private final float[] pw1B;
        private final float[] dw;
        private final int kernel;
        private final int inner;
        private final float[] bnW;
        private final float[] bnB;
        private final CpuTensorRef pw2;
        private final float[] pw2B;
        private final float[] ffNorm1W;
        private final float[] ffNorm1B;
        private final CpuTensorRef ffUp1;
        private final float[] ffUp1B;
        private final CpuTensorRef ffDown1;
        private final float[] ffDown1B;
        private final float[] ln2W;
        private final float[] ln2B;

        Block(TensorCatalog tc, ClipHparams hp, int il, Function<String, float[]> floats,
              Function<String, CpuTensorRef> refs) {
            String p = prefix(il);
            this.width = hp.nEmbd();
            this.heads = hp.nHead();
            this.dh = width / heads;
            this.context = hp.audio().chunkSize();
            this.maxPosEmb = hp.audio().maxPosEmb();
            this.eps = hp.eps();
            this.ffNormW = floats.apply(p + "ffn_norm.weight");
            this.ffNormB = floats.apply(p + "ffn_norm.bias");
            this.ffUp = refs.apply(p + "ffn_up.weight");
            this.ffUpB = floats.apply(p + "ffn_up.bias");
            this.ffDown = refs.apply(p + "ffn_down.weight");
            this.ffDownB = floats.apply(p + "ffn_down.bias");
            this.ln1W = floats.apply(p + "ln1.weight");
            this.ln1B = tc.has(p + "ln1.bias") ? floats.apply(p + "ln1.bias") : null;
            this.q = refs.apply(p + "attn_q.weight");
            this.k = refs.apply(p + "attn_k.weight");
            this.v = refs.apply(p + "attn_v.weight");
            this.o = refs.apply(p + "attn_out.weight");
            this.oB = floats.apply(p + "attn_out.bias");
            this.relPos = floats.apply(p + "attn_rel_pos_emb");
            this.normConvW = floats.apply(p + "norm_conv.weight");
            this.normConvB = floats.apply(p + "norm_conv.bias");
            this.pw1 = refs.apply(p + "conv_pw1.weight");
            this.pw1B = floats.apply(p + "conv_pw1.bias");
            this.dw = floats.apply(p + "conv_dw.weight");
            this.kernel = hp.audio().convKernel();
            this.inner = (int) tc.require(p + "conv_dw.weight").ne(1);
            this.bnW = floats.apply(p + "conv_norm.weight");
            this.bnB = floats.apply(p + "conv_norm.bias");
            this.pw2 = refs.apply(p + "conv_pw2.weight");
            this.pw2B = floats.apply(p + "conv_pw2.bias");
            this.ffNorm1W = floats.apply(p + "ffn_norm_1.weight");
            this.ffNorm1B = floats.apply(p + "ffn_norm_1.bias");
            this.ffUp1 = refs.apply(p + "ffn_up_1.weight");
            this.ffUp1B = floats.apply(p + "ffn_up_1.bias");
            this.ffDown1 = refs.apply(p + "ffn_down_1.weight");
            this.ffDown1B = floats.apply(p + "ffn_down_1.bias");
            this.ln2W = floats.apply(p + "ln2.weight");
            this.ln2B = tc.has(p + "ln2.bias") ? floats.apply(p + "ln2.bias") : null;
        }

        private static String prefix(int il) {
            return "a.blk." + il + ".";
        }

        static void validate(TensorCatalog tc, ClipHparams hp, int il) {
            String p = prefix(il);
            int e = hp.nEmbd();
            int c = hp.audio().chunkSize();
            for (String norm : new String[]{"ffn_norm", "ffn_norm_1", "norm_conv"}) {
                shape(tc, tc.require(p + norm + ".weight"), e);
                shape(tc, tc.require(p + norm + ".bias"), e);
            }
            for (String norm : new String[]{"ln1", "ln2"}) {
                shape(tc, tc.require(p + norm + ".weight"), e);
                checkVector(tc, tc.get(p + norm + ".bias"), e);
            }
            ffn(tc, p + "ffn_up", p + "ffn_down", e);
            ffn(tc, p + "ffn_up_1", p + "ffn_down_1", e);
            for (String proj : new String[]{"attn_q", "attn_k", "attn_v", "attn_out"}) {
                shape(tc, tc.require(p + proj + ".weight"), e, e);
            }
            shape(tc, tc.require(p + "attn_out.bias"), e);
            TensorRef rel = tc.require(p + "attn_rel_pos_emb");
            long rows = rel.ne(1);
            shape(tc, rel, e / hp.nHead(), rows);
            int maxPos = hp.audio().maxPosEmb();
            if (maxPos < c - 1 || maxPos + c - 1 >= rows) {
                throw new IllegalArgumentException("tensor " + rel.name() + " holds " + rows + " rows; the"
                        + " distances within a chunk of " + c + " frames read rows " + (maxPos - c + 1) + ".."
                        + (maxPos + c - 1) + " (clip.audio.max_pos_emb " + maxPos + ")");
            }
            int kernel = hp.audio().convKernel();
            TensorRef pw1 = tc.require(p + "conv_pw1.weight");
            long inner = pw1.ne(1) / 2;
            shape(tc, pw1, e, 2 * inner);
            shape(tc, tc.require(p + "conv_pw1.bias"), 2 * inner);
            shape(tc, tc.require(p + "conv_dw.weight"), kernel, inner);
            shape(tc, tc.require(p + "conv_norm.weight"), inner);
            shape(tc, tc.require(p + "conv_norm.bias"), inner);
            shape(tc, tc.require(p + "conv_pw2.weight"), inner, e);
            shape(tc, tc.require(p + "conv_pw2.bias"), e);
        }

        /** A biased up / down pair {@code [e, ff]} and {@code [ff, e]}. */
        private static void ffn(TensorCatalog tc, String up, String down, int e) {
            TensorRef u = tc.require(up + ".weight");
            long ff = u.ne(1);
            shape(tc, u, e, ff);
            shape(tc, tc.require(up + ".bias"), ff);
            shape(tc, tc.require(down + ".weight"), ff, e);
            shape(tc, tc.require(down + ".bias"), e);
        }

        /** The block over {@code [n][width]} rows of {@code x}, in place. */
        void forward(CpuMatVec matVec, float[] x, int n) {
            float[] h = normed(x, n, ffNormW, ffNormB);
            AudioOps.addScaled(x, AudioOps.ffn(matVec, h, n, ffUp, ffUpB, ActOp.SILU, ffDown, ffDownB), 0.5f);
            AudioOps.addScaled(x, attention(matVec, normed(x, n, ln1W, ln1B), n), 1f);
            AudioOps.addScaled(x, conv(matVec, normed(x, n, normConvW, normConvB), n), 1f);
            h = normed(x, n, ffNorm1W, ffNorm1B);
            AudioOps.addScaled(x, AudioOps.ffn(matVec, h, n, ffUp1, ffUp1B, ActOp.SILU, ffDown1, ffDown1B), 0.5f);
            AudioOps.layerNorm(x, n, width, ln2W, ln2B, eps);
        }

        private float[] normed(float[] x, int n, float[] w, float[] b) {
            float[] h = x.clone();
            AudioOps.layerNorm(h, n, width, w, b, eps);
            return h;
        }

        /** Attention within each chunk of {@code context} frames with the Shaw term; {@code [n][width]} rows. */
        private float[] attention(CpuMatVec matVec, float[] h, int n) {
            float[] qr = AudioOps.linear(matVec, q, null, h, n);
            float[] kr = AudioOps.linear(matVec, k, null, h, n);
            float[] vr = AudioOps.linear(matVec, v, null, h, n);
            int chunks = (n + context - 1) / context;
            float[] out = new float[n * width];
            float scale = (float) (1.0 / Math.sqrt(dh));
            CpuParallel.parallelFor(chunks * heads, 1, (from, to) -> {
                float[] s = new float[context];
                for (int task = from; task < to; task++) {
                    int start = (task / heads) * context;
                    int len = Math.min(context, n - start);
                    int off = (task % heads) * dh;
                    for (int i = 0; i < len; i++) {
                        attendRow(qr, kr, vr, out, start, len, i, off, scale, s);
                    }
                }
            });
            return AudioOps.linear(matVec, o, oB, out, n);
        }

        /** Query {@code i} of the chunk at {@code start} over its {@code len} keys for the head at {@code off}. */
        private void attendRow(float[] qr, float[] kr, float[] vr, float[] out, int start, int len, int i, int off,
                               float scale, float[] s) {
            int qi = (start + i) * width + off;
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < len; j++) {
                int kj = (start + j) * width + off;
                int rel = (i - j + maxPosEmb) * dh;
                float dot = 0f;
                for (int d = 0; d < dh; d++) {
                    dot += qr[qi + d] * (kr[kj + d] + relPos[rel + d]);
                }
                s[j] = dot * scale;
                max = Math.max(max, s[j]);
            }
            float sum = 0f;
            for (int j = 0; j < len; j++) {
                s[j] = (float) Math.exp(s[j] - max);
                sum += s[j];
            }
            for (int j = 0; j < len; j++) {
                float w = s[j] / sum;
                int vj = (start + j) * width + off;
                for (int d = 0; d < dh; d++) {
                    out[qi + d] += w * vr[vj + d];
                }
            }
        }

        /** Pointwise conv, sigmoid GLU, depthwise conv, folded batch norm, SiLU, pointwise conv. */
        private float[] conv(CpuMatVec matVec, float[] h, int n) {
            float[] glu = AudioOps.sigmoidGlu(AudioOps.linear(matVec, pw1, pw1B, h, n), n, inner);
            float[] c = AudioOps.depthwiseConv1d(glu, n, inner, dw, kernel, kernel / 2, kernel / 2);
            for (int i = 0; i < c.length; i++) {
                c[i] = c[i] * bnW[i % inner] + bnB[i % inner];
            }
            AudioOps.activate(ActOp.SILU, c);
            return AudioOps.linear(matVec, pw2, pw2B, c, n);
        }
    }

    /** The QFormer projector {@code a.proj_*}: learned queries per window, post-norm blocks, the projection. */
    private static final class QFormer {
        private final int width;
        private final int heads;
        private final int window;
        private final int nQueries;
        private final CpuMatVec matVec;
        private final float[] queries;
        private final QBlock[] blocks;
        private final CpuTensorRef linear;
        private final float[] linearB;

        QFormer(ClipHparams hp, CpuMatVec matVec, Function<String, float[]> floats,
                Function<String, CpuTensorRef> refs) {
            ClipAudioHparams a = hp.audio();
            this.width = hp.nEmbd();
            this.heads = a.projHeadCount();
            this.window = a.projWindowSize();
            this.nQueries = window / a.projDownsampleRate();
            this.matVec = matVec;
            this.queries = floats.apply("a.proj_query").clone();
            AudioOps.layerNorm(queries, nQueries, width, floats.apply("a.proj_norm.weight"),
                    floats.apply("a.proj_norm.bias"), PROJ_EPS);
            this.blocks = new QBlock[PROJ_LAYERS];
            for (int il = 0; il < blocks.length; il++) {
                blocks[il] = new QBlock(il, floats, refs);
            }
            this.linear = refs.apply("a.proj_linear.weight");
            this.linearB = floats.apply("a.proj_linear.bias");
        }

        static void validate(ClipHparams hp, TensorCatalog tc) {
            ClipAudioHparams a = hp.audio();
            int e = hp.nEmbd();
            if (e % a.projHeadCount() != 0) {
                throw new IllegalArgumentException("clip.audio.projector.head_count " + a.projHeadCount()
                        + " must divide embedding_length " + e);
            }
            int nq = a.projWindowSize() / a.projDownsampleRate();
            TensorRef query = tc.require("a.proj_query");
            if (query.rank() == 3) {
                shape(tc, query, e, nq, 1);
            } else {
                shape(tc, query, e, nq);
            }
            shape(tc, tc.require("a.proj_norm.weight"), e);
            shape(tc, tc.require("a.proj_norm.bias"), e);
            shape(tc, tc.require("a.proj_linear.weight"), e, hp.projectionDim());
            shape(tc, tc.require("a.proj_linear.bias"), hp.projectionDim());
            long encWidth = (long) e * (hp.featureLayers().length + 1);
            for (int il = 0; il < PROJ_LAYERS; il++) {
                QBlock.validate(tc, il, e, encWidth);
            }
        }

        /** The queries over the {@code [n][encWidth]} encoder rows; {@code [windows * queries][d]} rows. */
        float[][] forward(float[] enc, int n, int encWidth) {
            int windows = (n + window - 1) / window;
            float[] keys = new float[windows * window * encWidth];
            System.arraycopy(enc, 0, keys, 0, n * encWidth);
            int rows = windows * nQueries;
            float[] x = new float[rows * width];
            for (int w = 0; w < windows; w++) {
                System.arraycopy(queries, 0, x, w * nQueries * width, nQueries * width);
            }
            for (QBlock b : blocks) {
                b.forward(matVec, x, rows, keys, windows * window, nQueries, window, heads);
            }
            return VisionRows.tokenRows(AudioOps.linear(matVec, linear, linearB, x, rows), (int) linear.rows());
        }
    }

    /** One QFormer block {@code a.proj_blk.N}: self-attention, cross-attention and the GELU FFN, each post-norm. */
    private static final class QBlock {
        private static final String[] ATTN = {"self_attn_q", "self_attn_k", "self_attn_v", "self_attn_out",
                "cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_out"};
        private static final String[] NORMS = {"self_attn_norm", "cross_attn_norm", "ffn_norm"};

        private final CpuTensorRef[] w = new CpuTensorRef[ATTN.length];
        private final float[][] b = new float[ATTN.length][];
        private final float[][] normW = new float[NORMS.length][];
        private final float[][] normB = new float[NORMS.length][];
        private final CpuTensorRef up;
        private final float[] upB;
        private final CpuTensorRef down;
        private final float[] downB;

        QBlock(int il, Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
            String p = prefix(il);
            for (int i = 0; i < ATTN.length; i++) {
                w[i] = refs.apply(p + ATTN[i] + ".weight");
                b[i] = floats.apply(p + ATTN[i] + ".bias");
            }
            for (int i = 0; i < NORMS.length; i++) {
                normW[i] = floats.apply(p + NORMS[i] + ".weight");
                normB[i] = floats.apply(p + NORMS[i] + ".bias");
            }
            this.up = refs.apply(p + "ffn_up.weight");
            this.upB = floats.apply(p + "ffn_up.bias");
            this.down = refs.apply(p + "ffn_down.weight");
            this.downB = floats.apply(p + "ffn_down.bias");
        }

        private static String prefix(int il) {
            return "a.proj_blk." + il + ".";
        }

        /** The cross-attention K and V read the {@code encWidth}-wide encoder rows; all else is {@code e} wide. */
        static void validate(TensorCatalog tc, int il, int e, long encWidth) {
            String p = prefix(il);
            for (String t : ATTN) {
                long in = t.equals("cross_attn_k") || t.equals("cross_attn_v") ? encWidth : e;
                shape(tc, tc.require(p + t + ".weight"), in, e);
                shape(tc, tc.require(p + t + ".bias"), e);
            }
            for (String t : NORMS) {
                shape(tc, tc.require(p + t + ".weight"), e);
                shape(tc, tc.require(p + t + ".bias"), e);
            }
            Block.ffn(tc, p + "ffn_up", p + "ffn_down", e);
        }

        /** The block over the {@code [rows][e]} queries, {@code nq} per window, in place. */
        void forward(CpuMatVec matVec, float[] x, int rows, float[] keys, int nKeys, int nq, int window, int heads) {
            int e = x.length / rows;
            int dh = e / heads;
            float scale = (float) (1.0 / Math.sqrt(dh));
            VisionRows.KeyRanges ownWindow = ranges(nq, nq);
            float[] q = AudioOps.linear(matVec, w[0], b[0], x, rows);
            float[] k = AudioOps.linear(matVec, w[1], b[1], x, rows);
            float[] v = AudioOps.linear(matVec, w[2], b[2], x, rows);
            float[] a = VisionRows.attend(q, 0, rows, k, v, 0, rows, e, heads, 1, dh, scale, ownWindow);
            postNorm(x, AudioOps.linear(matVec, w[3], b[3], a, rows), rows, 0);
            q = AudioOps.linear(matVec, w[4], b[4], x, rows);
            k = AudioOps.linear(matVec, w[5], b[5], keys, nKeys);
            v = AudioOps.linear(matVec, w[6], b[6], keys, nKeys);
            a = VisionRows.attend(q, 0, rows, k, v, 0, nKeys, e, heads, 1, dh, scale, ranges(nq, window));
            postNorm(x, AudioOps.linear(matVec, w[7], b[7], a, rows), rows, 1);
            postNorm(x, AudioOps.ffn(matVec, x, rows, up, upB, ActOp.GELU_TANH, down, downB), rows, 2);
        }

        /** {@code x = norm(x + y)} with norm {@code i}. */
        private void postNorm(float[] x, float[] y, int rows, int i) {
            AudioOps.addScaled(x, y, 1f);
            AudioOps.layerNorm(x, rows, x.length / rows, normW[i], normB[i], PROJ_EPS);
        }

        /** Query row {@code r} of window {@code r / nq} reads the {@code keysPerWindow} keys of its window. */
        private static VisionRows.KeyRanges ranges(int nq, int keysPerWindow) {
            return new VisionRows.KeyRanges() {
                @Override
                public int begin(int row) {
                    return row / nq * keysPerWindow;
                }

                @Override
                public int end(int row) {
                    return row / nq * keysPerWindow + keysPerWindow;
                }
            };
        }
    }
}
