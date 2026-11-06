package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkVector;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * The LFM3-Audio encoder and its projector (llama.cpp tools/mtmd/models/lfm3a.cpp, {@code clip_graph_lfm3a}) from
 * a zero streaming state, which encodes a clip from its start: the downsampler over the {@code [frames, mel]}
 * image — three 3x3 stride-2 convs padded by one zero column and row each side (full, then depthwise followed by
 * a 1x1 conv, twice), every step biased and SiLU-activated — the {@code [mel / 8 * C]} flatten (mel bins minor)
 * through {@code a.pre_encode.out}, then the LFM2 stack: RMS-normed blocks that are either a short conv
 * ({@code c * causal_dwconv(b * x)} between {@code conv_pw1} and {@code conv_pw2}) or an attention block (RMS
 * q/k head norms, NEOX RoPE at base 1e6 over the frame positions, causal over the last
 * {@code clip.audio.window_size} positions), each followed by the gated SiLU FFN; the RMS {@code a.post_ln} and the
 * layer-norm GELU-erf MLP {@code mm.a.mlp.0/1/3}. Immutable weights; package-private executor support.
 */
final class Lfm3aTower implements FamilyEncoder {

    private static final int K = 3;

    private final int nEmbd;
    private final int nMel;
    private final int outWidth;
    private final float eps;
    private final CpuMatVec matVec;
    private final Stem stem;
    private final Block[] blocks;
    private final float[] postLn;
    private final float[] mm0W;
    private final float[] mm0B;
    private final CpuTensorRef mm1;
    private final float[] mm1B;
    private final CpuTensorRef mm3;
    private final float[] mm3B;

    private Lfm3aTower(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                       Function<String, CpuTensorRef> refs) {
        ClipHparams hp = model.hparams();
        TensorCatalog tc = model.catalog();
        this.nEmbd = hp.nEmbd();
        this.nMel = hp.audio().nMelBins();
        this.outWidth = model.outputWidth();
        this.eps = hp.eps();
        this.matVec = matVec;
        this.stem = new Stem(tc, floats, refs);
        this.blocks = new Block[hp.nLayer()];
        for (int il = 0; il < blocks.length; il++) {
            blocks[il] = new Block(tc, hp, il, floats, refs);
        }
        this.postLn = floats.apply("a.post_ln.weight");
        this.mm0W = floats.apply("mm.a.mlp.0.weight");
        this.mm0B = floats.apply("mm.a.mlp.0.bias");
        this.mm1 = refs.apply("mm.a.mlp.1.weight");
        this.mm1B = floats.apply("mm.a.mlp.1.bias");
        this.mm3 = refs.apply("mm.a.mlp.3.weight");
        this.mm3B = floats.apply("mm.a.mlp.3.bias");
    }

    /** Shape-checks the downsampler, every block, the trunk norm and the projector. */
    static void validate(ClipHparams hp, TensorCatalog tc) {
        int e = hp.nEmbd();
        if (e % hp.nHead() != 0 || (e / hp.nHead()) % 2 != 0) {
            throw new IllegalArgumentException("the lfm3a attention rotates whole heads in NEOX pairs, so"
                    + " clip.audio.embedding_length " + e + " must split into " + hp.nHead() + " even-width heads");
        }
        Stem.validate(tc, hp.audio().nMelBins(), e);
        for (int il = 0; il < hp.nLayer(); il++) {
            Block.validate(tc, hp, il);
        }
        shape(tc, tc.require("a.post_ln.weight"), e);
        shape(tc, tc.require("mm.a.mlp.0.weight"), e);
        shape(tc, tc.require("mm.a.mlp.0.bias"), e);
        TensorRef m1 = tc.require("mm.a.mlp.1.weight");
        long ff = m1.ne(1);
        shape(tc, m1, e, ff);
        shape(tc, tc.require("mm.a.mlp.1.bias"), ff);
        shape(tc, tc.require("mm.a.mlp.3.weight"), ff, hp.projectionDim());
        shape(tc, tc.require("mm.a.mlp.3.bias"), hp.projectionDim());
    }

    /** Binds the executor over {@code model} with the weights of {@code floats} / {@code refs}. */
    static Lfm3aTower bind(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                           Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(matVec, "matVec");
        return new Lfm3aTower(model, matVec, floats, refs);
    }

    /** The downsampler, the blocks, the trunk norm and the projector: {@code ConformerOps.subsampled(nLen)} rows. */
    @Override
    public float[][] encode(MelChunk chunk) {
        if (chunk.nMel() != nMel) {
            throw new IllegalArgumentException("mel chunk has " + chunk.nMel() + " bins, the encoder needs " + nMel);
        }
        float[] x = stem.run(matVec, chunk);
        int n = x.length / nEmbd;
        for (Block b : blocks) {
            b.forward(matVec, x, n, eps);
        }
        AudioOps.rmsNorm(x, n, nEmbd, postLn, null, eps);
        AudioOps.layerNorm(x, n, nEmbd, mm0W, mm0B, AudioOps.GRAPH_EPS);
        return VisionRows.tokenRows(AudioOps.ffn(matVec, x, n, mm1, mm1B, ActOp.GELU_ERF, mm3, mm3B), outWidth);
    }

    /** The downsampler {@code a.conv1d.{0,2,3,5,6}} and {@code a.pre_encode.out}. */
    private static final class Stem {
        private final CpuTensorRef conv0;
        private final float[] conv0B;
        private final float[] dw2;
        private final float[] dw2B;
        private final CpuTensorRef pw3;
        private final float[] pw3B;
        private final float[] dw5;
        private final float[] dw5B;
        private final CpuTensorRef pw6;
        private final float[] pw6B;
        private final CpuTensorRef out;
        private final float[] outB;

        Stem(TensorCatalog tc, Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
            int[] c = channels(tc);
            this.conv0 = CpuVitTower.matrix(refs.apply(name(0, "weight")), c[0], K * K);
            this.conv0B = floats.apply(name(0, "bias"));
            this.dw2 = floats.apply(name(2, "weight"));
            this.dw2B = floats.apply(name(2, "bias"));
            this.pw3 = CpuVitTower.matrix(refs.apply(name(3, "weight")), c[1], c[0]);
            this.pw3B = floats.apply(name(3, "bias"));
            this.dw5 = floats.apply(name(5, "weight"));
            this.dw5B = floats.apply(name(5, "bias"));
            this.pw6 = CpuVitTower.matrix(refs.apply(name(6, "weight")), c[2], c[1]);
            this.pw6B = floats.apply(name(6, "bias"));
            this.out = refs.apply("a.pre_encode.out.weight");
            this.outB = floats.apply("a.pre_encode.out.bias");
        }

        private static String name(int i, String suffix) {
            return "a.conv1d." + i + "." + suffix;
        }

        private static int[] channels(TensorCatalog tc) {
            return new int[]{(int) tc.require(name(0, "weight")).ne(3), (int) tc.require(name(3, "weight")).ne(3),
                    (int) tc.require(name(6, "weight")).ne(3)};
        }

        /**
         * Shape-checks the convs and the output projection. The streaming state lfm3a.cpp threads through the
         * strided convs holds {@code n_mel / 2} and {@code n_mel / 4} mel rows of {@code conv0}'s channels, so the
         * mel bins must be a multiple of four and the 1x1 conv 3 must keep conv 0's channel count.
         */
        static void validate(TensorCatalog tc, int nMel, int e) {
            int[] c = channels(tc);
            if (nMel % 4 != 0) {
                throw new IllegalArgumentException("the lfm3a downsampler state holds n_mel / 2 and n_mel / 4 mel"
                        + " rows, so clip.audio.num_mel_bins " + nMel + " must be a multiple of 4");
            }
            if (c[1] != c[0]) {
                throw new IllegalArgumentException("the lfm3a downsampler state keeps " + c[0] + " channels of "
                        + name(0, "weight") + ", but " + name(3, "weight") + " outputs " + c[1]);
            }
            shape(tc, tc.require(name(0, "weight")), K, K, 1, c[0]);
            shape(tc, tc.require(name(0, "bias")), 1, 1, c[0]);
            shape(tc, tc.require(name(2, "weight")), K, K, 1, c[0]);
            shape(tc, tc.require(name(2, "bias")), 1, 1, c[0]);
            shape(tc, tc.require(name(3, "weight")), 1, 1, c[0], c[1]);
            shape(tc, tc.require(name(3, "bias")), 1, 1, c[1]);
            shape(tc, tc.require(name(5, "weight")), K, K, 1, c[1]);
            shape(tc, tc.require(name(5, "bias")), 1, 1, c[1]);
            shape(tc, tc.require(name(6, "weight")), 1, 1, c[1], c[2]);
            shape(tc, tc.require(name(6, "bias")), 1, 1, c[2]);
            shape(tc, tc.require("a.pre_encode.out.weight"), (long) ConformerOps.subsampled(nMel) * c[2], e);
            shape(tc, tc.require("a.pre_encode.out.bias"), e);
        }

        /** The downsampler over the planar {@code [frames, mel, 1]} chunk: {@code [frames / 8][nEmbd]} rows. */
        float[] run(CpuMatVec matVec, MelChunk chunk) {
            int w = chunk.nLen();
            int h = chunk.nMel();
            float[] img = AudioOps.conv2d(matVec, chunk.data(), w, h, 1, conv0, K, K, 2, 1, conv0B);
            AudioOps.activate(ActOp.SILU, img);
            w = AudioOps.convOut(w, K, 2, 1);
            h = AudioOps.convOut(h, K, 2, 1);
            int c = (int) conv0.rows();
            img = AudioOps.depthwiseConv2d(img, w, h, c, dw2, K, K, 2, 1, dw2B);
            w = AudioOps.convOut(w, K, 2, 1);
            h = AudioOps.convOut(h, K, 2, 1);
            img = AudioOps.conv2d(matVec, img, w, h, c, pw3, 1, 1, 1, 0, pw3B);
            AudioOps.activate(ActOp.SILU, img);
            c = (int) pw3.rows();
            img = AudioOps.depthwiseConv2d(img, w, h, c, dw5, K, K, 2, 1, dw5B);
            w = AudioOps.convOut(w, K, 2, 1);
            h = AudioOps.convOut(h, K, 2, 1);
            img = AudioOps.conv2d(matVec, img, w, h, c, pw6, 1, 1, 1, 0, pw6B);
            AudioOps.activate(ActOp.SILU, img);
            c = (int) pw6.rows();
            return AudioOps.linear(matVec, out, outB, AudioOps.flattenFreqMajor(img, w, h, c), w);
        }
    }

    /** One LFM2 block {@code a.blk.N}: a short-conv or attention operator, then the gated SiLU FFN. */
    private static final class Block {
        private final int width;
        private final float[] ln1;
        private final float[] ln2;
        private final CpuTensorRef ffUp;
        private final CpuTensorRef ffGate;
        private final CpuTensorRef ffDown;
        // the short-conv operator
        private final CpuTensorRef pw1;
        private final float[] dw;
        private final int taps;
        private final CpuTensorRef pw2;
        // the attention operator
        private final CpuTensorRef q;
        private final CpuTensorRef k;
        private final CpuTensorRef v;
        private final CpuTensorRef o;
        private final float[] qNorm;
        private final float[] kNorm;
        private final int heads;
        private final int kvHeads;
        private final int window;
        private final float theta;

        Block(TensorCatalog tc, ClipHparams hp, int il, Function<String, float[]> floats,
              Function<String, CpuTensorRef> refs) {
            String p = prefix(il);
            boolean conv = isConv(tc, il);
            this.width = hp.nEmbd();
            this.ln1 = tc.get(p + "ln1.weight") == null ? null : floats.apply(p + "ln1.weight");
            this.ln2 = tc.get(p + "ln2.weight") == null ? null : floats.apply(p + "ln2.weight");
            this.ffUp = refs.apply(p + "ffn_up.weight");
            this.ffGate = tc.get(p + "ffn_gate.weight") == null ? null : refs.apply(p + "ffn_gate.weight");
            this.ffDown = refs.apply(p + "ffn_down.weight");
            this.pw1 = conv ? refs.apply(p + "conv_pw1.weight") : null;
            this.dw = conv ? floats.apply(p + "conv_dw.weight") : null;
            this.taps = conv ? (int) tc.require(p + "conv_dw.weight").ne(0) : 0;
            this.pw2 = conv ? refs.apply(p + "conv_pw2.weight") : null;
            this.q = conv ? null : refs.apply(p + "attn_q.weight");
            this.k = conv ? null : refs.apply(p + "attn_k.weight");
            this.v = conv ? null : refs.apply(p + "attn_v.weight");
            this.o = conv ? null : refs.apply(p + "attn_out.weight");
            this.qNorm = conv ? null : floats.apply(p + "attn_q_norm.weight");
            this.kNorm = conv ? null : floats.apply(p + "attn_k_norm.weight");
            this.heads = hp.nHead();
            this.kvHeads = hp.nHeadKv();
            this.window = hp.audio().attnWindow();
            this.theta = hp.audio().ropeTheta();
        }

        private static String prefix(int il) {
            return "a.blk." + il + ".";
        }

        /** A short-conv block has all of {@code conv_pw1}, {@code conv_dw} and {@code conv_pw2}. */
        private static boolean isConv(TensorCatalog tc, int il) {
            String p = prefix(il);
            return tc.has(p + "conv_pw1.weight") && tc.has(p + "conv_dw.weight") && tc.has(p + "conv_pw2.weight");
        }

        /** A complete attention block: q, k, v, out and the q / k head norms. */
        private static boolean isAttention(TensorCatalog tc, int il) {
            String p = prefix(il);
            for (String t : new String[]{"attn_q", "attn_k", "attn_v", "attn_out", "attn_q_norm", "attn_k_norm"}) {
                if (!tc.has(p + t + ".weight")) {
                    return false;
                }
            }
            return true;
        }

        static void validate(TensorCatalog tc, ClipHparams hp, int il) {
            String p = prefix(il);
            if (isConv(tc, il) == isAttention(tc, il)) {
                throw new IllegalArgumentException("lfm3a layer " + il + " is neither a complete conv block"
                        + " (conv_pw1, conv_dw, conv_pw2) nor a complete attention block (attn_q, attn_k, attn_v,"
                        + " attn_out, attn_q_norm, attn_k_norm), or is both");
            }
            int e = hp.nEmbd();
            int dh = e / hp.nHead();
            checkVector(tc, tc.get(p + "ln1.weight"), e);
            checkVector(tc, tc.get(p + "ln2.weight"), e);
            TensorRef up = tc.require(p + "ffn_up.weight");
            long ff = up.ne(1);
            shape(tc, up, e, ff);
            TensorRef gate = tc.get(p + "ffn_gate.weight");
            if (gate != null) {
                shape(tc, gate, e, ff);
            }
            shape(tc, tc.require(p + "ffn_down.weight"), ff, e);
            if (isConv(tc, il)) {
                shape(tc, tc.require(p + "conv_pw1.weight"), e, 3L * e);
                TensorRef dw = tc.require(p + "conv_dw.weight");
                if (dw.ne(0) < 2) {
                    throw new IllegalArgumentException("tensor " + dw.name() + " holds " + dw.ne(0)
                            + " taps, the lfm3a short conv keeps at least one past frame");
                }
                shape(tc, dw, dw.ne(0), e);
                shape(tc, tc.require(p + "conv_pw2.weight"), e, e);
                return;
            }
            int kv = dh * hp.nHeadKv();
            shape(tc, tc.require(p + "attn_q.weight"), e, e);
            shape(tc, tc.require(p + "attn_k.weight"), e, kv);
            shape(tc, tc.require(p + "attn_v.weight"), e, kv);
            shape(tc, tc.require(p + "attn_out.weight"), e, e);
            shape(tc, tc.require(p + "attn_q_norm.weight"), dh);
            shape(tc, tc.require(p + "attn_k_norm.weight"), dh);
        }

        /** The block over {@code [n][width]} rows of {@code x}, in place. */
        void forward(CpuMatVec matVec, float[] x, int n, float eps) {
            float[] h = x.clone();
            AudioOps.rmsNorm(h, n, width, ln1, null, eps);
            AudioOps.addScaled(x, pw1 != null ? shortConv(matVec, h, n) : attention(matVec, h, n, eps), 1f);
            h = x.clone();
            AudioOps.rmsNorm(h, n, width, ln2, null, eps);
            float[] up = AudioOps.linear(matVec, ffUp, null, h, n);
            if (ffGate != null) {
                float[] g = AudioOps.linear(matVec, ffGate, null, h, n);
                AudioOps.activate(ActOp.SILU, g);
                for (int i = 0; i < up.length; i++) {
                    up[i] *= g[i];
                }
            } else {
                AudioOps.activate(ActOp.SILU, up);
            }
            AudioOps.addScaled(x, AudioOps.linear(matVec, ffDown, null, up, n), 1f);
        }

        /** {@code pw2(c * conv(b * x))}: the depthwise conv reads {@code taps - 1} zero frames of state first. */
        private float[] shortConv(CpuMatVec matVec, float[] h, int n) {
            float[] bcx = AudioOps.linear(matVec, pw1, null, h, n);
            float[] bx = new float[n * width];
            for (int t = 0; t < n; t++) {
                for (int c = 0; c < width; c++) {
                    bx[t * width + c] = bcx[t * 3 * width + c] * bcx[t * 3 * width + 2 * width + c];
                }
            }
            float[] conv = AudioOps.depthwiseConv1d(bx, n, width, dw, taps, taps - 1, 0);
            for (int t = 0; t < n; t++) {
                for (int c = 0; c < width; c++) {
                    conv[t * width + c] *= bcx[t * 3 * width + width + c];
                }
            }
            return AudioOps.linear(matVec, pw2, null, conv, n);
        }

        /** RMS head norms, NEOX RoPE, attention to the last {@code window} positions, the output projection. */
        private float[] attention(CpuMatVec matVec, float[] h, int n, float eps) {
            int dh = width / heads;
            float[] qr = AudioOps.linear(matVec, q, null, h, n);
            float[] kr = AudioOps.linear(matVec, k, null, h, n);
            float[] vr = AudioOps.linear(matVec, v, null, h, n);
            AudioOps.rmsNorm(qr, n * heads, dh, qNorm, null, eps);
            AudioOps.rmsNorm(kr, n * kvHeads, dh, kNorm, null, eps);
            int[] pos = AudioOps.iota(n);
            AudioOps.ropeNeox(qr, n, heads, dh, dh, pos, theta);
            AudioOps.ropeNeox(kr, n, kvHeads, dh, dh, pos, theta);
            VisionRows.KeyRanges causal = new VisionRows.KeyRanges() {
                @Override
                public int begin(int row) {
                    return Math.max(0, row - window + 1);
                }

                @Override
                public int end(int row) {
                    return row + 1;
                }
            };
            float[] a = VisionRows.attend(qr, 0, n, kr, vr, 0, n, kvHeads * dh, heads, heads / kvHeads, dh,
                    (float) (1.0 / Math.sqrt(dh)), causal);
            return AudioOps.linear(matVec, o, null, a, n);
        }
    }
}
