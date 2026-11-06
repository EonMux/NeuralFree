package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkVector;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * The NeMo conformer pieces the lfm2a / d1omni_a graph (llama.cpp tools/mtmd/models/conformer.cpp) and the
 * parakeet graph (parakeet.cpp) share: the conv subsampling {@link Stem}, the sinusoidal relative-position input,
 * and the conformer {@link Block} with its relative-position attention and depthwise conv module. Static methods
 * and immutable weight holders; package-private executor support.
 */
final class ConformerOps {

    /** Stride-2 convs of the subsampling stem: eight frames per output row. */
    static final int STEM_STRIDES = 3;
    private static final int STEM_KERNEL = 3;
    private static final float HALF_STEP = 0.5f;

    private ConformerOps() {
    }

    /** Rows the stem leaves of {@code n}: three kernel-3, stride-2, padding-1 convs. */
    static int subsampled(int n) {
        int h = n;
        for (int i = 0; i < STEM_STRIDES; i++) {
            h = AudioOps.convOut(h, STEM_KERNEL, 2, 1);
        }
        return h;
    }

    /**
     * The relative-position input of {@code rows} rows and {@code width} columns: row {@code j} encodes the
     * distance {@code (rows - 1) / 2 - j} as interleaved {@code sin} (even) / {@code cos} (odd) of
     * {@code (float) (distance * freq[i])}.
     */
    static float[] posEmb(int rows, int width, double[] freq) {
        int center = (rows - 1) / 2;
        float[] out = new float[rows * width];
        for (int j = 0; j < rows; j++) {
            int dist = center - j;
            for (int i = 0; i < width / 2; i++) {
                float ang = (float) (dist * freq[i]);
                out[j * width + 2 * i] = (float) Math.sin(ang);
                out[j * width + 2 * i + 1] = (float) Math.cos(ang);
            }
        }
        return out;
    }

    /**
     * Relative-position attention of {@code [n][heads * dh]} rows: query {@code i} scores key {@code k} as
     * {@code ((q + u) . k + (q + v) . p[center - (i - k)]) * scale} over the keys within {@code center} of it,
     * where {@code p} holds the projected position rows and {@code u} / {@code v} the {@code [dh, heads]} biases.
     * This is the transformer-XL shift of conformer.cpp and parakeet.cpp: it reads the position row of the
     * distance {@code i - k}.
     */
    static float[] relPosAttend(float[] q, float[] k, float[] v, float[] p, float[] u, float[] vb, int n, int heads,
                                int dh, int center) {
        int width = heads * dh;
        float scale = (float) (1.0 / Math.sqrt(dh));
        float[] out = new float[n * width];
        CpuParallel.parallelFor(heads * n, Math.max(1, 4096 / Math.max(1, n)), (from, to) -> {
            float[] scores = new float[n];
            float[] qu = new float[dh];
            float[] qv = new float[dh];
            for (int idx = from; idx < to; idx++) {
                int h = idx / n;
                int i = idx - h * n;
                int base = i * width + h * dh;
                for (int d = 0; d < dh; d++) {
                    qu[d] = q[base + d] + u[h * dh + d];
                    qv[d] = q[base + d] + vb[h * dh + d];
                }
                int lo = Math.max(0, i - center);
                int hi = Math.min(n, i + center + 1);
                for (int t = lo; t < hi; t++) {
                    float ac = VisionRows.dot(qu, 0, k, t * width + h * dh, dh);
                    float bd = VisionRows.dot(qv, 0, p, (center - (i - t)) * width + h * dh, dh);
                    scores[t] = (ac + bd) * scale;
                }
                softmaxInto(scores, lo, hi, v, width, h * dh, dh, out, base);
            }
        });
        return out;
    }

    /** Softmax of {@code scores[lo, hi)} and the weighted sum of the value rows into {@code out[outBase]}. */
    private static void softmaxInto(float[] scores, int lo, int hi, float[] v, int width, int off, int dh,
                                    float[] out, int outBase) {
        float max = Float.NEGATIVE_INFINITY;
        for (int t = lo; t < hi; t++) {
            max = Math.max(max, scores[t]);
        }
        double sum = 0;
        for (int t = lo; t < hi; t++) {
            double e = Math.exp(scores[t] - max);
            scores[t] = (float) e;
            sum += e;
        }
        float inv = (float) (1.0 / sum);
        for (int t = lo; t < hi; t++) {
            float w = scores[t] * inv;
            int vBase = t * width + off;
            for (int d = 0; d < dh; d++) {
                out[outBase + d] += w * v[vBase + d];
            }
        }
    }

    /** The bias of {@code name} when the file has it, else null. */
    private static float[] optional(TensorCatalog tc, Function<String, float[]> floats, String name) {
        return tc.get(name) == null ? null : floats.apply(name);
    }

    /**
     * The conv subsampling stem {@code a.conv1d.{0,2,3,5,6}} and {@code a.pre_encode.out}: over the transposed
     * mel image ({@code [mel, frames]}), a full 3x3 stride-2 conv and ReLU, a depthwise 3x3 stride-2 conv, a 1x1
     * conv and ReLU, a depthwise 3x3 stride-2 conv, a 1x1 conv and ReLU (every conv with its {@code [1, 1, C]}
     * bias), then the {@code [mel / 8 * C]} flatten (mel bins minor) through the biased output projection.
     */
    static final class Stem {
        private final int nMel;
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

        private Stem(int nMel, Function<String, float[]> floats, Function<String, CpuTensorRef> refs, int[] c) {
            this.nMel = nMel;
            this.conv0 = CpuVitTower.matrix(refs.apply(name(0, "weight")), c[0], STEM_KERNEL * STEM_KERNEL);
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

        /** The channels after conv 0, conv 3 and conv 6. */
        private static int[] channels(TensorCatalog tc) {
            return new int[]{(int) tc.require(name(0, "weight")).ne(3), (int) tc.require(name(3, "weight")).ne(3),
                    (int) tc.require(name(6, "weight")).ne(3)};
        }

        /** Shape-checks the convs, their {@code [1, 1, C]} biases and the {@code [mel / 8 * C, nEmbd]} output. */
        static void validate(TensorCatalog tc, int nMel, int nEmbd) {
            int[] c = channels(tc);
            int k = STEM_KERNEL;
            shape(tc, tc.require(name(0, "weight")), k, k, 1, c[0]);
            shape(tc, tc.require(name(2, "weight")), k, k, 1, c[0]);
            shape(tc, tc.require(name(3, "weight")), 1, 1, c[0], c[1]);
            shape(tc, tc.require(name(5, "weight")), k, k, 1, c[1]);
            shape(tc, tc.require(name(6, "weight")), 1, 1, c[1], c[2]);
            shape(tc, tc.require(name(0, "bias")), 1, 1, c[0]);
            shape(tc, tc.require(name(2, "bias")), 1, 1, c[0]);
            shape(tc, tc.require(name(3, "bias")), 1, 1, c[1]);
            shape(tc, tc.require(name(5, "bias")), 1, 1, c[1]);
            shape(tc, tc.require(name(6, "bias")), 1, 1, c[2]);
            shape(tc, tc.require("a.pre_encode.out.weight"), (long) subsampled(nMel) * c[2], nEmbd);
            shape(tc, tc.require("a.pre_encode.out.bias"), nEmbd);
        }

        static Stem bind(TensorCatalog tc, int nMel, Function<String, float[]> floats,
                         Function<String, CpuTensorRef> refs) {
            return new Stem(nMel, floats, refs, channels(tc));
        }

        /** The stem over the mel-major chunk: {@code [subsampled(nLen)][nEmbd]} rows. */
        float[] run(CpuMatVec matVec, MelChunk chunk) {
            if (chunk.nMel() != nMel) {
                throw new IllegalArgumentException("mel chunk has " + chunk.nMel() + " bins, the encoder needs "
                        + nMel);
            }
            int w = nMel;
            int h = chunk.nLen();
            // the transposed mel is the planar [mel, frames, 1] image
            float[] img = AudioOps.transpose(chunk.data(), nMel, h);
            int k = STEM_KERNEL;
            img = AudioOps.conv2d(matVec, img, w, h, 1, conv0, k, k, 2, 1, conv0B);
            AudioOps.activate(ActOp.RELU, img);
            w = AudioOps.convOut(w, k, 2, 1);
            h = AudioOps.convOut(h, k, 2, 1);
            int c = (int) conv0.rows();
            img = AudioOps.depthwiseConv2d(img, w, h, c, dw2, k, k, 2, 1, dw2B);
            w = AudioOps.convOut(w, k, 2, 1);
            h = AudioOps.convOut(h, k, 2, 1);
            img = AudioOps.conv2d(matVec, img, w, h, c, pw3, 1, 1, 1, 0, pw3B);
            AudioOps.activate(ActOp.RELU, img);
            c = (int) pw3.rows();
            img = AudioOps.depthwiseConv2d(img, w, h, c, dw5, k, k, 2, 1, dw5B);
            w = AudioOps.convOut(w, k, 2, 1);
            h = AudioOps.convOut(h, k, 2, 1);
            img = AudioOps.conv2d(matVec, img, w, h, c, pw6, 1, 1, 1, 0, pw6B);
            AudioOps.activate(ActOp.RELU, img);
            c = (int) pw6.rows();
            return AudioOps.linear(matVec, out, outB, AudioOps.flattenTimeOnH(img, w, h, c), h);
        }
    }

    /**
     * The tensor layout of one family's blocks: lfm2a reads biased q/k/v/out, conv and second-FFN projections
     * (first-FFN and block-norm biases when present) and a plain affine conv norm; parakeet reads no projection
     * bias and a batch-norm conv norm over {@code conv_norm_mean} / {@code conv_norm_var}.
     */
    enum Layout {
        /** conformer.cpp. */
        LFM2A,
        /** parakeet.cpp. */
        PARAKEET
    }

    /**
     * One conformer block {@code a.blk.N}: half-step SiLU FFN, relative-position attention, depthwise conv
     * module, half-step SiLU FFN, each residual and pre-normed, then the block output norm.
     */
    static final class Block {
        private final int width;
        private final int heads;
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
        private final float[] qB;
        private final CpuTensorRef k;
        private final float[] kB;
        private final CpuTensorRef v;
        private final float[] vB;
        private final CpuTensorRef o;
        private final float[] oB;
        private final float[] posBiasU;
        private final float[] posBiasV;
        private final CpuTensorRef linearPos;
        private final float[] normConvW;
        private final float[] normConvB;
        private final CpuTensorRef pw1;
        private final float[] pw1B;
        private final float[] dw;
        private final float[] dwB;
        private final int kernel;
        private final float[] bnMean;
        private final float[] bnVar;
        private final float[] convNormW;
        private final float[] convNormB;
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

        private Block(TensorCatalog tc, String p, Layout layout, int heads, int kernel, float eps,
                      Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
            boolean lfm = layout == Layout.LFM2A;
            Function<String, float[]> bias = lfm ? n -> floats.apply(p + n) : n -> null;
            Function<String, float[]> opt = n -> optional(tc, floats, p + n);
            this.heads = heads;
            this.kernel = kernel;
            this.eps = eps;
            this.ffNormW = floats.apply(p + "ffn_norm.weight");
            this.ffNormB = floats.apply(p + "ffn_norm.bias");
            this.ffUp = refs.apply(p + "ffn_up.weight");
            this.ffUpB = lfm ? opt.apply("ffn_up.bias") : null;
            this.ffDown = refs.apply(p + "ffn_down.weight");
            this.ffDownB = lfm ? opt.apply("ffn_down.bias") : null;
            this.ln1W = floats.apply(p + "ln1.weight");
            this.ln1B = lfm ? opt.apply("ln1.bias") : floats.apply(p + "ln1.bias");
            this.q = refs.apply(p + "attn_q.weight");
            this.qB = bias.apply("attn_q.bias");
            this.k = refs.apply(p + "attn_k.weight");
            this.kB = bias.apply("attn_k.bias");
            this.v = refs.apply(p + "attn_v.weight");
            this.vB = bias.apply("attn_v.bias");
            this.o = refs.apply(p + "attn_out.weight");
            this.oB = bias.apply("attn_out.bias");
            this.width = (int) q.rows();
            this.posBiasU = floats.apply(p + "pos_bias_u");
            this.posBiasV = floats.apply(p + "pos_bias_v");
            this.linearPos = refs.apply(p + "linear_pos.weight");
            this.normConvW = floats.apply(p + "norm_conv.weight");
            this.normConvB = floats.apply(p + "norm_conv.bias");
            this.pw1 = refs.apply(p + "conv_pw1.weight");
            this.pw1B = bias.apply("conv_pw1.bias");
            this.dw = floats.apply(p + "conv_dw.weight");
            this.dwB = bias.apply("conv_dw.bias");
            this.bnMean = lfm ? null : floats.apply(p + "conv_norm_mean");
            this.bnVar = lfm ? null : floats.apply(p + "conv_norm_var");
            this.convNormW = floats.apply(p + "conv_norm.weight");
            this.convNormB = floats.apply(p + "conv_norm.bias");
            this.pw2 = refs.apply(p + "conv_pw2.weight");
            this.pw2B = bias.apply("conv_pw2.bias");
            this.ffNorm1W = floats.apply(p + "ffn_norm_1.weight");
            this.ffNorm1B = floats.apply(p + "ffn_norm_1.bias");
            this.ffUp1 = refs.apply(p + "ffn_up_1.weight");
            this.ffUp1B = bias.apply("ffn_up_1.bias");
            this.ffDown1 = refs.apply(p + "ffn_down_1.weight");
            this.ffDown1B = bias.apply("ffn_down_1.bias");
            this.ln2W = floats.apply(p + "ln2.weight");
            this.ln2B = lfm ? opt.apply("ln2.bias") : floats.apply(p + "ln2.bias");
        }

        private static String prefix(int il) {
            return "a.blk." + il + ".";
        }

        /**
         * Shape-checks block {@code il} of width {@code e} with {@code heads} heads, a {@code posWidth}-wide
         * relative-position input and a depthwise kernel of {@code kernel} taps.
         */
        static void validate(TensorCatalog tc, Layout layout, int il, int e, int heads, int posWidth, int kernel) {
            String p = prefix(il);
            boolean lfm = layout == Layout.LFM2A;
            norm(tc, p + "ffn_norm", e, true);
            ffn(tc, p, "", e, lfm, false);
            norm(tc, p + "ln1", e, !lfm);
            for (String t : new String[]{"attn_q", "attn_k", "attn_v", "attn_out"}) {
                shape(tc, tc.require(p + t + ".weight"), e, e);
                bias(tc, p + t + ".bias", e, lfm);
            }
            shape(tc, tc.require(p + "pos_bias_u"), e / heads, heads);
            shape(tc, tc.require(p + "pos_bias_v"), e / heads, heads);
            shape(tc, tc.require(p + "linear_pos.weight"), posWidth, e);
            norm(tc, p + "norm_conv", e, true);
            shape(tc, tc.require(p + "conv_pw1.weight"), e, 2L * e);
            bias(tc, p + "conv_pw1.bias", 2 * e, lfm);
            shape(tc, tc.require(p + "conv_dw.weight"), kernel, e);
            bias(tc, p + "conv_dw.bias", e, lfm);
            if (!lfm) {
                shape(tc, tc.require(p + "conv_norm_mean"), e);
                shape(tc, tc.require(p + "conv_norm_var"), e);
            }
            norm(tc, p + "conv_norm", e, true);
            shape(tc, tc.require(p + "conv_pw2.weight"), e, e);
            bias(tc, p + "conv_pw2.bias", e, lfm);
            norm(tc, p + "ffn_norm_1", e, true);
            ffn(tc, p, "_1", e, lfm, lfm);
            norm(tc, p + "ln2", e, !lfm);
        }

        /** A norm weight of {@code n} values and its bias, required or checked when present. */
        private static void norm(TensorCatalog tc, String base, int n, boolean biasRequired) {
            shape(tc, tc.require(base + ".weight"), n);
            if (biasRequired) {
                shape(tc, tc.require(base + ".bias"), n);
            } else {
                checkVector(tc, tc.get(base + ".bias"), n);
            }
        }

        /** A projection bias the layout reads: required when {@code read}, unread otherwise. */
        private static void bias(TensorCatalog tc, String name, int n, boolean read) {
            if (read) {
                shape(tc, tc.require(name), n);
            }
        }

        /**
         * The FFN {@code ffn_up<s>} / {@code ffn_down<s>} around its own hidden width: biases checked when
         * {@code read} (required when {@code required}).
         */
        private static void ffn(TensorCatalog tc, String p, String s, int e, boolean read, boolean required) {
            TensorRef up = tc.require(p + "ffn_up" + s + ".weight");
            long ff = up.ne(1);
            shape(tc, up, e, ff);
            shape(tc, tc.require(p + "ffn_down" + s + ".weight"), ff, e);
            if (required) {
                shape(tc, tc.require(p + "ffn_up" + s + ".bias"), ff);
                shape(tc, tc.require(p + "ffn_down" + s + ".bias"), e);
            } else if (read) {
                checkVector(tc, tc.get(p + "ffn_up" + s + ".bias"), (int) ff);
                checkVector(tc, tc.get(p + "ffn_down" + s + ".bias"), e);
            }
        }

        static Block bind(TensorCatalog tc, Layout layout, int il, int heads, int kernel, float eps,
                          Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
            return new Block(tc, prefix(il), layout, heads, kernel, eps, floats, refs);
        }

        /**
         * The block over {@code [n][width]} rows of {@code x}, in place, with the {@code posRows} rows of the
         * relative-position input {@code posEmb} (row {@code j} at distance {@code (posRows - 1) / 2 - j}).
         */
        void forward(CpuMatVec matVec, float[] x, int n, float[] posEmb, int posRows) {
            float[] h = normed(x, n, ffNormW, ffNormB);
            AudioOps.addScaled(x, AudioOps.ffn(matVec, h, n, ffUp, ffUpB, ActOp.SILU, ffDown, ffDownB), HALF_STEP);
            h = normed(x, n, ln1W, ln1B);
            AudioOps.addScaled(x, attention(matVec, h, n, posEmb, posRows), 1f);
            h = normed(x, n, normConvW, normConvB);
            AudioOps.addScaled(x, convModule(matVec, h, n), 1f);
            h = normed(x, n, ffNorm1W, ffNorm1B);
            AudioOps.addScaled(x, AudioOps.ffn(matVec, h, n, ffUp1, ffUp1B, ActOp.SILU, ffDown1, ffDown1B),
                    HALF_STEP);
            AudioOps.layerNorm(x, n, width, ln2W, ln2B, eps);
        }

        private float[] normed(float[] x, int n, float[] w, float[] b) {
            float[] h = x.clone();
            AudioOps.layerNorm(h, n, width, w, b, eps);
            return h;
        }

        private float[] attention(CpuMatVec matVec, float[] h, int n, float[] posEmb, int posRows) {
            float[] qr = AudioOps.linear(matVec, q, qB, h, n);
            float[] kr = AudioOps.linear(matVec, k, kB, h, n);
            float[] vr = AudioOps.linear(matVec, v, vB, h, n);
            float[] pos = AudioOps.linear(matVec, linearPos, null, posEmb, posRows);
            float[] a = relPosAttend(qr, kr, vr, pos, posBiasU, posBiasV, n, heads, width / heads,
                    (posRows - 1) / 2);
            return AudioOps.linear(matVec, o, oB, a, n);
        }

        /**
         * pw1, the sigmoid GLU, the centered depthwise conv along time (the {@code ggml_ssm_conv} over
         * {@code (kernel - 1) / 2} zero frames each side), the batch norm when the layout has one, the affine
         * {@code conv_norm}, SiLU and pw2.
         */
        private float[] convModule(CpuMatVec matVec, float[] h, int n) {
            float[] g = AudioOps.sigmoidGlu(AudioOps.linear(matVec, pw1, pw1B, h, n), n, width);
            int pad = (kernel - 1) / 2;
            float[] c = AudioOps.depthwiseConv1d(g, n, width, dw, kernel, pad, pad);
            VisionRows.addBias(c, dwB);
            for (int i = 0; i < c.length; i++) {
                int ch = i % width;
                float val = c[i];
                if (bnMean != null) {
                    val = (val - bnMean[ch]) / (float) Math.sqrt(bnVar[ch]);
                }
                c[i] = val * convNormW[ch] + convNormB[ch];
            }
            AudioOps.activate(ActOp.SILU, c);
            return AudioOps.linear(matVec, pw2, pw2B, c, n);
        }
    }
}
