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
 * The Gemma 4 audio conformer and its multimodal embedder (llama.cpp tools/mtmd/models/gemma4a.cpp,
 * {@code clip_graph_gemma4a}): over the transposed mel, two 3x3 stride-2 convs, each with its optional bias and
 * channel layer norm and a ReLU, the {@code [freq x channel]} flatten (channel minor) through
 * {@code a.input_projection}; then the blocks, every norm an RMS norm at 1e-6 — a half-step SiLU FFN with its
 * optional post-norm, the chunked local attention (12-row query chunks, each row seeing itself and the 11 rows
 * before it, Q scaled by {@code 1 / (sqrt(dh) ln 2)} and the optional {@code per_dim_scale}, K by
 * {@code ln(1 + e) / ln 2} and the optional {@code per_dim_k_scale}, the optional sinusoidal relative-position
 * term through {@code attn_k_rel}, a tanh softcap of 50), the causal kernel-5 conv module, a second half-step FFN
 * and the optional block norm; then the optional output projection {@code a.pre_encode.out}, an RMS norm with the
 * optional {@code mm.a.soft_emb_norm} and {@code mm.a.input_projection}. Every {@code build_mm} projection honors
 * the ClippableLinear bounds {@code .input_min/.input_max/.output_min/.output_max} the file carries.
 * Immutable weights; package-private executor support.
 */
final class Gemma4aTower implements FamilyEncoder {

    /** Query rows per attention chunk and the past rows each row sees (gemma4a.cpp). */
    static final int CHUNK = 12;
    static final int MAX_PAST = 12;
    /** Taps of the causal conv module: the graph pads four frames on the left only. */
    static final int CONV_KERNEL = 5;
    private static final int RPE_ROWS = MAX_PAST + 1;
    private static final float SOFTCAP = 50f;
    private static final float HALF_STEP = 0.5f;
    private static final float EPS = 1e-6f;
    private static final int STEM_CONVS = 2;

    private final int nMel;
    private final int nEmbd;
    private final int outWidth;
    private final CpuMatVec matVec;
    private final CpuTensorRef[] conv;
    private final float[][] convB;
    private final float[][] convNorm;
    private final Linear inProj;
    private final Block[] blocks;
    private final Linear outProj;
    private final float[] softEmbNorm;
    private final Linear embedProj;

    private Gemma4aTower(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                         Function<String, CpuTensorRef> refs) {
        ClipHparams hp = model.hparams();
        TensorCatalog tc = model.catalog();
        this.nMel = hp.audio().nMelBins();
        this.nEmbd = hp.nEmbd();
        this.outWidth = model.outputWidth();
        this.matVec = matVec;
        this.conv = new CpuTensorRef[STEM_CONVS];
        this.convB = new float[STEM_CONVS][];
        this.convNorm = new float[STEM_CONVS][];
        int cIn = 1;
        for (int i = 0; i < STEM_CONVS; i++) {
            TensorRef w = tc.require(stem(i, "weight"));
            conv[i] = CpuVitTower.matrix(refs.apply(w.name()), w.ne(3), 9L * cIn);
            convB[i] = optional(tc, floats, stem(i, "bias"));
            convNorm[i] = optional(tc, floats, "a.conv1d." + i + ".norm.weight");
            cIn = (int) w.ne(3);
        }
        this.inProj = Linear.of(tc, "a.input_projection", true, floats, refs);
        float[] posEmb = posEmb(nEmbd);
        this.blocks = new Block[hp.nLayer()];
        for (int il = 0; il < blocks.length; il++) {
            blocks[il] = new Block(tc, hp, il, matVec, posEmb, floats, refs);
        }
        this.outProj = tc.has("a.pre_encode.out.weight") ? Linear.of(tc, "a.pre_encode.out", true, floats, refs)
                : null;
        this.softEmbNorm = optional(tc, floats, "mm.a.soft_emb_norm.weight");
        this.embedProj = Linear.of(tc, "mm.a.input_projection", false, floats, refs);
    }

    private static String stem(int i, String suffix) {
        return "a.conv1d." + i + "." + suffix;
    }

    private static float[] optional(TensorCatalog tc, Function<String, float[]> floats, String name) {
        return tc.get(name) == null ? null : floats.apply(name);
    }

    /**
     * The 13 sinusoidal relative-position rows clip.cpp fills: row {@code p} at position {@code 12 - p}, the
     * sines in the first half of the row and the cosines in the second, timescales
     * {@code expf(-i * logf(1e4) / (e / 2 - 1))}, all in float.
     */
    static float[] posEmb(int e) {
        int half = e / 2;
        float inc = (float) Math.log(10000.0f) / Math.max(half - 1, 1);
        float[] out = new float[RPE_ROWS * e];
        for (int p = 0; p < RPE_ROWS; p++) {
            float position = MAX_PAST - p;
            for (int i = 0; i < half; i++) {
                float scaled = position * (float) Math.exp(-(float) i * inc);
                out[p * e + i] = (float) Math.sin(scaled);
                out[p * e + i + half] = (float) Math.cos(scaled);
            }
        }
        return out;
    }

    /** Rows two stride-2 kernel-3 padding-1 convs leave of {@code n} frames. */
    static int subsampled(int n) {
        int h = n;
        for (int i = 0; i < STEM_CONVS; i++) {
            h = AudioOps.convOut(h, 3, 2, 1);
        }
        return h;
    }

    /** Shape-checks the stem, the blocks, the output projection and the embedder against the hparams. */
    static void validate(ClipHparams hp, TensorCatalog tc) {
        int e = hp.nEmbd();
        if (e % hp.nHead() != 0) {
            throw new IllegalArgumentException("the gemma4a attention splits clip.audio.embedding_length " + e
                    + " over " + hp.nHead() + " heads unevenly");
        }
        int cIn = 1;
        int f = hp.audio().nMelBins();
        for (int i = 0; i < STEM_CONVS; i++) {
            TensorRef w = tc.require(stem(i, "weight"));
            int c = (int) w.ne(3);
            shape(tc, w, 3, 3, cIn, c);
            TensorRef b = tc.get(stem(i, "bias"));
            if (b != null) {
                shape(tc, b, 1, 1, c);
            }
            checkVector(tc, tc.get("a.conv1d." + i + ".norm.weight"), c);
            cIn = c;
            f = AudioOps.convOut(f, 3, 2, 1);
        }
        Linear.validate(tc, "a.input_projection", true, (long) f * cIn, e);
        for (int il = 0; il < hp.nLayer(); il++) {
            Block.validate(tc, hp, il);
        }
        if (tc.has("a.pre_encode.out.weight")) {
            Linear.validate(tc, "a.pre_encode.out", true, e, e);
        }
        checkVector(tc, tc.get("mm.a.soft_emb_norm.weight"), e);
        Linear.validate(tc, "mm.a.input_projection", false, e, hp.projectionDim());
    }

    /** Binds the executor over {@code model} with the weights of {@code floats} / {@code refs}. */
    static Gemma4aTower bind(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                             Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(matVec, "matVec");
        return new Gemma4aTower(model, matVec, floats, refs);
    }

    /** The stem, the blocks and the embedder: {@link #subsampled}{@code (nLen)} rows. */
    @Override
    public float[][] encode(MelChunk chunk) {
        if (chunk.nMel() != nMel) {
            throw new IllegalArgumentException("mel chunk has " + chunk.nMel() + " bins, the encoder needs " + nMel);
        }
        float[] x = stem(chunk);
        int n = x.length / nEmbd;
        for (Block b : blocks) {
            b.forward(matVec, x, n);
        }
        if (outProj != null) {
            x = outProj.apply(matVec, x, n);
        }
        AudioOps.rmsNorm(x, n, nEmbd, softEmbNorm, null, EPS);
        return VisionRows.tokenRows(embedProj.apply(matVec, x, n), outWidth);
    }

    /** The two convs over the transposed mel ({@code [mel, frames]}), the flatten and the input projection. */
    private float[] stem(MelChunk chunk) {
        int w = nMel;
        int h = chunk.nLen();
        int c = 1;
        float[] img = AudioOps.transpose(chunk.data(), nMel, h);
        for (int i = 0; i < STEM_CONVS; i++) {
            img = AudioOps.conv2d(matVec, img, w, h, c, conv[i], 3, 3, 2, 1, convB[i]);
            w = AudioOps.convOut(w, 3, 2, 1);
            h = AudioOps.convOut(h, 3, 2, 1);
            c = (int) conv[i].rows();
            if (convNorm[i] != null) {
                channelNorm(img, w * h, c, convNorm[i]);
            }
            AudioOps.activate(ActOp.RELU, img);
        }
        // [freq, time, ch] -> rows of [ch, freq]: feature ch + C * f
        float[] rows = new float[h * w * c];
        for (int t = 0; t < h; t++) {
            for (int f = 0; f < w; f++) {
                for (int ch = 0; ch < c; ch++) {
                    rows[(t * w + f) * c + ch] = img[f + w * (t + h * ch)];
                }
            }
        }
        return inProj.apply(matVec, rows, h);
    }

    /** The bias-free layer norm over the channels of every pixel of a planar image, times {@code w}. */
    private static void channelNorm(float[] img, int pixels, int c, float[] w) {
        float[] px = new float[c];
        for (int p = 0; p < pixels; p++) {
            for (int ch = 0; ch < c; ch++) {
                px[ch] = img[p + pixels * ch];
            }
            AudioOps.layerNorm(px, 1, c, w, null, EPS);
            for (int ch = 0; ch < c; ch++) {
                img[p + pixels * ch] = px[ch];
            }
        }
    }

    /**
     * A {@code build_mm} projection of gemma4a.cpp, with the optional bias where the graph adds one: the input
     * clamped to
     * {@code [input_min, input_max]} and the product to {@code [output_min, output_max]} when the file carries
     * those scalars (the ClippableLinear bounds), then the bias.
     */
    static final class Linear {
        private final CpuTensorRef w;
        private final float[] b;
        private final float[] bounds;

        private Linear(CpuTensorRef w, float[] b, float[] bounds) {
            this.w = w;
            this.b = b;
            this.bounds = bounds;
        }

        private static final String[] BOUNDS = {"input_min", "input_max", "output_min", "output_max"};
        private static final float[] DEFAULTS = {-Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE,
                Float.MAX_VALUE};

        /** {@code base.weight}, with {@code biased} {@code base.bias} when present, and the bounds of {@code base.*}. */
        static Linear of(TensorCatalog tc, String base, boolean biased, Function<String, float[]> floats,
                         Function<String, CpuTensorRef> refs) {
            float[] bounds = DEFAULTS.clone();
            boolean any = false;
            for (int i = 0; i < BOUNDS.length; i++) {
                String name = base + "." + BOUNDS[i];
                if (tc.has(name)) {
                    bounds[i] = floats.apply(name)[0];
                    any = true;
                }
            }
            return new Linear(refs.apply(base + ".weight"), biased ? optional(tc, floats, base + ".bias") : null,
                    any ? bounds : null);
        }

        /** Checks {@code base.weight} as {@code [in, out]}, the bias the graph reads and one-value bounds. */
        static void validate(TensorCatalog tc, String base, boolean biased, long in, long out) {
            shape(tc, tc.require(base + ".weight"), in, out);
            if (biased) {
                checkVector(tc, tc.get(base + ".bias"), (int) out);
            }
            for (String bound : BOUNDS) {
                TensorRef r = tc.get(base + "." + bound);
                if (r != null) {
                    ClipShapes.elements(tc, r, 1);
                }
            }
        }

        float[] apply(CpuMatVec matVec, float[] x, int n) {
            if (bounds == null) {
                return AudioOps.linear(matVec, w, b, x, n);
            }
            float[] in = x.clone();
            clamp(in, bounds[0], bounds[1]);
            float[] y = AudioOps.linear(matVec, w, null, in, n);
            clamp(y, bounds[2], bounds[3]);
            VisionRows.addBias(y, b);
            return y;
        }

        private static void clamp(float[] x, float lo, float hi) {
            for (int i = 0; i < x.length; i++) {
                x[i] = Math.max(Math.min(x[i], hi), lo);
            }
        }
    }

    /** One gemma4a block {@code a.blk.N}; every operator present in the file runs, in graph order. */
    private static final class Block {
        private final int width;
        private final int heads;
        private final float[] ffNorm;
        private final Linear ffUp;
        private final Linear ffDown;
        private final float[] ffPostNorm;
        private final float[] attnNorm;
        private final Linear q;
        private final Linear k;
        private final Linear v;
        private final Linear o;
        private final float[] qScale;
        private final float[] kScale;
        private final float[] relPos;
        private final float[] attnPostNorm;
        private final float[] convPreNorm;
        private final Linear pw1;
        private final float[] dw;
        private final float[] dwB;
        private final float[] convPostNorm;
        private final Linear pw2;
        private final float[] ffNorm1;
        private final Linear ffUp1;
        private final Linear ffDown1;
        private final float[] ffPostNorm1;
        private final float[] ln2;

        Block(TensorCatalog tc, ClipHparams hp, int il, CpuMatVec matVec, float[] posEmb,
              Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
            String p = prefix(il);
            Function<String, float[]> opt = name -> optional(tc, floats, p + name);
            Function<String, Linear> lin = name -> Linear.of(tc, p + name, false, floats, refs);
            this.width = hp.nEmbd();
            this.heads = hp.nHead();
            int dh = width / heads;
            this.ffNorm = floats.apply(p + "ffn_norm.weight");
            this.ffUp = lin.apply("ffn_up");
            this.ffDown = lin.apply("ffn_down");
            this.ffPostNorm = opt.apply("ffn_post_norm.weight");
            boolean attn = hasAttention(tc, il);
            float[] pre = opt.apply("attn_pre_norm.weight");
            this.attnNorm = pre != null ? pre : opt.apply("ln1.weight");
            this.q = attn ? lin.apply("attn_q") : null;
            this.k = attn ? lin.apply("attn_k") : null;
            this.v = attn ? lin.apply("attn_v") : null;
            this.o = Linear.of(tc, p + "attn_out", true, floats, refs);
            this.qScale = scales(opt.apply("per_dim_scale.weight"), dh, (1f / (float) Math.sqrt(dh)) / LN2);
            this.kScale = scales(opt.apply("per_dim_k_scale.weight"), dh, (float) Math.log(1f + (float) Math.E) / LN2);
            this.relPos = tc.has(p + "attn_k_rel.weight")
                    ? AudioOps.linear(matVec, refs.apply(p + "attn_k_rel.weight"), null, posEmb, RPE_ROWS) : null;
            this.attnPostNorm = opt.apply("attn_post_norm.weight");
            // gemma4a.cpp reads the conv pre-norm from conv_norm and the post-conv norm from norm_conv
            this.convPreNorm = opt.apply("conv_norm.weight");
            this.pw1 = lin.apply("conv_pw1");
            this.dw = floats.apply(p + "conv_dw.weight");
            this.dwB = opt.apply("conv_dw.bias");
            this.convPostNorm = opt.apply("norm_conv.weight");
            this.pw2 = lin.apply("conv_pw2");
            this.ffNorm1 = floats.apply(p + "ffn_norm_1.weight");
            this.ffUp1 = lin.apply("ffn_up_1");
            this.ffDown1 = lin.apply("ffn_down_1");
            this.ffPostNorm1 = opt.apply("ffn_post_norm_1.weight");
            this.ln2 = opt.apply("ln2.weight");
        }

        private static final float LN2 = (float) Math.log(2.0f);

        private static String prefix(int il) {
            return "a.blk." + il + ".";
        }

        /** The attention runs when q, k and v are all present (attn_out is always required). */
        private static boolean hasAttention(TensorCatalog tc, int il) {
            String p = prefix(il);
            return tc.has(p + "attn_q.weight") && tc.has(p + "attn_k.weight") && tc.has(p + "attn_v.weight");
        }

        /** Per head dim: {@code scale * perDim[d]}, or {@code scale} without the per-dim tensor. */
        private static float[] scales(float[] perDim, int dh, float scale) {
            float[] s = new float[dh];
            for (int d = 0; d < dh; d++) {
                s[d] = perDim == null ? scale : scale * perDim[d];
            }
            return s;
        }

        static void validate(TensorCatalog tc, ClipHparams hp, int il) {
            String p = prefix(il);
            int e = hp.nEmbd();
            int dh = e / hp.nHead();
            for (String n : new String[]{"ffn_norm", "ffn_post_norm", "attn_pre_norm", "ln1", "attn_post_norm",
                    "conv_norm", "norm_conv", "ffn_norm_1", "ffn_post_norm_1", "ln2"}) {
                checkVector(tc, tc.get(p + n + ".weight"), e);
            }
            tc.require(p + "ffn_norm.weight");
            tc.require(p + "ffn_norm_1.weight");
            for (String s : new String[]{"", "_1"}) {
                TensorRef up = tc.require(p + "ffn_up" + s + ".weight");
                Linear.validate(tc, p + "ffn_up" + s, false, e, up.ne(1));
                Linear.validate(tc, p + "ffn_down" + s, false, up.ne(1), e);
            }
            int qkv = (tc.has(p + "attn_q.weight") ? 1 : 0) + (tc.has(p + "attn_k.weight") ? 1 : 0)
                    + (tc.has(p + "attn_v.weight") ? 1 : 0);
            if (qkv != 0 && qkv != 3) {
                throw new IllegalArgumentException("gemma4a layer " + il + " carries only part of attn_q, attn_k"
                        + " and attn_v");
            }
            for (String t : new String[]{"attn_q", "attn_k", "attn_v"}) {
                if (qkv == 3) {
                    Linear.validate(tc, p + t, false, e, e);
                }
            }
            Linear.validate(tc, p + "attn_out", true, e, e);
            checkVector(tc, tc.get(p + "per_dim_scale.weight"), dh);
            checkVector(tc, tc.get(p + "per_dim_k_scale.weight"), dh);
            TensorRef rel = tc.get(p + "attn_k_rel.weight");
            if (rel != null) {
                shape(tc, rel, e, e);
            }
            Linear.validate(tc, p + "conv_pw1", false, e, 2L * e);
            shape(tc, tc.require(p + "conv_dw.weight"), CONV_KERNEL, e);
            checkVector(tc, tc.get(p + "conv_dw.bias"), e);
            Linear.validate(tc, p + "conv_pw2", false, e, e);
        }

        /** The block over {@code [n][width]} rows of {@code x} (the residual stream), in place. */
        void forward(CpuMatVec matVec, float[] x, int n) {
            AudioOps.addScaled(x, ffn(matVec, x, n, ffNorm, ffUp, ffDown, ffPostNorm), HALF_STEP);
            if (q != null) {
                AudioOps.addScaled(x, attention(matVec, x, n), 1f);
            }
            if (convPreNorm != null) {
                AudioOps.addScaled(x, convModule(matVec, x, n), 1f);
            }
            AudioOps.addScaled(x, ffn(matVec, x, n, ffNorm1, ffUp1, ffDown1, ffPostNorm1), HALF_STEP);
            if (ln2 != null) {
                AudioOps.rmsNorm(x, n, width, ln2, null, EPS);
            }
        }

        private float[] rms(float[] x, int n, float[] w) {
            float[] h = x.clone();
            AudioOps.rmsNorm(h, n, width, w, null, EPS);
            return h;
        }

        private float[] ffn(CpuMatVec matVec, float[] x, int n, float[] norm, Linear up, Linear down,
                            float[] post) {
            float[] h = up.apply(matVec, rms(x, n, norm), n);
            AudioOps.activate(ActOp.SILU, h);
            h = down.apply(matVec, h, n);
            if (post != null) {
                AudioOps.rmsNorm(h, n, width, post, null, EPS);
            }
            return h;
        }

        private float[] attention(CpuMatVec matVec, float[] x, int n) {
            float[] h = attnNorm != null ? rms(x, n, attnNorm) : x;
            int dh = width / heads;
            float[] qr = q.apply(matVec, h, n);
            float[] kr = k.apply(matVec, h, n);
            float[] vr = v.apply(matVec, h, n);
            for (int i = 0; i < qr.length; i++) {
                qr[i] *= qScale[i % dh];
                kr[i] *= kScale[i % dh];
            }
            float[] a = chunkedAttend(qr, kr, vr, n, dh);
            float[] out = o.apply(matVec, a, n);
            if (attnPostNorm != null) {
                AudioOps.rmsNorm(out, n, width, attnPostNorm, null, EPS);
            }
            return out;
        }

        /**
         * Row {@code i} attends to rows {@code i - 11 .. i}: score {@code q . k} plus, with {@code attn_k_rel},
         * {@code q . rel[12 - (i - j)]}, softcapped by {@code 50 tanh(s / 50)}.
         */
        private float[] chunkedAttend(float[] qr, float[] kr, float[] vr, int n, int dh) {
            float[] out = new float[n * width];
            CpuParallel.parallelFor(heads * n, 64, (from, to) -> {
                float[] scores = new float[MAX_PAST];
                for (int idx = from; idx < to; idx++) {
                    int hd = idx / n;
                    int i = idx - hd * n;
                    int qBase = i * width + hd * dh;
                    int lo = Math.max(0, i - MAX_PAST + 1);
                    float max = Float.NEGATIVE_INFINITY;
                    for (int j = lo; j <= i; j++) {
                        float s = VisionRows.dot(qr, qBase, kr, j * width + hd * dh, dh);
                        if (relPos != null) {
                            s += VisionRows.dot(qr, qBase, relPos, (MAX_PAST - (i - j)) * width + hd * dh, dh);
                        }
                        s = SOFTCAP * (float) Math.tanh(s / SOFTCAP);
                        scores[j - lo] = s;
                        max = Math.max(max, s);
                    }
                    double sum = 0;
                    for (int j = lo; j <= i; j++) {
                        double ex = Math.exp(scores[j - lo] - max);
                        scores[j - lo] = (float) ex;
                        sum += ex;
                    }
                    float inv = (float) (1.0 / sum);
                    for (int j = lo; j <= i; j++) {
                        float w = scores[j - lo] * inv;
                        int vBase = j * width + hd * dh;
                        for (int d = 0; d < dh; d++) {
                            out[qBase + d] += w * vr[vBase + d];
                        }
                    }
                }
            });
            return out;
        }

        /** pw1, the sigmoid GLU, the causal kernel-5 depthwise conv, the optional RMS norm, SiLU and pw2. */
        private float[] convModule(CpuMatVec matVec, float[] x, int n) {
            float[] g = AudioOps.sigmoidGlu(pw1.apply(matVec, rms(x, n, convPreNorm), n), n, width);
            float[] c = AudioOps.depthwiseConv1d(g, n, width, dw, CONV_KERNEL, CONV_KERNEL - 1, 0);
            VisionRows.addBias(c, dwB);
            if (convPostNorm != null) {
                AudioOps.rmsNorm(c, n, width, convPostNorm, null, EPS);
            }
            AudioOps.activate(ActOp.SILU, c);
            return pw2.apply(matVec, c, n);
        }
    }
}
