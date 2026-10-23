package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.CoefficientParams;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.PostParams;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.PreParams;

/**
 * CPU reference of {@code LLM_HC_PrePostComb}, {@code LLM_HC_Pre} and {@code LLM_HC_Post}: pure static math over
 * {@code float[]} with explicit offsets, using the kernel index formulas and the layouts of {@link HcKernels}
 * (stream {@code X[(t*hc + h)*n_embd + i]}, {@code pre/post[t*hc + h]}, {@code comb[t*hc*hc + dst + hc*src]}).
 * Arithmetic follows ggml-cpu/ops.cpp:11427-11733 in single precision and in the same summation order.
 */
public final class CpuHyperConnections {

    /** Minimum tokens per parallel range of the element-wise ops. */
    private static final int TOKEN_GRAIN = 1;

    private CpuHyperConnections() {
    }

    /**
     * {@code LLM_HC_PrePostComb} for tokens {@code [t0, t1)}. The mixes row of token t starts at
     * {@code mixes[mixOff + t*p.mixDim()]}; {@code scale} holds {@code p.scaleCount()} values and {@code base}
     * {@code p.baseCount()}. Outputs: {@code pre[preOff + t*hc + h]}, {@code post[postOff + t*hc + h]} (when
     * {@code nOut >= 2}) and {@code comb[combOff + t*hc*hc + dst + hc*src]} (when {@code nOut == 3}); an output the
     * variant does not produce may be null.
     *
     * @throws IllegalArgumentException when the token range is invalid or an array the variant needs is missing or
     *                                  too short
     */
    public static void coefficients(CoefficientParams p, int t0, int t1, float[] mixes, int mixOff, float[] scale,
                                    float[] base, float[] pre, int preOff, float[] post, int postOff,
                                    float[] comb, int combOff) {
        final int hc = p.hc();
        final int nMix = p.mixDim();
        requireTokenRange(t0, t1);
        requireLength("scale", scale, p.scaleCount());
        requireLength("base", base, p.baseCount());
        requireLength("mixes", mixes, (long) mixOff + (long) t1 * nMix);
        requireLength("pre", pre, (long) preOff + (long) t1 * hc);
        if (p.nOut() >= HcKernels.OUT_PRE_POST) {
            requireLength("post", post, (long) postOff + (long) t1 * hc);
        }
        if (p.nOut() == HcKernels.OUT_PRE_POST_COMB) {
            requireLength("comb", comb, (long) combOff + (long) t1 * hc * hc);
        }
        final float eps = p.eps();
        float[] c = new float[hc * hc];
        for (int t = t0; t < t1; t++) {
            final int m = mixOff + t * nMix;
            for (int h = 0; h < hc; h++) {
                pre[preOff + t * hc + h] = ActMath.sigmoid(mixes[m + h] * scale[0] + base[h]) + eps;
            }
            if (p.nOut() < HcKernels.OUT_PRE_POST) {
                continue;
            }
            for (int h = 0; h < hc; h++) {
                post[postOff + t * hc + h] = p.postMag() * ActMath.sigmoid(mixes[m + hc + h] * scale[1] + base[hc + h])
                        + p.postEps();
            }
            if (p.nOut() < HcKernels.OUT_PRE_POST_COMB) {
                continue;
            }
            combFromLogits(hc, eps, p.iters(), mixes, m + 2 * hc, scale[2], base, 2 * hc, c);
            System.arraycopy(c, 0, comb, combOff + t * hc * hc, hc * hc);
        }
    }

    /**
     * Sinkhorn mixing matrix of one token (ops.cpp:11505-11537): logits
     * {@code mixes[mixOff + idx]*scaleComb + base[baseOff + idx]} with {@code idx = dst + hc*src}; softmax over dst for
     * each src, {@code + eps}, then {@link #sinkhornNormalize}. Writes {@code out[dst + hc*src]}.
     */
    public static void combFromLogits(int hc, float eps, int iters, float[] mixes, int mixOff, float scaleComb,
                                      float[] base, int baseOff, float[] out) {
        for (int src = 0; src < hc; src++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int dst = 0; dst < hc; dst++) {
                final int idx = dst + hc * src;
                final float v = mixes[mixOff + idx] * scaleComb + base[baseOff + idx];
                out[idx] = v;
                max = Math.max(max, v);
            }
            float sum = 0.0f;
            for (int dst = 0; dst < hc; dst++) {
                final int idx = dst + hc * src;
                final float v = (float) Math.exp(out[idx] - max);
                out[idx] = v;
                sum += v;
            }
            final float invSum = 1.0f / sum;
            for (int dst = 0; dst < hc; dst++) {
                final int idx = dst + hc * src;
                out[idx] = out[idx] * invSum + eps;
            }
        }
        sinkhornNormalize(out, 0, hc, eps, iters);
    }

    /**
     * Sinkhorn normalization of {@code c[off + dst + hc*src]}: one {@code norm_cols} (each dst divided by
     * {@code eps + sum over src}), then {@code iters - 1} rounds of {@code norm_rows} (each src divided by
     * {@code eps + sum over dst}) followed by {@code norm_cols} (ops.cpp:11429-11459, 11533-11537).
     */
    public static void sinkhornNormalize(float[] c, int off, int hc, float eps, int iters) {
        normCols(c, off, hc, eps);
        for (int i = 1; i < iters; i++) {
            normRows(c, off, hc, eps);
            normCols(c, off, hc, eps);
        }
    }

    private static void normCols(float[] c, int off, int hc, float eps) {
        for (int dst = 0; dst < hc; dst++) {
            float sum = eps;
            for (int src = 0; src < hc; src++) {
                sum += c[off + dst + hc * src];
            }
            final float invSum = 1.0f / sum;
            for (int src = 0; src < hc; src++) {
                c[off + dst + hc * src] *= invSum;
            }
        }
    }

    private static void normRows(float[] c, int off, int hc, float eps) {
        for (int src = 0; src < hc; src++) {
            float sum = eps;
            for (int dst = 0; dst < hc; dst++) {
                sum += c[off + dst + hc * src];
            }
            final float invSum = 1.0f / sum;
            for (int dst = 0; dst < hc; dst++) {
                c[off + dst + hc * src] *= invSum;
            }
        }
    }

    /** {@link #preRange} over {@code [0, nTokens)}, split over tokens with {@link CpuParallel}. */
    public static void pre(PreParams p, int hc, int nEmbd, int nTokens, float[] x, int xOff, float[] w, int wOff,
                           float[] out, int outOff) {
        CpuParallel.parallelFor(nTokens, TOKEN_GRAIN,
                (from, to) -> preRange(p, hc, nEmbd, from, to, x, xOff, w, wOff, out, outOff));
    }

    /**
     * {@code LLM_HC_Pre} for tokens {@code [t0, t1)} (ops.cpp:11567-11626):
     * {@code out[outOff + t*nEmbd + i] = gscale * sum_h x[xOff + (t*hc + h)*nEmbd + i] * w_h}, with
     * {@code w_h = w[wOff + t*hc + h]} (weights), {@code sigmoid(w[wOff + (t*hc + h)*nEmbd + i])} (gated) or 1 (mean;
     * {@code w} may be null). {@code out} must not overlap {@code x}.
     *
     * @throws IllegalArgumentException when {@code hc} or the token range is invalid or an array is too short
     */
    public static void preRange(PreParams p, int hc, int nEmbd, int t0, int t1, float[] x, int xOff, float[] w,
                                int wOff, float[] out, int outOff) {
        HcKernels.validateStreams(hc);
        requireTokenRange(t0, t1);
        final int mode = p.mode();
        requireLength("x", x, (long) xOff + (long) t1 * hc * nEmbd);
        requireLength("out", out, (long) outOff + (long) t1 * nEmbd);
        if (mode == HcKernels.PRE_MODE_WEIGHTS) {
            requireLength("weights", w, (long) wOff + (long) t1 * hc);
        } else if (mode == HcKernels.PRE_MODE_GATED) {
            requireLength("gate", w, (long) wOff + (long) t1 * hc * nEmbd);
        }
        final float gscale = p.gscale();
        for (int t = t0; t < t1; t++) {
            final int row0 = t * hc;
            final int o = outOff + t * nEmbd;
            for (int i = 0; i < nEmbd; i++) {
                float sum = 0.0f;
                for (int h = 0; h < hc; h++) {
                    final int xi = (row0 + h) * nEmbd + i;
                    final float wv;
                    if (mode == HcKernels.PRE_MODE_WEIGHTS) {
                        wv = w[wOff + row0 + h];
                    } else if (mode == HcKernels.PRE_MODE_GATED) {
                        wv = ActMath.sigmoid(w[wOff + xi]);
                    } else {
                        wv = 1.0f;
                    }
                    sum += x[xOff + xi] * wv;
                }
                out[o + i] = gscale * sum;
            }
        }
    }

    /** {@link #postRange} over {@code [0, nTokens)}, split over tokens with {@link CpuParallel}. */
    public static void post(PostParams p, int hc, int nEmbd, int nTokens, float[] y, int yOff, float[] x, int xOff,
                            float[] post, int postOff, float[] comb, int combOff, float[] out, int outOff) {
        CpuParallel.parallelFor(nTokens, TOKEN_GRAIN, (from, to) -> postRange(p, hc, nEmbd, from, to, y, yOff, x, xOff,
                post, postOff, comb, combOff, out, outOff));
    }

    /**
     * {@code LLM_HC_Post} for tokens {@code [t0, t1)} (ops.cpp:11647-11720):
     * {@code out[outOff + (t*hc + dst)*nEmbd + i] = y[yOff + t*nEmbd + i] * p_dst + R}, where
     * {@code p_dst = p.coefficient(post[postOff + t*hc + dst])} and {@code R} is
     * {@code sum_src x[xOff + (t*hc + src)*nEmbd + i] * comb[combOff + t*hc*hc + dst + hc*src]} with a comb, or
     * {@code x[xOff + (t*hc + dst)*nEmbd + i]} without ({@code comb} may then be null). {@code out} may be {@code x}
     * with {@code outOff == xOff} (in-place stream update).
     *
     * @throws IllegalArgumentException when {@code hc} or the token range is invalid or an array is too short
     */
    public static void postRange(PostParams p, int hc, int nEmbd, int t0, int t1, float[] y, int yOff, float[] x,
                                 int xOff, float[] post, int postOff, float[] comb, int combOff, float[] out,
                                 int outOff) {
        HcKernels.validateStreams(hc);
        requireTokenRange(t0, t1);
        requireLength("y", y, (long) yOff + (long) t1 * nEmbd);
        requireLength("x", x, (long) xOff + (long) t1 * hc * nEmbd);
        requireLength("post", post, (long) postOff + (long) t1 * hc);
        requireLength("out", out, (long) outOff + (long) t1 * hc * nEmbd);
        if (p.hasComb()) {
            requireLength("comb", comb, (long) combOff + (long) t1 * hc * hc);
        }
        final float[] xs = new float[hc];
        final float[] pv = new float[hc];
        for (int t = t0; t < t1; t++) {
            final int row0 = t * hc;
            final int cb = combOff + t * hc * hc;
            for (int dst = 0; dst < hc; dst++) {
                pv[dst] = p.coefficient(post[postOff + row0 + dst]);
            }
            for (int i = 0; i < nEmbd; i++) {
                for (int h = 0; h < hc; h++) {
                    xs[h] = x[xOff + (row0 + h) * nEmbd + i];
                }
                final float yv = y[yOff + t * nEmbd + i];
                for (int dst = 0; dst < hc; dst++) {
                    float sum = yv * pv[dst];
                    if (p.hasComb()) {
                        for (int src = 0; src < hc; src++) {
                            sum += xs[src] * comb[cb + dst + hc * src];
                        }
                    } else {
                        sum += xs[dst];
                    }
                    out[outOff + (row0 + dst) * nEmbd + i] = sum;
                }
            }
        }
    }

    /**
     * Initial hyper-connection stream of each token: the embedding row repeated over the {@code hc} streams
     * ({@code ggml_repeat_4d} of {@code inp} to {@code [n_embd, hc, n_tokens]}, deepseek4.cpp:1241-1242).
     * {@code out[outOff + (t*hc + h)*nEmbd + i] = embd[embdOff + t*nEmbd + i]}; {@code out} must not overlap {@code embd}.
     */
    public static void expandStreams(int hc, int nEmbd, int nTokens, float[] embd, int embdOff, float[] out,
                                     int outOff) {
        HcKernels.validateStreams(hc);
        requireLength("embd", embd, (long) embdOff + (long) nTokens * nEmbd);
        requireLength("out", out, (long) outOff + (long) nTokens * hc * nEmbd);
        for (int t = 0; t < nTokens; t++) {
            for (int h = 0; h < hc; h++) {
                System.arraycopy(embd, embdOff + t * nEmbd, out, outOff + (t * hc + h) * nEmbd, nEmbd);
            }
        }
    }

    /** Rejects a token range {@code [t0, t1)} that is negative or reversed. */
    static void requireTokenRange(int t0, int t1) {
        if (t0 < 0 || t1 < t0) {
            throw new IllegalArgumentException("hyper-connection token range [" + t0 + ", " + t1 + ") is invalid");
        }
    }

    static void requireLength(String name, float[] a, long needed) {
        if (a == null) {
            throw new IllegalArgumentException("hyper-connection " + name + " array is missing");
        }
        if (a.length < needed) {
            throw new IllegalArgumentException("hyper-connection " + name + " array holds " + a.length
                    + " floats, needs " + needed);
        }
    }
}
