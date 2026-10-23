package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;

/**
 * CPU twins of the Gemma3n AltUp kernels ({@link Gemma3nKernels#ALTUP_PREDICT}, {@link Gemma3nKernels#ALTUP_CORRECT},
 * {@link Gemma3nKernels#MAGNITUDE_MATCH}, {@link Gemma3nKernels#STREAMS_MEAN}), following llama.cpp
 * {@code src/models/gemma3n.cpp} and the ggml-cpu operator arithmetic. Pure and stateless; the caller owns every
 * array.
 *
 * <p>Streams use the layout of {@link Gemma3nKernels}: element (stream {@code k}, token {@code t}, column {@code e})
 * at {@code off + k*streamStride + t*nEmbd + e}. Token-major views ({@code activated}, {@code ref}, {@code out}) are
 * {@code [nTokens][nEmbd]} at their offset. The aliasing rules are the kernels': an output may coincide with the input
 * it is computed from (same array, same offset); any other overlap within one array is rejected.
 *
 * <p>Arithmetic order: dot products over streams accumulate in {@code float} in ascending stream order
 * ({@code ggml_vec_dot_f32} leftover loop for rows shorter than one SIMD step); row sums of squares accumulate in
 * {@code double} and round once ({@code ggml_vec_sum_f32}); the device kernels reduce in {@code float}.
 *
 * <p>The layer-level pieces that use generic operators are composed by the AltUp extra: router modalities
 * {@code tanh(altup_router . (rms_norm(active) * altup_router_norm * }{@link #routerInputScale}{@code ))}
 * (gemma3n.cpp:404-412), the predict and correct coefficient matvecs, and LAuReL
 * {@code laurel_out = x + rms_norm(laurel_r . (laurel_l . x)) * laurel_post_norm} combined with the attention branch
 * as {@code (attn + laurel_out) * }{@link #ATTN_LAUREL_SCALE} (gemma3n.cpp:200-204, 377-385).
 */
public final class CpuAltUp {

    /** {@code 1 / sqrtf(2)} applied to {@code attn_gated + laurel_out} (gemma3n.cpp:203-204). */
    public static final float ATTN_LAUREL_SCALE = 1.0f / (float) Math.sqrt(2.0);

    /** Elements per parallel task of the element-wise ops. */
    private static final int ELEMENT_GRAIN = 4096;

    private CpuAltUp() {
    }

    /** Router input scale {@code 1 / n_embd} applied after the router RMS norm (gemma3n.cpp:407-408). */
    public static float routerInputScale(int nEmbd) {
        if (nEmbd <= 0) {
            throw new IllegalArgumentException("n_embd must be positive, got " + nEmbd);
        }
        return 1.0f / (float) nEmbd;
    }

    /**
     * AltUp predict (gemma3n.cpp:416-436):
     * {@code pred[j][t][e] = sum_k streams[k][t][e] * coefs[coefsOff + t*nAltup^2 + j*nAltup + k] + streams[j][t][e]},
     * the sum in ascending {@code k}. {@code pred} may be {@code streams} in place.
     *
     * @param coefs the {@code altup_predict_coef . modalities} matvec output, {@code nAltup^2} floats per token
     * @throws IllegalArgumentException on an invalid shape, an array too short for its view, or overlapping views
     */
    public static void predict(int nTokens, int nEmbd, int nAltup, int streamStride,
                               float[] coefs, int coefsOff, float[] streams, int streamsOff,
                               float[] pred, int predOff) {
        Gemma3nKernels.checkStreams(nTokens, nEmbd, nAltup, streamStride);
        final int rowFloats = nTokens * nEmbd;
        final int coefFloats = (int) Gemma3nKernels.elementCount(nTokens, (long) nAltup * nAltup);
        requireView("coefs", coefs, coefsOff, 0, 1, coefFloats);
        requireView("streams", streams, streamsOff, streamStride, nAltup, rowFloats);
        requireView("pred", pred, predOff, streamStride, nAltup, rowFloats);
        requireInPlaceOrDisjoint("pred", pred, predOff, "streams", streams, streamsOff, streamStride, nAltup, rowFloats);
        requireDisjoint("coefs", coefs, coefsOff, 0, 1, coefFloats, "pred", pred, predOff, streamStride, nAltup, rowFloats);

        CpuParallel.parallelFor(rowFloats, ELEMENT_GRAIN, (from, to) -> {
            final float[] s = new float[nAltup];
            for (int idx = from; idx < to; idx++) {
                final int c = coefsOff + (idx / nEmbd) * nAltup * nAltup;
                for (int k = 0; k < nAltup; k++) {
                    s[k] = streams[streamsOff + k * streamStride + idx];
                }
                for (int j = 0; j < nAltup; j++) {
                    float acc = 0.0f;
                    for (int k = 0; k < nAltup; k++) {
                        acc += s[k] * coefs[c + j * nAltup + k];
                    }
                    pred[predOff + j * streamStride + idx] = acc + s[j];
                }
            }
        });
    }

    /**
     * AltUp correct (gemma3n.cpp:441-461):
     * {@code innov = activated[t][e] - pred[act][t][e]},
     * {@code out[j][t][e] = innov * (coefs[coefsOff + t*nAltup + j] + 1) + pred[j][t][e]}.
     * {@code out} may be {@code pred} in place.
     *
     * @param coefs the raw {@code altup_correct_coef . modalities} matvec output, {@code nAltup} floats per token
     *              (the {@code +1} of {@code ggml_scale_bias} is applied here)
     * @throws IllegalArgumentException on an invalid shape or active index, an array too short for its view, or
     *                                  overlapping views
     */
    public static void correct(int nTokens, int nEmbd, int nAltup, int act, int streamStride,
                               float[] coefs, int coefsOff, float[] activated, int activatedOff,
                               float[] pred, int predOff, float[] out, int outOff) {
        Gemma3nKernels.checkStreams(nTokens, nEmbd, nAltup, streamStride);
        Gemma3nKernels.checkActive(act, nAltup);
        final int rowFloats = nTokens * nEmbd;
        final int coefFloats = (int) Gemma3nKernels.elementCount(nTokens, nAltup);
        requireView("coefs", coefs, coefsOff, 0, 1, coefFloats);
        requireView("activated", activated, activatedOff, 0, 1, rowFloats);
        requireView("pred", pred, predOff, streamStride, nAltup, rowFloats);
        requireView("streams_out", out, outOff, streamStride, nAltup, rowFloats);
        requireInPlaceOrDisjoint("streams_out", out, outOff, "pred", pred, predOff, streamStride, nAltup, rowFloats);
        requireDisjoint("coefs", coefs, coefsOff, 0, 1, coefFloats, "streams_out", out, outOff, streamStride, nAltup, rowFloats);
        requireDisjoint("activated", activated, activatedOff, 0, 1, rowFloats,
                "streams_out", out, outOff, streamStride, nAltup, rowFloats);

        CpuParallel.parallelFor(rowFloats, ELEMENT_GRAIN, (from, to) -> {
            for (int idx = from; idx < to; idx++) {
                final int c = coefsOff + (idx / nEmbd) * nAltup;
                final float innov = activated[activatedOff + idx] - pred[predOff + act * streamStride + idx];
                for (int j = 0; j < nAltup; j++) {
                    final float cj = coefs[c + j] + 1.0f;
                    final float p = pred[predOff + j * streamStride + idx];
                    out[outOff + j * streamStride + idx] = innov * cj + p;
                }
            }
        });
    }

    /**
     * Magnitude matching (gemma3n.cpp:117-123, 269-278): for every row {@code (j < nX, t < nTokens)} at
     * {@code xOff + j*xStride + t*nEmbd}, {@code x = (x * |ref[t]|) / |x|} element-wise, with
     * {@code |v| = }{@link #magnitude}. A zero-magnitude row yields inf/NaN exactly as {@code ggml_div}.
     *
     * @throws IllegalArgumentException on an invalid shape, an array too short for its view, overlapping {@code x}
     *                                  rows, or {@code x} overlapping {@code ref}
     */
    public static void magnitudeMatch(int nTokens, int nEmbd, int nX, int xStride,
                                      float[] ref, int refOff, float[] x, int xOff) {
        Gemma3nKernels.checkMagnitudeShape(nTokens, nEmbd, nX, xStride);
        final int rowFloats = nTokens * nEmbd;
        requireView("ref", ref, refOff, 0, 1, rowFloats);
        requireView("x", x, xOff, xStride, nX, rowFloats);
        requireDisjoint("ref", ref, refOff, 0, 1, rowFloats, "x", x, xOff, xStride, nX, rowFloats);

        CpuParallel.parallelFor(nX * nTokens, 1, (from, to) -> {
            for (int row = from; row < to; row++) {
                final int j = row / nTokens;
                final int t = row - j * nTokens;
                final float target = magnitude(ref, refOff + t * nEmbd, nEmbd);
                final int base = xOff + j * xStride + t * nEmbd;
                final float current = magnitude(x, base, nEmbd);
                for (int e = 0; e < nEmbd; e++) {
                    x[base + e] = (x[base + e] * target) / current;
                }
            }
        });
    }

    /**
     * Stream mean (gemma3n.cpp:281-286):
     * {@code out[t][e] = (((streams[0] + streams[1]) + ...) + streams[nAltup-1]) * (1 / nAltup)}, summed in ascending
     * stream order and scaled as {@code ggml_scale}. {@code out} may be stream 0 in place.
     *
     * @throws IllegalArgumentException on an invalid shape, an array too short for its view, or {@code out}
     *                                  overlapping a stream other than in place on stream 0
     */
    public static void streamsMean(int nTokens, int nEmbd, int nAltup, int streamStride,
                                   float[] streams, int streamsOff, float[] out, int outOff) {
        Gemma3nKernels.checkStreams(nTokens, nEmbd, nAltup, streamStride);
        final int rowFloats = nTokens * nEmbd;
        requireView("streams", streams, streamsOff, streamStride, nAltup, rowFloats);
        requireView("out", out, outOff, 0, 1, rowFloats);
        if (!(out == streams && outOff == streamsOff)) {
            requireDisjoint("out", out, outOff, 0, 1, rowFloats, "streams", streams, streamsOff, streamStride, nAltup, rowFloats);
        }
        final float scale = 1.0f / (float) nAltup;

        CpuParallel.parallelFor(rowFloats, ELEMENT_GRAIN, (from, to) -> {
            for (int idx = from; idx < to; idx++) {
                float acc = streams[streamsOff + idx];
                for (int k = 1; k < nAltup; k++) {
                    acc += streams[streamsOff + k * streamStride + idx];
                }
                out[outOff + idx] = acc * scale;
            }
        });
    }

    /**
     * {@code sqrtf(sum_e v[off + e]^2)} over {@code n} floats: squares in {@code float}, summed in {@code double}
     * and rounded once (gemma3n.cpp:314-316 {@code calc_magnitude}).
     *
     * @throws IllegalArgumentException when the span lies outside {@code v}
     */
    public static float magnitude(float[] v, int off, int n) {
        requireView("v", v, off, 0, 1, n);
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            final float a = v[off + i];
            sum += a * a;
        }
        return (float) Math.sqrt((float) sum);
    }

    static void requireView(String name, float[] a, long off, long stride, int count, long len) {
        if (a == null) {
            throw new IllegalArgumentException(name + " array is null");
        }
        if (off < 0 || len < 0) {
            throw new IllegalArgumentException(name + ": offset and length must be non-negative, got offset " + off
                    + ", length " + len);
        }
        if (len == 0) {
            return;
        }
        long end = off + (long) (count - 1) * stride + len;
        if (end > a.length) {
            throw new IllegalArgumentException(name + ": view ends at " + end + " but the array holds " + a.length);
        }
    }

    private static void requireInPlaceOrDisjoint(String outName, float[] out, int outOff, String inName, float[] in,
                                                 int inOff, int stride, int count, int len) {
        if (out == in && outOff != inOff
                && Gemma3nKernels.spansOverlap(outOff, stride, count, len, inOff, stride, count, len)) {
            throw new IllegalArgumentException(outName + " overlaps " + inName
                    + " without coinciding with it (in-place use needs equal offsets)");
        }
    }

    private static void requireDisjoint(String aName, float[] a, int aOff, int aStride, int aCount, int aLen,
                                        String bName, float[] b, int bOff, int bStride, int bCount, int bLen) {
        if (a == b && Gemma3nKernels.spansOverlap(aOff, aStride, aCount, aLen, bOff, bStride, bCount, bLen)) {
            throw new IllegalArgumentException(aName + " overlaps " + bName + " in the same array");
        }
    }
}
