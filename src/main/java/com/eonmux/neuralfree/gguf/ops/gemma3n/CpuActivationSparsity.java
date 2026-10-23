package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;

/**
 * CPU twin of {@link Gemma3nKernels#GAUSSIAN_TOPK}: Gemma3n activation sparsity on the FFN gate rows, applied between
 * the gate projection and the GELU (llama.cpp {@code src/models/gemma3n.cpp:215-219} and {@code gaussian_topk} at
 * :389-395). Per row of {@code n} values:
 * <pre>
 * mean   = sum(x) / n
 * std    = sqrt(sum((x - mean)^2) * (1 / (n - 1)))     unbiased: n - 1 denominator (llama.cpp; vLLM uses n)
 * cutoff = mean + std * kStd
 * x      = relu(x - cutoff)                            relu(v) = v > 0 ? v : 0
 * </pre>
 * Arithmetic follows ggml-cpu: element operations in {@code float}, row sums in {@code double} rounded once
 * ({@code ggml_vec_sum_f32}), the mean divided by {@code (float) n} ({@code ggml_compute_forward_mean_f32}). Pure and
 * stateless; the caller owns the array.
 */
public final class CpuActivationSparsity {

    /**
     * Gemma3n standard-deviation multiplier {@code normal_dist.icdf(0.95)}
     * ({@code f_sparsity_std_mul}, llama.cpp src/models/models.h:954).
     */
    public static final float GEMMA3N_STD_MUL = 1.6448533535003662f;

    /** Leading Gemma3n layers that apply activation sparsity ({@code n_layer_sparsity}, models.h:953). */
    public static final int GEMMA3N_SPARSE_LAYERS = 10;

    private CpuActivationSparsity() {
    }

    /**
     * Applies the Gaussian top-k filter in place to rows {@code t < nTokens} of {@code n} floats at
     * {@code xOff + t*xStride}.
     *
     * @param kStd standard-deviation multiplier ({@code FfnSpec.activationSparsityStdMul()})
     * @throws IllegalArgumentException on an invalid shape or multiplier, or an array too short for the rows
     */
    public static void gaussianTopK(int nTokens, int n, int xStride, int xOff, float kStd, float[] x) {
        Gemma3nKernels.checkTopKShape(nTokens, n, xStride, kStd);
        CpuAltUp.requireView("x", x, xOff, xStride, nTokens, n);
        CpuParallel.parallelFor(nTokens, 1, (from, to) -> {
            for (int t = from; t < to; t++) {
                final int base = xOff + t * xStride;
                final float cutoff = cutoff(x, base, n, kStd);
                for (int i = 0; i < n; i++) {
                    final float v = x[base + i] - cutoff;
                    x[base + i] = v > 0.0f ? v : 0.0f;
                }
            }
        });
    }

    /**
     * The row threshold {@code mean + std_unbiased * kStd} of {@code n >= 2} floats at {@code off}.
     *
     * @throws IllegalArgumentException when {@code n < 2} or the span lies outside {@code x}
     */
    public static float cutoff(float[] x, int off, int n, float kStd) {
        if (n < 2) {
            throw new IllegalArgumentException("Gaussian top-k needs a row width n >= 2, got " + n);
        }
        CpuAltUp.requireView("x", x, off, 0, 1, n);
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += x[off + i];
        }
        final float mean = (float) sum / (float) n;
        double sq = 0.0;
        for (int i = 0; i < n; i++) {
            final float d = x[off + i] - mean;
            sq += d * d;
        }
        final float var = (float) sq * (1.0f / (float) (n - 1));
        final float std = (float) Math.sqrt(var);
        return mean + std * kStd;
    }
}
