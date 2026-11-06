package com.eonmux.neuralfree.gguf.cpu.quant;

/**
 * Symmetric int8 quantization of one row of values, the activation format of the fused int8 attention: a scale of
 * {@code max(absmax / 127, 1e-6)} and one signed byte per value, encoded exactly as the per-token-head int8 KV
 * stores encode a cache row ({@code KvBlockCodec#perTokenHeadScale} / {@code #perTokenHeadCode}: the same qmax of
 * 127, the same epsilon and the same round-half-to-even).
 *
 * <p>The fused attention quantizes each query head to this format once per step and multiplies the exact int8 dot
 * product of a query head with a stored key row by the query scale and the row's stored scale, so the score error is
 * the query quantization error alone. Stateless; the input array is only read.</p>
 */
public final class CpuInt8Quantizer {

    /** Largest code magnitude, the qmax of the per-token-head int8 KV type. */
    public static final int QMAX = 127;

    /** Smallest scale of a row, the clamp of {@code KvBlockCodec#PER_TOKEN_HEAD_EPS}. */
    public static final float MIN_SCALE = 1e-6f;

    private CpuInt8Quantizer() {
    }

    /**
     * The scale of {@code x[xOff, xOff + n)}: {@code max(absmax / 127, 1e-6)}. A non-finite absmax falls back to
     * {@link #MIN_SCALE}, which encodes the non-finite values as saturated codes instead of propagating NaNs.
     *
     * @throws IllegalArgumentException for a non-positive count
     */
    public static float scale(float[] x, int xOff, int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("int8 scale needs a positive row length, got " + n);
        }
        float amax = 0.0f;
        for (int i = 0; i < n; i++) {
            float a = Math.abs(x[xOff + i]);
            amax = amax > a ? amax : a;
        }
        float s = amax / QMAX;
        if (!Float.isFinite(s)) {
            return MIN_SCALE;
        }
        return s > MIN_SCALE ? s : MIN_SCALE;
    }

    /**
     * The code of {@code value} under {@code scale}: round to nearest even of {@code value / scale}, clamped to
     * {@code [-127, 127]} and returned as a signed byte value.
     */
    public static int code(float value, float scale) {
        float scaled = value / scale;
        return (int) Math.rint(Math.max(-QMAX, Math.min(QMAX, scaled)));
    }

    /**
     * Encodes {@code x[xOff, xOff + n)} as the codes of {@code value / scale} into {@code dst[dstOff..]}. The scale
     * must be positive; {@link #scale(float[], int, int)} produces one.
     *
     * @throws IllegalArgumentException when {@code scale} is not positive
     */
    public static void quantize(float[] x, int xOff, int n, byte[] dst, int dstOff, float scale) {
        if (!(scale > 0f)) {
            throw new IllegalArgumentException("int8 row scale must be positive, got " + scale);
        }
        for (int i = 0; i < n; i++) {
            dst[dstOff + i] = (byte) code(x[xOff + i], scale);
        }
    }
}
