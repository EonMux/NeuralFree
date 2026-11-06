package com.eonmux.neuralfree.gguf.cpu.quant;

/**
 * The int8 primitives of the fused CPU paths ({@code CpuInt8Attention}, {@code CpuInt8Moe}): the exact int8 dot
 * product of the quantized attention scores, the fused dequantizing accumulation of the attention values and the
 * fused weighted accumulation of the expert outputs.
 *
 * <p>{@link ScalarInt8Kernels} is the reference implementation, and the kernels the engine runs when the Vector API
 * is unavailable. {@link CpuSimd#int8()} serves the Vector API kernels of {@code SimdInt8Kernels} when the JVM
 * offers them. Every kernel returns exactly the scalar result, so selection changes speed only: the dot is an exact
 * integer sum, and each fused accumulation is the same per-element {@code acc += weight * (code * scale)} as the
 * scalar loop, in the same operation order.</p>
 *
 * <p>Stateless and thread-safe; the arrays are read and written in place. Offsets and lengths must lie inside the
 * arrays the caller passes.</p>
 */
public interface CpuInt8Kernels {

    /**
     * {@code sum_i a[aOff + i] * b[bOff + i]} over {@code n} signed bytes, widened exactly into int lanes. Every
     * partial sum fits an int for the head widths attention uses: {@code n <= 4096} keeps the worst case below
     * {@code 2^31}.
     */
    int dot(byte[] a, int aOff, byte[] b, int bOff, int n);

    /**
     * {@code acc[accOff + i] += weight * (codes[cOff + i] * scale)}: one dequantized int8 value multiplied by a
     * weight and accumulated, the value half of the int8 attention.
     */
    void accumulateScaled(byte[] codes, int cOff, float scale, float weight, float[] acc, int accOff, int n);

    /** {@code acc[accOff + i] += weight * y[yOff + i]}: the weighted expert output of one pair. */
    void accumulateWeighted(float[] y, int yOff, float weight, float[] acc, int accOff, int n);

    /** One-line description of the implementation, for the engine's summary line. */
    String describe();
}
