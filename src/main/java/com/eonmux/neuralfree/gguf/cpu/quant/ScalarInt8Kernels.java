package com.eonmux.neuralfree.gguf.cpu.quant;

/**
 * The scalar reference of {@link CpuInt8Kernels}: plain loops in the order the Vector API kernels reproduce lane by
 * lane. Stateless; the singleton {@link #INSTANCE} is shared by every caller.
 */
public final class ScalarInt8Kernels implements CpuInt8Kernels {

    /** The scalar kernels, the reference every other implementation must equal. */
    public static final ScalarInt8Kernels INSTANCE = new ScalarInt8Kernels();

    private ScalarInt8Kernels() {
    }

    @Override
    public int dot(byte[] a, int aOff, byte[] b, int bOff, int n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    @Override
    public void accumulateScaled(byte[] codes, int cOff, float scale, float weight, float[] acc, int accOff, int n) {
        for (int i = 0; i < n; i++) {
            acc[accOff + i] += weight * (codes[cOff + i] * scale);
        }
    }

    @Override
    public void accumulateWeighted(float[] y, int yOff, float weight, float[] acc, int accOff, int n) {
        for (int i = 0; i < n; i++) {
            acc[accOff + i] += weight * y[yOff + i];
        }
    }

    @Override
    public String describe() {
        return "scalar";
    }

    @Override
    public String toString() {
        return "ScalarInt8Kernels{}";
    }
}
