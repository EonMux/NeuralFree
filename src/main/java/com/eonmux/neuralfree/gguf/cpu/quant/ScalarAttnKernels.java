package com.eonmux.neuralfree.gguf.cpu.quant;

/**
 * The scalar reference of {@link CpuAttnKernels}: plain loops in the order the Vector API kernels reproduce lane by
 * lane. Stateless; the singleton {@link #INSTANCE} is shared by every caller.
 */
public final class ScalarAttnKernels implements CpuAttnKernels {

    /** The scalar kernels, the reference every other implementation must equal. */
    public static final ScalarAttnKernels INSTANCE = new ScalarAttnKernels();

    /** Keys one lane scores side by side, to overlap their dependent chains of adds. */
    private static final int KEYS = 4;

    private ScalarAttnKernels() {
    }

    /**
     * Computes the used lanes only: the padding lanes keep whatever {@code out} held. Each lane scores four keys at
     * a time, four independent chains of {@code dot += q[d] * k[d]} that each keep the ascending-{@code d} order.
     */
    @Override
    public void dots(float[] qT, int lanes, int used, float[] k, int kOff, int kStride, int nKeys, int dK,
                     float[] out, int outOff) {
        CpuAttnKernels.checkLanes(lanes, used);
        for (int l = 0; l < used; l++) {
            int j = 0;
            for (; j + KEYS <= nKeys; j += KEYS) {
                final int k0 = kOff + j * kStride;
                final int k1 = k0 + kStride;
                final int k2 = k1 + kStride;
                final int k3 = k2 + kStride;
                float d0 = 0.0f;
                float d1 = 0.0f;
                float d2 = 0.0f;
                float d3 = 0.0f;
                for (int d = 0; d < dK; d++) {
                    final float qd = qT[d * lanes + l];
                    d0 += qd * k[k0 + d];
                    d1 += qd * k[k1 + d];
                    d2 += qd * k[k2 + d];
                    d3 += qd * k[k3 + d];
                }
                final int o = outOff + j * lanes + l;
                out[o] = d0;
                out[o + lanes] = d1;
                out[o + 2 * lanes] = d2;
                out[o + 3 * lanes] = d3;
            }
            for (; j < nKeys; j++) {
                final int kb = kOff + j * kStride;
                float dot = 0.0f;
                for (int d = 0; d < dK; d++) {
                    dot += qT[d * lanes + l] * k[kb + d];
                }
                out[outOff + j * lanes + l] = dot;
            }
        }
    }

    @Override
    public void accumulate(float[] v, int vOff, float weight, float[] acc, int accOff, int n) {
        for (int i = 0; i < n; i++) {
            acc[accOff + i] += weight * v[vOff + i];
        }
    }

    @Override
    public String describe() {
        return "scalar";
    }

    @Override
    public String toString() {
        return "ScalarAttnKernels";
    }
}
