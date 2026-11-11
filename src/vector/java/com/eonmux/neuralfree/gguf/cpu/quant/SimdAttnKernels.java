package com.eonmux.neuralfree.gguf.cpu.quant;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * The Vector API float primitives of the CPU attention core ({@link CpuAttnKernels}).
 *
 * <p>The dots run eight query lanes per vector and several keys side by side: every vector accumulator is eight
 * independent lane sums, each stepping through the head dimension in ascending order with a rounded multiply and a
 * rounded add, the scalar loop's arithmetic, so the results equal {@link ScalarAttnKernels} bit for bit. Four keys
 * by two lane vectors make eight independent accumulation chains, so the adds run at their throughput where the
 * scalar loop waits on one chain's latency. The accumulation is the scalar {@code acc += weight * v} lane by lane.
 * Instances are immutable and thread-safe.</p>
 */
public final class SimdAttnKernels implements CpuAttnKernels {

    /** The engine's single instance. */
    public static final SimdAttnKernels INSTANCE = new SimdAttnKernels();

    private static final VectorSpecies<Float> F8 = SimdFp.F8;

    /** Keys one register tile of {@link #dots} accumulates side by side. */
    private static final int KEYS = 4;

    private SimdAttnKernels() {
    }

    /** Computes the lane vectors that hold the used lanes; the vectors past them are not touched. */
    @Override
    public void dots(float[] qT, int lanes, int used, float[] k, int kOff, int kStride, int nKeys, int dK,
                     float[] out, int outOff) {
        CpuAttnKernels.checkLanes(lanes, used);
        final int width = (used + LANE_QUANTUM - 1) / LANE_QUANTUM * LANE_QUANTUM;
        int j = 0;
        for (; j + KEYS <= nKeys; j += KEYS) {
            final int kb = kOff + j * kStride;
            final int ob = outOff + j * lanes;
            int l = 0;
            for (; l + 2 * LANE_QUANTUM <= width; l += 2 * LANE_QUANTUM) {
                fourKeysTwoVectors(qT, lanes, l, k, kb, kStride, dK, out, ob);
            }
            if (l < width) {
                fourKeysOneVector(qT, lanes, l, k, kb, kStride, dK, out, ob);
            }
        }
        for (; j < nKeys; j++) {
            final int kb = kOff + j * kStride;
            for (int l = 0; l < width; l += LANE_QUANTUM) {
                FloatVector a = FloatVector.zero(F8);
                for (int d = 0; d < dK; d++) {
                    a = a.add(FloatVector.fromArray(F8, qT, d * lanes + l).mul(k[kb + d]));
                }
                a.intoArray(out, outOff + j * lanes + l);
            }
        }
    }

    /** Lanes {@code [l, l + 16)} of four keys: eight chains. */
    private static void fourKeysTwoVectors(float[] qT, int lanes, int l, float[] k, int kb, int kStride, int dK,
                                           float[] out, int ob) {
        FloatVector a0 = FloatVector.zero(F8);
        FloatVector a1 = a0;
        FloatVector a2 = a0;
        FloatVector a3 = a0;
        FloatVector b0 = a0;
        FloatVector b1 = a0;
        FloatVector b2 = a0;
        FloatVector b3 = a0;
        for (int d = 0; d < dK; d++) {
            final FloatVector x = FloatVector.fromArray(F8, qT, d * lanes + l);
            final FloatVector y = FloatVector.fromArray(F8, qT, d * lanes + l + LANE_QUANTUM);
            final float k0 = k[kb + d];
            a0 = a0.add(x.mul(k0));
            b0 = b0.add(y.mul(k0));
            final float k1 = k[kb + kStride + d];
            a1 = a1.add(x.mul(k1));
            b1 = b1.add(y.mul(k1));
            final float k2 = k[kb + 2 * kStride + d];
            a2 = a2.add(x.mul(k2));
            b2 = b2.add(y.mul(k2));
            final float k3 = k[kb + 3 * kStride + d];
            a3 = a3.add(x.mul(k3));
            b3 = b3.add(y.mul(k3));
        }
        a0.intoArray(out, ob + l);
        b0.intoArray(out, ob + l + LANE_QUANTUM);
        a1.intoArray(out, ob + lanes + l);
        b1.intoArray(out, ob + lanes + l + LANE_QUANTUM);
        a2.intoArray(out, ob + 2 * lanes + l);
        b2.intoArray(out, ob + 2 * lanes + l + LANE_QUANTUM);
        a3.intoArray(out, ob + 3 * lanes + l);
        b3.intoArray(out, ob + 3 * lanes + l + LANE_QUANTUM);
    }

    /** Lanes {@code [l, l + 8)} of four keys: four chains. */
    private static void fourKeysOneVector(float[] qT, int lanes, int l, float[] k, int kb, int kStride, int dK,
                                          float[] out, int ob) {
        FloatVector a0 = FloatVector.zero(F8);
        FloatVector a1 = a0;
        FloatVector a2 = a0;
        FloatVector a3 = a0;
        for (int d = 0; d < dK; d++) {
            final FloatVector x = FloatVector.fromArray(F8, qT, d * lanes + l);
            a0 = a0.add(x.mul(k[kb + d]));
            a1 = a1.add(x.mul(k[kb + kStride + d]));
            a2 = a2.add(x.mul(k[kb + 2 * kStride + d]));
            a3 = a3.add(x.mul(k[kb + 3 * kStride + d]));
        }
        a0.intoArray(out, ob + l);
        a1.intoArray(out, ob + lanes + l);
        a2.intoArray(out, ob + 2 * lanes + l);
        a3.intoArray(out, ob + 3 * lanes + l);
    }

    @Override
    public void accumulate(float[] v, int vOff, float weight, float[] acc, int accOff, int n) {
        int i = 0;
        for (; i + LANE_QUANTUM <= n; i += LANE_QUANTUM) {
            FloatVector.fromArray(F8, v, vOff + i).mul(weight).add(FloatVector.fromArray(F8, acc, accOff + i))
                    .intoArray(acc, accOff + i);
        }
        for (; i < n; i++) {
            acc[accOff + i] += weight * v[vOff + i];
        }
    }

    @Override
    public String describe() {
        return "vector/" + F8.vectorBitSize() + "b (jdk.incubator.vector)";
    }

    @Override
    public String toString() {
        return "SimdAttnKernels{" + describe() + '}';
    }
}
