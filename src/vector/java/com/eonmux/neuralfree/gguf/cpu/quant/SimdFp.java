package com.eonmux.neuralfree.gguf.cpu.quant;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * The float lanes and the bitcast loads shared by the float vec-dot kernels ({@link SimdF32}, {@link SimdBf16},
 * {@link SimdExt}): eight float lanes matching the eight scalar partial sums of {@code ggml_vec_dot_f32} /
 * {@code ggml_vec_dot_f16}, the per-thread window the weight elements widen into, and the pairwise lane
 * reduction in the scalar kernels' order.
 */
final class SimdFp {

    /** 8 float lanes. */
    static final VectorSpecies<Float> F8 = FloatVector.SPECIES_256;

    /** 32 byte lanes: the bitcast of eight floats, i.e. the eight F32 weights of one group. */
    static final VectorSpecies<Byte> B32 = ByteVector.SPECIES_256;

    /** 8 short lanes: the bitcast of 16 bytes, i.e. the eight bf16 values of one group. */
    static final VectorSpecies<Short> S8 = ShortVector.SPECIES_128;

    /** Tokens one pass of the batched float kernels accumulates at a time. */
    static final int TOKENS = 8;

    private static final ThreadLocal<float[]> ACTS = new ThreadLocal<>();

    private SimdFp() {
    }

    /** The calling thread's activation values, the signed quants of {@code qs} widened once per call. */
    static float[] activations(byte[] qs, int n) {
        float[] buf = ACTS.get();
        if (buf == null || buf.length < n) {
            buf = new float[Math.max(n, 1024)];
            ACTS.set(buf);
        }
        for (int i = 0; i < n; i++) {
            buf[i] = qs[i];
        }
        return buf;
    }

    /**
     * {@code ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))} of the eight accumulator lanes, in float.
     */
    static float pairwise(FloatVector acc) {
        return ((acc.lane(0) + acc.lane(1)) + (acc.lane(2) + acc.lane(3)))
                + ((acc.lane(4) + acc.lane(5)) + (acc.lane(6) + acc.lane(7)));
    }
}
