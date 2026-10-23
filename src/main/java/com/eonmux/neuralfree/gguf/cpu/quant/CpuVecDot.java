package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

/**
 * Dot product of one weight row, read in place from a {@link CpuTensorRef}, with an activation vector that
 * {@link CpuActivationQuantizer} prepared for the weight's {@code vec_dot_type}. Implementations mirror the
 * {@code ggml_vec_dot_<w>_<a>} kernels of llama.cpp ggml/src/ggml-cpu/quants.c, are stateless and thread-safe,
 * and allocate nothing per call.
 */
public interface CpuVecDot {

    /**
     * Returns {@code sum_j w[row][j] * x[j]} over the {@code w.cols()} elements of the row.
     *
     * @throws IllegalArgumentException when {@code x} has the wrong activation type or length for {@code w}
     */
    float dotRow(CpuTensorRef w, long row, QuantizedActivations x);

    /**
     * Writes {@code dotRow(w, rowStart + r, x)} to {@code y[yOff + r]} for {@code r} in {@code [0, rowCount)}.
     * Implementations may override this to hoist per-call work out of the row loop.
     */
    default void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        for (int r = 0; r < rowCount; r++) {
            y[yOff + r] = dotRow(w, rowStart + r, x);
        }
    }

    /**
     * One activation vector in the block format a weight type's vec-dot consumes: Q8_0, Q8_1 or Q8_K blocks,
     * or plain F32 for the float weight types. Instances are created only by {@link CpuActivationQuantizer}
     * from freshly allocated arrays that nothing else references, and are never modified afterwards.
     * <p>
     * The arrays are package-private so the vec-dot kernels of this package read them without call
     * overhead; code outside the package uses the accessors.
     */
    final class QuantizedActivations {
        /** Q8_0, Q8_1, Q8_K, or F32. */
        final GGMLType type;
        /** Element count; a whole number of {@link #blockSize()} blocks. */
        final int n;
        /** Quantized values, one per element (null for F32). */
        final byte[] qs;
        /**
         * Per-block scale as the vec-dot reads it: the half-rounded {@code d} for Q8_0 / Q8_1, the float
         * {@code d} for Q8_K (null for F32).
         */
        final float[] d;
        /** Q8_1 only: half-rounded {@code d * sum(qs)} per block ({@code block_q8_1.s}); null otherwise. */
        final float[] s;
        /** Q8_K only: sums of each 16 consecutive quants ({@code block_q8_K.bsums}); null otherwise. */
        final short[] bsums;
        /** F32 only: the activation values; null otherwise. */
        final float[] f32;

        QuantizedActivations(GGMLType type, int n, byte[] qs, float[] d, float[] s, short[] bsums, float[] f32) {
            this.type = type;
            this.n = n;
            this.qs = qs;
            this.d = d;
            this.s = s;
            this.bsums = bsums;
            this.f32 = f32;
        }

        public GGMLType type() {
            return type;
        }

        public int length() {
            return n;
        }

        /** Elements per activation block: 32 for Q8_0 / Q8_1, 256 for Q8_K, 1 for F32. */
        public int blockSize() {
            return type == GGMLType.F32 ? 1 : type.getBlockSize();
        }

        public int blockCount() {
            return n / blockSize();
        }

        /** Quantized value {@code i}. */
        public int quant(int i) {
            return qs[i];
        }

        /** Scale of block {@code b} as the vec-dot reads it. */
        public float scale(int b) {
            return d[b];
        }

        /** Q8_1 block sum term {@code s} of block {@code b}. */
        public float blockSum(int b) {
            return s[b];
        }

        /** Q8_K sum of quants {@code [16g, 16g + 16)}. */
        public int groupSum(int g) {
            return bsums[g];
        }

        /** F32 activation value {@code i}. */
        public float value(int i) {
            return f32[i];
        }

        /**
         * Validates that this vector feeds a vec-dot over {@code w}.
         *
         * @throws IllegalArgumentException naming the expected and actual type and length
         */
        public void requireCompatible(CpuTensorRef w, GGMLType expectedType) {
            if (type != expectedType) {
                throw new IllegalArgumentException(w.type() + " vec-dot needs " + expectedType
                        + " activations, got " + type);
            }
            if (n != w.cols()) {
                throw new IllegalArgumentException(w.type() + " row of " + w.cols()
                        + " elements cannot be dotted with " + n + " activations");
            }
        }

        @Override
        public String toString() {
            return "QuantizedActivations{" + type + ", n=" + n + "}";
        }
    }
}
