package com.eonmux.neuralfree.gguf.cpu.quant;

/**
 * The float primitives of the CPU attention core ({@code CpuAttentionCore}): the query-key dot products of a group
 * of keys against the query heads of a work unit laid out as lanes, and the weighted accumulation of a value row.
 *
 * <p>{@link ScalarAttnKernels} is the reference implementation, and the kernels the engine runs by default.
 * {@link CpuSimd#attention()} serves the Vector API kernels of {@code SimdAttnKernels} instead when
 * {@value CpuSimd#PROP_ATTENTION} is {@code true} and the JVM offers them. Every kernel returns exactly the scalar
 * result, so selection changes speed only: each lane of a dot is its own sum over the head dimension in ascending
 * order, a rounded product then a rounded add per step, which is the scalar loop {@code dot += q[d] * k[d]}; the
 * vector form runs lanes (query heads) and keys side by side, never reassociating one lane's sum.</p>
 *
 * <p>Stateless and thread-safe; the arrays are read and written in place. Offsets and lengths must lie inside the
 * arrays the caller passes.</p>
 */
public interface CpuAttnKernels {

    /** Lanes of the query layout of {@link #dots} are whole multiples of this. */
    int LANE_QUANTUM = 8;

    /**
     * {@code out[outOff + j * lanes + l] = sum_{d < dK} qT[d * lanes + l] * k[kOff + j * kStride + d]} for keys
     * {@code j < nKeys} and the lanes {@code l < used} the caller reads, each sum in ascending {@code d}, starting
     * from {@code 0.0f}, with the product rounded before the add. An implementation may also write the padding lanes
     * {@code [used, lanes)}, whose values are unspecified.
     *
     * @param qT    the query heads transposed: element {@code d} of lane {@code l} at {@code qT[d * lanes + l]}
     * @param lanes the layout's lane count, a positive multiple of {@link #LANE_QUANTUM}
     * @param used  the lanes that hold query heads, {@code 0 < used <= lanes}
     * @throws IllegalArgumentException for a lane count that is not a positive multiple of {@link #LANE_QUANTUM} or
     *                                  a used count outside {@code (0, lanes]}
     */
    void dots(float[] qT, int lanes, int used, float[] k, int kOff, int kStride, int nKeys, int dK, float[] out,
              int outOff);

    /**
     * Validates the lane counts of {@link #dots}.
     *
     * @throws IllegalArgumentException as {@link #dots} describes
     */
    static void checkLanes(int lanes, int used) {
        if (lanes <= 0 || lanes % LANE_QUANTUM != 0) {
            throw new IllegalArgumentException("query lanes must be a positive multiple of " + LANE_QUANTUM
                    + ", got " + lanes);
        }
        if (used <= 0 || used > lanes) {
            throw new IllegalArgumentException("used lanes must lie in (0, " + lanes + "], got " + used);
        }
    }

    /** {@code acc[accOff + i] += weight * v[vOff + i]}: one value row weighted and accumulated. */
    void accumulate(float[] v, int vOff, float weight, float[] acc, int accOff, int n);

    /** One-line description of the implementation, for the engine's summary line. */
    String describe();
}
