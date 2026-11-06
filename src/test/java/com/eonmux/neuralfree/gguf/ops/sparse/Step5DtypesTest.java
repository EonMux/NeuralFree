package com.eonmux.neuralfree.gguf.ops.sparse;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** The Step5 BF16 / FP8 E4M3 roundings against hand-computed values. */
public class Step5DtypesTest {

    @Test
    public void bf16RoundsToNearestEven() {
        assertEquals(1.0f, Step5Dtypes.bf16(1.0f), 0f);
        assertEquals(1.0f, Step5Dtypes.bf16(1.001953125f), 0f);       // 1 + 2^-9, below half of 2^-7
        assertEquals(1.0f, Step5Dtypes.bf16(1.00390625f), 0f);          // 1 + 2^-8, halfway, ties to even
        assertEquals(1.0078125f, Step5Dtypes.bf16(1.0078125f - 1e-7f), 0f);  // just above the tie point
        assertEquals(-1.0f, Step5Dtypes.bf16(-1.0f), 0f);
        assertEquals(0.0f, Step5Dtypes.bf16(0.0f), 0f);
        assertTrue(Float.isNaN(Step5Dtypes.bf16(Float.NaN)));
        assertEquals(Float.POSITIVE_INFINITY, Step5Dtypes.bf16(Float.POSITIVE_INFINITY), 0f);
    }

    @Test
    public void e4m3MatchesTheFiniteRange() {
        assertEquals(448f, Step5Dtypes.e4m3(448f), 0f);
        assertEquals(448f, Step5Dtypes.e4m3(1000f), 0f);
        assertEquals(-448f, Step5Dtypes.e4m3(-1000f), 0f);
        assertEquals(416f, Step5Dtypes.e4m3(416f), 0f);
        assertEquals(0.015625f, Step5Dtypes.e4m3(0.015625f), 0f);     // 2^-6, the smallest normal
        assertEquals(0.001953125f, Step5Dtypes.e4m3(0.001953125f), 0f);  // 2^-9, the smallest subnormal
        assertEquals(0.001953125f, Step5Dtypes.e4m3(0.00146484375f), 0f); // 0.75 of a subnormal step rounds up
        assertEquals(0.0f, Step5Dtypes.e4m3(0.0009765625f), 0f);      // half a subnormal step ties to even zero
        assertEquals(0.00390625f, Step5Dtypes.e4m3(0.0029296875f), 0f);  // 1.5 steps ties to two
        assertEquals(-0.0f, Step5Dtypes.e4m3(-0.0f), 0f);
    }

    @Test
    public void e4m3RoundsAValueWithoutAnExactRepresentation() {
        // 0.1 lies between 0.09375 (12 * 2^-7) and 0.1015625 (13 * 2^-7)
        assertEquals(0.1015625f, Step5Dtypes.e4m3(0.1f), 0f);
        assertEquals(1.0f, Step5Dtypes.e4m3(1.05f), 0f);
    }

    @Test
    public void gateSigmoidIsRoundedToBf16() {
        assertEquals(0.5f, Step5Dtypes.gateSigmoid(0f), 0f);
        assertEquals(Step5Dtypes.bf16(0.7310586f), Step5Dtypes.gateSigmoid(1f), 0f);
    }

    @Test
    public void exp2MatchesThePowerOfTwo() {
        assertEquals(1f, Step5Dtypes.exp2(0f), 0f);
        assertEquals(2f, Step5Dtypes.exp2(1f), 1e-7f);
        assertEquals(0.5f, Step5Dtypes.exp2(-1f), 1e-7f);
    }
}
