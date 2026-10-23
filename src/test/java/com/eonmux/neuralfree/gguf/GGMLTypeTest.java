package com.eonmux.neuralfree.gguf;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Unit tests for {@link GGMLType} — block sizing, byte accounting, id lookup, and the
 * GPU-native classification used to decide whether a tensor is uploaded as raw bytes or
 * dequantized to F32 on load. Pure logic; no GPU or model file required.
 */
public class GGMLTypeTest {

    @Test
    public void fromIdRoundTrips() {
        assertEquals(GGMLType.F32, GGMLType.fromId(0));
        assertEquals(GGMLType.F16, GGMLType.fromId(1));
        assertEquals(GGMLType.Q4_0, GGMLType.fromId(2));
        assertEquals(GGMLType.Q8_0, GGMLType.fromId(8));
        assertEquals(GGMLType.Q4_K, GGMLType.fromId(12));
        assertEquals(GGMLType.Q6_K, GGMLType.fromId(14));
        assertEquals(GGMLType.MXFP4, GGMLType.fromId(39));
        assertEquals(GGMLType.TQ1_0, GGMLType.fromId(34));
        assertEquals(GGMLType.TQ2_0, GGMLType.fromId(35));
        assertEquals(GGMLType.NVFP4, GGMLType.fromId(40));
        assertEquals(GGMLType.Q1_0, GGMLType.fromId(41));
        assertEquals(GGMLType.Q2_0, GGMLType.fromId(42));
    }

    @Test
    public void fromUnknownIdReturnsUnknown() {
        assertEquals(GGMLType.UNKNOWN, GGMLType.fromId(999));
        assertEquals(GGMLType.UNKNOWN, GGMLType.fromId(-7));
    }

    @Test
    public void isQuantizedReflectsBlockSize() {
        // Full-precision / single-element types are not block-quantized.
        assertFalse(GGMLType.F32.isQuantized());
        assertFalse(GGMLType.F16.isQuantized());
        assertFalse(GGMLType.BF16.isQuantized());
        // Block-quantized types are.
        assertTrue(GGMLType.Q4_0.isQuantized());
        assertTrue(GGMLType.Q8_0.isQuantized());
        assertTrue(GGMLType.Q6_K.isQuantized());
        assertTrue(GGMLType.MXFP4.isQuantized());
    }

    @Test
    public void byteAccountingForUnquantizedTypes() {
        // F32: 4 bytes/element, no blocking.
        assertEquals(40L, GGMLType.F32.getBytesForElements(10));
        // F16/BF16: 2 bytes/element.
        assertEquals(20L, GGMLType.F16.getBytesForElements(10));
        assertEquals(20L, GGMLType.BF16.getBytesForElements(10));
    }

    @Test
    public void byteAccountingForBlockQuantizedTypes() {
        // Q8_0: 32 elements per block, 34 bytes per block.
        assertEquals(34L, GGMLType.Q8_0.getBytesForElements(32));
        assertEquals(68L, GGMLType.Q8_0.getBytesForElements(64));
        // Partial block rounds up to a whole block.
        assertEquals(34L, GGMLType.Q8_0.getBytesForElements(1));
        assertEquals(68L, GGMLType.Q8_0.getBytesForElements(33));

        // Q4_0: 32 elements per block, 18 bytes per block.
        assertEquals(18L, GGMLType.Q4_0.getBytesForElements(32));
        assertEquals(36L, GGMLType.Q4_0.getBytesForElements(64));

        // Q6_K: 256 elements per block (the K-quant super-block), 210 bytes per block.
        assertEquals(210L, GGMLType.Q6_K.getBytesForElements(256));
        assertEquals(420L, GGMLType.Q6_K.getBytesForElements(512));
        assertEquals(210L, GGMLType.Q6_K.getBytesForElements(100));
    }

    @Test
    public void byteAccountingForTernaryAndNewTypes() {
        // block_tq1_0: qs[48] + qh[4] + half d = 54 bytes per 256 elements.
        assertEquals(54L, GGMLType.TQ1_0.getBytesForElements(256));
        assertEquals(256, GGMLType.TQ1_0.getBlockSize());
        // block_tq2_0: qs[64] + half d = 66 bytes per 256 elements.
        assertEquals(66L, GGMLType.TQ2_0.getBytesForElements(256));
        // block_nvfp4: 4 UE4M3 scales + 32 nibble bytes = 36 bytes per 64 elements.
        assertEquals(36L, GGMLType.NVFP4.getBytesForElements(64));
        assertEquals(72L, GGMLType.NVFP4.getBytesForElements(128));
        // block_q1_0: half d + 16 bit bytes = 18 bytes per 128 elements.
        assertEquals(18L, GGMLType.Q1_0.getBytesForElements(128));
        // block_q2_0: half d + 16 bytes of 2-bit codes = 18 bytes per 64 elements.
        assertEquals(18L, GGMLType.Q2_0.getBytesForElements(64));
        assertEquals(36L, GGMLType.Q2_0.getBytesForElements(128));
        assertTrue(GGMLType.NVFP4.isQuantized());
        assertTrue(GGMLType.Q1_0.isQuantized());
    }

    @Test
    public void zeroElementsIsZeroBytes() {
        assertEquals(0L, GGMLType.F32.getBytesForElements(0));
        assertEquals(0L, GGMLType.Q8_0.getBytesForElements(0));
    }
}
