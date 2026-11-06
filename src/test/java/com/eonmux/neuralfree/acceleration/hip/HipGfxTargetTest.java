package com.eonmux.neuralfree.acceleration.hip;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The gfx-target model against the AMD targets llama.cpp's {@code GGML_CUDA_CC_IS_*} macros and
 * the HIP kernel tuning key on: family ranges, the {@code 0x1000000 | gfx} compute-capability
 * encoding and the MFMA/WMMA/dp4a instruction sets.
 */
public class HipGfxTargetTest {

    @Test
    public void parsesGfxNamesAndStripsCapabilitySuffixes() {
        HipGfxTarget target = HipGfxTarget.parse("gfx90a:xnack-");
        assertEquals("gfx90a", target.gcnArchName());
        assertEquals(0x90a, target.gfxId());
        assertEquals(HipGfxTarget.Family.CDNA, target.family());
    }

    @Test
    public void classifiesTheReferenceFamilies() {
        assertEquals(HipGfxTarget.Family.GCN, HipGfxTarget.parse("gfx803").family());
        assertEquals(HipGfxTarget.Family.GCN, HipGfxTarget.parse("gfx900").family());
        assertEquals(HipGfxTarget.Family.GCN, HipGfxTarget.parse("gfx90c").family());
        assertEquals(HipGfxTarget.Family.GCN_DP4A, HipGfxTarget.parse("gfx906").family());
        assertEquals(HipGfxTarget.Family.CDNA, HipGfxTarget.parse("gfx908").family());
        assertEquals(HipGfxTarget.Family.CDNA, HipGfxTarget.parse("gfx90a").family());
        assertEquals(HipGfxTarget.Family.CDNA, HipGfxTarget.parse("gfx942").family());
        assertEquals(HipGfxTarget.Family.CDNA, HipGfxTarget.parse("gfx950").family());
        assertEquals(HipGfxTarget.Family.CDNA, HipGfxTarget.parse("gfx1250").family());
        assertEquals(HipGfxTarget.Family.RDNA1, HipGfxTarget.parse("gfx1010").family());
        assertEquals(HipGfxTarget.Family.RDNA2, HipGfxTarget.parse("gfx1030").family());
        assertEquals(HipGfxTarget.Family.RDNA3, HipGfxTarget.parse("gfx1100").family());
        assertEquals(HipGfxTarget.Family.RDNA3, HipGfxTarget.parse("gfx1101").family());
        assertEquals(HipGfxTarget.Family.RDNA3_5, HipGfxTarget.parse("gfx1150").family());
        assertEquals(HipGfxTarget.Family.RDNA3_5, HipGfxTarget.parse("gfx1151").family());
        assertEquals(HipGfxTarget.Family.RDNA4_M, HipGfxTarget.parse("gfx1170").family());
        assertEquals(HipGfxTarget.Family.RDNA4, HipGfxTarget.parse("gfx1200").family());
        assertEquals(HipGfxTarget.Family.RDNA4, HipGfxTarget.parse("gfx1201").family());
    }

    @Test
    public void encodesAmdComputeCapabilitiesLikeLlamaCpp() {
        // llama.cpp: GGML_CUDA_CC_OFFSET_AMD (0x1000000) plus the gfx id.
        assertEquals(0x1000000 | 0x942, HipGfxTarget.parse("gfx942").amdComputeCapability());
        assertEquals(0x1000000 | 0x90a, HipGfxTarget.parse("gfx90a").amdComputeCapability());
        assertEquals(0x1000000 | 0x1100, HipGfxTarget.parse("gfx1100").amdComputeCapability());
        assertEquals(0x1000000 | 0x1030, HipGfxTarget.parse("gfx1030").amdComputeCapability());
        assertEquals(0x1000000 | 0x1150, HipGfxTarget.parse("gfx1150").amdComputeCapability());
    }

    @Test
    public void decodesTheFamilyOfAnAmdComputeCapability() {
        assertEquals(HipGfxTarget.Family.CDNA, HipGfxTarget.familyOfComputeCapability(0x1000000 | 0x942));
        assertEquals(HipGfxTarget.Family.RDNA3, HipGfxTarget.familyOfComputeCapability(0x1000000 | 0x1100));
        assertEquals(HipGfxTarget.Family.RDNA4, HipGfxTarget.familyOfComputeCapability(0x1000000 | 0x1200));
        try {
            HipGfxTarget.familyOfComputeCapability(120);
            fail("an NVIDIA compute capability is not an AMD one");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("AMD compute capability"));
        }
        try {
            HipGfxTarget.familyOfComputeCapability(0x11000000);
            fail("bits outside the AMD range are not an AMD capability");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("AMD compute capability"));
        }
    }

    @Test
    public void instructionSetsFollowTheLlamaCppPredicates() {
        // amd_mfma_available: CDNA only.
        assertTrue(HipGfxTarget.parse("gfx908").mfmaAvailable());
        assertTrue(HipGfxTarget.parse("gfx90a").mfmaAvailable());
        assertTrue(HipGfxTarget.parse("gfx942").mfmaAvailable());
        assertTrue(HipGfxTarget.parse("gfx1250").mfmaAvailable());
        assertFalse(HipGfxTarget.parse("gfx1100").mfmaAvailable());
        assertFalse(HipGfxTarget.parse("gfx1030").mfmaAvailable());

        // amd_wmma_available: RDNA3, RDNA3.5, RDNA4-M and RDNA4.
        assertTrue(HipGfxTarget.parse("gfx1100").wmmaAvailable());
        assertTrue(HipGfxTarget.parse("gfx1150").wmmaAvailable());
        assertTrue(HipGfxTarget.parse("gfx1170").wmmaAvailable());
        assertTrue(HipGfxTarget.parse("gfx1201").wmmaAvailable());
        assertFalse(HipGfxTarget.parse("gfx1030").wmmaAvailable());
        assertFalse(HipGfxTarget.parse("gfx942").wmmaAvailable());

        // dp4a: Vega 20 (gfx906) and everything newer.
        assertTrue(HipGfxTarget.parse("gfx906").dp4aAvailable());
        assertTrue(HipGfxTarget.parse("gfx1030").dp4aAvailable());
        assertFalse(HipGfxTarget.parse("gfx900").dp4aAvailable());
        assertFalse(HipGfxTarget.parse("gfx803").dp4aAvailable());

        // bf16 matrix arithmetic: CDNA and RDNA3+.
        assertTrue(HipGfxTarget.parse("gfx942").bf16MmaAvailable());
        assertTrue(HipGfxTarget.parse("gfx1151").bf16MmaAvailable());
        assertFalse(HipGfxTarget.parse("gfx1030").bf16MmaAvailable());

        // fp16 everywhere on HIP.
        assertTrue(HipGfxTarget.parse("gfx803").float16Available());
    }

    @Test
    public void rejectsNonGfxNames() {
        assertRejected("");
        assertRejected("navigpu");
        assertRejected("gfx:");
        assertRejected("gfxxyz");
        assertRejected("gfx90a_extra");
        try {
            HipGfxTarget.parse(null);
            fail("a null gcnArchName must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected, and its message names the argument
        }
    }

    private static void assertRejected(String name) {
        try {
            HipGfxTarget.parse(name);
            fail("gcnArchName '" + name + "' must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue("the rejection must name the input", String.valueOf(expected.getMessage()).contains(name));
        }
    }

    @Test
    public void equalsComparesTheTarget() {
        assertEquals(HipGfxTarget.parse("gfx1100"), HipGfxTarget.parse("gfx1100:sramecc+"));
        assertFalse(HipGfxTarget.parse("gfx1100").equals(HipGfxTarget.parse("gfx1101")));
    }
}
