package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The row-quantizer facade: the supported types, the refusals by name, and the byte counts per row.
 */
public class RowQuantizersTest {

    @Test
    public void supportsTheProducibleTypes() {
        for (GGMLType type : new GGMLType[] {
                GGMLType.F32, GGMLType.F16, GGMLType.BF16, GGMLType.Q1_0, GGMLType.Q2_0, GGMLType.Q4_0,
                GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_K, GGMLType.Q2_K,
                GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.IQ4_NL,
                GGMLType.IQ4_XS, GGMLType.TQ1_0, GGMLType.TQ2_0, GGMLType.MXFP4, GGMLType.IQ1_S,
                GGMLType.IQ1_M, GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S, GGMLType.IQ3_XXS,
                GGMLType.IQ3_S}) {
            assertTrue(type + " should be supported", RowQuantizers.supports(type));
        }
        for (GGMLType type : new GGMLType[] {GGMLType.IQ3_K, GGMLType.Q4_2, GGMLType.UNKNOWN}) {
            assertFalse(type + " should be refused", RowQuantizers.supports(type));
        }
    }

    @Test
    public void weightRequirementFollowsGgml() {
        for (GGMLType type : new GGMLType[] {GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ1_S}) {
            assertTrue(type + " needs weights", RowQuantizers.requiresWeights(type));
        }
        for (GGMLType type : new GGMLType[] {GGMLType.IQ2_S, GGMLType.IQ3_XXS, GGMLType.IQ3_S,
                GGMLType.IQ1_M, GGMLType.Q2_K, GGMLType.IQ4_XS}) {
            assertFalse(type + " runs without weights", RowQuantizers.requiresWeights(type));
        }
    }

    @Test
    public void bytesPerRowMatchesTheBlockGeometry() {
        assertEquals(34 * 8, RowQuantizers.bytesPerRow(GGMLType.Q8_0, 256));
        assertEquals(144, RowQuantizers.bytesPerRow(GGMLType.Q4_K, 256));
        assertEquals(110 * 2, RowQuantizers.bytesPerRow(GGMLType.Q3_K, 512));
        assertEquals(136, RowQuantizers.bytesPerRow(GGMLType.IQ4_XS, 256));
        assertEquals(18 * 2, RowQuantizers.bytesPerRow(GGMLType.Q1_0, 256));
        assertEquals(18 * 4, RowQuantizers.bytesPerRow(GGMLType.Q2_0, 256));
        assertEquals(256 * 2, RowQuantizers.bytesPerRow(GGMLType.F16, 256));
        assertEquals(66 * 2, RowQuantizers.bytesPerRow(GGMLType.IQ2_XXS, 512));
        assertEquals(110, RowQuantizers.bytesPerRow(GGMLType.IQ3_S, 256));
        assertEquals(56, RowQuantizers.bytesPerRow(GGMLType.IQ1_M, 256));
    }

    @Test
    public void unsupportedTypeRefusesByName() {
        try {
            RowQuantizers.quantizeRow(GGMLType.IQ3_K, new float[256], 0, new byte[110], 0, 256, null);
            fail("IQ3_K has no encoder");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("IQ3_K"));
        }
        try {
            RowQuantizers.bytesPerRow(GGMLType.Q4_2, 256);
            fail("Q4_2 has no encoder");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Q4_2"));
        }
        try {
            RowQuantizers.quantizeRow(GGMLType.IQ2_XXS, new float[256], 0, new byte[66], 0, 256, null);
            fail("IQ2_XXS without weights must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("IQ2_XXS"));
            assertTrue(e.getMessage(), e.getMessage().contains("importance weights"));
        }
    }

    @Test
    public void partialRowRefuses() {
        try {
            RowQuantizers.quantizeRow(GGMLType.Q4_K, new float[100], 0, new byte[144], 0, 100, null);
            fail("a partial super-block must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("256"));
        }
    }

    @Test
    public void dispatchAgreesWithTheFamilies() {
        // one row through the facade equals the same row through the family class (Q4_K reference path)
        float[] x = new float[256];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) Math.sin(i * 0.7f) * 0.05f;
        }
        byte[] viaFacade = new byte[GGMLType.Q4_K.getTypeSize()];
        byte[] viaFamily = new byte[GGMLType.Q4_K.getTypeSize()];
        RowQuantizers.quantizeRow(GGMLType.Q4_K, x, 0, viaFacade, 0, 256, null);
        KQuantEncoders.quantizeRow(GGMLType.Q4_K, x, 0, viaFamily, 0, 256, null);
        org.junit.Assert.assertArrayEquals(viaFacade, viaFamily);
    }
}
