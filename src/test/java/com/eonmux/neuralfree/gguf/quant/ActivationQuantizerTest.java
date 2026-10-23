package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuActivationQuantizer;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import org.junit.Test;

import java.nio.ByteBuffer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for {@link CpuActivationQuantizer}: the weight-to-activation type pairing of ggml-cpu and the
 * rounding rules of {@code quantize_row_q8_0_ref}, {@code quantize_row_q8_1_ref} and
 * {@code quantize_row_q8_K_ref}.
 */
public class ActivationQuantizerTest {

    @Test
    public void activationTypeFollowsGgmlCpuTraits() {
        for (GGMLType t : new GGMLType[]{GGMLType.F32, GGMLType.F16, GGMLType.BF16}) {
            assertEquals(GGMLType.F32, CpuActivationQuantizer.activationType(t));
        }
        for (GGMLType t : new GGMLType[]{GGMLType.Q4_0, GGMLType.Q5_0, GGMLType.Q8_0, GGMLType.MXFP4,
                GGMLType.NVFP4, GGMLType.IQ4_NL, GGMLType.Q1_0, GGMLType.Q2_0}) {
            assertEquals(t.toString(), GGMLType.Q8_0, CpuActivationQuantizer.activationType(t));
        }
        assertEquals(GGMLType.Q8_1, CpuActivationQuantizer.activationType(GGMLType.Q4_1));
        assertEquals(GGMLType.Q8_1, CpuActivationQuantizer.activationType(GGMLType.Q5_1));
        for (GGMLType t : new GGMLType[]{GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K,
                GGMLType.Q6_K, GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S, GGMLType.IQ3_XXS,
                GGMLType.IQ3_S, GGMLType.IQ1_S, GGMLType.IQ1_M, GGMLType.IQ4_XS, GGMLType.TQ1_0,
                GGMLType.TQ2_0}) {
            assertEquals(t.toString(), GGMLType.Q8_K, CpuActivationQuantizer.activationType(t));
        }
        for (GGMLType t : new GGMLType[]{GGMLType.Q8_1, GGMLType.Q8_K, GGMLType.I32, GGMLType.F64,
                GGMLType.UNKNOWN}) {
            try {
                CpuActivationQuantizer.activationType(t);
                fail(t + " has no CPU vec-dot");
            } catch (UnsupportedOperationException expected) {
                assertTrue(expected.getMessage().contains(t.toString()));
            }
        }
    }

    @Test
    public void q8_0RoundsHalfAwayFromZeroAndStoresAHalfScale() {
        float[] x = new float[64];
        x[0] = 127f;     // amax 127 -> d = 1, id = 1
        x[1] = 2.5f;
        x[2] = -2.5f;
        x[3] = 0.5f;
        x[4] = -0.5f;
        x[5] = -127f;
        x[6] = 1.4999f;
        x[32] = 3f;      // second block: d = 3/127, not representable as a half
        QuantizedActivations q = CpuActivationQuantizer.quantizeQ8_0(x, 0, 64);
        assertEquals(GGMLType.Q8_0, q.type());
        assertEquals(2, q.blockCount());
        assertEquals(1.0f, q.scale(0), 0f);
        assertEquals(127, q.quant(0));
        assertEquals(3, q.quant(1));
        assertEquals(-3, q.quant(2));
        assertEquals(1, q.quant(3));
        assertEquals(-1, q.quant(4));
        assertEquals(-127, q.quant(5));
        assertEquals(1, q.quant(6));
        assertEquals(0, q.quant(7));
        assertEquals(FpBlockDecoders.roundToFp16(3f / 127), q.scale(1), 0f);
        assertEquals(127, q.quant(32));
    }

    @Test
    public void q8_1BlockSumUsesTheUnroundedScale() {
        float[] x = new float[32];
        x[0] = 127f;
        for (int j = 1; j < 32; j++) {
            x[j] = 1f;
        }
        QuantizedActivations q = CpuActivationQuantizer.quantizeQ8_1(x, 0, 32);
        assertEquals(GGMLType.Q8_1, q.type());
        assertEquals(1.0f, q.scale(0), 0f);
        assertEquals(158f, q.blockSum(0), 0f);       // (127 + 31) * 1
        for (int j = 1; j < 32; j++) {
            assertEquals(1, q.quant(j));
        }
    }

    @Test
    public void q8_KRoundsToNearestEvenWithASignedScale() {
        float[] x = new float[3 * 256];
        x[0] = -127f;    // largest magnitude, negative: iscale = 1, d = 1
        x[1] = 2.5f;
        x[2] = 3.5f;
        x[3] = -2.5f;
        x[256] = 127f;   // largest magnitude, positive: iscale = -1, d = -1
        x[257] = 2.5f;
        x[258] = -127f;  // equal magnitude later in the block does not replace the max
        QuantizedActivations q = CpuActivationQuantizer.quantizeQ8_K(x, 0, x.length);
        assertEquals(GGMLType.Q8_K, q.type());
        assertEquals(3, q.blockCount());

        assertEquals(1.0f, q.scale(0), 0f);
        assertEquals(-127, q.quant(0));
        assertEquals(2, q.quant(1));
        assertEquals(4, q.quant(2));
        assertEquals(-2, q.quant(3));
        assertEquals(-127 + 2 + 4 - 2, q.groupSum(0));
        assertEquals(0, q.groupSum(1));

        assertEquals(-1.0f, q.scale(1), 0f);
        assertEquals(-127, q.quant(256));
        assertEquals(-2, q.quant(257));
        assertEquals(127, q.quant(258));
        assertEquals(-127 - 2 + 127, q.groupSum(16));

        assertEquals(0.0f, q.scale(2), 0f);
        for (int i = 512; i < 768; i++) {
            assertEquals(0, q.quant(i));
        }
        assertEquals(0, q.groupSum(32));
    }

    @Test
    public void quantizeForDispatchesAndF32Copies() {
        float[] x = new float[256];
        x[5] = 4f;
        assertEquals(GGMLType.Q8_K, CpuActivationQuantizer.quantizeFor(GGMLType.Q4_K, x, 0, 256).type());
        assertEquals(GGMLType.Q8_1, CpuActivationQuantizer.quantizeFor(GGMLType.Q5_1, x, 0, 256).type());
        assertEquals(GGMLType.Q8_0, CpuActivationQuantizer.quantizeFor(GGMLType.MXFP4, x, 0, 256).type());
        QuantizedActivations f = CpuActivationQuantizer.quantizeFor(GGMLType.BF16, x, 4, 4);
        assertEquals(GGMLType.F32, f.type());
        x[5] = -1f;
        assertEquals(4f, f.value(1), 0f);
        assertEquals(4, f.length());
    }

    @Test
    public void invalidRangesAreRejected() {
        try {
            CpuActivationQuantizer.quantizeQ8_0(new float[32], 0, 16);
            fail("partial Q8_0 block");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("block size 32"));
        }
        try {
            CpuActivationQuantizer.quantizeQ8_K(new float[256], 1, 256);
            fail("range overrun");
        } catch (IndexOutOfBoundsException expected) {
            assertTrue(expected.getMessage().contains("256"));
        }
    }

    @Test
    public void compatibilityCheckNamesTypesAndLengths() {
        CpuTensorRef w = CpuTensorRef.of(GGMLType.Q4_0, ByteBuffer.allocate(2 * BlockLayout.Q4_0_BYTES), 1, 64);
        QuantizedActivations ok = CpuActivationQuantizer.quantizeQ8_0(new float[64], 0, 64);
        ok.requireCompatible(w, GGMLType.Q8_0);
        try {
            ok.requireCompatible(w, GGMLType.Q8_K);
            fail("wrong activation type");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("Q8_K"));
        }
        QuantizedActivations shorter = CpuActivationQuantizer.quantizeQ8_0(new float[32], 0, 32);
        try {
            shorter.requireCompatible(w, GGMLType.Q8_0);
            fail("wrong length");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("64"));
        }
    }
}
