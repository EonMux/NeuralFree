package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for the legacy, Q8, FP4 and K-quant CPU vec-dot kernels ({@link LegacyVecDot}, {@link Q8VecDot},
 * {@link Fp4VecDot}, {@link KQuantVecDot}): agreement with the dot product of the reference-decoded weights and the
 * quantized activations, a hand-computed Q4_K super-block, multi-shard rows and argument validation.
 */
public class VecDotLegacyTest {

    private static final GGMLType[] CQ1_TYPES = {
            GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.MXFP4,
            GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K
    };

    private static CpuVecDot vecDotFor(GGMLType t) {
        if (LegacyVecDot.supports(t)) {
            return LegacyVecDot.forType(t);
        }
        if (Q8VecDot.supports(t)) {
            return Q8VecDot.forType(t);
        }
        if (Fp4VecDot.supports(t)) {
            return Fp4VecDot.forType(t);
        }
        return KQuantVecDot.forType(t);
    }

    /** Byte offsets of the half-precision scale / min fields of each type's block. */
    private static int[] halfFields(GGMLType t) {
        switch (t) {
            case Q4_0:
                return new int[]{BlockLayout.Q4_0_D};
            case Q4_1:
                return new int[]{BlockLayout.Q4_1_D, BlockLayout.Q4_1_M};
            case Q5_0:
                return new int[]{BlockLayout.Q5_0_D};
            case Q5_1:
                return new int[]{BlockLayout.Q5_1_D, BlockLayout.Q5_1_M};
            case Q8_0:
                return new int[]{BlockLayout.Q8_0_D};
            case Q2_K:
                return new int[]{BlockLayout.Q2_K_D, BlockLayout.Q2_K_DMIN};
            case Q3_K:
                return new int[]{BlockLayout.Q3_K_D};
            case Q4_K:
                return new int[]{BlockLayout.Q4_K_D, BlockLayout.Q4_K_DMIN};
            case Q5_K:
                return new int[]{BlockLayout.Q5_K_D, BlockLayout.Q5_K_DMIN};
            case Q6_K:
                return new int[]{BlockLayout.Q6_K_D};
            default:
                return new int[0];
        }
    }

    /** Random blocks with finite, moderate scales (and E8M0 exponents near 1 for MXFP4). */
    private static ByteBuffer randomRows(GGMLType t, int rows, int cols, Random rnd) {
        BlockLayout layout = BlockLayout.of(t);
        int blocks = rows * (cols / layout.blockElements());
        byte[] raw = new byte[blocks * layout.blockBytes()];
        rnd.nextBytes(raw);
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < blocks; i++) {
            int o = i * layout.blockBytes();
            for (int f : halfFields(t)) {
                float v = (rnd.nextFloat() * 2f - 1f) * 0.1f;
                b.putShort(o + f, (short) FpBlockDecoders.fp32ToFp16(v));
            }
            if (t == GGMLType.MXFP4) {
                b.put(o + BlockLayout.MXFP4_E, (byte) (118 + rnd.nextInt(14)));
            }
        }
        return b;
    }

    private static float[] gaussian(int n, Random rnd) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) rnd.nextGaussian();
        }
        return x;
    }

    /** Offset of the min field of the Q8_1-paired types, whose min term ggml takes from {@code block_q8_1.s}. */
    private static int minField(GGMLType t) {
        return t == GGMLType.Q4_1 ? BlockLayout.Q4_1_M : BlockLayout.Q5_1_M;
    }

    @Test
    public void everyTypeMatchesTheDequantizedDotProduct() {
        final int rows = 3;
        final int cols = 512;
        Random rnd = new Random(0x5EED);
        for (GGMLType t : CQ1_TYPES) {
            ByteBuffer data = randomRows(t, rows, cols, rnd);
            CpuTensorRef w = CpuTensorRef.of(t, data, rows, cols);
            QuantizedActivations x = CpuActivationQuantizer.quantizeFor(t, gaussian(cols, rnd), 0, cols);
            CpuVecDot vd = vecDotFor(t);
            float[] wf = new float[cols];
            for (int r = 0; r < rows; r++) {
                w.decodeRow(r, wf, 0);
                double ref = 0;
                double maxW = 0;
                double sumA = 0;
                for (int j = 0; j < cols; j++) {
                    double a = (double) x.quant(j) * x.scale(j / x.blockSize());
                    ref += wf[j] * a;
                    maxW = Math.max(maxW, Math.abs(wf[j]));
                    sumA += Math.abs(a);
                }
                if (x.type() == GGMLType.Q8_1) {
                    ref += q8_1MinCorrection(t, data, w.rowOffset(r), x);
                }
                double tol = 1e-5 * maxW * sumA + 1e-6;
                float got = vd.dotRow(w, r, x);
                assertEquals(t + " row " + r, ref, got, tol);
            }
        }
    }

    /**
     * ggml pairs the min of Q4_1 / Q5_1 with {@code block_q8_1.s}, the half-rounded {@code d * sum(q)} of the
     * unrounded scale, instead of {@code half(d) * sum(q)}; returns the difference that choice makes.
     */
    private static double q8_1MinCorrection(GGMLType t, ByteBuffer data, int rowOff, QuantizedActivations x) {
        BlockLayout layout = BlockLayout.of(t);
        double c = 0;
        for (int b = 0; b < x.blockCount(); b++) {
            float m = FpBlockDecoders.fp16At(data, rowOff + b * layout.blockBytes() + minField(t));
            int sumQ = 0;
            for (int j = 0; j < x.blockSize(); j++) {
                sumQ += x.quant(b * x.blockSize() + j);
            }
            c += (double) m * (x.blockSum(b) - (double) x.scale(b) * sumQ);
        }
        return c;
    }

    /**
     * One Q4_K super-block with known 6-bit scales and mins (several above 15, exercising the high-bit
     * packing of {@code get_scale_min_k4}), known nibbles and integer activations that Q8_K quantizes with
     * {@code d = 1}, so the expected value is {@code sum_e (d * sc * q_e - dmin * m) * x_e}.
     */
    @Test
    public void q4_KSuperBlockMatchesTheHandComputedSum() {
        final int[] sc = {1, 2, 3, 4, 17, 33, 50, 63};
        final int[] mn = {3, 5, 7, 9, 20, 40, 61, 0};
        final float d = 1.0f;
        final float dmin = 0.5f;
        byte[] raw = new byte[BlockLayout.Q4_K_BYTES];
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        b.putShort(BlockLayout.Q4_K_D, (short) FpBlockDecoders.fp32ToFp16(d));
        b.putShort(BlockLayout.Q4_K_DMIN, (short) FpBlockDecoders.fp32ToFp16(dmin));
        int s = BlockLayout.Q4_K_SCALES;
        for (int j = 0; j < 4; j++) {
            raw[s + j] = (byte) (sc[j] | ((sc[j + 4] >> 4) << 6));
            raw[s + j + 4] = (byte) (mn[j] | ((mn[j + 4] >> 4) << 6));
            raw[s + j + 8] = (byte) ((sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4));
        }
        int[] q = new int[256];
        for (int e = 0; e < 256; e++) {
            q[e] = (e * 7 + 3) % 16;
        }
        for (int j = 0; j < 4; j++) {
            for (int l = 0; l < 32; l++) {
                raw[BlockLayout.Q4_K_QS + 32 * j + l] = (byte) (q[64 * j + l] | (q[64 * j + 32 + l] << 4));
            }
        }
        float[] x = new float[256];
        for (int e = 0; e < 256; e++) {
            x[e] = ((e * 37) % 255) - 127;
        }
        QuantizedActivations xq = CpuActivationQuantizer.quantizeQ8_K(x, 0, 256);
        assertEquals(1.0f, xq.scale(0), 0f);
        for (int e = 0; e < 256; e++) {
            assertEquals((int) x[e], xq.quant(e));
        }
        double expected = 0;
        for (int e = 0; e < 256; e++) {
            int sub = e / 32;
            expected += ((double) d * sc[sub] * q[e] - (double) dmin * mn[sub]) * x[e];
        }
        CpuTensorRef w = CpuTensorRef.of(GGMLType.Q4_K, b, 1, 256);
        float got = KQuantVecDot.Q4_K.dotRow(w, 0, xq);
        assertEquals(expected, got, Math.abs(expected) * 1e-6 + 1e-3);

        float[] decoded = new float[256];
        w.decodeRow(0, decoded, 0);
        double viaDecoder = 0;
        for (int e = 0; e < 256; e++) {
            viaDecoder += (double) decoded[e] * x[e];
        }
        assertEquals(expected, viaDecoder, Math.abs(expected) * 1e-6 + 1e-3);
    }

    @Test
    public void dotRowsAcrossShardsMatchesSingleRows() {
        final int rows = 4;
        final int cols = 256;
        Random rnd = new Random(42);
        for (GGMLType t : CQ1_TYPES) {
            ByteBuffer whole = randomRows(t, rows, cols, rnd);
            long rowBytes = BlockLayout.of(t).rowBytes(cols);
            ByteBuffer first = slice(whole, 0, (int) (2 * rowBytes));
            ByteBuffer second = slice(whole, (int) (2 * rowBytes), (int) (2 * rowBytes));
            CpuTensorRef sharded = new CpuTensorRef(t, new ByteBuffer[]{first, second},
                    new long[]{0L, 2 * rowBytes}, rows, cols, rowBytes);
            CpuTensorRef single = CpuTensorRef.of(t, whole, rows, cols);
            QuantizedActivations x = CpuActivationQuantizer.quantizeFor(t, gaussian(cols, rnd), 0, cols);
            CpuVecDot vd = vecDotFor(t);
            float[] y = new float[rows + 2];
            vd.dotRows(sharded, 0, rows, x, y, 1);
            assertEquals(0f, y[0], 0f);
            assertEquals(0f, y[rows + 1], 0f);
            for (int r = 0; r < rows; r++) {
                assertEquals(t + " row " + r, vd.dotRow(single, r, x), y[r + 1], 0f);
            }
        }
    }

    private static ByteBuffer slice(ByteBuffer src, int from, int len) {
        ByteBuffer d = src.duplicate();
        d.position(from);
        d.limit(from + len);
        return d.slice().order(ByteOrder.LITTLE_ENDIAN);
    }

    @Test
    public void factoriesCoverExactlyTheirTypes() {
        for (GGMLType t : GGMLType.values()) {
            int owners = (LegacyVecDot.supports(t) ? 1 : 0) + (Q8VecDot.supports(t) ? 1 : 0)
                    + (Fp4VecDot.supports(t) ? 1 : 0) + (KQuantVecDot.supports(t) ? 1 : 0);
            boolean cq1 = false;
            for (GGMLType c : CQ1_TYPES) {
                cq1 |= c == t;
            }
            assertEquals(t.toString(), cq1 ? 1 : 0, owners);
            int listed = (LegacyVecDot.supportedTypes().contains(t) ? 1 : 0)
                    + (Q8VecDot.supportedTypes().contains(t) ? 1 : 0)
                    + (Fp4VecDot.supportedTypes().contains(t) ? 1 : 0)
                    + (KQuantVecDot.supportedTypes().contains(t) ? 1 : 0);
            assertEquals(t.toString(), owners, listed);
        }
        for (GGMLType t : CQ1_TYPES) {
            GGMLType expected = CpuActivationQuantizer.activationType(t);
            if (LegacyVecDot.supports(t)) {
                assertEquals(t.toString(), expected, LegacyVecDot.forType(t).activationType());
                assertEquals(t, LegacyVecDot.forType(t).weightType());
            } else if (Q8VecDot.supports(t)) {
                assertEquals(t.toString(), expected, Q8VecDot.forType(t).activationType());
                assertEquals(t, Q8VecDot.forType(t).weightType());
            } else if (Fp4VecDot.supports(t)) {
                assertEquals(t.toString(), expected, Fp4VecDot.forType(t).activationType());
                assertEquals(t, Fp4VecDot.forType(t).weightType());
            } else {
                assertEquals(t.toString(), expected, KQuantVecDot.forType(t).activationType());
                assertEquals(t, KQuantVecDot.forType(t).weightType());
            }
        }
        assertSame(LegacyVecDot.Q5_1, LegacyVecDot.forType(GGMLType.Q5_1));
        assertSame(KQuantVecDot.Q6_K, KQuantVecDot.forType(GGMLType.Q6_K));
        assertEquals(GGMLType.Q8_1, LegacyVecDot.Q4_1.activationType());
        assertEquals(GGMLType.Q8_0, LegacyVecDot.Q5_0.activationType());
        for (GGMLType t : new GGMLType[]{GGMLType.Q8_K, GGMLType.IQ4_NL, GGMLType.F16}) {
            try {
                KQuantVecDot.forType(t);
                fail(t + " is not a K-quant handled by KQuantVecDot");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains(t.toString()));
            }
        }
    }

    @Test
    public void mismatchedArgumentsAreRejected() {
        Random rnd = new Random(7);
        CpuTensorRef q40 = CpuTensorRef.of(GGMLType.Q4_0, randomRows(GGMLType.Q4_0, 2, 64, rnd), 2, 64);
        float[] x64 = gaussian(64, rnd);
        expectIllegalArgument(() -> LegacyVecDot.Q4_0.dotRow(q40, 0, CpuActivationQuantizer.quantizeQ8_1(x64, 0, 64)));
        expectIllegalArgument(() -> LegacyVecDot.Q4_0.dotRow(q40, 0, CpuActivationQuantizer.quantizeQ8_0(x64, 0, 32)));
        expectIllegalArgument(() -> LegacyVecDot.Q4_1.dotRow(q40, 0, CpuActivationQuantizer.quantizeQ8_1(x64, 0, 64)));
        expectIllegalArgument(() -> KQuantVecDot.Q4_K.dotRow(q40, 0, CpuActivationQuantizer.quantizeQ8_0(x64, 0, 64)));
        QuantizedActivations ok = CpuActivationQuantizer.quantizeQ8_0(x64, 0, 64);
        try {
            LegacyVecDot.Q4_0.dotRows(q40, 1, 2, ok, new float[2], 0);
            fail("rows [1, 3) exceed the 2-row tensor");
        } catch (IndexOutOfBoundsException expected) {
            assertTrue(expected.getMessage().contains("2-row"));
        }
        try {
            LegacyVecDot.Q4_0.dotRows(q40, 0, 2, ok, new float[2], 1);
            fail("output range [1, 3) exceeds the array");
        } catch (IndexOutOfBoundsException expected) {
            assertTrue(expected.getMessage().contains("array length"));
        }
        assertFalse(Float.isNaN(LegacyVecDot.Q4_0.dotRow(q40, 1, ok)));
    }

    private static void expectIllegalArgument(Runnable r) {
        try {
            r.run();
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            assertFalse(expected.getMessage().isEmpty());
        }
    }
}
