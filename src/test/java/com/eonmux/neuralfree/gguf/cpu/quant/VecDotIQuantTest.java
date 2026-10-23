package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.Random;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for the i-quant, ternary and floating-point vec-dot kernels ({@link IQuantVecDot}, {@link TernaryVecDot},
 * {@link NewFpVecDot}, {@link Fp16VecDot}): a hand-traced IQ2_XXS x Q8_K block, agreement of every kernel with the
 * bit-exact CPU decoders on random blocks, multi-shard row iteration, and argument validation.
 */
public class VecDotIQuantTest {

    private static final GGMLType[] IQ_TYPES = {GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S,
            GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ1_S, GGMLType.IQ1_M, GGMLType.IQ4_NL, GGMLType.IQ4_XS};
    private static final GGMLType[] TERNARY_TYPES = {GGMLType.TQ1_0, GGMLType.TQ2_0};
    private static final GGMLType[] NEW_FP_TYPES = {GGMLType.NVFP4, GGMLType.Q1_0, GGMLType.Q2_0};
    private static final GGMLType[] FLOAT_TYPES = {GGMLType.F32, GGMLType.F16, GGMLType.BF16};

    private static final int HALF_ONE = 0x3C00;

    // ------------------------------------------------------------------ helpers

    private static CpuVecDot vecDot(GGMLType t) {
        if (IQuantVecDot.supports(t)) {
            return IQuantVecDot.forType(t);
        }
        if (TernaryVecDot.supports(t)) {
            return TernaryVecDot.forType(t);
        }
        if (NewFpVecDot.supports(t)) {
            return NewFpVecDot.forType(t);
        }
        return Fp16VecDot.forType(t);
    }

    private static void putHalf(byte[] row, int at, int h) {
        row[at] = (byte) h;
        row[at + 1] = (byte) (h >>> 8);
    }

    /** Byte offset of the half scale {@code d} inside one block, or -1 when the block has none. */
    private static int halfScaleOffset(GGMLType t) {
        switch (t) {
            case TQ1_0:
                return BlockLayout.TQ1_0_D;
            case TQ2_0:
                return BlockLayout.TQ2_0_D;
            case IQ1_M:
            case NVFP4:
                return -1;
            default:
                return 0;
        }
    }

    /** A random half in [0.01, 1), as bits. */
    private static int randomScale(Random rnd) {
        return FpBlockDecoders.fp32ToFp16(0.01f + rnd.nextFloat() * 0.99f);
    }

    /**
     * Random {@code rows x cols} tensor bytes of a quantized type with every half scale finite; all other
     * bits are random, which is a valid encoding for every type here.
     */
    private static byte[] randomQuantRows(GGMLType t, int rows, int cols, Random rnd) {
        BlockLayout layout = BlockLayout.of(t);
        int nBlocks = rows * (cols / layout.blockElements());
        byte[] data = new byte[nBlocks * layout.blockBytes()];
        rnd.nextBytes(data);
        int dOff = halfScaleOffset(t);
        for (int blk = 0; blk < nBlocks; blk++) {
            int base = blk * layout.blockBytes();
            if (dOff >= 0) {
                putHalf(data, base + dOff, randomScale(rnd));
            } else if (t == GGMLType.IQ1_M) {
                int h = randomScale(rnd);
                for (int k = 0; k < 4; k++) {
                    int at = base + BlockLayout.IQ1_M_SCALES + 2 * k;
                    int word = (data[at] & 0xFF) | ((data[at + 1] & 0xFF) << 8);
                    word = (word & 0x0FFF) | (((h >>> (4 * k)) & 0xF) << 12);
                    putHalf(data, at, word);
                }
            }
        }
        return data;
    }

    /** Random plain float rows (F32, F16 or BF16) with values in [-1, 1). */
    private static byte[] randomFloatRows(GGMLType t, int rows, int cols, Random rnd) {
        int elemBytes = BlockLayout.of(t).blockBytes();
        ByteBuffer bb = ByteBuffer.allocate(rows * cols * elemBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < rows * cols; i++) {
            float v = rnd.nextFloat() * 2f - 1f;
            if (t == GGMLType.F32) {
                bb.putFloat(v);
            } else if (t == GGMLType.F16) {
                bb.putShort((short) FpBlockDecoders.fp32ToFp16(v));
            } else {
                bb.putShort((short) (Float.floatToRawIntBits(v) >>> 16));
            }
        }
        return bb.array();
    }

    private static float[] randomActivations(int n, Random rnd) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (rnd.nextFloat() * 2f - 1f) * (rnd.nextInt(16) == 0 ? 8f : 1f);
        }
        return x;
    }

    /** The activation value the vec-dot sees for element {@code i}. */
    private static double dequantActivation(QuantizedActivations x, int i) {
        if (x.type() == GGMLType.F32) {
            return x.value(i);
        }
        return (double) x.quant(i) * x.scale(i / x.blockSize());
    }

    /**
     * Asserts {@code dotRow} against the decoder reference {@code sum(decode(w)[j] * x[j])}, with a tolerance
     * relative to {@code sum |w[j] * x[j]|} that covers float rounding only.
     */
    private static void assertMatchesDecoder(CpuVecDot dot, CpuTensorRef w, QuantizedActivations x) {
        int cols = (int) w.cols();
        float[] decoded = new float[cols];
        for (long r = 0; r < w.rows(); r++) {
            w.decodeRow(r, decoded, 0);
            double ref = 0;
            double mag = 0;
            for (int j = 0; j < cols; j++) {
                double p = decoded[j] * dequantActivation(x, j);
                ref += p;
                mag += Math.abs(p);
            }
            float got = dot.dotRow(w, r, x);
            assertEquals(w.type() + " row " + r, ref, got, 1e-4 * mag + 1e-5);
        }
    }

    // ------------------------------------------------------------------ hand trace

    /**
     * IQ2_XXS block with {@code d = 1}; sub-block 0 has scale nibble 3 ({@code ls = 7}) and sign index 1
     * ({@code ksigns_iq2xs[1] = 0x81}: lanes 0 and 7 negative) on its first 8-group; every grid index is 0
     * ({@code iq2xxs_grid[0]}: all lanes 8). Activations are all 1, so Q8_K gives {@code q = -127},
     * {@code d = -1/127}.
     * <p>
     * Sub-block 0: {@code sumi = -127 * 8 * (30 - 2) = -28448}, times 7 = -199136. Sub-blocks 1-7:
     * {@code sumi = -127 * 8 * 32 = -32512}, times {@code ls = 1}, seven times = -227584. {@code bsum =
     * -426720}; {@code sumf = (1 * -1/127) * -426720 = 3360}; result {@code 0.125 * 3360 = 420}, which equals
     * the decoded sum {@code 32 * 7 - 2 * 2 * 7 + 7 * 32 * 1}.
     */
    @Test
    public void iq2XxsTimesQ8KHandTrace() {
        byte[] block = new byte[BlockLayout.IQ2_XXS_BYTES];
        putHalf(block, BlockLayout.IQ2_XXS_D, HALF_ONE);
        int aux1 = (3 << 28) | 1;
        int at = BlockLayout.IQ2_XXS_QS + 4;
        for (int k = 0; k < 4; k++) {
            block[at + k] = (byte) (aux1 >>> (8 * k));
        }
        CpuTensorRef w = CpuTensorRef.of(GGMLType.IQ2_XXS, ByteBuffer.wrap(block), 1, BlockLayout.QK_K);
        float[] ones = new float[BlockLayout.QK_K];
        Arrays.fill(ones, 1f);
        QuantizedActivations x = CpuActivationQuantizer.quantizeFor(GGMLType.IQ2_XXS, ones, 0, ones.length);
        assertEquals(GGMLType.Q8_K, x.type());
        assertEquals(-127, x.quant(0));

        assertEquals(420f, IQuantVecDot.IQ2_XXS.dotRow(w, 0, x), 1e-3f);

        float[] decoded = new float[BlockLayout.QK_K];
        w.decodeRow(0, decoded, 0);
        assertEquals(-7f, decoded[0], 0f);
        assertEquals(7f, decoded[1], 0f);
        assertEquals(-7f, decoded[7], 0f);
        assertEquals(1f, decoded[32], 0f);
    }

    /** All-zero grid indices with every scale nibble 0 decode to 1.0; the dot with ones is 256. */
    @Test
    public void iq2XxsUniformBlock() {
        byte[] block = new byte[BlockLayout.IQ2_XXS_BYTES];
        putHalf(block, BlockLayout.IQ2_XXS_D, HALF_ONE);
        CpuTensorRef w = CpuTensorRef.of(GGMLType.IQ2_XXS, ByteBuffer.wrap(block), 1, BlockLayout.QK_K);
        float[] ones = new float[BlockLayout.QK_K];
        Arrays.fill(ones, 1f);
        QuantizedActivations x = CpuActivationQuantizer.quantizeQ8_K(ones, 0, ones.length);
        assertEquals(256f, IQuantVecDot.IQ2_XXS.dotRow(w, 0, x), 1e-3f);
    }

    // ------------------------------------------------------------------ decoder agreement

    private static void checkQuantType(GGMLType t, long seed) {
        Random rnd = new Random(seed);
        int cols = 2 * BlockLayout.of(t).blockElements();
        int rows = 3;
        byte[] data = randomQuantRows(t, rows, cols, rnd);
        CpuTensorRef w = CpuTensorRef.of(t, ByteBuffer.wrap(data), rows, cols);
        QuantizedActivations x = CpuActivationQuantizer.quantizeFor(t, randomActivations(cols, rnd), 0, cols);
        assertMatchesDecoder(vecDot(t), w, x);
    }

    @Test
    public void iQuantKernelsMatchDecoders() {
        for (GGMLType t : IQ_TYPES) {
            checkQuantType(t, 1000 + t.getId());
        }
    }

    @Test
    public void ternaryKernelsMatchDecoders() {
        for (GGMLType t : TERNARY_TYPES) {
            checkQuantType(t, 2000 + t.getId());
        }
    }

    @Test
    public void newFpKernelsMatchDecoders() {
        for (GGMLType t : NEW_FP_TYPES) {
            checkQuantType(t, 3000 + t.getId());
        }
    }

    @Test
    public void floatKernelsMatchDecodersIncludingTails() {
        for (GGMLType t : FLOAT_TYPES) {
            for (int cols : new int[]{1, 7, 8, 77, 256}) {
                Random rnd = new Random(4000 + 31L * t.getId() + cols);
                byte[] data = randomFloatRows(t, 2, cols, rnd);
                CpuTensorRef w = CpuTensorRef.of(t, ByteBuffer.wrap(data), 2, cols);
                QuantizedActivations x = CpuActivationQuantizer.quantizeFor(t, randomActivations(cols, rnd), 0, cols);
                assertEquals(GGMLType.F32, x.type());
                assertMatchesDecoder(vecDot(t), w, x);
            }
        }
    }

    /** Ternary weights with all-zero activation blocks give exactly 0 (bsums correction included). */
    @Test
    public void ternaryZeroActivationsGiveZero() {
        for (GGMLType t : TERNARY_TYPES) {
            Random rnd = new Random(5000 + t.getId());
            byte[] data = randomQuantRows(t, 1, BlockLayout.QK_K, rnd);
            CpuTensorRef w = CpuTensorRef.of(t, ByteBuffer.wrap(data), 1, BlockLayout.QK_K);
            QuantizedActivations x = CpuActivationQuantizer.quantizeFor(t, new float[BlockLayout.QK_K], 0,
                    BlockLayout.QK_K);
            assertEquals(t.toString(), 0f, vecDot(t).dotRow(w, 0, x), 0f);
        }
    }

    // ------------------------------------------------------------------ row iteration

    @Test
    public void dotRowsAcrossShardsMatchesDotRow() {
        GGMLType[] types = {GGMLType.IQ3_S, GGMLType.TQ1_0, GGMLType.Q1_0, GGMLType.BF16};
        for (GGMLType t : types) {
            Random rnd = new Random(6000 + t.getId());
            int cols = t == GGMLType.BF16 ? 40 : BlockLayout.of(t).blockElements() * 2;
            int rows = 5;
            byte[] data = t == GGMLType.BF16 ? randomFloatRows(t, rows, cols, rnd) : randomQuantRows(t, rows, cols, rnd);
            int rowBytes = (int) BlockLayout.of(t).rowBytes(cols);
            ByteBuffer first = ByteBuffer.wrap(Arrays.copyOfRange(data, 0, 2 * rowBytes));
            ByteBuffer second = ByteBuffer.wrap(Arrays.copyOfRange(data, 2 * rowBytes, data.length));
            CpuTensorRef sharded = new CpuTensorRef(t, new ByteBuffer[]{first, second},
                    new long[]{0L, 2L * rowBytes}, rows, cols, rowBytes);
            CpuTensorRef dense = CpuTensorRef.of(t, ByteBuffer.wrap(data), rows, cols);
            QuantizedActivations x = CpuActivationQuantizer.quantizeFor(t, randomActivations(cols, rnd), 0, cols);
            CpuVecDot dot = vecDot(t);

            float[] y = new float[rows + 2];
            dot.dotRows(sharded, 1, 4, x, y, 2);
            for (int r = 1; r < rows; r++) {
                assertEquals(t + " row " + r, dot.dotRow(dense, r, x), y[r + 1], 0f);
            }
            assertEquals(0f, y[0], 0f);
        }
    }

    // ------------------------------------------------------------------ validation and coverage

    @Test
    public void rejectsWrongActivationType() {
        byte[] block = new byte[BlockLayout.IQ2_XXS_BYTES];
        CpuTensorRef w = CpuTensorRef.of(GGMLType.IQ2_XXS, ByteBuffer.wrap(block), 1, BlockLayout.QK_K);
        QuantizedActivations q8_0 = CpuActivationQuantizer.quantizeQ8_0(new float[BlockLayout.QK_K], 0,
                BlockLayout.QK_K);
        try {
            IQuantVecDot.IQ2_XXS.dotRow(w, 0, q8_0);
            fail("Q8_0 activations must be rejected by the IQ2_XXS vec-dot");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("Q8_K"));
        }
    }

    @Test
    public void rejectsWrongWeightType() {
        byte[] block = new byte[BlockLayout.TQ2_0_BYTES];
        CpuTensorRef w = CpuTensorRef.of(GGMLType.TQ2_0, ByteBuffer.wrap(block), 1, BlockLayout.QK_K);
        QuantizedActivations x = CpuActivationQuantizer.quantizeQ8_K(new float[BlockLayout.QK_K], 0,
                BlockLayout.QK_K);
        try {
            IQuantVecDot.IQ2_XS.dotRow(w, 0, x);
            fail("a TQ2_0 tensor must be rejected by the IQ2_XS vec-dot");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("TQ2_0"));
        }
    }

    @Test
    public void rejectsOutOfRangeRows() {
        byte[] data = new byte[2 * BlockLayout.Q2_0_BYTES];
        CpuTensorRef w = CpuTensorRef.of(GGMLType.Q2_0, ByteBuffer.wrap(data), 2, BlockLayout.QK2_0);
        QuantizedActivations x = CpuActivationQuantizer.quantizeQ8_0(new float[BlockLayout.QK2_0], 0,
                BlockLayout.QK2_0);
        try {
            NewFpVecDot.Q2_0.dotRows(w, 1, 2, x, new float[2], 0);
            fail("rows [1, 3) of a 2-row tensor must be rejected");
        } catch (IndexOutOfBoundsException expected) {
            assertTrue(expected.getMessage().contains("2-row"));
        }
    }

    @Test
    public void forTypeRejectsForeignTypes() {
        try {
            IQuantVecDot.forType(GGMLType.Q4_K);
            fail("Q4_K is not an IQ type");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("Q4_K"));
        }
        assertFalse(TernaryVecDot.supports(GGMLType.Q8_0));
        assertFalse(NewFpVecDot.supports(GGMLType.MXFP4));
        assertFalse(Fp16VecDot.supports(GGMLType.F64));
        assertFalse(IQuantVecDot.supports(null));
    }

    /**
     * Every i-quant, ternary and floating-point type has exactly one kernel class, and each kernel consumes the
     * quantizer's activation type.
     */
    @Test
    public void coverageIsCompleteAndDisjoint() {
        Set<GGMLType> expected = EnumSet.noneOf(GGMLType.class);
        expected.addAll(Arrays.asList(IQ_TYPES));
        expected.addAll(Arrays.asList(TERNARY_TYPES));
        expected.addAll(Arrays.asList(NEW_FP_TYPES));
        expected.addAll(Arrays.asList(FLOAT_TYPES));
        for (GGMLType t : expected) {
            int owners = (IQuantVecDot.supports(t) ? 1 : 0) + (TernaryVecDot.supports(t) ? 1 : 0)
                    + (NewFpVecDot.supports(t) ? 1 : 0) + (Fp16VecDot.supports(t) ? 1 : 0);
            assertEquals(t + " owners", 1, owners);
        }
        assertEquals(EnumSet.copyOf(Arrays.asList(IQ_TYPES)), EnumSet.copyOf(IQuantVecDot.supportedTypes()));
        for (GGMLType t : IQ_TYPES) {
            assertEquals(CpuActivationQuantizer.activationType(t), IQuantVecDot.forType(t).activationType());
        }
        for (GGMLType t : TERNARY_TYPES) {
            assertEquals(CpuActivationQuantizer.activationType(t), TernaryVecDot.forType(t).activationType());
        }
        for (GGMLType t : NEW_FP_TYPES) {
            assertEquals(CpuActivationQuantizer.activationType(t), NewFpVecDot.forType(t).activationType());
        }
        for (GGMLType t : FLOAT_TYPES) {
            assertEquals(CpuActivationQuantizer.activationType(t), Fp16VecDot.forType(t).activationType());
        }
    }
}
