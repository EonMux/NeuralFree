package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.Assume;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for the Vector API CPU vec-dot selection ({@link CpuSimd}): agreement of every Vector API kernel with
 * its scalar counterpart bit for bit, the dispatch of {@link CpuMatVec#vecDot}, the property gate and its
 * rejection, and the fallback description. The kernel tests assume the engine ({@code assumeTrue}), so the
 * class also runs on a JVM without the incubator module.
 */
public class SimdVecDotTest {

    private static final GGMLType[] SIMD_TYPES = {
            GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0,
            GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
            GGMLType.F32, GGMLType.F16, GGMLType.BF16, GGMLType.IQ4_NL, GGMLType.IQ4_XS,
            GGMLType.TQ1_0, GGMLType.TQ2_0, GGMLType.Q1_0, GGMLType.Q2_0, GGMLType.MXFP4,
            GGMLType.I8_B128, GGMLType.NF4, GGMLType.FP4_BNB,
    };

    /** The scalar kernel of {@code t}, the reference the Vector API kernel must match exactly. */
    private static CpuVecDot scalarFor(GGMLType t) {
        if (LegacyVecDot.supports(t)) {
            return LegacyVecDot.forType(t);
        }
        if (Q8VecDot.supports(t)) {
            return Q8VecDot.forType(t);
        }
        if (KQuantVecDot.supports(t)) {
            return KQuantVecDot.forType(t);
        }
        if (Fp16VecDot.supports(t)) {
            return Fp16VecDot.forType(t);
        }
        if (IQuantVecDot.supports(t)) {
            return IQuantVecDot.forType(t);
        }
        if (Fp4VecDot.supports(t)) {
            return Fp4VecDot.forType(t);
        }
        if (NewFpVecDot.supports(t)) {
            return NewFpVecDot.forType(t);
        }
        if (ExtVecDot.supports(t)) {
            return ExtVecDot.forType(t);
        }
        return TernaryVecDot.forType(t);
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
            case IQ4_NL:
                return new int[]{BlockLayout.IQ4_NL_D};
            case IQ4_XS:
                return new int[]{BlockLayout.IQ4_XS_D};
            case MXFP4:
                return new int[0];
            case TQ1_0:
                return new int[]{BlockLayout.TQ1_0_D};
            case TQ2_0:
                return new int[]{BlockLayout.TQ2_0_D};
            case Q1_0:
                return new int[]{BlockLayout.Q1_0_D};
            case Q2_0:
                return new int[]{BlockLayout.Q2_0_D};
            default:
                return new int[0];
        }
    }

    /** Random blocks with finite, moderate scales, one row of {@code cols} elements per row index. */
    static ByteBuffer randomRows(GGMLType t, int rows, int cols, Random rnd) {
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
            if (t == GGMLType.Q6_K) {
                for (int g = 0; g < 16; g++) {
                    raw[o + BlockLayout.Q6_K_SCALES + g] = (byte) (rnd.nextInt(255) - 127);
                }
            }
            sanitizeBlock(t, raw, b, o, rnd);
        }
        if (t == GGMLType.F32) {
            for (int i = 0; i < rows * cols; i++) {
                b.putFloat(4 * i, (rnd.nextFloat() * 2f - 1f) * 0.1f);
            }
        } else if (t == GGMLType.F16) {
            for (int i = 0; i < rows * cols; i++) {
                b.putShort(2 * i, (short) FpBlockDecoders.fp32ToFp16((rnd.nextFloat() * 2f - 1f) * 0.1f));
            }
        } else if (t == GGMLType.BF16) {
            // finite bf16 values: a NaN lane's payload is not preserved identically by every JIT tier, so
            // random bytes would compare two NaN spellings
            for (int i = 0; i < rows * cols; i++) {
                b.putShort(2 * i, (short) (Float.floatToRawIntBits((rnd.nextFloat() * 2f - 1f) * 0.1f) >>> 16));
            }
        }
        return b;
    }

    /** Makes the scale bytes of one block finite, so a NaN never reaches the parity assertion. */
    private static void sanitizeBlock(GGMLType t, byte[] raw, ByteBuffer b, int o, Random rnd) {
        switch (t) {
            case MXFP4:
                raw[o + BlockLayout.MXFP4_E] = (byte) (118 + rnd.nextInt(14));
                break;
            case I8_B128:
                b.putFloat(o + BlockLayout.I8B128_D, (rnd.nextFloat() * 2f - 1f) * 0.01f);
                break;
            case NF4:
            case FP4_BNB:
                b.putFloat(o + BlockLayout.BNB4_D, 0.01f + rnd.nextFloat() * 2f);
                break;
            default:
                break;
        }
    }

    static float[] gaussian(int n, Random rnd) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) rnd.nextGaussian();
        }
        return x;
    }

    @Test
    public void everyVectorKernelMatchesItsScalarKernelBitForBit() {
        Assume.assumeTrue(CpuSimd.available());
        final int rows = 3;
        final int cols = 512;
        Random rnd = new Random(0x51D);
        for (GGMLType t : SIMD_TYPES) {
            ByteBuffer data = randomRows(t, rows, cols, rnd);
            CpuTensorRef w = CpuTensorRef.of(t, data, rows, cols);
            QuantizedActivations x = CpuActivationQuantizer.quantizeFor(t, gaussian(cols, rnd), 0, cols);
            CpuVecDot scalar = scalarFor(t);
            CpuVecDot simd = CpuSimd.vecDot(t);
            assertNotNull(t + " has no Vector API kernel", simd);
            assertNotSame(t + " dispatches the scalar kernel", scalar, simd);
            float[] ys = new float[rows];
            float[] yv = new float[rows];
            for (int r = 0; r < rows; r++) {
                assertEquals(t + " row " + r + ": Vector API kernel must match the scalar kernel exactly",
                        scalar.dotRow(w, r, x), simd.dotRow(w, r, x), 0f);
            }
            scalar.dotRows(w, 0, rows, x, ys, 0);
            simd.dotRows(w, 0, rows, x, yv, 0);
            assertArrayEquals(t + " dotRows", ys, yv);
        }
    }

    @Test
    public void batchedDotRowsMatchesOnePassPerToken() {
        Assume.assumeTrue(CpuSimd.available());
        final int rows = 3;
        final int cols = 512;
        Random rnd = new Random(0xB47);
        for (GGMLType t : SIMD_TYPES) {
            ByteBuffer data = randomRows(t, rows, cols, rnd);
            CpuTensorRef w = CpuTensorRef.of(t, data, rows, cols);
            CpuVecDot simd = CpuSimd.vecDot(t);
            for (int nTokens : new int[]{2, 3, 5, 8, 9, 16}) {
                QuantizedActivations[] acts = new QuantizedActivations[nTokens];
                float[][] expected = new float[nTokens][rows];
                for (int tok = 0; tok < nTokens; tok++) {
                    acts[tok] = CpuActivationQuantizer.quantizeFor(t, gaussian(cols, rnd), 0, cols);
                    simd.dotRows(w, 0, rows, acts[tok], expected[tok], 0);
                }
                float[] y = new float[nTokens * rows];
                simd.dotRows(w, 0, rows, acts, nTokens, y, 0, rows);
                for (int tok = 0; tok < nTokens; tok++) {
                    for (int r = 0; r < rows; r++) {
                        assertEquals(t + " batch of " + nTokens + ", token " + tok + ", row " + r,
                                expected[tok][r], y[tok * rows + r], 0f);
                    }
                }
            }
        }
    }

    @Test
    public void tiledProductMatchesTheRowLoopOverRemainderRowsAndTokenTiles() {
        Assume.assumeTrue(CpuSimd.available());
        Random rnd = new Random(0x711E);
        for (GGMLType t : SIMD_TYPES) {
            for (int rows : new int[]{1, 2, 3, 5}) {
                final int cols = 512;
                CpuTensorRef w = CpuTensorRef.of(t, randomRows(t, rows, cols, rnd), rows, cols);
                CpuVecDot simd = CpuSimd.vecDot(t);
                for (int nTokens : new int[]{1, 7, 8, 9}) {
                    QuantizedActivations[] acts = new QuantizedActivations[nTokens];
                    float[][] expected = new float[nTokens][rows];
                    for (int tok = 0; tok < nTokens; tok++) {
                        acts[tok] = CpuActivationQuantizer.quantizeFor(t, gaussian(cols, rnd), 0, cols);
                        simd.dotRows(w, 0, rows, acts[tok], expected[tok], 0);
                    }
                    float[] y = new float[nTokens * rows + 3];
                    simd.dotTiles(w, 0, rows, acts, nTokens, y, 2, rows);
                    for (int tok = 0; tok < nTokens; tok++) {
                        for (int r = 0; r < rows; r++) {
                            assertEquals(t + " rows " + rows + ", " + nTokens + " tokens, token " + tok
                                            + ", row " + r, expected[tok][r], y[2 + tok * rows + r], 0f);
                        }
                    }
                    assertEquals(0f, y[0], 0f);
                    assertEquals(0f, y[1], 0f);
                    assertEquals(0f, y[y.length - 1], 0f);
                }
            }
        }
    }

    /**
     * The prepared layout of the token-tile kernels (Q8_0, Q4_K, Q6_K) is built once and serves every tile-aligned
     * token range and every row range with the row loop's values, a null layout falls back to the activation rows,
     * and misaligned starts or a layout of another width are rejected by name.
     */
    @Test
    public void preparedTilesServeEveryTileStartWithTheRowLoopsValues() {
        Assume.assumeTrue(CpuSimd.available());
        Random rnd = new Random(0x7A1E5);
        final int rows = 5;
        final int cols = 512;
        final int nTokens = 21;
        for (GGMLType t : new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K}) {
            CpuTensorRef w = CpuTensorRef.of(t, randomRows(t, rows, cols, rnd), rows, cols);
            CpuVecDot simd = CpuSimd.vecDot(t);
            QuantizedActivations[] acts = new QuantizedActivations[nTokens];
            float[][] expected = new float[nTokens][rows];
            for (int tok = 0; tok < nTokens; tok++) {
                acts[tok] = CpuActivationQuantizer.quantizeFor(t, gaussian(cols, rnd), 0, cols);
                simd.dotRows(w, 0, rows, acts[tok], expected[tok], 0);
            }
            CpuVecDot.TileActivations tiles = simd.prepareTiles(acts, nTokens);
            assertNotNull(t + " layout", tiles);
            assertEquals(t + " layout tokens", nTokens, tiles.tokens());
            int step = simd.tileTokens();
            assertEquals(t + " tile", 8, step);
            for (int start = 0; start < nTokens; start += step) {
                for (int n : new int[]{1, step, nTokens - start}) {
                    for (int r0 = 0; r0 < 2 && n <= nTokens - start; r0++) {
                        float[] y = new float[n * rows + 1];
                        simd.dotTiles(w, r0, rows - r0, tiles, acts, start, n, y, 1, rows);
                        for (int tok = 0; tok < n; tok++) {
                            for (int r = 0; r < rows - r0; r++) {
                                assertEquals(t + " tokens [" + start + ", +" + n + "), row " + (r0 + r),
                                        expected[start + tok][r0 + r], y[1 + tok * rows + r], 0f);
                            }
                        }
                    }
                }
            }
            float[] y = new float[5 * rows];
            simd.dotTiles(w, 0, rows, null, acts, 3, 5, y, 0, rows);
            for (int tok = 0; tok < 5; tok++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(t + " without a layout, token " + (3 + tok), expected[3 + tok][r],
                            y[tok * rows + r], 0f);
                }
            }
            IllegalArgumentException misaligned = assertThrows(IllegalArgumentException.class,
                    () -> simd.dotTiles(w, 0, rows, tiles, acts, 3, 2, new float[2 * rows], 0, rows));
            assertTrue(misaligned.getMessage(), misaligned.getMessage().contains("multiple"));
            int wide = 2 * cols;
            CpuTensorRef w2 = CpuTensorRef.of(t, randomRows(t, rows, wide, rnd), rows, wide);
            IllegalArgumentException width = assertThrows(IllegalArgumentException.class,
                    () -> simd.dotTiles(w2, 0, rows, tiles, acts, 0, 2, new float[2 * rows], 0, rows));
            assertTrue(width.getMessage(), width.getMessage().contains("cannot be dotted"));
        }
    }

    @Test
    public void theTiledProductRejectsBadRangesByName() {
        Assume.assumeTrue(CpuSimd.available());
        Random rnd = new Random(0xBAD);
        final int rows = 2;
        final int cols = 512;
        CpuTensorRef w = CpuTensorRef.of(GGMLType.Q4_0, randomRows(GGMLType.Q4_0, rows, cols, rnd), rows, cols);
        CpuVecDot simd = CpuSimd.vecDot(GGMLType.Q4_0);
        QuantizedActivations[] acts = {CpuActivationQuantizer.quantizeFor(GGMLType.Q4_0, gaussian(cols, rnd), 0,
                cols), CpuActivationQuantizer.quantizeFor(GGMLType.Q4_0, gaussian(cols, rnd), 0, cols)};
        float[] y = new float[2 * rows];
        try {
            simd.dotTiles(w, 0, rows, acts, 3, y, 0, rows);
            fail("expected the rejection of a token count past the activation array");
        } catch (IndexOutOfBoundsException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("3 tokens"));
        }
        try {
            simd.dotTiles(w, 0, rows, acts, 2, y, 0, rows - 1);
            fail("expected the rejection of a stride below the row count");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("stride"));
        }
        try {
            simd.dotTiles(w, 1, rows, acts, 2, y, 0, rows);
            fail("expected the rejection of rows past the tensor");
        } catch (IndexOutOfBoundsException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("rows"));
        }
    }

    @Test
    public void floatKernelsMatchOnRowsThatDoNotFillTheEightLanes() {
        Assume.assumeTrue(CpuSimd.available());
        final int rows = 2;
        final int cols = 517;
        Random rnd = new Random(0x517);
        for (GGMLType t : new GGMLType[]{GGMLType.F32}) {
            CpuTensorRef w = CpuTensorRef.of(t, randomRows(t, rows, cols, rnd), rows, cols);
            QuantizedActivations x = CpuActivationQuantizer.quantizeFor(t, gaussian(cols, rnd), 0, cols);
            CpuVecDot scalar = scalarFor(t);
            CpuVecDot simd = CpuSimd.vecDot(t);
            for (int r = 0; r < rows; r++) {
                assertEquals(t + " row " + r, scalar.dotRow(w, r, x), simd.dotRow(w, r, x), 0f);
            }
        }
    }

    @Test
    public void vectorKernelsDotRowsAcrossShardsLikeTheScalarOnes() {
        Assume.assumeTrue(CpuSimd.available());
        final int rows = 4;
        final int cols = 512;
        Random rnd = new Random(42);
        for (GGMLType t : SIMD_TYPES) {
            ByteBuffer whole = randomRows(t, rows, cols, rnd);
            long rowBytes = BlockLayout.of(t).rowBytes(cols);
            ByteBuffer first = slice(whole, 0, (int) (2 * rowBytes));
            ByteBuffer second = slice(whole, (int) (2 * rowBytes), (int) (2 * rowBytes));
            CpuTensorRef sharded = new CpuTensorRef(t, new ByteBuffer[]{first, second},
                    new long[]{0L, 2 * rowBytes}, rows, cols, rowBytes);
            CpuTensorRef single = CpuTensorRef.of(t, whole, rows, cols);
            QuantizedActivations x = CpuActivationQuantizer.quantizeFor(t, gaussian(cols, rnd), 0, cols);
            CpuVecDot simd = CpuSimd.vecDot(t);
            CpuVecDot scalar = scalarFor(t);
            float[] y = new float[rows + 2];
            simd.dotRows(sharded, 0, rows, x, y, 1);
            assertEquals(0f, y[0], 0f);
            assertEquals(0f, y[rows + 1], 0f);
            for (int r = 0; r < rows; r++) {
                assertEquals(t + " row " + r, scalar.dotRow(single, r, x), y[r + 1], 0f);
            }
        }
    }

    /**
     * The prepared token-tile products read rows on both sides of a shard boundary, the boundary inside a two-row
     * register tile included, with the values of the unsharded tensor.
     */
    @Test
    public void preparedTilesDotRowsAcrossShardsLikeOneBuffer() {
        Assume.assumeTrue(CpuSimd.available());
        final int rows = 7;
        final int cols = 512;
        final int nTokens = 19;
        Random rnd = new Random(0x5AD);
        for (GGMLType t : new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K}) {
            ByteBuffer whole = randomRows(t, rows, cols, rnd);
            long rowBytes = BlockLayout.of(t).rowBytes(cols);
            ByteBuffer first = slice(whole, 0, (int) (3 * rowBytes));
            ByteBuffer second = slice(whole, (int) (3 * rowBytes), (int) (4 * rowBytes));
            CpuTensorRef sharded = new CpuTensorRef(t, new ByteBuffer[]{first, second},
                    new long[]{0L, 3 * rowBytes}, rows, cols, rowBytes);
            CpuTensorRef single = CpuTensorRef.of(t, whole, rows, cols);
            CpuVecDot simd = CpuSimd.vecDot(t);
            QuantizedActivations[] acts = new QuantizedActivations[nTokens];
            for (int tok = 0; tok < nTokens; tok++) {
                acts[tok] = CpuActivationQuantizer.quantizeFor(t, gaussian(cols, rnd), 0, cols);
            }
            CpuVecDot.TileActivations tiles = simd.prepareTiles(acts, nTokens);
            for (int r0 : new int[]{0, 1, 2}) {
                int nr = rows - r0;
                float[] expected = new float[nTokens * nr];
                float[] actual = new float[nTokens * nr];
                simd.dotTiles(single, r0, nr, tiles, acts, 0, nTokens, expected, 0, nr);
                simd.dotTiles(sharded, r0, nr, tiles, acts, 0, nTokens, actual, 0, nr);
                assertArrayEquals(t + " rows from " + r0 + " across the shard boundary at row 3", expected, actual);
                float[] rowLoop = new float[nr];
                simd.dotRows(single, r0, nr, acts[nTokens - 1], rowLoop, 0);
                for (int r = 0; r < nr; r++) {
                    assertEquals(t + " last token, row " + (r0 + r), rowLoop[r], actual[(nTokens - 1) * nr + r], 0f);
                }
            }
        }
    }

    @Test
    public void theDispatchSelectsTheVectorKernels() {
        Assume.assumeTrue(CpuSimd.available());
        for (GGMLType t : SIMD_TYPES) {
            CpuVecDot simd = CpuSimd.vecDot(t);
            assertSame(t + " must dispatch the Vector API kernel", simd, CpuMatVec.vecDot(t));
            assertTrue(CpuSimd.supports(t));
            assertTrue(CpuSimd.supportedTypes().contains(t));
        }
        assertTrue(CpuSimd.describe().startsWith("vector/"));
    }

    @Test
    public void thePropertyGateRejectsValuesOtherThanTrueAndFalseByName() {
        Properties p = new Properties();
        assertTrue("absent property selects the engines", CpuSimd.enabled(p));
        p.setProperty(CpuSimd.PROP_SIMD, "true");
        assertTrue(CpuSimd.enabled(p));
        p.setProperty(CpuSimd.PROP_SIMD, "false");
        assertTrue(!CpuSimd.enabled(p));
        p.setProperty(CpuSimd.PROP_SIMD, "maybe");
        try {
            CpuSimd.enabled(p);
            fail("expected the rejection of " + CpuSimd.PROP_SIMD + "=maybe");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains(CpuSimd.PROP_SIMD));
            assertTrue(expected.getMessage().contains("maybe"));
        }
    }

    @Test
    public void unsupportedTypesHaveNoVectorKernel() {
        for (GGMLType t : new GGMLType[]{GGMLType.Q8_K, GGMLType.Q8_1, GGMLType.I8, GGMLType.I16,
                GGMLType.I32, GGMLType.I64, GGMLType.F64, GGMLType.NVFP4,
                GGMLType.F8_E4M3, GGMLType.F8_E5M2, GGMLType.F8_E4M3_B128, GGMLType.F8_E4M3_B32,
                GGMLType.F8_E4M3FN, GGMLType.MXFP8, GGMLType.MXFP6_E3M2, GGMLType.MXFP6_E2M3,
                GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S, GGMLType.IQ3_XXS, GGMLType.IQ3_S,
                GGMLType.IQ3_K, GGMLType.IQ3_KS, GGMLType.IQ1_S, GGMLType.IQ1_M}) {
            assertTrue(t + " has no Vector API kernel here", !CpuSimd.supports(t));
        }
    }

    @Test
    public void withoutTheIncubatorModuleTheScalarKernelsRemain() throws Exception {
        String out = probe();
        assertTrue(out, out.contains("available=false"));
        assertTrue(out, out.contains("describe=scalar"));
        assertTrue(out, out.contains("q4_0=LegacyVecDot{Q4_0 x Q8_0}"));
        assertTrue(out, out.contains("q4_1=LegacyVecDot{Q4_1 x Q8_1}"));
        assertTrue(out, out.contains("q2_K=KQuantVecDot{Q2_K x Q8_K}"));
        assertTrue(out, out.contains("f32=Fp16VecDot{F32 x F32}"));
        assertTrue(out, out.contains("q1_0=NewFpVecDot{Q1_0 x Q8_0}"));
        assertTrue(out, out.contains("mxfp4=Fp4VecDot{MXFP4 x Q8_0}"));
        assertTrue(out, out.contains("repacked_q8=none"));
    }

    @Test
    public void theDisabledPropertyKeepsTheScalarKernelsEvenWithTheModule() throws Exception {
        Assume.assumeTrue(Runtime.version().feature() >= 17);
        String out = probe("--add-modules=jdk.incubator.vector", "-D" + CpuSimd.PROP_SIMD + "=false");
        assertTrue(out, out.contains("available=false"));
        assertTrue(out, out.contains("describe=scalar"));
        assertTrue(out, out.contains("q4_0=LegacyVecDot{Q4_0 x Q8_0}"));
        assertTrue(out, out.contains("q4_1=LegacyVecDot{Q4_1 x Q8_1}"));
        assertTrue(out, out.contains("q2_K=KQuantVecDot{Q2_K x Q8_K}"));
        assertTrue(out, out.contains("q1_0=NewFpVecDot{Q1_0 x Q8_0}"));
        assertTrue(out, out.contains("mxfp4=Fp4VecDot{MXFP4 x Q8_0}"));
        assertTrue(out, out.contains("repacked_q8=none"));
    }

    @Test
    public void anInvalidPropertyIsRejectedByNameInTheEngineQueries() throws Exception {
        String out = probe("-D" + CpuSimd.PROP_SIMD + "=maybe");
        assertTrue(out, out.contains("rejected=IllegalArgumentException"));
        assertTrue(out, out.contains(CpuSimd.PROP_SIMD));
        assertTrue(out, out.contains("maybe"));
    }

    /** Runs {@link SimdFallbackProbe} in a fresh JVM with {@code jvmArgs} before the classpath. */
    private static String probe(String... jvmArgs) throws Exception {
        List<String> cmd = new ArrayList<>();
        cmd.add(Paths.get(System.getProperty("java.home"), "bin", "java").toString());
        cmd.addAll(Arrays.asList(jvmArgs));
        cmd.add("-cp");
        cmd.add(System.getProperty("java.class.path"));
        cmd.add(SimdFallbackProbe.class.getName());
        ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
        // a host-level option must not inject the module the test withholds
        pb.environment().remove("JAVA_TOOL_OPTIONS");
        pb.environment().remove("_JAVA_OPTIONS");
        pb.environment().remove("JDK_JAVA_OPTIONS");
        Process p = pb.start();
        StringBuilder out = new StringBuilder();
        try (BufferedReader r = new BufferedReader(
                new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                out.append(line).append('\n');
            }
        }
        assertTrue("probe timed out: " + out, p.waitFor(30, TimeUnit.SECONDS));
        assertEquals("probe exit: " + out, 0, p.exitValue());
        return out.toString();
    }

    private static ByteBuffer slice(ByteBuffer src, int from, int len) {
        ByteBuffer d = src.duplicate();
        d.position(from);
        d.limit(from + len);
        return d.slice().order(ByteOrder.LITTLE_ENDIAN);
    }

    /** JUnit 4's assertArrayEquals lives in a differently named class than the float overload users expect. */
    private static void assertArrayEquals(String message, float[] expected, float[] actual) {
        assertEquals(message + ": lengths", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(message + " [" + i + "]", expected[i], actual[i], 0f);
        }
    }
}
