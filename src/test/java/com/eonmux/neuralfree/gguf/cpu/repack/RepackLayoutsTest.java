package com.eonmux.neuralfree.gguf.cpu.repack;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuActivationQuantizer;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuSimd;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.KQuantVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.Q4_Kx8VecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.Q8_0x4VecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * Tests for the repacked layouts beyond {@link Q4_0x8}: the byte layout of {@link Q8_0x4} and {@link Q4_Kx8}
 * against independent ports of {@code make_block_q8_0x4} and {@code make_block_q4_Kx8}, bit-exact agreement of
 * their scalar and Vector API kernels with the plain kernels, the {@link CpuRepack} plan dispatch, and the
 * load-time repacking of {@link CpuWeights}.
 */
public class RepackLayoutsTest {

    private static ByteBuffer randomQ8Rows(int rows, int cols, Random rnd) {
        BlockLayout layout = BlockLayout.of(GGMLType.Q8_0);
        int blocks = rows * (cols / layout.blockElements());
        byte[] raw = new byte[blocks * layout.blockBytes()];
        rnd.nextBytes(raw);
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < blocks; i++) {
            b.putShort(i * layout.blockBytes() + BlockLayout.Q8_0_D,
                    (short) FpBlockDecoders.fp32ToFp16((rnd.nextFloat() * 2f - 1f) * 0.1f));
        }
        return b;
    }

    private static ByteBuffer randomQ4KRows(int rows, int cols, Random rnd) {
        BlockLayout layout = BlockLayout.of(GGMLType.Q4_K);
        int blocks = rows * (cols / layout.blockElements());
        byte[] raw = new byte[blocks * layout.blockBytes()];
        rnd.nextBytes(raw);
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < blocks; i++) {
            int o = i * layout.blockBytes();
            b.putShort(o + BlockLayout.Q4_K_D,
                    (short) FpBlockDecoders.fp32ToFp16((rnd.nextFloat() * 2f - 1f) * 0.1f));
            b.putShort(o + BlockLayout.Q4_K_DMIN,
                    (short) FpBlockDecoders.fp32ToFp16((rnd.nextFloat() * 2f - 1f) * 0.1f));
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

    /** Repacks every row of {@code w} with {@code Q8_0x4} and returns a tensor view over the copy. */
    private static CpuTensorRef repackedQ8(CpuTensorRef w) {
        int groups = (int) (w.cols() / Q8_0x4.ELEMENTS);
        byte[] dst = new byte[(int) (w.rows() * w.rowBytes())];
        for (int r = 0; r < w.rows(); r++) {
            Q8_0x4.repackRows(w.rowBuffer(r), w.rowOffset(r), dst, (int) (r * w.rowBytes()), groups);
        }
        return CpuTensorRef.of(GGMLType.Q8_0, ByteBuffer.wrap(dst).order(ByteOrder.LITTLE_ENDIAN),
                w.rows(), w.cols());
    }

    /** Repacks every row of {@code w} with {@code Q4_Kx8} and returns a tensor view over the copy. */
    private static CpuTensorRef repackedQ4K(CpuTensorRef w) {
        int groups = (int) (w.cols() / Q4_Kx8.ELEMENTS);
        byte[] dst = new byte[(int) (w.rows() * w.rowBytes())];
        for (int r = 0; r < w.rows(); r++) {
            Q4_Kx8.repackRows(w.rowBuffer(r), w.rowOffset(r), dst, (int) (r * w.rowBytes()), groups);
        }
        return CpuTensorRef.of(GGMLType.Q4_K, ByteBuffer.wrap(dst).order(ByteOrder.LITTLE_ENDIAN),
                w.rows(), w.cols());
    }

    /** {@code make_block_q8_0x4} (repack.cpp:3022-3037) with the four-byte interleave. */
    private static byte[] referenceQ8x4(ByteBuffer src, int srcOff, int groups) {
        byte[] out = new byte[groups * Q8_0x4.BYTES];
        for (int g = 0; g < groups; g++) {
            int in = srcOff + g * 4 * BlockLayout.Q8_0_BYTES;
            int at = g * Q8_0x4.BYTES;
            for (int j = 0; j < 4; j++) {
                out[at + j * 2] = src.get(in + j * BlockLayout.Q8_0_BYTES + BlockLayout.Q8_0_D);
                out[at + j * 2 + 1] = src.get(in + j * BlockLayout.Q8_0_BYTES + BlockLayout.Q8_0_D + 1);
            }
            for (int c = 0; c < 8; c++) {
                for (int j = 0; j < 4; j++) {
                    for (int k = 0; k < 4; k++) {
                        out[at + Q8_0x4.QS + c * 16 + j * 4 + k] = src.get(
                                in + j * BlockLayout.Q8_0_BYTES + BlockLayout.Q8_0_QS + c * 4 + k);
                    }
                }
            }
        }
        return out;
    }

    /** {@code make_block_q4_Kx8} (repack.cpp:3177-3252) with the eight-byte interleave. */
    private static byte[] referenceQ4Kx8(ByteBuffer src, int srcOff, int groups) {
        byte[] out = new byte[groups * Q4_Kx8.BYTES];
        int[] s = new int[8];
        int[] m = new int[8];
        for (int g = 0; g < groups; g++) {
            int in = srcOff + g * 8 * BlockLayout.Q4_K_BYTES;
            int at = g * Q4_Kx8.BYTES;
            for (int j = 0; j < 8; j++) {
                int block = in + j * BlockLayout.Q4_K_BYTES;
                out[at + Q4_Kx8.D + 2 * j] = src.get(block + BlockLayout.Q4_K_D);
                out[at + Q4_Kx8.D + 2 * j + 1] = src.get(block + BlockLayout.Q4_K_D + 1);
                out[at + Q4_Kx8.DMIN + 2 * j] = src.get(block + BlockLayout.Q4_K_DMIN);
                out[at + Q4_Kx8.DMIN + 2 * j + 1] = src.get(block + BlockLayout.Q4_K_DMIN + 1);
            }
            for (int i = 0; i < 128; i++) {
                int block = in + (i % 8) * BlockLayout.Q4_K_BYTES + BlockLayout.Q4_K_QS + (i / 8) * 8;
                for (int k = 0; k < 8; k++) {
                    out[at + Q4_Kx8.QS + i * 8 + k] = src.get(block + k);
                }
            }
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 8; j++) {
                    int sc = in + j * BlockLayout.Q4_K_BYTES + BlockLayout.Q4_K_SCALES;
                    s[j] = src.get(sc + i) & 63;
                    m[j] = src.get(sc + i + 4) & 63;
                }
                referenceScales(out, at + Q4_Kx8.SCALES + i * 12, s, m);
            }
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 8; j++) {
                    int sc = in + j * BlockLayout.Q4_K_BYTES + BlockLayout.Q4_K_SCALES;
                    s[j] = ((src.get(sc + i) & 192) >> 2) | (src.get(sc + i + 8) & 15);
                    m[j] = ((src.get(sc + i + 4) & 192) >> 2) | ((src.get(sc + i + 8) & 240) >> 4);
                }
                referenceScales(out, at + Q4_Kx8.SCALES + 48 + i * 12, s, m);
            }
        }
        return out;
    }

    private static void referenceScales(byte[] out, int at, int[] s, int[] m) {
        out[at] = (byte) ((s[0] & 63) + ((s[4] & 48) << 2));
        out[at + 1] = (byte) ((s[1] & 63) + ((s[5] & 48) << 2));
        out[at + 2] = (byte) ((s[2] & 63) + ((s[6] & 48) << 2));
        out[at + 3] = (byte) ((s[3] & 63) + ((s[7] & 48) << 2));
        out[at + 4] = (byte) ((m[0] & 63) + ((m[4] & 48) << 2));
        out[at + 5] = (byte) ((m[1] & 63) + ((m[5] & 48) << 2));
        out[at + 6] = (byte) ((m[2] & 63) + ((m[6] & 48) << 2));
        out[at + 7] = (byte) ((m[3] & 63) + ((m[7] & 48) << 2));
        out[at + 8] = (byte) ((s[4] & 15) + ((m[4] & 15) << 4));
        out[at + 9] = (byte) ((s[5] & 15) + ((m[5] & 15) << 4));
        out[at + 10] = (byte) ((s[6] & 15) + ((m[6] & 15) << 4));
        out[at + 11] = (byte) ((s[7] & 15) + ((m[7] & 15) << 4));
    }

    @Test
    public void theQ8_0x4GroupLayoutMatchesLlamaCpp() {
        ByteBuffer src = randomQ8Rows(1, 4 * BlockLayout.QK8_0, new Random(0x84));
        byte[] got = new byte[Q8_0x4.BYTES];
        Q8_0x4.repackRows(src, 0, got, 0, 1);
        assertArrayEquals(referenceQ8x4(src, 0, 1), got);
    }

    @Test
    public void theQ4_Kx8GroupLayoutMatchesLlamaCpp() {
        ByteBuffer src = randomQ4KRows(1, 8 * BlockLayout.QK_K, new Random(0x4C));
        byte[] got = new byte[Q4_Kx8.BYTES];
        Q4_Kx8.repackRows(src, 0, got, 0, 1);
        assertArrayEquals(referenceQ4Kx8(src, 0, 1), got);
    }

    @Test
    public void theQ8_0x4ScalarKernelMatchesThePlainKernelBitForBit() {
        final int rows = 3;
        final int cols = 512;
        Random rnd = new Random(0x851);
        CpuTensorRef plain = CpuTensorRef.of(GGMLType.Q8_0, randomQ8Rows(rows, cols, rnd), rows, cols);
        QuantizedActivations x = CpuActivationQuantizer.quantizeQ8_0(gaussian(cols, rnd), 0, cols);
        CpuTensorRef repacked = repackedQ8(plain);
        for (int r = 0; r < rows; r++) {
            assertEquals("row " + r, Q8VecDot.Q8_0.dotRow(plain, r, x),
                    Q8_0x4VecDot.INSTANCE.dotRow(repacked, r, x), 0f);
        }
        float[] plainRows = new float[rows];
        float[] repackedRows = new float[rows];
        Q8VecDot.Q8_0.dotRows(plain, 0, rows, x, plainRows, 0);
        Q8_0x4VecDot.INSTANCE.dotRows(repacked, 0, rows, x, repackedRows, 0);
        assertArrayEquals("dotRows", plainRows, repackedRows, 0f);
    }

    @Test
    public void theQ4_Kx8ScalarKernelMatchesThePlainKernelBitForBit() {
        final int rows = 2;
        final int cols = 2048;
        Random rnd = new Random(0x4C8);
        CpuTensorRef plain = CpuTensorRef.of(GGMLType.Q4_K, randomQ4KRows(rows, cols, rnd), rows, cols);
        QuantizedActivations x = CpuActivationQuantizer.quantizeQ8_K(gaussian(cols, rnd), 0, cols);
        CpuTensorRef repacked = repackedQ4K(plain);
        for (int r = 0; r < rows; r++) {
            assertEquals("row " + r, KQuantVecDot.Q4_K.dotRow(plain, r, x),
                    Q4_Kx8VecDot.INSTANCE.dotRow(repacked, r, x), 0f);
        }
        float[] plainRows = new float[rows];
        float[] repackedRows = new float[rows];
        KQuantVecDot.Q4_K.dotRows(plain, 0, rows, x, plainRows, 0);
        Q4_Kx8VecDot.INSTANCE.dotRows(repacked, 0, rows, x, repackedRows, 0);
        assertArrayEquals("dotRows", plainRows, repackedRows, 0f);
    }

    @Test
    public void theVectorRepackedKernelsMatchTheScalarOnes() {
        Assume.assumeTrue(CpuSimd.available());
        Random rnd = new Random(0x852);
        int q8Cols = 512;
        CpuTensorRef q8 = repackedQ8(CpuTensorRef.of(GGMLType.Q8_0, randomQ8Rows(2, q8Cols, rnd), 2,
                q8Cols));
        QuantizedActivations x8 = CpuActivationQuantizer.quantizeQ8_0(gaussian(q8Cols, rnd), 0, q8Cols);
        CpuVecDot simdQ8 = CpuSimd.repacked(GGMLType.Q8_0);
        assertNotNull(simdQ8);
        for (int r = 0; r < 2; r++) {
            assertEquals(Q8_0x4VecDot.INSTANCE.dotRow(q8, r, x8), simdQ8.dotRow(q8, r, x8), 0f);
        }
        float[] scalarRows = new float[2];
        float[] vectorRows = new float[2];
        Q8_0x4VecDot.INSTANCE.dotRows(q8, 0, 2, x8, scalarRows, 0);
        simdQ8.dotRows(q8, 0, 2, x8, vectorRows, 0);
        assertArrayEquals("Q8_0x4 dotRows", scalarRows, vectorRows, 0f);

        int q4kCols = 2048;
        CpuTensorRef q4k = repackedQ4K(CpuTensorRef.of(GGMLType.Q4_K, randomQ4KRows(2, q4kCols, rnd), 2,
                q4kCols));
        QuantizedActivations xk = CpuActivationQuantizer.quantizeQ8_K(gaussian(q4kCols, rnd), 0, q4kCols);
        CpuVecDot simdQ4K = CpuSimd.repacked(GGMLType.Q4_K);
        assertNotNull(simdQ4K);
        for (int r = 0; r < 2; r++) {
            assertEquals(Q4_Kx8VecDot.INSTANCE.dotRow(q4k, r, xk), simdQ4K.dotRow(q4k, r, xk), 0f);
        }
        Q4_Kx8VecDot.INSTANCE.dotRows(q4k, 0, 2, xk, scalarRows, 0);
        simdQ4K.dotRows(q4k, 0, 2, xk, vectorRows, 0);
        assertArrayEquals("Q4_Kx8 dotRows", scalarRows, vectorRows, 0f);
    }

    @Test
    public void theBatchedRepackedKernelsMatchOnePassPerToken() {
        Assume.assumeTrue(CpuSimd.available());
        Random rnd = new Random(0x854);
        CpuRepack repack = new CpuRepack(CpuRepack.Mode.ON, Long.MAX_VALUE);
        assertBatches(repack, GGMLType.Q4_0, q4Rows(3, 512, rnd), 512, rnd);
        assertBatches(repack, GGMLType.Q8_0, randomQ8Rows(3, 512, rnd), 512, rnd);
        assertBatches(repack, GGMLType.Q4_K, randomQ4KRows(3, 2048, rnd), 2048, rnd);
    }

    /**
     * The batched product of the repacked kernel of {@code type} equals its per-token product bit for bit, for
     * every token count up to the kernel's tile: the batched path decodes a weight block once per tile, and the
     * integer sums and float combinations must still follow the one-token kernel exactly.
     */
    private static void assertBatches(CpuRepack repack, GGMLType type, ByteBuffer data, int cols, Random rnd) {
        final int rows = 3;
        CpuMatVec.MatvecPlan plan = repack.planFor(CpuTensorRef.of(type, data, rows, cols));
        assertNotNull(type + " must have a repacked plan", plan);
        CpuTensorRef w = plan.weight;
        CpuVecDot kernel = plan.kernel;
        for (int nTokens = 2; nTokens <= 5; nTokens++) {
            QuantizedActivations[] acts = new QuantizedActivations[nTokens];
            float[][] expected = new float[nTokens][];
            for (int t = 0; t < nTokens; t++) {
                acts[t] = CpuActivationQuantizer.quantizeFor(type, gaussian(cols, rnd), 0, cols);
                expected[t] = new float[rows];
                kernel.dotRows(w, 0, rows, acts[t], expected[t], 0);
            }
            float[] y = new float[nTokens * rows];
            kernel.dotRows(w, 0, rows, acts, nTokens, y, 0, rows);
            for (int t = 0; t < nTokens; t++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(type + " batch of " + nTokens + ", token " + t + ", row " + r,
                            expected[t][r], y[t * rows + r], 0f);
                }
            }
        }
    }

    @Test
    public void plansCoverTheRepackedLayoutsAndCacheThem() {
        Assume.assumeTrue(CpuSimd.available());
        CpuRepack repack = new CpuRepack(CpuRepack.Mode.ON);
        Random rnd = new Random(0x853);
        CpuTensorRef q4 = CpuTensorRef.of(GGMLType.Q4_0,
                q4Rows(2, 256, rnd), 2, 256);
        CpuTensorRef q8 = CpuTensorRef.of(GGMLType.Q8_0,
                randomQ8Rows(2, 256, rnd), 2, 256);
        CpuTensorRef q4k = CpuTensorRef.of(GGMLType.Q4_K,
                randomQ4KRows(2, 2048, rnd), 2, 2048);
        for (CpuTensorRef w : new CpuTensorRef[]{q4, q8, q4k}) {
            CpuMatVec.MatvecPlan plan = repack.planFor(w);
            assertNotNull(w.type() + " is eligible", plan);
            assertEquals(w.type(), plan.weight.type());
            assertEquals(w.rows(), plan.weight.rows());
            assertEquals(w.cols(), plan.weight.cols());
            assertEquals(w.rowBytes(), plan.weight.rowBytes());
            assertSame(w.type() + " plan is cached", plan, repack.planFor(w));
            assertFalse(w.type() + " copy differs from the original",
                    java.util.Arrays.equals(rowOf(w, 0), rowOf(plan.weight, 0)));
        }
        assertEquals(3, repack.repackedCount());
        assertNull("Q8_0 columns must split into 128-element groups",
                repack.planFor(CpuTensorRef.of(GGMLType.Q8_0, randomQ8Rows(1, 64, rnd), 1, 64)));
        assertNull("Q4_K columns must split into 2048-element groups",
                repack.planFor(CpuTensorRef.of(GGMLType.Q4_K, randomQ4KRows(1, 512, rnd), 1, 512)));
        assertNull("F16 has no repacked layout",
                repack.planFor(CpuTensorRef.of(GGMLType.F16, ByteBuffer.allocate(64), 1, 32)));
        assertNull("null gets no plan", repack.planFor(null));
    }

    private static ByteBuffer q4Rows(int rows, int cols, Random rnd) {
        BlockLayout layout = BlockLayout.of(GGMLType.Q4_0);
        int blocks = rows * (cols / layout.blockElements());
        byte[] raw = new byte[blocks * layout.blockBytes()];
        rnd.nextBytes(raw);
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < blocks; i++) {
            b.putShort(i * layout.blockBytes() + BlockLayout.Q4_0_D,
                    (short) FpBlockDecoders.fp32ToFp16((rnd.nextFloat() * 2f - 1f) * 0.1f));
        }
        return b;
    }

    private static byte[] rowOf(CpuTensorRef t, int r) {
        byte[] row = new byte[(int) t.rowBytes()];
        ByteBuffer d = t.rowBuffer(r).duplicate();
        d.position((int) t.rowOffset(r));
        d.get(row, 0, (int) t.rowBytes());
        return row;
    }

    @Test
    public void aKernelSourcePlanRunsRepackedWeightsThroughTheMatvec() {
        Assume.assumeTrue(CpuSimd.available());
        Random rnd = new Random(0x854);
        CpuRepack repack = new CpuRepack(CpuRepack.Mode.ON);
        for (CpuTensorRef w : new CpuTensorRef[]{
                CpuTensorRef.of(GGMLType.Q8_0, randomQ8Rows(3, 256, rnd), 3, 256),
                CpuTensorRef.of(GGMLType.Q4_K, randomQ4KRows(3, 2048, rnd), 3, 2048)}) {
            float[] x = gaussian((int) w.cols(), rnd);
            float[] plain = new float[(int) w.rows()];
            new CpuMatVec(weight -> CpuMatVec.Transforms.NONE).matvec(w, x, 0, plain, 0);
            CpuMatVec withPlan = new CpuMatVec(weight -> CpuMatVec.Transforms.NONE, repack::planFor);
            float[] viaPlan = new float[(int) w.rows()];
            withPlan.matvec(w, x, 0, viaPlan, 0);
            assertArrayEquals(w.type() + " product must equal the plain product exactly", plain, viaPlan, 0f);
        }
        assertEquals(2, repack.repackedCount());
    }

    @Test
    public void theOffModeKeepsEveryWeightInPlaceAndSaysSo() {
        Assume.assumeTrue(CpuSimd.available());
        CpuRepack repack = new CpuRepack(CpuRepack.Mode.OFF);
        CpuTensorRef w = CpuTensorRef.of(GGMLType.Q8_0, randomQ8Rows(2, 256, new Random(0x855)), 2, 256);
        assertNull(repack.planFor(w));
        assertEquals(0, repack.repackedCount());
        assertTrue(repack.summary(), repack.summary().contains("repacking off 1"));
    }

    @Test
    public void qualifyingWeightsAreRepackedByTheirFirstProductNotTheirLoad() throws IOException {
        Assume.assumeTrue(CpuSimd.available());
        GgufTestWriter model = new GgufTestWriter()
                .architecture("llama")
                .tensorZeros("blk.0.ffn_up.weight", GGMLType.Q4_0, 256, 2)
                .tensorZeros("blk.0.ffn_down.weight", GGMLType.Q8_0, 128, 2)
                .tensorZeros("blk.0.ffn_gate.weight", GGMLType.Q4_K, 2048, 2)
                .tensorZeros("output_norm.weight", GGMLType.F32, 64);
        Path path = model.writeTempFile();
        ModelSource source = null;
        String previous = System.getProperty(CpuRepack.PROP_REPACK);
        try {
            System.setProperty(CpuRepack.PROP_REPACK, "on");
            source = ModelSources.open(path.toString(), ModelSourceOptions.defaults());
            TensorCatalog catalog = TensorCatalog.of(source);
            try (CpuWeights weights = CpuWeights.open(source, catalog, null, TensorReadMode.MMAP)) {
                assertEquals("nothing is repacked before a tensor is loaded", 0,
                        weights.repack().repackedCount());
                CpuTensorRef up = weights.get("blk.0.ffn_up.weight");
                CpuTensorRef down = weights.get("blk.0.ffn_down.weight");
                CpuTensorRef gate = weights.get("blk.0.ffn_gate.weight");
                weights.get("output_norm.weight");
                assertEquals("loading alone repacks nothing", 0, weights.repack().repackedCount());
                for (CpuTensorRef w : new CpuTensorRef[]{up, down, gate}) {
                    float[] x = new float[(int) w.cols()];
                    float[] y = new float[(int) w.rows()];
                    weights.matVec().matvec(w, x, 0, y, 0);
                }
                assertEquals("each matmul weight is repacked by its first product", 3,
                        weights.repack().repackedCount());
            }
        } finally {
            if (previous == null) {
                System.clearProperty(CpuRepack.PROP_REPACK);
            } else {
                System.setProperty(CpuRepack.PROP_REPACK, previous);
            }
            try {
                if (source != null) {
                    source.close();
                }
            } finally {
                Files.deleteIfExists(path);
            }
        }
    }
}
