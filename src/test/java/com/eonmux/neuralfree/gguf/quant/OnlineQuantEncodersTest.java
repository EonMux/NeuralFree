package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.F8B128_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.F8B32_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.I8B128_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP8_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.NVFP4_BYTES;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link OnlineQuantEncoders}: the vLLM online weight quantizers against the reference scale rules (vLLM
 * online/fp8.py, int8.py, mxfp8_utils.py and nvfp4.py) — the stored scale values, the codes, the round-half-to-even
 * and clamp behavior, and the dequantization of the written blocks through the CPU decoders.
 */
public class OnlineQuantEncodersTest {

    /** {@code 1 / (448 * 512)} rounded to float (vLLM's {@code min_scaling_factor}). */
    private static final float FP8_MIN_SCALE = Float.intBitsToFloat(0x36924925);

    /** The E4M3 value of the code {@code f} encodes: the reference quantizer's element mapping. */
    private static float e4m3(float f) {
        return FpBlockDecoders.e4m3ToFp32(FpBlockDecoders.fp32ToE4m3(f));
    }

    private static float e2m1(float f) {
        return FpBlockDecoders.e2m1ToFp32(FpBlockDecoders.fp32ToE2m1(f));
    }

    /** Decodes one quantized row of {@code type} through the CPU decoders. */
    private static float[] decode(GGMLType type, byte[] row) {
        ByteBuffer b = ByteBuffer.wrap(row).order(ByteOrder.LITTLE_ENDIAN);
        float[] y = new float[type.getBlockSize() * (row.length / type.getTypeSize())];
        switch (type) {
            case F8_E4M3_B128:
                ExtBlockDecoders.decodeF8E4m3B128(b, 0, y, 0, row.length / F8B128_BYTES);
                break;
            case F8_E4M3_B32:
                ExtBlockDecoders.decodeF8E4m3B32(b, 0, y, 0, row.length / F8B32_BYTES);
                break;
            case I8_B128:
                ExtBlockDecoders.decodeI8B128(b, 0, y, 0, row.length / I8B128_BYTES);
                break;
            case MXFP8:
                ExtBlockDecoders.decodeMxfp8(b, 0, y, 0, row.length / MXFP8_BYTES);
                break;
            case NVFP4:
                FpBlockDecoders.decodeNvfp4(b, 0, y, 0, row.length / NVFP4_BYTES);
                break;
            default:
                throw new IllegalArgumentException(String.valueOf(type));
        }
        return y;
    }

    private static byte[] quantize(ScanningQuantizer prototype, long rows, int inner, long experts, float[][] tensor) {
        ScanningQuantizer s = prototype.open(rows, inner, experts);
        for (int r = 0; r < tensor.length; r++) {
            s.scanRow(r, tensor[r], 0, inner);
        }
        s.freeze();
        byte[] out = new byte[tensor.length * (int) prototype.type().getBytesForElements(inner)];
        for (int r = 0; r < tensor.length; r++) {
            s.quantizeRow(r, tensor[r], 0, inner, out, r * (out.length / tensor.length));
        }
        return out;
    }

    /** One row of {@code n} floats around a spread of magnitudes. */
    private static float[] row(int n, float scale, long seed) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            float v = (float) (Math.sin(seed + i * 0.731) * Math.pow(2, (i % 5) - 2));
            x[i] = v * scale;
        }
        return x;
    }

    // ------------------------------------------------------------------ fp8 per tensor

    @Test
    public void fp8PerTensorStoresTheTensorScaleAndTheReciprocalCodes() {
        int inner = 128;
        float[] x = row(inner, 3.5f, 1);
        float amax = 0.0f;
        for (float v : x) {
            amax = Math.max(amax, Math.abs(v));
        }
        float scale = amax / 448.0f;
        byte[] out = quantize(OnlineQuantEncoders.fp8PerTensor(), 1, inner, 1, new float[][]{x});
        ByteBuffer d = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals("the block scale is amax/448", scale, d.getFloat(0), 0.0f);
        float[] y = decode(GGMLType.F8_E4M3_B128, out);
        for (int i = 0; i < inner; i++) {
            assertEquals("element " + i, e4m3(x[i] * (1.0f / scale)) * scale, y[i], 0.0f);
            assertTrue("dequantized close to x",
                    Math.abs(y[i] - x[i]) <= 0.0625f * Math.abs(x[i]) + scale * 0x1p-9f);
        }
    }

    @Test
    public void fp8PerTensorScalesEachExpertByItsOwnAmax() {
        int inner = 128;
        float[] small = row(inner, 1.0f, 3);
        float[] large = row(inner, 100.0f, 4);
        byte[] out = quantize(OnlineQuantEncoders.fp8PerTensor(), 4, inner, 2,
                new float[][]{small, small, large, large});
        ByteBuffer b = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
        float d0 = b.getFloat(0);
        float d1 = b.getFloat(2 * F8B128_BYTES);
        assertNotEquals(d0, d1, 1e-30f);
        for (int r = 0; r < 4; r++) {
            assertEquals("row " + r + " keeps its expert's scale", r < 2 ? d0 : d1,
                    b.getFloat(r * F8B128_BYTES), 0.0f);
        }
    }

    @Test
    public void fp8PerTensorOfAnAllZeroTensorKeepsZeroCodes() {
        byte[] out = quantize(OnlineQuantEncoders.fp8PerTensor(), 1, 128, 1, new float[][]{new float[128]});
        assertArrayEquals(new byte[out.length], out);
    }

    // ------------------------------------------------------------------ fp8 per block

    @Test
    public void fp8PerBlockScalesEveryRowOfATileWithTheTileScale() {
        int inner = 256;
        int rows = 200;
        float[][] tensor = new float[rows][];
        for (int r = 0; r < rows; r++) {
            tensor[r] = row(inner, 1.0f + (r % 3), r + 11);
        }
        ScanningQuantizer s = OnlineQuantEncoders.fp8PerBlock().open(rows, inner, 1);
        for (int r = 0; r < rows; r++) {
            s.scanRow(r, tensor[r], 0, inner);
        }
        s.freeze();
        // The tile scales, by vLLM's rule: the clamped amax times the reciprocal of the FP8 maximum.
        float[][] want = new float[2][2];
        for (int rt = 0; rt < 2; rt++) {
            for (int ct = 0; ct < 2; ct++) {
                float amax = 0.0f;
                for (int r = rt * 128; r < Math.min(rows, rt * 128 + 128); r++) {
                    for (int c = ct * 128; c < ct * 128 + 128; c++) {
                        amax = Math.max(amax, Math.abs(tensor[r][c]));
                    }
                }
                want[rt][ct] = Math.max(amax, 1e-4f) * (1.0f / 448.0f);
            }
        }
        byte[] out = new byte[rows * 2 * F8B128_BYTES];
        for (int r = 0; r < rows; r++) {
            s.quantizeRow(r, tensor[r], 0, inner, out, r * 2 * F8B128_BYTES);
        }
        ByteBuffer b = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
        for (int r = 0; r < rows; r++) {
            int rt = r / 128;
            assertEquals("row " + r + " column tile 0", want[rt][0], b.getFloat(r * 2 * F8B128_BYTES), 0.0f);
            assertEquals("row " + r + " column tile 1", want[rt][1],
                    b.getFloat(r * 2 * F8B128_BYTES + F8B128_BYTES), 0.0f);
        }
        // The codes quantize through the reciprocal, and dequantize close.
        float[] y = decode(GGMLType.F8_E4M3_B128, out);
        for (int r = 0; r < rows; r++) {
            int rt = r / 128;
            for (int c = 0; c < inner; c++) {
                float scale = want[rt][c / 128];
                assertEquals("row " + r + " element " + c, e4m3(tensor[r][c] * (1.0f / scale)) * scale,
                        y[r * inner + c], 0.0f);
            }
        }
    }

    // ------------------------------------------------------------------ fp8 rows of no whole 128-value blocks

    @Test
    public void fp8PerBlockStoresRowsOfNoWhole128ValueBlocksAs32ValueRunsOfTheirTileScales() {
        int inner = 192;
        int rows = 130;
        float[][] tensor = new float[rows][];
        for (int r = 0; r < rows; r++) {
            tensor[r] = row(inner, 1.0f + (r % 7), r + 5);
        }
        assertEquals(GGMLType.F8_E4M3_B32, OnlineQuantEncoders.fp8PerBlock().typeFor(inner));
        assertEquals(GGMLType.F8_E4M3_B128, OnlineQuantEncoders.fp8PerBlock().typeFor(256));
        ScanningQuantizer s = OnlineQuantEncoders.fp8PerBlock().open(rows, inner, 1);
        for (int r = 0; r < rows; r++) {
            s.scanRow(r, tensor[r], 0, inner);
        }
        s.freeze();
        float[][] want = tileScales(tensor, inner);
        int rowBytes = (int) GGMLType.F8_E4M3_B32.getBytesForElements(inner);
        assertEquals(6 * F8B32_BYTES, rowBytes);
        for (int r = 0; r < rows; r++) {
            byte[] out = new byte[rowBytes];
            s.quantizeRow(r, tensor[r], 0, inner, out, 0);
            ByteBuffer b = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
            float[] y = decode(GGMLType.F8_E4M3_B32, out);
            for (int c = 0; c < inner; c++) {
                float scale = want[r / 128][c / 128];
                assertEquals("row " + r + " run " + c / 32, scale, b.getFloat(c / 32 * F8B32_BYTES), 0.0f);
                assertEquals("row " + r + " element " + c, e4m3(tensor[r][c] * (1.0f / scale)) * scale, y[c], 0.0f);
            }
        }
    }

    @Test
    public void fp8PerTensorAndPerChannelStoreRowsOfNoWhole128ValueBlocksAs32ValueRuns() {
        float[] x = row(160, 2.5f, 9);
        float amax = amaxOf(x);
        byte[] perTensor = quantizeRows(OnlineQuantEncoders.fp8PerTensor(), x);
        float[] y = decode(GGMLType.F8_E4M3_B32, perTensor);
        for (int c = 0; c < x.length; c++) {
            float scale = amax / 448.0f;
            assertEquals("run " + c / 32, scale,
                    ByteBuffer.wrap(perTensor).order(ByteOrder.LITTLE_ENDIAN).getFloat(c / 32 * F8B32_BYTES), 0.0f);
            assertEquals("element " + c, e4m3(x[c] * (1.0f / scale)) * scale, y[c], 0.0f);
        }

        float[] z = row(96, 0.75f, 13);
        OnlineQuantizer perChannel = OnlineQuantEncoders.fp8PerChannel();
        assertEquals(GGMLType.F8_E4M3_B32, perChannel.typeFor(96));
        byte[] out = new byte[3 * F8B32_BYTES];
        perChannel.quantizeRow(z, 0, z.length, out, 0);
        float scale = amaxOf(z) / 448.0f;
        y = decode(GGMLType.F8_E4M3_B32, out);
        for (int c = 0; c < z.length; c++) {
            assertEquals("element " + c, e4m3(z[c] / scale) * scale, y[c], 0.0f);
        }
    }

    @Test
    public void fp8ColumnTypesFollowTheRunsAndTheirAlignment() {
        ScanningQuantizer block = OnlineQuantEncoders.fp8PerBlock();
        assertEquals(GGMLType.F8_E4M3_B128, block.columnTypeFor(128, 256));
        assertEquals(GGMLType.F8_E4M3_B128, block.columnTypeFor(256, 0));
        assertEquals("a run starting inside a tile stores 32-value runs", GGMLType.F8_E4M3_B32,
                block.columnTypeFor(128, 192));
        assertEquals(GGMLType.F8_E4M3_B32, block.columnTypeFor(64, 128));
        assertNull(block.columnTypeFor(48, 0));
        assertNull("a 32-value run would straddle two tiles", block.columnTypeFor(64, 16));
        ScanningQuantizer tensor = OnlineQuantEncoders.fp8PerTensor();
        assertEquals(GGMLType.F8_E4M3_B128, tensor.columnTypeFor(128, 5));
        assertEquals(GGMLType.F8_E4M3_B32, tensor.columnTypeFor(64, 5));
        assertNull(tensor.columnTypeFor(40, 0));
    }

    /**
     * A tensor fed partly as rows and partly as the column runs of a transposed member scales by the same tiles
     * as the tensor fed by rows, and each column run's blocks carry the tile scales of the rows they cover.
     */
    @Test
    public void fp8PerBlockColumnRunsShareTheTilesOfTheRowsTheyCover() {
        int inner = 64;
        int rows = 256;
        float[][] tensor = new float[rows][];
        for (int r = 0; r < rows; r++) {
            tensor[r] = row(inner, r < 128 ? 0.5f : 3.0f, 31 + r);
        }
        ScanningQuantizer s = OnlineQuantEncoders.fp8PerBlock().open(rows, inner, 1);
        for (int h = 0; h < 2; h++) {
            for (int c = 0; c < inner; c++) {
                s.scanColumn(h * 128, c, column(tensor, h * 128, c, 64), 0, 64);
            }
            for (int r = h * 128 + 64; r < h * 128 + 128; r++) {
                s.scanRow(r, tensor[r], 0, inner);
            }
        }
        s.freeze();
        float[][] want = tileScales(tensor, inner);
        for (int h = 0; h < 2; h++) {
            for (int c = 0; c < inner; c++) {
                float[] x = column(tensor, h * 128, c, 64);
                byte[] out = new byte[2 * F8B32_BYTES];
                s.quantizeColumn(h * 128, c, x, 0, 64, GGMLType.F8_E4M3_B32, out, 0);
                float scale = want[h][0];
                float[] y = decode(GGMLType.F8_E4M3_B32, out);
                for (int i = 0; i < 64; i++) {
                    assertEquals("head " + h + " column " + c + " element " + i,
                            e4m3(x[i] * (1.0f / scale)) * scale, y[i], 0.0f);
                }
            }
        }
        float[] crossing = column(tensor, 64, 0, 128);
        assertThrows("a 128-value block crossing a row tile", IllegalArgumentException.class,
                () -> s.quantizeColumn(64, 0, crossing, 0, 128, GGMLType.F8_E4M3_B128, new byte[F8B128_BYTES], 0));
        assertThrows("not an FP8 block type", IllegalArgumentException.class,
                () -> s.quantizeColumn(0, 0, crossing, 0, 64, GGMLType.MXFP8, new byte[256], 0));
        assertThrows("a column outside the rows", IllegalArgumentException.class,
                () -> s.scanColumn(0, inner, crossing, 0, 32));
    }

    /** vLLM's 128x128 tile scales of a 2-D {@code tensor}: the clamped amax times the reciprocal of 448. */
    private static float[][] tileScales(float[][] tensor, int inner) {
        int rowTiles = (tensor.length + 127) / 128;
        int colTiles = (inner + 127) / 128;
        float[][] want = new float[rowTiles][colTiles];
        for (int r = 0; r < tensor.length; r++) {
            for (int c = 0; c < inner; c++) {
                want[r / 128][c / 128] = Math.max(want[r / 128][c / 128], Math.abs(tensor[r][c]));
            }
        }
        for (float[] w : want) {
            for (int ct = 0; ct < colTiles; ct++) {
                w[ct] = Math.max(w[ct], 1e-4f) * (1.0f / 448.0f);
            }
        }
        return want;
    }

    private static float[] column(float[][] tensor, int row, int col, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = tensor[row + i][col];
        }
        return x;
    }

    /** One row {@code x} through a scanning quantizer, in the quantizer's row type. */
    private static byte[] quantizeRows(ScanningQuantizer prototype, float[] x) {
        ScanningQuantizer s = prototype.open(1, x.length, 1);
        s.scanRow(0, x, 0, x.length);
        s.freeze();
        byte[] out = new byte[(int) prototype.typeFor(x.length).getBytesForElements(x.length)];
        s.quantizeRow(0, x, 0, x.length, out, 0);
        return out;
    }

    // ------------------------------------------------------------------ fp8 per channel

    @Test
    public void fp8PerChannelDividesByTheRowScaleAndFloorsTinyScales() {
        int inner = 128;
        float[] x = row(inner, 2.0f, 7);
        byte[] out = new byte[F8B128_BYTES];
        OnlineQuantEncoders.fp8PerChannel().quantizeRow(x, 0, inner, out, 0);
        float amax = 0.0f;
        for (float v : x) {
            amax = Math.max(amax, Math.abs(v));
        }
        ByteBuffer b = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(amax / 448.0f, b.getFloat(0), 0.0f);
        float[] y = decode(GGMLType.F8_E4M3_B128, out);
        for (int i = 0; i < inner; i++) {
            assertEquals(e4m3(x[i] / (amax / 448.0f)) * (amax / 448.0f), y[i], 0.0f);
        }
        float[] tiny = new float[inner];
        tiny[64] = 1e-12f;
        byte[] floor = new byte[F8B128_BYTES];
        OnlineQuantEncoders.fp8PerChannel().quantizeRow(tiny, 0, inner, floor, 0);
        assertEquals(FP8_MIN_SCALE,
                ByteBuffer.wrap(floor).order(ByteOrder.LITTLE_ENDIAN).getFloat(0), 0.0f);
    }

    // ------------------------------------------------------------------ int8 per channel

    @Test
    public void int8PerChannelRoundsHalfToEvenClampsToOneHundredTwentySeven() {
        int inner = 128;
        float[] x = new float[inner];
        x[0] = 0.5f;
        x[1] = 1.5f;
        x[2] = 2.5f;
        x[3] = -0.5f;
        x[4] = -1.5f;
        x[64] = 127.0f;
        x[65] = -127.0f;
        byte[] out = new byte[I8B128_BYTES];
        OnlineQuantEncoders.int8PerChannel().quantizeRow(x, 0, inner, out, 0);
        ByteBuffer b = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals("the row scale is amax/127", 1.0f, b.getFloat(0), 0.0f);
        assertEquals("0.5 rounds to even 0", 0, b.get(BlockLayout.I8B128_QS));
        assertEquals("1.5 rounds to even 2", 2, b.get(BlockLayout.I8B128_QS + 1));
        assertEquals("2.5 rounds to even 2", 2, b.get(BlockLayout.I8B128_QS + 2));
        assertEquals("-0.5 rounds to even 0", 0, b.get(BlockLayout.I8B128_QS + 3));
        assertEquals("-1.5 rounds to even -2", -2, b.get(BlockLayout.I8B128_QS + 4));
        assertEquals(127, b.get(BlockLayout.I8B128_QS + 64));
        assertEquals(-127, b.get(BlockLayout.I8B128_QS + 65));
        float[] y = decode(GGMLType.I8_B128, out);
        for (int i = 0; i < inner; i++) {
            int q = (int) Math.max(-127, Math.min(127, Math.rint(x[i])));
            assertEquals("element " + i, q, y[i], 0.0f);
        }
    }

    @Test
    public void int8PerChannelWritesEveryBlockOfALongRowWithTheRowScale() {
        int inner = 3 * 128;
        float[] x = row(inner, 1.5f, 5);
        x[300] = -9.0f;
        byte[] out = new byte[3 * I8B128_BYTES];
        OnlineQuantEncoders.int8PerChannel().quantizeRow(x, 0, inner, out, 0);
        ByteBuffer b = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
        float d = 9.0f / 127.0f;
        for (int blk = 0; blk < 3; blk++) {
            assertEquals("block " + blk + " carries the row scale", d, b.getFloat(blk * I8B128_BYTES), 0.0f);
        }
        float[] y = decode(GGMLType.I8_B128, out);
        for (int i = 0; i < inner; i++) {
            float q = (float) Math.max(-127, Math.min(127, Math.rint(x[i] / d)));
            assertEquals("element " + i, q * d, y[i], 0.0f);
        }
    }

    @Test
    public void rowQuantizersRefuseARowThatIsNotWholeBlocks() {
        float[] x = new float[200];
        byte[] out = new byte[1024];
        assertThrows(IllegalArgumentException.class,
                () -> OnlineQuantEncoders.int8PerChannel().quantizeRow(x, 0, 200, out, 0));
        assertThrows(IllegalArgumentException.class,
                () -> OnlineQuantEncoders.fp8PerChannel().quantizeRow(x, 0, 200, out, 0));
        assertThrows(IllegalArgumentException.class,
                () -> OnlineQuantEncoders.mxfp8().quantizeRow(x, 0, 40, out, 0));
        assertThrows(IllegalArgumentException.class,
                () -> OnlineQuantEncoders.nvfp4PerToken().open(2, 48, 1));
        assertThrows(IllegalArgumentException.class,
                () -> OnlineQuantEncoders.fp8PerBlock().open(3, 128, 2));
    }

    @Test
    public void fp8PerChannelKeepsEveryBlockOfALongRowOnTheRowScale() {
        int inner = 256;
        float[] x = row(inner, 0.5f, 8);
        x[200] = 44.8f;
        byte[] out = new byte[2 * F8B128_BYTES];
        OnlineQuantEncoders.fp8PerChannel().quantizeRow(x, 0, inner, out, 0);
        ByteBuffer b = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(44.8f / 448.0f, b.getFloat(0), 0.0f);
        assertEquals(44.8f / 448.0f, b.getFloat(F8B128_BYTES), 0.0f);
    }

    @Test
    public void int8PerChannelOfAnAllZeroRowKeepsZeroCodes() {
        byte[] out = new byte[I8B128_BYTES];
        OnlineQuantEncoders.int8PerChannel().quantizeRow(new float[128], 0, 128, out, 0);
        assertArrayEquals(new byte[out.length], out);
    }

    // ------------------------------------------------------------------ mxfp8

    @Test
    public void mxfp8ScalesEveryRunOf32ByAPowerOfTwo() {
        int inner = 64;
        float[] x = row(inner, 3.0f, 9);
        x[32] = 0.0f;
        byte[] out = new byte[2 * MXFP8_BYTES];
        OnlineQuantEncoders.mxfp8().quantizeRow(x, 0, inner, out, 0);
        float amax0 = 0.0f;
        float amax1 = 0.0f;
        for (int i = 0; i < 32; i++) {
            amax0 = Math.max(amax0, Math.abs(x[i]));
            amax1 = Math.max(amax1, Math.abs(x[32 + i]));
        }
        int e0 = ActFakeQuant.mxfp8Exponent(amax0);
        int e1 = ActFakeQuant.mxfp8Exponent(amax1);
        assertEquals("run 0 E8M0 byte", e0 + 127, out[BlockLayout.MXFP8_E] & 0xFF);
        assertEquals("run 1 E8M0 byte", e1 + 127, out[MXFP8_BYTES + BlockLayout.MXFP8_E] & 0xFF);
        float[] y = decode(GGMLType.MXFP8, out);
        for (int i = 0; i < inner; i++) {
            int e = i < 32 ? e0 : e1;
            assertEquals("element " + i, e4m3(Math.scalb(x[i], -e)) * Math.scalb(1.0f, e), y[i], 0.0f);
        }
    }

    // ------------------------------------------------------------------ nvfp4 per token

    @Test
    public void nvfp4PerTokenCarriesThePerExpertGlobalScaleAndRestoresTheValues() {
        int inner = 64;
        float[] small = row(inner, 1.0f, 13);
        float[] large = row(inner, 4.0f, 17);
        float[][] tensor = {small, small, large, large};
        ScanningQuantizer s = OnlineQuantEncoders.nvfp4PerToken().open(4, inner, 2);
        for (int r = 0; r < 4; r++) {
            s.scanRow(r, tensor[r], 0, inner);
        }
        s.freeze();
        // vLLM's (6 * 448) / amax is PyTorch's scalar-over-tensor: the rounded reciprocal of amax, times 2688.
        float gs0 = (1.0f / amaxOf(small)) * (6.0f * 448.0f);
        float gs1 = (1.0f / amaxOf(large)) * (6.0f * 448.0f);
        float ws0 = 1.0f / gs0;
        float ws1 = 1.0f / gs1;
        assertArrayEquals("the output multiplier is weight_scale_2 per expert",
                new float[]{ws0, ws1}, s.outputScale(), 0.0f);
        byte[] out = new byte[4 * NVFP4_BYTES];
        for (int r = 0; r < 4; r++) {
            s.quantizeRow(r, tensor[r], 0, inner, out, r * NVFP4_BYTES);
        }
        float[] y = decode(GGMLType.NVFP4, out);
        for (int r = 0; r < 4; r++) {
            float gs = r < 2 ? gs0 : gs1;
            float ws = r < 2 ? ws0 : ws1;
            float[] x = r < 2 ? small : large;
            for (int i = 0; i < inner; i++) {
                // ref_nvfp4_quant: scale = e4m3(min(gs * (vecMax * (1/6)), 448)); codes = e2m1(x * 1/(scale/gs))
                float s16 = e4m3(Math.min(gs * (amaxOfRun(x, i) * (1.0f / 6.0f)), 448.0f));
                float dequant = y[r * inner + i] * ws;
                float want = e2m1(x[i] * (1.0f / (s16 * ws))) * s16 * ws;
                assertEquals("row " + r + " element " + i, want, dequant, 0.0f);
                assertTrue("within one E2M1 step of the run scale",
                        Math.abs(dequant - x[i]) <= 0.18f * amaxOfRun(x, i) + 1e-30f);
            }
        }
    }

    /** {@code max|x|}. */
    private static float amaxOf(float[] x) {
        float m = 0.0f;
        for (float v : x) {
            m = Math.max(m, Math.abs(v));
        }
        return m;
    }

    private static float amaxOfRun(float[] x, int i) {
        int run = i / 16 * 16;
        return amaxOf(Arrays.copyOfRange(x, run, run + 16));
    }

    @Test
    public void nvfp4PerTokenOfAZeroRunInsideANonzeroExpertKeepsZeroCodes() {
        int inner = 64;
        float[] x = row(inner, 2.0f, 21);
        for (int i = 16; i < 32; i++) {
            x[i] = 0.0f;
        }
        byte[] out = quantize(OnlineQuantEncoders.nvfp4PerToken(), 1, inner, 1, new float[][]{x});
        // Run 1: scale byte 0 and zero codes; run 0 quantizes.
        assertEquals(0, out[1]);
        for (int j = 0; j < 8; j++) {
            assertEquals("zero run codes", 0, out[BlockLayout.NVFP4_QS + 8 + j] & 0xFF);
        }
        assertNotEquals(0, out[0]);
    }

    @Test
    public void mxfp4ExponentRoundsTheAmaxFromOnePointSevenFiveTimesAPowerOfTwoUp() {
        assertEquals("6 = 1.5 * 4 rounds down to 4", 0, OnlineQuantEncoders.mxfp4Exponent(6.0f));
        assertEquals("7 = 1.75 * 4 rounds up to 8", 1, OnlineQuantEncoders.mxfp4Exponent(7.0f));
        assertEquals(-1, OnlineQuantEncoders.mxfp4Exponent(1.75f));
        assertEquals(-2, OnlineQuantEncoders.mxfp4Exponent(Math.nextDown(1.75f)));
        assertEquals("an all-zero run takes log2(2^-126) - 2, clamped", -127, OnlineQuantEncoders.mxfp4Exponent(0.0f));
        assertEquals(-127, OnlineQuantEncoders.mxfp4Exponent(0x1p-126f));
        assertEquals(127, OnlineQuantEncoders.mxfp4Exponent(Float.MAX_VALUE));
        assertEquals(127, OnlineQuantEncoders.mxfp4Exponent(Float.POSITIVE_INFINITY));
    }

    @Test
    public void mxfp4CodeRoundsHalfToEvenWithAStickyBitAndSaturates() {
        // Below 1 the kernel drops the mantissa's lowest bit before the subnormal shift, so the float just above
        // the 0.25 tie still rounds to zero; the next mantissa bit reaches the sticky bit and rounds up.
        float[] values = {0.0f, 0.25f, Math.nextUp(0.25f), 0.25f + 0x1p-22f, 0.3f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f,
            1.75f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 5.0f, Math.nextUp(5.0f), 6.0f, 7.0f, 100.0f, 1e-30f, Float.MIN_VALUE};
        int[] codes = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 7, 7, 7, 0, 0};
        for (int i = 0; i < values.length; i++) {
            assertEquals(String.valueOf(values[i]), codes[i], OnlineQuantEncoders.mxfp4Code(values[i]));
            assertEquals("-" + values[i], codes[i] | 0x8, OnlineQuantEncoders.mxfp4Code(-values[i]));
        }
    }

    @Test
    public void mxfp4StoresTheE8m0ScaleAndSplitNibblesThatDecodeToTheGrid() {
        float[] x = new float[64];
        for (int j = 0; j < 32; j++) {
            x[j] = (j % 2 == 0 ? 1 : -1) * 0.5f * (j % 13);
        }
        x[40] = 0.1f;
        byte[] out = new byte[2 * BlockLayout.MXFP4_BYTES];
        OnlineQuantEncoders.mxfp4().quantizeRow(x, 0, 64, out, 0);
        assertEquals(GGMLType.MXFP4, OnlineQuantEncoders.mxfp4().type());
        // Block 0: amax 6 -> exponent 0 (byte 127), so the codes are the values' own E2M1 codes.
        assertEquals(127, out[BlockLayout.MXFP4_E] & 0xFF);
        for (int j = 0; j < 16; j++) {
            int q = out[BlockLayout.MXFP4_QS + j] & 0xFF;
            assertEquals("element " + j, OnlineQuantEncoders.mxfp4Code(x[j]), q & 0x0F);
            assertEquals("element " + (j + 16), OnlineQuantEncoders.mxfp4Code(x[j + 16]), q >>> 4);
        }
        // Block 1: amax 0.1 = 1.6 * 2^-4 keeps 2^-4 -> exponent -6; 0.1 * 64 = 6.4 saturates at 6.
        int b1 = BlockLayout.MXFP4_BYTES;
        assertEquals(127 - 6, out[b1 + BlockLayout.MXFP4_E] & 0xFF);
        float[] y = new float[64];
        FpBlockDecoders.decodeMxfp4(ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN), 0, y, 0, 2);
        for (int j = 0; j < 32; j++) {
            assertEquals("element " + j, e2m1(x[j]), y[j], 0.0f);
        }
        assertEquals(6.0f / 64, y[40], 0.0f);
        assertEquals(0.0f, y[41], 0.0f);
        assertThrows(IllegalArgumentException.class,
                () -> OnlineQuantEncoders.mxfp4().quantizeRow(new float[48], 0, 48, new byte[34], 0));
    }

    @Test
    public void aScannerIsIndependentOfItsPrototypeAndOfOtherScans() {
        ScanningQuantizer prototype = OnlineQuantEncoders.fp8PerTensor();
        ScanningQuantizer a = prototype.open(1, 128, 1);
        a.scanRow(0, row(128, 2.0f, 1), 0, 128);
        a.freeze();
        ScanningQuantizer b = prototype.open(1, 128, 1);
        b.scanRow(0, row(128, 6.0f, 2), 0, 128);
        b.freeze();
        byte[] outA = new byte[F8B128_BYTES];
        byte[] outB = new byte[F8B128_BYTES];
        a.quantizeRow(0, row(128, 2.0f, 1), 0, 128, outA, 0);
        b.quantizeRow(0, row(128, 6.0f, 2), 0, 128, outB, 0);
        assertNotEquals(ByteBuffer.wrap(outA).getFloat(0), ByteBuffer.wrap(outB).getFloat(0), 0.0f);
    }
}
