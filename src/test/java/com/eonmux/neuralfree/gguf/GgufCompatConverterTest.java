package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The MXFP8 exponent acceptance cases and the conversion proofs of {@link GgufCompatConverter}: an exponent
 * set wholly in {@code [3, 246]} exports as BF16, some exponent in {@code [0, 2]} makes the tensor F32, and
 * an exponent of {@code 247} or the reserved NaN byte fails naming the tensor. F8_E4M3 passes through to
 * llama.cpp's per-element F8_E4M3FN byte for byte, and
 * the Q8_0 requantization of the per-128 blocks matches the reference quantizer's bytes, cross-checked
 * through {@link GgufTestWriter#encodeFloats}. The tensor bytes are served by the same store double the
 * write tests use ({@code GgufWriterTest.TestWeightStore}'s shape, kept local for one-tensor scopes).
 */
public class GgufCompatConverterTest {

    private static final String NAME = "blk.1.attn_k.weight";

    // ================================================================== MXFP8 exponent cases

    @Test
    public void everyExponentInTheFittingRangeExportsAsBf16() throws IOException {
        // [3, 246]: single-block sets at both ends of the fitting range, and a mixed set
        assertEquals(GGMLType.BF16, mxfp8Target(3, 3, 3));
        assertEquals(GGMLType.BF16, mxfp8Target(246, 246, 246));
        assertEquals(GGMLType.BF16, mxfp8Target(3, 200, 246));
    }

    @Test
    public void anExponentBelowThreeNeedsTheSubnormalRange() throws IOException {
        // [0, 2]: one exponent below three makes the tensor F32, among fitting exponents or alone
        assertEquals(GGMLType.F32, mxfp8Target(0, 0, 0));
        assertEquals(GGMLType.F32, mxfp8Target(2, 2, 2));
        assertEquals(GGMLType.F32, mxfp8Target(246, 2, 10));
    }

    @Test
    public void anOverflowingExponentFailsNamingTheTensor() {
        for (int e : new int[]{247, 250, 0xFF}) {
            IOException failure = mxfp8TargetFails(e);
            assertTrue("tensor named in " + failure.getMessage(),
                    failure.getMessage().contains(NAME));
            assertTrue("exponent named in " + failure.getMessage(),
                    failure.getMessage().contains(String.valueOf(e & 0xFF)));
        }
    }

    // ================================================================== conversion proofs

    @Test
    public void theMxfp8RowConvertsToBf16Exactly() throws IOException {
        int[] codes = e4m3Codes(32);
        byte[] raw = mxfp8Bytes(codes, 10);
        byte[] out = convertTo(raw, new long[]{32}, GGMLType.MXFP8, GGMLType.BF16);
        for (int i = 0; i < 32; i++) {
            float source = FpBlockDecoders.e8m0ToFp32(10) * FpBlockDecoders.e4m3ToFp32(codes[i]);
            int bfBits = (out[2 * i] & 0xFF) | (out[2 * i + 1] & 0xFF) << 8;
            assertEquals("element " + i, source, FpBlockDecoders.bf16ToFp32(bfBits), 0f);
        }
    }

    @Test
    public void theF8E4m3CodesPassThroughToF8E4M3FnByteForByte() throws IOException {
        byte[] raw = bytes(e4m3Codes(96));
        GgufWriterTest.TestHfSource src = new GgufWriterTest.TestHfSource("synthetic://f8");
        int[] codes = new int[raw.length];
        for (int i = 0; i < raw.length; i++) {
            codes[i] = raw[i] & 0xFF;
        }
        src.tensor(NAME, GGMLType.F8_E4M3, new long[]{96}, codes);
        assertEquals(GGMLType.F8_E4M3FN, GgufCompatConverter.targetType(src.getTensors().get(0),
                src.createMappedStore(), GgufWriter.Mode.COMPAT));
        assertArrayEquals("codes are llama.cpp's per-element F8_E4M3FN bytes", raw,
                convertTo(raw, new long[]{96}, GGMLType.F8_E4M3, GGMLType.F8_E4M3FN));
        try {
            convertTo(raw, new long[]{96}, GGMLType.F8_E4M3, GGMLType.F16);
            fail("F8_E4M3 now exports as F8_E4M3FN, not F16");
        } catch (IOException expected) {
            assertTrue(expected.getMessage().contains("no COMPAT conversion"));
        }
    }

    @Test
    public void theQ8_0RequantizationOfAPer128ScaleMatchesTheReferenceQuantizer() throws IOException {
        // a per-128 tensor of four 32-value blocks, requantized per 32 exactly like quantize_row_q8_0_ref
        float d = 8.0f;
        int[] codes = e4m3Codes(128);
        byte[] out = convertTo(b128Bytes(d, codes), new long[]{128}, GGMLType.F8_E4M3_B128, GGMLType.Q8_0);
        float[] x = new float[128];
        for (int i = 0; i < 128; i++) {
            x[i] = d * FpBlockDecoders.e4m3ToFp32(codes[i]);
        }
        assertArrayEquals(GgufTestWriter.encodeFloats(GGMLType.Q8_0, x), out);
    }

    @Test
    public void theI8RequantizationKeepsTheExactScaleTheFp16FormatRounds() throws IOException {
        int q = 13;
        float d = 0.25f; // exactly representable in fp16, so the requantized block is checked tightly
        byte[] raw = i8Bytes(d, q);
        byte[] out = convertTo(raw, new long[]{128}, GGMLType.I8_B128, GGMLType.Q8_0);
        float[] x = new float[128];
        Arrays.fill(x, d * q);
        assertArrayEquals(GgufTestWriter.encodeFloats(GGMLType.Q8_0, x), out);
    }

    @Test
    public void theNf4RowConvertsThroughTheCodebook() throws IOException {
        float absmax = 8.0f;
        byte[] out = convertTo(bnb4Bytes(absmax, 5), new long[]{64}, GGMLType.NF4, GGMLType.F16);
        float[] x = new float[64];
        Arrays.fill(x, QuantTables.nf4Code(5) * absmax);
        assertArrayEquals(GgufTestWriter.encodeFloats(GGMLType.F16, x), out);
    }

    @Test
    public void theFp4BnbRowConvertsThroughItsCodebook() throws IOException {
        float absmax = 4.0f;
        byte[] out = convertTo(bnb4Bytes(absmax, 9), new long[]{64}, GGMLType.FP4_BNB, GGMLType.F16);
        float[] x = new float[64];
        Arrays.fill(x, QuantTables.fp4BnbCode(9) * absmax);
        assertArrayEquals(GgufTestWriter.encodeFloats(GGMLType.F16, x), out);
    }

    // ================================================================== helpers

    /** Three-block MXFP8 tensor of 96 elements, one per sub-test, and its target type. */
    private static GGMLType mxfp8Target(int e0, int e1, int e2) throws IOException {
        GgufWriterTest.TestHfSource src = new GgufWriterTest.TestHfSource("synthetic://mxfp8");
        src.tensor(NAME, GGMLType.MXFP8, new long[]{96}, mxfp8Blocks(e0, e1, e2));
        return GgufCompatConverter.targetType(src.getTensors().get(0), src.createMappedStore(),
                GgufWriter.Mode.COMPAT);
    }

    private static IOException mxfp8TargetFails(int e) {
        GgufWriterTest.TestHfSource src = new GgufWriterTest.TestHfSource("synthetic://mxfp8-overflow");
        src.tensor(NAME, GGMLType.MXFP8, new long[]{32}, mxfp8Block(e4m3Codes(32), e));
        try {
            GgufCompatConverter.targetType(src.getTensors().get(0), src.createMappedStore(),
                    GgufWriter.Mode.COMPAT);
            fail("target type of exponent " + e + " was expected to fail");
            return null; // not reached
        } catch (IOException failure) {
            return failure;
        }
    }

    /** The int codes of one MXFP8 block: the exponent byte then 32 E4M3FN codes. */
    private static int[] mxfp8Block(int[] codes, int e) {
        int[] bytes = new int[1 + codes.length];
        bytes[0] = e;
        System.arraycopy(codes, 0, bytes, 1, codes.length);
        return bytes;
    }

    /** The int codes of a three-block MXFP8 tensor, one exponent per block. */
    private static int[] mxfp8Blocks(int e0, int e1, int e2) {
        int[] codes = e4m3Codes(32);
        int[] bytes = new int[3 * (1 + codes.length)];
        System.arraycopy(mxfp8Block(codes, e0), 0, bytes, 0, 33);
        System.arraycopy(mxfp8Block(codes, e1), 0, bytes, 33, 33);
        System.arraycopy(mxfp8Block(codes, e2), 0, bytes, 66, 33);
        return bytes;
    }

    /** The byte form of one MXFP8 block. */
    private static byte[] mxfp8Bytes(int[] codes, int e) {
        byte[] b = new byte[1 + codes.length];
        b[0] = (byte) e;
        for (int i = 0; i < codes.length; i++) {
            b[1 + i] = (byte) codes[i];
        }
        return b;
    }

    /** The byte form of one F8_E4M3_B128 block: a little-endian f32 scale then 128 codes. */
    private static byte[] b128Bytes(float d, int[] codes) {
        byte[] b = new byte[4 + codes.length];
        for (int i = 0; i < codes.length; i++) {
            b[4 + i] = (byte) codes[i];
        }
        ByteBuffer le = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        le.putFloat(0, d);
        return b;
    }

    /** The byte form of one I8_B128 block: a little-endian f32 scale then 128 signed codes. */
    private static byte[] i8Bytes(float d, int q) {
        byte[] b = new byte[132];
        Arrays.fill(b, 4, b.length, (byte) q);
        ByteBuffer le = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        le.putFloat(0, d);
        return b;
    }

    /** The byte form of one NF4 / FP4_BNB block: a little-endian f32 absmax then 32 packed bytes. */
    private static byte[] bnb4Bytes(float absmax, int q) {
        byte[] b = new byte[36];
        Arrays.fill(b, 4, b.length, (byte) (q | q << 4));
        ByteBuffer le = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        le.putFloat(0, absmax);
        return b;
    }

    private static byte[] bytes(int[] codes) {
        byte[] b = new byte[codes.length];
        for (int i = 0; i < codes.length; i++) {
            b[i] = (byte) codes[i];
        }
        return b;
    }

    /** Small positive E4M3FN codes, never 0x7F (NaN). */
    private static int[] e4m3Codes(int n) {
        int[] codes = new int[n];
        for (int i = 0; i < n; i++) {
            codes[i] = (i % 30) + 0x08;
        }
        return codes;
    }

    /**
     * Bytes of one source tensor converted to {@code to}, captured through {@link GgufEncoder}: the exact
     * write path the writer takes, including the store reads by virtual offset.
     */
    private static byte[] convertTo(byte[] raw, long[] dims, GGMLType from, GGMLType to) throws IOException {
        GGUFTensorInfo source = new GGUFTensorInfo(NAME, dims, from, 0);
        GGUFTensorInfo target = new GGUFTensorInfo(NAME, dims, to, 0);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        GgufEncoder enc = new GgufEncoder(Channels.newChannel(out), "conversion");
        try (WeightStore store = new OneTensorStore(raw)) {
            GgufCompatConverter.convert(source, target, store, enc);
        }
        enc.flush();
        byte[] all = out.toByteArray();
        assertEquals("byte count of the " + to + " tensor", bytesFor(to, dims), all.length);
        return all;
    }

    private static int bytesFor(GGMLType type, long[] dims) {
        long n = 1;
        for (long d : dims) {
            n *= d;
        }
        return (int) type.getBytesForElements(n);
    }

    /** A store over exactly one tensor's bytes; virtual offsets of the tensor are the array offsets. */
    private static final class OneTensorStore implements WeightStore {
        private final byte[] data;

        OneTensorStore(byte[] data) {
            this.data = data;
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) throws IOException {
            readTensor(tensorOffset, size, dst, 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
            if (tensorOffset < 0 || size < 0 || size > data.length - tensorOffset) {
                throw new IOException("read [" + tensorOffset + ", +" + size + ") is outside the "
                        + data.length + " bytes of the tensor");
            }
            System.arraycopy(data, (int) tensorOffset, dst, dstOffset, size);
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
            byte[] chunk = new byte[size];
            readTensor(tensorOffset, size, chunk);
            dst.put(chunk);
        }

        @Override
        public void close() {
        }
    }
}
