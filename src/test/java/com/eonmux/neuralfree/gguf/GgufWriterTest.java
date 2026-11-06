package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The acceptance tests of {@link GgufWriter}: a round trip of a {@code GgufTestWriter} file with
 * byte-identical tensor data and an identical KV set and types; an HF synthetic source exported NATIVE and
 * COMPAT and re-opened with {@link GGUFFileParser}; the alignment rules and the atomic rename. The source
 * doubles are a tiny in-memory checkpoint in the GGUF contract exactly as an HF converter produces it
 * ({@link TestHfSource}) and the byte store behind it ({@link TestWeightStore}), re-used by
 * {@link GgufCompatConverterTest}.
 */
public class GgufWriterTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ================================================================== test doubles

    /** A tensor's bytes over {@code byte[]}; the writer reads it in a few sequential passes. */
    static final class TestWeightStore implements WeightStore {
        private final byte[] data;

        TestWeightStore(byte[] data) {
            this.data = Objects.requireNonNull(data);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) throws IOException {
            readTensor(tensorOffset, size, dst, 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
            check(tensorOffset, size);
            System.arraycopy(data, (int) tensorOffset, dst, dstOffset, size);
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
            check(tensorOffset, size);
            byte[] chunk = new byte[size];
            System.arraycopy(data, (int) tensorOffset, chunk, 0, size);
            dst.put(chunk);
        }

        private void check(long tensorOffset, int size) throws IOException {
            if (tensorOffset < 0 || size < 0 || size > data.length - tensorOffset) {
                throw new IOException("synthetic tensor read [" + tensorOffset + ", " + (tensorOffset + size)
                        + ") is outside the " + data.length + " stored bytes");
            }
        }

        @Override
        public void close() {
        }
    }

    /**
     * An in-memory checkpoint presented in the GGUF contract exactly as the HF converter would produce it:
     * metadata in write order, GGUF value types, a tensor table at virtual offsets over one byte array.
     */
    static final class TestHfSource implements ModelSource {

        final String path;
        final Map<String, Object> metadata = new LinkedHashMap<>();
        final Map<String, GGUFValueType> types = new LinkedHashMap<>();
        final Map<String, GGUFValueType> elementTypes = new LinkedHashMap<>();
        final List<GGUFTensorInfo> tensors = new ArrayList<>();
        final byte[] data = new byte[8192];
        int used;
        int tensorCount;

        TestHfSource(String path) {
            this.path = path;
        }

        /** Adds one scalar key/value pair in write order. */
        void kv(String key, GGUFValueType type, Object value) {
            metadata.put(key, value);
            types.put(key, type);
        }

        /** Adds one array key/value pair in write order. */
        void kvArray(String key, GGUFValueType elementType, Object... value) {
            List<Object> list = new ArrayList<>();
            Collections.addAll(list, value);
            metadata.put(key, list);
            types.put(key, GGUFValueType.ARRAY);
            elementTypes.put(key, elementType);
        }

        /** Appends {@code bytes} as the data of tensor {@code name}, at the next 32-aligned offset. */
        void tensor(String name, GGMLType type, long[] dims, int... bytes) {
            byte[] b = new byte[bytes.length];
            for (int i = 0; i < bytes.length; i++) {
                b[i] = (byte) bytes[i];
            }
            if (used + align32(b.length) > data.length) {
                throw new IllegalStateException("synthetic source is full after " + tensorCount + " tensors");
            }
            tensors.add(new GGUFTensorInfo(name, dims, type, used));
            System.arraycopy(b, 0, data, used, b.length);
            used += align32(b.length);
            tensorCount++;
        }

        private static long align32(long v) {
            return (v + 31) / 32 * 32;
        }

        @Override
        public ModelFormat format() {
            return ModelFormat.SAFETENSORS;
        }

        @Override
        public String getFilePath() {
            return path;
        }

        @Override
        public Map<String, Object> getMetadata() {
            return Collections.unmodifiableMap(metadata);
        }

        @Override
        public GGUFValueType getValueType(String key) {
            return types.get(key);
        }

        @Override
        public GGUFValueType getArrayElementType(String key) {
            return elementTypes.get(key);
        }

        @Override
        public String getString(String key) {
            Object v = metadata.get(key);
            return v instanceof String ? (String) v : null;
        }

        @Override
        public List<GGUFTensorInfo> getTensors() {
            return Collections.unmodifiableList(tensors);
        }

        @Override
        public long getTensorCount() {
            return tensorCount;
        }

        @Override
        public WeightStore createMappedStore(StoreUse use) {
            Objects.requireNonNull(use, "use");
            return new TestWeightStore(Arrays.copyOf(data, data.length));
        }

        @Override
        public TensorRegions openRegions(TensorReadMode mode) {
            throw new UnsupportedOperationException("no byte regions for a synthetic source; read the tensors "
                    + "through createMappedStore");
        }

        @Override
        public void close() {
        }
    }

    // ================================================================== fixtures

    /** E4M3FN codes that decode to small positive magnitudes, never 0x7F (NaN). */
    private static int[] e4m3Codes(int n) {
        int[] codes = new int[n];
        for (int i = 0; i < n; i++) {
            codes[i] = (i % 30) + 0x08;
        }
        return codes;
    }

    /** The storage bytes of {@link #e4m3Codes(int)}. */
    private static byte[] e4m3Bytes(int n) {
        int[] codes = e4m3Codes(n);
        byte[] b = new byte[codes.length];
        for (int i = 0; i < b.length; i++) {
            b[i] = (byte) codes[i];
        }
        return b;
    }

    /** One F32 little-endian value. */
    private static int[] f32(float v) {
        int bits = Float.floatToRawIntBits(v);
        return new int[]{bits & 0xFF, (bits >>> 8) & 0xFF, (bits >>> 16) & 0xFF, (bits >>> 24) & 0xFF};
    }

    /** One MXFP8 block: the E8M0 exponent {@code e} then 32 E4M3FN codes. */
    private static int[] mxfp8Block(int e, int[] codes) {
        int[] bytes = new int[1 + codes.length];
        bytes[0] = e;
        System.arraycopy(codes, 0, bytes, 1, codes.length);
        return bytes;
    }

    /** One F8_E4M3_B128 block: a little-endian f32 scale then 128 E4M3FN codes. */
    private static int[] scaleB128(float d, int[] codes) {
        int[] bytes = new int[4 + codes.length];
        System.arraycopy(f32(d), 0, bytes, 0, 4);
        System.arraycopy(codes, 0, bytes, 4, codes.length);
        return bytes;
    }

    /** One I8_B128 block: a little-endian f32 scale then 128 signed codes. */
    private static int[] scaleI8(float d, int q) {
        int[] bytes = new int[132];
        System.arraycopy(f32(d), 0, bytes, 0, 4);
        Arrays.fill(bytes, 4, bytes.length, q);
        return bytes;
    }

    /** One NF4 / FP4_BNB block: a little-endian f32 absmax then 32 nibble-packed bytes. */
    private static int[] bnb4(float absmax, int q) {
        int[] bytes = new int[36];
        System.arraycopy(f32(absmax), 0, bytes, 0, 4);
        Arrays.fill(bytes, 4, bytes.length, q | q << 4);
        return bytes;
    }

    /** The synthetic HF checkpoint of the export tests: one tensor of every extension type. */
    private TestHfSource hfSource(boolean withExtensionMarker) {
        TestHfSource src = new TestHfSource("synthetic://hf/model");
        src.kv(GgufKeys.GENERAL_ARCHITECTURE, GGUFValueType.STRING, "llama");
        src.kv(GgufKeys.CONTEXT_LENGTH, GGUFValueType.UINT32, 4096);
        src.kvArray("neuralfree.hadamard.weight_names", GGUFValueType.STRING, "blk.0.attn_q.weight");
        src.kvArray("tokenizer.ggml.tokens", GGUFValueType.STRING, "<unk>", "s");
        if (withExtensionMarker) {
            src.kv(GgufKeys.NF_EXTENSION_TYPES, GGUFValueType.BOOL, Boolean.TRUE);
        }
        src.tensor("blk.0.attn_q.weight", GGMLType.F8_E4M3, new long[]{32}, e4m3Codes(32));
        src.tensor("blk.0.attn_q.scale", GGMLType.F32, new long[]{1}, f32(1.25f));
        src.tensor("blk.1.attn_k.weight", GGMLType.MXFP8, new long[]{32}, mxfp8Block(10, e4m3Codes(32)));
        src.tensor("blk.2.attn_v.weight", GGMLType.F8_E4M3_B128, new long[]{128}, scaleB128(64.0f,
                e4m3Codes(128)));
        src.tensor("blk.3.attn_o.weight", GGMLType.I8_B128, new long[]{128}, scaleI8(0.5f, 7));
        src.tensor("blk.4.attn_f.weight", GGMLType.NF4, new long[]{64}, bnb4(1.0f, 3));
        src.tensor("blk.5.attn_g.weight", GGMLType.FP4_BNB, new long[]{64}, bnb4(2.0f, 9));
        return src;
    }

    /** A non-existing export target. */
    private Path out(String name) {
        return new File(tmp.getRoot(), name).toPath();
    }

    // ================================================================== round trip

    @Test
    public void roundTripKeepsTensorBytesAndEveryKeyValuePair() throws IOException {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("synthetic");
        w.u32(w.archKey("context_length"), 4096);
        w.f32(w.archKey("rope_freq_base"), 10000.0f);
        w.bool(w.archKey("causal"), true);
        w.str("tokenizer.chat_template", "{{ bos }}");
        w.strArray("tokenizer.ggml.tokens", "<unk>", "a", "b");
        w.u32Array("llama.attention.head_count", 32, 32);
        w.f64("custom.f64", -2.75);
        w.u64("custom.u64", -1L);
        w.alignment(64);
        w.tensorF32("a", new long[]{8}, 1.0f, 2.0f, 3.5f, -1.25f, 0f, 7f, 8f, 9f);
        w.tensorZeros("b", GGMLType.Q8_0, 64L);
        w.tensorFromFloats("c", GGMLType.F16, new long[]{16},
                new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f});
        w.tensorFromFloats("d", GGMLType.BF16, new long[]{8}, new float[]{-1f, 0.5f, 2f, 3f, 4f, 5f, 6f, 7f});
        w.remove("general.architecture");
        w.architecture("synthetic");
        Path fixture = w.writeTempFile();
        Path exported = out("export.gguf");
        try (GGUFFileParser in = new GGUFFileParser(fixture.toString())) {
            GgufWriter.write(in, exported, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (GGUFFileParser source = new GGUFFileParser(fixture.toString());
             GGUFFileParser export = new GGUFFileParser(exported.toString())) {
            assertEquals(source.getTensors().size(), export.getTensors().size());
            for (int i = 0; i < source.getTensors().size(); i++) {
                GGUFTensorInfo tIn = source.getTensors().get(i);
                GGUFTensorInfo tOut = export.getTensors().get(i);
                assertEquals(tIn.name(), tOut.name());
                assertEquals(tIn.type(), tOut.type());
                assertArrayEquals(tIn.dims(), tOut.dims());
                assertArrayEquals(source.readTensorData(tIn), export.readTensorData(tOut));
            }
            assertEquals(source.getMetadata().keySet(), export.getMetadata().keySet());
            for (String key : source.getMetadata().keySet()) {
                assertEquals("type of " + key, source.getValueType(key), export.getValueType(key));
                assertEquals("value of " + key, source.getMetadata().get(key), export.getMetadata().get(key));
            }
            assertEquals(source.getAlignment(), export.getAlignment());
            assertEquals(source.getTensorCount(), export.getTensorCount());
        }
    }

    @Test
    public void roundTripKeepsTheAlignmentOfTheSource() throws IOException {
        GgufTestWriter w = new GgufTestWriter().alignment(128);
        w.architecture("a");
        w.tensorF32("t", new long[]{4}, 1f, 2f, 3f, 4f);
        Path fixture = w.writeTempFile();
        Path exported = out("export128.gguf");
        try (GGUFFileParser in = new GGUFFileParser(fixture.toString())) {
            assertEquals(128, in.getAlignment());
            GgufWriter.write(in, exported, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (GGUFFileParser export = new GGUFFileParser(exported.toString())) {
            assertEquals(128, export.getAlignment());
            assertEquals(GGUFValueType.UINT32, export.getValueType("general.alignment"));
            assertEquals(Integer.valueOf(128), export.getInt("general.alignment"));
            for (GGUFTensorInfo t : export.getTensors()) {
                assertEquals(0, t.offset() % 128);
            }
        }
    }

    @Test
    public void aDeviceOnlyTensorIsRefusedByName() throws IOException {
        TestHfSource src = new TestHfSource("synthetic://device-only");
        src.kv(GgufKeys.GENERAL_ARCHITECTURE, GGUFValueType.STRING, "llama");
        src.tensor("w", GGMLType.W12, new long[]{64}, new int[96]);
        Path exported = out("device-only.gguf");
        try {
            GgufWriter.write(src, exported, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
            fail("a W12 tensor was written");
        } catch (IllegalArgumentException e) {
            assertEquals("tensor 'w' has type W12, a device-resident form that GGUF files never store",
                    e.getMessage());
        }
    }

    @Test
    public void optionsAlignmentIsWrittenWhenTheSourceHasNone() throws IOException {
        TestHfSource src = new TestHfSource("synthetic://no-alignment");
        src.kv(GgufKeys.GENERAL_ARCHITECTURE, GGUFValueType.STRING, "llama");
        src.tensor("t", GGMLType.F32, new long[]{4}, f32(1.0f));
        Path exported = out("align.gguf");
        GgufWriter.write(src, exported, new GgufWriter.Options(GgufWriter.Mode.NATIVE, false, 128));
        try (GGUFFileParser parser = new GGUFFileParser(exported.toString())) {
            assertEquals(128, parser.getAlignment());
            assertEquals(GGUFValueType.UINT32, parser.getValueType("general.alignment"));
        }
    }

    // ================================================================== atomic rename

    @Test
    public void exportFailsForAnExistingFileWithoutOverwriteAndWritesNoTmp() throws IOException {
        File existing = new File(tmp.getRoot(), "keep.gguf");
        existing.createNewFile();
        Files.write(existing.toPath(), new byte[]{1});
        TestHfSource src = simpleSource();
        IOException e = writeFails(src, existing.toPath(), new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        assertTrue(e.getMessage(), e.getMessage().contains("already exists"));
        assertArrayEquals(new byte[]{1}, Files.readAllBytes(existing.toPath()));
        assertFalse(Files.exists(out("keep.gguf.tmp")));
    }

    @Test
    public void exportWithOverwriteReplacesTheFileAtomically() throws IOException {
        File existing = new File(tmp.getRoot(), "replace.gguf");
        existing.createNewFile();
        Files.write(existing.toPath(), new byte[]{1});
        GgufWriter.write(simpleSource(), existing.toPath(),
                new GgufWriter.Options(GgufWriter.Mode.NATIVE).overwrite(true));
        assertFalse(Files.exists(out("replace.gguf.tmp")));
        try (GGUFFileParser parser = new GGUFFileParser(existing.toString())) {
            assertEquals(1, parser.getTensors().size());
            assertEquals("llama", parser.getString(GgufKeys.GENERAL_ARCHITECTURE));
        }
    }

    @Test
    public void aFailedExportRemovesItsTmpFileAndLeavesTheTarget() throws IOException {
        TestHfSource src = hfSource(false);
        src.tensor("blk.6.bad.weight", GGMLType.MXFP8, new long[]{32}, mxfp8Block(247, e4m3Codes(32)));
        IOException e = writeFails(src, out("failed.gguf"), new GgufWriter.Options(GgufWriter.Mode.COMPAT));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.6.bad.weight"));
        assertFalse(Files.exists(out("failed.gguf")));
        assertFalse(Files.exists(out("failed.gguf.tmp")));
    }

    /** A source of one F32 tensor and the architecture key. */
    private TestHfSource simpleSource() {
        TestHfSource src = new TestHfSource("synthetic://simple");
        src.kv(GgufKeys.GENERAL_ARCHITECTURE, GGUFValueType.STRING, "llama");
        src.tensor("t", GGMLType.F32, new long[]{4}, f32(1.0f));
        return src;
    }

    // ================================================================== HF synthetic exports

    @Test
    public void nativeExportKeepsExtensionTensorsAndWritesTheMarker() throws IOException {
        TestHfSource src = hfSource(false);
        Path exported = out("native.gguf");
        GgufWriter.write(src, exported, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        try (GGUFFileParser parser = new GGUFFileParser(exported.toString())) {
            assertEquals(GGUFValueType.BOOL, parser.getValueType(GgufKeys.NF_EXTENSION_TYPES));
            assertEquals(Boolean.TRUE, parser.getBool(GgufKeys.NF_EXTENSION_TYPES));
            assertEquals(GGMLType.F8_E4M3, parser.getTensor("blk.0.attn_q.weight").type());
            assertEquals(GGMLType.MXFP8, parser.getTensor("blk.1.attn_k.weight").type());
            assertEquals(GGMLType.F8_E4M3_B128, parser.getTensor("blk.2.attn_v.weight").type());
            assertEquals(GGMLType.I8_B128, parser.getTensor("blk.3.attn_o.weight").type());
            assertEquals(GGMLType.NF4, parser.getTensor("blk.4.attn_f.weight").type());
            assertEquals(GGMLType.FP4_BNB, parser.getTensor("blk.5.attn_g.weight").type());
            assertEquals(GGMLType.F32, parser.getTensor("blk.0.attn_q.scale").type());
        }
    }

    @Test
    public void nativeExportKeepsNativeOnlyTensors() throws IOException {
        TestHfSource src = hfSource(false);
        src.tensor("blk.6.attn_q.input_perm", GGMLType.F8_E4M3, new long[]{32}, e4m3Codes(32));
        Path exported = out("native-perm.gguf");
        GgufWriter.write(src, exported, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        try (GGUFFileParser parser = new GGUFFileParser(exported.toString())) {
            GGUFTensorInfo t = parser.getTensor("blk.6.attn_q.input_perm");
            assertNotNull(t);
            assertEquals(GGMLType.F8_E4M3, t.type());
            assertEquals(Boolean.TRUE, parser.getBool(GgufKeys.NF_EXTENSION_TYPES));
        }
    }

    @Test
    public void compatExportConvertsEveryExtensionTensorAndDropsTheMarker() throws IOException {
        TestHfSource src = hfSource(false);
        Path exported = out("compat.gguf");
        GgufWriter.write(src, exported, new GgufWriter.Options(GgufWriter.Mode.COMPAT));
        try (GGUFFileParser parser = new GGUFFileParser(exported.toString())) {
            assertNull(parser.getValueType(GgufKeys.NF_EXTENSION_TYPES));
            // the F8_E4M3 codes pass through as llama.cpp's per-element F8_E4M3FN
            assertEquals(GGMLType.F8_E4M3FN, parser.getTensor("blk.0.attn_q.weight").type());
            assertEquals(GGMLType.BF16, parser.getTensor("blk.1.attn_k.weight").type());
            assertEquals(GGMLType.Q8_0, parser.getTensor("blk.2.attn_v.weight").type());
            assertEquals(GGMLType.Q8_0, parser.getTensor("blk.3.attn_o.weight").type());
            assertEquals(GGMLType.F16, parser.getTensor("blk.4.attn_f.weight").type());
            assertEquals(GGMLType.F16, parser.getTensor("blk.5.attn_g.weight").type());
            assertEquals(GGMLType.F32, parser.getTensor("blk.0.attn_q.scale").type());
            // the values of the pass-through row are the codes' bytes
            assertArrayEquals(e4m3Bytes(32),
                    parser.readTensorData(parser.getTensor("blk.0.attn_q.weight")));
            // the neuralfree.* keys stay, llama.cpp ignores them
            assertEquals("llama", parser.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertNotNull(parser.getValueType("neuralfree.hadamard.weight_names"));
        }
    }

    @Test
    public void compatExportRefusesNativeOnlyTensorsWithTheNamedError() {
        TestHfSource perm = hfSource(false);
        perm.tensor("blk.6.attn_q.input_perm", GGMLType.F8_E4M3, new long[]{32}, e4m3Codes(32));
        IOException e = writeFails(perm, out("compat-perm.gguf"), new GgufWriter.Options(GgufWriter.Mode.COMPAT));
        assertTrue(e.getMessage(), e.getMessage().contains(".input_perm"));
        assertTrue(e.getMessage(), e.getMessage().contains("NATIVE"));

        TestHfSource zeroPoint = hfSource(false);
        zeroPoint.tensor("blk.6.attn_k.input_zero_point", GGMLType.F8_E4M3, new long[]{32}, e4m3Codes(32));
        IOException e2 = writeFails(zeroPoint, out("compat-zp.gguf"), new GgufWriter.Options(GgufWriter.Mode.COMPAT));
        assertTrue(e2.getMessage(), e2.getMessage().contains(".input_zero_point"));
    }

    @Test
    public void compatExactExportUsesF32ForEveryLossyRow() throws IOException {
        TestHfSource src = hfSource(false);
        Path exported = out("exact.gguf");
        GgufWriter.write(src, exported, new GgufWriter.Options(GgufWriter.Mode.COMPAT_EXACT));
        try (GGUFFileParser parser = new GGUFFileParser(exported.toString())) {
            assertNull(parser.getValueType(GgufKeys.NF_EXTENSION_TYPES));
            assertEquals(GGMLType.F32, parser.getTensor("blk.2.attn_v.weight").type());
            assertEquals(GGMLType.F32, parser.getTensor("blk.3.attn_o.weight").type());
            assertEquals(GGMLType.F32, parser.getTensor("blk.4.attn_f.weight").type());
            assertEquals(GGMLType.F32, parser.getTensor("blk.5.attn_g.weight").type());
            // the rows COMPAT keeps are still the COMPAT types
            assertEquals(GGMLType.F8_E4M3FN, parser.getTensor("blk.0.attn_q.weight").type());
            assertEquals(GGMLType.BF16, parser.getTensor("blk.1.attn_k.weight").type());
            // one lossy row exports as the exact floats of its blocks
            float[] v = parser.readTensorDataAsFloat(parser.getTensor("blk.3.attn_o.weight"));
            for (float x : v) {
                assertEquals(0.5f * 7f, x, 1e-6f);
            }
        }
    }

    @Test
    public void compatExportDropsTheMarkerTheSourceCarried() throws IOException {
        TestHfSource src = hfSource(true); // the HF converter already wrote the marker
        Path exported = out("compat-marked.gguf");
        GgufWriter.write(src, exported, new GgufWriter.Options(GgufWriter.Mode.COMPAT));
        try (GGUFFileParser parser = new GGUFFileParser(exported.toString())) {
            assertNull(parser.getValueType(GgufKeys.NF_EXTENSION_TYPES));
            assertEquals(GGMLType.Q8_0, parser.getTensor("blk.2.attn_v.weight").type());
        }
    }

    // ================================================================== helpers

    private static IOException writeFails(ModelSource src, Path out, GgufWriter.Options o) {
        try {
            GgufWriter.write(src, out, o);
            fail("export of " + o.mode + " was expected to fail");
            return null; // not reached
        } catch (IOException e) {
            return e;
        }
    }
}
