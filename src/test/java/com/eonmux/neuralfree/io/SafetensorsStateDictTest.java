package com.eonmux.neuralfree.io;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.norm.BatchNorm1d;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.HalfFloats;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SafetensorsStateDict} stores a module's state dict as safetensors and loads it back: a two-layer model round
 * trips in F32, BF16 and F16, a batch counter keeps its dtype under a dtype override, a stored tensor of another shape
 * or an FP8 dtype is refused without changing the model, strict loads report unexpected and missing keys, sharded
 * directories load through their index, the header carries the given metadata and {@code format = pt}, a PyTorch
 * batch-norm checkpoint with mixed dtypes and an I64 counter loads strictly, and every integer, BOOL and F64 dtype
 * decodes to its value. Runs on the host reference backend.
 */
public class SafetensorsStateDictTest {
    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    private ComputeBackend host;

    @Before
    public void setUp() {
        host = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        TrainGpu.release(host);
        host.close();
    }

    @Test
    public void stateInfoListsTheStateDictEntriesWithTheirShapesAndDtypes() {
        try (Sequential model = twoLayer(4, 3, 2, 1L)) {
            Map<String, Module.StateInfo> info = model.stateInfo();

            assertEquals(new ArrayList<>(model.stateDict().keySet()), new ArrayList<>(info.keySet()));
            assertEquals(Shape.matrix(3, 4), info.get("0.weight").shape());
            assertEquals(Shape.vector(2), info.get("1.bias").shape());
            assertEquals(DType.FLOAT32, info.get("1.weight").dtype());
            assertThrows(UnsupportedOperationException.class, () -> info.remove("0.bias"));
        }
    }

    @Test
    public void aTwoLayerModelRoundTripsInF32() throws IOException {
        assertRoundTrip(null, StDType.F32);
        assertRoundTrip(DType.FLOAT32, StDType.F32);
    }

    @Test
    public void aTwoLayerModelRoundTripsInBf16() throws IOException {
        assertRoundTrip(DType.BFLOAT16, StDType.BF16);
    }

    @Test
    public void aTwoLayerModelRoundTripsInF16() throws IOException {
        assertRoundTrip(DType.FLOAT16, StDType.F16);
    }

    @Test
    public void aDtypeOverrideLeavesTheBatchCounterAtItsOwnDtype() throws IOException {
        Path path = folder.getRoot().toPath().resolve("bn.safetensors");
        try (BatchNorm1d source = batchNorm(2); BatchNorm1d target = batchNorm(2)) {
            Map<String, Object> state = source.stateDict();
            state.put("running_mean", new float[]{0.1f, -0.3f});
            state.put("num_batches_tracked", new float[]{257f});
            source.loadStateDict(state, true);
            assertTrue(source.stateInfo().get("num_batches_tracked").keepsDtype());

            SafetensorsStateDict.save(source, path,
                SafetensorsStateDict.SaveOptions.defaults().withDtype(DType.BFLOAT16));

            try (SafetensorsFile file = SafetensorsFile.open(path)) {
                StTensor counter = file.tensor("num_batches_tracked");
                assertEquals(StDType.F32, counter.dtype());
                assertEquals(0, counter.rank());
                assertEquals(StDType.BF16, file.tensor("running_mean").dtype());
            }
            SafetensorsStateDict.load(target, path, true);

            Map<String, Object> loaded = target.stateDict();
            assertArrayEquals(new float[]{257f}, (float[]) loaded.get("num_batches_tracked"), 0f);
            assertArrayEquals(new float[]{bf16(0.1f), bf16(-0.3f)}, (float[]) loaded.get("running_mean"), 0f);
        }
    }

    @Test
    public void aStoredTensorOfAnotherShapeIsRefusedNamingTheKeyAndBothShapes() throws IOException {
        Path path = folder.getRoot().toPath().resolve("shape.safetensors");
        try (Sequential source = twoLayer(4, 3, 2, 1L); Sequential target = twoLayer(3, 4, 2, 2L)) {
            SafetensorsStateDict.save(source, path);
            Map<String, Object> before = target.stateDict();

            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> SafetensorsStateDict.load(target, path, false));

            assertTrue(e.getMessage(), e.getMessage().contains("'0.weight'"));
            assertTrue(e.getMessage(), e.getMessage().contains("[3, 4]"));
            assertTrue(e.getMessage(), e.getMessage().contains("[4, 3]"));
            assertStateEquals(before, target.stateDict());
        }
    }

    @Test
    public void strictLoadsReportUnexpectedAndMissingKeysWhileNonStrictLoadsIgnoreThem() throws IOException {
        Path extra = folder.getRoot().toPath().resolve("extra.safetensors");
        Path partial = folder.getRoot().toPath().resolve("partial.safetensors");
        try (Sequential source = twoLayer(4, 3, 2, 1L); Sequential target = twoLayer(4, 3, 2, 2L)) {
            SafetensorsWriter.Builder withExtra = stateWriter(source, Collections.<String>emptySet());
            withExtra.add("extra.scale", StDType.F8_E4M3, new long[]{2}, new byte[]{0x38, 0x40});
            withExtra.build().write(extra);
            stateWriter(source, Collections.singleton("1.bias")).build().write(partial);
            Map<String, Object> before = target.stateDict();

            IllegalArgumentException unexpected = assertThrows(IllegalArgumentException.class,
                () -> SafetensorsStateDict.load(target, extra, true));
            assertTrue(unexpected.getMessage(), unexpected.getMessage().contains("unexpected keys [extra.scale]"));
            IllegalArgumentException missing = assertThrows(IllegalArgumentException.class,
                () -> SafetensorsStateDict.load(target, partial, true));
            assertTrue(missing.getMessage(), missing.getMessage().contains("missing keys [1.bias]"));
            assertStateEquals(before, target.stateDict());

            SafetensorsStateDict.load(target, extra, false);
            assertStateEquals(source.stateDict(), target.stateDict());
        }
    }

    @Test
    public void aShardedDirectoryLoadsThroughItsIndex() throws IOException {
        Path dir = folder.newFolder("sharded").toPath();
        try (Sequential source = twoLayer(4, 3, 2, 1L); Sequential target = twoLayer(4, 3, 2, 2L)) {
            Set<String> second = new HashSet<>(Arrays.asList("1.weight", "1.bias"));
            Set<String> first = new HashSet<>(Arrays.asList("0.weight", "0.bias"));
            stateWriter(source, second).build().write(dir.resolve("model-00001-of-00002.safetensors"));
            stateWriter(source, first).build().write(dir.resolve("model-00002-of-00002.safetensors"));
            String index = "{\"metadata\": {\"total_size\": 76}, \"weight_map\": {"
                + "\"0.weight\": \"model-00001-of-00002.safetensors\","
                + "\"0.bias\": \"model-00001-of-00002.safetensors\","
                + "\"1.weight\": \"model-00002-of-00002.safetensors\","
                + "\"1.bias\": \"model-00002-of-00002.safetensors\"}}";
            Files.write(dir.resolve("model.safetensors.index.json"), index.getBytes(StandardCharsets.UTF_8));

            SafetensorsStateDict.load(target, dir, true);

            assertStateEquals(source.stateDict(), target.stateDict());
            SafetensorsStateDict.Contents contents = SafetensorsStateDict.read(dir);
            assertEquals(4, contents.size());
            assertArrayEquals(new long[]{2, 3}, contents.shape("1.weight"));
            assertEquals(StDType.F32, contents.dtype("0.bias"));
            assertEquals(SafetensorsStateDict.FORMAT_PT, contents.metadata().get(SafetensorsStateDict.FORMAT_KEY));
            assertStateEquals(source.stateDict(), contents.toStateDict());
        }
    }

    @Test
    public void theHeaderCarriesTheGivenMetadataAndFormatPt() throws IOException {
        Path path = folder.getRoot().toPath().resolve("meta.safetensors");
        Map<String, String> metadata = new HashMap<>();
        metadata.put("epoch", "3");
        metadata.put("note", "two layers");
        try (Sequential model = twoLayer(4, 3, 2, 1L)) {
            SafetensorsStateDict.save(model, path, SafetensorsStateDict.SaveOptions.of(null, metadata));
        }

        Map<String, String> expected = new HashMap<>(metadata);
        expected.put("format", "pt");
        try (SafetensorsFile file = SafetensorsFile.open(path)) {
            assertEquals(expected, file.metadata());
        }
        assertEquals(expected, SafetensorsStateDict.read(path).metadata());
    }

    @Test
    public void aStateTensorOfAnFp8DtypeIsRefusedByName() throws IOException {
        Path path = folder.getRoot().toPath().resolve("fp8.safetensors");
        try (Sequential target = twoLayer(4, 3, 2, 2L)) {
            SafetensorsWriter.builder()
                .add("0.weight", StDType.F8_E4M3, new long[]{3, 4}, new byte[12])
                .build().write(path);
            Map<String, Object> before = target.stateDict();

            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> SafetensorsStateDict.load(target, path, false));

            assertTrue(e.getMessage(), e.getMessage().contains("'0.weight'"));
            assertTrue(e.getMessage(), e.getMessage().contains("F8_E4M3"));
            assertStateEquals(before, target.stateDict());
            assertThrows(IllegalArgumentException.class, () -> SafetensorsStateDict.read(path));
        }
    }

    @Test
    public void aPyTorchBatchNormCheckpointWithAnInt64CounterLoadsStrictly() throws IOException {
        Path path = folder.getRoot().toPath().resolve("torch-bn.safetensors");
        float[] mean = {0.25f, -1.5f};
        float[] var = {2f, 0.5f};
        SafetensorsWriter.builder().metadata("format", "pt")
            .add("weight", StDType.BF16, new long[]{2}, bf16Bytes(new float[]{1.5f, -0.75f}))
            .add("bias", StDType.F16, new long[]{2}, f16Bytes(new float[]{0.125f, 3f}))
            .add("running_mean", StDType.F32, new long[]{2}, f32(mean))
            .add("running_var", StDType.F64, new long[]{2}, f64(var))
            .add("num_batches_tracked", StDType.I64, new long[0], i64(300L))
            .build().write(path);
        try (BatchNorm1d target = batchNorm(2)) {
            SafetensorsStateDict.load(target, path, true);

            Map<String, Object> loaded = target.stateDict();
            assertArrayEquals(new float[]{1.5f, -0.75f}, (float[]) loaded.get("weight"), 0f);
            assertArrayEquals(new float[]{0.125f, 3f}, (float[]) loaded.get("bias"), 0f);
            assertArrayEquals(mean, (float[]) loaded.get("running_mean"), 0f);
            assertArrayEquals(var, (float[]) loaded.get("running_var"), 0f);
            assertArrayEquals(new float[]{300f}, (float[]) loaded.get("num_batches_tracked"), 0f);
        }
    }

    @Test
    public void readDecodesTheIntegerBoolAndF64Dtypes() throws IOException {
        Path path = folder.getRoot().toPath().resolve("ints.safetensors");
        ByteBuffer u64 = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
        u64.putLong(0, -1L).putLong(8, 1L << 63);
        ByteBuffer u32 = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        u32.putInt(0, -1);
        ByteBuffer i16 = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        i16.putShort(0, (short) -32768).putShort(2, (short) 1234);
        SafetensorsWriter.builder()
            .add("b", StDType.BOOL, new long[]{3}, new byte[]{0, 1, 2})
            .add("i8", StDType.I8, new long[]{2}, new byte[]{-3, 5})
            .add("u8", StDType.U8, new long[]{1}, new byte[]{(byte) 250})
            .add("i16", StDType.I16, new long[]{2}, i16.array())
            .add("u16", StDType.U16, new long[]{1}, new byte[]{(byte) 0xff, (byte) 0xff})
            .add("u32", StDType.U32, new long[]{1}, u32.array())
            .add("i64", StDType.I64, new long[]{1}, i64(-7L))
            .add("u64", StDType.U64, new long[]{2}, u64.array())
            .add("f64", StDType.F64, new long[]{1}, f64(new float[]{0.1f}))
            .build().write(path);

        Map<String, float[]> values = SafetensorsStateDict.read(path).values();

        assertArrayEquals(new float[]{0f, 1f, 1f}, values.get("b"), 0f);
        assertArrayEquals(new float[]{-3f, 5f}, values.get("i8"), 0f);
        assertArrayEquals(new float[]{250f}, values.get("u8"), 0f);
        assertArrayEquals(new float[]{-32768f, 1234f}, values.get("i16"), 0f);
        assertArrayEquals(new float[]{65535f}, values.get("u16"), 0f);
        assertArrayEquals(new float[]{4294967295f}, values.get("u32"), 0f);
        assertArrayEquals(new float[]{-7f}, values.get("i64"), 0f);
        assertArrayEquals(new float[]{18446744073709551615f, 9223372036854775808f}, values.get("u64"), 0f);
        assertArrayEquals(new float[]{0.1f}, values.get("f64"), 0f);
    }

    @Test
    public void saveOptionsRefuseNonFloatDtypesAndAnotherFormat() {
        assertThrows(IllegalArgumentException.class,
            () -> SafetensorsStateDict.SaveOptions.of(DType.INT8, Collections.<String, String>emptyMap()));
        assertThrows(IllegalArgumentException.class,
            () -> SafetensorsStateDict.SaveOptions.of(DType.FLOAT64, Collections.<String, String>emptyMap()));
        assertThrows(IllegalArgumentException.class,
            () -> SafetensorsStateDict.SaveOptions.of(null, Collections.singletonMap("format", "tf")));
        SafetensorsStateDict.SaveOptions pt = SafetensorsStateDict.SaveOptions.of(DType.FLOAT16,
            Collections.singletonMap("format", "pt"));
        assertEquals(DType.FLOAT16, pt.dtype());
        assertNull(pt.withDtype(null).dtype());
        assertEquals(pt, SafetensorsStateDict.SaveOptions.of(DType.FLOAT16, Collections.singletonMap("format", "pt")));
        assertEquals(pt.hashCode(),
            SafetensorsStateDict.SaveOptions.of(DType.FLOAT16, Collections.singletonMap("format", "pt")).hashCode());
    }

    /** Saves a two-layer model with {@code dtype}, checks the stored types and shapes, and loads it strictly. */
    private void assertRoundTrip(DType dtype, StDType stored) throws IOException {
        Path path = folder.getRoot().toPath().resolve("two-layer-" + dtype + ".safetensors");
        try (Sequential source = twoLayer(4, 3, 2, 1L); Sequential target = twoLayer(4, 3, 2, 2L)) {
            SafetensorsStateDict.save(source, path, SafetensorsStateDict.SaveOptions.defaults().withDtype(dtype));

            Map<String, Module.StateInfo> info = source.stateInfo();
            try (SafetensorsFile file = SafetensorsFile.open(path)) {
                assertEquals(info.size(), file.tensors().size());
                for (StTensor t : file.tensors()) {
                    assertEquals(t.name(), stored, t.dtype());
                    assertArrayEquals(t.name(), dims(info.get(t.name()).shape()), t.shape());
                }
            }
            SafetensorsStateDict.load(target, path, true);

            Map<String, Object> expected = source.stateDict();
            for (Map.Entry<String, Object> e : expected.entrySet()) {
                float[] values = ((float[]) e.getValue()).clone();
                for (int i = 0; i < values.length; i++) {
                    values[i] = storedValue(dtype, values[i]);
                }
                e.setValue(values);
            }
            assertStateEquals(expected, target.stateDict());
        }
    }

    /** A writer of {@code model}'s state as F32 tensors, without the entries in {@code skip}. */
    private static SafetensorsWriter.Builder stateWriter(Module model, Set<String> skip) {
        SafetensorsWriter.Builder builder = SafetensorsWriter.builder().metadata("format", "pt");
        Map<String, Module.StateInfo> info = model.stateInfo();
        for (Map.Entry<String, Object> e : model.stateDict().entrySet()) {
            if (!skip.contains(e.getKey())) {
                builder.add(e.getKey(), StDType.F32, dims(info.get(e.getKey()).shape()), f32((float[]) e.getValue()));
            }
        }
        return builder;
    }

    private static void assertStateEquals(Map<String, Object> expected, Map<String, Object> actual) {
        assertEquals(expected.keySet(), actual.keySet());
        for (Map.Entry<String, Object> e : expected.entrySet()) {
            assertArrayEquals(e.getKey(), (float[]) e.getValue(), (float[]) actual.get(e.getKey()), 0f);
        }
    }

    private static float storedValue(DType dtype, float v) {
        if (dtype == DType.BFLOAT16) {
            return bf16(v);
        }
        if (dtype == DType.FLOAT16) {
            return HalfFloats.halfToFloat(HalfFloats.floatToHalf(v));
        }
        return v;
    }

    private static float bf16(float v) {
        return HalfFloats.bfloat16ToFloat(HalfFloats.floatToBFloat16(v));
    }

    private static byte[] f32(float[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(values);
        return bytes.array();
    }

    private static byte[] bf16Bytes(float[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < values.length; i++) {
            bytes.putShort(i * Short.BYTES, (short) HalfFloats.floatToBFloat16(values[i]));
        }
        return bytes.array();
    }

    private static byte[] f16Bytes(float[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < values.length; i++) {
            bytes.putShort(i * Short.BYTES, (short) HalfFloats.floatToHalf(values[i]));
        }
        return bytes.array();
    }

    private static byte[] f64(float[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Double.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < values.length; i++) {
            bytes.putDouble(i * Double.BYTES, values[i]);
        }
        return bytes.array();
    }

    private static byte[] i64(long value) {
        return ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN).putLong(0, value).array();
    }

    private static long[] dims(Shape shape) {
        int[] dims = shape.dims();
        long[] out = new long[dims.length];
        for (int i = 0; i < dims.length; i++) {
            out[i] = dims[i];
        }
        return out;
    }

    /** {@code Linear(in, hidden)} then {@code Linear(hidden, out)}, seeded. */
    private Sequential twoLayer(int in, int hidden, int out, long seed) {
        List<Module> layers = new ArrayList<>();
        layers.add(linear(in, hidden, seed));
        layers.add(linear(hidden, out, seed + 1));
        return new Sequential(layers);
    }

    private Linear linear(int in, int out, long seed) {
        Linear layer = new Linear(in, out);
        layer.initialize(host, new Random(seed));
        return layer;
    }

    private BatchNorm1d batchNorm(int features) {
        BatchNorm1d bn = new BatchNorm1d(features);
        bn.initialize(host);
        return bn;
    }
}
