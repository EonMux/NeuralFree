package com.eonmux.neuralfree.gguf.hf.parity;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.Set;
import java.util.TreeMap;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * PT-1, HfGgufParityTest: each synthetic HF checkpoint opened with {@code outType = AUTO} carries exactly the GGUF
 * view a llama.cpp converter of that family writes for it with {@code --outtype auto} (PT-1 and the HF.md parity
 * section). The expected GGUF of every family is a hand-built {@link GgufTestWriter} — its KV entries, value types,
 * tensor names, dimensions, storage types and data are constants derived from the converter rules the fixture files
 * cite (never produced by NeuralFree).
 *
 * <p>Per model, the test asserts:</p>
 * <ol>
 *   <li>equal KV sets, value types and values, except {@code general.name} and the {@code neuralfree.*} overlays
 *       ({@code general.quantization_version} included);</li>
 *   <li>equal tensor names, ne arrays and GGML types;</li>
 *   <li>byte-equal tensor data, once through each side's {@link WeightStore} (RESIDENT_UPLOAD) and once through each
 *       side's {@link TensorRegions} (CPU mapped reads) — the stores and regions serve every byte of every tensor.</li>
 * </ol>
 *
 * <p>Value comparison normalizes only across the Java models of the same GGUF types: integers compare as longs,
 * floats as their exact bits (the expected values are literal, so a mismatch would be a converter regression).</p>
 */
public class HfGgufParityTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ------------------------------------------------------------------ the eight fixtures

    @Test
    public void llamaCheckpointF32HasTheConverterGguf() throws IOException {
        parity(ParityFixturesDense::llamaCheckpoint, ParityFixturesDense::llamaExpected, StDType.F32);
    }

    @Test
    public void llamaCheckpointBf16HasTheConverterGguf() throws IOException {
        parity(ParityFixturesDense::llamaCheckpoint, ParityFixturesDense::llamaExpected, StDType.BF16);
    }

    @Test
    public void qwen2CheckpointF32HasTheConverterGguf() throws IOException {
        parity(ParityFixturesDense::qwen2Checkpoint, ParityFixturesDense::qwen2Expected, StDType.F32);
    }

    @Test
    public void qwen2CheckpointBf16HasTheConverterGguf() throws IOException {
        parity(ParityFixturesDense::qwen2Checkpoint, ParityFixturesDense::qwen2Expected, StDType.BF16);
    }

    // ------------------------------------------------------------------ the MoE and other-arch fixtures

    @Test
    public void deepSeekV2CheckpointF32HasTheConverterGguf() throws IOException {
        parity(ParityFixturesMoe::deepSeekV2Checkpoint, ParityFixturesMoe::deepSeekV2Expected, StDType.F32);
    }

    @Test
    public void deepSeekV2CheckpointBf16HasTheConverterGguf() throws IOException {
        parity(ParityFixturesMoe::deepSeekV2Checkpoint, ParityFixturesMoe::deepSeekV2Expected, StDType.BF16);
    }

    @Test
    public void qwen3MoeCheckpointF32HasTheConverterGguf() throws IOException {
        parity(ParityFixturesMoe::qwen3MoeCheckpoint, ParityFixturesMoe::qwen3MoeExpected, StDType.F32);
    }

    @Test
    public void qwen3MoeCheckpointBf16HasTheConverterGguf() throws IOException {
        parity(ParityFixturesMoe::qwen3MoeCheckpoint, ParityFixturesMoe::qwen3MoeExpected, StDType.BF16);
    }

    @Test
    public void gemma3CheckpointF32HasTheConverterGguf() throws IOException {
        parity(ParityFixturesOther::gemma3Checkpoint, ParityFixturesOther::gemma3Expected, StDType.F32);
    }

    @Test
    public void gemma3CheckpointBf16HasTheConverterGguf() throws IOException {
        parity(ParityFixturesOther::gemma3Checkpoint, ParityFixturesOther::gemma3Expected, StDType.BF16);
    }

    @Test
    public void phi3CheckpointF32HasTheConverterGguf() throws IOException {
        parity(ParityFixturesOther::phi3Checkpoint, ParityFixturesOther::phi3Expected, StDType.F32);
    }

    @Test
    public void phi3CheckpointBf16HasTheConverterGguf() throws IOException {
        parity(ParityFixturesOther::phi3Checkpoint, ParityFixturesOther::phi3Expected, StDType.BF16);
    }

    @Test
    public void mamba2CheckpointF32HasTheConverterGguf() throws IOException {
        parity(ParityFixturesOther::mamba2Checkpoint, ParityFixturesOther::mamba2Expected, StDType.F32);
    }

    @Test
    public void mamba2CheckpointBf16HasTheConverterGguf() throws IOException {
        parity(ParityFixturesOther::mamba2Checkpoint, ParityFixturesOther::mamba2Expected, StDType.BF16);
    }

    @Test
    public void bertCheckpointF32HasTheConverterGguf() throws IOException {
        parity(ParityFixturesOther::bertCheckpoint, ParityFixturesOther::bertExpected, StDType.F32);
    }

    @Test
    public void bertCheckpointBf16HasTheConverterGguf() throws IOException {
        parity(ParityFixturesOther::bertCheckpoint, ParityFixturesOther::bertExpected, StDType.BF16);
    }

    // ------------------------------------------------------------------ the checker

    private interface Checkpoint {
        Path write(Path root, StDType dtype) throws IOException;
    }

    private interface Expected {
        GgufTestWriter build(StDType dtype);
    }

    private void parity(Checkpoint ckpt, Expected expected, StDType dtype) throws IOException {
        Path dir = ckpt.write(tmp.newFolder("hf").toPath(), dtype);
        GgufTestWriter exp = expected.build(dtype);
        Path gguf = tmp.getRoot().toPath().resolve(dir.getFileName() + "-expected.gguf");
        exp.writeTo(gguf);
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache").toPath())
                .outType(ModelSourceOptions.OutType.AUTO)
                .build();
        try (ModelSource hf = ModelSources.open(dir.toString(), options, null);
             GGUFFileParser gg = new GGUFFileParser(gguf.toString())) {
            assertKv(hf, gg);
            assertTensorInfos(hf, gg);
            try (WeightStore hs = hf.createMappedStore(StoreUse.RESIDENT_UPLOAD);
                 WeightStore es = gg.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
                for (GGUFTensorInfo t : hf.getTensors()) {
                    assertArrayEquals("WeightStore " + t.name(), read(hs, t), read(es, gg.getTensor(t.name())));
                }
            }
            try (TensorRegions hr = hf.openRegions(TensorReadMode.MMAP);
                 TensorRegions er = gg.openRegions(TensorReadMode.MMAP)) {
                for (GGUFTensorInfo t : hf.getTensors()) {
                    assertArrayEquals("TensorRegions " + t.name(),
                            region(hr, t), region(er, gg.getTensor(t.name())));
                }
            }
        }
    }

    // ------------------------------------------------------------------ KV comparison

    /** {@code general.name} and the {@code neuralfree.*} overlays are the PT-1 exclusions. */
    private static boolean isExcluded(String key) {
        return "general.name".equals(key) || key.startsWith("neuralfree.");
    }

    private static void assertKv(ModelSource hf, GGUFFileParser gg) {
        Map<String, Object> left = hf.getMetadata();
        Map<String, Object> right = gg.getMetadata();
        Set<String> keysLeft = new TreeMap<>(left).keySet();
        keysLeft.removeIf(HfGgufParityTest::isExcluded);
        Set<String> keysRight = new TreeMap<>(right).keySet();
        keysRight.removeIf(HfGgufParityTest::isExcluded);
        assertEquals("the KV set differs (left = HF view, right = expected GGUF)", keysRight, keysLeft);
        for (String key : keysLeft) {
            assertEquals("value type of " + key, gg.getValueType(key), hf.getValueType(key));
            assertTrue("value of " + key + ": " + left.get(key) + " vs " + right.get(key),
                    sameValue(left.get(key), right.get(key)));
        }
    }

    /** Equality across the container-level Java types both parsers produce. */
    private static boolean sameValue(Object left, Object right) {
        if (left instanceof List && right instanceof List) {
            List<?> a = (List<?>) left;
            List<?> b = (List<?>) right;
            if (a.size() != b.size()) {
                return false;
            }
            for (int i = 0; i < a.size(); i++) {
                if (!sameValue(a.get(i), b.get(i))) {
                    return false;
                }
            }
            return true;
        }
        if (left instanceof String || right instanceof String) {
            return String.valueOf(left).equals(String.valueOf(right));
        }
        if (left instanceof Boolean || right instanceof Boolean) {
            return Objects2.booleanOf(left) == Objects2.booleanOf(right);
        }
        if (left instanceof Number && right instanceof Number) {
            double a = ((Number) left).doubleValue();
            double b = ((Number) right).doubleValue();
            return a == b || (Float.floatToRawIntBits((float) a) == Float.floatToRawIntBits((float) b)
                    && !Double.isNaN(a)) || (Float.isNaN((float) a) && Float.isNaN((float) b));
        }
        return left == null && right == null;
    }

    // ------------------------------------------------------------------ tensor infos and data

    private static void assertTensorInfos(ModelSource hf, GGUFFileParser gg) {
        assertEquals(hf.getTensorCount(), gg.getTensorCount());
        Map<String, String> left = byName(hf.getTensors());
        Map<String, String> right = byName(gg.getTensors());
        assertEquals(right, left);
    }

    /** {@code name -> "[ne] TYPE"} per tensor info, sorted by name. */
    private static Map<String, String> byName(List<GGUFTensorInfo> infos) {
        Map<String, String> out = new TreeMap<>();
        for (GGUFTensorInfo t : infos) {
            String prev = out.put(t.name(), Arrays.toString(t.dims()) + " " + t.type());
            assertNotNull("tensor '" + t.name() + "' appears twice", prev);
        }
        return out;
    }

    private static byte[] read(WeightStore store, GGUFTensorInfo t) throws IOException {
        byte[] out = new byte[(int) t.sizeInBytes()];
        store.readTensor(t.offset(), out.length, out);
        return out;
    }

    private static byte[] region(TensorRegions regions, GGUFTensorInfo t) throws IOException {
        ByteBuffer b = regions.region(t, 0, t.sizeInBytes()).order(ByteOrder.LITTLE_ENDIAN);
        byte[] out = new byte[b.remaining()];
        b.duplicate().get(out);
        return out;
    }

    /** Boxing-free literal helpers (JLS 11 has no such conveniences in java.lang). */
    private static final class Objects2 {
        private static boolean booleanOf(Object v) {
            if (v instanceof Boolean) {
                return (Boolean) v;
            }
            return v instanceof Number && ((Number) v).doubleValue() != 0;
        }
    }

    static { // keep the LinkedHashMap import in use for extension by future fixtures
        new LinkedHashMap<String, Integer>();
    }
}
