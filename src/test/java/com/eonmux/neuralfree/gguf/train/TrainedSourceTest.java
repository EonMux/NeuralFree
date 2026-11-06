package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The write-back view: trained bytes laid over one tensor of a file, exported through {@link GgufWriter}, come back
 * from the written file while every other tensor and the metadata are unchanged.
 */
public class TrainedSourceTest {

    private static final String TRAINED = "blk.0.ffn_norm.weight";
    private static final String UNTOUCHED = "blk.0.attn_q.weight";

    @Rule
    public final TemporaryFolder folder = new TemporaryFolder();

    private static Path file;
    private static Path quantizedFile;

    @BeforeClass
    public static void write() throws IOException {
        file = SyntheticModels.llama().writeTempFile();
        quantizedFile = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .weightType(GGMLType.Q8_0).build()).writeTempFile();
    }

    @AfterClass
    public static void close() throws IOException {
        if (file != null) {
            Files.deleteIfExists(file);
        }
        if (quantizedFile != null) {
            Files.deleteIfExists(quantizedFile);
        }
    }

    /** The F32 values of {@code name} in {@code source}, read through its store. */
    private static float[] values(ModelSource source, String name) throws IOException {
        GGUFTensorInfo info = tensor(source, name);
        byte[] bytes = new byte[(int) info.sizeInBytes()];
        try (WeightStore store = source.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            store.readTensor(info.offset(), bytes.length, bytes, 0);
        }
        float[] values = new float[(int) info.numElements()];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values);
        return values;
    }

    private static GGUFTensorInfo tensor(ModelSource source, String name) {
        for (GGUFTensorInfo info : source.getTensors()) {
            if (info.name().equals(name)) {
                return info;
            }
        }
        throw new AssertionError("no tensor " + name);
    }

    /** The base values with a shift applied, so a trained byte is distinguishable from a stored one. */
    private static float[] shifted(float[] values) {
        float[] out = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] + 0.5f;
        }
        return out;
    }

    @Test
    public void theWrittenFileCarriesTheTrainedValuesAndKeepsTheRest() throws IOException {
        ModelSource base = ModelSources.open(file.toString());
        Path out = folder.getRoot().toPath().resolve("trained.gguf");
        float[] before = values(base, TRAINED);
        float[] untouched = values(base, UNTOUCHED);
        float[] trained = shifted(before);
        Map<String, float[]> values = new LinkedHashMap<>();
        values.put(TRAINED, trained);
        try (TrainedSource source = TrainedSource.of(base, values)) {
            assertEquals(1, source.trainedTensorCount());
            assertEquals(base.getTensorCount(), source.getTensorCount());
            assertEquals(base.getTensors(), source.getTensors());
            assertEquals(base.getMetadata(), source.getMetadata());
            GgufWriter.write(source, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (ModelSource written = ModelSources.open(out.toString())) {
            assertArrayEquals(trained, values(written, TRAINED), 0f);
            assertArrayEquals(untouched, values(written, UNTOUCHED), 0f);
            // the writer adds general.alignment when the source carries none; every other key is the source's
            for (Map.Entry<String, Object> e : base.getMetadata().entrySet()) {
                assertEquals(e.getKey(), e.getValue(), written.getMetadata().get(e.getKey()));
            }
        }
    }

    @Test
    public void theTrainedWeightsChangeTheModelTheFileDescribes() throws IOException {
        Path out = folder.getRoot().toPath().resolve("logits.gguf");
        ModelSource base = ModelSources.open(file.toString());
        Map<String, float[]> values = new LinkedHashMap<>();
        values.put(TRAINED, shifted(values(base, TRAINED)));
        try (TrainedSource source = TrainedSource.of(base, values)) {
            GgufWriter.write(source, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        int[] tokens = {1, 7, 19, 42};
        float[] before = logits(file, tokens);
        float[] after = logits(out, tokens);
        boolean changed = false;
        for (int i = 0; i < before.length; i++) {
            changed |= before[i] != after[i];
        }
        assertTrue("the trained tensor does not reach the logits of the written file", changed);
    }

    private static float[] logits(Path model, int[] tokens) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(tokens.length).ubatch(tokens.length)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build())) {
            m.forwardBackend().reset();
            float[][] rows = m.forwardBackend().forwardRows(tokens, 0, null);
            return rows[rows.length - 1];
        }
    }

    @Test
    public void aReadThatCrossesTheEndOfTheTrainedTensorMixesBothSources() throws IOException {
        ModelSource base = ModelSources.open(file.toString());
        GGUFTensorInfo info = tensor(base, TRAINED);
        float[] trained = shifted(values(base, TRAINED));
        Map<String, float[]> values = new LinkedHashMap<>();
        values.put(TRAINED, trained);
        try (TrainedSource source = TrainedSource.of(base, values);
             WeightStore trainedStore = source.createMappedStore(StoreUse.RESIDENT_UPLOAD);
             WeightStore baseStore = base.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            int start = (int) info.sizeInBytes() - 3;
            int length = 6;
            byte[] mixed = new byte[length];
            trainedStore.readTensor(info.offset() + start, length, mixed, 0);
            byte[] stored = new byte[length];
            baseStore.readTensor(info.offset() + start, length, stored, 0);
            byte[] expected = stored.clone();
            System.arraycopy(floatBytes(trained), start, expected, 0, (int) info.sizeInBytes() - start);
            assertArrayEquals(expected, mixed);
            assertFalse("the trained bytes do not differ from the stored ones, so the range proves nothing",
                    Arrays.equals(stored, expected));
        }
    }

    private static byte[] floatBytes(float[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buffer.asFloatBuffer().put(values);
        return buffer.array();
    }

    @Test
    public void aNameTheModelDoesNotCarryIsRefused() throws IOException {
        ModelSource base = ModelSources.open(file.toString());
        Map<String, float[]> values = new LinkedHashMap<>();
        values.put("blk.99.ffn_norm.weight", new float[4]);
        try {
            TrainedSource.of(base, values);
            throw new AssertionError("an unknown tensor name was accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("no tensor"));
        } finally {
            base.close();
        }
    }

    @Test
    public void aTensorThatIsNotF32OrHasTheWrongLengthIsRefused() throws IOException {
        try (ModelSource base = ModelSources.open(quantizedFile.toString())) {
            Map<String, float[]> wrongType = new LinkedHashMap<>();
            wrongType.put(UNTOUCHED, new float[(int) tensor(base, UNTOUCHED).numElements()]);
            try {
                TrainedSource.of(base, wrongType);
                throw new AssertionError("a Q8_0 tensor was accepted as a trained F32 tensor");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("not F32"));
            }
        }
        try (ModelSource base = ModelSources.open(file.toString())) {
            Map<String, float[]> shortValues = new LinkedHashMap<>();
            shortValues.put(TRAINED, new float[3]);
            try {
                TrainedSource.of(base, shortValues);
                throw new AssertionError("a short value array was accepted");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("holds"));
            }
        }
    }

    @Test
    public void theRegionViewIsRefusedRatherThanServingUntrainedBytes() throws IOException {
        try (ModelSource base = ModelSources.open(file.toString());
             TrainedSource source = TrainedSource.of(base, new LinkedHashMap<>())) {
            try {
                source.openRegions(TensorReadMode.MMAP);
                throw new AssertionError("the region view of a trained source was served");
            } catch (UnsupportedOperationException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("untrained"));
            }
        }
    }
}
