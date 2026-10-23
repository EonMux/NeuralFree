package com.eonmux.neuralfree.io;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.norm.BatchNorm1d;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ModelSerializer} checkpoints carry the whole module state (parameters and persistent buffers such as
 * BatchNorm running statistics), refuse entries whose shape differs from the model's, and restore the optimizer's
 * state and step count. {@link ModelSerializer#inspect(Path)} reads long counts, scalar shapes and escaped strings
 * and refuses malformed numbers and strings. Runs on the host reference backend.
 */
public class ModelSerializerStateTest {
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
    public void batchNormRunningStatisticsSurviveASaveAndLoad() throws IOException {
        Path path = folder.getRoot().toPath().resolve("bn.zip");
        try (BatchNorm1d source = batchNorm(3); BatchNorm1d target = batchNorm(3)) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor in = matrix(new float[]{1f, 2f, 3f, 3f, 6f, 9f}, 2, 3);
                 Tensor out = source.forward(in)) {
                assertEquals(2, out.shape().dim(0));
            }
            Map<String, Object> state = source.stateDict();
            state.put("weight", new float[]{0.5f, 1.5f, 2.5f});
            source.loadStateDict(state, true);

            ModelSerializer.save(source, path);
            ModelSerializer.load(path, target, host);

            assertEquals(source.stateDict().keySet(), target.stateDict().keySet());
            assertArrayEquals(source.runningMean().toFloatArray(), target.runningMean().toFloatArray(), 0f);
            assertArrayEquals(source.runningVar().toFloatArray(), target.runningVar().toFloatArray(), 0f);
            assertArrayEquals(new float[]{0.5f, 1.5f, 2.5f}, target.weight().tensor().toFloatArray(), 0f);
        }
    }

    @Test
    public void anEntryOfAnotherShapeIsRefusedBeforeAnythingIsLoaded() throws IOException {
        Path path = folder.getRoot().toPath().resolve("linear.zip");
        try (Linear source = linear(2, 3, 1L); Linear target = linear(3, 2, 2L)) {
            ModelSerializer.save(source, path);
            float[] before = target.getParameter("weight").tensor().toFloatArray();

            IOException e = assertThrows(IOException.class, () -> ModelSerializer.load(path, target, host));

            assertTrue(e.getMessage(), e.getMessage().contains("'weight'"));
            assertArrayEquals(before, target.getParameter("weight").tensor().toFloatArray(), 0f);
        }
    }

    @Test
    public void optimizerStateAndStepCountAreRestored() throws IOException {
        Path path = folder.getRoot().toPath().resolve("adam.zip");
        float[] moment = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f};
        try (Linear model = linear(2, 3, 3L); Linear restored = linear(2, 3, 4L)) {
            Adam adam = new Adam(model.parameters(), 0.01f);
            Adam restoredAdam = new Adam(restored.parameters(), 0.01f);
            try {
                assertOptimizerRoundTrip(path, moment, model, restored, adam, restoredAdam);
            } finally {
                adam.close();
                restoredAdam.close();
            }
        }
    }

    /** Gives {@code adam} a step count and a first moment, saves and loads it, and checks {@code restoredAdam}. */
    private void assertOptimizerRoundTrip(Path path, float[] moment, Linear model, Linear restored, Adam adam,
                                          Adam restoredAdam) throws IOException {
        Map<String, Object> state = new HashMap<>();
        state.put(Optimizer.STEP_COUNT_KEY, 3);
        state.put("moment_0", moment);
        adam.loadStateDict(state);

        ModelSerializer.save(model, adam, path);
        ModelSerializer.load(path, restored, restoredAdam, host);

        assertEquals(3, restoredAdam.stepCount());
        Map<String, Object> restoredState = restoredAdam.stateDict();
        try {
            assertArrayEquals(moment, ((Tensor) restoredState.get("moment_0")).toFloatArray(), 0f);
        } finally {
            Optimizer.closeStateDict(restoredState);
        }
        assertArrayEquals(model.getParameter("weight").tensor().toFloatArray(),
                restored.getParameter("weight").tensor().toFloatArray(), 0f);
    }

    @Test
    public void aCheckpointWithoutTheBuffersStillLoadsTheParameters() throws IOException {
        Path path = folder.getRoot().toPath().resolve("params-only.zip");
        try (AffineOnly source = new AffineOnly(host, new float[]{2f, 3f}, new float[]{0.25f, 0.75f});
             BatchNorm1d target = batchNorm(2)) {
            ModelSerializer.save(source, path);

            ModelSerializer.load(path, target, host);

            assertArrayEquals(new float[]{2f, 3f}, target.weight().tensor().toFloatArray(), 0f);
            assertArrayEquals(new float[]{0.25f, 0.75f}, target.bias().tensor().toFloatArray(), 0f);
            assertArrayEquals(new float[]{0f, 0f}, target.runningMean().toFloatArray(), 0f);
            assertArrayEquals(new float[]{1f, 1f}, target.runningVar().toFloatArray(), 0f);
        }
    }

    @Test
    public void inspectReadsLongElementCountsAndScalarShapes() throws IOException {
        Path path = folder.getRoot().toPath().resolve("meta_long.zip");
        writeMeta(path, "{\n"
            + "  \"format_version\": 1,\n"
            + "  \"module_type\": \"Big\",\n"
            + "  \"parameter_count\": 3000000001,\n"
            + "  \"num_parameters\": 2,\n"
            + "  \"timestamp\": \"t\",\n"
            + "  \"parameters\": {\n"
            + "    \"table\": {\n"
            + "      \"shape\": [60000, 50000],\n"
            + "      \"dtype\": \"FLOAT32\",\n"
            + "      \"num_elements\": 3000000000\n"
            + "    },\n"
            + "    \"scale\": {\n"
            + "      \"shape\": [],\n"
            + "      \"dtype\": \"FLOAT32\",\n"
            + "      \"num_elements\": 1\n"
            + "    }\n"
            + "  }\n"
            + "}\n");

        ModelSerializer.CheckpointInfo info = ModelSerializer.inspect(path);

        assertEquals(3000000001L, info.parameterCount());
        assertEquals(2, info.numParameters());
        assertArrayEquals(new int[]{60000, 50000}, info.parameters().get("table").shape());
        assertEquals(3000000000L, info.parameters().get("table").numElements());
        assertArrayEquals(new int[0], info.parameters().get("scale").shape());
        assertEquals(1L, info.parameters().get("scale").numElements());
    }

    @Test
    public void inspectRefusesMalformedNumbers() throws IOException {
        Path notANumber = folder.getRoot().toPath().resolve("meta_nan.zip");
        writeMeta(notANumber, "{\n  \"format_version\": one,\n  \"module_type\": \"M\"\n}\n");
        IOException e = assertThrows(IOException.class, () -> ModelSerializer.inspect(notANumber));
        assertTrue(e.getMessage(), e.getMessage().contains("format_version"));

        Path outOfRange = folder.getRoot().toPath().resolve("meta_range.zip");
        writeMeta(outOfRange, "{\n  \"format_version\": 1,\n  \"num_parameters\": 4294967296\n}\n");
        e = assertThrows(IOException.class, () -> ModelSerializer.inspect(outOfRange));
        assertTrue(e.getMessage(), e.getMessage().contains("num_parameters"));

        Path badShape = folder.getRoot().toPath().resolve("meta_shape.zip");
        writeMeta(badShape, "{\n  \"format_version\": 1,\n  \"parameters\": {\n"
            + "    \"w\": {\n      \"shape\": [2, x],\n      \"num_elements\": 4\n    }\n  }\n}\n");
        e = assertThrows(IOException.class, () -> ModelSerializer.inspect(badShape));
        assertTrue(e.getMessage(), e.getMessage().contains("shape"));
    }

    @Test
    public void inspectUnescapesStringValues() throws IOException {
        Path path = folder.getRoot().toPath().resolve("meta_escaped.zip");
        writeMeta(path, "{\n  \"format_version\": 1,\n  \"module_type\": \"Wrap\\\"per\\\\\\u0041\",\n"
            + "  \"timestamp\": \"2026-01-01T00:00:00Z\",\n  \"parameters\": {\n"
            + "    \"w\": {\n      \"shape\": [2],\n      \"dtype\": \"FLOAT32\",\n      \"num_elements\": 2\n    }\n"
            + "  }\n}\n");

        ModelSerializer.CheckpointInfo info = ModelSerializer.inspect(path);

        assertEquals("Wrap\"per\\A", info.moduleType());
        assertEquals("2026-01-01T00:00:00Z", info.timestamp());
        assertEquals("FLOAT32", info.parameters().get("w").dtype());
    }

    @Test
    public void inspectRefusesMalformedStringValues() throws IOException {
        Path danglingEscape = folder.getRoot().toPath().resolve("meta_dangling.zip");
        writeMeta(danglingEscape, "{\n  \"format_version\": 1,\n  \"module_type\": \"a\\");
        IOException e = assertThrows(IOException.class, () -> ModelSerializer.inspect(danglingEscape));
        assertTrue(e.getMessage(), e.getMessage().contains("module_type"));

        Path unterminated = folder.getRoot().toPath().resolve("meta_unterminated.zip");
        writeMeta(unterminated, "{\n  \"format_version\": 1,\n  \"timestamp\": \"2026-01-01\n}\n");
        e = assertThrows(IOException.class, () -> ModelSerializer.inspect(unterminated));
        assertTrue(e.getMessage(), e.getMessage().contains("timestamp"));

        Path unknownEscape = folder.getRoot().toPath().resolve("meta_unknown_escape.zip");
        writeMeta(unknownEscape, "{\n  \"format_version\": 1,\n  \"module_type\": \"a\\qb\"\n}\n");
        e = assertThrows(IOException.class, () -> ModelSerializer.inspect(unknownEscape));
        assertTrue(e.getMessage(), e.getMessage().contains("unknown escape"));

        Path unterminatedDtype = folder.getRoot().toPath().resolve("meta_dtype.zip");
        writeMeta(unterminatedDtype, "{\n  \"format_version\": 1,\n  \"parameters\": {\n"
            + "    \"w\": {\n      \"shape\": [2],\n      \"dtype\": \"FLOAT32\n    }\n  }\n}\n");
        e = assertThrows(IOException.class, () -> ModelSerializer.inspect(unterminatedDtype));
        assertTrue(e.getMessage(), e.getMessage().contains("dtype"));
    }

    /** Writes a checkpoint archive holding only the metadata entry {@code model_meta.json} with {@code json}. */
    private static void writeMeta(Path path, String json) throws IOException {
        try (OutputStream file = Files.newOutputStream(path); ZipOutputStream zip = new ZipOutputStream(file)) {
            zip.putNextEntry(new ZipEntry("model_meta.json"));
            zip.write(json.getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
        }
    }

    /** The parameters {@code weight} and {@code bias} of a BatchNorm1d without its buffers. */
    private static final class AffineOnly extends Module {
        AffineOnly(ComputeBackend backend, float[] weight, float[] bias) {
            registerParameter("weight", new Parameter(Tensor.fromArray(backend, weight), "weight"));
            registerParameter("bias", new Parameter(Tensor.fromArray(backend, bias), "bias"));
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    private BatchNorm1d batchNorm(int features) {
        BatchNorm1d bn = new BatchNorm1d(features);
        bn.initialize(host);
        return bn;
    }

    private Linear linear(int in, int out, long seed) {
        Linear layer = new Linear(in, out);
        layer.initialize(host, new Random(seed));
        return layer;
    }

    private Tensor matrix(float[] data, int rows, int cols) {
        try (Tensor flat = Tensor.fromArray(host, data)) {
            return flat.reshape(Shape.matrix(rows, cols));
        }
    }
}
