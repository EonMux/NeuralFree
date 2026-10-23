package com.eonmux.neuralfree.io;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The optimizer entry of {@link ModelSerializer} checkpoints: format 2 keeps state tensor shapes and exact integer
 * step counts, format-1 files (all FLOAT32 tensor records) still load, and state that does not fit the optimizer or
 * a corrupt record is refused with an {@link IOException}. Runs on the host reference backend.
 */
public class ModelSerializerOptimizerStateTest {
    private static final String MODEL_ENTRY = "model_state.bin";
    private static final String OPTIMIZER_ENTRY = "optimizer_state.bin";
    private static final String TRAINING_ENTRY = "training_state.json";
    /** 2^24 + 1, the smallest positive integer a float cannot hold. */
    private static final int LARGE_STEP = 16_777_217;

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
    public void formatTwoKeepsShapesAndAnExactStepCount() throws IOException {
        Path saved = folder.getRoot().toPath().resolve("sgd.zip");
        Path optimizerOnly = folder.getRoot().toPath().resolve("sgd-optimizer-only.zip");
        try (Linear model = linear(1L); Linear restored = linear(2L)) {
            SGD sgd = new SGD(model.parameters(), 0.1f, 0.9f, 0f, false);
            SGD restoredSgd = new SGD(restored.parameters(), 0.1f, 0.9f, 0f, false);
            try {
                Map<String, Object> state = new HashMap<>();
                state.put(Optimizer.STEP_COUNT_KEY, LARGE_STEP);
                state.put("velocity_0", new float[]{1f, 2f, 3f, 4f, 5f, 6f});
                state.put("velocity_1", new float[]{-1f, -2f, -3f});
                sgd.loadStateDict(state);

                ModelSerializer.save(model, sgd, saved);
                copyWithout(saved, optimizerOnly, TRAINING_ENTRY);
                assertEquals(2, optimizerFormat(saved));
                ModelSerializer.load(optimizerOnly, restored, restoredSgd, host);

                assertEquals(LARGE_STEP, restoredSgd.stepCount());
                Map<String, Object> loaded = restoredSgd.stateDict();
                try {
                    Tensor velocity = (Tensor) loaded.get("velocity_0");
                    assertEquals(model.getParameter("weight").shape(), velocity.shape());
                    assertArrayEquals(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, velocity.toFloatArray(), 0f);
                    assertArrayEquals(new float[]{-1f, -2f, -3f},
                            ((Tensor) loaded.get("velocity_1")).toFloatArray(), 0f);
                } finally {
                    Optimizer.closeStateDict(loaded);
                }
            } finally {
                sgd.close();
                restoredSgd.close();
            }
        }
    }

    @Test
    public void formatOneCheckpointsStillLoad() throws IOException {
        Path path = folder.getRoot().toPath().resolve("legacy.zip");
        try (Linear model = linear(3L)) {
            int[] weightShape = model.getParameter("weight").shape().dims();
            float[] moment = {0.5f, -0.5f, 0.25f, -0.25f, 0.125f, -0.125f};
            writeOptimizerEntry(path, model, 1, out -> {
                writeTensorRecord(out, Optimizer.STEP_COUNT_KEY, new int[0], new float[]{5f});
                writeTensorRecord(out, "beta1", new int[0], new float[]{0.5f});
                writeTensorRecord(out, "moment_0", weightShape, moment);
            }, 3);
            Adam adam = new Adam(model.parameters(), 0.01f);
            try {
                ModelSerializer.load(path, model, adam, host);

                assertEquals(5, adam.stepCount());
                assertEquals(0.9f, adam.beta1(), 0f);
                Map<String, Object> state = adam.stateDict();
                try {
                    assertArrayEquals(moment, ((Tensor) state.get("moment_0")).toFloatArray(), 0f);
                } finally {
                    Optimizer.closeStateDict(state);
                }
            } finally {
                adam.close();
            }
        }
    }

    @Test
    public void stateThatDoesNotFitTheOptimizerIsRefused() throws IOException {
        Path path = folder.getRoot().toPath().resolve("momentum.zip");
        try (Linear model = linear(4L); Linear restored = linear(5L)) {
            SGD momentum = new SGD(model.parameters(), 0.1f, 0.9f, 0f, false);
            SGD plain = new SGD(restored.parameters(), 0.1f);
            try {
                ModelSerializer.save(model, momentum, path);
                float[] weightBefore = restored.getParameter("weight").tensor().toFloatArray();
                IOException e = assertThrows(IOException.class,
                        () -> ModelSerializer.load(path, restored, plain, host));
                assertTrue(e.getMessage(), e.getMessage().contains("'velocity_0'"));
                assertTrue(e.getCause() instanceof IllegalArgumentException);
                assertEquals(0, plain.stepCount());
                assertArrayEquals(weightBefore, restored.getParameter("weight").tensor().toFloatArray(), 0f);
            } finally {
                momentum.close();
                plain.close();
            }
        }

        Path wrongShape = folder.getRoot().toPath().resolve("wrong-shape.zip");
        try (Linear model = linear(6L)) {
            writeOptimizerEntry(wrongShape, model, 1,
                    out -> writeTensorRecord(out, "moment_0", new int[]{6}, new float[6]), 1);
            Adam adam = new Adam(model.parameters(), 0.01f);
            try {
                IOException e = assertThrows(IOException.class,
                        () -> ModelSerializer.load(wrongShape, model, adam, host));
                assertTrue(e.getMessage(), e.getMessage().contains("'moment_0'")
                        && e.getMessage().contains("shape"));
            } finally {
                adam.close();
            }
        }
    }

    @Test
    public void corruptOptimizerRecordsAreRefused() throws IOException {
        try (Linear model = linear(7L)) {
            Path unknownKind = folder.getRoot().toPath().resolve("unknown-kind.zip");
            writeOptimizerEntry(unknownKind, model, 2, out -> out.write(intBytes(9)), 1);
            Path badBoolean = folder.getRoot().toPath().resolve("bad-boolean.zip");
            writeOptimizerEntry(badBoolean, model, 2, out -> writeScalarRecord(out, 3, "nesterov", 2L), 1);
            Path duplicate = folder.getRoot().toPath().resolve("duplicate.zip");
            writeOptimizerEntry(duplicate, model, 2, out -> {
                writeScalarRecord(out, 1, Optimizer.STEP_COUNT_KEY, 1L);
                writeScalarRecord(out, 1, Optimizer.STEP_COUNT_KEY, 2L);
            }, 2);
            Path newer = folder.getRoot().toPath().resolve("newer.zip");
            writeOptimizerEntry(newer, model, 3, out -> { }, 0);

            SGD sgd = new SGD(model.parameters(), 0.1f, 0f, 0f, false);
            try {
                assertTrue(loadMessage(unknownKind, model, sgd).contains("unknown kind 9"));
                assertTrue(loadMessage(badBoolean, model, sgd).contains("'nesterov'"));
                assertTrue(loadMessage(duplicate, model, sgd).contains("appears twice"));
                assertTrue(loadMessage(newer, model, sgd).contains("version: 3"));
                assertEquals(0, sgd.stepCount());
            } finally {
                sgd.close();
            }
        }
    }

    private String loadMessage(Path path, Linear model, Optimizer optimizer) {
        return assertThrows(IOException.class, () -> ModelSerializer.load(path, model, optimizer, host))
                .getMessage();
    }

    private Linear linear(long seed) {
        Linear layer = new Linear(2, 3);
        layer.initialize(host, new Random(seed));
        return layer;
    }

    /** Writes the body of an optimizer entry. */
    private interface RecordWriter {
        void write(OutputStream out) throws IOException;
    }

    /**
     * Writes an archive holding the parameters of {@code model} as {@code model_state.bin} and an
     * {@code optimizer_state.bin} with the given header and records.
     */
    private static void writeOptimizerEntry(Path path, Linear model, int version, RecordWriter records, int count)
            throws IOException {
        try (OutputStream file = Files.newOutputStream(path); ZipOutputStream zip = new ZipOutputStream(file)) {
            Map<String, Parameter> parameters = model.namedParameters();
            zip.putNextEntry(new ZipEntry(MODEL_ENTRY));
            zip.write(intBytes(1));
            zip.write(intBytes(parameters.size()));
            for (Map.Entry<String, Parameter> parameter : parameters.entrySet()) {
                writeTensorRecord(zip, parameter.getKey(), parameter.getValue().shape().dims(),
                        parameter.getValue().tensor().toFloatArray());
            }
            zip.closeEntry();
            zip.putNextEntry(new ZipEntry(OPTIMIZER_ENTRY));
            zip.write(intBytes(version));
            zip.write(intBytes(count));
            records.write(zip);
            zip.closeEntry();
        }
    }

    /** A tensor record: name length, rank, element count, UTF-8 name, dims, little-endian floats. */
    private static void writeTensorRecord(OutputStream out, String name, int[] shape, float[] data)
            throws IOException {
        byte[] nameBytes = name.getBytes(StandardCharsets.UTF_8);
        ByteBuffer record = ByteBuffer.allocate(12 + nameBytes.length + 4 * shape.length + 4 * data.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        record.putInt(nameBytes.length).putInt(shape.length).putInt(data.length).put(nameBytes);
        for (int dim : shape) {
            record.putInt(dim);
        }
        for (float value : data) {
            record.putFloat(value);
        }
        out.write(record.array());
    }

    /** A format-2 scalar record: kind, UTF-8 name length, name, 8-byte little-endian payload. */
    private static void writeScalarRecord(OutputStream out, int kind, String name, long payload) throws IOException {
        byte[] nameBytes = name.getBytes(StandardCharsets.UTF_8);
        out.write(ByteBuffer.allocate(16 + nameBytes.length).order(ByteOrder.LITTLE_ENDIAN)
                .putInt(kind).putInt(nameBytes.length).put(nameBytes).putLong(payload).array());
    }

    private static byte[] intBytes(int value) {
        return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array();
    }

    /** The format version in the header of the archive's optimizer entry. */
    private static int optimizerFormat(Path path) throws IOException {
        try (InputStream file = Files.newInputStream(path); ZipInputStream zip = new ZipInputStream(file)) {
            ZipEntry entry;
            while ((entry = zip.getNextEntry()) != null) {
                if (OPTIMIZER_ENTRY.equals(entry.getName())) {
                    byte[] header = new byte[4];
                    int read = 0;
                    while (read < header.length) {
                        int n = zip.read(header, read, header.length - read);
                        if (n < 0) {
                            throw new IOException("optimizer entry ends inside its header");
                        }
                        read += n;
                    }
                    return ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN).getInt();
                }
            }
        }
        throw new IOException("no " + OPTIMIZER_ENTRY + " in " + path);
    }

    /** Copies every entry of the archive {@code from} except {@code skipped} into a new archive {@code to}. */
    private static void copyWithout(Path from, Path to, String skipped) throws IOException {
        try (InputStream in = Files.newInputStream(from); ZipInputStream source = new ZipInputStream(in);
             OutputStream out = Files.newOutputStream(to); ZipOutputStream target = new ZipOutputStream(out)) {
            ZipEntry entry;
            while ((entry = source.getNextEntry()) != null) {
                if (skipped.equals(entry.getName())) {
                    continue;
                }
                target.putNextEntry(new ZipEntry(entry.getName()));
                ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                byte[] buffer = new byte[8192];
                int n;
                while ((n = source.read(buffer)) > 0) {
                    bytes.write(buffer, 0, n);
                }
                target.write(bytes.toByteArray());
                target.closeEntry();
            }
        }
    }
}
