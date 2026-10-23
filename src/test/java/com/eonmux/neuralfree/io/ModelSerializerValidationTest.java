package com.eonmux.neuralfree.io;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.optim.SGD;
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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ModelSerializer#load} refuses, with an {@link IOException} and without changing the model, a file that is
 * not a checkpoint archive, an archive without the state asked for, model state of another model, and tensor records
 * whose lengths are corrupt, before allocating buffers of the sizes they name. Runs on the host reference backend.
 */
public class ModelSerializerValidationTest {
    private static final String MODEL_ENTRY = "model_state.bin";

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
    public void aFileThatIsNotAZipArchiveIsRefused() throws IOException {
        Path path = folder.getRoot().toPath().resolve("not-a-zip.zip");
        Files.write(path, "not a checkpoint".getBytes(StandardCharsets.UTF_8));
        try (Linear model = linear(1L)) {
            String message = refusal(path, model);
            assertTrue(message, message.contains("not a zip archive"));
            assertTrue(message, assertThrows(IOException.class, () -> ModelSerializer.inspect(path)).getMessage()
                .contains("not a zip archive"));
        }
    }

    @Test
    public void anArchiveWithoutModelStateIsRefused() throws IOException {
        Path path = folder.getRoot().toPath().resolve("meta-only.zip");
        try (OutputStream file = Files.newOutputStream(path); ZipOutputStream zip = new ZipOutputStream(file)) {
            zip.putNextEntry(new ZipEntry("model_meta.json"));
            zip.write("{}".getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
        }
        try (Linear model = linear(2L)) {
            String message = refusal(path, model);
            assertTrue(message, message.contains(MODEL_ENTRY));
        }
    }

    @Test
    public void anArchiveWithoutOptimizerStateIsRefusedWhenAnOptimizerIsGiven() throws IOException {
        Path path = folder.getRoot().toPath().resolve("model-only.zip");
        try (Linear source = linear(3L); Linear target = linear(4L)) {
            ModelSerializer.save(source, path);
            float[] before = weight(target);
            SGD sgd = new SGD(target.parameters(), 0.1f);
            try {
                IOException e = assertThrows(IOException.class, () -> ModelSerializer.load(path, target, sgd, host));
                assertTrue(e.getMessage(), e.getMessage().contains("optimizer_state.bin"));
                assertArrayEquals(before, weight(target), 0f);
            } finally {
                sgd.close();
            }
        }
    }

    @Test
    public void theCheckpointOfAnotherModelIsRefused() throws IOException {
        Path path = folder.getRoot().toPath().resolve("scale.zip");
        try (ScaleOnly source = new ScaleOnly(host); Linear target = linear(5L)) {
            ModelSerializer.save(source, path);
            float[] before = weight(target);

            String message = refusal(path, target);

            assertTrue(message, message.contains("missing parameters [weight, bias]"));
            assertTrue(message, message.contains("[scale]"));
            assertArrayEquals(before, weight(target), 0f);
        }
    }

    @Test
    public void corruptTensorRecordLengthsAreRefusedWithAnIOException() throws IOException {
        try (Linear model = linear(6L)) {
            assertRefused(model, "name-length", "name length", record(0x7FFF_FFF0, 0, 0));
            assertRefused(model, "rank", "rank", record(6, 0x1FFF_FFFF, 0));
            assertRefused(model, "element-count", "needs", record(6, 2, 0x1FFF_FFFF));
            assertRefused(model, "negative-dimension", "negative dimension",
                record(6, 2, 0), "weight".getBytes(StandardCharsets.UTF_8), ints(-1, 0));
            assertRefused(model, "shape-mismatch", "does not hold",
                record(6, 2, 5), "weight".getBytes(StandardCharsets.UTF_8), ints(3, 2), new byte[5 * Float.BYTES]);
            assertRefused(model, "truncated", "needs", record(6, 2, 6), "weight".getBytes(StandardCharsets.UTF_8),
                ints(3, 2), new byte[3 * Float.BYTES]);
        }
    }

    @Test
    public void aDuplicatedEntryIsRefused() throws IOException {
        Path path = folder.getRoot().toPath().resolve("duplicate.zip");
        try (Linear model = linear(7L)) {
            byte[] bias = concat(record(4, 1, 3), "bias".getBytes(StandardCharsets.UTF_8), ints(3),
                new byte[3 * Float.BYTES]);
            writeModelState(path, 2, concat(bias, bias));
            String message = refusal(path, model);
            assertTrue(message, message.contains("appears twice"));
        }
    }

    /** Writes a model state of one record made of {@code parts} and checks the load fails naming {@code expected}. */
    private void assertRefused(Linear model, String file, String expected, byte[]... parts) throws IOException {
        Path path = folder.getRoot().toPath().resolve(file + ".zip");
        writeModelState(path, 1, concat(parts));
        float[] before = weight(model);
        String message = refusal(path, model);
        assertTrue(message, message.contains(expected));
        assertArrayEquals(before, weight(model), 0f);
    }

    private String refusal(Path path, Module model) {
        return assertThrows(IOException.class, () -> ModelSerializer.load(path, model, host)).getMessage();
    }

    /** Writes an archive whose {@code model_state.bin} is a format-1 header for {@code count} records, then body. */
    private static void writeModelState(Path path, int count, byte[] body) throws IOException {
        try (OutputStream file = Files.newOutputStream(path); ZipOutputStream zip = new ZipOutputStream(file)) {
            zip.putNextEntry(new ZipEntry(MODEL_ENTRY));
            zip.write(ints(1, count));
            zip.write(body);
            zip.closeEntry();
        }
    }

    /** A tensor record header: name length, rank, element count. */
    private static byte[] record(int nameLength, int rank, int elements) {
        return ints(nameLength, rank, elements);
    }

    private static byte[] ints(int... values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int value : values) {
            buffer.putInt(value);
        }
        return buffer.array();
    }

    private static byte[] concat(byte[]... parts) {
        int length = 0;
        for (byte[] part : parts) {
            length += part.length;
        }
        ByteBuffer buffer = ByteBuffer.allocate(length);
        for (byte[] part : parts) {
            buffer.put(part);
        }
        return buffer.array();
    }

    private static float[] weight(Linear model) {
        return model.getParameter("weight").tensor().toFloatArray();
    }

    private Linear linear(long seed) {
        Linear layer = new Linear(2, 3);
        layer.initialize(host, new Random(seed));
        return layer;
    }

    /** A module whose only state is the parameter {@code scale}. */
    private static final class ScaleOnly extends Module {
        ScaleOnly(ComputeBackend backend) {
            registerParameter("scale", new Parameter(Tensor.fromArray(backend, new float[]{2f}), "scale"));
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }
}
