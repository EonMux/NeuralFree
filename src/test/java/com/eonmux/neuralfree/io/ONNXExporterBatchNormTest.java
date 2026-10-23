package com.eonmux.neuralfree.io;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.norm.BatchNorm1d;
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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertTrue;

/**
 * {@link ONNXExporter} exports BatchNorm1d as BatchNormalization with its four statistics inputs, also without affine
 * parameters, where the required scale and B inputs are constant initializers. Runs on the host reference backend.
 */
public class ONNXExporterBatchNormTest {
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
    public void batchNormWithoutAffineParametersExportsConstantScaleAndBias() throws IOException {
        assertExportsBatchNormalization(false);
    }

    @Test
    public void batchNormWithAffineParametersExportsThem() throws IOException {
        assertExportsBatchNormalization(true);
    }

    private void assertExportsBatchNormalization(boolean affine) throws IOException {
        Path path = folder.getRoot().toPath().resolve("bn-" + affine + ".onnx");
        try (BatchNorm1d layer = new BatchNorm1d(3, 1e-5f, 0.1f, affine);
             Tensor sample = Tensor.zeros(host, Shape.matrix(2, 3))) {
            layer.initialize(host);

            new ONNXExporter("BatchNormModel").export(layer, sample, path);

            String content = new String(Files.readAllBytes(path), StandardCharsets.ISO_8859_1);
            assertTrue(content.contains("BatchNormalization"));
            for (String input : new String[]{"weight", "bias", "running_mean", "running_var"}) {
                assertTrue("missing initializer " + input, content.contains("BatchNorm1d_" + input + "_"));
            }
        }
    }
}
