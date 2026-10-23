package com.eonmux.neuralfree.io;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.WeightInit;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.nn.activations.Sigmoid;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;

import org.junit.Test;
import org.junit.Before;
import org.junit.After;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

public class ModelIOTest {
    
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private Path tempDir;
    private boolean gpuAvailable = false;
    
    @Before
    public void setUp() throws IOException {
        tempDir = Files.createTempDirectory("model_io_test");
        
        try {
            hwAccel = new HWAcceleration();
            
            if (hwAccel.getCLDevices().isEmpty()) {
                System.out.println("No OpenCL GPU devices available, skipping tests");
                return;
            }
            
            CLDevice clDevice = hwAccel.getCLDevices().get(0);
            backend = new OclComputeBackend(clDevice);
            gpuAvailable = true;
            
            System.out.println("GPU available for tests: " + backend.capabilities().getDeviceName());
        } catch (Exception e) {
            System.out.println("GPU initialization failed, skipping tests: " + e.getMessage());
            gpuAvailable = false;
        }
    }
    
    @After
    public void tearDown() throws IOException {
        if (backend != null) {
            backend.close();
            backend = null;
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
            hwAccel = null;
        }
        if (tempDir != null) {
            deleteDirectory(tempDir.toFile());
        }
    }
    
    private void assumeGpuAvailable() {
        if (!gpuAvailable) {
            System.out.println("Skipping test - no GPU available");
            org.junit.Assume.assumeTrue("GPU not available", false);
        }
    }
    
    private void deleteDirectory(File dir) {
        if (dir.isDirectory()) {
            File[] files = dir.listFiles();
            if (files != null) {
                for (File file : files) {
                    deleteDirectory(file);
                }
            }
        }
        dir.delete();
    }

    @Test
    public void testWeightInitCreation() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend);
        assertNotNull(init);
        assertEquals(backend, init.backend());
    }

    @Test
    public void testWeightInitWithSeed() {
        assumeGpuAvailable();
        WeightInit init1 = new WeightInit(backend, 42L);
        WeightInit init2 = new WeightInit(backend, 42L);
        
        Tensor t1 = init1.normal(Shape.matrix(3, 4));
        Tensor t2 = init2.normal(Shape.matrix(3, 4));
        
        float[] data1 = t1.toFloatArray();
        float[] data2 = t2.toFloatArray();
        
        assertArrayEquals(data1, data2, 1e-6f);
    }

    @Test
    public void testXavierInitialization() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend, 42L);
        Tensor t = init.xavier(Shape.matrix(100, 50));
        
        assertEquals(2, t.shape().rank());
        assertEquals(100, t.shape().dim(0));
        assertEquals(50, t.shape().dim(1));
        
        float[] data = t.toFloatArray();
        float mean = computeMean(data);
        float std = computeStd(data, mean);
        
        float expectedStd = (float) Math.sqrt(2.0 / (100 + 50));
        assertTrue("Xavier std should be close to expected", 
            Math.abs(std - expectedStd) < 0.1);
    }

    @Test
    public void testXavierUniform() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend, 42L);
        Tensor t = init.xavier(Shape.matrix(100, 50), WeightInit.XavierMode.UNIFORM);
        
        assertEquals(2, t.shape().rank());
        
        float[] data = t.toFloatArray();
        for (float val : data) {
            double limit = Math.sqrt(6.0 / (100 + 50));
            assertTrue("Xavier uniform values should be in range", 
                val >= -limit - 0.01 && val <= limit + 0.01);
        }
    }

    @Test
    public void testKaimingInitialization() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend, 42L);
        Tensor t = init.kaiming(Shape.matrix(100, 50));
        
        assertEquals(2, t.shape().rank());
        assertEquals(100, t.shape().dim(0));
        assertEquals(50, t.shape().dim(1));
        
        float[] data = t.toFloatArray();
        float mean = computeMean(data);
        float std = computeStd(data, mean);
        
        float gain = (float) Math.sqrt(2.0);
        float expectedStd = gain / (float) Math.sqrt(50);
        assertTrue("Kaiming std should be close to expected", 
            Math.abs(std - expectedStd) < 0.15);
    }

    @Test
    public void testKaimingFanOut() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend, 42L);
        Tensor t = init.kaiming(Shape.matrix(100, 50), 
            WeightInit.KaimingMode.FAN_OUT, WeightInit.Nonlinearity.RELU);
        
        assertNotNull(t);
        assertEquals(5000, t.numElements());
    }

    @Test
    public void testNormalInitialization() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend, 42L);
        Tensor t = init.normal(Shape.matrix(1000, 1), 0.0, 2.0);
        
        float[] data = t.toFloatArray();
        float mean = computeMean(data);
        float std = computeStd(data, mean);
        
        assertTrue("Normal mean should be close to 0", Math.abs(mean) < 0.2);
        assertTrue("Normal std should be close to 2", Math.abs(std - 2.0) < 0.2);
    }

    @Test
    public void testUniformInitialization() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend, 42L);
        Tensor t = init.uniform(Shape.vector(1000), -1.0, 3.0);
        
        float[] data = t.toFloatArray();
        for (float val : data) {
            assertTrue("Uniform values should be in range", val >= -1.0 && val <= 3.0);
        }
        
        float mean = computeMean(data);
        assertTrue("Uniform mean should be close to 1.0", Math.abs(mean - 1.0) < 0.1);
    }

    @Test
    public void testConstantInitialization() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend);
        Tensor t = init.constant(Shape.matrix(3, 4), 5.0f);
        
        float[] data = t.toFloatArray();
        for (float val : data) {
            assertEquals(5.0f, val, 1e-6f);
        }
    }

    @Test
    public void testZerosInitialization() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend);
        Tensor t = init.zeros(Shape.matrix(3, 4));
        
        float[] data = t.toFloatArray();
        for (float val : data) {
            assertEquals(0.0f, val, 1e-6f);
        }
    }

    @Test
    public void testOnesInitialization() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend);
        Tensor t = init.ones(Shape.matrix(3, 4));
        
        float[] data = t.toFloatArray();
        for (float val : data) {
            assertEquals(1.0f, val, 1e-6f);
        }
    }

    @Test
    public void testOrthogonalInitialization() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend, 42L);
        Tensor t = init.orthogonal(Shape.matrix(10, 5));
        
        assertEquals(2, t.shape().rank());
        assertEquals(10, t.shape().dim(0));
        assertEquals(5, t.shape().dim(1));
    }

    @Test
    public void testOrthogonalWithGain() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend, 42L);
        Tensor t = init.orthogonal(Shape.matrix(10, 5), 2.0);
        
        assertNotNull(t);
        assertEquals(50, t.numElements());
    }

    @Test
    public void testTruncatedNormal() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend, 42L);
        Tensor t = init.truncatedNormal(Shape.vector(1000), 0.0, 1.0, -2.0, 2.0);
        
        float[] data = t.toFloatArray();
        for (float val : data) {
            assertTrue("Truncated normal values should be in range", val >= -2.0 && val <= 2.0);
        }
    }

    @Test
    public void testSparseInitialization() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend, 42L);
        Tensor t = init.sparse(Shape.matrix(10, 10), 0.9, 1.0);
        
        float[] data = t.toFloatArray();
        int zeros = 0;
        for (float val : data) {
            if (Math.abs(val) < 1e-6f) zeros++;
        }
        
        float sparsity = (float) zeros / data.length;
        assertTrue("Sparse initialization should have ~90% zeros", sparsity >= 0.85);
    }

    @Test
    public void testEyeInitialization() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend);
        Tensor t = init.eye(5);
        
        float[] data = t.toFloatArray();
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                float expected = (i == j) ? 1.0f : 0.0f;
                assertEquals(expected, data[i * 5 + j], 1e-6f);
            }
        }
    }

    @Test
    public void testWeightInitInvalidShape() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend);
        try {
            init.xavier(Shape.vector(0));
            fail("Should throw for empty shape");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("at least one element"));
        }
    }

    @Test
    public void testWeightInitNegativeStd() {
        assumeGpuAvailable();
        WeightInit init = new WeightInit(backend);
        try {
            init.normal(Shape.vector(10), 0.0, -1.0);
            fail("Should throw for negative std");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("positive"));
        }
    }

    @Test
    public void testModelSerializerSaveModel() throws IOException {
        assumeGpuAvailable();
        Linear layer = new Linear(10, 5);
        layer.initialize(backend);
        
        Path modelPath = tempDir.resolve("model.zip");
        ModelSerializer.save(layer, modelPath);
        
        assertTrue("Model file should exist", Files.exists(modelPath));
        assertTrue("Model file should not be empty", Files.size(modelPath) > 0);
    }

    @Test
    public void testModelSerializerLoadModel() throws IOException {
        assumeGpuAvailable();
        Linear layer = new Linear(10, 5);
        layer.initialize(backend);
        
        float[] originalWeight = layer.weight().tensor().toFloatArray().clone();
        
        Path modelPath = tempDir.resolve("model.zip");
        ModelSerializer.save(layer, modelPath);
        
        Linear loadedLayer = new Linear(10, 5);
        loadedLayer.initialize(backend);
        ModelSerializer.load(modelPath, loadedLayer, backend);
        
        float[] loadedWeight = loadedLayer.weight().tensor().toFloatArray();
        assertArrayEquals("Weights should be preserved", originalWeight, loadedWeight, 1e-6f);
    }

    @Test
    public void testModelSerializerWithOptimizer() throws IOException {
        assumeGpuAvailable();
        Linear layer = new Linear(10, 5);
        layer.initialize(backend);
        Adam optimizer = new Adam(layer.parameters(), 0.001f);
        
        Path modelPath = tempDir.resolve("model_with_optimizer.zip");
        ModelSerializer.save(layer, optimizer, modelPath);
        
        assertTrue("Model file should exist", Files.exists(modelPath));
    }

    @Test
    public void testModelSerializerSequential() throws IOException {
        assumeGpuAvailable();
        Sequential model = new Sequential();
        Linear l1 = new Linear(10, 20);
        l1.initialize(backend);
        Linear l2 = new Linear(20, 5);
        l2.initialize(backend);
        model.add(l1);
        model.add(new ReLU());
        model.add(l2);
        
        Map<String, Parameter> originalParams = model.namedParameters();
        float[] originalWeight1 = originalParams.get("0.weight").tensor().toFloatArray().clone();
        float[] originalWeight2 = originalParams.get("2.weight").tensor().toFloatArray().clone();
        
        Path modelPath = tempDir.resolve("sequential.zip");
        ModelSerializer.save(model, modelPath);
        
        Sequential loadedModel = new Sequential();
        Linear loadedL1 = new Linear(10, 20);
        loadedL1.initialize(backend);
        Linear loadedL2 = new Linear(20, 5);
        loadedL2.initialize(backend);
        loadedModel.add(loadedL1);
        loadedModel.add(new ReLU());
        loadedModel.add(loadedL2);
        
        ModelSerializer.load(modelPath, loadedModel, backend);
        
        Map<String, Parameter> loadedParams = loadedModel.namedParameters();
        float[] loadedWeight1 = loadedParams.get("0.weight").tensor().toFloatArray();
        float[] loadedWeight2 = loadedParams.get("2.weight").tensor().toFloatArray();
        
        assertArrayEquals("Layer 0 weight should be preserved", originalWeight1, loadedWeight1, 1e-6f);
        assertArrayEquals("Layer 2 weight should be preserved", originalWeight2, loadedWeight2, 1e-6f);
    }

    @Test
    public void testModelSerializerInspect() throws IOException {
        assumeGpuAvailable();
        Linear layer = new Linear(10, 5);
        layer.initialize(backend);
        
        Path modelPath = tempDir.resolve("inspect_model.zip");
        ModelSerializer.save(layer, modelPath);
        
        ModelSerializer.CheckpointInfo info = ModelSerializer.inspect(modelPath);
        
        assertNotNull(info);
        assertEquals(1, info.formatVersion());
        assertEquals("Linear", info.moduleType());
        assertEquals(55, info.parameterCount());
        assertEquals(2, info.numParameters());
        assertNotNull(info.timestamp());
        assertNotNull(info.parameters());
    }

    @Test
    public void testModelSerializerFileNotFound() throws IOException {
        assumeGpuAvailable();
        Linear layer = new Linear(10, 5);
        layer.initialize(backend);
        
        Path nonExistentPath = tempDir.resolve("nonexistent.zip");
        try {
            ModelSerializer.load(nonExistentPath, layer, backend);
            fail("Should throw FileNotFoundException");
        } catch (IOException e) {
            assertTrue(e.getMessage().contains("not found"));
        }
    }

    @Test
    public void testONNXExporterCreation() {
        ONNXExporter exporter = new ONNXExporter();
        assertNotNull(exporter);
    }

    @Test
    public void testONNXExporterWithModelName() {
        ONNXExporter exporter = new ONNXExporter("TestModel");
        assertNotNull(exporter);
    }

    @Test
    public void testONNXExporterLinear() throws IOException {
        assumeGpuAvailable();
        Linear layer = new Linear(10, 5);
        layer.initialize(backend);
        
        Tensor sampleInput = Tensor.zeros(backend, Shape.matrix(1, 10));
        
        Path outputPath = tempDir.resolve("linear.onnx");
        ONNXExporter exporter = new ONNXExporter();
        exporter.export(layer, sampleInput, outputPath);
        
        assertTrue("ONNX file should exist", Files.exists(outputPath));
        assertTrue("ONNX file should not be empty", Files.size(outputPath) > 0);
    }

    @Test
    public void testONNXExporterSequential() throws IOException {
        assumeGpuAvailable();
        Sequential model = new Sequential();
        Linear l1 = new Linear(10, 20);
        l1.initialize(backend);
        Linear l2 = new Linear(20, 5);
        l2.initialize(backend);
        model.add(l1);
        model.add(new ReLU());
        model.add(l2);
        model.add(new Sigmoid());
        
        Tensor sampleInput = Tensor.zeros(backend, Shape.matrix(1, 10));
        
        Path outputPath = tempDir.resolve("sequential.onnx");
        ONNXExporter exporter = new ONNXExporter("SequentialModel");
        exporter.export(model, sampleInput, outputPath);
        
        assertTrue("ONNX file should exist", Files.exists(outputPath));
        assertTrue("ONNX file should not be empty", Files.size(outputPath) > 100);
    }

    @Test
    public void testONNXExporterCreatesDirectory() throws IOException {
        assumeGpuAvailable();
        Linear layer = new Linear(10, 5);
        layer.initialize(backend);
        
        Tensor sampleInput = Tensor.zeros(backend, Shape.matrix(1, 10));
        
        Path nestedPath = tempDir.resolve("nested").resolve("dir").resolve("model.onnx");
        ONNXExporter exporter = new ONNXExporter();
        exporter.export(layer, sampleInput, nestedPath);
        
        assertTrue("ONNX file should exist in nested directory", Files.exists(nestedPath));
    }

    @Test
    public void testONNXExporterNullModel() {
        assumeGpuAvailable();
        ONNXExporter exporter = new ONNXExporter();
        Tensor sampleInput = Tensor.zeros(backend, Shape.matrix(1, 10));
        Path outputPath = tempDir.resolve("null_model.onnx");
        
        try {
            exporter.export(null, sampleInput, outputPath);
            fail("Should throw NullPointerException");
        } catch (NullPointerException e) {
        } catch (IOException e) {
            fail("Should not throw IOException");
        }
    }

    @Test
    public void testONNXExporterNullInput() {
        assumeGpuAvailable();
        Linear layer = new Linear(10, 5);
        layer.initialize(backend);
        
        ONNXExporter exporter = new ONNXExporter();
        Path outputPath = tempDir.resolve("null_input.onnx");
        
        try {
            exporter.export(layer, null, outputPath);
            fail("Should throw NullPointerException");
        } catch (NullPointerException e) {
        } catch (IOException e) {
            fail("Should not throw IOException");
        }
    }

    @Test
    public void testRoundTripPreservesGradients() throws IOException {
        assumeGpuAvailable();
        Linear layer = new Linear(10, 5);
        layer.initialize(backend);
        
        assertTrue("Layer weight should require gradients", 
            layer.weight().tensor().requiresGrad());
        
        Path modelPath = tempDir.resolve("gradients.zip");
        ModelSerializer.save(layer, modelPath);
        
        Linear loadedLayer = new Linear(10, 5);
        loadedLayer.initialize(backend);
        ModelSerializer.load(modelPath, loadedLayer, backend);
        
        assertTrue("Loaded layer should have requiresGrad", 
            loadedLayer.weight().tensor().requiresGrad());
    }

    private float computeMean(float[] data) {
        float sum = 0;
        for (float val : data) {
            sum += val;
        }
        return sum / data.length;
    }

    private float computeStd(float[] data, float mean) {
        float sumSq = 0;
        for (float val : data) {
            sumSq += (val - mean) * (val - mean);
        }
        return (float) Math.sqrt(sumSq / data.length);
    }
}
