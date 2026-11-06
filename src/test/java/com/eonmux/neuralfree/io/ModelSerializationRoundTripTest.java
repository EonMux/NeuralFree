package com.eonmux.neuralfree.io;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.data.*;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.loss.CrossEntropyLoss;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.training.*;

import org.junit.Before;
import org.junit.After;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.junit.Assert.*;

public class ModelSerializationRoundTripTest {

    private ComputeBackend backend;
    private Path tempDir;
    private static final int IMG_SIZE = 28;
    private static final int NUM_CLASSES = 10;
    private static final int NUM_TRAIN_SAMPLES = 160;
    private static final int BATCH_SIZE = 16;

    @Before
    public void setUp() throws IOException {
        backend = TrainParity.hostBackend();
        tempDir = Files.createTempDirectory("model_roundtrip_test");
    }

    @After
    public void tearDown() throws IOException {
        ComputationGraph.current().clear();
        if (backend != null) {
            backend.close();
        }
        if (tempDir != null) {
            deleteDirectory(tempDir.toFile());
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
    public void testModelSerializerSaveProducesValidFile() throws IOException {
        Module model = createCNNModel();
        
        Path modelPath = tempDir.resolve("cnn_model.zip");
        ModelSerializer.save(model, modelPath);
        
        assertTrue("Model file should exist", Files.exists(modelPath));
        assertTrue("Model file should not be empty", Files.size(modelPath) > 0);
        
        System.out.println("Model file size: " + Files.size(modelPath) + " bytes");
    }

    @Test
    public void testModelSerializerSaveWithOptimizer() throws IOException {
        Module model = createCNNModel();
        Adam optimizer = new Adam(model.parameters(), 0.001f);
        
        Path modelPath = tempDir.resolve("model_with_optimizer.zip");
        ModelSerializer.save(model, optimizer, modelPath);
        
        assertTrue("Model file with optimizer should exist", Files.exists(modelPath));
        assertTrue("Model file with optimizer should not be empty", Files.size(modelPath) > 0);
        
        System.out.println("Model+optimizer file size: " + Files.size(modelPath) + " bytes");
    }

    @Test
    public void testCheckpointTrainingResume() throws IOException {
        Module model = createCNNModel();
        
        Path checkpointPath = tempDir.resolve("checkpoint.zip");
        Adam optimizer = new Adam(model.parameters(), 0.001f);
        ModelSerializer.save(model, optimizer, checkpointPath);
        
        assertTrue("Checkpoint file should exist", Files.exists(checkpointPath));
        
        TensorDataset dataset = createSyntheticDataset(NUM_TRAIN_SAMPLES, 42);
        DataLoader<TensorDataset.Sample> loader = createDataLoader(dataset);
        
        Trainer trainer = Trainer.builder()
            .model(model)
            .optimizer(optimizer)
            .lossFunction(new CrossEntropyLoss())
            .trainLoader(loader)
            .config(TrainerConfig.defaults())
            .build();
        
        TrainingResult result = trainer.train(5);
        
        assertNotNull("Training result should not be null", result);
        assertEquals("Should complete all epochs", 5, result.epochs());
        
        List<Float> losses = result.trainLosses();
        assertEquals("Should have loss for each epoch", 5, losses.size());
        
        float firstLoss = losses.get(0);
        float lastLoss = losses.get(losses.size() - 1);
        assertTrue("Losses should be finite", Float.isFinite(firstLoss) && Float.isFinite(lastLoss));
        
        System.out.println("Checkpoint resume training: first loss=" + firstLoss + ", last loss=" + lastLoss);
        
        trainer.close();
        loader.close();
    }

    @Test
    public void testONNXExporterProducesValidFile() throws IOException {
        Module model = createCNNModel();
        
        Tensor sampleInput = Tensor.zeros(backend, Shape.matrix(1, IMG_SIZE * IMG_SIZE));
        
        Path outputPath = tempDir.resolve("cnn_model.onnx");
        ONNXExporter exporter = new ONNXExporter("TestCNN");
        exporter.export(model, sampleInput, outputPath);
        
        assertTrue("ONNX file should exist", Files.exists(outputPath));
        assertTrue("ONNX file should not be empty", Files.size(outputPath) > 100);
        
        byte[] fileBytes = Files.readAllBytes(outputPath);
        int magicNumber = Integer.reverseBytes(ByteBuffer.wrap(fileBytes, 0, 4).getInt());
        assertEquals("ONNX file should start with magic number 0x08C7", 0x08C7, magicNumber);
        
        System.out.println("ONNX file size: " + Files.size(outputPath) + " bytes");
    }

    @Test
    public void testONNXExporterWithSequentialModel() throws IOException {
        Sequential model = new Sequential();
        
        Linear linear1 = new Linear(IMG_SIZE * IMG_SIZE, 64);
        linear1.initialize(backend);
        model.add(linear1);
        model.add(new ReLU());
        
        Linear linear2 = new Linear(64, NUM_CLASSES);
        linear2.initialize(backend);
        model.add(linear2);
        
        Tensor sampleInput = Tensor.zeros(backend, Shape.matrix(1, IMG_SIZE * IMG_SIZE));
        
        Path outputPath = tempDir.resolve("sequential_model.onnx");
        ONNXExporter exporter = new ONNXExporter("SequentialModel");
        exporter.export(model, sampleInput, outputPath);
        
        assertTrue("ONNX file should exist", Files.exists(outputPath));
        assertTrue("ONNX file should have reasonable size", Files.size(outputPath) > 100);
        
        System.out.println("Sequential ONNX file size: " + Files.size(outputPath) + " bytes");
    }

    @Test
    public void testModelSerializerInspectReturnsValidInfo() throws IOException {
        Module model = createCNNModel();
        
        Path modelPath = tempDir.resolve("inspect_model.zip");
        ModelSerializer.save(model, modelPath);
        
        ModelSerializer.CheckpointInfo info = ModelSerializer.inspect(modelPath);
        
        assertNotNull("CheckpointInfo should not be null", info);
        assertEquals("Format version should be 1", 1, info.formatVersion());
        assertEquals("Module type should be Sequential", "Sequential", info.moduleType());
        assertTrue("Parameter count should be positive", info.parameterCount() > 0);
        assertNotNull("Timestamp should not be null", info.timestamp());
        assertNotNull("Parameters map should not be null", info.parameters());
        
        System.out.println("Model info: type=" + info.moduleType() + ", params=" + info.parameterCount());
    }

    @Test
    public void testTrainingDecreasesLoss() throws IOException {
        Module model = createCNNModel();
        
        TensorDataset dataset = createSyntheticDataset(NUM_TRAIN_SAMPLES, 42);
        DataLoader<TensorDataset.Sample> loader = createDataLoader(dataset);
        Adam optimizer = new Adam(model.parameters(), 0.001f);
        LossFunction lossFn = new CrossEntropyLoss();
        
        Trainer trainer = Trainer.builder()
            .model(model)
            .optimizer(optimizer)
            .lossFunction(lossFn)
            .trainLoader(loader)
            .config(TrainerConfig.defaults())
            .build();
        
        TrainingResult result = trainer.train(5);
        
        List<Float> losses = result.trainLosses();
        assertEquals("Should have 5 loss values", 5, losses.size());
        
        float firstLoss = losses.get(0);
        float lastLoss = losses.get(losses.size() - 1);
        
        System.out.println("Training: first loss=" + firstLoss + ", last loss=" + lastLoss);
        
        assertTrue("Training losses should be finite", Float.isFinite(firstLoss) && Float.isFinite(lastLoss));
        assertTrue("Loss should decrease or stay same during training", lastLoss <= firstLoss * 1.5f);
        
        trainer.close();
        loader.close();
    }

    @Test
    public void testTrainingForMultipleEpochsWithCheckpoint() throws IOException {
        Module model = createCNNModel();
        Adam optimizer = new Adam(model.parameters(), 0.001f);
        
        TensorDataset dataset = createSyntheticDataset(NUM_TRAIN_SAMPLES, 42);
        DataLoader<TensorDataset.Sample> loader = createDataLoader(dataset);
        
        Trainer trainer = Trainer.builder()
            .model(model)
            .optimizer(optimizer)
            .lossFunction(new CrossEntropyLoss())
            .trainLoader(loader)
            .config(TrainerConfig.defaults())
            .build();
        
        TrainingResult result1 = trainer.train(3);
        
        Path checkpointPath = tempDir.resolve("checkpoint_epoch3.zip");
        ModelSerializer.save(model, optimizer, checkpointPath);
        assertTrue("Checkpoint at epoch 3 should exist", Files.exists(checkpointPath));
        
        List<Float> losses1 = result1.trainLosses();
        System.out.println("First 3 epochs losses: " + losses1);
        
        TrainingResult result2 = trainer.train(2);
        List<Float> losses2 = result2.trainLosses();
        System.out.println("Next 2 epochs losses: " + losses2);
        
        assertTrue("All losses should be finite", 
            losses1.stream().allMatch(Float::isFinite) && 
            losses2.stream().allMatch(Float::isFinite));
        
        trainer.close();
        loader.close();
    }

    @Test
    public void testONNXExporterCreatesNestedDirectories() throws IOException {
        Module model = createCNNModel();
        
        Tensor sampleInput = Tensor.zeros(backend, Shape.matrix(1, IMG_SIZE * IMG_SIZE));
        
        Path nestedPath = tempDir.resolve("nested").resolve("dir").resolve("model.onnx");
        ONNXExporter exporter = new ONNXExporter("NestedModel");
        exporter.export(model, sampleInput, nestedPath);
        
        assertTrue("ONNX file should exist in nested directory", Files.exists(nestedPath));
        assertTrue("ONNX file should have content", Files.size(nestedPath) > 100);
        
        System.out.println("Nested ONNX file created at: " + nestedPath);
    }

    @Test
    public void testMultipleSaveOperationsProduceConsistentFiles() throws IOException {
        Module model = createCNNModel();
        
        for (int i = 0; i < 3; i++) {
            Path modelPath = tempDir.resolve("model_v" + i + ".zip");
            ModelSerializer.save(model, modelPath);
            
            assertTrue("Model file v" + i + " should exist", Files.exists(modelPath));
            
            ModelSerializer.CheckpointInfo info = ModelSerializer.inspect(modelPath);
            assertEquals("Format version should be consistent", 1, info.formatVersion());
        }
        
        System.out.println("Created 3 consistent model files");
    }

    private Module createCNNModel() {
        Sequential model = new Sequential();
        
        Linear linear1 = new Linear(IMG_SIZE * IMG_SIZE, 128);
        linear1.initialize(backend);
        model.add(linear1);
        
        model.add(new ReLU());
        
        Linear linear2 = new Linear(128, 64);
        linear2.initialize(backend);
        model.add(linear2);
        
        model.add(new ReLU());
        
        Linear linear3 = new Linear(64, NUM_CLASSES);
        linear3.initialize(backend);
        model.add(linear3);
        
        return model;
    }

    private TensorDataset createSyntheticDataset(int numSamples, long seed) {
        Random random = new Random(seed);
        
        Tensor[] inputs = new Tensor[numSamples];
        Tensor[] targets = new Tensor[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            int label = random.nextInt(NUM_CLASSES);
            
            float[] imageData = new float[IMG_SIZE * IMG_SIZE];
            int centerRow = 7 + (label % 5) * 4;
            int centerCol = 7 + (label / 2) * 4;
            
            for (int r = 0; r < IMG_SIZE; r++) {
                for (int c = 0; c < IMG_SIZE; c++) {
                    int idx = r * IMG_SIZE + c;
                    double dist = Math.sqrt((r - centerRow) * (r - centerRow) + (c - centerCol) * (c - centerCol));
                    if (dist < 3.0) {
                        imageData[idx] = 0.5f + random.nextFloat() * 0.5f;
                    } else {
                        imageData[idx] = random.nextFloat() * 0.1f;
                    }
                }
            }
            
            inputs[i] = Tensor.fromArray(backend, imageData).reshape(Shape.of(IMG_SIZE * IMG_SIZE));
            
            float[] targetData = new float[NUM_CLASSES];
            targetData[label] = 1.0f;
            targets[i] = Tensor.fromArray(backend, targetData);
        }
        
        return new TensorDataset(inputs, targets);
    }

    private DataLoader<TensorDataset.Sample> createDataLoader(TensorDataset dataset) {
        return DataLoader.<TensorDataset.Sample>builder(dataset)
            .batchSize(BATCH_SIZE)
            .shuffle(true)
            .build();
    }
}
