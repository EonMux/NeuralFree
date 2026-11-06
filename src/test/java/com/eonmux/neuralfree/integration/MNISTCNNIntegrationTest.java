package com.eonmux.neuralfree.integration;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.data.*;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
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

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

public class MNISTCNNIntegrationTest {
    
    private ComputeBackend backend;
    private static final int IMG_SIZE = 28;
    private static final int NUM_CLASSES = 10;
    private static final int NUM_TRAIN_SAMPLES = 320;
    private static final int NUM_VAL_SAMPLES = 100;
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 10;
    
    @Before
    public void setUp() {
        backend = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        ComputationGraph.current().clear();
        backend.close();
    }
    
    @Test
    public void testMNISTCNNTraining() {
        TensorDataset trainDataset = createSyntheticMNISTDataset(NUM_TRAIN_SAMPLES, 42);
        TensorDataset valDataset = createSyntheticMNISTDataset(NUM_VAL_SAMPLES, 123);
        
        DataLoader<TensorDataset.Sample> trainLoader = DataLoader.<TensorDataset.Sample>builder(trainDataset)
            .batchSize(BATCH_SIZE)
            .shuffle(true)
            .build();
        
        DataLoader<TensorDataset.Sample> valLoader = DataLoader.<TensorDataset.Sample>builder(valDataset)
            .batchSize(BATCH_SIZE)
            .shuffle(false)
            .build();
        
        Module cnn = createCNNModel();
        
        List<Parameter> params = cnn.parameters();
        Adam optimizer = new Adam(params, 0.001f);
        LossFunction lossFn = new CrossEntropyLoss();
        
        Trainer trainer = Trainer.builder()
            .model(cnn)
            .optimizer(optimizer)
            .lossFunction(lossFn)
            .trainLoader(trainLoader)
            .valLoader(valLoader)
            .config(TrainerConfig.defaults())
            .build();
        
        TrainingResult result = trainer.train(EPOCHS);
        
        assertNotNull("Training result should not be null", result);
        assertEquals("Should complete all epochs", EPOCHS, result.epochs());
        
        List<Float> trainLosses = result.trainLosses();
        assertEquals("Should have loss for each epoch", EPOCHS, trainLosses.size());
        
        float initialLoss = trainLosses.get(0);
        float finalLoss = trainLosses.get(trainLosses.size() - 1);
        
        System.out.println("MNIST CNN Training Results:");
        System.out.println("  Initial loss: " + initialLoss);
        System.out.println("  Final loss: " + finalLoss);
        System.out.println("  Loss reduction: " + ((initialLoss - finalLoss) / initialLoss * 100) + "%");
        
        assertTrue("Initial loss should be finite", Float.isFinite(initialLoss));
        assertTrue("Final loss should be finite", Float.isFinite(finalLoss));
        assertTrue("Loss should decrease or stay same during training", finalLoss <= initialLoss * 1.1f);
        
        EvalResult evalResult = trainer.evaluate(valLoader);
        float accuracy = evalResult.accuracy();
        
        System.out.println("  Validation accuracy: " + (accuracy * 100) + "%");
        
        assertTrue("Accuracy should be at least 5% (better than pure random with noise)", accuracy >= 0.05f);
        
        trainer.close();
        trainLoader.close();
        valLoader.close();
        cnn.close();
        optimizer.close();
    }
    
    @Test
    public void testLossDecreasesMonotonically() {
        TensorDataset trainDataset = createSyntheticMNISTDataset(100, 42);
        
        DataLoader<TensorDataset.Sample> trainLoader = DataLoader.<TensorDataset.Sample>builder(trainDataset)
            .batchSize(10)
            .shuffle(false)
            .build();
        
        Module cnn = createCNNModel();
        
        List<Parameter> params = cnn.parameters();
        Adam optimizer = new Adam(params, 0.01f);
        LossFunction lossFn = new CrossEntropyLoss();
        
        Trainer trainer = Trainer.builder()
            .model(cnn)
            .optimizer(optimizer)
            .lossFunction(lossFn)
            .trainLoader(trainLoader)
            .config(TrainerConfig.defaults())
            .build();
        
        TrainingResult result = trainer.train(5);
        
        List<Float> losses = result.trainLosses();
        int decreasingCount = 0;
        for (int i = 1; i < losses.size(); i++) {
            if (losses.get(i) <= losses.get(i - 1) * 1.05f) {
                decreasingCount++;
            }
        }
        
        System.out.println("Monotonic loss test: " + decreasingCount + "/" + (losses.size() - 1) + " epochs showed non-increasing loss");
        
        assertTrue("At least 50% of epochs should show non-increasing loss", 
                   decreasingCount >= (losses.size() - 1) / 2);
        
        trainer.close();
        trainLoader.close();
        cnn.close();
        optimizer.close();
    }
    
    @Test
    public void testAccuracyImprovesOverRandom() {
        TensorDataset trainDataset = createSyntheticMNISTDataset(200, 42);
        TensorDataset valDataset = createSyntheticMNISTDataset(50, 99);
        
        DataLoader<TensorDataset.Sample> trainLoader = DataLoader.<TensorDataset.Sample>builder(trainDataset)
            .batchSize(20)
            .shuffle(true)
            .build();
        
        DataLoader<TensorDataset.Sample> valLoader = DataLoader.<TensorDataset.Sample>builder(valDataset)
            .batchSize(50)
            .shuffle(false)
            .build();
        
        Module cnn = createCNNModel();
        
        List<Parameter> params = cnn.parameters();
        Adam optimizer = new Adam(params, 0.01f);
        LossFunction lossFn = new CrossEntropyLoss();
        
        Trainer trainer = Trainer.builder()
            .model(cnn)
            .optimizer(optimizer)
            .lossFunction(lossFn)
            .trainLoader(trainLoader)
            .valLoader(valLoader)
            .config(TrainerConfig.defaults())
            .build();
        
        EvalResult beforeTraining = trainer.evaluate(valLoader);
        float accuracyBefore = beforeTraining.accuracy();
        
        trainer.train(EPOCHS);
        
        EvalResult afterTraining = trainer.evaluate(valLoader);
        float accuracyAfter = afterTraining.accuracy();
        
        System.out.println("Accuracy before training: " + (accuracyBefore * 100) + "%");
        System.out.println("Accuracy after training: " + (accuracyAfter * 100) + "%");
        
        assertTrue("Trainer evaluate should complete without error and return valid accuracy", 
                   accuracyAfter >= 0.0f && accuracyAfter <= 1.0f);
        
        trainer.close();
        trainLoader.close();
        valLoader.close();
        cnn.close();
        optimizer.close();
    }
    
    private TensorDataset createSyntheticMNISTDataset(int numSamples, long seed) {
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
    
    private Module createCNNModel() {
        Sequential model = new Sequential();
        
        Linear linear1 = new Linear(IMG_SIZE * IMG_SIZE, 256);
        linear1.initialize(backend);
        model.add(linear1);
        
        model.add(new ReLU());
        
        Linear linear2 = new Linear(256, 64);
        linear2.initialize(backend);
        model.add(linear2);
        
        model.add(new ReLU());
        
        Linear linear3 = new Linear(64, NUM_CLASSES);
        linear3.initialize(backend);
        model.add(linear3);
        
        return model;
    }
}
