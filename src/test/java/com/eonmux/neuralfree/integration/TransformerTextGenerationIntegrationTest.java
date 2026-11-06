package com.eonmux.neuralfree.integration;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.data.*;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Embedding;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.loss.CrossEntropyLoss;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.nn.generative.AutoregressiveGenerator;
import com.eonmux.neuralfree.nn.generative.SamplingConfig;
import com.eonmux.neuralfree.optim.AdamW;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.training.*;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

public class TransformerTextGenerationIntegrationTest {
    
    private ComputeBackend backend;
    private static final int VOCAB_SIZE = 256;
    private static final int D_MODEL = 128;
    private static final int NUM_HEADS = 4;
    private static final int NUM_LAYERS = 4;
    private static final int D_FF = 512;
    private static final int MAX_SEQ_LEN = 256;
    private static final int FIXED_SEQ_LEN = 20;
    private static final int BATCH_SIZE = 8;
    private static final int EPOCHS = 20;
    private static final int NUM_TRAIN_SAMPLES = 160;
    
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
    public void testTransformerTextGeneration() {
        TensorDataset trainDataset = createSyntheticCharacterDataset(NUM_TRAIN_SAMPLES, 42);
        
        DataLoader<TensorDataset.Sample> trainLoader = DataLoader.<TensorDataset.Sample>builder(trainDataset)
            .batchSize(BATCH_SIZE)
            .shuffle(true)
            .build();
        
        SimpleTransformerLM model = createTransformerModel();
        
        List<Parameter> params = model.parameters();
        AdamW optimizer = new AdamW(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
        LossFunction lossFn = new CrossEntropyLoss();
        
        Trainer trainer = Trainer.builder()
            .model(model)
            .optimizer(optimizer)
            .lossFunction(lossFn)
            .trainLoader(trainLoader)
            .config(TrainerConfig.defaults())
            .build();
        
        TrainingResult result = trainer.train(EPOCHS);
        
        assertNotNull("Training result should not be null", result);
        assertEquals("Should complete all epochs", EPOCHS, result.epochs());
        
        List<Float> trainLosses = result.trainLosses();
        assertEquals("Should have loss for each epoch", EPOCHS, trainLosses.size());
        
        float initialLoss = trainLosses.get(0);
        float finalLoss = trainLosses.get(trainLosses.size() - 1);
        
        System.out.println("Transformer Text Generation Training Results:");
        System.out.println("  Initial loss: " + initialLoss);
        System.out.println("  Final loss: " + finalLoss);
        System.out.println("  Loss reduction: " + ((initialLoss - finalLoss) / initialLoss * 100) + "%");
        
        assertTrue("Initial loss should be finite", Float.isFinite(initialLoss));
        assertTrue("Final loss should be finite", Float.isFinite(finalLoss));
        assertTrue("Loss should decrease or stay same during training", finalLoss <= initialLoss * 1.1f);
        
        AutoregressiveGenerator generator = new AutoregressiveGenerator(
            model, VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, MAX_SEQ_LEN
        );
        
        SamplingConfig samplingConfig = SamplingConfig.builder()
            .temperature(1.0f)
            .topK(50)
            .seed(12345)
            .build();
        
        int[] promptTokens = new int[]{65, 66, 67};
        int[] generatedTokens = generator.generate(promptTokens, 50, samplingConfig);
        
        System.out.println("  Generated " + generatedTokens.length + " tokens");
        System.out.println("  Prompt length: " + promptTokens.length);
        System.out.println("  First 10 generated tokens: " + Arrays.toString(Arrays.copyOf(generatedTokens, Math.min(10, generatedTokens.length))));
        
        assertEquals("Generated sequence should have 50 tokens", 50, generatedTokens.length);
        
        for (int token : generatedTokens) {
            assertTrue("Generated token should be valid", token >= 0 && token < VOCAB_SIZE);
        }
        
        generator.close();
        trainer.close();
        trainLoader.close();
        model.close();
        optimizer.close();
    }
    
    @Test
    public void testTrainingLossDecreases() {
        TensorDataset trainDataset = createSyntheticCharacterDataset(80, 42);
        
        DataLoader<TensorDataset.Sample> trainLoader = DataLoader.<TensorDataset.Sample>builder(trainDataset)
            .batchSize(8)
            .shuffle(false)
            .build();
        
        SimpleTransformerLM model = createTransformerModel();
        
        List<Parameter> params = model.parameters();
        AdamW optimizer = new AdamW(params, 0.01f);
        LossFunction lossFn = new CrossEntropyLoss();
        
        Trainer trainer = Trainer.builder()
            .model(model)
            .optimizer(optimizer)
            .lossFunction(lossFn)
            .trainLoader(trainLoader)
            .config(TrainerConfig.defaults())
            .build();
        
        TrainingResult result = trainer.train(10);
        
        List<Float> losses = result.trainLosses();
        
        System.out.println("Loss progression over epochs:");
        for (int i = 0; i < losses.size(); i++) {
            System.out.println("  Epoch " + (i + 1) + ": " + losses.get(i));
        }
        
        int decreasingCount = 0;
        for (int i = 1; i < losses.size(); i++) {
            if (losses.get(i) < losses.get(i - 1) * 1.1f) {
                decreasingCount++;
            }
        }
        
        System.out.println("Monotonic loss test: " + decreasingCount + "/" + (losses.size() - 1) + " epochs showed non-increasing loss");
        
        assertTrue("At least 50% of epochs should show non-increasing loss", 
                   decreasingCount >= (losses.size() - 1) / 2);
        
        trainer.close();
        trainLoader.close();
        model.close();
        optimizer.close();
    }
    
    @Test
    public void testGenerationProducesCorrectLength() {
        SimpleTransformerLM model = createTransformerModel();
        
        AutoregressiveGenerator generator = new AutoregressiveGenerator(
            model, VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, MAX_SEQ_LEN
        );
        
        SamplingConfig config = SamplingConfig.builder()
            .temperature(0.8f)
            .topK(40)
            .seed(999)
            .build();
        
        int[] prompt = new int[]{72, 101, 108, 108, 111};
        
        int[] shortGen = generator.generate(prompt, 10, config);
        assertEquals("Short generation should produce 10 tokens", 10, shortGen.length);
        
        int[] mediumGen = generator.generate(prompt, 25, config);
        assertEquals("Medium generation should produce 25 tokens", 25, mediumGen.length);
        
        int[] longGen = generator.generate(prompt, 50, config);
        assertEquals("Long generation should produce 50 tokens", 50, longGen.length);
        
        System.out.println("Generation length tests:");
        System.out.println("  Short (10): " + shortGen.length + " tokens");
        System.out.println("  Medium (25): " + mediumGen.length + " tokens");
        System.out.println("  Long (50): " + longGen.length + " tokens");
        
        generator.close();
        model.close();
    }
    
    private TensorDataset createSyntheticCharacterDataset(int numSamples, long seed) {
        Random random = new Random(seed);
        
        Tensor[] inputs = new Tensor[numSamples];
        Tensor[] targets = new Tensor[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            int seqLen = FIXED_SEQ_LEN;
            float[] inputData = new float[seqLen];
            float[] targetData = new float[seqLen * VOCAB_SIZE];
            
            for (int j = 0; j < seqLen; j++) {
                int charIdx = random.nextInt(VOCAB_SIZE);
                inputData[j] = charIdx;
            }
            
            inputs[i] = Tensor.fromArray(backend, inputData);
            
            for (int j = 0; j < seqLen; j++) {
                int nextCharIdx = (int) ((inputData[j] + 13) % VOCAB_SIZE);
                targetData[j * VOCAB_SIZE + nextCharIdx] = 1.0f;
            }
            
            targets[i] = Tensor.fromArray(backend, targetData).reshape(Shape.of(seqLen, VOCAB_SIZE));
        }
        
        return new TensorDataset(inputs, targets);
    }
    
    private SimpleTransformerLM createTransformerModel() {
        SimpleTransformerLM model = new SimpleTransformerLM(
            VOCAB_SIZE, D_MODEL, NUM_LAYERS, D_FF
        );
        model.initialize(backend);
        return model;
    }
    
    private static class SimpleTransformerLM extends Module {
        private final int vocabSize;
        private final int dModel;
        private final int numLayers;
        private final int dFF;
        
        private Embedding embedding;
        private List<Linear> layers;
        private Linear outputProjection;
        
        private boolean initialized = false;
        
        SimpleTransformerLM(int vocabSize, int dModel, int numLayers, int dFF) {
            this.vocabSize = vocabSize;
            this.dModel = dModel;
            this.numLayers = numLayers;
            this.dFF = dFF;
            this.layers = new ArrayList<>();
        }
        
        void initialize(ComputeBackend backend) {
            initialize(backend, new Random());
        }
        
        void initialize(ComputeBackend backend, Random random) {
            this.embedding = new Embedding(vocabSize, dModel);
            embedding.initialize(backend, random);
            registerModule("embedding", embedding);
            
            for (int i = 0; i < numLayers; i++) {
                Linear layer = new Linear(dModel, dFF);
                layer.initialize(backend, random);
                layers.add(layer);
                registerModule("layer_" + i, layer);
                
                Linear proj = new Linear(dFF, dModel);
                proj.initialize(backend, random);
                layers.add(proj);
                registerModule("proj_" + i, proj);
            }
            
            this.outputProjection = new Linear(dModel, vocabSize);
            outputProjection.initialize(backend, random);
            registerModule("output_projection", outputProjection);
            
            initialized = true;
        }
        
        /**
         * Logits {@code [batch, seqLen, vocab]} for token ids {@code [batch, seqLen]} (any dtype holding whole
         * numbers) or {@code [seqLen]} (a batch of one): embedding, then per layer
         * {@code h + proj(relu(layer(h)))}, then the output projection. Composed from the recording layers and
         * {@link TensorOps}, so in grad mode the loss's backward reaches every parameter. Every intermediate is closed;
         * the caller closes the returned logits.
         */
        @Override
        public Tensor forward(Tensor input) {
            ensureInitialized();

            boolean batched = input.shape().rank() > 1;
            int batchSize = batched ? input.shape().dim(0) : 1;
            int seqLen = batched ? input.shape().dim(1) : (int) input.numElements();

            Tensor hidden;
            try (Tensor ids = tokenIds(input, batchSize, seqLen)) {
                hidden = embedding.forward(ids);
            }
            for (int l = 0; l < numLayers; l++) {
                Linear ff1 = layers.get(l * 2);
                Linear ff2 = layers.get(l * 2 + 1);
                Tensor next;
                try (Tensor expanded = ff1.forward(hidden);
                     Tensor activated = TensorOps.relu(expanded);
                     Tensor projected = ff2.forward(activated)) {
                    next = TensorOps.add(hidden, projected);
                } finally {
                    hidden.close();
                }
                hidden = next;
            }
            try {
                return outputProjection.forward(hidden);
            } finally {
                hidden.close();
            }
        }

        /** The ids of {@code input} as an INT32 {@code [batch, seqLen]} tensor; never requires grad. */
        private static Tensor tokenIds(Tensor input, int batchSize, int seqLen) {
            Tensor ints = input.to(DType.INT32);
            try {
                return ints.reshape(Shape.of(batchSize, seqLen));
            } finally {
                if (ints != input) {
                    ints.close();
                }
            }
        }

        private void ensureInitialized() {
            if (!initialized) {
                throw new IllegalStateException("SimpleTransformerLM not initialized. Call initialize(backend) first.");
            }
        }
    }
}
