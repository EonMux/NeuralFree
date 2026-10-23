package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Embedding;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.loss.CrossEntropyLoss;
import com.eonmux.neuralfree.nn.transformer.SinusoidalPE;
import com.eonmux.neuralfree.nn.transformer.TransformerEncoder;
import com.eonmux.neuralfree.nn.transformer.TransformerEncoderLayer;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Random;

public final class TransformerDemo {

    private static final int VOCAB_SIZE = 1000;
    private static final int D_MODEL = 64;
    private static final int NUM_HEADS = 4;
    private static final int NUM_LAYERS = 2;
    private static final int D_FF = 128;
    private static final int MAX_SEQ_LEN = 32;
    private static final int NUM_CLASSES = 5;
    private static final int BATCH_SIZE = 4;
    private static final int EPOCHS = 5;

    public static void run(ComputeBackend backend) {
        DemoUtils.printHeader("Transformer Text Classification Demo");
        System.out.println("Training a Transformer encoder for sequence classification\n");

        Random random = new Random(42);

        TransformerClassifier model = new TransformerClassifier();
        model.initialize(backend, random);
        System.out.println("Model: " + model.name());
        System.out.println("Parameters: " + model.parameterCount() + "\n");

        System.out.println("Generating synthetic sequence data...");
        int[][] sequences = new int[BATCH_SIZE][MAX_SEQ_LEN];
        float[] labels = new float[BATCH_SIZE];
        
        for (int b = 0; b < BATCH_SIZE; b++) {
            labels[b] = random.nextInt(NUM_CLASSES);
            for (int t = 0; t < MAX_SEQ_LEN; t++) {
                sequences[b][t] = random.nextInt(VOCAB_SIZE);
            }
        }

        Adam optimizer = new Adam(model.parameters(), 0.001f);
        try (Tensor inputBatch = createIntTensor(backend, sequences);
             Tensor labelBatch = Tensor.fromArray(backend, labels)) {
            CrossEntropyLoss lossFn = new CrossEntropyLoss();

            System.out.println("Starting training loop...");
            model.train();

            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                float lossVal = DemoUtils.trainStep(model, optimizer, () -> model.forward(inputBatch),
                        output -> lossFn.compute(output, labelBatch), epoch == 0);
                System.out.printf("Epoch %d/%d - Loss: %.4f%n", epoch + 1, EPOCHS, lossVal);
            }

            System.out.println("\nRunning inference...");
            model.eval();

            int[] testSeq = new int[MAX_SEQ_LEN];
            for (int i = 0; i < MAX_SEQ_LEN; i++) {
                testSeq[i] = random.nextInt(VOCAB_SIZE);
            }

            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor testInput = createIntTensor(backend, new int[][]{testSeq})) {
                long inferStart = System.nanoTime();
                float[] probs;
                try (Tensor output = model.forward(testInput)) {
                    long inferTime = System.nanoTime() - inferStart;
                    probs = DemoUtils.probabilities(output, NUM_CLASSES);
                    System.out.println("Inference time: " + (inferTime / 1000.0) + " μs");
                }
                System.out.println("Predicted class: " + DemoUtils.argmax(probs));
                System.out.println("Probabilities: " + DemoUtils.formatProbs(probs));
            }
        } finally {
            optimizer.close();
            model.close();
        }
    }

    private static Tensor createIntTensor(ComputeBackend backend, int[][] data) {
        int batchSize = data.length;
        int seqLen = data[0].length;
        
        int[] flatData = new int[batchSize * seqLen];
        int idx = 0;
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                flatData[idx++] = data[b][t];
            }
        }
        
        return DemoUtils.intTensor(backend, flatData, batchSize, seqLen);
    }

    private static class TransformerClassifier extends Module {
        private Embedding embedding;
        private SinusoidalPE posEncoding;
        private TransformerEncoder encoder;
        private Linear classifier;

        /** Closes every intermediate it creates; the caller closes the returned logits. */
        @Override
        public Tensor forward(Tensor input) {
            Tensor x = embedding.forward(input);
            
            x = DemoUtils.consume(x, posEncoding::apply);
            
            x = DemoUtils.consume(x, encoder::forward);
            
            x = DemoUtils.consume(x, t -> TensorOps.mean(t, 1));
            
            return DemoUtils.consume(x, classifier::forward);
        }

        public void initialize(ComputeBackend backend, Random random) {
            embedding = new Embedding(VOCAB_SIZE, D_MODEL);
            embedding.initialize(backend, random);
            registerModule("embedding", embedding);

            posEncoding = new SinusoidalPE(D_MODEL, MAX_SEQ_LEN);
            posEncoding.initialize(backend);

            encoder = new TransformerEncoder(
                NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, 0.1f, true,
                TransformerEncoderLayer.NormType.LAYER_NORM,
                TransformerEncoderLayer.FFNType.MLP, false
            );
            encoder.initialize(backend, random);
            registerModule("encoder", encoder);

            classifier = new Linear(D_MODEL, NUM_CLASSES);
            classifier.initialize(backend, random);
            registerModule("classifier", classifier);
        }

        /** Also releases the positional-encoding table, which is not a registered submodule. */
        @Override
        public void close() {
            try {
                if (posEncoding != null) {
                    posEncoding.close();
                }
            } finally {
                super.close();
            }
        }

        public String name() {
            return "TransformerClassifier(vocab=" + VOCAB_SIZE + ", d_model=" + D_MODEL + 
                   ", heads=" + NUM_HEADS + ", layers=" + NUM_LAYERS + ")";
        }
    }
}
