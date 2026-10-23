package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.layers.MaxPool2d;
import com.eonmux.neuralfree.nn.loss.CrossEntropyLoss;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Random;

public final class CNNDemo {

    private static final int IN_CHANNELS = 3;
    private static final int OUT_CHANNELS = 16;
    private static final int IMAGE_SIZE = 16;
    private static final int NUM_CLASSES = 4;
    private static final int BATCH_SIZE = 8;
    private static final int EPOCHS = 5;

    public static void run(ComputeBackend backend) {
        DemoUtils.printHeader("CNN Image Classification Demo");
        System.out.println("Training a simple CNN on synthetic image data\n");

        Random random = new Random(42);

        SimpleCNN model = new SimpleCNN();
        model.initialize(backend, random);
        System.out.println("Model: " + model.name());
        System.out.println("Parameters: " + model.parameterCount() + "\n");

        System.out.println("Generating synthetic image data...");
        float[][][][] images = new float[BATCH_SIZE][IN_CHANNELS][IMAGE_SIZE][IMAGE_SIZE];
        float[] labels = new float[BATCH_SIZE];
        
        for (int b = 0; b < BATCH_SIZE; b++) {
            labels[b] = random.nextInt(NUM_CLASSES);
            for (int c = 0; c < IN_CHANNELS; c++) {
                for (int h = 0; h < IMAGE_SIZE; h++) {
                    for (int w = 0; w < IMAGE_SIZE; w++) {
                        images[b][c][h][w] = (float) (random.nextGaussian() * 0.1);
                    }
                }
            }
        }

        float[] flatImages = flattenImages(images);
        Adam optimizer = new Adam(model.parameters(), 0.001f);
        try (Tensor inputBatch = DemoUtils.tensor(backend, flatImages,
                 BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
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

            float[] singleImage = new float[IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE];
            for (int i = 0; i < singleImage.length; i++) {
                singleImage[i] = (float) (random.nextGaussian() * 0.1);
            }

            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor testInput = DemoUtils.tensor(backend, singleImage, 1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)) {
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

    private static float[] flattenImages(float[][][][] images) {
        int batchSize = images.length;
        int channels = images[0].length;
        int height = images[0][0].length;
        int width = images[0][0][0].length;
        
        float[] flat = new float[batchSize * channels * height * width];
        int idx = 0;
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        flat[idx++] = images[b][c][h][w];
                    }
                }
            }
        }
        return flat;
    }

    private static class SimpleCNN extends Module {
        private Conv2d conv1;
        private Conv2d conv2;
        private MaxPool2d pool1;
        private MaxPool2d pool2;
        private Linear fc;

        /** Closes every intermediate it creates; the caller closes the returned logits. */
        @Override
        public Tensor forward(Tensor input) {
            int batchSize = (int) input.shape().dim(0);
            
            Tensor x = conv1.forward(input);
            x = DemoUtils.consume(x, TensorOps::relu);
            x = DemoUtils.consume(x, pool1::forward);
            
            x = DemoUtils.consume(x, conv2::forward);
            x = DemoUtils.consume(x, TensorOps::relu);
            x = DemoUtils.consume(x, pool2::forward);
            
            int flatSize = (int) x.shape().numElements() / batchSize;
            x = DemoUtils.consume(x, t -> t.reshape(batchSize, flatSize));
            
            return DemoUtils.consume(x, fc::forward);
        }

        public void initialize(ComputeBackend backend, Random random) {
            conv1 = new Conv2d(IN_CHANNELS, OUT_CHANNELS, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1});
            conv1.initialize(backend, random);
            registerModule("conv1", conv1);

            pool1 = new MaxPool2d(new int[]{2, 2});
            registerModule("pool1", pool1);

            conv2 = new Conv2d(OUT_CHANNELS, OUT_CHANNELS * 2, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1});
            conv2.initialize(backend, random);
            registerModule("conv2", conv2);

            pool2 = new MaxPool2d(new int[]{2, 2});
            registerModule("pool2", pool2);

            int finalSize = (IMAGE_SIZE / 4) * (IMAGE_SIZE / 4) * OUT_CHANNELS * 2;
            fc = new Linear(finalSize, NUM_CLASSES);
            fc.initialize(backend, random);
            registerModule("fc", fc);
        }

        public String name() {
            return "SimpleCNN(channels=" + IN_CHANNELS + "->" + (OUT_CHANNELS * 2) + 
                   ", classes=" + NUM_CLASSES + ")";
        }
    }
}
