package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.loss.CrossEntropyLoss;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.training.Trainer;
import com.eonmux.neuralfree.training.TrainerConfig;
import com.eonmux.neuralfree.training.TrainingResult;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.List;
import java.util.Map;
import java.util.Random;

public final class MLPDemo {

    private static final int INPUT_DIM = 20;
    private static final int HIDDEN_DIM = 64;
    private static final int NUM_CLASSES = 5;
    private static final int NUM_SAMPLES = 500;
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 10;

    public static void run(ComputeBackend backend) {
        DemoUtils.printHeader("MLP Classification Demo");
        System.out.println("Training a simple MLP classifier on synthetic data\n");

        Random random = new Random(42);

        SimpleMLP model = new SimpleMLP();
        model.initialize(backend, random);
        System.out.println("Model: " + model.name());
        System.out.println("Parameters: " + model.parameterCount() + "\n");

        Tensor[] inputs = new Tensor[NUM_SAMPLES];
        Tensor[] targets = new Tensor[NUM_SAMPLES];
        Optimizer optimizer = null;
        try {
            for (int i = 0; i < NUM_SAMPLES; i++) {
                float[] data = DemoUtils.generateRandomData(INPUT_DIM, random);
                inputs[i] = Tensor.fromArray(backend, data);

                int label = random.nextInt(NUM_CLASSES);
                float[] oneHot = new float[NUM_CLASSES];
                oneHot[label] = 1.0f;
                targets[i] = Tensor.fromArray(backend, oneHot);
            }

            TensorDataset dataset = new TensorDataset(inputs, targets);
            DataLoader<TensorDataset.Sample> trainLoader = new DataLoader<>(dataset, BATCH_SIZE, true);

            optimizer = new FirstStepGradientNorm(new Adam(model.parameters(), 0.01f));
            CrossEntropyLoss lossFn = new CrossEntropyLoss();

            try (Trainer trainer = new Trainer(model, optimizer, lossFn, trainLoader, null,
                    List.of(), null, TrainerConfig.defaults())) {
                System.out.println("Starting training...");
                long start = System.currentTimeMillis();
                TrainingResult result = trainer.train(EPOCHS);
                long elapsed = System.currentTimeMillis() - start;

                System.out.println("\nTraining completed!");
                System.out.println("Time: " + elapsed + "ms");
                System.out.println("Final loss: " + String.format("%.4f",
                    result.trainLosses().get(result.trainLosses().size() - 1)));
                System.out.println("Total steps: " + result.totalSteps());
            } finally {
                trainLoader.close();
            }

            System.out.println("\nRunning inference...");
            model.eval();
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor testInput = DemoUtils.tensor(backend,
                    DemoUtils.generateRandomData(INPUT_DIM, random), 1, INPUT_DIM)) {
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
            if (optimizer != null) {
                optimizer.close();
            }
            DemoUtils.closeAll(inputs);
            DemoUtils.closeAll(targets);
            model.close();
        }
    }

    /**
     * The optimizer the {@link Trainer} steps: delegates every call to {@code delegate} and, before the first
     * {@link #step()}, prints the global gradient norm of the parameters it updates
     * ({@link DemoUtils#printGradientNorm}), the gradients that first step applies, so a loss that reaches no
     * parameter is visible. Closing it closes {@code delegate}.
     */
    private static final class FirstStepGradientNorm extends Optimizer {
        private final Optimizer delegate;
        private boolean reported;

        FirstStepGradientNorm(Optimizer delegate) {
            super(delegate.parameters(), delegate.defaults());
            this.delegate = delegate;
        }

        @Override
        public void step() {
            if (!reported) {
                reported = true;
                DemoUtils.printGradientNorm("First step", delegate.parameters());
            }
            delegate.step();
        }

        @Override
        public void zeroGrad() {
            delegate.zeroGrad();
        }

        @Override
        public List<Parameter> parameters() {
            return delegate.parameters();
        }

        @Override
        public Map<String, Object> defaults() {
            return delegate.defaults();
        }

        @Override
        public int stepCount() {
            return delegate.stepCount();
        }

        @Override
        public Map<String, Object> stateDict() {
            return delegate.stateDict();
        }

        @Override
        public void loadStateDict(Map<String, ?> state) {
            delegate.loadStateDict(state);
        }

        @Override
        public float getLearningRate() {
            return delegate.getLearningRate();
        }

        @Override
        public void setLearningRate(float lr) {
            delegate.setLearningRate(lr);
        }

        @Override
        public void close() {
            try {
                delegate.close();
            } finally {
                super.close();
            }
        }

        @Override
        public String toString() {
            return delegate.toString();
        }
    }

    private static class SimpleMLP extends Module {
        private Linear fc1;
        private Linear fc2;
        private Linear fc3;

        /** Closes every intermediate it creates; the caller closes the returned logits. */
        @Override
        public Tensor forward(Tensor input) {
            Tensor x = DemoUtils.consume(fc1.forward(input), TensorOps::relu);
            x = DemoUtils.consume(x, fc2::forward);
            x = DemoUtils.consume(x, TensorOps::relu);
            return DemoUtils.consume(x, fc3::forward);
        }

        public void initialize(ComputeBackend backend, Random random) {
            fc1 = new Linear(INPUT_DIM, HIDDEN_DIM);
            fc1.initialize(backend, random);
            registerModule("fc1", fc1);

            fc2 = new Linear(HIDDEN_DIM, HIDDEN_DIM);
            fc2.initialize(backend, random);
            registerModule("fc2", fc2);

            fc3 = new Linear(HIDDEN_DIM, NUM_CLASSES);
            fc3.initialize(backend, random);
            registerModule("fc3", fc3);
        }

        public String name() {
            return "SimpleMLP(INPUT_DIM=" + INPUT_DIM + ", HIDDEN=" + HIDDEN_DIM + 
                   ", OUTPUT=" + NUM_CLASSES + ")";
        }
    }
}
