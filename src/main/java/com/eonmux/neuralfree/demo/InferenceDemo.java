package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Random;

/**
 * Inference throughput of a three-layer MLP. Every forward pass runs under {@link ComputationGraph#noGrad()}, so
 * nothing is recorded although the parameters require grad.
 */
public final class InferenceDemo {

    private static final int INPUT_DIM = 512;
    private static final int HIDDEN_DIM = 1024;
    private static final int OUTPUT_DIM = 256;
    private static final int WARMUP_ITERATIONS = 5;
    private static final int BENCHMARK_ITERATIONS = 100;

    public static void run(ComputeBackend backend) {
        DemoUtils.printHeader("Inference Performance Demo");
        System.out.println("Benchmarking model inference throughput\n");

        Random random = new Random(42);

        InferenceModel model = new InferenceModel();
        model.initialize(backend, random);
        System.out.println("Model: " + model.name());
        System.out.println("Parameters: " + model.parameterCount() + "\n");

        model.eval();

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            float[] inputData = new float[INPUT_DIM];
            for (int i = 0; i < INPUT_DIM; i++) {
                inputData[i] = (float) (random.nextGaussian() * 0.1);
            }
            try (Tensor input = Tensor.fromArray(backend, inputData)) {
                benchmarkSingle(model, input);
            }

            System.out.println("\n--- Batch Inference ---");
            int[] batchSizes = {1, 4, 8, 16, 32};

            System.out.printf("%-12s %-15s %-15s%n", "Batch Size", "Latency (ms)", "Samples/sec");
            System.out.println("-".repeat(42));

            for (int batchSize : batchSizes) {
                float[] batchData = new float[batchSize * INPUT_DIM];
                for (int i = 0; i < batchData.length; i++) {
                    batchData[i] = (float) (random.nextGaussian() * 0.1);
                }
                try (Tensor batchInput = DemoUtils.tensor(backend, batchData, batchSize, INPUT_DIM)) {
                    for (int i = 0; i < 3; i++) {
                        infer(model, batchInput);
                    }

                    long start = System.nanoTime();
                    for (int i = 0; i < 10; i++) {
                        infer(model, batchInput);
                    }
                    long elapsed = System.nanoTime() - start;

                    double avgLatency = (elapsed / 10.0) / 1_000_000.0;
                    double samplesPerSec = (batchSize * 10.0 * 1_000_000_000.0) / elapsed;

                    System.out.printf("%-12d %-15.3f %-15.1f%n", batchSize, avgLatency, samplesPerSec);
                }
            }
        } finally {
            model.close();
        }
    }

    /** Warms up, then times single-sample inference. */
    private static void benchmarkSingle(InferenceModel model, Tensor input) {
        System.out.println("--- Warmup Phase ---");
        System.out.println("Running " + WARMUP_ITERATIONS + " warmup iterations...\n");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            infer(model, input);
        }

        System.out.println("--- Benchmark Phase ---");
        System.out.println("Running " + BENCHMARK_ITERATIONS + " benchmark iterations...\n");

        long totalInferenceTime = 0;
        long minTime = Long.MAX_VALUE;
        long maxTime = Long.MIN_VALUE;

        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            infer(model, input);
            long elapsed = System.nanoTime() - start;

            totalInferenceTime += elapsed;
            minTime = Math.min(minTime, elapsed);
            maxTime = Math.max(maxTime, elapsed);
        }

        double avgTimeMs = (totalInferenceTime / (double) BENCHMARK_ITERATIONS) / 1_000_000.0;
        double minTimeMs = minTime / 1_000_000.0;
        double maxTimeMs = maxTime / 1_000_000.0;
        double throughput = 1000.0 / avgTimeMs;

        System.out.println("Results:");
        System.out.println("  Average latency: " + String.format("%.3f", avgTimeMs) + " ms");
        System.out.println("  Min latency:     " + String.format("%.3f", minTimeMs) + " ms");
        System.out.println("  Max latency:     " + String.format("%.3f", maxTimeMs) + " ms");
        System.out.println("  Throughput:      " + String.format("%.1f", throughput) + " inferences/sec");
    }

    /** One forward pass, read back to the host so the timing covers the device work; the output is closed. */
    private static void infer(InferenceModel model, Tensor input) {
        try (Tensor output = model.forward(input)) {
            output.toFloatArray();
        }
    }

    private static class InferenceModel extends Module {
        private Linear fc1;
        private Linear fc2;
        private Linear fc3;

        /** Closes every intermediate it creates; the caller closes the returned output. */
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

            fc3 = new Linear(HIDDEN_DIM, OUTPUT_DIM);
            fc3.initialize(backend, random);
            registerModule("fc3", fc3);
        }

        public String name() {
            return "InferenceModel(input=" + INPUT_DIM + ", hidden=" + HIDDEN_DIM + 
                   ", output=" + OUTPUT_DIM + ")";
        }
    }
}
