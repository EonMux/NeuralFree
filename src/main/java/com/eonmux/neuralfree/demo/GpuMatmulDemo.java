package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainKernels;

import java.util.Arrays;
import java.util.Random;

/**
 * Times {@link TensorOps#matmul} on 2-D, batched 3-D and broadcast N-D FLOAT32 operands. Matmul runs on the train
 * library's GEMM whenever the library is available on the backend, at every size; otherwise it takes its CPU path.
 */
public final class GpuMatmulDemo {

    private static final int WARMUP_ITERATIONS = 3;
    private static final int BENCHMARK_ITERATIONS = 10;

    private GpuMatmulDemo() {}

    public static void run(ComputeBackend backend) {
        DemoUtils.printHeader("GPU-Accelerated N-Dimensional Matmul Demo");

        System.out.println("Backend: " + backend.backendType());
        System.out.println("Device: " + backend.capabilities().getDeviceName());
        System.out.println("Compute Units: " + backend.capabilities().getComputeUnits());
        Availability train = TrainKernels.availability(backend);
        System.out.println("Matmul path: " + (train.available()
                ? "GPU (train library GEMM)"
                : "CPU (train library unavailable: " + train.reason() + ")"));
        System.out.println();

        System.out.println("=== Matrix Multiplication Benchmarks ===\n");

        benchmark2D(backend, 128, 128, 128);
        benchmark2D(backend, 256, 256, 256);
        benchmark2D(backend, 512, 256, 512);

        System.out.println("\n=== Batched 3D Matrix Multiplication ===\n");

        benchmarkBatched3D(backend, 8, 64, 64, 64);
        benchmarkBatched3D(backend, 16, 64, 64, 64);
        benchmarkBatched3D(backend, 32, 32, 32, 32);

        System.out.println("\n=== N-Dimensional Batch Broadcasting ===\n");

        benchmarkND(backend, new int[]{4, 4}, 32, 32, 32);
        benchmarkND(backend, new int[]{2, 4, 4}, 32, 32, 32);

        System.out.println("\n=== Large-Scale Matrix Multiplication ===\n");

        benchmark2D(backend, 1024, 1024, 1024);
    }

    private static void benchmark2D(ComputeBackend backend, int M, int N, int K) {
        System.out.printf("2D Matmul: (%d x %d) @ (%d x %d)%n", M, K, K, N);
        benchmark(backend, new int[]{M, K}, new int[]{K, N}, 2.0 * M * N * K);
    }

    private static void benchmarkBatched3D(ComputeBackend backend, int batch, int M, int N, int K) {
        System.out.printf("Batched 3D Matmul: (%d x %d x %d) @ (%d x %d x %d)%n",
            batch, M, K, batch, K, N);
        benchmark(backend, new int[]{batch, M, K}, new int[]{batch, K, N}, 2.0 * batch * M * N * K);
    }

    private static void benchmarkND(ComputeBackend backend, int[] batchDims, int M, int N, int K) {
        int batchSize = 1;
        for (int d : batchDims) {
            batchSize *= d;
        }
        String batchStr = Arrays.toString(batchDims);
        System.out.printf("N-D Matmul: %s @ (%d x %d) @ (%d x %d) -> %s @ (%d x %d)%n",
            batchStr, M, K, K, N, batchStr, M, N);
        benchmark(backend, withMatrix(batchDims, M, K), withMatrix(batchDims, K, N), 2.0 * batchSize * M * N * K);
    }

    /** {@code batchDims} followed by {@code rows, cols}. */
    private static int[] withMatrix(int[] batchDims, int rows, int cols) {
        int[] shape = Arrays.copyOf(batchDims, batchDims.length + 2);
        shape[batchDims.length] = rows;
        shape[batchDims.length + 1] = cols;
        return shape;
    }

    /**
     * Times {@code a @ b} for random operands of the given shapes. Each product is read back to the host, so a
     * timing includes the device work and the readback.
     */
    private static void benchmark(ComputeBackend backend, int[] aShape, int[] bShape, double flops) {
        Random random = new Random(42);
        try (Tensor a = DemoUtils.tensor(backend, uniform(random, aShape), aShape);
             Tensor b = DemoUtils.tensor(backend, uniform(random, bShape), bShape)) {
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                try (Tensor warmup = TensorOps.matmul(a, b)) {
                    warmup.toFloatArray();
                }
            }

            long totalNanos = 0;
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                long start = System.nanoTime();
                try (Tensor result = TensorOps.matmul(a, b)) {
                    result.toFloatArray();
                }
                totalNanos += System.nanoTime() - start;
            }

            double avgMs = totalNanos / (double) BENCHMARK_ITERATIONS / 1_000_000.0;
            double gflops = flops / (avgMs / 1000.0) / 1e9;

            System.out.printf("  Avg Time: %.2f ms%n", avgMs);
            System.out.printf("  Throughput: %.2f GFLOPS%n%n", gflops);
        }
    }

    /** Values uniform in [-1, 1) for a tensor of the given shape. */
    private static float[] uniform(Random random, int[] shape) {
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = random.nextFloat() * 2 - 1;
        }
        return data;
    }
}
