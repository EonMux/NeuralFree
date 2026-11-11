package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Arrays;
import java.util.Random;

/**
 * Runs the Q8_0 batched product in the JVM the tests launch, so its speed and values are observed on a JVM whose
 * options (such as {@code -XX:-UseFMA}) the test chooses. Output lines are {@code engine=} (the kernel
 * description), {@code equal=} (the 64-token product against one matvec per token, bit for bit) and {@code ms=}
 * (the best of three timed products).
 */
public final class TileFmaProbe {

    private TileFmaProbe() {
    }

    public static void main(String[] args) {
        final int rows = 512;
        final int cols = 4096;
        final int tokens = 64;
        Random rnd = new Random(29);
        CpuTensorRef w = CpuTensorRef.of(GGMLType.Q8_0, SimdVecDotTest.randomRows(GGMLType.Q8_0, rows, cols, rnd),
                rows, cols);
        float[] x = SimdVecDotTest.gaussian(tokens * cols, rnd);
        float[] y = new float[tokens * rows];
        float[] ref = new float[tokens * rows];
        for (int t = 0; t < tokens; t++) {
            CpuMatVec.PLAIN.matvec(w, x, t * cols, ref, t * rows);
        }
        long best = Long.MAX_VALUE;
        for (int i = 0; i < 3; i++) {
            long t0 = System.nanoTime();
            CpuMatVec.PLAIN.matmul(w, x, 0, y, 0, tokens);
            best = Math.min(best, System.nanoTime() - t0);
        }
        System.out.println("engine=" + CpuSimd.describe());
        System.out.println("equal=" + Arrays.equals(ref, y));
        System.out.println("ms=" + best / 1_000_000);
    }
}
