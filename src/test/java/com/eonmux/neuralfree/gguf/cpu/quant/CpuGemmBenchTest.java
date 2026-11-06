package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.Assume;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

/**
 * Measures the batched CPU matmul kernels (the prefill path) so a tile or kernel change can be compared against
 * the one it replaces. Runs only when {@code NF_GEMM_BENCH} is set in the environment, because it is a
 * measurement and not a check; the numbers are written to stdout and land in the test XML report.
 */
public class CpuGemmBenchTest {

    /** Output rows of the measured product. */
    private static final int ROWS = 2560;
    /** Input columns of the measured product. */
    private static final int COLS = 2560;
    /** Activation rows of the measured product. */
    private static final int TOKENS = 64;

    private static final GGMLType[] TYPES = {
            GGMLType.Q4_0, GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K, GGMLType.Q5_K,
            GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.IQ4_NL, GGMLType.IQ4_XS, GGMLType.F16,
    };

    @Test
    public void bench() {
        Assume.assumeTrue("set NF_GEMM_BENCH to measure", System.getenv("NF_GEMM_BENCH") != null);
        Assume.assumeTrue(CpuSimd.available());
        Random rnd = new Random(7);
        System.out.println("type,backend,threads,gmacs,ms");
        for (GGMLType t : TYPES) {
            CpuTensorRef w = tensor(t, rnd);
            QuantizedActivations[] acts = activations(t, rnd);
            CpuVecDot kernel = CpuMatVec.vecDot(t);
            float[] y = new float[TOKENS * ROWS];
            // single-threaded: one whole product per iteration
            double best = Double.MAX_VALUE;
            for (int it = 0; it < 8; it++) {
                long t0 = System.nanoTime();
                kernel.dotRows(w, 0, ROWS, acts, TOKENS, y, 0, ROWS);
                long t1 = System.nanoTime();
                if (it >= 2) {
                    best = Math.min(best, (t1 - t0) / 1e6);
                }
            }
            System.out.printf("%s,%s,1,%.2f,%.2f%n", t, kernel, gmacs(best), best);
            CpuVecDot scalar = scalarFor(t);
            double sbest = Double.MAX_VALUE;
            for (int it = 0; it < 6; it++) {
                long t0 = System.nanoTime();
                scalar.dotRows(w, 0, ROWS, acts, TOKENS, y, 0, ROWS);
                long t1 = System.nanoTime();
                if (it >= 2) {
                    sbest = Math.min(sbest, (t1 - t0) / 1e6);
                }
            }
            System.out.printf("%s,%s,1,%.2f,%.2f%n", t, scalar, gmacs(sbest), sbest);
        }
    }

    @Test
    public void benchThreads() {
        Assume.assumeTrue("set NF_GEMM_BENCH to measure", System.getenv("NF_GEMM_BENCH") != null);
        Assume.assumeTrue(CpuSimd.available());
        Random rnd = new Random(11);
        System.out.println("type,backend,threads,gmacs,ms");
        CpuMatVec plain = new CpuMatVec(w -> CpuMatVec.Transforms.NONE);
        for (GGMLType t : TYPES) {
            CpuTensorRef w = tensor(t, rnd);
            float[] x = new float[TOKENS * COLS];
            for (int i = 0; i < x.length; i++) {
                x[i] = (float) rnd.nextGaussian();
            }
            float[] y = new float[TOKENS * ROWS];
            double best = Double.MAX_VALUE;
            for (int it = 0; it < 8; it++) {
                long t0 = System.nanoTime();
                plain.matmul(w, x, 0, COLS, y, 0, ROWS, TOKENS);
                long t1 = System.nanoTime();
                if (it >= 2) {
                    best = Math.min(best, (t1 - t0) / 1e6);
                }
            }
            System.out.printf("%s,CpuMatVec,%d,%.2f,%.2f%n", t, CpuParallelThreads.get(), gmacs(best), best);
        }
    }

    /** Millions of multiply-accumulates of one whole product, divided by a time in milliseconds. */
    private static double gmacs(double ms) {
        return (double) ROWS * COLS * TOKENS / 1e6 / ms;
    }

    private static CpuVecDot scalarFor(GGMLType t) {
        if (LegacyVecDot.supports(t)) {
            return LegacyVecDot.forType(t);
        }
        if (Q8VecDot.supports(t)) {
            return Q8VecDot.forType(t);
        }
        if (KQuantVecDot.supports(t)) {
            return KQuantVecDot.forType(t);
        }
        if (Fp16VecDot.supports(t)) {
            return Fp16VecDot.forType(t);
        }
        if (IQuantVecDot.supports(t)) {
            return IQuantVecDot.forType(t);
        }
        if (Fp4VecDot.supports(t)) {
            return Fp4VecDot.forType(t);
        }
        if (NewFpVecDot.supports(t)) {
            return NewFpVecDot.forType(t);
        }
        if (ExtVecDot.supports(t)) {
            return ExtVecDot.forType(t);
        }
        return TernaryVecDot.forType(t);
    }

    private static QuantizedActivations[] activations(GGMLType t, Random rnd) {
        QuantizedActivations[] acts = new QuantizedActivations[TOKENS];
        for (int i = 0; i < TOKENS; i++) {
            float[] x = new float[COLS];
            for (int j = 0; j < COLS; j++) {
                x[j] = (float) rnd.nextGaussian();
            }
            acts[i] = CpuActivationQuantizer.quantizeFor(t, x, 0, COLS);
        }
        return acts;
    }

    /** Random weights of {@code t} with well-formed scales, one row per weight row. */
    private static CpuTensorRef tensor(GGMLType t, Random rnd) {
        BlockLayout layout = BlockLayout.of(t);
        int blocks = ROWS * (COLS / layout.blockElements());
        byte[] raw = new byte[blocks * layout.blockBytes()];
        rnd.nextBytes(raw);
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < blocks; i++) {
            int o = i * layout.blockBytes();
            for (int f : halfFields(t)) {
                float v = (rnd.nextFloat() * 2f - 1f) * 0.1f;
                b.putShort(o + f, (short) FpBlockDecoders.fp32ToFp16(v));
            }
            if (t == GGMLType.Q6_K) {
                for (int g = 0; g < 16; g++) {
                    raw[o + BlockLayout.Q6_K_SCALES + g] = (byte) (rnd.nextInt(255) - 127);
                }
            }
        }
        if (t == GGMLType.F16) {
            for (int i = 0; i < ROWS * COLS; i++) {
                b.putShort(2 * i, (short) FpBlockDecoders.fp32ToFp16((rnd.nextFloat() * 2f - 1f) * 0.1f));
            }
        }
        return CpuTensorRef.of(t, b, ROWS, COLS);
    }

    private static int[] halfFields(GGMLType t) {
        switch (t) {
            case Q4_0:
                return new int[]{BlockLayout.Q4_0_D};
            case Q4_1:
                return new int[]{BlockLayout.Q4_1_D, BlockLayout.Q4_1_M};
            case Q5_0:
                return new int[]{BlockLayout.Q5_0_D};
            case Q5_1:
                return new int[]{BlockLayout.Q5_1_D, BlockLayout.Q5_1_M};
            case Q8_0:
                return new int[]{BlockLayout.Q8_0_D};
            case Q2_K:
                return new int[]{BlockLayout.Q2_K_D, BlockLayout.Q2_K_DMIN};
            case Q3_K:
                return new int[]{BlockLayout.Q3_K_D};
            case Q4_K:
                return new int[]{BlockLayout.Q4_K_D, BlockLayout.Q4_K_DMIN};
            case Q5_K:
                return new int[]{BlockLayout.Q5_K_D, BlockLayout.Q5_K_DMIN};
            case Q6_K:
                return new int[]{BlockLayout.Q6_K_D};
            case IQ4_NL:
                return new int[]{BlockLayout.IQ4_NL_D};
            case IQ4_XS:
                return new int[]{BlockLayout.IQ4_XS_D};
            default:
                return new int[0];
        }
    }

    /** The pool size the matmul dispatch uses, without starting a second pool. */
    private static final class CpuParallelThreads {
        static int get() {
            return com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel.parallelism();
        }
    }
}
