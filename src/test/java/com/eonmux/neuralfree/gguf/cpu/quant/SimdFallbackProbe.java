package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

/**
 * Prints the CPU kernel selection of the JVM the tests launch, so the scalar fallback is observable in a child
 * process whose incubator module and property gate the test chooses per run. Output lines are
 * {@code available=}, {@code describe=}, {@code reason=}, {@code status=}, {@code q8_0=}, {@code q4_0=},
 * {@code q4_1=}, {@code q2_K=}, {@code f32=}, {@code q1_0=}, {@code mxfp4=} and {@code repacked_q8=}, or
 * {@code rejected=} when a query throws.
 */
public final class SimdFallbackProbe {

    private SimdFallbackProbe() {
    }

    public static void main(String[] args) {
        try {
            System.out.println("available=" + CpuSimd.available());
            System.out.println("describe=" + CpuSimd.describe());
            System.out.println("reason=" + CpuSimd.reason());
            System.out.println("status=" + CpuSimd.status());
            System.out.println("q8_0=" + CpuMatVec.vecDot(GGMLType.Q8_0));
            System.out.println("q4_0=" + CpuMatVec.vecDot(GGMLType.Q4_0));
            System.out.println("q4_1=" + CpuMatVec.vecDot(GGMLType.Q4_1));
            System.out.println("q2_K=" + CpuMatVec.vecDot(GGMLType.Q2_K));
            System.out.println("f32=" + CpuMatVec.vecDot(GGMLType.F32));
            System.out.println("q1_0=" + CpuMatVec.vecDot(GGMLType.Q1_0));
            System.out.println("mxfp4=" + CpuMatVec.vecDot(GGMLType.MXFP4));
            System.out.println("repacked_q8=" + (CpuSimd.repacked(GGMLType.Q8_0) == null
                    ? "none" : CpuSimd.repacked(GGMLType.Q8_0)));
        } catch (RuntimeException e) {
            System.out.println("rejected=" + e.getClass().getSimpleName() + ":" + e.getMessage());
        }
    }
}
