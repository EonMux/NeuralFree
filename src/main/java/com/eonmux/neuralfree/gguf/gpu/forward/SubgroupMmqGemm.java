package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.kernels.OpenClKernelLibrary;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * Launch plan of the subgroup-row int8 MMQ GEMM ({@code kernels/llm/matmul_mmq_sg.cl}): {@code {T}_MatMulSg_MMQ},
 * the multi-token product of {@link #TYPES} weights (Q8_0, Q4_K, Q6_K) with the staged Q8_1 activation records of
 * {@code {T}_MatMul_MMQ}, under the same argument list. A work-group of 16-lane subgroups owns {@link #TILE_ROWS}
 * weight rows by {@link #TILE_TOKENS} tokens: each lane keeps one row's token sums in registers,
 * reads its weights with 16-byte loads straight from the GGUF blocks, and takes every token's activation words from
 * one uniform local-memory block read shared by subgroup shuffles, over {@link #STAGE_DEPTH}-deep stages with one
 * barrier each (a Q8_0 block pair, or a quarter of a K-quant super-block).
 *
 * <p>The form exists where {@link SubgroupRowForms#served} holds: in OpenCL programs built with
 * {@code NF_SG_ROW_SIZE} ({@link OpenClKernelLibrary#subgroupRowSize}: a device that accepts a required 16-lane
 * subgroup, the Intel GPUs) whose compiler offers subgroup block reads of local memory, and on the Intel Vulkan
 * devices described below. Under OpenCL the portable kernel's 64-row by 16-token tile
 * spends about 300 instructions and three barriers per 16 dp4a, which left the Arrow-Lake iGPU's 512-token Q8_0
 * projections at 6.0-10.2 ms against llama.cpp Vulkan's 2.3-6.4 ms; the subgroup form issues about 2 instructions
 * per dp4a and holds its 32 accumulators per lane without spilling at SIMD16. The K-quant forms keep the same tile
 * without spilling and measured, at 512 tokens on the same shapes, 1.5-1.9 TFLOPS for Q4_K and 0.8-1.2 TFLOPS for
 * Q6_K against 270-330 and 200-240 GFLOPS for their portable kernels. The dispatcher
 * ({@link FastGemmLanes#subgroupMmqKernel}) gates on the program, the shape and the staging, and the split-K
 * launches keep the portable twin.</p>
 *
 * <p>On Vulkan the form is the pinned 16-lane module of {@code shaders/llm/matmul_mmq_sg.comp}
 * ({@link SubgroupRowForms}: the Intel GPUs), which hands each token's activation words to the lanes by
 * {@code subgroupBroadcast} of one shared-memory load and splits a work-group's tokens between two halves
 * ({@link #VULKAN_TOKEN_SPLIT}). On the Arrow-Lake iGPU
 * (Mesa ANV) the portable GLSL kernel took 42.8 ms for a 512-token Q8_0 projection of 3072 x 1024, the form 1.86 ms
 * (1.7 TFLOPS); Q4_K runs 9728 x 2560 in 15.6 ms (1.6 TFLOPS) and Q6_K 2560 x 9728 in 25.7 ms (1.0), the rates
 * llama.cpp's Vulkan MMQ reaches on the same shapes and device.</p>
 *
 * <p>The constants mirror {@code NF_SGM_TM}, {@code NF_SGM_TN}, {@code NF_SGM_TK} and {@code NF_SGM_SPLIT} of the
 * kernel sources.</p>
 */
final class SubgroupMmqGemm {

    /** Weight rows per work-group ({@code NF_SGM_TM}): each work-item owns one. */
    static final int TILE_ROWS = 128;
    /**
     * Work-items per weight row of the Vulkan Q8_0 and Q4_K forms ({@code NF_SGM_SPLIT} of
     * shaders/llm/matmul_mmq_sg.comp): the GLSL form splits a work-group's tokens between two halves of
     * {@link #TILE_ROWS} work-items, so each keeps 16 token sums instead of 32. Mesa's Intel compiler gives a SIMD16
     * thread 128 registers, in which the 32-sum Q4_K tile spilled 76 of them; split, the Arrow-Lake iGPU's 512-token
     * projections took 1.15-1.86 ms (Q8_0, Qwen3 0.6B) against 1.43-2.41 and 15.6 ms (Q4_K, 9728 x 2560) against
     * 16.9. Q6_K keeps one work-item per row: its heavier block decode, repeated by each half, cost more than the
     * spills it saved (27.1 ms against 25.7 on 2560 x 9728).
     */
    static final int VULKAN_TOKEN_SPLIT = 2;
    /** Tokens per work-group ({@code NF_SGM_TN}). */
    static final int TILE_TOKENS = 32;
    /**
     * Reduction depth of one stage ({@code NF_SGM_TK}): a Q8_0 block pair, a quarter of a K-quant super-block. The
     * row length must hold whole stages and whole weight blocks ({@link #shapeServes}).
     */
    static final int STAGE_DEPTH = 64;
    /** Depth of the staged activation records the kernel reads ({@code NF_MMQ_TK} it is built for). */
    static final int RECORD_DEPTH = 16;

    /** The weight types with a subgroup MMQ form. */
    static final Set<GGMLType> TYPES =
            Collections.unmodifiableSet(EnumSet.of(GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K));

    private static final String SUFFIX = "_MatMulSg_MMQ";

    private SubgroupMmqGemm() {
    }

    /** The subgroup MMQ kernel of a weight type, or null when the type has no subgroup form. */
    static String kernel(GGMLType type) {
        return TYPES.contains(type) ? type.name() + SUFFIX : null;
    }

    /**
     * Whether rows of {@code cols} elements of {@code type} meet the kernel's layout: whole stages of whole blocks, a
     * multiple of 64 for Q8_0 (its 4-byte-aligned block pairs) and of 256 for the K-quants (their super-blocks).
     */
    static boolean shapeServes(GGMLType type, long cols) {
        if (!TYPES.contains(type) || cols <= 0) {
            return false;
        }
        return cols % Math.max(STAGE_DEPTH, type.getBlockSize()) == 0;
    }

    /**
     * Whether the backend's kernels can hold the subgroup forms ({@link SubgroupRowForms#served}: an OpenCL program
     * built for a required 16-lane subgroup, or an Intel Vulkan device whose subgroup range includes 16 lanes). The
     * kernel itself is still looked up by name, since the OpenCL compiler must also offer local-memory block reads and
     * the Vulkan device must pin the subgroup width for compute pipelines.
     */
    static boolean programBuilds(LlmKernels kernels) {
        return SubgroupRowForms.served(kernels);
    }

    /** Work-group size of the form of {@code type} on a backend: {@link #TILE_ROWS} work-items per token split. */
    static int workGroup(BackendType backend, GGMLType type) {
        boolean split = backend == BackendType.VULKAN && type != GGMLType.Q6_K;
        return TILE_ROWS * (split ? VULKAN_TOKEN_SPLIT : 1);
    }

    /** Work-group size of the form of {@code type} on {@code kernels}' backend. */
    static int workGroup(LlmKernels kernels, GGMLType type) {
        return workGroup(kernels.backend().backendType(), type);
    }

    /** Global size along the rows of a launch over {@code rows} weight rows in work-groups of {@code workGroup}. */
    static long globalRows(int rows, int workGroup) {
        if (rows <= 0) {
            throw new IllegalArgumentException("row count must be positive: " + rows);
        }
        return (long) GpuOps.ceilDiv(rows, TILE_ROWS) * workGroup;
    }

    /** Work-groups along the tokens of a launch over {@code nTokens} tokens. */
    static long tokenGroups(int nTokens) {
        if (nTokens <= 0) {
            throw new IllegalArgumentException("token count must be positive: " + nTokens);
        }
        return GpuOps.ceilDiv(nTokens, TILE_TOKENS);
    }
}
