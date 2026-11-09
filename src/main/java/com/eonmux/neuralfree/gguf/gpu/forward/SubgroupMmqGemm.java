package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.OpenClKernelLibrary;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * Launch plan of the subgroup-row int8 MMQ GEMM ({@code kernels/llm/matmul_mmq_sg.cl}): {@code {T}_MatMulSg_MMQ},
 * the multi-token product of {@link #TYPES} weights (Q8_0, Q4_K, Q6_K) with the staged Q8_1 activation records of
 * {@code {T}_MatMul_MMQ}, under the same argument list. A work-group of {@link #SUBGROUPS} 16-lane subgroups owns
 * {@link #TILE_ROWS} weight rows by {@link #TILE_TOKENS} tokens: each lane keeps one row's token sums in registers,
 * reads its weights with 16-byte loads straight from the GGUF blocks, and takes every token's activation words from
 * one uniform local-memory block read shared by subgroup shuffles, over {@link #STAGE_DEPTH}-deep stages with one
 * barrier each (a Q8_0 block pair, or a quarter of a K-quant super-block).
 *
 * <p>The form exists only in OpenCL programs built with {@code NF_SG_ROW_SIZE}
 * ({@link OpenClKernelLibrary#subgroupRowSize}: a device that accepts a required 16-lane subgroup, the Intel GPUs)
 * whose compiler offers subgroup block reads of local memory. There the portable kernel's 64-row by 16-token tile
 * spends about 300 instructions and three barriers per 16 dp4a, which left the Arrow-Lake iGPU's 512-token Q8_0
 * projections at 6.0-10.2 ms against llama.cpp Vulkan's 2.3-6.4 ms; the subgroup form issues about 2 instructions
 * per dp4a and holds its 32 accumulators per lane without spilling at SIMD16. The K-quant forms keep the same tile
 * without spilling and measured, at 512 tokens on the same shapes, 1.5-1.9 TFLOPS for Q4_K and 0.8-1.2 TFLOPS for
 * Q6_K against 270-330 and 200-240 GFLOPS for their portable kernels. The dispatcher
 * ({@link FastGemmLanes#subgroupMmqKernel}) gates on the program, the shape and the staging, and the split-K
 * launches keep the portable twin.</p>
 *
 * <p>The constants mirror {@code NF_SGM_SUBGROUPS}, {@code NF_SGM_TN} and {@code NF_SGM_TK} of the kernel source.</p>
 */
final class SubgroupMmqGemm {

    /** Subgroups per work-group ({@code NF_SGM_SUBGROUPS}). */
    static final int SUBGROUPS = 8;
    /** Work-group size: {@link #SUBGROUPS} subgroups of {@link OpenClKernelLibrary#SUBGROUP_ROW_SIZE} lanes. */
    static final int WORK_GROUP = SUBGROUPS * OpenClKernelLibrary.SUBGROUP_ROW_SIZE;
    /** Weight rows per work-group: one per work-item ({@code NF_SGM_TM}). */
    static final int TILE_ROWS = WORK_GROUP;
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
     * Whether the backend's program can hold the subgroup forms: an OpenCL device that accepts the required 16-lane
     * subgroup ({@link OpenClKernelLibrary#subgroupRowSize}). The kernel itself is still looked up by name, since its
     * compiler must also offer local-memory block reads.
     */
    static boolean programBuilds(LlmKernels kernels) {
        return kernels.backend().backendType() == ComputeBackend.BackendType.OPENCL
                && OpenClKernelLibrary.subgroupRowSize(kernels.caps(), kernels.subgroups()) > 0;
    }

    /** Global size along the rows of a launch over {@code rows} weight rows. */
    static long globalRows(int rows) {
        if (rows <= 0) {
            throw new IllegalArgumentException("row count must be positive: " + rows);
        }
        return (long) GpuOps.ceilDiv(rows, TILE_ROWS) * WORK_GROUP;
    }

    /** Work-groups along the tokens of a launch over {@code nTokens} tokens. */
    static long tokenGroups(int nTokens) {
        if (nTokens <= 0) {
            throw new IllegalArgumentException("token count must be positive: " + nTokens);
        }
        return GpuOps.ceilDiv(nTokens, TILE_TOKENS);
    }
}
