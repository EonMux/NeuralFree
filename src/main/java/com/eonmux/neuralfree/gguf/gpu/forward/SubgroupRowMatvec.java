package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.kernels.OpenClKernelLibrary;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * Launch plan of the subgroup-row integer-dot matvec forms ({@code kernels/llm/matvec_sg.cl}):
 * {@code {T}_MatVecSg_Q8}, {@code {T}_MatVecAddSg_Q8} and {@code {T}_MatVecFusedGateUpSg_Q8}, the single-token
 * Q8_1-activation products of {@link #TYPES} with one 16-lane subgroup per {@link #ROWS_PER_SUBGROUP} output rows,
 * 16-byte weight loads and a subgroup reduction instead of the family's one work-group per row.
 *
 * <p>The forms exist in OpenCL programs built with {@code NF_SG_ROW_SIZE}
 * ({@link OpenClKernelLibrary#subgroupRowSize}: a device that accepts a required 16-lane subgroup, the Intel
 * GPUs) and, as the pinned 16-lane modules of {@code shaders/llm/matvec_sg.comp}, on Intel Vulkan devices. The
 * dispatcher gates on {@link SubgroupRowForms#served} before it looks the kernel up, so every other backend and
 * device keeps the family's forms. On the Arrow-Lake iGPU the OpenCL forms stream the weights at
 * 41-61 GB/s (Q8_0, Q4_K and Q6_K alike) against the 5-12 GB/s of the one-work-group-per-row family (its
 * streaming-read ceiling is 55-68 GB/s), because that device's loads are limited by load messages (2-byte
 * loads of 34-byte Q8_0 blocks) rather than by DRAM.</p>
 *
 * <p>Under Vulkan (Mesa ANV) on the same device, a Qwen3 4B Q4_K_M decode step's fused Q4_K gate/up product
 * (2 x 9728 x 2560) takes 417 us against the portable form's 1442 (67 against 19 GB/s) and its Q6_K down product
 * (2560 x 9728) 414 us against 959.</p>
 *
 * <p>The constants mirror {@code NF_SGR_ROWS}, {@code NF_SGR_SUBGROUPS} and {@code NF_SGR_UNIT} of the kernel
 * source.</p>
 */
final class SubgroupRowMatvec {

    /** Output rows per subgroup ({@code NF_SGR_ROWS}). */
    static final int ROWS_PER_SUBGROUP = 2;
    /** Subgroups per work-group ({@code NF_SGR_SUBGROUPS}). */
    static final int SUBGROUPS = 4;
    /** Work-group size: {@link #SUBGROUPS} subgroups of {@link SubgroupRowForms#SIZE} lanes. */
    static final int WORK_GROUP = SUBGROUPS * SubgroupRowForms.SIZE;
    /** Output rows per work-group. */
    static final int ROWS_PER_GROUP = SUBGROUPS * ROWS_PER_SUBGROUP;
    /** Input elements per lane unit ({@code NF_SGR_UNIT}): a Q8_0 block pair, a K-quant quarter block. */
    static final int UNIT = 64;

    /** The weight types with subgroup-row forms. */
    static final Set<GGMLType> TYPES =
            Collections.unmodifiableSet(EnumSet.of(GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K));

    private static final String Q8_SUFFIX = "_Q8";

    private SubgroupRowMatvec() {
    }

    /**
     * The subgroup-row twin of an integer-dot single-token kernel of {@code type}
     * ({@code {T}_MatVec_Q8}, {@code {T}_MatVecAdd_Q8} or {@code {T}_MatVecFusedGateUp_Q8}: the same name with
     * {@code Sg} before {@code _Q8}), or null when the type has no subgroup-row forms or the kernel is not one of
     * those three.
     */
    static String twinName(String int8Kernel, GGMLType type) {
        if (int8Kernel == null || !TYPES.contains(type)) {
            return null;
        }
        String prefix = type.name() + "_";
        String form = int8Kernel.startsWith(prefix) && int8Kernel.endsWith(Q8_SUFFIX)
                ? int8Kernel.substring(prefix.length(), int8Kernel.length() - Q8_SUFFIX.length()) : null;
        if (!"MatVec".equals(form) && !"MatVecAdd".equals(form) && !"MatVecFusedGateUp".equals(form)) {
            return null;
        }
        return prefix + form + "Sg" + Q8_SUFFIX;
    }

    /**
     * Whether rows of {@code cols} elements of {@code type} meet the forms' layout preconditions: whole lane
     * units, and for Q8_0 an even block count so every row starts 4-byte aligned and holds whole block pairs
     * (the K-quant super-blocks are whole units).
     */
    static boolean shapeServes(GGMLType type, int cols) {
        return TYPES.contains(type) && cols > 0 && cols % UNIT == 0 && cols % type.getBlockSize() == 0;
    }

    /** Global size of a launch over {@code rows} output rows. */
    static long globalSize(int rows) {
        if (rows <= 0) {
            throw new IllegalArgumentException("row count must be positive: " + rows);
        }
        return (long) GpuOps.ceilDiv(rows, ROWS_PER_GROUP) * WORK_GROUP;
    }
}
