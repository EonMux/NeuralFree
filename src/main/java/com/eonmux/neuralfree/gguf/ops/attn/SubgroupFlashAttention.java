package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.kernels.OpenClKernelLibrary;
import com.eonmux.neuralfree.gguf.gpu.forward.SubgroupRowForms;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;

/**
 * The subgroup flash attention of {@code kernels/llm/flash_decode_sg.cl} and {@code kernels/llm/flash_prefill_sg.cl}
 * for plain F32, F16 and BF16 caches and Q8_0 caches: the decode kernels {@code LLM_FlashDecodeSg_<KV>} /
 * {@code LLM_FlashDecodePagedSg_<KV>} and the prefill kernels {@code LLM_FlashAttnSg_<KV>} /
 * {@code LLM_FlashAttnPagedSg_<KV>}. Each takes the arguments of its portable counterpart ({@code LLM_FlashDecode*},
 * {@code LLM_FlashAttn*}) and writes the same outputs and partials, with work-groups of {@link #SUBGROUPS} 16-lane
 * subgroups that keep their running state per lane ({@link #DIMS_PER_LANE} dims of the output per lane; in the
 * Intel runtime build the decode kernels hold it in registers, the prefill kernels spill part of it; a Q8_0 lane
 * reads its 8 codes and their block's scale):
 * <ul>
 *   <li>decode: one query head per work-group; each subgroup scans every {@link #SUBGROUPS}-th key of the split and
 *       the subgroups merge through about 2 KB of local memory;</li>
 *   <li>prefill: {@link #PREFILL_ROWS} query rows of one KV head per work-group, {@link #ROWS_PER_SUBGROUP} per
 *       subgroup; a lane scores one key of a 16-key block against both rows (full dot products in the lane, the
 *       rows read as uniform values from 4 KB of local memory) and accumulates its output dims from the block's V
 *       rows.</li>
 * </ul>
 *
 * <p>The kernels exist in OpenCL programs built with {@code NF_SG_ROW_SIZE}
 * ({@link OpenClKernelLibrary#subgroupRowSize}: a device that accepts a required 16-lane subgroup, the Intel GPUs)
 * and, as the pinned 16-lane modules of {@code shaders/llm/flash_decode_sg.comp} and
 * {@code shaders/llm/flash_prefill_sg.comp}, on Intel Vulkan devices ({@link SubgroupRowForms}). On OpenCL the
 * portable kernels' {@code NF_FA_SMEM_FLOATS}-float staging holds about one work-group per Xe-core, and
 * their 128-item work-groups compile at SIMD8 and spill: on the Arrow-Lake iGPU the decode attention took 106-252 us
 * per layer against llama.cpp Vulkan's 56-149 us, and the 512-token prefill attention of a 16-head layer 417 ms
 * against its 10.7 ms. {@link FlashPlan} picks the forms for a layer ({@link #serves}) and its dense and paged plans
 * then all use them, so a single-row step and a paged batch row over the same keys keep one summation order.</p>
 *
 * <p>Under Vulkan (Mesa ANV) on the same device, the 512-token prefill attention of a Qwen3-Embedding 0.6B layer
 * takes 2.4-2.8 ms against the portable GLSL kernel's 210 ms, and of a Qwen3-Embedding 4B layer (32 heads) 4.3-4.5 ms
 * against llama.cpp Vulkan's 8.3 ms; a Qwen3-Embedding 4B decode step's attention takes 16 us per layer against
 * 990.</p>
 *
 * <p>The constants mirror {@code NF_FDS_SUBGROUPS}, {@code NF_FDS_DPL} and {@code NF_FAS_ROWS} of the kernel
 * sources.</p>
 */
final class SubgroupFlashAttention {

    /** Subgroups per work-group ({@code NF_FDS_SUBGROUPS}). */
    static final int SUBGROUPS = 4;
    /** Head dims each lane holds ({@code NF_FDS_DPL}). */
    static final int DIMS_PER_LANE = 8;
    /** Work-group size: {@link #SUBGROUPS} subgroups of {@link SubgroupRowForms#SIZE} lanes. */
    static final int WORK_GROUP = SUBGROUPS * SubgroupRowForms.SIZE;
    /** Largest head dim: one subgroup's lanes hold the whole row ({@code NF_FDS_MAX_D}). */
    static final int MAX_HEAD_DIM = DIMS_PER_LANE * SubgroupRowForms.SIZE;
    /** Query rows per prefill subgroup ({@code NF_FAS_ROWS}). */
    static final int ROWS_PER_SUBGROUP = 2;
    /** Query rows per prefill work-group ({@code NF_FAS_WG_ROWS}): {@code bq * g_tile} of the prefill plan. */
    static final int PREFILL_ROWS = SUBGROUPS * ROWS_PER_SUBGROUP;
    /** Keys per prefill block: one per lane ({@code NF_SG_ROW_SIZE}); the prefill plan's {@code bk}. */
    static final int PREFILL_BLOCK = SubgroupRowForms.SIZE;

    /** Name prefix of the {@code LLM_FlashDecodeSg_<KV>} variants. */
    static final String PREFIX = "LLM_FlashDecodeSg_";
    /** Name prefix of the {@code LLM_FlashDecodePagedSg_<KV>} variants. */
    static final String PAGED_PREFIX = "LLM_FlashDecodePagedSg_";
    /** Name prefix of the {@code LLM_FlashAttnSg_<KV>} variants. */
    static final String PREFILL_PREFIX = "LLM_FlashAttnSg_";
    /** Name prefix of the {@code LLM_FlashAttnPagedSg_<KV>} variants. */
    static final String PAGED_PREFILL_PREFIX = "LLM_FlashAttnPagedSg_";

    private SubgroupFlashAttention() {
    }

    /**
     * Whether the subgroup kernels serve a layer with these caches and head dims on {@code kernels}' device: a
     * backend with the pinned subgroup forms ({@link SubgroupRowForms#served}), equal F32, F16, BF16 or Q8_0 K and V
     * caches, equal head dims that are positive multiples of {@link #DIMS_PER_LANE} (of a Q8_0 block for Q8_0) of at
     * most {@link #MAX_HEAD_DIM}, and the four kernels of the type (dense and paged decode and prefill) present.
     */
    static boolean serves(LlmKernels kernels, KvDType typeK, KvDType typeV, int dK, int dV) {
        return SubgroupRowForms.served(kernels)
                && shapeServes(typeK, typeV, dK, dV)
                && kernels.has(kernel(typeK))
                && kernels.has(pagedKernel(typeK))
                && kernels.has(prefillKernel(typeK))
                && kernels.has(pagedPrefillKernel(typeK));
    }

    /** The cache-type and head-dim conditions of {@link #serves}, which the kernels' plan check repeats. */
    static boolean shapeServes(KvDType typeK, KvDType typeV, int dK, int dV) {
        return typeK == typeV
                && (typeK == KvDType.F32 || typeK == KvDType.F16 || typeK == KvDType.BF16 || typeK == KvDType.Q8_0)
                && dK == dV && dK > 0 && dK % strideUnit(typeK) == 0 && dK <= MAX_HEAD_DIM;
    }

    /**
     * Whether K and V cache rows of these strides (elements) start where the kernels' loads need them: on whole
     * {@link #strideUnit} multiples, as the kernels' plan check requires of every row and head offset.
     */
    static boolean stridesServe(KvDType type, int kRowStride, int vRowStride) {
        int unit = strideUnit(type);
        return kRowStride % unit == 0 && vRowStride % unit == 0;
    }

    /** The element alignment of the kernels' row loads: {@link #DIMS_PER_LANE}, a whole 32-value block for Q8_0. */
    static int strideUnit(KvDType type) {
        return type == KvDType.Q8_0 ? 32 : DIMS_PER_LANE;
    }

    /** The dense decode kernel of a cache type. */
    static String kernel(KvDType kv) {
        return kv.kernel(PREFIX);
    }

    /** The paged decode kernel of a cache type. */
    static String pagedKernel(KvDType kv) {
        return kv.kernel(PAGED_PREFIX);
    }

    /** The dense prefill kernel of a cache type. */
    static String prefillKernel(KvDType kv) {
        return kv.kernel(PREFILL_PREFIX);
    }

    /** The paged prefill kernel of a cache type. */
    static String pagedPrefillKernel(KvDType kv) {
        return kv.kernel(PAGED_PREFILL_PREFIX);
    }
}
