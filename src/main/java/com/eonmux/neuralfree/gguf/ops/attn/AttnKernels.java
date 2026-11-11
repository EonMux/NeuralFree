package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.KvDType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Names and argument indices of the flash-attention kernels ({@code kernels/llm/flash_decode.cl},
 * {@code flash_prefill.cl}, {@code flash_combine.cl} and their GLSL twins). Argument indices are positions in the
 * full Java argument list (every parameter in declared order: the scalars, then the buffers, then, for the mixed
 * kernels only, the trailing {@code kv_types} scalar), as {@code PreparedDispatch.setInt} expects. Grid sizes and the tile parameters come from {@link FlashAttnTiling}; the CPU reference is
 * {@link CpuAttentionCore}.
 *
 * <p>Conventions shared by all three kernels:
 * <ul>
 *   <li>offsets ({@code q_off}, {@code out_off}) and strides count elements; a KV element index is
 *       {@code row * row_stride + kvHead * head_stride + d}, and caches of a block type ({@link KvDType#quantized()})
 *       require row and head strides that are multiples of 32 ({@link KvDType#rowBytes});</li>
 *   <li>the decode and prefill kernels exist per KV element type, {@code <prefix><KvDType#kernelSuffix()>}, for K and
 *       V caches of one type; K and V caches of different types (non-MLA attention, as llama.cpp's flash attention
 *       accepts them: ggml-cpu/ops.cpp:9007-9013 reads K and V through their own type traits) use
 *       {@link #FLASH_DECODE_MIXED} / {@link #FLASH_ATTN_MIXED}, which take the same arguments followed by one
 *       {@code kv_types} scalar ({@link #kvTypes}) after the buffers. {@link #flashDecode(KvDType, KvDType)},
 *       {@link #flashAttn(KvDType, KvDType)} and {@link #withKvTypes} select the kernel and its arguments, so the
 *       same-type kernels stay the path of equal types;</li>
 *   <li>{@code swa_type} is {@code SwaType.code()}, {@code bias_mode} is {@code AttentionSpec.BiasMode.code()};
 *       the packed {@code flags} scalar carries causal masking ({@link #FLAG_CAUSAL}), the sink logits of buffer
 *       {@code sinks} ({@link #FLAG_SINKS}), direct normalized output with {@code n_split == 1}
 *       ({@link #FLAG_DIRECT}) and a tree verify step's mask ({@link #FLAG_TREE}) with the step's fresh-row count
 *       packed in bits 5-31 ({@code NF_ATTN_FLAG_FRESH_SHIFT}; {@link #treeFlags} builds it, the kernels decode it
 *       with {@code nf_attn_flag_fresh}), as {@link #attnFlags} builds the base bits;</li>
 *   <li>over a paginated cache, {@link #FLASH_DECODE_PAGED_PREFIX} / {@link #FLASH_ATTN_PAGED_PREFIX} address K, V
 *       and {@code kv_pos} at the physical row {@code r = (page << page_shift) | (slot & ((1 << page_shift) - 1))}
 *       of {@code common/attn_common.h} nf_kv_row: the paged decode takes the base decode arguments, then the page
 *       geometry and the per-sequence buffers (the {@code ARG_FDP_*} indices; {@code page_table} is the
 *       {@code int[nSeq x maxPages]} of {@code PagedStep.pageTable(g)}), and the paged prefill takes the base
 *       prefill arguments, then {@code page_shift}, {@code pt_off} and {@code page_table} (a page id below 0 masks
 *       the slot's keys, never required). The base kernels are the identity map {@code page_shift = -1};</li>
 *   <li>{@code kv_pos[row]} is the position stored in a cache row, -1 for an empty row;</li>
 *   <li>unused {@code sinks}, {@code rel_bias}, {@code kv_scales} and {@code partial} buffers are bound to placeholders
 *       (read-only for the first two, writable for {@code partial}); {@code kv_scales} holds the FP8 cache's per-head
 *       checkpoint scales, or the interleaved per-(row, KV head) scale rows of a per-token-head cache;</li>
 *   <li>{@link #FLAG_DIRECT} requires {@code n_split = 1}; otherwise the kernel writes {@code (o, m, l)} partials
 *       that {@link #FLASH_COMBINE} merges, adding the sink there.</li>
 * </ul>
 */
public final class AttnKernels {

    /** Name prefix of the {@code LLM_FlashDecode_<KV>} variants. */
    public static final String FLASH_DECODE_PREFIX = "LLM_FlashDecode_";
    /** Name prefix of the {@code LLM_FlashAttn_<KV>} variants. */
    public static final String FLASH_ATTN_PREFIX = "LLM_FlashAttn_";
    public static final String FLASH_COMBINE = "LLM_FlashAttn_Combine";
    /**
     * Merged log-sum-exp of the split-K partials ({@code kernels/llm/flash_combine.cl}): the same merge as
     * {@link #FLASH_COMBINE}, but it writes one {@code lse = m + log(l)} per row instead of the normalized output.
     * Context-parallel attention merges the per-rank outputs with these values ({@link #CP_MERGE}).
     */
    public static final String FLASH_LSE = "LLM_FlashAttn_Lse";
    /**
     * Context-parallel attention merge ({@code kernels/llm/cp_merge.cl}): combines the normalized outputs and
     * log-sum-exp values of every rank into the one softmax over their concatenated key ranges.
     */
    public static final String CP_MERGE = "LLM_AttnCpMerge";
    /**
     * Learned static sink-token partial of the openPangu sink attention ({@code kernels/llm/flash_combine.cl}): the
     * (o, m, l) split-K partial of the sink keys and values of every (token, head), written at one extra split
     * index so {@link #FLASH_COMBINE} merges it with the cached-key splits.
     */
    public static final String STATIC_SINK_PARTIAL = "LLM_AttnStaticSinkPartial";
    /** Decode kernel over K and V caches of different types: the {@code LLM_FlashDecode_<KV>} arguments + kv_types. */
    public static final String FLASH_DECODE_MIXED = "LLM_FlashDecode_Mixed";
    /** Prefill kernel over K and V caches of different types: the {@code LLM_FlashAttn_<KV>} arguments + kv_types. */
    public static final String FLASH_ATTN_MIXED = "LLM_FlashAttn_Mixed";
    /** Name prefix of the paged decode variants over the batch's sequences ({@code PagedStep.pageTable(g)}). */
    public static final String FLASH_DECODE_PAGED_PREFIX = "LLM_FlashDecodePaged_";
    /** Name prefix of the paged prefill variants over one paginated KV segment. */
    public static final String FLASH_ATTN_PAGED_PREFIX = "LLM_FlashAttnPaged_";
    /** Paged decode over K and V caches of different types: the {@code LLM_FlashDecodePaged_<KV>} args + kv_types. */
    public static final String FLASH_DECODE_PAGED_MIXED = "LLM_FlashDecodePaged_Mixed";
    /** Paged prefill over K and V caches of different types: the {@code LLM_FlashAttnPaged_<KV>} args + kv_types. */
    public static final String FLASH_ATTN_PAGED_MIXED = "LLM_FlashAttnPaged_Mixed";
    /** Name prefix of the tensor-core ({@code LLM_FlashAttnMma_<KV>}) tiled prefill variants; CUDA and Vulkan. */
    public static final String FLASH_ATTN_MMA_PREFIX = "LLM_FlashAttnMma_";
    /** Name prefix of the tensor-core paged ({@code LLM_FlashAttnMmaPaged_<KV>}) variants; CUDA and Vulkan. */
    public static final String FLASH_ATTN_MMA_PAGED_PREFIX = "LLM_FlashAttnMmaPaged_";
    /**
     * The one-row decode attention of the plain step over a dense F16 cache ({@code kernels/llm/flash_decode_row.cl}
     * on CUDA, its twin {@code shaders/llm/flash_decode_row.comp} on Vulkan): {@code LLM_FlashDecode_F16}'s
     * arguments, grid and outputs, in a body that keeps only what that step needs. {@link FlashPlan#rowKernelName}
     * decides where it serves.
     */
    public static final String FLASH_DECODE_ROW_F16 = "LLM_FlashDecodeRow_F16";
    /** Largest head dim of {@link #FLASH_DECODE_ROW_F16} ({@code NF_FDR_MAX_D}). */
    public static final int FLASH_DECODE_ROW_MAX_D = 256;

    /**
     * Name prefix of the grouped-query paged decode variants ({@code LLM_FlashDecodePagedGqa_<KV>}, the
     * {@code LLM_FlashDecodePaged_<KV>} arguments, every query head of a KV group in one work-group); CUDA only.
     */
    public static final String FLASH_DECODE_PAGED_GQA_PREFIX = "LLM_FlashDecodePagedGqa_";
    /**
     * Name prefix of the grouped-query dense decode variants ({@code LLM_FlashDecodeGqa_<KV>}, the
     * {@code LLM_FlashDecode_<KV>} arguments, every query head of a KV group in one work-group); CUDA and Vulkan.
     */
    public static final String FLASH_DECODE_GQA_PREFIX = "LLM_FlashDecodeGqa_";

    /** Bits of one type code in the packed {@code kv_types} argument ({@code NF_ATTN_KV_TYPE_BITS}). */
    public static final int KV_TYPE_BITS = 8;

    private static final List<String> ALL;

    static {
        List<String> all = new ArrayList<>(KvDType.kernelVariants(FLASH_DECODE_PREFIX));
        all.add(FLASH_DECODE_MIXED);
        all.addAll(KvDType.kernelVariants(FLASH_ATTN_PREFIX));
        all.add(FLASH_ATTN_MIXED);
        all.add(FLASH_COMBINE);
        all.add(FLASH_LSE);
        all.add(CP_MERGE);
        all.add(STATIC_SINK_PARTIAL);
        all.add(AttnKernels.DUAL_CHUNK_MERGE);
        all.addAll(KvDType.kernelVariants(FLASH_DECODE_PAGED_PREFIX));
        all.add(FLASH_DECODE_PAGED_MIXED);
        all.addAll(KvDType.kernelVariants(FLASH_ATTN_PAGED_PREFIX));
        all.add(FLASH_ATTN_PAGED_MIXED);
        ALL = Collections.unmodifiableList(all);
    }

    // ---- LLM_FlashDecode_*: 25 scalars, then 10 buffers ----
    /** Bits of the packed {@code flags} argument ({@code NF_ATTN_FLAG_*} of {@code common/attn_common.h}). */
    public static final int FLAG_CAUSAL = 1;
    public static final int FLAG_SINKS = 2;
    public static final int FLAG_DIRECT = 4;
    /**
     * The mask buffer of a tree verify step names the query row's visible fresh slots: bit {@code i} of
     * {@code tree_mask[t]} admits fresh slot {@code i} of the step's last fresh slots (the count packed in the
     * {@code flags} bits above {@link #FLAG_FRESH_SHIFT}, {@link #treeFlags}); the slots before them keep the
     * position rules.
     */
    public static final int FLAG_TREE = 16;
    public static final int ARG_FD_N_HEAD = 0;
    public static final int ARG_FD_N_HEAD_KV = 1;
    public static final int ARG_FD_D_K = 2;
    public static final int ARG_FD_D_V = 3;
    public static final int ARG_FD_G_TILE = 4;
    /** Per step: split count of the current key range. */
    public static final int ARG_FD_N_SPLIT = 5;
    /** Per step: keys per split of the current key range. */
    public static final int ARG_FD_SPLIT_LEN = 6;
    /** Per step: first cache row scanned. */
    public static final int ARG_FD_KV_START = 7;
    /** Per step: one past the last cache row scanned. */
    public static final int ARG_FD_KV_END = 8;
    /** Per step: position of the query token. */
    public static final int ARG_FD_Q_POS = 9;
    public static final int ARG_FD_K_ROW_STRIDE = 10;
    public static final int ARG_FD_K_HEAD_STRIDE = 11;
    public static final int ARG_FD_V_ROW_STRIDE = 12;
    public static final int ARG_FD_V_HEAD_STRIDE = 13;
    public static final int ARG_FD_SCALE = 14;
    public static final int ARG_FD_SOFTCAP = 15;
    public static final int ARG_FD_MAX_BIAS = 16;
    /** Packed causal / has_sinks / direct_out / tree bits; the per-step bit is {@link #FLAG_DIRECT}. */
    public static final int ARG_FD_FLAGS = 17;
    public static final int ARG_FD_SWA_TYPE = 18;
    public static final int ARG_FD_N_SWA = 19;
    public static final int ARG_FD_BIAS_MODE = 20;
    public static final int ARG_FD_N_BUCKETS = 21;
    public static final int ARG_FD_MAX_DISTANCE = 22;
    /** Per row of a per-row prefill: element offset of the query row. */
    public static final int ARG_FD_Q_OFF = 23;
    /** Per row of a per-row prefill: element offset of the output row. */
    public static final int ARG_FD_OUT_OFF = 24;
    public static final int ARG_FD_BUF_Q = 25;
    public static final int ARG_FD_BUF_K = 26;
    public static final int ARG_FD_BUF_V = 27;
    public static final int ARG_FD_BUF_KV_POS = 28;
    public static final int ARG_FD_BUF_SINKS = 29;
    public static final int ARG_FD_BUF_REL_BIAS = 30;
    public static final int ARG_FD_BUF_OUT = 31;
    public static final int ARG_FD_BUF_PARTIAL = 32;
    /**
     * The FP8 KV-cache scales ({@code n_head_kv} K values then {@code n_head_kv} V values), or the interleaved
     * per-(row, KV head) scale rows of a per-token-head cache ({@code common/kv_types.h}).
     */
    public static final int ARG_FD_BUF_KV_SCALES = 33;
    /**
     * Under {@link #FLAG_TREE}, one {@code ulong} per query row whose fresh-slot bits
     * ({@code common/attn_common.h} nf_attn_tree_visible) name the row's visible slots of the last fresh keys (the
     * count packed in the {@code flags} bits above {@link #FLAG_FRESH_SHIFT}).
     */
    public static final int ARG_FD_BUF_TREE_MASK = 34;
    public static final int FD_SCALAR_COUNT = 25;
    public static final int FD_ARG_COUNT = 35;
    /** {@link #FLASH_DECODE_MIXED} only: {@link #kvTypes} of the K and V caches, after the buffers. */
    public static final int ARG_FD_KV_TYPES = 35;
    public static final int FD_MIXED_SCALAR_COUNT = 26;
    public static final int FD_MIXED_ARG_COUNT = 36;

    // ---- LLM_FlashAttn_*: 29 scalars, then 11 buffers ----
    /** Per call: query rows. */
    public static final int ARG_FA_N_TOKENS = 0;
    public static final int ARG_FA_N_HEAD = 1;
    public static final int ARG_FA_N_HEAD_KV = 2;
    public static final int ARG_FA_D_K = 3;
    public static final int ARG_FA_D_V = 4;
    public static final int ARG_FA_BQ = 5;
    public static final int ARG_FA_BK = 6;
    public static final int ARG_FA_G_TILE = 7;
    /** Per call: split count ({@link FlashAttnTiling#prefillSplit}). */
    public static final int ARG_FA_N_SPLIT = 8;
    /** Per call: keys per split. */
    public static final int ARG_FA_SPLIT_LEN = 9;
    /** Per call: first key row. */
    public static final int ARG_FA_KV_START = 10;
    /** Per call: one past the last key row. */
    public static final int ARG_FA_KV_END = 11;
    public static final int ARG_FA_Q_ROW_STRIDE = 12;
    public static final int ARG_FA_OUT_ROW_STRIDE = 13;
    public static final int ARG_FA_K_ROW_STRIDE = 14;
    public static final int ARG_FA_K_HEAD_STRIDE = 15;
    public static final int ARG_FA_V_ROW_STRIDE = 16;
    public static final int ARG_FA_V_HEAD_STRIDE = 17;
    public static final int ARG_FA_SCALE = 18;
    public static final int ARG_FA_SOFTCAP = 19;
    public static final int ARG_FA_MAX_BIAS = 20;
    /** Packed causal / has_sinks / direct_out / tree bits; the per-step bit is {@link #FLAG_DIRECT}. */
    public static final int ARG_FA_FLAGS = 21;
    public static final int ARG_FA_SWA_TYPE = 22;
    public static final int ARG_FA_N_SWA = 23;
    public static final int ARG_FA_BIAS_MODE = 24;
    public static final int ARG_FA_N_BUCKETS = 25;
    public static final int ARG_FA_MAX_DISTANCE = 26;
    public static final int ARG_FA_Q_OFF = 27;
    public static final int ARG_FA_OUT_OFF = 28;
    public static final int ARG_FA_BUF_Q = 29;
    public static final int ARG_FA_BUF_K = 30;
    public static final int ARG_FA_BUF_V = 31;
    public static final int ARG_FA_BUF_Q_POS = 32;
    public static final int ARG_FA_BUF_KV_POS = 33;
    public static final int ARG_FA_BUF_SINKS = 34;
    public static final int ARG_FA_BUF_REL_BIAS = 35;
    public static final int ARG_FA_BUF_OUT = 36;
    public static final int ARG_FA_BUF_PARTIAL = 37;
    /**
     * The FP8 KV-cache scales ({@code n_head_kv} K values then {@code n_head_kv} V values), or the interleaved
     * per-(row, KV head) scale rows of a per-token-head cache ({@code common/kv_types.h}).
     */
    public static final int ARG_FA_BUF_KV_SCALES = 38;
    /** Under {@link #FLAG_TREE}, the per-query-row fresh-slot bitmask ({@link #ARG_FD_BUF_TREE_MASK}). */
    public static final int ARG_FA_BUF_TREE_MASK = 39;
    public static final int FA_SCALAR_COUNT = 29;
    public static final int FA_ARG_COUNT = 40;
    /** {@link #FLASH_ATTN_MIXED} only: {@link #kvTypes} of the K and V caches, after the buffers. */
    public static final int ARG_FA_KV_TYPES = 40;
    public static final int FA_MIXED_SCALAR_COUNT = 30;
    public static final int FA_MIXED_ARG_COUNT = 41;
    /** {@code LLM_FlashAttnMma_*}: the prefill's 29 scalars and 10 buffers, without the KV scale buffer. */
    public static final int FA_MMA_ARG_COUNT = 39;
    /**
     * {@code LLM_FlashAttnMma_*}: the prefill's buffer list drops {@code kv_scales}, so its tree mask sits at the
     * prefill's scale index.
     */
    public static final int ARG_FA_MMA_BUF_TREE_MASK = ARG_FA_BUF_KV_SCALES;

    // ---- LLM_FlashDecodePaged_*: 30 scalars, then 13 buffers ----
    /**
     * Paged decode over the batch's sequences one launch ({@code LLM_FlashDecodePaged_<KV>}): the
     * {@code LLM_FlashDecode_<KV>} arguments in order (their {@code kv_start}, {@code kv_end} and {@code q_pos} are
     * ignored; the host passes 0), then the page geometry and the per-sequence buffers:
     * {@code page_shift} (log2 page size), {@code max_pages} (page-table entries per sequence:
     * {@code PagedStep.maxPages()}), {@code n_seq}, {@code q_seq_stride}, {@code out_seq_stride}, and buffers
     * {@code page_table} (the host-validated int[nSeq x maxPages], sequence {@code s}'s block at
     * {@code s * max_pages}), {@code seq_kv_len} (per-sequence slot count), {@code q_pos_seq} (per-sequence query
     * position). {@code kv_start}/{@code kv_end}/{@code q_pos} and the buffers they address stay in the list so
     * the same-type argument list is the base decode list's prefix; under {@link #FLAG_TREE} sequence {@code s}'s
     * mask word is {@code tree_mask + s} ({@link #ARG_FDP_BUF_TREE_MASK}).
     */
    public static final int ARG_FDP_PAGE_SHIFT = 25;
    public static final int ARG_FDP_MAX_PAGES = 26;
    public static final int ARG_FDP_N_SEQ = 27;
    public static final int ARG_FDP_Q_SEQ_STRIDE = 28;
    public static final int ARG_FDP_OUT_SEQ_STRIDE = 29;
    public static final int ARG_FDP_BUF_Q = 30;
    public static final int ARG_FDP_BUF_K = 31;
    public static final int ARG_FDP_BUF_V = 32;
    public static final int ARG_FDP_BUF_KV_POS = 33;
    public static final int ARG_FDP_BUF_SINKS = 34;
    public static final int ARG_FDP_BUF_REL_BIAS = 35;
    public static final int ARG_FDP_BUF_OUT = 36;
    public static final int ARG_FDP_BUF_PARTIAL = 37;
    public static final int ARG_FDP_BUF_PAGE_TABLE = 38;
    public static final int ARG_FDP_BUF_SEQ_KV_LEN = 39;
    public static final int ARG_FDP_BUF_Q_POS_SEQ = 40;
    /**
     * The FP8 KV-cache scales ({@code n_head_kv} K values then {@code n_head_kv} V values), or the interleaved
     * per-(row, KV head) scale rows of a per-token-head cache ({@code common/kv_types.h}).
     */
    public static final int ARG_FDP_BUF_KV_SCALES = 41;
    /** Under {@link #FLAG_TREE}, the per-sequence mask words ({@link #ARG_FD_BUF_TREE_MASK}). */
    public static final int ARG_FDP_BUF_TREE_MASK = 42;
    public static final int FDP_SCALAR_COUNT = 30;
    public static final int FDP_ARG_COUNT = 43;
    /** {@link #FLASH_DECODE_PAGED_MIXED} only: {@link #kvTypes} of the K and V caches, after the buffers. */
    public static final int ARG_FDP_KV_TYPES = 43;
    public static final int FDP_MIXED_SCALAR_COUNT = 31;
    public static final int FDP_MIXED_ARG_COUNT = 44;

    // ---- LLM_FlashAttnPaged_*: 31 scalars, then 12 buffers ----
    /**
     * Paged prefill over one multi-token segment of a paginated KV cache ({@code LLM_FlashAttnPaged_<KV>}): the
     * {@code LLM_FlashAttn_<KV>} arguments in order (their {@code kv_start} / {@code kv_end} count slots in the
     * segment), then {@code page_shift} (log2 page size) and {@code pt_off} (the element offset of the segment's
     * page-table block in {@code page_table}, the segment's sequence resolved by the caller).
     */
    public static final int ARG_FAP_PAGE_SHIFT = 29;
    public static final int ARG_FAP_PT_OFF = 30;
    public static final int ARG_FAP_BUF_Q = 31;
    public static final int ARG_FAP_BUF_K = 32;
    public static final int ARG_FAP_BUF_V = 33;
    public static final int ARG_FAP_BUF_Q_POS = 34;
    public static final int ARG_FAP_BUF_KV_POS = 35;
    public static final int ARG_FAP_BUF_SINKS = 36;
    public static final int ARG_FAP_BUF_REL_BIAS = 37;
    public static final int ARG_FAP_BUF_OUT = 38;
    public static final int ARG_FAP_BUF_PARTIAL = 39;
    public static final int ARG_FAP_BUF_PAGE_TABLE = 40;
    /**
     * The FP8 KV-cache scales ({@code n_head_kv} K values then {@code n_head_kv} V values), or the interleaved
     * per-(row, KV head) scale rows of a per-token-head cache ({@code common/kv_types.h}).
     */
    public static final int ARG_FAP_BUF_KV_SCALES = 41;
    /** Under {@link #FLAG_TREE}, the per-query-row fresh-slot bitmask ({@link #ARG_FD_BUF_TREE_MASK}). */
    public static final int ARG_FAP_BUF_TREE_MASK = 42;
    public static final int FAP_SCALAR_COUNT = 31;
    public static final int FAP_ARG_COUNT = 43;
    /** {@link #FLASH_ATTN_PAGED_MIXED} only: {@link #kvTypes} of the K and V caches, after the buffers. */
    public static final int ARG_FAP_KV_TYPES = 43;
    public static final int FAP_MIXED_SCALAR_COUNT = 32;
    public static final int FAP_MIXED_ARG_COUNT = 44;
    /** {@code LLM_FlashAttnMmaPaged_F16}: the paged prefill's 31 scalars and 11 buffers, without KV scales. */
    public static final int FAP_MMA_ARG_COUNT = 42;
    /**
     * {@code LLM_FlashAttnMmaPaged_F16}: the paged prefill's buffer list drops {@code kv_scales}, so its tree mask
     * sits at the paged prefill's scale index.
     */
    public static final int ARG_FAP_MMA_BUF_TREE_MASK = ARG_FAP_BUF_KV_SCALES;

    // ---- LLM_FlashAttn_Combine: 6 scalars, then 3 buffers ----
    /** Per call: rows to merge (heads for decode, tokens x heads for prefill). */
    public static final int ARG_FC_N_ROWS = 0;
    public static final int ARG_FC_D_V = 1;
    /** Per call: split count of the producing call. */
    public static final int ARG_FC_N_SPLIT = 2;
    public static final int ARG_FC_N_HEAD = 3;
    public static final int ARG_FC_HAS_SINKS = 4;
    public static final int ARG_FC_OUT_OFF = 5;
    public static final int ARG_FC_BUF_PARTIAL = 6;
    public static final int ARG_FC_BUF_SINKS = 7;
    public static final int ARG_FC_BUF_OUT = 8;
    public static final int FC_SCALAR_COUNT = 6;
    public static final int FC_ARG_COUNT = 9;

    // ---- LLM_AttnCpMerge: 3 scalars, then 3 buffers ----
    /** Per call: rows to merge (decode token x heads, prefill tokens x heads). */
    public static final int ARG_CM_N_ROWS = 0;
    public static final int ARG_CM_D_V = 1;
    /** Per call: context-parallel ranks whose outputs are merged. */
    public static final int ARG_CM_N_RANK = 2;
    /** {@code n_rank * n_rows * d_v}; rank {@code r}'s rows start at {@code r * n_rows * d_v}. */
    public static final int ARG_CM_BUF_OUTS = 3;
    /** {@code n_rank * n_rows} merged log-sum-exp values, rank-major. */
    public static final int ARG_CM_BUF_LSES = 4;
    public static final int ARG_CM_BUF_OUT = 5;
    public static final int CM_SCALAR_COUNT = 3;
    public static final int CM_ARG_COUNT = 6;

    // ---- LLM_DualChunkMerge: 5 scalars, then 4 buffers ----
    /**
     * Merge of a token's three dual-chunk stage partials ({@code kernels/llm/dual_chunk_merge.cl}): each stage is an
     * {@code LLM_FlashAttn_*} / {@code LLM_FlashAttnPaged_*} launch with {@code direct_out = 0} and
     * {@code n_split = 1}, and the kernel resolves the one softmax over their concatenated logits.
     */
    public static final String DUAL_CHUNK_MERGE = "LLM_DualChunkMerge";
    /** Per call: query rows of the launch ({@code t = row / n_head}). */
    public static final int ARG_DCM_N_ROWS = 0;
    public static final int ARG_DCM_N_HEAD = 1;
    public static final int ARG_DCM_D_V = 2;
    /** Per call: element stride between output rows (the dense attention's {@code out_dim}). */
    public static final int ARG_DCM_OUT_ROW_STRIDE = 3;
    /** Per call: element offset of the first output row. */
    public static final int ARG_DCM_OUT_OFF = 4;
    public static final int ARG_DCM_BUF_P0 = 5;
    public static final int ARG_DCM_BUF_P1 = 6;
    public static final int ARG_DCM_BUF_P2 = 7;
    public static final int ARG_DCM_BUF_OUT = 8;
    public static final int DCM_SCALAR_COUNT = 5;
    public static final int DCM_ARG_COUNT = 9;

    // ---- LLM_AttnStaticSinkPartial: 18 scalars, then 5 buffers ----
    /** Per call: the (token, head) rows {@code n_tokens * n_head}. */
    public static final int ARG_SS_N_ROWS = 0;
    public static final int ARG_SS_N_HEAD = 1;
    public static final int ARG_SS_N_HEAD_KV = 2;
    public static final int ARG_SS_D_K = 3;
    public static final int ARG_SS_D_V = 4;
    public static final int ARG_SS_N_SINK = 5;
    /** Per call: element offset of the first query row. */
    public static final int ARG_SS_Q_OFF = 6;
    /** Per call: element stride between query rows ({@code q_dim}). */
    public static final int ARG_SS_Q_ROW_STRIDE = 7;
    /** Per call: first position entry of the first row. */
    public static final int ARG_SS_POS_OFF = 8;
    /** Per call: position entries per row; 0 makes every row use {@link #ARG_SS_FIXED_POS}. */
    public static final int ARG_SS_POS_STRIDE = 9;
    /** Per call: the position of every row when {@link #ARG_SS_POS_STRIDE} is 0. */
    public static final int ARG_SS_FIXED_POS = 10;
    /** Per call: split count the partial belongs to (the cached-key splits plus this one). */
    public static final int ARG_SS_N_SPLIT = 11;
    /** Per call: split index this partial is written at. */
    public static final int ARG_SS_SPLIT_INDEX = 12;
    public static final int ARG_SS_SCALE = 13;
    public static final int ARG_SS_SOFTCAP = 14;
    public static final int ARG_SS_CAUSAL = 15;
    public static final int ARG_SS_SWA_TYPE = 16;
    public static final int ARG_SS_N_SWA = 17;
    public static final int ARG_SS_BUF_Q = 18;
    public static final int ARG_SS_BUF_Q_POS = 19;
    public static final int ARG_SS_BUF_SINK_K = 20;
    public static final int ARG_SS_BUF_SINK_V = 21;
    public static final int ARG_SS_BUF_PARTIAL = 22;
    public static final int SS_SCALAR_COUNT = 18;
    public static final int SS_ARG_COUNT = 23;

    private AttnKernels() {
    }

    /** Decode kernel for a KV element type. */
    public static String flashDecode(KvDType kv) {
        return Objects.requireNonNull(kv, "kv").kernel(FLASH_DECODE_PREFIX);
    }

    /** Tiled prefill kernel for a KV element type. */
    public static String flashAttn(KvDType kv) {
        return Objects.requireNonNull(kv, "kv").kernel(FLASH_ATTN_PREFIX);
    }

    /**
     * Decode kernel for a K cache of type {@code k} and a V cache of type {@code v}: {@link #flashDecode(KvDType)}
     * when they are equal, {@link #FLASH_DECODE_MIXED} otherwise.
     */
    public static String flashDecode(KvDType k, KvDType v) {
        return mixed(k, v) ? FLASH_DECODE_MIXED : flashDecode(k);
    }

    /**
     * Tiled prefill kernel for a K cache of type {@code k} and a V cache of type {@code v}:
     * {@link #flashAttn(KvDType)} when they are equal, {@link #FLASH_ATTN_MIXED} otherwise.
     */
    public static String flashAttn(KvDType k, KvDType v) {
        return mixed(k, v) ? FLASH_ATTN_MIXED : flashAttn(k);
    }

    /**
     * The packed {@code kv_types} argument of the mixed kernels: the K type's {@link KvDType#code()} in bits 0-7 and
     * the V type's in bits 8-15 ({@code nf_attn_kv_type_k} / {@code nf_attn_kv_type_v} of
     * {@code kernels/llm/common/attn_common.h}).
     */
    public static int kvTypes(KvDType k, KvDType v) {
        return Objects.requireNonNull(k, "k").code() | (Objects.requireNonNull(v, "v").code() << KV_TYPE_BITS);
    }

    /** The packed {@code flags} argument of the flash attention kernels ({@code NF_ATTN_FLAG_*} bits). */
    /** V of an MLA latent cache reads the K rows, so it decodes through the K side planes. */
    public static final int FLAG_V_ALIAS = 8;
    /**
     * Bit the fresh-row count of a tree verify step shifts into {@code flags}
     * ({@code NF_ATTN_FLAG_FRESH_SHIFT} of {@code common/attn_common.h}); the count fills bits 5-31 next to
     * {@link #FLAG_TREE} and keeps its scalar argument out of the kernels.
     */
    public static final int FLAG_FRESH_SHIFT = 5;
    /** The bits 5-31 of {@code flags} ({@code treeFlags} packs the fresh-row count there). */
    public static final int FLAG_FRESH_MASK = ((1 << 32 - FLAG_FRESH_SHIFT) - 1) << FLAG_FRESH_SHIFT;

    public static int attnFlags(boolean causal, boolean sinks, boolean direct) {
        return (causal ? FLAG_CAUSAL : 0) | (sinks ? FLAG_SINKS : 0) | (direct ? FLAG_DIRECT : 0);
    }

    /**
     * The {@code flags} value of a tree verify step: {@code base} (the {@link #attnFlags} bits and
     * {@link #FLAG_V_ALIAS}) with {@link #FLAG_TREE} and the fresh-row count {@code n_fresh} packed in bits 5-31,
     * which the kernels decode with {@code nf_attn_flag_fresh} ({@code common/attn_common.h}). Without
     * {@link #FLAG_TREE} no kernel reads the packed count, and the count of a step without fresh rows is 0.
     *
     * @throws IllegalArgumentException when {@code n_fresh} is negative or leaves bits 5-31, or {@code base}
     *         already carries fresh-count bits
     */
    public static int treeFlags(int base, int nFresh) {
        if ((base & FLAG_FRESH_MASK) != 0) {
            throw new IllegalArgumentException("flags " + base + " already carry a fresh-row count");
        }
        if (nFresh < 0 || nFresh > FLAG_FRESH_MASK >>> FLAG_FRESH_SHIFT) {
            throw new IllegalArgumentException("fresh-row count " + nFresh + " does not fit the "
                    + (32 - FLAG_FRESH_SHIFT) + " flags bits");
        }
        return (base & ~FLAG_FRESH_MASK) | FLAG_TREE | nFresh << FLAG_FRESH_SHIFT;
    }

    /** The fresh-row count packed in {@code flags} ({@code nf_attn_flag_fresh}): 0 without {@link #FLAG_TREE}. */
    public static int flagFresh(int flags) {
        return (flags & FLAG_TREE) != 0 ? flags >>> FLAG_FRESH_SHIFT : 0;
    }

    /**
     * Arguments of the kernel that {@link #flashDecode(KvDType, KvDType)} or {@link #flashAttn(KvDType, KvDType)}
     * selects, built from the same-type argument list {@code args} (all {@link #FD_ARG_COUNT} or
     * {@link #FA_ARG_COUNT} arguments in signature order): {@code args} itself when {@code k == v}, otherwise a copy
     * with {@link #kvTypes} appended at {@link #ARG_FD_KV_TYPES} / {@link #ARG_FA_KV_TYPES}.
     *
     * @throws IllegalArgumentException when {@code args} is not a decode or prefill argument list
     */
    public static Object[] withKvTypes(Object[] args, KvDType k, KvDType v) {
        Objects.requireNonNull(args, "args");
        if (args.length != FD_ARG_COUNT && args.length != FA_ARG_COUNT) {
            throw new IllegalArgumentException("flash attention takes " + FD_ARG_COUNT + " (decode) or "
                    + FA_ARG_COUNT + " (prefill) arguments before kv_types, got " + args.length);
        }
        if (!mixed(k, v)) {
            return args;
        }
        Object[] out = Arrays.copyOf(args, args.length + 1);
        out[args.length] = kvTypes(k, v);
        return out;
    }

    private static boolean mixed(KvDType k, KvDType v) {
        return Objects.requireNonNull(k, "k") != Objects.requireNonNull(v, "v");
    }

    /** Paged decode kernel for a KV element type: one launch over the batch's decode sequences. */
    public static String flashDecodePaged(KvDType kv) {
        return Objects.requireNonNull(kv, "kv").kernel(FLASH_DECODE_PAGED_PREFIX);
    }

    /** Paged tiled prefill kernel for a KV element type: one launch per multi-token segment. */
    public static String flashAttnPaged(KvDType kv) {
        return Objects.requireNonNull(kv, "kv").kernel(FLASH_ATTN_PAGED_PREFIX);
    }

    /** Paged decode kernel for a K cache of type {@code k} and a V cache of type {@code v} (as {@link #flashDecode(KvDType, KvDType)}). */
    public static String flashDecodePaged(KvDType k, KvDType v) {
        return mixed(k, v) ? FLASH_DECODE_PAGED_MIXED : flashDecodePaged(k);
    }

    /** Paged prefill kernel for a K cache of type {@code k} and a V cache of type {@code v} (as {@link #flashAttn(KvDType, KvDType)}). */
    public static String flashAttnPaged(KvDType k, KvDType v) {
        return mixed(k, v) ? FLASH_ATTN_PAGED_MIXED : flashAttnPaged(k);
    }

    /**
     * Tensor-core tiled prefill kernel ({@code LLM_FlashAttnMma_<KV>}) for a KV element type whose mma variant
     * exists: the F16 cache (CUDA and Vulkan) and the Q8_0 cache (CUDA, which dequantizes the blocks to F16 as it
     * stages the tiles); null for every other type, so only a caller that has checked the mma plan reaches it. The
     * caller still requires the backend to have the kernel.
     */
    public static String flashAttnMma(KvDType kv) {
        return hasMmaVariant(kv) ? kv.kernel(FLASH_ATTN_MMA_PREFIX) : null;
    }

    /** Tensor-core paged tiled prefill kernel (as {@link #flashAttnMma(KvDType)}); null without an mma variant. */
    public static String flashAttnMmaPaged(KvDType kv) {
        return hasMmaVariant(kv) ? kv.kernel(FLASH_ATTN_MMA_PAGED_PREFIX) : null;
    }

    /** Whether the tensor-core prefill has a variant for caches of type {@code kv}: F16 and Q8_0. */
    private static boolean hasMmaVariant(KvDType kv) {
        return kv == KvDType.F16 || kv == KvDType.Q8_0;
    }

    /**
     * Grouped-query paged decode kernel ({@code LLM_FlashDecodePagedGqa_<KV>}, CUDA only) over a K cache of type
     * {@code k} and a V cache of type {@code v}: the F16, BF16 and Q8_0 caches with both caches of one type; null
     * for every other pair, so only a caller that has checked {@link FlashAttnTiling#gqaDecodeServes} reaches it.
     */
    public static String flashDecodePagedGqa(KvDType k, KvDType v) {
        Objects.requireNonNull(k, "k");
        Objects.requireNonNull(v, "v");
        return k == v && gqaType(k) ? k.kernel(FLASH_DECODE_PAGED_GQA_PREFIX) : null;
    }

    /**
     * Grouped-query dense decode kernel ({@code LLM_FlashDecodeGqa_<KV>}, CUDA and Vulkan): the dense twin of
     * {@link #flashDecodePagedGqa(KvDType, KvDType)} over the same cache pairs, null for every other pair.
     */
    public static String flashDecodeGqa(KvDType k, KvDType v) {
        Objects.requireNonNull(k, "k");
        Objects.requireNonNull(v, "v");
        return k == v && gqaType(k) ? k.kernel(FLASH_DECODE_GQA_PREFIX) : null;
    }

    /**
     * The row-stride granularity, in elements, of the grouped-query decode kernels over a cache of type {@code t}:
     * a whole Q8_0 block (32) or a whole 16-byte vector of F16 / BF16 (8).
     */
    public static int gqaStrideUnit(KvDType t) {
        return t == KvDType.Q8_0 ? 32 : 8;
    }

    private static boolean gqaType(KvDType t) {
        return t == KvDType.F16 || t == KvDType.BF16 || t == KvDType.Q8_0;
    }

    /**
     * Arguments of the paged kernel that {@link #flashDecodePaged(KvDType, KvDType)} or
     * {@link #flashAttnPaged(KvDType, KvDType)} selects, built from the same-type paged argument list {@code args}
     * (all {@link #FDP_ARG_COUNT} or {@link #FAP_ARG_COUNT} arguments in signature order): {@code args} itself when
     * {@code k == v}, otherwise a copy with {@link #kvTypes} appended at {@link #ARG_FDP_KV_TYPES} /
     * {@link #ARG_FAP_KV_TYPES}.
     *
     * @throws IllegalArgumentException when {@code args} is not a paged decode or paged prefill argument list
     */
    public static Object[] pagedWithKvTypes(Object[] args, KvDType k, KvDType v) {
        Objects.requireNonNull(args, "args");
        if (args.length != FDP_ARG_COUNT && args.length != FAP_ARG_COUNT) {
            throw new IllegalArgumentException("paged flash attention takes " + FDP_ARG_COUNT + " (paged decode) or "
                    + FAP_ARG_COUNT + " (paged prefill) arguments before kv_types, got " + args.length);
        }
        if (!mixed(k, v)) {
            return args;
        }
        Object[] out = Arrays.copyOf(args, args.length + 1);
        out[args.length] = kvTypes(k, v);
        return out;
    }

    /**
     * Every kernel of this family that each backend provides, for {@code LlmKernels.prewarm}. The kernels only some
     * backends build are absent: hosts look the tensor-core mma kernels (CUDA compute capability 8.0+, Vulkan with
     * {@code VK_NV_cooperative_matrix2}) up through {@link #flashAttnMma(KvDType)} and
     * {@link #flashAttnMmaPaged(KvDType)}, and the grouped-query decodes (the dense one CUDA and Vulkan, the paged
     * one CUDA) through {@link #flashDecodeGqa(KvDType, KvDType)} and {@link #flashDecodePagedGqa(KvDType, KvDType)}.
     */
    public static List<String> all() {
        return ALL;
    }
}
