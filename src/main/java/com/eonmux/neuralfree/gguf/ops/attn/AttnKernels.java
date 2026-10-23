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
 *   <li>{@code swa_type} is {@code SwaType.code()}, {@code bias_mode} is {@code AttentionSpec.BiasMode.code()};</li>
 *   <li>{@code kv_pos[row]} is the position stored in a cache row, -1 for an empty row;</li>
 *   <li>unused {@code sinks}, {@code rel_bias} and {@code partial} buffers are bound to placeholders
 *       (read-only for the first two, writable for {@code partial});</li>
 *   <li>{@code direct_out = 1} requires {@code n_split = 1}; otherwise the kernel writes {@code (o, m, l)} partials
 *       that {@link #FLASH_COMBINE} merges, adding the sink there.</li>
 * </ul>
 */
public final class AttnKernels {

    /** Name prefix of the {@code LLM_FlashDecode_<KV>} variants. */
    public static final String FLASH_DECODE_PREFIX = "LLM_FlashDecode_";
    /** Name prefix of the {@code LLM_FlashAttn_<KV>} variants. */
    public static final String FLASH_ATTN_PREFIX = "LLM_FlashAttn_";
    public static final String FLASH_COMBINE = "LLM_FlashAttn_Combine";
    /** Decode kernel over K and V caches of different types: the {@code LLM_FlashDecode_<KV>} arguments + kv_types. */
    public static final String FLASH_DECODE_MIXED = "LLM_FlashDecode_Mixed";
    /** Prefill kernel over K and V caches of different types: the {@code LLM_FlashAttn_<KV>} arguments + kv_types. */
    public static final String FLASH_ATTN_MIXED = "LLM_FlashAttn_Mixed";

    /** Bits of one type code in the packed {@code kv_types} argument ({@code NF_ATTN_KV_TYPE_BITS}). */
    public static final int KV_TYPE_BITS = 8;

    private static final List<String> ALL;

    static {
        List<String> all = new ArrayList<>(KvDType.kernelVariants(FLASH_DECODE_PREFIX));
        all.add(FLASH_DECODE_MIXED);
        all.addAll(KvDType.kernelVariants(FLASH_ATTN_PREFIX));
        all.add(FLASH_ATTN_MIXED);
        all.add(FLASH_COMBINE);
        ALL = Collections.unmodifiableList(all);
    }

    // ---- LLM_FlashDecode_*: 27 scalars, then 8 buffers ----
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
    public static final int ARG_FD_CAUSAL = 17;
    public static final int ARG_FD_SWA_TYPE = 18;
    public static final int ARG_FD_N_SWA = 19;
    public static final int ARG_FD_BIAS_MODE = 20;
    public static final int ARG_FD_N_BUCKETS = 21;
    public static final int ARG_FD_MAX_DISTANCE = 22;
    public static final int ARG_FD_HAS_SINKS = 23;
    /** Per step: 1 when {@link #ARG_FD_N_SPLIT} is 1 and the kernel writes the normalized output itself. */
    public static final int ARG_FD_DIRECT_OUT = 24;
    /** Per row of a per-row prefill: element offset of the query row. */
    public static final int ARG_FD_Q_OFF = 25;
    /** Per row of a per-row prefill: element offset of the output row. */
    public static final int ARG_FD_OUT_OFF = 26;
    public static final int ARG_FD_BUF_Q = 27;
    public static final int ARG_FD_BUF_K = 28;
    public static final int ARG_FD_BUF_V = 29;
    public static final int ARG_FD_BUF_KV_POS = 30;
    public static final int ARG_FD_BUF_SINKS = 31;
    public static final int ARG_FD_BUF_REL_BIAS = 32;
    public static final int ARG_FD_BUF_OUT = 33;
    public static final int ARG_FD_BUF_PARTIAL = 34;
    public static final int FD_SCALAR_COUNT = 27;
    public static final int FD_ARG_COUNT = 35;
    /** {@link #FLASH_DECODE_MIXED} only: {@link #kvTypes} of the K and V caches, after the buffers. */
    public static final int ARG_FD_KV_TYPES = 35;
    public static final int FD_MIXED_SCALAR_COUNT = 28;
    public static final int FD_MIXED_ARG_COUNT = 36;

    // ---- LLM_FlashAttn_*: 31 scalars, then 9 buffers ----
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
    public static final int ARG_FA_CAUSAL = 21;
    public static final int ARG_FA_SWA_TYPE = 22;
    public static final int ARG_FA_N_SWA = 23;
    public static final int ARG_FA_BIAS_MODE = 24;
    public static final int ARG_FA_N_BUCKETS = 25;
    public static final int ARG_FA_MAX_DISTANCE = 26;
    public static final int ARG_FA_HAS_SINKS = 27;
    /** Per call: 1 when {@link #ARG_FA_N_SPLIT} is 1. */
    public static final int ARG_FA_DIRECT_OUT = 28;
    public static final int ARG_FA_Q_OFF = 29;
    public static final int ARG_FA_OUT_OFF = 30;
    public static final int ARG_FA_BUF_Q = 31;
    public static final int ARG_FA_BUF_K = 32;
    public static final int ARG_FA_BUF_V = 33;
    public static final int ARG_FA_BUF_Q_POS = 34;
    public static final int ARG_FA_BUF_KV_POS = 35;
    public static final int ARG_FA_BUF_SINKS = 36;
    public static final int ARG_FA_BUF_REL_BIAS = 37;
    public static final int ARG_FA_BUF_OUT = 38;
    public static final int ARG_FA_BUF_PARTIAL = 39;
    public static final int FA_SCALAR_COUNT = 31;
    public static final int FA_ARG_COUNT = 40;
    /** {@link #FLASH_ATTN_MIXED} only: {@link #kvTypes} of the K and V caches, after the buffers. */
    public static final int ARG_FA_KV_TYPES = 40;
    public static final int FA_MIXED_SCALAR_COUNT = 32;
    public static final int FA_MIXED_ARG_COUNT = 41;

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

    /** Every kernel of this family, for {@code LlmKernels.prewarm}. */
    public static List<String> all() {
        return ALL;
    }
}
