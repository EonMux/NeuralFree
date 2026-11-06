package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.TiledGemmPlan;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Names, argument indices, codes, limits and argument lists of the MoE kernels in
 * {@code kernels/llm/moe_matvec.cl} (family {@code common/matvec_id_family.h}) and {@code kernels/llm/moe.cl}
 * (GLSL {@code matvec_id.comp}, {@code moe_router.comp}, {@code moe_wsum.comp}, {@code moe_misc.comp}):
 * <pre>
 * {T}_MatVecId(i out_dim, i in_dim, i n_sel, i x_stride, i in_off, i out_off, i has_scale, i expert_base,
 *              i expert_count, R&lt;int&gt; slots, R&lt;uchar&gt; weight, R input, R scales, W output)
 * {T}_MatVecIdT(same arguments as {T}_MatVecId; the transposed slot read, common/matvec_id_family.h
 *              {T}_MatVecIdT)
 * {T}_MatVecIdGateUp(i out_dim, i in_dim, i n_sel, i x_stride, i in_off, i glu_op, f limit, f alpha, f beta,
 *              i fused, i has_scale, i expert_base, i expert_count, R&lt;int&gt; slots, Rn&lt;uchar&gt; gate_w,
 *              Rn&lt;uchar&gt; up_w, R input, R scales, W output)
 * {T}_BatchMatMulId(i out_dim, i in_dim, i n_rows, i x_div, i in_off, i has_scale, i expert_base, i expert_count,
 *              R&lt;int&gt; row_token, R&lt;int&gt; row_slot, R&lt;uchar&gt; weight, R input, R scales, W output)
 * {T}_GroupedGemm(i out_dim, i in_dim, i n_tiles, i x_div, i x_stride, i in_off, i out_off, i has_scale,
 *              i expert_base, i expert_count, R&lt;int&gt; gemm_token, R&lt;int&gt; gemm_slot, R&lt;uchar&gt; weight,
 *              R input, R scales, W output)
 * {T}_GroupedGemmMma(same arguments as {T}_GroupedGemm; CUDA sm_80+ (kernels/llm/moe_gemm_mma.cl) and Vulkan
 *              cooperative matrices (shaders/llm/moe_gemm_mma.comp) only)
 * {T}_GroupedGemmInd(i out_dim, i in_dim, i n_tiles, i x_div, i x_stride, i in_off, i out_off, i has_scale,
 *              R&lt;int&gt; gemm_token, R&lt;int&gt; gemm_slot, R&lt;int&gt; map, R&lt;uchar&gt; weight, R input,
 *              R scales, W output)
 * {T}_GroupedGemmMmaInd(same arguments as {T}_GroupedGemmInd; where {T}_GroupedGemmMma exists)
 * {T}_MatVecIdInd(i out_dim, i in_dim, i n_sel, i x_stride, i in_off, i out_off, i has_scale,
 *              R&lt;int&gt; slots, R&lt;int&gt; map, R&lt;uchar&gt; weight, R input, R scales, W output)
 * {T}_MatVecIdGateUpInd(i out_dim, i in_dim, i n_sel, i x_stride, i in_off, i glu_op, f limit, f alpha, f beta,
 *              i fused, i has_scale, R&lt;int&gt; slots, R&lt;int&gt; map, Rn&lt;uchar&gt; gate_w,
 *              Rn&lt;uchar&gt; up_w, R input, R scales, W output)
 * {T}_BatchMatMulIdInd(i out_dim, i in_dim, i n_rows, i x_div, i in_off, i has_scale,
 *              R&lt;int&gt; row_token, R&lt;int&gt; row_slot, R&lt;int&gt; map, R&lt;uchar&gt; weight,
 *              R input, R scales, W output)
 * LLM_MoE_Router(i n_tokens, i n_expert, i k, i gating, i n_groups, i n_group_used, i norm_w, f w_scale,
 *              i has_bias, i sel_mode, i id_div, R logits, R exp_probs_b, RW&lt;int&gt; ids, W weights)
 * LLM_MoE_HashIds(i n_tokens, i k, R&lt;int&gt; tokens, R&lt;int&gt; tid2eid, W&lt;int&gt; ids)
 * LLM_MoE_WeightedSum(i n_tokens, i n_embd, i k, i flags, R expert_out, R weights, RWn dst)
 * LLM_AddId(i n_sel, i dim, R&lt;int&gt; slots, R bias, RWn data)
 * LLM_MoE_Bucketize(i n_tokens, i k, i n_expert, R&lt;int&gt; ids, W&lt;int&gt; counts, W&lt;int&gt; offsets,
 *              W&lt;int&gt; row_token, W&lt;int&gt; row_slot)
 * LLM_MoE_BucketCount(i n_tokens, i k, i n_expert, i chunk, R&lt;int&gt; ids, W&lt;int&gt; hist)
 * LLM_MoE_BucketScan(i n_tokens, i k, i n_expert, i chunk, RW&lt;int&gt; hist, W&lt;int&gt; counts, W&lt;int&gt; offsets)
 * LLM_MoE_BucketScatter(i n_tokens, i k, i n_expert, i chunk, R&lt;int&gt; ids, R&lt;int&gt; hist,
 *              W&lt;int&gt; row_token, W&lt;int&gt; row_slot)
 * LLM_MoE_AlignBlocks(i n_tokens, i k, i n_expert, i n_tiles, R&lt;int&gt; counts, R&lt;int&gt; offsets,
 *              R&lt;int&gt; row_token, R&lt;int&gt; row_slot, W&lt;int&gt; gemm_token, W&lt;int&gt; gemm_slot)
 * LLM_MoE_ScaleInput(i n_tokens, i n_embd, i k, R x, R weights, W out)
 * </pre>
 * Semantics are documented in the kernel sources; the CPU twins are {@link MoeRouting} and {@link MoeCombine}.
 * Pair index {@code p = t * k + j}. Optional buffers ({@code scales}, {@code exp_probs_b}) are bound to a
 * read-only placeholder when unused. Expert biases are applied with {@code LLM_AddId} between separate
 * {@code {T}_MatVecId} calls and {@code LLM_GLU}, because they precede the activation.
 *
 * <p>Expert shards: the weight buffers of the range-based id-matvec kernels bind the experts
 * {@code [expert_base, expert_base + expert_count)} of an expert tensor (one shard of whole experts, see
 * {@code GpuWeight}); a slot whose expert lies outside that range is left to the dispatch of the shard that holds
 * it, so one dispatch per shard writes every output row exactly once. Per-expert scales stay indexed by the global
 * expert. A tensor in one buffer, and every non-MoE caller (MLA head absorption: the "experts" are heads), passes
 * {@code expert_base = 0, expert_count = n}. The {@code Ind} variants replace the expert range with a per-layer
 * indirection map ({@code map[expert] = local expert of an expert cache bank, or -1}); pairs whose map entry is
 * negative are left to the host, which computes the cache misses.</p>
 *
 * <p>Launch: the id-matvec kernels are 2-D with {@code local = {L, 1}}, L a power of two at most
 * {@code LLM_DIMX}, a multiple of {@link #threadsPerBlock} and of 32 ({@link #idMatVecGlobal},
 * {@link #batchMatMulIdGlobal}); the grouped GEMM is 2-D with the tiled GEMM's work-group of
 * {@link #GROUPED_WORKGROUP} ({@link #groupedGemmGlobal}); the router uses one work-group of L (32 to
 * 256) per token; the bucketize and bucket-scan kernels one work-group, the bucket count and scatter kernels
 * one work-group per {@code chunk} pairs ({@link #bucketGroups}), the align kernel any number of work-groups
 * ({@link #alignGlobal}); the rest are 1-D element grids.
 */
public final class MoeKernels {

    public static final String ROUTER = "LLM_MoE_Router";
    public static final String HASH_IDS = "LLM_MoE_HashIds";
    public static final String WEIGHTED_SUM = "LLM_MoE_WeightedSum";
    public static final String ADD_ID = "LLM_AddId";
    public static final String BUCKETIZE = "LLM_MoE_Bucketize";
    public static final String ALIGN_BLOCKS = "LLM_MoE_AlignBlocks";
    /** Phase 1 of the multi-work-group expert sort: per-group bucket counts. */
    public static final String BUCKET_COUNT = "LLM_MoE_BucketCount";
    /** Phase 2 of the multi-work-group expert sort: counts, offsets and per-group row starts. */
    public static final String BUCKET_SCAN = "LLM_MoE_BucketScan";
    /** Phase 3 of the multi-work-group expert sort: the sorted rows. */
    public static final String BUCKET_SCATTER = "LLM_MoE_BucketScatter";
    public static final String SCALE_INPUT = "LLM_MoE_ScaleInput";
    /** Takes the identity experts out of a routing result, summing their weights per token. */
    public static final String ZERO_EXPERTS = "LLM_MoE_ZeroExperts";
    /** Adds {@code zero_weights[t] * x[t]} to a routed-expert output. */
    public static final String ZERO_APPLY = "LLM_MoE_ZeroApply";
    /** Maps logical routed ids to physical replica slots (EPLB). */
    public static final String EPLB_MAP = "LLM_MoE_EplbMap";

    public static final String MATVEC_ID_SUFFIX = "_MatVecId";
    /** Suffix of the transposed slot read ({@code {T}_MatVecIdT}). */
    public static final String MATVEC_ID_T_SUFFIX = "_MatVecIdT";
    public static final String MATVEC_ID_GATE_UP_SUFFIX = "_MatVecIdGateUp";
    public static final String BATCH_MATMUL_ID_SUFFIX = "_BatchMatMulId";
    public static final String GROUPED_GEMM_SUFFIX = "_GroupedGemm";
    /** Suffix of the tensor-core grouped GEMM kernels ({@code {T}_GroupedGemmMma}, CUDA sm_80+ and Vulkan). */
    public static final String GROUPED_GEMM_MMA_SUFFIX = "_GroupedGemmMma";
    /** Suffix of the indirection-map variants the MoE expert cache dispatches ({@code ...Ind}). */
    public static final String IND_SUFFIX = "Ind";

    /** Largest router / bucketize expert count ({@code NF_MOE_MAX_EXPERTS}). */
    public static final int MAX_EXPERTS = 1024;
    /** Largest experts-per-token count of the router ({@code NF_MOE_MAX_K}). */
    public static final int MAX_K = 32;
    /** Largest expert-group count of the router ({@code NF_MOE_MAX_GROUPS}). */
    public static final int MAX_GROUPS = 64;
    /** Sorted rows per {@code {T}_BatchMatMulId} work-group ({@code NF_MOE_BMM_ROWS}). */
    public static final int BMM_ROWS = 8;
    /** Aligned positions per {@code {T}_GroupedGemm} tile ({@code NF_GEMM_TN} of the tiled GEMM family). */
    public static final int ALIGN_TILE = TiledGemmPlan.TILE_TOKENS;
    /** Work-items per {@code {T}_GroupedGemm} work-group ({@code NF_GEMM_WG} of the tiled GEMM family). */
    public static final int GROUPED_WORKGROUP = TiledGemmPlan.WORKGROUP_SIZE;
    /**
     * Pairs per work-group of {@code LLM_MoE_BucketCount} / {@code LLM_MoE_BucketScatter}; a step of at most this
     * many pairs sorts in the one work-group of {@code LLM_MoE_Bucketize}.
     */
    public static final int BUCKET_CHUNK = 2048;
    /** Largest work-group count of an {@code LLM_MoE_AlignBlocks} launch; the kernel's grid-stride loop covers the rest. */
    public static final int MAX_ALIGN_GROUPS = 256;
    /** Largest work-group size of the reduce kernels ({@code LLM_DIMX}). */
    public static final int MAX_LOCAL = 256;
    /** Smallest router work-group size: each work-item tracks up to 32 experts in a bit mask. */
    public static final int MIN_ROUTER_LOCAL = 32;

    /** {@code sel_mode}: gating probabilities (+ {@code exp_probs_b}); {@code MoeSpec.SelectionMode.PROBS}. */
    public static final int SEL_PROBS = 0;
    /** {@code sel_mode}: raw logits (llama4); {@code MoeSpec.SelectionMode.RAW_LOGITS}. */
    public static final int SEL_RAW_LOGITS = 1;
    /** {@code sel_mode}: sigmoid of the logits (grovemoe); {@code MoeSpec.SelectionMode.SIGMOID_LOGITS}. */
    public static final int SEL_SIGMOID_LOGITS = 2;
    /** {@code sel_mode}: the ids are given in {@code ids} (deepseek4 hash layers, after {@code LLM_MoE_HashIds}). */
    public static final int SEL_PRESELECTED = 3;

    /** {@code has_scale}: no expert scale. */
    public static final int SCALE_NONE = 0;
    /** {@code has_scale}: {@code scales[e]} scales the output (gate and up alike for the gate-up kernel). */
    public static final int SCALE_SHARED = 1;
    /** {@code has_scale} of the gate-up kernel: {@code scales[2e]} gate, {@code scales[2e + 1]} up. */
    public static final int SCALE_GATE_UP_PAIRS = 2;

    /** {@code LLM_MoE_WeightedSum} flag: add the sum to {@code dst}. */
    public static final int WSUM_ACCUMULATE = 1;
    /** {@code LLM_MoE_WeightedSum} flag: ignore the weights (llama4 weight-before-FFN). */
    public static final int WSUM_UNWEIGHTED = 2;

    public static final int ARG_ID_OUT_DIM = 0;
    public static final int ARG_ID_IN_DIM = 1;
    /** Per-step {@code n_sel} of {@code {T}_MatVecId} and {@code {T}_MatVecIdGateUp}. */
    public static final int ARG_ID_N_SEL = 2;
    public static final int ARG_ID_X_STRIDE = 3;
    public static final int ARG_ID_IN_OFF = 4;
    public static final int ARG_ID_OUT_OFF = 5;
    public static final int ARG_ID_HAS_SCALE = 6;
    public static final int ARG_ID_EXPERT_BASE = 7;
    public static final int ARG_ID_EXPERT_COUNT = 8;
    public static final int ARG_ID_SLOTS = 9;
    public static final int ARG_ID_WEIGHT = 10;
    public static final int ARG_ID_INPUT = 11;
    public static final int ARG_ID_SCALES = 12;
    public static final int ARG_ID_OUTPUT = 13;
    public static final int ID_ARG_COUNT = 14;

    /** {@code map} of {@code {T}_MatVecIdInd} (one entry per expert: local bank expert, or -1). */
    public static final int ARG_IDI_SLOTS = 7;
    public static final int ARG_IDI_MAP = 8;
    public static final int ARG_IDI_WEIGHT = 9;
    public static final int ARG_IDI_INPUT = 10;
    public static final int ARG_IDI_SCALES = 11;
    public static final int ARG_IDI_OUTPUT = 12;
    public static final int ID_IND_ARG_COUNT = 13;
    /** {@code in_off} of {@code {T}_MatVecIdInd} and {@code {T}_BatchMatMulIdInd} (scalar 4, as without the map). */
    public static final int ARG_IND_IN_OFF = 4;

    /** {@code in_off} of {@code {T}_MatVecIdGateUp}. */
    public static final int ARG_GU_IN_OFF = 4;
    public static final int ARG_GU_GLU_OP = 5;
    public static final int ARG_GU_LIMIT = 6;
    public static final int ARG_GU_ALPHA = 7;
    public static final int ARG_GU_BETA = 8;
    public static final int ARG_GU_FUSED = 9;
    public static final int ARG_GU_HAS_SCALE = 10;
    public static final int ARG_GU_EXPERT_BASE = 11;
    public static final int ARG_GU_EXPERT_COUNT = 12;
    public static final int ARG_GU_SLOTS = 13;
    public static final int ARG_GU_GATE = 14;
    public static final int ARG_GU_UP = 15;
    public static final int ARG_GU_INPUT = 16;
    public static final int ARG_GU_SCALES = 17;
    public static final int ARG_GU_OUTPUT = 18;
    public static final int GU_ARG_COUNT = 19;

    /** {@code map} of {@code {T}_MatVecIdGateUpInd}. */
    public static final int ARG_GUI_SLOTS = 11;
    public static final int ARG_GUI_MAP = 12;
    public static final int ARG_GUI_GATE = 13;
    public static final int ARG_GUI_UP = 14;
    public static final int ARG_GUI_INPUT = 15;
    public static final int ARG_GUI_SCALES = 16;
    public static final int ARG_GUI_OUTPUT = 17;
    public static final int GU_IND_ARG_COUNT = 18;

    /** Per-step {@code n_rows} of {@code {T}_BatchMatMulId}. */
    public static final int ARG_BMM_N_ROWS = 2;
    public static final int ARG_BMM_X_DIV = 3;
    public static final int ARG_BMM_IN_OFF = 4;
    public static final int ARG_BMM_HAS_SCALE = 5;
    public static final int ARG_BMM_EXPERT_BASE = 6;
    public static final int ARG_BMM_EXPERT_COUNT = 7;
    public static final int ARG_BMM_ROW_TOKEN = 8;
    public static final int ARG_BMM_ROW_SLOT = 9;
    public static final int ARG_BMM_WEIGHT = 10;
    public static final int ARG_BMM_INPUT = 11;
    public static final int ARG_BMM_SCALES = 12;
    public static final int ARG_BMM_OUTPUT = 13;
    public static final int BMM_ARG_COUNT = 14;

    /** {@code map} of {@code {T}_BatchMatMulIdInd}. */
    public static final int ARG_BMI_ROW_TOKEN = 6;
    public static final int ARG_BMI_ROW_SLOT = 7;
    public static final int ARG_BMI_MAP = 8;
    public static final int ARG_BMI_WEIGHT = 9;
    public static final int ARG_BMI_INPUT = 10;
    public static final int ARG_BMI_SCALES = 11;
    public static final int ARG_BMI_OUTPUT = 12;
    public static final int BMM_IND_ARG_COUNT = 13;

    /** Bind-time tile count of {@code {T}_GroupedGemm} and {@code {T}_GroupedGemmInd}, constant across steps. */
    public static final int ARG_GG_N_TILES = 2;
    public static final int ARG_GG_X_DIV = 3;
    public static final int ARG_GG_X_STRIDE = 4;
    public static final int ARG_GG_IN_OFF = 5;
    public static final int ARG_GG_OUT_OFF = 6;
    public static final int ARG_GG_HAS_SCALE = 7;
    public static final int ARG_GG_EXPERT_BASE = 8;
    public static final int ARG_GG_EXPERT_COUNT = 9;
    public static final int ARG_GG_GEMM_TOKEN = 10;
    public static final int ARG_GG_GEMM_SLOT = 11;
    public static final int ARG_GG_WEIGHT = 12;
    public static final int ARG_GG_INPUT = 13;
    public static final int ARG_GG_SCALES = 14;
    public static final int ARG_GG_OUTPUT = 15;
    public static final int GG_ARG_COUNT = 16;

    /**
     * {@code {T}_GroupedGemmInd}: the scalars of {@code {T}_GroupedGemm} up to {@code has_scale}
     * ({@link #ARG_GG_N_TILES} to {@link #ARG_GG_HAS_SCALE}), then the aligned rows, the map and the bank.
     */
    public static final int ARG_GGI_GEMM_TOKEN = 8;
    public static final int ARG_GGI_GEMM_SLOT = 9;
    public static final int ARG_GGI_MAP = 10;
    public static final int ARG_GGI_WEIGHT = 11;
    public static final int ARG_GGI_INPUT = 12;
    public static final int ARG_GGI_SCALES = 13;
    public static final int ARG_GGI_OUTPUT = 14;
    public static final int GG_IND_ARG_COUNT = 15;

    /** Per-step token count, argument 0 of the router, hash-ids, weighted-sum, bucketize, align and
     * scale-input kernels. */
    public static final int ARG_N_TOKENS = 0;
    /** {@code n_pairs} of {@code LLM_MoE_EplbMap}. */
    public static final int ARG_EPLB_N_PAIRS = 0;
    /** Per-step {@code n_sel} of {@code LLM_AddId}. */
    public static final int ARG_ADD_ID_N_SEL = 0;
    public static final int ROUTER_ARG_COUNT = 16;
    /** Scalar index of the per-group selection flag of {@code LLM_MoE_Router}. */
    public static final int ARG_ROUTER_PER_GROUP = 11;
    /** Buffer indices of {@code LLM_MoE_Router} in signature order. */
    public static final int ARG_ROUTER_LOGITS = 12;
    public static final int ARG_ROUTER_EXP_PROBS_B = 13;
    public static final int ARG_ROUTER_IDS = 14;
    public static final int ARG_ROUTER_WEIGHTS = 15;
    public static final int HASH_IDS_ARG_COUNT = 5;
    public static final int WEIGHTED_SUM_ARG_COUNT = 7;
    public static final int ADD_ID_ARG_COUNT = 5;
    public static final int BUCKETIZE_ARG_COUNT = 8;
    public static final int BUCKET_COUNT_ARG_COUNT = 6;
    public static final int BUCKET_SCAN_ARG_COUNT = 7;
    public static final int BUCKET_SCATTER_ARG_COUNT = 8;
    public static final int ALIGN_ARG_COUNT = 10;
    public static final int SCALE_INPUT_ARG_COUNT = 6;
    public static final int ZERO_EXPERTS_ARG_COUNT = 6;
    public static final int ZERO_APPLY_ARG_COUNT = 7;

    private static final Map<GGMLType, Integer> TPS = buildTps();

    private MoeKernels() {
    }

    private static Map<GGMLType, Integer> buildTps() {
        Map<GGMLType, Integer> m = new EnumMap<>(GGMLType.class);
        for (GGMLType t : new GGMLType[]{GGMLType.F32, GGMLType.F16, GGMLType.BF16}) {
            m.put(t, 1);
        }
        for (GGMLType t : new GGMLType[]{GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0,
                GGMLType.IQ4_NL, GGMLType.MXFP4, GGMLType.F8_E5M2, GGMLType.F8_E4M3, GGMLType.F8_E4M3_B32,
                GGMLType.MXFP8, GGMLType.MXFP6_E3M2, GGMLType.MXFP6_E2M3}) {
            m.put(t, 4);
        }
        for (GGMLType t : new GGMLType[]{GGMLType.NVFP4, GGMLType.Q2_0, GGMLType.NF4, GGMLType.FP4_BNB}) {
            m.put(t, 8);
        }
        for (GGMLType t : new GGMLType[]{GGMLType.Q1_0, GGMLType.F8_E4M3_B128, GGMLType.I8_B128}) {
            m.put(t, 16);
        }
        for (GGMLType t : new GGMLType[]{GGMLType.IQ4_XS, GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K,
                GGMLType.Q6_K, GGMLType.Q8_K, GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ3_K, GGMLType.IQ3_KS,
                GGMLType.IQ2_XXS, GGMLType.IQ2_XS,
                GGMLType.IQ2_S, GGMLType.IQ1_S, GGMLType.IQ1_M, GGMLType.TQ1_0, GGMLType.TQ2_0}) {
            m.put(t, 32);
        }
        return Collections.unmodifiableMap(m);
    }

    /** The 41 weight types with id-matvec kernels (the F8_E5M2, MXFP6_E3M2 and MXFP6_E2M3 extension types
     * included). */
    public static Set<GGMLType> weightTypes() {
        return TPS.keySet();
    }

    public static boolean supports(GGMLType type) {
        return type != null && TPS.containsKey(type);
    }

    /**
     * Work-items that cooperate on one block of {@code type} (the family's TPS); the local size must be a multiple.
     *
     * @throws UnsupportedOperationException for a type without id-matvec kernels
     */
    public static int threadsPerBlock(GGMLType type) {
        return TPS.get(requireSupported(type));
    }

    /** {@code {T}_MatVecId} for a weight type. @throws UnsupportedOperationException for an unsupported type */
    public static String matVecId(GGMLType type) {
        return requireSupported(type).name() + MATVEC_ID_SUFFIX;
    }

    /**
     * {@code {T}_MatVecIdT} for a weight type: the transposed slot read of {@link #matVecId} over a slot block
     * stored {@code [in_dim x out_dim]} row-major (the deepseek2 absorbed {@code attn_k_b}), with the same
     * arguments, launch contract and {@link #matVecIdArgs} argument list.
     *
     * @throws UnsupportedOperationException for an unsupported type
     */
    public static String matVecIdT(GGMLType type) {
        return requireSupported(type).name() + MATVEC_ID_T_SUFFIX;
    }

    /** {@code {T}_MatVecIdGateUp} for a weight type. @throws UnsupportedOperationException for an unsupported type */
    public static String matVecIdGateUp(GGMLType type) {
        return requireSupported(type).name() + MATVEC_ID_GATE_UP_SUFFIX;
    }

    /** {@code {T}_BatchMatMulId} for a weight type. @throws UnsupportedOperationException for an unsupported type */
    public static String batchMatMulId(GGMLType type) {
        return requireSupported(type).name() + BATCH_MATMUL_ID_SUFFIX;
    }

    /**
     * {@code {T}_GroupedGemm} for a weight type: the type needs an expert and a tiled kernel.
     *
     * @throws UnsupportedOperationException for a type without one of them
     */
    public static String groupedGemm(GGMLType type) {
        requireSupported(type);
        if (!GemmKernels.hasTiled(type)) {
            throw new UnsupportedOperationException("no tiled GEMM kernel for the grouped expert GEMM of " + type);
        }
        return type.name() + GROUPED_GEMM_SUFFIX;
    }

    /**
     * Whether {@code type} has a tensor-core grouped GEMM: every grouped GEMM type except F32. The Vulkan shader set
     * omits BF16 too (its cooperative matrices take fp16 operands), so a backend's kernel set decides at last.
     */
    public static boolean hasGroupedGemmMma(GGMLType type) {
        return supports(type) && GemmKernels.hasTiled(type) && type != GGMLType.F32;
    }

    /**
     * {@code {T}_GroupedGemmMma} for a weight type: the tensor-core twin of {@link #groupedGemm}, with the same
     * arguments and geometry (fp16 operands, bfloat16 for BF16 experts, fp32 accumulation).
     *
     * @throws UnsupportedOperationException for a type without one ({@link #hasGroupedGemmMma})
     */
    public static String groupedGemmMma(GGMLType type) {
        if (!hasGroupedGemmMma(type)) {
            throw new UnsupportedOperationException("no tensor-core grouped expert GEMM for weight type " + type);
        }
        return type.name() + GROUPED_GEMM_MMA_SUFFIX;
    }

    /**
     * {@code {T}_GroupedGemmInd} for a weight type: the expert-cache form of {@link #groupedGemm}.
     *
     * @throws UnsupportedOperationException for a type without a grouped GEMM
     */
    public static String groupedGemmInd(GGMLType type) {
        return groupedGemm(type) + IND_SUFFIX;
    }

    /**
     * {@code {T}_GroupedGemmMmaInd} for a weight type: the expert-cache form of {@link #groupedGemmMma}.
     *
     * @throws UnsupportedOperationException for a type without one ({@link #hasGroupedGemmMma})
     */
    public static String groupedGemmMmaInd(GGMLType type) {
        return groupedGemmMma(type) + IND_SUFFIX;
    }

    /** {@code {T}_MatVecIdInd} for a weight type. @throws UnsupportedOperationException for an unsupported type */
    public static String matVecIdInd(GGMLType type) {
        return requireSupported(type).name() + MATVEC_ID_SUFFIX + IND_SUFFIX;
    }

    /** {@code {T}_MatVecIdGateUpInd} for a weight type. @throws UnsupportedOperationException for an unsupported type */
    public static String matVecIdGateUpInd(GGMLType type) {
        return requireSupported(type).name() + MATVEC_ID_GATE_UP_SUFFIX + IND_SUFFIX;
    }

    /** {@code {T}_BatchMatMulIdInd} for a weight type. @throws UnsupportedOperationException for an unsupported type */
    public static String batchMatMulIdInd(GGMLType type) {
        return requireSupported(type).name() + BATCH_MATMUL_ID_SUFFIX + IND_SUFFIX;
    }

    private static GGMLType requireSupported(GGMLType type) {
        if (!supports(type)) {
            throw new UnsupportedOperationException("no MoE expert matvec kernels for weight type " + type);
        }
        return type;
    }

    /**
     * Validates the local size of an id-matvec dispatch: a power of two, at most {@link #MAX_LOCAL}, a multiple of
     * 32 (full CUDA warps) and of the type's {@link #threadsPerBlock}.
     *
     * @throws IllegalArgumentException naming the type and the size
     */
    public static void checkIdLocal(GGMLType type, int local) {
        int tps = threadsPerBlock(type);
        if (local < 32 || local > MAX_LOCAL || Integer.bitCount(local) != 1 || local % tps != 0) {
            throw new IllegalArgumentException("id-matvec local size for " + type + " must be a power of two in [32, "
                    + MAX_LOCAL + "] and a multiple of " + tps + ", got " + local);
        }
    }

    /** {@code {outDim * local, nSel}} for {@code {T}_MatVecId} / {@code {T}_MatVecIdGateUp}. */
    public static long[] idMatVecGlobal(int outDim, int nSel, int local) {
        return new long[]{(long) positive("out_dim", outDim) * local, positive("n_sel", nSel)};
    }

    /** {@code {outDim * local, ceil(nRows / BMM_ROWS)}} for {@code {T}_BatchMatMulId}. */
    public static long[] batchMatMulIdGlobal(int outDim, int nRows, int local) {
        long tiles = (positive("n_rows", nRows) + (long) BMM_ROWS - 1) / BMM_ROWS;
        return new long[]{(long) positive("out_dim", outDim) * local, tiles};
    }

    /**
     * Worst-case number of aligned tiles of {@code nRows} sorted rows over {@code nExpert} experts: every bucket,
     * the out-of-range one included, rounds up to a multiple of {@link #ALIGN_TILE} and adds at most one whole tile
     * (the bound of {@code LLM_MoE_AlignBlocks} and the second grid dimension of {@code {T}_GroupedGemm}).
     *
     * @throws IllegalArgumentException for more than {@link #MAX_EXPERTS} experts
     */
    public static int groupedTiles(int nRows, int nExpert) {
        if (nExpert > MAX_EXPERTS) {
            throw new IllegalArgumentException("the grouped expert GEMM supports at most " + MAX_EXPERTS
                    + " experts, got " + nExpert);
        }
        positive("n_expert", nExpert);
        long nb = (long) positive("n_expert", nExpert) + 1;
        long tiles = ((long) positive("n_rows", nRows) + nb * (ALIGN_TILE - 1)) / ALIGN_TILE;
        if (tiles > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("grouped tile count overflows int for " + nRows + " rows and "
                    + nExpert + " experts");
        }
        return (int) tiles;
    }

    /** Positions of the aligned layout: {@code groupedTiles(nRows, nExpert) * ALIGN_TILE}. */
    public static long alignedPositions(int nRows, int nExpert) {
        return (long) groupedTiles(nRows, nExpert) * ALIGN_TILE;
    }

    /** {@code {ceil(outDim / 64) * 256, nTiles}} for {@code {T}_GroupedGemm}. */
    public static long[] groupedGemmGlobal(int outDim, int nTiles) {
        long rowTiles = ((long) positive("out_dim", outDim) + TiledGemmPlan.TILE_ROWS - 1) / TiledGemmPlan.TILE_ROWS;
        return new long[]{rowTiles * GROUPED_WORKGROUP, positive("n_tiles", nTiles)};
    }

    /** {@code {256, 1}}. */
    public static long[] groupedGemmLocal() {
        return new long[]{GROUPED_WORKGROUP, 1};
    }

    /**
     * Work-groups of {@code LLM_MoE_BucketCount} / {@code LLM_MoE_BucketScatter}: {@code ceil(nRows / chunk)}.
     *
     * @throws IllegalArgumentException for a non-positive row count or chunk
     */
    public static int bucketGroups(int nRows, int chunk) {
        return (int) (((long) positive("n_rows", nRows) + positive("chunk", chunk) - 1) / chunk);
    }

    /** Ints of the per-group bucket table of the multi-work-group sort: {@code bucketGroups * (nExpert + 1)}. */
    public static long bucketHistInts(int nRows, int nExpert, int chunk) {
        return (long) bucketGroups(nRows, chunk) * (positive("n_expert", nExpert) + 1L);
    }

    /**
     * {@code {groups * local}} for {@code LLM_MoE_AlignBlocks} over {@code nTiles} aligned tiles: one position per
     * work-item, at most {@link #MAX_ALIGN_GROUPS} work-groups.
     */
    public static long[] alignGlobal(int nTiles, int local) {
        long positions = (long) positive("n_tiles", nTiles) * ALIGN_TILE;
        long groups = Math.min(MAX_ALIGN_GROUPS, (positions + positive("local", local) - 1) / local);
        return new long[]{groups * local};
    }

    /** {@code nTokens * local}: one router work-group per token. */
    public static long[] routerGlobal(int nTokens, int local) {
        checkRouterLocal(local);
        return new long[]{(long) positive("n_tokens", nTokens) * local};
    }

    /** {@code roundUp(elements, local)} for the 1-D element kernels, at least one work-group. */
    public static long[] elementGlobal(long elements, int local) {
        if (local <= 0 || elements < 0) {
            throw new IllegalArgumentException("element grid needs local > 0 and elements >= 0, got " + local + ", "
                    + elements);
        }
        long groups = Math.max(1L, (elements + local - 1) / local);
        return new long[]{groups * local};
    }

    private static void checkRouterLocal(int local) {
        if (local < MIN_ROUTER_LOCAL || local > MAX_LOCAL || Integer.bitCount(local) != 1) {
            throw new IllegalArgumentException("router local size must be a power of two in [" + MIN_ROUTER_LOCAL
                    + ", " + MAX_LOCAL + "], got " + local);
        }
    }

    /**
     * Checks the router limits of the GPU kernel for a routing configuration and work-group size.
     *
     * @throws IllegalArgumentException when the configuration exceeds a kernel limit
     */
    public static void checkRouter(MoeRouting.Params p, int local) {
        checkRouterLimits(p);
        checkRouterLocal(local);
        if (p.nExpert() > 32 * local) {
            throw new IllegalArgumentException("router with local size " + local + " supports at most 32 * local = "
                    + 32 * local + " experts, got " + p.nExpert());
        }
    }

    /** The router limits that do not depend on the work-group size (the kernel's local-memory table sizes). */
    private static void checkRouterLimits(MoeRouting.Params p) {
        Objects.requireNonNull(p, "params");
        if (p.nExpert() > MAX_EXPERTS) {
            throw new IllegalArgumentException("router supports at most " + MAX_EXPERTS + " experts, got "
                    + p.nExpert());
        }
        if (p.k() > MAX_K) {
            throw new IllegalArgumentException("router supports at most " + MAX_K + " experts per token, got " + p.k());
        }
        if (p.nGroups() > MAX_GROUPS) {
            throw new IllegalArgumentException("router supports at most " + MAX_GROUPS + " expert groups, got "
                    + p.nGroups());
        }
    }

    /**
     * The {@code (limit, alpha, beta)} scalars of the gate-up kernel, with the {@code LLM_GLU} mapping: SITU passes
     * {@code (limit, situ_beta = p.beta(), situ_linear_beta = p.lambda())}, every other op
     * {@code (limit, p.alpha(), p.lambda())}.
     */
    public static float[] gluScalars(GluOp op, ActParams p) {
        return ActKernels.gluScalars(op, p);
    }

    /**
     * Arguments of {@code {T}_MatVecIdInd} in signature order; {@code weight} is an expert cache bank
     * whose local expert {@code map[expert]} holds the routed expert's rows.
     *
     * @throws IllegalArgumentException for a negative size or offset
     */
    public static Object[] matVecIdIndArgs(int outDim, int inDim, int nSel, int xStride, int inOff, int outOff,
                                           boolean scaled, GpuBuffer slots, GpuBuffer map, GpuBuffer weight,
                                           GpuBuffer input, GpuBuffer scales, GpuBuffer output) {
        nonNegative("x_stride", xStride);
        nonNegative("in_off", inOff);
        nonNegative("out_off", outOff);
        return new Object[]{positive("out_dim", outDim), positive("in_dim", inDim), positive("n_sel", nSel), xStride,
                inOff, outOff, scaled ? SCALE_SHARED : SCALE_NONE, req("slots", slots), req("map", map),
                req("weight", weight), req("input", input), req("scales", scales), req("output", output)};
    }

    /**
     * Arguments of {@code {T}_MatVecIdGateUpInd} in signature order; {@code scaleMode} is {@link #SCALE_NONE},
     * {@link #SCALE_SHARED} (fused tensor with {@code up_exps_s}) or {@link #SCALE_GATE_UP_PAIRS}.
     *
     * @throws IllegalArgumentException for {@link #SCALE_SHARED} on separate gate/up tensors, or a negative size or
     *                                  offset
     */
    public static Object[] matVecIdGateUpIndArgs(int outDim, int inDim, int nSel, int xStride, int inOff, GluOp op,
                                                 ActParams p, boolean fused, int scaleMode, GpuBuffer slots,
                                                 GpuBuffer map, GpuBuffer gate, GpuBuffer up, GpuBuffer input,
                                                 GpuBuffer scales, GpuBuffer output) {
        if (scaleMode < SCALE_NONE || scaleMode > SCALE_GATE_UP_PAIRS || (scaleMode == SCALE_SHARED && !fused)) {
            throw new IllegalArgumentException("gate-up scale mode " + scaleMode + " is invalid for "
                    + (fused ? "a fused" : "separate") + " gate/up tensor");
        }
        nonNegative("x_stride", xStride);
        nonNegative("in_off", inOff);
        float[] s = gluScalars(op, p);
        return new Object[]{positive("out_dim", outDim), positive("in_dim", inDim), positive("n_sel", nSel), xStride,
                inOff, op.code(), s[0], s[1], s[2], fused ? 1 : 0, scaleMode, req("slots", slots),
                req("map", map), req("gate", gate), req("up", up), req("input", input), req("scales", scales),
                req("output", output)};
    }

    /**
     * Arguments of {@code {T}_BatchMatMulIdInd} in signature order; {@code xDiv} is k for token rows, 1 for pair
     * rows; {@code weight} is an expert cache bank.
     *
     * @throws IllegalArgumentException for a non-positive size or a negative offset
     */
    public static Object[] batchMatMulIdIndArgs(int outDim, int inDim, int nRows, int xDiv, int inOff, boolean scaled,
                                                GpuBuffer rowToken, GpuBuffer rowSlot, GpuBuffer map,
                                                GpuBuffer weight, GpuBuffer input, GpuBuffer scales,
                                                GpuBuffer output) {
        nonNegative("in_off", inOff);
        return new Object[]{positive("out_dim", outDim), positive("in_dim", inDim), positive("n_rows", nRows),
                positive("x_div", xDiv), inOff, scaled ? SCALE_SHARED : SCALE_NONE, req("row_token", rowToken),
                req("row_slot", rowSlot), req("map", map), req("weight", weight), req("input", input),
                req("scales", scales), req("output", output)};
    }

    /**
     * Arguments of {@code {T}_MatVecId} in signature order; {@code weight} holds the experts
     * {@code [expertBase, expertBase + expertCount)} ({@code 0, n} for a tensor in one buffer).
     *
     * @throws IllegalArgumentException for a negative size or offset, or an empty or negative expert range
     */
    public static Object[] matVecIdArgs(int outDim, int inDim, int nSel, int xStride, int inOff, int outOff,
                                        boolean scaled, int expertBase, int expertCount, GpuBuffer slots,
                                        GpuBuffer weight, GpuBuffer input, GpuBuffer scales, GpuBuffer output) {
        nonNegative("x_stride", xStride);
        nonNegative("in_off", inOff);
        nonNegative("out_off", outOff);
        nonNegative("expert_base", expertBase);
        return new Object[]{positive("out_dim", outDim), positive("in_dim", inDim), positive("n_sel", nSel), xStride,
                inOff, outOff, scaled ? SCALE_SHARED : SCALE_NONE, expertBase, positive("expert_count", expertCount),
                req("slots", slots), req("weight", weight), req("input", input), req("scales", scales),
                req("output", output)};
    }

    /**
     * Arguments of {@code {T}_MatVecIdGateUp} in signature order; {@code scaleMode} is {@link #SCALE_NONE},
     * {@link #SCALE_SHARED} (fused tensor with {@code up_exps_s}) or {@link #SCALE_GATE_UP_PAIRS}; {@code gate} and
     * {@code up} both hold the experts {@code [expertBase, expertBase + expertCount)}.
     *
     * @throws IllegalArgumentException for {@link #SCALE_SHARED} on separate gate/up tensors, which llama.cpp
     *                                  scales independently, or for an empty or negative expert range
     */
    public static Object[] gateUpArgs(int outDim, int inDim, int nSel, int xStride, int inOff, GluOp op,
                                      ActParams p, boolean fused, int scaleMode, int expertBase, int expertCount,
                                      GpuBuffer slots, GpuBuffer gate, GpuBuffer up, GpuBuffer input,
                                      GpuBuffer scales, GpuBuffer output) {
        if (scaleMode < SCALE_NONE || scaleMode > SCALE_GATE_UP_PAIRS || (scaleMode == SCALE_SHARED && !fused)) {
            throw new IllegalArgumentException("gate-up scale mode " + scaleMode + " is invalid for "
                    + (fused ? "a fused" : "separate") + " gate/up tensor");
        }
        nonNegative("x_stride", xStride);
        nonNegative("in_off", inOff);
        nonNegative("expert_base", expertBase);
        float[] s = gluScalars(op, p);
        return new Object[]{positive("out_dim", outDim), positive("in_dim", inDim), positive("n_sel", nSel), xStride,
                inOff, op.code(), s[0], s[1], s[2], fused ? 1 : 0, scaleMode, expertBase,
                positive("expert_count", expertCount), req("slots", slots), req("gate", gate), req("up", up),
                req("input", input), req("scales", scales), req("output", output)};
    }

    /**
     * Arguments of {@code {T}_BatchMatMulId} in signature order; {@code xDiv} is k for token rows, 1 for pair rows;
     * {@code weight} holds the experts {@code [expertBase, expertBase + expertCount)}.
     *
     * @throws IllegalArgumentException for a non-positive size, a negative offset, or an empty or negative expert
     *                                  range
     */
    public static Object[] batchMatMulIdArgs(int outDim, int inDim, int nRows, int xDiv, int inOff, boolean scaled,
                                             int expertBase, int expertCount, GpuBuffer rowToken, GpuBuffer rowSlot,
                                             GpuBuffer weight, GpuBuffer input, GpuBuffer scales, GpuBuffer output) {
        nonNegative("in_off", inOff);
        nonNegative("expert_base", expertBase);
        return new Object[]{positive("out_dim", outDim), positive("in_dim", inDim), positive("n_rows", nRows),
                positive("x_div", xDiv), inOff, scaled ? SCALE_SHARED : SCALE_NONE, expertBase,
                positive("expert_count", expertCount), req("row_token", rowToken), req("row_slot", rowSlot),
                req("weight", weight), req("input", input), req("scales", scales), req("output", output)};
    }

    /**
     * Arguments of {@code {T}_GroupedGemm} (and {@code {T}_GroupedGemmMma}) in signature order; {@code xDiv} is k
     * for token rows, 1 for pair rows, and row {@code p / xDiv} starts at {@code input[inOff + (p / xDiv) * xStride]};
     * pair {@code p}'s output row starts at {@code output[outOff + p * outDim]}; {@code weight} holds the experts
     * {@code [expertBase, expertBase + expertCount)}; {@code nTiles} is the bind-time bound {@link #groupedTiles}.
     *
     * @throws IllegalArgumentException for a non-positive size, a row stride below {@code inDim}, a negative offset,
     *                                  or an empty or negative expert range
     */
    public static Object[] groupedGemmArgs(int outDim, int inDim, int nTiles, int xDiv, int xStride, int inOff,
                                           int outOff, boolean scaled, int expertBase, int expertCount,
                                           GpuBuffer gemmToken, GpuBuffer gemmSlot, GpuBuffer weight, GpuBuffer input,
                                           GpuBuffer scales, GpuBuffer output) {
        if (xStride < inDim) {
            throw new IllegalArgumentException("x_stride " + xStride + " is shorter than in_dim " + inDim);
        }
        nonNegative("in_off", inOff);
        nonNegative("out_off", outOff);
        nonNegative("expert_base", expertBase);
        return new Object[]{positive("out_dim", outDim), positive("in_dim", inDim), positive("n_tiles", nTiles),
                positive("x_div", xDiv), xStride, inOff, outOff, scaled ? SCALE_SHARED : SCALE_NONE, expertBase,
                positive("expert_count", expertCount), req("gemm_token", gemmToken), req("gemm_slot", gemmSlot),
                req("weight", weight), req("input", input), req("scales", scales), req("output", output)};
    }

    /**
     * Arguments of {@code {T}_GroupedGemmInd} (and {@code {T}_GroupedGemmMmaInd}) in signature order: those of
     * {@link #groupedGemmArgs} with the expert cache's {@code map} in place of the expert range; {@code weight} is
     * a bank whose local expert {@code map[e]} holds expert {@code e}, and tiles of an expert with a negative entry
     * are left to the host.
     *
     * @throws IllegalArgumentException for a non-positive size, a row stride below {@code inDim} or a negative offset
     */
    public static Object[] groupedGemmIndArgs(int outDim, int inDim, int nTiles, int xDiv, int xStride, int inOff,
                                              int outOff, boolean scaled, GpuBuffer gemmToken, GpuBuffer gemmSlot,
                                              GpuBuffer map, GpuBuffer weight, GpuBuffer input, GpuBuffer scales,
                                              GpuBuffer output) {
        if (xStride < inDim) {
            throw new IllegalArgumentException("x_stride " + xStride + " is shorter than in_dim " + inDim);
        }
        nonNegative("in_off", inOff);
        nonNegative("out_off", outOff);
        return new Object[]{positive("out_dim", outDim), positive("in_dim", inDim), positive("n_tiles", nTiles),
                positive("x_div", xDiv), xStride, inOff, outOff, scaled ? SCALE_SHARED : SCALE_NONE,
                req("gemm_token", gemmToken), req("gemm_slot", gemmSlot), req("map", map), req("weight", weight),
                req("input", input), req("scales", scales), req("output", output)};
    }

    /**
     * Arguments of {@code LLM_MoE_Router} in signature order.
     *
     * @throws IllegalArgumentException when the configuration exceeds the kernel's expert, k or group tables
     */
    public static Object[] routerArgs(MoeRouting.Params p, int nTokens, GpuBuffer logits, GpuBuffer expProbsB,
                                      GpuBuffer ids, GpuBuffer weights) {
        checkRouterLimits(p);
        return new Object[]{positive("n_tokens", nTokens), p.nExpert(), p.k(), p.gating().code(), p.nGroups(),
                p.nGroupUsed(), p.normWeights() ? 1 : 0, p.weightsScale(), p.hasBias() ? 1 : 0, p.selMode(),
                p.idDiv(), p.perGroup() ? 1 : 0, req("logits", logits), req("exp_probs_b", expProbsB),
                req("ids", ids), req("weights", weights)};
    }

    /** Arguments of {@code LLM_MoE_HashIds} in signature order. */
    public static Object[] hashIdsArgs(int nTokens, int k, GpuBuffer tokens, GpuBuffer tid2eid, GpuBuffer ids) {
        return new Object[]{positive("n_tokens", nTokens), positive("k", k), req("tokens", tokens),
                req("tid2eid", tid2eid), req("ids", ids)};
    }

    /** Arguments of {@code LLM_MoE_WeightedSum} in signature order; {@code flags} of {@link #WSUM_ACCUMULATE} / {@link #WSUM_UNWEIGHTED}. */
    public static Object[] weightedSumArgs(int nTokens, int nEmbd, int k, int flags, GpuBuffer expertOut,
                                           GpuBuffer weights, GpuBuffer dst) {
        if ((flags & ~(WSUM_ACCUMULATE | WSUM_UNWEIGHTED)) != 0) {
            throw new IllegalArgumentException("unknown weighted-sum flags " + flags);
        }
        return new Object[]{positive("n_tokens", nTokens), positive("n_embd", nEmbd), positive("k", k), flags,
                req("expert_out", expertOut), req("weights", weights), req("dst", dst)};
    }

    /** Arguments of {@code LLM_AddId} in signature order. */
    public static Object[] addIdArgs(int nSel, int dim, GpuBuffer slots, GpuBuffer bias, GpuBuffer data) {
        return new Object[]{positive("n_sel", nSel), positive("dim", dim), req("slots", slots), req("bias", bias),
                req("data", data)};
    }

    /** Arguments of {@code LLM_MoE_ZeroExperts} in signature order. */
    public static Object[] zeroExpertsArgs(int nTokens, int k, int nRealExperts, GpuBuffer ids, GpuBuffer weights,
                                           GpuBuffer zeroWeights) {
        if (nRealExperts <= 0) {
            throw new IllegalArgumentException("zero experts need a positive real expert count, got " + nRealExperts);
        }
        return new Object[]{positive("n_tokens", nTokens), positive("k", k), nRealExperts, req("ids", ids),
                req("weights", weights), req("zero_weights", zeroWeights)};
    }

    /** Arguments of {@code LLM_MoE_EplbMap} in signature order. */
    public static Object[] eplbMapArgs(int nPairs, int k, int nLogical, int mapStride, GpuBuffer ids,
                                       GpuBuffer replicaCount, GpuBuffer logicalToPhysical, GpuBuffer out) {
        if (mapStride < 1) {
            throw new IllegalArgumentException("the replica map stride must be positive, got " + mapStride);
        }
        return new Object[]{positive("n_pairs", nPairs), positive("k", k), positive("n_logical", nLogical), mapStride,
                req("ids", ids), req("replica_count", replicaCount),
                req("logical_to_physical", logicalToPhysical), req("out", out)};
    }

    /** Arguments of {@code LLM_MoE_ZeroApply} in signature order. */
    public static Object[] zeroApplyArgs(int nTokens, int nEmbd, int xOff, int dstOff, GpuBuffer x,
                                         GpuBuffer zeroWeights, GpuBuffer dst) {
        if (xOff < 0 || dstOff < 0) {
            throw new IllegalArgumentException("zero-expert offsets must be >= 0, got " + xOff + ", " + dstOff);
        }
        return new Object[]{positive("n_tokens", nTokens), positive("n_embd", nEmbd), xOff, dstOff,
                req("x", x), req("zero_weights", zeroWeights), req("dst", dst)};
    }

    /** Arguments of {@code LLM_MoE_Bucketize} in signature order. */
    public static Object[] bucketizeArgs(int nTokens, int k, int nExpert, GpuBuffer ids, GpuBuffer counts,
                                         GpuBuffer offsets, GpuBuffer rowToken, GpuBuffer rowSlot) {
        if (nExpert > MAX_EXPERTS) {
            throw new IllegalArgumentException("bucketize supports at most " + MAX_EXPERTS + " experts, got " + nExpert);
        }
        return new Object[]{positive("n_tokens", nTokens), positive("k", k), positive("n_expert", nExpert),
                req("ids", ids), req("counts", counts), req("offsets", offsets), req("row_token", rowToken),
                req("row_slot", rowSlot)};
    }

    /** Arguments of {@code LLM_MoE_BucketCount} in signature order. */
    public static Object[] bucketCountArgs(int nTokens, int k, int nExpert, int chunk, GpuBuffer ids, GpuBuffer hist) {
        checkSortExperts(nExpert);
        return new Object[]{positive("n_tokens", nTokens), positive("k", k), positive("n_expert", nExpert),
                positive("chunk", chunk), req("ids", ids), req("hist", hist)};
    }

    /** Arguments of {@code LLM_MoE_BucketScan} in signature order. */
    public static Object[] bucketScanArgs(int nTokens, int k, int nExpert, int chunk, GpuBuffer hist,
                                          GpuBuffer counts, GpuBuffer offsets) {
        checkSortExperts(nExpert);
        return new Object[]{positive("n_tokens", nTokens), positive("k", k), positive("n_expert", nExpert),
                positive("chunk", chunk), req("hist", hist), req("counts", counts), req("offsets", offsets)};
    }

    /** Arguments of {@code LLM_MoE_BucketScatter} in signature order. */
    public static Object[] bucketScatterArgs(int nTokens, int k, int nExpert, int chunk, GpuBuffer ids,
                                             GpuBuffer hist, GpuBuffer rowToken, GpuBuffer rowSlot) {
        checkSortExperts(nExpert);
        return new Object[]{positive("n_tokens", nTokens), positive("k", k), positive("n_expert", nExpert),
                positive("chunk", chunk), req("ids", ids), req("hist", hist), req("row_token", rowToken),
                req("row_slot", rowSlot)};
    }

    private static void checkSortExperts(int nExpert) {
        if (nExpert > MAX_EXPERTS) {
            throw new IllegalArgumentException("the expert sort supports at most " + MAX_EXPERTS + " experts, got "
                    + nExpert);
        }
    }

    /**
     * Arguments of {@code LLM_MoE_AlignBlocks} in signature order; {@code nTiles} is the bind-time bound
     * {@link #groupedTiles}.
     */
    public static Object[] alignBlocksArgs(int nTokens, int k, int nExpert, int nTiles, GpuBuffer counts,
                                           GpuBuffer offsets, GpuBuffer rowToken, GpuBuffer rowSlot,
                                           GpuBuffer gemmToken, GpuBuffer gemmSlot) {
        if (nExpert > MAX_EXPERTS) {
            throw new IllegalArgumentException("align supports at most " + MAX_EXPERTS + " experts, got " + nExpert);
        }
        return new Object[]{positive("n_tokens", nTokens), positive("k", k), positive("n_expert", nExpert),
                positive("n_tiles", nTiles), req("counts", counts), req("offsets", offsets),
                req("row_token", rowToken), req("row_slot", rowSlot), req("gemm_token", gemmToken),
                req("gemm_slot", gemmSlot)};
    }

    /** Arguments of {@code LLM_MoE_ScaleInput} in signature order. */
    public static Object[] scaleInputArgs(int nTokens, int nEmbd, int k, GpuBuffer x, GpuBuffer weights,
                                          GpuBuffer out) {
        return new Object[]{positive("n_tokens", nTokens), positive("n_embd", nEmbd), positive("k", k), req("x", x),
                req("weights", weights), req("out", out)};
    }

    private static int positive(String name, int v) {
        if (v <= 0) {
            throw new IllegalArgumentException(name + " must be positive, got " + v);
        }
        return v;
    }

    private static void nonNegative(String name, int v) {
        if (v < 0) {
            throw new IllegalArgumentException(name + " must be non-negative, got " + v);
        }
    }

    private static GpuBuffer req(String name, GpuBuffer b) {
        return Objects.requireNonNull(b, name + " buffer must not be null (bind a placeholder for unused inputs)");
    }
}
