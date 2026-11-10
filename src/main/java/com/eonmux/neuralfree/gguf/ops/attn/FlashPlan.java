package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;

/**
 * The flash-attention plans of one softmax-attention layer, made once at bind with
 * {@code smemFloats = LlmKernels.faSmemFloats()}: the decode plan, the prefill plan
 * ({@link FlashAttnTiling.Plan#TILED}, the tensor-core {@link FlashAttnTiling.Plan#MMA} on a backend with
 * {@link DeviceFeature#COOPERATIVE_MATRIX} and F16 caches, or the per-row decode fallback), the paged decode plan
 * and paged prefill kernel of a paged cache (with the CUDA grouped-query paged decode kernel where it serves the
 * caches), and the split-K partial capacity every plan's launches fit in. The
 * dense and paged decode plans and the tiled prefill plan share one form: the subgroup kernels where
 * {@link SubgroupFlashAttention#serves} holds, the portable ones otherwise, with the kernel names of
 * {@link #decodeKernel}, {@link #pagedDecodeKernel}, {@link #tiledKernel} and {@link #pagedPrefillKernel}.
 * Immutable.
 */
final class FlashPlan {

    private final FlashAttnTiling decode;
    private final String decodeKernel;
    private final String pagedDecodeKernel;
    private final FlashAttnTiling prefill;
    private final String tiledKernel;
    private final String mmaKernel;
    private final FlashAttnTiling paged;
    private final String pagedPrefillKernel;
    private final String pagedGqaKernel;
    private final long partialCapacity;
    private final boolean depthRouted;

    /**
     * Largest query heads per KV head for which the one-row step leaves the MMA prefill past one workgroup of
     * keys. The decode plan runs one query head per workgroup and reads the group's K/V rows once per head, while
     * the MMA prefill stages each K/V tile once for the whole group, so the decode route's gain shrinks as the
     * group grows. Paired alternating CUDA runs (RTX 5080 Laptop, 128 tokens generated after 512 cached): the
     * 0.6B Q8_0 model (16 query and 8 KV heads) ran 320 against 277 t/s on the decode route, the 4B models (32
     * and 8) 105 against 122 t/s (Q4_K_M) and 95 against 100 t/s (Q8_0).
     */
    static final int DEPTH_ROUTE_MAX_GROUP = 2;

    private FlashPlan(FlashAttnTiling decode, String decodeKernel, String pagedDecodeKernel, FlashAttnTiling prefill,
                      String tiledKernel, String mmaKernel, FlashAttnTiling paged, String pagedPrefillKernel,
                      String pagedGqaKernel, long partialCapacity, boolean depthRouted) {
        this.decode = decode;
        this.decodeKernel = decodeKernel;
        this.pagedDecodeKernel = pagedDecodeKernel;
        this.prefill = prefill;
        this.tiledKernel = tiledKernel;
        this.mmaKernel = mmaKernel;
        this.paged = paged;
        this.pagedPrefillKernel = pagedPrefillKernel;
        this.pagedGqaKernel = pagedGqaKernel;
        this.partialCapacity = partialCapacity;
        this.depthRouted = depthRouted;
    }

    /**
     * The plans of attention {@code a} over {@code keys}, with the paged plans when {@code pg} is not null.
     *
     * @throws IllegalArgumentException when the geometry fits no kernel, or the partials exceed one buffer
     */
    static FlashPlan of(LlmKernels kernels, AttentionSpec a, FlashKeys keys, PagedFlashGeometry pg, int ubatch) {
        int smem = kernels.faSmemFloats();
        int cu = Math.max(1, kernels.caps().getComputeUnits());
        int rows = keys.rows();
        boolean fast = FlashAttnTiling.decodeFastPath(kernels.caps().hasFeature(DeviceFeature.SUBGROUP_SHUFFLE),
                keys.typeK(), keys.typeV(), a.headDimK(), a.headDimV());
        boolean subgroup = SubgroupFlashAttention.serves(kernels, keys.typeK(), keys.typeV(), a.headDimK(),
                a.headDimV());
        FlashAttnTiling decode = subgroup
                ? FlashAttnTiling.decodeSubgroup(a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(), rows, smem, cu)
                : FlashAttnTiling.decode(a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(), rows, smem, cu, fast);
        String decodeKernel = subgroup ? SubgroupFlashAttention.kernel(keys.typeK())
                : AttnKernels.flashDecode(keys.typeK(), keys.typeV());
        String pagedDecodeKernel = subgroup ? SubgroupFlashAttention.pagedKernel(keys.typeK())
                : AttnKernels.flashDecodePaged(keys.typeK(), keys.typeV());
        String mmaName = mmaKernelName(kernels.caps().hasFeature(DeviceFeature.COOPERATIVE_MATRIX), keys.typeK(),
                keys.typeV(), a.headDimK(), a.headDimV());
        String mma = mmaName != null && kernels.has(mmaName) ? mmaName : null;
        FlashAttnTiling prefill;
        if (mma != null) {
            prefill = FlashAttnTiling.mma(a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(), rows, smem, cu);
        } else if (subgroup) {
            prefill = FlashAttnTiling.prefillSubgroup(a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(), rows, smem,
                    cu);
        } else {
            prefill = FlashAttnTiling.prefill(a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(), rows, smem, cu);
        }
        String tiledKernel = prefill.subgroup() ? SubgroupFlashAttention.prefillKernel(keys.typeK())
                : AttnKernels.flashAttn(keys.typeK(), keys.typeV());
        long cap = decode.decodePartialFloats();
        if (prefill.plan() != FlashAttnTiling.Plan.PER_ROW_DECODE) {
            cap = Math.max(cap, prefillPartialFloats(prefill, ubatch, cu));
        }
        FlashAttnTiling paged = null;
        String pagedPrefill = null;
        String pagedGqa = null;
        if (pg != null) {
            pagedPrefill = pagedPrefillKernel(kernels, prefill, keys.typeK(), keys.typeV());
            paged = subgroup
                    ? FlashAttnTiling.decodePagedSubgroup(pg.maxLaunchSeq(), a.nHead(), a.nHeadKv(), a.headDimK(),
                            a.headDimV(), pg.slotCapacity(), smem, cu)
                    : FlashAttnTiling.decodePaged(pg.maxLaunchSeq(), a.nHead(), a.nHeadKv(), a.headDimK(),
                            a.headDimV(), pg.slotCapacity(), smem, cu, fast);
            pagedGqa = subgroup ? null : pagedGqaKernel(kernels, paged, keys);
            cap = Math.max(cap, paged.decodePagedPartialFloats(pg.maxLaunchSeq()));
        }
        if (cap > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("flash partials of " + cap + " floats exceed one buffer");
        }
        return new FlashPlan(decode, decodeKernel, pagedDecodeKernel, prefill, tiledKernel, mma, paged, pagedPrefill,
                pagedGqa, cap,
                depthRouted(mma, fast, pg != null, a.nHead(), a.nHeadKv()));
    }

    /**
     * The tensor-core prefill kernel name for these caches and head dims when the backend reports the
     * {@link DeviceFeature#COOPERATIVE_MATRIX} feature: F16 K and V caches only, and head dims that are multiples of
     * 16 of at most {@link FlashAttnTiling#MMA_MAX_HEAD_DIM}. Null otherwise. The caller still requires
     * {@code kernels.has(name)} before preparing it. The dense launches run the base kernel (its identity paging,
     * page_shift = -1) because a dense cache has no page table; the paged kernel {@code LLM_FlashAttnMmaPaged_F16}
     * serves launches over a paginated cache.
     */
    static String mmaKernelName(boolean cooperativeMatrix, KvDType typeK, KvDType typeV, int dK, int dV) {
        if (!cooperativeMatrix || typeK != KvDType.F16 || typeV != KvDType.F16) {
            return null;
        }
        if (dK > FlashAttnTiling.MMA_MAX_HEAD_DIM || dV > FlashAttnTiling.MMA_MAX_HEAD_DIM
                || dK % 16 != 0 || dV % 16 != 0) {
            return null;
        }
        return AttnKernels.flashAttnMma(typeK);
    }

    /**
     * The paged prefill kernel of the prefill plan: the tiled {@code LLM_FlashAttnPaged_<KV>} (its subgroup form
     * {@code LLM_FlashAttnPagedSg_<KV>} under a subgroup plan), or the tensor-core {@code LLM_FlashAttnMmaPaged_F16}
     * of an MMA plan (same arguments) when the backend has it; null when the plan's segments run the per-row paged
     * decode fallback ({@link PagedRowFallback}).
     */
    private static String pagedPrefillKernel(LlmKernels kernels, FlashAttnTiling prefill, KvDType typeK,
                                             KvDType typeV) {
        if (prefill.plan() == FlashAttnTiling.Plan.TILED) {
            return prefill.subgroup() ? SubgroupFlashAttention.pagedPrefillKernel(typeK)
                    : AttnKernels.flashAttnPaged(typeK, typeV);
        }
        String mma = prefill.plan() == FlashAttnTiling.Plan.MMA ? AttnKernels.flashAttnMmaPaged(typeK) : null;
        return mma != null && kernels.has(mma) ? mma : null;
    }

    /**
     * The grouped-query paged decode kernel ({@link AttnKernels#flashDecodePagedGqa}) of the paged decode plan when
     * the backend has it and it serves the caches: one F16 or BF16 type for K and V, no V alias, row strides of
     * whole 16-byte vectors, and a plan whose geometry and partial buffer it fits
     * ({@link FlashAttnTiling#gqaDecodeUsable}); null otherwise.
     */
    static String pagedGqaKernel(LlmKernels kernels, FlashAttnTiling paged, FlashKeys keys) {
        String name = AttnKernels.flashDecodePagedGqa(keys.typeK(), keys.typeV());
        if (name == null || keys.vAliasesK() || keys.kRowStride() % 8 != 0 || keys.vRowStride() % 8 != 0
                || !paged.gqaDecodeUsable()) {
            return null;
        }
        return kernels.has(name) ? name : null;
    }

    /**
     * Partial floats a tiled prefill may split into: splits are chosen only while the call has fewer than two
     * workgroups per compute unit ({@link FlashAttnTiling#prefillSplit}), i.e. for at most
     * {@code ceil(2 * CU / (n_head_kv * groupTiles)) * bq} tokens.
     */
    private static long prefillPartialFloats(FlashAttnTiling p, int ubatch, int cu) {
        long groupsPerBlock = (long) p.nHeadKv() * p.groupTiles();
        long blocks = (2L * cu + groupsPerBlock - 1) / groupsPerBlock;
        long tokens = Math.min(ubatch, blocks * p.bq());
        return FlashAttnTiling.partialFloats(tokens * p.nHead(), FlashAttnTiling.MAX_PREFILL_SPLITS, p.dV());
    }

    /** The decode plan of one query row. */
    FlashAttnTiling decode() {
        return decode;
    }

    /** The kernel of {@link #decode}'s launches. */
    String decodeKernel() {
        return decodeKernel;
    }

    /** The kernel of {@link #paged}'s launches (named whether or not the layer has a paged cache). */
    String pagedDecodeKernel() {
        return pagedDecodeKernel;
    }

    /** The prefill plan of a multi-token step. */
    FlashAttnTiling prefill() {
        return prefill;
    }

    /** The kernel of a {@link FlashAttnTiling.Plan#TILED} prefill plan's launches. */
    String tiledKernel() {
        return tiledKernel;
    }

    /** The tensor-core prefill kernel of an MMA plan; null otherwise. */
    String mmaKernel() {
        return mmaKernel;
    }

    /**
     * Whether every one-row step of this plan runs the MMA prefill kernel, whatever the step's depth: the plan has
     * an mma kernel and its one-row route does not follow the step's depth ({@link #depthRouted}). The step-level
     * overload {@link #routesDecodeThroughPrefill(int)} adds the per-step rule on top of this.
     */
    boolean routesDecodeThroughPrefill() {
        return mmaKernel != null && !depthRouted;
    }

    /** Whether the one-row step at {@code liveKeys} live keys runs the MMA prefill kernel ({@link #depthRouted}). */
    boolean routesDecodeThroughPrefill(int liveKeys) {
        return routesDecodeThroughPrefill(mmaKernel, depthRouted, liveKeys);
    }

    /**
     * The one-row route rule: with an mma kernel, a depth-routed plan runs the MMA prefill for steps over at most
     * {@link FlashAttnTiling#WG} live keys (the workgroup bound at which the decode plan stops splitting) and the
     * decode kernel past it; any other plan with an mma kernel runs the MMA prefill at every step, and a plan
     * without one runs the decode kernel. Paired CUDA runs over 128 generated tokens (every step at or under
     * the bound) put the MMA route 3-7% ahead of the decode route on every model tried.
     */
    static boolean routesDecodeThroughPrefill(String mmaKernel, boolean depthRouted, int liveKeys) {
        return mmaKernel != null && (!depthRouted || liveKeys <= FlashAttnTiling.WG);
    }

    /**
     * Whether the one-row route follows the step's live key count: an mma kernel exists, the decode kernels' shuffle
     * fast path serves the caches and head dims, the cache is dense (a paged cache's per-sequence key count is not
     * derivable from the step position) and the GQA group has at most {@link #DEPTH_ROUTE_MAX_GROUP} query heads.
     * The decode replay recaptures the one trunk whose route flips (at most once per generation, at the bound).
     */
    static boolean depthRouted(String mmaKernel, boolean decodeFastPath, boolean paged, int nHead, int nHeadKv) {
        return mmaKernel != null && decodeFastPath && !paged && nHead <= DEPTH_ROUTE_MAX_GROUP * nHeadKv;
    }

    /** Whether the one-row route follows the step's live key count ({@link #depthRouted}). */
    boolean routesDecodeByDepth() {
        return depthRouted;
    }

    /** The decode plan of the paged launches; null without a paged cache. */
    FlashAttnTiling paged() {
        return paged;
    }

    /** The paged prefill kernel; null without a paged cache or when segments run the per-row fallback. */
    String pagedPrefillKernel() {
        return pagedPrefillKernel;
    }

    /** The grouped-query paged decode kernel ({@link #pagedGqaKernel}); null when the paged decode serves alone. */
    String pagedGqaKernel() {
        return pagedGqaKernel;
    }

    /** Floats of the split-K partial buffer every launch of these plans fits in. */
    long partialCapacity() {
        return partialCapacity;
    }
}
