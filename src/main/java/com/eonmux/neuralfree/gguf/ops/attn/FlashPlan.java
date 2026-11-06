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
 * and paged prefill kernel of a paged cache, and the split-K partial capacity every plan's launches fit in.
 * Immutable.
 */
final class FlashPlan {

    private final FlashAttnTiling decode;
    private final FlashAttnTiling prefill;
    private final String mmaKernel;
    private final FlashAttnTiling paged;
    private final String pagedPrefillKernel;
    private final long partialCapacity;

    private FlashPlan(FlashAttnTiling decode, FlashAttnTiling prefill, String mmaKernel, FlashAttnTiling paged,
                      String pagedPrefillKernel, long partialCapacity) {
        this.decode = decode;
        this.prefill = prefill;
        this.mmaKernel = mmaKernel;
        this.paged = paged;
        this.pagedPrefillKernel = pagedPrefillKernel;
        this.partialCapacity = partialCapacity;
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
        FlashAttnTiling decode = FlashAttnTiling.decode(a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(), rows,
                smem, cu);
        String mmaName = mmaKernelName(kernels.caps().hasFeature(DeviceFeature.COOPERATIVE_MATRIX), keys.typeK(),
                keys.typeV(), a.headDimK(), a.headDimV());
        String mma = mmaName != null && kernels.has(mmaName) ? mmaName : null;
        FlashAttnTiling prefill = mma != null
                ? FlashAttnTiling.mma(a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(), rows, smem, cu)
                : FlashAttnTiling.prefill(a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(), rows, smem, cu);
        long cap = decode.decodePartialFloats();
        if (prefill.plan() != FlashAttnTiling.Plan.PER_ROW_DECODE) {
            cap = Math.max(cap, prefillPartialFloats(prefill, ubatch, cu));
        }
        FlashAttnTiling paged = null;
        String pagedPrefill = null;
        if (pg != null) {
            pagedPrefill = pagedPrefillKernel(kernels, prefill, keys.typeK(), keys.typeV());
            paged = FlashAttnTiling.decodePaged(pg.maxLaunchSeq(), a.nHead(), a.nHeadKv(), a.headDimK(),
                    a.headDimV(), pg.slotCapacity(), smem, cu);
            cap = Math.max(cap, paged.decodePagedPartialFloats(pg.maxLaunchSeq()));
        }
        if (cap > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("flash partials of " + cap + " floats exceed one buffer");
        }
        return new FlashPlan(decode, prefill, mma, paged, pagedPrefill, cap);
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
     * The paged prefill kernel of the prefill plan: the tiled {@code LLM_FlashAttnPaged_<KV>}, or the tensor-core
     * {@code LLM_FlashAttnMmaPaged_F16} of an MMA plan (same arguments) when the backend has it; null when the
     * plan's segments run the per-row paged decode fallback ({@link PagedRowFallback}).
     */
    private static String pagedPrefillKernel(LlmKernels kernels, FlashAttnTiling prefill, KvDType typeK,
                                             KvDType typeV) {
        if (prefill.plan() == FlashAttnTiling.Plan.TILED) {
            return AttnKernels.flashAttnPaged(typeK, typeV);
        }
        String mma = prefill.plan() == FlashAttnTiling.Plan.MMA ? AttnKernels.flashAttnMmaPaged(typeK) : null;
        return mma != null && kernels.has(mma) ? mma : null;
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

    /** The prefill plan of a multi-token step. */
    FlashAttnTiling prefill() {
        return prefill;
    }

    /** The tensor-core prefill kernel of an MMA plan; null otherwise. */
    String mmaKernel() {
        return mmaKernel;
    }

    /** The decode plan of the paged launches; null without a paged cache. */
    FlashAttnTiling paged() {
        return paged;
    }

    /** The paged prefill kernel; null without a paged cache or when segments run the per-row fallback. */
    String pagedPrefillKernel() {
        return pagedPrefillKernel;
    }

    /** Floats of the split-K partial buffer every launch of these plans fits in. */
    long partialCapacity() {
        return partialCapacity;
    }
}
