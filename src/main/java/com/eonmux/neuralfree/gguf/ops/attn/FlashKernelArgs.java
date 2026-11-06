package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;

import java.util.Objects;

/**
 * The bound argument lists of one layer's flash kernels, packed at the {@code ARG_FD_*}, {@code ARG_FA_*},
 * {@code ARG_FDP_*}, {@code ARG_FAP_*} and {@code ARG_FC_*} indices of {@link AttnKernels}: the attention's head
 * geometry, score scale, softcap, ALiBi bias, sliding window and T5 bias mode, the key source ({@link FlashKeys}),
 * the Q and output rows, the split-K partials, the sinks, relative-bias and FP8 KV-scale tables (a read-only
 * placeholder for an absent one). Per-launch scalars (token and split counts, key range, offsets) start at the values
 * the first launch re-sets. Immutable.
 */
final class FlashKernelArgs {

    private final AttentionSpec a;
    private final FlashKeys keys;
    private final GpuBuffer q;
    private final GpuBuffer qPos;
    private final GpuBuffer out;
    private final GpuBuffer partial;
    private final GpuBuffer sinks;
    private final GpuBuffer sinkK;
    private final GpuBuffer sinkV;
    private final GpuBuffer relBias;
    private final GpuBuffer kvScales;
    private final GpuBuffer treeMask;
    private final boolean hasSinks;
    private final int flags;

    /**
     * @param sinks       the F32 {@code attn_sinks}, or null without sinks
     * @param sinkK       the F32 learned static sink keys {@code [n_sink][n_head_kv][d_k]}, or null without them
     * @param sinkV       the F32 learned static sink values {@code [n_sink][n_head_kv][d_v]}, or null without them
     * @param relBias     the F32 T5 relative-bias table, or the Inkling per-row relative-logit table, or null
     *                    without a relative bias
     * @param kvScales    the F32 FP8 KV-cache scales ({@code n_head_kv} K values then {@code n_head_kv} V values,
     *                    {@code common/kv_types.h}); never null, {@code kvScalesOnes()} without scales (the kernels
     *                    read the scales unconditionally, so the identity ones keep an unscaled cache unchanged). A
     *                    per-token-head cache provides its interleaved row scales and a vendor cache its side buffer
     *                    through {@link FlashKeys} instead.
     * @param placeholder the read-only float buffer bound in place of an absent table, and in place of a tree mask
     *                    outside a tree verify step (there {@code n_fresh} is 0 and the kernels never read it)
     */
    FlashKernelArgs(AttentionSpec a, FlashKeys keys, GpuBuffer q, GpuBuffer qPos, GpuBuffer out, GpuBuffer partial,
                    GpuBuffer sinks, GpuBuffer sinkK, GpuBuffer sinkV, GpuBuffer relBias, GpuBuffer kvScales,
                    GpuBuffer placeholder) {
        this.a = Objects.requireNonNull(a, "a");
        this.keys = Objects.requireNonNull(keys, "keys");
        this.q = Objects.requireNonNull(q, "q");
        this.qPos = Objects.requireNonNull(qPos, "qPos");
        this.out = Objects.requireNonNull(out, "out");
        this.partial = Objects.requireNonNull(partial, "partial");
        Objects.requireNonNull(placeholder, "placeholder");
        this.hasSinks = sinks != null;
        this.sinks = hasSinks ? sinks : placeholder;
        this.sinkK = sinkK;
        this.sinkV = sinkV;
        this.relBias = relBias != null ? relBias : placeholder;
        this.kvScales = keys.vendorSide() != null ? keys.vendorSide()
                : keys.rowScales() != null ? keys.rowScales()
                : Objects.requireNonNull(kvScales, "kvScales");
        this.treeMask = placeholder;
        this.flags = AttnKernels.attnFlags(a.causal(), hasSinks, false)
                | (keys.vAliasesK() ? AttnKernels.FLAG_V_ALIAS : 0);
    }

    /** The attention whose kernels these arguments drive. */
    AttentionSpec attention() {
        return a;
    }

    /**
     * The same arguments with another split-K partial buffer: the dual-chunk stages each write their own
     * {@code n_split = 1} partial, and one launch per stage binds its buffer here.
     */
    FlashKernelArgs withPartial(GpuBuffer p) {
        Objects.requireNonNull(p, "p");
        return p == partial ? this : new FlashKernelArgs(this, p);
    }

    private FlashKernelArgs(FlashKernelArgs o, GpuBuffer p) {
        this.a = o.a;
        this.keys = o.keys;
        this.q = o.q;
        this.qPos = o.qPos;
        this.out = o.out;
        this.partial = p;
        this.sinks = o.sinks;
        this.sinkK = o.sinkK;
        this.sinkV = o.sinkV;
        this.relBias = o.relBias;
        this.kvScales = o.kvScales;
        this.treeMask = o.treeMask;
        this.hasSinks = o.hasSinks;
        this.flags = o.flags;
    }

    /** The key and value rows the kernels read. */
    FlashKeys keys() {
        return keys;
    }

    /** The causal and sink bits of the {@code flags} argument; each launch adds its direct bit. */
    int flags() {
        return flags;
    }

    /** The attention rows the launches (and the dual-chunk merge) write. */
    GpuBuffer out() {
        return out;
    }

    /** The query rows the kernels read. */
    GpuBuffer q() {
        return q;
    }

    /** The query-position buffer the prefill kernels read. */
    GpuBuffer qPos() {
        return qPos;
    }

    /** The split-K partial buffer. */
    GpuBuffer partial() {
        return partial;
    }

    /** The learned static sink keys, or null without them. */
    GpuBuffer staticSinkK() {
        return sinkK;
    }

    /** The learned static sink values, or null without them. */
    GpuBuffer staticSinkV() {
        return sinkV;
    }

    /** {@code LLM_FlashDecode_*} of one query row at offset 0, one split, the key range {@code [0, 1)}. */
    Object[] decode(FlashAttnTiling plan) {
        Object[] args = new Object[AttnKernels.FD_ARG_COUNT];
        decodeScalars(args, plan, 1);
        args[AttnKernels.ARG_FD_BUF_Q] = q;
        args[AttnKernels.ARG_FD_BUF_K] = keys.k();
        args[AttnKernels.ARG_FD_BUF_V] = keys.v();
        args[AttnKernels.ARG_FD_BUF_KV_POS] = keys.kvPos();
        args[AttnKernels.ARG_FD_BUF_SINKS] = sinks;
        args[AttnKernels.ARG_FD_BUF_REL_BIAS] = relBias;
        args[AttnKernels.ARG_FD_BUF_OUT] = out;
        args[AttnKernels.ARG_FD_BUF_PARTIAL] = partial;
        args[AttnKernels.ARG_FD_BUF_KV_SCALES] = kvScales;
        args[AttnKernels.ARG_FD_BUF_TREE_MASK] = treeMask;
        return AttnKernels.withKvTypes(args, keys.typeK(), keys.typeV());
    }

    /** {@code LLM_FlashAttn_*} over {@code nTokens} rows, one split, the key range {@code [0, 1)}. */
    Object[] tiledPrefill(FlashAttnTiling plan, int nTokens) {
        return AttnKernels.withKvTypes(prefill(plan, nTokens), keys.typeK(), keys.typeV());
    }

    /** {@code LLM_FlashAttnMma_*}: the tiled prefill's arguments, without the KV type codes or the scale buffers. */
    Object[] mmaPrefill(FlashAttnTiling plan, int nTokens) {
        Object[] args = new Object[AttnKernels.FA_MMA_ARG_COUNT];
        prefillScalars(args, plan, nTokens, 1);
        args[AttnKernels.ARG_FA_BUF_Q] = q;
        args[AttnKernels.ARG_FA_BUF_K] = keys.k();
        args[AttnKernels.ARG_FA_BUF_V] = keys.v();
        args[AttnKernels.ARG_FA_BUF_Q_POS] = qPos;
        args[AttnKernels.ARG_FA_BUF_KV_POS] = keys.kvPos();
        args[AttnKernels.ARG_FA_BUF_SINKS] = sinks;
        args[AttnKernels.ARG_FA_BUF_REL_BIAS] = relBias;
        args[AttnKernels.ARG_FA_BUF_OUT] = out;
        args[AttnKernels.ARG_FA_BUF_PARTIAL] = partial;
        args[AttnKernels.ARG_FA_MMA_BUF_TREE_MASK] = treeMask;
        return args;
    }

    private Object[] prefill(FlashAttnTiling plan, int nTokens) {
        return prefill(plan, nTokens, AttnKernels.FA_ARG_COUNT, true);
    }

    private Object[] prefill(FlashAttnTiling plan, int nTokens, int length, boolean withKvScales) {
        Object[] args = new Object[length];
        prefillScalars(args, plan, nTokens, 1);
        args[AttnKernels.ARG_FA_BUF_Q] = q;
        args[AttnKernels.ARG_FA_BUF_K] = keys.k();
        args[AttnKernels.ARG_FA_BUF_V] = keys.v();
        args[AttnKernels.ARG_FA_BUF_Q_POS] = qPos;
        args[AttnKernels.ARG_FA_BUF_KV_POS] = keys.kvPos();
        args[AttnKernels.ARG_FA_BUF_SINKS] = sinks;
        args[AttnKernels.ARG_FA_BUF_REL_BIAS] = relBias;
        args[AttnKernels.ARG_FA_BUF_OUT] = out;
        args[AttnKernels.ARG_FA_BUF_PARTIAL] = partial;
        if (withKvScales) {
            args[AttnKernels.ARG_FA_BUF_KV_SCALES] = kvScales;
            args[AttnKernels.ARG_FA_BUF_TREE_MASK] = treeMask;
        }
        return args;
    }

    /**
     * {@code LLM_FlashDecodePaged_*}: the decode scalars with {@code kv_start}/{@code kv_end}/{@code q_pos} at 0
     * (the kernel reads the per-sequence buffers instead), one sequence, then the page geometry and the
     * per-sequence buffers of {@code pg}.
     */
    Object[] pagedDecode(FlashAttnTiling plan, PagedFlashGeometry pg) {
        Object[] args = new Object[AttnKernels.FDP_ARG_COUNT];
        decodeScalars(args, plan, 0);
        args[AttnKernels.ARG_FDP_PAGE_SHIFT] = pg.pageShift();
        args[AttnKernels.ARG_FDP_MAX_PAGES] = pg.maxPages();
        args[AttnKernels.ARG_FDP_N_SEQ] = 1;
        args[AttnKernels.ARG_FDP_Q_SEQ_STRIDE] = a.qDim();
        args[AttnKernels.ARG_FDP_OUT_SEQ_STRIDE] = a.outDim();
        args[AttnKernels.ARG_FDP_BUF_Q] = q;
        args[AttnKernels.ARG_FDP_BUF_K] = keys.k();
        args[AttnKernels.ARG_FDP_BUF_V] = keys.v();
        args[AttnKernels.ARG_FDP_BUF_KV_POS] = keys.kvPos();
        args[AttnKernels.ARG_FDP_BUF_SINKS] = sinks;
        args[AttnKernels.ARG_FDP_BUF_REL_BIAS] = relBias;
        args[AttnKernels.ARG_FDP_BUF_OUT] = out;
        args[AttnKernels.ARG_FDP_BUF_PARTIAL] = partial;
        args[AttnKernels.ARG_FDP_BUF_PAGE_TABLE] = pg.pageTable();
        args[AttnKernels.ARG_FDP_BUF_SEQ_KV_LEN] = pg.seqKvLen();
        args[AttnKernels.ARG_FDP_BUF_Q_POS_SEQ] = pg.qPosSeq();
        args[AttnKernels.ARG_FDP_BUF_KV_SCALES] = kvScales;
        args[AttnKernels.ARG_FDP_BUF_TREE_MASK] = treeMask;
        return AttnKernels.pagedWithKvTypes(args, keys.typeK(), keys.typeV());
    }

    /**
     * {@code LLM_FlashAttnPaged_*} (and {@code LLM_FlashAttnMmaPaged_F16}): the prefill scalars of one row and an
     * empty key range, then {@code page_shift}, {@code pt_off} and the page table; the query positions are the
     * segment positions of {@code pg}. The MMA kernel takes no FP8 scales ({@code mma} true drops the
     * {@code kv_scales} buffer).
     */
    Object[] pagedPrefill(FlashAttnTiling plan, PagedFlashGeometry pg, boolean mma) {
        Object[] args = new Object[mma ? AttnKernels.FAP_MMA_ARG_COUNT : AttnKernels.FAP_ARG_COUNT];
        prefillScalars(args, plan, 1, 0);
        args[AttnKernels.ARG_FAP_PAGE_SHIFT] = pg.pageShift();
        args[AttnKernels.ARG_FAP_PT_OFF] = 0;
        args[AttnKernels.ARG_FAP_BUF_Q] = q;
        args[AttnKernels.ARG_FAP_BUF_K] = keys.k();
        args[AttnKernels.ARG_FAP_BUF_V] = keys.v();
        args[AttnKernels.ARG_FAP_BUF_Q_POS] = pg.segPos();
        args[AttnKernels.ARG_FAP_BUF_KV_POS] = keys.kvPos();
        args[AttnKernels.ARG_FAP_BUF_SINKS] = sinks;
        args[AttnKernels.ARG_FAP_BUF_REL_BIAS] = relBias;
        args[AttnKernels.ARG_FAP_BUF_OUT] = out;
        args[AttnKernels.ARG_FAP_BUF_PARTIAL] = partial;
        args[AttnKernels.ARG_FAP_BUF_PAGE_TABLE] = pg.pageTable();
        if (!mma) {
            args[AttnKernels.ARG_FAP_BUF_KV_SCALES] = kvScales;
            args[AttnKernels.ARG_FAP_BUF_TREE_MASK] = treeMask;
        } else {
            args[AttnKernels.ARG_FAP_MMA_BUF_TREE_MASK] = treeMask;
        }
        return mma ? args : AttnKernels.pagedWithKvTypes(args, keys.typeK(), keys.typeV());
    }

    /** {@code LLM_FlashAttn_Combine} of {@code n_head} rows, one split, at output offset 0. */
    Object[] combine() {
        return new Object[]{a.nHead(), a.headDimV(), 1, a.nHead(), hasSinks ? 1 : 0, 0, partial, sinks, out};
    }

    /**
     * {@code LLM_FlashAttn_Lse} over the same partials as {@link #combine()}: the identical argument list with the
     * log-sum-exp output in place of the normalized output.
     */
    Object[] lse(GpuBuffer lse) {
        return new Object[]{a.nHead(), a.headDimV(), 1, a.nHead(), hasSinks ? 1 : 0, 0, partial, sinks, lse};
    }

    /** The scalars {@code ARG_FD_N_HEAD .. ARG_FD_OUT_OFF} shared by the dense and paged decode kernels. */
    private void decodeScalars(Object[] args, FlashAttnTiling plan, int kvEnd) {
        args[AttnKernels.ARG_FD_N_HEAD] = a.nHead();
        args[AttnKernels.ARG_FD_N_HEAD_KV] = a.nHeadKv();
        args[AttnKernels.ARG_FD_D_K] = a.headDimK();
        args[AttnKernels.ARG_FD_D_V] = a.headDimV();
        args[AttnKernels.ARG_FD_G_TILE] = plan.gTile();
        args[AttnKernels.ARG_FD_N_SPLIT] = 1;
        args[AttnKernels.ARG_FD_SPLIT_LEN] = plan.maxSplitLen();
        args[AttnKernels.ARG_FD_KV_START] = 0;
        args[AttnKernels.ARG_FD_KV_END] = kvEnd;
        args[AttnKernels.ARG_FD_Q_POS] = 0;
        args[AttnKernels.ARG_FD_K_ROW_STRIDE] = keys.kRowStride();
        args[AttnKernels.ARG_FD_K_HEAD_STRIDE] = a.headDimK();
        args[AttnKernels.ARG_FD_V_ROW_STRIDE] = keys.vRowStride();
        args[AttnKernels.ARG_FD_V_HEAD_STRIDE] = a.headDimV();
        args[AttnKernels.ARG_FD_SCALE] = a.scoreScale();
        args[AttnKernels.ARG_FD_SOFTCAP] = a.softcap();
        args[AttnKernels.ARG_FD_MAX_BIAS] = maxBias();
        args[AttnKernels.ARG_FD_FLAGS] = flags | AttnKernels.FLAG_DIRECT;
        args[AttnKernels.ARG_FD_SWA_TYPE] = a.effectiveSwaType().code();
        args[AttnKernels.ARG_FD_N_SWA] = nSwa();
        args[AttnKernels.ARG_FD_BIAS_MODE] = a.biasMode().code();
        args[AttnKernels.ARG_FD_N_BUCKETS] = biasBuckets(a);
        args[AttnKernels.ARG_FD_MAX_DISTANCE] = a.t5MaxDistance();
        args[AttnKernels.ARG_FD_Q_OFF] = 0;
        args[AttnKernels.ARG_FD_OUT_OFF] = 0;
    }

    /** The scalars {@code ARG_FA_N_TOKENS .. ARG_FA_OUT_OFF} shared by the dense and paged prefill kernels. */
    private void prefillScalars(Object[] args, FlashAttnTiling plan, int nTokens, int kvEnd) {
        args[AttnKernels.ARG_FA_N_TOKENS] = nTokens;
        args[AttnKernels.ARG_FA_N_HEAD] = a.nHead();
        args[AttnKernels.ARG_FA_N_HEAD_KV] = a.nHeadKv();
        args[AttnKernels.ARG_FA_D_K] = a.headDimK();
        args[AttnKernels.ARG_FA_D_V] = a.headDimV();
        args[AttnKernels.ARG_FA_BQ] = plan.bq();
        args[AttnKernels.ARG_FA_BK] = plan.bk();
        args[AttnKernels.ARG_FA_G_TILE] = plan.gTile();
        args[AttnKernels.ARG_FA_N_SPLIT] = 1;
        args[AttnKernels.ARG_FA_SPLIT_LEN] = plan.bk();
        args[AttnKernels.ARG_FA_KV_START] = 0;
        args[AttnKernels.ARG_FA_KV_END] = kvEnd;
        args[AttnKernels.ARG_FA_Q_ROW_STRIDE] = a.qDim();
        args[AttnKernels.ARG_FA_OUT_ROW_STRIDE] = a.outDim();
        args[AttnKernels.ARG_FA_K_ROW_STRIDE] = keys.kRowStride();
        args[AttnKernels.ARG_FA_K_HEAD_STRIDE] = a.headDimK();
        args[AttnKernels.ARG_FA_V_ROW_STRIDE] = keys.vRowStride();
        args[AttnKernels.ARG_FA_V_HEAD_STRIDE] = a.headDimV();
        args[AttnKernels.ARG_FA_SCALE] = a.scoreScale();
        args[AttnKernels.ARG_FA_SOFTCAP] = a.softcap();
        args[AttnKernels.ARG_FA_MAX_BIAS] = maxBias();
        args[AttnKernels.ARG_FA_FLAGS] = flags | AttnKernels.FLAG_DIRECT;
        args[AttnKernels.ARG_FA_SWA_TYPE] = a.effectiveSwaType().code();
        args[AttnKernels.ARG_FA_N_SWA] = nSwa();
        args[AttnKernels.ARG_FA_BIAS_MODE] = a.biasMode().code();
        args[AttnKernels.ARG_FA_N_BUCKETS] = biasBuckets(a);
        args[AttnKernels.ARG_FA_MAX_DISTANCE] = a.t5MaxDistance();
        args[AttnKernels.ARG_FA_Q_OFF] = 0;
        args[AttnKernels.ARG_FA_OUT_OFF] = 0;
    }

    /** The ALiBi maximum bias, 0 unless the bias mode is an ALiBi mode (linear or square-root). */
    private float maxBias() {
        return a.biasMode() == AttentionSpec.BiasMode.ALIBI
                || a.biasMode() == AttentionSpec.BiasMode.ALIBI_SQRT ? a.alibiMaxBias() : 0f;
    }

    /** The {@code n_buckets} argument: the T5 bucket count, or the relative extent of an Inkling table. */
    private static int biasBuckets(AttentionSpec a) {
        return a.biasMode() == AttentionSpec.BiasMode.INKLING ? a.relBiasExtent() : a.t5Buckets();
    }

    /** The sliding-window size, 0 without a window. */
    private int nSwa() {
        return a.effectiveSwaType() == SwaType.NONE ? 0 : a.nSwa();
    }
}
