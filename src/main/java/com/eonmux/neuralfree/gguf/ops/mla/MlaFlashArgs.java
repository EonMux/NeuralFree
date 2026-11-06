package com.eonmux.neuralfree.gguf.ops.mla;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.attn.FlashAttnTiling;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;

/**
 * The flash-attention argument arrays of one bound {@link GpuMlaMixer} layer: decode, tiled prefill and their paged
 * forms. Every array follows its kernel's signature and starts from one sequence, one row and one direct split; the
 * step re-sets the per-call token counts, key ranges, positions, splits and flags on the prepared dispatches. The
 * tree-mask slots bind the read-only placeholder, and the flags these steps prepare carry no tree bit, so the
 * fresh-row count the flags pack stays 0 and the kernels never read the mask word — exactly the dense path's
 * convention outside a tree verify step; a tree verify step re-binds the mask and packs its fresh-row count into the
 * flags as standard attention does.
 */
final class MlaFlashArgs {

    private final MlaBlocks.Geometry g;
    private final KvLayout layout;
    private final GpuKvLayer kv;
    private final GpuStepInputs steps;
    private final GpuBuffer kvScalesPlaceholder;
    private final FlashAttnTiling prefillTiling;
    private final FlashAttnTiling pagedTiling;
    private final int flashFlags;
    private final int pageShift;
    private final int maxPages;

    MlaFlashArgs(MlaBlocks.Geometry g, KvLayout layout, GpuKvLayer kv, GpuStepInputs steps,
                 GpuBuffer kvScalesPlaceholder, FlashAttnTiling prefillTiling, FlashAttnTiling pagedTiling,
                 int flashFlags, int pageShift, int maxPages) {
        this.g = g;
        this.layout = layout;
        this.kv = kv;
        this.steps = steps;
        this.kvScalesPlaceholder = kvScalesPlaceholder;
        this.prefillTiling = prefillTiling;
        this.pagedTiling = pagedTiling;
        this.flashFlags = flashFlags;
        this.pageShift = pageShift;
        this.maxPages = maxPages;
    }

    /**
     * {@code LLM_FlashDecodePaged_*} arguments in kernel signature order; the per-launch scalars start at one
     * sequence, one direct split.
     */
    Object[] pagedDecodeArgs(ScratchSlot qs, ScratchSlot os, float softcap, GpuBuffer sinks, GpuBuffer ph,
                             GpuBuffer partials) {
        KvWindow w = layout.window();
        Object[] a = new Object[AttnKernels.FDP_ARG_COUNT];
        a[AttnKernels.ARG_FD_N_HEAD] = g.nHead;
        a[AttnKernels.ARG_FD_N_HEAD_KV] = g.nHeadKv;
        a[AttnKernels.ARG_FD_D_K] = g.dQ;
        a[AttnKernels.ARG_FD_D_V] = g.dV;
        a[AttnKernels.ARG_FD_G_TILE] = pagedTiling.gTile();
        a[AttnKernels.ARG_FD_N_SPLIT] = 1;
        a[AttnKernels.ARG_FD_SPLIT_LEN] = pagedTiling.maxSplitLen();
        a[AttnKernels.ARG_FD_KV_START] = 0;
        a[AttnKernels.ARG_FD_KV_END] = 0;
        a[AttnKernels.ARG_FD_Q_POS] = 0;
        a[AttnKernels.ARG_FD_K_ROW_STRIDE] = g.kRowStride;
        a[AttnKernels.ARG_FD_K_HEAD_STRIDE] = g.kHeadStride;
        a[AttnKernels.ARG_FD_V_ROW_STRIDE] = g.vRowStride;
        a[AttnKernels.ARG_FD_V_HEAD_STRIDE] = g.vHeadStride;
        a[AttnKernels.ARG_FD_SCALE] = g.kqScale;
        a[AttnKernels.ARG_FD_SOFTCAP] = softcap;
        a[AttnKernels.ARG_FD_MAX_BIAS] = 0.0f;
        a[AttnKernels.ARG_FD_FLAGS] = flashFlags | AttnKernels.FLAG_DIRECT;
        a[AttnKernels.ARG_FD_SWA_TYPE] = w.type().code();
        a[AttnKernels.ARG_FD_N_SWA] = w.nSwa();
        a[AttnKernels.ARG_FD_BIAS_MODE] = AttentionSpec.BiasMode.NONE.code();
        a[AttnKernels.ARG_FD_N_BUCKETS] = 0;
        a[AttnKernels.ARG_FD_MAX_DISTANCE] = 0;
        a[AttnKernels.ARG_FD_Q_OFF] = qs.offset();
        a[AttnKernels.ARG_FD_OUT_OFF] = os.offset();
        a[AttnKernels.ARG_FDP_PAGE_SHIFT] = pageShift;
        a[AttnKernels.ARG_FDP_MAX_PAGES] = maxPages;
        a[AttnKernels.ARG_FDP_N_SEQ] = 1;
        a[AttnKernels.ARG_FDP_Q_SEQ_STRIDE] = g.nHead * g.dQ;
        a[AttnKernels.ARG_FDP_OUT_SEQ_STRIDE] = g.nHead * g.dV;
        a[AttnKernels.ARG_FDP_BUF_Q] = qs.buffer();
        a[AttnKernels.ARG_FDP_BUF_K] = kv.k();
        a[AttnKernels.ARG_FDP_BUF_V] = kv.v();
        a[AttnKernels.ARG_FDP_BUF_KV_POS] = steps.kvPos(kv.group());
        a[AttnKernels.ARG_FDP_BUF_SINKS] = sinks;
        a[AttnKernels.ARG_FDP_BUF_REL_BIAS] = ph;
        a[AttnKernels.ARG_FDP_BUF_OUT] = os.buffer();
        a[AttnKernels.ARG_FDP_BUF_PARTIAL] = partials;
        a[AttnKernels.ARG_FDP_BUF_PAGE_TABLE] = steps.pageTable(kv.group());
        a[AttnKernels.ARG_FDP_BUF_SEQ_KV_LEN] = steps.seqKvLen(kv.group());
        a[AttnKernels.ARG_FDP_BUF_Q_POS_SEQ] = steps.qPosSeq();
        a[AttnKernels.ARG_FDP_BUF_KV_SCALES] = kvScales();
        a[AttnKernels.ARG_FDP_BUF_TREE_MASK] = ph;
        return a;
    }

    /** {@code LLM_FlashAttnPaged_*} arguments in kernel signature order; per-segment scalars start at one row. */
    Object[] pagedPrefillArgs(ScratchSlot qs, ScratchSlot os, float softcap, GpuBuffer sinks, GpuBuffer ph,
                              GpuBuffer partials) {
        KvWindow w = layout.window();
        Object[] a = new Object[AttnKernels.FAP_ARG_COUNT];
        a[AttnKernels.ARG_FA_N_TOKENS] = 1;
        a[AttnKernels.ARG_FA_N_HEAD] = g.nHead;
        a[AttnKernels.ARG_FA_N_HEAD_KV] = g.nHeadKv;
        a[AttnKernels.ARG_FA_D_K] = g.dQ;
        a[AttnKernels.ARG_FA_D_V] = g.dV;
        a[AttnKernels.ARG_FA_BQ] = prefillTiling.bq();
        a[AttnKernels.ARG_FA_BK] = prefillTiling.bk();
        a[AttnKernels.ARG_FA_G_TILE] = prefillTiling.gTile();
        a[AttnKernels.ARG_FA_N_SPLIT] = 1;
        a[AttnKernels.ARG_FA_SPLIT_LEN] = prefillTiling.bk();
        a[AttnKernels.ARG_FA_KV_START] = 0;
        a[AttnKernels.ARG_FA_KV_END] = 0;
        a[AttnKernels.ARG_FA_Q_ROW_STRIDE] = g.nHead * g.dQ;
        a[AttnKernels.ARG_FA_OUT_ROW_STRIDE] = g.nHead * g.dV;
        a[AttnKernels.ARG_FA_K_ROW_STRIDE] = g.kRowStride;
        a[AttnKernels.ARG_FA_K_HEAD_STRIDE] = g.kHeadStride;
        a[AttnKernels.ARG_FA_V_ROW_STRIDE] = g.vRowStride;
        a[AttnKernels.ARG_FA_V_HEAD_STRIDE] = g.vHeadStride;
        a[AttnKernels.ARG_FA_SCALE] = g.kqScale;
        a[AttnKernels.ARG_FA_SOFTCAP] = softcap;
        a[AttnKernels.ARG_FA_MAX_BIAS] = 0.0f;
        a[AttnKernels.ARG_FA_FLAGS] = flashFlags | AttnKernels.FLAG_DIRECT;
        a[AttnKernels.ARG_FA_SWA_TYPE] = w.type().code();
        a[AttnKernels.ARG_FA_N_SWA] = w.nSwa();
        a[AttnKernels.ARG_FA_BIAS_MODE] = AttentionSpec.BiasMode.NONE.code();
        a[AttnKernels.ARG_FA_N_BUCKETS] = 0;
        a[AttnKernels.ARG_FA_MAX_DISTANCE] = 0;
        a[AttnKernels.ARG_FA_Q_OFF] = qs.offset();
        a[AttnKernels.ARG_FA_OUT_OFF] = os.offset();
        a[AttnKernels.ARG_FAP_PAGE_SHIFT] = pageShift;
        a[AttnKernels.ARG_FAP_PT_OFF] = 0;
        a[AttnKernels.ARG_FAP_BUF_Q] = qs.buffer();
        a[AttnKernels.ARG_FAP_BUF_K] = kv.k();
        a[AttnKernels.ARG_FAP_BUF_V] = kv.v();
        a[AttnKernels.ARG_FAP_BUF_Q_POS] = steps.segmentPositions();
        a[AttnKernels.ARG_FAP_BUF_KV_POS] = steps.kvPos(kv.group());
        a[AttnKernels.ARG_FAP_BUF_SINKS] = sinks;
        a[AttnKernels.ARG_FAP_BUF_REL_BIAS] = ph;
        a[AttnKernels.ARG_FAP_BUF_OUT] = os.buffer();
        a[AttnKernels.ARG_FAP_BUF_PARTIAL] = partials;
        a[AttnKernels.ARG_FAP_BUF_PAGE_TABLE] = steps.pageTable(kv.group());
        a[AttnKernels.ARG_FAP_BUF_KV_SCALES] = kvScales();
        a[AttnKernels.ARG_FAP_BUF_TREE_MASK] = ph;
        return a;
    }

    /**
     * Partial floats the largest split of a tiled prefill needs: the most {@link FlashAttnTiling#prefillSplit} chooses
     * for any call of up to {@code ubatch} tokens over the whole cache, unconstrained by capacity. A smaller capacity
     * only lowers the split count, because {@code prefillSplit} fits its splits to the capacity it is given.
     */
    static long prefillPartialFloats(FlashAttnTiling p, int ubatch, int nKvMax) {
        long cap = 0;
        for (int n = 1; n <= ubatch; n++) {
            FlashAttnTiling.Split s = p.prefillSplit(n, nKvMax, Long.MAX_VALUE);
            if (!s.direct()) {
                cap = Math.max(cap, FlashAttnTiling.partialFloats((long) n * p.nHead(), s.count(), p.dV()));
            }
        }
        return cap;
    }

    /**
     * {@code LLM_FlashDecode_*} arguments in kernel signature order; the per-step values start as one direct split at
     * row 0.
     */
    Object[] decodeArgs(FlashAttnTiling t, ScratchSlot qs, ScratchSlot os, float softcap, GpuBuffer sinks,
                        GpuBuffer ph, GpuBuffer partials) {
        KvWindow w = layout.window();
        Object[] a = new Object[AttnKernels.FD_ARG_COUNT];
        a[AttnKernels.ARG_FD_N_HEAD] = g.nHead;
        a[AttnKernels.ARG_FD_N_HEAD_KV] = g.nHeadKv;
        a[AttnKernels.ARG_FD_D_K] = g.dQ;
        a[AttnKernels.ARG_FD_D_V] = g.dV;
        a[AttnKernels.ARG_FD_G_TILE] = t.gTile();
        a[AttnKernels.ARG_FD_N_SPLIT] = 1;
        a[AttnKernels.ARG_FD_SPLIT_LEN] = t.maxSplitLen();
        a[AttnKernels.ARG_FD_KV_START] = 0;
        a[AttnKernels.ARG_FD_KV_END] = 1;
        a[AttnKernels.ARG_FD_Q_POS] = 0;
        a[AttnKernels.ARG_FD_K_ROW_STRIDE] = g.kRowStride;
        a[AttnKernels.ARG_FD_K_HEAD_STRIDE] = g.kHeadStride;
        a[AttnKernels.ARG_FD_V_ROW_STRIDE] = g.vRowStride;
        a[AttnKernels.ARG_FD_V_HEAD_STRIDE] = g.vHeadStride;
        a[AttnKernels.ARG_FD_SCALE] = g.kqScale;
        a[AttnKernels.ARG_FD_SOFTCAP] = softcap;
        a[AttnKernels.ARG_FD_MAX_BIAS] = 0.0f;
        a[AttnKernels.ARG_FD_FLAGS] = flashFlags | AttnKernels.FLAG_DIRECT;
        a[AttnKernels.ARG_FD_SWA_TYPE] = w.type().code();
        a[AttnKernels.ARG_FD_N_SWA] = w.nSwa();
        a[AttnKernels.ARG_FD_BIAS_MODE] = AttentionSpec.BiasMode.NONE.code();
        a[AttnKernels.ARG_FD_N_BUCKETS] = 0;
        a[AttnKernels.ARG_FD_MAX_DISTANCE] = 0;
        a[AttnKernels.ARG_FD_Q_OFF] = qs.offset();
        a[AttnKernels.ARG_FD_OUT_OFF] = os.offset();
        a[AttnKernels.ARG_FD_BUF_Q] = qs.buffer();
        a[AttnKernels.ARG_FD_BUF_K] = kv.k();
        a[AttnKernels.ARG_FD_BUF_V] = kv.v();
        a[AttnKernels.ARG_FD_BUF_KV_POS] = steps.kvPos(kv.group());
        a[AttnKernels.ARG_FD_BUF_SINKS] = sinks;
        a[AttnKernels.ARG_FD_BUF_REL_BIAS] = ph;
        a[AttnKernels.ARG_FD_BUF_OUT] = os.buffer();
        a[AttnKernels.ARG_FD_BUF_PARTIAL] = partials;
        a[AttnKernels.ARG_FD_BUF_KV_SCALES] = kvScales();
        a[AttnKernels.ARG_FD_BUF_TREE_MASK] = ph;
        return a;
    }

    /**
     * {@code LLM_FlashAttn_*} arguments in kernel signature order; the per-call values start as one token, one
     * split.
     */
    Object[] prefillArgs(ScratchSlot qs, ScratchSlot os, float softcap, GpuBuffer sinks, GpuBuffer ph,
                         GpuBuffer partials) {
        KvWindow w = layout.window();
        Object[] a = new Object[AttnKernels.FA_ARG_COUNT];
        a[AttnKernels.ARG_FA_N_TOKENS] = 1;
        a[AttnKernels.ARG_FA_N_HEAD] = g.nHead;
        a[AttnKernels.ARG_FA_N_HEAD_KV] = g.nHeadKv;
        a[AttnKernels.ARG_FA_D_K] = g.dQ;
        a[AttnKernels.ARG_FA_D_V] = g.dV;
        a[AttnKernels.ARG_FA_BQ] = prefillTiling.bq();
        a[AttnKernels.ARG_FA_BK] = prefillTiling.bk();
        a[AttnKernels.ARG_FA_G_TILE] = prefillTiling.gTile();
        a[AttnKernels.ARG_FA_N_SPLIT] = 1;
        a[AttnKernels.ARG_FA_SPLIT_LEN] = prefillTiling.bk();
        a[AttnKernels.ARG_FA_KV_START] = 0;
        a[AttnKernels.ARG_FA_KV_END] = 1;
        a[AttnKernels.ARG_FA_Q_ROW_STRIDE] = g.nHead * g.dQ;
        a[AttnKernels.ARG_FA_OUT_ROW_STRIDE] = g.nHead * g.dV;
        a[AttnKernels.ARG_FA_K_ROW_STRIDE] = g.kRowStride;
        a[AttnKernels.ARG_FA_K_HEAD_STRIDE] = g.kHeadStride;
        a[AttnKernels.ARG_FA_V_ROW_STRIDE] = g.vRowStride;
        a[AttnKernels.ARG_FA_V_HEAD_STRIDE] = g.vHeadStride;
        a[AttnKernels.ARG_FA_SCALE] = g.kqScale;
        a[AttnKernels.ARG_FA_SOFTCAP] = softcap;
        a[AttnKernels.ARG_FA_MAX_BIAS] = 0.0f;
        a[AttnKernels.ARG_FA_FLAGS] = flashFlags | AttnKernels.FLAG_DIRECT;
        a[AttnKernels.ARG_FA_SWA_TYPE] = w.type().code();
        a[AttnKernels.ARG_FA_N_SWA] = w.nSwa();
        a[AttnKernels.ARG_FA_BIAS_MODE] = AttentionSpec.BiasMode.NONE.code();
        a[AttnKernels.ARG_FA_N_BUCKETS] = 0;
        a[AttnKernels.ARG_FA_MAX_DISTANCE] = 0;
        a[AttnKernels.ARG_FA_Q_OFF] = qs.offset();
        a[AttnKernels.ARG_FA_OUT_OFF] = os.offset();
        a[AttnKernels.ARG_FA_BUF_Q] = qs.buffer();
        a[AttnKernels.ARG_FA_BUF_K] = kv.k();
        a[AttnKernels.ARG_FA_BUF_V] = kv.v();
        a[AttnKernels.ARG_FA_BUF_Q_POS] = steps.qPos();
        a[AttnKernels.ARG_FA_BUF_KV_POS] = steps.kvPos(kv.group());
        a[AttnKernels.ARG_FA_BUF_SINKS] = sinks;
        a[AttnKernels.ARG_FA_BUF_REL_BIAS] = ph;
        a[AttnKernels.ARG_FA_BUF_OUT] = os.buffer();
        a[AttnKernels.ARG_FA_BUF_PARTIAL] = partials;
        a[AttnKernels.ARG_FA_BUF_KV_SCALES] = kvScales();
        a[AttnKernels.ARG_FA_BUF_TREE_MASK] = ph;
        return a;
    }

    /**
     * The {@code kv_scales} binding: the side buffer of a vendor latent cache (its checkpoint scales, Lloyd-Max table
     * and packed planes), else the unit-scale placeholder.
     */
    private GpuBuffer kvScales() {
        return kv.vendorSide() != null ? kv.vendorSide() : kvScalesPlaceholder;
    }
}
