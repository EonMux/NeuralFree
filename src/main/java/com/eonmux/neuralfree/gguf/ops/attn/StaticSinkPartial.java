package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;

import java.util.Objects;

/**
 * The split-K partial of a layer's learned static sink tokens ({@code LLM_AttnStaticSinkPartial}, vLLM
 * {@code StaticSinkAttention}): one (o, m, l) row per active (token, head), written at the split index after the
 * cached-key splits so the unchanged {@code LLM_FlashAttn_Combine} merges it. The dispatch is prepared once at
 * bind; a step re-sets the row count, the query offsets, the positions and the split. Not thread-safe.
 */
final class StaticSinkPartial {

    private final PreparedDispatch dispatch;
    private int rows = -1;

    private StaticSinkPartial(PreparedDispatch dispatch) {
        this.dispatch = dispatch;
    }

    /** The sink launch whose arguments a step re-sets. */
    PreparedDispatch dispatch() {
        return dispatch;
    }

    /**
     * Prepares the launch of {@code a}'s sink tokens.
     *
     * @param sinkK the F32 {@code [n_sink][n_head_kv][d_k]} keys, already normalized and un-rotated
     * @param sinkV the F32 {@code [n_sink][n_head_kv][d_v]} values
     * @throws IllegalArgumentException when a sink buffer is missing or the attention has no sink tokens
     */
    static StaticSinkPartial prepare(GpuMixerResources res, AttentionSpec a, GpuBuffer q, GpuBuffer qPos,
                                     GpuBuffer sinkK, GpuBuffer sinkV, GpuBuffer partial) {
        Objects.requireNonNull(res, "res");
        Objects.requireNonNull(a, "a");
        if (a.sinkTokens() <= 0) {
            throw new IllegalArgumentException("attention " + a + " has no static sink tokens");
        }
        Object[] args = new Object[AttnKernels.SS_ARG_COUNT];
        args[AttnKernels.ARG_SS_N_ROWS] = 0;
        args[AttnKernels.ARG_SS_N_HEAD] = a.nHead();
        args[AttnKernels.ARG_SS_N_HEAD_KV] = a.nHeadKv();
        args[AttnKernels.ARG_SS_D_K] = a.headDimK();
        args[AttnKernels.ARG_SS_D_V] = a.headDimV();
        args[AttnKernels.ARG_SS_N_SINK] = a.sinkTokens();
        args[AttnKernels.ARG_SS_Q_OFF] = 0;
        args[AttnKernels.ARG_SS_Q_ROW_STRIDE] = a.qDim();
        args[AttnKernels.ARG_SS_POS_OFF] = 0;
        args[AttnKernels.ARG_SS_POS_STRIDE] = 1;
        args[AttnKernels.ARG_SS_FIXED_POS] = 0;
        args[AttnKernels.ARG_SS_N_SPLIT] = 1;
        args[AttnKernels.ARG_SS_SPLIT_INDEX] = 0;
        args[AttnKernels.ARG_SS_SCALE] = a.scoreScale();
        args[AttnKernels.ARG_SS_SOFTCAP] = a.softcap();
        args[AttnKernels.ARG_SS_CAUSAL] = a.causal() ? 1 : 0;
        args[AttnKernels.ARG_SS_SWA_TYPE] = a.effectiveSwaType().code();
        args[AttnKernels.ARG_SS_N_SWA] = a.effectiveSwaType() == SwaType.NONE ? 0 : a.nSwa();
        args[AttnKernels.ARG_SS_BUF_Q] = Objects.requireNonNull(q, "q");
        args[AttnKernels.ARG_SS_BUF_Q_POS] = Objects.requireNonNull(qPos, "qPos");
        args[AttnKernels.ARG_SS_BUF_SINK_K] = Objects.requireNonNull(sinkK, "sinkK");
        args[AttnKernels.ARG_SS_BUF_SINK_V] = Objects.requireNonNull(sinkV, "sinkV");
        args[AttnKernels.ARG_SS_BUF_PARTIAL] = Objects.requireNonNull(partial, "partial");
        PreparedDispatch dispatch = res.prepare(AttnKernels.STATIC_SINK_PARTIAL,
                FlashAttnTiling.combineGlobal(1), FlashAttnTiling.combineLocal(), args);
        return new StaticSinkPartial(dispatch);
    }

    /**
     * Enqueues the sink partial of {@code rows} (token, head) rows at split index {@code cachedSplits}, and makes
     * the merge that follows read {@code cachedSplits + 1} splits.
     *
     * @param qOff         element offset of the first query row
     * @param qRowStride   element stride between query rows
     * @param posOff       first position entry
     * @param posStride    position entries per row; 0 reads {@code fixedPos} for every row
     * @param fixedPos     the position of every row when {@code posStride} is 0
     * @param posBuffer    the position buffer to bind for this launch, or null for the one bound at prepare
     * @param cachedSplits the split count of the cached-key launches; the sink partial becomes split
     *                     {@code cachedSplits}
     * @return the split count the merge must read ({@code cachedSplits + 1})
     */
    int run(int rows, int qOff, int qRowStride, int posOff, int posStride, int fixedPos, GpuBuffer posBuffer,
            int cachedSplits) {
        if (rows < 0 || qOff < 0 || qRowStride < 0 || posOff < 0 || posStride < 0 || cachedSplits < 1) {
            throw new IllegalArgumentException("static sink partial of " + rows + " rows at " + qOff + "/"
                    + qRowStride + ", positions " + posOff + "/" + posStride + "/" + fixedPos + ", " + cachedSplits
                    + " cached splits");
        }
        if (rows == 0) {
            return cachedSplits + 1;
        }
        if (posBuffer != null) {
            dispatch.setBuffer(AttnKernels.ARG_SS_BUF_Q_POS, posBuffer);
        }
        dispatch.setInt(AttnKernels.ARG_SS_N_ROWS, rows);
        dispatch.setInt(AttnKernels.ARG_SS_Q_OFF, qOff);
        dispatch.setInt(AttnKernels.ARG_SS_Q_ROW_STRIDE, qRowStride);
        dispatch.setInt(AttnKernels.ARG_SS_POS_OFF, posOff);
        dispatch.setInt(AttnKernels.ARG_SS_POS_STRIDE, posStride);
        dispatch.setInt(AttnKernels.ARG_SS_FIXED_POS, fixedPos);
        dispatch.setInt(AttnKernels.ARG_SS_N_SPLIT, cachedSplits + 1);
        dispatch.setInt(AttnKernels.ARG_SS_SPLIT_INDEX, cachedSplits);
        if (rows != this.rows) {
            /* One work-item per (token, head) row, rounded up to whole workgroups. */
            long grid = (long) ((rows + FlashAttnTiling.WG - 1) / FlashAttnTiling.WG) * FlashAttnTiling.WG;
            dispatch.setGlobal(new long[]{grid});
            this.rows = rows;
        }
        dispatch.dispatch();
        return cachedSplits + 1;
    }
}
