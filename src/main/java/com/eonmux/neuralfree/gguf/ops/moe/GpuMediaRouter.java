package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.Objects;

/**
 * The expert selection of a step's embedding rows on a deepseek4 hash or {@code exp_probs_b_vl} layer
 * ({@link MoeSpec#mediaRouting()}, deepseek4.cpp:1290-1301), applied after the layer's own router: a second
 * {@code LLM_MoE_Router} over the step's logits with the media parameters ({@link MoeRouting.Params#forMedia}) and
 * {@code exp_probs_b_vl} or no selection bias writes its own ids and weights, and each run of embedding rows (host
 * step-token id {@code -1}, {@link GpuScratch#readStepTokens}) is copied bit for bit over the router's ids and
 * weights. Token rows keep the router's selection, so a mixed paged ubatch routes every row as llama.cpp routes its
 * ubatch kind. A step without embedding rows dispatches nothing. Not thread-safe: one forward pass at a time.
 */
final class GpuMediaRouter implements AutoCloseable {

    static final String T_IDS = "moe.media_ids";
    static final String T_WEIGHTS = "moe.media_weights";

    private final GpuForwardContext ctx;
    private final int k;
    private final int routerLocal;
    private final ScratchSlot ids;
    private final ScratchSlot weights;
    private final ScratchSlot mediaIds;
    private final ScratchSlot mediaWeights;
    private final int[] stepTokens;
    private final PreparedDispatch select;
    private int lastN;

    /**
     * @param params      the media routing ({@link MoeRouting.Params#forMedia})
     * @param logits      the step's router logits, the router's input
     * @param bias        {@code exp_probs_b_vl} when {@code params} has a bias, else a placeholder
     * @param ids         the router's ids, overwritten on the embedding rows
     * @param weights     the router's weights, overwritten on the embedding rows
     * @param routerLocal the router's work-group size
     */
    GpuMediaRouter(GpuForwardContext ctx, MoeRouting.Params params, ScratchSlot logits, GpuBuffer bias,
                   ScratchSlot ids, ScratchSlot weights, int routerLocal) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(params, "params");
        this.k = params.k();
        this.routerLocal = routerLocal;
        this.ids = Objects.requireNonNull(ids, "ids");
        this.weights = Objects.requireNonNull(weights, "weights");
        MoeKernels.checkRouter(params, routerLocal);
        GpuScratch scratch = ctx.scratch();
        int rows = Math.multiplyExact(scratch.ubatch(), k);
        this.mediaIds = scratch.temp(T_IDS, rows);
        this.mediaWeights = scratch.temp(T_WEIGHTS, rows);
        this.stepTokens = new int[scratch.ubatch()];
        this.select = ctx.ops().prepare(MoeKernels.ROUTER, MoeKernels.routerGlobal(1, routerLocal),
                new long[]{routerLocal}, MoeKernels.routerArgs(params, 1, logits.buffer(), bias, mediaIds.buffer(),
                        mediaWeights.buffer()));
        this.lastN = 1;
    }

    /** Routes the embedding rows among the step's {@code n} rows and copies their selection over the router's. */
    void apply(int n) {
        ctx.scratch().readStepTokens(stepTokens, 0, n);
        int t = 0;
        while (t < n && stepTokens[t] >= 0) {
            t++;
        }
        if (t == n) {
            return;
        }
        if (lastN != n) {
            select.setInt(MoeKernels.ARG_N_TOKENS, n);
            select.setGlobal(MoeKernels.routerGlobal(n, routerLocal));
            lastN = n;
        }
        select.dispatch();
        while (t < n) {
            int end = t;
            while (end < n && stepTokens[end] < 0) {
                end++;
            }
            if (end > t) {
                copyRows(mediaIds, ids, t, end - t);
                copyRows(mediaWeights, weights, t, end - t);
            }
            t = end + 1;
        }
    }

    /** Copies the {@code k}-wide selection of rows {@code [row, row + rows)} byte for byte. */
    private void copyRows(ScratchSlot src, ScratchSlot dst, int row, int rows) {
        long at = (long) row * k * Float.BYTES;
        ctx.queue().enqueueCopyBuffer(src.buffer(), (long) src.offset() * Float.BYTES + at, dst.buffer(),
                (long) dst.offset() * Float.BYTES + at, (long) rows * k * Float.BYTES);
    }

    @Override
    public void close() {
        RuntimeException failure = Closeables.close(select, null);
        if (failure != null) {
            throw failure;
        }
    }
}
