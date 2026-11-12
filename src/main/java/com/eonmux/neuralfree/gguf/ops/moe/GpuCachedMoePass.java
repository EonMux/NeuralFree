package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertPrefetch;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * The routed expert pass of one hosted MoE layer, Avalon's hosted MoE: with an expert cache the cached experts are
 * computed on the GPU from the banks and the per-layer indirection map ({@code {T}_MatVecId*Ind} and
 * {@code {T}_BatchMatMulIdInd}) and the misses on the host ({@link HostedMoeExperts}, the CPU computing while the
 * GPU dispatches); with no cache, or a layer whose cache group is disabled, every pair is a miss the host computes.
 * The miss rows are uploaded before the post-scale and the weighted sum, which run over all pairs on the GPU exactly
 * as without the cache. The per-expert scales, the expert biases and the post-expert norm come from the resident
 * weights; the biases are added over every pair after their product, and the host adds them to the miss rows it
 * uploads, whose device writes it overwrites.
 *
 * <p>The id readback waits for the compute queue after the router, as llama.cpp's cache copy callback does; the
 * read ids of every step, hosted or prefetched, are also counted into the context's {@link MoeExpertHeat} when a
 * profile records (Avalon's hosted plans {@code moe_cache_heat_note}), so the router reads no second copy. Fills
 * of newly admitted experts are enqueued at {@link MoeExpertCache#endStep} and become visible at the layer's next
 * step; until then their experts stay misses the host computes. When no pair hits, the GPU expert kernels are not
 * dispatched and the host rows are the whole expert output.</p>
 *
 * <p>One decode path ({@code {T}_MatVecId*Ind}, one token, without expert biases) and one batched path, prepared at
 * bind with fixed buffers whose token counts, global sizes and input view are re-set per step. The batched path sorts
 * the pairs by expert ({@link GpuGroupedRows}) and runs the grouped GEMM over the expert-aligned tiles through the
 * map ({@code LLM_MoE_AlignBlocks} + {@code {T}_GroupedGemmInd}, {@code {T}_GroupedGemmMmaInd} on CUDA or Vulkan
 * tensor cores) under the eligibility of {@link GpuMoeFfn}'s grouped GEMM: the align kernel and the tile grid fit
 * the device ({@link GpuGroupedRows#alignUsable}) and every bank type passes
 * {@link GpuGroupedRows#groupedUsable}; otherwise it multiplies per pair ({@code {T}_BatchMatMulIdInd}). A tile
 * whose expert the map does not hold is skipped whole, so the host's miss rows stay the only writes of those
 * pairs. Not thread-safe.</p>
 *
 * <p>With a {@link MoeExpertPrefetch}, a step of a served layer the pool takes (prefill-sized: at least two routed
 * pairs per expert) computes no pair on the host: the pool uploads the step's missing experts into slots it borrows
 * from the cache's banks, and the batched products (and their biases) run once over the banks through the pool's
 * prefill map, which points each cached expert at its own slot (read in place) and each other expert at its borrowed
 * slot.</p>
 *
 * <p>With a hosted tail-skip rank ({@code neuralfree.moe.cache.tail.skip}, {@link MoeTailSkip}, Avalon
 * {@code GGML_CUDA_MOE_TAIL_SKIP}) a missed expert that every token of the step routes only at that 0-based top-k
 * position or lower is neither computed on the host nor moved to the device: its expert output rows are zeroed before
 * the weighted sum, so the other experts keep the weights the router gave them. The rule applies to this model's
 * trunk layers, never to a NextN / MTP layer's, and never to a step the prefetch pool takes.</p>
 */
final class GpuCachedMoePass implements AutoCloseable {

    static final String T_SCALE_PAIRS_IND = "moe.scale_pairs.ind.";
    /** Scratch prefix of the expert sort of the batched path ({@link GpuGroupedRows}). */
    static final String T_ROWS_IND = "moe.ind";
    static final String T_GATE_IND = "moe.gate.ind";
    static final String T_UP_IND = "moe.up.ind";
    static final String T_ACT_IND = "moe.act.ind";
    static final String T_EXPERT_OUT_IND = "moe.expert_out.ind";
    static final String T_WEIGHTED_IN_IND = "moe.weighted_in.ind";
    static final String T_INPUT_IND = "moe.input.ind";

    private static final int WSUM_FLAGS_ARG = 3;
    private static final int WSUM_DST_ARG = 6;
    private static final int SCALE_INPUT_X_ARG = 3;

    private final GpuForwardContext ctx;
    private final LayerSpec layer;
    private final MoeExpertWeights plan;
    private final GpuMoeFfn.Router router;
    /** The layer's expert cache, or null; {@link #served} is false when it is null or the group is disabled. */
    private final MoeExpertCache cache;
    private final boolean served;
    /** The prefill expert prefetch pool of a served layer, or null when none is configured. */
    private final MoeExpertPrefetch prefetch;
    private final GpuWeight gateBias;
    private final GpuWeight upBias;
    private final GpuWeight gateUpBias;
    private final GpuWeight downBias;
    private final int weightLayer;
    /** The routed-row recorder of the profile save, or null. */
    private final MoeExpertHeat heat;
    private final GpuOps ops;
    private final boolean weightBefore;
    private final int k;
    private final int ubatch;
    private final int nFf;
    private final int inDim;
    private final int outDim;
    private final int nExpert;
    private final GpuWeight upScale;
    private final GpuWeight gateScale;
    private final GpuWeight downScale;
    private final GpuWeight postNorm;
    private final float postEps;

    private final ScratchSlot act;
    private final ScratchSlot gate;
    private final ScratchSlot up;
    private final ScratchSlot expertOut;
    private final ScratchSlot weightedIn;
    private final ScratchSlot input0;
    /** The expert sort of the batched path, aligned when {@link #grouped}; null when the cache serves no expert. */
    private final GpuGroupedRows sort;
    /** Whether the batched products run the grouped GEMM over the aligned tiles ({@code {T}_GroupedGemmInd}). */
    private final boolean grouped;
    /** Bind-time tile bound of the grouped GEMM; 0 when not {@link #grouped}. */
    private final int nTiles;
    private final PreparedDispatch postScale;
    private final PreparedDispatch wsum;
    private final GpuBuffer scalePairs;
    private final HostedMoeExperts hosted;
    private final byte[] idBytes;
    private final float[] xsHost;
    private float[] rowHost;
    private final int idLocal;
    /**
     * The hosted tail-skip rank of this layer ({@link MoeTailSkip}); {@link MoeTailSkip#OFF} on an MTP layer, which
     * the rule never drops (Avalon applies it to the main model's layers only).
     */
    private final int tailSkipRank;
    private final Path decode;
    private final Path batch;
    private boolean closed;

    /**
     * Binds the pass of one hosted layer. With {@code cache} the cached experts run on the device from the banks
     * and the misses on the host; with null (no cache configured, or the layer's group is disabled) every pair is
     * a miss and the host computes them all.
     *
     * @param weights  resident weights of the model: the scales, the expert biases and the post-expert norm (the
     *                 expert stacks are the cache's or the host's)
     * @param ubatch   largest token count of one step
     * @param postNorm post-expert norm vector, or null
     * @param prefetch prefill expert prefetch pool over {@code cache}, or null to leave every step's misses to the
     *                 host
     * @param tailSkipRank the hosted tail-skip rank ({@link MoeTailSkip}); 0 keeps every miss
     * @throws IllegalArgumentException when a bank's type has no id-matvec kernels, a bank disagrees with the
     *                                  expert plan, a scale vector has the wrong shape, or the rank is invalid
     * @throws IllegalStateException    when the model was opened without the host tensors the misses compute from
     */
    GpuCachedMoePass(GpuForwardContext ctx, LayerSpec layer, MoeExpertWeights plan, GpuMoeFfn.Router router,
                     MoeExpertCache cache, GpuWeights weights, int ubatch, boolean weightBefore, GpuWeight postNorm,
                     float postEps, MoeExpertPrefetch prefetch, int tailSkipRank) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.layer = Objects.requireNonNull(layer, "layer");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.router = Objects.requireNonNull(router, "router");
        this.cache = cache;
        this.weightLayer = layer.weightLayer();
        this.heat = ctx.heat();
        this.ops = ctx.ops();
        this.weightBefore = weightBefore;
        this.k = router.params.k();
        this.ubatch = ubatch;
        this.nFf = plan.nFf;
        this.inDim = plan.inDim;
        this.outDim = plan.outDim;
        this.nExpert = plan.nExpert;
        this.postNorm = postNorm;
        this.postEps = postEps;
        if (ctx.hostWeights() == null) {
            throw new IllegalStateException("layer " + layer.il() + " keeps its routed experts in host RAM but the"
                    + " model was opened without the host tensors the misses compute from");
        }
        this.served = cache != null && cache.serves(weightLayer);
        this.prefetch = served ? prefetch : null;
        this.tailSkipRank = mainModelLayer(ctx, layer) ? checkTailSkipRank(tailSkipRank) : MoeTailSkip.OFF;
        this.gateBias = plan.gated && !plan.fused
                ? resolveVector(weights, TensorNames.bias(TensorNames.FFN_GATE_EXPS), nFf) : null;
        this.upBias = plan.fused ? null
                : resolveVector(weights, TensorNames.bias(TensorNames.FFN_UP_EXPS), nFf);
        this.gateUpBias = plan.fused
                ? resolveVector(weights, TensorNames.bias(TensorNames.FFN_GATE_UP_EXPS), 2 * nFf) : null;
        this.downBias = resolveVector(weights, TensorNames.bias(TensorNames.FFN_DOWN_EXPS), outDim);
        this.upScale = resolveVector(weights, TensorNames.scale(TensorNames.FFN_UP_EXPS), 1);
        this.gateScale = plan.gated && !plan.fused
                ? resolveVector(weights, TensorNames.scale(TensorNames.FFN_GATE_EXPS), 1) : null;
        this.downScale = resolveVector(weights, TensorNames.scale(TensorNames.FFN_DOWN_EXPS), 1);
        if (postNorm != null) {
            checkVector(postNorm, outDim);
        }
        if (served) {
            checkBanks();
        }
        GpuScratch scratch = ctx.scratch();
        int pairs = Math.multiplyExact(ubatch, k);
        this.idLocal = served ? idLocal() : 0;
        this.expertOut = scratch.temp(T_EXPERT_OUT_IND, Math.multiplyExact((long) pairs, outDim));
        this.weightedIn = weightBefore ? scratch.temp(T_WEIGHTED_IN_IND, Math.multiplyExact((long) pairs, inDim))
                : null;
        this.input0 = weightBefore ? scratch.temp(T_INPUT_IND, Math.multiplyExact((long) ubatch, inDim)) : null;
        this.hosted = new HostedMoeExperts(plan, ctx.hostWeights(), k, ubatch, layer);
        this.xsHost = new float[Math.toIntExact(Math.multiplyExact((long) pairs, inDim))];
        this.rowHost = new float[outDim];
        this.idBytes = new byte[Math.multiplyExact(pairs, Integer.BYTES)];
        if (!served) {
            // every pair is a miss: the host rows upload into expertOut and the weighted sum covers the step
            this.scalePairs = null;
            this.act = null;
            this.gate = null;
            this.up = null;
            this.sort = null;
            this.grouped = false;
            this.nTiles = 0;
            this.postScale = postNorm == null ? null : ops.prepare(NormKernels.MUL_ROW_SCALE,
                    ActKernels.global1d((long) k * outDim, ops.local()), new long[]{ops.local()},
                    NormKernels.mulRowScaleArgs(k, outDim, postNorm.shard(0), outDim, expertOut.buffer()));
            this.wsum = ops.prepare(MoeKernels.WEIGHTED_SUM, MoeKernels.elementGlobal(outDim, ops.local()),
                    new long[]{ops.local()},
                    MoeKernels.weightedSumArgs(1, outDim, k, 0, expertOut.buffer(), router.weights.buffer(),
                            expertOut.buffer()));
            this.decode = null;
            this.batch = null;
            return;
        }
        this.act = scratch.temp(T_ACT_IND, Math.multiplyExact((long) pairs, nFf));
        this.gate = plan.gated ? scratch.temp(T_GATE_IND, Math.multiplyExact((long) pairs, plan.gateUpRows()))
                : null;
        this.up = plan.gated && !plan.fused ? scratch.temp(T_UP_IND, Math.multiplyExact((long) pairs, nFf)) : null;
        this.grouped = !BatchInvariance.enabled() && groupedUsable(pairs);
        this.sort = new GpuGroupedRows(ops, scratch, ctx.kernels(), T_ROWS_IND, ubatch, k, nExpert,
                router.ids.buffer(), grouped);
        this.nTiles = sort.nTiles();
        Path d = null;
        Path b = null;
        try {
            this.scalePairs = plan.gated && !plan.fused && (gateScale != null || upScale != null)
                    ? scalePairs(scratch) : null;
            d = hasBiases() ? null : new Path(false);
            b = new Path(true);
            this.decode = d;
            this.batch = b;
        } catch (RuntimeException e) {
            throw Closeables.close(this, Closeables.close(d, Closeables.close(b, e)));
        }
        if (postNorm != null) {
            this.postScale = ops.prepare(NormKernels.MUL_ROW_SCALE,
                    ActKernels.global1d((long) k * outDim, ops.local()), new long[]{ops.local()},
                    NormKernels.mulRowScaleArgs(k, outDim, postNorm.shard(0), outDim, expertOut.buffer()));
        } else {
            this.postScale = null;
        }
        this.wsum = ops.prepare(MoeKernels.WEIGHTED_SUM, MoeKernels.elementGlobal(outDim, ops.local()),
                new long[]{ops.local()},
                MoeKernels.weightedSumArgs(1, outDim, k, 0, expertOut.buffer(), router.weights.buffer(),
                        expertOut.buffer()));
    }

    /**
     * Whether the batched products can run as the grouped GEMM over the banks, the rule of {@link GpuMoeFfn}: the
     * align kernel and both grid dimensions are available for {@code pairs} rows ({@link GpuGroupedRows#alignUsable})
     * and every bank's type has the map form of the grouped GEMM for its reduction length
     * ({@link GpuGroupedRows#groupedUsable}).
     */
    private boolean groupedUsable(int pairs) {
        if (!GpuGroupedRows.alignUsable(ctx.kernels(), pairs, nExpert, Math.max(plan.gateUpRows(), outDim))) {
            return false;
        }
        int count = cache.tensorCount(weightLayer);
        for (int i = 0; i < count; i++) {
            int reduction = i == count - 1 ? nFf : inDim;
            if (!GpuGroupedRows.groupedUsable(ctx.kernels(), cache.tensorType(weightLayer, i), reduction, true)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Local size of the id-matvec kernels: the preferred row-reduce size, raised to the largest threads-per-block
     * of the bank types, as a power of two in {@code [32, MAX_LOCAL]}.
     */
    private int idLocal() {
        int l = Math.max(32, ceilPow2(ops.local()));
        for (int i = 0; i < cache.tensorCount(weightLayer); i++) {
            l = Math.max(l, ceilPow2(MoeKernels.threadsPerBlock(cache.tensorType(weightLayer, i))));
        }
        return Math.min(l, MoeKernels.MAX_LOCAL);
    }

    private static int ceilPow2(int v) {
        return v <= 1 ? 1 : Integer.highestOneBit(v - 1) << 1;
    }

    private GpuBuffer placeholder() {
        return ctx.placeholders().readOnlyFloats();
    }

    private GpuBuffer bufOrPlaceholder(GpuWeight w) {
        return w != null ? w.shard(0) : placeholder();
    }

    /** The bank index of each expert tensor: gate_up or gate or up first, down last. */
    private int downBank() {
        return 1;
    }

    private void checkBanks() {
        int count = cache.tensorCount(weightLayer);
        int expected = plan.fused ? 2 : plan.gated ? 3 : 2;
        if (count != expected) {
            throw new IllegalArgumentException("layer " + layer.il() + ": the cache holds " + count
                    + " expert tensor(s) of the layer, the plan needs " + expected);
        }
        String firstName = TensorNames.layerName(weightLayer, TensorNames.weight(plan.fused
                ? TensorNames.FFN_GATE_UP_EXPS
                : plan.gated ? TensorNames.FFN_GATE_EXPS : TensorNames.FFN_UP_EXPS));
        if (!cache.tensor(weightLayer, 0).name().equals(firstName)) {
            throw new IllegalArgumentException("layer " + layer.il() + ": the cache's first expert tensor is "
                    + cache.tensor(weightLayer, 0).name() + ", the plan expects " + firstName);
        }
        String lastName = TensorNames.layerName(weightLayer, TensorNames.weight(TensorNames.FFN_DOWN_EXPS));
        if (!cache.tensor(weightLayer, count - 1).name().equals(lastName)) {
            throw new IllegalArgumentException("layer " + layer.il() + ": the cache's last expert tensor is "
                    + cache.tensor(weightLayer, count - 1).name() + ", the plan expects " + lastName);
        }
        for (int i = 0; i < count; i++) {
            MoeExpertCache.TensorPlan t = cache.tensor(weightLayer, i);
            if (!MoeKernels.supports(t.type())) {
                throw new IllegalArgumentException("layer " + layer.il() + ": cached expert tensor " + t.name()
                        + " has type " + t.type() + ", which has no expert matvec kernels");
            }
            if (t.nExpert() != nExpert) {
                throw new IllegalArgumentException("layer " + layer.il() + ": cached expert tensor " + t.name()
                        + " holds " + t.nExpert() + " experts, the plan routes over " + nExpert);
            }
            boolean down = i == count - 1;
            long planRows = !down && plan.fused ? 2L * nFf : down ? outDim : nFf;
            long planCols = down ? nFf : inDim;
            if (t.rowsPerExpert() != planRows || t.cols() != planCols) {
                throw new IllegalArgumentException("layer " + layer.il() + ": cached expert tensor " + t.name()
                        + " is [" + t.cols() + ", " + t.rowsPerExpert() + ", " + nExpert + "], the plan expects ["
                        + planCols + ", " + planRows + ", " + nExpert + "]");
            }
        }
    }

    /** True when the layer's routed experts carry biases, which the paths add over all pairs. */
    private boolean hasBiases() {
        return gateBias != null || upBias != null || gateUpBias != null || downBias != null;
    }

    /** True for a trunk layer: the tail-skip rule never drops a NextN / MTP layer's experts. */
    private static boolean mainModelLayer(GpuForwardContext ctx, LayerSpec layer) {
        return !ctx.spec().nextnLayers().contains(layer);
    }

    private static int checkTailSkipRank(int rank) {
        if (rank < MoeTailSkip.OFF || rank > MoeTailSkip.MAX_RANK) {
            throw new IllegalArgumentException("tail-skip rank must be " + MoeTailSkip.OFF + " (off) or 1-"
                    + MoeTailSkip.MAX_RANK + ", got " + rank);
        }
        return rank;
    }

    /** An F32 per-expert vector of {@code width * n_expert} values in one buffer, or null when absent. */
    private GpuWeight resolveVector(GpuWeights w, String suffix, int width) {
        GpuWeight v = w.layerOrNull(layer, suffix);
        if (v == null) {
            return null;
        }
        checkVector(v, width);
        return v;
    }

    private void checkVector(GpuWeight v, int width) {
        long n = v.rows() * v.cols();
        if (v.type() != GGMLType.F32 || !v.isSingleShard() || n != (long) width * nExpert) {
            throw new IllegalStateException("layer " + layer.il() + ": " + v.name() + " must be one F32 buffer of "
                    + width + " x " + nExpert + " values, got " + v);
        }
    }

    /**
     * Interleaved {@code [gate_s[e], up_s[e]]} pairs for the separate gate/up gate-up kernel
     * ({@code has_scale} 2), 1 where a tensor has no scale; built once into a per-layer temporary.
     */
    private GpuBuffer scalePairs(GpuScratch scratch) {
        ScratchSlot pairs = scratch.temp(T_SCALE_PAIRS_IND + cache.tensor(weightLayer, 0).name(), 2L * nExpert);
        ops.fill(pairs.buffer(), 4L * pairs.offset(), 8L * nExpert, Float.floatToRawIntBits(1.0f));
        if (gateScale != null) {
            ops.copy2d(gateScale.shard(0), 0, 1, pairs.buffer(), pairs.offset(), 2, nExpert, 1);
        }
        if (upScale != null) {
            ops.copy2d(upScale.shard(0), 0, 1, pairs.buffer(), pairs.offset() + 1, 2, nExpert, 1);
        }
        return pairs.buffer();
    }

    /**
     * The routed experts of {@code n} tokens over the input rows at {@code x[xOff]}: cached experts on the GPU,
     * misses on the host, the weighted sum into {@code dst} from element 0 with {@code flags}. A step the prefetch
     * pool takes reads every expert from VRAM instead. While an activation collection's CPU matmul tap wants the
     * layer's expert products, every pair is computed on the host, so the tap sees each routed pair exactly as on
     * the CPU engine. Batch invariance also computes every pair on the host and skips the prefetch: a device
     * product and a host product of one expert accumulate differently, so which experts another sequence left in
     * the cache would otherwise move a row's logits.
     */
    void run(int n, GpuBuffer x, int xOff, GpuBuffer dst, int flags) {
        ensureOpen();
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("token count " + n + " outside [1, " + ubatch + "]");
        }
        int pairs = n * k;
        boolean invariant = BatchInvariance.enabled();
        Path p = served && n == 1 && !hasBiases() && !invariant ? decode : batch;
        GpuBuffer xs = x;
        int xsOff = xOff;
        if (weightBefore) {
            if (xOff != 0) {
                ops.copy(x, xOff, input0.buffer(), input0.offset(), n * inDim);
                xs = input0.buffer();
                xsOff = input0.offset();
            }
            p.scaleInput.setBuffer(SCALE_INPUT_X_ARG, xs);
            p.scaleInput.setInt(MoeKernels.ARG_N_TOKENS, n);
            p.scaleInput.setGlobal(MoeKernels.elementGlobal((long) pairs * inDim, ops.local()));
            p.scaleInput.dispatch();
            xs = weightedIn.buffer();
            xsOff = weightedIn.offset();
        }
        boolean collected = hosted.observed();
        /* Batch invariance computes every routed pair on the host: a device-computed cache hit and a host-computed
         * miss evaluate one expert product with different accumulation orders, so which experts other sequences
         * left in the cache would otherwise move a row's logits by an ulp. The prefetch's device dispatch is
         * skipped for the same reason. */
        if (!invariant && !collected && prefetch != null && prefetch.begin(weightLayer, n, pairs)) {
            runPrefetched(n, p, xs, xsOff);
        } else {
            runHosted(n, p, xs, xsOff, collected || invariant);
        }
        finish(n, dst, flags);
        if (served && !invariant) {
            cache.endStep(weightLayer);
        }
    }

    /** The cached experts on the GPU over the cache's map, the misses on the host; every pair when {@code allHost}. */
    private void runHosted(int n, Path p, GpuBuffer xs, int xsOff, boolean allHost) {
        int pairs = n * k;
        int xsCount = Math.toIntExact((long) (weightBefore ? pairs : n) * inDim);
        // both reads behind one synchronization: the input rows queue ahead of the ids and land first
        GpuEvent xsRead = ctx.queue().enqueueReadBuffer(xs, 4L * xsOff, 4L * xsCount, xsHost);
        int[] ids = readIds(pairs);
        await(xsRead);
        float[] xsValues = xsCount == xsHost.length ? xsHost : Arrays.copyOf(xsHost, xsCount);
        MoeExpertCache.Step step = served && !allHost
                ? cache.beginStep(weightLayer, ids, pairs, n, true, tailSkipRank)
                : MoeExpertCache.Step.allMiss(pairs);
        if (step.hitCount() > 0) {
            p.update(n);
            p.bindInput(xs, xsOff);
            p.dispatch(n);
            if (step.missCount() > 0) {
                // The id readback drained the queue, so a batching queue would otherwise hold the hit pass until
                // the misses' uploads: submitting it lets the device run the hits while the host computes.
                ctx.queue().flush();
            }
        }
        if (step.missCount() > 0) {
            float[] rows = hosted.run(ids, step.missPairs(), step.missCount(), xsValues, weightBefore ? 1 : k);
            uploadMissRows(step, rows);
        }
        zeroSkippedRows(step);
    }

    /** Every expert on the GPU in one batched dispatch over the prefetch pool's prefill map. */
    private void runPrefetched(int n, Path p, GpuBuffer xs, int xsOff) {
        int pairs = n * k;
        int[] ids = readIds(pairs);
        MoeExpertCache.Step step = cache.beginStep(weightLayer, ids, pairs, n, false);
        GpuBuffer map = prefetch.cover(weightLayer, ids, step.missPairs(), step.missCount());
        p.update(n);
        p.bindInput(xs, xsOff);
        p.bindMap(map);
        try {
            p.dispatch(n);
        } finally {
            p.bindMap(cache.mapBuffer(weightLayer));
        }
        prefetch.end(weightLayer);
    }

    /** The step's routed expert ids, read back after the router and counted into the heat recorder if any. */
    private int[] readIds(int pairs) {
        GpuEvent evt = ctx.queue().enqueueReadBuffer(router.ids.buffer(), 4L * router.ids.offset(),
                4L * pairs, idBytes);
        await(evt);
        int[] ids = new int[pairs];
        ByteBuffer.wrap(idBytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(ids);
        if (heat != null) {
            heat.note(weightLayer, ids, pairs);
        }
        return ids;
    }

    private void await(GpuEvent evt) {
        if (evt == null) {
            ctx.queue().finish();
            return;
        }
        try {
            evt.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting for the routed reads of layer "
                    + layer.il(), e);
        } finally {
            evt.close();
        }
    }

    /**
     * Uploads the miss rows into the expert output, one write per run of consecutive pairs (a prefill whose experts
     * miss for consecutive slots writes them together).
     */
    private void uploadMissRows(MoeExpertCache.Step step, float[] rows) {
        int[] pairs = step.missPairs();
        int count = step.missCount();
        for (int q = 0; q < count; ) {
            int end = runEnd(pairs, q, count);
            int len = (end - q) * outDim;
            float[] src = rows;
            if (q > 0) {
                src = runStaging(len);
                System.arraycopy(rows, q * outDim, src, 0, len);
            }
            GpuEvent evt = ctx.queue().enqueueWriteBuffer(expertOut.buffer(),
                    4L * (expertOut.offset() + (long) pairs[q] * outDim), 4L * len, src);
            if (evt != null) {
                evt.close();
            }
            q = end;
        }
    }

    /**
     * Zeros the expert output rows of the pairs the tail-skip rule drops ({@link MoeTailSkip}) on the device, one
     * fill per run of consecutive pairs: the weighted sum then adds nothing for them while the other experts keep
     * their router weights (no renormalisation). The rows are misses, so no product wrote them this step; the fill
     * clears whatever an earlier step left in the scratch.
     */
    private void zeroSkippedRows(MoeExpertCache.Step step) {
        int[] skipped = step.skippedPairs();
        int count = step.skippedCount();
        for (int q = 0; q < count; ) {
            int end = runEnd(skipped, q, count);
            ops.fill(expertOut.buffer(), 4L * (expertOut.offset() + (long) skipped[q] * outDim),
                    4L * (end - q) * outDim, 0);
            q = end;
        }
    }

    /** End of the run of consecutive pair indices starting at {@code pairs[from]}, exclusive, within {@code count}. */
    static int runEnd(int[] pairs, int from, int count) {
        int end = from + 1;
        while (end < count && pairs[end] == pairs[end - 1] + 1) {
            end++;
        }
        return end;
    }

    /** A host array of at least {@code len} floats for a run of miss rows past the first, grown on demand. */
    private float[] runStaging(int len) {
        if (rowHost.length < len) {
            rowHost = new float[len];
        }
        return rowHost;
    }

    /** Post-expert scale and norm over all pair rows, then the weighted sum into {@code dst}. */
    private void finish(int n, GpuBuffer dst, int flags) {
        if (postScale != null) {
            postScale.setInt(NormKernels.ARG_ROW_SCALE_N_ROWS, n * k);
            postScale.setGlobal(ActKernels.global1d((long) n * k * outDim, ops.local()));
            postScale.dispatch();
            ops.norm(NormKind.RMS, null, null, postEps, expertOut.buffer(), expertOut.offset(),
                    expertOut.buffer(), expertOut.offset(), n * k, outDim);
        }
        wsum.setInt(MoeKernels.ARG_N_TOKENS, n);
        wsum.setGlobal(MoeKernels.elementGlobal((long) n * outDim, ops.local()));
        wsum.setBuffer(WSUM_DST_ARG, dst);
        wsum.setInt(WSUM_FLAGS_ARG, flags | (weightBefore ? MoeKernels.WSUM_UNWEIGHTED : 0));
        wsum.dispatch();
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = Closeables.closeAll(Arrays.asList(decode, batch, postScale, wsum, sort), null);
        if (failure != null) {
            throw failure;
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the cached MoE expert pass of layer " + layer.il() + " is closed");
        }
    }

    @Override
    public String toString() {
        return "GpuCachedMoePass{layer=" + layer.il() + ", " + plan + ", slots="
                + (served ? cache.slotCount(weightLayer) : 0) + ", "
                + (weightBefore ? "weight-before" : "weight-after") + "}";
    }

    /**
     * The prepared dispatches of one path: the one-token products ({@code {T}_MatVecId*Ind}, one dispatch per
     * product), or the sorted batch ({@code {T}_GroupedGemmInd} over the aligned tiles when
     * {@link GpuCachedMoePass#grouped}, else {@code {T}_BatchMatMulIdInd}, plus the {@code LLM_GLU} / activation
     * between the products).
     */
    private final class Path implements AutoCloseable {
        final boolean batched;
        final PreparedDispatch scaleInput;
        PreparedDispatch gateUp;
        PreparedDispatch gateMm;
        PreparedDispatch upMm;
        PreparedDispatch gateBiasAdd;
        PreparedDispatch upBiasAdd;
        PreparedDispatch downBiasAdd;
        PreparedDispatch down;
        int lastN = 1;

        /** One path: the decode products, or the batch products over {@link GpuCachedMoePass#sort}'s rows. */
        Path(boolean batched) {
            this.batched = batched;
            PreparedDispatch si = null;
            try {
                si = weightBefore ? prepareScaleInput() : null;
                this.scaleInput = si;
                build();
            } catch (RuntimeException e) {
                throw Closeables.close(si, e);
            }
        }

        private PreparedDispatch prepareScaleInput() {
            return ops.prepare(MoeKernels.SCALE_INPUT,
                    MoeKernels.elementGlobal((long) k * inDim, ops.local()), new long[]{ops.local()},
                    MoeKernels.scaleInputArgs(1, inDim, k, input0.buffer(), router.weights.buffer(),
                            weightedIn.buffer()));
        }

        /** The kernels of the path over the banks; one dispatch per product. */
        private void build() {
            int pairs = k;
            GpuBuffer map = cache.mapBuffer(weightLayer);
            GpuBuffer slots = router.ids.buffer();
            int dnBank = dnBank();
            int xDiv = weightBefore ? 1 : k;
            int xStride = weightBefore ? inDim : 0;
            if (batched) {
                if (plan.fused) {
                    gateMm = prepareBatch(0, plan.gateUpRows(), inDim, xDiv, upScale, placeholder(), gate.buffer());
                } else if (plan.gated) {
                    gateMm = prepareBatch(0, nFf, inDim, xDiv, gateScale, placeholder(), gate.buffer());
                    upMm = prepareBatch(1, nFf, inDim, xDiv, upScale, placeholder(), up.buffer());
                } else {
                    upMm = prepareBatch(0, nFf, inDim, xDiv, upScale, placeholder(), act.buffer());
                }
            } else if (plan.fused) {
                gateUp = ops.prepare(MoeKernels.matVecIdGateUpInd(cache.tensorType(weightLayer, 0)),
                        idGlobal(nFf, pairs), idLocal2(),
                        MoeKernels.matVecIdGateUpIndArgs(nFf, inDim, pairs, xStride, 0, plan.glu, plan.params, true,
                                upScale != null ? MoeKernels.SCALE_SHARED : MoeKernels.SCALE_NONE, slots, map,
                                cache.bank(weightLayer, 0), cache.bank(weightLayer, 0), placeholder(),
                                bufOrPlaceholder(upScale), act.buffer()));
            } else if (plan.gated) {
                gateUp = ops.prepare(MoeKernels.matVecIdGateUpInd(cache.tensorType(weightLayer, 0)),
                        idGlobal(nFf, pairs), idLocal2(),
                        MoeKernels.matVecIdGateUpIndArgs(nFf, inDim, pairs, xStride, 0, plan.glu, plan.params, false,
                                gateScale != null || upScale != null ? MoeKernels.SCALE_GATE_UP_PAIRS
                                        : MoeKernels.SCALE_NONE, slots, map,
                                cache.bank(weightLayer, 0), cache.bank(weightLayer, 1), placeholder(),
                                scalePairs != null ? scalePairs : placeholder(), act.buffer()));
            } else {
                upMm = ops.prepare(MoeKernels.matVecIdInd(cache.tensorType(weightLayer, 0)),
                        idGlobal(nFf, pairs), idLocal2(),
                        MoeKernels.matVecIdIndArgs(nFf, inDim, pairs, xStride, 0, 0, upScale != null, slots, map,
                                cache.bank(weightLayer, 0), placeholder(), bufOrPlaceholder(upScale),
                                act.buffer()));
            }
            if (batched) {
                down = prepareBatch(dnBank, outDim, nFf, 1, downScale, act.buffer(), expertOut.buffer());
            } else {
                down = ops.prepare(MoeKernels.matVecIdInd(cache.tensorType(weightLayer, dnBank)),
                        idGlobal(outDim, pairs), idLocal2(),
                        MoeKernels.matVecIdIndArgs(outDim, nFf, pairs, nFf, 0, 0, downScale != null, slots, map,
                                cache.bank(weightLayer, dnBank), placeholder(), bufOrPlaceholder(downScale),
                                expertOut.buffer()));
            }
            // the biases apply after their product and before the activation, over every pair (the host applies
            // them to the miss rows it uploads, whose device writes it overwrites)
            PreparedDispatch gb = null;
            PreparedDispatch ub = null;
            PreparedDispatch db = null;
            if (batched) {
                if (plan.fused) {
                    gb = addId(gateUpBias, plan.gateUpRows(), gate.buffer());
                } else if (plan.gated) {
                    gb = addId(gateBias, nFf, gate.buffer());
                    ub = addId(upBias, nFf, up.buffer());
                } else {
                    ub = addId(upBias, nFf, act.buffer());
                }
                db = addId(downBias, outDim, expertOut.buffer());
            }
            this.gateBiasAdd = gb;
            this.upBiasAdd = ub;
            this.downBiasAdd = db;
        }

        /**
         * One batched product of bank {@code bank} over the sorted rows into {@code out}: the grouped GEMM over the
         * aligned tiles when {@link GpuCachedMoePass#grouped}, else the per-pair {@code {T}_BatchMatMulIdInd}.
         * {@code xDiv = k} reads the token rows (gate and up), {@code xDiv = 1} the pair rows (down, and gate and up
         * of a weight-before-FFN layer), each {@code inDim2} values long; {@code scale} is the product's per-expert
         * scale vector, or null.
         */
        private PreparedDispatch prepareBatch(int bank, int outDim2, int inDim2, int xDiv, GpuWeight scale,
                                              GpuBuffer in, GpuBuffer out) {
            GGMLType type = cache.tensorType(weightLayer, bank);
            GpuBuffer map = cache.mapBuffer(weightLayer);
            if (grouped) {
                return ops.prepare(GpuGroupedRows.groupedKernel(ops, type, true),
                        MoeKernels.groupedGemmGlobal(outDim2, nTiles), MoeKernels.groupedGemmLocal(),
                        MoeKernels.groupedGemmIndArgs(outDim2, inDim2, nTiles, xDiv, inDim2, 0, 0, scale != null,
                                sort.gemmToken(), sort.gemmSlot(), map, cache.bank(weightLayer, bank), in,
                                bufOrPlaceholder(scale), out));
            }
            return ops.prepare(MoeKernels.batchMatMulIdInd(type),
                    MoeKernels.batchMatMulIdGlobal(outDim2, k, idLocal), new long[]{idLocal, 1},
                    MoeKernels.batchMatMulIdIndArgs(outDim2, inDim2, k, xDiv, 0, scale != null, sort.rowToken(),
                            sort.rowSlot(), map, cache.bank(weightLayer, bank), in, bufOrPlaceholder(scale), out));
        }

        /** The bank index of the down product: 1 after a fused or ungated first tensor, 2 after gate and up. */
        private int dnBank() {
            return plan.fused || !plan.gated ? 1 : 2;
        }

        private long[] idGlobal(int outDim2, int nSel) {
            return MoeKernels.idMatVecGlobal(outDim2, nSel, idLocal);
        }

        private long[] idLocal2() {
            return new long[]{idLocal, 1};
        }

        /** Token-count dependent arguments and global sizes of the batched path. */
        void update(int n) {
            if (!batched || n == lastN) {
                return;
            }
            int pairs = n * k;
            sort.update(n);
            if (!grouped) {
                setRows(gateMm, plan.fused ? plan.gateUpRows() : nFf, pairs);
                setRows(upMm, nFf, pairs);
                setRows(down, outDim, pairs);
            }
            setPairs(gateBiasAdd, plan.fused ? plan.gateUpRows() : nFf, pairs);
            setPairs(upBiasAdd, nFf, pairs);
            setPairs(downBiasAdd, outDim, pairs);
            lastN = n;
        }

        private void setRows(PreparedDispatch d, int rows, int pairs) {
            if (d != null) {
                d.setInt(MoeKernels.ARG_BMM_N_ROWS, pairs);
                d.setGlobal(MoeKernels.batchMatMulIdGlobal(rows, pairs, idLocal));
            }
        }

        /** {@code LLM_AddId} of a per-expert bias over the path's pairs, or null without the bias. */
        private PreparedDispatch addId(GpuWeight bias, int dim, GpuBuffer data) {
            if (bias == null) {
                return null;
            }
            return ops.prepare(MoeKernels.ADD_ID, MoeKernels.elementGlobal((long) k * dim, ops.local()),
                    new long[]{ops.local()},
                    MoeKernels.addIdArgs(k, dim, router.ids.buffer(), bias.shard(0), data));
        }

        private void setPairs(PreparedDispatch d, int dim, int pairs) {
            if (d != null) {
                d.setInt(MoeKernels.ARG_ADD_ID_N_SEL, pairs);
                d.setGlobal(MoeKernels.elementGlobal((long) pairs * dim, ops.local()));
            }
        }

        /** Input view of every product: the tokens' rows, or the weighted pair rows before the FFN. */
        void bindInput(GpuBuffer x, int xOff) {
            if (gateUp != null) {
                gateUp.setBuffer(MoeKernels.ARG_GUI_INPUT, x);
                gateUp.setInt(MoeKernels.ARG_GU_IN_OFF, xOff);
            }
            setInput(gateMm, x, xOff);
            setInput(upMm, x, xOff);
            setInput(down, act.buffer(), act.offset());
        }

        /** The input view of one product of either path. */
        private void setInput(PreparedDispatch d, GpuBuffer x, int xOff) {
            if (d == null) {
                return;
            }
            boolean gemm = batched && grouped;
            d.setBuffer(gemm ? MoeKernels.ARG_GGI_INPUT : batched ? MoeKernels.ARG_BMI_INPUT
                    : MoeKernels.ARG_IDI_INPUT, x);
            d.setInt(gemm ? MoeKernels.ARG_GG_IN_OFF : MoeKernels.ARG_IND_IN_OFF, xOff);
        }

        /**
         * Binds the expert map every batched product reads: the cache's map, or a prefetch pool's prefill map over
         * the same banks.
         *
         * @throws IllegalStateException on the one-token path, whose products take no prefill map
         */
        void bindMap(GpuBuffer map) {
            if (!batched) {
                throw new IllegalStateException("the one-token path reads the cache's map only");
            }
            int mapArg = grouped ? MoeKernels.ARG_GGI_MAP : MoeKernels.ARG_BMI_MAP;
            for (PreparedDispatch d : new PreparedDispatch[]{gateMm, upMm, down}) {
                if (d != null) {
                    d.setBuffer(mapArg, map);
                }
            }
        }

        /** Enqueues the products, the biases and the activation steps between them. */
        void dispatch(int n) {
            int pairs = n * k;
            if (batched) {
                sort.dispatch();
            }
            if (gateUp != null) {
                gateUp.dispatch();
            } else {
                run(gateMm);
                run(gateBiasAdd);
                run(upMm);
                run(upBiasAdd);
                activate(pairs);
            }
            down.dispatch();
            run(downBiasAdd);
        }

        /** {@code LLM_GLU} of the projected rows into {@link GpuCachedMoePass#act}, or the activation. */
        private void activate(int pairs) {
            int nFf = GpuCachedMoePass.this.nFf;
            if (plan.fused) {
                ops.glu(plan.glu, plan.params, gate.buffer(), gate.offset(), 2 * nFf, gate.buffer(),
                        gate.offset() + nFf, 2 * nFf, act.buffer(), act.offset(), nFf, pairs, nFf);
            } else if (plan.gated) {
                ops.glu(plan.glu, plan.params, gate.buffer(), gate.offset(), nFf, up.buffer(), up.offset(), nFf,
                        act.buffer(), act.offset(), nFf, pairs, nFf);
            } else {
                ops.unary(plan.act, plan.params, act.buffer(), act.offset(), act.buffer(), act.offset(),
                        pairs * nFf);
            }
        }

        private void run(PreparedDispatch d) {
            if (d != null) {
                d.dispatch();
            }
        }

        @Override
        public void close() {
            RuntimeException failure = Closeables.closeAll(Arrays.asList(scaleInput, gateUp, gateMm, upMm,
                    gateBiasAdd, upBiasAdd, downBiasAdd, down), null);
            if (failure != null) {
                throw failure;
            }
        }
    }
}
