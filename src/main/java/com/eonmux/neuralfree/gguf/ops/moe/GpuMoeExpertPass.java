package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuMatmulTap;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.Arrays;
import java.util.List;
import java.util.function.IntFunction;

/**
 * One expert set over the ids and weights of a {@link GpuMoeFfn.Router}: the one-token path
 * ({@code {T}_MatVecId*}) and the batched path ({@code LLM_MoE_Bucketize}, then {@code {T}_GroupedGemm} over the
 * aligned rows of {@code LLM_MoE_AlignBlocks} when {@link #grouped} holds, else {@code {T}_BatchMatMulId}), each a
 * fixed list of prepared dispatches whose token counts, global sizes, input view and destination are re-set per step.
 * The routed experts of a hosted layer run over the expert cache through {@link GpuCachedMoePass} instead.
 */
final class GpuMoeExpertPass implements GpuMoeFfn.Routed {

    private static final String T_GATE = "moe.gate";
    private static final String T_UP = "moe.up";
    private static final String T_ACT = "moe.act";
    private static final String T_EXPERT_OUT = "moe.expert_out";
    private static final String T_WEIGHTED_IN = "moe.weighted_in";
    private static final String T_INPUT = "moe.input";
    /** Scratch prefix of the expert sort ({@link GpuGroupedRows}). */
    private static final String T_ROWS = "moe";
    /**
     * Prefix of the interleaved gate/up expert scales ({@code SCALE_GATE_UP_PAIRS}), one temporary per gate tensor
     * name, written once at bind and never shared with another tensor.
     */
    private static final String T_SCALE_PAIRS = "moe.scale_pairs.";

    /** {@code x} of {@code LLM_MoE_ScaleInput(i n_tokens, i n_embd, i k, R x, R weights, W out)}. */
    private static final int ARG_SCALE_INPUT_X = 3;
    /**
     * {@code flags} of
     * {@code LLM_MoE_WeightedSum(i n_tokens, i n_embd, i k, i flags, R expert_out, R weights, RWn dst)}.
     */
    private static final int ARG_WSUM_FLAGS = 3;
    /** {@code dst} of {@code LLM_MoE_WeightedSum}. */
    private static final int ARG_WSUM_DST = 6;

    private final GpuForwardContext ctx;
    private final GpuOps ops;
    private final LayerSpec layer;
    private final int ubatch;
    final MoeExpertWeights plan;
    final MoeExpertWeights.Gpu w;
    final GpuMoeFfn.Router src;
    final int k;
    final boolean weightBefore;
    final int idLocal;
    final GpuWeight postNorm;
    final float postEps;
    /** Batched path over the aligned tiles ({@code {T}_GroupedGemm}) instead of {@code {T}_BatchMatMulId}. */
    final boolean grouped;
    final int nTiles;
    /** The expert sort of the batched path, and its aligned rows when {@link #grouped}. */
    final GpuGroupedRows rows;
    final ScratchSlot gate;
    final ScratchSlot up;
    final ScratchSlot act;
    final ScratchSlot expertOut;
    final ScratchSlot weightedIn;
    final ScratchSlot input0;
    final boolean gateUpKernel;
    /**
     * True when the expert shards cover only part of the expert set (expert parallel): the pairs of the other
     * ranks' experts are then zeroed before the products, whose id dispatches skip experts outside the shard.
     */
    final boolean zeroSkippedExperts;
    final GpuProjectionDeltas.ExpertSite gateSite;
    final GpuProjectionDeltas.ExpertSite upSite;
    final GpuProjectionDeltas.ExpertSite gateUpSite;
    final GpuProjectionDeltas.ExpertSite downSite;
    final Path decode;
    final Path batch;

    GpuMoeExpertPass(GpuForwardContext ctx, LayerSpec layer, MoeExpertWeights plan, GpuWeights weights,
                     GpuMoeFfn.Router src, boolean weightBefore, GpuWeight postNorm, float postEps) {
        this.ctx = ctx;
        this.ops = ctx.ops();
        this.layer = layer;
        this.ubatch = ctx.scratch().ubatch();
        this.plan = plan;
        this.w = plan.resolve(weights, layer);
        GpuProjectionDeltas deltas = ctx.projectionDeltas();
        this.gateSite = w.gate != null ? deltas.serveExperts(layer, plan.gateName(), w.gate) : null;
        this.upSite = w.up != null ? deltas.serveExperts(layer, plan.upName(), w.up) : null;
        this.gateUpSite = w.gateUp != null ? deltas.serveExperts(layer, plan.gateUpName(), w.gateUp) : null;
        this.downSite = deltas.serveExperts(layer, plan.downName(), w.down);
        this.src = src;
        this.k = src.params.k();
        this.weightBefore = weightBefore;
        this.postNorm = postNorm;
        this.postEps = postEps;
        if (postNorm != null) {
            f32Vector(postNorm, plan.outDim);
        }
        this.idLocal = idLocal(w.gate, w.up, w.gateUp, w.down);
        this.gateUpKernel = plan.gated && !w.hasBiases() && (plan.fused || (w.gate.type() == w.up.type()
                && MoeExpertWeights.samePartition(w.gateShards, w.upShards)));
        this.zeroSkippedExperts = !w.coversEveryExpert();
        checkLocal(w.gate);
        checkLocal(w.up);
        checkLocal(w.gateUp);
        checkLocal(w.down);
        /* Batch invariance keeps the sorted per-pair product for every row count; the grouped expert GEMM
         * tiles K with its own split, so it would change a row's reduction order with the batch. */
        this.grouped = !BatchInvariance.enabled() && groupedUsable();
        checkGrid();
        GpuScratch scratch = ctx.scratch();
        long pairs = rows(k);
        this.gate = plan.gated ? scratch.temp(T_GATE, pairs * plan.gateUpRows()) : null;
        this.up = plan.gated && !plan.fused ? scratch.temp(T_UP, pairs * plan.nFf) : null;
        this.act = scratch.temp(T_ACT, pairs * plan.nFf);
        this.expertOut = scratch.temp(T_EXPERT_OUT, pairs * plan.outDim);
        this.weightedIn = weightBefore ? scratch.temp(T_WEIGHTED_IN, pairs * plan.inDim) : null;
        this.input0 = weightBefore ? scratch.temp(T_INPUT, rows(plan.inDim)) : null;
        GpuGroupedRows r = null;
        Path d = null;
        try {
            r = new GpuGroupedRows(ops, scratch, ctx.kernels(), T_ROWS, ubatch, k, plan.nExpert,
                    src.expertIds(), grouped);
            this.nTiles = grouped ? r.nTiles() : 1;
            this.rows = r;
            d = new Path(false);
            this.batch = new Path(true);
        } catch (RuntimeException e) {
            throw Closeables.close(r, Closeables.close(d, e));
        }
        this.decode = d;
    }

    /**
     * Whether the batched products can run as the grouped GEMM: the align kernel and both grid dimensions are
     * available ({@link GpuGroupedRows#alignUsable}) and every expert tensor passes
     * {@link GpuGroupedRows#groupedUsable}. Anything else keeps {@code {T}_BatchMatMulId}.
     */
    private boolean groupedUsable() {
        if (!GpuGroupedRows.alignUsable(ctx.kernels(), Math.toIntExact(rows(k)), plan.nExpert,
                Math.max(plan.gateUpRows(), plan.outDim))) {
            return false;
        }
        return tensorGrouped(w.gate, plan.inDim) && tensorGrouped(w.up, plan.inDim)
                && tensorGrouped(w.gateUp, plan.inDim) && tensorGrouped(w.down, plan.nFf);
    }

    private boolean tensorGrouped(GpuWeight t, int kDim) {
        return t == null || GpuGroupedRows.groupedUsable(ctx.kernels(), t.type(), kDim);
    }

    /**
     * Local size of the id-matvec kernels: the preferred row-reduce size, raised to the largest threads-per-block
     * of the expert tensor types, as a power of two in {@code [32, MAX_LOCAL]}.
     */
    private int idLocal(GpuWeight... tensors) {
        int l = Math.max(32, floorPow2(ops.local()));
        for (GpuWeight t : tensors) {
            if (t != null) {
                l = Math.max(l, ceilPow2(MoeKernels.threadsPerBlock(t.type())));
            }
        }
        return Math.min(l, MoeKernels.MAX_LOCAL);
    }

    private void checkLocal(GpuWeight t) {
        if (t != null) {
            MoeKernels.checkIdLocal(t.type(), idLocal);
        }
    }

    /** The id-matvec work-group counts must fit the device's per-dimension limits at {@code ubatch}. */
    private void checkGrid() {
        long[] max = ctx.kernels().caps().getMaxWorkGroupCount();
        long rowsX = Math.max(plan.gateUpRows(), plan.outDim);
        long tilesY = (rows(k) + MoeKernels.BMM_ROWS - 1) / MoeKernels.BMM_ROWS;
        if (max.length >= 2 && ((max[0] > 0 && rowsX > max[0]) || (max[1] > 0 && Math.max(tilesY, k) > max[1]))) {
            throw new UnsupportedOperationException("layer " + layer.il() + " " + plan + ": " + rowsX
                    + " output rows or " + tilesY + " row tiles at ubatch " + ubatch
                    + " exceed the device's work-group counts " + max[0] + " x " + max[1]
                    + "; lower neuralfree.ubatch");
        }
    }

    /**
     * Experts over {@code n} tokens whose rows of {@code in_dim} start at {@code x[xOff]}; the weighted sum goes
     * to {@code dst} from element 0 with {@code flags}.
     */
    @Override
    public void run(int n, GpuBuffer x, int xOff, GpuBuffer dst, int flags) {
        /* Batch invariance pins every row count to the sorted batch path: the one-token matvec path is a
         * different kernel, and a row's value must not depend on its neighbours. */
        Path p = n == 1 && !BatchInvariance.enabled() ? decode : batch;
        p.update(n);
        GpuBuffer in = x;
        int inOff = xOff;
        if (weightBefore) {
            if (xOff != 0) {
                ops.copy(x, xOff, input0.buffer(), input0.offset(), n * plan.inDim);
                in = input0.buffer();
            }
            p.bindScaleInput(in);
            in = weightedIn.buffer();
            inOff = 0;
        }
        p.bindInput(in, inOff);
        p.bindDst(dst, flags);
        p.dispatch(n);
    }

    /** True when a delta is registered on the gate or up expert products. */
    boolean gateUpDeltas() {
        return hasDeltas(gateSite) || hasDeltas(upSite) || hasDeltas(gateUpSite);
    }

    private boolean hasDeltas(GpuProjectionDeltas.ExpertSite site) {
        return site != null && !site.isEmpty();
    }

    @Override
    public void close() {
        closeAll(decode, batch, rows);
    }

    /** The prepared dispatches of one path (one token, or several). */
    private final class Path implements AutoCloseable {
        final boolean batched;
        PreparedDispatch scaleInput;
        ShardedDispatch gateUp;
        ShardedDispatch gateMm;
        ShardedDispatch upMm;
        PreparedDispatch gateBias;
        PreparedDispatch upBias;
        ShardedDispatch down;
        PreparedDispatch downBias;
        PreparedDispatch postScale;
        PreparedDispatch wsum;
        GpuBuffer lastIn;
        int lastInOff = -1;
        GpuBuffer lastScaleX;
        GpuBuffer lastDst;
        int lastFlags;
        int lastN;

        Path(boolean batched) {
            this.batched = batched;
            try {
                build();
            } catch (RuntimeException e) {
                throw Closeables.close(this, e);
            }
            lastN = 1;
        }

        /** The pair count at bind is {@code k} (one token); {@link #update} re-sets it per step. */
        private void build() {
            GpuBuffer ph = ctx.placeholders().readOnlyFloats();
            GpuBuffer wts = src.weights.buffer();
            GpuBuffer x0 = weightBefore ? weightedIn.buffer() : ph;
            int xStride = weightBefore ? plan.inDim : 0;
            if (weightBefore) {
                scaleInput = ops.prepare(MoeKernels.SCALE_INPUT,
                        MoeKernels.elementGlobal((long) k * plan.inDim, ops.local()), local1(),
                        MoeKernels.scaleInputArgs(1, plan.inDim, k, input0.buffer(), wts, weightedIn.buffer()));
            }
            if (gateUpKernel && !plan.fused && !batched) {
                boolean scaled = w.gateScale != null || w.upScale != null;
                gateUp = gateUpShards(w.gate.type(), false,
                        scaled ? MoeKernels.SCALE_GATE_UP_PAIRS : MoeKernels.SCALE_NONE, w.gateShards, w.upShards,
                        x0, xStride, scaled ? scalePairs() : ph);
            } else if (gateUpKernel && !batched) {
                gateUp = gateUpShards(w.gateUp.type(), true,
                        w.upScale != null ? MoeKernels.SCALE_SHARED : MoeKernels.SCALE_NONE, w.gateUpShards,
                        w.gateUpShards, x0, xStride, buf(w.upScale, ph));
            }
            buildSplitGateUp();
            down = project(w.down, w.downShards, w.downScale, plan.outDim, plan.nFf, plan.nFf, 1, act.buffer(),
                    expertOut);
            downBias = addId(w.downBias, plan.outDim, expertOut);
            if (postNorm != null) {
                postScale = ops.prepare(NormKernels.MUL_ROW_SCALE,
                        ActKernels.global1d((long) k * plan.outDim, ops.local()), local1(),
                        NormKernels.mulRowScaleArgs(k, plan.outDim, postNorm.shard(0), plan.outDim,
                                expertOut.buffer()));
            }
            wsum = ops.prepare(MoeKernels.WEIGHTED_SUM, MoeKernels.elementGlobal(plan.outDim, ops.local()),
                    local1(), MoeKernels.weightedSumArgs(1, plan.outDim, k, 0, expertOut.buffer(), wts,
                            expertOut.buffer()));
            lastDst = expertOut.buffer();
            lastFlags = 0;
        }

        /**
         * The separate gate and up products ({@code {T}_MatVecId}, {@code {T}_BatchMatMulId} or
         * {@code {T}_GroupedGemm}) and their bias additions: the batched path, and the one-token path without
         * the fused gate-up kernel or, beside it, for the steps that run with gate/up deltas.
         */
        private void buildSplitGateUp() {
            GpuBuffer x0 = weightBefore ? weightedIn.buffer() : ctx.placeholders().readOnlyFloats();
            int xStride = weightBefore ? plan.inDim : 0;
            int xDiv = weightBefore ? 1 : k;
            if (plan.fused) {
                gateMm = project(w.gateUp, w.gateUpShards, w.upScale, plan.gateUpRows(), plan.inDim, xStride, xDiv,
                        x0, gate);
                gateBias = addId(w.gateUpBias, plan.gateUpRows(), gate);
            } else if (plan.gated) {
                gateMm = project(w.gate, w.gateShards, w.gateScale, plan.nFf, plan.inDim, xStride, xDiv, x0, gate);
                upMm = project(w.up, w.upShards, w.upScale, plan.nFf, plan.inDim, xStride, xDiv, x0, up);
                gateBias = addId(w.gateBias, plan.nFf, gate);
                upBias = addId(w.upBias, plan.nFf, up);
            } else {
                upMm = project(w.up, w.upShards, w.upScale, plan.nFf, plan.inDim, xStride, xDiv, x0, act);
                upBias = addId(w.upBias, plan.nFf, act);
            }
        }

        /**
         * {@code {T}_MatVecIdGateUp} of the one-token path into {@link #act}, one dispatch per expert shard;
         * {@code gates} and {@code ups} split the experts identically (the same list for the fused tensor).
         */
        private ShardedDispatch gateUpShards(GGMLType type, boolean fused, int scaleMode,
                                             List<MoeExpertWeights.Shard> gates, List<MoeExpertWeights.Shard> ups,
                                             GpuBuffer x0, int xStride, GpuBuffer scales) {
            GpuBuffer ids = src.expertIds();
            return prepareShards(gates.size(), i -> {
                MoeExpertWeights.Shard g = gates.get(i);
                return ops.prepare(MoeKernels.matVecIdGateUp(type), idGlobal(plan.nFf), idLocal2(),
                        MoeKernels.gateUpArgs(plan.nFf, plan.inDim, k, xStride, 0, plan.glu, plan.params, fused,
                                scaleMode, g.expertBase, g.expertCount, ids, g.buffer, ups.get(i).buffer, x0,
                                scales, act.buffer()));
            });
        }

        /**
         * {@code {T}_MatVecId} (one token), {@code {T}_BatchMatMulId} or {@code {T}_GroupedGemm} (several) into
         * {@code out} from element 0, one dispatch per expert shard of {@code t}.
         */
        private ShardedDispatch project(GpuWeight t, List<MoeExpertWeights.Shard> shards, GpuWeight scale,
                                        int outDim, int inDim, int xStride, int xDiv, GpuBuffer x0,
                                        ScratchSlot out) {
            GpuBuffer s = buf(scale, ctx.placeholders().readOnlyFloats());
            boolean scaled = scale != null;
            return prepareShards(shards.size(), i -> {
                MoeExpertWeights.Shard sh = shards.get(i);
                if (batched) {
                    if (grouped) {
                        return ops.prepare(GpuGroupedRows.groupedKernel(ops, t.type(), false),
                                MoeKernels.groupedGemmGlobal(outDim, nTiles), MoeKernels.groupedGemmLocal(),
                                MoeKernels.groupedGemmArgs(outDim, inDim, nTiles, xDiv, inDim, 0, 0, scaled,
                                        sh.expertBase, sh.expertCount, rows.gemmToken(), rows.gemmSlot(),
                                        sh.buffer, x0, s, out.buffer()));
                    }
                    return ops.prepare(MoeKernels.batchMatMulId(t.type()),
                            MoeKernels.batchMatMulIdGlobal(outDim, k, idLocal), idLocal2(),
                            MoeKernels.batchMatMulIdArgs(outDim, inDim, k, xDiv, 0, scaled, sh.expertBase,
                                    sh.expertCount, rows.rowToken(), rows.rowSlot(), sh.buffer, x0, s,
                                    out.buffer()));
                }
                return ops.prepare(MoeKernels.matVecId(t.type()), idGlobal(outDim), idLocal2(),
                        MoeKernels.matVecIdArgs(outDim, inDim, k, xStride, 0, 0, scaled, sh.expertBase,
                                sh.expertCount, src.expertIds(), sh.buffer, x0, s, out.buffer()));
            });
        }

        private PreparedDispatch addId(GpuWeight bias, int dim, ScratchSlot data) {
            if (bias == null) {
                return null;
            }
            return ops.prepare(MoeKernels.ADD_ID, MoeKernels.elementGlobal((long) k * dim, ops.local()), local1(),
                    MoeKernels.addIdArgs(k, dim, src.expertIds(), bias.shard(0), data.buffer()));
        }

        private long[] idGlobal(int outDim) {
            return MoeKernels.idMatVecGlobal(outDim, k, idLocal);
        }

        private long[] idLocal2() {
            return new long[]{idLocal, 1};
        }

        /** Token-count dependent arguments and global sizes; the one-token path never changes. */
        void update(int n) {
            if (n == lastN) {
                return;
            }
            if (!batched) {
                throw new IllegalStateException("one-token expert path run with " + n + " tokens");
            }
            int pairs = n * k;
            rows.update(n);
            if (scaleInput != null) {
                scaleInput.setInt(MoeKernels.ARG_N_TOKENS, n);
                scaleInput.setGlobal(MoeKernels.elementGlobal((long) pairs * plan.inDim, ops.local()));
            }
            if (!grouped) {
                setRows(gateMm, plan.gateUpRows(), pairs);
                setRows(upMm, plan.nFf, pairs);
                setRows(down, plan.outDim, pairs);
            }
            setPairs(gateBias, plan.gateUpRows(), pairs);
            setPairs(upBias, plan.nFf, pairs);
            setPairs(downBias, plan.outDim, pairs);
            if (postScale != null) {
                postScale.setInt(NormKernels.ARG_ROW_SCALE_N_ROWS, pairs);
                postScale.setGlobal(ActKernels.global1d((long) pairs * plan.outDim, ops.local()));
            }
            wsum.setInt(MoeKernels.ARG_N_TOKENS, n);
            wsum.setGlobal(MoeKernels.elementGlobal((long) n * plan.outDim, ops.local()));
            lastN = n;
        }

        private void setRows(ShardedDispatch d, int outDim, int pairs) {
            if (d != null) {
                d.setInt(MoeKernels.ARG_BMM_N_ROWS, pairs);
                d.setGlobal(MoeKernels.batchMatMulIdGlobal(outDim, pairs, idLocal));
            }
        }

        private void setPairs(PreparedDispatch d, int dim, int pairs) {
            if (d != null) {
                d.setInt(MoeKernels.ARG_ADD_ID_N_SEL, pairs);
                d.setGlobal(MoeKernels.elementGlobal((long) pairs * dim, ops.local()));
            }
        }

        void bindScaleInput(GpuBuffer x) {
            if (x != lastScaleX) {
                scaleInput.setBuffer(ARG_SCALE_INPUT_X, x);
                lastScaleX = x;
            }
        }

        /** Input view of the gate/up projections (the weighted pair rows for weight-before-FFN). */
        void bindInput(GpuBuffer x, int xOff) {
            if (x == lastIn && xOff == lastInOff) {
                return;
            }
            if (gateUp != null) {
                gateUp.setBuffer(MoeKernels.ARG_GU_INPUT, x);
                gateUp.setInt(MoeKernels.ARG_GU_IN_OFF, xOff);
            }
            setInput(gateMm, x, xOff);
            setInput(upMm, x, xOff);
            lastIn = x;
            lastInOff = xOff;
        }

        private void setInput(ShardedDispatch d, GpuBuffer x, int xOff) {
            if (d != null) {
                int inArg = !batched ? MoeKernels.ARG_ID_INPUT : grouped ? MoeKernels.ARG_GG_INPUT
                        : MoeKernels.ARG_BMM_INPUT;
                int offArg = !batched ? MoeKernels.ARG_ID_IN_OFF : grouped ? MoeKernels.ARG_GG_IN_OFF
                        : MoeKernels.ARG_BMM_IN_OFF;
                d.setBuffer(inArg, x);
                d.setInt(offArg, xOff);
            }
        }

        void bindDst(GpuBuffer dst, int flags) {
            int f = flags | (weightBefore ? MoeKernels.WSUM_UNWEIGHTED : 0);
            if (dst != lastDst) {
                wsum.setBuffer(ARG_WSUM_DST, dst);
                lastDst = dst;
            }
            if (f != lastFlags) {
                wsum.setInt(ARG_WSUM_FLAGS, f);
                lastFlags = f;
            }
        }

        /**
         * Enqueues the path for {@code n} tokens ({@code n * k} pairs). With a {@link GpuMatmulTap} that wants an
         * expert tensor of the layer, the routed ids and the inputs of the gate/up and down products are read
         * back and reported per routed pair.
         */
        void dispatch(int n) {
            int pairs = n * k;
            if (zeroSkippedExperts) {
                // the pairs of experts another rank holds take no product here: their summand is zero
                ops.fill(expertOut.buffer(), expertOut.offset() * Float.BYTES,
                        (long) pairs * plan.outDim * Float.BYTES, 0);
            }
            if (batched) {
                rows.dispatch();
            }
            if (scaleInput != null) {
                scaleInput.dispatch();
            }
            int[] routed = tappedIds(pairs);
            if (routed != null) {
                int rows = weightBefore ? pairs : n;
                for (GpuWeight t : new GpuWeight[]{w.gate, w.up, w.gateUp}) {
                    GpuMatmulTap.experts(ctx.queue(), t, plan.nExpert, routed, k, lastIn, lastInOff, rows,
                            plan.inDim, weightBefore);
                }
            }
            boolean gateUpDeltas = gateUpDeltas();
            if (gateUp != null && !gateUpDeltas) {
                gateUp.dispatch();
            } else {
                run(gateMm);
                if (gateUpDeltas) {
                    addDeltas(plan.fused ? gateUpSite : gateSite, n, lastIn, lastInOff, !weightBefore, gate);
                }
                run(gateBias);
                run(upMm);
                if (gateUpDeltas) {
                    addDeltas(upSite, n, lastIn, lastInOff, !weightBefore, plan.gated ? up : act);
                }
                run(upBias);
                activate(pairs);
            }
            if (routed != null) {
                GpuMatmulTap.experts(ctx.queue(), w.down, plan.nExpert, routed, k, act.buffer(), act.offset(),
                        pairs, plan.nFf, true);
            }
            down.dispatch();
            addDeltas(downSite, n, act.buffer(), act.offset(), false, expertOut);
            run(downBias);
            if (postScale != null) {
                postScale.dispatch();
                ops.norm(NormKind.RMS, null, null, postEps, expertOut.buffer(), expertOut.offset(),
                        expertOut.buffer(), expertOut.offset(), pairs, plan.outDim);
            }
            wsum.dispatch();
        }

        /** The routed expert of every pair when the tap wants an expert tensor of the layer, else null. */
        private int[] tappedIds(int pairs) {
            if (!GpuMatmulTap.wantsExperts(w.gate) && !GpuMatmulTap.wantsExperts(w.up)
                    && !GpuMatmulTap.wantsExperts(w.gateUp) && !GpuMatmulTap.wantsExperts(w.down)) {
                return null;
            }
            return GpuMatmulTap.readInts(ctx.queue(), src.ids.buffer(), src.ids.offset(), pairs);
        }

        /** {@code LLM_GLU} or {@code LLM_Unary} of the projected rows into {@link #act}. */
        private void activate(int pairs) {
            int nFf = plan.nFf;
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

        /** The deltas of an expert product over this step's pairs, after its per-expert scale. */
        private void addDeltas(GpuProjectionDeltas.ExpertSite site, int n, GpuBuffer x, int xOff,
                               boolean tokenRows, ScratchSlot out) {
            if (site != null && !site.isEmpty()) {
                site.apply(n, k, src.expertIds(), x, xOff, tokenRows, out.buffer(), out.offset());
            }
        }

        private void run(ShardedDispatch d) {
            if (d != null) {
                d.dispatch();
            }
        }

        @Override
        public void close() {
            closeAll(scaleInput, gateBias, upBias, downBias, postScale, wsum, gateUp, gateMm, upMm, down);
        }
    }

    /**
     * Interleaved {@code [gate_s[e], up_s[e]]} pairs for {@code {T}_MatVecIdGateUp} {@code has_scale} 2, 1 where
     * a tensor has no scale; built once into a per-layer temporary.
     */
    private GpuBuffer scalePairs() {
        int n = plan.nExpert;
        ScratchSlot pairs = ctx.scratch().temp(T_SCALE_PAIRS + w.gate.name(), 2L * n);
        ops.fill(pairs.buffer(), 4L * pairs.offset(), 8L * n, Float.floatToRawIntBits(1.0f));
        if (w.gateScale != null) {
            ops.copy2d(w.gateScale.shard(0), 0, 1, pairs.buffer(), pairs.offset(), 2, n, 1);
        }
        if (w.upScale != null) {
            ops.copy2d(w.upScale.shard(0), 0, 1, pairs.buffer(), pairs.offset() + 1, 2, n, 1);
        }
        return pairs.buffer();
    }

    private GpuBuffer buf(GpuWeight t, GpuBuffer placeholder) {
        return t != null ? t.shard(0) : placeholder;
    }

    private long rows(long width) {
        return Math.multiplyExact((long) ubatch, width);
    }

    private long[] local1() {
        return new long[]{ops.local()};
    }

    private void f32Vector(GpuWeight v, int n) {
        if (v.type() != GGMLType.F32 || !v.isSingleShard() || v.rows() * v.cols() != n) {
            throw new IllegalStateException("layer " + layer.il() + ": " + v.name() + " must be one F32 buffer of "
                    + n + " values, got " + v);
        }
    }

    /** Closes every non-null resource, then throws the first failure with the later ones suppressed. */
    private static void closeAll(AutoCloseable... resources) {
        RuntimeException failure = Closeables.closeAll(Arrays.asList(resources), null);
        if (failure != null) {
            throw failure;
        }
    }

    /** Largest power of two not above {@code v} ({@code v >= 1}). */
    private static int floorPow2(int v) {
        return Integer.highestOneBit(v);
    }

    /** Smallest power of two not below {@code v} ({@code v >= 1}). */
    private static int ceilPow2(int v) {
        return v <= 1 ? 1 : Integer.highestOneBit(v - 1) << 1;
    }

    // ================================================================== expert shards

    /**
     * Prepares one dispatch per expert shard ({@code part.apply(i)} for shard {@code i}); a failure closes the parts
     * already prepared before it propagates.
     */
    private static ShardedDispatch prepareShards(int count, IntFunction<PreparedDispatch> part) {
        if (count < 1) {
            throw new IllegalArgumentException("an expert product needs at least one shard, got " + count);
        }
        PreparedDispatch[] parts = new PreparedDispatch[count];
        try {
            for (int i = 0; i < count; i++) {
                parts[i] = part.apply(i);
            }
        } catch (RuntimeException e) {
            throw Closeables.closeAll(Arrays.asList(parts), e);
        }
        return new ShardedDispatch(parts);
    }

    /**
     * One expert product as one prepared dispatch per expert shard of its tensor, in shard order. The parts differ
     * only in their weight buffers and expert ranges, write disjoint output rows and share every per-step argument,
     * which the setters apply to all of them.
     */
    private static final class ShardedDispatch implements AutoCloseable {
        private final PreparedDispatch[] parts;

        ShardedDispatch(PreparedDispatch[] parts) {
            this.parts = parts;
        }

        void setInt(int argIndex, int v) {
            for (PreparedDispatch d : parts) {
                d.setInt(argIndex, v);
            }
        }

        void setBuffer(int argIndex, GpuBuffer b) {
            for (PreparedDispatch d : parts) {
                d.setBuffer(argIndex, b);
            }
        }

        void setGlobal(long[] global) {
            for (PreparedDispatch d : parts) {
                d.setGlobal(global);
            }
        }

        void dispatch() {
            for (PreparedDispatch d : parts) {
                d.dispatch();
            }
        }

        @Override
        public void close() {
            closeAll(parts);
        }
    }
}
