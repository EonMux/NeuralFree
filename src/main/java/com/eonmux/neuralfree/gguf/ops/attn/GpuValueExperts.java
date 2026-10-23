package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeRouting;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The MoVA value projection of a k2-horizon attention layer on the GPU, {@code build_routed_value}
 * (models/k2-horizon.cpp:137-205), all on the context queue:
 * <ol>
 *   <li>{@code logits = attn_v_gate . x} ({@link GpuOps#matmul}) plus its LoRA deltas ({@code build_lora_mm});</li>
 *   <li>{@code LLM_MoE_Router}: softmax or sigmoid probabilities, top-k selection on {@code probs + attn_v_gate.bias}
 *       (the bias selects only), weights {@code probs[id]}, renormalized with the sum clamped to
 *       {@link MoeSpec#NORM_SUM_MIN} when {@code expert_weights_norm}, scaled when {@code expert_weights_scale} is
 *       not 0 or 1;</li>
 *   <li>the selected experts' products {@code attn_v_exps[id] . x}: {@code {T}_MatVecId} with a broadcast input row
 *       for one token, {@code LLM_MoE_Bucketize} + {@code {T}_BatchMatMulId} (token rows, {@code x_div = k}) for
 *       several, one dispatch per expert shard of the tensor; then their LoRA deltas ({@code build_lora_mm_id});</li>
 *   <li>{@code silu} ({@code LLM_Unary}) and {@code LLM_MoE_WeightedSum}, which adds the weighted experts in
 *       selection order into the V rows.</li>
 * </ol>
 * Every per-step value is re-set on dispatches prepared at bind. Temporaries are the {@code attn.vexp.*}
 * {@link GpuScratch} buffers sized for {@code ubatch}. Not thread-safe.
 */
final class GpuValueExperts implements AutoCloseable {

    private static final String T_LOGITS = "attn.vexp.logits";
    private static final String T_IDS = "attn.vexp.ids";
    private static final String T_WEIGHTS = "attn.vexp.weights";
    private static final String T_OUT = "attn.vexp.out";
    private static final String T_COUNTS = "attn.vexp.counts";
    private static final String T_OFFSETS = "attn.vexp.offsets";
    private static final String T_ROW_TOKEN = "attn.vexp.row_token";
    private static final String T_ROW_SLOT = "attn.vexp.row_slot";

    private final GpuOps ops;
    private final int layer;
    /** Input width ({@code attn_v_gate.ne[0]}); the mixer checks it against the Q projection's. */
    private final int nEmbd;
    private final int vDim;
    private final int nExpert;
    private final int k;
    private final int idLocal;
    private final int routerLocal;
    private final GpuWeight router;
    private final GpuProjectionDeltas.Site routerDeltas;
    private final GpuProjectionDeltas.ExpertSite expertDeltas;
    private final GpuBuffer logits;
    private final GpuBuffer ids;
    private final GpuBuffer weights;
    private final GpuBuffer expertOut;
    private final List<PreparedDispatch> prepared = new ArrayList<>();
    private final PreparedDispatch select;
    private final PreparedDispatch bucket;
    private final PreparedDispatch[] decode;
    private final PreparedDispatch[] batch;
    private final PreparedDispatch wsum;
    private int lastN = 1;
    private int lastXOff;
    private GpuBuffer lastX;

    /**
     * Prepares the projection of {@code layer} into the dense V rows {@code v} ({@code [ubatch][vDim]} from element 0).
     *
     * @param gateBias {@code attn_v_gate.bias} as one F32 buffer of {@code n_value_expert} values; required when the
     *                 spec's router has a selection bias, else null
     * @throws IllegalStateException         when a tensor's shape does not match the spec
     * @throws UnsupportedOperationException when the expert tensor's type has no expert kernels, or the grid
     *                                       exceeds the device's work-group counts
     * @throws IllegalArgumentException      when the routing exceeds the router kernel's limits
     */
    GpuValueExperts(GpuForwardContext ctx, LayerSpec layer, QkvProjection proj, MoeSpec spec, GpuBuffer gateBias,
                    GpuBuffer v) {
        this.ops = ctx.ops();
        this.layer = layer.il();
        this.vDim = spec.nFfExp();
        this.nExpert = spec.nExpert();
        this.k = spec.nUsed();
        GpuWeights w = ctx.weightSource().weightsFor(layer.weightLayer());
        this.router = w.layer(layer, proj.vGateWeight());
        GpuWeight exps = w.layer(layer, proj.vExpsWeight());
        this.nEmbd = Math.toIntExact(router.cols());
        checkTensors(exps);
        if (spec.probBias() != (gateBias != null)) {
            throw new IllegalStateException("layer " + this.layer + ": the value router's selection bias "
                    + (gateBias == null ? "is missing" : "is not part of the spec"));
        }
        this.routerDeltas = ctx.projectionDeltas().serve(layer, proj.vGateWeight(), router);
        this.expertDeltas = ctx.projectionDeltas().serveExperts(layer, proj.vExpsWeight(), exps);
        MoeRouting.Params params = MoeRouting.Params.forExperts(spec);
        this.routerLocal = routerLocal(nExpert);
        MoeKernels.checkRouter(params, routerLocal);
        this.idLocal = idLocal(exps);
        MoeKernels.checkIdLocal(exps.type(), idLocal);
        int ubatch = ctx.scratch().ubatch();
        checkGrid(ctx, ubatch);
        GpuScratch scratch = ctx.scratch();
        long pairs = (long) ubatch * k;
        this.logits = scratch.temp(T_LOGITS, (long) ubatch * nExpert).buffer();
        this.ids = scratch.temp(T_IDS, pairs).buffer();
        this.weights = scratch.temp(T_WEIGHTS, pairs).buffer();
        this.expertOut = scratch.temp(T_OUT, pairs * vDim).buffer();
        GpuBuffer rowToken = scratch.temp(T_ROW_TOKEN, pairs).buffer();
        GpuBuffer rowSlot = scratch.temp(T_ROW_SLOT, pairs).buffer();
        GpuBuffer ph = ctx.placeholders().readOnlyFloats();
        try {
            select = prepare(MoeKernels.ROUTER, MoeKernels.routerGlobal(1, routerLocal), new long[]{routerLocal},
                    MoeKernels.routerArgs(params, 1, logits, gateBias != null ? gateBias : ph, ids, weights));
            long[] group = {Integer.highestOneBit(ops.local())};
            bucket = prepare(MoeKernels.BUCKETIZE, group, group, MoeKernels.bucketizeArgs(1, k, nExpert, ids,
                    scratch.temp(T_COUNTS, nExpert).buffer(), scratch.temp(T_OFFSETS, nExpert).buffer(), rowToken,
                    rowSlot));
            int shards = exps.shardCount();
            decode = new PreparedDispatch[shards];
            batch = new PreparedDispatch[shards];
            long[] local2 = {idLocal, 1};
            for (int i = 0; i < shards; i++) {
                int base = expertOf(exps, exps.shardRowStart(i), i);
                int count = expertOf(exps, exps.shardRows(i), i);
                decode[i] = prepare(MoeKernels.matVecId(exps.type()), MoeKernels.idMatVecGlobal(vDim, k, idLocal),
                        local2, MoeKernels.matVecIdArgs(vDim, nEmbd, k, 0, 0, 0, false, base, count, ids,
                                exps.shard(i), ph, ph, expertOut));
                batch[i] = prepare(MoeKernels.batchMatMulId(exps.type()),
                        MoeKernels.batchMatMulIdGlobal(vDim, k, idLocal), local2,
                        MoeKernels.batchMatMulIdArgs(vDim, nEmbd, k, k, 0, false, base, count, rowToken, rowSlot,
                                exps.shard(i), ph, ph, expertOut));
            }
            wsum = prepare(MoeKernels.WEIGHTED_SUM, MoeKernels.elementGlobal(vDim, ops.local()),
                    new long[]{ops.local()}, MoeKernels.weightedSumArgs(1, vDim, k, 0, expertOut, weights,
                            Objects.requireNonNull(v, "v")));
        } catch (RuntimeException e) {
            close();
            throw e;
        }
    }

    private void checkTensors(GpuWeight exps) {
        if (router.rank() != 2 || router.rows() != nExpert || router.hasScale()) {
            throw new IllegalStateException("layer " + layer + ": " + router.name() + " must be [n_embd, " + nExpert
                    + "] without an output scale (build_lora_mm passes none), got " + router);
        }
        long[] ne = exps.ne();
        if (ne.length != 3 || ne[0] != nEmbd || ne[1] != vDim || ne[2] != nExpert) {
            throw new IllegalStateException("layer " + layer + ": " + exps.name() + " has shape " + Arrays.toString(ne)
                    + ", expected [" + nEmbd + ", " + vDim + ", " + nExpert + "]");
        }
        if (!MoeKernels.supports(exps.type())) {
            throw new UnsupportedOperationException("layer " + layer + ": " + exps.name() + " has type "
                    + exps.type() + ", which has no expert matvec kernels");
        }
        if (exps.hasScale() || exps.hadamard() != null) {
            throw new IllegalStateException("layer " + layer + ": " + exps.name() + " carries a per-tensor scale or"
                    + " Hadamard transform, which build_lora_mm_id does not apply to value experts");
        }
    }

    /** Whole experts of {@code rows} expert-tensor rows of shard {@code shard}. */
    private int expertOf(GpuWeight exps, long rows, int shard) {
        if (rows % vDim != 0) {
            throw new IllegalStateException("layer " + layer + ": " + exps.name() + " shard " + shard + " splits an"
                    + " expert of " + vDim + " rows at row " + rows);
        }
        return Math.toIntExact(rows / vDim);
    }

    /** Router work-group: a power of two in {@code [32, 256]} with at most 32 experts per work-item. */
    private int routerLocal(int experts) {
        int needed = ceilPow2((experts + 31) / 32);
        int preferred = Integer.highestOneBit(Math.max(ops.local(), MoeKernels.MIN_ROUTER_LOCAL));
        return Math.min(MoeKernels.MAX_LOCAL, Math.max(needed, preferred));
    }

    /** Id-matvec work-group: the row-reduce size raised to the type's threads per block, in {@code [32, 256]}. */
    private int idLocal(GpuWeight exps) {
        int l = Math.max(32, Integer.highestOneBit(ops.local()));
        l = Math.max(l, ceilPow2(MoeKernels.threadsPerBlock(exps.type())));
        return Math.min(l, MoeKernels.MAX_LOCAL);
    }

    private void checkGrid(GpuForwardContext ctx, int ubatch) {
        long[] max = ctx.kernels().caps().getMaxWorkGroupCount();
        long tiles = ((long) ubatch * k + MoeKernels.BMM_ROWS - 1) / MoeKernels.BMM_ROWS;
        if (max.length >= 2 && ((max[0] > 0 && vDim > max[0]) || (max[1] > 0 && Math.max(tiles, k) > max[1]))) {
            throw new UnsupportedOperationException("layer " + layer + ": value experts of " + vDim + " rows over "
                    + tiles + " row tiles at ubatch " + ubatch + " exceed the device's work-group counts " + max[0]
                    + " x " + max[1] + "; lower neuralfree.ubatch");
        }
    }

    private static int ceilPow2(int v) {
        return v <= 1 ? 1 : Integer.highestOneBit(v - 1) << 1;
    }

    private PreparedDispatch prepare(String kernel, long[] global, long[] local, Object[] args) {
        PreparedDispatch d = ops.prepare(kernel, global, local, args);
        prepared.add(d);
        return d;
    }

    /**
     * Writes V of {@code n} tokens, whose input rows of {@code n_embd} values start at {@code x[xOff]}, to the V rows
     * bound at construction.
     */
    void run(int n, GpuBuffer x, int xOff) {
        ops.matmul(router, x, xOff, logits, 0, n);
        routerDeltas.applyRows(n, x, xOff, logits, 0);
        if (n != lastN) {
            select.setInt(MoeKernels.ARG_N_TOKENS, n);
            select.setGlobal(MoeKernels.routerGlobal(n, routerLocal));
            bucket.setInt(MoeKernels.ARG_N_TOKENS, n);
            for (PreparedDispatch d : batch) {
                d.setInt(MoeKernels.ARG_BMM_N_ROWS, n * k);
                d.setGlobal(MoeKernels.batchMatMulIdGlobal(vDim, n * k, idLocal));
            }
            wsum.setInt(MoeKernels.ARG_N_TOKENS, n);
            wsum.setGlobal(MoeKernels.elementGlobal((long) n * vDim, ops.local()));
            lastN = n;
        }
        if (x != lastX || xOff != lastXOff) {
            for (PreparedDispatch d : decode) {
                d.setBuffer(MoeKernels.ARG_ID_INPUT, x);
                d.setInt(MoeKernels.ARG_ID_IN_OFF, xOff);
            }
            for (PreparedDispatch d : batch) {
                d.setBuffer(MoeKernels.ARG_BMM_INPUT, x);
                d.setInt(MoeKernels.ARG_BMM_IN_OFF, xOff);
            }
            lastX = x;
            lastXOff = xOff;
        }
        select.dispatch();
        if (n == 1) {
            for (PreparedDispatch d : decode) {
                d.dispatch();
            }
        } else {
            bucket.dispatch();
            for (PreparedDispatch d : batch) {
                d.dispatch();
            }
        }
        if (!expertDeltas.isEmpty()) {
            expertDeltas.apply(n, k, ids, x, xOff, true, expertOut, 0);
        }
        ops.unary(ActOp.SILU, ActParams.NONE, expertOut, 0, expertOut, 0, n * k * vDim);
        wsum.dispatch();
    }

    @Override
    public void close() {
        RuntimeException failure = Closeables.closeAll(prepared, null);
        prepared.clear();
        if (failure != null) {
            throw failure;
        }
    }

    /** Values per input row. */
    int inputWidth() {
        return nEmbd;
    }

    @Override
    public String toString() {
        return "GpuValueExperts{layer " + layer + ", " + k + " of " + nExpert + " experts of " + vDim + '}';
    }
}
