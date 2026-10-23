package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * CPU twin of {@link GpuMoeFfn}: the same steps in the same order with the host MoE math ({@link MoeRouting},
 * {@link MoeCombine}), expert products grouped by expert ({@link CpuExpertMatmul}, the twin of
 * {@code ggml_mul_mat_id}: each selected expert's weights are streamed once per step for all of its pairs, and the
 * per-expert {@code <exps>.scale} of separate tensors is applied through the {@link CpuWeights} transforms of
 * {@link CpuMatVec}) and {@link CpuOps} norms and activations. Rows are row-major {@code [t][n_embd]} from index 0;
 * working arrays are {@link CpuScratch} temporaries sized for {@code ubatch} at bind.
 *
 * <p>Expert tensors may be row-sharded {@link CpuTensorRef}s (mapped regions cut on any row boundary, so an expert
 * may span two shards): the grouped products read expert {@code e}'s rows {@code [e * rows, (e + 1) * rows)} through
 * the vec-dots, which follow the shards that hold them, and the hash table is read row by row through
 * {@link CpuTensorRef#rowBuffer}. A sharded tensor therefore gives the same products as one buffer, the twin of the
 * per-shard {@code expert_base, expert_count} dispatches of {@link GpuMoeFfn}.</p>
 *
 * <p>Every weight product adds the {@link CpuProjectionDeltas} registered on its tensor (LoRA) right after the
 * product and before any bias: {@code build_lora_mm} for the router, latent, shared-gate and dense products, with a
 * dense FFN's {@code w_s} applied after the delta as {@code build_ffn} does (llama-graph.cpp:1979-2010), and
 * {@code build_lora_mm_id} per selected pair for the expert products after their per-expert scale
 * (llama-graph.cpp:1706-1749).</p>
 */
public final class CpuMoeFfn implements CpuFfnBlock {

    private static final String T_LOGITS = "moe.logits";
    private static final String T_IDS = "moe.ids";
    private static final String T_WEIGHTS = "moe.weights";
    private static final String T_ROUTER_IN = "moe.router_in";
    private static final String T_SELECTION = "moe.selection";
    private static final String T_MOE_IN = "moe.moe_in";
    private static final String T_LATENT_IN = "moe.latent_in";
    private static final String T_LATENT_SUM = "moe.latent_sum";
    private static final String T_GATE = "moe.gate";
    private static final String T_UP = "moe.up";
    private static final String T_ACT = "moe.act";
    private static final String T_EXPERT_OUT = "moe.expert_out";
    private static final String T_CHUNK_OUT = "moe.chunk_out";
    private static final String T_FFN_GATE = "moe.ffn_gate";
    private static final String T_FFN_UP = "moe.ffn_up";
    private static final String T_FFN_OUT = "moe.ffn_out";
    private static final String T_SHEXP_GATE = "moe.shexp_gate";
    /** grok-2 parallel combine factor {@code std::sqrt(2) / 2} (grok.cpp:178). */
    private static final float GROK_COMBINE = (float) (Math.sqrt(2.0) / 2.0);

    private final FfnKind kind;
    private final LayerSpec layer;
    private final MoeSpec spec;

    private CpuForwardContext ctx;
    private CpuMatVec mv;
    private MoeBlocks.Wiring wiring;
    private int ubatch;
    private int nEmbd;
    private float rmsEps;

    private CpuTensorRef gateInp;
    private CpuProjectionDeltas.Site gateInpDeltas;
    private CpuProjectionDeltas.Site latentDownDeltas;
    private CpuProjectionDeltas.Site latentUpDeltas;
    private CpuProjectionDeltas.Site sharedGateInpDeltas;
    private float[] routerBias;
    private float[] probBias;
    private float[] routerScale;
    private int[] tid2eid;
    private float[] logits;
    private float[] routerIn;
    private float[] layerInput;
    private MoeRouting.Params routing;
    private MoeRouting.Params selectionRouting;
    private float[] selection;
    private MoeRouting.Params chunkRouting;

    private float[] expertNorm;
    private float[] moeIn;
    private CpuTensorRef latentDown;
    private CpuTensorRef latentUp;
    private float[] preExpNorm;
    private float[] routedNorm;
    private float[] latentIn;
    private float[] latentSum;

    private ExpertPass routed;
    private ExpertPass chunk;
    private float[] chunkOut;

    private DenseFfn shared;
    private CpuTensorRef sharedGateInp;
    private float[] sharedGate;
    private DenseFfn parallel;
    private float[] postNorm1;
    private float[] postNorm2;

    /**
     * An unbound block for a layer of {@code kind}.
     *
     * @throws IllegalArgumentException when the layer's spec does not match the kind
     */
    public CpuMoeFfn(FfnKind kind, LayerSpec layer) {
        this.kind = Objects.requireNonNull(kind, "kind");
        this.layer = layer;
        this.spec = MoeBlocks.checkLayer(kind, layer);
    }

    // ================================================================== bind

    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        if (this.ctx != null) {
            throw new IllegalStateException("MoE block of layer " + this.layer.il() + " is already bound");
        }
        if (layer != this.layer) {
            throw new IllegalArgumentException("block created for layer " + this.layer.il() + " bound to layer "
                    + layer.il());
        }
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.mv = ctx.matVec();
        CpuScratch scratch = ctx.scratch();
        CpuWeights w = ctx.weights();
        this.ubatch = scratch.ubatch();
        this.rmsEps = ctx.spec().hparams().normRmsEps();
        this.wiring = MoeBlocks.Wiring.of(ctx.spec().arch(), kind, layer);

        gateInp = w.layer(layer, TensorNames.weight(TensorNames.FFN_GATE_INP));
        TensorRef gi = w.describe(gateInp);
        if (gi.rank() != 2 || gi.ne(1) != spec.nExpert()) {
            throw new IllegalStateException("layer " + layer.il() + ": " + gi.name() + " must be [n_embd, "
                    + spec.nExpert() + "], got " + gi);
        }
        nEmbd = Math.toIntExact(gi.ne(0));
        gateInpDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.FFN_GATE_INP), gateInp, mv);
        bindRouter(w, scratch);
        bindExpertInput(w, scratch);
        int expertDim = spec.latent() ? spec.latentSize() : nEmbd;
        routed = new ExpertPass(MoeExpertWeights.routed(spec, expertDim), w, routing.k(), spec.weightBeforeFfn(),
                spec.postExpertNorm() ? vector(w, TensorNames.weight(TensorNames.FFN_POST_EXP_NORM), expertDim) : null,
                ctx.spec().hparams().postNormEps());
        if (spec.chunked()) {
            chunkRouting = MoeRouting.Params.forChunks(spec);
            chunk = new ExpertPass(MoeExpertWeights.chunks(spec, nEmbd), w, chunkRouting.k(), false, null, 0f);
            chunkOut = scratch.temp(T_CHUNK_OUT, rows(nEmbd));
        }
        if (spec.sharedExpert() != null) {
            shared = new DenseFfn(spec.sharedExpert(), w);
            if (spec.sharedExpertGated()) {
                sharedGateInp = w.layer(layer, TensorNames.weight(TensorNames.FFN_GATE_INP_SHEXP));
                if (sharedGateInp.rows() != 1 || sharedGateInp.cols() != nEmbd) {
                    throw new IllegalStateException("layer " + layer.il() + ": "
                            + layer.tensorName(TensorNames.weight(TensorNames.FFN_GATE_INP_SHEXP))
                            + " must hold one row of " + nEmbd + ", got " + sharedGateInp);
                }
                sharedGateInpDeltas = ctx.projectionDeltas().serve(layer,
                        TensorNames.weight(TensorNames.FFN_GATE_INP_SHEXP), sharedGateInp, mv);
                sharedGate = scratch.temp(T_SHEXP_GATE, ubatch);
            }
        }
        if (kind == FfnKind.MOE_PARALLEL_DENSE) {
            parallel = new DenseFfn(layer.ffn(), w);
            if (wiring.combine == MoeBlocks.Combine.NORMED_SUM) {
                if (shared != null || chunk != null) {
                    throw new UnsupportedOperationException("layer " + layer.il() + ": the normed parallel combine"
                            + " of gemma4 has no shared or chunk experts in llama.cpp");
                }
                postNorm1 = vector(w, TensorNames.weight(TensorNames.FFN_POST_NORM_1), nEmbd);
                postNorm2 = vector(w, TensorNames.weight(TensorNames.FFN_POST_NORM_2), nEmbd);
            }
        }
    }

    private void bindRouter(CpuWeights w, CpuScratch scratch) {
        logits = scratch.temp(T_LOGITS, rows(spec.nExpert()));
        if (wiring.selection == MoeBlocks.Selection.BIASED_LOGITS) {
            routing = MoeRouting.Params.preselected(spec);
            selectionRouting = new MoeRouting.Params(spec.nExpert(), spec.nUsed(), spec.gating(), 1, 1, false, 1.0f,
                    false, MoeKernels.SEL_RAW_LOGITS, 0);
            selection = scratch.temp(T_SELECTION, rows(spec.nExpert()));
        } else {
            routing = MoeRouting.Params.forExperts(spec);
        }
        if (spec.routerBias()) {
            routerBias = vector(w, TensorNames.bias(TensorNames.FFN_GATE_INP), spec.nExpert());
        }
        if (spec.probBias()) {
            float[] b = w.layerFloatsOrNull(layer, TensorNames.bias(TensorNames.FFN_EXP_PROBS_B));
            probBias = b != null ? checkLength(b, spec.nExpert(), TensorNames.bias(TensorNames.FFN_EXP_PROBS_B))
                    : vector(w, TensorNames.FFN_EXP_PROBS_B, spec.nExpert());
        }
        if (wiring.router == MoeBlocks.RouterInput.SCALED_RESIDUAL) {
            routerIn = scratch.temp(T_ROUTER_IN, rows(nEmbd));
            routerScale = vector(w, TensorNames.scale(TensorNames.FFN_GATE_INP), nEmbd);
        }
        if (wiring.needsLayerInput()) {
            if (layer.residual().streamWidth() != nEmbd) {
                throw new IllegalStateException("layer " + layer.il() + ": the layer input has "
                        + layer.residual().streamWidth() + " columns, the MoE reads " + nEmbd);
            }
            layerInput = scratch.temp(MoeBlocks.LAYER_INPUT_TEMP,
                    Math.toIntExact(MoeBlocks.layerInputFloats(ubatch, ctx.spec())));
        }
        if (spec.hashRouting()) {
            tid2eid = readHashTable(w);
        }
    }

    /** {@code ffn_gate_tid2eid} ({@code [k, n_vocab]} I32), validated once (deepseek4.cpp:1297-1299). */
    private int[] readHashTable(CpuWeights w) {
        CpuTensorRef table = w.layer(layer, TensorNames.weight(TensorNames.FFN_GATE_TID2EID));
        TensorRef ref = w.describe(table);
        int k = routing.k();
        if (ref.type() != GGMLType.I32 || ref.ne(0) != k) {
            throw new IllegalStateException("layer " + layer.il() + ": " + ref.name() + " must be I32 [" + k
                    + ", n_vocab], got " + ref);
        }
        int nVocab = Math.toIntExact(table.rows());
        int[] ids = new int[Math.multiplyExact(nVocab, k)];
        for (int r = 0; r < nVocab; r++) {
            ByteBuffer b = table.rowBuffer(r).duplicate().order(ByteOrder.LITTLE_ENDIAN);
            int off = table.rowOffset(r);
            for (int j = 0; j < k; j++) {
                ids[r * k + j] = b.getInt(off + 4 * j);
            }
        }
        MoeCombine.validateHashTable(ids, k, nVocab, spec.nExpert());
        return ids;
    }

    private void bindExpertInput(CpuWeights w, CpuScratch scratch) {
        if (wiring.experts == MoeBlocks.ExpertInput.NORM_EXPS_OF_LAYER_INPUT) {
            expertNorm = vector(w, TensorNames.weight(TensorNames.FFN_NORM_EXPS), nEmbd);
        } else if (wiring.experts == MoeBlocks.ExpertInput.PRE_NORM_2_OF_RESIDUAL) {
            expertNorm = vector(w, TensorNames.weight(TensorNames.FFN_PRE_NORM_2), nEmbd);
        }
        if (expertNorm != null) {
            moeIn = scratch.temp(T_MOE_IN, rows(nEmbd));
        }
        if (!spec.latent()) {
            return;
        }
        String down = TensorNames.weight(TensorNames.FFN_LATENT_DOWN);
        String up = TensorNames.weight(TensorNames.FFN_LATENT_UP);
        if (w.layerOrNull(layer, down) == null) {
            down = TensorNames.weight(TensorNames.FFN_ROUTED_DOWN);
            up = TensorNames.weight(TensorNames.FFN_ROUTED_UP);
        }
        latentDown = w.layer(layer, down);
        latentUp = w.layer(layer, up);
        int latent = spec.latentSize();
        if (latentDown.cols() != nEmbd || latentDown.rows() != latent || latentUp.cols() != latent
                || latentUp.rows() != nEmbd) {
            throw new IllegalStateException("layer " + layer.il() + ": latent projections " + latentDown + " / "
                    + latentUp + " do not map " + nEmbd + " <-> " + latent);
        }
        latentDownDeltas = ctx.projectionDeltas().serve(layer, down, latentDown, mv);
        latentUpDeltas = ctx.projectionDeltas().serve(layer, up, latentUp, mv);
        float[] pre = w.layerFloatsOrNull(layer, TensorNames.weight(TensorNames.FFN_PRE_EXP_NORM));
        preExpNorm = pre != null ? checkLength(pre, latent, TensorNames.weight(TensorNames.FFN_PRE_EXP_NORM)) : null;
        float[] rn = w.layerFloatsOrNull(layer, TensorNames.weight(TensorNames.FFN_ROUTED_NORM));
        routedNorm = rn != null ? checkLength(rn, latent, TensorNames.weight(TensorNames.FFN_ROUTED_NORM)) : null;
        latentIn = scratch.temp(T_LATENT_IN, rows(latent));
        latentSum = scratch.temp(T_LATENT_SUM, rows(latent));
    }

    // ================================================================== forward

    @Override
    public void forward(int n, float[] in, float[] out) {
        if (ctx == null) {
            throw new IllegalStateException("MoE block of layer " + layer.il() + " used before bind");
        }
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("token count " + n + " outside [1, " + ubatch + "]");
        }
        int nx = Math.multiplyExact(n, nEmbd);
        if (in.length < nx || out.length < nx) {
            throw new IllegalArgumentException("MoE rows need " + nx + " values, got in " + in.length + ", out "
                    + out.length);
        }
        float[] x = in;
        if (expertNorm != null) {
            float[] src = wiring.experts == MoeBlocks.ExpertInput.NORM_EXPS_OF_LAYER_INPUT ? layerInput : residual();
            CpuOps.rmsNorm(src, 0, moeIn, 0, n, nEmbd, expertNorm, rmsEps);
            x = moeIn;
        }
        MoeRouting.Selection sel = route(n, in, x);
        if (spec.latent()) {
            int latent = spec.latentSize();
            mv.matmul(latentDown, x, 0, latentIn, 0, n);
            latentDownDeltas.applyRows(n, x, 0, latentIn, 0);
            if (preExpNorm != null) {
                CpuOps.rmsNorm(latentIn, 0, latentIn, 0, n, latent, preExpNorm, rmsEps);
            }
            routed.run(n, latentIn, sel, latentSum, 0);
            if (routedNorm != null) {
                CpuOps.rmsNorm(latentSum, 0, latentSum, 0, n, latent, routedNorm, rmsEps);
            }
            mv.matmul(latentUp, latentSum, 0, out, 0, n);
            latentUpDeltas.applyRows(n, latentSum, 0, out, 0);
        } else {
            routed.run(n, x, sel, out, 0);
        }
        if (chunk != null) {
            MoeRouting.Selection chunkSel = MoeRouting.route(chunkRouting, logits, n, null);
            chunk.run(n, out, chunkSel, chunkOut, 0);
            CpuOps.axpy(spec.chunkScale(), chunkOut, 0, out, 0, nx);
        }
        if (shared != null) {
            addShared(n, in, out);
        }
        if (parallel != null) {
            combineParallel(n, in, out);
        }
    }

    /**
     * Router logits, then the expert selection of this step's tokens: hash ids, ids picked by
     * {@code logits + exp_probs_b} ({@link MoeBlocks.Selection#BIASED_LOGITS}), or the spec's routing; {@code expertIn}
     * is the experts' input.
     */
    private MoeRouting.Selection route(int n, float[] in, float[] expertIn) {
        switch (wiring.router) {
            case FFN_INPUT:
                routerLogits(n, in);
                break;
            case EXPERT_INPUT:
                routerLogits(n, expertIn);
                break;
            case LAYER_INPUT:
                routerLogits(n, layerInput);
                break;
            case SCALED_RESIDUAL:
                CpuOps.rmsNorm(residual(), 0, routerIn, 0, n, nEmbd, null, rmsEps);
                CpuOps.scale(routerIn, 0, n * nEmbd, 1.0f / (float) Math.sqrt(nEmbd));
                NormMath.mulRowScale(n, nEmbd, routerScale, 0, nEmbd, routerIn, 0);
                routerLogits(n, routerIn);
                break;
            default:
                throw new IllegalStateException("unhandled router input " + wiring.router);
        }
        if (routerBias != null) {
            CpuOps.addRowBias(logits, 0, n, spec.nExpert(), routerBias, 0);
        }
        if (tid2eid != null) {
            int[] ids = MoeCombine.hashIds(ctx.scratch().ints(CpuScratch.TOKEN_IDS), n, routing.k(), tid2eid);
            return MoeRouting.routePreselected(routing, logits, n, ids);
        }
        if (selection != null) {
            int ne = n * spec.nExpert();
            System.arraycopy(logits, 0, selection, 0, ne);
            CpuOps.addRowBias(selection, 0, n, spec.nExpert(), probBias, 0);
            int[] ids = MoeRouting.route(selectionRouting, selection, n, null).ids();
            return MoeRouting.routePreselected(routing, logits, n, ids);
        }
        return MoeRouting.route(routing, logits, n, probBias);
    }

    /**
     * {@code out += shared(x)}, plain or gated by {@code sigmoid(ffn_gate_inp_shexp . x)}, then
     * {@code out *= sharedCombineScale} when that scale is not 1 ({@code ggml_add} then {@code ggml_scale},
     * cohere2moe.cpp:242-243 and the MTP block at :396-397).
     */
    private void addShared(int n, float[] in, float[] out) {
        int nx = n * nEmbd;
        shared.run(n, in);
        if (sharedGate == null) {
            CpuOps.add(out, 0, shared.out, 0, out, 0, nx);
        } else {
            mv.matmul(sharedGateInp, in, 0, sharedGate, 0, n);
            sharedGateInpDeltas.applyRows(n, in, 0, sharedGate, 0);
            CpuOps.unary(ActOp.SIGMOID, ActParams.NONE, sharedGate, 0, sharedGate, 0, n);
            MoeCombine.weightedSum(n, nEmbd, 1, MoeKernels.WSUM_ACCUMULATE, shared.out, sharedGate, out);
        }
        float combine = spec.sharedCombineScale();
        if (combine != 1.0f) {
            CpuOps.scale(out, 0, nx, combine);
        }
    }

    /** {@code logits = ffn_gate_inp . x} plus its deltas ({@code build_lora_mm}, llama-graph.cpp:2223). */
    private void routerLogits(int n, float[] x) {
        mv.matmul(gateInp, x, 0, logits, 0, n);
        gateInpDeltas.applyRows(n, x, 0, logits, 0);
    }

    private void combineParallel(int n, float[] in, float[] out) {
        int nx = n * nEmbd;
        parallel.run(n, in);
        float[] ffn = parallel.out;
        switch (wiring.combine) {
            case SUM:
                CpuOps.add(out, 0, ffn, 0, out, 0, nx);
                break;
            case SCALED_SUM:
                CpuOps.add(ffn, 0, out, 0, out, 0, nx);
                CpuOps.scale(out, 0, nx, GROK_COMBINE);
                break;
            case NORMED_SUM:
                CpuOps.rmsNorm(out, 0, out, 0, n, nEmbd, postNorm2, rmsEps);
                CpuOps.rmsNorm(ffn, 0, ffn, 0, n, nEmbd, postNorm1, rmsEps);
                CpuOps.add(ffn, 0, out, 0, out, 0, nx);
                break;
            default:
                throw new IllegalStateException("unhandled parallel combine " + wiring.combine);
        }
    }

    private float[] residual() {
        return ctx.scratch().slot(CpuScratch.X);
    }

    @Override
    public String toString() {
        return "CpuMoeFfn{layer=" + layer.il() + ", kind=" + kind + ", " + spec + (wiring != null ? ", " + wiring : "")
                + '}';
    }

    // ================================================================== helpers

    private int rows(long width) {
        return Math.toIntExact(Math.multiplyExact((long) ubatch, width));
    }

    private float[] vector(CpuWeights w, String suffix, int n) {
        float[] v = w.layerFloatsOrNull(layer, suffix);
        if (v == null) {
            w.layer(layer, suffix);
            throw new IllegalStateException("layer " + layer.il() + ": " + layer.tensorName(suffix) + " is absent");
        }
        return checkLength(v, n, suffix);
    }

    private float[] checkLength(float[] v, int n, String suffix) {
        if (v.length != n) {
            throw new IllegalStateException("layer " + layer.il() + ": " + layer.tensorName(suffix) + " holds "
                    + v.length + " values, expected " + n);
        }
        return v;
    }

    // ================================================================== experts

    /**
     * One expert set: the projections of the step's selected pairs grouped by expert, then the per-pair activation,
     * norm and weighted-sum steps.
     */
    private final class ExpertPass {
        final MoeExpertWeights plan;
        final MoeExpertWeights.Cpu w;
        final CpuExpertMatmul products;
        final int k;
        final boolean weightBefore;
        final float[] postNorm;
        final float postEps;
        final int[] ids;
        final float[] weights;
        final float[] gate;
        final float[] up;
        final float[] act;
        final float[] expertOut;
        final CpuProjectionDeltas.ExpertSite gateSite;
        final CpuProjectionDeltas.ExpertSite upSite;
        final CpuProjectionDeltas.ExpertSite gateUpSite;
        final CpuProjectionDeltas.ExpertSite downSite;

        ExpertPass(MoeExpertWeights plan, CpuWeights weights, int k, boolean weightBefore, float[] postNorm,
                   float postEps) {
            this.plan = plan;
            this.w = plan.resolve(weights, layer);
            this.products = new CpuExpertMatmul(mv);
            this.k = k;
            this.weightBefore = weightBefore;
            this.postNorm = postNorm;
            this.postEps = postEps;
            CpuScratch scratch = ctx.scratch();
            int pairs = rows(k);
            this.ids = scratch.intTemp(T_IDS, pairs);
            this.weights = scratch.temp(T_WEIGHTS, pairs);
            this.gate = plan.gated ? scratch.temp(T_GATE, Math.multiplyExact(pairs, plan.gateUpRows())) : null;
            this.up = plan.gated && !plan.fused ? scratch.temp(T_UP, Math.multiplyExact(pairs, plan.nFf)) : null;
            this.act = scratch.temp(T_ACT, Math.multiplyExact(pairs, plan.nFf));
            this.expertOut = scratch.temp(T_EXPERT_OUT, Math.multiplyExact(pairs, plan.outDim));
            CpuProjectionDeltas deltas = ctx.projectionDeltas();
            this.gateSite = plan.gated && !plan.fused
                    ? deltas.serveExperts(layer, plan.gateName(), w.gate, plan.nExpert, mv) : null;
            this.upSite = plan.fused ? null : deltas.serveExperts(layer, plan.upName(), w.up, plan.nExpert, mv);
            this.gateUpSite = plan.fused
                    ? deltas.serveExperts(layer, plan.gateUpName(), w.gateUp, plan.nExpert, mv) : null;
            this.downSite = deltas.serveExperts(layer, plan.downName(), w.down, plan.nExpert, mv);
        }

        /**
         * Experts of {@code sel} over {@code n} tokens whose rows of {@code in_dim} start at {@code x[0]}; the
         * weighted sum goes to {@code dst} from index 0 with {@code flags}.
         */
        void run(int n, float[] x, MoeRouting.Selection sel, float[] dst, int flags) {
            int pairs = n * k;
            for (int t = 0; t < n; t++) {
                for (int j = 0; j < k; j++) {
                    ids[t * k + j] = sel.id(t, j);
                    weights[t * k + j] = sel.weight(t, j);
                }
            }
            float[] xs = weightBefore ? MoeCombine.scaleInput(n, plan.inDim, k, x, weights) : x;
            int perRow = weightBefore ? 1 : k;
            int nFf = plan.nFf;
            int gu = plan.gateUpRows();
            CpuExpertMatmul.Groups groups = CpuExpertMatmul.Groups.of(ids, pairs, plan.nExpert);
            if (plan.fused) {
                products.run(w.gateUp, plan.nExpert, groups, xs, 0, perRow, gate, 0);
                if (w.fusedUpScale != null) {
                    for (int p = 0; p < pairs; p++) {
                        NormMath.mulRowScale(1, gu, w.fusedUpScale, ids[p], 1, gate, p * gu);
                    }
                }
            } else if (plan.gated) {
                products.run(w.up, plan.nExpert, groups, xs, 0, perRow, up, 0);
                products.run(w.gate, plan.nExpert, groups, xs, 0, perRow, gate, 0);
            } else {
                products.run(w.up, plan.nExpert, groups, xs, 0, perRow, act, 0);
            }
            if (plan.fused) {
                addDeltas(gateUpSite, pairs, xs, !weightBefore, gate);
            } else if (plan.gated) {
                addDeltas(upSite, pairs, xs, !weightBefore, up);
                addDeltas(gateSite, pairs, xs, !weightBefore, gate);
            } else {
                addDeltas(upSite, pairs, xs, !weightBefore, act);
            }
            if (plan.fused) {
                addBias(pairs, gu, w.gateUpBias, gate);
                CpuOps.gluRows(plan.glu, plan.params, pairs, nFf, gate, 0, gu, gate, nFf, gu, act, 0, nFf);
            } else if (plan.gated) {
                addBias(pairs, nFf, w.upBias, up);
                addBias(pairs, nFf, w.gateBias, gate);
                CpuOps.gluRows(plan.glu, plan.params, pairs, nFf, gate, 0, nFf, up, 0, nFf, act, 0, nFf);
            } else {
                addBias(pairs, nFf, w.upBias, act);
                CpuOps.unary(plan.act, plan.params, act, 0, act, 0, pairs * nFf);
            }
            products.run(w.down, plan.nExpert, groups, act, 0, 1, expertOut, 0);
            addDeltas(downSite, pairs, act, false, expertOut);
            addBias(pairs, plan.outDim, w.downBias, expertOut);
            if (postNorm != null) {
                NormMath.mulRowScale(pairs, plan.outDim, postNorm, 0, plan.outDim, expertOut, 0);
                CpuOps.rmsNorm(expertOut, 0, expertOut, 0, pairs, plan.outDim, null, postEps);
            }
            int f = flags | (weightBefore ? MoeKernels.WSUM_UNWEIGHTED : 0);
            MoeCombine.weightedSum(n, plan.outDim, k, f, expertOut, weights, dst);
        }

        /** The deltas of an expert product over the step's pairs, after its per-expert scale. */
        private void addDeltas(CpuProjectionDeltas.ExpertSite site, int pairs, float[] x, boolean tokenRows,
                               float[] y) {
            if (site != null && !site.isEmpty()) {
                site.apply(pairs, k, ids, x, 0, tokenRows, y, 0);
            }
        }

        private void addBias(int pairs, int dim, float[] bias, float[] data) {
            if (bias != null) {
                MoeCombine.addId(pairs, dim, ids, bias, data);
            }
        }
    }

    // ================================================================== dense FFN

    /**
     * A dense FFN ({@code build_ffn}): the shared experts and the parallel dense FFN. A product with deltas runs
     * without its {@code w_s}, adds them and then applies {@code w_s}, as {@code build_ffn} scales after
     * {@code build_lora_mm} (llama-graph.cpp:1979-2010).
     */
    private final class DenseFfn {
        final FfnSpec f;
        final CpuTensorRef gate;
        final CpuTensorRef up;
        final CpuTensorRef down;
        final CpuProjectionDeltas.Site gateSite;
        final CpuProjectionDeltas.Site upSite;
        final CpuProjectionDeltas.Site downSite;
        final CpuMatVec unscaled;
        final float[] g;
        final float[] u;
        final float[] out;

        DenseFfn(FfnSpec f, CpuWeights w) {
            this.f = Objects.requireNonNull(f, "ffn");
            if (f.activationSparsityStdMul() != 0f) {
                throw new UnsupportedOperationException("layer " + layer.il() + ": " + f.role()
                        + " FFN with activation sparsity is not part of a MoE graph");
            }
            if (f.gating() == FfnGating.SEQ && f.gated()) {
                throw new UnsupportedOperationException("layer " + layer.il() + ": sequentially gated " + f.role()
                        + " FFN is not built by the MoE graphs");
            }
            int nFf = f.nFf();
            boolean fused = f.gating() == FfnGating.FUSED_UP;
            this.gate = f.gating() == FfnGating.PAR ? w.layer(layer, f.gateWeight()) : null;
            this.up = w.layer(layer, f.upWeight());
            this.down = w.layer(layer, f.downWeight());
            long upRows = fused ? 2L * nFf : nFf;
            if (up.cols() != nEmbd || up.rows() != upRows || down.cols() != nFf || down.rows() != nEmbd
                    || (gate != null && (gate.cols() != nEmbd || gate.rows() != nFf))) {
                throw new IllegalStateException("layer " + layer.il() + ": " + f.role() + " FFN tensors " + gate + ", "
                        + up + ", " + down + " do not match n_embd " + nEmbd + " and n_ff " + nFf);
            }
            CpuProjectionDeltas deltas = ctx.projectionDeltas();
            this.gateSite = gate != null ? deltas.serve(layer, f.gateWeight(), gate, mv) : null;
            this.upSite = deltas.serve(layer, f.upWeight(), up, mv);
            this.downSite = deltas.serve(layer, f.downWeight(), down, mv);
            this.unscaled = new CpuMatVec(t -> {
                CpuMatVec.Transforms tr = mv.transformsOf(t);
                return tr.hasScale() ? CpuMatVec.Transforms.of(tr.input(), null, tr.experts()) : tr;
            });
            CpuScratch scratch = ctx.scratch();
            this.g = gate != null || fused ? scratch.temp(T_FFN_GATE, rows(upRows)) : null;
            this.u = scratch.temp(T_FFN_UP, rows(nFf));
            this.out = scratch.temp(T_FFN_OUT, rows(nEmbd));
        }

        /** {@link #out} = FFN of {@code n} rows of {@code x}. */
        void run(int n, float[] x) {
            int nFf = f.nFf();
            switch (f.gating()) {
                case PAR:
                    project(up, upSite, x, u, n);
                    project(gate, gateSite, x, g, n);
                    CpuOps.gluRows(f.glu(), f.params(), n, nFf, g, 0, nFf, u, 0, nFf, u, 0, nFf);
                    break;
                case FUSED_UP:
                    project(up, upSite, x, g, n);
                    CpuOps.gluRows(f.glu(), f.params(), n, nFf, g, 0, 2 * nFf, g, nFf, 2 * nFf, u, 0, nFf);
                    break;
                case SEQ:
                    project(up, upSite, x, u, n);
                    CpuOps.unary(f.act(), f.params(), u, 0, u, 0, n * nFf);
                    break;
                default:
                    throw new IllegalStateException("unhandled FFN gating " + f.gating());
            }
            project(down, downSite, u, out, n);
        }

        /** {@code y = (W x + deltas) * w_s} over {@code n} dense rows from index 0. */
        private void project(CpuTensorRef w, CpuProjectionDeltas.Site site, float[] x, float[] y, int n) {
            if (site.isEmpty()) {
                mv.matmul(w, x, 0, y, 0, n);
                return;
            }
            CpuMatVec.Transforms tr = mv.transformsOf(w);
            unscaled.matmul(w, x, 0, y, 0, n);
            site.applyRows(n, x, 0, y, 0);
            if (tr.hasScale()) {
                CpuOps.scale(y, 0, Math.toIntExact((long) n * w.rows()), tr.scale(0));
            }
        }
    }
}
