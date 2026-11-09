package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * GPU block of the routed mixture-of-experts kinds {@link FfnKind#MOE}, {@link FfnKind#MOE_LATENT},
 * {@link FfnKind#MOE_CHUNKED} and {@link FfnKind#MOE_PARALLEL_DENSE}, following llama.cpp {@code build_moe_ffn}
 * (src/llama-graph.cpp:2189-2580) and the graphs that call it. One step, all on the context queue with no host
 * synchronization:
 * <ol>
 *   <li><b>Router.</b> {@code logits = ffn_gate_inp . r} with {@code r} per {@link MoeBlocks.RouterInput}
 *       ({@code GpuOps.matmul}), {@code + ffn_gate_inp.bias} ({@code LLM_AddId} over an all-zero slot list), then
 *       {@code LLM_MoE_Router} (gating, {@code exp_probs_b}, groups, top-k, weight normalization and scale). deepseek4
 *       hash layers first write the ids with {@code LLM_MoE_HashIds} from the step's token ids, kolibri1 picks them
 *       by {@code logits + exp_probs_b} ({@link MoeBlocks.Selection#BIASED_LOGITS}); both then route with
 *       {@code sel_mode} 3.</li>
 *   <li><b>Experts</b> (one token): {@code {T}_MatVecIdGateUp} for the gated GLU with per-expert scales and the
 *       expert clamp of the spec, or, with expert biases, {@code {T}_MatVecId} gate and up, {@code LLM_AddId},
 *       {@code LLM_GLU}; an ungated set runs {@code {T}_MatVecId} up and {@code LLM_Unary}. Then
 *       {@code {T}_MatVecId} down, {@code LLM_AddId} down bias, the post-expert norm
 *       ({@code LLM_MulRowScale}, {@code LLM_RowNorm}) and {@code LLM_MoE_WeightedSum}. Several tokens:
 *       the expert sort ({@link GpuGroupedRows}: {@code LLM_MoE_Bucketize}, or its multi-work-group phases on
 *       large steps), then either the grouped GEMM ({@code LLM_MoE_AlignBlocks} + {@code {T}_GroupedGemm}, the
 *       expert-sorted tiled GEMM over the block-aligned rows, {@code {T}_GroupedGemmMma} on CUDA or Vulkan tensor
 *       cores, whenever the device runs the tiled GEMM kernels and every expert dimension is a multiple of the
 *       weight types' block sizes) or
 *       {@code {T}_BatchMatMulId}, for gate, up (or fused gate_up) and down, with the same bias, activation and norm
 *       steps. llama4 weights the inputs first ({@code LLM_MoE_ScaleInput}) and sums unweighted. An expert tensor
 *       held in several device shards of whole experts (a tensor above one binding, {@link MoeExpertWeights.Shard})
 *       runs each of these products as one dispatch per shard with that shard's
 *       {@code expert_base, expert_count}; the shards write disjoint (pair, row) outputs, so nothing is cleared or
 *       combined, and a single-shard tensor is one dispatch. When the shards cover only part of the expert set
 *       (expert parallel), the pairs of the missing experts are zeroed before the products, whose id dispatches skip
 *       experts outside the bound range, so the block's output is that rank's summand for the group all-reduce.</li>
 *   <li><b>Latent</b> experts run between {@code ffn_latent_down/up} or {@code ffn_routed_down/up}, with the optional
 *       {@code ffn_pre_exp_norm} (muse-spark.cpp:175-179) and {@code ffn_routed_norm} (kimi-k3.cpp:597-602).</li>
 *   <li><b>Chunk experts</b> (grovemoe.cpp:154-171) route again on the same logits with divided ids, read the routed
 *       output and are added scaled by {@code expert_group_scale}.</li>
 *   <li><b>Shared experts</b> are a dense FFN on the FFN input through {@link GpuOps}, gated by
 *       {@code sigmoid(ffn_gate_inp_shexp . x)} (added with {@code LLM_MoE_WeightedSum}, k = 1) or added plainly;
 *       the sum is then multiplied by {@link MoeSpec#sharedCombineScale()} when it is not 1 (cohere2moe 0.5,
 *       cohere2moe.cpp:242-243).</li>
 *   <li><b>Parallel dense FFN</b> per {@link MoeBlocks.Combine}.</li>
 * </ol>
 *
 * <p>When a placement rule keeps a layer's routed expert stacks in RAM ({@code --cpu-moe}, {@code --n-cpu-moe},
 * {@code -ot}, {@code neuralfree.moe.cache}), the routed pass runs over the MoE expert cache
 * ({@link GpuCachedMoePass}): the cached experts are computed on the device from the cache banks, the misses on the
 * host; the layer kinds the hosted path does not build are refused by {@link #checkSupported} (the host-expert FFN
 * decorator serves them under a hybrid layer split).</p>
 *
 * <p>When the context carries a {@link MoeExpertHeat} recorder (a ranking profile save is configured), each router's
 * selected ids are read back asynchronously after its kernels and counted per weight layer as the reads complete
 * (at most {@link GpuRouterHeat#PENDING} stay pending; close drains them); a hosted layer's ids are counted by
 * {@link GpuCachedMoePass} from the read it already makes.</p>
 *
 * <p>Every weight product adds the {@link GpuProjectionDeltas} registered on its tensor (LoRA) right after the
 * product and before any bias: {@code build_lora_mm} for the router, latent, shared-gate and dense products, with
 * a dense FFN's {@code w_s} applied after the delta as {@code build_ffn} does (llama-graph.cpp:1979-2010), and
 * {@code build_lora_mm_id} for the expert products after their per-expert scale (llama-graph.cpp:1706-1749). An
 * expert delta on gate or up runs the one-token experts through separate {@code {T}_MatVecId} gate and up products
 * instead of the fused {@code {T}_MatVecIdGateUp}; both are prepared at bind, as an adapter can attach to a bound
 * model at any time and no step may build a pipeline.</p>
 *
 * <p>The block reads FFN-input rows of {@code n_embd = ffn_gate_inp.ne[0]} values and writes the branch rows in the
 * same layout. Every working buffer is a {@link GpuScratch} temporary sized for {@code ubatch} at bind; the
 * per-step work only re-sets token counts, global sizes and the caller's slot buffers on the prepared
 * dispatches.</p>
 */
public final class GpuMoeFfn implements GpuFfnBlock {

    static final String T_LOGITS = "moe.logits";
    static final String T_IDS = "moe.ids";
    static final String T_WEIGHTS = "moe.weights";
    static final String T_ZERO_IDS = "moe.zero_ids";
    static final String T_SELECTION = "moe.selection";
    static final String T_ROUTER_IN = "moe.router_in";
    static final String T_MOE_IN = "moe.moe_in";
    static final String T_LATENT_IN = "moe.latent_in";
    static final String T_LATENT_SUM = "moe.latent_sum";
    static final String T_CHUNK_OUT = "moe.chunk_out";
    static final String T_SUM = "moe.sum";
    static final String T_SHEXP_GATE = "moe.shexp_gate";
    /** Per-token weight sum of the selected identity experts. */
    static final String T_ZERO_WEIGHTS = "moe.zero_weights";
    /** Physical replica ids the expert kernels address when the layer remaps its routed experts. */
    static final String T_EXPERT_IDS = "moe.expert_ids";

    /** {@code x} of {@code LLM_MoE_ZeroApply}. */
    private static final int ARG_ZERO_X = 4;
    /** {@code dst} of {@code LLM_MoE_ZeroApply}. */
    private static final int ARG_ZERO_DST = 6;

    /** {@code dst} of {@code LLM_MoE_WeightedSum}. */
    private static final int ARG_WSUM_DST = 6;
    /** grok-2 parallel combine factor {@code std::sqrt(2) / 2} (grok.cpp:178). */
    private static final float GROK_COMBINE = (float) (Math.sqrt(2.0) / 2.0);

    private final FfnKind kind;
    private final LayerSpec layer;
    private final MoeSpec spec;

    private GpuForwardContext ctx;
    private GpuOps ops;
    private MoeBlocks.Wiring wiring;
    private int ubatch;
    private int nEmbd;
    private float rmsEps;

    private GpuWeight gateInp;
    private GpuProjectionDeltas.Site gateInpDeltas;
    private GpuProjectionDeltas.Site latentDownDeltas;
    private GpuProjectionDeltas.Site latentUpDeltas;
    private GpuProjectionDeltas.Site sharedGateInpDeltas;
    private ScratchSlot logits;
    private ScratchSlot zeroIds;
    private PreparedDispatch routerBias;
    private ScratchSlot routerIn;
    private GpuWeight routerScaleVec;
    private PreparedDispatch routerScale;
    private ScratchSlot layerInput;
    private Router router;
    private Router chunkRouter;
    /** Identity-expert masking of the routed selection; null without zero experts. */
    private ScratchSlot zeroWeights;
    private PreparedDispatch zeroMask;
    private PreparedDispatch zeroApply;
    private int zeroN = -1;
    private int zeroApplyN = -1;
    private GpuBuffer zeroX;
    private int zeroXOff = -1;
    private GpuBuffer zeroDst;

    private GpuWeight expertNorm;
    private ScratchSlot moeIn;
    private GpuWeight latentDown;
    private GpuWeight latentUp;
    private GpuWeight preExpNorm;
    private GpuWeight routedNorm;
    private ScratchSlot latentIn;
    private ScratchSlot latentSum;

    /**
     * The routed (or chunk) expert set of one step: the products and the weighted sum into the destination. The
     * routed experts of a hosted layer run over the expert cache through {@link GpuCachedMoePass}; resident ones
     * through {@link GpuMoeExpertPass}.
     */
    interface Routed extends AutoCloseable {
        void run(int n, GpuBuffer x, int xOff, GpuBuffer dst, int flags);
    }

    /** The routed experts' logical-to-physical replica mapping of this layer, or null. */
    private MoeExpertRemap.Layer layerRemap;

    private Routed routed;
    private Routed chunk;
    private ScratchSlot chunkOut;
    /** Whether a decode step reads the routed ids (and a hosted layer's input rows) back to the host. */
    private boolean decodeReadsHost = true;

    private GpuDenseFfn shared;
    private GpuWeight sharedGateInp;
    private ScratchSlot sharedGate;
    private PreparedDispatch sharedSum;
    private GpuDenseFfn parallel;
    private GpuWeight postNorm1;
    private GpuWeight postNorm2;

    private ScratchSlot sum;
    private GpuBuffer sharedSumDst;
    private int sharedSumN = -1;
    private int routerBiasN = -1;
    private int routerScaleN = -1;

    /**
     * An unbound block for a layer of {@code kind}.
     *
     * @throws IllegalArgumentException when the layer's spec does not match the kind
     */
    public GpuMoeFfn(FfnKind kind, LayerSpec layer) {
        this.kind = Objects.requireNonNull(kind, "kind");
        this.layer = layer;
        this.spec = MoeBlocks.checkLayer(kind, layer);
    }

    // ================================================================== bind

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        if (this.ctx != null) {
            throw new IllegalStateException("MoE block of layer " + this.layer.il() + " is already bound");
        }
        if (layer != this.layer) {
            throw new IllegalArgumentException("block created for layer " + this.layer.il() + " bound to layer "
                    + layer.il());
        }
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.ops = ctx.ops();
        GpuScratch scratch = ctx.scratch();
        this.ubatch = scratch.ubatch();
        this.rmsEps = ctx.spec().hparams().normRmsEps();
        this.wiring = MoeBlocks.Wiring.of(ctx.spec().arch(), kind, layer);
        GpuWeights w = ctx.weightSource().weightsFor(layer.weightLayer());
        boolean hosted = !expertsOnDevice(w, layer) && hostHasExperts(ctx, layer);
        if (hosted) {
            checkSupported(ctx.spec(), layer);
        }
        decodeReadsHost = hosted || ctx.heat() != null;

        gateInp = w.layer(layer, TensorNames.weight(TensorNames.FFN_GATE_INP));
        if (gateInp.rank() != 2 || gateInp.rows() != spec.nRouterExpert() || gateInp.hasScale()) {
            throw new IllegalStateException("layer " + layer.il() + ": " + gateInp.name() + " must be [n_embd, "
                    + spec.nRouterExpert() + "] without an output scale, got " + gateInp);
        }
        nEmbd = Math.toIntExact(gateInp.cols());
        gateInpDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.FFN_GATE_INP), gateInp);
        this.layerRemap = ctx.moeExpertRemap() != null && ctx.moeExpertRemap().covers(layer.weightLayer())
                ? ctx.moeExpertRemap().layer(layer.weightLayer()) : null;
        bindRouter(w, scratch, hosted ? null : ctx.heat());
        if (spec.zeroExperts() > 0) {
            bindZeroExperts(scratch);
        }
        bindExpertInput(w, scratch);
        int expertDim = spec.latent() ? spec.latentSize() : nEmbd;
        MoeExpertWeights expertPlan = MoeExpertWeights.routed(spec, expertDim,
                layerRemap != null ? layerRemap.physical() : spec.nExpert());
        if (hosted) {
            MoeExpertCache cache = ctx.moeExpertCache();
            routed = new CachedRouted(new GpuCachedMoePass(ctx, layer, expertPlan, router,
                    cache != null && cache.serves(layer.weightLayer()) ? cache : null, w,
                    ctx.scratch().ubatch(), spec.weightBeforeFfn(),
                    spec.postExpertNorm() ? w.layer(layer, TensorNames.weight(TensorNames.FFN_POST_EXP_NORM)) : null,
                    ctx.spec().hparams().postNormEps(), ctx.moeExpertPrefetch(), ctx.moeTailSkipRank()));
        } else {
            routed = new GpuMoeExpertPass(ctx, layer, expertPlan, w, router, spec.weightBeforeFfn(),
                    spec.postExpertNorm() ? w.layer(layer, TensorNames.weight(TensorNames.FFN_POST_EXP_NORM)) : null,
                    ctx.spec().hparams().postNormEps());
        }
        if (spec.chunked()) {
            chunkRouter = new Router(MoeRouting.Params.forChunks(spec), w, false, false, ctx.heat(), false, null);
            chunk = new GpuMoeExpertPass(ctx, layer, MoeExpertWeights.chunks(spec, nEmbd), w, chunkRouter,
                    false, null, 0f);
            chunkOut = scratch.temp(T_CHUNK_OUT, rows(nEmbd));
        }
        if (spec.sharedExpert() != null) {
            shared = new GpuDenseFfn(ctx, w, layer, spec.sharedExpert(), nEmbd);
            if (spec.sharedExpertGated()) {
                sharedGateInp = w.layer(layer, TensorNames.weight(TensorNames.FFN_GATE_INP_SHEXP));
                if (sharedGateInp.rows() != 1 || sharedGateInp.cols() != nEmbd) {
                    throw new IllegalStateException("layer " + layer.il() + ": " + sharedGateInp.name()
                            + " must hold one row of " + nEmbd + ", got " + sharedGateInp);
                }
                sharedGateInpDeltas = ctx.projectionDeltas().serve(layer,
                        TensorNames.weight(TensorNames.FFN_GATE_INP_SHEXP), sharedGateInp);
                sharedGate = scratch.temp(T_SHEXP_GATE, ubatch);
            }
        }
        if (kind == FfnKind.MOE_PARALLEL_DENSE) {
            parallel = new GpuDenseFfn(ctx, w, layer, layer.ffn(), nEmbd);
            if (wiring.combine == MoeBlocks.Combine.NORMED_SUM) {
                if (shared != null || chunk != null) {
                    throw new UnsupportedOperationException("layer " + layer.il() + ": the normed parallel combine"
                            + " of gemma4 has no shared or chunk experts in llama.cpp");
                }
                postNorm1 = w.layer(layer, TensorNames.weight(TensorNames.FFN_POST_NORM_1));
                postNorm2 = w.layer(layer, TensorNames.weight(TensorNames.FFN_POST_NORM_2));
            }
        }
        sum = scratch.temp(T_SUM, rows(nEmbd));
        if (shared != null && spec.sharedExpertGated()) {
            sharedSum = ops.prepare(MoeKernels.WEIGHTED_SUM, MoeKernels.elementGlobal(nEmbd, ops.local()),
                    local1(), MoeKernels.weightedSumArgs(1, nEmbd, 1, MoeKernels.WSUM_ACCUMULATE,
                            shared.out().buffer(), sharedGate.buffer(), sum.buffer()));
            sharedSumN = 1;
            sharedSumDst = sum.buffer();
        }
    }

    /**
     * Binds the router logits and the router; {@code heat} is the recorder its selections are read back into, null
     * when a hosted pass counts them from its own ids read or no profile records.
     */
    private void bindRouter(GpuWeights w, GpuScratch scratch, MoeExpertHeat heat) {
        logits = scratch.temp(T_LOGITS, rows(spec.nRouterExpert()));
        if (spec.routerBias()) {
            GpuWeight bias = f32Vector(w.layer(layer, TensorNames.bias(TensorNames.FFN_GATE_INP)),
                    spec.nRouterExpert());
            routerBias = ops.prepare(MoeKernels.ADD_ID,
                    MoeKernels.elementGlobal((long) spec.nRouterExpert(), ops.local()),
                    local1(), MoeKernels.addIdArgs(1, spec.nRouterExpert(), zeroIds(scratch).buffer(),
                            bias.shard(0), logits.buffer()));
            routerBiasN = 1;
        }
        if (wiring.router == MoeBlocks.RouterInput.SCALED_RESIDUAL) {
            routerIn = scratch.temp(T_ROUTER_IN, rows(nEmbd));
            routerScaleVec = f32Vector(w.layer(layer, TensorNames.scale(TensorNames.FFN_GATE_INP)), nEmbd);
            routerScale = ops.prepare(NormKernels.MUL_ROW_SCALE, ActKernels.global1d(nEmbd, ops.local()), local1(),
                    NormKernels.mulRowScaleArgs(1, nEmbd, routerScaleVec.shard(0), nEmbd, routerIn.buffer()));
            routerScaleN = 1;
        }
        if (wiring.needsLayerInput()) {
            if (layer.residual().streamWidth() != nEmbd) {
                throw new IllegalStateException("layer " + layer.il() + ": the layer input has "
                        + layer.residual().streamWidth() + " columns, the MoE reads " + nEmbd);
            }
            layerInput = scratch.temp(MoeBlocks.LAYER_INPUT_TEMP, MoeBlocks.layerInputFloats(ubatch, ctx.spec()));
        }
        boolean biasedLogits = wiring.selection == MoeBlocks.Selection.BIASED_LOGITS;
        if (biasedLogits && spec.mediaRouting()) {
            throw new UnsupportedOperationException("layer " + layer.il() + ": embedding-row routing over the"
                    + " logits + exp_probs_b selection has no llama.cpp graph");
        }
        router = biasedLogits
                ? new Router(MoeRouting.Params.preselected(spec), w, false, true, heat, false, layerRemap)
                : new Router(MoeRouting.Params.forExperts(spec), w, spec.probBias(), false, heat,
                        spec.mediaRouting(), layerRemap);
    }

    /**
     * Prepares the identity-expert post-processing: the mask over the routed ids and weights, and the per-token
     * identity contribution added to the routed output.
     */
    private void bindZeroExperts(GpuScratch scratch) {
        zeroWeights = scratch.temp(T_ZERO_WEIGHTS, ubatch);
        zeroMask = ops.prepare(MoeKernels.ZERO_EXPERTS,
                MoeKernels.elementGlobal(ubatch, ops.local()), local1(),
                MoeKernels.zeroExpertsArgs(1, router.params.k(), spec.nExpert(), router.ids.buffer(),
                        router.weights.buffer(), zeroWeights.buffer()));
        zeroApply = ops.prepare(MoeKernels.ZERO_APPLY,
                MoeKernels.elementGlobal((long) ubatch * nEmbd, ops.local()), local1(),
                MoeKernels.zeroApplyArgs(1, nEmbd, 0, 0, ctx.placeholders().readOnlyFloats(), zeroWeights.buffer(),
                        ctx.placeholders().readOnlyFloats()));
        zeroN = 1;
        zeroApplyN = 1;
    }

    /** {@code zero_weights}: takes the identity experts out of the routing result of {@code n} tokens. */
    private void maskZeroExperts(int n) {
        if (zeroMask == null) {
            return;
        }
        if (zeroN != n) {
            zeroMask.setInt(MoeKernels.ARG_N_TOKENS, n);
            zeroMask.setGlobal(MoeKernels.elementGlobal(n, ops.local()));
            zeroN = n;
        }
        zeroMask.dispatch();
    }

    /** {@code dst += zero_weights[t] * x[t]}: the identity experts' contribution to the routed output. */
    private void applyZeroExperts(int n, GpuBuffer x, int xOff, GpuBuffer dst) {
        if (zeroApply == null) {
            return;
        }
        if (zeroApplyN != n) {
            zeroApply.setInt(MoeKernels.ARG_N_TOKENS, n);
            zeroApply.setGlobal(MoeKernels.elementGlobal((long) n * nEmbd, ops.local()));
            zeroApplyN = n;
        }
        if (x != zeroX) {
            zeroApply.setBuffer(ARG_ZERO_X, x);
            zeroX = x;
        }
        if (xOff != zeroXOff) {
            zeroApply.setInt(2, xOff);
            zeroXOff = xOff;
        }
        if (dst != zeroDst) {
            zeroApply.setBuffer(ARG_ZERO_DST, dst);
            zeroDst = dst;
        }
        zeroApply.dispatch();
    }

    /** The embedding rows' selection bias: {@code exp_probs_b_vl} as one F32 buffer, or a placeholder for none. */
    private GpuBuffer mediaBias(GpuWeights w) {
        return spec.mediaProbBias()
                ? f32Vector(w.layer(layer, TensorNames.bias(TensorNames.FFN_EXP_PROBS_B_VL)), spec.nExpert()).shard(0)
                : ctx.placeholders().readOnlyFloats();
    }

    /**
     * An all-zero slot list of {@code ubatch} entries for {@code LLM_AddId} of one bias row over every token, filled
     * once at bind; only this block family reads the name and never writes it.
     */
    private ScratchSlot zeroIds(GpuScratch scratch) {
        if (zeroIds == null) {
            zeroIds = scratch.temp(T_ZERO_IDS, ubatch);
            ops.fill(zeroIds.buffer(), 4L * zeroIds.offset(), 4L * ubatch, 0);
        }
        return zeroIds;
    }

    /** {@code exp_probs_b} ({@code .bias}, or suffix-less for hy_v3, hy-v3.cpp:59) as one F32 buffer. */
    private GpuBuffer expProbsB(GpuWeights w) {
        GpuWeight b = w.layerOrNull(layer, TensorNames.bias(TensorNames.FFN_EXP_PROBS_B));
        if (b == null) {
            b = w.layer(layer, TensorNames.FFN_EXP_PROBS_B);
        }
        return f32Vector(b, spec.nRouterExpert()).shard(0);
    }

    private void bindExpertInput(GpuWeights w, GpuScratch scratch) {
        if (wiring.experts == MoeBlocks.ExpertInput.NORM_EXPS_OF_LAYER_INPUT) {
            expertNorm = w.layer(layer, TensorNames.weight(TensorNames.FFN_NORM_EXPS));
        } else if (wiring.experts == MoeBlocks.ExpertInput.PRE_NORM_2_OF_RESIDUAL) {
            expertNorm = w.layer(layer, TensorNames.weight(TensorNames.FFN_PRE_NORM_2));
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
        latentDownDeltas = ctx.projectionDeltas().serve(layer, down, latentDown);
        latentUpDeltas = ctx.projectionDeltas().serve(layer, up, latentUp);
        preExpNorm = w.layerOrNull(layer, TensorNames.weight(TensorNames.FFN_PRE_EXP_NORM));
        routedNorm = w.layerOrNull(layer, TensorNames.weight(TensorNames.FFN_ROUTED_NORM));
        latentIn = scratch.temp(T_LATENT_IN, rows(latent));
        latentSum = scratch.temp(T_LATENT_SUM, rows(latent));
    }

    // ================================================================== forward

    @Override
    public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
        if (ctx == null) {
            throw new IllegalStateException("MoE block of layer " + layer.il() + " used before bind");
        }
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException("token count " + nTokens + " outside [1, " + ubatch + "]");
        }
        int nx = Math.multiplyExact(nTokens, nEmbd);
        in.require(nx);
        out.require(nx);
        GpuBuffer dst = out.offset() == 0 ? out.buffer() : sum.buffer();

        GpuBuffer xBuf = in.buffer();
        int xOff = in.offset();
        if (expertNorm != null) {
            ScratchSlot src = wiring.experts == MoeBlocks.ExpertInput.NORM_EXPS_OF_LAYER_INPUT ? layerInput : residual();
            ops.norm(NormKind.RMS, expertNorm, null, rmsEps, src.buffer(), src.offset(), moeIn.buffer(),
                    moeIn.offset(), nTokens, nEmbd);
            xBuf = moeIn.buffer();
            xOff = moeIn.offset();
        }
        route(nTokens, in, xBuf, xOff);
        if (spec.latent()) {
            int latent = spec.latentSize();
            ops.matmul(latentDown, xBuf, xOff, latentIn.buffer(), latentIn.offset(), nTokens);
            latentDownDeltas.applyRows(nTokens, xBuf, xOff, latentIn.buffer(), latentIn.offset());
            if (preExpNorm != null) {
                ops.norm(NormKind.RMS, preExpNorm, null, rmsEps, latentIn.buffer(), latentIn.offset(),
                        latentIn.buffer(), latentIn.offset(), nTokens, latent);
            }
            routed.run(nTokens, latentIn.buffer(), latentIn.offset(), latentSum.buffer(), 0);
            if (routedNorm != null) {
                ops.norm(NormKind.RMS, routedNorm, null, rmsEps, latentSum.buffer(), latentSum.offset(),
                        latentSum.buffer(), latentSum.offset(), nTokens, latent);
            }
            ops.matmul(latentUp, latentSum.buffer(), latentSum.offset(), dst, 0, nTokens);
            latentUpDeltas.applyRows(nTokens, latentSum.buffer(), latentSum.offset(), dst, 0);
        } else {
            routed.run(nTokens, xBuf, xOff, dst, 0);
            applyZeroExperts(nTokens, xBuf, xOff, dst);
        }
        if (chunk != null) {
            chunkRouter.run(nTokens);
            chunk.run(nTokens, dst, 0, chunkOut.buffer(), 0);
            ops.scaleAdd(chunkOut.buffer(), chunkOut.offset(), spec.chunkScale(), dst, 0, nx);
        }
        if (shared != null) {
            addShared(nTokens, in, dst);
        }
        if (parallel != null) {
            combineParallel(nTokens, in, dst);
        }
        if (dst != out.buffer()) {
            ops.copy(dst, 0, out.buffer(), out.offset(), nx);
        }
    }

    /**
     * False when a decode step reads back to the host: a hosted layer's expert pass reads the routed ids and the
     * expert input rows every step, and a heat profile reads the ids of a resident routing too.
     */
    @Override
    public boolean capturableDecode() {
        return !decodeReadsHost;
    }

    /**
     * Router logits for this step's tokens, then expert ids and weights; {@code expertIn} at {@code expertOff} is the
     * routed experts' input (before any latent projection).
     */
    private void route(int n, ScratchSlot in, GpuBuffer expertIn, int expertOff) {
        switch (wiring.router) {
            case FFN_INPUT:
                routerLogits(n, in.buffer(), in.offset());
                break;
            case EXPERT_INPUT:
                routerLogits(n, expertIn, expertOff);
                break;
            case LAYER_INPUT:
                routerLogits(n, layerInput.buffer(), layerInput.offset());
                break;
            case SCALED_RESIDUAL:
                ScratchSlot x = residual();
                ops.norm(NormKind.RMS, null, null, rmsEps, x.buffer(), x.offset(), routerIn.buffer(),
                        routerIn.offset(), n, nEmbd);
                ops.scale(routerIn.buffer(), routerIn.offset(), n * nEmbd, 1.0f / (float) Math.sqrt(nEmbd));
                if (routerScaleN != n) {
                    routerScale.setInt(NormKernels.ARG_ROW_SCALE_N_ROWS, n);
                    routerScale.setGlobal(ActKernels.global1d((long) n * nEmbd, ops.local()));
                    routerScaleN = n;
                }
                routerScale.dispatch();
                routerLogits(n, routerIn.buffer(), routerIn.offset());
                break;
            default:
                throw new IllegalStateException("unhandled router input " + wiring.router);
        }
        if (routerBias != null) {
            if (routerBiasN != n) {
                routerBias.setInt(MoeKernels.ARG_ADD_ID_N_SEL, n);
                routerBias.setGlobal(MoeKernels.elementGlobal((long) n * spec.nExpert(), ops.local()));
                routerBiasN = n;
            }
            routerBias.dispatch();
        }
        router.run(n);
        maskZeroExperts(n);
    }

    /** {@code logits = ffn_gate_inp . x} plus its deltas ({@code build_lora_mm}, llama-graph.cpp:2223). */
    private void routerLogits(int n, GpuBuffer x, int xOff) {
        ops.matmul(gateInp, x, xOff, logits.buffer(), logits.offset(), n);
        gateInpDeltas.applyRows(n, x, xOff, logits.buffer(), logits.offset());
    }

    /** {@code dst += shared(x)}, gated or plain, then {@code dst *= sharedCombineScale} when that scale is not 1
     * ({@code ggml_add} then {@code ggml_scale}, cohere2moe.cpp:242-243 and the MTP block at :396-397).
     */
    private void addShared(int n, ScratchSlot in, GpuBuffer dst) {
        shared.run(n, in.buffer(), in.offset());
        if (sharedSum == null) {
            ops.add(dst, 0, shared.out().buffer(), shared.out().offset(), dst, 0, n * nEmbd);
        } else {
            addGatedShared(n, in, dst);
        }
        float combine = spec.sharedCombineScale();
        if (combine != 1.0f) {
            ops.scale(dst, 0, n * nEmbd, combine);
        }
    }

    /** {@code dst += sigmoid(ffn_gate_inp_shexp . x) * shared(x)} through the one-slot weighted sum. */
    private void addGatedShared(int n, ScratchSlot in, GpuBuffer dst) {
        ops.matmul(sharedGateInp, in.buffer(), in.offset(), sharedGate.buffer(), sharedGate.offset(), n);
        sharedGateInpDeltas.applyRows(n, in.buffer(), in.offset(), sharedGate.buffer(), sharedGate.offset());
        ops.unary(ActOp.SIGMOID, ActParams.NONE, sharedGate.buffer(), sharedGate.offset(), sharedGate.buffer(),
                sharedGate.offset(), n);
        if (sharedSumN != n) {
            sharedSum.setInt(MoeKernels.ARG_N_TOKENS, n);
            sharedSum.setGlobal(MoeKernels.elementGlobal((long) n * nEmbd, ops.local()));
            sharedSumN = n;
        }
        if (sharedSumDst != dst) {
            sharedSum.setBuffer(ARG_WSUM_DST, dst);
            sharedSumDst = dst;
        }
        sharedSum.dispatch();
    }

    private void combineParallel(int n, ScratchSlot in, GpuBuffer dst) {
        int nx = n * nEmbd;
        parallel.run(n, in.buffer(), in.offset());
        GpuBuffer ffn = parallel.out().buffer();
        int ffnOff = parallel.out().offset();
        switch (wiring.combine) {
            case SUM:
                ops.add(dst, 0, ffn, ffnOff, dst, 0, nx);
                break;
            case SCALED_SUM:
                ops.add(ffn, ffnOff, dst, 0, dst, 0, nx);
                ops.scale(dst, 0, nx, GROK_COMBINE);
                break;
            case NORMED_SUM:
                ops.norm(NormKind.RMS, postNorm2, null, rmsEps, dst, 0, dst, 0, n, nEmbd);
                ops.norm(NormKind.RMS, postNorm1, null, rmsEps, ffn, ffnOff, ffn, ffnOff, n, nEmbd);
                ops.add(ffn, ffnOff, dst, 0, dst, 0, nx);
                break;
            default:
                throw new IllegalStateException("unhandled parallel combine " + wiring.combine);
        }
    }

    private ScratchSlot residual() {
        return ctx.scratch().slot(ScratchSlots.X);
    }

    /** True when the layer's routed expert tensors are on the device; a host placement runs the cached pass. */
    private static boolean expertsOnDevice(GpuWeights w, LayerSpec layer) {
        return w.find(layer.tensorName(TensorNames.weight(TensorNames.FFN_UP_EXPS))) != null
                || w.find(layer.tensorName(TensorNames.weight(TensorNames.FFN_GATE_UP_EXPS))) != null;
    }

    /** True when the context's host view holds the layer's routed experts, so the cached pass can run them. */
    private static boolean hostHasExperts(GpuForwardContext ctx, LayerSpec layer) {
        com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights host = ctx.hostWeights();
        return host != null
                && (host.has(layer.tensorName(TensorNames.weight(TensorNames.FFN_UP_EXPS)))
                || host.has(layer.tensorName(TensorNames.weight(TensorNames.FFN_GATE_UP_EXPS))));
    }

    /**
     * Whether a layer can run its routed experts as a hosted block (the routed experts in RAM, the misses computed
     * on the host while the cached pairs run on the device): the refusals checked before the weights resolve, so a
     * load fails with the flag in the message instead of at the first forward pass. Latent or chunk experts, hash
     * routing and biased-logit selection keep their expert set on the device and their layer runs the host-expert
     * FFN decorator when a hybrid layer split hosts it.
     *
     * @throws UnsupportedOperationException for a layer kind, routing or wiring the hosted path does not build
     */
    public static void checkSupported(ModelSpec spec, LayerSpec layer) {
        MoeBlocks.Wiring wiring = MoeBlocks.Wiring.of(spec.arch(), layer.ffnKind(), layer);
        MoeSpec moe = layer.moe();
        String where = spec.arch().ggufName() + " layer " + layer.il() + ": ";
        if (moe.latent() || moe.chunked()) {
            throw new UnsupportedOperationException(where + "latent or chunk experts stay on the device;"
                    + " neuralfree.cpu-moe / neuralfree.n-cpu-moe / neuralfree.override-tensor /"
                    + " neuralfree.moe.cache host the plain routed experts only");
        }
        if (moe.hashRouting()) {
            throw new UnsupportedOperationException(where + "hash routing (ffn_gate_tid2eid) runs on one device;"
                    + " the hosted path routes with the MoE spec");
        }
        if (wiring.selection != MoeBlocks.Selection.SPEC) {
            throw new UnsupportedOperationException(where + "biased-logit selection runs on one device;"
                    + " the hosted path routes with the MoE spec");
        }
        if (wiring.router != MoeBlocks.RouterInput.FFN_INPUT || wiring.experts != MoeBlocks.ExpertInput.FFN_INPUT) {
            throw new UnsupportedOperationException(where + "the hosted path routes and reads the experts on the"
                    + " normalized FFN input; this architecture routes on " + wiring.router + " and reads "
                    + wiring.experts);
        }
        if (moe.weightBeforeFfn()) {
            throw new UnsupportedOperationException(where + "weight-before-FFN experts (llama4) are not part of"
                    + " the hosted path");
        }
    }

    @Override
    public void close() {
        closeAll(router, chunkRouter, routed, chunk, routerBias, routerScale, sharedSum, zeroMask, zeroApply);
    }

    @Override
    public String toString() {
        return "GpuMoeFfn{layer=" + layer.il() + ", kind=" + kind + ", " + spec + (wiring != null ? ", " + wiring : "")
                + '}';
    }

    // ================================================================== helpers

    private long rows(long width) {
        return Math.multiplyExact((long) ubatch, width);
    }

    private long[] local1() {
        return new long[]{ops.local()};
    }

    private GpuWeight f32Vector(GpuWeight v, int n) {
        if (v.type() != GGMLType.F32 || !v.isSingleShard() || v.rows() * v.cols() != n) {
            throw new IllegalStateException("layer " + layer.il() + ": " + v.name() + " must be one F32 buffer of "
                    + n + " values, got " + v);
        }
        return v;
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

    // ================================================================== routed experts over the cache

    /** The routed experts computed over a {@link MoeExpertCache}; the block owns the pass's lifetime. */
    private static final class CachedRouted implements Routed {
        private final GpuCachedMoePass pass;

        CachedRouted(GpuCachedMoePass pass) {
            this.pass = pass;
        }

        @Override
        public void run(int n, GpuBuffer x, int xOff, GpuBuffer dst, int flags) {
            pass.run(n, x, xOff, dst, flags);
        }

        @Override
        public void close() {
            pass.close();
        }

        @Override
        public String toString() {
            return pass.toString();
        }
    }

    // ================================================================== router

    /**
     * {@code LLM_MoE_Router} over the shared logits, preceded on deepseek4 hash layers by {@code LLM_MoE_HashIds}
     * and for {@link MoeBlocks.Selection#BIASED_LOGITS} by a raw-logit top-k over {@code logits + exp_probs_b}
     * (a copy of the logits, {@code LLM_AddId}, {@code LLM_MoE_Router} with {@code sel_mode} 1); in both cases the
     * final router call takes the ids as given ({@code sel_mode} 3) and computes their weights. On a
     * {@link MoeSpec#mediaRouting()} layer the step's embedding rows then take the {@link GpuMediaRouter} selection.
     */
    final class Router implements AutoCloseable {
        final MoeRouting.Params params;
        final ScratchSlot ids;
        final ScratchSlot weights;
        final MoeExpertRemap.Layer remap;
        final ScratchSlot physicalIds;
        PreparedDispatch remapDispatch;
        int lastRemapN = -1;
        final int routerLocal;
        final GpuRouterHeat heat;
        GpuMediaRouter media;
        PreparedDispatch hash;
        ScratchSlot selection;
        PreparedDispatch selectionBias;
        PreparedDispatch preselect;
        PreparedDispatch select;
        int lastN = -1;

        /**
         * @param probBias     bind {@code exp_probs_b} as the PROBS selection bias of {@code params}
         * @param biasedLogits pick the ids by {@code logits + exp_probs_b} first; {@code params} must be preselected
         * @param heat         recorder the selected ids are read back into asynchronously, or null for none
         * @param mediaRows    route the step's embedding rows by the layer's media routing ({@link GpuMediaRouter})
         */
        Router(MoeRouting.Params params, GpuWeights w, boolean probBias, boolean biasedLogits, MoeExpertHeat heat,
               boolean mediaRows, MoeExpertRemap.Layer remap) {
            this.params = params;
            this.remap = remap;
            GpuScratch scratch = ctx.scratch();
            this.ids = scratch.temp(T_IDS, rows(params.k()));
            this.weights = scratch.temp(T_WEIGHTS, rows(params.k()));
            this.physicalIds = remap == null ? null : scratch.temp(T_EXPERT_IDS, rows(params.k()));
            EplbLoadWindow window = ctx.moeLoadWindow();
            this.heat = heat != null || window != null
                    ? new GpuRouterHeat(ctx.queue(), heat, window, layer.weightLayer(), layer.il())
                    : null;
            int needed = ceilPow2((params.nExpert() + 31) / 32);
            int preferred = floorPow2(Math.max(ops.local(), MoeKernels.MIN_ROUTER_LOCAL));
            this.routerLocal = Math.min(MoeKernels.MAX_LOCAL, Math.max(needed, preferred));
            MoeKernels.checkRouter(params, routerLocal);
            if (biasedLogits && params.selMode() != MoeKernels.SEL_PRESELECTED) {
                throw new IllegalArgumentException("biased-logit selection needs preselected weight params, got "
                        + params);
            }
            try {
                if (biasedLogits) {
                    prepareBiasedSelection(w, scratch);
                } else if (params.selMode() == MoeKernels.SEL_PRESELECTED) {
                    hash = prepareHash(w);
                }
                GpuBuffer bias = probBias ? expProbsB(w) : ctx.placeholders().readOnlyFloats();
                select = ops.prepare(MoeKernels.ROUTER, MoeKernels.routerGlobal(1, routerLocal),
                        new long[]{routerLocal}, MoeKernels.routerArgs(params, 1, logits.buffer(), bias,
                                ids.buffer(), weights.buffer()));
                if (remap != null) {
                    remapDispatch = ops.prepare(MoeKernels.EPLB_MAP, MoeKernels.elementGlobal(params.k(), ops.local()),
                            local1(), MoeKernels.eplbMapArgs(1, params.k(), remap.logical(), remap.maxReplicas(),
                                    ids.buffer(), remap.replicaCount(), remap.logicalToPhysical(),
                                    physicalIds.buffer()));
                }
                if (mediaRows) {
                    media = new GpuMediaRouter(ctx, MoeRouting.Params.forMedia(spec), logits, mediaBias(w), ids,
                            weights, routerLocal);
                }
            } catch (RuntimeException e) {
                throw Closeables.close(this, e);
            }
            lastN = 1;
        }

        /** kolibri1 (kolibri1.cpp:134-143): ids = top-k of {@code logits + exp_probs_b} by raw value. */
        private void prepareBiasedSelection(GpuWeights w, GpuScratch scratch) {
            int nExpert = params.nExpert();
            selection = scratch.temp(T_SELECTION, rows(nExpert));
            selectionBias = ops.prepare(MoeKernels.ADD_ID, MoeKernels.elementGlobal(nExpert, ops.local()), local1(),
                    MoeKernels.addIdArgs(1, nExpert, zeroIds(scratch).buffer(), expProbsB(w), selection.buffer()));
            MoeRouting.Params raw = new MoeRouting.Params(nExpert, params.k(), params.gating(), 1, 1, false, 1.0f,
                    false, MoeKernels.SEL_RAW_LOGITS, 0, false);
            preselect = ops.prepare(MoeKernels.ROUTER, MoeKernels.routerGlobal(1, routerLocal),
                    new long[]{routerLocal}, MoeKernels.routerArgs(raw, 1, selection.buffer(),
                            ctx.placeholders().readOnlyFloats(), ids.buffer(), weights.buffer()));
        }

        /** deepseek4 hash layer: validates {@code ffn_gate_tid2eid} once and prepares {@code LLM_MoE_HashIds}. */
        private PreparedDispatch prepareHash(GpuWeights w) {
            GpuWeight table = w.layer(layer, TensorNames.weight(TensorNames.FFN_GATE_TID2EID));
            int k = params.k();
            if (table.type() != GGMLType.I32 || !table.isSingleShard() || table.cols() != k) {
                throw new IllegalStateException("layer " + layer.il() + ": " + table.name() + " must be one I32"
                        + " buffer of [" + k + ", n_vocab], got " + table);
            }
            int nVocab = Math.toIntExact(table.rows());
            byte[] raw = new byte[Math.multiplyExact(Math.multiplyExact(nVocab, k), 4)];
            GpuEvent read = ctx.queue().enqueueReadBuffer(table.shard(0), 0L, raw.length, raw);
            ctx.queue().finish();
            if (read != null) {
                read.close();
            }
            int[] tid2eid = new int[nVocab * k];
            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(tid2eid);
            MoeCombine.validateHashTable(tid2eid, k, nVocab, params.nExpert());
            ScratchSlot tokens = ctx.scratch().slot(ScratchSlots.TOKEN_IDS);
            if (tokens.offset() != 0) {
                throw new IllegalStateException("the token-id slot must start at element 0 for LLM_MoE_HashIds");
            }
            return ops.prepare(MoeKernels.HASH_IDS, MoeKernels.elementGlobal(k, ops.local()), local1(),
                    MoeKernels.hashIdsArgs(1, k, tokens.buffer(), table.shard(0), ids.buffer()));
        }

        void run(int n) {
            if (lastN != n) {
                long[] global = MoeKernels.routerGlobal(n, routerLocal);
                if (hash != null) {
                    hash.setInt(MoeKernels.ARG_N_TOKENS, n);
                    hash.setGlobal(MoeKernels.elementGlobal((long) n * params.k(), ops.local()));
                }
                if (preselect != null) {
                    selectionBias.setInt(MoeKernels.ARG_ADD_ID_N_SEL, n);
                    selectionBias.setGlobal(MoeKernels.elementGlobal((long) n * params.nExpert(), ops.local()));
                    preselect.setInt(MoeKernels.ARG_N_TOKENS, n);
                    preselect.setGlobal(global);
                }
                select.setInt(MoeKernels.ARG_N_TOKENS, n);
                select.setGlobal(global);
                lastN = n;
            }
            if (hash != null) {
                hash.dispatch();
            }
            if (preselect != null) {
                ops.copy(logits.buffer(), logits.offset(), selection.buffer(), selection.offset(),
                        n * params.nExpert());
                selectionBias.dispatch();
                preselect.dispatch();
            }
            select.dispatch();
            if (media != null) {
                media.apply(n);
            }
            if (remapDispatch != null) {
                if (lastRemapN != n) {
                    remapDispatch.setInt(MoeKernels.ARG_EPLB_N_PAIRS, n * params.k());
                    remapDispatch.setGlobal(MoeKernels.elementGlobal((long) n * params.k(), ops.local()));
                    lastRemapN = n;
                }
                remapDispatch.dispatch();
            }
            if (heat != null) {
                heat.note(ids, n * params.k());
            }
        }

        /** The routed pair ids the expert kernels address: physical replica slots, or the logical ids. */
        GpuBuffer expertIds() {
            return remap == null ? ids.buffer() : physicalIds.buffer();
        }

        @Override
        public void close() {
            RuntimeException failure = heat != null ? heat.drain() : null;
            failure = Closeables.close(media, failure);
            failure = Closeables.close(hash, failure);
            failure = Closeables.close(selectionBias, failure);
            failure = Closeables.close(preselect, failure);
            failure = Closeables.close(select, failure);
            failure = Closeables.close(remapDispatch, failure);
            if (failure != null) {
                throw failure;
            }
        }
    }

}
