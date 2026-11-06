package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuMatmulTap;
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
import java.util.List;
import java.util.Objects;
import java.util.function.IntFunction;

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
    static final String T_INPUT = "moe.input";
    static final String T_WEIGHTED_IN = "moe.weighted_in";
    static final String T_GATE = "moe.gate";
    static final String T_UP = "moe.up";
    static final String T_ACT = "moe.act";
    static final String T_EXPERT_OUT = "moe.expert_out";
    /** Scratch prefix of the expert sort ({@link GpuGroupedRows}). */
    static final String T_ROWS = "moe";
    static final String T_CHUNK_OUT = "moe.chunk_out";
    static final String T_SUM = "moe.sum";
    static final String T_SHEXP_GATE = "moe.shexp_gate";
    /**
     * Prefix of the interleaved gate/up expert scales ({@code SCALE_GATE_UP_PAIRS}), one temporary per gate tensor
     * name, written once at bind and never shared with another tensor.
     */
    static final String T_SCALE_PAIRS = "moe.scale_pairs.";
    /** Per-token weight sum of the selected identity experts. */
    static final String T_ZERO_WEIGHTS = "moe.zero_weights";
    /** Physical replica ids the expert kernels address when the layer remaps its routed experts. */
    static final String T_EXPERT_IDS = "moe.expert_ids";

    /** {@code x} of {@code LLM_MoE_ZeroApply}. */
    private static final int ARG_ZERO_X = 4;
    /** {@code dst} of {@code LLM_MoE_ZeroApply}. */
    private static final int ARG_ZERO_DST = 6;

    /** {@code x} of {@code LLM_MoE_ScaleInput(i n_tokens, i n_embd, i k, R x, R weights, W out)}. */
    private static final int ARG_SCALE_INPUT_X = 3;
    /**
     * {@code flags} of
     * {@code LLM_MoE_WeightedSum(i n_tokens, i n_embd, i k, i flags, R expert_out, R weights, RWn dst)}.
     */
    private static final int ARG_WSUM_FLAGS = 3;
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
     * through {@link ExpertPass}.
     */
    private interface Routed extends AutoCloseable {
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
            routed = new ExpertPass(expertPlan, w, router, spec.weightBeforeFfn(),
                    spec.postExpertNorm() ? w.layer(layer, TensorNames.weight(TensorNames.FFN_POST_EXP_NORM)) : null,
                    ctx.spec().hparams().postNormEps());
        }
        if (spec.chunked()) {
            chunkRouter = new Router(MoeRouting.Params.forChunks(spec), w, false, false, ctx.heat(), false, null);
            chunk = new ExpertPass(MoeExpertWeights.chunks(spec, nEmbd), w, chunkRouter, false, null, 0f);
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

    /** Work-group size of the one-group kernels ({@code LLM_MoE_Bucketize}): a power of two. */
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

    // ================================================================== experts

    /**
     * One expert set over the ids and weights of a {@link Router}: the one-token path ({@code {T}_MatVecId*}) and
     * the batched path ({@code LLM_MoE_Bucketize}, then {@code {T}_GroupedGemm} over the aligned rows of
     * {@code LLM_MoE_AlignBlocks} when {@link #grouped} holds, else {@code {T}_BatchMatMulId}), each a fixed list of
     * prepared dispatches whose token counts, global sizes, input view and destination are re-set per step.
     */
    private final class ExpertPass implements Routed {
        final MoeExpertWeights plan;
        final MoeExpertWeights.Gpu w;
        final Router src;
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

        ExpertPass(MoeExpertWeights plan, GpuWeights weights, Router src, boolean weightBefore, GpuWeight postNorm,
                   float postEps) {
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
    }

}
