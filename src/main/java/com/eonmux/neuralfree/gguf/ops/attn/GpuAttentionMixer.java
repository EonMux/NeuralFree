package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.encdec.T5RelativeBias;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.DualChunkRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DualChunkSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Softmax-attention mixer on the GPU ({@code MixerKind.ATTENTION} and deci's {@code ATTENTION_LINEAR_REPLACE}),
 * following llama.cpp {@code build_qkv} + {@code build_attn} (src/llama-graph.cpp:1807-1912, 2946-3100, 3115-3290).
 *
 * <p>Per step, for {@code n} tokens of the normalized input {@code in}:
 * <ol>
 *   <li>projections ({@link GpuQkvProjections}, shaped by {@link QkvProjection}): Q, K and V land densely in the {@code Q}, {@code K} and {@code V}
 *       scratch slots ({@code [t][n_head * d]}); a fused {@code attn_qkv} product is split there with
 *       {@code LLM_Copy2D_F32}, as is V from the raw K projection when the layer has no {@code attn_v}
 *       ({@link QkvProjection#vFromK()}); on a MoVA layer V is the weighted sum of routed value experts
 *       ({@link GpuValueExperts}); biases with {@code LLM_Batch_BiasAdd}; the KQV clamp with
 *       {@code LLM_Unary CLAMP};
 *       an interleaved output gate is split off Q; other gates are projected from {@code in}; limite's value
 *       embeddings ({@link AttentionSpec#valueEmbeddings()}) add the token's {@code value_embd} row, gated per KV
 *       head by {@code sigmoid(attn_ve_gate @ in)} when the layer carries the gate, to V with
 *       {@value AttnExtKernels#VALUE_EMBEDDINGS} ({@link GpuValueEmbeddings});</li>
 *   <li>Q/K norms of every {@link QkNormKind} ({@link GpuNormRopeStages}; geometry {@link QkNormGeometry}: per head,
 *       per half head, or the whole
 *       projection; their biases unless {@link AttentionSpec#qkNormBias()} is off) and gemma's weightless V norm with
 *       {@code LLM_RowNorm} over the dense head rows;</li>
 *   <li>RoPE with {@code LLM_RoPE_Ext} (NoPE layers skip it); the Q/K norms run after it instead when
 *       {@link AttentionSpec#qkNormAfterRope()} (hunyuan, llama4, ...); then the attention temperature
 *       ({@code LLM_AttnTempScale});</li>
 *   <li>with a cache ({@link AttentionSpec#usesMemory()}): the quantized-KV Hadamard rotation of Q, K and V
 *       ({@link GpuQuantizedKvRotation}), {@code LLM_KV_Store_<type>} of K and V at {@code cell_idx} (only on the layer
 *       that owns the cache; a shared-KV layer reads its owner's {@link GpuKvLayer}, the store steps in
 *       {@link GpuKvStoreStages}), and flash attention over the
 *       cache rows with {@code kv_pos} visibility; without a cache (encoders) flash attention reads K and V of the
 *       current batch with {@code kv_pos = q_pos};</li>
 *   <li>flash attention ({@link GpuFlashAttention}, planned by {@link FlashPlan}, its arguments packed by
 *       {@link FlashKernelArgs}, the paged launches in {@link GpuPagedFlashAttention}): flash decode / tiled prefill
 *       per {@link FlashAttnTiling} with {@code smemFloats = LlmKernels.faSmemFloats()}, the per-row decode
 *       fallback ({@link FlashAttnTiling.Plan#PER_ROW_DECODE}, over a paged cache {@link PagedRowFallback}),
 *       split-K partials merged by
 *       {@code LLM_FlashAttn_Combine}; on a device that runs the tensor-core attention
 *       ({@code LlmKernels.flashMmaPath()}) and F16 caches the prefill runs the mma kernel
 *       ({@link FlashAttnTiling.Plan#MMA}, {@code LLM_FlashAttnMma*}); sinks, softcap, ALiBi and the T5 relative
 *       bias are kernel arguments; K and V caches of different types use the mixed-type flash kernels
 *       ({@link AttnKernels#flashDecode(KvDType, KvDType)}); the static sinks are bound by {@link GpuStaticSinks};</li>
 *   <li>output ({@link GpuOutputStages}): the V rotation undone, muse-spark's weightless per-head RMS norm
 *       ({@code LLM_RowNorm}), limite's
 *       exclusive self attention ({@link AttentionSpec#xsaEps()}, {@value AttnExtKernels#XSA}) that removes each
 *       head's projection on the token's own value, the output
 *       gate ({@code LLM_AttnOutGate}), bitnet's sub-norm, {@code wo} (with
 *       its {@code .scale}), its bias, and the value scale ({@code f_attn_value_scale}, models/mimo2.cpp:174-176,
 *       models/dflash.cpp:992-993), written to {@code out}.</li>
 * </ol>
 * Every projection ({@code attn_qkv}, {@code attn_q}/{@code k}/{@code v} also as parts of a fused product,
 * {@code attn_gate}, {@code wo}) adds the {@link GpuProjectionDeltas} registered on it (LoRA) right after its base
 * product and before its bias, as {@code build_lora_mm} does (src/llama-graph.cpp:1629-1667).
 * The linear replacement is {@code out = wo @ in} (models/deci.cpp:114-117).
 *
 * <p>With {@code neuralfree.kv.offload=false} ({@code -nkvo}) and the layer's cache in host RAM the mixer binds
 * without a device cache: {@link #runWithHostAttention} runs the same projections, norms and RoPE, leaves the Q/K/V
 * rows to a host attention over the host cache, then runs the output part, and
 * {@link #projectKvForHostAttention} runs the cache writes alone. The device flash attention, KV store and cache
 * rotation do not run; the host stage ({@link GpuHostKvStage}) applies the rotation and store.</p>
 *
 * <p>Positions: with a cache the mixer reads {@link GpuStepInputs} ({@code qPos}, {@code ropePos}, {@code kv_pos},
 * {@code cell_idx}), which the executor uploads once per step before the first layer; without a cache it reads the
 * {@link ScratchSlots#POS} slot, which the executor fills with the step's positions in the same layout
 * ({@code pos[t + k * nTokens]}, section 0 = {@code pos0 + t}).</p>
 *
 * <p>Under {@code neuralfree.gpu.fused} the norm, rotation and K-row store launches of the list above collapse into
 * {@link FusedKernels}: {@link FusedKernels#RMS_NORM_ROPE} or the persistent multi-task kernel for the Q and K norms
 * with the V norm, and {@link FusedKernels#NORM_ROPE_STORE} or {@link FusedKernels#ROPE_STORE} for the K row written
 * straight to the cache. The unfused kernels remain the fallback where a fused kernel is unavailable or the cache
 * layout is not one the fused store covers.</p>
 *
 * <p>Every kernel this mixer launches itself is a {@link PreparedDispatch} built in {@link #bind}; a step re-sets the
 * token count, positions and key range only. Projections go through {@link GpuOps#matmul}, which picks the kernel
 * for the weight type and stages offsets. The mixer owns the {@code Q}, {@code K}, {@code V} and {@code ATTN_OUT}
 * slots and its {@code attn.*} temporaries for the duration of a call; {@code out} must be none of them. Not
 * thread-safe.</p>
 */
public final class GpuAttentionMixer implements GpuMixer {

    /** {@code data[s * dim + d] += bias[d]} ({@code kernels/llm/eltwise.cl}); arguments
     * {@code (dim, seq_len, bias, data)}. */
    public static final String BIAS_ADD = GpuOps.BATCH_BIAS_ADD;
    private static final int ARG_BIAS_SEQ = 1;
    private static final int ARG_COPY_ROWS = 0;

    private final List<TokenStep> inputSteps = new ArrayList<>();
    /** The subset of {@link #inputSteps} that feeds the stored K and V rows (the cache-write-only step). */
    private final List<TokenStep> kvInputSteps = new ArrayList<>();
    private final List<TokenStep> outputSteps = new ArrayList<>();

    private GpuForwardContext ctx;
    private GpuOps ops;
    private GpuMixerResources resources;
    private LayerSpec layer;
    private AttentionSpec spec;
    private QkvProjection proj;
    private int local;
    private int ubatch;
    private boolean bound;
    private boolean closed;

    private GpuWeight wo;

    private GpuProjectionDeltas.Site outSite;

    private GpuBuffer qBuf;
    private GpuBuffer kBuf;
    private GpuBuffer vBuf;
    private GpuBuffer attnBuf;

    private GpuKvLayer kvLayer;
    private GpuStepInputs stepInputs;
    /** True when the layer's KV cache is in host RAM and its attention runs on the host ({@code -nkvo}). */
    private boolean hostKv;
    private GpuQuantizedKvRotation rotation;
    /** The K/V half of {@link #rotation}, for the cache-write-only step. */
    private GpuQuantizedKvRotation storeRotation;
    private GpuFlashAttention flash;
    /** The hook that sees each contiguous step's raw head outputs; null for a plain attention. */
    private AttentionTap tap;
    /** The per-row log-sum-exp a {@link #tap} reads; null without a tap. */
    private GpuBuffer lseBuf;

    /** Dual-chunk attention of the layer (Qwen2.5-1M), or null. */
    private DualChunkSpec dualChunk;
    private GpuDualChunkAttention dualChunkAttention;
    /** The dual-chunk query planes, their positions and the original-context scaling. */
    private GpuDualChunkPlanes dcPlanes;
    /** IQuestLoopCoder's read-only first-pass cache and its flash attention; null without dual attention. */
    private GpuKvLayer dualKvLayer;
    private GpuFlashAttention dualFlash;
    private GpuBuffer loopGateW;
    private GpuBuffer loopGateB;
    private GpuBuffer dualAttnBuf;
    private TokenStep loopMixStep;
    private GpuAttentionShift shift;
    private T5RelativeBias.Device t5Bias;
    /** The TurboQuant query rotation ({@code LLM_KvTqRotateQ}) over the projected rows, and its output buffer. */
    private TokenStep tqRotate;
    private GpuBuffer qRot;
    /** The FP8 KV-cache scales of the layer's cache, device and host. */
    private GpuKvScales kvScales;
    /** The static sink tokens: the learned {@code attn_sinks} and openPangu's sink keys and values. */
    private GpuStaticSinks sinks;
    /** limite's value embeddings and exclusive self attention. */
    private GpuValueEmbeddings valueEmbeddings;
    /** The per-use Q/K/V adapters of a shared attention block (Zamba2); null without them. */
    private GpuQkvAdapters adapters;
    /** The Q/K/V norms, the RoPE rotation and their fused forms. */
    private GpuNormRopeStages normRope;
    /** The cache stores of K and V. */
    private GpuKvStoreStages kvStore;
    /** The layer's Q/K/V projections: weights, LoRA sites, split steps and the per-step passes. */
    private GpuQkvProjections projections;
    /** The host-KV halves ({@code -nkvo}): the projection-only passes and the row copies for a host attention. */
    private GpuHostKvStage hostKvStage;
    /** Inkling's relative projection, short convolutions and log scaling; null on a plain layer. */
    private GpuInklingStages inkling;
    /** The output stages between the attention rows and the {@code wo} projection. */
    private GpuOutputStages output;
    /** The step's query-position buffer the attention and the Inkling steps read. */
    private GpuBuffer posBuffer;

    /** An unbound mixer; {@link #bind} reads the layer's attention spec. */
    public GpuAttentionMixer() {
    }

    /**
     * Installs the hook that sees every contiguous step's raw head outputs, before the output stages. The mixer then
     * captures the per-row log-sum-exp the hook receives; the layer must be plain softmax attention without
     * dual-chunk or dual attention.
     *
     * @throws IllegalStateException when the mixer is already bound
     * @throws NullPointerException  when {@code tap} is null
     */
    public void setAttentionTap(AttentionTap tap) {
        if (bound) {
            throw new IllegalStateException("the attention mixer of layer " + layer.il() + " is already bound");
        }
        this.tap = Objects.requireNonNull(tap, "tap");
    }

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException      when the layer has no attention spec or a tensor has the wrong shape
     * @throws GpuModelUnsupportedException when the attention geometry fits no GPU flash kernel
     *                                      ({@link FlashBindGuard#requireServable})
     * @throws IllegalStateException         when the layer uses a cache but the context has no memory, or the
     *                                       well-known slots are not offset-0 buffers
     * @throws UnsupportedOperationException for a latent cache, or value experts on a fused projection or of a type
     *                                       without expert kernels
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        if (bound) {
            throw new IllegalStateException("attention mixer of layer " + this.layer.il() + " is already bound");
        }
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.layer = Objects.requireNonNull(layer, "layer");
        this.ops = ctx.ops();
        this.resources = new GpuMixerResources(ops, ctx.backend());
        this.local = ops.local();
        this.ubatch = ctx.scratch().ubatch();
        this.hostKv = ctx.hostKv(layer.il()) && layer.usesKv();
        this.proj = QkvProjection.of(layer, s -> weights().layerOrNull(layer, s) != null, ctx.spec().arch());
        this.spec = layer.attention();
        this.hostKvStage = new GpuHostKvStage(this);
        if (layer.mixerKind() == MixerKind.INKLING && hostKv) {
            throw new UnsupportedOperationException("layer " + layer.il() + ": Inkling's short convolutions keep"
                    + " their state on the device and feed the device KV cache; run without host KV offload");
        }
        try {
            if (proj.source() == QkvProjection.Source.NONE) {
                wo = weights().layer(layer, proj.outWeight());
                outSite = serveDeltas(proj.outWeight(), wo);
                shift = GpuAttentionShift.none();
            } else {
                bindAttention();
            }
        } catch (RuntimeException e) {
            releaseAll();
            throw e;
        }
        bound = true;
    }

    GpuWeights weights() {
        return ctx.weightSource().weightsFor(layer.weightLayer());
    }

    private void bindAttention() {
        GpuScratch scratch = ctx.scratch();
        this.dualChunk = spec.dualChunk();
        if (tap != null) {
            if (dualChunk != null || spec.dual()) {
                throw new UnsupportedOperationException("layer " + layer.il() + " runs dual-chunk or dual attention,"
                        + " whose output stages a context-parallel merge does not cover");
            }
            if (hostKv) {
                throw new UnsupportedOperationException("layer " + layer.il() + " keeps its KV cache on the host;"
                        + " context-parallel attention needs the device cache");
            }
        }
        if (dualChunk != null && hostKv) {
            throw new UnsupportedOperationException("layer " + layer.il() + " runs dual-chunk attention with a host"
                    + " KV cache; run without host KV offload");
        }
        qBuf = offsetFree(scratch.slot(ScratchSlots.Q), (long) ubatch * spec.qRopeDim());
        attnBuf = offsetFree(scratch.slot(ScratchSlots.ATTN_OUT), (long) ubatch * spec.outDim());
        if (tap != null) {
            lseBuf = scratch.temp("cp.attn.lse", (long) ubatch * spec.nHead()).buffer();
        }
        if (proj.projectsKv()) {
            kBuf = offsetFree(scratch.slot(ScratchSlots.K), (long) ubatch * spec.kDim());
            vBuf = offsetFree(scratch.slot(ScratchSlots.V), (long) ubatch * spec.vDim());
        }
        GpuKvLayer kv = bindCache();
        kvScales = new GpuKvScales(this);
        kvScales.bind(kv, hostKv);
        boolean cached = kv != null || hostKv;
        GpuBuffer qPos = cached ? stepInputs.qPos() : offsetFree(scratch.slot(ScratchSlots.POS), (long) ubatch * 4);
        this.posBuffer = qPos;
        GpuBuffer ropePos = cached ? stepInputs.ropePos() : qPos;
        adapters = new GpuQkvAdapters(this);
        normRope = new GpuNormRopeStages(this);
        kvStore = new GpuKvStoreStages(this);
        output = new GpuOutputStages(this);
        projections = new GpuQkvProjections(this);
        projections.bind(scratch);
        adapters.bind(scratch);
        if (layer.mixerKind() == MixerKind.INKLING) {
            inkling = new GpuInklingStages(this);
        }
        valueEmbeddings = new GpuValueEmbeddings(this);
        if (dualChunk != null) {
            dcPlanes = new GpuDualChunkPlanes(this);
            dcPlanes.bindBuffers();
        }
        // Inkling's log scaling runs on the normed queries and its post norms follow RoPE, so its norms, rotation
        // and K-row store stay separate steps. Dual-chunk attention rotates five query planes and scales the K rows
        // before the store, so it keeps the unfused path too.
        String fusedKStore = inkling != null || dualChunk != null ? null : kvStore.fusedKernel(kv);
        boolean fused = inkling == null && dualChunk == null && !spec.qkNormAfterRope()
                && normRope.bindFusedNormRope(ropePos, fusedKStore != null);
        if (!fused) {
            if (!spec.qkNormAfterRope()) {
                normRope.bindNorms(fusedKStore != null);
            }
            normRope.bindVNorm();
            normRope.bindRope(ropePos, fusedKStore != null);
        }
        sinks = new GpuStaticSinks(this);
        sinks.bind();
        if (spec.qkNormAfterRope()) {
            normRope.bindNorms(false);
        }
        if (inkling != null) {
            inkling.bindPostNorms();
        }
        if (spec.temperature().enabled()) {
            input(step(AttnExtKernels.TEMP_SCALE,
                    AttnExtKernels.tempScaleArgs(spec.temperature(), ubatch, spec.nHead(), spec.headDimK(), qBuf, 0,
                            spec.qDim(), qPos),
                    AttnExtKernels.ARG_TEMP_N_TOKENS, 1, (long) spec.qDim(), 0, -1), false);
        }
        if (kv != null) {
            rotation = dualChunk != null
                    ? GpuQuantizedKvRotation.ofPlanes(ops, kv.rotation(), qBuf, spec.qDim(), DualChunkRope.VARIANTS,
                            dcPlanes.planePitch(), kBuf, spec.kDim(), vBuf, spec.vDim(), attnBuf, spec.outDim(),
                            ubatch)
                    : GpuQuantizedKvRotation.of(ops, kv.rotation(), qBuf, spec.qDim(), kBuf, spec.kDim(), vBuf,
                            spec.vDim(), attnBuf, spec.outDim(), ubatch);
            if (proj.projectsKv()) {
                storeRotation = GpuQuantizedKvRotation.forStore(ops, kv.rotation(), kBuf, spec.kDim(), vBuf,
                        spec.vDim(), ubatch);
                kvStore.bind(kv, fusedKStore, ropePos);
            }
        }
        if (hostKv) {
            sinks.bindHostLogits();
        } else if (dualChunk != null) {
            dualChunkAttention = bindDualChunkAttention(kv, qPos);
        } else if (spec.dual()) {
            bindDualFlash(kv, qPos);
        } else {
            flash = bindFlash(kv, qPos);
        }
        output.bindSteps();
        bindOutputWeights();
        output.bindProjection((int) wo.rows());
        shift = GpuAttentionShift.bind(ctx, layer, hostKv, kv, normRope.factors());
    }

    /**
     * The layer's KV cache and the memory's step inputs when the layer attends through a cache
     * ({@link AttentionSpec#usesMemory()}); null for a layer that attends only to the current batch.
     */
    private GpuKvLayer bindCache() {
        if (!spec.usesMemory()) {
            return null;
        }
        if (!ctx.hasMemory()) {
            throw new IllegalStateException("layer " + layer.il() + " attends through a KV cache but the model "
                    + ctx.spec().arch().ggufName() + " has no memory");
        }
        GpuMemory memory = ctx.memory();
        stepInputs = memory.stepInputs();
        if (hostKv) {
            // the cache lives in host RAM; the device memory allocates none for this layer
            return null;
        }
        GpuKvLayer kv = memory.kv(layer.il());
        kvLayer = kv;
        if (kv == null) {
            throw new IllegalStateException("layer " + layer.il() + " has no KV cache in the memory plan");
        }
        checkCache(kv.layout());
        return kv;
    }

    /**
     * The two flash attentions of IQuestLoopCoder and the per-head mix: the local one over the layer's own cache with
     * a sliding window ({@code attn_loop_gate}), and the global one read-only over the first pass's cache with the
     * layer's full causal window. The mix kernel combines them into the attention rows.
     */
    private void bindDualFlash(GpuKvLayer own, GpuBuffer qPos) {
        FlashBindGuard.requireServable(ctx.kernels(), "layer " + layer.il(), spec.nHead(), spec.nHeadKv(),
                spec.headDimK(), spec.headDimV());
        GpuMemory memory = ctx.memory();
        dualKvLayer = memory.kv(spec.dualReadLayer());
        if (dualKvLayer == null) {
            throw new IllegalStateException("layer " + layer.il() + " reads the cache of layer "
                    + spec.dualReadLayer() + ", which the memory plan does not hold");
        }
        if (dualKvLayer == own) {
            throw new IllegalStateException("layer " + layer.il() + " reads its own cache as the global attention");
        }
        checkCache(dualKvLayer.layout());
        AttentionSpec localSpec = spec.toBuilder().swaLayer(true).swaType(SwaType.STANDARD)
                .nSwa(spec.dualWindow()).build();
        FlashKeys localKeys = FlashKeys.cached(own, stepInputs.kvPos(own.group()));
        FlashKeys globalKeys = FlashKeys.cached(dualKvLayer, stepInputs.kvPos(dualKvLayer.group()));
        FlashPlan localPlan = FlashPlan.of(ctx.kernels(), localSpec, localKeys, null, ubatch);
        FlashPlan globalPlan = FlashPlan.of(ctx.kernels(), spec, globalKeys, null, ubatch);
        GpuBuffer partial = ctx.scratch().partials(
                (int) Math.max(localPlan.partialCapacity(), globalPlan.partialCapacity()));
        GpuBuffer placeholder = ops.placeholders().readOnlyFloats();
        GpuBuffer ones = ops.placeholders().kvScalesOnes();
        GpuBuffer fq = flashQuery(localKeys);
        flash = GpuFlashAttention.prepare(resources, ctx.queue(), localPlan,
                new FlashKernelArgs(localSpec, localKeys, fq, qPos, attnBuf, partial, sinks.logits(),
                        sinks.sinkKeys(), sinks.sinkValues(), relBias(),
                        kvScales.device() != null ? kvScales.device() : ones, placeholder), null, stepInputs, ubatch,
                null);
        dualAttnBuf = ctx.scratch().temp("attn.dual", (long) ubatch * spec.outDim()).buffer();
        dualFlash = GpuFlashAttention.prepare(resources, ctx.queue(), globalPlan,
                new FlashKernelArgs(spec, globalKeys, fq, qPos, dualAttnBuf, partial, sinks.logits(),
                        sinks.sinkKeys(), sinks.sinkValues(), relBias(), ones, placeholder), null, stepInputs, ubatch,
                null);
        loopGateW = vector(TensorNames.weight(TensorNames.ATTN_LOOP_GATE), (long) spec.nHead() * spec.headDimK());
        loopGateB = vector(TensorNames.bias(TensorNames.ATTN_LOOP_GATE), spec.nHead());
        loopMixStep = step(LoopAttnKernels.MIX,
                LoopAttnKernels.mixArgs(ubatch, spec.nHead(), spec.headDimK(), spec.qDim(), 0, spec.outDim(), 0,
                        qBuf, loopGateW, loopGateB, dualAttnBuf, attnBuf),
                LoopAttnKernels.ARG_N_TOKENS, 1, (long) spec.nHead() * spec.headDimK(), 0, -1);
    }

    /**
     * The layer's flash attention ({@link GpuFlashAttention}): over the cache rows of {@code kv} with
     * {@code kv_pos} visibility (and the paged launches of a paged memory), or over the current batch's K and V rows
     * with {@code kv_pos = q_pos} when {@code kv} is null.
     */
    private GpuFlashAttention bindFlash(GpuKvLayer kv, GpuBuffer qPos) {
        FlashBindGuard.requireServable(ctx.kernels(), "layer " + layer.il(), spec.nHead(), spec.nHeadKv(),
                spec.headDimK(), spec.headDimV());
        FlashKeys keys = kv != null ? FlashKeys.cached(kv, stepInputs.kvPos(kv.group()))
                : FlashKeys.batch(kBuf, spec.kDim(), vBuf, spec.vDim(), qPos, ubatch);
        PagedFlashGeometry pg = kv != null ? PagedFlashGeometry.of(stepInputs, kv.group(), ubatch) : null;
        FlashPlan plan = FlashPlan.of(ctx.kernels(), spec, keys, pg, ubatch);
        long capacity = plan.partialCapacity();
        if (spec.sinkTokens() > 0) {
            /* One extra split holds the static sink partial of the largest launch. */
            capacity += (long) ubatch * spec.nHead() * (spec.headDimV() + 2L);
        }
        GpuBuffer partial = ctx.scratch().partials((int) capacity);
        FlashKernelArgs args = new FlashKernelArgs(spec, keys, flashQuery(keys), qPos, attnBuf, partial,
                sinks.logits(), sinks.sinkKeys(), sinks.sinkValues(), relBias(),
                kvScales.device() != null ? kvScales.device() : ops.placeholders().kvScalesOnes(),
                ops.placeholders().readOnlyFloats());
        return GpuFlashAttention.prepare(resources, ctx.queue(), plan, args, pg, stepInputs, ubatch, lseBuf);
    }

    // ------------------------------------------------------------------ dual-chunk attention

    /**
     * The layer's dual-chunk stage launches ({@link GpuDualChunkAttention}): the flash plan of its cache, dense and
     * paged, with one {@code n_split = 1} partial buffer per stage and the merge kernel.
     */
    private GpuDualChunkAttention bindDualChunkAttention(GpuKvLayer kv, GpuBuffer qPos) {
        FlashBindGuard.requireServable(ctx.kernels(), "layer " + layer.il(), spec.nHead(), spec.nHeadKv(),
                spec.headDimK(), spec.headDimV());
        FlashKeys keys = FlashKeys.cached(kv, stepInputs.kvPos(kv.group()));
        if (keys.typeK().turboQuantMse()) {
            throw new UnsupportedOperationException("layer " + layer.il() + " runs dual-chunk attention over a "
                    + keys.typeK() + " key cache: its keys are stored in the Hadamard-rotated basis and the GPU"
                    + " dual-chunk stages score the unrotated query planes; use another --cache-type-k or the CPU backend");
        }
        PagedFlashGeometry pg = PagedFlashGeometry.of(stepInputs, kv.group(), ubatch);
        FlashPlan plan = FlashPlan.of(ctx.kernels(), spec, keys, pg, ubatch);
        /* The stage launches replace this buffer with their own partial, so the attention rows serve as the default. */
        FlashKernelArgs args = new FlashKernelArgs(spec, keys, qBuf, qPos, attnBuf, attnBuf, sinks.logits(),
                sinks.sinkKeys(), sinks.sinkValues(), relBias(),
                kvScales.device() != null ? kvScales.device() : ops.placeholders().kvScalesOnes(),
                ops.placeholders().readOnlyFloats());
        GpuDualChunkAttention attention = GpuDualChunkAttention.prepare(resources, ctx.queue(), plan, args, pg,
                ubatch, dcPlanes.planePitch(), dcPlanes.origPos());
        dcPlanes.attach(attention);
        return attention;
    }

    /**
     * The query rows the flash kernels read. A TurboQuant key cache stores the centroid indices of each key's
     * Hadamard rotation and attention scores the centroid vector, so the query is rotated by the same normalized
     * Sylvester matrix first ({@link RopeKernels#KV_TQ_ROTATE_Q}); every other cache reads the projected rows.
     */
    private GpuBuffer flashQuery(FlashKeys keys) {
        if (!keys.typeK().turboQuantMse()) {
            return qBuf;
        }
        if (tqRotate == null) {
            qRot = resources.allocate((long) ubatch * spec.qDim() * Float.BYTES);
            Object[] args = new Object[]{ubatch, spec.nHead(), spec.headDimK(), 0, spec.qDim(), 0, qBuf, qRot};
            tqRotate = step(RopeKernels.KV_TQ_ROTATE_Q, args, 0, 1, (long) spec.nHead() * local, 0, -1);
        }
        return qRot;
    }

    // ------------------------------------------------------------------ cache

    private void checkCache(KvLayout layout) {
        if (layout.vAliasesK()) {
            throw new UnsupportedOperationException("layer " + layer.il() + " has a latent (MLA) cache; the softmax"
                    + " attention mixer reads separate K and V rows");
        }
        if (layout.widthK() != spec.kDim() || layout.widthV() != spec.vDim()) {
            throw new IllegalArgumentException("layer " + layer.il() + ": cache rows of K " + layout.widthK() + " / V "
                    + layout.widthV() + " do not hold " + spec.nHeadKv() + " heads of (" + spec.headDimK() + ", "
                    + spec.headDimV() + ")");
        }
    }

    ModelHparams hparams() {
        return ctx.spec().hparams();
    }

    float rmsEps() {
        return hparams().normRmsEps();
    }

    // ------------------------------------------------------------------ output

    /** Loads the {@code wo} weight and its LoRA site, and guards the projection width against one step's limits. */
    private void bindOutputWeights() {
        wo = weights().layer(layer, proj.outWeight());
        QkvProjection.checkShape(wo.name(), wo.cols(), wo.rows(), spec.outDim(), wo.rows());
        outSite = serveDeltas(proj.outWeight(), wo);
        if (wo.rows() > Integer.MAX_VALUE / Math.max(1, ubatch)) {
            throw new IllegalArgumentException("output projection " + wo.name() + " is too wide for one step");
        }
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        run(pos, 1, in, out);
    }

    @Override
    public List<GpuStepPatcher> decodeReplayPatchers() {
        if (dualChunk != null || flash == null) {
            return stepReplayPatchers();
        }
        if (flash.routesDecodeByDepth()) {
            /* The mixer's route (decided by the main attention) holds for the dual attention too. */
            List<GpuStepPatcher> routed = new ArrayList<>(
                    DepthRoutedPatcher.of(flash, this::routesDecodeThroughPrefill));
            if (dualFlash != null) {
                routed.addAll(DepthRoutedPatcher.of(dualFlash, this::routesDecodeThroughPrefill));
            }
            return List.copyOf(routed);
        }
        if (routesDecodeThroughPrefill()) {
            return stepReplayPatchers();
        }
        return dualFlash == null
                ? List.of(flash.patcher())
                : List.of(flash.patcher(), dualFlash.patcher());
    }

    @Override
    public List<GpuStepPatcher> stepReplayPatchers() {
        if (dualChunk != null || flash == null) {
            /* The dense dual-chunk step never enqueues the flash group a patcher would claim. */
            return List.of();
        }
        return dualFlash == null
                ? List.of(flash.prefillPatcher())
                : List.of(flash.prefillPatcher(), dualFlash.prefillPatcher());
    }

    /** True when this mixer's one-row step runs the tensor-core prefill launch, the plan's fastest kernel. */
    private boolean routesDecodeThroughPrefill() {
        return flash != null && flash.routesDecodeThroughPrefill();
    }

    /** True when the one-row step at {@code pos0} runs the tensor-core prefill launch: the plan-level route plus, on
     * a depth-routed plan, at most one workgroup of live keys ({@code pos0 + 1} of a dense causal cache). */
    private boolean routesDecodeThroughPrefill(int pos0) {
        return flash != null && flash.routesDecodeThroughPrefill(pos0 + 1);
    }

    @Override
    public List<GpuStepPatcher> pagedDecodeReplayPatchers() {
        return dualChunk != null || flash == null ? null : flash.pagedPatchers();
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        run(pos0, nTokens, in, out);
    }

    /**
     * The cache writes of a step: the K and V projections (every part of a fused product, so the K/V rows are the
     * ones the whole step writes), their biases, clamps, norms and RoPE, the store rotation and the store. The Q
     * projection, its norms and RoPE, the attention, the output and the output projection do not run.
     */
    @Override
    public void prefillKV(int pos0, int nTokens, ScratchSlot in) {
        ensureBound();
        ensureDeviceKv();
        Objects.requireNonNull(in, "in");
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": step of " + nTokens + " tokens outside [1, "
                    + ubatch + "]");
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("layer " + layer.il() + ": negative position " + pos0);
        }
        if (proj.source() == QkvProjection.Source.NONE || !spec.usesMemory() || !proj.projectsKv()) {
            return;
        }
        checkStepInputs(pos0, nTokens);
        PagedStep paged = stepInputs != null ? stepInputs.currentPagedStep() : null;
        if (dualChunk != null) {
            if (paged != null) {
                dcPlanes.uploadKeyPositions(paged, nTokens);
            } else {
                dcPlanes.uploadKeyPositions(pos0, nTokens);
            }
        }
        projections.projectKv(nTokens, in);
        adapters.run(nTokens, in);
        runAll(kvInputSteps, nTokens, paged);
        if (dualChunk != null) {
            if (paged != null) {
                dcPlanes.scaleK(paged);
            } else {
                dcPlanes.scaleK(0, nTokens, dualChunk.scalingFactor(pos0 + nTokens));
            }
        }
        if (storeRotation != null) {
            storeRotation.beforeStore(nTokens);
        }
        runAll(kvStore.steps(), nTokens);
    }

    @Override
    public String kvRefusal() {
        ensureBound();
        return null;
    }

    // ================================================================== host KV

    /**
     * Why this mixer cannot run its attention over a host KV cache, or null when it can: only a mixer bound for a
     * host-KV layer can, and its attention must project its own rows ({@link AttentionSpec#usesMemory()}).
     */
    public String hostAttentionRefusal() {
        ensureBound();
        if (!hostKv) {
            return "layer " + layer.il() + " keeps its KV cache on the device";
        }
        if (proj.source() == QkvProjection.Source.NONE) {
            return "layer " + layer.il() + " projects no attention rows";
        }
        return null;
    }

    /** True when this mixer projects and stores K and V of its own; a shared-cache reader does not. */
    public boolean projectsKv() {
        ensureBound();
        return proj.projectsKv();
    }

    /** First context row this mixer's cache holds: the start of its context shard ({@link KvLayout#rowBase()}). */
    public int kvRowBase() {
        ensureBound();
        return kvLayer.layout().rowBase();
    }

    /** The per-KV-head FP8 K cache scales of a host-KV layer, or null for an unscaled cache. */
    public float[] hostKScale() {
        ensureBound();
        return kvScales.hostK();
    }

    /** The per-KV-head FP8 V cache scales of a host-KV layer, or null for an unscaled cache. */
    public float[] hostVScale() {
        ensureBound();
        return kvScales.hostV();
    }

    /** The per-head attention sinks of a host-KV layer, or null without sinks. */
    public float[] hostSinkValues() {
        ensureBound();
        return sinks.hostSinkLogits();
    }

    /** The normalized static sink keys of a host-KV layer {@code [n_sink][n_head_kv][d_k]}, or null. */
    public float[] hostSinkK() {
        ensureBound();
        return sinks.hostKeys();
    }

    /** The static sink values of a host-KV layer {@code [n_sink][n_head_kv][d_v]}, or null. */
    public float[] hostSinkV() {
        ensureBound();
        return sinks.hostValues();
    }

    /**
     * The K and V rows of a step on the device, for the host cache to store: the K/V projections, their biases,
     * clamps, norms and RoPE. The Q projection, the attention, the cache store and the output projection do not run.
     * A shared-cache layer projects none and leaves the arrays untouched.
     *
     * @param kOut {@code nTokens * kDim} values, or null for a layer that stores no cache
     * @param vOut {@code nTokens * vDim} values, or null for a layer that stores no cache
     */
    public void projectKvForHostAttention(int pos0, int nTokens, ScratchSlot in, float[] kOut, float[] vOut) {
        ensureBound();
        hostKvStage.projectKvForHostAttention(pos0, nTokens, in, kOut, vOut);
    }

    /**
     * One step whose attention runs on the host over the host KV cache: the device projections, norms and RoPE, the
     * caller's host attention over the copied Q/K/V rows ({@code q} is always projected; {@code k} and {@code v} are
     * null for a shared-cache layer), then the device output norm, gate, sub-norm, output projection and value scale
     * over the returned attention rows. The device cache store and rotation do not run; the host stage applies the
     * cache rotation and store.
     *
     * @param q    {@code nTokens * qDim} values
     * @param k    {@code nTokens * kDim} values, or null for a layer that stores no cache
     * @param v    {@code nTokens * vDim} values, or null for a layer that stores no cache
     * @param attn {@code nTokens * outDim} values the host attention writes
     */
    public void runWithHostAttention(int pos0, int nTokens, ScratchSlot in, ScratchSlot out, float[] q, float[] k,
                                     float[] v, float[] attn, HostAttention host) {
        ensureBound();
        hostKvStage.runWithHostAttention(pos0, nTokens, in, out, q, k, v, attn, host);
    }

    /** The host attention over the projected rows of one step. */
    @FunctionalInterface
    public interface HostAttention {
        /**
         * Stores the projected K/V rows in the host cache and writes the attention of the step's rows to
         * {@code out}. {@code k} and {@code v} are null for a shared-cache layer.
         */
        void attend(int pos0, int nTokens, float[] q, float[] k, float[] v, float[] out);
    }

    /**
     * One step over the rows of {@code s}: the contiguous step at {@code s.pos0(0)} when no paged step is current —
     * decode for one token with memory, prefill otherwise — the paged batched step through the memory's current
     * {@code PagedStep} otherwise, whose segments the flash kernels address through the step's page tables.
     *
     * @throws UnsupportedOperationException when the step is not contiguous and the memory holds no paged step
     */
    @Override
    public void batch(StepLayout s, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        Objects.requireNonNull(s, "s");
        PagedStep step = stepInputs != null ? stepInputs.currentPagedStep() : null;
        if (step == null) {
            if (!s.contiguousSingle()) {
                throw new UnsupportedOperationException(batchRefusal());
            }
            if (s.nTokens() == 1 && spec.usesMemory()) {
                run(s.pos0(0), 1, in, out);
            } else {
                run(s.pos0(0), s.nTokens(), in, out);
            }
            return;
        }
        runBatched(s, step, in, out);
    }

    /**
     * The batched pipeline: the projections, norms, RoPE and KV store run over the whole step (their per-row
     * inputs — positions, {@code cell_idx} — are the step inputs' device buffers), and the flash attention
     * dispatches the paged kernels: one launch over the step's decode rows, or one paged prefill per
     * multi-token segment. A bound attention tap receives the step's raw head outputs and log-sum-exp over all
     * its rows, as in the contiguous step.
     */
    private void runBatched(StepLayout s, PagedStep step, ScratchSlot in, ScratchSlot out) {
        ensureDeviceKv();
        if (stepTreeMask() != null) {
            String refusal = treeRefusal();
            if (refusal != null) {
                throw new UnsupportedOperationException(refusal);
            }
        }
        int n = s.nTokens();
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": step of " + n + " tokens outside [1, "
                    + ubatch + "]");
        }
        if (proj.source() == QkvProjection.Source.NONE) {
            ops.matmul(wo, in.buffer(), in.offset(), out.buffer(), out.offset(), n);
            addDeltas(outSite, n, in.buffer(), in.offset(), (int) wo.cols(), out.buffer(), out.offset(),
                    (int) wo.rows());
            return;
        }
        projections.project(n, in);
        adapters.run(n, in);
        if (dualChunk != null) {
            dcPlanes.uploadPositions(step, n);
        }
        runAll(inputSteps, n, step);
        if (dualChunk != null) {
            dcPlanes.scale(step);
        }
        if (rotation != null) {
            rotation.beforeStore(n);
        }
        runAll(kvStore.steps(), n);
        if (dualChunk != null) {
            dualChunkAttention.runPaged(step, kvLayer.group());
        } else {
            if (tqRotate != null) {
                tqRotate.run(n);
            }
            flash.batched(step, kvLayer.group());
        }
        if (tap != null) {
            tap.afterAttention(step.segPos0(0), n, attnBuf, lseBuf, spec.nHead(), spec.headDimV());
            if (!tap.ownsOutput()) {
                // a context-parallel peer contributes its partition and its projection would be discarded
                return;
            }
        }
        if (rotation != null) {
            rotation.afterAttention(n);
        }
        runAll(outputSteps, n);
        output.projectOutput(n, out);
    }

    private void run(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        ensureDeviceKv();
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": step of " + n + " tokens outside [1, "
                    + ubatch + "]");
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("layer " + layer.il() + ": negative position " + pos0);
        }
        if (proj.source() == QkvProjection.Source.NONE) {
            ops.matmul(wo, in.buffer(), in.offset(), out.buffer(), out.offset(), n);
            addDeltas(outSite, n, in.buffer(), in.offset(), (int) wo.cols(), out.buffer(), out.offset(),
                    (int) wo.rows());
            return;
        }
        checkStepInputs(pos0, n);
        prepareStep(pos0, n, in);
        if (!attendStep(pos0, n, null, null)) {
            // a context-parallel peer contributes its partition and its projection would be discarded
            return;
        }
        finishStep(n, out);
    }

    /**
     * Runs the projections, the adapters, the rope and the cache stores of {@code n} rows of a prepared step: the
     * half of {@link #run} between the step-input check and the attention launches. The rows are the step's own
     * packed rows of a prefill context-parallel step when the caller uploaded them through
     * {@code DeviceStepInputs.uploadPacked}. The caller runs {@link #attendStep} and {@link #finishStep} after.
     *
     * @throws UnsupportedOperationException when the running step carries a tree mask this layer's attention
     *                                       cannot take (a dual-chunk, dual-attention or position-bias stage)
     */
    public void prepareStep(int pos0, int n, ScratchSlot in) {
        ensureBound();
        ensureDeviceKv();
        Objects.requireNonNull(in, "in");
        if (n < 0 || n > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": step of " + n + " tokens outside [0, "
                    + ubatch + "]");
        }
        if (proj.source() == QkvProjection.Source.NONE) {
            throw new UnsupportedOperationException("layer " + layer.il() + " has no Q/K/V projections, so a"
                    + " prefill context-parallel step has no attention to exchange");
        }
        checkStepInputs(pos0, n);
        if (stepTreeMask() != null) {
            String refusal = treeRefusal();
            if (refusal != null) {
                throw new UnsupportedOperationException(refusal);
            }
        }
        if (n == 0) {
            // the rank owns no row of this step: nothing to project, rope or store
            return;
        }
        projections.project(n, in);
        adapters.run(n, in);
        if (dualChunk != null) {
            dcPlanes.uploadPositions(pos0, n);
        }
        runAll(inputSteps, n);
        if (dualChunk != null) {
            dcPlanes.scale(0, n, dualChunk.scalingFactor(pos0 + n));
        }
        if (rotation != null) {
            rotation.beforeStore(n);
        }
        runAll(kvStore.steps(), n);
        if (dualChunk == null && tqRotate != null) {
            tqRotate.run(n);
        }
    }

    /**
     * Runs the attention launches of a prepared step: the flash over the cache (or the dense batch attention), then
     * the layer's tap. When {@code q} and {@code qPos} are given they hold the whole step's query rows in the
     * query's global row order with their true positions (a prefill context-parallel attention over the gathered
     * rows), and the flash launches read them instead of the packed rows this mixer projected; the mixer's own
     * buffers are restored before the call returns. The launches apply the running step's tree mask when one is
     * bound (a tree verify step). Returns false when a tap took the attention output (a context-parallel rank
     * whose partition the merge consumes), so the caller skips the output half for it.
     */
    public boolean attendStep(int pos0, int n, GpuBuffer q, GpuBuffer qPos) {
        ensureBound();
        ensureDeviceKv();
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": step of " + n + " tokens outside [1, "
                    + ubatch + "]");
        }
        GpuBuffer tree = stepTreeMask();
        boolean rebound = q != null;
        if (rebound) {
            flash.bindStepQueries(q, qPos);
        }
        try {
            if (dualChunk != null) {
                dualChunkAttention.runDense(0, n, Math.min(kvLayer.layout().rows(), pos0 + n));
            } else if (n == 1 && !routesDecodeThroughPrefill(pos0)) {
                flash.decode(pos0, tree);
                if (dualFlash != null) {
                    dualFlash.decode(pos0);
                }
            } else {
                flash.prefill(pos0, n, tree);
                if (dualFlash != null) {
                    dualFlash.prefill(pos0, n);
                }
            }
        } finally {
            if (rebound) {
                flash.restoreStepQueries(qBuf, posBuffer);
            }
        }
        if (tap != null) {
            tap.afterAttention(pos0, n, attnBuf, lseBuf, spec.nHead(), spec.headDimV());
            if (!tap.ownsOutput()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Runs the attention launches of a prepared step over the paged step {@code step}, reading the query rows from
     * {@code q} instead of the packed rows this mixer projected: the gathered rows of a prefill context-parallel
     * batched step, which lie in the step's global row order like the translated step's segments do. The launches
     * take the query positions from the step's own segment buffers (the per-segment uploads and the per-sequence
     * query positions, which carry the true positions). The paged kernels address the KV rows through the step's
     * page tables of this rank's own page shard; the mixer's own query buffer is restored before the call returns.
     * Returns false when a tap took the attention output (a context-parallel rank whose partition the merge
     * consumes), so the caller skips the output half for it.
     *
     * @throws UnsupportedOperationException when the keys are not a paged cache
     */
    public boolean attendPagedStep(PagedStep step, GpuBuffer q) {
        ensureBound();
        ensureDeviceKv();
        Objects.requireNonNull(step, "step");
        Objects.requireNonNull(q, "q");
        if (step.nTokens() < 1 || step.nTokens() > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": paged step of " + step.nTokens()
                    + " tokens outside [1, " + ubatch + "]");
        }
        flash.bindPagedStepQueries(q);
        try {
            flash.batched(step, kvLayer.group());
        } finally {
            flash.restorePagedStepQueries(qBuf);
        }
        if (tap != null) {
            tap.afterAttention(step.segPos0(0), step.nTokens(), attnBuf, lseBuf, spec.nHead(), spec.headDimV());
            return tap.ownsOutput();
        }
        return true;
    }

    /**
     * Runs the post-attention half of a prepared step over its first {@code n} packed rows: the dual attention's
     * value mix, the cache rotation's output half, the output stages and the {@code wo} projection into
     * {@code out}. A prefill context-parallel mixer calls it with the rank's own row count.
     */
    public void finishStep(int n, ScratchSlot out) {
        ensureBound();
        ensureDeviceKv();
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": step of " + n + " tokens outside [1, "
                    + ubatch + "]");
        }
        if (loopMixStep != null) {
            loopMixStep.run(n);
        }
        if (rotation != null) {
            rotation.afterAttention(n);
        }
        runAll(outputSteps, n);
        output.projectOutput(n, out);
    }

    /**
     * Why this layer's attention cannot take part in a prefill context-parallel step, or null when it can: a host
     * KV cache, dual-chunk or dual attention stages, static sink tokens, an Inkling stage, cache side planes the
     * query exchange does not copy, or missing Q/K/V projections. The quantized-KV Hadamard rotation needs no
     * refusal: every rank rotates its own packed rows in place before the stores, and the shadow store re-runs the
     * same store kernels over the rotated rows, so the gathered queries and the scattered cache rows are the
     * rotated ones. The vendor side planes are what the exchange cannot carry (the store kernels write them
     * beside the rows, which the scatter moves alone).
     */
    public String pcpRefusal() {
        if (hostKv) {
            return "its KV cache is on the host";
        }
        if (dualChunk != null) {
            return "it runs dual-chunk attention";
        }
        if (dualFlash != null || loopMixStep != null) {
            return "it runs a dual attention stage";
        }
        if (spec.sinkTokens() > 0) {
            return "it has static sink tokens";
        }
        if (inkling != null) {
            return "it runs Inkling attention stages";
        }
        if (proj.source() == QkvProjection.Source.NONE) {
            return "it has no Q/K/V projections";
        }
        if (kvLayer == null) {
            return "it has no KV cache";
        }
        if (kvLayer.rowScales() != null || kvLayer.vendorSide() != null) {
            return "its cache carries per-row scales or vendor planes the query exchange does not copy";
        }
        return null;
    }

    /** The raw head outputs of the prepared step ({@code n} rows of {@code n_head * d_v}), before the tap. */
    public GpuBuffer attentionBuffer() {
        return attnBuf;
    }

    /**
     * Re-runs this bound mixer's cache stores of the prepared step with rebound destinations: the shadow store of a
     * prefill context-parallel split. Every store kernel reads the same projected and rotated key/value rows it
     * already stored — a fused K store re-runs its rope over the same rows and positions — so with {@code cellIdx}
     * naming the destination rows of another rank's cache block and {@code kDst}/{@code vDst} holding those rebound
     * rows, a chunk owner writes the cache rows another rank holds into the caller's shadow buffers, which the
     * coordinator then moves into that rank's cache. The bound steps and the mixer's own cache are unchanged after
     * the call. {@code n} is the prepared step's token count; the shadow buffers must hold {@code n} rows of the
     * cache layout's row width.
     */
    public void shadowStore(int n, GpuBuffer cellIdx, GpuBuffer kDst, GpuBuffer vDst) {
        ensureBound();
        Objects.requireNonNull(cellIdx, "cellIdx");
        Objects.requireNonNull(kDst, "kDst");
        Objects.requireNonNull(vDst, "vDst");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": shadow store of " + n + " tokens outside"
                    + " [1, " + ubatch + "]");
        }
        for (StoreRebind rebind : kvStore.rebinds()) {
            GpuBuffer dst = rebind.vSide ? vDst : kDst;
            rebind.step.dispatch.setBuffer(rebind.cellArg, cellIdx);
            rebind.step.dispatch.setBuffer(rebind.dstArg, dst);
            rebind.step.run(n);
            rebind.step.dispatch.setBuffer(rebind.cellArg, rebind.cell);
            rebind.step.dispatch.setBuffer(rebind.dstArg, rebind.dst);
        }
    }

    /** The projected query rows of the prepared step, in this mixer's packed row order. */
    public GpuBuffer queryBuffer() {
        return qBuf;
    }

    /** The layer's KV cache: its K and V buffers and the row layout of this rank's shard. */
    public GpuKvLayer kvLayer() {
        return kvLayer;
    }

    /** The device step paths need the device cache; a host-KV layer runs through {@link #runWithHostAttention}. */
    private void ensureDeviceKv() {
        if (hostKv) {
            throw new IllegalStateException("layer " + layer.il() + " keeps its KV cache on the host; its attention"
                    + " runs through the host attention stage");
        }
    }

    void checkStepInputs(int pos0, int n) {
        if (stepInputs != null && (stepInputs.pos0() != pos0 || stepInputs.nTokens() != n)) {
            throw new IllegalStateException("layer " + layer.il() + " runs positions [" + pos0 + ", " + (pos0 + n)
                    + ") but the step inputs hold [" + stepInputs.pos0() + ", "
                    + (stepInputs.pos0() + stepInputs.nTokens()) + "); upload them before the first layer");
        }
    }

    /** The running step's tree mask buffer over its rows, or null outside a tree verify step. */
    private GpuBuffer stepTreeMask() {
        return stepInputs != null ? stepInputs.treeMask() : null;
    }

    /** Why this layer's attention cannot run under the step's tree mask, or null when it can. */
    private String treeRefusal() {
        if (dualChunk != null) {
            return "layer " + layer.il() + " runs dual-chunk attention, which takes no tree mask";
        }
        if (dualFlash != null) {
            return "layer " + layer.il() + " runs dual-attention layers, which take no tree mask";
        }
        AttentionSpec.BiasMode mode = spec.biasMode();
        if (mode == AttentionSpec.BiasMode.T5_CAUSAL || mode == AttentionSpec.BiasMode.T5_BIDIR
                || mode == AttentionSpec.BiasMode.INKLING) {
            return "layer " + layer.il() + " runs " + mode + " position bias, which reads a relative table over the"
                    + " position rules a tree mask replaces";
        }
        return null;
    }

    static void addDeltas(GpuProjectionDeltas.Site site, int n, GpuBuffer x, int xOff, int xStride,
                          GpuBuffer y, int yOff, int yStride) {
        if (site != null && !site.isEmpty()) {
            site.apply(n, x, xOff, xStride, y, yOff, yStride);
        }
    }

    /** The delta site of projection {@code weight}; a matmul that transforms its input (PRISM) refuses deltas. */
    GpuProjectionDeltas.Site serveDeltas(String weight, GpuWeight w) {
        String refusal = w.hadamard() != null || (w.inputTransform() != null && !w.inputTransform().isIdentity())
                ? w.name() + " transforms its input (PRISM Hadamard or input transform) inside the matmul; a LoRA"
                + " delta would need the transformed rows" : null;
        return ctx.projectionDeltas().serve(layer, weight, refusal);
    }

    static void runAll(List<TokenStep> steps, int n) {
        runAll(steps, n, null);
    }

    static void runAll(List<TokenStep> steps, int n, PagedStep paged) {
        for (int i = 0; i < steps.size(); i++) {
            steps.get(i).run(n, paged);
        }
    }

    /** Records an input step for the whole step and, when {@code cacheWrite}, for the cache-write-only step. */
    void input(TokenStep step, boolean cacheWrite) {
        inputSteps.add(step);
        if (cacheWrite) {
            kvInputSteps.add(step);
        }
    }

    // ================================================================== context shift

    /**
     * Applies the RoPE delta of a context shift to this layer's own K cache ({@link GpuContextShift}). The memory has
     * already moved the rows ({@code DeviceModelMemory.shiftPositions}); layers without RoPE, shared-KV readers and
     * the linear replacement have nothing to rotate.
     *
     * @throws UnsupportedOperationException for a layer without a cache, a model llama.cpp cannot shift
     *                                       ({@link GpuContextShift#refusal}), or a cache whose geometry the shift
     *                                       kernel cannot address
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        ensureBound();
        shift.apply(nKeep, nDiscard);
    }

    /**
     * Null when {@link #shiftContext} can run: a layer without its own projections, or a cached layer of a model
     * llama.cpp can shift whose cache geometry the shift kernel addresses.
     */
    @Override
    public String shiftRefusal() {
        ensureBound();
        return shift.refusal();
    }

    // ================================================================== dispatch helpers

    TokenStep step(String kernel, Object[] args, int countArg, long countPerToken, long itemsPerToken,
                   long itemsFixed, int posStrideArg) {
        long items = (long) ubatch * itemsPerToken + itemsFixed;
        PreparedDispatch d = prepare(kernel, roundUp(items), args);
        return new TokenStep(d, countArg, countPerToken, itemsPerToken, itemsFixed, posStrideArg, local);
    }

    TokenStep biasStep(GpuBuffer bias, int dim, GpuBuffer data) {
        return step(BIAS_ADD, bindArgs(dim, ubatch, bias, data), ARG_BIAS_SEQ, 1, 0, dim, -1);
    }

    TokenStep copyStep(GpuBuffer src, int srcOff, int srcStride, GpuBuffer dst, int dstOff, int dstStride,
                       int rowsPerToken, int cols) {
        Object[] args = bindArgs((int) ((long) ubatch * rowsPerToken), cols, srcStride, srcOff, dstStride, dstOff,
                src, dst);
        return step(GpuOps.COPY2D, args, ARG_COPY_ROWS, rowsPerToken, (long) rowsPerToken * cols, 0, -1);
    }

    TokenStep clampStep(ActParams clamp, GpuBuffer x, int width) {
        int n = ActKernels.elementCount(ubatch, width);
        Object[] args = ActKernels.unaryArgs(ActOp.CLAMP, clamp, n, x, 0, x, 0);
        return step(ActKernels.UNARY, args, ActKernels.ARG_UNARY_N, width, width, 0, -1);
    }

    TokenStep normStep(NormKind kind, float eps, GpuBuffer x, int rowsPerToken, int rowLen, GpuBuffer w,
                       GpuBuffer b, int wRows) {
        int rows = ActKernels.elementCount(ubatch, rowsPerToken);
        Object[] args = NormKernels.rowNormArgs(kind, eps, rows, rowLen, rowLen, x, 0, x, 0, w, b, wRows,
                ops.placeholders().readOnlyFloats());
        return step(NormKernels.ROW_NORM, args, NormKernels.ARG_ROW_NORM_N_ROWS, rowsPerToken,
                (long) rowsPerToken * local, 0, -1);
    }

    PreparedDispatch prepare(String kernel, long global, Object[] args) {
        return resources.prepare(kernel, new long[]{global}, new long[]{local}, args);
    }

    long roundUp(long items) {
        return Math.max(1, (items + local - 1) / local) * local;
    }

    static Object[] bindArgs(Object... args) {
        return args;
    }

    /** The F32 values of a layer vector tensor ({@code suffix}), checked to hold {@code n} values. */
    GpuBuffer vector(String suffix, long n) {
        return vector(weights().layer(layer, suffix), n);
    }

    /**
     * The F32 values of a vector tensor: the tensor's own buffer when it is a single F32 buffer, otherwise its rows
     * decoded once into a buffer this mixer owns.
     */
    GpuBuffer vector(GpuWeight w, long n) {
        long total = w.rows() * w.cols();
        QkvProjection.checkLength(w.name(), total, n);
        if (w.type() == GGMLType.F32 && w.isSingleShard() && w.inverseHadamard() == null && !w.hasScale()) {
            return w.shard(0);
        }
        if (w.rows() > Integer.MAX_VALUE || total > Integer.MAX_VALUE / Float.BYTES) {
            throw new IllegalArgumentException("vector tensor " + w.name() + " is too large to decode");
        }
        GpuBuffer b = resources.allocate(total * Float.BYTES);
        int[] rows = new int[(int) w.rows()];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = i;
        }
        ops.embed(w, rows, b, 0);
        return b;
    }

    private static GpuBuffer offsetFree(ScratchSlot slot, long floats) {
        if (slot.offset() != 0) {
            throw new IllegalStateException("scratch slot " + slot.name() + " must start at offset 0, got "
                    + slot.offset());
        }
        slot.require(floats);
        return slot.buffer();
    }

    // ------------------------------------------------------------------ stage accessors

    GpuForwardContext ctx() {
        return ctx;
    }

    GpuOps ops() {
        return ops;
    }

    GpuMixerResources resources() {
        return resources;
    }

    AttentionSpec spec() {
        return spec;
    }

    QkvProjection proj() {
        return proj;
    }

    int ubatch() {
        return ubatch;
    }

    int local() {
        return local;
    }

    boolean hostKv() {
        return hostKv;
    }

    DualChunkSpec dualChunk() {
        return dualChunk;
    }

    GpuStepInputs stepInputs() {
        return stepInputs;
    }

    GpuKvScales kvScales() {
        return kvScales;
    }

    GpuNormRopeStages normRope() {
        return normRope;
    }

    GpuDualChunkPlanes dcPlanes() {
        return dcPlanes;
    }

    GpuQkvProjections projections() {
        return projections;
    }

    GpuInklingStages inkling() {
        return inkling;
    }

    GpuValueEmbeddings valueEmbeddings() {
        return valueEmbeddings;
    }

    GpuQkvAdapters adapters() {
        return adapters;
    }

    GpuOutputStages output() {
        return output;
    }

    List<TokenStep> inputSteps() {
        return inputSteps;
    }

    List<TokenStep> kvInputSteps() {
        return kvInputSteps;
    }

    List<TokenStep> outputSteps() {
        return outputSteps;
    }

    GpuWeight wq() {
        return projections.wq();
    }

    GpuWeight wo() {
        return wo;
    }

    GpuProjectionDeltas.Site outSite() {
        return outSite;
    }

    GpuBuffer qBuf() {
        return qBuf;
    }

    GpuBuffer kBuf() {
        return kBuf;
    }

    GpuBuffer vBuf() {
        return vBuf;
    }

    GpuBuffer attnBuf() {
        return attnBuf;
    }

    GpuBuffer qProjBuf() {
        return projections.qProjBuf();
    }

    GpuBuffer gateTemp() {
        return projections.gateTemp();
    }

    GpuBuffer posBuffer() {
        return posBuffer;
    }

    /** Records an output step that runs over the attention rows before the {@code wo} projection. */
    void addOutputStep(TokenStep step) {
        outputSteps.add(step);
    }

    private void ensureBound() {
        if (closed) {
            throw new IllegalStateException("attention mixer is closed");
        }
        if (!bound) {
            throw new IllegalStateException("attention mixer is not bound");
        }
    }

    /** The layer this mixer is bound to; null before {@link #bind}. */
    public LayerSpec layer() {
        return layer;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        releaseAll();
    }

    private void releaseAll() {
        RuntimeException failure = Closeables.closeAll(
                Arrays.<AutoCloseable>asList(shift, rotation, storeRotation, t5Bias, projections, normRope), null);
        if (resources != null) {
            failure = resources.release(failure);
        }
        shift = null;
        flash = null;
        dualFlash = null;
        tap = null;
        lseBuf = null;
        dualKvLayer = null;
        dualAttnBuf = null;
        loopGateW = null;
        loopGateB = null;
        loopMixStep = null;
        rotation = null;
        storeRotation = null;
        t5Bias = null;
        projections = null;
        normRope = null;
        kvStore = null;
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "GpuAttentionMixer{" + (layer == null ? "unbound" : "layer " + layer.il() + ", " + proj) + '}';
    }

    /**
     * The T5 relative-position bias table of the stack's bias layer ({@link T5RelativeBias#device}: F32
     * {@code [n_buckets][n_head]}, copied once at bind and owned by this mixer), or null without a T5 bias.
     */
    private GpuBuffer relBias() {
        if (spec.biasMode() == AttentionSpec.BiasMode.INKLING) {
            return inkling.relTable();
        }
        if (!T5RelativeBias.uses(spec)) {
            return null;
        }
        t5Bias = T5RelativeBias.device(ctx, layer, spec);
        return t5Bias.buffer();
    }
}
