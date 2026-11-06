package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
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
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceKvFormats;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuRecurrentState;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.encdec.T5RelativeBias;
import com.eonmux.neuralfree.gguf.ops.fused.FusedKernels;
import com.eonmux.neuralfree.gguf.ops.fused.GpuMultiTask;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.rope.DualChunkRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DualChunkSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;
import com.eonmux.neuralfree.utils.Closeables;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.IntConsumer;

/**
 * Softmax-attention mixer on the GPU ({@code MixerKind.ATTENTION} and deci's {@code ATTENTION_LINEAR_REPLACE}),
 * following llama.cpp {@code build_qkv} + {@code build_attn} (src/llama-graph.cpp:1807-1912, 2946-3100, 3115-3290).
 *
 * <p>Per step, for {@code n} tokens of the normalized input {@code in}:
 * <ol>
 *   <li>projections ({@link QkvProjection}): Q, K and V land densely in the {@code Q}, {@code K} and {@code V}
 *       scratch slots ({@code [t][n_head * d]}); a fused {@code attn_qkv} product is split there with
 *       {@code LLM_Copy2D_F32}, as is V from the raw K projection when the layer has no {@code attn_v}
 *       ({@link QkvProjection#vFromK()}); on a MoVA layer V is the weighted sum of routed value experts
 *       ({@link GpuValueExperts}); biases with {@code LLM_Batch_BiasAdd}; the KQV clamp with
 *       {@code LLM_Unary CLAMP};
 *       an interleaved output gate is split off Q; other gates are projected from {@code in}; limite's value
 *       embeddings ({@link AttentionSpec#valueEmbeddings()}) add the token's {@code value_embd} row, gated per KV
 *       head by {@code sigmoid(attn_ve_gate @ in)} when the layer carries the gate, to V with
 *       {@value AttnExtKernels#VALUE_EMBEDDINGS};</li>
 *   <li>Q/K norms of every {@link QkNormKind} ({@link QkNormGeometry}: per head, per half head, or the whole
 *       projection; their biases unless {@link AttentionSpec#qkNormBias()} is off) and gemma's weightless V norm with
 *       {@code LLM_RowNorm} over the dense head rows;</li>
 *   <li>RoPE with {@code LLM_RoPE_Ext} (NoPE layers skip it); the Q/K norms run after it instead when
 *       {@link AttentionSpec#qkNormAfterRope()} (hunyuan, llama4, ...); then the attention temperature
 *       ({@code LLM_AttnTempScale});</li>
 *   <li>with a cache ({@link AttentionSpec#usesMemory()}): the quantized-KV Hadamard rotation of Q, K and V
 *       ({@link GpuQuantizedKvRotation}), {@code LLM_KV_Store_<type>} of K and V at {@code cell_idx} (only on the layer
 *       that owns the cache; a shared-KV layer reads its owner's {@link GpuKvLayer}), and flash attention over the
 *       cache rows with {@code kv_pos} visibility; without a cache (encoders) flash attention reads K and V of the
 *       current batch with {@code kv_pos = q_pos};</li>
 *   <li>flash attention ({@link GpuFlashAttention}, planned by {@link FlashPlan}, its arguments packed by
 *       {@link FlashKernelArgs}, the paged launches in {@link GpuPagedFlashAttention}): flash decode / tiled prefill
 *       per {@link FlashAttnTiling} with {@code smemFloats = LlmKernels.faSmemFloats()}, the per-row decode
 *       fallback ({@link FlashAttnTiling.Plan#PER_ROW_DECODE}, over a paged cache {@link PagedRowFallback}),
 *       split-K partials merged by
 *       {@code LLM_FlashAttn_Combine}; on a backend with tensor cores ({@link DeviceFeature#COOPERATIVE_MATRIX})
 *       and F16 caches the prefill runs the mma kernel ({@link FlashAttnTiling.Plan#MMA}, {@code
 *       LLM_FlashAttnMma*}); sinks, softcap, ALiBi and the T5 relative bias are kernel arguments; K and V
 *       caches of different types use the mixed-type flash kernels
 *       ({@link AttnKernels#flashDecode(KvDType, KvDType)});</li>
 *   <li>output: the V rotation undone, muse-spark's weightless per-head RMS norm ({@code LLM_RowNorm}), limite's
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
 * rotation do not run; the host stage applies the rotation and store.</p>
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

    /**
     * {@code data[s * dim + d] += bias[d]} ({@code kernels/llm/eltwise.cl}); arguments
     * {@code (dim, seq_len, bias, data)}.
     */
    public static final String BIAS_ADD = GpuOps.BATCH_BIAS_ADD;
    private static final int ARG_BIAS_SEQ = 1;
    private static final int ARG_BIAS_DATA = 3;
    private static final int ARG_COPY_ROWS = 0;
    private static final int ARG_AXPBY_N = 0;
    private static final int ARG_AXPBY_B_OFF = 4;
    private static final int ARG_AXPBY_OUT_OFF = 5;
    private static final int ARG_AXPBY_B = 7;
    private static final int ARG_AXPBY_OUT = 8;

    private static final String T_QKV = "attn.qkv";
    private static final String T_QLAT = "attn.qlat";
    private static final String T_QGATE = "attn.qgate";
    private static final String T_GATE = "attn.gate";
    private static final String T_VE = "attn.ve";
    private static final String T_VE_GATE = "attn.ve.gate";
    private static final String T_OUT = "attn.out";
    private static final String T_ADAPTER = "attn.adapter_rank";
    private static final String T_REL_OUT = "attn.rel_out";
    private static final String T_REL_TABLE = "attn.rel_table";
    private static final String T_CONV = "attn.sconv";
    /** Argument indices of {@code LLM_SSM_Conv}'s row offsets (kernels/llm/recurrent/ssm_conv.cl). */
    private static final int ARG_CONV_X_OFF = 5;
    private static final int ARG_CONV_OUT_OFF = 7;

    private final List<TokenStep> inputSteps = new ArrayList<>();
    /** The subset of {@link #inputSteps} that feeds the stored K and V rows (the cache-write-only step). */
    private final List<TokenStep> kvInputSteps = new ArrayList<>();
    private final List<TokenStep> storeSteps = new ArrayList<>();
    /** The cache stores' rebind points a prefill context-parallel shadow store overrides. */
    private final List<StoreRebind> storeRebinds = new ArrayList<>();
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

    private GpuWeight wq;
    private GpuWeight wqUp;
    private GpuWeight wk;
    private GpuWeight wv;
    private GpuWeight wqkv;
    private GpuWeight wgate;
    private GpuWeight wo;
    /** Per-use Q/K/V adapters of a shared attention block (Zamba2); null without them. */
    private GpuWeight wqAdapterA;
    private GpuWeight wqAdapterB;
    private GpuWeight wkAdapterA;
    private GpuWeight wkAdapterB;
    private GpuWeight wvAdapterA;
    private GpuWeight wvAdapterB;
    private ScratchSlot adapterRankBuf;

    /** limite value-embedding table and its per-KV-head gate projection; null without value embeddings. */
    private GpuWeight valueEmbd;
    private GpuWeight veGateW;
    private ScratchSlot veValues;
    private ScratchSlot veGate;
    /** Token ids and host-id fallback of the value-embedding lookup; device ids serve a single-shard table. */
    private ScratchSlot stepTokenIds;
    private int[] hostStepIds;
    /** The per-head {@code attn_xsa_alpha} of the exclusive self attention; null without XSA. */
    private GpuBuffer xsaAlpha;

    private GpuProjectionDeltas.Site qSite;
    private GpuProjectionDeltas.Site kSite;
    private GpuProjectionDeltas.Site vSite;
    private GpuProjectionDeltas.Site qkvSite;
    private GpuProjectionDeltas.Site gateSite;
    private GpuProjectionDeltas.Site outSite;
    private GpuValueExperts valueExperts;

    private GpuBuffer qBuf;
    private GpuBuffer kBuf;
    private GpuBuffer vBuf;
    private GpuBuffer attnBuf;
    private GpuBuffer qProjBuf;
    private GpuBuffer qLatentBuf;
    /** RMS post-norm of the shared query projection (Step3 text), null for a direct Q projection. */
    private TokenStep qLatentNorm;
    private GpuBuffer qkvTemp;
    private GpuBuffer gateTemp;
    private GpuBuffer outTemp;

    private GpuKvLayer kvLayer;
    private GpuStepInputs stepInputs;
    /** True when the layer's KV cache is in host RAM and its attention runs on the host ({@code -nkvo}). */
    private boolean hostKv;
    /** FP8 cache scales in host memory for a host attention, or null. */
    private float[] hostKScale;
    private float[] hostVScale;
    /** Per-head attention sinks in host memory for a host attention, or null. */
    private float[] hostSinks;
    /** Normalized learned static sink keys and values of the openPangu sink attention, or null. */
    private float[] hostSinkK;
    private float[] hostSinkV;
    /** The device copies of the static sink keys and values; null without sink tokens. */
    private GpuBuffer sinkKBuf;
    private GpuBuffer sinkVBuf;
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
    /** The five query-plane position buffers, the key position buffer and the chunk positions the stages read. */
    private final List<GpuBuffer> dcPlanePos = new ArrayList<>(DualChunkRope.VARIANTS);
    private GpuBuffer dcKeyPos;
    private GpuBuffer dcOrigPos;
    private int[] dcHostPlanes;
    private int[] dcHostKeys;
    /** IQuestLoopCoder's read-only first-pass cache and its flash attention; null without dual attention. */
    private GpuKvLayer dualKvLayer;
    private GpuFlashAttention dualFlash;
    private GpuBuffer loopGateW;
    private GpuBuffer loopGateB;
    private GpuBuffer dualAttnBuf;
    private TokenStep loopMixStep;
    /** Persistent launch of the Q/K norms with their rotations and the V norm; null when unused. */
    private GpuMultiTask normRopeBatch;
    private GpuBuffer ropeFactorBuf;
    private GpuBuffer computedFactorBuf;
    private float[] computedFactorValues;
    private GpuBuffer fopeCosBuf;
    private GpuBuffer fopeSinBuf;
    private GpuBuffer fopeInvBuf;
    private float[] fopeInvValues;
    private GpuContextShift shift;
    private T5RelativeBias.Device t5Bias;
    private String shiftBlocker;
    /** The TurboQuant query rotation ({@code LLM_KvTqRotateQ}) over the projected rows, and its output buffer. */
    private TokenStep tqRotate;
    private GpuBuffer qRot;
    private PreparedDispatch outBias;
    private PreparedDispatch valueScale;
    private long valueScaleCount = -1;
    /** The FP8 KV-cache scale buffers of the layer's cache (n_head_kv values each), null without an FP8 cache. */
    private GpuBuffer kvScales;
    private GpuBuffer kScales;
    private GpuBuffer vScales;
    /** The step's query-position buffer the attention and the Inkling steps read. */
    private GpuBuffer posBuffer;

    /** Inkling's query-side relative projection, log scaling and the short convolutions of the K/V streams. */
    private boolean inkling;
    private GpuWeight wr;
    private GpuProjectionDeltas.Site rSite;
    private int relDim;
    private int relExtent;
    private int logFloor;
    private float logAlpha;
    private GpuBuffer rBuf;
    private GpuBuffer relTableBuf;
    private GpuBuffer relProjBuf;
    private GpuBuffer rBiasBuf;
    private PreparedDispatch rBiasAdd;
    private TokenStep relLogitsStep;
    private TokenStep queryScaleStep;
    private PreparedDispatch convK;
    private PreparedDispatch convV;
    private TokenStep convKStep;
    private TokenStep convVStep;
    private GpuBuffer convTemp;
    private GpuBuffer convStateBuf;
    private GpuBuffer convStateRows;
    private boolean routeConvState;
    private int convStateStride;
    private int convKChannels;
    private int convVChannels;

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
     * @throws IllegalArgumentException      when the layer has no attention spec, a tensor has the wrong shape, or the
     *                                       geometry does not fit the kernels
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
        if (layer.mixerKind() == MixerKind.INKLING && hostKv) {
            throw new UnsupportedOperationException("layer " + layer.il() + ": Inkling's short convolutions keep"
                    + " their state on the device and feed the device KV cache; run without host KV offload");
        }
        try {
            if (proj.source() == QkvProjection.Source.NONE) {
                wo = weights().layer(layer, proj.outWeight());
                outSite = serveDeltas(proj.outWeight(), wo);
            } else {
                bindAttention();
            }
        } catch (RuntimeException e) {
            releaseAll();
            throw e;
        }
        bound = true;
    }

    private GpuWeights weights() {
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
        if (kv != null) {
            bindKvScales(kv);
        } else if (hostKv) {
            bindHostKvScales();
        }
        boolean cached = kv != null || hostKv;
        GpuBuffer qPos = cached ? stepInputs.qPos() : offsetFree(scratch.slot(ScratchSlots.POS), (long) ubatch * 4);
        this.posBuffer = qPos;
        GpuBuffer ropePos = cached ? stepInputs.ropePos() : qPos;
        bindProjections(scratch);
        if (layer.mixerKind() == MixerKind.INKLING) {
            bindInklingInputs(scratch);
        }
        bindValueEmbeddings(scratch);
        if (dualChunk != null) {
            bindDualChunkBuffers();
        }
        // Inkling's log scaling runs on the normed queries and its post norms follow RoPE, so its norms, rotation
        // and K-row store stay separate steps. Dual-chunk attention rotates five query planes and scales the K rows
        // before the store, so it keeps the unfused path too.
        String fusedKStore = inkling || dualChunk != null ? null : fusedKStoreKernel(kv);
        boolean fused = !inkling && dualChunk == null && !spec.qkNormAfterRope()
                && bindFusedNormRope(ropePos, fusedKStore != null);
        if (!fused) {
            if (!spec.qkNormAfterRope()) {
                bindQkNorms(fusedKStore != null);
            }
            bindVNorm();
            bindRope(ropePos, fusedKStore != null);
        }
        bindStaticSinks();
        if (spec.qkNormAfterRope()) {
            bindQkNorms(false);
        }
        if (inkling) {
            bindInklingPostNorms();
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
                            dualChunkPlanePitch(), kBuf, spec.kDim(), vBuf, spec.vDim(), attnBuf, spec.outDim(),
                            ubatch)
                    : GpuQuantizedKvRotation.of(ops, kv.rotation(), qBuf, spec.qDim(), kBuf, spec.kDim(), vBuf,
                            spec.vDim(), attnBuf, spec.outDim(), ubatch);
            if (proj.projectsKv()) {
                storeRotation = GpuQuantizedKvRotation.forStore(ops, kv.rotation(), kBuf, spec.kDim(), vBuf,
                        spec.vDim(), ubatch);
                bindStore(kv, fusedKStore, ropePos);
            }
        }
        if (hostKv) {
            hostSinks = hostSinks();
        } else if (dualChunk != null) {
            dualChunkAttention = bindDualChunkAttention(kv, qPos);
        } else if (spec.dual()) {
            bindDualFlash(kv, qPos);
        } else {
            flash = bindFlash(kv, qPos);
        }
        bindOutput();
        if (kv != null) {
            bindShift(kv);
        }
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
                new FlashKernelArgs(localSpec, localKeys, fq, qPos, attnBuf, partial, sinks(), sinkKBuf, sinkVBuf,
                        relBias(), kvScales != null ? kvScales : ones, placeholder), null, stepInputs, ubatch, null);
        dualAttnBuf = ctx.scratch().temp("attn.dual", (long) ubatch * spec.outDim()).buffer();
        dualFlash = GpuFlashAttention.prepare(resources, ctx.queue(), globalPlan,
                new FlashKernelArgs(spec, globalKeys, fq, qPos, dualAttnBuf, partial, sinks(), sinkKBuf, sinkVBuf,
                        relBias(), ones, placeholder), null, stepInputs, ubatch, null);
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
        FlashKernelArgs args = new FlashKernelArgs(spec, keys, flashQuery(keys), qPos, attnBuf, partial, sinks(),
                sinkKBuf, sinkVBuf, relBias(), kvScales != null ? kvScales : ops.placeholders().kvScalesOnes(),
                ops.placeholders().readOnlyFloats());
        return GpuFlashAttention.prepare(resources, ctx.queue(), plan, args, pg, stepInputs, ubatch, lseBuf);
    }

    // ------------------------------------------------------------------ dual-chunk attention

    /** Floats between consecutive query rotation planes ({@code [5][ubatch][qDim]} in the Q slot). */
    private int dualChunkPlanePitch() {
        return ubatch * spec.qDim();
    }

    /**
     * Allocates the position buffers of the five query rotations, the key rotation and the stage launches; they are
     * uploaded per step before the bound RoPE steps run.
     */
    private void bindDualChunkBuffers() {
        if (kvLayer == null) {
            throw new IllegalStateException("layer " + layer.il() + " runs dual-chunk attention without a KV cache");
        }
        dcHostPlanes = new int[DualChunkRope.VARIANTS * ubatch];
        dcHostKeys = new int[ubatch];
        for (int plane = 0; plane < DualChunkRope.VARIANTS; plane++) {
            dcPlanePos.add(resources.allocate((long) ubatch * Integer.BYTES));
        }
        dcKeyPos = resources.allocate((long) ubatch * Integer.BYTES);
        dcOrigPos = resources.allocate((long) ubatch * Integer.BYTES);
    }

    /**
     * The layer's dual-chunk stage launches ({@link GpuDualChunkAttention}): the flash plan of its cache, dense and
     * paged, with one {@code n_split = 1} partial buffer per stage and the merge kernel.
     */
    private GpuDualChunkAttention bindDualChunkAttention(GpuKvLayer kv, GpuBuffer qPos) {
        FlashKeys keys = FlashKeys.cached(kv, stepInputs.kvPos(kv.group()));
        if (keys.typeK().turboQuantMse()) {
            throw new UnsupportedOperationException("layer " + layer.il() + " runs dual-chunk attention over a "
                    + keys.typeK() + " key cache: its keys are stored in the Hadamard-rotated basis and the GPU"
                    + " dual-chunk stages score the unrotated query planes; use another --cache-type-k or the CPU backend");
        }
        PagedFlashGeometry pg = PagedFlashGeometry.of(stepInputs, kv.group(), ubatch);
        FlashPlan plan = FlashPlan.of(ctx.kernels(), spec, keys, pg, ubatch);
        /* The stage launches replace this buffer with their own partial, so the attention rows serve as the default. */
        FlashKernelArgs args = new FlashKernelArgs(spec, keys, qBuf, qPos, attnBuf, attnBuf, sinks(), sinkKBuf, sinkVBuf,
                relBias(), kvScales != null ? kvScales : ops.placeholders().kvScalesOnes(),
                ops.placeholders().readOnlyFloats());
        return GpuDualChunkAttention.prepare(resources, ctx.queue(), plan, args, pg, ubatch, dualChunkPlanePitch(),
                dcOrigPos);
    }

    /**
     * Uploads the step positions of the rotation planes and of the key rotation from the original positions
     * {@code pos0 + t} of a contiguous step ({@link DualChunkRope}).
     */
    private void uploadDualChunkPositions(int pos0, int n) {
        int[] host = dualChunkAttention.hostPositions();
        for (int t = 0; t < n; t++) {
            host[t] = pos0 + t;
        }
        uploadDualChunkPlanes(n);
    }

    /** Uploads the step positions of a paged step from its segments' first positions. */
    private void uploadDualChunkPositions(PagedStep step, int n) {
        int[] host = dualChunkAttention.hostPositions();
        for (int seg = 0; seg < step.nSeg(); seg++) {
            int start = step.segStart(seg);
            int pos0 = step.segPos0(seg);
            for (int t = 0; t < step.segLen(seg); t++) {
                host[start + t] = pos0 + t;
            }
        }
        uploadDualChunkPlanes(n);
    }

    /** The five query-plane position arrays and the key positions of the first {@code n} host rows. */
    private void uploadDualChunkPlanes(int n) {
        int[] host = dualChunkAttention.hostPositions();
        DualChunkRope.queryPositions(dualChunk, host, 0, n, dcHostPlanes);
        for (int plane = 0; plane < DualChunkRope.VARIANTS; plane++) {
            writeIntSlice(dcPlanePos.get(plane), dcHostPlanes, plane * n, n);
        }
        DualChunkRope.keyPositions(dualChunk, host, 0, n, dcHostKeys);
        writeIntSlice(dcKeyPos, dcHostKeys, 0, n);
    }

    /** The key positions of the first {@code n} rows of a contiguous step (the cache-write-only pass). */
    private void uploadDualChunkKeyPositions(int pos0, int n) {
        int[] host = dualChunkAttention.hostPositions();
        for (int t = 0; t < n; t++) {
            host[t] = pos0 + t;
        }
        uploadDualChunkKeyPositions(n);
    }

    /** The key positions of a paged step's rows (the cache-write-only pass). */
    private void uploadDualChunkKeyPositions(PagedStep step, int n) {
        int[] host = dualChunkAttention.hostPositions();
        for (int seg = 0; seg < step.nSeg(); seg++) {
            int start = step.segStart(seg);
            int pos0 = step.segPos0(seg);
            for (int t = 0; t < step.segLen(seg); t++) {
                host[start + t] = pos0 + t;
            }
        }
        uploadDualChunkKeyPositions(n);
    }

    private void uploadDualChunkKeyPositions(int n) {
        DualChunkRope.keyPositions(dualChunk, dualChunkAttention.hostPositions(), 0, n, dcHostKeys);
        writeIntSlice(dcKeyPos, dcHostKeys, 0, n);
    }

    /**
     * Applies the original-context logit scaling of a step (vLLM multiplies the new K rows before caching and the
     * query too): rows {@code [rowStart, rowStart + n)} of K and of the three used query planes, by the factor of
     * the sequence's length at this step.
     */
    private void scaleDualChunk(int rowStart, int n, float factor) {
        if (factor == 1.0f) {
            return;
        }
        ops.scale(kBuf, rowStart * spec.kDim(), n * spec.kDim(), factor);
        for (int plane = 0; plane < 3; plane++) {
            ops.scale(qBuf, plane * dualChunkPlanePitch() + rowStart * spec.qDim(), n * spec.qDim(), factor);
        }
    }

    /** The K half of {@link #scaleDualChunk} for the cache-write-only pass. */
    private void scaleDualChunkK(int rowStart, int n, float factor) {
        if (factor != 1.0f) {
            ops.scale(kBuf, rowStart * spec.kDim(), n * spec.kDim(), factor);
        }
    }

    /** {@link #scaleDualChunk} over a paged step, each segment by its sequence's slot count. */
    private void scaleDualChunk(PagedStep step) {
        int[] slots = step.seqSlots(kvLayer.group());
        for (int seg = 0; seg < step.nSeg(); seg++) {
            scaleDualChunk(step.segStart(seg), step.segLen(seg),
                    dualChunk.scalingFactor(slots[step.segSeq(seg)]));
        }
    }

    /** {@link #scaleDualChunkK} over a paged step, each segment by its sequence's slot count. */
    private void scaleDualChunkK(PagedStep step) {
        int[] slots = step.seqSlots(kvLayer.group());
        for (int seg = 0; seg < step.nSeg(); seg++) {
            scaleDualChunkK(step.segStart(seg), step.segLen(seg),
                    dualChunk.scalingFactor(slots[step.segSeq(seg)]));
        }
    }

    /** Writes {@code count} ints of {@code src} from {@code off} into {@code dst}. */
    private void writeIntSlice(GpuBuffer dst, int[] src, int off, int count) {
        ByteBuffer bb = ByteBuffer.allocate(count * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int t = 0; t < count; t++) {
            bb.putInt(src[off + t]);
        }
        GpuEvent e = ctx.queue().enqueueWriteBuffer(dst, 0, (long) count * Integer.BYTES, bb.array());
        if (e != null) {
            e.close();
        }
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

    // ------------------------------------------------------------------ Inkling

    /**
     * The steps that run before the K norm: the two short convolutions of the K and V streams, each over the layer's
     * conv state (vLLM {@code InklingAttention.k_sconv}/{@code v_sconv}, {@code fused_qkvr_prep}). The convolution
     * adds back into the projection ({@code kv = conv(kv) + kv}), so it reads the raw rows into a temporary and adds
     * them, and it must run before the K norm normalizes the rows it stores.
     */
    private void bindInklingInputs(GpuScratch scratch) {
        ModelHparams hp = ctx.spec().hparams();
        inkling = true;
        relDim = hp.inklingRelDim();
        relExtent = spec.relBiasExtent();
        logFloor = hp.inklingLogScalingFloor();
        logAlpha = hp.inklingLogScalingAlpha();
        int rWidth = spec.nHead() * relDim;
        wr = weights().layer(layer, TensorNames.weight(TensorNames.ATTN_REL));
        QkvProjection.checkShape(TensorNames.weight(TensorNames.ATTN_REL), wr.cols(), wr.rows(), hp.nEmbd(), rWidth);
        rBuf = inklingTemp(scratch, T_REL_OUT, (long) ubatch * rWidth).buffer();
        rSite = serveDeltas(TensorNames.weight(TensorNames.ATTN_REL), wr);
        if (weights().layerOrNull(layer, TensorNames.bias(TensorNames.ATTN_REL)) != null) {
            GpuBuffer rBiasValues = vector(TensorNames.bias(TensorNames.ATTN_REL), rWidth);
            rBiasAdd = resources.prepare(BIAS_ADD, new long[]{roundUp((long) ubatch * rWidth)}, new long[]{local},
                    bindArgs(rWidth, ubatch, rBiasValues, rBuf));
        }
        GpuWeight proj = weights().layer(layer, TensorNames.weight(TensorNames.ATTN_REL_PROJ));
        relProjBuf = vector(proj, (long) relDim * relExtent);
        relTableBuf = inklingTemp(scratch, T_REL_TABLE, (long) ubatch * spec.nHead() * relExtent).buffer();
        bindInklingConvs(scratch);
    }

    /** The K/V short convolutions and the layer's packed conv state. */
    private void bindInklingConvs(GpuScratch scratch) {
        RecurrentSpec r = layer.recurrent();
        int width = r.dConv();
        convKChannels = spec.nHeadKv() * spec.headDimK();
        convVChannels = spec.nHeadKv() * spec.headDimV();
        GpuBuffer kw = vector(TensorNames.weight(TensorNames.ATTN_SCONV_K), (long) convKChannels * width);
        GpuBuffer vw = vector(TensorNames.weight(TensorNames.ATTN_SCONV_V), (long) convVChannels * width);
        convTemp = inklingTemp(scratch, T_CONV, (long) ubatch * Math.max(convKChannels, convVChannels)).buffer();
        GpuRecurrentState state = ctx.memory().recurrent(layer.il());
        if (state == null) {
            throw new IllegalStateException("layer " + layer.il() + " runs Inkling short convolutions but has no"
                    + " recurrent state");
        }
        convStateStride = -1;
        for (LayerMemoryPlan.StateTensor t : state.tensors()) {
            if (t.name().equals(LayerMemoryPlan.STATE_CONV)) {
                convStateStride = (int) t.floats();
            }
        }
        if (convStateStride < 0) {
            throw new IllegalStateException("layer " + layer.il() + " has no conv state for its short"
                    + " convolutions");
        }
        if (convStateStride != r.convStateFloats()) {
            throw new IllegalStateException("layer " + layer.il() + ": conv state holds " + convStateStride
                    + " floats, the layer geometry sizes it at " + r.convStateFloats());
        }
        convStateBuf = state.tensor(LayerMemoryPlan.STATE_CONV);
        GpuMemory memory = ctx.memory();
        convStateRows = memory.stepInputs().stateRows();
        routeConvState = memory instanceof com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory
                && memory.pagedSequences() != null;
        convK = inklingConv(kBuf, kw, convKChannels, r.inklingStreamOffset(RecurrentSpec.INKLING_K));
        convV = inklingConv(vBuf, vw, convVChannels, r.inklingStreamOffset(RecurrentSpec.INKLING_V));
        convKStep = inklingConvStep(convK);
        convVStep = inklingConvStep(convV);
        input(convKStep, true);
        input(convVStep, true);
    }

    /**
     * One stream's {@code LLM_SSM_Conv} over the layer's conv state: {@code x} is the stream's projection buffer
     * (its raw rows), the convolution writes {@link #convTemp} and the step adds it back into {@code x}. The token
     * count, the row offsets and the state row are re-set per segment.
     */
    private PreparedDispatch inklingConv(GpuBuffer x, GpuBuffer w, int channels, int stateOff) {
        Object[] args = {channels, layer.recurrent().dConv(), 1, 0, x == kBuf ? spec.kDim() : spec.vDim(), 0,
            channels, 0, stateOff, convStateStride, x, w, ctx.placeholders().readOnlyFloats(), convStateBuf,
            convStateRows, convTemp};
        if (args.length != SsmKernels.CONV_ARG_COUNT) {
            throw new IllegalStateException(SsmKernels.SSM_CONV + " takes " + SsmKernels.CONV_ARG_COUNT
                    + " arguments, " + args.length + " built");
        }
        return resources.prepare(SsmKernels.SSM_CONV,
                new long[]{SsmKernels.elementwiseGlobal(channels, SsmKernels.ELEMENTWISE_LOCAL)},
                new long[]{SsmKernels.ELEMENTWISE_LOCAL}, args);
    }

    /** The step of one conv dispatch: one launch per step segment, each over its sequence's state row. */
    private TokenStep inklingConvStep(PreparedDispatch conv) {
        TokenStep step = new TokenStep(conv, 0, 1, 0, 0, -1, local);
        step.segments = (n, paged) -> runConv(conv, n, paged);
        return step;
    }

    /**
     * Runs a conv dispatch over the step's segments: one launch per segment with its token count and row offsets,
     * routed to its sequence's state row. A step without a paged layout is one segment at row 0.
     */
    private void runConv(PreparedDispatch conv, int n, PagedStep paged) {
        if (paged == null || paged.nSeg() <= 1) {
            convSegment(conv, 0, n, 0);
            return;
        }
        for (int seg = 0; seg < paged.nSeg(); seg++) {
            convSegment(conv, paged.segSeq(seg), paged.segLen(seg), paged.segStart(seg));
        }
    }

    private void convSegment(PreparedDispatch conv, int seq, int len, int start) {
        if (len < 1) {
            return;
        }
        int channels = conv == convK ? convKChannels : convVChannels;
        int stride = conv == convK ? spec.kDim() : spec.vDim();
        if (routeConvState) {
            GpuMixer.routeStateRow(seq, convStateRows, ops.queue());
        }
        conv.setInt(SsmKernels.ARG_CONV_N_TOKENS, len);
        conv.setInt(ARG_CONV_X_OFF, start * stride);
        conv.setInt(ARG_CONV_OUT_OFF, 0);
        conv.dispatch();
        GpuBuffer x = conv == convK ? kBuf : vBuf;
        ops.add(x, start * stride, convTemp, 0, x, start * stride, len * channels);
    }

    /**
     * The steps that run after the norms: the relative-logit table of the step's tokens and, on a full-attention
     * layer, the log scaling of the normalized queries (vLLM {@code fused_qkvr_prep}).
     */
    private void bindInklingPostNorms() {
        queryScaleStep = step(AttnExtKernels.INKLING_QSCALE,
                AttnExtKernels.inklingScaleArgs(ubatch, spec.nHead(), spec.headDimK(), spec.qDim(), 0,
                        spec.isSwaLayer() ? 0 : logFloor, logAlpha, posBuffer, qBuf),
                AttnExtKernels.ARG_QSCALE_N_TOKENS, 1, (long) spec.nHead() * spec.headDimK(), 0, -1);
        relLogitsStep = step(AttnExtKernels.INKLING_REL,
                AttnExtKernels.inklingRelArgs(ubatch, spec.nHead(), relDim, relExtent,
                        spec.isSwaLayer() ? 0 : logFloor, logAlpha, rBuf, relProjBuf, posBuffer, relTableBuf),
                AttnExtKernels.ARG_REL_N_TOKENS, 1, (long) spec.nHead() * relExtent, 0, -1);
        input(queryScaleStep, false);
        input(relLogitsStep, false);
    }

    /** A scratch temporary of {@code floats}, which must start at element 0. */
    private ScratchSlot inklingTemp(GpuScratch scratch, String name, long floats) {
        ScratchSlot slot = scratch.temp(name, floats);
        if (slot.offset() != 0) {
            throw new IllegalStateException("scratch slot " + slot.name() + " starts at element " + slot.offset());
        }
        return slot;
    }


    /**
     * Loads the FP8 KV-cache scales of the layer's cache: the per-head K and V scale buffers the FP8 store kernels
     * divide by and the combined {@code kv_scales} buffer (K values then V values) the flash kernels multiply the
     * dequantized rows by. No-op unless a cache dtype is checkpoint-scaled (an FP8 type or NVFP4): vLLM consumes the
     * checkpoint scales of an FP8 cache only and enforces 1.0 for every other (vllm kv_cache.py
     * process_weights_after_loading); NVFP4 follows the same rule through its store kernel's global scale
     * (vllm csrc/nvfp4_kv_cache_kernels.cu).
     */
    private void bindKvScales(GpuKvLayer kv) {
        KvLayout layout = kv.layout();
        float[] k = kvScaleValues(TensorNames.ATTN_K + "." + TensorClaims.KV_SCALE, layout.dtypeK());
        float[] v = kvScaleValues(TensorNames.ATTN_V + "." + TensorClaims.KV_SCALE, layout.dtypeV());
        if (k == null && v == null) {
            return;
        }
        int heads = spec.nHeadKv();
        float[] both = new float[2 * heads];
        for (int h = 0; h < heads; h++) {
            both[h] = k != null ? k[h] : 1.0f;
            both[heads + h] = v != null ? v[h] : 1.0f;
        }
        kvScales = floatBuffer(both);
        if (kv.vendorSide() != null) {
            GpuEvent e = ctx.queue().enqueueWriteBuffer(kv.vendorSide(), 0, (long) both.length * Float.BYTES, both);
            if (e != null) {
                e.close();
            }
        }
        kScales = k != null ? floatBuffer(k) : null;
        vScales = v != null ? floatBuffer(v) : null;
    }

    /**
     * The FP8 cache scales of a host-KV layer's planned cache: the host attention stores and decodes with the same
     * per-head values the device kernels would.
     */
    private void bindHostKvScales() {
        KvLayout layout = ctx.memory().plan().layer(layer.il()).kv();
        if (layout == null) {
            // a shared-cache reader stores no rows and its scale tensors are absent, as on the device path: ones
            return;
        }
        hostKScale = kvScaleValues(TensorNames.ATTN_K + "." + TensorClaims.KV_SCALE, layout.dtypeK());
        hostVScale = kvScaleValues(TensorNames.ATTN_V + "." + TensorClaims.KV_SCALE, layout.dtypeV());
    }

    /**
     * Loads the learned static sink tokens of the openPangu sink attention: the {@code [n_sink * n_head_kv,
     * head_dim]} keys normalized per KV head by the layer's K norm (vLLM
     * {@code OpenPanguSinkAttention.post_weight_load}) and left un-rotated, and the matching values (zeros when the
     * checkpoint omits {@code attn_sink_v}). Host copies serve a host attention; the device buffers serve flash
     * attention.
     *
     * @throws IllegalStateException    when the sink keys are missing
     * @throws IllegalArgumentException when a sink tensor or the K norm weight has the wrong size
     */
    private void bindStaticSinks() {
        int n = spec.sinkTokens();
        if (n == 0) {
            return;
        }
        int heads = spec.nHeadKv();
        int dK = spec.headDimK();
        int dV = spec.headDimV();
        GpuWeight keyWeight = weights().layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_SINK_K));
        if (keyWeight == null) {
            throw new IllegalStateException("layer " + layer.il() + ": tensor "
                    + TensorNames.weight(TensorNames.ATTN_SINK_K) + " is missing");
        }
        float[] rawKeys = hostVector(keyWeight, (long) n * heads * dK);
        GpuWeight normWeight = weights().layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_K_NORM));
        float[] weight = normWeight != null ? hostVector(normWeight, dK) : null;
        hostSinkK = new float[n * heads * dK];
        NormMath.rowNorm(NormKind.RMS, rmsEps(), n * heads, dK, dK, 1, rawKeys, 0, weight, null, hostSinkK, 0);
        GpuWeight valueWeight = weights().layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_SINK_V));
        hostSinkV = valueWeight != null
                ? hostVector(valueWeight, (long) n * heads * dV) : new float[n * heads * dV];
        if (!hostKv) {
            sinkKBuf = floatBuffer(hostSinkK);
            sinkVBuf = floatBuffer(hostSinkV);
        }
    }

    /** The F32 values of a weight as a host array of {@code n} values, after the read has completed. */
    private float[] hostVector(GpuWeight w, long n) {
        GpuBuffer b = vector(w, n);
        float[] out = new float[(int) n];
        GpuEvent read = ctx.queue().enqueueReadBuffer(b, 0, n * Float.BYTES, out);
        if (read != null) {
            try {
                read.awaitCompletion();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while reading " + w.name()
                        + " for the static sink tokens", e);
            } finally {
                read.close();
            }
        }
        return out;
    }

    /**
     * The checkpoint KV-cache scale of {@code suffix} as one value per KV head (a {@code [1]} scale repeated), or
     * null when {@code type} is not an FP8 cache or the checkpoint carries none (the store and attention then scale
     * by 1.0, as vLLM does for an FP8 cache without {@code k_scale}/{@code v_scale}).
     *
     * @throws IllegalArgumentException when the tensor is not a plain F32 vector of 1 or {@code n_head_kv} values or
     *                                 holds a non-finite or non-positive scale
     */
    private float[] kvScaleValues(String suffix, KvDType type) {
        if (!type.checkpointScaled()) {
            return null;
        }
        GpuWeight w = weights().layerOrNull(layer, suffix);
        int heads = spec.nHeadKv();
        if (w == null) {
            float[] ones = new float[heads];
            Arrays.fill(ones, 1.0f);
            return ones;
        }
        long total = w.rows() * w.cols();
        if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.hasScale() || w.inverseHadamard() != null
                || (total != 1 && total != heads)) {
            throw new IllegalArgumentException("layer " + layer.il() + ": " + w.name() + " must be a plain F32 tensor"
                    + " of 1 or " + heads + " values (one KV-cache scale), holds " + total + " values as " + w.type());
        }
        float[] v = new float[(int) total];
        GpuEvent e = ctx.queue().enqueueReadBuffer(w.shard(0), 0, total * Float.BYTES, v);
        if (e != null) {
            e.close();
        }
        for (float value : v) {
            if (!Float.isFinite(value) || value <= 0f) {
                throw new IllegalArgumentException("layer " + layer.il() + ": " + w.name() + " holds " + value
                        + "; KV-cache scales must be finite and positive");
            }
        }
        if (total == heads) {
            return v;
        }
        float[] out = new float[heads];
        Arrays.fill(out, v[0]);
        return out;
    }

    /** An F32 device buffer of {@code values}, owned by this mixer's resources. */
    private GpuBuffer floatBuffer(float[] values) {
        GpuBuffer b = resources.allocate((long) values.length * Float.BYTES);
        GpuEvent e = ctx.queue().enqueueWriteBuffer(b, 0, (long) values.length * Float.BYTES, values);
        if (e != null) {
            e.close();
        }
        return b;
    }

    /**
     * Prepares the RoPE context shift of a layer that owns a RoPE cache in a model llama.cpp can shift. A cache whose
     * geometry the shift kernel cannot address (for example a block-type cache whose head stride is not a multiple of
     * 32)
     * does not fail the bind, because the shift is optional: {@link #shiftContext} reports it instead.
     */
    private void bindShift(GpuKvLayer kv) {
        if (!GpuContextShift.shifts(layer) || GpuContextShift.refusal(ctx.spec(), layer) != null) {
            return;
        }
        try {
            shift = GpuContextShift.of(ops, ctx.memory(), layer, kv, ropeFactorBuf);
        } catch (IllegalArgumentException e) {
            shiftBlocker = e.getMessage();
        }
    }

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

    // ------------------------------------------------------------------ projections

    private void bindProjections(GpuScratch scratch) {
        int qw = proj.qProjWidth();
        qProjBuf = proj.interleavedGate() ? scratch.temp(T_QGATE, (long) ubatch * qw).buffer() : qBuf;
        if (proj.qLatentDim() > 0) {
            qLatentBuf = scratch.temp(T_QLAT, (long) ubatch * qw).buffer();
            qProjBuf = qLatentBuf;
        }
        if (proj.source() == QkvProjection.Source.FUSED) {
            wqkv = weights().layer(layer, proj.qkvWeight());
            QkvProjection.checkShape(wqkv.name(), wqkv.cols(), wqkv.rows(), wqkv.cols(), proj.fusedWidth());
            qkvSite = serveDeltas(proj.qkvWeight(), wqkv);
            qSite = serveDeltas(TensorNames.weight(TensorNames.ATTN_Q), wqkv);
            kSite = serveDeltas(TensorNames.weight(TensorNames.ATTN_K), wqkv);
            vSite = serveDeltas(TensorNames.weight(TensorNames.ATTN_V), wqkv);
            int fw = proj.fusedWidth();
            qkvTemp = scratch.temp(T_QKV, (long) ubatch * fw).buffer();
            if (proj.qkvBias() != null) {
                input(biasStep(vector(proj.qkvBias(), fw), fw, qkvTemp), true);
            }
            input(copyStep(qkvTemp, proj.qOffset(), fw, qProjBuf, 0, qw, 1, qw), false);
            if (proj.projectsKv()) {
                input(copyStep(qkvTemp, proj.kOffset(), fw, kBuf, 0, spec.kDim(), 1, spec.kDim()), true);
                input(copyStep(qkvTemp, proj.vOffset(), fw, vBuf, 0, spec.vDim(), 1, spec.vDim()), true);
            }
        } else {
            wq = weights().layer(layer, proj.qWeight());
            QkvProjection.checkShape(wq.name(), wq.cols(), wq.rows(), wq.cols(), qw);
            qSite = serveDeltas(proj.qWeight(), wq);
            if (proj.qLatentDim() > 0) {
                wqUp = weights().layer(layer, proj.qUpWeight());
                QkvProjection.checkShape(wqUp.name(), wqUp.cols(), wqUp.rows(), proj.qLatentDim(), spec.qDim());
                GpuBuffer qn = vector(proj.qNormWeight(), proj.qLatentDim());
                qLatentNorm = normStep(NormKind.RMS, rmsEps(), qLatentBuf, 1, proj.qLatentDim(), qn, null, 1);
            }
            if (proj.projectsKv()) {
                wk = weights().layer(layer, proj.kWeight());
                QkvProjection.checkShape(wk.name(), wk.cols(), wk.rows(), wq.cols(), spec.kDim());
                kSite = serveDeltas(proj.kWeight(), wk);
                if (proj.valueExperts()) {
                    bindValueExperts();
                } else if (proj.vFromK()) {
                    input(copyStep(kBuf, 0, spec.kDim(), vBuf, 0, spec.vDim(), 1, spec.kDim()), true);
                } else {
                    wv = weights().layer(layer, proj.vWeight());
                    QkvProjection.checkShape(wv.name(), wv.cols(), wv.rows(), wq.cols(), spec.vDim());
                    vSite = serveDeltas(proj.vWeight(), wv);
                }
            }
        }
        if (proj.qBias() != null) {
            input(biasStep(vector(proj.qBias(), qw), qw, qProjBuf), false);
        }
        if (proj.projectsKv()) {
            if (proj.kBias() != null) {
                input(biasStep(vector(proj.kBias(), spec.kDim()), spec.kDim(), kBuf), true);
            }
            if (proj.vBias() != null) {
                input(biasStep(vector(proj.vBias(), spec.vDim()), spec.vDim(), vBuf), true);
            }
        }
        if (spec.clampKqv() > 0f) {
            ActParams clamp = ActParams.builder().clamp(-spec.clampKqv(), spec.clampKqv()).build();
            input(clampStep(clamp, qProjBuf, qw), false);
            if (proj.projectsKv()) {
                input(clampStep(clamp, kBuf, spec.kDim()), true);
                input(clampStep(clamp, vBuf, spec.vDim()), true);
            }
        }
        int hd = spec.headDimK();
        if (proj.interleavedGate()) {
            gateTemp = scratch.temp(T_GATE, (long) ubatch * spec.qDim()).buffer();
            input(copyStep(qProjBuf, 0, 2 * hd, qBuf, 0, hd, spec.nHead(), hd), false);
            input(copyStep(qProjBuf, hd, 2 * hd, gateTemp, 0, hd, spec.nHead(), hd), false);
        } else if (proj.gateWeight() != null) {
            wgate = weights().layer(layer, proj.gateWeight());
            int gw = AttnExtMath.gateWidth(spec.gate(), spec.nHead(), spec.headDimV());
            QkvProjection.checkShape(wgate.name(), wgate.cols(), wgate.rows(), wgate.cols(), gw);
            gateSite = serveDeltas(proj.gateWeight(), wgate);
            gateTemp = scratch.temp(T_GATE, (long) ubatch * gw).buffer();
        }
        bindAdapters(scratch);
    }

    /**
     * The per-use Q/K/V adapters of a shared attention block: {@code blk.<qkvAdapterLayer>.attn_{q,k,v}_adapter_a}
     * maps the block input to the adapter rank and {@code _b} back to the projection width; their product is added
     * to the base projections before RoPE (vLLM {@code Zamba2LoRA}).
     *
     * @throws IllegalArgumentException when an adapter has the wrong shape
     */
    private void bindAdapters(GpuScratch scratch) {
        int rank = spec.qkvAdapterRank();
        if (rank <= 0) {
            return;
        }
        GpuWeights w = ctx.weightSource().weightsFor(spec.qkvAdapterLayer());
        int il = spec.qkvAdapterLayer();
        int in = (int) wq.cols();
        wqAdapterA = adapter(w, il, "attn_q_adapter_a", in, rank);
        wqAdapterB = adapter(w, il, "attn_q_adapter_b", rank, spec.qDim());
        wkAdapterA = adapter(w, il, "attn_k_adapter_a", in, rank);
        wkAdapterB = adapter(w, il, "attn_k_adapter_b", rank, spec.kDim());
        wvAdapterA = adapter(w, il, "attn_v_adapter_a", in, rank);
        wvAdapterB = adapter(w, il, "attn_v_adapter_b", rank, spec.vDim());
        adapterRankBuf = scratch.temp(T_ADAPTER, (long) ubatch * rank);
    }

    private GpuWeight adapter(GpuWeights w, int il, String base, long cols, long rows) {
        GpuWeight a = w.layer(il, TensorNames.weight(base));
        QkvProjection.checkShape(a.name(), a.cols(), a.rows(), cols, rows);
        return a;
    }

    /**
     * limite value embeddings (models/limite.cpp:87-89, 149-155): the lookup of each token's {@code value_embd} row
     * and the per-KV-head {@code attn_ve_gate} projection of {@code in} are the inputs of the add step that runs
     * after the V projection. The lookup reads the step's token ids from the device when the table is one buffer and
     * from the step's host record otherwise; the gate product runs with the other projections.
     *
     * @throws IllegalArgumentException when the table or the gate has the wrong shape, or the token-id slot is not
     *                                  at offset 0
     */
    private void bindValueEmbeddings(GpuScratch scratch) {
        if (!spec.valueEmbeddings()) {
            return;
        }
        if (!proj.projectsKv()) {
            throw new IllegalArgumentException("layer " + layer.il() + ": value embeddings need the layer's own V"
                    + " projection");
        }
        valueEmbd = weights().get(TensorNames.weight(TensorNames.VALUE_EMBD));
        if (valueEmbd.cols() != spec.vDim()) {
            throw new IllegalArgumentException("layer " + layer.il() + ": " + valueEmbd.name() + " holds rows of "
                    + valueEmbd.cols() + " values, V is " + spec.vDim() + " wide");
        }
        int vDim = spec.vDim();
        veValues = scratch.temp(T_VE, (long) ubatch * vDim);
        GpuBuffer gateBuf;
        if (spec.valueEmbeddingGate()) {
            veGateW = weights().layer(layer, TensorNames.weight(TensorNames.ATTN_VE_GATE));
            int nEmbd = hparams().nEmbd();
            if (veGateW.cols() != nEmbd || veGateW.rows() != spec.nHeadKv()) {
                throw new IllegalArgumentException("layer " + layer.il() + ": "
                        + TensorNames.weight(TensorNames.ATTN_VE_GATE) + " must be [" + nEmbd + ", "
                        + spec.nHeadKv() + "], got [" + veGateW.cols() + ", " + veGateW.rows() + "]");
            }
            veGate = scratch.temp(T_VE_GATE, (long) ubatch * spec.nHeadKv());
            gateBuf = veGate.buffer();
        } else {
            gateBuf = ops.placeholders().readOnlyFloats();
        }
        if (valueEmbd.isSingleShard()) {
            stepTokenIds = scratch.slot(ScratchSlots.TOKEN_IDS);
            if (stepTokenIds.offset() != 0) {
                throw new IllegalStateException("the token-id slot must start at element 0 of its buffer for the "
                        + "value-embedding lookup in " + valueEmbd.name() + ", got offset " + stepTokenIds.offset());
            }
        } else {
            hostStepIds = new int[ubatch];
        }
        input(TokenStep.action(n -> lookupValueEmbeddings(n)), true);
        Object[] args = AttnExtKernels.valueEmbeddingArgs(ubatch, vDim, spec.nHeadKv(), spec.headDimV(),
                spec.valueEmbeddingGate(), veValues.buffer(), gateBuf, vBuf);
        input(step(AttnExtKernels.VALUE_EMBEDDINGS, args, AttnExtKernels.ARG_VE_N_TOKENS, 1, vDim, 0, -1), true);
    }

    /** Loads the step's {@code value_embd} rows into {@link #veValues} by the step's token ids. */
    private void lookupValueEmbeddings(int n) {
        if (stepTokenIds != null) {
            ops.embed(valueEmbd, stepTokenIds.buffer(), n, veValues.buffer(), veValues.offset());
        } else {
            ctx.scratch().readStepTokens(hostStepIds, 0, n);
            ops.embed(valueEmbd, hostStepIds, n, veValues.buffer(), veValues.offset());
        }
    }

    /** The {@code attn_ve_gate @ in} product of the value-embedding gate; a no-op without the gate. */
    private void projectValueEmbeddingGate(int n, ScratchSlot in) {
        if (veGateW != null) {
            ops.matmul(veGateW, in.buffer(), in.offset(), veGate.buffer(), veGate.offset(), n);
        }
    }

    /**
     * limite exclusive self attention (models/limite.cpp:177-191): the per-head {@code attn_xsa_alpha} and the
     * output step that removes each head's projection on the token's own value, before the output norm.
     *
     * @throws IllegalArgumentException when the alpha tensor does not hold one value per query head
     */
    private void bindXsa() {
        if (!(spec.xsaEps() > 0f)) {
            return;
        }
        if (!proj.projectsKv()) {
            throw new IllegalArgumentException("layer " + layer.il() + ": the exclusive self attention needs the"
                    + " layer's own V projection");
        }
        xsaAlpha = vector(TensorNames.weight(TensorNames.ATTN_XSA_ALPHA), spec.nHead());
        Object[] args = AttnExtKernels.xsaArgs(ubatch, spec.nHeadKv(), spec.gqaRatio(), spec.headDimV(), spec.vDim(),
                spec.xsaEps(), xsaAlpha, vBuf, attnBuf);
        outputSteps.add(step(AttnExtKernels.XSA, args, AttnExtKernels.ARG_XSA_N_TOKENS, 1, spec.nHeadKv(), 0, -1));
    }

    /** Adds the per-use adapter products to the projected Q, K and V rows of the step. */
    private void runAdapters(int n, ScratchSlot in) {
        if (adapterRankBuf == null) {
            return;
        }
        adapterRankBuf.require((long) n * spec.qkvAdapterRank());
        addAdapter(wqAdapterA, wqAdapterB, n, in, qProjBuf);
        addAdapter(wkAdapterA, wkAdapterB, n, in, kBuf);
        addAdapter(wvAdapterA, wvAdapterB, n, in, vBuf);
    }

    private void addAdapter(GpuWeight a, GpuWeight b, int n, ScratchSlot in, GpuBuffer target) {
        ops.matmul(a, in.buffer(), in.offset(), adapterRankBuf.buffer(), adapterRankBuf.offset(), n);
        ops.matvecAdd(b, adapterRankBuf.buffer(), adapterRankBuf.offset(), target, 0, n);
    }

    // ------------------------------------------------------------------ norms

    /** MoVA: V from the routed value experts on the layer input. */
    private void bindValueExperts() {
        GpuBuffer gateBias = proj.vGateBias() != null
                ? vector(proj.vGateBias(), spec.valueExperts().nExpert()) : null;
        valueExperts = new GpuValueExperts(ctx, layer, proj, spec.valueExperts(), gateBias, vBuf);
        if (valueExperts.inputWidth() != wq.cols()) {
            throw new IllegalArgumentException("layer " + layer.il() + ": value router " + proj.vGateWeight()
                    + " reads " + valueExperts.inputWidth() + " values, the Q projection " + wq.cols());
        }
    }

    // ------------------------------------------------------------------ norms

    /** The Q and K norms the graph applies ({@link QkNormGeometry#applies}); {@code skipK} leaves K to the store. */
    private void bindQkNorms(boolean skipK) {
        QkNormKind kind = spec.qkNorm();
        if (kind == QkNormKind.NONE) {
            return;
        }
        boolean hasQ = weights().layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_Q_NORM)) != null;
        boolean hasK = weights().layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_K_NORM)) != null;
        if (QkNormGeometry.applies(kind, ctx.spec().arch(), false, hasQ, hasK)) {
            addQkNorm(geometry(kind, spec.nHead()), TensorNames.ATTN_Q_NORM, qBuf, false);
        }
        if (!skipK && proj.projectsKv() && QkNormGeometry.applies(kind, ctx.spec().arch(), true, hasQ, hasK)) {
            addQkNorm(geometry(kind, spec.nHeadKv()), TensorNames.ATTN_K_NORM, kBuf, true);
        }
    }

    private QkNormGeometry geometry(QkNormKind kind, int nHeads) {
        return QkNormGeometry.of(kind, spec.headDimK(), nHeads, hparams().normEps(), rmsEps(), layer.il());
    }

    /** Gemma's weightless per-head RMS norm of V. */
    private void bindVNorm() {
        if (spec.vNorm() && proj.projectsKv()) {
            input(normStep(NormKind.RMS, rmsEps(), vBuf, spec.nHeadKv(), spec.headDimV(), null, null, 1), true);
        }
    }

    /**
     * Q or K norm over the dense rows of {@code x} in geometry {@code g}; a weight of one scalar per head is expanded
     * to {@code n_heads} rows ({@link #broadcastPerHead}). The norm bias is loaded only when the graph applies it.
     */
    private void addQkNorm(QkNormGeometry g, String base, GpuBuffer x, boolean cacheWrite) {
        QkNormOperand op = qkNormOperand(g, base);
        input(normStep(g.normKind(), g.eps(), x, g.rowsPerToken(), g.rowLen(), op.weight, op.bias, op.wRows),
                cacheWrite);
    }

    /** The normalized weight and bias buffers of a Q/K norm, expanded to per-head rows when needed. */
    private QkNormOperand qkNormOperand(QkNormGeometry g, String base) {
        GpuWeight w = weights().layerOrNull(layer, TensorNames.weight(base));
        GpuWeight b = spec.qkNormBias() ? weights().layerOrNull(layer, TensorNames.bias(base)) : null;
        int rowLen = g.rowLen();
        int wRows = 1;
        GpuBuffer wb = null;
        GpuBuffer bb = null;
        boolean perHeadScalar = w != null && g.perHeadScalar(w.cols(), w.rows());
        if (perHeadScalar) {
            wRows = g.nHeads();
            wb = broadcastPerHead(w, g.nHeads(), rowLen);
        } else if (w != null) {
            wRows = g.weightRows(w.name(), w.rows() * w.cols());
            wb = vector(w, (long) wRows * rowLen);
        }
        if (b != null) {
            bb = perHeadScalar && g.perHeadScalar(b.cols(), b.rows())
                    ? broadcastPerHead(b, g.nHeads(), rowLen) : vector(b, (long) wRows * rowLen);
        }
        return new QkNormOperand(wb, bb, wRows);
    }

    /** Per-head Q/K norm weight and bias views, or placeholders when the norm is weightless. */
    private static final class QkNormOperand {
        final GpuBuffer weight;
        final GpuBuffer bias;
        final int wRows;

        QkNormOperand(GpuBuffer weight, GpuBuffer bias, int wRows) {
            this.weight = weight;
            this.bias = bias;
            this.wRows = wRows;
        }

        int flags() {
            int f = 0;
            if (weight != null) {
                f |= NormKernels.FLAG_HAS_W;
            }
            if (bias != null) {
                f |= NormKernels.FLAG_HAS_B;
            }
            return f;
        }
    }

    /**
     * Binds the fused norm and rotation path of a layer whose Q and K norms are per-head RMS norms applied before
     * RoPE. With the multi-task kernel the Q norm+rotation, the K norm+rotation and the V norm run as one launch;
     * without it the norm+rotation of each projection is one {@link FusedKernels#RMS_NORM_ROPE} launch. {@code skipK}
     * leaves the K norm and rotation to the fused KV store; when that leaves the multi-task launch with a single
     * body (Q only, no V norm) the plain {@code RMS_NORM_ROPE} is used instead. Returns false when the layer's shapes
     * or the backend do not admit either form, leaving the unfused pair to the caller.
     */
    private boolean bindFusedNormRope(GpuBuffer ropePos, boolean skipK) {
        RopeSpec rope = layer.rope();
        if (rope == null || spec.noRope() || rope.type() == RopeType.NONE) {
            return false;
        }
        QkNormKind kind = spec.qkNorm();
        if (kind == QkNormKind.NONE || rope.fope() != null) {
            /* The fused norm+rope kernels evaluate the analytic rotation; FoPE pairs come from coefficients. */
            return false;
        }
        boolean hasQ = weights().layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_Q_NORM)) != null;
        boolean hasK = weights().layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_K_NORM)) != null;
        boolean kv = proj.projectsKv();
        if (!QkNormGeometry.applies(kind, ctx.spec().arch(), false, hasQ, hasK)) {
            return false;
        }
        QkNormGeometry qg = geometry(kind, spec.nHead());
        QkNormGeometry kg = kv && QkNormGeometry.applies(kind, ctx.spec().arch(), true, hasQ, hasK)
                ? geometry(kind, spec.nHeadKv()) : null;
        if (!perHeadRms(qg, spec.nHead(), spec.headDimK())
                || (kv && !skipK && kg == null)
                || (kg != null && !perHeadRms(kg, spec.nHeadKv(), spec.headDimK()))) {
            return false;
        }
        GpuBuffer ff = ropeFactors();
        ropeFactorBuf = ff;
        boolean hasFf = ff != null;
        GpuBuffer ffArg = hasFf ? ff : ops.placeholders().readOnlyFloats();
        QkNormOperand qop = qkNormOperand(qg, TensorNames.ATTN_Q_NORM);
        QkNormOperand kop = kg != null ? qkNormOperand(kg, TensorNames.ATTN_K_NORM) : null;
        GpuBuffer none = ops.placeholders().readOnlyFloats();

        boolean packed = (!skipK && kop != null) || spec.vNorm();
        if (GpuMultiTask.usable(ops) && GpuMultiTask.localUsable(local) && packed) {
            GpuMultiTask batch = new GpuMultiTask(ops, ubatch, "attn.normrope." + layer.il());
            try {
                batch.normRope(rope, qg.eps(), spec.nHead(), spec.headDimK(), spec.qDim(), spec.headDimK(),
                        rope.nOffs(), 0, 0, qop.flags(), qop.wRows, hasFf, ropePos, ffArg,
                        qop.weight != null ? qop.weight : none, qop.bias != null ? qop.bias : none, qBuf, qBuf);
                if (!skipK && kop != null) {
                    batch.normRope(rope, kg.eps(), spec.nHeadKv(), spec.headDimK(), spec.kDim(), spec.headDimK(),
                            rope.nOffs(), 0, 0, kop.flags(), kop.wRows, hasFf, ropePos, ffArg,
                            kop.weight != null ? kop.weight : none, kop.bias != null ? kop.bias : none, kBuf, kBuf);
                }
                if (spec.vNorm()) {
                    batch.rowNorm(spec.nHeadKv(), spec.headDimV(), spec.headDimV(), NormKind.RMS, 0, 1, 0, 0,
                            rmsEps(), vBuf, none, none, vBuf);
                }
                batch.build();
                normRopeBatch = batch;
                input(TokenStep.multiTask(batch), true);
                return true;
            } catch (RuntimeException e) {
                batch.close();
                if (ops.fusedMode().requiresFusion()) {
                    throw e;
                }
            }
        }
        if (!ops.fused(FusedKernels.RMS_NORM_ROPE)) {
            return false;
        }
        addFusedNormRope(rope, qg, qop, spec.qDim(), qBuf, ropePos, ffArg, hasFf, none);
        if (!skipK && kg != null) {
            addFusedNormRope(rope, kg, kop, spec.kDim(), kBuf, ropePos, ffArg, hasFf, none);
        }
        if (spec.vNorm()) {
            input(normStep(NormKind.RMS, rmsEps(), vBuf, spec.nHeadKv(), spec.headDimV(), null, null, 1), true);
        }
        return true;
    }

    private void addFusedNormRope(RopeSpec rope, QkNormGeometry g, QkNormOperand op, int rowStride, GpuBuffer x,
                                  GpuBuffer pos, GpuBuffer ff, boolean hasFf, GpuBuffer none) {
        Object[] args = FusedKernels.rmsNormRopeArgs(rope, ubatch, g.nHeads(), spec.headDimK(), rowStride,
                g.rowLen(), rope.nOffs(), ubatch, 0, 0, op.flags(), op.wRows, g.eps(), hasFf, pos, ff,
                op.weight != null ? op.weight : none, op.bias != null ? op.bias : none, x, x);
        input(step(FusedKernels.RMS_NORM_ROPE, args, FusedKernels.ROPE_ARG_N_TOKENS, 1,
                (long) g.nHeads() * local, 0, FusedKernels.ROPE_ARG_POS_STRIDE), true);
    }

    private static boolean perHeadRms(QkNormGeometry g, int nHeads, int headDim) {
        return g.normKind() == NormKind.RMS && g.rowsPerToken() == nHeads && g.rowLen() == headDim;
    }

    /**
     * A {@code [nHeads][rowLen]} F32 buffer, owned by this mixer, whose row {@code h} repeats head {@code h}'s scalar
     * of a {@code {1, nHeads}} tensor: the same product as ggml's broadcast multiply of the normalized
     * {@code [rowLen, nHeads, n_tokens]} rows. Filled once with one {@value GpuOps#COPY2D} per head reading its
     * scalar with source stride 0.
     */
    private GpuBuffer broadcastPerHead(GpuWeight w, int nHeads, int rowLen) {
        GpuBuffer scalars = vector(w, nHeads);
        GpuBuffer rows = resources.allocate((long) nHeads * rowLen * Float.BYTES);
        long[] global = {roundUp(rowLen)};
        long[] localSize = {local};
        for (int h = 0; h < nHeads; h++) {
            ops.enqueue(GpuOps.COPY2D, global, localSize, rowLen, 1, 0, h, 1, h * rowLen, scalars, rows);
        }
        return rows;
    }

    private ModelHparams hparams() {
        return ctx.spec().hparams();
    }

    private float rmsEps() {
        return hparams().normRmsEps();
    }

    // ------------------------------------------------------------------ RoPE

    private void bindRope(GpuBuffer ropePos, boolean skipK) {
        RopeSpec rope = layer.rope();
        if (rope == null || spec.noRope() || rope.type() == RopeType.NONE) {
            return;
        }
        if (rope.fope() != null) {
            bindFope(rope, ropePos, skipK);
            return;
        }
        GpuBuffer ff = ropeFactors();
        ropeFactorBuf = ff;
        boolean hasFf = ff != null;
        GpuBuffer ffArg = hasFf ? ff : ops.placeholders().readOnlyFloats();
        if (dualChunk != null) {
            bindDualChunkRope(rope, ffArg, hasFf);
            return;
        }
        input(ropeStep(rope, spec.nHead(), spec.qDim(), ropePos, ffArg, hasFf, qBuf), false);
        if (!skipK && proj.projectsKv()) {
            input(ropeStep(rope, spec.nHeadKv(), spec.kDim(), ropePos, ffArg, hasFf, kBuf), true);
        }
    }

    /** Binds the FoPE rotation of Q and K ({@link RopeKernels#ROPE_FOPE}) with the model's coefficient tensors. */
    private void bindFope(RopeSpec rope, GpuBuffer ropePos, boolean skipK) {
        TensorRef cosRef = rope.fope().cosCoef();
        TensorRef sinRef = rope.fope().sinCoef();
        GpuWeight cos = weights().getOrNull(cosRef.name());
        GpuWeight sin = weights().getOrNull(sinRef.name());
        if (cos == null || sin == null) {
            throw new IllegalStateException("layer " + layer.il() + ": FoPE coefficients " + cosRef.name() + " / "
                    + sinRef.name() + " are not resident");
        }
        long n = cosRef.numElements();
        fopeCosBuf = vector(cos, n);
        fopeSinBuf = vector(sin, n);
        float[] inv = rope.fope().invFreq();
        if (fopeInvBuf == null || !Arrays.equals(fopeInvValues, inv)) {
            fopeInvBuf = resources.allocate((long) inv.length * Float.BYTES);
            GpuEvent e = ctx.queue().enqueueWriteBuffer(fopeInvBuf, 0, (long) inv.length * Float.BYTES, inv);
            if (e != null) {
                e.close();
            }
            fopeInvValues = inv;
        }
        input(fopeStep(rope, spec.nHead(), spec.qDim(), ropePos, qBuf), false);
        if (!skipK && proj.projectsKv()) {
            input(fopeStep(rope, spec.nHeadKv(), spec.kDim(), ropePos, kBuf), true);
        }
    }

    /** The {@link RopeKernels#ROPE_FOPE} step of {@code nHeads} heads over {@code x} ({@code rowStride} wide). */
    private TokenStep fopeStep(RopeSpec rope, int nHeads, int rowStride, GpuBuffer pos, GpuBuffer x) {
        Object[] args = RopeKernels.fopeArgs(ubatch, nHeads, spec.headDimK(), rowStride, rope.nDims(), 0,
                rope.fope().numInvFreq(), rope.fope().numKeyValueHeads(), rope.fope().separateHeads(), pos,
                fopeCosBuf, fopeSinBuf, fopeInvBuf, x);
        return step(RopeKernels.ROPE_FOPE, args, RopeKernels.ARG_N_TOKENS, 1, nHeads, 0, -1);
    }

    /**
     * The five query rotations and the key rotation of dual-chunk attention: the projected query is copied into the
     * four spare planes, each plane rotates at its own derived position with the same RoPE, and the keys rotate at
     * {@code position % chunk_length} ({@link DualChunkRope}, vLLM {@code DualChunkRotaryEmbedding.forward}).
     */
    private void bindDualChunkRope(RopeSpec rope, GpuBuffer ff, boolean hasFf) {
        int pitch = dualChunkPlanePitch();
        for (int plane = 1; plane < DualChunkRope.VARIANTS; plane++) {
            input(copyStep(qBuf, 0, spec.qDim(), qBuf, plane * pitch, spec.qDim(), 1, spec.qDim()), false);
        }
        for (int plane = 0; plane < DualChunkRope.VARIANTS; plane++) {
            input(ropeStep(rope, spec.nHead(), spec.qDim(), dcPlanePos.get(plane), ff, hasFf, qBuf,
                    plane * pitch), false);
        }
        if (proj.projectsKv()) {
            input(ropeStep(rope, spec.nHeadKv(), spec.kDim(), dcKeyPos, ff, hasFf, kBuf, 0), true);
        }
    }

    /**
     * The layer's F32 frequency factors: the divisors the resolver computed (vLLM's fixed and mixed NTK scaling) or
     * the {@code rope_freqs} / long / short factors, or null when the rotation has none.
     */
    private GpuBuffer ropeFactors() {
        RopeSpec rope = layer.rope();
        if (rope == null) {
            return null;
        }
        float[] computed = rope.computedFreqFactors();
        if (computed != null) {
            if (computedFactorBuf == null || !Arrays.equals(computedFactorValues, computed)) {
                computedFactorBuf = resources.allocate((long) computed.length * Float.BYTES);
                GpuEvent e = ctx.queue().enqueueWriteBuffer(computedFactorBuf, 0,
                        (long) computed.length * Float.BYTES, computed);
                if (e != null) {
                    e.close();
                }
                computedFactorValues = computed;
            }
            return computedFactorBuf;
        }
        if (rope.freqFactors() == null) {
            return null;
        }
        TensorRef ref = rope.freqFactors();
        GpuWeight w = weights().getOrNull(ref.name());
        if (w == null) {
            throw new IllegalStateException("layer " + layer.il() + ": RoPE frequency factors " + ref.name()
                    + " are not resident");
        }
        return vector(w, rope.nDims() / 2);
    }

    private TokenStep ropeStep(RopeSpec rope, int nHeads, int rowStride, GpuBuffer pos, GpuBuffer ff, boolean hasFf,
                               GpuBuffer x) {
        return ropeStep(rope, nHeads, rowStride, pos, ff, hasFf, x, 0);
    }

    private TokenStep ropeStep(RopeSpec rope, int nHeads, int rowStride, GpuBuffer pos, GpuBuffer ff, boolean hasFf,
                               GpuBuffer x, int xOff) {
        Object[] args = RopeKernels.ropeExtArgs(rope, ubatch, nHeads, spec.headDimK(), rowStride, rope.nOffs(),
                ubatch, xOff, hasFf, pos, ff, x);
        long perToken = RopeKernels.ropeExtWorkItems(rope.type(), 1, nHeads, rope.nDims());
        return step(RopeKernels.ROPE_EXT, args, RopeKernels.ARG_N_TOKENS, 1, perToken, 0, RopeKernels.ARG_POS_STRIDE);
    }

    // ------------------------------------------------------------------ KV store

    /**
     * The fused K-row kernel this layer can use, or null: {@link FusedKernels#NORM_ROPE_STORE} when the K rows are
     * per-head RMS normalized before RoPE, {@link FusedKernels#ROPE_STORE} without a QK norm. Both write the cache
     * rows directly, so the layer skips its separate K norm/rotation and K store. Only plain F32/F16 caches without
     * a K rotation or per-token-head scale qualify, as llama.cpp's rope + set_rows fusion does; a V-only rotation is
     * still applied before the store.
     */
    private String fusedKStoreKernel(GpuKvLayer kv) {
        RotationPlan rotation = kv != null ? kv.rotation() : null;
        if (kv == null || hostKv || !proj.projectsKv() || proj.vFromK() || spec.qkNormAfterRope()
                || (rotation != null && rotation.rotateK())) {
            return null;
        }
        KvLayout layout = kv.layout();
        if (layout.vAliasesK() || (layout.dtypeK() != KvDType.F32 && layout.dtypeK() != KvDType.F16)
                || layout.widthK() != spec.kDim() || kScales != null || kv.rowScales() != null) {
            return null;
        }
        RopeSpec rope = layer.rope();
        if (rope == null || spec.noRope() || rope.type() == RopeType.NONE || rope.fope() != null) {
            /* The fused store kernels evaluate the analytic rotation; FoPE pairs come from coefficients. */
            return null;
        }
        QkNormKind kind = spec.qkNorm();
        if (kind == QkNormKind.NONE) {
            return ops.fused(FusedKernels.ROPE_STORE) ? FusedKernels.ROPE_STORE : null;
        }
        boolean hasQ = weights().layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_Q_NORM)) != null;
        boolean hasK = weights().layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_K_NORM)) != null;
        if (!QkNormGeometry.applies(kind, ctx.spec().arch(), true, hasQ, hasK)
                || !perHeadRms(geometry(kind, spec.nHeadKv()), spec.nHeadKv(), spec.headDimK())) {
            return null;
        }
        return ops.fused(FusedKernels.NORM_ROPE_STORE) ? FusedKernels.NORM_ROPE_STORE : null;
    }

    private void bindStore(GpuKvLayer kv, String fusedKernel, GpuBuffer ropePos) {
        KvLayout layout = kv.layout();
        GpuBuffer cellIdx = stepInputs.cellIdx(kv.group());
        GpuBuffer side = kv.vendorSide();
        int sideStride = side != null ? DeviceKvFormats.sideRowBytes(layout) : 0;
        if (fusedKernel == null) {
            storeSteps.add(storeStep(layout.dtypeK(), spec.kDim(), layout.widthK(), kBuf, cellIdx, kv.k(), kScales,
                    kv.rowScales(), side, sideStride, DeviceKvFormats.sidePlaneOffset(layout, false),
                    spec.headDimK(), 0));
        } else {
            storeSteps.add(fusedKStoreStep(kv, fusedKernel, ropePos));
        }
        storeSteps.add(storeStep(layout.dtypeV(), spec.vDim(), layout.widthV(), vBuf, cellIdx, kv.v(), vScales,
                kv.rowScales(), side, sideStride, DeviceKvFormats.sidePlaneOffset(layout, true),
                spec.headDimV(), 1));
    }

    /** The fused K-row step of {@link #fusedKStoreKernel}: normalize (first form only) and rotate into the cache. */
    private TokenStep fusedKStoreStep(GpuKvLayer kv, String kernel, GpuBuffer ropePos) {
        RopeSpec rope = layer.rope();
        KvLayout layout = kv.layout();
        int nHeads = spec.nHeadKv();
        int headDim = spec.headDimK();
        GpuBuffer ff = ropeFactorBuf;
        boolean hasFf = ff != null;
        GpuBuffer ffArg = hasFf ? ff : ops.placeholders().readOnlyFloats();
        GpuBuffer none = ops.placeholders().readOnlyFloats();
        GpuBuffer cellIdx = stepInputs.cellIdx(kv.group());
        int dstType = layout.dtypeK() == KvDType.F16 ? FusedKernels.KV_STORE_F16 : FusedKernels.KV_STORE_F32;
        Object[] args;
        int cellArg;
        int dstArg;
        int posStrideArg;
        if (FusedKernels.NORM_ROPE_STORE.equals(kernel)) {
            QkNormGeometry kg = geometry(spec.qkNorm(), nHeads);
            QkNormOperand kop = qkNormOperand(kg, TensorNames.ATTN_K_NORM);
            args = FusedKernels.normRopeStoreArgs(rope, ubatch, nHeads, headDim, spec.kDim(), headDim,
                    rope.nOffs(), ubatch, 0, kop.flags(), kop.wRows, kg.eps(), hasFf, headDim, layout.widthK(),
                    dstType, ropePos, ffArg, kop.weight != null ? kop.weight : none,
                    kop.bias != null ? kop.bias : none, kBuf, cellIdx, kv.k());
            cellArg = FusedKernels.STORE_ARG_CELL;
            dstArg = FusedKernels.STORE_ARG_DST;
            posStrideArg = FusedKernels.STORE_ARG_POS_STRIDE;
        } else {
            args = FusedKernels.ropeStoreArgs(rope, ubatch, nHeads, headDim, spec.kDim(), rope.nOffs(), ubatch, 0,
                    hasFf, headDim, layout.widthK(), dstType, ropePos, ffArg, kBuf, cellIdx, kv.k());
            cellArg = FusedKernels.ROPE_STORE_ARG_CELL;
            dstArg = FusedKernels.ROPE_STORE_ARG_DST;
            posStrideArg = FusedKernels.ROPE_STORE_ARG_POS_STRIDE;
        }
        TokenStep store = step(kernel, args, FusedKernels.STORE_ARG_N_TOKENS, 1, (long) nHeads * local, 0,
                posStrideArg);
        storeRebinds.add(new StoreRebind(store, cellArg, cellIdx, dstArg, kv.k(), false));
        return store;
    }

    private TokenStep storeStep(KvDType type, int rowElems, int dstRowStride, GpuBuffer src, GpuBuffer cellIdx,
                                GpuBuffer dst, GpuBuffer kvScale, GpuBuffer rowScales, GpuBuffer side,
                                int sideStride, int sideOff, int headElems, int vSide) {
        GpuBuffer scales = kvScale != null ? kvScale : ops.placeholders().kvScalesOnes();
        GpuBuffer rowScaleArg = rowScales != null ? rowScales : ops.placeholders().kvScalesOnes();
        GpuBuffer sideArg = side != null ? side : ops.placeholders().scratchInts();
        Object[] args = RopeKernels.kvStoreArgs(type, ubatch, rowElems, rowElems, 0, dstRowStride, headElems, src,
                cellIdx, scales, dst, rowScaleArg, sideArg, sideStride, sideOff, vSide);
        long perToken = type.perTokenHead() || type.vendorFormat()
                ? RopeKernels.kvStoreWorkItemsPerTokenHead(type, 1, rowElems, headElems)
                : RopeKernels.kvStoreWorkItems(type, 1, rowElems);
        TokenStep store = step(RopeKernels.kvStore(type), args, RopeKernels.ARG_N_TOKENS, 1, perToken, 0, -1);
        storeRebinds.add(new StoreRebind(store, RopeKernels.kvStoreCellArg(type), cellIdx,
                RopeKernels.kvStoreDstArg(type), dst, vSide != 0));
        return store;
    }

    // ------------------------------------------------------------------ output

    private void bindOutput() {
        GpuScratch scratch = ctx.scratch();
        bindXsa();
        if (proj.outHeadNorm()) {
            outputSteps.add(normStep(NormKind.RMS, rmsEps(), attnBuf, spec.nHead(), spec.headDimV(), null, null, 1));
        }
        AttnGateKind gate = spec.gate();
        if (gate != AttnGateKind.NONE) {
            int gw = AttnExtMath.gateWidth(gate, spec.nHead(), spec.headDimV());
            GpuBuffer bias = null;
            if (proj.gateBias() != null) {
                bias = vector(proj.gateBias(), spec.nHead());
            }
            boolean hasBias = bias != null;
            Object[] args = AttnExtKernels.outGateArgs(gate, ubatch, spec.nHead(), spec.headDimV(), attnBuf, 0,
                    spec.outDim(), gateTemp, 0, gw, hasBias ? bias : ops.placeholders().readOnlyFloats(), hasBias);
            outputSteps.add(step(AttnExtKernels.OUT_GATE, args, AttnExtKernels.ARG_GATE_N_TOKENS, 1,
                    spec.outDim(), 0, -1));
        }
        if (spec.outNormKind() != null) {
            outputSteps.add(normStep(spec.outNormKind(), spec.outNormEps(), attnBuf, 1, spec.outDim(), null, null, 1));
        }
        if (proj.subNormWeight() != null) {
            NormKind kind = layer.norm().kind();
            float eps = kind == NormKind.LAYER ? hparams().normEps() : rmsEps();
            GpuBuffer w = vector(proj.subNormWeight(), spec.outDim());
            GpuBuffer b = proj.subNormBias() != null ? vector(proj.subNormBias(), spec.outDim()) : null;
            outputSteps.add(normStep(kind, eps, attnBuf, 1, spec.outDim(), w, b, 1));
        }
        wo = weights().layer(layer, proj.outWeight());
        QkvProjection.checkShape(wo.name(), wo.cols(), wo.rows(), spec.outDim(), wo.rows());
        outSite = serveDeltas(proj.outWeight(), wo);
        if (wo.rows() > Integer.MAX_VALUE / Math.max(1, ubatch)) {
            throw new IllegalArgumentException("output projection " + wo.name() + " is too wide for one step");
        }
        int outRows = (int) wo.rows();
        if (proj.outBias() != null) {
            GpuBuffer bias = vector(proj.outBias(), outRows);
            outTemp = scratch.temp(T_OUT, (long) ubatch * outRows).buffer();
            outBias = prepare(BIAS_ADD, roundUp(outRows), bindArgs(outRows, ubatch, bias, outTemp));
        }
        if (spec.valueScale() != 0f) {
            long n = (long) ubatch * outRows;
            valueScale = prepare(GpuOps.AXPBY, roundUp(n), bindArgs((int) n, 0.0f, spec.valueScale(), 0, 0, 0,
                    ops.placeholders().readOnlyFloats(), attnBuf, attnBuf));
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
            /* The dense dual-chunk step never enqueues the flash group a patcher would claim. */
            return List.of();
        }
        return dualFlash == null
                ? List.of(flash.patcher())
                : List.of(flash.patcher(), dualFlash.patcher());
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
                uploadDualChunkKeyPositions(paged, nTokens);
            } else {
                uploadDualChunkKeyPositions(pos0, nTokens);
            }
        }
        projectKV(nTokens, in);
        runAdapters(nTokens, in);
        runAll(kvInputSteps, nTokens, paged);
        if (dualChunk != null) {
            if (paged != null) {
                scaleDualChunkK(paged);
            } else {
                scaleDualChunkK(0, nTokens, dualChunk.scalingFactor(pos0 + nTokens));
            }
        }
        if (storeRotation != null) {
            storeRotation.beforeStore(nTokens);
        }
        runAll(storeSteps, nTokens);
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
        return hostKScale;
    }

    /** The per-KV-head FP8 V cache scales of a host-KV layer, or null for an unscaled cache. */
    public float[] hostVScale() {
        ensureBound();
        return hostVScale;
    }

    /** The per-head attention sinks of a host-KV layer, or null without sinks. */
    public float[] hostSinkValues() {
        ensureBound();
        return hostSinks;
    }

    /** The normalized static sink keys of a host-KV layer {@code [n_sink][n_head_kv][d_k]}, or null. */
    public float[] hostSinkK() {
        ensureBound();
        return hostSinkK;
    }

    /** The static sink values of a host-KV layer {@code [n_sink][n_head_kv][d_v]}, or null. */
    public float[] hostSinkV() {
        ensureBound();
        return hostSinkV;
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
        requireHostKv();
        Objects.requireNonNull(in, "in");
        checkStep(pos0, nTokens);
        if (!proj.projectsKv()) {
            return;
        }
        requireLength(kOut, (long) nTokens * spec.kDim(), "K rows");
        requireLength(vOut, (long) nTokens * spec.vDim(), "V rows");
        checkStepInputs(pos0, nTokens);
        projectKV(nTokens, in);
        runAdapters(nTokens, in);
        runAll(kvInputSteps, nTokens, stepInputs != null ? stepInputs.currentPagedStep() : null);
        readInto(kBuf, kOut, (long) nTokens * spec.kDim());
        readInto(vBuf, vOut, (long) nTokens * spec.vDim());
        ctx.queue().finish();
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
        requireHostKv();
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(host, "host");
        checkStep(pos0, nTokens);
        requireLength(q, (long) nTokens * spec.qDim(), "Q rows");
        requireLength(attn, (long) nTokens * spec.outDim(), "attention rows");
        if (proj.projectsKv()) {
            requireLength(k, (long) nTokens * spec.kDim(), "K rows");
            requireLength(v, (long) nTokens * spec.vDim(), "V rows");
        } else if (k != null || v != null) {
            throw new IllegalArgumentException("layer " + layer.il() + " stores no K/V rows of its own; pass null"
                    + " rows");
        }
        project(nTokens, in);
        runAdapters(nTokens, in);
        runAll(inputSteps, nTokens);
        readInto(qBuf, q, (long) nTokens * spec.qDim());
        if (proj.projectsKv()) {
            readInto(kBuf, k, (long) nTokens * spec.kDim());
            readInto(vBuf, v, (long) nTokens * spec.vDim());
        }
        ctx.queue().finish();
        host.attend(pos0, nTokens, q, k, v, attn);
        writeFrom(attnBuf, attn, (long) nTokens * spec.outDim());
        runAll(outputSteps, nTokens);
        projectOutput(nTokens, out);
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

    private void requireHostKv() {
        String refusal = hostAttentionRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
    }

    private void requireLength(float[] rows, long expected, String what) {
        if (rows == null || rows.length < expected) {
            throw new IllegalArgumentException("layer " + layer.il() + ": " + what + " need " + expected
                    + " floats, got " + (rows == null ? "null" : rows.length));
        }
    }

    private void checkStep(int pos0, int nTokens) {
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": step of " + nTokens + " tokens outside [1, "
                    + ubatch + "]");
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("layer " + layer.il() + ": negative position " + pos0);
        }
    }

    /** Reads {@code count} floats from the start of a dense device buffer into the host rows. */
    private void readInto(GpuBuffer src, float[] dst, long count) {
        GpuEvent event = ctx.queue().enqueueReadBuffer(src, 0L, count * Float.BYTES, dst);
        if (event != null) {
            event.close();
        }
    }

    /** Writes host rows to the start of a dense device buffer. */
    private void writeFrom(GpuBuffer dst, float[] src, long count) {
        GpuEvent event = ctx.queue().enqueueWriteBuffer(dst, 0L, count * Float.BYTES, src);
        if (event != null) {
            event.close();
        }
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
        project(n, in);
        runAdapters(n, in);
        if (dualChunk != null) {
            uploadDualChunkPositions(step, n);
        }
        runAll(inputSteps, n, step);
        if (dualChunk != null) {
            scaleDualChunk(step);
        }
        if (rotation != null) {
            rotation.beforeStore(n);
        }
        runAll(storeSteps, n);
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
        projectOutput(n, out);
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
        project(n, in);
        runAdapters(n, in);
        if (dualChunk != null) {
            uploadDualChunkPositions(pos0, n);
        }
        runAll(inputSteps, n);
        if (dualChunk != null) {
            scaleDualChunk(0, n, dualChunk.scalingFactor(pos0 + n));
        }
        if (rotation != null) {
            rotation.beforeStore(n);
        }
        runAll(storeSteps, n);
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
            } else if (n == 1) {
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
        projectOutput(n, out);
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
        if (inkling) {
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
        for (StoreRebind rebind : storeRebinds) {
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

    private void checkStepInputs(int pos0, int n) {
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

    /**
     * The base products and their deltas. The deltas of a fused product's parts land in the fused rows, before the
     * split and the biases, matching graniteswitch's {@code Q/K/V += delta} on the sliced views
     * (models/granite-switch.cpp:350-359).
     */
    private void project(int n, ScratchSlot in) {
        GpuBuffer x = in.buffer();
        int off = in.offset();
        // The attention projections of a single token all read this row, so its Q8_1 blocks are quantized once.
        int width = (int) (wqkv != null ? wqkv.cols() : wq.cols());
        GpuOps.SharedQ8 share = n == 1 && GpuOps.shareableQ8Activations(off, width)
                ? ops.shareQ8Activations(x, off, width) : null;
        try {
            if (wqkv != null) {
                int cols = (int) wqkv.cols();
                int fw = proj.fusedWidth();
                ops.matmul(wqkv, x, off, qkvTemp, 0, n);
                addDeltas(qkvSite, n, x, off, cols, qkvTemp, 0, fw);
                addDeltas(qSite, n, x, off, cols, qkvTemp, proj.qOffset(), fw);
                addDeltas(kSite, n, x, off, cols, qkvTemp, proj.kOffset(), fw);
                addDeltas(vSite, n, x, off, cols, qkvTemp, proj.vOffset(), fw);
            } else {
                int cols = (int) wq.cols();
                ops.matmul(wq, x, off, qProjBuf, 0, n);
                addDeltas(qSite, n, x, off, cols, qProjBuf, 0, proj.qProjWidth());
                if (wk != null) {
                    ops.matmul(wk, x, off, kBuf, 0, n);
                    addDeltas(kSite, n, x, off, cols, kBuf, 0, spec.kDim());
                }
                if (wv != null) {
                    ops.matmul(wv, x, off, vBuf, 0, n);
                    addDeltas(vSite, n, x, off, cols, vBuf, 0, spec.vDim());
                }
                if (valueExperts != null) {
                    valueExperts.run(n, x, off);
                }
            }
            if (wqUp != null) {
                qLatentNorm.run(n);
                ops.matmul(wqUp, qProjBuf, 0, qBuf, 0, n);
            }
            if (wgate != null) {
                ops.matmul(wgate, x, off, gateTemp, 0, n);
                addDeltas(gateSite, n, x, off, (int) wgate.cols(), gateTemp, 0, (int) wgate.rows());
            }
            if (inkling) {
                ops.matmul(wr, x, off, rBuf, 0, n);
                addDeltas(rSite, n, x, off, (int) wr.cols(), rBuf, 0, spec.nHead() * relDim);
                if (rBiasAdd != null) {
                    rBiasAdd.setBuffer(ARG_BIAS_DATA, rBuf);
                    rBiasAdd.setInt(ARG_BIAS_SEQ, n);
                    rBiasAdd.dispatch();
                }
            }
            projectValueEmbeddingGate(n, in);
        } finally {
            if (share != null) {
                share.close();
            }
        }
    }

    /**
     * The base products the cache writes need: every part of a fused projection (its K and V rows are read from the
     * fused rows, so the product runs whole), otherwise K and V alone. The Q product, the gate and their deltas do
     * not run.
     */
    private void projectKV(int n, ScratchSlot in) {
        GpuBuffer x = in.buffer();
        int off = in.offset();
        projectValueEmbeddingGate(n, in);
        if (wqkv != null) {
            int cols = (int) wqkv.cols();
            int fw = proj.fusedWidth();
            ops.matmul(wqkv, x, off, qkvTemp, 0, n);
            addDeltas(qkvSite, n, x, off, cols, qkvTemp, 0, fw);
            addDeltas(kSite, n, x, off, cols, qkvTemp, proj.kOffset(), fw);
            addDeltas(vSite, n, x, off, cols, qkvTemp, proj.vOffset(), fw);
            return;
        }
        int cols = (int) wq.cols();
        if (wk != null) {
            ops.matmul(wk, x, off, kBuf, 0, n);
            addDeltas(kSite, n, x, off, cols, kBuf, 0, spec.kDim());
        }
        if (wv != null) {
            ops.matmul(wv, x, off, vBuf, 0, n);
            addDeltas(vSite, n, x, off, cols, vBuf, 0, spec.vDim());
        }
        if (valueExperts != null) {
            valueExperts.run(n, x, off);
        }
    }

    private static void addDeltas(GpuProjectionDeltas.Site site, int n, GpuBuffer x, int xOff, int xStride,
                                  GpuBuffer y, int yOff, int yStride) {
        if (site != null && !site.isEmpty()) {
            site.apply(n, x, xOff, xStride, y, yOff, yStride);
        }
    }

    /** The delta site of projection {@code weight}; a matmul that transforms its input (PRISM) refuses deltas. */
    private GpuProjectionDeltas.Site serveDeltas(String weight, GpuWeight w) {
        String refusal = w.hadamard() != null || (w.inputTransform() != null && !w.inputTransform().isIdentity())
                ? w.name() + " transforms its input (PRISM Hadamard or input transform) inside the matmul; a LoRA"
                + " delta would need the transformed rows" : null;
        return ctx.projectionDeltas().serve(layer, weight, refusal);
    }

    private void projectOutput(int n, ScratchSlot out) {
        int rows = (int) wo.rows();
        long count = (long) n * rows;
        out.require(count);
        GpuBuffer dst = out.buffer();
        int dstOff = out.offset();
        int cols = spec.outDim();
        if (outBias != null) {
            GpuBuffer biased = dstOff == 0 ? dst : outTemp;
            ops.matmul(wo, attnBuf, 0, biased, 0, n);
            addDeltas(outSite, n, attnBuf, 0, cols, biased, 0, rows);
            outBias.setBuffer(ARG_BIAS_DATA, biased);
            outBias.setInt(ARG_BIAS_SEQ, n);
            outBias.dispatch();
            if (biased != dst) {
                ops.copy(outTemp, 0, dst, dstOff, (int) count);
            }
        } else {
            ops.matmul(wo, attnBuf, 0, dst, dstOff, n);
            addDeltas(outSite, n, attnBuf, 0, cols, dst, dstOff, rows);
        }
        if (valueScale != null) {
            valueScale.setInt(ARG_AXPBY_B_OFF, dstOff);
            valueScale.setInt(ARG_AXPBY_OUT_OFF, dstOff);
            valueScale.setBuffer(ARG_AXPBY_B, dst);
            valueScale.setBuffer(ARG_AXPBY_OUT, dst);
            if (count != valueScaleCount) {
                valueScale.setInt(ARG_AXPBY_N, (int) count);
                valueScale.setGlobal(roundUp(count));
                valueScaleCount = count;
            }
            valueScale.dispatch();
        }
    }

    private static void runAll(List<TokenStep> steps, int n) {
        runAll(steps, n, null);
    }

    private static void runAll(List<TokenStep> steps, int n, PagedStep paged) {
        for (int i = 0; i < steps.size(); i++) {
            steps.get(i).run(n, paged);
        }
    }

    /** Records an input step for the whole step and, when {@code cacheWrite}, for the cache-write-only step. */
    private void input(TokenStep step, boolean cacheWrite) {
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
        if (nKeep < 0 || nDiscard <= 0) {
            throw new IllegalArgumentException("context shift needs nKeep >= 0 and nDiscard > 0, got " + nKeep + "/"
                    + nDiscard);
        }
        String refusal = shiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("context shift unsupported: " + refusal);
        }
        if (shift != null) {
            shift.apply();
        }
    }

    /**
     * Null when {@link #shiftContext} can run: a layer without its own projections, or a cached layer of a model
     * llama.cpp can shift whose cache geometry the shift kernel addresses.
     */
    @Override
    public String shiftRefusal() {
        ensureBound();
        if (hostKv) {
            return "layer " + layer.il() + " keeps its KV cache on the host; context shift over a host cache is"
                    + " not supported";
        }
        if (proj.source() == QkvProjection.Source.NONE) {
            return null;
        }
        if (!spec.usesMemory()) {
            return "layer " + layer.il() + " attends only to the current batch; it has no context to shift";
        }
        String refusal = GpuContextShift.refusal(ctx.spec(), layer);
        if (refusal != null) {
            return refusal;
        }
        return shiftBlocker != null ? "layer " + layer.il() + ": " + shiftBlocker : null;
    }

    // ================================================================== dispatch helpers

    private TokenStep step(String kernel, Object[] args, int countArg, long countPerToken, long itemsPerToken,
                           long itemsFixed, int posStrideArg) {
        long items = (long) ubatch * itemsPerToken + itemsFixed;
        PreparedDispatch d = prepare(kernel, roundUp(items), args);
        return new TokenStep(d, countArg, countPerToken, itemsPerToken, itemsFixed, posStrideArg, local);
    }

    private TokenStep biasStep(GpuBuffer bias, int dim, GpuBuffer data) {
        return step(BIAS_ADD, bindArgs(dim, ubatch, bias, data), ARG_BIAS_SEQ, 1, 0, dim, -1);
    }

    private TokenStep copyStep(GpuBuffer src, int srcOff, int srcStride, GpuBuffer dst, int dstOff, int dstStride,
                               int rowsPerToken, int cols) {
        Object[] args = bindArgs((int) ((long) ubatch * rowsPerToken), cols, srcStride, srcOff, dstStride, dstOff,
                src, dst);
        return step(GpuOps.COPY2D, args, ARG_COPY_ROWS, rowsPerToken, (long) rowsPerToken * cols, 0, -1);
    }

    private TokenStep clampStep(ActParams clamp, GpuBuffer x, int width) {
        int n = ActKernels.elementCount(ubatch, width);
        Object[] args = ActKernels.unaryArgs(ActOp.CLAMP, clamp, n, x, 0, x, 0);
        return step(ActKernels.UNARY, args, ActKernels.ARG_UNARY_N, width, width, 0, -1);
    }

    private TokenStep normStep(NormKind kind, float eps, GpuBuffer x, int rowsPerToken, int rowLen, GpuBuffer w,
                               GpuBuffer b, int wRows) {
        int rows = ActKernels.elementCount(ubatch, rowsPerToken);
        Object[] args = NormKernels.rowNormArgs(kind, eps, rows, rowLen, rowLen, x, 0, x, 0, w, b, wRows,
                ops.placeholders().readOnlyFloats());
        return step(NormKernels.ROW_NORM, args, NormKernels.ARG_ROW_NORM_N_ROWS, rowsPerToken,
                (long) rowsPerToken * local, 0, -1);
    }

    private PreparedDispatch prepare(String kernel, long global, Object[] args) {
        return resources.prepare(kernel, new long[]{global}, new long[]{local}, args);
    }

    private long roundUp(long items) {
        return Math.max(1, (items + local - 1) / local) * local;
    }

    private static Object[] bindArgs(Object... args) {
        return args;
    }

    /** The F32 values of a layer vector tensor ({@code suffix}), checked to hold {@code n} values. */
    private GpuBuffer vector(String suffix, long n) {
        return vector(weights().layer(layer, suffix), n);
    }

    /**
     * The F32 values of a vector tensor: the tensor's own buffer when it is a single F32 buffer, otherwise its rows
     * decoded once into a buffer this mixer owns.
     */
    private GpuBuffer vector(GpuWeight w, long n) {
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
                Arrays.<AutoCloseable>asList(shift, rotation, storeRotation, t5Bias, valueExperts,
                        normRopeBatch), null);
        normRopeBatch = null;
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
        valueExperts = null;
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "GpuAttentionMixer{" + (layer == null ? "unbound" : "layer " + layer.il() + ", " + proj) + '}';
    }

    // ================================================================== nested types

    /**
     * One cache store's rebind points: the prepared step, the argument slots its {@code cell_idx} and
     * {@code dst} buffers sit in, and the buffers they hold when bound. {@link #shadowStore} swaps the two buffers
     * for the rerun and restores them.
     */
    private static final class StoreRebind {
        final TokenStep step;
        final int cellArg;
        final GpuBuffer cell;
        final int dstArg;
        final GpuBuffer dst;
        /** Whether the store writes the V row ({@code false} writes K). */
        final boolean vSide;

        StoreRebind(TokenStep step, int cellArg, GpuBuffer cell, int dstArg, GpuBuffer dst, boolean vSide) {
            this.step = step;
            this.cellArg = cellArg;
            this.cell = cell;
            this.dstArg = dstArg;
            this.dst = dst;
            this.vSide = vSide;
        }
    }

    /** A prepared dispatch whose count argument and 1-D grid follow the step's token count. */
    private static final class TokenStep {
        private final PreparedDispatch dispatch;
        /** Persistent multi-task launch; when set the step runs it instead of {@link #dispatch}. */
        private final GpuMultiTask batch;
        /** Arbitrary per-step action (an embedding lookup); when set it replaces {@link #dispatch}. */
        private final IntConsumer action;
        private final int countArg;
        private final long countPerToken;
        private final long itemsPerToken;
        private final long itemsFixed;
        private final int posStrideArg;
        private final int local;
        private int lastTokens = -1;
        /** Runs the step over the step's segments instead of one launch over {@code n} tokens when set. */
        private SegmentStep segments;

        /** A step whose launches follow the step's sequence segments (Inkling's short convolutions). */
        private interface SegmentStep {
            void run(int n, PagedStep paged);
        }

        private TokenStep(PreparedDispatch dispatch, int countArg, long countPerToken, long itemsPerToken,
                          long itemsFixed, int posStrideArg, int local) {
            this(dispatch, null, null, countArg, countPerToken, itemsPerToken, itemsFixed, posStrideArg, local);
        }

        private TokenStep(PreparedDispatch dispatch, GpuMultiTask batch, int countArg, long countPerToken,
                          long itemsPerToken, long itemsFixed, int posStrideArg, int local) {
            this(dispatch, batch, null, countArg, countPerToken, itemsPerToken, itemsFixed, posStrideArg, local);
        }

        private TokenStep(PreparedDispatch dispatch, GpuMultiTask batch, IntConsumer action, int countArg,
                          long countPerToken, long itemsPerToken, long itemsFixed, int posStrideArg, int local) {
            this.dispatch = dispatch;
            this.batch = batch;
            this.action = action;
            this.countArg = countArg;
            this.countPerToken = countPerToken;
            this.itemsPerToken = itemsPerToken;
            this.itemsFixed = itemsFixed;
            this.posStrideArg = posStrideArg;
            this.local = local;
        }

        private static TokenStep multiTask(GpuMultiTask batch) {
            return new TokenStep(null, batch, 0, 1, 1, 0, -1, 1);
        }

        /** A step that runs {@code action} with the token count and launches no kernel of its own. */
        private static TokenStep action(IntConsumer action) {
            return new TokenStep(null, null, Objects.requireNonNull(action, "action"), 0, 1, 1, 0, -1, 1);
        }

        private void run(int n) {
            run(n, null);
        }

        private void run(int n, PagedStep paged) {
            if (segments != null) {
                segments.run(n, paged);
                return;
            }
            if (action != null) {
                action.accept(n);
                return;
            }
            if (batch != null) {
                batch.run(n);
                return;
            }
            if (n != lastTokens) {
                dispatch.setInt(countArg, (int) (n * countPerToken));
                if (posStrideArg >= 0) {
                    dispatch.setInt(posStrideArg, n);
                }
                long items = n * itemsPerToken + itemsFixed;
                dispatch.setGlobal(Math.max(1, (items + local - 1) / local) * local);
                lastTokens = n;
            }
            dispatch.dispatch();
        }
    }

    /** {@code attn_sinks} as F32 ({@code n_head} logits), or null without sinks. */
    private GpuBuffer sinks() {
        if (!spec.sinks()) {
            return null;
        }
        return vector(TensorNames.weight(TensorNames.ATTN_SINKS), spec.nHead());
    }

    /** {@code attn_sinks} in host memory for a host attention, or null without sinks. */
    private float[] hostSinks() {
        if (!spec.sinks()) {
            return null;
        }
        GpuWeight w = weights().layer(layer, TensorNames.weight(TensorNames.ATTN_SINKS));
        int n = spec.nHead();
        if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.rows() * w.cols() != n) {
            throw new UnsupportedOperationException("layer " + layer.il() + ": " + w.name() + " must be one F32"
                    + " buffer of " + n + " values for a host attention, got " + w);
        }
        float[] out = new float[n];
        GpuEvent read = ctx.queue().enqueueReadBuffer(w.shard(0), w.shardRowStart(0) * w.rowBytes(),
                (long) n * Float.BYTES, out);
        if (read != null) {
            try {
                read.awaitCompletion();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while reading " + w.name()
                        + " for the host attention", e);
            } finally {
                read.close();
            }
        }
        return out;
    }

    /**
     * The T5 relative-position bias table of the stack's bias layer ({@link T5RelativeBias#device}: F32
     * {@code [n_buckets][n_head]}, copied once at bind and owned by this mixer), or null without a T5 bias.
     */
    private GpuBuffer relBias() {
        if (spec.biasMode() == AttentionSpec.BiasMode.INKLING) {
            return relTableBuf;
        }
        if (!T5RelativeBias.uses(spec)) {
            return null;
        }
        t5Bias = T5RelativeBias.device(ctx, layer, spec);
        return t5Bias.buffer();
    }
}
