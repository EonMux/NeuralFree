package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
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
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.encdec.T5RelativeBias;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
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
 *   <li>projections ({@link QkvProjection}): Q, K and V land densely in the {@code Q}, {@code K} and {@code V}
 *       scratch slots ({@code [t][n_head * d]}); a fused {@code attn_qkv} product is split there with
 *       {@code LLM_Copy2D_F32}, as is V from the raw K projection when the layer has no {@code attn_v}
 *       ({@link QkvProjection#vFromK()}); on a MoVA layer V is the weighted sum of routed value experts
 *       ({@link GpuValueExperts}); biases with {@code LLM_Batch_BiasAdd}; the KQV clamp with
 *       {@code LLM_Unary CLAMP};
 *       an interleaved output gate is split off Q; other gates are projected from {@code in};</li>
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
 *   <li>flash decode / tiled prefill per {@link FlashAttnTiling} with {@code smemFloats = LlmKernels.faSmemFloats()},
 *       the per-row decode fallback ({@link FlashAttnTiling.Plan#PER_ROW_DECODE}), split-K partials merged by
 *       {@code LLM_FlashAttn_Combine}; sinks, softcap, ALiBi and the T5 relative bias are kernel arguments; K and V
 *       caches of different types use the mixed-type flash kernels
 *       ({@link AttnKernels#flashDecode(KvDType, KvDType)});</li>
 *   <li>output: the V rotation undone, muse-spark's weightless per-head RMS norm ({@code LLM_RowNorm}), the output
 *       gate ({@code LLM_AttnOutGate}), bitnet's sub-norm, {@code wo} (with
 *       its {@code .scale}), its bias, and the value scale ({@code f_attn_value_scale}, models/mimo2.cpp:174-176,
 *       models/dflash.cpp:992-993), written to {@code out}.</li>
 * </ol>
 * Every projection ({@code attn_qkv}, {@code attn_q}/{@code k}/{@code v} also as parts of a fused product,
 * {@code attn_gate}, {@code wo}) adds the {@link GpuProjectionDeltas} registered on it (LoRA) right after its base
 * product and before its bias, as {@code build_lora_mm} does (src/llama-graph.cpp:1629-1667).
 * The linear replacement is {@code out = wo @ in} (models/deci.cpp:114-117).
 *
 * <p>Positions: with a cache the mixer reads {@link GpuStepInputs} ({@code qPos}, {@code ropePos}, {@code kv_pos},
 * {@code cell_idx}), which the executor uploads once per step before the first layer; without a cache it reads the
 * {@link ScratchSlots#POS} slot, which the executor fills with the step's positions in the same layout
 * ({@code pos[t + k * nTokens]}, section 0 = {@code pos0 + t}).</p>
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
    private static final String T_QGATE = "attn.qgate";
    private static final String T_GATE = "attn.gate";
    private static final String T_OUT = "attn.out";

    private final List<PreparedDispatch> prepared = new ArrayList<>();
    private final List<GpuBuffer> owned = new ArrayList<>();
    private final List<TokenStep> inputSteps = new ArrayList<>();
    private final List<TokenStep> storeSteps = new ArrayList<>();
    private final List<TokenStep> outputSteps = new ArrayList<>();

    private GpuForwardContext ctx;
    private GpuOps ops;
    private LayerSpec layer;
    private AttentionSpec spec;
    private QkvProjection proj;
    private int local;
    private int ubatch;
    private boolean bound;
    private boolean closed;

    private GpuWeight wq;
    private GpuWeight wk;
    private GpuWeight wv;
    private GpuWeight wqkv;
    private GpuWeight wgate;
    private GpuWeight wo;

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
    private GpuBuffer qkvTemp;
    private GpuBuffer gateTemp;
    private GpuBuffer outTemp;

    private GpuStepInputs stepInputs;
    private GpuQuantizedKvRotation rotation;
    private Flash flash;
    private GpuBuffer ropeFactorBuf;
    private GpuContextShift shift;
    private T5RelativeBias.Device t5Bias;
    private String shiftBlocker;
    private PreparedDispatch outBias;
    private PreparedDispatch valueScale;
    private long valueScaleCount = -1;

    /** An unbound mixer; {@link #bind} reads the layer's attention spec. */
    public GpuAttentionMixer() {
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
        this.local = ops.local();
        this.ubatch = ctx.scratch().ubatch();
        this.proj = QkvProjection.of(layer, s -> weights().layerOrNull(layer, s) != null, ctx.spec().arch());
        this.spec = layer.attention();
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
        qBuf = offsetFree(scratch.slot(ScratchSlots.Q), (long) ubatch * spec.qDim());
        attnBuf = offsetFree(scratch.slot(ScratchSlots.ATTN_OUT), (long) ubatch * spec.outDim());
        if (proj.projectsKv()) {
            kBuf = offsetFree(scratch.slot(ScratchSlots.K), (long) ubatch * spec.kDim());
            vBuf = offsetFree(scratch.slot(ScratchSlots.V), (long) ubatch * spec.vDim());
        }
        GpuBuffer qPos;
        GpuBuffer ropePos;
        GpuKvLayer kv = null;
        if (spec.usesMemory()) {
            if (!ctx.hasMemory()) {
                throw new IllegalStateException("layer " + layer.il() + " attends through a KV cache but the model "
                        + ctx.spec().arch().ggufName() + " has no memory");
            }
            GpuMemory memory = ctx.memory();
            stepInputs = memory.stepInputs();
            kv = memory.kv(layer.il());
            if (kv == null) {
                throw new IllegalStateException("layer " + layer.il() + " has no KV cache in the memory plan");
            }
            checkCache(kv.layout());
            qPos = stepInputs.qPos();
            ropePos = stepInputs.ropePos();
        } else {
            ScratchSlot pos = scratch.slot(ScratchSlots.POS);
            qPos = offsetFree(pos, (long) ubatch * 4);
            ropePos = qPos;
        }
        bindProjections(scratch);
        if (!spec.qkNormAfterRope()) {
            bindQkNorms();
        }
        bindVNorm();
        bindRope(ropePos);
        if (spec.qkNormAfterRope()) {
            bindQkNorms();
        }
        if (spec.temperature().enabled()) {
            inputSteps.add(step(AttnExtKernels.TEMP_SCALE,
                    AttnExtKernels.tempScaleArgs(spec.temperature(), ubatch, spec.nHead(), spec.headDimK(), qBuf, 0,
                            spec.qDim(), qPos),
                    AttnExtKernels.ARG_TEMP_N_TOKENS, 1, (long) spec.qDim(), 0, -1));
        }
        if (kv != null) {
            rotation = GpuQuantizedKvRotation.of(ops, kv.rotation(), qBuf, spec.qDim(), kBuf, spec.kDim(), vBuf,
                    spec.vDim(), attnBuf, spec.outDim(), ubatch);
            if (proj.projectsKv()) {
                bindStore(kv);
            }
        }
        flash = kv != null ? Flash.cached(this, kv, qPos) : Flash.batch(this, qPos);
        bindOutput();
        bindShift(kv);
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
                inputSteps.add(biasStep(vector(proj.qkvBias(), fw), fw, qkvTemp));
            }
            inputSteps.add(copyStep(qkvTemp, proj.qOffset(), fw, qProjBuf, 0, qw, 1, qw));
            if (proj.projectsKv()) {
                inputSteps.add(copyStep(qkvTemp, proj.kOffset(), fw, kBuf, 0, spec.kDim(), 1, spec.kDim()));
                inputSteps.add(copyStep(qkvTemp, proj.vOffset(), fw, vBuf, 0, spec.vDim(), 1, spec.vDim()));
            }
        } else {
            wq = weights().layer(layer, proj.qWeight());
            QkvProjection.checkShape(wq.name(), wq.cols(), wq.rows(), wq.cols(), qw);
            qSite = serveDeltas(proj.qWeight(), wq);
            if (proj.projectsKv()) {
                wk = weights().layer(layer, proj.kWeight());
                QkvProjection.checkShape(wk.name(), wk.cols(), wk.rows(), wq.cols(), spec.kDim());
                kSite = serveDeltas(proj.kWeight(), wk);
                if (proj.valueExperts()) {
                    bindValueExperts();
                } else if (proj.vFromK()) {
                    inputSteps.add(copyStep(kBuf, 0, spec.kDim(), vBuf, 0, spec.vDim(), 1, spec.kDim()));
                } else {
                    wv = weights().layer(layer, proj.vWeight());
                    QkvProjection.checkShape(wv.name(), wv.cols(), wv.rows(), wq.cols(), spec.vDim());
                    vSite = serveDeltas(proj.vWeight(), wv);
                }
            }
        }
        if (proj.qBias() != null) {
            inputSteps.add(biasStep(vector(proj.qBias(), qw), qw, qProjBuf));
        }
        if (proj.projectsKv()) {
            if (proj.kBias() != null) {
                inputSteps.add(biasStep(vector(proj.kBias(), spec.kDim()), spec.kDim(), kBuf));
            }
            if (proj.vBias() != null) {
                inputSteps.add(biasStep(vector(proj.vBias(), spec.vDim()), spec.vDim(), vBuf));
            }
        }
        if (spec.clampKqv() > 0f) {
            ActParams clamp = ActParams.builder().clamp(-spec.clampKqv(), spec.clampKqv()).build();
            inputSteps.add(clampStep(clamp, qProjBuf, qw));
            if (proj.projectsKv()) {
                inputSteps.add(clampStep(clamp, kBuf, spec.kDim()));
                inputSteps.add(clampStep(clamp, vBuf, spec.vDim()));
            }
        }
        int hd = spec.headDimK();
        if (proj.interleavedGate()) {
            gateTemp = scratch.temp(T_GATE, (long) ubatch * spec.qDim()).buffer();
            inputSteps.add(copyStep(qProjBuf, 0, 2 * hd, qBuf, 0, hd, spec.nHead(), hd));
            inputSteps.add(copyStep(qProjBuf, hd, 2 * hd, gateTemp, 0, hd, spec.nHead(), hd));
        } else if (proj.gateWeight() != null) {
            wgate = weights().layer(layer, proj.gateWeight());
            int gw = AttnExtMath.gateWidth(spec.gate(), spec.nHead(), spec.headDimV());
            QkvProjection.checkShape(wgate.name(), wgate.cols(), wgate.rows(), wgate.cols(), gw);
            gateSite = serveDeltas(proj.gateWeight(), wgate);
            gateTemp = scratch.temp(T_GATE, (long) ubatch * gw).buffer();
        }
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

    /** The Q and K norms the graph applies ({@link QkNormGeometry#applies}). */
    private void bindQkNorms() {
        QkNormKind kind = spec.qkNorm();
        if (kind == QkNormKind.NONE) {
            return;
        }
        boolean hasQ = weights().layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_Q_NORM)) != null;
        boolean hasK = weights().layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_K_NORM)) != null;
        if (QkNormGeometry.applies(kind, ctx.spec().arch(), false, hasQ, hasK)) {
            addQkNorm(geometry(kind, spec.nHead()), TensorNames.ATTN_Q_NORM, qBuf);
        }
        if (proj.projectsKv() && QkNormGeometry.applies(kind, ctx.spec().arch(), true, hasQ, hasK)) {
            addQkNorm(geometry(kind, spec.nHeadKv()), TensorNames.ATTN_K_NORM, kBuf);
        }
    }

    private QkNormGeometry geometry(QkNormKind kind, int nHeads) {
        return QkNormGeometry.of(kind, spec.headDimK(), nHeads, hparams().normEps(), rmsEps(), layer.il());
    }

    /** Gemma's weightless per-head RMS norm of V. */
    private void bindVNorm() {
        if (spec.vNorm() && proj.projectsKv()) {
            inputSteps.add(normStep(NormKind.RMS, rmsEps(), vBuf, spec.nHeadKv(), spec.headDimV(), null, null, 1));
        }
    }

    /**
     * Q or K norm over the dense rows of {@code x} in geometry {@code g}; a weight of one scalar per head is expanded
     * to {@code n_heads} rows ({@link #broadcastPerHead}). The norm bias is loaded only when the graph applies it.
     */
    private void addQkNorm(QkNormGeometry g, String base, GpuBuffer x) {
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
        inputSteps.add(normStep(g.normKind(), g.eps(), x, g.rowsPerToken(), rowLen, wb, bb, wRows));
    }

    /**
     * A {@code [nHeads][rowLen]} F32 buffer, owned by this mixer, whose row {@code h} repeats head {@code h}'s scalar
     * of a {@code {1, nHeads}} tensor: the same product as ggml's broadcast multiply of the normalized
     * {@code [rowLen, nHeads, n_tokens]} rows. Filled once with one {@value GpuOps#COPY2D} per head reading its
     * scalar with source stride 0.
     */
    private GpuBuffer broadcastPerHead(GpuWeight w, int nHeads, int rowLen) {
        GpuBuffer scalars = vector(w, nHeads);
        GpuBuffer rows = ctx.backend().allocateBuffer((long) nHeads * rowLen * Float.BYTES);
        owned.add(rows);
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

    private void bindRope(GpuBuffer ropePos) {
        RopeSpec rope = layer.rope();
        if (rope == null || spec.noRope() || rope.type() == RopeType.NONE) {
            return;
        }
        GpuBuffer ff = ropeFactors();
        ropeFactorBuf = ff;
        boolean hasFf = ff != null;
        GpuBuffer ffArg = hasFf ? ff : ops.placeholders().readOnlyFloats();
        inputSteps.add(ropeStep(rope, spec.nHead(), spec.qDim(), ropePos, ffArg, hasFf, qBuf));
        if (proj.projectsKv()) {
            inputSteps.add(ropeStep(rope, spec.nHeadKv(), spec.kDim(), ropePos, ffArg, hasFf, kBuf));
        }
    }

    /** The layer's F32 frequency factors ({@code rope_freqs} or the selected long/short factors), or null. */
    private GpuBuffer ropeFactors() {
        RopeSpec rope = layer.rope();
        if (rope == null || rope.freqFactors() == null) {
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
        Object[] args = RopeKernels.ropeExtArgs(rope, ubatch, nHeads, spec.headDimK(), rowStride, rope.nOffs(),
                ubatch, 0, hasFf, pos, ff, x);
        long perToken = RopeKernels.ropeExtWorkItems(rope.type(), 1, nHeads, rope.nDims());
        return step(RopeKernels.ROPE_EXT, args, RopeKernels.ARG_N_TOKENS, 1, perToken, 0, RopeKernels.ARG_POS_STRIDE);
    }

    // ------------------------------------------------------------------ KV store

    private void bindStore(GpuKvLayer kv) {
        KvLayout layout = kv.layout();
        GpuBuffer cellIdx = stepInputs.cellIdx(kv.group());
        storeSteps.add(storeStep(layout.dtypeK(), spec.kDim(), layout.widthK(), kBuf, cellIdx, kv.k()));
        storeSteps.add(storeStep(layout.dtypeV(), spec.vDim(), layout.widthV(), vBuf, cellIdx, kv.v()));
    }

    private TokenStep storeStep(KvDType type, int rowElems, int dstRowStride, GpuBuffer src, GpuBuffer cellIdx,
                                GpuBuffer dst) {
        Object[] args = RopeKernels.kvStoreArgs(type, ubatch, rowElems, rowElems, 0, dstRowStride, src, cellIdx, dst);
        long perToken = RopeKernels.kvStoreWorkItems(type, 1, rowElems);
        return step(RopeKernels.kvStore(type), args, RopeKernels.ARG_N_TOKENS, 1, perToken, 0, -1);
    }

    // ------------------------------------------------------------------ output

    private void bindOutput() {
        GpuScratch scratch = ctx.scratch();
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
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        run(pos0, nTokens, in, out);
    }

    private void run(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        ensureBound();
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
        project(n, in);
        runAll(inputSteps, n);
        if (rotation != null) {
            rotation.beforeStore(n);
        }
        runAll(storeSteps, n);
        if (n == 1) {
            flash.decode(pos0);
        } else {
            flash.prefill(pos0, n);
        }
        if (rotation != null) {
            rotation.afterAttention(n);
        }
        runAll(outputSteps, n);
        projectOutput(n, out);
    }

    private void checkStepInputs(int pos0, int n) {
        if (stepInputs != null && (stepInputs.pos0() != pos0 || stepInputs.nTokens() != n)) {
            throw new IllegalStateException("layer " + layer.il() + " runs positions [" + pos0 + ", " + (pos0 + n)
                    + ") but the step inputs hold [" + stepInputs.pos0() + ", "
                    + (stepInputs.pos0() + stepInputs.nTokens()) + "); upload them before the first layer");
        }
    }

    /**
     * The base products and their deltas. The deltas of a fused product's parts land in the fused rows, before the
     * split and the biases, matching graniteswitch's {@code Q/K/V += delta} on the sliced views
     * (models/granite-switch.cpp:350-359).
     */
    private void project(int n, ScratchSlot in) {
        GpuBuffer x = in.buffer();
        int off = in.offset();
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
        if (wgate != null) {
            ops.matmul(wgate, x, off, gateTemp, 0, n);
            addDeltas(gateSite, n, x, off, (int) wgate.cols(), gateTemp, 0, (int) wgate.rows());
        }
    }

    private static void addDeltas(GpuProjectionDeltas.Site site, int n, GpuBuffer x, int xOff, int xStride,
                                  GpuBuffer y, int yOff, int yStride) {
        if (site != null && !site.isEmpty()) {
            site.apply(n, x, xOff, xStride, y, yOff, yStride);
        }
    }

    /** The delta site of projection {@code weight}; a matmul that rotates its input (PRISM) refuses deltas. */
    private GpuProjectionDeltas.Site serveDeltas(String weight, GpuWeight w) {
        String refusal = w.hadamard() != null ? w.name() + " rotates its input (PRISM Hadamard) inside the matmul;"
                + " a LoRA delta would need the rotated rows" : null;
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
        for (int i = 0; i < steps.size(); i++) {
            steps.get(i).run(n);
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
        PreparedDispatch d = ops.prepare(kernel, new long[]{global}, new long[]{local}, args);
        prepared.add(d);
        return d;
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
        GpuBuffer b = ctx.backend().allocateBuffer(total * Float.BYTES);
        owned.add(b);
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
                Arrays.<AutoCloseable>asList(shift, rotation, t5Bias, valueExperts), null);
        failure = Closeables.closeAll(prepared, failure);
        prepared.clear();
        failure = Closeables.closeAll(owned, failure);
        owned.clear();
        shift = null;
        flash = null;
        rotation = null;
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

    /** A prepared dispatch whose count argument and 1-D grid follow the step's token count. */
    private static final class TokenStep {
        private final PreparedDispatch dispatch;
        private final int countArg;
        private final long countPerToken;
        private final long itemsPerToken;
        private final long itemsFixed;
        private final int posStrideArg;
        private final int local;
        private int lastTokens = -1;

        private TokenStep(PreparedDispatch dispatch, int countArg, long countPerToken, long itemsPerToken,
                          long itemsFixed, int posStrideArg, int local) {
            this.dispatch = dispatch;
            this.countArg = countArg;
            this.countPerToken = countPerToken;
            this.itemsPerToken = itemsPerToken;
            this.itemsFixed = itemsFixed;
            this.posStrideArg = posStrideArg;
            this.local = local;
        }

        private void run(int n) {
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

    /**
     * Flash attention of one layer: the decode and prefill plans, their prepared kernels and the split-K combine.
     * Keys come from the cache rows ({@link #cached}) or from the current batch ({@link #batch}). Its dispatches are
     * registered with, and closed by, the mixer.
     */
    private static final class Flash {
        private final FlashAttnTiling decodePlan;
        private final FlashAttnTiling prefillPlan;
        private final PreparedDispatch decode;
        private final PreparedDispatch prefill;
        private final PreparedDispatch combine;
        private final boolean cached;
        private final boolean causal;
        private final int keyRows;
        private final int nHead;
        private final int qDim;
        private final int outDim;
        private final long partialCapacity;
        private int decodeSplits = 1;
        private int combineRows;

        private Flash(GpuAttentionMixer m, KvDType typeK, KvDType typeV, GpuBuffer k, int kRowStride, GpuBuffer v,
                      int vRowStride, GpuBuffer kvPos, GpuBuffer qPos, int keyRows, boolean cached) {
            AttentionSpec a = m.spec;
            this.cached = cached;
            this.causal = a.causal();
            this.keyRows = keyRows;
            this.nHead = a.nHead();
            this.qDim = a.qDim();
            this.outDim = a.outDim();
            int smem = m.ctx.kernels().faSmemFloats();
            int cu = Math.max(1, m.ctx.kernels().caps().getComputeUnits());
            decodePlan = FlashAttnTiling.decode(a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(), keyRows, smem, cu);
            prefillPlan = FlashAttnTiling.prefill(a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(), keyRows, smem,
                    cu);
            long cap = decodePlan.decodePartialFloats();
            if (prefillPlan.plan() == FlashAttnTiling.Plan.TILED) {
                cap = Math.max(cap, prefillPartialFloats(prefillPlan, m.ubatch, cu));
            }
            if (cap > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("flash partials of " + cap + " floats exceed one buffer");
            }
            this.partialCapacity = cap;
            GpuBuffer partial = m.ctx.scratch().partials((int) cap);
            GpuBuffer sinks = m.sinks();
            GpuBuffer relBias = m.relBias();
            GpuBuffer roFloats = m.ops.placeholders().readOnlyFloats();
            boolean hasSinks = sinks != null;
            GpuBuffer sinkArg = hasSinks ? sinks : roFloats;
            GpuBuffer biasArg = relBias != null ? relBias : roFloats;
            float maxBias = a.biasMode() == AttentionSpec.BiasMode.ALIBI ? a.alibiMaxBias() : 0f;
            int swa = a.effectiveSwaType().code();
            int nSwa = a.effectiveSwaType() == SwaType.NONE ? 0 : a.nSwa();
            int biasMode = a.biasMode().code();
            int buckets = a.t5Buckets();
            int maxDist = a.t5MaxDistance();
            decode = m.prepare2d(AttnKernels.flashDecode(typeK, typeV), decodePlan.decodeGlobal(1),
                    FlashAttnTiling.decodeLocal(), AttnKernels.withKvTypes(new Object[]{
                            a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(), decodePlan.gTile(), 1,
                            decodePlan.maxSplitLen(), 0, 1, 0, kRowStride, a.headDimK(), vRowStride, a.headDimV(),
                            a.scoreScale(), a.softcap(), maxBias, a.causal() ? 1 : 0, swa, nSwa, biasMode, buckets,
                            maxDist, hasSinks ? 1 : 0, 1, 0, 0,
                            m.qBuf, k, v, kvPos, sinkArg, biasArg, m.attnBuf, partial}, typeK, typeV));
            if (prefillPlan.plan() == FlashAttnTiling.Plan.TILED) {
                prefill = m.prepare2d(AttnKernels.flashAttn(typeK, typeV), prefillPlan.prefillGlobal(m.ubatch, 1),
                        FlashAttnTiling.prefillLocal(), AttnKernels.withKvTypes(new Object[]{
                                m.ubatch, a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(), prefillPlan.bq(),
                                prefillPlan.bk(), prefillPlan.gTile(), 1, prefillPlan.bk(), 0, 1, a.qDim(),
                                a.outDim(), kRowStride, a.headDimK(), vRowStride, a.headDimV(), a.scoreScale(),
                                a.softcap(), maxBias, a.causal() ? 1 : 0, swa, nSwa, biasMode, buckets, maxDist,
                                hasSinks ? 1 : 0, 1, 0, 0,
                                m.qBuf, k, v, qPos, kvPos, sinkArg, biasArg, m.attnBuf, partial}, typeK, typeV));
            } else {
                prefill = null;
            }
            combine = m.prepare2d(AttnKernels.FLASH_COMBINE, FlashAttnTiling.combineGlobal(a.nHead()),
                    FlashAttnTiling.combineLocal(), new Object[]{
                            a.nHead(), a.headDimV(), 1, a.nHead(), hasSinks ? 1 : 0, 0, partial, sinkArg, m.attnBuf});
            combineRows = a.nHead();
        }

        /** Flash over the layer's cache rows with {@code kv_pos} visibility. */
        static Flash cached(GpuAttentionMixer m, GpuKvLayer kv, GpuBuffer qPos) {
            KvLayout layout = kv.layout();
            GpuBuffer kvPos = m.stepInputs.kvPos(kv.group());
            return new Flash(m, layout.dtypeK(), layout.dtypeV(), kv.k(), layout.widthK(), kv.v(), layout.widthV(),
                    kvPos, qPos, layout.rows(), true);
        }

        /** Flash over the current batch's K and V rows (no cache): {@code kv_pos = q_pos}. */
        static Flash batch(GpuAttentionMixer m, GpuBuffer pos) {
            AttentionSpec a = m.spec;
            return new Flash(m, KvDType.F32, KvDType.F32, m.kBuf, a.kDim(), m.vBuf, a.vDim(), pos, pos, m.ubatch,
                    false);
        }

        /**
         * Partial floats a tiled prefill may split into: splits are chosen only while the call has fewer than two
         * workgroups per compute unit ({@link FlashAttnTiling#prefillSplit}), i.e. for at most
         * {@code ceil(2 * CU / (n_head_kv * groupTiles)) * bq} tokens.
         */
        private static long prefillPartialFloats(FlashAttnTiling p, int ubatch, int cu) {
            long groupsPerBlock = (long) p.nHeadKv() * p.groupTiles();
            long blocks = (2L * cu + groupsPerBlock - 1) / groupsPerBlock;
            long tokens = Math.min(ubatch, blocks * p.bq());
            return FlashAttnTiling.partialFloats(tokens * p.nHead(), FlashAttnTiling.MAX_PREFILL_SPLITS, p.dV());
        }

        void decode(int pos) {
            decodeRow(pos, 0, cached ? Math.min(keyRows, pos + 1) : 1);
        }

        void prefill(int pos0, int n) {
            int nKeys = cached ? Math.min(keyRows, pos0 + n) : n;
            if (prefill != null) {
                FlashAttnTiling.Split s = prefillPlan.prefillSplit(n, nKeys, partialCapacity);
                prefill.setInt(AttnKernels.ARG_FA_N_TOKENS, n);
                prefill.setInt(AttnKernels.ARG_FA_N_SPLIT, s.count());
                prefill.setInt(AttnKernels.ARG_FA_SPLIT_LEN, s.length());
                prefill.setInt(AttnKernels.ARG_FA_KV_END, nKeys);
                prefill.setInt(AttnKernels.ARG_FA_DIRECT_OUT, s.direct() ? 1 : 0);
                prefill.setGlobal(prefillPlan.prefillGlobal(n, s.count()));
                prefill.dispatch();
                if (!s.direct()) {
                    combine(n * nHead, s.count(), 0);
                }
                return;
            }
            for (int t = 0; t < n; t++) {
                int qPos = pos0 + t;
                int keys = cached && causal ? Math.min(keyRows, qPos + 1) : nKeys;
                decodeRow(qPos, t, keys);
            }
        }

        /**
         * One query row of the decode kernel. Split count and length follow the live key count
         * ({@link FlashAttnTiling#decodeSplit}); the grid is re-set only when the count changes.
         */
        private void decodeRow(int qPos, int row, int nKeys) {
            FlashAttnTiling.Split s = decodePlan.decodeSplit(nKeys);
            int count = s.count();
            boolean direct = s.direct();
            decode.setInt(AttnKernels.ARG_FD_N_SPLIT, count);
            decode.setInt(AttnKernels.ARG_FD_SPLIT_LEN, s.length());
            decode.setInt(AttnKernels.ARG_FD_KV_END, nKeys);
            decode.setInt(AttnKernels.ARG_FD_Q_POS, qPos);
            decode.setInt(AttnKernels.ARG_FD_DIRECT_OUT, direct ? 1 : 0);
            decode.setInt(AttnKernels.ARG_FD_Q_OFF, row * qDim);
            decode.setInt(AttnKernels.ARG_FD_OUT_OFF, direct ? row * outDim : 0);
            if (count != decodeSplits) {
                decode.setGlobal(decodePlan.decodeGlobal(count));
                decodeSplits = count;
            }
            decode.dispatch();
            if (!direct) {
                combine(nHead, count, row * outDim);
            }
        }

        private void combine(int rows, int nSplit, int outOff) {
            combine.setInt(AttnKernels.ARG_FC_N_ROWS, rows);
            combine.setInt(AttnKernels.ARG_FC_N_SPLIT, nSplit);
            combine.setInt(AttnKernels.ARG_FC_OUT_OFF, outOff);
            if (rows != combineRows) {
                combine.setGlobal(FlashAttnTiling.combineGlobal(rows));
                combineRows = rows;
            }
            combine.dispatch();
        }
    }

    private PreparedDispatch prepare2d(String kernel, long[] global, long[] localSize, Object[] args) {
        PreparedDispatch d = ops.prepare(kernel, global, localSize, args);
        prepared.add(d);
        return d;
    }

    /** {@code attn_sinks} as F32 ({@code n_head} logits), or null without sinks. */
    private GpuBuffer sinks() {
        if (!spec.sinks()) {
            return null;
        }
        return vector(TensorNames.weight(TensorNames.ATTN_SINKS), spec.nHead());
    }

    /**
     * The T5 relative-position bias table of the stack's bias layer ({@link T5RelativeBias#device}: F32
     * {@code [n_buckets][n_head]}, copied once at bind and owned by this mixer), or null without a T5 bias.
     */
    private GpuBuffer relBias() {
        if (!T5RelativeBias.uses(spec)) {
            return null;
        }
        t5Bias = T5RelativeBias.device(ctx, layer, spec);
        return t5Bias.buffer();
    }
}
