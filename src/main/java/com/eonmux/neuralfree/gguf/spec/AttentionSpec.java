package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.ops.attn.AttnExtMath;

import java.util.Objects;

/**
 * Softmax attention of one layer (MixerKind ATTENTION, ATTENTION_LINEAR_REPLACE, CROSS_ATTENTION_T5 and the
 * attention half of hybrid layers). Head counts and dims are per layer and already SWA-aware.
 *
 * <p>Score of query {@code q} against key {@code k} in the flash-attention kernels:
 * {@code s = softcap > 0 ? softcap * tanh(dot * scoreScale() / softcap) : dot * scoreScale()}, plus the
 * position bias of {@link #biasMode()}, then clamped by the window and causal mask, then softmax with
 * optional sinks. Immutable.
 */
public final class AttentionSpec {

    /**
     * Additive position bias; the code is the flash kernel's {@code bias_mode} argument. {@link #ALIBI_SQRT} is
     * vLLM's {@code use_alibi_sqrt} (Step1): the slope times the negative square root of the distance.
     */
    public enum BiasMode {
        NONE(0), ALIBI(1), T5_CAUSAL(2), T5_BIDIR(3), ALIBI_SQRT(4),
        /**
         * Inkling's per-query relative logits: a query row's head {@code h} adds
         * {@code rel_bias[(row * n_head + h) * rel_extent + (p_q - p_k)]} when {@code 0 <= p_q - p_k < rel_extent},
         * and nothing outside that range. The table is computed from the query's relative projection each step, so
         * {@code rel_bias} holds the whole step's rows and {@code n_buckets} is the layer's relative extent.
         */
        INKLING(5);

        private final int code;

        BiasMode(int code) {
            this.code = code;
        }

        public int code() {
            return code;
        }
    }

    /**
     * Position-dependent query scaling (llama4 / mistral3 / deepseek2 / beam attention temperature tuning,
     * llama-graph.cpp:213-238). The scale of a token at position {@code pos} is
     * {@code exp(scale * log1p((pos + offset) / floorScale))} when {@code pow}, else
     * {@code log(floor((pos + offset) / floorScale) + 1) * scale + 1}.
     */
    public static final class Temperature {
        /** No temperature tuning (scale 1 for every position). */
        public static final Temperature NONE = new Temperature(0f, 0, 0f, false);

        private final float scale;
        private final int floorScale;
        private final float offset;
        private final boolean pow;

        private Temperature(float scale, int floorScale, float offset, boolean pow) {
            this.scale = scale;
            this.floorScale = floorScale;
            this.offset = offset;
            this.pow = pow;
        }

        /**
         * @param scale      {@code f_attn_temp_scale}, non-zero
         * @param floorScale {@code n_attn_temp_floor_scale}, positive
         * @param offset     {@code f_attn_temp_offset}
         * @param pow        {@code attn_temp_pow}
         * @throws IllegalArgumentException when scale is 0 or NaN, or floorScale is not positive
         */
        public static Temperature of(float scale, int floorScale, float offset, boolean pow) {
            if (scale == 0f || Float.isNaN(scale)) {
                throw new IllegalArgumentException("attention temperature scale must be non-zero, got " + scale);
            }
            if (floorScale <= 0) {
                throw new IllegalArgumentException("attention temperature floor scale must be positive, got " + floorScale);
            }
            return new Temperature(scale, floorScale, offset, pow);
        }

        public boolean enabled() {
            return floorScale > 0;
        }

        public float scale() { return scale; }
        public int floorScale() { return floorScale; }
        public float offset() { return offset; }
        public boolean pow() { return pow; }

        /**
         * Query multiplier at an absolute position ({@code llm_graph_input_attn_temp::set_input},
         * llama-graph.cpp:213-238), computed by {@link AttnExtMath#temperatureFactor(Temperature, int)}; 1 when
         * disabled.
         */
        public float at(int pos) {
            return AttnExtMath.temperatureFactor(this, pos);
        }

        @Override
        public String toString() {
            return enabled() ? "Temperature{scale=" + scale + ", floor=" + floorScale + ", offset=" + offset
                    + ", pow=" + pow + '}' : "Temperature.NONE";
        }
    }

    private final int nHead;
    private final int nHeadKv;
    private final int headDimK;
    private final int headDimV;
    private final int nRot;
    private final boolean noRope;
    private final boolean causal;
    private final SwaType swaType;
    private final int nSwa;
    private final boolean swaLayer;
    private final QkNormKind qkNorm;
    private final boolean qkNormAfterRope;
    private final boolean qkNormBias;
    private final boolean inBias;
    private final boolean outBias;
    private final boolean vNorm;
    private final AttnGateKind gate;
    private final Temperature temperature;
    private final float clampKqv;
    private final boolean sinks;
    private final int sinkTokens;
    private final float softcap;
    private final float valueScale;
    private final float alibiMaxBias;
    private final BiasMode biasMode;
    private final int t5Buckets;
    private final int t5MaxDistance;
    private final int t5BiasLayer;
    private final int relBiasExtent;
    private final float kqScale;
    private final float outScale;
    private final boolean usesMemory;
    private final int kvOwnerLayer;
    private final boolean fusedQkv;
    private final boolean linearReplace;
    private final int qLatentDim;
    private final MoeSpec valueExperts;
    private final boolean valueEmbeddings;
    private final boolean valueEmbeddingGate;
    private final float xsaEps;
    private final int qkvAdapterRank;
    private final int qkvAdapterLayer;
    private final NormKind outNormKind;
    private final float outNormEps;
    private final int dualReadLayer;
    private final int dualWindow;
    private final DualChunkSpec dualChunk;

    private AttentionSpec(Builder b) {
        this.nHead = b.nHead;
        this.nHeadKv = b.nHeadKv;
        this.headDimK = b.headDimK;
        this.headDimV = b.headDimV;
        this.nRot = b.nRot;
        this.noRope = b.noRope;
        this.causal = b.causal;
        this.swaType = b.swaType;
        this.nSwa = b.nSwa;
        this.swaLayer = b.swaLayer;
        this.qkNorm = b.qkNorm;
        this.qkNormAfterRope = b.qkNormAfterRope;
        this.qkNormBias = b.qkNormBias;
        this.inBias = b.inBias;
        this.outBias = b.outBias;
        this.vNorm = b.vNorm;
        this.gate = b.gate;
        this.temperature = b.temperature;
        this.clampKqv = b.clampKqv;
        this.sinks = b.sinks;
        this.sinkTokens = b.sinkTokens;
        this.softcap = b.softcap;
        this.valueScale = b.valueScale;
        this.alibiMaxBias = b.alibiMaxBias;
        this.biasMode = b.biasMode;
        this.t5Buckets = b.t5Buckets;
        this.t5MaxDistance = b.t5MaxDistance;
        this.t5BiasLayer = b.t5BiasLayer;
        this.relBiasExtent = b.relBiasExtent;
        this.kqScale = b.kqScale;
        this.outScale = b.outScale;
        this.usesMemory = b.usesMemory;
        this.kvOwnerLayer = b.kvOwnerLayer;
        this.fusedQkv = b.fusedQkv;
        this.linearReplace = b.linearReplace;
        this.qLatentDim = b.qLatentDim;
        this.valueExperts = b.valueExperts;
        this.valueEmbeddings = b.valueEmbeddings;
        this.valueEmbeddingGate = b.valueEmbeddingGate;
        this.xsaEps = b.xsaEps;
        this.qkvAdapterRank = b.qkvAdapterRank;
        this.qkvAdapterLayer = b.qkvAdapterLayer;
        this.outNormKind = b.outNormKind;
        this.outNormEps = b.outNormEps;
        this.dualReadLayer = b.dualReadLayer;
        this.dualWindow = b.dualWindow;
        this.dualChunk = b.dualChunk;
    }

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        return new Builder(this);
    }

    public int nHead() { return nHead; }
    public int nHeadKv() { return nHeadKv; }
    public int headDimK() { return headDimK; }
    public int headDimV() { return headDimV; }
    /** RoPE dims per head (partial rotary); 0 when {@link #noRope()}. */
    public int nRot() { return nRot; }
    /** NoPE layer: no rotary embedding on Q/K. */
    public boolean noRope() { return noRope; }
    public boolean causal() { return causal; }
    /** Model SWA type; it applies to this layer only when {@link #isSwaLayer()}. */
    public SwaType swaType() { return swaType; }
    public int nSwa() { return nSwa; }
    public boolean isSwaLayer() { return swaLayer; }
    public QkNormKind qkNorm() { return qkNorm; }
    /** The Q/K norm runs on the rotated heads (after RoPE) rather than before RoPE. */
    public boolean qkNormAfterRope() { return qkNormAfterRope; }
    /** The Q/K norm adds {@code attn_q_norm.bias} / {@code attn_k_norm.bias} when the file carries them. */
    public boolean qkNormBias() { return qkNormBias; }
    /**
     * The Q/K/V projections add {@code attn_qkv.bias} or {@code attn_q/k/v.bias} when the file carries them (false
     * where the graph projects without them, {@code UnusedBiases.attnInputUnused}).
     */
    public boolean inBias() { return inBias; }
    /** The attention output adds {@code attn_output.bias} when the file carries it. */
    public boolean outBias() { return outBias; }
    /** Weightless RMS norm on V per head (gemma4). */
    public boolean vNorm() { return vNorm; }
    public AttnGateKind gate() { return gate; }
    public Temperature temperature() { return temperature; }
    /** Clamp of the Q/K/V projections to {@code [-clampKqv, clampKqv]}; 0 disables. */
    public float clampKqv() { return clampKqv; }
    /** Per-head attention sinks ({@code attn_sinks}) join the softmax denominator. */
    public boolean sinks() { return sinks; }

    /**
     * Learned static sink tokens of the openPangu sink attention ({@code param_sink_number}): every query also
     * attends the {@code param_sink_number} sink key/value pairs of {@code attn_sink_k}/{@code attn_sink_v}; 0 when
     * the layer has none. The sink keys are normalized by the layer's K norm and carry no RoPE, and the sinks always
     * precede the cached keys (vLLM {@code StaticSinkAttention}).
     */
    public int sinkTokens() { return sinkTokens; }
    /** Attention logit softcap; 0 disables. */
    public float softcap() { return softcap; }
    /** Multiplier on the attention output before {@code wo} ({@code f_attn_value_scale}); 0 disables. */
    public float valueScale() { return valueScale; }
    public float alibiMaxBias() { return alibiMaxBias; }
    public BiasMode biasMode() { return biasMode; }
    /** T5 relative-position bucket count ({@code n_rel_attn_bkts}); 0 unless a T5 bias mode. */
    public int t5Buckets() { return t5Buckets; }
    /** T5 relative-position max distance (llama.cpp uses 128). */
    public int t5MaxDistance() { return t5MaxDistance; }
    /** Layer of the stack whose {@code attn_rel_b} tensor holds the shared T5 bias (layer 0 in llama.cpp). */
    public int t5BiasLayer() { return t5BiasLayer; }
    /**
     * Distances of the {@link BiasMode#INKLING} relative-logit table: the layer's sliding window on a local layer,
     * the model's relative extent on a full-attention one. 0 for every other bias mode.
     */
    public int relBiasExtent() { return relBiasExtent; }
    /** {@code kq_scale} passed to softmax / flash attention (1/sqrt(headDimK), attention.scale or MLA mscale). */
    public float kqScale() { return kqScale; }
    /**
     * Grok {@code f_attn_out_scale}: logit multiplier applied together with the softcap
     * ({@code kq = cap * tanh(kq * outScale / cap)}, llama-graph.cpp:3024-3035); 0 when unused. Grok passes
     * {@code kq_scale = 1} (models/grok.cpp:135), so its resolved {@link #kqScale()} is 1 and
     * {@link #scoreScale()} equals {@code outScale}.
     */
    public float outScale() { return outScale; }
    /** False for encoder / non-causal passes that attend only to the current batch without a cache. */
    public boolean usesMemory() { return usesMemory; }
    /** Layer whose KV cache this layer reads; equals the layer itself when it owns its cache. */
    public int kvOwnerLayer() { return kvOwnerLayer; }
    /** Q/K/V come from one {@code attn_qkv} projection ({@link TensorCatalog#fusedQkv}). */
    public boolean fusedQkv() { return fusedQkv; }
    /** deci {@code n_head_kv == 0}: the mixer is {@code wo @ attn_norm(x)} without attention. */
    public boolean linearReplace() { return linearReplace; }
    /**
     * Width of the shared query projection when Q is low-rank (Step3 text): the layer projects
     * {@code attn_q_a}, RMS-normalizes it with {@code attn_q_a_norm} and widens it to {@link #qDim()} with
     * {@code attn_q_b} before RoPE. 0 when Q comes from a single projection.
     */
    public int qLatentDim() { return qLatentDim; }
    /**
     * MoVA routed value experts (k2-horizon, models/k2-horizon.cpp:137-205); null when V is a plain projection.
     * Routing follows the MoeSpec fields over {@code attn_v_gate} ({@code attn_v_gate.bias} biases selection only);
     * expert {@code k} computes {@code silu(attn_v_exps[k] · x)} of width {@link #vDim()} (no down projection) and
     * V is the weighted sum over the selected experts.
     */
    public MoeSpec valueExperts() { return valueExperts; }
    /**
     * V gains the token's {@code value_embd} row (limite, models/limite.cpp:149-155); the row is looked up by
     * token id, so embeddings input is unsupported.
     */
    public boolean valueEmbeddings() { return valueEmbeddings; }
    /** The value-embedding add is gated per KV head by {@code sigmoid(attn_ve_gate @ x)}. */
    public boolean valueEmbeddingGate() { return valueEmbeddingGate; }
    /**
     * Epsilon of the exclusive self attention (limite.cpp:177-191): each head output loses its projection on
     * the token's own value, scaled by {@code attn_xsa_alpha}; 0 disables.
     */
    public float xsaEps() { return xsaEps; }
    /**
     * Rank of the per-use adapters added to the Q, K and V projections of a shared attention block (Zamba2,
     * {@code use_shared_attention_adapter}); 0 without adapters. Adapter {@code A} projects the layer input to the
     * rank and {@code B} back to the projection width, and their product is added to each projection before RoPE.
     */
    public int qkvAdapterRank() { return qkvAdapterRank; }
    /**
     * Main-stack layer holding the adapter tensors {@code blk.<layer>.attn_{q,k,v}_adapter_{a,b}.weight}; -1
     * without adapters. The shared block's weights live in another stack, so the adapters are named by their use.
     */
    public int qkvAdapterLayer() { return qkvAdapterLayer; }
    /** A weightless norm over the whole attention output, applied after the output gate and before {@code wo}; null when absent. */
    public NormKind outNormKind() { return outNormKind; }
    /** Epsilon of {@link #outNormKind()}. */
    public float outNormEps() { return outNormEps; }

    /**
     * Layer whose cache this layer also attends read-only over a full causal window, or -1 when it attends its own
     * cache alone. IQuestLoopCoder's passes after the first mix this global attention with a sliding-window
     * attention over their own cache, gated per head ({@code forward_loop > 0} of {@code iquest_loopcoder.py}).
     */
    public int dualReadLayer() { return dualReadLayer; }
    /** Sliding window of the local attention over this layer's own cache when {@link #dualReadLayer()} is set. */
    public int dualWindow() { return dualWindow; }
    /** True when the layer mixes a global (read-only) and a local attention, gated per head. */
    public boolean dual() { return dualReadLayer >= 0; }
    /** Dual-chunk attention (Qwen2.5-1M), or null for a plain softmax layer. */
    public DualChunkSpec dualChunk() { return dualChunk; }
    /** True when the query carries the five dual-chunk rotations. */
    public boolean hasDualChunk() { return dualChunk != null; }

    /** Q projection width. */
    public int qDim() { return nHead * headDimK; }
    /**
     * Width of the query rows the attention reads: {@code 5 * qDim} for dual-chunk attention (the five rotations,
     * one plane after another, each a {@code [tokens][nHead][headDimK]} block), else {@link #qDim()}.
     */
    public int qRopeDim() { return dualChunk != null ? 5 * qDim() : qDim(); }
    /** K projection width. */
    public int kDim() { return nHeadKv * headDimK; }
    /** V projection width. */
    public int vDim() { return nHeadKv * headDimV; }
    /** Attention output width before {@code wo}. */
    public int outDim() { return nHead * headDimV; }
    /** Query heads per KV head. */
    public int gqaRatio() { return nHeadKv > 0 ? nHead / nHeadKv : 0; }
    /** Window type in effect for this layer: {@link SwaType#NONE} unless it is an SWA layer. */
    public SwaType effectiveSwaType() { return swaLayer ? swaType : SwaType.NONE; }
    /** True when this layer reads another layer's cache. */
    public boolean sharesKv(int il) { return kvOwnerLayer != il; }

    /** Logit multiplier fed to the kernels: {@code kqScale * outScale} when {@code outScale != 0}. */
    public float scoreScale() {
        return outScale != 0f ? kqScale * outScale : kqScale;
    }

    @Override
    public String toString() {
        return "AttentionSpec{nHead=" + nHead + ", nHeadKv=" + nHeadKv + ", headK=" + headDimK + ", headV=" + headDimV
                + ", nRot=" + nRot + ", noRope=" + noRope + ", causal=" + causal + ", swa=" + effectiveSwaType()
                + "/" + nSwa + ", qkNorm=" + qkNorm + (qkNormAfterRope ? " after rope" : "")
                + (qkNormBias ? "" : " no bias") + ", inBias=" + inBias + ", outBias=" + outBias + ", gate=" + gate
                + ", softcap=" + softcap + ", bias=" + biasMode
                + ", kqScale=" + kqScale + ", kvOwner=" + kvOwnerLayer + ", fusedQkv=" + fusedQkv
                + (valueExperts != null ? ", valueExperts=" + valueExperts : "") + '}';
    }

    public static final class Builder {
        private int nHead;
        private int nHeadKv;
        private int headDimK;
        private int headDimV;
        private int nRot;
        private boolean noRope;
        private boolean causal = true;
        private SwaType swaType = SwaType.NONE;
        private int nSwa;
        private boolean swaLayer;
        private QkNormKind qkNorm = QkNormKind.NONE;
        private boolean qkNormAfterRope;
        private boolean qkNormBias = true;
        private boolean inBias = true;
        private boolean outBias = true;
        private boolean vNorm;
        private AttnGateKind gate = AttnGateKind.NONE;
        private Temperature temperature = Temperature.NONE;
        private float clampKqv;
        private boolean sinks;
        private int sinkTokens;
        private float softcap;
        private float valueScale;
        private float alibiMaxBias;
        private BiasMode biasMode = BiasMode.NONE;
        private int t5Buckets;
        private int t5MaxDistance = 128;
        private int t5BiasLayer;
        private int relBiasExtent = 0;
        private float kqScale = Float.NaN;
        private float outScale;
        private boolean usesMemory = true;
        private int kvOwnerLayer = -1;
        private boolean fusedQkv;
        private boolean linearReplace;
        private int qLatentDim;
        private MoeSpec valueExperts;
        private boolean valueEmbeddings;
        private boolean valueEmbeddingGate;
        private float xsaEps;
        private int qkvAdapterRank;
        private int qkvAdapterLayer = -1;
        private NormKind outNormKind;
        private float outNormEps;
        private int dualReadLayer = -1;
        private int dualWindow;
        private DualChunkSpec dualChunk;

        private Builder() {
        }

        private Builder(AttentionSpec s) {
            nHead = s.nHead; nHeadKv = s.nHeadKv; headDimK = s.headDimK; headDimV = s.headDimV; nRot = s.nRot;
            noRope = s.noRope; causal = s.causal; swaType = s.swaType; nSwa = s.nSwa; swaLayer = s.swaLayer;
            qkNorm = s.qkNorm; qkNormAfterRope = s.qkNormAfterRope; qkNormBias = s.qkNormBias; inBias = s.inBias;
            outBias = s.outBias;
            vNorm = s.vNorm; gate = s.gate; temperature = s.temperature; clampKqv = s.clampKqv;
            sinks = s.sinks; sinkTokens = s.sinkTokens; softcap = s.softcap;
            valueScale = s.valueScale; alibiMaxBias = s.alibiMaxBias;
            biasMode = s.biasMode; t5Buckets = s.t5Buckets; t5MaxDistance = s.t5MaxDistance;
            t5BiasLayer = s.t5BiasLayer; kqScale = s.kqScale; outScale = s.outScale; usesMemory = s.usesMemory;
            relBiasExtent = s.relBiasExtent;
            kvOwnerLayer = s.kvOwnerLayer; fusedQkv = s.fusedQkv; linearReplace = s.linearReplace;
            qLatentDim = s.qLatentDim;
            valueExperts = s.valueExperts;
            valueEmbeddings = s.valueEmbeddings; valueEmbeddingGate = s.valueEmbeddingGate; xsaEps = s.xsaEps;
            qkvAdapterRank = s.qkvAdapterRank; qkvAdapterLayer = s.qkvAdapterLayer;
            outNormKind = s.outNormKind; outNormEps = s.outNormEps;
            dualReadLayer = s.dualReadLayer; dualWindow = s.dualWindow;
            dualChunk = s.dualChunk;
        }

        public Builder nHead(int v) { this.nHead = v; return this; }
        public Builder nHeadKv(int v) { this.nHeadKv = v; return this; }
        public Builder headDimK(int v) { this.headDimK = v; return this; }
        public Builder headDimV(int v) { this.headDimV = v; return this; }
        public Builder nRot(int v) { this.nRot = v; return this; }
        public Builder noRope(boolean v) { this.noRope = v; return this; }
        public Builder causal(boolean v) { this.causal = v; return this; }
        public Builder swaType(SwaType v) { this.swaType = Objects.requireNonNull(v, "swaType"); return this; }
        public Builder nSwa(int v) { this.nSwa = v; return this; }
        public Builder swaLayer(boolean v) { this.swaLayer = v; return this; }
        public Builder qkNorm(QkNormKind v) { this.qkNorm = Objects.requireNonNull(v, "qkNorm"); return this; }
        public Builder qkNormAfterRope(boolean v) { this.qkNormAfterRope = v; return this; }
        public Builder qkNormBias(boolean v) { this.qkNormBias = v; return this; }
        public Builder inBias(boolean v) { this.inBias = v; return this; }
        public Builder outBias(boolean v) { this.outBias = v; return this; }
        public Builder vNorm(boolean v) { this.vNorm = v; return this; }
        public Builder gate(AttnGateKind v) { this.gate = Objects.requireNonNull(v, "gate"); return this; }
        public Builder temperature(Temperature v) {
            this.temperature = Objects.requireNonNull(v, "temperature");
            return this;
        }
        public Builder clampKqv(float v) { this.clampKqv = v; return this; }
        public Builder sinks(boolean v) { this.sinks = v; return this; }
        /** Learned static sink tokens ({@code attn_sink_k}/{@code attn_sink_v}); 0 for none. */
        public Builder sinkTokens(int v) { this.sinkTokens = v; return this; }
        public Builder softcap(float v) { this.softcap = v; return this; }
        public Builder valueScale(float v) { this.valueScale = v; return this; }
        public Builder alibiMaxBias(float v) { this.alibiMaxBias = v; return this; }
        public Builder biasMode(BiasMode v) { this.biasMode = Objects.requireNonNull(v, "biasMode"); return this; }
        public Builder t5Buckets(int v) { this.t5Buckets = v; return this; }
        public Builder t5MaxDistance(int v) { this.t5MaxDistance = v; return this; }
        public Builder t5BiasLayer(int v) { this.t5BiasLayer = v; return this; }
        /** Distances of the {@link BiasMode#INKLING} relative-logit table; 0 for every other bias mode. */
        public Builder relBiasExtent(int v) { this.relBiasExtent = v; return this; }
        public Builder kqScale(float v) { this.kqScale = v; return this; }
        public Builder outScale(float v) { this.outScale = v; return this; }
        public Builder usesMemory(boolean v) { this.usesMemory = v; return this; }
        public Builder kvOwnerLayer(int v) { this.kvOwnerLayer = v; return this; }
        public Builder fusedQkv(boolean v) { this.fusedQkv = v; return this; }
        public Builder linearReplace(boolean v) { this.linearReplace = v; return this; }
        /** Low-rank query width ({@code attn_q_a} rows); 0 for a single Q projection. */
        public Builder qLatentDim(int v) { this.qLatentDim = v; return this; }
        public Builder valueExperts(MoeSpec v) { this.valueExperts = v; return this; }
        public Builder valueEmbeddings(boolean v) { this.valueEmbeddings = v; return this; }
        public Builder valueEmbeddingGate(boolean v) { this.valueEmbeddingGate = v; return this; }
        public Builder xsaEps(float v) { this.xsaEps = v; return this; }
        /** Per-use Q/K/V adapter rank of a shared attention block; 0 disables. */
        public Builder qkvAdapterRank(int v) { this.qkvAdapterRank = v; return this; }
        /** Main-stack layer holding the adapter tensors; -1 without adapters. */
        public Builder qkvAdapterLayer(int v) { this.qkvAdapterLayer = v; return this; }
        /**
         * The gated mixed attention of IQuestLoopCoder: the layer attends {@code globalLayer}'s cache read-only over
         * a full causal window and its own cache over a sliding window of {@code window} positions, then mixes the
         * two per head with {@code sigmoid(dot(q_head, attn_loop_gate_head) + attn_loop_gate_bias_head)}.
         *
         * @param globalLayer layer whose cache the global attention reads
         * @param window      sliding window of the local attention, positive
         */
        public Builder dualRead(int globalLayer, int window) {
            this.dualReadLayer = globalLayer;
            this.dualWindow = window;
            return this;
        }

        /**
         * Dual-chunk attention (Qwen2.5-1M): the query is rotated five ways and the keys are grouped into the token's
         * own chunk, the preceding chunk and the earlier chunks, merged by one softmax. The layer must be a causal,
         * cached, full-attention layer without sinks, softcap or a position bias.
         */
        public Builder dualChunk(DualChunkSpec v) {
            this.dualChunk = v;
            return this;
        }

        /** Weightless norm over the whole attention output, after the output gate and before {@code wo}. */
        public Builder outNorm(NormKind kind, float eps) {
            this.outNormKind = Objects.requireNonNull(kind, "outNormKind");
            this.outNormEps = eps;
            return this;
        }

        /**
         * @throws IllegalArgumentException on inconsistent dimensions, a missing kqScale, an SWA layer without a
         *                                  window, a bias mode without its parameters or an unset KV owner
         */
        public AttentionSpec build() {
            if (nHead <= 0) {
                throw new IllegalArgumentException("attention nHead must be positive, got " + nHead);
            }
            if (headDimK <= 0 || headDimV <= 0) {
                throw new IllegalArgumentException("attention head dims must be positive, got K=" + headDimK + " V=" + headDimV);
            }
            if (linearReplace) {
                if (nHeadKv != 0) {
                    throw new IllegalArgumentException("linear-replace attention has nHeadKv 0, got " + nHeadKv);
                }
            } else if (nHeadKv <= 0 || nHead % nHeadKv != 0) {
                throw new IllegalArgumentException("attention nHead " + nHead + " must be a positive multiple of nHeadKv "
                        + nHeadKv);
            }
            if (noRope ? nRot != 0 : (nRot <= 0 || nRot > headDimK || nRot % 2 != 0)) {
                throw new IllegalArgumentException("attention nRot " + nRot + " is invalid for headDimK " + headDimK
                        + (noRope ? " on a NoPE layer (must be 0)" : " (must be even, in (0, headDimK])"));
            }
            if (!(kqScale > 0f) || Float.isInfinite(kqScale)) {
                throw new IllegalArgumentException("attention kqScale must be a positive finite value, got " + kqScale);
            }
            if (swaLayer && (swaType == SwaType.NONE || nSwa <= 0)) {
                throw new IllegalArgumentException("SWA layer needs a window type and nSwa > 0, got " + swaType + "/" + nSwa);
            }
            if (!(clampKqv >= 0f) || !(softcap >= 0f) || !(alibiMaxBias >= 0f)) {
                throw new IllegalArgumentException("attention clampKqv, softcap and alibiMaxBias must be >= 0: "
                        + clampKqv + ", " + softcap + ", " + alibiMaxBias);
            }
            if (Float.isNaN(outScale) || Float.isNaN(valueScale)) {
                throw new IllegalArgumentException("attention outScale and valueScale must not be NaN: "
                        + outScale + ", " + valueScale);
            }
            if ((biasMode == BiasMode.ALIBI || biasMode == BiasMode.ALIBI_SQRT) && !(alibiMaxBias > 0f)) {
                throw new IllegalArgumentException("ALiBi bias mode needs alibiMaxBias > 0");
            }
            boolean t5 = biasMode == BiasMode.T5_CAUSAL || biasMode == BiasMode.T5_BIDIR;
            if (t5 && (t5Buckets <= 0 || t5MaxDistance <= 0 || t5BiasLayer < 0)) {
                throw new IllegalArgumentException("T5 bias mode needs buckets, max distance and a bias layer, got "
                        + t5Buckets + "/" + t5MaxDistance + "/" + t5BiasLayer);
            }
            if (biasMode == BiasMode.T5_CAUSAL && !causal) {
                throw new IllegalArgumentException("T5_CAUSAL bias mode on a non-causal layer");
            }
            if (biasMode == BiasMode.INKLING) {
                if (relBiasExtent <= 0) {
                    throw new IllegalArgumentException("INKLING bias mode needs a positive relative extent, got "
                            + relBiasExtent);
                }
                if (!noRope) {
                    throw new IllegalArgumentException("INKLING bias mode needs a NoPE layer");
                }
            } else if (relBiasExtent != 0) {
                throw new IllegalArgumentException("a relative extent needs the INKLING bias mode, got " + biasMode);
            }
            if (valueExperts != null && (valueExperts.gated() || valueExperts.nFfExp() != nHeadKv * headDimV)) {
                throw new IllegalArgumentException("value experts must be ungated with width vDim " + (nHeadKv * headDimV)
                        + ", got " + valueExperts);
            }
            if (valueEmbeddingGate && !valueEmbeddings) {
                throw new IllegalArgumentException("a value-embedding gate needs valueEmbeddings");
            }
            if (Float.isNaN(xsaEps) || xsaEps < 0f) {
                throw new IllegalArgumentException("attention xsaEps must be >= 0, got " + xsaEps);
            }
            if (outNormKind != null && (!outNormKind.isRowNorm() || !(outNormEps > 0f) || Float.isInfinite(outNormEps))) {
                throw new IllegalArgumentException("attention outNorm needs a row norm and a positive finite eps, got "
                        + outNormKind + "/" + outNormEps);
            }
            if (qkNormAfterRope && (qkNorm == QkNormKind.NONE || noRope)) {
                throw new IllegalArgumentException("a Q/K norm after RoPE needs a Q/K norm and a roped layer, got "
                        + qkNorm + (noRope ? " on a NoPE layer" : ""));
            }
            if (qLatentDim < 0) {
                throw new IllegalArgumentException("attention qLatentDim must be >= 0, got " + qLatentDim);
            }
            if (qLatentDim > 0 && fusedQkv) {
                throw new IllegalArgumentException("a low-rank query projection needs separate Q/K/V weights,"
                        + " not a fused attn_qkv");
            }
            if (qLatentDim > 0 && gate == AttnGateKind.Q_INTERLEAVED) {
                throw new IllegalArgumentException("a low-rank query projection cannot carry an interleaved gate");
            }
            if (qLatentDim > 0 && (clampKqv > 0f || inBias)) {
                throw new IllegalArgumentException("a low-rank query projection carries no input clamp or bias");
            }
            if (sinkTokens < 0) {
                throw new IllegalArgumentException("attention sinkTokens must be >= 0, got " + sinkTokens);
            }
            if (sinkTokens > 0) {
                if (linearReplace || nHeadKv <= 0) {
                    throw new IllegalArgumentException("static sink attention needs KV heads, got nHeadKv " + nHeadKv);
                }
                if (!causal) {
                    throw new IllegalArgumentException("static sink attention needs causal attention");
                }
                if (sinks) {
                    throw new IllegalArgumentException("a layer cannot carry both scalar and static attention sinks");
                }
                if (biasMode != BiasMode.NONE) {
                    throw new IllegalArgumentException("static sink attention takes no position bias, got " + biasMode);
                }
            }
            if (kvOwnerLayer < 0) {
                throw new IllegalArgumentException("attention kvOwnerLayer must be set (the layer itself when it owns its cache)");
            }
            if (qkvAdapterRank < 0) {
                throw new IllegalArgumentException("attention qkvAdapterRank must be >= 0, got " + qkvAdapterRank);
            }
            if (qkvAdapterRank > 0) {
                if (qkvAdapterLayer < 0) {
                    throw new IllegalArgumentException("Q/K/V adapters need the layer holding their tensors");
                }
                if (linearReplace || fusedQkv || qLatentDim > 0) {
                    throw new IllegalArgumentException("Q/K/V adapters need separate, single-step Q, K and V"
                            + " projections");
                }
            }
            if (dualReadLayer >= 0) {
                if (dualWindow <= 0) {
                    throw new IllegalArgumentException("dual attention needs a positive local window, got " + dualWindow);
                }
                if (!usesMemory || !causal) {
                    throw new IllegalArgumentException("dual attention needs a causal layer with a KV cache");
                }
                if (linearReplace) {
                    throw new IllegalArgumentException("dual attention cannot replace the attention with a linear map");
                }
            } else if (dualWindow != 0) {
                throw new IllegalArgumentException("attention dualWindow must be 0 without dualReadLayer, got " + dualWindow);
            }
            validateDualChunk();
            return new AttentionSpec(this);
        }

        /**
         * Dual-chunk attention runs over a causal KV cache and merges one softmax over the three stages. It has no
         * sliding window, sinks, softcap, position bias or second cache, and its Q rotation needs a roped layer with
         * the default NORM/NEOX pair layout (vLLM refuses the same combinations in its dual-chunk backend).
         *
         * @throws IllegalArgumentException naming the conflicting field
         */
        private void validateDualChunk() {
            if (dualChunk == null) {
                return;
            }
            if (!causal) {
                throw new IllegalArgumentException("dual-chunk attention needs a causal layer");
            }
            if (!usesMemory) {
                throw new IllegalArgumentException("dual-chunk attention needs a KV cache");
            }
            if (swaLayer) {
                throw new IllegalArgumentException("dual-chunk attention has no sliding window, got " + swaType);
            }
            if (dualReadLayer >= 0) {
                throw new IllegalArgumentException("dual-chunk attention cannot also read a second cache");
            }
            if (sinks || sinkTokens > 0) {
                throw new IllegalArgumentException("dual-chunk attention does not support attention sinks");
            }
            if (softcap != 0f) {
                throw new IllegalArgumentException("dual-chunk attention does not support a logit softcap, got "
                        + softcap);
            }
            if (temperature.enabled()) {
                throw new IllegalArgumentException("dual-chunk attention does not support an attention temperature");
            }
            if (outScale != 0f) {
                throw new IllegalArgumentException("dual-chunk attention does not support an output logit scale, got "
                        + outScale);
            }
            if (biasMode != BiasMode.NONE) {
                throw new IllegalArgumentException("dual-chunk attention does not support a position bias, got "
                        + biasMode);
            }
            if (noRope) {
                throw new IllegalArgumentException("dual-chunk attention needs a roped layer");
            }
            if (linearReplace) {
                throw new IllegalArgumentException("dual-chunk attention cannot replace the attention with a linear map");
            }
            if (qkNormAfterRope) {
                throw new IllegalArgumentException("dual-chunk attention applies its Q/K norm before RoPE only");
            }
        }
    }
}
