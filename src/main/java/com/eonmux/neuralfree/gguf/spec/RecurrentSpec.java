package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.arch.KdaGateMode;
import com.eonmux.neuralfree.gguf.arch.MixerKind;

import java.util.Objects;

/**
 * Recurrent or linear-attention mixer of one layer: Mamba1/2, PLaMo-2 Mamba, LFM2 short convolution, RWKV6/7
 * (and the Qwen-hybrid QRWKV variants), gated delta-net (GDN), Kimi delta attention (KDA), MiniMax lightning
 * attention and Inkling's packed K/V/attention-output/MLP-output convolutions. Only the fields of {@link #kind()}
 * are meaningful; the others stay 0 / false / null.
 *
 * <p>Lightning attention ({@code LIGHTNING}) carries the decay {@link #slopes()} of every head and, for Bailing MoE
 * V2.5, the mixer steps vLLM's {@code BailingMoELinearAttention} runs before the recurrence: the fused-QKV SiLU
 * ({@link #linearSilu()}), the per-head Q/K RMS norms ({@link #qkNorm()}), the head-local rotary embedding
 * ({@link #rope()}) and the {@code head_dim^-0.5} Q scale ({@link #qScale()}), followed by the output norm over
 * {@link #nNormGroups()} groups.
 *
 * <p>{@link #convStateFloats()} and {@link #stateFloats()} equal llama.cpp {@code n_embd_r()} and
 * {@code n_embd_s()} (src/llama-hparams.cpp:208-255) for the layer kind, except that the PLaMo-2 conv state holds
 * only the channels its convolution reads. Immutable.
 */
public final class RecurrentSpec {

    /** KDA / GDN decay-gate parameterisation; the code is {@code LLM_GDN_GatePrep}'s {@code kda_mode}. */
    public enum DecayGate {
        /** GDN: {@code g = softplus(alpha + dt_b) * ssm_a}. */
        GDN(-1),
        /** KDA: {@code g = softplus(f + dt_b) * ssm_a} (kimi-linear; kimi-k3 / glm5-next with lb = -inf). */
        SOFTPLUS(0),
        /** KDA: {@code g = lb * sigmoid(-(ssm_a * (f + dt_b)))} (kimi-k3, glm5-next, src/models/kimi-k3.cpp:418-436). */
        LB_NEG(1),
        /** KDA: {@code g = lb * sigmoid(ssm_a * (f + dt_b))} (bailingmoe3, src/models/bailingmoe3.cpp:270-276). */
        LB_POS(2);

        private final int code;

        DecayGate(int code) {
            this.code = code;
        }

        public int code() {
            return code;
        }
    }

    private final MixerKind kind;
    private final int nEmbd;
    private final int dConv;
    private final int dInner;
    private final int dState;
    private final int nGroup;
    private final int nHead;
    private final int headDim;
    private final int dtRank;
    private final boolean dtBcRms;
    private final boolean convBias;
    private final int nKHeads;
    private final int nVHeads;
    private final int headKDim;
    private final int headVDim;
    private final GdnHeadMap headMap;
    private final boolean fusedBetaAlpha;
    private final boolean concatBetaAlpha;
    private final boolean mergedQkvGate;
    private final boolean legacySsmIn;
    private final KdaGateMode kdaGateMode;
    private final DecayGate decayGate;
    private final boolean safeGate;
    private final float gateLowerBound;
    private final float betaScale;
    private final int wkvHeadSize;
    private final int tokenShiftCount;
    private final boolean qrwkv;
    private final int rescaleEveryN;
    private final int timeMixExtraDim;
    private final int timeDecayExtraDim;
    private final int nLoraDecay;
    private final int nLoraIclr;
    private final int nLoraValueResMix;
    private final int nLoraGate;
    private final int lCache;
    private final int nNormGroups;
    private final boolean linearSilu;
    private final boolean qkNorm;
    private final float qScale;
    private final RopeSpec rope;
    private final float[] slopes;

    private RecurrentSpec(Builder b) {
        this.kind = b.kind;
        this.nEmbd = b.nEmbd;
        this.dConv = b.dConv;
        this.dInner = b.dInner;
        this.dState = b.dState;
        this.nGroup = b.nGroup;
        this.nHead = b.nHead;
        this.headDim = b.headDim;
        this.dtRank = b.dtRank;
        this.dtBcRms = b.dtBcRms;
        this.convBias = b.convBias;
        this.nKHeads = b.nKHeads;
        this.nVHeads = b.nVHeads;
        this.headKDim = b.headKDim;
        this.headVDim = b.headVDim;
        this.headMap = b.headMap;
        this.fusedBetaAlpha = b.fusedBetaAlpha;
        this.concatBetaAlpha = b.concatBetaAlpha;
        this.mergedQkvGate = b.mergedQkvGate;
        this.legacySsmIn = b.legacySsmIn;
        this.kdaGateMode = b.kdaGateMode;
        this.decayGate = b.decayGate;
        this.safeGate = b.safeGate;
        this.gateLowerBound = b.gateLowerBound;
        this.betaScale = b.betaScale;
        this.wkvHeadSize = b.wkvHeadSize;
        this.tokenShiftCount = b.tokenShiftCount;
        this.qrwkv = b.qrwkv;
        this.rescaleEveryN = b.rescaleEveryN;
        this.timeMixExtraDim = b.timeMixExtraDim;
        this.timeDecayExtraDim = b.timeDecayExtraDim;
        this.nLoraDecay = b.nLoraDecay;
        this.nLoraIclr = b.nLoraIclr;
        this.nLoraValueResMix = b.nLoraValueResMix;
        this.nLoraGate = b.nLoraGate;
        this.lCache = b.lCache;
        this.nNormGroups = b.nNormGroups;
        this.linearSilu = b.linearSilu;
        this.qkNorm = b.qkNorm;
        this.qScale = b.qScale;
        this.rope = b.rope;
        this.slopes = b.slopes != null ? b.slopes.clone() : null;
    }

    /**
     * @param kind a recurrent mixer kind
     * @throws IllegalArgumentException when {@code kind} is not recurrent
     */
    public static Builder builder(MixerKind kind) {
        Objects.requireNonNull(kind, "kind");
        if (!isRecurrentKind(kind)) {
            throw new IllegalArgumentException("mixer kind " + kind + " is not a recurrent kind");
        }
        return new Builder(kind);
    }

    /** True for the mixer kinds described by this class. */
    public static boolean isRecurrentKind(MixerKind k) {
        switch (k) {
            case MAMBA1: case MAMBA2: case ZAMBA2: case PLAMO2_MAMBA: case SHORTCONV: case SHORTCONV_CENTERED:
            case RWKV6: case RWKV6_QWEN2: case RWKV7: case GDN: case KDA: case LIGHTNING: case INKLING:
                return true;
            default:
                return false;
        }
    }

    /**
     * The four convolution streams of an Inkling layer, in the order their slices pack into the layer's conv state
     * (vLLM {@code sconv_swa_attn._K, _V, _ATTN, _MLP}).
     */
    public static final int INKLING_K = 0;
    public static final int INKLING_V = 1;
    public static final int INKLING_ATTN = 2;
    public static final int INKLING_MLP = 3;

    public MixerKind kind() { return kind; }
    /** Model width; the channel count of RWKV token shift and short convolution. */
    public int nEmbd() { return nEmbd; }
    /** Convolution kernel width ({@code ssm_d_conv}); the conv state keeps {@code dConv - 1} steps. */
    public int dConv() { return dConv; }
    public int dInner() { return dInner; }
    public int dState() { return dState; }
    public int nGroup() { return nGroup; }
    /** SSM heads (Mamba2), KDA heads or lightning heads. */
    public int nHead() { return nHead; }
    /** Per-head dim (Mamba2 {@code d_inner / n_head}, KDA {@code n_embd_head_kda}, lightning {@code n_embd_head_la}). */
    public int headDim() { return headDim; }
    public int dtRank() { return dtRank; }
    public boolean dtBcRms() { return dtBcRms; }
    public boolean convBias() { return convBias; }
    /** GDN key heads. */
    public int nKHeads() { return nKHeads; }
    /** GDN value heads. */
    public int nVHeads() { return nVHeads; }
    public int headKDim() { return headKDim; }
    public int headVDim() { return headVDim; }
    /** GDN value-head to key-head mapping; null for other kinds. */
    public GdnHeadMap headMap() { return headMap; }
    /** qwen3next fused {@code ssm_ba} beta/alpha projection. */
    public boolean fusedBetaAlpha() { return fusedBetaAlpha; }
    /** qwen4exp merged {@code ssm_beta_alpha} beta/alpha projection ({@code [beta | alpha]} rows). */
    public boolean concatBetaAlpha() { return concatBetaAlpha; }
    /** qwen4exp merged linear-attention input {@code attn_qkv_gate} ({@code [attn_qkv | attn_gate]} rows). */
    public boolean mergedQkvGate() { return mergedQkvGate; }
    /** Legacy qwen3next per-k-head {@code [q|k|v|z]} {@code ssm_in} packing. */
    public boolean legacySsmIn() { return legacySsmIn; }
    /** KDA gate projection form; null for other kinds. */
    public KdaGateMode kdaGateMode() { return kdaGateMode; }
    /** Decay gate formula (GDN, or the KDA mode); null for non-delta kinds. */
    public DecayGate decayGate() { return decayGate; }
    public boolean safeGate() { return safeGate; }
    /** KDA gate lower bound {@code lb}; {@code -inf} when absent. */
    public float gateLowerBound() { return gateLowerBound; }
    /** Multiplier of the delta-rule update rate {@code beta} after its sigmoid; 1 unless the family doubles it. */
    public float betaScale() { return betaScale; }
    public int wkvHeadSize() { return wkvHeadSize; }
    /** RWKV token-shift slots: 2 (attention + channel mix) or 1 (QRWKV, no channel mix). */
    public int tokenShiftCount() { return tokenShiftCount; }
    /** Qwen-hybrid RWKV (rwkv6qwen2, arwkv7): no channel mix, GQA-style projections. */
    public boolean qrwkv() { return qrwkv; }
    /** RWKV6 halves the residual stream every {@code rescaleEveryN} layers; 0 disables. */
    public int rescaleEveryN() { return rescaleEveryN; }
    public int timeMixExtraDim() { return timeMixExtraDim; }
    public int timeDecayExtraDim() { return timeDecayExtraDim; }
    public int nLoraDecay() { return nLoraDecay; }
    public int nLoraIclr() { return nLoraIclr; }
    public int nLoraValueResMix() { return nLoraValueResMix; }
    public int nLoraGate() { return nLoraGate; }
    /** LFM2 short-convolution cache length ({@code n_shortconv_l_cache}). */
    public int lCache() { return lCache; }
    /** Groups of the lightning-attention output RMS norm; 1 normalizes the whole {@code nHead*headDim} row. */
    public int nNormGroups() { return nNormGroups; }
    /** True when the fused lightning-attention Q/K/V projection is passed through SiLU. */
    public boolean linearSilu() { return linearSilu; }
    /** True when the lightning-attention Q and K are RMS-normalized per head before RoPE. */
    public boolean qkNorm() { return qkNorm; }
    /** Scale applied to the lightning-attention Q after RoPE; 0 leaves it unchanged. */
    public float qScale() { return qScale; }
    /** Rotary embedding of the lightning-attention heads; null when they do not rotate. */
    public RopeSpec rope() { return rope; }

    /** Lightning per-head decay slopes for this layer, already layer-scaled (copy); null for other kinds. */
    public float[] slopes() {
        return slopes != null ? slopes.clone() : null;
    }

    /** Lightning slope of one head. */
    public float slope(int h) {
        if (slopes == null) {
            throw new IllegalStateException(kind + " has no lightning slopes");
        }
        return slopes[h];
    }

    /**
     * This spec with {@link #kind()} replaced by {@code kind} and every other field unchanged. The Zamba2 composite
     * mixer binds the Mamba-2 mixer with {@link MixerKind#MAMBA2} over its layer's recurrent spec.
     *
     * @throws IllegalArgumentException when the fields do not satisfy {@code kind}
     */
    public RecurrentSpec asKind(MixerKind kind) {
        Objects.requireNonNull(kind, "kind");
        Builder b = builder(kind);
        b.nEmbd(nEmbd);
        b.dConv(dConv);
        b.dInner(dInner);
        b.dState(dState);
        b.nGroup(nGroup);
        b.nHead(nHead);
        b.headDim(headDim);
        b.dtRank(dtRank);
        b.dtBcRms(dtBcRms);
        b.convBias(convBias);
        b.gdnHeads(nKHeads, nVHeads, headKDim, headVDim);
        b.headMap(headMap);
        b.fusedBetaAlpha(fusedBetaAlpha);
        b.concatBetaAlpha(concatBetaAlpha);
        b.mergedQkvGate(mergedQkvGate);
        b.legacySsmIn(legacySsmIn);
        b.kdaGateMode(kdaGateMode);
        b.decayGate(decayGate);
        b.safeGate(safeGate);
        b.gateLowerBound(gateLowerBound);
        b.wkvHeadSize(wkvHeadSize);
        b.tokenShiftCount(tokenShiftCount);
        b.qrwkv(qrwkv);
        b.rescaleEveryN(rescaleEveryN);
        b.timeMixExtraDim(timeMixExtraDim);
        b.timeDecayExtraDim(timeDecayExtraDim);
        b.rwkv7Lora(nLoraDecay, nLoraIclr, nLoraValueResMix, nLoraGate);
        b.lCache(lCache);
        b.slopes(slopes);
        return b.build();
    }

    /**
     * Channels of one conv-state time step ({@code conv[c][j]} has this many {@code c}). PLaMo-2 convolves only
     * the {@code d_inner} x branch ({@code ssm_conv1d} is {@code {d_conv, d_inner}}, src/models/plamo2.cpp:67).
     */
    public int convChannels() {
        switch (kind) {
            case MAMBA1: case MAMBA2: case ZAMBA2:
                return dInner + 2 * nGroup * dState;
            case PLAMO2_MAMBA:
                return dInner;
            case GDN:
                return nVHeads * headVDim + 2 * nKHeads * headKDim;
            case KDA:
                return 3 * nHead * headDim;
            case SHORTCONV: case SHORTCONV_CENTERED:
                return nEmbd;
            case INKLING:
                return 2 * nHead * headDim + 2 * nEmbd;
            default:
                return 0;
        }
    }

    /** Conv-state time steps kept: {@code dConv - 1} (KDA defaults to 3, LFM2 uses {@code lCache - 1}). */
    public int convSteps() {
        switch (kind) {
            case KDA:
                return dConv > 0 ? dConv - 1 : 3;
            case SHORTCONV: case SHORTCONV_CENTERED:
                return lCache - 1;
            case MAMBA1: case MAMBA2: case ZAMBA2: case PLAMO2_MAMBA: case GDN: case INKLING:
                return dConv > 0 ? dConv - 1 : 0;
            default:
                return 0;
        }
    }

    /** Channel offset of an Inkling convolution stream inside the layer's packed conv state. */
    public int inklingStreamOffset(int stream) {
        checkInklingStream(stream);
        switch (stream) {
            case INKLING_K:
                return 0;
            case INKLING_V:
                return nHead * headDim;
            case INKLING_ATTN:
                return 2 * nHead * headDim;
            default:
                return 2 * nHead * headDim + nEmbd;
        }
    }

    private void checkInklingStream(int stream) {
        if (kind != MixerKind.INKLING) {
            throw new IllegalStateException(kind + " has no Inkling convolution streams");
        }
        if (stream < INKLING_K || stream > INKLING_MLP) {
            throw new IllegalArgumentException("Inkling stream " + stream + " is not one of "
                    + INKLING_K + ".." + INKLING_MLP);
        }
    }

    /** Floats of the "r" state: conv history, or the RWKV token shift ({@code n_embd_r}). */
    public long convStateFloats() {
        switch (kind) {
            case RWKV6: case RWKV6_QWEN2: case RWKV7:
                return (long) tokenShiftCount * nEmbd;
            case LIGHTNING:
                return 0;
            default:
                return (long) convSteps() * convChannels();
        }
    }

    /** Floats of the "s" state: SSM scan, WKV, delta-rule or lightning matrices ({@code n_embd_s}). */
    public long stateFloats() {
        switch (kind) {
            case RWKV6: case RWKV6_QWEN2: case RWKV7:
                return (long) nEmbd * wkvHeadSize;
            case KDA: case LIGHTNING:
                return (long) headDim * headDim * nHead;
            case GDN:
                return (long) headKDim * headVDim * nVHeads;
            case MAMBA1: case MAMBA2: case ZAMBA2: case PLAMO2_MAMBA:
                return (long) dState * dInner;
            default:
                return 0;
        }
    }

    @Override
    public String toString() {
        return "RecurrentSpec{" + kind + ", r=" + convStateFloats() + ", s=" + stateFloats() + '}';
    }

    public static final class Builder {
        private final MixerKind kind;
        private int nEmbd;
        private int dConv;
        private int dInner;
        private int dState;
        private int nGroup;
        private int nHead;
        private int headDim;
        private int dtRank;
        private boolean dtBcRms;
        private boolean convBias;
        private int nKHeads;
        private int nVHeads;
        private int headKDim;
        private int headVDim;
        private GdnHeadMap headMap;
        private boolean fusedBetaAlpha;
        private boolean concatBetaAlpha;
        private boolean mergedQkvGate;
        private boolean legacySsmIn;
        private KdaGateMode kdaGateMode;
        private DecayGate decayGate;
        private boolean safeGate;
        private float gateLowerBound = Float.NEGATIVE_INFINITY;
        private float betaScale = 1f;
        private int wkvHeadSize;
        private int tokenShiftCount;
        private boolean qrwkv;
        private int rescaleEveryN;
        private int timeMixExtraDim;
        private int timeDecayExtraDim;
        private int nLoraDecay;
        private int nLoraIclr;
        private int nLoraValueResMix;
        private int nLoraGate;
        private int lCache;
        private int nNormGroups = 1;
        private boolean linearSilu = true;
        private boolean qkNorm;
        private float qScale;
        private RopeSpec rope;
        private float[] slopes;

        private Builder(MixerKind kind) {
            this.kind = kind;
            if (kind == MixerKind.GDN) {
                this.decayGate = DecayGate.GDN;
            }
        }

        public Builder nEmbd(int v) { this.nEmbd = v; return this; }
        public Builder dConv(int v) { this.dConv = v; return this; }
        public Builder dInner(int v) { this.dInner = v; return this; }
        public Builder dState(int v) { this.dState = v; return this; }
        public Builder nGroup(int v) { this.nGroup = v; return this; }
        public Builder nHead(int v) { this.nHead = v; return this; }
        public Builder headDim(int v) { this.headDim = v; return this; }
        public Builder dtRank(int v) { this.dtRank = v; return this; }
        public Builder dtBcRms(boolean v) { this.dtBcRms = v; return this; }
        public Builder convBias(boolean v) { this.convBias = v; return this; }
        public Builder gdnHeads(int nK, int nV, int headK, int headV) {
            this.nKHeads = nK;
            this.nVHeads = nV;
            this.headKDim = headK;
            this.headVDim = headV;
            return this;
        }
        public Builder headMap(GdnHeadMap v) { this.headMap = v; return this; }
        public Builder fusedBetaAlpha(boolean v) { this.fusedBetaAlpha = v; return this; }
        public Builder concatBetaAlpha(boolean v) { this.concatBetaAlpha = v; return this; }
        public Builder mergedQkvGate(boolean v) { this.mergedQkvGate = v; return this; }
        public Builder legacySsmIn(boolean v) { this.legacySsmIn = v; return this; }
        public Builder kdaGateMode(KdaGateMode v) { this.kdaGateMode = v; return this; }
        public Builder decayGate(DecayGate v) { this.decayGate = v; return this; }
        public Builder safeGate(boolean v) { this.safeGate = v; return this; }
        public Builder gateLowerBound(float v) { this.gateLowerBound = v; return this; }
        public Builder betaScale(float v) { this.betaScale = v; return this; }
        public Builder wkvHeadSize(int v) { this.wkvHeadSize = v; return this; }
        public Builder tokenShiftCount(int v) { this.tokenShiftCount = v; return this; }
        public Builder qrwkv(boolean v) { this.qrwkv = v; return this; }
        public Builder rescaleEveryN(int v) { this.rescaleEveryN = v; return this; }
        public Builder timeMixExtraDim(int v) { this.timeMixExtraDim = v; return this; }
        public Builder timeDecayExtraDim(int v) { this.timeDecayExtraDim = v; return this; }
        public Builder rwkv7Lora(int decay, int iclr, int valueResMix, int gate) {
            this.nLoraDecay = decay;
            this.nLoraIclr = iclr;
            this.nLoraValueResMix = valueResMix;
            this.nLoraGate = gate;
            return this;
        }
        public Builder lCache(int v) { this.lCache = v; return this; }
        /** Groups of the lightning-attention output norm; 1 (the default) normalizes the whole row. */
        public Builder nNormGroups(int v) { this.nNormGroups = v; return this; }
        /** SiLU on the fused lightning-attention Q/K/V projection (the mixers' default). */
        public Builder linearSilu(boolean v) { this.linearSilu = v; return this; }
        /** Per-head RMS norm of the lightning-attention Q and K before RoPE. */
        public Builder qkNorm(boolean v) { this.qkNorm = v; return this; }
        /** Scale of the lightning-attention Q after RoPE; 0 disables. */
        public Builder qScale(float v) { this.qScale = v; return this; }
        /** Rotary embedding of the lightning-attention heads, or null for none. */
        public Builder rope(RopeSpec v) { this.rope = v; return this; }
        public Builder slopes(float[] v) { this.slopes = v != null ? v.clone() : null; return this; }

        /** @throws IllegalArgumentException when a field the kind needs is missing or inconsistent */
        public RecurrentSpec build() {
            switch (kind) {
                case MAMBA1: case MAMBA2: case ZAMBA2: case PLAMO2_MAMBA:
                    require(dConv > 1 && dInner > 0 && dState > 0 && nGroup >= 0, "dConv > 1, dInner, dState");
                    if (kind != MixerKind.MAMBA1) {
                        require(nHead > 0 && headDim > 0 && nHead * headDim == dInner, "nHead * headDim == dInner");
                    }
                    if (kind == MixerKind.MAMBA2 || kind == MixerKind.ZAMBA2) {
                        require(nGroup > 0 && nHead % nGroup == 0, "nGroup > 0 dividing nHead");
                    }
                    break;
                case SHORTCONV: case SHORTCONV_CENTERED:
                    require(nEmbd > 0 && lCache > 1, "nEmbd and lCache > 1");
                    break;
                case RWKV6: case RWKV6_QWEN2: case RWKV7:
                    require(nEmbd > 0 && wkvHeadSize > 0 && nEmbd % wkvHeadSize == 0, "nEmbd divisible by wkvHeadSize");
                    require(tokenShiftCount == 1 || tokenShiftCount == 2, "tokenShiftCount 1 or 2");
                    require(!qrwkv || tokenShiftCount == 1, "QRWKV uses one token-shift slot");
                    break;
                case GDN:
                    require(nKHeads > 0 && nVHeads > 0 && nVHeads % nKHeads == 0, "nVHeads a multiple of nKHeads");
                    require(headKDim > 0 && headVDim > 0 && dConv > 1, "head dims and dConv > 1");
                    require(headMap != null, "GDN head map");
                    require(decayGate == DecayGate.GDN, "GDN decay gate");
                    require(!fusedBetaAlpha || !concatBetaAlpha, "one beta/alpha projection form");
                    require(betaScale > 0f, "a positive beta scale");
                    break;
                case KDA:
                    require(nHead > 0 && headDim > 0 && dConv >= 0, "nHead and headDim");
                    require(kdaGateMode != null, "KDA gate mode");
                    require(decayGate != null && decayGate != DecayGate.GDN, "a KDA decay gate");
                    require(decayGate == DecayGate.SOFTPLUS || gateLowerBound < 0f, "lb < 0 for bounded KDA gates");
                    break;
                case LIGHTNING:
                    require(nHead > 0 && headDim > 0, "nHead and headDim");
                    require(slopes != null && slopes.length == nHead, "one slope per head");
                    require(nNormGroups > 0 && (long) nHead * headDim % nNormGroups == 0,
                            "nHead * headDim divisible by the output norm groups");
                    require(qScale >= 0f && !Float.isNaN(qScale), "a non-negative Q scale");
                    require(rope == null || (rope.nDims() > 0 && rope.nDims() % 2 == 0
                            && rope.nOffs() + rope.nDims() <= headDim), "rotary dims inside the head");
                    break;
                case INKLING:
                    require(nEmbd > 0 && nHead > 0 && headDim > 0 && dConv > 1,
                            "nEmbd, the KV heads and head dim, and dConv > 1 (the convolution width)");
                    break;
                default:
                    throw new IllegalStateException("unreachable recurrent kind " + kind);
            }
            return new RecurrentSpec(this);
        }

        private void require(boolean ok, String what) {
            if (!ok) {
                throw new IllegalArgumentException(kind + " recurrent spec needs " + what + ": " + describe());
            }
        }

        private String describe() {
            return "dConv=" + dConv + " dInner=" + dInner + " dState=" + dState + " nGroup=" + nGroup + " nHead=" + nHead
                    + " headDim=" + headDim + " nEmbd=" + nEmbd + " gdn=" + nKHeads + "x" + headKDim + "/" + nVHeads + "x"
                    + headVDim + " wkv=" + wkvHeadSize + " shift=" + tokenShiftCount + " lCache=" + lCache
                    + " decay=" + decayGate + " lb=" + gateLowerBound;
        }
    }
}
