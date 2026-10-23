package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.arch.KdaGateMode;
import com.eonmux.neuralfree.gguf.arch.MixerKind;

import java.util.Objects;

/**
 * Recurrent or linear-attention mixer of one layer: Mamba1/2, PLaMo-2 Mamba, LFM2 short convolution, RWKV6/7
 * (and the Qwen-hybrid QRWKV variants), gated delta-net (GDN), Kimi delta attention (KDA) and MiniMax lightning
 * attention. Only the fields of {@link #kind()} are meaningful; the others stay 0 / false / null.
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
    private final boolean legacySsmIn;
    private final KdaGateMode kdaGateMode;
    private final DecayGate decayGate;
    private final boolean safeGate;
    private final float gateLowerBound;
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
        this.legacySsmIn = b.legacySsmIn;
        this.kdaGateMode = b.kdaGateMode;
        this.decayGate = b.decayGate;
        this.safeGate = b.safeGate;
        this.gateLowerBound = b.gateLowerBound;
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
            case MAMBA1: case MAMBA2: case PLAMO2_MAMBA: case SHORTCONV: case SHORTCONV_CENTERED:
            case RWKV6: case RWKV6_QWEN2: case RWKV7: case GDN: case KDA: case LIGHTNING:
                return true;
            default:
                return false;
        }
    }

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
    /** Legacy qwen3next per-k-head {@code [q|k|v|z]} {@code ssm_in} packing. */
    public boolean legacySsmIn() { return legacySsmIn; }
    /** KDA gate projection form; null for other kinds. */
    public KdaGateMode kdaGateMode() { return kdaGateMode; }
    /** Decay gate formula (GDN, or the KDA mode); null for non-delta kinds. */
    public DecayGate decayGate() { return decayGate; }
    public boolean safeGate() { return safeGate; }
    /** KDA gate lower bound {@code lb}; {@code -inf} when absent. */
    public float gateLowerBound() { return gateLowerBound; }
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
     * Channels of one conv-state time step ({@code conv[c][j]} has this many {@code c}). PLaMo-2 convolves only
     * the {@code d_inner} x branch ({@code ssm_conv1d} is {@code {d_conv, d_inner}}, src/models/plamo2.cpp:67).
     */
    public int convChannels() {
        switch (kind) {
            case MAMBA1: case MAMBA2:
                return dInner + 2 * nGroup * dState;
            case PLAMO2_MAMBA:
                return dInner;
            case GDN:
                return nVHeads * headVDim + 2 * nKHeads * headKDim;
            case KDA:
                return 3 * nHead * headDim;
            case SHORTCONV: case SHORTCONV_CENTERED:
                return nEmbd;
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
            case MAMBA1: case MAMBA2: case PLAMO2_MAMBA: case GDN:
                return dConv > 0 ? dConv - 1 : 0;
            default:
                return 0;
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
            case MAMBA1: case MAMBA2: case PLAMO2_MAMBA:
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
        private boolean legacySsmIn;
        private KdaGateMode kdaGateMode;
        private DecayGate decayGate;
        private boolean safeGate;
        private float gateLowerBound = Float.NEGATIVE_INFINITY;
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
        public Builder legacySsmIn(boolean v) { this.legacySsmIn = v; return this; }
        public Builder kdaGateMode(KdaGateMode v) { this.kdaGateMode = v; return this; }
        public Builder decayGate(DecayGate v) { this.decayGate = v; return this; }
        public Builder safeGate(boolean v) { this.safeGate = v; return this; }
        public Builder gateLowerBound(float v) { this.gateLowerBound = v; return this; }
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
        public Builder slopes(float[] v) { this.slopes = v != null ? v.clone() : null; return this; }

        /** @throws IllegalArgumentException when a field the kind needs is missing or inconsistent */
        public RecurrentSpec build() {
            switch (kind) {
                case MAMBA1: case MAMBA2: case PLAMO2_MAMBA:
                    require(dConv > 1 && dInner > 0 && dState > 0 && nGroup >= 0, "dConv > 1, dInner, dState");
                    if (kind != MixerKind.MAMBA1) {
                        require(nHead > 0 && headDim > 0 && nHead * headDim == dInner, "nHead * headDim == dInner");
                    }
                    if (kind == MixerKind.MAMBA2) {
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
