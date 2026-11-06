package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;

import java.util.Objects;

/**
 * Routed mixture-of-experts FFN of one layer, following llama.cpp {@code build_moe_ffn}
 * (src/llama-graph.cpp:2190-2440):
 * <ol>
 *   <li>logits = {@code ffn_gate_inp · x} (+ {@code ffn_gate_inp.bias}); probs by {@link #gating()};</li>
 *   <li>selection scores per {@link #selectionMode()} (+ {@code exp_probs_b} when {@link #probBias()}), with
 *       group-limited selection when {@code nGroups > 1}; top {@link #nUsed()}; or hash ids from
 *       {@code ffn_gate_tid2eid} when {@link #hashRouting()}; an embedding row of a {@link #mediaRouting()} layer
 *       routes through the router instead, its selection bias {@code exp_probs_b_vl} when
 *       {@link #mediaProbBias()} (deepseek4.cpp:1290-1301);</li>
 *   <li>weights = probs of the selected experts, softmaxed for SOFTMAX_WEIGHT, normalized by their sum clamped to
 *       {@link #NORM_SUM_MIN} when {@link #normWeights()}, scaled by {@link #weightsScale()} when not 0 or 1;</li>
 *   <li>experts: gate/up (fused or separate, with {@code *_exps.scale}), GLU or activation, down; weighted sum,
 *       or input weighting before the experts when {@link #weightBeforeFfn()} (llama4);</li>
 *   <li>plus the {@link #sharedExpert()} FFN, gated by {@code sigmoid(ffn_gate_inp_shexp · x)} when
 *       {@link #sharedExpertGated()}; the sum of routed and shared output is multiplied by
 *       {@link #sharedCombineScale()} (cohere2moe halves it, models/cohere2moe.cpp:242-243, 396-397).</li>
 * </ol>
 * The expert clamp ({@code ActParams.limit}) applies only with a separate {@code ffn_gate_exps}; the fused
 * gate_up path scales by {@code ffn_up_exps.scale} only. Immutable.
 */
public final class MoeSpec {

    /** Smallest weight sum used when normalizing (smallest normal F16, llama-graph.cpp:2339). */
    public static final float NORM_SUM_MIN = 6.103515625e-5f;

    /** Scores used to pick experts; the code is the router kernel's {@code sel_mode}. */
    public enum SelectionMode {
        /** Gating probabilities (+ {@code exp_probs_b}). */
        PROBS(0),
        /** Raw router logits (llama4, llama-graph.cpp:2270-2272). */
        RAW_LOGITS(1),
        /** {@code sigmoid(logits)} (grovemoe, llama-graph.cpp:2274-2277). */
        SIGMOID_LOGITS(2);

        private final int code;

        SelectionMode(int code) {
            this.code = code;
        }

        public int code() {
            return code;
        }
    }

    private final int nExpert;
    private final int zeroExperts;
    private final int nUsed;
    private final int nFfExp;
    private final GatingFunc gating;
    private final SelectionMode selectionMode;
    private final boolean probBias;
    private final boolean routerBias;
    private final int nGroups;
    private final int nGroupUsed;
    private final boolean perGroupTopK;
    private final boolean normWeights;
    private final float weightsScale;
    private final boolean gated;
    private final boolean fusedGateUp;
    private final GluOp glu;
    private final ActOp act;
    private final ActParams params;
    private final boolean expertScales;
    private final boolean expertBiases;
    private final int latentSize;
    private final int idDiv;
    private final int nChunkExperts;
    private final int nFfChunk;
    private final float chunkScale;
    private final boolean hashRouting;
    private final boolean mediaProbBias;
    private final boolean weightBeforeFfn;
    private final FfnSpec sharedExpert;
    private final boolean sharedExpertGated;
    private final boolean postExpertNorm;
    private final float sharedCombineScale;

    private MoeSpec(Builder b) {
        this.nExpert = b.nExpert;
        this.zeroExperts = b.zeroExperts;
        this.nUsed = b.nUsed;
        this.nFfExp = b.nFfExp;
        this.gating = b.gating;
        this.selectionMode = b.selectionMode;
        this.probBias = b.probBias;
        this.routerBias = b.routerBias;
        this.nGroups = b.nGroups;
        this.nGroupUsed = b.nGroupUsed;
        this.perGroupTopK = b.perGroupTopK;
        this.normWeights = b.normWeights;
        this.weightsScale = b.weightsScale;
        this.gated = b.gated;
        this.fusedGateUp = b.fusedGateUp;
        this.glu = b.glu;
        this.act = b.act;
        this.params = b.params;
        this.expertScales = b.expertScales;
        this.expertBiases = b.expertBiases;
        this.latentSize = b.latentSize;
        this.idDiv = b.idDiv;
        this.nChunkExperts = b.nChunkExperts;
        this.nFfChunk = b.nFfChunk;
        this.chunkScale = b.chunkScale;
        this.hashRouting = b.hashRouting;
        this.mediaProbBias = b.mediaProbBias;
        this.weightBeforeFfn = b.weightBeforeFfn;
        this.sharedExpert = b.sharedExpert;
        this.sharedExpertGated = b.sharedExpertGated;
        this.postExpertNorm = b.postExpertNorm;
        this.sharedCombineScale = b.sharedCombineScale;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** A builder holding every field of {@code of}, so a split can replace one dimension of the expert set. */
    public static Builder builder(MoeSpec of) {
        Objects.requireNonNull(of, "of");
        Builder b = new Builder();
        b.nExpert = of.nExpert;
        b.zeroExperts = of.zeroExperts;
        b.nUsed = of.nUsed;
        b.nFfExp = of.nFfExp;
        b.gating = of.gating;
        b.selectionMode = of.selectionMode;
        b.probBias = of.probBias;
        b.routerBias = of.routerBias;
        b.nGroups = of.nGroups;
        b.nGroupUsed = of.nGroupUsed;
        b.normWeights = of.normWeights;
        b.weightsScale = of.weightsScale;
        b.gated = of.gated;
        b.fusedGateUp = of.fusedGateUp;
        b.glu = of.glu;
        b.act = of.act;
        b.params = of.params;
        b.expertScales = of.expertScales;
        b.expertBiases = of.expertBiases;
        b.latentSize = of.latentSize;
        b.idDiv = of.idDiv;
        b.nChunkExperts = of.nChunkExperts;
        b.nFfChunk = of.nFfChunk;
        b.chunkScale = of.chunkScale;
        b.hashRouting = of.hashRouting;
        b.mediaProbBias = of.mediaProbBias;
        b.weightBeforeFfn = of.weightBeforeFfn;
        b.sharedExpert = of.sharedExpert;
        b.sharedExpertGated = of.sharedExpertGated;
        b.postExpertNorm = of.postExpertNorm;
        b.sharedCombineScale = of.sharedCombineScale;
        return b;
    }

    /**
     * Whether layer {@code il} runs routed experts: {@code ffn_gate_inp} is present (authoritative),
     * {@code il >= nDenseLead}, {@code moeLayerStep == 0 || (il + 1) % moeLayerStep == 0}, and
     * {@code moeEveryN == 0 || il % moeEveryN == 1}. Only bert / nomic-bert-moe use the {@code il % n == 1} rule
     * (models/bert.cpp:171); callers pass {@code moeEveryN = 0} for every other arch, including k2-horizon,
     * which reads the key but selects by dense lead only (models/k2-horizon.cpp:67).
     *
     * @throws IllegalStateException when the hparams rule and the tensor presence disagree
     */
    public static boolean isMoeLayer(int il, boolean hasGateInp, int nDenseLead, int moeLayerStep, int moeEveryN) {
        boolean byHparams = il >= nDenseLead
                && (moeLayerStep <= 0 || (il + 1) % moeLayerStep == 0)
                && (moeEveryN <= 0 || il % moeEveryN == 1);
        if (hasGateInp && !byHparams) {
            throw new IllegalStateException("layer " + il + " has ffn_gate_inp but the hparams mark it dense (dense lead "
                    + nDenseLead + ", step " + moeLayerStep + ", every " + moeEveryN + ")");
        }
        return hasGateInp;
    }

    /** Real experts, the ones whose FFN weights the expert tensors hold. */
    public int nExpert() { return nExpert; }
    /**
     * Routed slots that compute the identity instead of an expert FFN, appended after the real experts (vLLM
     * {@code zero_expert_num} of longcat_flash.py, {@code FusedMoE(zero_expert_type="identity")}). Their share of the
     * routing weight multiplies the expert input, and the expert tensors hold only {@link #nExpert()} experts.
     */
    public int zeroExperts() { return zeroExperts; }
    /**
     * Experts the router chooses among: {@code nExpert + zeroExperts}. The router weight, its bias and every routing
     * decision have this width.
     */
    public int nRouterExpert() { return nExpert + zeroExperts; }
    /** Experts per token ({@code n_expert_used}, per layer when the GGUF has an array). */
    public int nUsed() { return nUsed; }
    /** Expert FFN width ({@code n_ff_exp}). */
    public int nFfExp() { return nFfExp; }
    public GatingFunc gating() { return gating; }
    public SelectionMode selectionMode() { return selectionMode; }
    /** {@code exp_probs_b} selection bias present. */
    public boolean probBias() { return probBias; }
    /** {@code ffn_gate_inp.bias} present. */
    public boolean routerBias() { return routerBias; }
    public int nGroups() { return nGroups; }
    public int nGroupUsed() { return nGroupUsed; }
    /**
     * True when every group contributes {@code nUsed / nGroups} experts (vLLM's InternS1-Pro router); false for the
     * llama.cpp group-limited selection that keeps the {@link #nGroupUsed()} best groups and takes the top
     * {@link #nUsed()} of the rest.
     */
    public boolean perGroupTopK() { return perGroupTopK; }
    public boolean normWeights() { return normWeights; }
    /** Weight multiplier; 0 or 1 leaves the weights unscaled. */
    public float weightsScale() { return weightsScale; }
    /** Experts have a gate ({@code ffn_gate_exps} or the fused gate half). */
    public boolean gated() { return gated; }
    public boolean fusedGateUp() { return fusedGateUp; }
    /** Expert GLU op when gated; null otherwise. */
    public GluOp glu() { return glu; }
    /** Expert activation when ungated; null otherwise. */
    public ActOp act() { return act; }
    public ActParams params() { return params; }
    /** {@code *_exps.scale} per-expert scales present. */
    public boolean expertScales() { return expertScales; }
    /** {@code *_exps.bias} per-expert biases present (gpt-oss). */
    public boolean expertBiases() { return expertBiases; }
    /** Latent MoE width; 0 when the experts run on the model width. */
    public int latentSize() { return latentSize; }
    /** grovemoe chunk experts: selected ids are divided by this ({@code n_group_experts}); 0 without chunks. */
    public int idDiv() { return idDiv; }
    /** Number of chunk experts ({@code n_expert / n_group_experts}); 0 without chunks. */
    public int nChunkExperts() { return nChunkExperts; }
    /** Chunk expert FFN width ({@code n_ff_chexp}). */
    public int nFfChunk() { return nFfChunk; }
    /** Chunk branch multiplier ({@code expert_group_scale}). */
    public float chunkScale() { return chunkScale; }
    /** Expert ids come from the {@code ffn_gate_tid2eid} token table (deepseek4 hash layers). */
    public boolean hashRouting() { return hashRouting; }
    /** Embedding rows select experts with {@code exp_probs_b_vl} as the selection bias (deepseek4 vision files). */
    public boolean mediaProbBias() { return mediaProbBias; }
    /**
     * Embedding rows route otherwise than token rows: through the router with {@code exp_probs_b_vl} or no bias on a
     * hash layer, or with {@code exp_probs_b_vl} in place of {@code exp_probs_b} (deepseek4.cpp:1290-1301).
     */
    public boolean mediaRouting() { return hashRouting || mediaProbBias; }
    /** Router weights multiply the expert input instead of the output (llama4). */
    public boolean weightBeforeFfn() { return weightBeforeFfn; }
    /** Shared expert FFN; null when absent. */
    public FfnSpec sharedExpert() { return sharedExpert; }
    /** Shared expert output is multiplied by {@code sigmoid(ffn_gate_inp_shexp · x)}. */
    public boolean sharedExpertGated() { return sharedExpertGated; }
    /** A norm is applied to the expert output before the down/latent-up projection ({@code post_exp_norm}). */
    public boolean postExpertNorm() { return postExpertNorm; }
    /** Multiplier on {@code routed + shared} when a {@link #sharedExpert()} exists; 1 leaves the sum unchanged. */
    public float sharedCombineScale() { return sharedCombineScale; }

    public boolean latent() { return latentSize > 0; }
    public boolean chunked() { return idDiv > 0; }

    @Override
    public String toString() {
        return "MoeSpec{experts=" + nExpert + (zeroExperts > 0 ? " (" + zeroExperts + " zero)" : "") + "/" + nUsed + ", nFfExp=" + nFfExp + ", gating=" + gating + ", sel="
                + selectionMode + ", groups=" + nGroups + "/" + nGroupUsed + ", norm=" + normWeights + ", scale="
                + weightsScale + ", op=" + (glu != null ? glu : act) + ", fused=" + fusedGateUp + ", latent=" + latentSize
                + ", chunks=" + nChunkExperts + ", hash=" + hashRouting + (mediaProbBias ? ", mediaBias" : "") + ", shared=" + (sharedExpert != null)
                + (sharedCombineScale != 1.0f ? ", sharedCombineScale=" + sharedCombineScale : "") + '}';
    }

    public static final class Builder {
        private int nExpert;
        private int zeroExperts;
        private int nUsed;
        private int nFfExp;
        private GatingFunc gating = GatingFunc.SOFTMAX;
        private SelectionMode selectionMode = SelectionMode.PROBS;
        private boolean probBias;
        private boolean routerBias;
        private int nGroups = 1;
        private int nGroupUsed = 1;
        private boolean perGroupTopK;
        private boolean normWeights;
        private float weightsScale = 1.0f;
        private boolean gated = true;
        private boolean fusedGateUp;
        private GluOp glu = GluOp.SWIGLU;
        private ActOp act;
        private ActParams params = ActParams.NONE;
        private boolean expertScales;
        private boolean expertBiases;
        private int latentSize;
        private int idDiv;
        private int nChunkExperts;
        private int nFfChunk;
        private float chunkScale = 1.0f;
        private boolean hashRouting;
        private boolean mediaProbBias;
        private boolean weightBeforeFfn;
        private FfnSpec sharedExpert;
        private boolean sharedExpertGated;
        private boolean postExpertNorm;
        private float sharedCombineScale = 1.0f;

        private Builder() {
        }

        public Builder experts(int nExpert, int nUsed, int nFfExp) {
            this.nExpert = nExpert;
            this.nUsed = nUsed;
            this.nFfExp = nFfExp;
            return this;
        }
        public Builder gating(GatingFunc v) { this.gating = Objects.requireNonNull(v, "gating"); return this; }
        /**
         * Identity slots appended after the real experts: the router and its biases hold
         * {@code nExpert + zeroExperts} entries while the expert tensors keep {@code nExpert}.
         */
        public Builder zeroExperts(int v) { this.zeroExperts = v; return this; }
        public Builder selectionMode(SelectionMode v) {
            this.selectionMode = Objects.requireNonNull(v, "selectionMode");
            return this;
        }
        public Builder probBias(boolean v) { this.probBias = v; return this; }
        public Builder routerBias(boolean v) { this.routerBias = v; return this; }
        public Builder groups(int nGroups, int nGroupUsed) {
            this.nGroups = nGroups;
            this.nGroupUsed = nGroupUsed;
            return this;
        }

        /** Selection that takes {@code nUsed / nGroups} experts from every group (InternS1-Pro's router). */
        public Builder perGroupTopK(boolean v) {
            this.perGroupTopK = v;
            return this;
        }
        public Builder normWeights(boolean v) { this.normWeights = v; return this; }
        public Builder weightsScale(float v) { this.weightsScale = v; return this; }
        /** Gated experts with a GLU op. */
        public Builder glu(GluOp op) {
            this.glu = Objects.requireNonNull(op, "glu");
            this.act = null;
            this.gated = true;
            return this;
        }
        /** Ungated experts ({@code down(act(up(x)))}). */
        public Builder act(ActOp op) {
            this.act = Objects.requireNonNull(op, "act");
            this.glu = null;
            this.gated = false;
            return this;
        }
        public Builder fusedGateUp(boolean v) { this.fusedGateUp = v; return this; }
        public Builder params(ActParams v) { this.params = Objects.requireNonNull(v, "params"); return this; }
        public Builder expertScales(boolean v) { this.expertScales = v; return this; }
        public Builder expertBiases(boolean v) { this.expertBiases = v; return this; }
        public Builder latentSize(int v) { this.latentSize = v; return this; }
        public Builder chunks(int idDiv, int nChunkExperts, int nFfChunk, float chunkScale) {
            this.idDiv = idDiv;
            this.nChunkExperts = nChunkExperts;
            this.nFfChunk = nFfChunk;
            this.chunkScale = chunkScale;
            return this;
        }
        public Builder hashRouting(boolean v) { this.hashRouting = v; return this; }
        public Builder mediaProbBias(boolean v) { this.mediaProbBias = v; return this; }
        public Builder weightBeforeFfn(boolean v) { this.weightBeforeFfn = v; return this; }
        public Builder sharedExpert(FfnSpec ffn, boolean gatedByRouter) {
            this.sharedExpert = ffn;
            this.sharedExpertGated = gatedByRouter;
            return this;
        }
        public Builder postExpertNorm(boolean v) { this.postExpertNorm = v; return this; }
        public Builder sharedCombineScale(float v) { this.sharedCombineScale = v; return this; }

        /** @throws IllegalArgumentException for inconsistent expert counts, groups, chunks or op choice */
        public MoeSpec build() {
            if (nExpert <= 0 || nUsed <= 0 || nUsed > nExpert + zeroExperts || nFfExp <= 0) {
                throw new IllegalArgumentException("MoE needs 0 < nUsed <= nExpert + zeroExperts and nFfExp > 0,"
                        + " got " + nUsed + "/" + nExpert + " + " + zeroExperts + ", nFfExp " + nFfExp);
            }
            if (zeroExperts < 0 || zeroExperts > Integer.MAX_VALUE - nExpert) {
                throw new IllegalArgumentException("MoE zero-expert count must be in [0, " + (Integer.MAX_VALUE - nExpert)
                        + "], got " + zeroExperts);
            }
            if (zeroExperts > 0 && (nGroups > 1 || hashRouting || mediaProbBias)) {
                throw new IllegalArgumentException("MoE zero experts select over the whole router width; grouped,"
                        + " hash and embedding-row routing do not define their identity share: " + nExpert
                        + " experts with " + zeroExperts + " zero");
            }
            if (zeroExperts > 0 && (latentSize > 0 || idDiv > 0 || weightBeforeFfn)) {
                throw new IllegalArgumentException("MoE zero experts need the expert input as their identity"
                        + " output, which the latent, chunked and input-weighted graphs do not define: " + nExpert
                        + " experts with " + zeroExperts + " zero (latent " + latentSize + ", chunks " + idDiv
                        + ", weightBefore " + weightBeforeFfn + ")");
            }
            if (nGroups < 1 || nGroupUsed < 1 || nGroupUsed > nGroups || nExpert % nGroups != 0) {
                throw new IllegalArgumentException("MoE groups " + nGroupUsed + "/" + nGroups + " are invalid for "
                        + nExpert + " experts");
            }
            if (perGroupTopK && (nGroups <= 1 || nUsed % nGroups != 0 || nUsed / nGroups < 1)) {
                throw new IllegalArgumentException("per-group top-k needs more than one group and a selected count"
                        + " divisible by the group count, got " + nUsed + " of " + nGroups + " groups");
            }
            if (fusedGateUp && !gated) {
                throw new IllegalArgumentException("fused gate_up experts must be gated");
            }
            if (sharedExpert != null && sharedExpert.role() != FfnSpec.Role.SHARED_EXPERT) {
                throw new IllegalArgumentException("MoE shared expert must have role SHARED_EXPERT, got " + sharedExpert.role());
            }
            if (sharedExpertGated && sharedExpert == null) {
                throw new IllegalArgumentException("MoE shared-expert gate without a shared expert");
            }
            if (idDiv < 0 || (idDiv > 0 && (nChunkExperts <= 0 || nFfChunk <= 0 || nExpert / idDiv != nChunkExperts))) {
                throw new IllegalArgumentException("MoE chunk experts need nChunkExperts == nExpert / idDiv and nFfChunk > 0, got "
                        + nChunkExperts + ", " + nExpert + "/" + idDiv + ", " + nFfChunk);
            }
            if (!(sharedCombineScale > 0f) || Float.isInfinite(sharedCombineScale)
                    || (sharedCombineScale != 1.0f && sharedExpert == null)) {
                throw new IllegalArgumentException("MoE shared combine scale must be positive and finite and needs a"
                        + " shared expert when not 1, got " + sharedCombineScale);
            }
            if (latentSize < 0 || Float.isNaN(weightsScale) || Float.isNaN(chunkScale)) {
                throw new IllegalArgumentException("MoE latent size must be >= 0 and scales must not be NaN");
            }
            return new MoeSpec(this);
        }
    }
}
