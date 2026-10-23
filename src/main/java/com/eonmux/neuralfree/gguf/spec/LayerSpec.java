package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Everything the forward pass needs to run one virtual layer, resolved from hparams, arch traits and tensor
 * presence. Forward code reads only this and {@link ModelSpec}, never GGUF keys or arch names.
 *
 * <p>The mixer is selected by {@link #mixerKind()} and described by the matching sub-spec:
 * <ul>
 *   <li>ATTENTION / ATTENTION_LINEAR_REPLACE: {@link #attention()};</li>
 *   <li>MLA: {@link #mla()} (plus {@link #attention()} for the shared causal / window / sinks / KV-owner fields
 *       when present); DSA_MLA: {@link #mla()} and {@link #dsa()};</li>
 *   <li>MSA: {@link #attention()} and {@link #msa()} (MiniMax-M3), or a QSA layer: {@link #dsa()} with a k-pool
 *       over {@link #attention()} (qwen4exp) or {@link #mla()} (glm5-next); DSV4: {@link #dsv4()};</li>
 *   <li>recurrent kinds: {@link #recurrent()}; with {@link #parallelAttention()} (falcon-h1) the
 *       {@link #attention()} branch runs on the same normed input and the two outputs are summed;</li>
 *   <li>CROSS_ATTENTION_T5 (T5 decoder): self-attention {@link #attention()}, then cross-attention
 *       {@link #crossAttention()} over the encoder output;</li>
 *   <li>NONE: no mixer.</li>
 * </ul>
 * The FFN is selected by {@link #ffnKind()}: DENSE / RWKV_CHANNEL_MIX use {@link #ffn()}; MOE, MOE_LATENT and
 * MOE_CHUNKED use {@link #moe()}; MOE_PARALLEL_DENSE uses both. {@link #extras()} lists the layer-extra feature
 * ids ({@link Extras}) bound around the block. Immutable.
 */
public final class LayerSpec {

    /** Layer-extra feature ids, matched against {@code GpuLayerExtraFactory.featureId()} and its CPU twin. */
    public static final class Extras {
        /** Hyper-connections (deepseek4, glm5-next, hy_v4, qwen4exp, dflash). */
        public static final String HC = "hc";
        /** kimi-k3 attention-residual stack. */
        public static final String ATTN_RES = "attn_res";
        /** gemma3n AltUp predict / correct. */
        public static final String ALTUP = "altup";
        /** gemma3n Laurel low-rank residual. */
        public static final String LAUREL = "laurel";
        /** Per-layer embeddings (gemma3n, gemma4, gemma-embedding2). */
        public static final String PLE = "ple";
        /** qwen4exp n-gram hash per-layer embeddings. */
        public static final String NGRAM_PLE = "ngram_ple";
        /** Embedding skip added to the layer output (talkie). */
        public static final String EMBD_SKIP = "embd_skip";
        /** Norm-preserving alpha / beta residual (muse-spark). */
        public static final String RESID_ALPHA_BETA = "resid_alpha_beta";
        /** nanbeige loop output norm ({@link LayerSchedule#normAfterSlot(int)}). */
        public static final String LOOP_NORM = "loop_norm";
        /** granite-switch switched LoRA. */
        public static final String SWITCHED_LORA = "switched_lora";
        /** Keeps the layer input for MoE blocks that read it after the mixer (arctic experts, smallthinker router). */
        public static final String MOE_LAYER_INPUT = "moe_layer_input";

        private Extras() {
        }
    }

    private final int il;
    private final int weightLayer;
    private final String stack;
    private final MixerKind mixerKind;
    private final FfnKind ffnKind;
    private final NormSpec norm;
    private final ResidualSpec residual;
    private final AttentionSpec attention;
    private final AttentionSpec crossAttention;
    private final boolean parallelAttention;
    private final MlaSpec mla;
    private final DsaSpec dsa;
    private final MsaSpec msa;
    private final Dsv4Spec dsv4;
    private final RecurrentSpec recurrent;
    private final FfnSpec ffn;
    private final MoeSpec moe;
    private final RopeSpec rope;
    private final List<String> extras;

    private LayerSpec(Builder b) {
        this.il = b.il;
        this.weightLayer = b.weightLayer;
        this.stack = b.stack;
        this.mixerKind = b.mixerKind;
        this.ffnKind = b.ffnKind;
        this.norm = b.norm;
        this.residual = b.residual;
        this.attention = b.attention;
        this.crossAttention = b.crossAttention;
        this.parallelAttention = b.parallelAttention;
        this.mla = b.mla;
        this.dsa = b.dsa;
        this.msa = b.msa;
        this.dsv4 = b.dsv4;
        this.recurrent = b.recurrent;
        this.ffn = b.ffn;
        this.moe = b.moe;
        this.rope = b.rope;
        this.extras = Collections.unmodifiableList(new ArrayList<>(b.extras));
    }

    /**
     * @param il          virtual layer index (execution order, KV slot)
     * @param mixerKind   mixer selector
     * @param ffnKind     FFN selector
     */
    public static Builder builder(int il, MixerKind mixerKind, FfnKind ffnKind) {
        return new Builder(il, mixerKind, ffnKind);
    }

    /** Virtual layer index. */
    public int il() { return il; }
    /** Physical layer whose tensors this layer uses ({@link LayerSchedule#weightLayer(int)}). */
    public int weightLayer() { return weightLayer; }
    /** Tensor stack prefix ({@link TensorNames#STACK_BLK}, {@link TensorNames#STACK_ENC}, {@link TensorNames#STACK_DEC}). */
    public String stack() { return stack; }
    public MixerKind mixerKind() { return mixerKind; }
    public FfnKind ffnKind() { return ffnKind; }
    public NormSpec norm() { return norm; }
    public ResidualSpec residual() { return residual; }
    /** Self-attention; null for layers without softmax attention. */
    public AttentionSpec attention() { return attention; }
    /** T5 decoder cross-attention; null otherwise. */
    public AttentionSpec crossAttention() { return crossAttention; }
    /** falcon-h1: attention runs in parallel with the recurrent mixer and the outputs are summed. */
    public boolean parallelAttention() { return parallelAttention; }
    public MlaSpec mla() { return mla; }
    public DsaSpec dsa() { return dsa; }
    public MsaSpec msa() { return msa; }
    public Dsv4Spec dsv4() { return dsv4; }
    public RecurrentSpec recurrent() { return recurrent; }
    public FfnSpec ffn() { return ffn; }
    public MoeSpec moe() { return moe; }
    /** RoPE of the attention branch; null for NoPE and non-attention layers. */
    public RopeSpec rope() { return rope; }
    /** Layer-extra feature ids, in binding order (unmodifiable). */
    public List<String> extras() { return extras; }

    public boolean hasExtra(String id) {
        return extras.contains(id);
    }

    /** True when the layer keeps recurrent state. */
    public boolean isRecurrent() {
        return recurrent != null;
    }

    /** True when the layer reads or writes a KV cache. */
    public boolean usesKv() {
        if (dsv4 != null) {
            return true;
        }
        if (mla != null) {
            return attention == null || attention.usesMemory();
        }
        return attention != null && attention.usesMemory() && !attention.linearReplace();
    }

    /** Full tensor name {@code <stack>.<weightLayer>.<suffix>}. */
    public String tensorName(String suffix) {
        return TensorNames.layerName(stack, weightLayer, suffix);
    }

    @Override
    public String toString() {
        return "LayerSpec{il=" + il + (weightLayer != il ? "->" + weightLayer : "") + ", " + stack + ", mixer=" + mixerKind
                + (parallelAttention ? "+ATTENTION" : "") + ", ffn=" + ffnKind + ", extras=" + extras + '}';
    }

    public static final class Builder {
        private final int il;
        private final MixerKind mixerKind;
        private final FfnKind ffnKind;
        private int weightLayer;
        private String stack = TensorNames.STACK_BLK;
        private NormSpec norm;
        private ResidualSpec residual;
        private AttentionSpec attention;
        private AttentionSpec crossAttention;
        private boolean parallelAttention;
        private MlaSpec mla;
        private DsaSpec dsa;
        private MsaSpec msa;
        private Dsv4Spec dsv4;
        private RecurrentSpec recurrent;
        private FfnSpec ffn;
        private MoeSpec moe;
        private RopeSpec rope;
        private final List<String> extras = new ArrayList<>();

        private Builder(int il, MixerKind mixerKind, FfnKind ffnKind) {
            this.il = il;
            this.weightLayer = il;
            this.mixerKind = Objects.requireNonNull(mixerKind, "mixerKind");
            this.ffnKind = Objects.requireNonNull(ffnKind, "ffnKind");
        }

        public Builder weightLayer(int v) { this.weightLayer = v; return this; }
        public Builder stack(String v) { this.stack = Objects.requireNonNull(v, "stack"); return this; }
        public Builder norm(NormSpec v) { this.norm = v; return this; }
        public Builder residual(ResidualSpec v) { this.residual = v; return this; }
        public Builder attention(AttentionSpec v) { this.attention = v; return this; }
        public Builder crossAttention(AttentionSpec v) { this.crossAttention = v; return this; }
        public Builder parallelAttention(boolean v) { this.parallelAttention = v; return this; }
        public Builder mla(MlaSpec v) { this.mla = v; return this; }
        public Builder dsa(DsaSpec v) { this.dsa = v; return this; }
        public Builder msa(MsaSpec v) { this.msa = v; return this; }
        public Builder dsv4(Dsv4Spec v) { this.dsv4 = v; return this; }
        public Builder recurrent(RecurrentSpec v) { this.recurrent = v; return this; }
        public Builder ffn(FfnSpec v) { this.ffn = v; return this; }
        public Builder moe(MoeSpec v) { this.moe = v; return this; }
        public Builder rope(RopeSpec v) { this.rope = v; return this; }

        /** Appends a layer-extra feature id; duplicates are rejected at {@link #build()}. */
        public Builder extra(String id) {
            if (id == null || id.isEmpty()) {
                throw new IllegalArgumentException("layer extra id is empty");
            }
            this.extras.add(id);
            return this;
        }

        /**
         * @throws IllegalArgumentException naming the layer when a sub-spec required by the mixer or FFN kind is
         *                                  missing, a sub-spec of another kind is present, or shared facets disagree
         */
        public LayerSpec build() {
            if (il < 0 || weightLayer < 0) {
                throw new IllegalArgumentException("layer index " + il + " / weight layer " + weightLayer + " must be >= 0");
            }
            require(norm != null, "a NormSpec");
            require(residual != null, "a ResidualSpec");
            checkMixer();
            checkFfn();
            if (attention != null) {
                require(norm.qkNorm() == attention.qkNorm(), "matching Q/K norm kinds in NormSpec ("
                        + norm.qkNorm() + ") and AttentionSpec (" + attention.qkNorm() + ")");
            }
            checkRope();
            if (dsa != null) {
                require(dsa.full() == (dsa.sourceLayer() == il), "a DSA source layer equal to the layer exactly when full");
            }
            require(new HashSet<>(extras).size() == extras.size(), "distinct extras " + extras);
            return new LayerSpec(this);
        }

        private void checkRope() {
            if (mla != null) {
                require(mla.ropeApplied() == (rope != null), "a RopeSpec exactly when the MLA applies RoPE");
                if (rope != null) {
                    require(rope.nDims() == mla.headKRope(), "MLA rope dims " + mla.headKRope() + " matching RopeSpec.nDims "
                            + rope.nDims());
                }
            } else if (dsv4 != null) {
                if (rope != null) {
                    require(rope.nDims() == dsv4.ropeDim() && rope.nOffs() == dsv4.headDim() - dsv4.ropeDim(),
                            "a DSV4 RopeSpec over the last " + dsv4.ropeDim() + " head dims");
                }
            } else if (attention != null) {
                boolean roped = !attention.noRope() && !attention.linearReplace();
                require(roped == (rope != null), "a RopeSpec exactly when the attention applies RoPE");
                if (rope != null) {
                    require(rope.nDims() == attention.nRot(), "RopeSpec.nDims " + rope.nDims() + " == AttentionSpec.nRot "
                            + attention.nRot());
                }
            } else {
                require(rope == null, "no RopeSpec without an attention branch");
            }
        }

        private void checkMixer() {
            switch (mixerKind) {
                case ATTENTION:
                    require(attention != null && !attention.linearReplace(), "an AttentionSpec");
                    absent(mla == null && dsa == null && msa == null && dsv4 == null && recurrent == null
                            && crossAttention == null);
                    break;
                case ATTENTION_LINEAR_REPLACE:
                    require(attention != null && attention.linearReplace(), "a linear-replace AttentionSpec");
                    absent(mla == null && dsa == null && msa == null && dsv4 == null && recurrent == null
                            && crossAttention == null);
                    break;
                case MLA:
                    require(mla != null, "an MlaSpec");
                    absent(dsa == null && msa == null && dsv4 == null && recurrent == null && crossAttention == null);
                    break;
                case DSA_MLA:
                    require(mla != null && dsa != null, "an MlaSpec and a DsaSpec");
                    absent(msa == null && dsv4 == null && recurrent == null && crossAttention == null);
                    break;
                case MSA:
                    checkMsa();
                    break;
                case DSV4:
                    require(dsv4 != null, "a Dsv4Spec");
                    absent(mla == null && dsa == null && msa == null && recurrent == null && crossAttention == null);
                    break;
                case CROSS_ATTENTION_T5:
                    require(attention != null && crossAttention != null, "self- and cross-AttentionSpecs");
                    absent(mla == null && dsa == null && msa == null && dsv4 == null && recurrent == null);
                    break;
                case NONE:
                    absent(attention == null && mla == null && dsa == null && msa == null && dsv4 == null
                            && recurrent == null && crossAttention == null);
                    break;
                default:
                    require(recurrent != null && recurrent.kind() == mixerKind, "a " + mixerKind + " RecurrentSpec");
                    require(parallelAttention == (attention != null), "an AttentionSpec exactly when parallelAttention");
                    absent(mla == null && dsa == null && msa == null && dsv4 == null && crossAttention == null);
                    return;
            }
            require(!parallelAttention, "parallelAttention only on recurrent mixers");
        }

        /**
         * The two {@link MixerKind#MSA} forms: MiniMax-M3 block-sparse attention ({@code attention} + {@code msa},
         * models/minimax-m3.cpp), or a QSA pooled-indexer layer ({@code dsa} with {@code kpool > 0}) over gated GQA
         * ({@code attention}, models/qwen4exp.cpp:741-1026) or nope MLA ({@code mla}, with an optional attention
         * facet, models/glm5-next.cpp:892-1108).
         */
        private void checkMsa() {
            absent(dsv4 == null && recurrent == null && crossAttention == null);
            if (msa != null) {
                require(attention != null, "an AttentionSpec with the MsaSpec");
                absent(mla == null && dsa == null);
                return;
            }
            require(dsa != null && dsa.kpool() > 0, "an MsaSpec, or a DsaSpec with a k-pool");
            require(mla != null || attention != null, "an MlaSpec or an AttentionSpec under the k-pool indexer");
        }

        private void checkFfn() {
            switch (ffnKind) {
                case DENSE:
                    require(ffn != null && ffn.role() == FfnSpec.Role.DENSE && moe == null, "a DENSE FfnSpec and no MoeSpec");
                    break;
                case RWKV_CHANNEL_MIX:
                    require(ffn != null && ffn.role() == FfnSpec.Role.RWKV_CHANNEL_MIX && moe == null,
                            "an RWKV_CHANNEL_MIX FfnSpec and no MoeSpec");
                    break;
                case MOE:
                    require(moe != null && ffn == null && !moe.latent() && !moe.chunked(), "a plain MoeSpec and no FfnSpec");
                    break;
                case MOE_LATENT:
                    require(moe != null && ffn == null && moe.latent(), "a latent MoeSpec and no FfnSpec");
                    break;
                case MOE_CHUNKED:
                    require(moe != null && ffn == null && moe.chunked(), "a chunked MoeSpec and no FfnSpec");
                    break;
                case MOE_PARALLEL_DENSE:
                    require(moe != null && ffn != null && ffn.role() == FfnSpec.Role.PARALLEL_DENSE,
                            "a MoeSpec and a PARALLEL_DENSE FfnSpec");
                    break;
                case NONE:
                    require(moe == null && ffn == null, "no FfnSpec or MoeSpec");
                    break;
                default:
                    throw new IllegalStateException("unhandled FFN kind " + ffnKind);
            }
        }

        private void require(boolean ok, String what) {
            if (!ok) {
                throw new IllegalArgumentException("layer " + il + " (" + mixerKind + "/" + ffnKind + ") needs " + what);
            }
        }

        private void absent(boolean ok) {
            require(ok, "no sub-specs of other mixer kinds");
        }
    }
}
