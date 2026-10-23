package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * Resolves softmax-attention layers into {@link AttentionSpec}s: per-layer heads and dims (SWA-aware), NoPE layers
 * and partial rotary dims (via {@link RopeResolver}), causality, the sliding window, Q/K norms (via
 * {@link NormResolver}), output gates, attention temperature, KQV clamp, sinks, logit softcap, value scale, ALiBi,
 * T5 relative-position bias, the KQ scale, cache use and sharing, fused QKV, deci's linear replacement, the
 * {@code attn_output.bias} switch and k2-horizon's MoVA routed value experts.
 *
 * <p>The caller decides which layers are softmax-attention layers (recurrent, MLA, DSA, MSA and DSV4 mixers have
 * their own resolvers); {@link #attentionKind(int)} refines an attention layer into
 * {@link MixerKind#ATTENTION}, {@link MixerKind#ATTENTION_LINEAR_REPLACE} or {@link MixerKind#NONE} (deci), and
 * the T5 decoder stack is {@link MixerKind#CROSS_ATTENTION_T5}. Immutable.
 */
public final class AttentionResolver {

    /** T5 relative-position max distance (llama-graph.cpp {@code llama_relative_position_bucket}, max 128). */
    private static final int T5_MAX_DISTANCE = 128;

    private final ModelHparams hp;
    private final TensorCatalog tc;
    private final RopeResolver rope;
    private final NormResolver norm;
    private final Arch arch;

    public AttentionResolver(ModelHparams hp, TensorCatalog tc, RopeResolver rope, NormResolver norm) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.tc = Objects.requireNonNull(tc, "tc");
        this.rope = Objects.requireNonNull(rope, "rope");
        this.norm = Objects.requireNonNull(norm, "norm");
        this.arch = hp.arch();
    }

    /**
     * Mixer kind of attention layer {@code il} of the main stack. deci (models/deci.cpp:106-118):
     * {@code n_head == 0} is attention-free ({@link MixerKind#NONE}), {@code n_head_kv == 0} is the linear
     * replacement {@code wo @ attn_norm(x)}; every other layer is {@link MixerKind#ATTENTION}.
     */
    public MixerKind attentionKind(int il) {
        checkLayer(il);
        if (arch == Arch.DECI) {
            if (hp.nHead(il) == 0) {
                return MixerKind.NONE;
            }
            if (hp.nHeadKv(il) == 0) {
                return MixerKind.ATTENTION_LINEAR_REPLACE;
            }
        }
        return MixerKind.ATTENTION;
    }

    /**
     * Mixer kind of layer {@code il} of {@code stack}: the T5 decoder stack is
     * {@link MixerKind#CROSS_ATTENTION_T5}, encoder stacks are {@link MixerKind#ATTENTION}, and the main stack
     * follows {@link #attentionKind(int)}.
     */
    public MixerKind attentionKind(String stack, int il) {
        Objects.requireNonNull(stack, "stack");
        if (TensorNames.STACK_DEC.equals(stack)) {
            return MixerKind.CROSS_ATTENTION_T5;
        }
        if (TensorNames.STACK_ENC.equals(stack)) {
            return MixerKind.ATTENTION;
        }
        return attentionKind(il);
    }

    /**
     * Self-attention of main-stack layer {@code il}.
     *
     * @return the spec, or null for an attention-free layer ({@link MixerKind#NONE})
     * @throws IllegalStateException    for a gate or norm layout no spec field describes
     * @throws IllegalArgumentException when the resolved values are inconsistent
     */
    public AttentionSpec resolve(int il) {
        MixerKind kind = attentionKind(il);
        if (kind == MixerKind.NONE) {
            return null;
        }
        boolean linear = kind == MixerKind.ATTENTION_LINEAR_REPLACE;
        boolean swaLayer = hp.isSwa(il) && hp.swaType() != SwaType.NONE;
        int headK = hp.headK(il);
        int headV = hp.headV(il);
        int nHead = hp.nHead(il);
        int nRot = linear ? 0 : rope.nRot(il);
        AttentionSpec.Builder b = AttentionSpec.builder()
                .nHead(nHead)
                .nHeadKv(linear ? 0 : hp.nHeadKv(il))
                .headDimK(headK)
                .headDimV(headV)
                .nRot(nRot)
                .noRope(nRot == 0)
                .causal(causal(il))
                .swaType(hp.swaType())
                .nSwa(hp.nSwa())
                .swaLayer(swaLayer)
                .qkNorm(norm.qkNorm(il))
                .qkNormAfterRope(nRot != 0 && norm.qkNormAfterRope(il))
                .qkNormBias(norm.qkNormBias(il))
                .inBias(!UnusedBiases.attnInputUnused(arch))
                .outBias(!UnusedBiases.attnOutputUnused(arch))
                .vNorm(vNorm(il))
                .gate(linear ? AttnGateKind.NONE : gate(il, nHead, headV))
                .temperature(temperature(il))
                .clampKqv(hp.clampKqv() > 0f ? hp.clampKqv() : 0f)
                .sinks(has(il, TensorNames.ATTN_SINKS))
                .softcap(softcap())
                .valueScale(hp.attnValueScale())
                .kqScale(kqScale(il))
                .outScale(arch == Arch.GROK ? hp.attnOutScale() : 0f)
                .usesMemory(usesMemory(false))
                .kvOwnerLayer(hp.kvOwnerLayer(il))
                .fusedQkv(tc.fusedQkv(il) != null)
                .linearReplace(linear)
                .valueExperts(linear ? null : valueExperts(il, hp.nHeadKv(il) * headV));
        if (hp.maxAlibiBias() > 0f) {
            b.biasMode(AttentionSpec.BiasMode.ALIBI).alibiMaxBias(hp.maxAlibiBias());
        }
        return b.build();
    }

    /**
     * Self-attention of layer {@code il} of a T5 stack ({@link TensorNames#STACK_ENC} bidirectional,
     * {@link TensorNames#STACK_DEC} causal), or of the main stack via {@link #resolve(int)}. T5 attention has no
     * RoPE and {@code kq_scale 1}; the relative-position bias comes from the layer's {@code attn_rel_b}, else
     * layer 0's (models/t5.cpp:146-151, 303-305).
     *
     * @throws IllegalArgumentException when the resolved values are inconsistent
     */
    public AttentionSpec resolve(String stack, int il) {
        Objects.requireNonNull(stack, "stack");
        if (TensorNames.STACK_BLK.equals(stack)) {
            return resolve(il);
        }
        boolean decoder = TensorNames.STACK_DEC.equals(stack);
        if (!decoder && !TensorNames.STACK_ENC.equals(stack)) {
            throw new IllegalArgumentException(arch.ggufName() + ": no self-attention for stack " + stack);
        }
        String relB = TensorNames.weight(decoder ? TensorNames.DEC_ATTN_REL_B : TensorNames.ENC_ATTN_REL_B);
        int biasLayer = tc.layer(stack, il, relB) != null ? il : 0;
        return t5Base(il, decoder ? stack + " self" : stack)
                .causal(decoder)
                .biasMode(decoder ? AttentionSpec.BiasMode.T5_CAUSAL : AttentionSpec.BiasMode.T5_BIDIR)
                .t5Buckets(hp.nRelAttnBuckets())
                .t5MaxDistance(T5_MAX_DISTANCE)
                .t5BiasLayer(biasLayer)
                .usesMemory(usesMemory(!decoder))
                .build();
    }

    /**
     * Cross-attention of T5 decoder layer {@code il} over the encoder output: bidirectional, no position bias,
     * {@code kq_scale 1} (models/t5.cpp:170-182).
     */
    public AttentionSpec crossAttention(int il) {
        return t5Base(il, TensorNames.STACK_DEC + " cross").causal(false).usesMemory(true).build();
    }

    /**
     * Attention facet of MLA layer {@code il} with geometry {@code m}: the fields the MLA graphs share with softmax
     * attention (causality, sliding window, Q/K norm kind, output gate, temperature, sinks, logit softcap, cache use
     * and KV owner) over the decompressed per-head view of {@code m}: {@code nHead} query and KV heads,
     * {@code [nope | rope]} keys, {@code headV} values, {@link MlaSpec#kqScale()}. The query temperature applies to
     * the deepseek2 graph family only (models/deepseek2.cpp:446-449, 577-580, 618-621); the gate is resolved over
     * the MLA heads and value width (per-head sigmoid of dots3note and bailingmoe3, element-wise sigmoid of kimi-k3
     * and hy_v4).
     *
     * @throws IllegalStateException    for a gate layout no {@link AttnGateKind} describes
     * @throws IllegalArgumentException when the resolved values are inconsistent
     */
    public AttentionSpec mlaFacet(int il, MlaSpec m) {
        checkLayer(il);
        Objects.requireNonNull(m, "m");
        boolean roped = m.ropeApplied();
        AttentionSpec.Temperature temp = hp.traits().graphFamily() == Arch.DEEPSEEK2
                ? temperature() : AttentionSpec.Temperature.NONE;
        return AttentionSpec.builder()
                .nHead(m.nHead())
                .nHeadKv(m.nHead())
                .headDimK(m.headKNope() + m.headKRope())
                .headDimV(m.headV())
                .nRot(roped ? m.headKRope() : 0)
                .noRope(!roped)
                .causal(causal(il))
                .swaType(hp.swaType())
                .nSwa(hp.nSwa())
                .swaLayer(hp.isSwa(il) && hp.swaType() != SwaType.NONE)
                .qkNorm(norm.qkNorm(il))
                .qkNormAfterRope(roped && norm.qkNormAfterRope(il))
                .qkNormBias(norm.qkNormBias(il))
                .gate(gate(il, m.nHead(), m.headV()))
                .temperature(temp)
                .sinks(has(il, TensorNames.ATTN_SINKS))
                .softcap(softcap())
                .kqScale(m.kqScale())
                .usesMemory(usesMemory(false))
                .kvOwnerLayer(hp.kvOwnerLayer(il))
                .build();
    }

    /** T5 heads use the model-level (layer 0) counts and dims (models/t5.cpp:111-114, 144). */
    private AttentionSpec.Builder t5Base(int il, String what) {
        if (il < 0) {
            throw new IndexOutOfBoundsException(arch.ggufName() + ": " + what + " layer " + il);
        }
        return AttentionSpec.builder()
                .nHead(hp.nHead(0))
                .nHeadKv(hp.nHeadKv(0))
                .headDimK(hp.headK(0))
                .headDimV(hp.headV(0))
                .nRot(0)
                .noRope(true)
                .kqScale(1.0f)
                .kvOwnerLayer(il);
    }

    /**
     * Causal mask of main-stack layer {@code il}: the model's causality, except that with
     * {@code non_causal_type SWA_ONLY} the dense layers of a non-causal model stay causal
     * (llama-kv-cache.cpp:1777-1781).
     */
    public boolean causal(int il) {
        if (hp.causal()) {
            return true;
        }
        return hp.nonCausalType() == ModelHparams.NON_CAUSAL_SWA_ONLY && !(hp.isSwa(il) && hp.swaType() != SwaType.NONE);
    }

    /**
     * False when the attention reads only the current batch: models without memory (encoders, classifiers,
     * diffusion), the LFM2 decision graph (non-causal with decision layers, llama-model.cpp:2782-2784) and the T5
     * encoder stack.
     */
    private boolean usesMemory(boolean encoderStack) {
        if (encoderStack || hp.traits().memoryKind() == MemoryKind.NONE) {
            return false;
        }
        return !(arch == Arch.LFM2 && !hp.causal() && hp.nLayerDecision() > 0);
    }

    /**
     * {@code kq_scale} of main-stack layer {@code il}, including a query pre-scale the graph folds into Q:
     * <ul>
     *   <li>grok, t5: 1 (grok applies its scale with the softcap, see {@link AttentionSpec#outScale()});</li>
     *   <li>jais: {@code 1/head_dim} (models/jais.cpp);</li>
     *   <li>plamo2: {@code 1/sqrt(head_dim_v)};</li>
     *   <li>bailingmoe, gpt-oss: {@code 1/sqrt(n_rot)} with the graph-level {@code hparams.n_rot()} of layer 0
     *       (models/bailingmoe.cpp:111, models/openai-moe.cpp:113);</li>
     *   <li>deepseek2-ocr: the deepseek2 graph's YaRN {@code mscale^2 / sqrt(head_k)}, ignoring
     *       {@code attention.scale} (models/deepseek2.cpp:427-437, 484-486);</li>
     *   <li>otherwise {@code attention.scale} when set (granite family, llama-style graphs, gemma Q pre-scale,
     *       gemma3n/4 {@code f_attention_scale}), else {@code 1/sqrt(head_dim_k)}.</li>
     * </ul>
     */
    private float kqScale(int il) {
        switch (arch) {
            case GROK:
            case T5:
            case T5ENCODER:
                return 1.0f;
            case JAIS:
                return 1.0f / hp.headK(il);
            case PLAMO2:
                return invSqrt(hp.headV(il));
            case BAILINGMOE:
            case OPENAI_MOE:
                return invSqrt(hp.nRot(0));
            case DEEPSEEK2OCR:
                return DeepseekScale.yarnKqScale(hp, hp.headK(il));
            default:
                return hp.attentionScale() != 0f ? hp.attentionScale() : invSqrt(hp.headK(il));
        }
    }

    private static float invSqrt(int n) {
        if (n <= 0) {
            throw new IllegalStateException("attention scale needs a positive dimension, got " + n);
        }
        return 1.0f / (float) Math.sqrt((float) n);
    }

    /**
     * Attention logit softcap: {@code f_attn_logit_softcapping} when {@code attn_soft_cap} is set
     * (llama-graph.cpp:3038-3045), and always for grok (:3024-3035).
     */
    private float softcap() {
        if (hp.attnSoftcapEnabled() || arch == Arch.GROK) {
            return hp.attnSoftcap();
        }
        return 0f;
    }

    /** Weightless RMS norm on V per head: gemma3n, gemma4, gemma-embedding2; dflash when V reuses K. */
    private boolean vNorm(int il) {
        switch (arch) {
            case GEMMA3N:
            case GEMMA4:
            case GEMMA_EMBEDDING2:
                return true;
            case DFLASH:
                return !has(il, TensorNames.ATTN_V);
            default:
                return false;
        }
    }

    /**
     * Output gate of layer {@code il} with {@code nHead} heads of value width {@code headDimV}:
     * <ul>
     *   <li>a Q projection of width {@code 2 * n_head * head_dim} carries an interleaved sigmoid gate (qwen3next,
     *       qwen35, qwen35moe, qwen4exp, clef);</li>
     *   <li>{@code attn_gate} with {@code n_head * head_dim} rows: element-wise softplus for laguna (models/laguna.cpp:
     *       239-259), element-wise base-2 softplus {@code log2(1 + 2^x)} for k2-horizon (models/k2-horizon.cpp:
     *       281-294), element-wise sigmoid otherwise (afmoe, beam, hrm_text, muse-*, kimi-k3, hy_v4); beam adds the
     *       per-head {@code attn_gate.bias} before the sigmoid (models/beam.cpp:180-188), which the executor reads
     *       when present;</li>
     *   <li>{@code attn_gate} with {@code n_head} rows: one softplus per head for laguna XS.2
     *       (models/laguna.cpp:247-254); one sigmoid per head otherwise (step35, spark2_5, dots3note, bailingmoe3),
     *       with a per-head bias when {@code attn_gate.bias} exists.</li>
     * </ul>
     * Public for the MLA and DSA resolvers, whose layers carry the same gates.
     *
     * @throws IllegalStateException for a gate tensor of any other width, and for a k2-horizon gate that is not
     *                               element-wise (models/k2-horizon.cpp:96)
     */
    public AttnGateKind gate(int il, int nHead, int headDimV) {
        TensorRef q = tc.layer(il, TensorNames.weight(TensorNames.ATTN_Q));
        if (q != null && q.rank() == 2 && q.ne(1) == 2L * nHead * hp.headK(il)) {
            return AttnGateKind.Q_INTERLEAVED;
        }
        TensorRef g = tc.layer(il, TensorNames.weight(TensorNames.ATTN_GATE));
        if (g == null) {
            return AttnGateKind.NONE;
        }
        long width = g.rank() >= 2 ? g.ne(1) : 1;
        boolean perHead = width == nHead;
        boolean perElement = width == (long) nHead * headDimV;
        if (!perHead && !perElement) {
            throw new IllegalStateException(arch.ggufName() + " layer " + il + ": attention gate " + g + " is neither "
                    + nHead + " (per head) nor " + ((long) nHead * headDimV) + " (per element) wide");
        }
        if (arch == Arch.K2_HORIZON) {
            if (!perElement) {
                throw new IllegalStateException(arch.ggufName() + " layer " + il + ": attention gate " + g
                        + " must be " + ((long) nHead * headDimV) + " (per element) wide");
            }
            return AttnGateKind.SOFTPLUS_BASE2;
        }
        if (arch == Arch.LAGUNA) {
            return perHead ? AttnGateKind.PER_HEAD_SOFTPLUS : AttnGateKind.ELEMENTWISE_SOFTPLUS;
        }
        if (perElement) {
            return AttnGateKind.ELEMENTWISE_SIGMOID;
        }
        return tc.layer(il, TensorNames.bias(TensorNames.ATTN_GATE)) != null
                ? AttnGateKind.PER_HEAD_SIGMOID_BIAS : AttnGateKind.PER_HEAD_SIGMOID;
    }

    /**
     * MoVA routed value experts of layer {@code il} with value width {@code vDim}, or null when V is a plain
     * projection (models/k2-horizon.cpp:66-84, 137-205): {@code attn_v_gate} routes {@code n_value_expert} experts
     * with the model's gating function (softmax or sigmoid), {@code attn_v_gate.bias} biasing selection only, top
     * {@code n_value_expert_used}, weights renormalized when {@code expert_weights_norm} (sum clamped to
     * {@link MoeSpec#NORM_SUM_MIN}) and scaled by {@code expert_weights_scale} when not 0 or 1; expert {@code k}
     * computes {@code silu(attn_v_exps[k] · x)} and V is the weighted sum.
     *
     * @throws IllegalStateException when the expert tensors do not match the hparams, or the gating function is
     *                               neither softmax nor sigmoid (k2-horizon.cpp:148-152)
     */
    private MoeSpec valueExperts(int il, int vDim) {
        TensorRef exps = tc.layer(il, TensorNames.weight(TensorNames.ATTN_V_EXPS));
        if (exps == null) {
            return null;
        }
        if (arch != Arch.K2_HORIZON) {
            throw new IllegalStateException(arch.ggufName() + " layer " + il + ": routed value experts ("
                    + exps.name() + ") exist only in k2-horizon");
        }
        int nValue = hp.nValueExpert();
        TensorRef router = tc.layer(il, TensorNames.weight(TensorNames.ATTN_V_GATE));
        if (nValue <= 0 || router == null || exps.rank() != 3 || exps.ne(1) != vDim || exps.ne(2) != nValue
                || router.ne(1) != nValue) {
            throw new IllegalStateException(arch.ggufName() + " layer " + il + ": value experts " + exps
                    + " with router " + router + " do not match " + nValue + " experts of width " + vDim);
        }
        GatingFunc gating = hp.gating() != null ? hp.gating() : hp.traits().defaultGating();
        if (gating != GatingFunc.SOFTMAX && gating != GatingFunc.SIGMOID) {
            throw new IllegalStateException(arch.ggufName() + ": value-router gating " + gating
                    + " is neither softmax nor sigmoid");
        }
        return MoeSpec.builder()
                .experts(nValue, hp.nValueExpertUsed(), vDim)
                .gating(gating)
                .probBias(tc.layer(il, TensorNames.bias(TensorNames.ATTN_V_GATE)) != null)
                .normWeights(hp.expertWeightsNorm())
                .weightsScale(hp.expertWeightsScale() == 0f ? 1f : hp.expertWeightsScale())
                .act(ActOp.SILU)
                .build();
    }

    /**
     * Position-dependent query scale of layer {@code il} (llm_graph_input_attn_temp, llama-graph.cpp:213-238):
     * llama4 on its NoPE layers (models/llama4.cpp:141-175), beam on its full-attention (NoPE) layers
     * (models/beam.cpp:124-173), mistral3 on every layer when {@code attention.temperature_scale} is set
     * (models/mistral3.cpp:144-205); deepseek2 MLA layers use {@link #temperature()}.
     */
    public AttentionSpec.Temperature temperature(int il) {
        switch (arch) {
            case LLAMA4:
                return rope.appliesRope(il) ? AttentionSpec.Temperature.NONE : temperature();
            case BEAM:
                return hp.isSwa(il) ? AttentionSpec.Temperature.NONE : temperature();
            case MISTRAL3:
                return temperature();
            default:
                return AttentionSpec.Temperature.NONE;
        }
    }

    /** The model's temperature parameters, or {@link AttentionSpec.Temperature#NONE} when its scale is 0. */
    public AttentionSpec.Temperature temperature() {
        if (hp.attnTempScale() == 0f) {
            return AttentionSpec.Temperature.NONE;
        }
        if (hp.attnTempFloorScale() <= 0) {
            throw new IllegalStateException(arch.ggufName() + ": attention temperature scale " + hp.attnTempScale()
                    + " without a positive floor scale");
        }
        return AttentionSpec.Temperature.of(hp.attnTempScale(), hp.attnTempFloorScale(), hp.attnTempOffset(),
                hp.attnTempPow());
    }

    private boolean has(int il, String base) {
        return tc.layer(il, TensorNames.weight(base)) != null;
    }

    private void checkLayer(int il) {
        if (il < 0 || il >= hp.nLayerAll()) {
            throw new IndexOutOfBoundsException(arch.ggufName() + ": layer " + il + " outside [0, " + hp.nLayerAll() + ")");
        }
    }

    @Override
    public String toString() {
        return "AttentionResolver{" + arch.ggufName() + '}';
    }
}
