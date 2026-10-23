package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * Resolves the routed-expert FFN of a layer into a {@link MoeSpec}, following llama.cpp {@code build_moe_ffn}
 * (src/llama-graph.cpp:2189-2580) and the per-architecture calls into it.
 *
 * <p>A layer is a MoE layer exactly when {@code ffn_gate_inp.weight} is present; the hparams rule
 * of the architecture's graph (dense lead, {@code n_moe_layer_step}, the bert {@code il % n == 1} rule) is checked
 * against it by {@link MoeSpec#isMoeLayer}. Architectures whose graph tests tensor presence (jamba, mimo2, step35,
 * hy_v3, gemma4, ...) pass no hparams rule, so the expert count never marks every layer MoE.
 *
 * <p>Experts: a fused {@code ffn_gate_up_exps} is scaled by {@code ffn_up_exps.scale} and never clamped; the
 * SwiGLU clamp {@code swiglu_clamp_exp[il]} applies only with a separate {@code ffn_gate_exps} (:2412-2440).
 * Router: gating from the hparams, raw-logit selection for llama4 and sigmoid selection for grovemoe
 * (:2270-2277), {@code exp_probs_b} selection bias, {@code ffn_gate_inp.bias} logit bias, group-limited routing
 * when {@code n_expert_groups > 1} (:2281-2305), weight normalisation clamped to
 * {@link MoeSpec#NORM_SUM_MIN} (:2326-2339) and scaling. Immutable.
 */
public final class MoeResolver {

    /** cohere2moe {@code ggml_scale(routed + shared, 0.5f)} (models/cohere2moe.cpp:243). */
    static final float COHERE2MOE_SHARED_COMBINE_SCALE = 0.5f;

    private final ModelHparams hp;
    private final TensorCatalog tc;

    public MoeResolver(ModelHparams hp, TensorCatalog tc) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.tc = Objects.requireNonNull(tc, "tc");
    }

    /**
     * True when main-stack layer {@code il} runs routed experts.
     *
     * @throws IllegalStateException when {@code ffn_gate_inp} sits on a layer the graph's hparams rule keeps dense
     */
    public boolean isMoeLayer(int il) {
        checkLayer(il);
        boolean hasGateInp = tc.hasLayer(il, TensorNames.weight(TensorNames.FFN_GATE_INP));
        return MoeSpec.isMoeLayer(il, hasGateInp, denseLeadRule(), stepRule(), everyNRule());
    }

    /**
     * Routed experts of MoE layer {@code il}.
     *
     * @throws IllegalArgumentException when the layer is not a MoE layer
     * @throws MissingTensorException   when an expert or router tensor is missing
     * @throws IllegalStateException    for inconsistent expert shapes or an unsupported expert op
     */
    public MoeSpec resolve(int il) {
        if (!isMoeLayer(il)) {
            throw new IllegalArgumentException(arch() + ": layer " + il + " has no routed experts");
        }
        TensorCatalog.GateUpExps gu = tc.gateUpExps(il);
        if (gu == null) {
            throw new MissingTensorException(tc.archName(), il,
                    TensorNames.layerName(il, TensorNames.weight(TensorNames.FFN_UP_EXPS)));
        }
        long nExpert = gu.nExpert();
        long nFfExp = gu.nFf();
        TensorRef gateInp = tc.require(il, TensorNames.weight(TensorNames.FFN_GATE_INP));
        TensorRef down = tc.require(il, TensorNames.weight(TensorNames.FFN_DOWN_EXPS));
        checkShapes(il, gateInp, down, nExpert, nFfExp);

        MoeSpec.Builder b = MoeSpec.builder()
                .experts(toInt(nExpert, il), hp.nExpertUsed(il), toInt(nFfExp, il))
                .gating(gating())
                .selectionMode(selectionMode())
                .routerBias(tc.hasLayer(il, TensorNames.bias(TensorNames.FFN_GATE_INP)))
                .normWeights(hp.expertWeightsNorm())
                .weightsScale(hp.expertWeightsScale() == 0f ? 1f : hp.expertWeightsScale())
                .weightBeforeFfn(hp.arch() == Arch.LLAMA4)
                .postExpertNorm(tc.hasLayer(il, TensorNames.weight(TensorNames.FFN_POST_EXP_NORM)));
        if (hp.nExpertGroups() > 1) {
            b.groups(hp.nExpertGroups(), hp.nGroupUsed());
        }
        boolean hash = hashRouted(il);
        b.hashRouting(hash).probBias(!hash && hasProbBias(il));
        expertOp(b, il, gu);
        b.expertScales(expertScales(il, gu, nExpert));
        b.expertBiases(gu.gateUpBias() != null || gu.gateBias() != null || gu.upBias() != null
                || tc.hasLayer(il, TensorNames.bias(TensorNames.FFN_DOWN_EXPS)));
        b.latentSize(latentSize(il));
        if (hp.arch() == Arch.GROVEMOE) {
            chunks(b, il, nExpert);
        }
        sharedExpert(b, il);
        return b.build();
    }

    // ---- MoE layer rule ----

    /**
     * Leading dense blocks the graph tests ({@code il < n_layer_dense_lead}); 0 for graphs that test only tensor
     * presence or {@code n_moe_layer_step}.
     */
    private int denseLeadRule() {
        switch (hp.arch()) {
            case GRANITE_SWA:
            case MIMO2:
            case GEMMA4:
            case JAMBA:
            case LLAMA4:
            case LLAMA:
            case NEMOTRON_H:
            case NEMOTRON_H_MOE:
            case DECI:
            case MAION_CODER:
            case MISTRAL3:
            case GRANITE:
            case GRANITE_HYBRID:
            case HY_V3:
            case STEP35:
            case PHI3:
            case QWEN35:
            case QWEN3NEXT:
            case QWEN35MOE:
            case QWEN4EXP:
                return 0;
            default:
                return hp.nLayerDenseLead();
        }
    }

    /** {@code (il + 1) % n_moe_layer_step == 0} (llama4.cpp:62, ernie4-5-moe.cpp:64, maion-coder.cpp:38). */
    private int stepRule() {
        switch (hp.arch()) {
            case LLAMA4:
            case ERNIE4_5_MOE:
            case MAION_CODER:
                return hp.nMoeLayerStep();
            default:
                return 0;
        }
    }

    /** {@code il % moe_every_n_layers == 1} of the bert graph only (bert.cpp:171, nomic-bert-moe.cpp:37). */
    private int everyNRule() {
        return hp.traits().graphFamily() == Arch.BERT ? hp.moeEveryN() : 0;
    }

    // ---- router ----

    private GatingFunc gating() {
        GatingFunc g = hp.gating();
        return g != null ? g : hp.traits().defaultGating();
    }

    private MoeSpec.SelectionMode selectionMode() {
        switch (hp.arch()) {
            case LLAMA4:
                return MoeSpec.SelectionMode.RAW_LOGITS;
            case GROVEMOE:
                return MoeSpec.SelectionMode.SIGMOID_LOGITS;
            default:
                return MoeSpec.SelectionMode.PROBS;
        }
    }

    /** {@code exp_probs_b.bias}, or the suffix-less {@code exp_probs_b} of hy_v3 (hy-v3.cpp:59). */
    private boolean hasProbBias(int il) {
        return tc.hasLayer(il, TensorNames.bias(TensorNames.FFN_EXP_PROBS_B))
                || tc.hasLayer(il, TensorNames.FFN_EXP_PROBS_B);
    }

    /** deepseek4 layers below {@code hash_layer_count} take ids from {@code ffn_gate_tid2eid[token]} (deepseek4.cpp:1297-1299). */
    private boolean hashRouted(int il) {
        if (hp.arch() != Arch.DEEPSEEK4 || il >= hp.dsv4HashLayerCount()) {
            return false;
        }
        tc.require(il, TensorNames.weight(TensorNames.FFN_GATE_TID2EID));
        return true;
    }

    // ---- experts ----

    /** {@code llm_ffn_op_type} each graph passes to {@code build_moe_ffn}. */
    private FfnResolver.Op expertOpType() {
        switch (hp.arch()) {
            case KIMI_K3:
                return FfnResolver.Op.SITU;                 // kimi-k3.cpp build_latent_moe
            case OPENAI_MOE:
            case MINIMAX_M3:
                return FfnResolver.Op.SWIGLU_OAI;           // openai-moe.cpp:139, minimax-m3.cpp
            case NEMOTRON_H:
            case NEMOTRON_H_MOE:
                return FfnResolver.Op.RELU_SQR;             // nemotron-h.cpp, nemotron-h-moe.cpp
            case SMALLTHINKER:
                return FfnResolver.Op.RELU;                 // smallthinker.cpp:156
            case GEMMA4:
            case GROK:
                return FfnResolver.Op.GELU;                 // gemma4.cpp:338, grok.cpp:165
            default:
                return hp.traits().graphFamily() == Arch.BERT ? FfnResolver.Op.GELU : FfnResolver.Op.SILU;
        }
    }

    private void expertOp(MoeSpec.Builder b, int il, TensorCatalog.GateUpExps gu) {
        FfnResolver.Op op = expertOpType();
        if (!gu.gated()) {
            if (op == FfnResolver.Op.SWIGLU_OAI || op == FfnResolver.Op.SITU) {
                throw new IllegalStateException(arch() + ": layer " + il + " " + op + " experts need ffn_gate_exps");
            }
            b.act(FfnResolver.actOp(arch(), op, il));
            return;
        }
        float limit = gu.fused() ? 0f : hp.swigluClampExp(il);
        FfnResolver.GluWithParams g = FfnResolver.parallelGlu(hp, op, il, limit, expertClampArch());
        b.glu(g.op).params(g.params).fusedGateUp(gu.fused());
    }

    /** {@code ggml_swiglu_clamp} archs of the expert clamp (llama-graph.cpp:2429); the others use the limited form. */
    private boolean expertClampArch() {
        switch (hp.arch()) {
            case MAPLE:
            case DEEPSEEK4:
            case GLM5_NEXT:
            case HY_V4:
                return true;
            case DFLASH:
                return hp.dsv4HcMult() > 0;
            default:
                return false;
        }
    }

    /**
     * {@code ffn_{gate,up,down}_exps.scale}: one value per expert (llama-model.cpp:1764-1773). The fused gate_up
     * path applies {@code ffn_up_exps.scale} to the whole product (llama-graph.cpp:2372).
     */
    private boolean expertScales(int il, TensorCatalog.GateUpExps gu, long nExpert) {
        TensorRef[] scales = {gu.gateScale(), gu.upScale(), tc.layer(il, TensorNames.scale(TensorNames.FFN_DOWN_EXPS))};
        boolean any = false;
        for (TensorRef s : scales) {
            if (s == null) {
                continue;
            }
            if (s.numElements() != nExpert) {
                throw new IllegalStateException(arch() + ": " + s.name() + " holds " + s.numElements()
                        + " scales for " + nExpert + " experts");
            }
            any = true;
        }
        return any;
    }

    /**
     * Latent width of experts run between {@code ffn_latent_down/up} (muse-spark, maion-coder, nemotron-h) or
     * {@code ffn_routed_down/up} (kimi-k3); 0 without a latent projection.
     */
    private int latentSize(int il) {
        String[][] pairs = {
                {TensorNames.FFN_LATENT_DOWN, TensorNames.FFN_LATENT_UP},
                {TensorNames.FFN_ROUTED_DOWN, TensorNames.FFN_ROUTED_UP},
        };
        for (String[] p : pairs) {
            TensorRef downProj = tc.layer(il, TensorNames.weight(p[0]));
            if (downProj == null) {
                continue;
            }
            TensorRef upProj = tc.require(il, TensorNames.weight(p[1]));
            long latent = downProj.ne(1);
            if (upProj.ne(0) != latent) {
                throw new IllegalStateException(arch() + ": " + upProj.name() + " reads " + upProj.ne(0)
                        + " latent values but " + downProj.name() + " writes " + latent);
            }
            return toInt(latent, il);
        }
        return 0;
    }

    /**
     * grovemoe chunk experts: a second pass over {@code n_expert / n_group_experts} chunk experts whose selected ids
     * are divided by {@code n_group_experts}, added with {@code expert_group_scale} (grovemoe.cpp:50-59, 137-166).
     */
    private void chunks(MoeSpec.Builder b, int il, long nExpert) {
        int idDiv = hp.nGroupExperts();
        if (idDiv <= 0 || nExpert % idDiv != 0) {
            throw new IllegalStateException(arch() + ": experts_per_group " + idDiv + " must divide " + nExpert
                    + " experts");
        }
        TensorRef upCh = tc.require(il, TensorNames.weight(TensorNames.FFN_UP_CHEXPS));
        tc.require(il, TensorNames.weight(TensorNames.FFN_GATE_CHEXPS));
        tc.require(il, TensorNames.weight(TensorNames.FFN_DOWN_CHEXPS));
        int nChunk = (int) (nExpert / idDiv);
        if (upCh.ne(2) != nChunk) {
            throw new IllegalStateException(arch() + ": " + upCh.name() + " holds " + upCh.ne(2) + " chunk experts, expected "
                    + nChunk);
        }
        b.chunks(idDiv, nChunk, toInt(upCh.ne(1), il), hp.expertGroupScale());
    }

    /**
     * Shared experts {@code ffn_*_shexp}, gated by {@code sigmoid(ffn_gate_inp_shexp . x)} when that tensor exists
     * (qwen2moe, qwen3next). Their op is SiTU for kimi-k3, SwiGLU-OAI for minimax-m3, ungated squared ReLU for
     * nemotron-h, otherwise SiLU (fused when {@code ffn_up_shexp} packs the gate, granite-swa). cohere2moe halves the
     * routed plus shared sum (models/cohere2moe.cpp:242-243, 396-397). mistral3 has no shared expert: its loader
     * never sizes one ({@code n_ff_shexp} is not read, models/mistral3.cpp:3-29) and its graph never builds one
     * (:249-266).
     */
    private void sharedExpert(MoeSpec.Builder b, int il) {
        if (hp.arch() == Arch.MISTRAL3) {
            return;
        }
        if (!tc.hasLayer(il, TensorNames.weight(TensorNames.FFN_UP_SHEXP))
                && !tc.hasLayer(il, TensorNames.weight(TensorNames.FFN_DOWN_SHEXP))) {
            return;
        }
        FfnResolver.Op op;
        switch (hp.arch()) {
            case KIMI_K3:
                op = FfnResolver.Op.SITU;
                break;
            case MINIMAX_M3:
                op = FfnResolver.Op.SWIGLU_OAI;
                break;
            case NEMOTRON_H:
            case NEMOTRON_H_MOE:
                op = FfnResolver.Op.RELU_SQR;
                break;
            default:
                op = FfnResolver.Op.SILU;
                break;
        }
        FfnSpec shexp = FfnResolver.buildFfn(hp, tc, FfnSpec.Role.SHARED_EXPERT, TensorNames.STACK_BLK, il, op,
                FfnGating.PAR, 0f);
        b.sharedExpert(shexp, tc.hasLayer(il, TensorNames.weight(TensorNames.FFN_GATE_INP_SHEXP)));
        if (hp.arch() == Arch.COHERE2MOE) {
            b.sharedCombineScale(COHERE2MOE_SHARED_COMBINE_SCALE);
        }
    }

    private void checkShapes(int il, TensorRef gateInp, TensorRef down, long nExpert, long nFfExp) {
        if (gateInp.ne(1) != nExpert) {
            throw new IllegalStateException(arch() + ": " + gateInp.name() + " routes " + gateInp.ne(1)
                    + " experts but the expert tensors hold " + nExpert);
        }
        if (down.ne(0) != nFfExp || down.ne(2) != nExpert) {
            throw new IllegalStateException(arch() + ": " + down.name() + " is " + down.ne(0) + " x " + down.ne(1) + " x "
                    + down.ne(2) + ", expected n_ff_exp " + nFfExp + " and " + nExpert + " experts");
        }
    }

    private void checkLayer(int il) {
        if (il < 0 || il >= hp.nLayerAll()) {
            throw new IndexOutOfBoundsException("layer " + il + " outside [0, " + hp.nLayerAll() + ")");
        }
    }

    private static int toInt(long v, int il) {
        return FfnResolver.toInt(v, il);
    }

    private String arch() {
        return hp.arch().ggufName();
    }
}
