package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.gemma3n.CpuActivationSparsity;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * Resolves the feed-forward block of one layer into an {@link FfnKind} with its {@link FfnSpec} and/or
 * {@link MoeSpec}. Tensor presence is authoritative: {@code channel_mix_key} selects the RWKV channel mix,
 * {@code ffn_gate_inp} a routed MoE (via {@link MoeResolver}), {@code ffn_up}/{@code ffn_down} a dense FFN, and a
 * layer without any of them has no FFN ({@link FfnKind#NONE}, deci {@code n_ff == 0}, Nemotron-H mixer layers).
 *
 * <p>The dense FFN mirrors llama.cpp {@code build_ffn} (src/llama-graph.cpp:1922-2140):
 * <ul>
 *   <li>a separate gate with {@code LLM_FFN_PAR} becomes a GLU: SILU swiglu (clamped by
 *       {@code swiglu_clamp_shexp[il]} when above {@code 1e-6}: {@code ggml_swiglu_clamp} for deepseek4, glm5-next and
 *       hyper-connected dflash, the limited form otherwise, :1999-2026), GELU geglu, GELU_ERF geglu-erf, RELU reglu,
 *       SWIGLU_OAI_MOE ({@code alpha 1.702, limit 7}, :2080-2091);</li>
 *   <li>a gate with {@code LLM_FFN_SEQ} is {@code act(gate(up(x)))};</li>
 *   <li>no gate: the fused {@code LLM_FFN_SWIGLU/GEGLU/GEGLU_ERF/REGLU} ops split {@code ffn_up} into
 *       {@code [gate | up]} halves (llama-graph.cpp:2071-2105); any other op is the ungated {@code down(act(up(x)))}. An
 *       {@code ffn_up} of {@code 2 * n_ff} rows without a gate is the fused form (jina-bert-v2, models/bert.cpp:194).</li>
 * </ul>
 * {@code n_ff} is the {@code ffn_down} input width. Per-tensor {@code *.scale} weights must hold one value
 * (llama-model.cpp:1744-1762); they are applied with the weights and are not part of the spec. Immutable.
 */
public final class FfnResolver {

    /** Clamp limits at or below this are disabled (llama-graph.cpp:2003, 2427). */
    static final float CLAMP_EPS = 1e-6f;
    /** {@code LLM_FFN_SWIGLU_OAI_MOE} constants (llama-graph.cpp:2083-2084, 2479-2480). */
    static final float OAI_ALPHA = 1.702f;
    static final float OAI_LIMIT = 7.0f;

    /** llama.cpp {@code llm_ffn_op_type} plus the custom kimi-k3 SiTU and apertus xIELU FFNs. */
    enum Op {
        SILU, GELU, GELU_ERF, RELU, RELU_SQR, XIELU, SWIGLU_OAI, SITU,
        /** Fused forms: one {@code ffn_up} of {@code [gate | up]} rows, no separate gate. */
        SWIGLU, GEGLU, GEGLU_ERF, REGLU;

        boolean fused() {
            return this == SWIGLU || this == GEGLU || this == GEGLU_ERF || this == REGLU;
        }
    }

    /** FFN facet of one layer. */
    public static final class Result {
        private final FfnKind kind;
        private final FfnSpec ffn;
        private final MoeSpec moe;

        Result(FfnKind kind, FfnSpec ffn, MoeSpec moe) {
            this.kind = Objects.requireNonNull(kind, "kind");
            this.ffn = ffn;
            this.moe = moe;
        }

        public FfnKind kind() { return kind; }
        /** Dense, parallel-dense or channel-mix FFN; null for plain MoE and {@link FfnKind#NONE}. */
        public FfnSpec ffn() { return ffn; }
        /** Routed experts; null unless the kind is a MoE kind. */
        public MoeSpec moe() { return moe; }

        @Override
        public String toString() {
            return "FfnResolver.Result{" + kind + ", ffn=" + ffn + ", moe=" + moe + '}';
        }
    }

    private final ModelHparams hp;
    private final TensorCatalog tc;
    private final MoeResolver moe;

    public FfnResolver(ModelHparams hp, TensorCatalog tc) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.tc = Objects.requireNonNull(tc, "tc");
        this.moe = new MoeResolver(hp, tc);
    }

    /**
     * FFN of main-stack layer {@code il} (trunk or NextN block).
     *
     * @throws MissingTensorException when a partially present FFN lacks a tensor
     * @throws IllegalStateException   for an op/tensor combination llama.cpp does not build
     */
    public Result resolve(int il) {
        checkLayer(il);
        if (tc.hasLayer(il, TensorNames.weight(TensorNames.CHANNEL_MIX_KEY))) {
            return new Result(FfnKind.RWKV_CHANNEL_MIX, channelMix(il), null);
        }
        if (moe.isMoeLayer(il)) {
            return moeLayer(il);
        }
        requireRouterForExperts(il);
        return dense(TensorNames.STACK_BLK, il);
    }

    /**
     * Every llama.cpp loader that creates routed-expert tensors also creates {@code ffn_gate_inp}; expert weights
     * without the router mean a damaged file, never a dense or FFN-less layer.
     *
     * @throws MissingTensorException naming {@code ffn_gate_inp} when expert tensors exist without it
     */
    private void requireRouterForExperts(int il) {
        String[] experts = {TensorNames.FFN_GATE_UP_EXPS, TensorNames.FFN_GATE_EXPS, TensorNames.FFN_UP_EXPS,
                TensorNames.FFN_DOWN_EXPS};
        for (String base : experts) {
            if (tc.hasLayer(il, TensorNames.weight(base))) {
                throw new MissingTensorException(tc.archName(), il,
                        TensorNames.layerName(il, TensorNames.weight(TensorNames.FFN_GATE_INP)));
            }
        }
    }

    /**
     * Dense FFN of a layer of any stack ({@link TensorNames#STACK_ENC}, {@link TensorNames#STACK_DEC}, ...); the main
     * stack delegates to {@link #resolve(int)}.
     *
     * @throws IllegalStateException when a non-main stack carries routed experts
     */
    public Result resolve(String stack, int il) {
        Objects.requireNonNull(stack, "stack");
        if (TensorNames.STACK_BLK.equals(stack)) {
            return resolve(il);
        }
        if (il < 0) {
            throw new IndexOutOfBoundsException("layer " + il + " of stack " + stack);
        }
        if (tc.layer(stack, il, TensorNames.weight(TensorNames.FFN_GATE_INP)) != null) {
            throw new IllegalStateException(arch() + ": routed experts in the " + stack + " stack (layer " + il
                    + ") are not built by llama.cpp");
        }
        return dense(stack, il);
    }

    private Result moeLayer(int il) {
        MoeSpec spec = moe.resolve(il);
        if (parallelDense(il)) {
            FfnSpec ffn = buildFfn(hp, tc, FfnSpec.Role.PARALLEL_DENSE, TensorNames.STACK_BLK, il,
                    denseOp(true), FfnGating.PAR, 0f);
            return new Result(FfnKind.MOE_PARALLEL_DENSE, ffn, spec);
        }
        FfnKind kind = spec.latent() ? FfnKind.MOE_LATENT : spec.chunked() ? FfnKind.MOE_CHUNKED : FfnKind.MOE;
        return new Result(kind, null, spec);
    }

    /**
     * A dense FFN summed with the experts: arctic's residual MLP (models/arctic.cpp:124-150), gemma4's
     * {@code ffn_mlp + ffn_moe} (gemma4.cpp:299-351), grok-2's {@code (ffn + moe) / sqrt(2)} when {@code ffn_up}
     * exists (grok.cpp:171-180). Other architectures never read dense FFN tensors on MoE layers.
     */
    private boolean parallelDense(int il) {
        switch (hp.arch()) {
            case ARCTIC:
            case GEMMA4:
                return true;
            case GROK:
                return tc.hasLayer(il, TensorNames.weight(TensorNames.FFN_UP));
            default:
                return false;
        }
    }

    private Result dense(String stack, int il) {
        TensorRef down = tc.layer(stack, il, TensorNames.weight(TensorNames.FFN_DOWN));
        TensorRef up = tc.layer(stack, il, TensorNames.weight(TensorNames.FFN_UP));
        TensorRef gate = tc.layer(stack, il, TensorNames.weight(TensorNames.FFN_GATE));
        if (down == null && up == null && gate == null) {
            return new Result(FfnKind.NONE, null, null);
        }
        boolean gated = gate != null;
        boolean decision = TensorNames.STACK_BLK.equals(stack) && isDecisionLayer(il);
        Op op = decision ? Op.RELU : denseOp(gated);
        FfnGating gating = decision ? FfnGating.SEQ : denseGating(gated);
        float sparsity = hp.arch() == Arch.GEMMA3N && il < CpuActivationSparsity.GEMMA3N_SPARSE_LAYERS
                ? CpuActivationSparsity.GEMMA3N_STD_MUL : 0f;
        return new Result(FfnKind.DENSE, buildFfn(hp, tc, FfnSpec.Role.DENSE, stack, il, op, gating, sparsity), null);
    }

    /** RWKV6/7 channel mix: {@code value(relu(key(xk))^2)}, gated by the RWKV6 receptance (rwkv6-base.cpp, rwkv7-base.cpp). */
    private FfnSpec channelMix(int il) {
        TensorRef key = tc.require(il, TensorNames.weight(TensorNames.CHANNEL_MIX_KEY));
        TensorRef value = tc.require(il, TensorNames.weight(TensorNames.CHANNEL_MIX_VALUE));
        long nFf = value.ne(0);
        if (key.ne(1) != nFf) {
            throw new IllegalStateException(arch() + ": layer " + il + " channel_mix_key has " + key.ne(1)
                    + " rows but channel_mix_value reads " + nFf);
        }
        return FfnSpec.builder(FfnSpec.Role.RWKV_CHANNEL_MIX).nFf(toInt(nFf, il)).act(false, ActOp.RELU_SQR).build();
    }

    /**
     * {@code llm_ffn_op_type} of a dense FFN outside the decision head (also the arctic/gemma4/grok parallel FFN).
     * Architectures whose traits pack the gate in {@code ffn_up} get the fused op (phi3, chatglm, glm4, plamo2/3,
     * neo-bert: {@code LLM_FFN_SWIGLU, LLM_FFN_SEQ}).
     */
    private Op denseOp(boolean gated) {
        switch (hp.arch()) {
            case KIMI_K3:
                return Op.SITU;                                // kimi-k3.cpp:315-319
            case MINIMAX_M3:
                return Op.SWIGLU_OAI;                          // minimax-m3.cpp:550-560
            case MODERN_BERT:
                return modernBertOp();
            case DFLASH:
                return dflashOp();
            case T5:
            case T5ENCODER:
                return gated ? Op.GELU : Op.RELU;              // t5.cpp:226, 326
            default:
                Op op = traitsOp();
                return hp.traits().ffnGating() == FfnGating.FUSED_UP ? fusedOf(op) : op;
        }
    }

    /** Gate type the graph passes: SEQ for modern-bert and fused-up archs, PAR/SEQ by gate for T5, else the traits'. */
    private FfnGating denseGating(boolean gated) {
        if (isT5()) {
            return gated ? FfnGating.PAR : FfnGating.SEQ;
        }
        FfnGating g = hp.traits().ffnGating();
        return g == FfnGating.FUSED_UP || hp.arch() == Arch.MODERN_BERT ? FfnGating.SEQ : g;
    }

    private boolean isT5() {
        return hp.arch() == Arch.T5 || hp.arch() == Arch.T5ENCODER;
    }

    /**
     * Trunk layers of the modern-bert / lfm2 decision head ({@code il >= n_layer - n_layer_decision}), whose FFN is
     * the ungated ReLU MLP (modern-bert.cpp:288-294, lfm2.cpp:462).
     */
    private boolean isDecisionLayer(int il) {
        int nDec = hp.nLayerDecision();
        return nDec > 0 && il >= hp.nLayer() - nDec && il < hp.nLayer();
    }

    private Op traitsOp() {
        ActOp a = hp.traits().ffnAct();
        switch (a) {
            case SILU: return Op.SILU;
            case GELU_TANH: return Op.GELU;
            case GELU_ERF: return Op.GELU_ERF;
            case RELU: return Op.RELU;
            case RELU_SQR: return Op.RELU_SQR;
            case XIELU: return Op.XIELU;
            default:
                throw new IllegalStateException(arch() + ": FFN activation " + a + " has no llama.cpp FFN op");
        }
    }

    /** {@code LLM_FFN_OP_TYPES_FROM_STRING} with the GEGLU-erf fallback (llama-model.cpp:1109-1138, modern-bert.cpp:20-24). */
    private Op modernBertOp() {
        String act = hp.hiddenActivation();
        if (act == null) {
            return Op.GEGLU_ERF;
        }
        switch (act) {
            case "gelu":
            case "gelu_python":
                return Op.GEGLU_ERF;
            case "gelu_pytorch_tanh":
            case "gelu_new":
            case "gelu_fast":
            case "gelu_accurate":
            case "gelu_python_tanh":
            case "geglu":
                return Op.GEGLU;
            case "silu":
            case "swish":
            case "swiglu":
                return Op.SWIGLU;
            case "relu":
                return Op.RELU;
            case "reglu":
                return Op.REGLU;
            default:
                return Op.GEGLU_ERF;
        }
    }

    /** dflash.cpp:13-21: SILU unless {@code hidden_act} is gelu / gelu_pytorch_tanh. */
    private Op dflashOp() {
        String act = hp.hiddenActivation();
        if (act == null || "silu".equals(act)) {
            return Op.SILU;
        }
        if ("gelu".equals(act) || "gelu_pytorch_tanh".equals(act)) {
            return Op.GELU;
        }
        throw new IllegalStateException(arch() + ": unsupported DFlash hidden activation '" + act + "'");
    }

    /** The fused single-tensor op of a GLU-able op ({@code LLM_FFN_SWIGLU} for SILU, ...). */
    private Op fusedOf(Op op) {
        switch (op) {
            case SILU: return Op.SWIGLU;
            case GELU: return Op.GEGLU;
            case GELU_ERF: return Op.GEGLU_ERF;
            case RELU: return Op.REGLU;
            default:
                throw new IllegalStateException(arch() + ": FFN op " + op + " has no fused gate/up form");
        }
    }

    private void checkLayer(int il) {
        if (il < 0 || il >= hp.nLayerAll()) {
            throw new IndexOutOfBoundsException("layer " + il + " outside [0, " + hp.nLayerAll() + ")");
        }
    }

    private String arch() {
        return hp.arch().ggufName();
    }

    // ---- shared with MoeResolver ----

    /**
     * Builds a dense-form FFN ({@code build_ffn}) of {@code role} for layer {@code il} of {@code stack}.
     *
     * @param op       llama.cpp FFN op
     * @param gateType {@link FfnGating#PAR} or {@link FfnGating#SEQ} as the graph passes it
     * @param sparsity gemma3n Gaussian top-k multiplier, 0 for none
     * @throws MissingTensorException when the up or down projection is missing
     * @throws IllegalStateException   for shapes or ops llama.cpp does not build
     */
    static FfnSpec buildFfn(ModelHparams hp, TensorCatalog tc, FfnSpec.Role role, String stack, int il, Op op,
                            FfnGating gateType, float sparsity) {
        String arch = hp.arch().ggufName();
        TensorRef down = tc.require(stack, il, TensorNames.weight(role.down()));
        TensorRef up = tc.require(stack, il, TensorNames.weight(role.up()));
        boolean gated = tc.layer(stack, il, TensorNames.weight(role.gate())) != null;
        long nFf = down.ne(0);
        checkScales(tc, stack, il, role);
        FfnSpec.Builder b = FfnSpec.builder(role).nFf(toInt(nFf, il)).activationSparsityStdMul(sparsity)
                .biases(!UnusedBiases.ffnUnused(hp.arch()));
        boolean fusedShape = up.ne(1) == 2 * nFf;
        if (op.fused() || (!gated && fusedShape)) {
            if (gated) {
                throw new IllegalStateException(arch + ": layer " + il + " " + role + " FFN op " + op
                        + " packs the gate in " + up.name() + " but a separate gate exists");
            }
            if (!fusedShape) {
                throw new IllegalStateException(arch + ": layer " + il + " fused " + role + " FFN needs " + up.name()
                        + " with 2 * n_ff = " + 2 * nFf + " rows, got " + up.ne(1));
            }
            return b.glu(FfnGating.FUSED_UP, fusedGlu(arch, op, il)).build();
        }
        boolean seqGated = gated && gateType == FfnGating.SEQ;
        if (!seqGated && up.ne(1) != nFf) {
            throw new IllegalStateException(arch + ": layer " + il + " " + up.name() + " has " + up.ne(1)
                    + " rows but the down projection reads " + nFf);
        }
        if (!gated || seqGated) {
            if (op == Op.SWIGLU_OAI || op == Op.SITU) {
                throw new IllegalStateException(arch + ": layer " + il + " " + role + " FFN op " + op
                        + " needs a parallel gate");
            }
            ActParams params = op == Op.XIELU ? xieluParams(hp, il) : ActParams.NONE;
            return b.act(gated, actOp(arch, op, il)).params(params).build();
        }
        GluWithParams g = parallelGlu(hp, op, il, hp.swigluClampShexp(il), shexpClampArch(hp));
        return b.glu(FfnGating.PAR, g.op).params(g.params).build();
    }

    /** A GLU op with its parameters. */
    static final class GluWithParams {
        final GluOp op;
        final ActParams params;

        GluWithParams(GluOp op, ActParams params) {
            this.op = op;
            this.params = params;
        }
    }

    /**
     * GLU of a separate gate and up: {@code build_ffn} PAR (llama-graph.cpp:1999-2091) and the gated
     * {@code build_moe_ffn} ops (:2412-2490).
     *
     * @param limit         SwiGLU clamp; at or below {@link #CLAMP_EPS} disables it
     * @param clampVariant  true for {@code ggml_swiglu_clamp}, false for the limited form
     */
    static GluWithParams parallelGlu(ModelHparams hp, Op op, int il, float limit, boolean clampVariant) {
        switch (op) {
            case SILU:
                if (limit > CLAMP_EPS) {
                    return new GluWithParams(clampVariant ? GluOp.SWIGLU_CLAMP : GluOp.SWIGLU_LIMITED,
                            ActParams.builder().limit(limit).build());
                }
                return new GluWithParams(GluOp.SWIGLU, ActParams.NONE);
            case GELU:
                return new GluWithParams(GluOp.GEGLU_TANH, ActParams.NONE);
            case GELU_ERF:
                return new GluWithParams(GluOp.GEGLU_ERF, ActParams.NONE);
            case RELU:
                return new GluWithParams(GluOp.REGLU, ActParams.NONE);
            case SWIGLU_OAI:
                return new GluWithParams(GluOp.SWIGLU_OAI, ActParams.builder().alpha(OAI_ALPHA).limit(OAI_LIMIT).build());
            case SITU:
                return new GluWithParams(GluOp.SITU, situParams(hp));
            default:
                throw new IllegalStateException(hp.arch().ggufName() + ": layer " + il + " gated " + op
                        + " FFN is not built by llama.cpp");
        }
    }

    /**
     * SiTU {@code beta * tanh(g / beta) * sigmoid(g) * (lb > 0 ? lb * tanh(u / lb) : u)} with
     * {@code beta = situ_beta}, {@code lb = situ_linear_beta} (kimi-k3.cpp:173-184, llama-graph.cpp:2453-2467).
     */
    static ActParams situParams(ModelHparams hp) {
        return ActParams.builder().beta(hp.situBeta()).lambda(Math.max(0f, hp.situLinearBeta())).build();
    }

    /** {@code build_ffn} uses {@code ggml_swiglu_clamp} for deepseek4, glm5-next and dflash with hyper-connections (:2004). */
    static boolean shexpClampArch(ModelHparams hp) {
        switch (hp.arch()) {
            case DEEPSEEK4:
            case GLM5_NEXT:
                return true;
            case DFLASH:
                return hp.dsv4HcMult() > 0;
            default:
                return false;
        }
    }

    /** Unary activation of an ungated or SEQ-gated op. */
    static ActOp actOp(String arch, Op op, int il) {
        switch (op) {
            case SILU: return ActOp.SILU;
            case GELU: return ActOp.GELU_TANH;
            case GELU_ERF: return ActOp.GELU_ERF;
            case RELU: return ActOp.RELU;
            case RELU_SQR: return ActOp.RELU_SQR;
            case XIELU: return ActOp.XIELU;
            default:
                throw new IllegalStateException(arch + ": layer " + il + " FFN op " + op + " is not a unary activation");
        }
    }

    private static GluOp fusedGlu(String arch, Op op, int il) {
        switch (op) {
            case SILU: case SWIGLU: return GluOp.SWIGLU;
            case GELU: case GEGLU: return GluOp.GEGLU_TANH;
            case GELU_ERF: case GEGLU_ERF: return GluOp.GEGLU_ERF;
            case RELU: case REGLU: return GluOp.REGLU;
            default:
                throw new IllegalStateException(arch + ": layer " + il + " FFN op " + op + " has no fused gate/up form");
        }
    }

    /** Per-layer xIELU parameters, prepared as {@code ggml_xielu} does (apertus.cpp:132-138). */
    private static ActParams xieluParams(ModelHparams hp, int il) {
        return ActParams.builder().xieluRaw(hp.xieluAlphaN(il), hp.xieluAlphaP(il), hp.xieluBeta(il), hp.xieluEps(il))
                .build();
    }

    /** Dense and shared-expert per-tensor {@code *.scale} tensors hold one value (llama-model.cpp:1744-1762). */
    private static void checkScales(TensorCatalog tc, String stack, int il, FfnSpec.Role role) {
        String[] bases = {role.gate(), role.up(), role.down()};
        for (String base : bases) {
            TensorRef s = tc.layer(stack, il, TensorNames.scale(base));
            if (s != null && s.numElements() != 1) {
                throw new IllegalStateException("architecture '" + tc.archName() + "': " + s.name()
                        + " must hold one per-tensor scale, got " + s.numElements() + " values");
            }
        }
    }

    static int toInt(long v, int il) {
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new IllegalStateException("layer " + il + " FFN width " + v + " is out of range");
        }
        return (int) v;
    }
}
