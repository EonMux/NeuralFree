package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.arch.KdaGateMode;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.hooks.HybridHooksLightning;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * Resolves the recurrent mixer of a layer ({@link ModelHparams#isRecurrent(int)}) into a {@link RecurrentSpec}: the
 * kind follows the architecture's graph, the dimensions its hparams and the variant flags the tensors present.
 *
 * <p>The resolved spec's per-sequence state sizes must equal the hparams' {@code n_embd_r} / {@code n_embd_s}
 * ({@link ModelHparams#nEmbdR()}, {@link ModelHparams#nEmbdS()}), which size the recurrent cache rows
 * ({@code llama_memory_recurrent}, llama-memory-recurrent.cpp); PLaMo-2 is exempt from the conv check because it
 * convolves only the {@code d_inner} branch while its cache rows keep the generic Mamba-2 width. Immutable.
 */
public final class RecurrentResolver {

    /** PLaMo-2 {@code dt_dim = max(64, n_embd / 16)} (plamo2.cpp:43). */
    private static final int PLAMO2_MIN_DT_RANK = 64;
    private static final int PLAMO2_DT_RANK_DIV = 16;

    private final ModelHparams hp;
    private final TensorCatalog tc;

    public RecurrentResolver(ModelHparams hp, TensorCatalog tc) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.tc = Objects.requireNonNull(tc, "tc");
    }

    /** True when layer {@code il} runs a recurrent mixer. */
    public boolean isRecurrentLayer(int il) {
        checkLayer(il);
        return hp.isRecurrent(il);
    }

    /**
     * Recurrent mixer kind of this architecture's recurrent layers. A non-causal LFM2 runs the centered short
     * convolution: the decision layers by construction (lfm2.cpp:355-381), the other non-causal models because
     * {@code build_lfm_shortconv}'s non-causal form (lfm-base.cpp:49-56) pads the window on the left with the last
     * {@code (L-1)/2} columns of a conv state that is only ever written from the zero right padding, so both pads are
     * zero and no state is carried.
     */
    public MixerKind kind() {
        switch (hp.arch()) {
            case MAMBA:
            case JAMBA:
                return MixerKind.MAMBA1;
            case MAMBA2:
            case FALCON_H1:
            case GRANITE_HYBRID:
            case NEMOTRON_H:
            case NEMOTRON_H_MOE:
                return MixerKind.MAMBA2;
            case PLAMO2:
                return MixerKind.PLAMO2_MAMBA;
            case LFM2:
            case LFM2MOE:
                return hp.causal() ? MixerKind.SHORTCONV : MixerKind.SHORTCONV_CENTERED;
            case RWKV6:
                return MixerKind.RWKV6;
            case RWKV6QWEN2:
                return MixerKind.RWKV6_QWEN2;
            case RWKV7:
            case ARWKV7:
                return MixerKind.RWKV7;
            case QWEN3NEXT:
            case QWEN35:
            case QWEN35MOE:
            case QWEN4EXP:
            case CLEF:
                return MixerKind.GDN;
            case KIMI_LINEAR:
            case KIMI_K3:
            case BAILINGMOE3:
            case GLM5_NEXT:
                return MixerKind.KDA;
            case MINIMAX_01:
                return MixerKind.LIGHTNING;
            default:
                throw new IllegalStateException(arch() + ": architecture has no recurrent mixer");
        }
    }

    /**
     * Recurrent mixer of layer {@code il}.
     *
     * @throws IllegalArgumentException when the layer is not recurrent
     * @throws MissingTensorException   when a tensor that selects the mixer variant is missing
     * @throws IllegalStateException    when the spec's state sizes disagree with {@code n_embd_r} / {@code n_embd_s}
     */
    public RecurrentSpec resolve(int il) {
        if (!isRecurrentLayer(il)) {
            throw new IllegalArgumentException(arch() + ": layer " + il + " is not recurrent");
        }
        MixerKind kind = kind();
        RecurrentSpec.Builder b = RecurrentSpec.builder(kind).nEmbd(hp.nEmbd());
        switch (kind) {
            case MAMBA1:
                mamba1(b, il);
                break;
            case MAMBA2:
                mamba2(b, il);
                break;
            case PLAMO2_MAMBA:
                plamo2(b, il);
                break;
            case SHORTCONV:
            case SHORTCONV_CENTERED:
                tc.require(il, TensorNames.weight(TensorNames.SHORTCONV_CONV));
                b.lCache(hp.shortconvLCache());
                break;
            case RWKV6:
            case RWKV6_QWEN2:
            case RWKV7:
                rwkv(b, il, kind);
                break;
            case GDN:
                gdn(b, il);
                break;
            case KDA:
                kda(b, il);
                break;
            case LIGHTNING:
                lightning(b, il);
                break;
            default:
                throw new IllegalStateException(arch() + ": unhandled recurrent kind " + kind);
        }
        RecurrentSpec spec = b.build();
        checkStateRows(spec, il);
        return spec;
    }

    /** Mamba-1 selective scan (mamba.cpp, jamba.cpp; llama-graph.cpp build_mamba_layer). */
    private void mamba1(RecurrentSpec.Builder b, int il) {
        b.dConv(hp.ssmDConv()).dInner(hp.ssmDInner()).dState(hp.ssmDState()).dtRank(hp.ssmDtRank())
                .nGroup(hp.ssmNGroup()).dtBcRms(hp.ssmDtBcRms()).convBias(hasConvBias(il));
    }

    /** Mamba-2 SSD with {@code n_head = ssm_dt_rank} heads of {@code d_inner / n_head} (mamba2-base). */
    private void mamba2(RecurrentSpec.Builder b, int il) {
        int nHead = hp.ssmDtRank();
        b.dConv(hp.ssmDConv()).dInner(hp.ssmDInner()).dState(hp.ssmDState()).nGroup(hp.ssmNGroup())
                .nHead(nHead).headDim(divide(hp.ssmDInner(), nHead, "ssm.inner_size", "ssm.time_step_rank heads"))
                .convBias(hasConvBias(il));
    }

    /** PLaMo-2 Mamba with {@code ssm_dt_rank} heads and a {@code dt_dim}-wide dt projection (plamo2.cpp:41-43, 268-340). */
    private void plamo2(RecurrentSpec.Builder b, int il) {
        int nHead = hp.ssmDtRank();
        int dtRank = Math.max(PLAMO2_MIN_DT_RANK, hp.nEmbd() / PLAMO2_DT_RANK_DIV);
        TensorRef dt = tc.layer(il, TensorNames.weight(TensorNames.SSM_DT));
        if (dt != null && dt.ne(0) != dtRank) {
            throw new IllegalStateException(arch() + ": " + dt.name() + " reads " + dt.ne(0) + " values, expected dt_dim "
                    + dtRank);
        }
        b.dConv(hp.ssmDConv()).dInner(hp.ssmDInner()).dState(hp.ssmDState()).nGroup(0).nHead(nHead)
                .headDim(divide(hp.ssmDInner(), nHead, "ssm.inner_size", "ssm.time_step_rank heads")).dtRank(dtRank)
                .convBias(hasConvBias(il));
    }

    /**
     * RWKV6/6-qwen2/7 time mix. QRWKV (RWKV6-qwen2 converts) has no {@code time_mix_first} bonus and runs the GLA
     * pre-step instead (rwkv6-base.cpp:50, 139; rwkv6qwen2.cpp:57).
     */
    private void rwkv(RecurrentSpec.Builder b, int il, MixerKind kind) {
        boolean qrwkv = kind != MixerKind.RWKV7 && !tc.hasLayer(il, TensorNames.weight(TensorNames.TIME_MIX_FIRST));
        b.wkvHeadSize(hp.wkvHeadSize()).tokenShiftCount(hp.tokenShiftCount()).qrwkv(qrwkv)
                .rescaleEveryN(hp.rescaleEveryN()).timeMixExtraDim(hp.timeMixExtraDim())
                .timeDecayExtraDim(hp.timeDecayExtraDim());
        if (kind == MixerKind.RWKV7) {
            b.rwkv7Lora(hp.nLoraDecay(), hp.nLoraIclr(), hp.nLoraValueResMix(), hp.nLoraGate());
        }
    }

    /**
     * Gated DeltaNet: K heads {@code ssm.group_count}, V heads {@code ssm.time_step_rank}, K head dim
     * {@code ssm.state_size}, V head dim {@code ssm.inner_size / V heads} (qwen3next.cpp:505-515,
     * qwen35.cpp:355-360). The head map comes from the hparams hook (qwen3next GROUPED, qwen35 TILED; ADD A4).
     * Qwen3-Next checkpoints without {@code attn_qkv} project through the legacy {@code ssm_in}
     * (qwen3next.cpp:85-87, 322-334); {@code ssm_ba} fuses the beta/alpha projections.
     */
    private void gdn(RecurrentSpec.Builder b, int il) {
        int nK = hp.ssmNGroup();
        int nV = hp.ssmDtRank();
        GdnHeadMap map = hp.gdnHeadMap();
        if (map == null) {
            throw new IllegalStateException(arch() + ": gated DeltaNet needs a head map from the hparams");
        }
        boolean legacy = false;
        if (!tc.hasLayer(il, TensorNames.weight(TensorNames.ATTN_QKV))) {
            if (hp.arch() != Arch.QWEN3NEXT) {
                tc.require(il, TensorNames.weight(TensorNames.ATTN_QKV));
            }
            tc.require(il, TensorNames.weight(TensorNames.SSM_IN));
            legacy = true;
        }
        b.dConv(hp.ssmDConv())
                .gdnHeads(nK, nV, hp.ssmDState(), divide(hp.ssmDInner(), nV, "ssm.inner_size", "V heads"))
                .headMap(map)
                .fusedBetaAlpha(tc.hasLayer(il, TensorNames.weight(TensorNames.SSM_BETA_ALPHA)))
                .legacySsmIn(legacy);
    }

    /**
     * Kimi delta attention: the decay gate is {@code softplus} for kimi-linear
     * (kimi-linear.cpp), {@code lb * sigmoid(-(a (f + dt_b)))} for kimi-k3 / glm5-next with a finite lower bound
     * (kimi-k3.cpp:418-436) and {@code lb * sigmoid(a (f + dt_b))} for bailingmoe3 (bailingmoe3.cpp:270-276). The
     * gate projection form comes from the hparams: low-rank {@code ssm_g_a . ssm_g_b}, full {@code ssm_g}, or full
     * {@code ssm_g_a}.
     */
    private void kda(RecurrentSpec.Builder b, int il) {
        KdaGateMode mode = hp.kdaGateMode();
        if (mode == null) {
            throw new IllegalStateException(arch() + ": KDA needs a gate mode from the hparams");
        }
        switch (mode) {
            case LOWRANK:
                tc.require(il, TensorNames.weight(TensorNames.SSM_G_A));
                tc.require(il, TensorNames.weight(TensorNames.SSM_G_B));
                break;
            case FULL_SSM_G:
                tc.require(il, TensorNames.weight(TensorNames.SSM_G));
                break;
            case FULL_SSM_G_A:
                tc.require(il, TensorNames.weight(TensorNames.SSM_G_A));
                break;
            default:
                throw new IllegalStateException(arch() + ": unhandled KDA gate mode " + mode);
        }
        float lb = hp.kdaGateLowerBound();
        b.nHead(hp.nHead(0)).headDim(hp.kdaHeadDim()).dConv(hp.ssmDConv()).kdaGateMode(mode)
                .decayGate(decayGate(lb)).safeGate(hp.kdaSafeGate()).gateLowerBound(lb);
    }

    private RecurrentSpec.DecayGate decayGate(float lb) {
        switch (hp.arch()) {
            case KIMI_LINEAR:
                return RecurrentSpec.DecayGate.SOFTPLUS;
            case KIMI_K3:
            case GLM5_NEXT:
                return lb > Float.NEGATIVE_INFINITY ? RecurrentSpec.DecayGate.LB_NEG : RecurrentSpec.DecayGate.SOFTPLUS;
            case BAILINGMOE3:
                return RecurrentSpec.DecayGate.LB_POS;
            default:
                throw new IllegalStateException(arch() + ": no KDA decay gate rule for this architecture");
        }
    }

    /** Lightning attention: per-head slopes {@code base[h] * (1 - il / (n_layer - 1) + 1e-5)} (minimax-01.cpp:98-103, 288). */
    private void lightning(RecurrentSpec.Builder b, int il) {
        float[] base = HybridHooksLightning.lightningSlopes(hp);
        float scale = HybridHooksLightning.lightningSlopeScale(hp, il);
        float[] slopes = new float[base.length];
        for (int h = 0; h < base.length; h++) {
            slopes[h] = base[h] * scale;
        }
        b.nHead(hp.nHead(0)).headDim(hp.headLinearAttn()).slopes(slopes);
    }

    private void checkStateRows(RecurrentSpec spec, int il) {
        if (spec.kind() != MixerKind.PLAMO2_MAMBA && spec.convStateFloats() != hp.nEmbdR()) {
            throw new IllegalStateException(arch() + ": layer " + il + " " + spec.kind() + " conv state holds "
                    + spec.convStateFloats() + " floats but n_embd_r is " + hp.nEmbdR());
        }
        if (spec.stateFloats() != hp.nEmbdS()) {
            throw new IllegalStateException(arch() + ": layer " + il + " " + spec.kind() + " state holds "
                    + spec.stateFloats() + " floats but n_embd_s is " + hp.nEmbdS());
        }
    }

    private boolean hasConvBias(int il) {
        return tc.hasLayer(il, TensorNames.bias(TensorNames.SSM_CONV1D));
    }

    private int divide(int value, int by, String what, String byWhat) {
        if (by <= 0 || value % by != 0) {
            throw new IllegalStateException(arch() + ": " + what + " " + value + " does not split into " + by + " "
                    + byWhat);
        }
        return value / by;
    }

    private void checkLayer(int il) {
        if (il < 0 || il >= hp.nLayerAll()) {
            throw new IndexOutOfBoundsException("layer " + il + " outside [0, " + hp.nLayerAll() + ")");
        }
    }

    private String arch() {
        return hp.arch().ggufName();
    }
}
