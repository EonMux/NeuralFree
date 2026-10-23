package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * Resolves how each layer's branches join the residual stream into a {@link ResidualSpec}:
 * <ul>
 *   <li>kind: hyper-connections when {@code hc_mult > 1} (deepseek4, hy_v4, glm5-next, qwen4exp, hyper-connected
 *       dflash; replaced around both sub-blocks), except the glm5-next NextN blocks, which have no mHC mixers and add
 *       both branches plainly (glm5-next.cpp:85-92, 626, 656); otherwise the trait's kind, with gptneox's parallel
 *       residual taken from {@code use_parallel_residual} (models/gptneox.cpp:143), stablelm parallel when the layer
 *       has no {@code ffn_norm} (models/stablelm.cpp:129-138), and SCALED dropped when {@code residual_scale} is 0 (the
 *       granite graphs scale only when it is set, models/granite.cpp:248, 308);</li>
 *   <li>NORMED_INPUT_SCALED for minimax-01 ({@code x = branch + residual_scale * norm(x)},
 *       models/minimax-01.cpp:428-458);</li>
 *   <li>DEPTH_SCALED for beam: the {@code i}-th residual add of the stack is
 *       {@code branch * (1 / sqrt(i)) + x * sqrt((i - 1) / i)}, scales computed in double and rounded to float, with
 *       {@code i = 2 * (il + 1)} for the mixer and {@code i + 1} for the FFN (models/beam.cpp:109-113, 203-204,
 *       237);</li>
 *   <li>deci: {@code n_ff == 0} discards the layer output (the layer passes its input on,
 *       models/deci.cpp:145-148), {@code n_head == 0} has no mixer residual (:150-153);</li>
 *   <li>muse-spark residual alpha/beta tensors; the falcon-40b / RWKV second input norm {@code attn_norm_2}; talkie embedding
 *       skip; RWKV6 halving every {@code rescale_every_n_layers} layers (models/rwkv6.cpp:164-166).</li>
 * </ul>
 * Immutable.
 */
public final class ResidualResolver {

    /** RWKV6 rescale factor (models/rwkv6.cpp:165). */
    private static final float RWKV6_RESCALE = 0.5f;

    private final ModelHparams hp;
    private final TensorCatalog tc;
    private final Arch arch;

    public ResidualResolver(ModelHparams hp, TensorCatalog tc) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.tc = Objects.requireNonNull(tc, "tc");
        this.arch = hp.arch();
    }

    /**
     * Residual wiring of main-stack layer {@code il}.
     *
     * @throws IllegalStateException    when the trait requires hyper-connections but {@code hc_mult <= 1}
     * @throws IllegalArgumentException when the resolved values are inconsistent
     */
    public ResidualSpec resolve(int il) {
        checkLayer(il);
        ResidualKind kind = kind(il);
        ResidualSpec.Builder b = ResidualSpec.builder(kind, hp.nEmbd());
        if (kind == ResidualKind.SCALED || kind == ResidualKind.NORMED_INPUT_SCALED) {
            b.residualScale(hp.residualScale());
        }
        if (kind == ResidualKind.HYPER_CONNECTIONS) {
            b.hyperConnections(hp.dsv4HcMult(), true);
        }
        if (kind == ResidualKind.DEPTH_SCALED) {
            int i = 2 * (il + 1);
            b.depthScales(beamBranchScale(i), beamStreamScale(i), beamBranchScale(i + 1), beamStreamScale(i + 1));
        }
        if (arch == Arch.DECI) {
            b.discardLayerOutput(hp.nFf(il) == 0);
            b.mixerResidual(hp.nHead(il) != 0);
        }
        b.attnAlphaBeta(weight(il, TensorNames.ATTN_RESID_ALPHA), weight(il, TensorNames.ATTN_RESID_BETA));
        b.ffnAlphaBeta(weight(il, TensorNames.FFN_RESID_ALPHA), weight(il, TensorNames.FFN_RESID_BETA));
        b.ffnUsesAttnNorm2(ffnUsesAttnNorm2(il));
        if (arch == Arch.TALKIE) {
            b.embeddingSkip(true, 1.0f);
        }
        if (arch == Arch.RWKV6 && hp.rescaleEveryN() > 0 && (il + 1) % hp.rescaleEveryN() == 0) {
            b.blockOutputScale(RWKV6_RESCALE);
        }
        return b.build();
    }

    /**
     * Residual of layer {@code il} of a T5 encoder or decoder stack: plain {@code x += branch} after each
     * sub-block, including the decoder's cross-attention (models/t5.cpp).
     */
    public ResidualSpec resolve(String stack, int il) {
        Objects.requireNonNull(stack, "stack");
        if (TensorNames.STACK_BLK.equals(stack)) {
            return resolve(il);
        }
        if (il < 0) {
            throw new IndexOutOfBoundsException(arch.ggufName() + ": " + stack + " layer " + il);
        }
        return ResidualSpec.builder(ResidualKind.STANDARD, hp.nEmbd()).build();
    }

    /** {@code h_scale} of {@code beam_residual_add} (models/beam.cpp:110). */
    static float beamBranchScale(int i) {
        return (float) (1.0 / Math.sqrt((double) i));
    }

    /** {@code r_scale} of {@code beam_residual_add} (models/beam.cpp:111). */
    static float beamStreamScale(int i) {
        return (float) Math.sqrt((double) (i - 1) / (double) i);
    }

    private ResidualKind kind(int il) {
        ResidualKind trait = hp.traits().residualKind();
        if (hp.dsv4HcMult() > 1) {
            return plainNextnBlock(il) ? ResidualKind.STANDARD : ResidualKind.HYPER_CONNECTIONS;
        }
        if (trait == ResidualKind.DEPTH_SCALED && arch != Arch.BEAM) {
            throw new IllegalStateException(arch.ggufName() + ": no depth schedule for a depth-scaled residual");
        }
        if (trait == ResidualKind.HYPER_CONNECTIONS) {
            throw new IllegalStateException(arch.ggufName() + ": hyper-connections need hc_mult > 1, got "
                    + hp.dsv4HcMult());
        }
        if (arch == Arch.GPTNEOX) {
            return hp.useParallelResidual() ? ResidualKind.PARALLEL : ResidualKind.STANDARD;
        }
        if (arch == Arch.STABLELM) {
            return weight(il, TensorNames.FFN_NORM) == null ? ResidualKind.PARALLEL : ResidualKind.STANDARD;
        }
        if (trait == ResidualKind.SCALED && hp.residualScale() == 0f) {
            return ResidualKind.STANDARD;
        }
        if (trait == ResidualKind.NORMED_INPUT_SCALED && hp.residualScale() == 0f) {
            throw new IllegalStateException(arch.ggufName() + ": residual_scale must be set for a normed-input residual");
        }
        return trait;
    }

    /**
     * glm5-next creates the {@code hc_*} mixer tensors for trunk blocks only (glm5-next.cpp:85-92); its MTP graph runs
     * the NextN block over one {@code n_embd} stream with plain residual adds (glm5-next.cpp:626, 656).
     */
    private boolean plainNextnBlock(int il) {
        return arch == Arch.GLM5_NEXT && il >= hp.nLayer();
    }

    /**
     * falcon-40b (attention from {@code attn_norm_2}, FFN from {@code attn_norm}, models/falcon.cpp:70-86, 125) and
     * RWKV6/7 (channel mix from {@code attn_norm_2}) have the second input norm.
     */
    private boolean ffnUsesAttnNorm2(int il) {
        boolean family = arch == Arch.FALCON || arch == Arch.RWKV6 || arch == Arch.RWKV7;
        return family && weight(il, TensorNames.ATTN_NORM) != null && weight(il, TensorNames.ATTN_NORM_2) != null;
    }

    private TensorRef weight(int il, String base) {
        return tc.layer(il, TensorNames.weight(base));
    }

    private void checkLayer(int il) {
        if (il < 0 || il >= hp.nLayerAll()) {
            throw new IndexOutOfBoundsException(arch.ggufName() + ": layer " + il + " outside [0, " + hp.nLayerAll() + ")");
        }
    }

    @Override
    public String toString() {
        return "ResidualResolver{" + arch.ggufName() + '}';
    }
}
