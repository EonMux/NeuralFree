package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;
import java.util.function.Predicate;

/**
 * The MTP-head family of every NextN/MTP architecture: whether its {@code h_nextn} is the post-norm state or the
 * pre-norm stream, whether several blocks chain, cycle or refuse to run, and which heads read hyper-connection
 * streams. Also the tensor naming the heads share: a block's {@code nextn.*} tensor name, its head-norm tensor
 * resolution and the norm epsilon a norm kind reads from the hparams.
 */
final class NextnArchitectures {

    /**
     * Targets whose {@code h_nextn} is the output-norm state (the target graphs' {@code t_h_nextn}; MiMo's
     * {@code mimo_mtp.py} head reads the Qwen2 model output after its final norm).
     */
    private static final Set<Arch> POST_NORM_TARGET = Collections.unmodifiableSet(EnumSet.of(Arch.DEEPSEEK2,
            Arch.DEEPSEEK32, Arch.GLM4_MOE, Arch.GLM_DSA, Arch.MELLUM, Arch.BAILINGMOE2, Arch.BAILINGMOE3,
            Arch.COHERE2MOE, Arch.HY_V3, Arch.QWEN3NEXT, Arch.QWEN35, Arch.QWEN35MOE, Arch.NEMOTRON_H,
            Arch.NEMOTRON_H_MOE, Arch.GLM5_NEXT, Arch.QWEN2, Arch.EXAONE4, Arch.EXAONE_MOE, Arch.DOTS3NOTE));
    /**
     * Targets and heads whose {@code h_nextn} is the pre-norm stream (mimo2.cpp:236-238, step35.cpp:339-341; the
     * Step5 predictor returns the decoder block's residual output and the shared head norms it afterwards).
     */
    private static final Set<Arch> PRE_NORM = Collections.unmodifiableSet(EnumSet.of(Arch.MIMO2, Arch.STEP35,
            Arch.STEP5));
    /**
     * MTP graphs that select their block by {@code nextn_layer_offset} without asserting a single block
     * (hy-v3.cpp:236, mimo2.cpp:265, step35.cpp:365); every other MTP graph asserts {@code n_layer_nextn == 1}.
     */
    private static final Set<Arch> CHAINED_HEADS = Collections.unmodifiableSet(EnumSet.of(Arch.HY_V3, Arch.MIMO2,
            Arch.STEP35));
    /**
     * MTP heads that cycle their blocks (vLLM {@code spec_step_idx % num_mtp_layers}): draft step {@code k} runs
     * block {@code k mod n}, each block with its own caches: exaone4_5_mtp.py:119, exaone_moe_mtp.py:114,
     * bailing_moe_mtp.py:177, deepseek_mtp.py:217 (GLM-4 MoE Lite) and the dots3_note predictor over
     * deepseek_v32's.
     */
    private static final Set<Arch> CYCLING_HEADS = Collections.unmodifiableSet(EnumSet.of(Arch.EXAONE4,
            Arch.EXAONE_MOE, Arch.BAILINGMOE2, Arch.DOTS3NOTE, Arch.DEEPSEEK2, Arch.STEP5));
    /**
     * MTP heads over the flattened hyper-connection streams of the trunk output, whose block keeps the streams
     * (deepseek4.cpp:1376, 1407-1427; qwen4exp.cpp:530, 581-588).
     */
    private static final Set<Arch> HC_STREAM = Collections.unmodifiableSet(EnumSet.of(Arch.DEEPSEEK4,
            Arch.QWEN4EXP));
    /**
     * Hyper-connection targets whose MTP head reads the stream mean after the output norm and whose block has no
     * hyper-connections (glm5-next.cpp:566, 785-798).
     */
    private static final Set<Arch> HC_MEAN_TARGET = Collections.unmodifiableSet(EnumSet.of(Arch.GLM5_NEXT));

    private NextnArchitectures() {
    }

    /** Whether {@code arch}'s {@code h_nextn} is the target's output-norm state. */
    static boolean postNormTarget(Arch arch) {
        return POST_NORM_TARGET.contains(arch);
    }

    /** Whether {@code arch}'s {@code h_nextn} is the pre-norm stream. */
    static boolean preNorm(Arch arch) {
        return PRE_NORM.contains(arch);
    }

    /** Whether {@code arch}'s MTP graph chains several blocks by {@code nextn_layer_offset}. */
    static boolean chainsHeads(Arch arch) {
        return CHAINED_HEADS.contains(arch);
    }

    /** Whether {@code arch}'s MTP head cycles its blocks, one block per draft step. */
    static boolean cyclesHeads(Arch arch) {
        return CYCLING_HEADS.contains(arch);
    }

    /** Whether {@code arch}'s MTP head runs over the flattened hyper-connection streams of the trunk output. */
    static boolean hcStream(Arch arch) {
        return HC_STREAM.contains(arch);
    }

    /** Whether {@code arch}'s MTP head reads the stream mean after the output norm of a hyper-connection trunk. */
    static boolean hcMeanTarget(Arch arch) {
        return HC_MEAN_TARGET.contains(arch);
    }

    /** A NextN block's {@code base} tensor at weight layer {@code wl}. */
    static String nextnTensor(int wl, String base) {
        return TensorNames.layerName(wl, TensorNames.weight(base));
    }

    /** The epsilon {@code kind}'s norm reads from {@code hp}. */
    static float eps(ModelHparams hp, NormKind kind) {
        return kind == NormKind.LAYER ? hp.normEps() : hp.normRmsEps();
    }

    /**
     * The head-norm tensor of NextN block {@code wl}. bailingmoe3 loads its head norm from the block's
     * {@code layer_output_norm} and requires it (bailingmoe3.cpp:152, 430, 548). Every other head takes
     * {@code nextn.shared_head_norm}, which nemotron_h_moe requires; else mimo2's or bailingmoe2's
     * {@code layer_output_norm} (mimo2.cpp:384-386, bailing_moe_mtp.py {@code final_layernorm}); else the target's
     * {@code output_norm}.
     *
     * @param present    whether the target file holds a tensor
     * @param outputNorm name of the target's {@code output_norm}, or null when it has none
     * @throws IllegalArgumentException when the tensor the architecture needs is missing
     */
    static String headNormWeight(Arch arch, int wl, Predicate<String> present, String outputNorm) {
        String layerOut = TensorNames.layerName(wl, TensorNames.weight(TensorNames.LAYER_OUT_NORM));
        if (arch == Arch.BAILINGMOE3) {
            return requireHeadNorm(arch, present, layerOut, layerOut);
        }
        String shared = nextnTensor(wl, TensorNames.NEXTN_SHARED_HEAD_NORM);
        if (arch == Arch.NEMOTRON_H_MOE) {
            return requireHeadNorm(arch, present, shared, shared);
        }
        if (present.test(shared)) {
            return shared;
        }
        if ((arch == Arch.MIMO2 || arch == Arch.BAILINGMOE2) && present.test(layerOut)) {
            return layerOut;
        }
        return requireHeadNorm(arch, present, outputNorm, "a head norm");
    }

    private static String requireHeadNorm(Arch arch, Predicate<String> present, String tensor, String what) {
        if (tensor == null || !present.test(tensor)) {
            throw new IllegalArgumentException(arch.ggufName() + ": the MTP head needs " + what);
        }
        return tensor;
    }
}
