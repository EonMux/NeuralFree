package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * Resolves the model output into a {@link HeadSpec}:
 * <ul>
 *   <li>final norm: {@code output_norm} (+ bias), grouped like the block norms for k2-horizon
 *       ({@link NormResolver#normGroups(ModelHparams)}); LFM2's {@code token_embd_norm}; the T5 decoder's
 *       {@code dec.output_norm}, the {@code enc.output_norm} of the T5 encoder and neo-bert
 *       (models/neo-bert.cpp:21, 126-128);</li>
 *   <li>LM head: {@code output}, else the token table (tied) for every model that generates and for clef, whose
 *       lexical option term reads the tied table (models/qwen35.cpp:43-53, models/clef.cpp:567); other embedding and
 *       classifier models get an LM head only when the file stores {@code output}; gemma-embedding2's
 *       {@code output} is the embedding projection, not an LM head (models/gemma-embedding2.cpp:41, 206-212);</li>
 *   <li>qwen35's optional {@code cls.output} {@code {n_embd, n_embd_out}} (+ bias) forms the embedding rows
 *       ({@code t_embd}) that pooling reads, while the LM head keeps the normed rows (models/qwen35.cpp:47-48,
 *       222-229); build_pooling applies the same {@code cls.output} once more to a RANK-pooled vector
 *       (llama-graph.cpp:4166-4171, llama-model.cpp:3330), so it is also the classifier output;</li>
 *   <li>{@code output.bias} only where the graph loads and adds it: qwen2, also over a tied head (models/qwen2.cpp:
 *       26-31, 147-148), phi2 (phi2.cpp:22, 136), phimoe through the phi3 graph (phimoe.cpp:23, phi3.cpp:184-186) and
 *       wavtokenizer-dec (wavtokenizer-dec.cpp:111, 258); phi3 and pangu-embedded add a bias their loaders never
 *       create, and dream and qwen2vl load it without using it;</li>
 *   <li>chameleon masks the image-token logits {@code [4, 8196)} with {@code -FLT_MAX} (models/chameleon.cpp:186-198).
 *       llama.cpp writes the mask into the first output row only ({@code ggml_set_1d}); the mask applies to every
 *       logits row here, so text generation never samples an image token at any output position;</li>
 *   <li>logit scale in the architecture's direction ({@code HparamsSupport.ScalingDefaults}: the granite family
 *       and minicpm divide; command-r, cohere2, cohere2moe, grok, talkie, muse-*, dflash and minicpm3 multiply,
 *       minicpm3's {@code n_embd_base / n_embd} on the hidden state before the linear head being the same product,
 *       models/minicpm3.cpp:240-246), final softcap where the graph applies one (gemma2/3/3n/4, grok, beam, muse-*,
 *       dflash), the qwen3tts codec-head shape check, the hyper-connection head, eagle3 {@code d2t}, pooling,
 *       {@code dense_2}/{@code dense_3} and the {@code cls} classifier.</li>
 * </ul>
 * Immutable.
 */
public final class HeadResolver {

    /** Graphs that apply {@code f_final_logit_softcapping} to the logits. */
    private static final Set<Arch> FINAL_SOFTCAP = Collections.unmodifiableSet(EnumSet.of(
            Arch.GEMMA2, Arch.GEMMA3, Arch.GEMMA3N, Arch.GEMMA4, Arch.GROK, Arch.BEAM, Arch.MUSE_SPARK,
            Arch.MUSE_GLIMMER, Arch.DFLASH));

    /** Graphs that load {@code output.bias} and add it to the logits. */
    private static final Set<Arch> OUTPUT_BIAS = Collections.unmodifiableSet(EnumSet.of(
            Arch.QWEN2, Arch.PHI2, Arch.PHIMOE, Arch.WAVTOKENIZER_DEC));

    /** chameleon image-token logit range set to {@code -FLT_MAX} (models/chameleon.cpp:188-189). */
    static final int CHAMELEON_IMAGE_TOKEN_START = 4;
    static final int CHAMELEON_IMAGE_TOKEN_END = 8196;

    private final ModelHparams hp;
    private final TensorCatalog tc;
    private final Arch arch;

    public HeadResolver(ModelHparams hp, TensorCatalog tc) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.tc = Objects.requireNonNull(tc, "tc");
        this.arch = hp.arch();
    }

    /**
     * The model's output spec.
     *
     * @throws IllegalArgumentException when the resolved values are inconsistent
     */
    public HeadSpec resolve() {
        HeadSpec.Builder b = HeadSpec.builder();
        String normName = outputNormName();
        b.outputNorm(tc.get(TensorNames.weight(normName)), tc.get(TensorNames.bias(normName)));
        b.outputNormGroups(NormResolver.normGroups(hp));

        TensorRef output = tc.get(TensorNames.weight(TensorNames.OUTPUT));
        if (arch == Arch.GEMMA_EMBEDDING2) {
            b.embeddingProjection(output);
            output = null;
        }
        boolean tied = false;
        if (output == null && arch != Arch.GEMMA_EMBEDDING2 && (generates() || arch == Arch.CLEF)) {
            output = tc.get(TensorNames.weight(TensorNames.TOKEN_EMBD));
            tied = output != null;
        }
        b.output(output, tied);
        if (output != null && OUTPUT_BIAS.contains(arch)) {
            b.outputBias(tc.get(TensorNames.bias(TensorNames.OUTPUT)));
        }
        if (arch == Arch.CHAMELEON && output != null) {
            int end = (int) Math.min(CHAMELEON_IMAGE_TOKEN_END, output.ne(1));
            b.maskedLogits(Math.min(CHAMELEON_IMAGE_TOKEN_START, end), end);
        }
        if (FINAL_SOFTCAP.contains(arch)) {
            b.finalSoftcap(hp.finalSoftcap());
        }
        boolean multiplies = HparamsSupport.ScalingDefaults.multipliesLogits(arch);
        boolean divides = HparamsSupport.ScalingDefaults.dividesLogits(arch);
        if ((multiplies || divides) && hp.logitScale() != 0f) {
            b.logitScale(hp.logitScale(), hp.logitScaleDivides());
        }
        if (arch == Arch.QWEN3TTS) {
            checkCodecHead(output);
        }
        b.hcHead(hp.dsv4HcMult() > 1 && tc.get(TensorNames.weight(TensorNames.HC_HEAD_FN)) != null);
        TensorRef d2t = tc.get(TensorNames.D2T);
        b.eagle3D2t(d2t != null ? d2t : tc.get(TensorNames.weight(TensorNames.D2T)));
        b.pooling(hp.pooling());
        b.dense(tc.get(TensorNames.weight(TensorNames.DENSE_2_OUT)), tc.get(TensorNames.weight(TensorNames.DENSE_3_OUT)));
        b.classifier(tc.get(TensorNames.weight(TensorNames.CLS)), tc.get(TensorNames.weight(TensorNames.CLS_OUT)),
                tc.get(TensorNames.weight(TensorNames.CLS_NORM)));
        if (arch == Arch.QWEN35) {
            resolveEmbeddingOutput(b);
        }
        return b.build();
    }

    /**
     * qwen35's embedding output {@code cls.output} {@code {n_embd, n_embd_out}} and its bias {@code {n_embd_out}},
     * both optional (models/qwen35.cpp:47-48); the graph uses the bias only with the weight (qwen35.cpp:222-226).
     *
     * @throws GgufFormatException when a tensor's shape differs from the one llama.cpp creates it with
     */
    private void resolveEmbeddingOutput(HeadSpec.Builder b) {
        TensorRef weight = tc.get(TensorNames.weight(TensorNames.CLS_OUT));
        if (weight == null) {
            return;
        }
        long nEmbd = hp.nEmbd();
        long nEmbdOut = hp.nEmbdOut();
        if (weight.rank() != 2 || weight.ne(0) != nEmbd || weight.ne(1) != nEmbdOut) {
            throw new GgufFormatException(arch.ggufName() + ": " + weight.name() + " is "
                    + Arrays.toString(weight.ne()) + ", expected [" + nEmbd + ", " + nEmbdOut
                    + "] (n_embd, n_embd_out; qwen35.cpp:47)");
        }
        TensorRef bias = tc.get(TensorNames.bias(TensorNames.CLS_OUT));
        if (bias != null && (bias.ne(0) != nEmbdOut || bias.ne(1) != 1 || bias.ne(2) != 1 || bias.ne(3) != 1)) {
            throw new GgufFormatException(arch.ggufName() + ": " + bias.name() + " is "
                    + Arrays.toString(bias.ne()) + ", expected [" + nEmbdOut + "] (n_embd_out;"
                    + " qwen35.cpp:48)");
        }
        b.embeddingOutput(weight, bias);
    }

    /**
     * qwen3tts's codec head may have fewer rows than {@code token_embd}; the logits are then padded in front with
     * {@code -inf}, which needs {@code output_norm}, and more head rows than table rows is invalid
     * (qwen3vl.cpp:175-191).
     *
     * @throws GgufFormatException for a head wider than the token table, or a padded head without output_norm
     */
    private void checkCodecHead(TensorRef output) {
        TensorRef tok = tc.get(TensorNames.weight(TensorNames.TOKEN_EMBD));
        if (output == null || tok == null) {
            return;
        }
        if (tok.ne(1) < output.ne(1)) {
            throw new GgufFormatException(arch.ggufName() + ": the LM head has " + output.ne(1) + " rows, more than the "
                    + tok.ne(1) + " rows of " + tok.name() + " (qwen3vl.cpp:189-191)");
        }
        if (tok.ne(1) > output.ne(1) && tc.get(TensorNames.weight(outputNormName())) == null) {
            throw new GgufFormatException(arch.ggufName() + ": a codec head padded to the token table needs "
                    + outputNormName() + " (qwen3vl.cpp:178)");
        }
    }

    /** Name of the final norm tensor. */
    private String outputNormName() {
        switch (arch) {
            case LFM2:
            case LFM2MOE:
                return TensorNames.OUTPUT_NORM_LFM2;
            case T5:
                return TensorNames.DEC_OUTPUT_NORM;
            case T5ENCODER:
            case NEO_BERT:
                return TensorNames.ENC_OUTPUT_NORM;
            default:
                return TensorNames.OUTPUT_NORM;
        }
    }

    /** True when the model produces token logits and so ties a missing {@code output} to the token table. */
    private boolean generates() {
        SupportClass s = hp.traits().support();
        return s != SupportClass.EMBED && s != SupportClass.CLS;
    }

    @Override
    public String toString() {
        return "HeadResolver{" + arch.ggufName() + '}';
    }
}
