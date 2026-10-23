package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * Resolves the model input into an {@link EmbeddingSpec}: token table, learned position
 * and token-type embeddings, the input scale, {@code token_embd_norm}, the weightless input RMS norm of talkie,
 * muse-spark and muse-glimmer (talkie.cpp:50-51, muse-spark.cpp:86-87, muse-glimmer.cpp:69), the deepstack input
 * width and the per-layer embedding tables.
 *
 * <p>The input scale is {@code attention.embedding_scale} when set (applied in {@code build_inp_embd},
 * llama-graph.cpp:2761-2772: granite, grok, hrm_text, ...) times the graph's own factor: {@code sqrt(n_embd)} for
 * gemma, gemma2, gemma3, gemma3n, gemma4, gemma-embedding(2) and afmoe; 12 for minicpm3 (models/minicpm3.cpp:66).
 * LFM2's {@code token_embd_norm} is its output norm, not an input norm (see {@link HeadResolver}).
 *
 * <p>The token-type table joins the input only on the BERT graph ({@link #addsInputTokenTypes}). Other graphs that
 * load {@code token_types} use it in their heads (modern-bert.cpp:260, lfm2.cpp:434, clef.cpp:600), whose rows may
 * have a width other than {@code n_embd}. Immutable.
 */
public final class EmbeddingResolver {

    /** minicpm3 input scale (models/minicpm3.cpp:66). */
    private static final float MINICPM3_EMBD_SCALE = 12.0f;

    private final ModelHparams hp;
    private final TensorCatalog tc;
    private final Arch arch;

    public EmbeddingResolver(ModelHparams hp, TensorCatalog tc) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.tc = Objects.requireNonNull(tc, "tc");
        this.arch = hp.arch();
    }

    /**
     * The model's input spec. The token table is null only for drafters that embed through their target model.
     * codeshell without {@code token_embd} embeds through its LM head {@code output} (models/codeshell.cpp:15-20,
     * {@code TENSOR_DUPLICATED}).
     *
     * @throws IllegalArgumentException when the token table width does not match {@code n_embd}
     */
    public EmbeddingSpec resolve() {
        TensorRef tok = tc.get(TensorNames.weight(TensorNames.TOKEN_EMBD));
        if (tok == null && arch == Arch.CODESHELL) {
            tok = tc.get(TensorNames.weight(TensorNames.OUTPUT));
        }
        EmbeddingSpec.Builder b = EmbeddingSpec.builder(tok, hp.nEmbd())
                .posEmbd(tc.get(TensorNames.weight(TensorNames.POS_EMBD)))
                .embdScale(embeddingScale())
                .nEmbdInp(hp.nEmbdInp());
        if (addsInputTokenTypes(hp.traits())) {
            b.tokenTypes(tc.get(TensorNames.weight(TensorNames.TOKEN_TYPES)));
        }
        if (arch != Arch.LFM2 && arch != Arch.LFM2MOE) {
            b.tokenEmbdNorm(tc.get(TensorNames.weight(TensorNames.TOKEN_EMBD_NORM)));
        }
        b.inputRmsNorm(arch == Arch.TALKIE || arch == Arch.MUSE_SPARK || arch == Arch.MUSE_GLIMMER);
        TensorRef ple = tc.get(TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD));
        if (ple != null) {
            b.perLayer(ple, tc.get(TensorNames.weight(TensorNames.PER_LAYER_MODEL_PROJ)),
                    tc.get(TensorNames.weight(TensorNames.PER_LAYER_PROJ_NORM)), perLayerWidth());
        }
        if (hp.pleNHeads() > 0 && hp.pleNgramSize() > 0) {
            b.ngramPle(true);
            if (ple == null) {
                b.perLayer(null, null, null, perLayerWidth());
            }
        }
        return b.build();
    }

    /**
     * True when the architecture runs {@code llama_model_bert::graph} (bert, and through models.h:341, 352, 363, 374
     * jina-bert-v2, jina-bert-v3, nomic-bert and nomic-bert-moe), the only graph that adds token-type row 0 to every
     * input embedding (bert.cpp:91-94).
     */
    public static boolean addsInputTokenTypes(ArchTraits traits) {
        return Objects.requireNonNull(traits, "traits").graphFamily() == Arch.BERT;
    }

    /**
     * Width of one layer's slice of the per-layer embeddings: {@code n_embd_altup} for gemma3n, whose tables are
     * {@code {n_embd_altup * n_layer, n_vocab}} (models/gemma3n.cpp:39-41), {@code n_embd_per_layer} otherwise.
     */
    private int perLayerWidth() {
        return arch == Arch.GEMMA3N ? hp.nEmbdAltup() : hp.nEmbdPerLayer();
    }

    /** Multiplier on the token embeddings; 1 when the graph applies none. */
    public float embeddingScale() {
        float scale = hp.embeddingScale() != 0f ? hp.embeddingScale() : 1.0f;
        switch (arch) {
            case GEMMA:
            case GEMMA2:
            case GEMMA3:
            case GEMMA3N:
            case GEMMA4:
            case GEMMA_EMBEDDING:
            case GEMMA_EMBEDDING2:
            case AFMOE:
                return scale * (float) Math.sqrt((float) hp.nEmbd());
            case MINICPM3:
                return scale * MINICPM3_EMBD_SCALE;
            default:
                return scale;
        }
    }

    @Override
    public String toString() {
        return "EmbeddingResolver{" + arch.ggufName() + '}';
    }
}
