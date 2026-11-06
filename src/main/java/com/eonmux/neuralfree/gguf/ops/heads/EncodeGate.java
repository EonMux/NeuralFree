package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ClassifierScope;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;

/**
 * Which embedding outputs a model offers, shared by the GPU and CPU orchestrators so both accept and reject the same
 * models with the same messages:
 * <ul>
 *   <li>an embedding model ({@link SupportClass#EMBED}) is an encoder;</li>
 *   <li>a decision model ({@link DecisionScorer#present}) has no embedding output: llama.cpp's {@code t_embd} of
 *       vllm-decision2 and clef is the decision score vector (vllm-decision2.cpp:137-146, clef.cpp:312), and the
 *       modern-bert / lfm2 decision blocks consume the hidden rows, so all of them score through classify;</li>
 *   <li>{@code encode} needs an encoder or a pooling type set by the GGUF or the override (llama.cpp embeddings of
 *       any model); {@code encodeBatch} also needs a pooling that answers one row per sequence;</li>
 *   <li>{@code encodeTokens} needs NONE pooling and an encoder, or pockettts whose backbone exposes its
 *       {@code t_embd} rows;</li>
 *   <li>{@code rerank} needs a sequence classifier (RANK pooling or a pooled classifier), as the llama.cpp
 *       server's rerank endpoint does.</li>
 * </ul>
 */
public final class EncodeGate {

    private EncodeGate() {
    }

    /** True for an embedding model: its output is embeddings, not token logits or decision scores. */
    public static boolean isEncoder(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return spec.supportClass() == SupportClass.EMBED;
    }

    /**
     * True when NONE pooling runs the classifier chain over every token (vLLM's {@code token_classify}): a
     * {@link ClassifierScope#TOKEN} classifier head. A sparse model's classifier reduces into its {@code embed}
     * output, so its per-token rows stay embeddings (vLLM's {@code token_embed} on a SPLADE model).
     */
    public static boolean hasTokenClassifier(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return !DecisionScorer.present(spec) && spec.head().classifierScope() == ClassifierScope.TOKEN
                && sparsePooling(spec) == null && spec.head().sparseOutput() == null;
    }

    /**
     * True when the model stores a separate per-token score projection ({@code cls.sparse}, vLLM's BGE-M3
     * {@code sparse_linear}): its {@code token_classify} rows are that projection's ReLU scores, while the pooled
     * embedding stays the sequence output.
     */
    public static boolean hasTokenScoreProjection(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return !DecisionScorer.present(spec) && spec.head().sparseOutput() != null;
    }

    /**
     * True when the model stores a separate per-token embedding projection ({@code cls.colbert}, vLLM's BGE-M3
     * {@code colbert_linear}): {@code token_embed} answers its rows.
     */
    public static boolean hasTokenEmbeddingProjection(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return !DecisionScorer.present(spec) && spec.head().colbertOutput() != null;
    }

    /**
     * True when a pooled type (MEAN, CLS, LAST or RANK) runs the classifier chain over the pooled row: a
     * {@link ClassifierScope#SEQUENCE} classifier. RANK pooling answers one row per sequence even without an
     * explicit scope, which llama.cpp's legacy RANK classifier and the RANK override allow.
     */
    public static boolean hasSequenceClassifier(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        if (DecisionScorer.present(spec) || spec.head().pooling() == PoolingKind.NONE) {
            return false;
        }
        return spec.head().classifierScope() == ClassifierScope.SEQUENCE
                || spec.head().classifierScope() == null && spec.head().pooling() == PoolingKind.RANK;
    }

    /**
     * True when {@code spec} reads a fixed-question decision answer over the pooled classifier output: the pooled
     * read of a decision model without a per-span head — LAST pooling with a classifier output head whose output
     * width names one score per class (llama.cpp's minicpm-ocr-jev: {@code n_classes == 2} scores of
     * {@code n_embd_out}). Unlike a sequence classifier it answers only the decision routes, so {@code rerank} and
     * {@code encode} still refuse it.
     */
    public static boolean hasDecisionClassifier(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return spec.head().pooling() != PoolingKind.NONE
                && spec.head().clsOutput() != null && spec.hparams().decisionType() != null;
    }

    /**
     * True when {@code token_classify} applies to {@code spec}: a token classifier, or a sequence classifier,
     * whose score projection vLLM's {@code for_seq_cls} dispatch also runs over every row.
     */
    public static boolean canTokenClassify(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return hasTokenClassifier(spec) || hasTokenScoreProjection(spec)
                || spec.head().classifierScope() == ClassifierScope.SEQUENCE;
    }

    /**
     * Checks that {@code token_classify} applies to {@code spec}.
     *
     * @throws UnsupportedOperationException for a model without a classifier head
     */
    public static void checkTokenClassify(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        if (DecisionScorer.present(spec)) {
            throw new UnsupportedOperationException("token_classify: " + spec.arch().ggufName()
                    + " is a decision model; score it with classify");
        }
        if (sparsePooling(spec) != null) {
            throw new UnsupportedOperationException("token_classify: " + spec.arch().ggufName()
                    + " pools a sparse embedding (" + spec.hparams().classifierSparse() + "); the per-token"
                    + " classifier scores reduce into the embed output");
        }
        if (!canTokenClassify(spec)) {
            throw new UnsupportedOperationException("token_classify needs a classifier head; "
                    + spec.arch().ggufName() + " (" + spec.supportClass() + ", pooling " + spec.head().pooling()
                    + ") stores none");
        }
    }

    /**
     * Checks that {@code encode} applies to {@code spec}.
     *
     * @throws UnsupportedOperationException for a decision model, a token classifier, and a model that is not an
     *                                       encoder and sets no pooling type
     */
    public static void checkEncode(ModelSpec spec) {
        rejectDecision(spec, "encode");
        if (hasTokenClassifier(spec)) {
            throw new UnsupportedOperationException("encode: " + spec.arch().ggufName() + " is a token classifier;"
                    + " its per-token rows are classifier scores (use token_classify)");
        }
        if (spec.head().classifierScope() == ClassifierScope.SEQUENCE) {
            throw new UnsupportedOperationException("encode: " + spec.arch().ggufName() + " is a sequence"
                    + " classifier; its pooled row is a classifier score (use classify)");
        }
        if (!isEncoder(spec) && spec.head().pooling() == PoolingKind.NONE) {
            throw new UnsupportedOperationException("encode needs an embedding model or a pooling type; "
                    + spec.arch().ggufName() + " (" + spec.supportClass() + ") has neither");
        }
    }

    /**
     * Checks that {@code encodeBatch} applies to {@code spec}: {@link #checkEncode}, and a pooling that answers one
     * row per sequence.
     *
     * @throws UnsupportedOperationException as {@link #checkEncode}, and for NONE pooling, whose output is one
     *                                       embedding per token
     */
    public static void checkEncodeBatch(ModelSpec spec) {
        checkEncode(spec);
        if (spec.head().pooling() == PoolingKind.NONE) {
            throw new UnsupportedOperationException("batched encode pools one row per sequence; "
                    + spec.arch().ggufName() + " pools by NONE and embeds per token");
        }
    }

    /**
     * Checks that {@code encodeTokens} applies to {@code spec}.
     *
     * @throws UnsupportedOperationException for a decision model, a model that pools per sequence, and a model with
     *                                       no per-token embedding output
     */
    public static void checkEncodeTokens(ModelSpec spec) {
        rejectDecision(spec, "encodeTokens");
        if (hasTokenClassifier(spec)) {
            throw new UnsupportedOperationException("encodeTokens: " + spec.arch().ggufName()
                    + " is a token classifier; its per-token rows are classifier scores (use token_classify)");
        }
        PoolingKind pooling = spec.head().pooling();
        if (!perToken(spec) && !hasTokenEmbeddingProjection(spec)) {
            throw new UnsupportedOperationException("per-token embeddings not supported by " + spec.arch().ggufName()
                    + " (" + spec.supportClass() + ", pooling " + pooling + ")");
        }
    }

    /**
     * Whether {@code spec}'s embeddings are per token: no pooling, on an embedding model or pockettts's latent
     * output. A decision model has none ({@link #checkEncodeTokens} refuses it by name).
     */
    public static boolean embedsPerToken(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        if (DecisionScorer.present(spec)) {
            return false;
        }
        if (hasTokenEmbeddingProjection(spec)) {
            return true;
        }
        return !hasTokenClassifier(spec) && perToken(spec);
    }

    /**
     * The sparse pooling over the per-token classifier scores ({@code classifier.sparse}, vLLM's
     * {@code SPLADESparsePooler}), or null: a sparse model's {@code embed} output is the reduced score vector
     * ({@code log1p(relu(x))} max or sum), and its per-token classifier scores are not offered.
     */
    public static String sparsePooling(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return spec.hparams().classifierSparse();
    }

    private static boolean perToken(ModelSpec spec) {
        return spec.head().pooling() == PoolingKind.NONE && (isEncoder(spec) || spec.arch() == Arch.POCKETTTS);
    }

    /**
     * Checks that {@code rerank} applies to {@code spec}: a sequence classifier or RANK pooling answers the score
     * of a pair.
     *
     * @throws UnsupportedOperationException when the model has no sequence classifier
     */
    public static void checkRerank(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        if (!hasSequenceClassifier(spec)) {
            throw new UnsupportedOperationException("reranking needs a sequence classifier (RANK pooling or a"
                    + " pooled classifier); " + spec.arch().ggufName() + " pools by " + spec.head().pooling());
        }
    }

    private static void rejectDecision(ModelSpec spec, String what) {
        Objects.requireNonNull(spec, "spec");
        if (DecisionScorer.present(spec)) {
            throw new UnsupportedOperationException(what + ": " + spec.arch().ggufName() + " is a decision model;"
                    + " score it with classify");
        }
    }
}
