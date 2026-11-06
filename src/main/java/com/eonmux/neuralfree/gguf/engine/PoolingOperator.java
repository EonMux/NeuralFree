package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionScorer;
import com.eonmux.neuralfree.gguf.ops.heads.EncodeGate;
import com.eonmux.neuralfree.gguf.ops.heads.PoolingHead;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;
import java.util.function.Supplier;

/**
 * The pooling heads of one model over final hidden rows (after the output norm), the operator a batch engine's
 * pooling requests apply at their prompt's completion: the pooled vector, the per-token embeddings, the classifier
 * or decision scores, the classifier output and the rerank score — the same outputs {@code LLMModel.encode},
 * {@code encodeTokens}, {@code classify}, {@code classifyOutput} and {@code rerank} answer, but over rows a batched
 * forward already computed instead of a memory reset. The head suppliers run lazily on the model's engine thread
 * (the engine's pooling completion runs there); a model without pooling heads has no operator, and its pooling
 * requests refuse by name.
 */
public interface PoolingOperator {

    /**
     * Refuses by name the outputs {@code output} the model cannot answer (vLLM validates a request's pooling task
     * against the model at submit): a pooled vector on per-token pooling, per-token embeddings on a pooled model,
     * classifier outputs on a model without the RANK head.
     *
     * @throws UnsupportedOperationException when the model does not answer {@code output}
     */
    void validate(PoolingOutput output);

    /**
     * Whether {@code output} reads every prompt row's hidden state (MEAN, CLS, per-token and decision scores), or
     * only the last computed row (LAST pooling and its RANK form): a request whose output reads every row cannot
     * reuse a cached prefix, because the reused rows' hidden states are not computed in its own forward.
     */
    boolean readsAllRows(PoolingOutput output);

    /**
     * The output {@code output} over the hidden rows of the prompt rows this run computed, in prompt order, and the
     * prompt's token ids (the decision heads read them): {@code float[]} for {@link PoolingOutput#POOLED} and
     * {@link PoolingOutput#SCORES} and {@link PoolingOutput#RERANK_SCORE} (boxed), {@code float[][]} for
     * {@link PoolingOutput#TOKEN_EMBEDDINGS} and {@link PoolingOutput#TOKEN_SCORES}, {@code ClassifierOutput} for
     * {@link PoolingOutput#CLASSIFIER_OUTPUT}. {@code activation} applies or skips the classifier's final
     * probability activation (vLLM's {@code use_activation}); {@code false} answers the raw head output, an
     * embedding output pools the same either way.
     *
     * @throws UnsupportedOperationException when the model does not answer {@code output}
     */
    Object apply(PoolingOutput output, float[][] hiddenRows, int[] tokens, boolean activation);

    /**
     * {@link #apply(PoolingOutput, float[][], int[], boolean)} with the head's activation applied (the output the
     * model answers by default).
     */
    default Object apply(PoolingOutput output, float[][] hiddenRows, int[] tokens) {
        return apply(output, hiddenRows, tokens, true);
    }

    /**
     * The operator of {@code spec} over the lazily built pooling head and decision scorer of one model: the shared
     * implementation behind {@code GGUFModel} and {@code GGUFModelGPU}, so every backend pools with the same
     * semantics.
     */
    static PoolingOperator of(ModelSpec spec, Supplier<PoolingHead> head, Supplier<DecisionScorer> scorer) {
        Objects.requireNonNull(spec, "spec");
        return new PoolingOperator() {
            @Override
            public void validate(PoolingOutput output) {
                switch (output) {
                    case POOLED:
                        EncodeGate.checkEncode(spec);
                        if (spec.head().pooling() == PoolingKind.NONE && EncodeGate.sparsePooling(spec) == null) {
                            throw new UnsupportedOperationException("the pooled vector needs a pooled embedding; "
                                    + spec.arch().ggufName() + " pools per token (none)");
                        }
                        return;
                    case TOKEN_EMBEDDINGS:
                        EncodeGate.checkEncodeTokens(spec);
                        return;
                    case TOKEN_SCORES:
                        EncodeGate.checkTokenClassify(spec);
                        return;
                    case SCORES:
                        if (DecisionScorer.present(spec)) {
                            return;
                        }
                        requireSequenceClassifier("classification scores need");
                        return;
                    case CLASSIFIER_OUTPUT:
                        if (!EncodeGate.hasDecisionClassifier(spec)) {
                            if (DecisionScorer.present(spec)) {
                                throw new UnsupportedOperationException(spec.arch().ggufName() + " is a decision"
                                        + " model; its decision scores come from classify, not a classifier output");
                            }
                            requireSequenceClassifier("a classifier output needs");
                        }
                        return;
                    case RERANK_SCORE:
                        EncodeGate.checkRerank(spec);
                        return;
                    default:
                        throw new IllegalArgumentException("unknown pooling output " + output);
                }
            }

            private void requireSequenceClassifier(String what) {
                if (!EncodeGate.hasSequenceClassifier(spec)) {
                    throw new UnsupportedOperationException(what + " a sequence classifier; "
                            + spec.arch().ggufName() + " pools by " + spec.head().pooling()
                            + (EncodeGate.hasTokenClassifier(spec) ? " and classifies per token; use token_classify"
                                    : " and has no classifier head"));
                }
            }

            @Override
            public boolean readsAllRows(PoolingOutput output) {
                if (output == PoolingOutput.TOKEN_EMBEDDINGS || output == PoolingOutput.TOKEN_SCORES
                        || DecisionScorer.present(spec) || EncodeGate.sparsePooling(spec) != null) {
                    return true;
                }
                return head.get().rowPooling() != PoolingKind.LAST;
            }

            @Override
            public Object apply(PoolingOutput output, float[][] hiddenRows, int[] tokens, boolean activation) {
                switch (output) {
                    case POOLED:
                        if (EncodeGate.sparsePooling(spec) != null) {
                            return head.get().sparseEmbed(hiddenRows, tokens);
                        }
                        return head.get().pool(hiddenRows, activation);
                    case TOKEN_EMBEDDINGS:
                        return head.get().poolTokens(hiddenRows, tokens);
                    case TOKEN_SCORES:
                        return head.get().classifyTokens(hiddenRows, tokens);
                    case SCORES:
                        DecisionScorer decision = scorer.get();
                        if (decision != null) {
                            return decision.score(hiddenRows, tokens, null);
                        }
                        return head.get().pool(hiddenRows, activation);
                    case CLASSIFIER_OUTPUT:
                        return head.get().classify(hiddenRows, activation);
                    case RERANK_SCORE:
                        return head.get().rerank(hiddenRows, activation);
                    default:
                        throw new IllegalArgumentException("unknown pooling output " + output);
                }
            }
        };
    }
}
