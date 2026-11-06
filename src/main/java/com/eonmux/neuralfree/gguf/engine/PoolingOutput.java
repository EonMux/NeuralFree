package com.eonmux.neuralfree.gguf.engine;

/**
 * The pooled output one pooling request completes with (vLLM's pooling tasks): the kind a
 * {@link PoolingOperator} applies to the prompt's final hidden rows. A batch engine runs one pooling request
 * per sequence beside its generations, so the kinds are what the server's pooling routes answer.
 */
public enum PoolingOutput {

    /** The pooled vector: MEAN, CLS or LAST pooled through the dense projections (task {@code embed}). */
    POOLED,

    /** Per-token embedding rows of NONE pooling (task {@code token_embed}). */
    TOKEN_EMBEDDINGS,

    /** Per-token classifier rows of NONE pooling (vLLM's {@code token_classify}). */
    TOKEN_SCORES,

    /** The decision head's per-token scores, else the sequence classifier's scores (task {@code classify}). */
    SCORES,

    /** The RANK classifier output: scores with their labels (a rank model's {@code classify}). */
    CLASSIFIER_OUTPUT,

    /** The first classifier output's score, the relevance score of a rerank input ({@code /rerank}, {@code /score}). */
    RERANK_SCORE
}
