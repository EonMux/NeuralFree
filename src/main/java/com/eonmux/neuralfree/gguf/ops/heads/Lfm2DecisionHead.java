package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;

/**
 * The lfm2 decision head (LFM2-D1, models/lfm2.cpp:29-36, 53-86, 422-482): the modern-bert decision blocks
 * ({@link ModernBertDecisionHead}) with the query and KV head counts of each head layer
 * ({@code hparams.n_head(il)}, {@code n_head_kv(il)}), on the final hidden rows of the non-causal body (centred
 * short convolutions and bidirectional attention, :365-418) after {@code token_embd_norm}.
 *
 * <p>The head's attention mask lets text read only text and media only media (llm_graph_input_attn_media with
 * {@code is_head}, :251-283); the rows of {@code ForwardBackend.hiddenRows} are all text tokens, so every key is
 * visible. Immutable; {@link #score} allocates its result.</p>
 */
public final class Lfm2DecisionHead {

    private final ModernBertDecisionHead blocks;

    private Lfm2DecisionHead(ModelSpec spec, CpuWeights w) {
        ModelHparams hp = spec.hparams();
        this.blocks = new ModernBertDecisionHead(spec, w, hp::nHead, hp::nHeadKv);
    }

    /** The decision head of an lfm2 decision model over the tensors of {@code weights}. */
    public static Lfm2DecisionHead of(ModelSpec spec, CpuWeights weights) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(weights, "weights");
        return new Lfm2DecisionHead(spec, weights);
    }

    /**
     * Scores of every token: {@code [t][it]} for question type {@code it} in
     * {@code 0 .. ModernBertDecisionHead.N_DECISION_TYPES - 1}.
     *
     * @param hiddenRows the body's final hidden rows of every text token, after {@code token_embd_norm}
     */
    public float[][] score(float[][] hiddenRows) {
        return blocks.score(hiddenRows);
    }
}
