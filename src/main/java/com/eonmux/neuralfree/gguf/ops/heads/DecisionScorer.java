package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;

/**
 * The host decision head of a model over its final hidden rows, shared by the GPU and CPU orchestrators:
 * vllm-decision2 ({@link VllmDecision2Head}) and clef ({@link ClefDecoder}) score one value per token from a decision
 * order; the modern-bert and lfm2 decision models ({@link DecisionBody}) score
 * {@link ModernBertDecisionHead#N_DECISION_TYPES} question types per token, flattened {@code [t][type]} row-major;
 * kev ({@link KevScorer}) dots the last token's hidden state with every {@code <|box_end|>} option end's.
 * Immutable.
 */
public final class DecisionScorer {

    /** Scores of one input: final hidden rows, token ids and the decision order (null when absent). */
    @FunctionalInterface
    private interface Score {
        float[] apply(float[][] hidden, int[] tokens, int[] order);
    }

    private final String arch;
    private final Score score;
    private final boolean readsOrder;

    private DecisionScorer(String arch, Score score, boolean readsOrder) {
        this.arch = arch;
        this.score = score;
        this.readsOrder = readsOrder;
    }

    /**
     * The decision head of {@code spec}, read from {@code weights}, or null when the model has none. A kev-typed
     * model takes its marker from {@code tokenizer}'s vocabulary.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a head tensor is absent
     * @throws IllegalStateException                                  for a kev model whose vocabulary holds no
     *                                                                single {@code <|box_end|>} token
     */
    public static DecisionScorer forModel(ModelSpec spec, CpuWeights weights, GGUFTokenizer tokenizer) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(tokenizer, "tokenizer");
        String name = spec.arch().ggufName();
        if ("kev".equals(spec.hparams().decisionType())) {
            KevScorer h = KevScorer.of(spec, tokenizer);
            return new DecisionScorer(name, (hidden, tokens, order) -> h.score(hidden, tokens), false);
        }
        if (spec.arch() == Arch.VLLM_DECISION2) {
            VllmDecision2Head h = VllmDecision2Head.of(spec, weights);
            return new DecisionScorer(name, (hidden, tokens, order) -> h.score(hidden, order), true);
        }
        if (spec.arch() == Arch.CLEF) {
            ClefDecoder h = ClefDecoder.of(spec, weights);
            return new DecisionScorer(name, h::score, true);
        }
        if (DecisionBody.applies(spec) && spec.arch() == Arch.LFM2) {
            Lfm2DecisionHead h = Lfm2DecisionHead.of(spec, weights);
            return new DecisionScorer(name, (hidden, tokens, order) -> flatten(h.score(hidden)), false);
        }
        if (DecisionBody.applies(spec)) {
            ModernBertDecisionHead h = ModernBertDecisionHead.of(spec, weights);
            return new DecisionScorer(name, (hidden, tokens, order) -> flatten(h.score(hidden)), false);
        }
        return null;
    }

    /**
     * True when {@code spec} has a decision head: vllm-decision2, clef, a modern-bert / lfm2 decision model, or a
     * model whose decision type is kev, whose markers score over its own hidden rows.
     */
    public static boolean present(ModelSpec spec) {
        return spec.arch() == Arch.VLLM_DECISION2 || spec.arch() == Arch.CLEF || DecisionBody.applies(spec)
                || "kev".equals(spec.hparams().decisionType());
    }

    /** True when this head reads a decision order (vllm-decision2, clef). */
    public boolean readsOrder() {
        return readsOrder;
    }

    /**
     * Validates a decision order before the forward pass: null, or one code per token for a head that reads one.
     *
     * @throws IllegalArgumentException for an order given to a head that reads none, or of the wrong length
     */
    public void checkOrder(int nTokens, int[] order) {
        if (order == null) {
            return;
        }
        if (!readsOrder) {
            throw new IllegalArgumentException(arch + " decision blocks read no decision order; they score every"
                    + " question type per token");
        }
        if (order.length != nTokens) {
            throw new IllegalArgumentException(order.length + " decision codes for " + nTokens + " tokens");
        }
    }

    /**
     * Scores of one input.
     *
     * @param hidden final hidden rows, one per token
     * @param tokens the input token ids
     * @param order  one decision code per token for a head that reads one, or null
     */
    public float[] score(float[][] hidden, int[] tokens, int[] order) {
        checkOrder(tokens.length, order);
        return score.apply(hidden, tokens, order);
    }

    private static float[] flatten(float[][] rows) {
        int total = 0;
        for (float[] r : rows) {
            total = Math.addExact(total, r.length);
        }
        float[] out = new float[total];
        int off = 0;
        for (float[] r : rows) {
            System.arraycopy(r, 0, out, off, r.length);
            off += r.length;
        }
        return out;
    }

    @Override
    public String toString() {
        return "DecisionScorer{" + arch + (readsOrder ? ", reads order" : "") + '}';
    }
}
