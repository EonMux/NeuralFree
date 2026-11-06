package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

/**
 * llama.cpp's mirostat sampler, both versions (llama_sampler_mirostat_apply, llama.cpp src/llama-sampler.cpp:2466, and
 * llama_sampler_mirostat_v2_apply, :2579): a terminal selector - it softmaxes, keeps or recomputes the candidate set
 * around a target surprise {@code tau}, draws one token with its own std::mt19937 through llama_sample_dist, sets that
 * candidate selected, and adapts {@code mu} toward the target by the learning rate {@code eta}. It replaces the chain's
 * dist stage, as llama.cpp builds it ({@code temp + mirostat}, common/sampling.cpp:408-413).
 *
 * <p>{@code mu} adapts on accepted tokens: apply keeps its final candidate set, and {@link #accept} moves {@code mu}
 * by the accepted token's surprise in that set, once per apply. Plain decoding accepts the token apply selected, which
 * is llama.cpp's in-apply update; speculative verification reads rows ahead of their confirmation, and a row whose
 * draft token is rejected never moves {@code mu}. A token outside the kept set (one apply never offered) leaves
 * {@code mu} as it is. {@link #saveState} keeps {@code mu} with the pending set, so a restored stage adapts on the
 * next accepted token as it would have when the state was saved.
 *
 * <p>v1 ({@code mirostat = 1}, {@code m = 100}) estimates the tail exponent {@code s_hat} over the first {@code m - 1}
 * probabilities, cuts the set to max(1, int(k)) candidates via llama.top_k_impl with k from mu and the vocabulary size,
 * then draws; the vocabulary size comes from the first candidate set (llama.cpp reads it from the vocabulary at chain
 * construction; here the chain is {@code temperature + mirostat}, so the first full row is exactly that size).
 * v2 ({@code mirostat = 2}) truncates at the first candidate with {@code -log2(p) > mu}, collapsing to one candidate
 * when none survives, then draws.
 */
public final class MirostatStage extends SamplerStage {

    private final int version;
    private final float tau;
    private final float eta;
    private final long seed;
    private int vocabSize;
    private float mu;
    /** The final candidate set of the last apply that no accepted token has adapted {@code mu} with, or null. */
    private Offer pending;
    private final Mt19937 rng;

    public MirostatStage(int version, float tau, float eta, long seed) {
        this.version = version;
        this.tau = tau;
        this.eta = eta;
        this.seed = seed;
        this.vocabSize = 0;
        this.mu = 2.0f * tau;
        this.rng = new Mt19937(seed);
    }

    @Override
    public StageKind kind() {
        return StageKind.MIROSTAT;
    }

    @Override
    public void apply(Candidates candidates) {
        if (vocabSize == 0) {
            vocabSize = candidates.size();
        }
        if (version == 1) {
            applyV1(candidates);
        } else {
            applyV2(candidates);
        }
    }

    private void applyV1(Candidates candidates) {
        if (!candidates.sorted()) {
            candidates.sortDescending();
        }
        candidates.softmax();

        // Estimate s_hat using the most probable m tokens
        float sHat = 0f;
        float sumTiBi = 0f;
        float sumTiSq = 0f;
        int m = 100;
        for (int i = 0; i < m - 1 && i < candidates.size() - 1; i++) {
            float ti = (float) Math.log((float) (i + 2) / (float) (i + 1));
            float bi = (float) Math.log(candidates.prob(i) / candidates.prob(i + 1));
            sumTiBi += ti * bi;
            sumTiSq += ti * ti;
        }
        sHat = sumTiBi / sumTiSq;

        // Compute k from the estimated s_hat and target surprise value
        float epsilonHat = sHat - 1;
        float k = (float) Math.pow((epsilonHat * Math.pow(2.0, mu)) / (1.0 - Math.pow(vocabSize, -epsilonHat)),
                1.0 / sHat);
        int keep = Math.max((int) k, 1);
        int kImpl = Math.min(keep, candidates.size());
        candidates.partialSortDescending(kImpl);
        candidates.truncate(kImpl);

        if (!candidates.sorted()) {
            candidates.sortDescending();
        }
        candidates.softmax();

        float[] probs = probs(candidates);
        int idx = rng.distDraw(probs);
        candidates.select(idx);
        pending = Offer.of(candidates, probs);
    }

    private void applyV2(Candidates candidates) {
        if (!candidates.sorted()) {
            candidates.sortDescending();
        }
        candidates.softmax();

        // Truncate the words with surprise values greater than mu
        int size = 0;
        for (int i = 0; i < candidates.size(); i++) {
            if (-log2f(candidates.prob(i)) > mu) {
                break;
            }
            size++;
        }
        candidates.truncate(Math.max(size, 1));

        // Normalize the probabilities of the remaining words
        if (!candidates.sorted()) {
            candidates.sortDescending();
        }
        candidates.softmax();

        float[] probs = probs(candidates);
        int idx = rng.distDraw(probs);
        candidates.select(idx);
        pending = Offer.of(candidates, probs);
    }

    private static float log2f(float v) {
        return (float) (Math.log(v) / Math.log(2.0));
    }

    /** The probabilities of the current candidates (llama_sample_dist's iterator range). */
    private static float[] probs(Candidates candidates) {
        float[] probs = new float[candidates.size()];
        for (int i = 0; i < probs.length; i++) {
            probs[i] = candidates.prob(i);
        }
        return probs;
    }

    /**
     * Adapts {@code mu} toward the target by the surprise of {@code token} in the last apply's candidate set
     * (llama.cpp's {@code mu = mu - eta * (-log2(p) - tau)}); nothing without a pending apply or for a token that
     * set did not keep.
     */
    @Override
    public void accept(int token) {
        Offer offer = pending;
        pending = null;
        if (offer == null) {
            return;
        }
        float p = offer.probOf(token);
        if (p > 0f) {
            float observedSurprise = -log2f(p);
            float e = observedSurprise - tau;
            mu = mu - eta * e;
        }
    }

    @Override
    public void reset() {
        mu = 2.0f * tau;
        pending = null;
        rng.seed(seed);
    }

    /**
     * {@code mu}, the vocabulary size the first apply measured and the pending candidate set; the draw's random
     * stream is not part of the state.
     */
    @Override
    public Object saveState() {
        return new State(mu, vocabSize, pending);
    }

    @Override
    public void restoreState(Object state) {
        if (!(state instanceof State)) {
            throw new IllegalArgumentException("not a mirostat stage state: " + state);
        }
        State s = (State) state;
        mu = s.mu;
        vocabSize = s.vocabSize;
        pending = s.pending;
    }

    /** A saved stage state ({@link #saveState}). */
    private static final class State {
        final float mu;
        final int vocabSize;
        final Offer pending;

        State(float mu, int vocabSize, Offer pending) {
            this.mu = mu;
            this.vocabSize = vocabSize;
            this.pending = pending;
        }
    }

    /** The token ids and probabilities of one apply's final candidate set; immutable. */
    private static final class Offer {
        private final int[] ids;
        private final float[] probs;

        private Offer(int[] ids, float[] probs) {
            this.ids = ids;
            this.probs = probs;
        }

        /** The set of {@code candidates}, whose probabilities are {@code probs} (owned by the new offer). */
        static Offer of(Candidates candidates, float[] probs) {
            int[] ids = new int[candidates.size()];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = candidates.id(i);
            }
            return new Offer(ids, probs);
        }

        /** The probability of {@code token} in the set, 0 when the set does not hold it. */
        float probOf(int token) {
            for (int i = 0; i < ids.length; i++) {
                if (ids[i] == token) {
                    return probs[i];
                }
            }
            return 0f;
        }
    }
}
