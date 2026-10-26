package com.eonmux.neuralfree.gguf.engine.draft;

/**
 * FLy's entropy-gated deferred verification (vLLM {@code v1/spec_decode/fly.py}, ModelRunner pre-passes). A native
 * rejection at position {@code i} is <em>deferred</em> — its draft token is emitted instead of the standard
 * replacement — when the position's top-k entropy of the target's processed distribution reaches
 * {@code fly_entropy_threshold} and the following {@code W} draft tokens would be accepted natively
 * ({@code fly_window_size}); the deferral needs the {@code W} future draft tokens inside the block
 * ({@code i + W < k}). Native acceptance is the {@link RejectionVerifier} test between the target's and the
 * drafter's distribution, or, for a greedy target chain, the target's argmax match; in the greedy case the row's
 * entropy is taken over the softmax of the processed logits, as fly.py does for target-only acceptance.
 *
 * <p><b>FLy is intentionally lossy</b>: the deferred token replaces the target's own choice or residual and is
 * documented as changing the output distribution (vLLM's FLy docs measure the quality it retains). It consumes one
 * uniform per draft position in position order (greedy chains draw none for the native checks) plus one for a
 * replacement's residual and none for a bonus, which is the chain's own draw of the bonus row.
 */
public final class FlyVerifier implements Verifier {

    private final int window;
    private final double threshold;
    private final int entropyTopK;
    private final boolean greedyTarget;

    /**
     * @param window       {@code fly_window_size}: the future native acceptance decisions a deferral checks, or 0
     *                     for vLLM's default of {@code min(6, k - 1)} of the round's proposal
     * @param threshold    {@code fly_entropy_threshold}: the entropy gate
     * @param entropyTopK  {@code fly_entropy_top_k}: the processed probabilities the top-k entropy covers
     * @param greedyTarget whether the target chain is greedy, in which case verification is target-only (argmax)
     * @throws IllegalArgumentException for a negative window, a negative threshold or a non-positive entropy top-k
     */
    public FlyVerifier(int window, double threshold, int entropyTopK, boolean greedyTarget) {
        if (window < 0) {
            throw new IllegalArgumentException("FLy window size must not be negative, got " + window);
        }
        if (!(threshold >= 0.0) || Double.isNaN(threshold)) {
            throw new IllegalArgumentException("FLy entropy threshold must not be negative, got " + threshold);
        }
        if (entropyTopK <= 0) {
            throw new IllegalArgumentException("FLy entropy top-k must be positive, got " + entropyTopK);
        }
        this.window = window;
        this.threshold = threshold;
        this.entropyTopK = entropyTopK;
        this.greedyTarget = greedyTarget;
    }

    @Override
    public void verify(Round round) {
        int k = round.draftCount();
        int w = window == 0 ? Math.min(6, k - 1) : Math.min(window, Math.max(k - 1, 0));
        Distribution[] ps = new Distribution[k + 1];
        Distribution[] qs = new Distribution[k + 1];
        boolean[] nativeAccepted = new boolean[k];
        for (int i = 0; i < k; i++) {
            ps[i] = round.target(i);
            qs[i] = round.draft(i);
            int d = round.draftToken(i);
            if (greedyTarget) {
                nativeAccepted[i] = ps[i].selectedToken() == d;
            } else {
                double pd = ps[i].probOf(d);
                double qd = qs[i].probOf(d);
                // as fly.py: only a proposal with positive draft probability can be accepted natively at all.
                double u = round.uniform();
                nativeAccepted[i] = qd > 0.0 && pd > 0.0 && u * qd <= pd;
            }
        }
        for (int i = 0; i < k; i++) {
            if (nativeAccepted[i] || defers(round, ps, nativeAccepted, i, k, w)) {
                if (round.confirm(i) != null) {
                    return;
                }
                continue;
            }
            if (round.emit(Verifier.residual(round, ps[i], qs[i], 1.0, 1.0)) != null) {
                return;
            }
            return;
        }
        if (round.emit(round.target(k).selectedToken()) != null) {
            return;
        }
    }

    /** fly.py's {@code can_defer}: a not-natively-accepted position survives the entropy gate and W-orbit window. */
    private boolean defers(Round round, Distribution[] ps, boolean[] nativeAccepted, int i, int k, int w) {
        if (i + w >= k) {
            return false;
        }
        // The decision needs the future draft tokens' native acceptances, which the pre-pass already drew.
        for (int o = 1; o <= w; o++) {
            if (!nativeAccepted[i + o]) {
                return false;
            }
        }
        Distribution p = ps[i];
        int d = round.draftToken(i);
        if (greedyTarget) {
            if (p.logitOf(d) == Float.NEGATIVE_INFINITY) {
                return false;
            }
        } else if (p.probOf(d) <= 0f) {
            return false;
        }
        return entropy(p, entropyTopK, greedyTarget) >= threshold;
    }

    /**
     * fly.py's {@code compute_fly_entropy}: the top-k entropy {@code -sum p·ln(p)} over the {@code topK} largest
     * processed probabilities, or, for a greedy target chain, over the softmax of the processed logits.
     */
    static double entropy(Distribution p, int topK, boolean fromLogits) {
        int n = p.size();
        int top = Math.min(topK, n);
        if (fromLogits) {
            double max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < n; i++) {
                if (p.logit(i) > max) {
                    max = p.logit(i);
                }
            }
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += Math.exp(p.logit(i) - max);
            }
            double lse = max + Math.log(sum);
            double h = 0.0;
            int[] order = topByLogit(p, top);
            for (int j = 0; j < top; j++) {
                if (order[j] < 0) {
                    continue;
                }
                double prob = Math.exp(p.logit(order[j]) - lse);
                if (prob > 0.0) {
                    h -= prob * Math.log(prob);
                }
            }
            return h;
        }
        int[] order = topByProb(p, top);
        double h = 0.0;
        for (int j = 0; j < top; j++) {
            if (order[j] < 0) {
                continue;
            }
            double prob = p.prob(order[j]);
            if (prob > 0.0) {
                h -= prob * Math.log(prob);
            }
        }
        return h;
    }

    /** The indices of the {@code top} largest processed probabilities, best first. */
    private static int[] topByProb(Distribution p, int top) {
        double[] keys = new double[p.size()];
        for (int i = 0; i < keys.length; i++) {
            keys[i] = p.prob(i);
        }
        return topIndices(keys, top);
    }

    /** The indices of the {@code top} largest processed logits, best first. */
    private static int[] topByLogit(Distribution p, int top) {
        double[] keys = new double[p.size()];
        for (int i = 0; i < keys.length; i++) {
            keys[i] = p.logit(i);
        }
        return topIndices(keys, top);
    }

    /** Selection sort over descending keys; ties keep their candidate order, which equal keys make immaterial. */
    private static int[] topIndices(double[] keys, int top) {
        int[] order = new int[top];
        for (int j = 0; j < top; j++) {
            int best = -1;
            for (int i = 0; i < keys.length; i++) {
                if (keys[i] <= Double.NEGATIVE_INFINITY) {
                    continue;
                }
                if (best < 0 || keys[i] > keys[best]) {
                    best = i;
                }
            }
            order[j] = best;
            if (best >= 0) {
                keys[best] = Double.NEGATIVE_INFINITY;
            }
        }
        return order;
    }

    @Override
    public String toString() {
        return "FlyVerifier{window=" + window + ", threshold=" + threshold + ", entropyTopK=" + entropyTopK
                + (greedyTarget ? ", target-only" : "") + '}';
    }
}
