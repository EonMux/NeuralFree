package com.eonmux.neuralfree.gguf.engine.draft;

/**
 * Benchmarking verification (vLLM {@code rejection_sample_method} {@code synthetic} with
 * {@code synthetic_acceptance_rates} / {@code synthetic_acceptance_length}): draft tokens are accepted with
 * configured per-position rates instead of a real comparison against the target. Entry {@code i} of the rates is
 * the marginal probability that the first {@code i + 1} draft tokens are all accepted; the round then accepts
 * position {@code i} with the conditional rate {@code rates[i]/rates[i-1]} (one uniform per position, in position
 * order). A rejection at position {@code i} — or the bonus after a fully accepted block — emits the same token the
 * target chain itself sampled; the rates only fake the speed-up. A draft token the target chain excludes at its
 * row (a {@code -inf} processed logit: a grammar mask, a ban, bad words, min-tokens, a truncation) is rejected
 * whatever its uniform, so the output stays one the request's chain admits (vLLM trims the proposal to its
 * grammar-valid prefix before verification). The mode changes no emission law: it is for throughput benchmarking with
 * calibrated acceptance rates, and {@code SpeculativeDecoder} refuses it unless
 * {@code neuralfree.spec.synthetic=true}.
 */
public final class SyntheticVerifier implements Verifier {

    /** The opt-in property ({@code neuralfree.spec.synthetic}) that unlocks synthetic verification. */
    public static final String PROP_ENABLED = "neuralfree.spec.synthetic";

    private final double[] rates;

    /**
     * @param rates the unconditional per-position acceptance rates; entry {@code i} is the marginal probability
     *              that the first {@code i + 1} draft tokens are all accepted: length at least 1, each entry in
     *              {@code [0, 1]}, monotonically non-increasing (copied)
     */
    public SyntheticVerifier(double[] rates) {
        if (rates == null || rates.length == 0) {
            throw new IllegalArgumentException("synthetic acceptance rates must have at least one entry");
        }
        for (int i = 0; i < rates.length; i++) {
            double r = rates[i];
            if (!(r >= 0.0) || !(r <= 1.0) || Double.isNaN(r)) {
                throw new IllegalArgumentException("synthetic acceptance rate " + i + " = " + r
                        + " is outside [0, 1]");
            }
            if (i > 0 && r > rates[i - 1]) {
                throw new IllegalArgumentException("synthetic acceptance rates must be non-increasing, got entry "
                        + i + " = " + r + " after " + rates[i - 1]);
            }
        }
        this.rates = rates.clone();
    }

    /**
     * vLLM {@code _acceptance_length_to_rates}: the unconditional per-position rates whose expected acceptance
     * length is {@code length}: {@code length - 1} full positions at 1.0, then the remainder, then entries beyond
     * the mean acceptance length at 0 (the minimum-variance schedule).
     *
     * @param length the target mean acceptance length in {@code [1, numProposals + 1]}
     */
    public static double[] ratesForLength(double length, int numProposals) {
        if (!(length >= 1.0) || length > numProposals + 1.0) {
            throw new IllegalArgumentException("synthetic acceptance length must be in [1, " + (numProposals + 1)
                    + "], got " + length);
        }
        double drafts = length - 1.0;
        int full = (int) drafts;
        double[] rates = new double[numProposals];
        int i = 0;
        for (; i < full && i < numProposals; i++) {
            rates[i] = 1.0;
        }
        if (i < numProposals) {
            rates[i++] = drafts - full;
        }
        return rates;
    }

    @Override
    public void verify(Round round) {
        int k = round.draftCount();
        double previous = 1.0;
        for (int i = 0; i < k; i++) {
            double rate = i < rates.length ? rates[i] : 0.0;
            double conditional = previous > 0.0 ? rate / previous : 0.0;
            if (conditional > 1.0) {
                conditional = 1.0; // monotone rates keep this at 1.0; rounding may not
            }
            double u = round.uniform();
            Distribution p = round.target(i);
            if (u <= conditional && p.logitOf(round.draftToken(i)) != Float.NEGATIVE_INFINITY) {
                if (round.confirm(i) != null) {
                    return;
                }
                previous = rate;
                continue;
            }
            // the acceptance decision was synthetic: the replacement is a plain target draw.
            if (round.emit(p.selectedToken()) != null) {
                return;
            }
            return;
        }
        if (round.emit(round.target(k).selectedToken()) != null) {
            return;
        }
    }

    /** The configured rates (unmodifiable copy). */
    public double[] rates() {
        return rates.clone();
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder("SyntheticVerifier{rates=[");
        for (int i = 0; i < rates.length; i++) {
            if (i > 0) {
                s.append(", ");
            }
            s.append(rates[i]);
        }
        return s.append("]}").toString();
    }
}
