package com.eonmux.neuralfree.gguf.engine.draft;

/**
 * How a draft proposal is verified against the target (vLLM {@code config/speculative.py:83,538-545}
 * {@code rejection_sample_method}), as SpeculativeDecoder's {@code neuralfree.draft.verify} mode. Every mode offers
 * the drafter's proposal the same verify call; the modes differ in how the target's processed distributions turn
 * into emissions:
 *
 * <ul>
 *   <li>{@link #MATCH}: llama.cpp's target-sample-match (common/sampling.cpp:736
 *       {@code common_sampler_sample_and_accept_n}): each verify row is sampled through the unmodified chain and the
 *       longest prefix whose samples equal the draft tokens is kept, with the mismatching row's sample emitted. The
 *       default and the fallback of the probabilistic modes.</li>
 *   <li>{@link #REJECTION}: vLLM {@code standard} rejection sampling (v1/sample/rejection_sampler.py): a draft token
 *       is confirmed with probability {@code min(1, p/q)}, {@code p} the processed target distribution of its verify
 *       row and {@code q} the drafter's proposal distribution (one-hot for drafters without probabilities); the
 *       token after the first rejection is drawn from {@code norm(max(0, p − q))}; with the whole block confirmed
 *       the bonus token comes from {@code p}. Lossless: every emitted token is a draw from the processed target
 *       distribution at its position.</li>
 *   <li>{@link #BLOCK}: block verification (Sun et al., arXiv:2403.10444, Algorithm 3, as vLLM {@code block}): the
 *       draft is verified jointly as a block; lossless like {@link #REJECTION}.</li>
 *   <li>{@link #FLY}: FLy's entropy-gated deferred verification (vLLM {@code v1/spec_decode/fly.py}): a high-entropy
 *       rejection at a position whose next {@code W} draft tokens would be accepted natively is deferred, so its
 *       draft token is emitted instead of the standard replacement. <b>Lossy</b>: documented as changing the output
 *       distribution, never selective by default.</li>
 *   <li>{@link #SYNTHETIC}: benchmarking mode (vLLM {@code synthetic_acceptance_rates}/{@code
 *       synthetic_acceptance_length}): draft tokens are accepted with configured per-position rates instead of a
 *       real comparison; refused unless {@code neuralfree.spec.synthetic=true}.</li>
 * </ul>
 *
 * <p>A target chain that is greedy (temperature 0 or an explicit greedy request) degenerates every lossless mode
 * into {@link #MATCH}'s behavior: the processed target distribution is one-hot on the argmax, so the acceptance
 * ratio and residual draws reduce to the argmax match. {@link #FLY} keeps its deferral under a greedy chain, with
 * target-only acceptance and the entropy gate over the softmax of the processed logits.
 */
public enum VerifyMode {

    /** Longest-match verification, llama.cpp's target-sample-match (the default). */
    MATCH("match"),

    /** vLLM {@code standard}: probabilistic rejection sampling, lossless. */
    REJECTION("rejection"),

    /** Block verification of the whole draft (Sun et al.), lossless. */
    BLOCK("block"),

    /** FLy's entropy-gated deferred verification; lossy. */
    FLY("fly"),

    /** Benchmarking mode with configured acceptance rates; refused unless {@code neuralfree.spec.synthetic=true}. */
    SYNTHETIC("synthetic");

    private final String name;

    VerifyMode(String name) {
        this.name = name;
    }

    /** {@code mode}, case-insensitively; null recovers the default {@link #MATCH}. */
    public static VerifyMode parse(String mode) {
        if (mode == null) {
            return MATCH;
        }
        String s = mode.trim();
        for (VerifyMode v : values()) {
            if (v.name.equalsIgnoreCase(s)) {
                return v;
            }
        }
        StringBuilder known = new StringBuilder();
        for (VerifyMode v : values()) {
            if (known.length() > 0) {
                known.append(", ");
            }
            known.append(v.name);
        }
        throw new IllegalArgumentException("unknown verification mode '" + mode.trim() + "'; expected one of: "
                + known);
    }

    @Override
    public String toString() {
        return name;
    }
}
