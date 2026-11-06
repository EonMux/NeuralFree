package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.VerifyRound;
import com.eonmux.neuralfree.gguf.engine.draft.SyntheticVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.Verifier;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;

import java.util.Objects;

/**
 * The speculative settings of a {@link BatchEngine}: the most tokens a round drafts ({@code draftMax}, vLLM
 * {@code num_speculative_tokens}, llama.cpp {@code --draft-max}), the shortest proposal still verified
 * ({@code draftMin}, {@code --draft-min}), and how every sequence's verify rows decide its emissions
 * ({@link VerifyMode}, vLLM {@code rejection_sample_method}, with FLy's window and entropy gate and the synthetic
 * acceptance rates). Immutable; validated at construction.
 */
public final class BatchSpec {

    private final int draftMax;
    private final int draftMin;
    private final VerifyMode verifyMode;
    private final int flyWindow;
    private final double flyEntropyThreshold;
    private final int flyEntropyTopK;
    private final double[] syntheticRates;
    private final boolean treeVerify;
    private final int treeWidth;
    private final double treeTemp;
    private final TreeBudget treeBudget;
    private final int treeBranches;

    /**
     * Lengths verified by {@link VerifyMode#MATCH}.
     *
     * @throws IllegalArgumentException for a {@code draftMax} below 1, a negative {@code draftMin}, or a
     *                                  {@code draftMin} above {@code draftMax}
     */
    public BatchSpec(int draftMax, int draftMin) {
        this(draftMax, draftMin, VerifyMode.MATCH, SpeculativeOptions.defaults(), null);
    }

    /** @param rates the resolved synthetic acceptance rates, null outside {@link VerifyMode#SYNTHETIC} */
    private BatchSpec(int draftMax, int draftMin, VerifyMode mode, SpeculativeOptions options, double[] rates) {
        if (draftMax < 1) {
            throw new IllegalArgumentException("draft max must be at least 1, got " + draftMax);
        }
        if (draftMin < 0) {
            throw new IllegalArgumentException("draft min must not be negative, got " + draftMin);
        }
        if (draftMin > draftMax) {
            throw new IllegalArgumentException("draft min " + draftMin + " exceeds draft max " + draftMax);
        }
        this.draftMax = draftMax;
        this.draftMin = draftMin;
        this.verifyMode = Objects.requireNonNull(mode, "verify mode");
        this.flyWindow = options.flyWindow();
        this.flyEntropyThreshold = options.flyEntropyThreshold();
        this.flyEntropyTopK = options.flyEntropyTopK();
        this.syntheticRates = rates;
        this.treeVerify = options.treeVerify();
        if (treeVerify && mode != VerifyMode.MATCH) {
            throw new IllegalArgumentException(SpeculativeOptions.PROP_TREE + "=true verifies tree proposals by"
                    + " target-sample match, which " + SpeculativeOptions.PROP_VERIFY + "=" + mode + " does not"
                    + " run over a tree's rows; set " + SpeculativeOptions.PROP_VERIFY + "=match or turn the tree"
                    + " off");
        }
        this.treeWidth = Math.min(draftMax, options.treeNodes());
        this.treeTemp = options.treeTemp();
        this.treeBudget = treeVerify ? TreeBudget.of(options) : null;
        this.treeBranches = options.treeBranches();
    }

    /**
     * The lengths with the verification of {@code options} ({@code neuralfree.draft.verify} and its FLy and
     * synthetic settings): synthetic acceptance resolves its length over {@code draftMax} positions and needs the
     * {@code neuralfree.spec.synthetic=true} opt-in.
     *
     * @throws IllegalArgumentException for invalid lengths or synthetic settings
     * @throws IllegalStateException    for synthetic verification without the opt-in, naming the property
     */
    public static BatchSpec of(int draftMax, int draftMin, SpeculativeOptions options) {
        Objects.requireNonNull(options, "options");
        VerifyMode mode = options.verifyMode();
        double[] rates = null;
        if (mode == VerifyMode.SYNTHETIC) {
            if (!Boolean.getBoolean(SyntheticVerifier.PROP_ENABLED)) {
                throw new IllegalStateException(SyntheticVerifier.PROP_ENABLED + "=true is required for synthetic"
                        + " verification (" + SpeculativeOptions.PROP_VERIFY + "=" + mode + ")");
            }
            rates = options.synthRates() != null ? options.synthRates()
                    : SyntheticVerifier.ratesForLength(options.synthLength(), draftMax);
            new SyntheticVerifier(rates); // validates the resolved rates
        }
        return new BatchSpec(draftMax, draftMin, mode, options, rates);
    }

    /** Most tokens proposed per round. */
    public int draftMax() {
        return draftMax;
    }

    /** Shortest proposal that is verified; shorter proposals decode plain. */
    public int draftMin() {
        return draftMin;
    }

    /** How every sequence's verify rows decide its emissions. */
    public VerifyMode verifyMode() {
        return verifyMode;
    }

    /** Whether every decoding sequence's round verifies a draft tree (one masked forward per sequence). */
    public boolean treeVerify() {
        return treeVerify;
    }

    /** The most nodes one sequence's tree proposal holds: {@code draftMax} within {@code treeNodes}. */
    public int treeWidth() {
        return treeWidth;
    }

    /** The temperature a tree builder scales its path scores at. */
    public double treeTemp() {
        return treeTemp;
    }

    /** The step's verify budget over the tree proposals, or null outside tree verify. */
    TreeBudget treeBudget() {
        return treeBudget;
    }

    /** The branch budget a draft model splits a request's tree over. */
    public int treeBranches() {
        return treeBranches;
    }

    /** The budget policy's name for the engine's startup log ({@code full} outside tree verify). */
    public String treeBudgetPolicyName() {
        return treeBudget == null ? TreeBudgetPolicy.FULL.name().toLowerCase(java.util.Locale.ROOT)
                : treeBudget.policy().name().toLowerCase(java.util.Locale.ROOT);
    }

    /**
     * The verifier of one sequence (null for {@link VerifyMode#MATCH}, whose rounds sample through the chain).
     *
     * @throws IllegalArgumentException when the sequence's chain ends in {@code adaptive_p} under another mode
     *                                  (a mirostat chain, which adapts on the accepted token and builds no
     *                                  {@code adaptive_p}, runs in every mode)
     */
    Verifier verifierFor(SamplerConfig config, Sampler sampler) {
        return VerifyRound.verifierFor(verifyMode, config, sampler, flyWindow, flyEntropyThreshold, flyEntropyTopK,
                syntheticRates);
    }

    @Override
    public String toString() {
        return "BatchSpec{draftMax=" + draftMax + ", draftMin=" + draftMin + ", verify=" + verifyMode + "}";
    }
}
