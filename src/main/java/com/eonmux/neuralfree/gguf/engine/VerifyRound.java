package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.draft.BlockVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.FlyVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.RejectionVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.SyntheticVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.Verifier;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;

import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.BooleanSupplier;
import java.util.function.IntFunction;

/**
 * One verify round of a non-{@code MATCH} {@link Verifier} over a sequence's request chain: the processed row
 * distributions on demand, the drafter's proposal distributions, the round's uniform stream and its emissions
 * through the sequence's stop session. {@link SpeculativeDecoder} runs one per round of its single sequence, and
 * the batch engine one per speculating sequence of a verify step ({@link #forRun}). Rows are handed out in
 * ascending order at most once each; an emission's non-null stop reason ends the round, and so does the emission of
 * a replacement or bonus.
 *
 * <p><b>Rows under the request's chain.</b> Row {@code r} is the chain's distribution after the batched tokens
 * before it, so its stateful stages (grammar, penalties, DRY, min-tokens, bad-words, reasoning budget) must have
 * seen draft tokens {@code 0 .. r-1}, and mirostat must hold the surprise target those tokens left. A verifier that
 * asks for a row past the confirmed prefix (block verification needs every row before deciding) has the sampler
 * accept the draft tokens before that row ahead of their confirmation, with a {@link Sampler#saveState} before
 * each, and an emission at a position whose draft token is not kept restores the chain to the state saved at that
 * position before accepting the emitted token (vLLM's grammar advance over the speculative tokens with its
 * {@code rollback}). Every row is run through the chain exactly once, in order, so the chain state behind every
 * emitted token equals the plain loop's; a restored state also holds mirostat's candidate set of the row it was
 * saved after, so mirostat adapts {@code mu} on each emitted token from the row the token was emitted at, and on no
 * row read ahead. A draft token the chain excludes at its row (a {@code -inf} processed logit: masked by the
 * grammar, a ban, bad words, min-tokens or truncated away) is never accepted ahead; the rows after it are scored
 * without it and can decide nothing, since no lossless verifier confirms past an excluded token (vLLM likewise
 * leaves the rows after a grammar-invalid draft token unconstrained). A confirm of an excluded token is a verifier
 * fault. Every emission is scored over its verify row (the raw row, or its chain distribution {@code scored[r]}
 * under a processed mode) by the request's {@link LogprobCapture}.
 *
 * <p>Mutable, used once, on the thread that owns the sequence.
 */
public final class VerifyRound implements Verifier.Round {

    /** Where a round's emissions go: the sequence's stop session with its logprobs capture. */
    interface Emitter {
        /** Whether the request was cancelled; a cancelled emission stops the round with {@code CANCELLED}. */
        boolean cancelled();

        /**
         * Offers an emitted token, already accepted into the sampler, with the raw verify row it was emitted at and
         * that row's chain distribution — what the request's {@link LogprobCapture} scores it over; returns the
         * stop reason, or null to continue.
         */
        StopReason offer(int token, float[] row, Verifier.Distribution processed);
    }

    private final Sampler sampler;
    private final Emitter emitter;
    private final int[] draft;
    private final int k;
    private final float[][] rows;
    private final int rowBase;
    private final SplittableRandom rng;
    private final IntFunction<float[]> draftProbs;
    private final Verifier.Distribution[] draftDistributions;
    /** Rows {@code [0, scoredRows)} ran through the chain; their distributions. */
    private final Verifier.Distribution[] scored;
    /** {@code ahead[j]}: the chain's saved state before draft token {@code j} was accepted ahead. */
    private final Object[] ahead;
    private int scoredRows;
    /** Draft tokens the sampler holds, confirmed or accepted ahead. */
    private int absorbed;
    /** The first draft token the chain excludes at its row, or -1. */
    private int excluded = -1;
    private int lastRow = -1;
    private boolean closed;
    private int accepted;
    private int emitted;
    private int next = -1;
    private StopReason reason;

    /**
     * @param rows       the verify rows: row {@code i} of the round is {@code rows[rowBase + i]}, {@code i <= k}
     * @param draftProbs the drafter's proposal row of a draft position, or null for one-hot proposals (drafters
     *                   without probabilities); a function answering null for a position makes that position
     *                   one-hot
     */
    VerifyRound(Sampler sampler, Emitter emitter, int[] draft, float[][] rows, int rowBase, SplittableRandom rng,
                IntFunction<float[]> draftProbs) {
        this.sampler = Objects.requireNonNull(sampler, "sampler");
        this.emitter = Objects.requireNonNull(emitter, "emitter");
        this.draft = Objects.requireNonNull(draft, "draft");
        this.k = draft.length;
        this.rows = Objects.requireNonNull(rows, "rows");
        if (rowBase < 0 || rowBase + k >= rows.length) {
            throw new IllegalArgumentException("verify rows " + rowBase + ".." + (rowBase + k) + " outside the "
                    + rows.length + " rows given");
        }
        this.rowBase = rowBase;
        this.rng = Objects.requireNonNull(rng, "rng");
        this.draftProbs = draftProbs;
        this.draftDistributions = new Verifier.Distribution[k];
        this.scored = new Verifier.Distribution[k + 1];
        this.ahead = new Object[k];
    }

    /**
     * A round over a {@link SequenceRun}'s chain and stop session (the batch engine's sequences): every emission
     * is counted by the run, captures its row's logprobs when the request asked for them, and is offered to the
     * run's session.
     *
     * @param cancelled  the round's cancellation poll (the request's hook and any engine-wide cancel)
     * @param rows       the verify rows: row {@code i} of the round is {@code rows[rowBase + i]}
     * @param draftProbs the drafter's proposal row of a draft position, or null for one-hot proposals
     */
    public static VerifyRound forRun(SequenceRun run, BooleanSupplier cancelled, int[] draft, float[][] rows,
                                     int rowBase, SplittableRandom rng, IntFunction<float[]> draftProbs) {
        Objects.requireNonNull(run, "run");
        Objects.requireNonNull(cancelled, "cancelled");
        Emitter emitter = new Emitter() {
            @Override
            public boolean cancelled() {
                return cancelled.getAsBoolean();
            }

            @Override
            public StopReason offer(int token, float[] row, Verifier.Distribution processed) {
                return run.offerVerified(token, row, processed).reason();
            }
        };
        return new VerifyRound(run.sampler(), emitter, draft, rows, rowBase, rng, draftProbs);
    }

    /**
     * The verifier of {@code mode} for one generation (stateless, shared across its rounds), or null for
     * {@link VerifyMode#MATCH}, whose rounds sample through the chain itself.
     *
     * @param syntheticRates the resolved per-position rates of {@link VerifyMode#SYNTHETIC}; unused otherwise
     * @throws IllegalArgumentException when the chain ends in {@code adaptive_p}: it folds only its own draw into
     *                                  its moving average, which a verifier's emissions replace, so only
     *                                  {@code MATCH} reproduces it (a mirostat chain, which adapts on the accepted
     *                                  token and builds no {@code adaptive_p}, runs in every mode)
     */
    public static Verifier verifierFor(VerifyMode mode, SamplerConfig config, Sampler sampler, int flyWindow,
                                       double flyEntropyThreshold, int flyEntropyTopK, double[] syntheticRates) {
        Objects.requireNonNull(mode, "mode");
        if (mode == VerifyMode.MATCH) {
            return null;
        }
        refuseAdaptiveP(mode, config);
        switch (mode) {
            case REJECTION:
                return new RejectionVerifier();
            case BLOCK:
                return new BlockVerifier();
            case FLY:
                return new FlyVerifier(flyWindow, flyEntropyThreshold, flyEntropyTopK, sampler.greedy());
            case SYNTHETIC:
                return new SyntheticVerifier(syntheticRates);
            default:
                throw new IllegalStateException("no verifier for verification mode " + mode);
        }
    }

    /**
     * Refuses a chain that ends in {@code adaptive_p} under a verify mode other than {@link VerifyMode#MATCH},
     * naming the setting. Mirostat runs in every mode: it adapts {@code mu} on the emitted token's surprise in its
     * row's candidate set, so a verifier's emissions adapt it as the plain loop's draws would.
     *
     * @throws IllegalArgumentException for such a chain
     */
    public static void refuseAdaptiveP(VerifyMode mode, SamplerConfig config) {
        if (mode == VerifyMode.MATCH) {
            return;
        }
        boolean mirostat = config.mirostat() == 1 || config.mirostat() == 2;
        if (!mirostat && config.order().contains(StageKind.ADAPTIVE_P)) {
            throw new IllegalArgumentException("adaptive_p selects its token itself; verify mode " + mode
                    + " cannot verify it (use verify mode MATCH)");
        }
    }

    @Override public int draftCount() {
        return k;
    }

    @Override public int draftToken(int position) {
        checkDraftPosition(position);
        return draft[position];
    }

    @Override
    public Verifier.Distribution target(int row) {
        if (row <= lastRow || row > k) {
            throw new IllegalStateException("verify row " + row + " after last-row " + lastRow + " (may be"
                    + " requested in ascending order, once, up to " + k + ")");
        }
        lastRow = row;
        return score(row);
    }

    /** Row {@code row}'s distribution, running the chain over the rows up to it first. */
    private Verifier.Distribution score(int row) {
        while (scoredRows <= row) {
            scoreNext();
        }
        return scored[row];
    }

    /**
     * Runs the chain over the next row. A row past the confirmed prefix first saves the chain's state, then accepts
     * the unconfirmed draft token before it ahead when the chain admits it.
     */
    private void scoreNext() {
        int r = scoredRows;
        int j = r - 1;
        if (j >= accepted) {
            ahead[j] = sampler.saveState();
            if (excluded < 0 && admits(j)) {
                sampler.accept(draft[j]);
                absorbed++;
            } else if (excluded < 0) {
                excluded = j;
            }
        }
        Sampler.Candidates cur = sampler.apply(rows[rowBase + r]);
        scored[r] = distribution(cur, sampler);
        scoredRows++;
    }

    /** Whether the chain keeps draft token {@code position} in its scored row (a finite processed logit). */
    private boolean admits(int position) {
        return scored[position].logitOf(draft[position]) != Float.NEGATIVE_INFINITY;
    }

    @Override
    public Verifier.Distribution draft(int position) {
        checkDraftPosition(position);
        Verifier.Distribution built = draftDistributions[position];
        if (built == null) {
            float[] row = draftProbs == null ? null : draftProbs.apply(position);
            built = row == null ? Verifier.Distribution.oneHot(draft[position])
                    : proposalDistribution(row, draft[position], position);
            draftDistributions[position] = built;
        }
        return built;
    }

    /**
     * The drafter's proposal row of {@code position} as a distribution over its positive entries.
     *
     * @throws IllegalStateException when the row does not cover the drafted token, holds NaN, or gives the drafted
     *                               token no probability
     */
    private static Verifier.Distribution proposalDistribution(float[] row, int token, int position) {
        if (token >= row.length) {
            throw new IllegalStateException("the drafter proposed token " + token + " at position " + position
                    + " but offered " + row.length + " proposal probabilities");
        }
        int support = 0;
        for (float v : row) {
            if (Float.isNaN(v)) {
                throw new IllegalStateException("the drafter's proposal probability row contains NaN");
            }
            support += v > 0f ? 1 : 0;
        }
        if (row[token] <= 0f) {
            throw new IllegalStateException("the drafter drafted token " + token + " with proposal probability "
                    + row[token]);
        }
        int[] tokens = new int[support];
        float[] probs = new float[support];
        float[] logits = new float[support];
        int j = 0;
        for (int i = 0; i < row.length; i++) {
            if (row[i] > 0f) {
                tokens[j] = i;
                probs[j] = row[i];
                j++;
            }
        }
        return new Verifier.Distribution(tokens, probs, logits, -1);
    }

    @Override public double uniform() {
        return rng.nextDouble();
    }

    /**
     * The replacement of a rejection under a watermarked chain: vLLM's resample kernel draws the residual with the
     * target key, so the emitted token is detectable like the chain's own draws. The keyed draw replaces {@link
     * Verifier#residual}'s uniform draw; a repeated context keeps the ordinary draw, as the sampler's own selection
     * does.
     */
    @Override
    public int resample(Verifier.Distribution p, Verifier.Distribution q, double a, double b) {
        if (sampler.watermarked() && !sampler.watermarkRepeatedContext()) {
            int n = p.size();
            int[] tokenIds = new int[n];
            float[] scores = new float[n];
            int count = 0;
            for (int i = 0; i < n; i++) {
                double weight = a * p.prob(i) - b * q.probOf(p.token(i));
                if (weight > 0.0) {
                    tokenIds[count] = p.token(i);
                    scores[count] = (float) Math.log(weight);
                    count++;
                }
            }
            if (count > 0) {
                return sampler.watermarkedDraw(tokenIds, scores, count);
            }
        }
        return Verifier.residual(this, p, q, a, b);
    }

    @Override public StopReason confirm(int position) {
        requireOpen();
        if (position != accepted) {
            throw new IllegalStateException("confirm(" + position + ") after " + accepted
                    + " confirmed draft tokens");
        }
        score(position);
        if (!admits(position)) {
            throw new IllegalStateException("the verifier confirmed draft token " + draft[position]
                    + " at position " + position + ", which the request's sampler chain excludes there");
        }
        // the confirmed position is kept in the memory either way; the rollback accounting
        // (drop = k - accepted) counts it as kept
        accepted = position + 1;
        if (emitter.cancelled()) {
            reason = StopReason.CANCELLED;
            return reason;
        }
        if (absorbed == position) {
            sampler.accept(draft[position]);
            absorbed++;
        }
        return offer(position, draft[position]);
    }

    @Override public StopReason emit(int token) {
        requireOpen();
        int position = accepted;
        score(position);
        closed = true;
        rewind(position);
        if (emitter.cancelled()) {
            reason = StopReason.CANCELLED;
            return reason;
        }
        sampler.accept(token);
        return offer(position, token);
    }

    /** Emits {@code token} at {@code position}: its logprobs capture, then the session. */
    private StopReason offer(int position, int token) {
        emitted++;
        next = token;
        reason = emitter.offer(token, rows[rowBase + position], scored[position]);
        return reason;
    }

    @Override public int accepted() {
        return accepted;
    }

    @Override public boolean stopped() {
        return reason != null;
    }

    /** Tokens the round emitted: confirmed draft tokens plus a replacement or bonus. */
    public int emitted() {
        return emitted;
    }

    /** The last emitted token (the target's next token when the round did not stop), or -1. */
    public int next() {
        return next;
    }

    /** The stop reason an emission or a cancellation reported, or null while the generation continues. */
    public StopReason reason() {
        return reason;
    }

    /**
     * Ends the round after {@code verifier} returned: the chain drops the draft tokens it holds past the confirmed
     * prefix.
     *
     * @throws IllegalStateException when the verifier neither stopped nor emitted the target's next token
     */
    public void finish(Verifier verifier) {
        if (reason == null && !closed) {
            throw new IllegalStateException(verifier + " ended the verify round without emitting the target's"
                    + " next token");
        }
        if (!closed) {
            rewind(accepted);
        }
    }

    /**
     * Returns the chain to its state after row {@code position} when rows past it ran: the draft tokens from
     * {@code position} on leave the chain, and so does every later row's apply.
     */
    private void rewind(int position) {
        if (position < k && scoredRows > position + 1) {
            sampler.restoreState(ahead[position]);
            absorbed = Math.min(absorbed, position);
        }
    }

    private void requireOpen() {
        if (reason != null) {
            throw new IllegalStateException("the verify round already stopped with " + reason);
        }
        if (closed) {
            throw new IllegalStateException("the verify round already emitted the target's next token");
        }
    }

    private void checkDraftPosition(int position) {
        if (position < 0 || position >= k) {
            throw new IndexOutOfBoundsException("draft position " + position + " outside [0, " + k + ")");
        }
    }

    /**
     * The processed distribution of the chain run that left {@code cur}: the surviving candidates with their
     * post-chain probabilities and logits and the chain's own draw. A greedy chain keeps no normalized
     * probabilities; its distribution is one-hot on the argmax (which leaves the argmax rule of the lossless modes
     * intact), while the logits stay the chain's for FLy's entropy gate.
     */
    static Verifier.Distribution distribution(Sampler.Candidates cur, Sampler sampler) {
        int n = cur.size();
        int[] tokens = new int[n];
        float[] probs = new float[n];
        float[] logits = new float[n];
        for (int i = 0; i < n; i++) {
            tokens[i] = cur.id(i);
            logits[i] = cur.logit(i);
        }
        if (sampler.greedy()) {
            probs[cur.selected()] = 1f;
        } else {
            for (int i = 0; i < n; i++) {
                probs[i] = cur.prob(i);
            }
        }
        return new Verifier.Distribution(tokens, probs, logits, cur.selected());
    }
}
