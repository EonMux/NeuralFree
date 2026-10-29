package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.draft.BlockVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.DraftProbs;
import com.eonmux.neuralfree.gguf.engine.draft.FlyVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.RejectionVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.SyntheticVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.Verifier;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Consumer;

/**
 * Speculative generation over a target {@link ForwardBackend} and a {@link DraftModel}: the drafter proposes tokens,
 * the target scores them all in one forward call and keeps the longest prefix it would have sampled itself (llama.cpp
 * tools/server speculative loop with {@code common_sampler_sample_and_accept_n}, common/sampling.cpp). Immutable;
 * one generation runs at a time, on the backend's thread.
 *
 * <p>Per {@link #generate} call:
 * <ol>
 *   <li><b>Model defaults.</b> The request takes the decoder's {@link SamplingDefaults} for every sampler setting
 *       and the token budget it leaves unset, as in {@link GenerationLoop}.</li>
 *   <li><b>Memory.</b> The target memory and the drafter are reset: the drafter's state follows the target's hidden
 *       rows from the first prompt token, so a reused prefix would leave it without them. A prompt longer than the
 *       context throws {@link ContextFullException}.</li>
 *   <li><b>Prefill.</b> The prompt is prefilled and enters the penalty history, as in {@link GenerationLoop}. The
 *       first token is sampled from its logits and offered to the {@link StopCriteria} session.</li>
 *   <li><b>Rounds.</b> With the last sampled token {@code t} at the next position {@code P = nPast}, the drafter
 *       proposes up to {@code k} tokens {@code d}. Without a usable proposal (empty, or shorter than
 *       {@code draftMin}) the round decodes {@code t} and samples one token. Otherwise
 *       {@link ForwardBackend#forwardRows} evaluates {@code [t, d_0 .. d_{k-1}]} at {@code P ..} and returns all
 *       {@code k + 1} logits rows, whose processed distributions the verify mode decides over:</li>
 * </ol>
 * <p><b>Verification modes</b> ({@link VerifyMode}, the builder's {@code verifyMode}, default {@link VerifyMode#MATCH}):
 * <ul>
 *   <li>{@link VerifyMode#MATCH}: the longest matching prefix of the chain's own samples (llama.cpp
 *       common/sampling.cpp:736 {@code common_sampler_sample_and_accept_n}), as described next.</li>
 *   <li>{@link VerifyMode#REJECTION}: vLLM {@code standard} ({@link RejectionVerifier});</li>
 *   <li>{@link VerifyMode#BLOCK}: Sun et al. block verification ({@link BlockVerifier});</li>
 *   <li>{@link VerifyMode#FLY}: entropy-gated deferred verification ({@link FlyVerifier}, lossy);</li>
 *   <li>{@link VerifyMode#SYNTHETIC}: benchmarking acceptance ({@link SyntheticVerifier}), refused unless the
 *       system property {@code neuralfree.spec.synthetic} is {@code true} at build time.</li>
 * </ul>
 * <p>In {@code MATCH} row {@code i} is sampled, accepted into the sampler and offered to the session,
 * continuing while the sample equals {@code d_i}. In every other mode a {@link Verifier} runs the round against
 * the row distributions; each verifier documents its own RNG stream, and a drafter implementing
 * {@link DraftProbs} supplies its proposal distributions (one-hot otherwise). A greedy target chain degenerates
 * the lossless modes into {@code MATCH}'s argmax rule. A request with {@code logprobs(n)} captures the post-chain
 * distribution of every emitted token under the default {@code MATCH} mode and the plain decode steps, the same
 * capture the plain loop makes; the lossless verifier modes' emissions report no scores yet ([GAPS.md](GAPS.md)).
 *
 * <p>With {@code a} drafted tokens confirmed, the round emits
 * {@code d_0 .. d_{a-1}} and the target's next token, and the last {@code k - a} positions leave memory:
 * by {@link ForwardBackend#rollback}, or for recurrent memory by restoring the snapshot taken before the
 * call and prefilling {@code [t, d_0 .. d_{a-1}]} again. A stop inside the verify (end of generation, stop string,
 * budget, cancel) leaves the same state as the ordinary rejection path: the extra verified positions are dropped
 * or replayed before the round returns, never left in the memory as an overshot state (PARITY §5.5.1 S-1). The
 * drafter then hears {@code accept(a)}.
 * </ol>
 * Every emitted token passes through the same {@link Sampler} and {@link StopCriteria.Session} as in
 * {@link GenerationLoop}: banned tokens, penalties, stop strings, the token budget and reasoning routing are
 * unchanged. {@code MATCH} with an argmax sampler equals plain greedy decoding; with a stochastic chain the
 * lossless modes ({@code MATCH}, {@code REJECTION}, {@code BLOCK}) keep every emitted token a draw from the
 * target's distribution at its position.
 *
 * <p>{@code k = min(draftMax, nCtx - P - 1, ubatch - 1, remaining - 1)}, lowered to the largest {@code k} for which
 * every sliding-window ring still holds the window of the new end after storing {@code k + 1} positions and dropping
 * {@code k} ({@link ModelMemory#ringsKeepWindow}). With DSV4 compressed caches {@code k} is at most the memory's
 * compressor rollback snapshots ({@link MemoryPlan#rollbackSnapshots()}, llama.cpp {@code n_rs_seq}, which the server
 * likewise bounds the draft by, tools/server/server-context.cpp:3805-3830), so 0 without snapshots; recurrent state
 * that cannot be snapshot ({@link ModelMemory#canSnapshot}) gives 0.
 * A full context ends generation with {@link StopReason#CONTEXT_FULL}; the context does not shift. When a
 * round stops (end of generation, stop string, budget), the verify call's extra positions stay in memory; the next
 * generation resets it.
 *
 * <p>When a forward call, the drafter or the sampler fails, the target memory and the drafter are reset before the
 * exception propagates.
 */
public final class SpeculativeDecoder {

    private static final ReasoningFilter.Sink DISCARD = (channel, tokenId, text) -> { };

    /** Counts of one generation's speculation. Immutable. */
    public static final class Stats {
        private final int rounds;
        private final int drafted;
        private final int accepted;

        Stats(int rounds, int drafted, int accepted) {
            this.rounds = rounds;
            this.drafted = drafted;
            this.accepted = accepted;
        }

        /** Verify calls made (rounds with a usable proposal). */
        public int rounds() { return rounds; }

        /** Proposed tokens the target verified. */
        public int drafted() { return drafted; }

        /** Proposed tokens the target confirmed. */
        public int accepted() { return accepted; }

        /** {@code accepted / drafted}, 0 when nothing was drafted. */
        public double acceptanceRate() {
            return drafted == 0 ? 0.0 : (double) accepted / drafted;
        }

        @Override
        public String toString() {
            return "Stats{rounds=" + rounds + ", drafted=" + drafted + ", accepted=" + accepted + '}';
        }
    }

    private final ForwardBackend backend;
    private final DraftModel drafter;
    private final GGUFTokenizer tokenizer;
    private final ReasoningFilter reasoning;
    private final int draftMax;
    private final int draftMin;
    private final Consumer<Stats> statsSink;
    private final SamplingDefaults samplingDefaults;
    private final VerifyMode verifyMode;
    private final int flyWindow;
    private final double flyEntropyThreshold;
    private final int flyEntropyTopK;
    private final double[] syntheticRates;
    private final Double syntheticLength;

    private SpeculativeDecoder(Builder b) {
        this.backend = b.backend;
        this.drafter = b.drafter;
        this.tokenizer = b.tokenizer;
        this.reasoning = b.reasoning;
        this.draftMax = b.draftMax;
        this.draftMin = b.draftMin;
        this.statsSink = b.statsSink;
        this.samplingDefaults = b.samplingDefaults;
        this.verifyMode = b.verifyMode;
        this.flyWindow = b.flyWindow;
        this.flyEntropyThreshold = b.flyEntropyThreshold;
        this.flyEntropyTopK = b.flyEntropyTopK;
        this.syntheticRates = b.syntheticRates;
        this.syntheticLength = b.syntheticLength;
        if (draftMin > draftMax) {
            throw new IllegalArgumentException("draft min " + draftMin + " exceeds draft max " + draftMax);
        }
    }

    public VerifyMode verifyMode() {
        return verifyMode;
    }

    /**
     * @param backend the target's forward backend
     * @param drafter a drafter bound to that target ({@link DraftModel#bindTarget})
     */
    public static Builder builder(ForwardBackend backend, DraftModel drafter, GGUFTokenizer tokenizer) {
        return new Builder(backend, drafter, tokenizer);
    }

    /**
     * The decoder of an engine model and its bound drafter: the reasoning parser of the model's chat template (with
     * its tag tokens), the draft lengths of {@code config} ({@code neuralfree.draft.max},
     * {@code neuralfree.draft.min}) and the model's sampling defaults unless {@code config} turns them off
     * ({@link SamplingDefaults#forModel}).
     *
     * @throws UnsupportedOperationException for models that do not generate autoregressively with a decoder-only
     *                                       memory
     */
    public static SpeculativeDecoder forModel(EngineModel model, DraftModel drafter, EngineConfig config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(drafter, "drafter");
        Objects.requireNonNull(config, "config");
        SupportClass support = model.supportClass();
        switch (support) {
            case EMBED:
            case CLS:
            case NA:
            case DIFF:
            case ENC_DEC:
                throw new UnsupportedOperationException(model.architecture() + " (" + support
                        + ") has no decoder-only token loop to speculate on");
            default:
                break;
        }
        GGUFTokenizer tok = model.tokenizer();
        ReasoningFilter rf = ReasoningFilter.forChatTemplate(tok.chatTemplate()).withTagTokens(tok::getTokenId);
        return builder(model.forwardBackend(), drafter, tok)
                .reasoning(rf)
                .draftMax(config.draftMax())
                .draftMin(config.draftMin())
                .samplingDefaults(SamplingDefaults.forModel(model, config))
                .build();
    }

    public ReasoningFilter reasoning() { return reasoning; }
    public int draftMax() { return draftMax; }
    public int draftMin() { return draftMin; }
    public SamplingDefaults samplingDefaults() { return samplingDefaults; }

    /**
     * Generates after {@code prompt}, streaming the content channel to {@code callback}.
     *
     * @see GenerationLoop#generate(int[], GenerationRequest, TokenCallback)
     */
    public GenerationResult generate(int[] prompt, GenerationRequest request, TokenCallback callback) {
        return generate(prompt, request, callback == null ? DISCARD : ReasoningFilter.contentTo(callback), null);
    }

    /**
     * Generates after {@code prompt}, routing every piece of output to {@code sink} with its channel.
     *
     * @param renderedPrompt the prompt text used to detect a generation prefix that opens reasoning; null renders it
     *                       from the prompt tokens
     * @throws ContextFullException          when the prompt is longer than the context
     * @throws UnsupportedOperationException when the backend has no memory
     * @throws IllegalArgumentException      for an empty prompt, a negative token id, or a model sampling default
     *                                       the sampler rejects
     */
    public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                     String renderedPrompt) {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(sink, "sink");
        requirePrompt(prompt);
        ModelMemory memory = backend.memory();
        if (memory == null) {
            throw new UnsupportedOperationException("speculative generation needs model memory; this backend runs"
                    + " without a cache");
        }
        SequenceState seq = memory.sequence();
        int nCtx = seq.nCtx();
        if (prompt.length > nCtx) {
            throw new ContextFullException(prompt.length, nCtx);
        }
        GenerationRequest effective = request.withModelDefaults(samplingDefaults);
        StopCriteria stop = StopCriteria.of(effective, tokenizer, reasoning);
        Sampler sampler = Sampler.create(samplerConfig(effective.sampler(), nCtx), stop.bannedTokens());
        String rendered = renderedPrompt != null || reasoning.mode() == ReasoningFilter.Mode.NONE
                ? renderedPrompt : stop.renderPrompt(prompt);
        StopCriteria.Session session = stop.start(sink, rendered);
        try {
            return run(prompt, memory, seq, nCtx, sampler, session, effective, sink);
        } catch (RuntimeException e) {
            try {
                backend.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            try {
                drafter.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            throw e;
        }
    }

    private GenerationResult run(int[] prompt, ModelMemory memory, SequenceState seq, int nCtx, Sampler sampler,
                                 StopCriteria.Session session, GenerationRequest effective,
                                 ReasoningFilter.Sink sink) {
        long t0 = System.nanoTime();
        backend.reset();
        drafter.reset();
        int n = effective.logprobs();
        List<TokenScore[]> tokenLogprobs = new ArrayList<>();
        float[] logits = backend.prefill(prompt, 0);
        sampler.accept(prompt, 0, prompt.length);
        long t1 = System.nanoTime();
        // Non-MATCH verification runs per generation with its own uniform stream, seeded like the sampler's chain.
        Verifier verifier = verifyMode == VerifyMode.MATCH ? null : verifierFor(sampler);
        SplittableRandom roundsRng = verifier == null ? null : new SplittableRandom(sampler.seed());

        int rounds = 0;
        int drafted = 0;
        int accepted = 0;
        int sampled = 0;
        int token = -1;
        StopReason reason;
        if (session.cancelled()) {
            reason = StopReason.CANCELLED;
        } else {
            Sampler.Candidates cur = sampler.apply(logits);
            token = cur.id(cur.selected());
            capture(tokenLogprobs, sink, cur, token, n);
            sampler.accept(token);
            sampled++;
            reason = session.offer(token);
        }
        while (reason == null) {
            if (session.cancelled()) {
                reason = StopReason.CANCELLED;
                break;
            }
            int pos = seq.nPast();
            if (pos >= nCtx) {
                reason = StopReason.CONTEXT_FULL;
                break;
            }
            int k = draftLength(memory, pos, nCtx, session.remaining());
            int[] draft = k > 0 ? checkDraft(drafter.draft(token, k), k) : new int[0];
            if (draft.length == 0 || draft.length < draftMin) {
                logits = backend.decode(token, pos);
                drafter.accept(0);
                Sampler.Candidates cur = sampler.apply(logits);
                token = cur.id(cur.selected());
                capture(tokenLogprobs, sink, cur, token, n);
                sampler.accept(token);
                sampled++;
                reason = session.offer(token);
                continue;
            }
            Verified v = verify(memory, verifier, roundsRng, sampler, session, token, draft, pos, n,
                    tokenLogprobs, sink);
            rounds++;
            drafted += draft.length;
            accepted += v.accepted;
            sampled += v.emitted;
            token = v.next;
            reason = v.reason;
        }
        long t2 = System.nanoTime();
        statsSink.accept(new Stats(rounds, drafted, accepted));
        GenerationResult result = session.finish(reason,
                GenerationResult.Timings.of(prompt.length, t1 - t0, sampled, t2 - t1));
        return tokenLogprobs.isEmpty() ? result : result.withLogprobs(tokenLogprobs);
    }

    /**
     * The captured distribution of one emitted token: the token first with its post-chain log probability, then
     * the top alternatives by probability — the same shape {@code SequenceRun} captures for the plain loop. Routed
     * to the sink as an empty content piece carrying the capture, so a stream carries one entry per token.
     */
    private static void capture(List<TokenScore[]> out, ReasoningFilter.Sink sink, Sampler.Candidates cur,
                                int token, int n) {
        if (n <= 0) {
            return;
        }
        int size = cur.size();
        int[] ids = new int[size];
        float[] probs = new float[size];
        float[] logits = new float[size];
        int live = 0;
        double sum = 0;
        for (int i = 0; i < size; i++) {
            float p = cur.prob(i);
            if (p <= 0f) {
                continue;
            }
            ids[live] = cur.id(i);
            probs[live] = p;
            sum += p;
            live++;
        }
        if (live == 0 || Math.abs(sum - 1f) > 1e-3) {
            live = 0;
            sum = 0;
            double max = -Double.MAX_VALUE;
            for (int i = 0; i < size; i++) {
                float logit = cur.logit(i);
                if (logit != Float.NEGATIVE_INFINITY) {
                    max = Math.max(max, logit);
                }
            }
            for (int i = 0; i < size; i++) {
                float logit = cur.logit(i);
                if (logit == Float.NEGATIVE_INFINITY) {
                    continue;
                }
                double e = Math.exp(logit - max);
                ids[live] = cur.id(i);
                probs[live] = (float) e;
                sum += e;
                live++;
            }
            for (int i = 0; i < live; i++) {
                probs[i] = (float) (probs[i] / sum);
            }
        }
        int sampledIndex = -1;
        for (int i = 0; i < live; i++) {
            if (ids[i] == token) {
                sampledIndex = i;
                break;
            }
        }
        int count = Math.min(n, live);
        TokenScore[] scores = new TokenScore[count];
        scores[0] = new TokenScore(token,
                (float) Math.log(sampledIndex >= 0 ? probs[sampledIndex] : 0f));
        boolean[] used = new boolean[live];
        if (sampledIndex >= 0) {
            used[sampledIndex] = true;
        }
        for (int k = 1; k < count; k++) {
            int best = -1;
            for (int i = 0; i < live; i++) {
                if (!used[i] && (best < 0 || probs[i] > probs[best])) {
                    best = i;
                }
            }
            used[best] = true;
            scores[k] = new TokenScore(ids[best], (float) Math.log(probs[best]));
        }
        out.add(scores);
        sink.accept(ReasoningFilter.Channel.CONTENT, token, "", scores);
    }

    /** Outcome of one verify call. */
    private static final class Verified {
        final int accepted;
        final int emitted;
        final int next;
        final StopReason reason;

        Verified(int accepted, int emitted, int next, StopReason reason) {
            this.accepted = accepted;
            this.emitted = emitted;
            this.next = next;
            this.reason = reason;
        }
    }

    /**
     * Scores {@code [token, draft..]} at {@code pos} and emits the confirmation the verify mode decides over, then
     * drops the rejected positions from memory. A stop inside the verify (a non-null reason) leaves the same state
     * as the ordinary rejection path: the extra verified positions are replayed away, never left as an overshot
     * state (PARITY §5.5.1 S-1).
     */
    private Verified verify(ModelMemory memory, Verifier verifier, SplittableRandom roundsRng, Sampler sampler,
                            StopCriteria.Session session, int token, int[] draft, int pos, int n,
                            List<TokenScore[]> tokenLogprobs, ReasoningFilter.Sink sink) {
        int k = draft.length;
        int[] batch = new int[k + 1];
        batch[0] = token;
        System.arraycopy(draft, 0, batch, 1, k);
        boolean recurrent = memory.hasRecurrentState();
        if (recurrent) {
            memory.snapshot();
        }
        float[][] rows = backend.forwardRows(batch, pos, null);
        if (rows.length != k + 1) {
            throw new IllegalStateException("verification of " + (k + 1) + " tokens returned " + rows.length
                    + " logits rows");
        }
        Verified outcome = verifier == null
                ? matchVerify(sampler, session, draft, rows, n, tokenLogprobs, sink)
                : verifierVerify(verifier, roundsRng, sampler, session, draft, rows);
        int drop = k - outcome.accepted;
        if (drop > 0) {
            if (recurrent) {
                memory.restore();
                backend.prefill(Arrays.copyOf(batch, outcome.accepted + 1), pos);
            } else {
                if (!backend.canRollback(drop)) {
                    throw new IllegalStateException("the target memory cannot drop the " + drop
                            + " rejected draft positions");
                }
                backend.rollback(drop);
            }
        }
        drafter.accept(outcome.accepted);
        return outcome;
    }

    /**
     * {@link VerifyMode#MATCH}: the longest prefix whose chain samples equal the draft (llama.cpp
     * common/sampling.cpp:736); the first mismatching row's sample, or the last row's, is the target's next token.
     */
    private Verified matchVerify(Sampler sampler, StopCriteria.Session session, int[] draft, float[][] rows,
                                 int n, List<TokenScore[]> tokenLogprobs, ReasoningFilter.Sink sink) {
        int k = draft.length;
        int nAccepted = 0;
        int emitted = 0;
        int next = -1;
        StopReason reason = null;
        for (int i = 0; i <= k; i++) {
            if (i > 0 && session.cancelled()) {
                reason = StopReason.CANCELLED;
                break;
            }
            Sampler.Candidates cur = sampler.apply(rows[i]);
            next = cur.id(cur.selected());
            capture(tokenLogprobs, sink, cur, next, n);
            sampler.accept(next);
            emitted++;
            reason = session.offer(next);
            if (reason != null || i == k || next != draft[i]) {
                break;
            }
            nAccepted++;
        }
        return new Verified(nAccepted, emitted, next, reason);
    }

    /** A non-MATCH verifier decides the round over the rows' processed distributions. */
    private Verified verifierVerify(Verifier verifier, SplittableRandom roundsRng, Sampler sampler,
                                    StopCriteria.Session session, int[] draft, float[][] rows) {
        int k = draft.length;
        Round round = new Round(sampler, session, draft, k, rows, roundsRng);
        verifier.verify(round);
        return new Verified(round.accepted, round.emitted, round.next, round.reason);
    }

    /**
     * One verify round of a non-MATCH verifier: the processed row distributions on demand, the drafter's proposal
     * distributions, the round's uniform stream and its emissions through the session. Rows are handed out in
     * ascending order at most once each; an emission's non-null stop reason ends the round.
     */
    private final class Round implements Verifier.Round {

        private final Sampler sampler;
        private final StopCriteria.Session session;
        private final int[] draft;
        private final int k;
        private final float[][] rows;
        private final SplittableRandom rng;
        private final Verifier.Distribution[] draftDistributions;
        private int lastRow = -1;
        int accepted;
        int emitted;
        int next = -1;
        private StopReason reason;

        private Round(Sampler sampler, StopCriteria.Session session, int[] draft, int k, float[][] rows,
                      SplittableRandom rng) {
            this.sampler = sampler;
            this.session = session;
            this.draft = draft;
            this.k = k;
            this.rows = rows;
            this.rng = rng;
            this.draftDistributions = new Verifier.Distribution[k];
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
            return distribution(rows[row], sampler);
        }

        @Override
        public Verifier.Distribution draft(int position) {
            checkDraftPosition(position);
            return draftDistribution(position);
        }

        private Verifier.Distribution draftDistribution(int position) {
            Verifier.Distribution built = draftDistributions[position];
            if (built != null) {
                return built;
            }
            if (!(drafter instanceof DraftProbs)) {
                built = Verifier.Distribution.oneHot(draft[position]);
            } else {
                float[] row = ((DraftProbs) drafter).draftProbs(position);
                if (row == null) {
                    throw new IllegalStateException("the drafter returned no proposal probabilities for position "
                            + position);
                }
                int token = draft[position];
                if (token >= row.length) {
                    throw new IllegalStateException("the drafter proposed token " + token
                            + " but offered " + row.length + " proposal probabilities");
                }
                for (float v : row) {
                    if (Float.isNaN(v)) {
                        throw new IllegalStateException("the drafter's proposal probability row contains NaN");
                    }
                }
                if (row[token] <= 0f) {
                    throw new IllegalStateException("the drafter drafted token " + token + " with proposal"
                            + " probability " + row[token]);
                }
                int support = 0;
                for (float v : row) {
                    support += v > 0f ? 1 : 0;
                }
                int[] tokens = new int[support];
                float[] probs = new float[support];
                float[] logits = new float[support];
                int j = 0;
                for (int i = 0; i < row.length; i++) {
                    if (row[i] > 0f) {
                        tokens[j] = i;
                        probs[j] = row[i];
                        logits[j] = 0f;
                        j++;
                    }
                }
                built = new Verifier.Distribution(tokens, probs, logits, -1);
            }
            draftDistributions[position] = built;
            return built;
        }

        @Override public double uniform() {
            return rng.nextDouble();
        }

        @Override public StopReason confirm(int position) {
            if (position != accepted) {
                throw new IllegalStateException("confirm(" + position + ") after " + accepted
                        + " confirmed draft tokens");
            }
            return emitOne(draft[position]);
        }

        @Override public StopReason emit(int token) {
            return emitOne(token);
        }

        @Override public int accepted() {
            return accepted;
        }

        @Override public boolean stopped() {
            return reason != null;
        }

        private StopReason emitOne(int token) {
            if (reason != null) {
                throw new IllegalStateException("the verify round already stopped with " + reason);
            }
            if (session.cancelled()) {
                reason = StopReason.CANCELLED;
                return reason;
            }
            sampler.accept(token);
            emitted++;
            next = token;
            reason = session.offer(token);
            return reason;
        }

        private void checkDraftPosition(int position) {
            if (position < 0 || position >= k) {
                throw new IndexOutOfBoundsException("draft position " + position + " outside [0, " + k + ")");
            }
        }
    }

    /**
     * The processed distribution of one verify row: the surviving candidates with their post-chain probabilities
     * and logits and the chain's own draw. A greedy chain keeps no normalized probabilities; its distribution is
     * one-hot on the argmax (which leaves the argmax rule of the lossless modes intact), while the logits stay the
     * chain's for FLy's entropy gate.
     */
    private static Verifier.Distribution distribution(float[] row, Sampler sampler) {
        Sampler.Candidates cur = sampler.apply(row);
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

    /** The verify mode's verifier for one generation (stateless, shared across the generation's rounds). */
    private Verifier verifierFor(Sampler sampler) {
        switch (verifyMode) {
            case REJECTION:
                return new RejectionVerifier();
            case BLOCK:
                return new BlockVerifier();
            case FLY:
                return new FlyVerifier(flyWindow, flyEntropyThreshold, flyEntropyTopK, sampler.greedy());
            case SYNTHETIC:
                return new SyntheticVerifier(syntheticRates != null
                        ? syntheticRates
                        : SyntheticVerifier.ratesForLength(syntheticLength, draftMax));
            case MATCH:
            default:
                throw new IllegalStateException("no verifier for verification mode " + verifyMode);
        }
    }

    /**
     * The longest proposal the round can verify and then drop: bounded by {@code draftMax}, the free context after
     * {@code token}, one ubatch and the remaining budget, then the largest {@code k} for which the memory keeps every
     * sliding window after storing {@code k + 1} positions and dropping {@code k}
     * ({@link ModelMemory#ringsKeepWindow}, monotone in {@code k}). DSV4 compressed caches bound {@code k} by their
     * compressor rollback snapshots ({@link MemoryPlan#rollbackSnapshots()}): the verify stores {@code k + 1 <= ubatch}
     * positions in one step, whose snapshot planes cover dropping any {@code k} of them. 0 for recurrent state without
     * snapshot shadows.
     */
    private int draftLength(ModelMemory memory, int pos, int nCtx, int remaining) {
        MemoryPlan plan = memory.plan();
        int k = Math.min(draftMax, nCtx - pos - 1);
        k = Math.min(k, plan.ubatch() - 1);
        k = Math.min(k, remaining - 1);
        if (plan.hasCompressed()) {
            k = Math.min(k, plan.rollbackSnapshots());
        }
        if (k <= 0 || memory.hasRecurrentState() && !memory.canSnapshot()) {
            return 0;
        }
        if (memory.ringsKeepWindow(k + 1, k)) {
            return k;
        }
        int lo = 0;
        int hi = k;
        while (hi - lo > 1) {
            int mid = (lo + hi) >>> 1;
            if (memory.ringsKeepWindow(mid + 1, mid)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    private static int[] checkDraft(int[] draft, int k) {
        if (draft == null) {
            throw new IllegalStateException("the drafter returned no proposal array");
        }
        if (draft.length > k) {
            throw new IllegalStateException("the drafter proposed " + draft.length + " tokens, at most " + k
                    + " were asked");
        }
        for (int i = 0; i < draft.length; i++) {
            if (draft[i] < 0) {
                throw new IllegalStateException("the drafter proposed negative token id " + draft[i]);
            }
        }
        return draft;
    }

    /** {@code config} with a whole-context penalty window bounded to the context length. */
    private static SamplerConfig samplerConfig(SamplerConfig config, int nCtx) {
        return config.penaltyLastN() == SamplerConfig.PENALTY_WHOLE_CONTEXT
                ? config.toBuilder().penaltyLastN(nCtx).build() : config;
    }

    private static void requirePrompt(int[] prompt) {
        if (prompt.length == 0) {
            throw new IllegalArgumentException("prompt must not be empty");
        }
        for (int i = 0; i < prompt.length; i++) {
            if (prompt[i] < 0) {
                throw new IllegalArgumentException("prompt token " + i + " has negative id " + prompt[i]);
            }
        }
    }

    @Override
    public String toString() {
        return "SpeculativeDecoder{" + drafter + ", draftMax=" + draftMax + ", draftMin=" + draftMin + ", "
                + "verify=" + verifyMode + ", " + reasoning
                + (samplingDefaults.isEmpty() ? "" : ", " + samplingDefaults) + '}';
    }

    public static final class Builder {
        private final ForwardBackend backend;
        private final DraftModel drafter;
        private final GGUFTokenizer tokenizer;
        private ReasoningFilter reasoning = ReasoningFilter.none();
        private int draftMax = 3;
        private int draftMin;
        private Consumer<Stats> statsSink = s -> { };
        private SamplingDefaults samplingDefaults = SamplingDefaults.NONE;
        private VerifyMode verifyMode = VerifyMode.MATCH;
        private int flyWindow;
        private double flyEntropyThreshold = 0.3;
        private int flyEntropyTopK = 3;
        private double[] syntheticRates;
        private Double syntheticLength;

        private Builder(ForwardBackend backend, DraftModel drafter, GGUFTokenizer tokenizer) {
            this.backend = Objects.requireNonNull(backend, "backend");
            this.drafter = Objects.requireNonNull(drafter, "drafter");
            this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        }

        /** Reasoning parser of the output (default {@link ReasoningFilter#none()}). */
        public Builder reasoning(ReasoningFilter v) { this.reasoning = Objects.requireNonNull(v, "reasoning"); return this; }

        /** Most tokens proposed per round (default 3; {@code neuralfree.draft.max}). */
        public Builder draftMax(int v) {
            if (v < 1) {
                throw new IllegalArgumentException("draft max must be at least 1, got " + v);
            }
            this.draftMax = v;
            return this;
        }

        /** Shorter proposals are not verified (default 0; {@code neuralfree.draft.min}). */
        public Builder draftMin(int v) {
            if (v < 0) {
                throw new IllegalArgumentException("draft min must not be negative, got " + v);
            }
            this.draftMin = v;
            return this;
        }

        /** Receives each generation's {@link Stats} before its result is built (default: discarded). */
        public Builder statsSink(Consumer<Stats> v) { this.statsSink = Objects.requireNonNull(v, "statsSink"); return this; }

        /**
         * The target model's sampling defaults applied to every request (default {@link SamplingDefaults#NONE};
         * {@link SamplingDefaults#forModel} gives the target's under an engine configuration).
         */
        public Builder samplingDefaults(SamplingDefaults v) {
            this.samplingDefaults = Objects.requireNonNull(v, "samplingDefaults");
            return this;
        }

        /** The verification mode (default {@link VerifyMode#MATCH}). */
        public Builder verifyMode(VerifyMode v) {
            this.verifyMode = Objects.requireNonNull(v, "verifyMode");
            return this;
        }

        /**
         * {@code fly_window_size}: FLy's future native acceptance decisions to check, or 0 for vLLM's default of
         * {@code min(6, k - 1)} of the round's proposal (default 0; only {@link VerifyMode#FLY} reads it).
         */
        public Builder flyWindow(int v) {
            if (v < 0) {
                throw new IllegalArgumentException("fly window must not be negative, got " + v);
            }
            this.flyWindow = v;
            return this;
        }

        /** {@code fly_entropy_threshold}: the entropy a deferral needs (default 0.3; only {@link VerifyMode#FLY}). */
        public Builder flyEntropyThreshold(double v) {
            if (!(v >= 0.0)) {
                throw new IllegalArgumentException("fly entropy threshold must not be negative, got " + v);
            }
            this.flyEntropyThreshold = v;
            return this;
        }

        /** {@code fly_entropy_top_k}: the processed probabilities the entropy covers (default 3; only {@link VerifyMode#FLY}). */
        public Builder flyEntropyTopK(int v) {
            if (v <= 0) {
                throw new IllegalArgumentException("fly entropy top-k must be positive, got " + v);
            }
            this.flyEntropyTopK = v;
            return this;
        }

        /**
         * {@code synthetic_acceptance_rates}: the unconditional per-position acceptance rates of
         * {@link VerifyMode#SYNTHETIC}; mutually exclusive with {@link #syntheticAcceptanceLength(double)}, which
         * vLLM's minimum-variance schedule resolves to these rates.
         */
        public Builder syntheticAcceptanceRates(double... rates) {
            new SyntheticVerifier(rates); // validation
            this.syntheticRates = rates.clone();
            return this;
        }

        /** @{@code synthetic_acceptance_length}: the target mean acceptance length (a shorthand for the rates). */
        public Builder syntheticAcceptanceLength(double length) {
            SyntheticVerifier.ratesForLength(length, draftMax); // validation against the proposal budget
            this.syntheticLength = length;
            return this;
        }

        /**
         * The configured unconditional rates: the explicit {@code synthetic_acceptance_rates}, or the
         * {@code synthetic_acceptance_length} resolved over {@code draftMax} proposal positions (vLLM's
         * minimum-variance schedule).
         */
        private double[] resolvedSyntheticRates() {
            return syntheticRates != null
                    ? syntheticRates
                    : SyntheticVerifier.ratesForLength(syntheticLength, draftMax);
        }

        /**
         * Returns the decoder.
         *
         * @throws IllegalStateException when {@link VerifyMode#SYNTHETIC} is selected without the opt-in property
         *                               {@code neuralfree.spec.synthetic}
         * @throws IllegalArgumentException when both synthetic settings are given, or none outside synthetic mode,
         *                                  when the synthetic settings are themselves invalid, or when the draft
         *                                  lengths disagree
         */
        public SpeculativeDecoder build() {
            boolean synthetic = verifyMode == VerifyMode.SYNTHETIC;
            if (syntheticRates != null && syntheticLength != null) {
                throw new IllegalArgumentException("synthetic_acceptance_rates and synthetic_acceptance_length"
                        + " are mutually exclusive; configure one");
            }
            if (synthetic && syntheticRates == null && syntheticLength == null) {
                throw new IllegalArgumentException("synthetic verification needs synthetic_acceptance_rates or"
                        + " synthetic_acceptance_length");
            }
            if (synthetic) {
                if (!Boolean.getBoolean(SyntheticVerifier.PROP_ENABLED)) {
                    throw new IllegalStateException(SyntheticVerifier.PROP_ENABLED + "=true is required for"
                            + " synthetic verification");
                }
                new SyntheticVerifier(resolvedSyntheticRates()); // validates the resolved rates
            }
            return new SpeculativeDecoder(this);
        }
    }
}
