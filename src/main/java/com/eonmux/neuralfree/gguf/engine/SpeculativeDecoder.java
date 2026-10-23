package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Speculative generation over a target {@link ForwardBackend} and a {@link DraftModel}: the drafter proposes tokens,
 * the target scores them all in one forward call and keeps the longest prefix it would have sampled itself (llama.cpp
 * tools/server speculative loop with {@code common_sampler_sample_and_accept_n}, common/sampling.cpp). Immutable;
 * one generation runs at a time, on the backend's thread.
 *
 * <p>Per {@link #generate} call:
 * <ol>
 *   <li><b>Memory.</b> The target memory and the drafter are reset: the drafter's state follows the target's hidden
 *       rows from the first prompt token, so a reused prefix would leave it without them. A prompt longer than the
 *       context throws {@link ContextFullException}.</li>
 *   <li><b>Prefill.</b> The prompt is prefilled and enters the penalty history, as in {@link GenerationLoop}. The
 *       first token is sampled from its logits and offered to the {@link StopCriteria} session.</li>
 *   <li><b>Rounds.</b> With the last sampled token {@code t} at the next position {@code P = nPast}, the drafter
 *       proposes up to {@code k} tokens {@code d}. Without a usable proposal (empty, or shorter than
 *       {@code draftMin}) the round decodes {@code t} and samples one token. Otherwise
 *       {@link ForwardBackend#forwardRows} evaluates {@code [t, d_0 .. d_{k-1}]} at {@code P ..} and returns all
 *       {@code k + 1} logits rows; row {@code i} is sampled, accepted into the sampler and offered to the session,
 *       continuing while the sample equals {@code d_i}. With {@code a} drafted tokens confirmed, the round emits
 *       {@code d_0 .. d_{a-1}} and the sample of row {@code a}, and the last {@code k - a} positions leave memory:
 *       by {@link ForwardBackend#rollback}, or for recurrent memory by restoring the snapshot taken before the
 *       call and prefilling {@code [t, d_0 .. d_{a-1}]} again. The drafter then hears {@code accept(a)}.</li>
 * </ol>
 * Every emitted token passes through the same {@link Sampler} and {@link StopCriteria.Session} as in
 * {@link GenerationLoop}: banned tokens, penalties, stop strings, the token budget and reasoning routing are
 * unchanged. With an argmax sampler the output equals plain greedy decoding; with a stochastic chain each emitted
 * token is a draw from the target's distribution at its position.
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

    private SpeculativeDecoder(Builder b) {
        this.backend = b.backend;
        this.drafter = b.drafter;
        this.tokenizer = b.tokenizer;
        this.reasoning = b.reasoning;
        this.draftMax = b.draftMax;
        this.draftMin = b.draftMin;
        this.statsSink = b.statsSink;
        if (draftMin > draftMax) {
            throw new IllegalArgumentException("draft min " + draftMin + " exceeds draft max " + draftMax);
        }
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
     * its tag tokens) and the draft lengths of {@code config} ({@code neuralfree.draft.max},
     * {@code neuralfree.draft.min}).
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
                .build();
    }

    public ReasoningFilter reasoning() { return reasoning; }
    public int draftMax() { return draftMax; }
    public int draftMin() { return draftMin; }

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
     * @throws IllegalArgumentException      for an empty prompt or a negative token id
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
        StopCriteria stop = StopCriteria.of(request, tokenizer, reasoning);
        Sampler sampler = Sampler.create(samplerConfig(request.sampler(), nCtx), stop.bannedTokens());
        String rendered = renderedPrompt != null || reasoning.mode() == ReasoningFilter.Mode.NONE
                ? renderedPrompt : stop.renderPrompt(prompt);
        StopCriteria.Session session = stop.start(sink, rendered);
        try {
            return run(prompt, memory, seq, nCtx, sampler, session);
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
                                 StopCriteria.Session session) {
        long t0 = System.nanoTime();
        backend.reset();
        drafter.reset();
        float[] logits = backend.prefill(prompt, 0);
        sampler.accept(prompt, 0, prompt.length);
        long t1 = System.nanoTime();

        int rounds = 0;
        int drafted = 0;
        int accepted = 0;
        int sampled = 0;
        int token = -1;
        StopReason reason;
        if (session.cancelled()) {
            reason = StopReason.CANCELLED;
        } else {
            token = sampler.sample(logits);
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
                token = sampler.sample(logits);
                sampler.accept(token);
                sampled++;
                reason = session.offer(token);
                continue;
            }
            Verified v = verify(memory, sampler, session, token, draft, pos);
            rounds++;
            drafted += draft.length;
            accepted += v.accepted;
            sampled += v.emitted;
            token = v.next;
            reason = v.reason;
        }
        long t2 = System.nanoTime();
        statsSink.accept(new Stats(rounds, drafted, accepted));
        return session.finish(reason, GenerationResult.Timings.of(prompt.length, t1 - t0, sampled, t2 - t1));
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
     * Scores {@code [token, draft..]} at {@code pos}, emits the confirmed prefix and the target's own next token,
     * and drops the rejected positions from memory.
     */
    private Verified verify(ModelMemory memory, Sampler sampler, StopCriteria.Session session, int token,
                            int[] draft, int pos) {
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
        int nAccepted = 0;
        int emitted = 0;
        int next = -1;
        StopReason reason = null;
        for (int i = 0; i <= k; i++) {
            if (i > 0 && session.cancelled()) {
                reason = StopReason.CANCELLED;
                break;
            }
            next = sampler.sample(rows[i]);
            sampler.accept(next);
            emitted++;
            reason = session.offer(next);
            if (reason != null || i == k || next != draft[i]) {
                break;
            }
            nAccepted++;
        }
        if (reason != null) {
            return new Verified(nAccepted, emitted, next, reason);
        }
        int drop = k - nAccepted;
        if (drop > 0) {
            if (recurrent) {
                memory.restore();
                backend.prefill(Arrays.copyOf(batch, nAccepted + 1), pos);
            } else {
                if (!backend.canRollback(drop)) {
                    throw new IllegalStateException("the target memory cannot drop the " + drop
                            + " rejected draft positions");
                }
                backend.rollback(drop);
            }
        }
        drafter.accept(nAccepted);
        return new Verified(nAccepted, emitted, next, null);
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
                + reasoning + '}';
    }

    public static final class Builder {
        private final ForwardBackend backend;
        private final DraftModel drafter;
        private final GGUFTokenizer tokenizer;
        private ReasoningFilter reasoning = ReasoningFilter.none();
        private int draftMax = 3;
        private int draftMin;
        private Consumer<Stats> statsSink = s -> { };

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

        public SpeculativeDecoder build() {
            return new SpeculativeDecoder(this);
        }
    }
}
