package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.draft.Verifier;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The per-sequence token pipeline of one autoregressive generation: the stages llama.cpp runs inside
 * its completion token loop, shared by {@link GenerationLoop} and the batch engine that builds on it. Mutable, owned
 * by one thread at a time, never thread-safe.
 *
 * <p>Built from a {@link GenerationRequest} (with the model's sampling defaults applied), the model's tokenizer and
 * the model's end-of-generation set (the tokenizer's {@link com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens};
 * {@link StopCriteria} derives it), its stages per sampled token are:
 * <ol>
 *   <li><b>sample</b>: run the {@link Sampler} chain over the logits row ({@link #next}) and accept the token into
 *       the chain's history stages (a token the caller already has is accepted with {@link #accept});</li>
 *   <li><b>stop</b>: {@link StopCriteria.Session#offer} — an end-of-generation token, a completed stop string or
 *       the token budget ends the generation ({@link #offer}, {@link Step#stopped()});</li>
 *   <li><b>route and call</b>: the token's decoded text runs the stop-string filter and the reasoning parser inside
 *       the session, which forwards each routed piece with its {@link com.eonmux.neuralfree.gguf.engine.ReasoningFilter.Channel}
 *       to the {@link SequenceRun} builder's sink (the caller's callback); the text routed by this step is reported
 *       back in the step ({@link Step#text()});</li>
 *   <li><b>counters and timings</b>: tokens accepted, prompt tokens and both phase times for the result
 *       ({@link #prefilled}, {@link #result()}).</li>
 * </ol>
 * A stop reason no sampled token produced — the request's cancellation hook or a context the memory cannot make
 * room for — is recorded with {@link #stop(StopReason)}; {@link #done()} turns true after any stop and
 * {@link GenerationResult result()} finishes the {@link GenerationResult} text with the session.
 */
public final class SequenceRun {

    private static final ReasoningFilter.Sink DISCARD = (channel, tokenId, text) -> { };

    private final StopCriteria stop;
    private final Sampler sampler;
    private final ReasoningFilter.Sink sink;
    private final StopCriteria.Session session;

    private final StringBuilder delta = new StringBuilder();
    private boolean recording;
    private int sampled;
    private int promptTokens;
    private long promptNanos;
    private long generationStart;
    private StopReason stopReason;
    private final LogprobCapture capture;
    /** Whether the current offer's capture already rode a routed piece (one capture per token). */
    private boolean logprobsRouted;
    private static final TokenScore[] EMPTY_LOGPROBS = new TokenScore[0];

    private TokenScore[] pendingLogprobs = EMPTY_LOGPROBS;
    private final List<TokenScore[]> tokenLogprobs = new ArrayList<>();

    private SequenceRun(Builder b) {
        this.stop = b.stop;
        this.sampler = b.sampler;
        this.sink = b.sink;
        this.capture = LogprobCapture.of(b.request);
        this.session = stop.start(new ReasoningFilter.Sink() {
            @Override
            public void accept(ReasoningFilter.Channel channel, int tokenId, String text) {
                routed(channel, tokenId, text);
            }

            @Override
            public void toolCallStarted(String recipient) {
                sink.toolCallStarted(recipient);
            }

            @Override
            public void stepped(int tokenId) {
                sink.stepped(tokenId);
            }

            @Override
            public void contentFallback() {
                sink.contentFallback();
            }
        }, b.renderedPrompt);
        this.generationStart = System.nanoTime();
    }

    /** A run for {@code request}: the builder takes {@link #sampler}, {@link #sink} and the rest of the output side. */
    public static Builder over(GenerationRequest request, GGUFTokenizer tokenizer) {
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(tokenizer, "tokenizer");
        return new Builder(request, tokenizer);
    }

    /**
     * Runs the sampler chain over {@code logits} and accepts the selected token into the chain's history stages —
     * llama.cpp sample step and its sequence accept in one call. When the request asked for logprobs
     * ({@code n_probs}, {@code logprob_token_ids}), the step carries the {@link LogprobCapture} of the row in the
     * request's {@link LogprobsMode}: the sampled token first, then its best alternatives or the requested ids
     * ({@link Step#logprobs()}).
     *
     * @throws IllegalArgumentException      when the row is empty
     * @throws ConstraintDeadEndException    when the chain's constraints admit no candidate
     * @throws IllegalStateException         when no candidate has a finite logit
     */
    public int next(float[] logits) {
        int token;
        if (capture.active()) {
            Sampler.Candidates candidates = sampler.apply(logits);
            token = candidates.selectedId();
            pendingLogprobs = capture.capture(logits, candidates, token);
        } else {
            token = sampler.sample(logits);
        }
        accept(token);
        return token;
    }

    /**
     * Runs the sampler chain over a row's device-read top candidates ({@link RowTopK}, the backend-sampled decode of
     * a truncated chain) and returns the selected token; the caller accepts it with {@link #accept}, as with the
     * {@link ForwardBackend#decodeGreedyOnDevice} fast path. Never called when the request captures logprobs.
     *
     * @throws IllegalArgumentException when the candidates disagree in length
     * @throws IllegalStateException    when no candidate has a finite logit
     */
    public int sampleTopK(RowTopK top) {
        return sampleTopK(top, 0);
    }

    /**
     * Returns the token the chain selects over row {@code row} of a device top-k readback; the caller accepts it with
     * {@link #accept}, as with {@link #sampleTopK(RowTopK)}. A batched step samples every output row from the one
     * readback this way.
     *
     * @throws IllegalArgumentException when {@code row} is outside the readback
     * @throws IllegalStateException    when no candidate has a finite logit
     */
    public int sampleTopK(RowTopK top, int row) {
        return sampleTopK(top, row, false);
    }

    /**
     * {@link #sampleTopK(RowTopK, int)} over a readout that applied the row's logit prefilter before its top-k
     * (the device path of {@code Sampler#deviceConeWidth()}): the chain skips the fixed bias stage and the
     * pre-top-k penalties and DRY it already reproduced ({@code Sampler#applyTopK(int[], float[], boolean)}).
     */
    public int sampleTopK(RowTopK top, int row, boolean prefiltered) {
        Objects.requireNonNull(top, "top");
        if (row < 0 || row >= top.rows()) {
            throw new IllegalArgumentException("row " + row + " outside a top-k readback of " + top.rows() + " rows");
        }
        Sampler.Candidates candidates = sampler.applyTopK(top.ids(row), top.logits(row), prefiltered);
        return candidates.id(candidates.selected());
    }

    /** Whether the request's logprobs setting needs host-side logits every step (disables on-device greedy). */
    public boolean wantsHostSampling() {
        return capture.active();
    }

    /**
     * Accepts a token that was not sampled host-side (the {@link ForwardBackend#decodeGreedyOnDevice} fast path with
     * it) into the chain's history stages and the token counter.
     */
    public void accept(int token) {
        sampled++;
        sampler.accept(token);
    }

    /**
     * Feeds one sampled token to the stop decision and routes its text (stage 2 to 3).
     *
     * @return the step's stop decision and the text its session routed to the sink, or {@link Step#stopped()} false
     *         while the generation continues
     * @throws IllegalStateException after {@link #result()}
     */
    public Step offer(int token) {
        delta.setLength(0);
        StopReason reason;
        recording = true;
        logprobsRouted = false;
        try {
            reason = session.offer(token);
        } finally {
            recording = false;
        }
        String routed = delta.length() == 0 ? "" : delta.toString();
        delta.setLength(0);
        if (reason != null) {
            this.stopReason = reason;
        }
        Step step = Step.of(reason, routed, pendingLogprobs);
        if (pendingLogprobs.length > 0) {
            tokenLogprobs.add(pendingLogprobs);
            if (!logprobsRouted) {
                // the token's text is held back whole: its capture rides an empty piece on the content channel
                sink.accept(ReasoningFilter.Channel.CONTENT, token, "", pendingLogprobs);
            }
        }
        pendingLogprobs = EMPTY_LOGPROBS;
        return step;
    }

    /**
     * Offers a token a speculative verifier emitted ({@link VerifyRound}): the round already accepted it into the
     * sampler, so this counts it, captures its logprobs from the verify row it was emitted at when the request
     * asked for them, and runs {@link #offer}.
     *
     * @param row       the raw verify row the token was emitted at
     * @param processed the row's chain distribution the verifier decided over
     */
    Step offerVerified(int token, float[] row, Verifier.Distribution processed) {
        sampled++;
        if (capture.active()) {
            pendingLogprobs = capture.capture(row, processed, token);
        }
        return offer(token);
    }

    /**
     * Records a stop decision no sampled token produced: {@link StopReason#CANCELLED} when the request's
     * cancellation hook fired ({@link #cancelled()}), {@link StopReason#CONTEXT_FULL} when the context could not
     * make room for the next token. Token-level reasons come from {@link #offer} only.
     *
     * @throws IllegalStateException after a previous stop
     */
    public Step stop(StopReason reason) {
        if (reason == null || reason == StopReason.EOS || reason == StopReason.STOP_STRING
                || reason == StopReason.MAX_TOKENS || reason == StopReason.REPETITION) {
            throw new IllegalArgumentException("token-level stop reasons come from offer(), got " + reason);
        }
        if (stopReason != null) {
            throw new IllegalStateException("the sequence already stopped with " + stopReason);
        }
        this.stopReason = reason;
        if (pendingLogprobs.length > 0) {
            tokenLogprobs.add(pendingLogprobs);
        }
        Step step = new Step(reason, "", pendingLogprobs);
        pendingLogprobs = EMPTY_LOGPROBS;
        return step;
    }

    /** The request's cancellation hook; true stops the generation with {@link StopReason#CANCELLED}. */
    public boolean cancelled() {
        return session.cancelled();
    }

    /** True once a stop decision was recorded, by {@link #offer} or {@link #stop}. */
    public boolean done() {
        return stopReason != null;
    }

    /** The sampler: the driver's prefill accepts the prompt tokens into its history stages (llama.cpp completion). */
    public Sampler sampler() { return sampler; }

    /**
     * Records the prompt phase and starts the generation clock for {@link GenerationResult#timings()}: the prompt
     * tokens evaluated (after any reuse) and the time spent evaluating them.
     */
    public void prefilled(int promptTokens, long promptNanos) {
        prefilled(promptTokens, promptNanos, System.nanoTime());
    }

    /**
     * {@link #prefilled(int, long)} with the generation clock started at {@code generationStart} (a
     * {@link System#nanoTime} reading): a driver that records the prompt phase after the first token's sampling
     * passes the moment the prefill ended.
     */
    public void prefilled(int promptTokens, long promptNanos, long generationStart) {
        if (promptTokens < 0 || promptNanos < 0) {
            throw new IllegalArgumentException("negative prompt timing: " + promptTokens + " tokens, "
                    + promptNanos + " ns");
        }
        this.promptTokens = promptTokens;
        this.promptNanos = promptNanos;
        this.generationStart = generationStart;
    }

    /**
     * Releases the withheld output text (the decoder's partial character, the stop filter's held-back tail, the
     * reasoning parser's buffer) and builds the result over the recorded stop reason and timings. The released tail
     * running past a stop string still ends the generation with {@link StopReason#STOP_STRING}.
     *
     * @throws IllegalStateException before a stop, or twice
     */
    public GenerationResult result() {
        if (stopReason == null) {
            throw new IllegalStateException("the sequence has no stop decision yet");
        }
        long generationNanos = System.nanoTime() - generationStart;
        GenerationResult result = session.finish(stopReason,
                GenerationResult.Timings.of(promptTokens, promptNanos, sampled, generationNanos));
        return tokenLogprobs.isEmpty() ? result : result.withLogprobs(tokenLogprobs);
    }

    /** The per-token scores the run captured ({@link GenerationRequest#wantsLogprobs()}), one entry per token. */
    public List<TokenScore[]> tokenLogprobs() {
        return tokenLogprobs;
    }

    @Override
    public String toString() {
        return "SequenceRun{sampled=" + sampled + ", stop=" + stopReason + '}';
    }

    /** Forwards routed text to the builder's sink and records what the current {@link #offer} routed. */
    private void routed(ReasoningFilter.Channel channel, int tokenId, String text) {
        if (recording && text != null && !text.isEmpty()) {
            delta.append(text);
        }
        if (capture.active()) {
            // the capture rides the first piece the token releases; later pieces of the same offer carry none
            sink.accept(channel, tokenId, text, logprobsRouted ? EMPTY_LOGPROBS : pendingLogprobs);
            logprobsRouted = true;
        } else {
            sink.accept(channel, tokenId, text);
        }
    }

    /**
     * One token's offer: the stop decision ({@link #stopped()}, {@link #reason()} non-null exactly then) and the
     * output text the session routed to the sink for this token. Immutable.
     */
    public static final class Step {

        private static final Step EMPTY = new Step(null, "", EMPTY_LOGPROBS);

        private final StopReason reason;
        private final String text;
        private final TokenScore[] logprobs;

        private Step(StopReason reason, String text, TokenScore[] logprobs) {
            this.reason = reason;
            this.text = text;
            this.logprobs = logprobs;
        }

        private static Step of(StopReason reason, String text, TokenScore[] logprobs) {
            return reason == null && text.isEmpty() && logprobs.length == 0
                    ? EMPTY : new Step(reason, text, logprobs);
        }

        /** Whether the token ended the generation. */
        public boolean stopped() {
            return reason != null;
        }

        /** Why the token ended the generation, or null while it continues. */
        public StopReason reason() {
            return reason;
        }

        /** The output text routed to the sink by this token's offer, across every channel. */
        public String text() {
            return text;
        }

        /**
         * The step's {@link LogprobCapture}: the sampled token first, then its best alternatives or the requested
         * token ids, when the request asked for them; empty otherwise.
         */
        public TokenScore[] logprobs() {
            return logprobs;
        }

        @Override
        public String toString() {
            return "Step{" + (reason == null ? "continue" : reason) + (text.isEmpty() ? "" : " '" + text + "'") + '}';
        }
    }

    /** Configures a {@link SequenceRun} over {@link #over}. */
    public static final class Builder {

        private final GenerationRequest request;
        private final GGUFTokenizer tokenizer;
        private ReasoningFilter reasoning = ReasoningFilter.none();
        private StopCriteria stop;
        private Sampler sampler;
        private ReasoningFilter.Sink sink = DISCARD;
        private String renderedPrompt;

        private Builder(GenerationRequest request, GGUFTokenizer tokenizer) {
            this.request = request;
            this.tokenizer = tokenizer;
        }

        /**
         * Stop criteria of the request over the tokenizer's special tokens (default:
         * {@link StopCriteria#of} over the request, the tokenizer and the builder's {@link #reasoning}).
         */
        public Builder stopCriteria(StopCriteria v) {
            Objects.requireNonNull(v, "stopCriteria");
            this.stop = v;
            return this;
        }

        /** Reasoning parser of the output for the default stop criteria (default {@link ReasoningFilter#none()}). */
        public Builder reasoning(ReasoningFilter v) {
            Objects.requireNonNull(v, "reasoning");
            this.reasoning = v;
            return this;
        }

        /**
         * The sampler chain, built for the request's configuration and the criteria's
         * {@link StopCriteria#bannedTokens()}.
         */
        public Builder sampler(Sampler v) {
            Objects.requireNonNull(v, "sampler");
            this.sampler = v;
            return this;
        }

        /** Receives the routed output text (null discards it; channel routing upstream of the sink is unchanged). */
        public Builder sink(ReasoningFilter.Sink v) {
            this.sink = v == null ? DISCARD : v;
            return this;
        }

        /** The prompt as text for the reasoning parser's generation-prefix detection (null when unknown). */
        public Builder renderedPrompt(String v) {
            this.renderedPrompt = v;
            return this;
        }

        public SequenceRun build() {
            if (stop == null) {
                this.stop = StopCriteria.of(request, tokenizer, reasoning);
            }
            return new SequenceRun(this);
        }
    }
}
