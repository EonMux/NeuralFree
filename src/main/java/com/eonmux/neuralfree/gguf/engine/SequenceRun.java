package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;

import java.util.Objects;

/**
 * The per-sequence token pipeline of one autoregressive generation (PARITY §BE-1): the stages llama.cpp runs inside
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

    private SequenceRun(Builder b) {
        this.stop = b.stop;
        this.sampler = b.sampler;
        this.sink = b.sink;
        this.session = stop.start(this::routed, b.renderedPrompt);
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
     * llama.cpp sample step and its sequence accept in one call.
     *
     * @throws IllegalArgumentException when the row is empty
     * @throws IllegalStateException    when no candidate has a finite logit
     */
    public int next(float[] logits) {
        int token = sampler.sample(logits);
        accept(token);
        return token;
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
        return Step.of(reason, routed);
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
                || reason == StopReason.MAX_TOKENS) {
            throw new IllegalArgumentException("token-level stop reasons come from offer(), got " + reason);
        }
        if (stopReason != null) {
            throw new IllegalStateException("the sequence already stopped with " + stopReason);
        }
        this.stopReason = reason;
        return new Step(reason, "");
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
        if (promptTokens < 0 || promptNanos < 0) {
            throw new IllegalArgumentException("negative prompt timing: " + promptTokens + " tokens, "
                    + promptNanos + " ns");
        }
        this.promptTokens = promptTokens;
        this.promptNanos = promptNanos;
        this.generationStart = System.nanoTime();
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
        return session.finish(stopReason,
                GenerationResult.Timings.of(promptTokens, promptNanos, sampled, generationNanos));
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
        sink.accept(channel, tokenId, text);
    }

    /**
     * One token's offer: the stop decision ({@link #stopped()}, {@link #reason()} non-null exactly then) and the
     * output text the session routed to the sink for this token. Immutable.
     */
    public static final class Step {

        private static final Step EMPTY = new Step(null, "");

        private final StopReason reason;
        private final String text;

        private Step(StopReason reason, String text) {
            this.reason = reason;
            this.text = text;
        }

        private static Step of(StopReason reason, String text) {
            return reason == null && text.isEmpty() ? EMPTY : new Step(reason, text);
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
