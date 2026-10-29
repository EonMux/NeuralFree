package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.sampling.RequestStages;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.Objects;

/**
 * Autoregressive generation over a {@link ForwardBackend}: prompt prefill, then sample, emit and decode one token
 * at a time (the token loop of llama.cpp tools/completion/completion.cpp). Immutable; one generation runs at a time, on
 * the backend's thread.
 *
 * <p>Per {@link #generate} call:
 * <ol>
 *   <li><b>Model defaults.</b> The request takes the loop's {@link SamplingDefaults} for every sampler setting and
 *       the token budget it leaves unset ({@link GenerationRequest#withModelDefaults}).</li>
 *   <li><b>Memory.</b> Every call starts with a reset (the stored sequence is cleared with it). With prefix reuse,
 *       the longest common prefix of the stored tokens and the prompt would be kept (at most
 *       {@code prompt.length - 1} tokens, so the last prompt token is evaluated for logits) when the memory can
 *       drop the rest ({@link ForwardBackend#canRollback}) — until PC-9 the contiguous path reuses nothing
 *       (PARITY §5.5.1 E-2); {@link #prefixReuse} and {@link SequenceState#reusable} stay as its switches.</li>
 *   <li><b>Prefill.</b> A decoder prompt longer than the context throws {@link ContextFullException} before any
 *       work. The unreused prompt tail is prefilled; every prompt token enters the penalty history (llama.cpp
 *       completion accepts prompt tokens into the sampler). For encoder-decoder models the history starts with the
 *       decoder tokens the prefill stored (the decoder start token), not the encoder source, as llama.cpp
 *       completion replaces the prompt by the decoder start token after encoding. A penalty window of
 *       {@link SamplerConfig#PENALTY_WHOLE_CONTEXT} covers the last {@code nCtx} tokens.</li>
 *   <li><b>Token loop.</b> Poll cancellation; sample ({@link Sampler}, llama.cpp chain); offer the token to the
 *       {@link SequenceRun} (its {@link StopCriteria} session: end of generation, stop strings, budget); if the
 *       memory is full, shift the context or stop with {@link StopReason#CONTEXT_FULL}; decode the token at the
 *       next position. When the chain is plain argmax ({@link Sampler#argmaxOnly()}), decoding uses
 *       {@link ForwardBackend#decodeGreedyOnDevice} and no logits are read back.</li>
 * </ol>
 * Positions come from the memory's {@link SequenceState}: the next position is {@code nPast}. For encoder-decoder
 * backends, {@code prefill(source, 0)} encodes the source and evaluates the decoder start, and the decoder
 * sequence supplies the positions.
 *
 * <p><b>Context shift</b> (llama.cpp completion.cpp:562-588, enabled by {@code neuralfree.context.shift}): when
 * the next token has no position left, {@code n_keep} positions are kept (1 when the prompt starts with BOS,
 * llama.cpp's {@code n_keep = add_bos}), half of the rest is discarded and later positions move down. A backend
 * whose memory cannot shift (recurrent state, sparse caches, multi-section RoPE, step35) ends generation with
 * {@link StopReason#CONTEXT_FULL}.
 *
 * <p>When a forward call or the sampler fails, the memory is reset before the exception propagates and the memory's
 * {@link SequenceState} is cleared with it, so the next generation never reuses a half-written sequence; the state
 * is cleared even when the reset itself fails.
 */
public final class GenerationLoop {

    private static final Logger LOGGER = LogManager.getLogger(GenerationLoop.class);
    private static final ReasoningFilter.Sink DISCARD = (channel, tokenId, text) -> { };

    private final ForwardBackend backend;
    private final GGUFTokenizer tokenizer;
    private final ReasoningFilter reasoning;
    private final boolean prefixReuse;
    private final boolean contextShift;
    private final boolean encoderDecoder;
    private final SamplingDefaults samplingDefaults;

    private GenerationLoop(Builder b) {
        this.backend = b.backend;
        this.tokenizer = b.tokenizer;
        this.reasoning = b.reasoning;
        this.prefixReuse = b.prefixReuse;
        this.contextShift = b.contextShift;
        this.encoderDecoder = b.encoderDecoder;
        this.samplingDefaults = b.samplingDefaults;
    }

    public static Builder builder(ForwardBackend backend, GGUFTokenizer tokenizer) {
        return new Builder(backend, tokenizer);
    }

    /**
     * The loop of an engine model: its backend and tokenizer, the reasoning parser of its chat template (with its
     * tag tokens), prefix reuse and context shift from {@code config}'s memory settings, and the model's sampling
     * defaults unless {@code config} turns them off ({@link SamplingDefaults#forModel}).
     *
     * @throws UnsupportedOperationException for models that do not generate autoregressively (embedding,
     *                                       classifier, diffusion and unsupported architectures)
     */
    public static GenerationLoop forModel(EngineModel model, EngineConfig config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        SupportClass support = model.supportClass();
        switch (support) {
            case EMBED:
            case CLS:
            case NA:
                throw new UnsupportedOperationException(model.architecture() + " (" + support
                        + ") does not generate text autoregressively");
            case DIFF:
                throw new UnsupportedOperationException(model.architecture()
                        + " is a masked-diffusion model; generate with DiffusionGenerator");
            default:
                break;
        }
        GGUFTokenizer tok = model.tokenizer();
        ReasoningFilter rf = ReasoningFilter.forChatTemplate(tok.chatTemplate()).withTagTokens(tok::getTokenId);
        return builder(model.forwardBackend(), tok)
                .reasoning(rf)
                .memoryConfig(config.memory())
                .encoderDecoder(support == SupportClass.ENC_DEC)
                .samplingDefaults(SamplingDefaults.forModel(model, config))
                .build();
    }

    public ReasoningFilter reasoning() { return reasoning; }
    public boolean prefixReuse() { return prefixReuse; }
    public boolean contextShift() { return contextShift; }
    public boolean encoderDecoder() { return encoderDecoder; }
    public SamplingDefaults samplingDefaults() { return samplingDefaults; }

    /**
     * Generates after {@code prompt}, streaming the content channel of the output to {@code callback} (reasoning
     * and tool-call text are parsed out by the reasoning filter; with {@link ReasoningFilter#none()} everything is
     * content). Text released at the end carries token id -1.
     *
     * @param callback receives content text; may be null
     * @return the generated tokens, the raw output text (cut before a matched stop string), stop reason, timings
     * @throws ContextFullException          when a decoder prompt is longer than the context
     * @throws UnsupportedOperationException when the backend has no memory
     * @throws IllegalArgumentException      for an empty prompt, a negative token id, or a model sampling default
     *                                       the sampler rejects
     */
    public GenerationResult generate(int[] prompt, GenerationRequest request, TokenCallback callback) {
        return generate(prompt, request, callback == null ? DISCARD : ReasoningFilter.contentTo(callback), null);
    }

    /**
     * Generates after {@code prompt}, routing every piece of output to {@code sink} with its channel.
     *
     * @param renderedPrompt the prompt text (for example the rendered chat template) used to detect a
     *                       generation prefix that opens reasoning; null renders it from the prompt tokens
     * @see #generate(int[], GenerationRequest, TokenCallback)
     */
    public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                     String renderedPrompt) {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(sink, "sink");
        requirePrompt(prompt);
        ModelMemory memory = backend.memory();
        if (memory == null) {
            throw new UnsupportedOperationException("autoregressive generation needs model memory; this backend"
                    + " runs without a cache (encoder, classifier or diffusion model)");
        }
        SequenceState seq = memory.sequence();
        int nCtx = seq.nCtx();
        if (!encoderDecoder && prompt.length > nCtx) {
            throw new ContextFullException(prompt.length, nCtx);
        }
        GenerationRequest effective = request.withModelDefaults(samplingDefaults);
        StopCriteria stop = StopCriteria.of(effective, tokenizer, reasoning);
        SamplerVocab vocab = Sampler.vocab(tokenizer);
        Sampler sampler = Sampler.create(samplerConfig(effective.sampler(), nCtx), stop.bannedTokens(),
                effective.logitBias(), RequestStages.of(effective, vocab), vocab);
        String rendered = renderedPrompt != null || reasoning.mode() == ReasoningFilter.Mode.NONE
                ? renderedPrompt : stop.renderPrompt(prompt);
        SequenceRun seqRun = SequenceRun.over(effective, tokenizer)
                .stopCriteria(stop)
                .sampler(sampler)
                .sink(sink)
                .renderedPrompt(rendered)
                .build();
        try {
            return run(prompt, seq, nCtx, seqRun);
        } catch (RuntimeException e) {
            try {
                backend.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            } finally {
                // PARITY §5.5.1 E-2: a failed step leaves no recoverable sequence; the state is cleared even when
                // the memory reset above throws, and never left half-written under a stored prefix.
                try {
                    seq.clear();
                } catch (RuntimeException clearFailure) {
                    e.addSuppressed(clearFailure);
                }
            }
            throw e;
        }
    }

    private GenerationResult run(int[] prompt, SequenceState seq, int nCtx, SequenceRun seqRun) {
        long t0 = System.nanoTime();
        int reused = prepareMemory(prompt, seq);
        int[] tail = reused == 0 ? prompt : Arrays.copyOfRange(prompt, reused, prompt.length);
        float[] logits = backend.prefill(tail, reused);
        int[] history = encoderDecoder ? seq.tokens() : prompt;
        seqRun.sampler().accept(history, 0, history.length);
        seqRun.prefilled(tail.length, System.nanoTime() - t0);

        boolean onDevice = seqRun.sampler().argmaxOnly() && !seqRun.wantsHostSampling();
        int deviceToken = -1;
        while (!seqRun.done()) {
            if (seqRun.cancelled()) {
                seqRun.stop(StopReason.CANCELLED);
                break;
            }
            int token = deviceToken >= 0 ? deviceToken : seqRun.next(logits);
            if (deviceToken >= 0) {
                seqRun.accept(token);
            }
            SequenceRun.Step step = seqRun.offer(token);
            if (step.stopped()) {
                break;
            }
            if (seq.nPast() >= nCtx && !shiftContext(prompt, seq)) {
                seqRun.stop(StopReason.CONTEXT_FULL);
                break;
            }
            int pos = seq.nPast();
            if (onDevice) {
                deviceToken = backend.decodeGreedyOnDevice(token, pos);
            } else {
                logits = backend.decode(token, pos);
            }
        }
        return seqRun.result();
    }

    /** {@code config} with a whole-context penalty window bounded to the context length. */
    private static SamplerConfig samplerConfig(SamplerConfig config, int nCtx) {
        return config.penaltyLastN() == SamplerConfig.PENALTY_WHOLE_CONTEXT
                ? config.toBuilder().penaltyLastN(nCtx).build() : config;
    }

    /**
     * Resets the memory for every request (the stored sequence is cleared with it).
     *
     * <p>The contiguous reuse path stays off until PC-9 turns it on (PARITY §5.5.1 E-2, no reuse before D-2's path
     * enablement): prefix reuse would have kept the longest common prefix ({@link SequenceState#commonPrefix}, at
     * most {@code prompt.length - 1} so the last prompt token is evaluated for logits) behind a
     * {@link ForwardBackend#canRollback} check; until then {@link #prefixReuse} and
     * {@link SequenceState#reusable} are only maintained for it, so a later enablement caps the reuse rows at the
     * rows computed with exactly that prefix.
     *
     * @return the number of prompt tokens already in memory ({@code 0} until PC-9)
     */
    private int prepareMemory(int[] prompt, SequenceState seq) {
        backend.reset();
        return 0;
    }

    /**
     * Frees positions by a context shift.
     *
     * @return false when shifting is disabled, nothing can be discarded or the memory cannot shift
     */
    private boolean shiftContext(int[] prompt, SequenceState seq) {
        if (!contextShift) {
            return false;
        }
        int nPast = seq.nPast();
        int bos = tokenizer.specialTokens().bos();
        int nKeep = Math.min(bos != SpecialTokens.NULL && prompt[0] == bos ? 1 : 0, nPast);
        int nDiscard = (nPast - nKeep) / 2;
        if (nDiscard <= 0) {
            return false;
        }
        try {
            backend.shiftContext(nKeep, nDiscard);
        } catch (UnsupportedOperationException e) {
            LOGGER.warn("context full and the context shift is unsupported: {}", e.getMessage());
            return false;
        }
        return true;
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
        return "GenerationLoop{" + reasoning + ", prefixReuse=" + prefixReuse + ", contextShift=" + contextShift
                + (encoderDecoder ? ", encoder-decoder" : "")
                + (samplingDefaults.isEmpty() ? "" : ", " + samplingDefaults) + '}';
    }

    public static final class Builder {
        private final ForwardBackend backend;
        private final GGUFTokenizer tokenizer;
        private ReasoningFilter reasoning = ReasoningFilter.none();
        private boolean prefixReuse = true;
        private boolean contextShift;
        private boolean encoderDecoder;
        private SamplingDefaults samplingDefaults = SamplingDefaults.NONE;

        private Builder(ForwardBackend backend, GGUFTokenizer tokenizer) {
            this.backend = Objects.requireNonNull(backend, "backend");
            this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        }

        /** Reasoning parser of the output (default {@link ReasoningFilter#none()}). */
        public Builder reasoning(ReasoningFilter v) { this.reasoning = Objects.requireNonNull(v, "reasoning"); return this; }

        /** Keep the common prompt prefix between generations (default true; {@code neuralfree.prefix.reuse}). */
        public Builder prefixReuse(boolean v) { this.prefixReuse = v; return this; }

        /** Shift the context instead of stopping when it is full (default false; {@code neuralfree.context.shift}). */
        public Builder contextShift(boolean v) { this.contextShift = v; return this; }

        /** Prefix reuse and context shift from memory settings. */
        public Builder memoryConfig(MemoryConfig v) {
            Objects.requireNonNull(v, "memoryConfig");
            this.prefixReuse = v.prefixReuse();
            this.contextShift = v.contextShift();
            return this;
        }

        /** The backend is an encoder-decoder (the prompt is the encoder source; memory is reset per call). */
        public Builder encoderDecoder(boolean v) { this.encoderDecoder = v; return this; }

        /** The model's sampling defaults applied to every request (default {@link SamplingDefaults#NONE}). */
        public Builder samplingDefaults(SamplingDefaults v) {
            this.samplingDefaults = Objects.requireNonNull(v, "samplingDefaults");
            return this;
        }

        public GenerationLoop build() {
            return new GenerationLoop(this);
        }
    }
}
