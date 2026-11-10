package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessors;
import com.eonmux.neuralfree.gguf.engine.sampling.RequestStages;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.grammar.ConstraintCompiler;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Lookahead (Jacobi) decoding over a target {@link ForwardBackend}, the decoding scheme of Fu et al. (arXiv
 * 2402.02057) and llama.cpp's {@code examples/lookahead/lookahead.cpp}: every step maintains a window of {@code W}
 * Jacobi guesses at the next positions, refines them through {@code N - 1} Jacobi iterations run in parallel, and
 * verifies the n-grams observed in the history against the target in the same forward batch, so a step emits the
 * longest observed n-gram the target agrees with.
 *
 * <p><b>Sequences.</b> The scheme runs over a paged memory and needs {@code W + G + 1} sequences
 * ({@link LookaheadOptions#sequenceCount()}, llama.cpp's {@code n_parallel = W + G + 1}) with unified KV, so every
 * sequence spans the whole context. Sequence 0 is the accepted sequence; sequences {@code 1 .. W} are the Jacobi
 * diagonals; sequences {@code W + 1 .. W + G} verify one observed n-gram each. A step copies sequence 0 into all
 * the others (page sharing, llama.cpp's {@code llama_memory_seq_cp}), writes one batched forward call of all of them
 * ({@link ForwardBackend#forwardBatch}), samples the accepted token from the first verification row and compares it
 * with every still-active n-gram, then keeps the sequence that matched and copies it back over sequence 0.
 *
 * <p><b>Batch.</b> The step's batch holds, in this order, the input token at each sequence's next position, the
 * {@code N - 1} following tokens of every active n-gram, the window's first-level guesses (each read by every
 * sequence that starts after it) and the {@code N - 2} later Jacobi levels. Row {@code i} of level {@code j}
 * carries guess {@code i} of level {@code j} at position {@code n_past + j + i} of sequence {@code i + 1}, so each
 * sequence holds one diagonal of the Jacobi grid. Positions never exceed {@link LookaheadOptions#positionSpan()}.
 *
 * <p><b>Acceptance.</b> A step samples the token after each verification row and keeps the active n-grams whose next
 * token it equals; the walk continues while an n-gram matches, and the last sampled token becomes the next input
 * token. This is the match rule of llama.cpp's example and of {@link SpeculativeDecoder}'s {@code MATCH} mode, over
 * the n-gram tree: a step emits a confirmed n-gram prefix plus the target's own next token. The n-gram table is fed
 * with the Jacobi diagonals of every step (llama.cpp's {@code ngram_container}), so observed continuations become
 * verification candidates.
 *
 * <p>Stops, sampling, penalties, grammars, logprobs and the reasoning parser are the plain loop's: every emitted token
 * runs through the request's {@link Sampler} and {@link SequenceRun}, so the output side matches
 * {@link GenerationLoop}. The request's per-request LoRA adapters and prompt-row scoring are refused by name, as are
 * a target without paged memory, with fewer sequences than {@link LookaheadOptions#sequenceCount()}, and a memory
 * whose recurrent, compressed or context-shifted state cannot rewind the lookahead rows. Each generation resets the
 * memory: lookahead decoding keeps no reusable prefix. A step whose rows do not fit the page pools ends the
 * generation with {@link StopReason#CONTEXT_FULL}.
 *
 * <p>The settings are immutable and the decoder keeps no per-generation state; the last generation's
 * {@link Stats} stay readable through {@link #lastStats()}. One generation runs at a time, on the backend's thread.
 */
public final class LookaheadDecoder {

    private static final ReasoningFilter.Sink DISCARD = (channel, tokenId, text) -> { };

    /** Counts of one generation's lookahead steps. Immutable. */
    public static final class Stats {
        private final int rounds;
        private final int emitted;
        private final int accepted;

        Stats(int rounds, int emitted, int accepted) {
            this.rounds = rounds;
            this.emitted = emitted;
            this.accepted = accepted;
        }

        /** Lookahead steps run (llama.cpp's example {@code n_drafted}). */
        public int rounds() { return rounds; }

        /** Tokens the generation emitted. */
        public int emitted() { return emitted; }

        /** Tokens accepted from an observed n-gram beyond each step's input token (llama.cpp {@code n_accept}). */
        public int accepted() { return accepted; }

        /** {@code accepted / rounds}, the mean accepted tokens per step beyond the input token, 0 without steps. */
        public double acceptanceRate() {
            return rounds == 0 ? 0.0 : (double) accepted / rounds;
        }

        @Override
        public String toString() {
            return "Stats{rounds=" + rounds + ", emitted=" + emitted + ", accepted=" + accepted + '}';
        }
    }

    private final ForwardBackend backend;
    private final GGUFTokenizer tokenizer;
    /** The sampler's view of {@link #tokenizer}, shared by every request so its piece caches build once. */
    private final SamplerVocab samplerVocab;
    /** The request constraints compiled over {@link #tokenizer}, remembered across requests. */
    private final ConstraintCompiler constraints;
    private final ReasoningFilter reasoning;
    private final SamplingDefaults samplingDefaults;
    private final LogitsProcessors logitsProcessors;
    private final Consumer<Stats> statsSink;
    private final int window;
    private final int ngramSize;
    private final int maxNgrams;
    private final int sequences;
    private final LookaheadOptions options;
    /** Positions a step spans at most, from its input row through its deepest row ({@link LookaheadOptions#positionSpan()}). */
    private final int positionSpan;
    /** The last generation's stats, null before the first. */
    private volatile Stats lastStats;

    private LookaheadDecoder(Builder b) {
        this.backend = b.backend;
        this.tokenizer = b.tokenizer;
        this.samplerVocab = Sampler.vocab(b.tokenizer);
        this.constraints = new ConstraintCompiler(text -> tokenizer.encode(text, false, true));
        this.reasoning = b.reasoning;
        this.samplingDefaults = b.samplingDefaults;
        this.logitsProcessors = b.logitsProcessors;
        this.statsSink = b.statsSink;
        this.window = b.options.window();
        this.ngramSize = b.options.ngramSize();
        this.maxNgrams = b.options.maxNgrams();
        this.sequences = b.options.sequenceCount();
        this.positionSpan = b.options.positionSpan();
        this.options = b.options;
    }

    /**
     * @param backend the target's forward backend, over a paged memory
     */
    public static Builder builder(ForwardBackend backend, GGUFTokenizer tokenizer, LookaheadOptions options) {
        return new Builder(backend, tokenizer, options);
    }

    /**
     * The decoder of an engine model and its configuration: the configured reasoning parser
     * ({@link EngineConfig#reasoningParser()}, the chat template's by default), the target's sampling defaults unless
     * {@code config} turns them off ({@link SamplingDefaults#forModel}) and the lookahead sizes of
     * {@code config.lookaheadOptions()}.
     *
     * @throws UnsupportedOperationException for models that do not generate autoregressively
     * @throws IllegalArgumentException      when {@code config} does not enable lookahead decoding
     */
    public static LookaheadDecoder forModel(EngineModel model, EngineConfig config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        if (!config.lookaheadOptions().enabled()) {
            throw new IllegalArgumentException(LookaheadOptions.PROP_ENABLED + " is off; lookahead decoding needs"
                    + " it on (or build the decoder with an enabled LookaheadOptions)");
        }
        SupportClass support = model.supportClass();
        switch (support) {
            case EMBED:
            case CLS:
            case NA:
            case DIFF:
            case ENC_DEC:
                throw new UnsupportedOperationException(model.architecture() + " (" + support
                        + ") has no decoder-only token loop to decode with lookahead");
            default:
                break;
        }
        GGUFTokenizer tok = model.tokenizer();
        ReasoningFilter rf = ReasoningParsers.forModel(config.reasoningParser(), config.toolParser(), tok);
        return builder(model.forwardBackend(), tok, config.lookaheadOptions())
                .reasoning(rf)
                .samplingDefaults(SamplingDefaults.forModel(model, config))
                .logitsProcessors(LogitsProcessors.forConfig(config))
                .build();
    }

    public ReasoningFilter reasoning() { return reasoning; }
    public SamplingDefaults samplingDefaults() { return samplingDefaults; }
    public int window() { return window; }
    public int ngramSize() { return ngramSize; }
    public int maxNgrams() { return maxNgrams; }
    /** The sequences a step runs, {@code W + G + 1}. */
    public int sequences() { return sequences; }

    /** The last generation's {@link Stats}, or null before the first generation. */
    public Stats lastStats() { return lastStats; }

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
     * @throws ContextFullException          when the prompt is longer than a sequence's context, or the memory cannot
     *                                       hold the prompt (the lookahead rows end the generation with
     *                                       {@link StopReason#CONTEXT_FULL})
     * @throws UnsupportedOperationException when the backend has no paged multi-sequence memory, when the memory
     *                                       holds fewer than {@code W + G + 1} sequences, has recurrent or compressed
     *                                       state, or the request scores its prompt rows or selects LoRA adapters
     * @throws IllegalArgumentException      for an empty prompt, a negative token id, a prompt that leaves no room
     *                                       for the lookahead window, or a model sampling default the sampler rejects
     */
    public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                     String renderedPrompt) {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(sink, "sink");
        requirePrompt(prompt);
        request.requireNoPromptRows("lookahead decoding");
        if (request.lora() != null) {
            throw new UnsupportedOperationException("lora: lookahead decoding runs the target's startup adapters"
                    + " only; it does not select per-request adapters (" + request.lora() + ")");
        }
        ModelMemory memory = backend.memory();
        if (memory == null) {
            throw new UnsupportedOperationException("lookahead decoding needs model memory; this backend runs"
                    + " without a cache");
        }
        SequenceMemory seqs = requirePaged(memory);
        int cap = seqs.sequence(0).nCtx();
        if (prompt.length > cap) {
            throw new ContextFullException(prompt.length, cap);
        }
        int span = positionSpan;
        if (prompt.length + span > cap) {
            throw new IllegalArgumentException("lookahead decoding writes " + span + " positions past the prompt,"
                    + " which a sequence context of " + cap + " holds for a prompt of " + prompt.length
                    + "; raise neuralfree.ctx or lower the lookahead window and n-gram size");
        }
        GenerationRequest effective = request.withModelDefaults(samplingDefaults);
        SequenceRun run = sequenceRun(effective, sink, renderedPrompt, prompt, cap);
        try {
            return run(prompt, seqs, run);
        } catch (RuntimeException e) {
            try {
                backend.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            throw e;
        }
    }

    /**
     * The paged sequence memory a lookahead step runs over: it needs {@link #sequences()} sequences, a paged memory
     * the batched forward can address, and a state that rewinding the lookahead rows leaves consistent.
     */
    private SequenceMemory requirePaged(ModelMemory memory) {
        SequenceMemory seqs = memory.sequences();
        if (memory.pagedSequences() == null) {
            throw new UnsupportedOperationException("lookahead decoding needs paged memory with " + sequences
                    + " sequences; set neuralfree.parallel=" + sequences + " and neuralfree.kv.unified=true");
        }
        if (seqs.maxSequences() < sequences) {
            throw new UnsupportedOperationException("lookahead decoding needs " + sequences + " sequences ("
                    + window + " window + " + maxNgrams + " n-grams + 1); the memory holds "
                    + seqs.maxSequences() + ". Set neuralfree.parallel=" + sequences
                    + " and neuralfree.kv.unified=true");
        }
        if (memory.hasRecurrentState()) {
            throw new UnsupportedOperationException("lookahead decoding rewinds the rows of a step, which state a"
                    + " model with recurrent state cannot restore");
        }
        if (memory.plan().hasCompressed()) {
            throw new UnsupportedOperationException("lookahead decoding rewinds the rows of a step, which a memory"
                    + " with compressed caches cannot restore");
        }
        com.eonmux.neuralfree.gguf.memory.paged.PagedSequences pages = memory.pagedSequences();
        for (int g = 0; g < pages.groups(); g++) {
            if (pages.trims(g)) {
                throw new UnsupportedOperationException("lookahead decoding rewinds the rows of a step, which a"
                        + " sliding-window memory trims away");
            }
        }
        return seqs;
    }

    /** The request's output side, built as {@link GenerationLoop} builds it. */
    private SequenceRun sequenceRun(GenerationRequest effective, ReasoningFilter.Sink sink, String renderedPrompt,
                                    int[] prompt, int nCtx) {
        ReasoningFilter filter = GenerationLoop.forRequest(reasoning, effective);
        StopCriteria stop = StopCriteria.of(effective, tokenizer, filter);
        String rendered = renderedPrompt != null || filter.mode() == ReasoningFilter.Mode.NONE
                ? renderedPrompt : stop.renderPrompt(prompt);
        SamplerVocab vocab = samplerVocab;
        List<SamplerStage> extra = new ArrayList<>();
        SamplerStage grammar = RequestStages.constraintStage(effective.constraint(), constraints, vocab);
        if (grammar != null) {
            extra.add(grammar);
        }
        extra.addAll(RequestStages.of(effective, vocab, filter, rendered));
        Sampler sampler = Sampler.create(samplerConfig(effective.sampler(), nCtx), stop.bannedTokens(),
                effective.logitBias(), extra, logitsProcessors.stages(effective), vocab);
        return SequenceRun.over(effective, tokenizer)
                .stopCriteria(stop)
                .sampler(sampler)
                .sink(sink)
                .renderedPrompt(rendered)
                .build();
    }

    /** The generation: prefill, the lookahead steps, and the result over the run. */
    private GenerationResult run(int[] prompt, SequenceMemory seqs, SequenceRun run) {
        backend.reset();
        long t0 = System.nanoTime();
        float[] logits = prefill(prompt);
        long t1 = System.nanoTime();
        run.sampler().accept(prompt, 0, prompt.length);
        run.prefilled(prompt.length, t1 - t0);
        LookaheadRun lookahead = new LookaheadRun(seqs, 0, options, run, tokenizer.vocabSize());
        try {
            lookahead.initialize();
        } catch (MemoryFullException e) {
            run.stop(StopReason.CONTEXT_FULL);
            return finish(run, 0, 0, 0);
        }
        int nPast = prompt.length;
        int rounds = 0;
        int emitted = 0;
        int accepted = 0;
        if (run.cancelled()) {
            run.stop(StopReason.CANCELLED);
            return finish(run, rounds, emitted, accepted);
        }
        int token = run.next(logits);
        SequenceRun.Step step = run.offer(token);
        emitted++;
        if (step.stopped()) {
            return finish(run, rounds, emitted, accepted);
        }
        while (true) {
            if (run.cancelled()) {
                run.stop(StopReason.CANCELLED);
                break;
            }
            LookaheadRun.Step rows = lookahead.step(token, nPast);
            float[][] out;
            try {
                out = backend.forwardBatch(rows.batch());
            } catch (MemoryFullException e) {
                run.stop(StopReason.CONTEXT_FULL);
                break;
            }
            rounds++;
            LookaheadRun.Result stepResult = lookahead.walk(rows, out, run::cancelled);
            emitted += stepResult.emitted();
            nPast += stepResult.emitted();
            accepted += stepResult.accepted();
            if (stepResult.stopped()) {
                if (stepResult.cancelled()) {
                    run.stop(StopReason.CANCELLED);
                }
                break;
            }
            lookahead.commit(nPast, stepResult.best());
            int[] stepTokens = stepResult.tokens();
            if (stepTokens.length == 0) {
                throw new IllegalStateException("a lookahead step that did not stop emitted no token");
            }
            token = stepTokens[stepTokens.length - 1];
        }
        return finish(run, rounds, emitted, accepted);
    }

    /** Records the generation's stats and builds its result. */
    private GenerationResult finish(SequenceRun run, int rounds, int emitted, int accepted) {
        Stats stats = new Stats(rounds, emitted, accepted);
        lastStats = stats;
        statsSink.accept(stats);
        return run.result();
    }

    /** Prefills the prompt into sequence 0 through one batched forward and returns the last row's logits. */
    private float[] prefill(int[] prompt) {
        ForwardBatch.Builder b = ForwardBatch.builder(prompt.length);
        for (int i = 0; i < prompt.length; i++) {
            b.add(prompt[i], i, 0, i == prompt.length - 1);
        }
        float[][] rows = backend.forwardBatch(b.build());
        if (rows.length != 1) {
            throw new IllegalStateException("the prompt prefill returned " + rows.length + " logits rows for one"
                    + " requested row");
        }
        return rows[0];
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
        return "LookaheadDecoder{window=" + window + ", ngram=" + ngramSize + ", maxNgrams=" + maxNgrams
                + ", sequences=" + sequences + '}';
    }

    public static final class Builder {
        private final ForwardBackend backend;
        private final GGUFTokenizer tokenizer;
        private final LookaheadOptions options;
        private ReasoningFilter reasoning = ReasoningFilter.none();
        private SamplingDefaults samplingDefaults = SamplingDefaults.NONE;
        private LogitsProcessors logitsProcessors = LogitsProcessors.empty();
        private Consumer<Stats> statsSink = s -> { };

        private Builder(ForwardBackend backend, GGUFTokenizer tokenizer, LookaheadOptions options) {
            this.backend = Objects.requireNonNull(backend, "backend");
            this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
            this.options = Objects.requireNonNull(options, "options");
            if (!options.enabled()) {
                throw new IllegalArgumentException("lookahead options are off; enable them to build the decoder");
            }
        }

        /** Reasoning parser of the output (default {@link ReasoningFilter#none()}). */
        public Builder reasoning(ReasoningFilter v) {
            this.reasoning = Objects.requireNonNull(v, "reasoning");
            return this;
        }

        /** The target model's sampling defaults applied to every request (default {@link SamplingDefaults#NONE}). */
        public Builder samplingDefaults(SamplingDefaults v) {
            this.samplingDefaults = Objects.requireNonNull(v, "samplingDefaults");
            return this;
        }

        /** The custom logits processors every step consults (default none). */
        public Builder logitsProcessors(LogitsProcessors v) {
            this.logitsProcessors = Objects.requireNonNull(v, "logitsProcessors");
            return this;
        }

        /** Receives each generation's {@link Stats} before its result is built (default: discarded). */
        public Builder statsSink(Consumer<Stats> v) {
            this.statsSink = Objects.requireNonNull(v, "statsSink");
            return this;
        }

        public LookaheadDecoder build() {
            return new LookaheadDecoder(this);
        }
    }
}
