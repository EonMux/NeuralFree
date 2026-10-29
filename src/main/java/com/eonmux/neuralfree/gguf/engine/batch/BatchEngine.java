package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SamplingDefaults;
import com.eonmux.neuralfree.gguf.engine.SequenceRun;
import com.eonmux.neuralfree.gguf.engine.StopCriteria;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Objects;

/**
 * Continuous batching over the paged memory (PARITY plan P2): one open model, submitted prompts and pumped engine
 * steps. Every {@link #pump} call runs exactly one step — admitting queued requests, then one prompt prefill, or
 * one decode row per running sequence — through {@code ForwardBackend.forwardBatch}, which splits every batch into
 * one-sequence ubatches over the page tables. Each sequence runs its own {@link SequenceRun} pipeline (sample,
 * stop, route), so a batch computes the same texts as the same requests run one at a time through
 * {@link com.eonmux.neuralfree.gguf.engine.GenerationLoop}.
 *
 * <p>Lifecycle of a submission ({@link #submit}): the prompt is checked against the context length (an oversize
 * prompt throws {@link ContextFullException} to the caller, as {@code GenerationLoop} does), then queued FIFO.
 * {@code pump} admits the head of the queue when a sequence slot is free and the pools hold enough free cells for
 * the prompt (an admission hint; the per-ubatch reservation stays all-or-nothing), prefills it (output logits of
 * the last prompt row), and decodes one token per pump per running sequence. A finished sequence reports its
 * {@link GenerationResult} on its {@link BatchSequence} handle and its slot and pages free at once, so the next
 * pump can admit a queued request into the same slot.
 *
 * <p>Per request the engine mirrors {@code GenerationLoop}: the request takes the model's sampling defaults for
 * every unset setting, {@link StopCriteria} derives the end-of-generation set, the penalty history starts with the
 * prompt tokens, and the reasoning parser routes output pieces to the submission's sink. A sequence whose next
 * position would leave the context stops with {@link StopReason#CONTEXT_FULL}: the batch engine runs without
 * context shift ({@code neuralfree.context.shift} stays a single-sequence {@code GenerationLoop} feature); a
 * cancellation poll per step stops with {@link StopReason#CANCELLED}.
 *
 * <p>Failures are per sequence where the engine can attribute them: a prefill failure lands on that submission
 * alone ({@code PagedSequences.prepare} is all-or-nothing, nothing was written), while a decode step that throws
 * mid-batch fails every sequence of that batch — the split runs per-sequence ubatches in order and a throw
 * abandons the rest, so no returned row can be attributed. The failing sequences retire with
 * {@link BatchSequence.State#FAILED}; the memory slot and pages free either way.
 *
 * <p>Not thread-safe: one caller thread submits and pumps (the adapters drive it from a single worker). The
 * {@link BatchSequence} handles are safe to read from other threads. The engine owns no model: the caller opens
 * and closes it. The memory must be the paged form ({@code neuralfree.parallel > 1} or
 * {@code neuralfree.kv.paged=true}); every other memory refuses at construction by name, and a backend without a
 * batched forward (the GPU path today) refuses through its own {@code forwardBatch}.
 */
public final class BatchEngine implements AutoCloseable {

    private static final ReasoningFilter.Sink DISCARD = (channel, tokenId, text) -> { };

    private final ForwardBackend backend;
    private final GGUFTokenizer tokenizer;
    private final ReasoningFilter reasoning;
    private final SamplingDefaults samplingDefaults;
    private final SequenceMemory sequences;
    private final int nCtx;
    private final Slot[] slots;
    private final Deque<Submission> queue = new ArrayDeque<>();
    private final List<Slot> active = new ArrayList<>();
    private final PrefixCache cache;
    private int nextSubmitId;
    private boolean closed;

    private BatchEngine(ForwardBackend backend, GGUFTokenizer tokenizer, ReasoningFilter reasoning,
                        SamplingDefaults samplingDefaults, SequenceMemory sequences, int maxSequences,
                        boolean prefixReuse, boolean windowed) {
        this.backend = backend;
        this.tokenizer = tokenizer;
        this.reasoning = reasoning;
        this.samplingDefaults = samplingDefaults;
        this.sequences = sequences;
        this.nCtx = sequences.sequence(0).nCtx();
        this.slots = new Slot[maxSequences];
        // the prefix cache holds retired sequences whose pages stay valid: a windowed (SWA) memory trims old
        // positions per commit, so a cached prefix's rows can sit outside the window the next request reads
        this.cache = !prefixReuse || windowed ? null : new PrefixCache();
    }

    /**
     * A batch engine over {@code model}: refuses models that do not generate text (embeddings, classifiers,
     * diffusion, no support), encoder-decoders (the cross-attention batch path is not built), and memories that
     * hold no paged sequences, naming the properties that select them.
     *
     * @throws UnsupportedOperationException when the model or its memory cannot run batched generation
     */
    public static BatchEngine forModel(EngineModel model, EngineConfig config) {
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
            case ENC_DEC:
                throw new UnsupportedOperationException(model.architecture()
                        + " is an encoder-decoder; the batch engine runs decoder-only models (the cross-attention"
                        + " batch path is not built)");
            default:
                break;
        }
        ForwardBackend backend = model.forwardBackend();
        ModelMemory memory = backend.memory();
        if (!(memory instanceof CpuMemory)) {
            throw new UnsupportedOperationException(model.architecture() + ": batched generation needs paged memory"
                    + " on a CpuMemory backend, got "
                    + (memory == null ? "none" : memory.getClass().getSimpleName()));
        }
        PagedSequences paged = ((CpuMemory) memory).pagedSequences();
        if (paged == null) {
            throw new UnsupportedOperationException("batched generation needs paged memory"
                    + " (neuralfree.parallel > 1 or neuralfree.kv.paged=true)");
        }
        GGUFTokenizer tok = model.tokenizer();
        ReasoningFilter rf = ReasoningFilter.forChatTemplate(tok.chatTemplate()).withTagTokens(tok::getTokenId);
        boolean windowed = false;
        for (int g = 0; g < paged.plan().groups().size(); g++) {
            windowed |= paged.plan().group(g).window().type() != com.eonmux.neuralfree.gguf.arch.SwaType.NONE;
        }
        return new BatchEngine(backend, tok, rf, SamplingDefaults.forModel(model, config), paged,
                paged.maxSequences(), config.memory().prefixReuse(), windowed);
    }

    /** The model's tokenizer. */
    public GGUFTokenizer tokenizer() {
        return tokenizer;
    }

    /** Sequences the paged memory holds ({@code neuralfree.parallel}). */
    public int maxSequences() {
        return slots.length;
    }

    /** Submissions waiting for a slot. */
    public int queued() {
        return queue.size();
    }

    /** Positions the paged memory holds per sequence (the plan's {@code nCtx}). */
    public int contextLength() {
        return nCtx;
    }

    /** Submissions currently prefilling or decoding. */
    public int running() {
        return active.size();
    }

    /** Whether no submission is queued or running. */
    public boolean idle() {
        return queue.isEmpty() && active.isEmpty();
    }

    /**
     * Submits {@code prompt} for generation: validated against the context length now (an oversize or empty
     * prompt throws to the caller), then queued FIFO behind the admissions already waiting. The sink receives the
     * routed output pieces with their {@link ReasoningFilter.Channel} as the engine pumps; the finished text and
     * timings land on the returned handle's {@link BatchSequence#result()}.
     *
     * @param renderedPrompt the rendered prompt text for reasoning-prefix detection (for example the chat
     *                       template's rendering); null renders it from the prompt tokens
     * @throws IllegalStateException    when the engine is closed
     * @throws ContextFullException     when the prompt does not fit the context
     * @throws IllegalArgumentException when the prompt is empty or holds a negative token id
     */
    public BatchSequence submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                String renderedPrompt) {
        return submit(prompt, request, sink, renderedPrompt, null);
    }

    /**
     * Submits with a grammar constraint: {@code constraint} compiles now ({@link
     * com.eonmux.neuralfree.gguf.grammar.GbnfBuilder#grammarOf}), so an invalid or untranslatable spec throws to
     * the caller before anything is queued; a compiled grammar runs as the sequence's sampling stage
     * ({@link com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage}), masking every token the grammar cannot
     * accept next (guided_json, guided_regex, guided_choice map onto the spec kinds).
     *
     * @throws IllegalArgumentException when the constraint compiles to an unusable grammar
     */
    public BatchSequence submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                String renderedPrompt, com.eonmux.neuralfree.gguf.grammar.ConstraintSpec constraint) {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(sink, "sink");
        checkOpen();
        requirePrompt(prompt);
        if (prompt.length > nCtx) {
            throw new ContextFullException(prompt.length, nCtx);
        }
        com.eonmux.neuralfree.gguf.grammar.Grammar grammar = constraint == null ? null
                : com.eonmux.neuralfree.gguf.grammar.GbnfBuilder.grammarOf(constraint);
        BatchSequence handle = new BatchSequence(nextSubmitId++);
        queue.add(new Submission(handle, prompt, request, sink, renderedPrompt, grammar));
        return handle;
    }

    /** {@link #submit(int[], GenerationRequest, ReasoningFilter.Sink, String)} with a content-only callback. */
    public BatchSequence submit(int[] prompt, GenerationRequest request, TokenCallback callback) {
        return submit(prompt, request, callback == null ? DISCARD : ReasoningFilter.contentTo(callback), null, null);
    }

    /**
     * Runs one engine step: admits queued requests while a slot and the pages of its prompt are free, then
     * prefills the oldest admitted request, or — when none waits for prefill — decodes one token per running
     * sequence. A pump over no submissions does nothing. Every routed piece reaches the submission's sink before
     * the call returns; finished sequences carry their result on their handle before it returns too.
     */
    public void pump() {
        checkOpen();
        admit();
        Slot prefilling = firstWaitingPrefill();
        if (prefilling != null) {
            prefill(prefilling);
            return;
        }
        if (active.isEmpty()) {
            return;
        }
        decodeStep();
    }

    /**
     * Pumps until every submission finished, failed or was cancelled.
     *
     * @throws RuntimeException when a pump step throws after retiring the sequences it could not attribute (see
     *                          the class doc); queued submissions then still drain through later calls
     */
    public void drain() {
        while (!idle()) {
            pump();
        }
    }

    /**
     * Frees every sequence slot and queued submission. The backing model stays open (the caller owns it); further
     * submits refuse.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        for (Slot slot : new ArrayList<>(active)) {
            retire(slot, null, false, new IllegalStateException("batch engine closed"));
        }
        active.clear();
        for (Submission sub : queue) {
            sub.handle.fail(new IllegalStateException("batch engine closed"));
        }
        queue.clear();
    }

    // ------------------------------------------------------------------ engine steps

    private void admit() {
        while (!queue.isEmpty()) {
            Submission head = queue.peek();
            int slot = claimSlot(head);
            if (slot < 0 || sequences.freeCells() < head.prompt.length) {
                return; // FIFO: the head waits until its prompt fits whole
            }
            queue.poll();
            active.add(openSlot(slot, head));
        }
    }

    /**
     * The slot for the head submission: when the prompt extends a cached prefix, that cached sequence is adopted
     * in place — its slot, its pages, its divergent tail cut at the reused length ({@code remove(seq, reused, -1)})
     * — so the request continues the exact rows its prompt's prefix already computed. Without a match a slot is
     * freed by evicting the least recently used cached prefix when none is idle.
     */
    private int claimSlot(Submission head) {
        if (cache != null) {
            PrefixCache.Entry entry = cache.longestPrefix(head.prompt);
            if (entry != null) {
                int reused = Math.min(PrefixCache.commonLength(entry.tokens, head.prompt),
                        head.prompt.length - 1);
                cache.discard(entry.seq);
                if (reused >= 1) {
                    sequences.remove(entry.seq, reused, -1);
                    head.reused = reused;
                    head.handle.reused(reused);
                    return entry.seq;
                }
                sequences.remove(entry.seq, 0, -1);
            }
        }
        if (freeSlot() < 0 && cache != null) {
            PrefixCache.Entry oldest = cache.evictOldest();
            if (oldest != null) {
                sequences.remove(oldest.seq, 0, -1);
            }
        }
        int slot = freeSlot();
        if (slot >= 0 && cache != null && cache.holds(slot)) {
            // the new generation overwrites this slot: drop its cached entry and free its pages
            cache.discard(slot);
            sequences.remove(slot, 0, -1);
        }
        return slot;
    }

    private Slot firstWaitingPrefill() {
        for (Slot slot : active) {
            if (slot.pendingToken < 0) {
                return slot;
            }
        }
        return null;
    }

    /** Prefills the oldest admitted submission: the prompt's tail after any reused prefix, logits of its last row. */
    private void prefill(Slot slot) {
        if (slot.run.cancelled()) {
            finish(slot, StopReason.CANCELLED);
            return;
        }
        Submission sub = slot.submission;
        int reused = sub.reused;
        int[] tail = reused == 0 ? sub.prompt
                : java.util.Arrays.copyOfRange(sub.prompt, reused, sub.prompt.length);
        try {
            long t0 = System.nanoTime();
            ForwardBatch.Builder batch = ForwardBatch.builder(tail.length);
            for (int i = 0; i < tail.length; i++) {
                batch.add(tail[i], reused + i, slot.seq, i == tail.length - 1);
            }
            float[] logits = backend.forwardBatch(batch.build())[0];
            slot.run.prefilled(tail.length, System.nanoTime() - t0);
            slot.run.sampler().accept(sub.prompt, 0, sub.prompt.length);
            slot.nextPos = reused + tail.length;
            sampleAndOffer(slot, logits, false);
        } catch (RuntimeException e) {
            // prepare is all-or-nothing: the failed prompt wrote nothing, the other sequences are untouched
            retire(slot, null, false, e);
        }
    }

    /** Decodes one token per running sequence: one output row each, sampled through the sequence's own chain. */
    private void decodeStep() {
        List<Slot> batch = new ArrayList<>(active);
        ForwardBatch.Builder fb = ForwardBatch.builder(batch.size());
        for (Slot slot : batch) {
            fb.add(slot.pendingToken, slot.nextPos, slot.seq, true);
        }
        float[][] rows;
        try {
            rows = backend.forwardBatch(fb.build());
        } catch (RuntimeException e) {
            // the split runs per-sequence ubatches in order; a throw abandons the rest of the batch, so no row
            // can be attributed: the whole pump's sequences fail together (rare path, see class doc)
            for (Slot slot : batch) {
                retire(slot, null, false, e);
            }
            return;
        }
        for (int i = 0; i < batch.size(); i++) {
            Slot slot = batch.get(i);
            try {
                sampleAndOffer(slot, rows[i], true);
            } catch (RuntimeException e) {
                // this slot's row is unpersisted and every later slot went unsampled; the earlier slots sampled
                // fine: retire from here, naming the sampling failure (rare path, see class doc)
                retire(slot, null, false, e);
                for (int j = i + 1; j < batch.size(); j++) {
                    retire(batch.get(j), null, false, e);
                }
                return;
            }
        }
    }

    /**
     * Samples one logits row through the sequence's chain, offers the token and retires or arms the slot. After a
     * prefill ({@code decoded} false) the sampled token waits to be fed at {@code nextPos}; a decoded token arms
     * at the next position.
     */
    private void sampleAndOffer(Slot slot, float[] logits, boolean decoded) {
        if (slot.run.cancelled()) {
            finish(slot, StopReason.CANCELLED);
            return;
        }
        int token = slot.run.next(logits);
        SequenceRun.Step step = slot.run.offer(token);
        if (step.stopped()) {
            // offer recorded the stop (end of generation, stop string, budget): the result carries its reason
            retire(slot, slot.run.result(), false, null);
            return;
        }
        if (decoded) {
            slot.nextPos++;
        }
        slot.pendingToken = token;
        if (slot.nextPos >= slot.nCtx) {
            finish(slot, StopReason.CONTEXT_FULL);
        }
    }

    /** Stops a sequence with an external reason: its result lands on the handle and its slot frees at once. */
    private void finish(Slot slot, StopReason reason) {
        slot.run.stop(reason);
        retire(slot, slot.run.result(), reason == StopReason.CANCELLED, null);
    }

    /** Frees the paged sequence and the slot, and reports the outcome on the handle. */
    private void retire(Slot slot, GenerationResult result, boolean cancelled, Throwable error) {
        active.remove(slot);
        slots[slot.seq] = null;
        try {
            if (error == null && cache != null && slot.nextPos > 0) {
                // the sequence's pages stay valid: cache it by its exact tokens. The memory stores the prompt
                // plus the fed decodes; the final sampled token armed but unfed is not in the pages.
                int[] generated = result == null ? new int[0] : result.tokens();
                int decoded = Math.min(slot.nextPos - slot.submission.prompt.length, generated.length);
                int[] full = new int[slot.submission.prompt.length + Math.max(decoded, 0)];
                System.arraycopy(slot.submission.prompt, 0, full, 0, slot.submission.prompt.length);
                System.arraycopy(generated, 0, full, slot.submission.prompt.length, Math.max(decoded, 0));
                cache.offer(slot.seq, full);
                slot.submission.handle.complete(result, cancelled);
                return;
            }
            sequences.remove(slot.seq, 0, -1);
        } catch (RuntimeException e) {
            if (error != null) {
                e.addSuppressed(error);
            }
            slot.submission.handle.fail(e);
            return;
        }
        if (error != null) {
            slot.submission.handle.fail(error);
        } else {
            slot.submission.handle.complete(result, cancelled);
        }
    }

    // ------------------------------------------------------------------ slots and submissions

    private int freeSlot() {
        for (int i = 0; i < slots.length; i++) {
            if (slots[i] == null) {
                return i;
            }
        }
        return -1;
    }

    /** Opens a slot for an admitted submission: the sequence's own stop criteria, sampler and run. */
    private Slot openSlot(int seq, Submission sub) {
        GenerationRequest effective = sub.request.withModelDefaults(samplingDefaults);
        StopCriteria stop = StopCriteria.of(effective, tokenizer, reasoning);
        Sampler sampler;
        if (sub.grammar != null) {
            SamplerVocab vocab = Sampler.vocab(tokenizer);
            sampler = Sampler.create(samplerConfig(effective.sampler(), nCtx), stop.bannedTokens(),
                    List.of(new GrammarStage(sub.grammar, vocab)), vocab);
        } else {
            sampler = Sampler.create(samplerConfig(effective.sampler(), nCtx), stop.bannedTokens());
        }
        String rendered = sub.renderedPrompt != null || reasoning.mode() == ReasoningFilter.Mode.NONE
                ? sub.renderedPrompt : stop.renderPrompt(sub.prompt);
        SequenceRun run = SequenceRun.over(effective, tokenizer)
                .stopCriteria(stop)
                .sampler(sampler)
                .sink(sub.sink)
                .renderedPrompt(rendered)
                .build();
        Slot slot = new Slot(seq, sub, run, nCtx);
        slots[seq] = slot;
        sub.handle.admit(seq);
        return slot;
    }

    /** {@code config} with a whole-context penalty window bounded to the context length (as GenerationLoop). */
    private static SamplerConfig samplerConfig(SamplerConfig config, int nCtx) {
        return config.penaltyLastN() == SamplerConfig.PENALTY_WHOLE_CONTEXT
                ? config.toBuilder().penaltyLastN(nCtx).build() : config;
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("batch engine is closed");
        }
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

    /** One submitted request: its prompt and output side, queued until a slot admits it. */
    private static final class Submission {
        final BatchSequence handle;
        final int[] prompt;
        final GenerationRequest request;
        final ReasoningFilter.Sink sink;
        final String renderedPrompt;
        final com.eonmux.neuralfree.gguf.grammar.Grammar grammar;
        int reused;

        Submission(BatchSequence handle, int[] prompt, GenerationRequest request,
                   ReasoningFilter.Sink sink, String renderedPrompt,
                   com.eonmux.neuralfree.gguf.grammar.Grammar grammar) {
            this.handle = handle;
            this.prompt = prompt;
            this.request = request;
            this.sink = sink;
            this.renderedPrompt = renderedPrompt;
            this.grammar = grammar;
        }
    }

    /** One running sequence: its slot id, run pipeline and decode cursor. */
    private static final class Slot {
        final int seq;
        final Submission submission;
        final SequenceRun run;
        final int nCtx;
        int nextPos;
        int pendingToken = -1;

        Slot(int seq, Submission submission, SequenceRun run, int nCtx) {
            this.seq = seq;
            this.submission = submission;
            this.run = run;
            this.nCtx = nCtx;
        }
    }
}
