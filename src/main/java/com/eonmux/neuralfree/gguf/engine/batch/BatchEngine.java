package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
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
import com.eonmux.neuralfree.gguf.engine.sampling.TriggeredGrammarStage;
import com.eonmux.neuralfree.gguf.engine.sampling.RequestStages;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.Iterator;
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
 * the unreused part of its prompt next to the admitted prompts still waiting for prefill, evicting least recently
 * used cached prefixes to make room. That count is a hint: the paged reservation of each ubatch is all-or-nothing
 * and stays the authority. The engine prefills the admitted request (output logits of the last prompt row) and
 * decodes one token per pump per running sequence. A finished sequence reports its {@link GenerationResult} on its
 * {@link BatchSequence} handle and its slot and pages free at once, so the next pump can admit a queued request
 * into the same slot.
 *
 * <p>Running out of pages is recoverable, as vLLM's scheduler makes it. A prefill that finds the pools full rolls
 * back what it wrote and queues again at the head; admission then waits until a retirement or eviction frees
 * memory, so a prompt that cannot fit never starves the decodes. A decode step that finds the pools full rolls back
 * the rows it committed, then frees memory by evicting a cached prefix or, with none left, by preempting the
 * youngest running sequence (recompute preemption): its pages free while its run keeps its sampler, grammar, stop
 * filter and streamed text. Once a running sequence retires it is re-admitted ahead of the queue, its prompt and
 * fed tokens prefill again, and it continues with the token it had already sampled, so preemption never changes a
 * result. A sequence that cannot
 * fit even with every other sequence's memory freed fails with the {@link MemoryFullException}.
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
    /** Preempted sequences waiting to recompute, oldest first, re-admitted ahead of the queue. */
    private final Deque<Slot> preempted = new ArrayDeque<>();
    /**
     * Set when a prefill ran out of pages beside running sequences or a decode step preempted one: admission waits
     * until a retirement or an eviction frees memory.
     */
    private boolean memoryBlocked;
    private final PrefixCache cache;
    private final BatchStats.Builder stats = new BatchStats.Builder();
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
        ReasoningFilter rf = ReasoningParsers.forModel(config.reasoningParser(), tok);
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

    /** The engine's counters and current state; call it on the engine's thread, then share the snapshot freely. */
    public BatchStats stats() {
        stats.running = active.size();
        stats.queued = queue.size();
        stats.preempted = preempted.size();
        stats.totalCells = sequences.totalCells();
        stats.usedCells = sequences.totalCells() - sequences.freeCells();
        return stats.build();
    }

    /**
     * Drops every cached prefix and frees its pages: another user of the model reset the memory under them (a
     * pooling forward on the same backend), so their rows no longer hold the cached tokens. Call it only while the
     * engine is {@link #idle()}.
     *
     * @throws IllegalStateException when a submission is queued, running or preempted
     */
    public void clearPrefixCache() {
        if (!idle()) {
            throw new IllegalStateException("the prefix cache clears only on an idle engine");
        }
        while (evictCached()) {
            // each eviction frees one cached sequence
        }
    }

    /** Whether no submission is queued, running or preempted. */
    public boolean idle() {
        return queue.isEmpty() && active.isEmpty() && preempted.isEmpty();
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
     * com.eonmux.neuralfree.gguf.grammar.GbnfBuilder#compile}), so an invalid or untranslatable spec throws to
     * the caller before anything is queued; a compiled grammar runs as the sequence's sampling stage
     * ({@link com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage}), masking every token the grammar cannot
     * accept next (guided_json, guided_regex, guided_choice map onto the spec kinds). A trigger-gated constraint
     * runs as a {@link com.eonmux.neuralfree.gguf.engine.sampling.TriggeredGrammarStage}: free text until a
     * trigger fires, the grammar from the trigger on.
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
        com.eonmux.neuralfree.gguf.grammar.CompiledConstraint grammar = constraint == null ? null
                : com.eonmux.neuralfree.gguf.grammar.GbnfBuilder.compile(constraint,
                        text -> tokenizer.encode(text, false, true));
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
        for (Slot slot : preempted) {
            failed(slot.submission, new IllegalStateException("batch engine closed"));
        }
        preempted.clear();
        for (Submission sub : queue) {
            failed(sub, new IllegalStateException("batch engine closed"));
        }
        queue.clear();
    }

    // ------------------------------------------------------------------ engine steps

    /** Re-admits preempted sequences, then queued submissions, in order, while slots and memory allow. */
    private void admit() {
        if (memoryBlocked) {
            if (!active.isEmpty()) {
                return;
            }
            // nothing runs that could free memory: the gate would never open
            memoryBlocked = false;
        }
        dropCancelledPreempted();
        while (!preempted.isEmpty()) {
            Slot waiting = preempted.peek();
            if (!makeRoom(waiting.nextPos, null)) {
                return;
            }
            int seq = claimFreeSlot();
            if (seq < 0) {
                return;
            }
            preempted.poll();
            Slot moved = waiting.moveTo(seq);
            slots[seq] = moved;
            active.add(moved);
            moved.submission.handle.admit(seq);
        }
        while (!queue.isEmpty()) {
            Submission head = queue.peek();
            if (!makeRoom(head.prompt.length, head.prompt)) {
                return; // FIFO: the head waits until its prompt fits
            }
            int slot = claimSlot(head);
            if (slot < 0) {
                return;
            }
            queue.poll();
            try {
                active.add(openSlot(slot, head));
            } catch (RuntimeException e) {
                // the request's own pipeline refused to build: free what the claim adopted and report it
                slots[slot] = null;
                sequences.remove(slot, 0, -1);
                failed(head, e);
            }
        }
    }

    /**
     * The admission hint: evicts least recently used cached prefixes until the free cells cover the candidate's
     * {@code cells} beyond the prefix it would reuse ({@code prompt}, null for a recompute) plus the admitted
     * prompts still waiting for prefill. With nothing left to evict it admits only onto an engine with no running
     * sequence (the prefill then fits or fails by name); otherwise the candidate waits for running sequences.
     */
    private boolean makeRoom(int cells, int[] prompt) {
        while (true) {
            int need = cells - reusable(prompt) + pendingPrefillCells();
            if (sequences.freeCells() >= need) {
                return true;
            }
            if (!evictCached()) {
                return active.isEmpty();
            }
        }
    }

    /** The prompt tokens a cached prefix would let {@code prompt} skip (as {@link #claimSlot} adopts them). */
    private int reusable(int[] prompt) {
        if (prompt == null || cache == null) {
            return 0;
        }
        PrefixCache.Entry match = cache.longestPrefix(prompt);
        return match == null ? 0 : Math.min(PrefixCache.commonLength(match.tokens, prompt), prompt.length - 1);
    }

    /** Cells the admitted sequences that still wait for their prefill will write. */
    private int pendingPrefillCells() {
        int cells = 0;
        for (Slot slot : active) {
            if (slot.needsPrefill) {
                cells += slot.recompute ? slot.nextPos : slot.submission.prompt.length - slot.submission.reused;
            }
        }
        return cells;
    }

    /** Evicts the least recently used cached prefix and frees its pages; false when nothing is cached. */
    private boolean evictCached() {
        PrefixCache.Entry oldest = cache == null ? null : cache.evictOldest();
        if (oldest == null) {
            return false;
        }
        sequences.remove(oldest.seq, 0, -1);
        memoryBlocked = false;
        return true;
    }

    /** Finishes the preempted sequences whose request was cancelled while they waited (they hold no pages). */
    private void dropCancelledPreempted() {
        for (Iterator<Slot> it = preempted.iterator(); it.hasNext(); ) {
            Slot slot = it.next();
            if (slot.run.cancelled()) {
                it.remove();
                slot.run.stop(StopReason.CANCELLED);
                completed(slot.submission, slot.run.result(), true);
            }
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
        return claimFreeSlot();
    }

    /**
     * A free slot holding no pages: evicts the least recently used cached prefix when every slot is taken, and
     * drops the cached entry of the slot it hands out.
     */
    private int claimFreeSlot() {
        if (freeSlot() < 0) {
            evictCached();
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
            if (slot.needsPrefill) {
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
        if (slot.recompute) {
            recompute(slot);
            return;
        }
        Submission sub = slot.submission;
        int reused = sub.reused;
        int[] tail = reused == 0 ? sub.prompt
                : Arrays.copyOfRange(sub.prompt, reused, sub.prompt.length);
        try {
            long t0 = System.nanoTime();
            ForwardBatch.Builder batch = ForwardBatch.builder(tail.length);
            for (int i = 0; i < tail.length; i++) {
                batch.add(tail[i], reused + i, slot.seq, i == tail.length - 1);
            }
            float[] logits = backend.forwardBatch(batch.build())[0];
            long elapsed = System.nanoTime() - t0;
            slot.needsPrefill = false;
            slot.run.prefilled(tail.length, elapsed);
            stats.promptTokens += tail.length;
            stats.promptTokensCached += reused;
            stats.promptNanos += elapsed;
            slot.run.sampler().accept(sub.prompt, 0, sub.prompt.length);
            slot.nextPos = reused + tail.length;
            sampleAndOffer(slot, logits, false);
        } catch (MemoryFullException e) {
            requeue(slot, e);
        } catch (RuntimeException e) {
            // prepare is all-or-nothing: the failed prompt wrote nothing, the other sequences are untouched
            retire(slot, null, false, e);
        }
    }

    /**
     * Puts a submission whose prefill ran out of pages back at the head of the queue: the rows it wrote roll back
     * (the reused prefix returns to the cache), then a cached prefix is evicted, or admission waits for the running
     * sequences. With neither, nothing else holds memory and the request fails with the exception.
     */
    private void requeue(Slot slot, MemoryFullException e) {
        Submission sub = slot.submission;
        detach(slot);
        if (cache != null && sub.reused > 0) {
            sequences.remove(slot.seq, sub.reused, -1);
            cache.offer(slot.seq, Arrays.copyOf(sub.prompt, sub.reused));
        } else {
            sequences.remove(slot.seq, 0, -1);
        }
        sub.reused = 0;
        if (!evictCached()) {
            if (active.isEmpty()) {
                failed(sub, e);
                return;
            }
            memoryBlocked = true;
        }
        sub.handle.requeued();
        queue.addFirst(sub);
    }

    /**
     * Re-prefills a preempted sequence's prompt and fed tokens under its new slot; the token it sampled before the
     * preemption stays armed, so decoding continues where it stopped. Out of pages, it is preempted again.
     */
    private void recompute(Slot slot) {
        int[] history = slot.history();
        try {
            ForwardBatch.Builder batch = ForwardBatch.builder(history.length);
            for (int i = 0; i < history.length; i++) {
                batch.add(history[i], i, slot.seq, i == history.length - 1);
            }
            long t0 = System.nanoTime();
            backend.forwardBatch(batch.build());
            stats.promptNanos += System.nanoTime() - t0;
            stats.recomputedTokens += history.length;
            slot.recompute = false;
            slot.needsPrefill = false;
        } catch (MemoryFullException e) {
            detach(slot);
            sequences.remove(slot.seq, 0, -1);
            if (!evictCached()) {
                if (active.isEmpty()) {
                    failed(slot.submission, e);
                    return;
                }
                memoryBlocked = true;
            }
            preempted.addFirst(slot);
        } catch (RuntimeException e) {
            retire(slot, null, false, e);
        }
    }

    /**
     * Frees memory after a decode step ran out of pages: a cached prefix goes first, else the youngest running
     * sequence is preempted, and admission then waits for a retirement, so the survivors decode before the
     * preempted sequence recomputes (re-admitting it at once would refill the pool and preempt it again). False
     * when the only running sequence holds all the memory.
     */
    private boolean relieve() {
        if (evictCached()) {
            return true;
        }
        if (active.size() < 2) {
            return false;
        }
        Slot youngest = active.get(active.size() - 1);
        detach(youngest);
        sequences.remove(youngest.seq, 0, -1);
        preempted.addFirst(youngest);
        stats.preemptions++;
        memoryBlocked = true;
        return true;
    }

    /** Takes a slot off the running set without reporting on its handle; its pages are the caller's to free. */
    private void detach(Slot slot) {
        active.remove(slot);
        slots[slot.seq] = null;
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
            long t0 = System.nanoTime();
            rows = backend.forwardBatch(fb.build());
            stats.decodeNanos += System.nanoTime() - t0;
            stats.decodeSteps++;
            stats.decodeRows += batch.size();
        } catch (MemoryFullException e) {
            // the per-sequence ubatches before the full one committed their row: roll every row of the step back,
            // free memory, and let the next pump decode the same tokens again
            for (Slot slot : batch) {
                sequences.remove(slot.seq, slot.nextPos, -1);
            }
            if (!relieve()) {
                for (Slot slot : batch) {
                    retire(slot, null, false, e);
                }
            }
            return;
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
        stats.generatedTokens++;
        SequenceRun.Step step = slot.run.offer(token);
        if (step.stopped()) {
            // offer recorded the stop (end of generation, stop string, budget): the result carries its reason
            retire(slot, slot.run.result(), false, null);
            return;
        }
        if (decoded) {
            slot.fed(slot.pendingToken);
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
        detach(slot);
        memoryBlocked = false;
        try {
            if (error == null && cache != null && slot.nextPos > 0 && !slot.needsPrefill) {
                // the sequence's pages stay valid (a slot still waiting for its prefill or recompute holds
                // none): cache it by its exact tokens. The memory stores the prompt plus the fed decodes; the
                // final sampled token armed but unfed is not in the pages.
                int[] generated = result == null ? new int[0] : result.tokens();
                int decoded = Math.min(slot.nextPos - slot.submission.prompt.length, generated.length);
                int[] full = new int[slot.submission.prompt.length + Math.max(decoded, 0)];
                System.arraycopy(slot.submission.prompt, 0, full, 0, slot.submission.prompt.length);
                System.arraycopy(generated, 0, full, slot.submission.prompt.length, Math.max(decoded, 0));
                cache.offer(slot.seq, full);
                completed(slot.submission, result, cancelled);
                return;
            }
            sequences.remove(slot.seq, 0, -1);
        } catch (RuntimeException e) {
            if (error != null) {
                e.addSuppressed(error);
            }
            failed(slot.submission, e);
            return;
        }
        if (error != null) {
            failed(slot.submission, error);
        } else {
            completed(slot.submission, result, cancelled);
        }
    }

    /** Reports a finished submission on its handle and counts it under its stop reason and its length. */
    private void completed(Submission sub, GenerationResult result, boolean cancelled) {
        StopReason reason = cancelled || result == null ? StopReason.CANCELLED : result.stopReason();
        stats.finished[reason.ordinal()]++;
        int length = sub.prompt.length + (result == null ? 0 : result.tokens().length);
        stats.maxSequenceLength = Math.max(stats.maxSequenceLength, length);
        sub.handle.complete(result, cancelled);
    }

    /** Reports a failed submission on its handle and counts it. */
    private void failed(Submission sub, Throwable error) {
        stats.failed++;
        sub.handle.fail(error);
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
        SamplerVocab vocab = Sampler.vocab(tokenizer);
        List<SamplerStage> stages = new ArrayList<>(RequestStages.of(effective, vocab));
        Sampler sampler;
        if (sub.grammar != null) {
            stages.add(sub.grammar.gated()
                    ? new TriggeredGrammarStage(sub.grammar.grammar(), sub.grammar.triggers(), vocab)
                    : new GrammarStage(sub.grammar.grammar(), vocab));
        }
        if (stages.isEmpty()) {
            sampler = Sampler.create(samplerConfig(effective.sampler(), nCtx), stop.bannedTokens(),
                    effective.logitBias(), vocab);
        } else {
            sampler = Sampler.create(samplerConfig(effective.sampler(), nCtx), stop.bannedTokens(),
                    effective.logitBias(), stages, vocab);
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
        final com.eonmux.neuralfree.gguf.grammar.CompiledConstraint grammar;
        int reused;

        Submission(BatchSequence handle, int[] prompt, GenerationRequest request,
                   ReasoningFilter.Sink sink, String renderedPrompt,
                   com.eonmux.neuralfree.gguf.grammar.CompiledConstraint grammar) {
            this.handle = handle;
            this.prompt = prompt;
            this.request = request;
            this.sink = sink;
            this.renderedPrompt = renderedPrompt;
            this.grammar = grammar;
        }
    }

    /**
     * One running sequence: its slot id, run pipeline and decode cursor, and the decode tokens it fed (with the
     * prompt, the history a recompute after preemption prefills again).
     */
    private static final class Slot {
        final int seq;
        final Submission submission;
        final SequenceRun run;
        final int nCtx;
        int nextPos;
        int pendingToken = -1;
        boolean needsPrefill = true;
        /** Whether the pending prefill replays the history of a preempted run instead of a fresh prompt. */
        boolean recompute;
        private int[] fed = new int[16];
        private int fedCount;

        Slot(int seq, Submission submission, SequenceRun run, int nCtx) {
            this.seq = seq;
            this.submission = submission;
            this.run = run;
            this.nCtx = nCtx;
        }

        /** Records a decode token fed at the next position. */
        void fed(int token) {
            if (fedCount == fed.length) {
                fed = Arrays.copyOf(fed, fed.length * 2);
            }
            fed[fedCount++] = token;
        }

        /** The prompt and the fed decode tokens: the {@code nextPos} positions the sequence stores. */
        int[] history() {
            int[] out = Arrays.copyOf(submission.prompt, submission.prompt.length + fedCount);
            System.arraycopy(fed, 0, out, submission.prompt.length, fedCount);
            return out;
        }

        /** The same run re-admitted under {@code newSeq}, waiting to recompute its history. */
        Slot moveTo(int newSeq) {
            Slot moved = new Slot(newSeq, submission, run, nCtx);
            moved.fed = fed.clone();
            moved.fedCount = fedCount;
            moved.nextPos = nextPos;
            moved.pendingToken = pendingToken;
            moved.recompute = true;
            return moved;
        }
    }
}
