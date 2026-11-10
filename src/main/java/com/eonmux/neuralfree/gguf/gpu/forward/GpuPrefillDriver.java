package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.engine.EmbeddingRowSupport;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.DraftTree;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.engine.OutputKind;
import com.eonmux.neuralfree.gguf.engine.TreeAccept;
import com.eonmux.neuralfree.gguf.engine.BatchSplitter;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.UBatch;
import com.eonmux.neuralfree.gguf.engine.lora.AdapterRows;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PageCopy;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Runs one forward call over any number of tokens as a sequence of ubatch steps (llama.cpp {@code llama_decode}
 * splitting a batch into ubatches) and reads back only what the caller asked for.
 *
 * <p>Per step of {@code n} tokens at position {@code p} ({@code n} at most the scratch ubatch and, when the call
 * writes memory, the memory plan's ubatch, split by {@code MemoryPlan.stepTokens} so the trailing tokens a
 * compressor rollback may drop stay in the last step): {@code GpuStepInputs.upload(p, n)} when the call
 * writes memory, the four-section positions into {@link ScratchSlots#POS}, the embedding stage, the layer executor
 * (DECODE for a single token against memory, PREFILL otherwise), then the head on the requested rows of that step:
 * their logits, their hidden rows ({@code t_embd}) or their greedy ids, each enqueued as a read into a host array.
 * Tap points the {@link HiddenTap} wants are enqueued as reads too. After the last step the queue is finished once;
 * that is the only host synchronization of the call (a call that fails while enqueuing drains the queue before its
 * pending reads are released). Memory bookkeeping ({@code SequenceState.append}) follows each
 * step.</p>
 *
 * <p>A call without memory evaluates all tokens in one step (non-causal attention cannot be split), so it needs
 * {@code tokens.length <= ubatch}. A one-token causal step ({@code StepMode.DECODE}) runs its trunk through the
 * {@link GpuDecodeReplay}: after {@code neuralfree.gpu.graphWarmupSteps} consecutive steps of the same shape the
 * trunk's dispatches are captured, and the following ones replay them as a single submission. A single-sequence
 * prompt pass and a memory reset keep the capture and re-verify it on the next steps; a batched call invalidates
 * it. Not thread-safe.</p>
 */
public final class GpuPrefillDriver {

    /** What a call reads back per requested row. */
    public enum Output {
        /** {@code vocab} logits. */
        LOGITS,
        /** {@code n_embd} hidden values after the output norm. */
        HIDDEN,
        /** The greedy token id. */
        ARGMAX,
        /** The row's {@code k} best candidates in rank order and their logits ({@code LLM_Row_TopK}). */
        TOPK
    }

    /** Rows read back by a call, in request order. */
    public static final class Result {
        private final float[][] rows;
        private final int[] ids;
        private final int topK;
        private final int[] topIds;
        private final float[] topLogits;

        Result(float[][] rows, int[] ids) {
            this(rows, ids, 0, null, null);
        }

        Result(float[][] rows, int[] ids, int topK, int[] topIds, float[] topLogits) {
            this.rows = rows;
            this.ids = ids;
            this.topK = topK;
            this.topIds = topIds;
            this.topLogits = topLogits;
        }

        /** Logits or hidden rows ({@link Output#LOGITS}, {@link Output#HIDDEN}); null for ARGMAX and TOPK. */
        public float[][] rows() {
            return rows;
        }

        /** Greedy ids ({@link Output#ARGMAX}); null otherwise. */
        public int[] ids() {
            return ids;
        }

        /** Candidates per row ({@link Output#TOPK}); 0 otherwise. */
        public int topK() {
            return topK;
        }

        /** Candidate ids in rank order, {@code rows * topK} ({@link Output#TOPK}); null otherwise. */
        public int[] topIds() {
            return topIds;
        }

        /** Candidate logits in rank order, {@code rows * topK} ({@link Output#TOPK}); null otherwise. */
        public float[] topLogits() {
            return topLogits;
        }
    }

    private final GpuForwardContext ctx;
    private final GpuEmbeddingStage embedding;
    private final GpuLayerExecutor executor;
    private final GpuLogitsHead head;
    private final ComputeQueue queue;
    private final GpuStepPositions positions;
    private final GpuDecodeReplay decodeReplay;
    /** The trunk replay of the memory-backed multi-row causal steps (a chain round's verify of k + 1 rows). */
    private final GpuDecodeReplay stepReplay;
    /** Whether the speculative round's steps capture their trunks ({@code neuralfree.gpu.chainReplay}). */
    private final boolean chainReplay;
    private final int ubatch;
    /** The id an embedding row records in the token record ({@link EmbeddingRowSupport#recordToken}). */
    private final int embeddingRowToken;
    private StepMirror mirror;
    /** The runner of a PREFILL step's trunk (the prefill context-parallel coordinator), or null. */
    private TrunkRunner trunkRunner;
    /** The runner of a batched PREFILL step's trunk (the coordinator's paged mode), or null. */
    private BatchTrunkRunner batchTrunkRunner;
    /** The open recurrent tree round relayed from the backend, or null when no tree verify runs. */
    private GpuRecurrentTreeRound recurrentRound;
    /** True when a cross-device collective keeps the decode trunk out of a single-queue graph replay. */
    private boolean replayDisabled;
    /**
     * The shape of the plain paged decode step (every segment one row) whose trunk {@link #stepReplay} captured or
     * is warming up for, or null when it holds no such step: a paged decode capture serves only the consecutive
     * paged decode steps of its shape, and any other step drops it first.
     */
    private PagedDecodeShape pagedDecodeShape;

    /**
     * A receiver of each step's positions and tokens, called after the main memory uploaded the step and after the
     * step was recorded: a tensor-parallel group mirrors both onto its per-rank caches.
     */
    public interface StepMirror {
        /** The main memory uploaded positions {@code [pos0, pos0 + nTokens)}. */
        void beforeStep(int pos0, int nTokens);

        /** The main memory recorded {@code n} token rows from {@code tokens} at {@code off}. */
        void afterTokens(int[] tokens, int off, int n);

        /** The main memory recorded {@code n} embedding rows carrying {@code token} ({@code -1} for no id). */
        void afterEmbeddings(int n, int token);

        /** The main memory ran {@code copy} on its page storage (a paged step's copy). */
        default void copyPage(com.eonmux.neuralfree.gguf.memory.paged.PageCopy copy) {
        }

        /** The main memory uploaded the paged step and its row positions; the step runs until the end hook. */
        default void beginPagedStep(com.eonmux.neuralfree.gguf.memory.paged.PagedStep step, int[] positions,
                                    int[][] ropeSections, boolean nonCausal) {
        }

        /** The paged step began by {@link #beginPagedStep} is over. */
        default void endPagedStep() {
        }
    }

    public GpuPrefillDriver(GpuForwardContext ctx, GpuEmbeddingStage embedding, GpuLayerExecutor executor,
                            GpuLogitsHead head) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.embedding = Objects.requireNonNull(embedding, "embedding");
        this.executor = Objects.requireNonNull(executor, "executor");
        this.head = Objects.requireNonNull(head, "head");
        this.queue = ctx.queue();
        this.positions = new GpuStepPositions(ctx.scratch(), queue, ctx.spec().hparams().mropeTextPositions());
        this.decodeReplay = new GpuDecodeReplay(queue);
        this.stepReplay = new GpuDecodeReplay(queue);
        this.chainReplay = GpuForwardBackend.chainReplayConfigured(ctx.backend().backendType());
        this.ubatch = ctx.scratch().ubatch();
        this.embeddingRowToken = EmbeddingRowSupport.recordToken(ctx.spec());
    }

    /** Installs the per-step mirror, or null to remove it. */
    public void stepMirror(StepMirror mirror) {
        this.mirror = mirror;
    }

    /**
     * The runner of one PREFILL step's trunk between the embedding stage and the head: the prefill context-parallel
     * coordinator, which packs each rank's DualChunkSwap rows, runs the layers over the ranks and restores the
     * query order before the head reads the step's rows. A trunk step skips the plain step-input upload and the
     * {@link StepMirror#beforeStep} hook; the runner uploads every rank's packed step inputs itself.
     */
    public interface TrunkRunner {

        /**
         * Runs the trunk of the PREFILL step of {@code nTokens} tokens at {@code pos0} in {@code mode}. The stream
         * {@code X} holds the step's embedding rows on entry and must hold the stack output in row order when the
         * runner returns.
         *
         * @throws UnsupportedOperationException when the runner cannot carry the step (a hidden tap the split
         *                                       cannot serve, a memoryless call, a looped schedule)
         */
        void runTrunk(int pos0, int nTokens, GpuLayerExecutor.StepMode mode, GpuLayerExecutor.TapSink taps);
    }

    /** Installs the prefill trunk runner, or null to remove it. */
    public void trunkRunner(TrunkRunner runner) {
        this.trunkRunner = runner;
    }

    /**
     * The runner of one batched PREFILL step's trunk between the embedding stage and the head: the prefill
     * context-parallel coordinator's paged mode, which partitions every multi-row segment of the step over the
     * ranks' DualChunkSwap chunk pairs, replicates the step's single-row (decode) segments onto every rank, and
     * runs the layers over the ranks' packed rows through the per-rank translated page tables. A batched trunk
     * step skips the plain executor run; the runner uploads every rank's packed step inputs itself.
     */
    public interface BatchTrunkRunner {

        /**
         * Whether this runner carries the batched step of {@code layout}: a paged step that holds at least one
         * multi-row segment. A step of single-row segments only (the pure decode batch) keeps the plain executor
         * run and the decode-context-parallel protocol of the mixers.
         */
        boolean runsBatchedTrunk(StepLayout layout);

        /**
         * Runs the trunk of the batched step of {@code layout}. The stream {@code X} holds the step's embedding
         * rows in the layout's row order on entry and must hold the stack output in the same row order when the
         * runner returns.
         *
         * @throws UnsupportedOperationException when the runner cannot carry the step (a hidden tap the split
         *                                       cannot serve)
         */
        void runBatchTrunk(StepLayout layout, GpuLayerExecutor.StepMode mode, GpuLayerExecutor.TapSink taps);
    }

    /** Installs the batched prefill trunk runner, or null to remove it. */
    public void batchTrunkRunner(BatchTrunkRunner runner) {
        this.batchTrunkRunner = runner;
    }

    /**
     * Relays the open recurrent tree round to the executor: while it is set, the batched steps whose tree mask the
     * caller opens a round for run the recurrent layers' mixer blocks through it. The backend sets it around one
     * masked forward and clears it after.
     */
    public void recurrentTreeRound(GpuRecurrentTreeRound round) {
        this.recurrentRound = round;
    }

    /**
     * Keeps the decode trunk out of a captured graph: a split whose mixers synchronize or enqueue on peer queues
     * between the main queue's dispatches cannot be recorded by a one-queue graph, so every decode step runs
     * directly. Idempotent.
     */
    public void disableDecodeReplay() {
        replayDisabled = true;
        decodeReplay.invalidate();
        invalidateStepReplay();
    }

    /**
     * The decode-step replay's state: the captured graph's dispatch count once one exists, else the warmup progress
     * and whether the replay is disabled.
     */
    String decodeReplayStats() {
        return decodeReplay + " " + stepReplay;
    }

    /**
     * Evaluates {@code tokens} at positions {@code pos0 ..} and reads back {@code output} for {@code outputRows}.
     *
     * @param types      token types (BERT segments) parallel to {@code tokens}, or null
     * @param outputRows indices into {@code tokens} in result order (repeats allowed), or null for every row
     * @param useMemory  true to write the model memory (causal calls); the memory must hold exactly {@code pos0}
     *                   positions
     * @param tap        hidden-state observer of this call
     * @throws ContextFullException     when the positions exceed the memory's context length; nothing is enqueued
     * @throws IllegalArgumentException for empty tokens, a row outside the call, or a memoryless call longer than
     *                                  one ubatch
     * @throws IllegalStateException    when the memory does not end at {@code pos0}
     * @throws UnsupportedOperationException when logits or ids are asked of a model without an LM head
     */
    public Result run(int[] tokens, int[] types, int pos0, int[] outputRows, Output output, boolean useMemory,
                      HiddenTap tap) {
        Objects.requireNonNull(tokens, "tokens");
        return run(tokens, null, types, pos0, outputRows, output, 0, useMemory, tap);
    }

    /**
     * {@link #run(int[], int[], int, int[], Output, boolean, HiddenTap)} with a top-k readback: per requested row
     * the {@code topK} best candidates in rank order and their logits ({@link Output#TOPK}, {@code LLM_Row_TopK}).
     *
     * @throws IllegalArgumentException when {@code topK} is below one
     */
    public Result runTopK(int[] tokens, int pos0, int[] outputRows, int topK, HiddenTap tap) {
        return runTopK(tokens, pos0, outputRows, topK, tap, null);
    }

    /**
     * {@link #runTopK(int[], int, int[], int, HiddenTap)} with per-row pre-top-k logit edits applied on the device
     * before the readout ({@code LLM_LogitsPrefilter}): the logit-bias table, the penalties and DRY. {@code
     * prefilter} holds one row per requested output row (null or empty for none).
     */
    public Result runTopK(int[] tokens, int pos0, int[] outputRows, int topK, HiddenTap tap,
                          LogitPrefilter prefilter) {
        Objects.requireNonNull(tokens, "tokens");
        if (topK < 1) {
            throw new IllegalArgumentException("top-k readback of " + topK + " candidates");
        }
        return run(tokens, null, null, pos0, outputRows, Output.TOPK, topK, true, tap, prefilter);
    }

    /**
     * {@link #run(int[], int[], int, int[], Output, boolean, HiddenTap)} over precomputed embedding rows instead of
     * tokens ({@code GpuEmbeddingStage.embedRows}): one row per position from {@code pos0}, recorded in the memory
     * as embedding positions.
     */
    public Result runRows(float[][] rows, int pos0, int[] outputRows, Output output, HiddenTap tap) {
        Objects.requireNonNull(rows, "rows");
        return run(null, rows, null, pos0, outputRows, output, 0, true, tap);
    }

    private Result run(int[] tokens, float[][] embd, int[] types, int pos0, int[] outputRows, Output output,
                       int topK, boolean useMemory, HiddenTap tap) {
        return run(tokens, embd, types, pos0, outputRows, output, topK, useMemory, tap, null);
    }

    private Result run(int[] tokens, float[][] embd, int[] types, int pos0, int[] outputRows, Output output,
                       int topK, boolean useMemory, HiddenTap tap, LogitPrefilter prefilter) {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(tap, "tap");
        if (prefilter != null && prefilter.rows() != 0 && topK < 1) {
            throw new IllegalArgumentException("a logit prefilter needs a top-k readback");
        }
        int total = embd != null ? embd.length : tokens.length;
        if (total == 0) {
            throw new IllegalArgumentException("forward call without tokens");
        }
        if (types != null && types.length != total) {
            throw new IllegalArgumentException(types.length + " token types for " + total + " tokens");
        }
        int[] rows = outputRows != null ? outputRows.clone() : identity(total);
        if (rows.length == 0) {
            throw new IllegalArgumentException("forward call without output rows");
        }
        for (int r : rows) {
            if (r < 0 || r >= total) {
                throw new IllegalArgumentException("output row " + r + " is outside the " + total + " tokens");
            }
        }
        if (output != Output.HIDDEN && !head.hasLmHead()) {
            throw new UnsupportedOperationException(ctx.spec().arch().ggufName() + " has no LM head; only hidden"
                    + " rows can be read");
        }
        GpuMemory memory = useMemory ? ctx.memory() : null;
        checkSpan(memory, pos0, total);
        GpuCallReads calls = new GpuCallReads(head, queue, rows.length, output, topK, prefilter);
        GpuTapReads taps = new GpuTapReads(queue, tap, total, 0);
        boolean finished = false;
        try {
            int chunk = memory != null ? Math.min(ubatch, memory.plan().ubatch()) : ubatch;
            for (int c0 = 0; c0 < total; ) {
                int n = memory != null ? memory.plan().stepTokens(total - c0, chunk) : Math.min(chunk, total - c0);
                step(tokens, embd, types, pos0 + c0, c0, n, rows, output, topK, memory, calls, taps);
                if (memory != null && embd != null) {
                    memory.sequence().appendEmbeddings(n, embeddingRowToken);
                    if (mirror != null) {
                        mirror.afterEmbeddings(n, embeddingRowToken);
                    }
                } else if (memory != null) {
                    memory.sequence().append(tokens, c0, n);
                    if (mirror != null) {
                        mirror.afterTokens(tokens, c0, n);
                    }
                }
                c0 += n;
            }
            finished = true;
            queue.finish();
            taps.deliver();
            return calls.result();
        } catch (RuntimeException e) {
            if (!finished) {
                try {
                    queue.finish();
                } catch (RuntimeException drain) {
                    e.addSuppressed(drain);
                }
            }
            throw e;
        } finally {
            calls.closeEvents();
            taps.closeEvents();
        }
    }

    private void checkSpan(GpuMemory memory, int pos0, int total) {
        if (memory == null) {
            if (total > ubatch) {
                throw new IllegalArgumentException(total + " tokens exceed the ubatch of " + ubatch + "; a call without"
                        + " memory runs in one step, raise neuralfree.ubatch");
            }
            if (pos0 != 0) {
                throw new IllegalArgumentException("a call without memory starts at position 0, got " + pos0);
            }
            return;
        }
        int nCtx = memory.plan().nCtx();
        long end = (long) pos0 + total;
        if (end > nCtx) {
            throw new ContextFullException((int) Math.min(Integer.MAX_VALUE, end), nCtx);
        }
        int past = memory.sequence().nPast();
        if (pos0 != past) {
            throw new IllegalStateException("the memory holds " + past + " positions, the call starts at " + pos0);
        }
    }

    /**
     * Evaluates a whole {@link ForwardBatch} as one memoryless step (a model planned without memory: an encoder or
     * a decision graph that keeps no cache, so every row attends every row): per-row tokens and embedding rows at
     * the batch's own positions, every requested output row read back in batch order. The batch must hold one
     * sequence and at most one ubatch, and carry no adapter slots outside slot 0. M-RoPE sections are refused: the
     * memoryless graphs read normal rope (llama.cpp's cache-less decision models), so a batch without
     * {@code ropePositions} fills the rows' positions from the text pattern.
     *
     * @param output {@link Output#LOGITS} or {@link Output#HIDDEN}
     * @throws IllegalArgumentException  when the batch holds several sequences, exceeds one ubatch, or the model
     *                                   holds memory
     * @throws UnsupportedOperationException when logits are asked of a model without an LM head, a row runs in an
     *                                       adapter slot that holds no adapter, the batch carries M-RoPE sections,
     *                                       or embedding rows the model cannot take
     */
    public Result runMemorylessBatch(ForwardBatch batch, Output output, HiddenTap tap) {
        Objects.requireNonNull(batch, "batch");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(tap, "tap");
        if (output != Output.HIDDEN && output != Output.LOGITS) {
            throw new IllegalArgumentException("a memoryless batch reads logits or hidden rows, not " + output);
        }
        if (ctx.hasMemory()) {
            throw new IllegalArgumentException(ctx.spec().arch().ggufName() + " is planned with memory; a"
                    + " memoryless batch needs a model without memory");
        }
        int n = batch.nTokens();
        if (n > ubatch) {
            throw new IllegalArgumentException("a memoryless batch of " + n + " rows exceeds one ubatch of "
                    + ubatch + "; raise neuralfree.ubatch");
        }
        int[] seqIds = batch.seqIds();
        for (int t = 0; t < seqIds.length; t++) {
            if (seqIds[t] != 0) {
                throw new UnsupportedOperationException("a memoryless batch holds one sequence, row " + t
                        + " runs in sequence " + seqIds[t]);
            }
        }
        if (output != Output.HIDDEN && !head.hasLmHead()) {
            throw new UnsupportedOperationException(ctx.spec().arch().ggufName() + " has no LM head; only hidden"
                    + " rows can be read");
        }
        String refusal = batch.adapterSlotRefusal(getClass().getSimpleName(), ctx.projectionDeltas()::holdsSlot);
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        int[][] ropeSections = batch.ropePositions();
        if (ropeSections != null) {
            throw new UnsupportedOperationException("a memoryless batch reads its rows' positions from"
                    + " ForwardBatch.positions(); " + ctx.spec().arch().ggufName() + " takes no M-RoPE sections");
        }
        if (batch.treeMask() != null) {
            throw new UnsupportedOperationException("a memoryless batch runs dense rows; a tree verify batch needs"
                    + " a memory with a KV cache to mask");
        }
        float[][] embd = batch.embeddings();
        if (embd != null) {
            String embdRefusal = EmbeddingRowSupport.refusal(ctx.spec());
            if (embdRefusal != null) {
                throw new UnsupportedOperationException(embdRefusal);
            }
        }
        for (int i = 0; i < n; i++) {
            if (batch.produces(i, OutputKind.ROUTING)) {
                throw new UnsupportedOperationException("row " + i + " requests " + OutputKind.ROUTING
                        + "; the memoryless batch reads logits (LOGITS) or hidden rows (HIDDEN)");
            }
        }
        int[] rows = batch.outputRows();
        GpuCallReads calls = new GpuCallReads(head, queue, rows.length, output, 0, null);
        GpuTapReads taps = new GpuTapReads(queue, tap, n, 0);
        boolean finished = false;
        try {
            executor.prefetchFirst();
            positions.write(batch.positions(), n);
            embedding.embedBatched(batch.tokens(), batch.positions(), null, embd);
            taps.batchRows(identity(n));
            decodeReplay.invalidate();
            invalidateStepReplay();
            executor.run(0, n, GpuLayerExecutor.StepMode.PREFILL, taps);
            ScratchSlot hidden = head.hidden(n, rows, rows.length);
            calls.read(output, hidden, rows.length, rows);
            finished = true;
            queue.finish();
            taps.deliver();
            return calls.result();
        } catch (RuntimeException e) {
            if (!finished) {
                try {
                    queue.finish();
                } catch (RuntimeException drain) {
                    e.addSuppressed(drain);
                }
            }
            throw e;
        } finally {
            calls.closeEvents();
            taps.closeEvents();
        }
    }

    /**
     * Evaluates a {@link ForwardBatch} over the paged memory: the rows split into multi-sequence ubatches
     * (llama.cpp {@code split_equal}, vLLM's ragged batches — one segment per sequence, sequences advancing
     * together), each ubatch reserving its pages all-or-nothing ({@code PagedSequences.prepare}), running the
     * device page copies (the step's own and the ones engine-side mutations still owe), committing the
     * reservation so the trunk's queries see the copied rows' positions (the CPU backend's ordering), uploading
     * its page tables, per-sequence addressing and its rows' RoPE sections ({@code beginPagedStep}, with the
     * batch's mask rule), writing its token and embedding rows by kind and stepping the trunk through them. The
     * output rows' logits or hidden rows come back in batch order; the hidden tap receives every row at its batch
     * row. Each ubatch sets the adapter slot of its rows ({@code ForwardBatch.adapterSlots}) in the projection
     * deltas' {@link AdapterRows} table, and the output rows' slots before the head; a model whose blocks cannot mix
     * slots in one product ({@link AdapterRows#mixable} false, or recurrent state) splits into slot-uniform ubatches.
     *
     * @param ropeSections section-major M-RoPE positions of every batch row, or null to derive them from the
     *                     segments
     * @param output       {@link Output#LOGITS}, {@link Output#HIDDEN} or (through
     *                     {@link #runBatchTopK}) {@link Output#TOPK}
     * @throws UnsupportedOperationException when the memory holds no paged sequences, or logits are asked of a
     *                                       model without an LM head
     * @throws ContextFullException          (as {@code MemoryFullException}) when a ubatch cannot reserve its pages
     */
    public Result runBatch(ForwardBatch batch, int[][] ropeSections, Output output, HiddenTap tap) {
        return runBatch(batch, ropeSections, output, 0, tap);
    }

    /**
     * {@link #runBatch(ForwardBatch, int[][], Output, HiddenTap)} with a top-k readback: per output row the
     * {@code topK} best candidates in rank order and their logits ({@link Output#TOPK}, {@code LLM_Row_TopK}).
     *
     * @throws IllegalArgumentException when {@code topK} is below one
     */
    public Result runBatchTopK(ForwardBatch batch, int[][] ropeSections, int topK, HiddenTap tap) {
        return runBatchTopK(batch, ropeSections, topK, tap, null);
    }

    /**
     * {@link #runBatchTopK(ForwardBatch, int[][], int, HiddenTap)} with per-output-row pre-top-k logit edits
     * applied on the device before the readout ({@code LLM_LogitsPrefilter}). {@code prefilter} holds one row per
     * output row of the batch (null or empty for none).
     */
    public Result runBatchTopK(ForwardBatch batch, int[][] ropeSections, int topK, HiddenTap tap,
                               LogitPrefilter prefilter) {
        if (topK < 1) {
            throw new IllegalArgumentException("top-k readback of " + topK + " candidates");
        }
        return runBatch(batch, ropeSections, Output.TOPK, topK, tap, prefilter);
    }

    private Result runBatch(ForwardBatch batch, int[][] ropeSections, Output output, int topK, HiddenTap tap) {
        return runBatch(batch, ropeSections, output, topK, tap, null);
    }

    private Result runBatch(ForwardBatch batch, int[][] ropeSections, Output output, int topK, HiddenTap tap,
                            LogitPrefilter prefilter) {
        Objects.requireNonNull(batch, "batch");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(tap, "tap");
        if (output == Output.ARGMAX) {
            throw new IllegalArgumentException("a batched call reads logits, hidden rows or top-k candidates,"
                    + " not " + output);
        }
        if (output == Output.TOPK && topK < 1) {
            throw new IllegalArgumentException("a batched top-k readback needs at least one candidate");
        }
        if (output != Output.HIDDEN && !head.hasLmHead()) {
            throw new UnsupportedOperationException(ctx.spec().arch().ggufName() + " has no LM head; only hidden"
                    + " rows can be read");
        }
        DeviceModelMemory memory = pagedMemory();
        PagedSequences paged = memory.pagedSequences();
        if (batch.treeMask() != null) {
            if (BatchInvariance.enabled()) {
                throw new UnsupportedOperationException("a tree verify batch needs the paged prefill kernel, and"
                        + " batch invariance (neuralfree.batch-invariant) runs every segment through the per-row"
                        + " fallback");
            }
            if (batch.embeddings() != null) {
                throw new UnsupportedOperationException("a tree verify batch runs token rows only, got embedding"
                        + " rows");
            }
        }
        BatchSteps steps = new BatchSteps(batch, memory, ropeSections, output, topK, tap, prefilter);
        decodeReplay.invalidate();
        int chunk = Math.min(ubatch, memory.plan().ubatch());
        boolean slotUniform = batch.adapterSlots() != null
                && (memory.plan().hasRecurrentState() || !AdapterRows.mixable(ctx.spec()));
        // one single-ubatch masked tree verify step keeps its step-replay capture across rounds, and so does one
        // single-ubatch plain paged decode step across steps of its shape (BatchSteps.ubatch drops a capture of
        // another shape or kind): their uploads (page copies, step inputs, tree mask, embeddings, row appends) run
        // outside the captured trunk and a commit's row moves dispatch directly between captures
        // (GpuForwardBackend.carryRowMoves), so a compaction does not force a recapture. Anything else
        // invalidates as before.
        boolean singleUbatch = batch.nTokens() <= chunk && !slotUniform;
        if (!(singleUbatch && (batch.treeMask() == null || recurrentRound == null))) {
            invalidateStepReplay();
        }
        // the copies engine-side mutations still owe (an adopt cut, a fork's prefix copy) run before the step's
        // own, as the CPU backend's ubatch orders them; their sources release once the queue finished
        for (PageCopy copy : paged.pendingCopies()) {
            memory.copyPage(copy);
            if (mirror != null) {
                mirror.copyPage(copy);
            }
            steps.pendingCopies.add(copy);
        }
        boolean finished = false;
        try {
            BatchSplitter splitter = new BatchSplitter(true);
            for (UBatch u : splitter.split(batch, chunk, BatchSplitter.Mode.EQUAL, slotUniform)) {
                steps.ubatch(u);
            }
            finished = true;
            queue.finish();
            steps.taps.deliver();
            return steps.calls.result();
        } catch (RuntimeException e) {
            if (!finished) {
                try {
                    queue.finish();
                } catch (RuntimeException drain) {
                    e.addSuppressed(drain);
                }
            }
            throw e;
        } finally {
            releaseCopies(paged, steps.pendingCopies);
            steps.calls.closeEvents();
            steps.taps.closeEvents();
        }
    }

    /**
     * One masked single-pass tree verify batch with its acceptance walked on the device: the batch runs as the
     * tree verify's one ubatch exactly as {@link #runBatch} (the same memory writes, the same trunk), the head
     * enqueues the rows' logits and their greedy ids, and the {@code LLM_TreeAccept} kernel follows the accepted
     * path through the proposal's parent array over them; only {@code 3 + nNodes} ints come back, against the
     * {@code (width + 1) * vocab} floats of a logits readback.
     *
     * @param batch    the tree verify batch {@code TreeVerify.batch} builds: the pending token plus the proposal's
     *                 nodes (shorter trees padded) under the per-row ancestor masks, at most one ubatch's rows
     * @param proposal the proposal the batch verifies, at most 63 nodes (the kernel's frontier bit set)
     * @return the device's acceptance walk of the batch
     * @throws UnsupportedOperationException when the batch is not the one masked single-ubatch step the walk runs
     *                                       over (no tree mask, embedding rows, batch invariance, a plan whose
     *                                       ubatch cannot hold the batch, per-row adapter slots), the backend
     *                                       provides no {@code LLM_TreeAccept} kernel, or the proposal exceeds 63
     *                                       nodes
     */
    public TreeAccept runBatchTreeAccept(ForwardBatch batch, int[][] ropeSections, DraftTree proposal, HiddenTap tap) {
        Objects.requireNonNull(batch, "batch");
        Objects.requireNonNull(proposal, "proposal");
        Objects.requireNonNull(tap, "tap");
        if (!head.hasLmHead()) {
            throw new UnsupportedOperationException(ctx.spec().arch().ggufName() + " has no LM head; the device"
                    + " accept walk reads the rows' greedy ids");
        }
        DeviceModelMemory memory = pagedMemory();
        PagedSequences paged = memory.pagedSequences();
        if (batch.treeMask() == null) {
            throw new UnsupportedOperationException("the device accept walks a tree verify batch's ancestor masks,"
                    + " got a batch without a tree mask (forwardBatch reads the rows for the host walk)");
        }
        if (batch.embeddings() != null) {
            throw new UnsupportedOperationException("a tree verify batch runs token rows only, got embedding rows");
        }
        if (BatchInvariance.enabled()) {
            throw new UnsupportedOperationException("a tree verify batch needs the paged prefill kernel, and batch"
                    + " invariance (neuralfree.batch-invariant) runs every segment through the per-row fallback");
        }
        int chunk = Math.min(batch.nTokens(), memory.plan().ubatch());
        if (batch.nTokens() > chunk) {
            throw new UnsupportedOperationException("the device accept runs the tree verify batch as one ubatch,"
                    + " got " + batch.nTokens() + " rows for a plan of " + chunk);
        }
        boolean slotUniform = batch.adapterSlots() != null
                && (memory.plan().hasRecurrentState() || !AdapterRows.mixable(ctx.spec()));
        if (slotUniform) {
            throw new UnsupportedOperationException("the device accept runs a batch whose rows carry no per-row"
                    + " adapter slots");
        }
        GpuTreeAcceptReadout accept = GpuTreeAcceptReadout.bind(ctx.ops(), queue, ctx.scratch(), proposal);
        BatchSteps steps = new BatchSteps(batch, memory, ropeSections, Output.LOGITS, 0, tap, null, accept);
        decodeReplay.invalidate();
        // the capture from the previous round survives: see runBatch
        for (PageCopy copy : paged.pendingCopies()) {
            memory.copyPage(copy);
            if (mirror != null) {
                mirror.copyPage(copy);
            }
            steps.pendingCopies.add(copy);
        }
        boolean finished = false;
        try {
            BatchSplitter splitter = new BatchSplitter(true);
            for (UBatch u : splitter.split(batch, chunk, BatchSplitter.Mode.EQUAL, slotUniform)) {
                steps.ubatch(u);
            }
            finished = true;
            queue.finish();
            steps.taps.deliver();
            return accept.result();
        } catch (RuntimeException e) {
            if (!finished) {
                try {
                    queue.finish();
                } catch (RuntimeException drain) {
                    e.addSuppressed(drain);
                }
            }
            throw e;
        } finally {
            releaseCopies(paged, steps.pendingCopies);
            steps.calls.closeEvents();
            steps.taps.closeEvents();
            accept.closeEvents();
        }
    }

    /** The context's device memory with paged sequences, or the refusal by name. */
    private DeviceModelMemory pagedMemory() {
        GpuMemory m = ctx.hasMemory() ? ctx.memory() : null;
        if (!(m instanceof DeviceModelMemory)) {
            throw new UnsupportedOperationException(ctx.spec().arch().ggufName()
                    + ": forwardBatch needs a device memory with paged sequences, got "
                    + (m == null ? "none" : m.getClass().getSimpleName()));
        }
        DeviceModelMemory memory = (DeviceModelMemory) m;
        if (memory.pagedSequences() == null) {
            throw new UnsupportedOperationException(ctx.spec().arch().ggufName()
                    + ": forwardBatch needs paged memory (neuralfree.parallel > 1 or neuralfree.kv.paged=true)");
        }
        return memory;
    }

    /**
     * Drops every captured trunk-replay graph, so the next step captures again: anything that can change a
     * trunk's dispatch sequence outside its steps (a compressed plan's rollback, context shift, a new hidden tap)
     * calls this.
     */
    void invalidateReplay() {
        decodeReplay.invalidate();
        invalidateStepReplay();
    }

    /**
     * Keeps the one-token decode capture through a memory reset ({@link GpuDecodeReplay#distrust()}: the next decode
     * steps verify it against the body before it is trusted again) and drops the multi-row step captures, whose
     * shapes follow the batches.
     */
    void distrustReplay() {
        decodeReplay.distrust();
        invalidateStepReplay();
    }

    /**
     * Re-arms the decode replay's continuity guard for a rollback that kept the captures (the speculative
     * round's replay, {@code chainReplay}): the rollback only rewound the host's sequence bookkeeping, so
     * the next decode step runs at {@code nextPos} against the same capture.
     */
    void rearmDecodeReplay(int nextPos) {
        decodeReplay.rearm(nextPos);
    }

    /** Releases the captured trunk-replay graphs, before the blocks they dispatch are closed. */
    void closeReplay() {
        decodeReplay.close();
        stepReplay.close();
    }

    /** The per-call state of {@link #runBatch} and its ubatch step. */
    private final class BatchSteps {
        final ForwardBatch batch;
        final DeviceModelMemory memory;
        final PagedSequences paged;
        final int[][] ropeSections;
        final float[][] embeddings;
        final Output output;
        final GpuCallReads calls;
        final GpuTapReads taps;
        final List<PageCopy> pendingCopies = new ArrayList<>();
        /** The device accept walk a tree verify round stages per ubatch instead of an output readback. */
        final GpuTreeAcceptReadout accept;

        BatchSteps(ForwardBatch batch, DeviceModelMemory memory, int[][] ropeSections, Output output, int topK,
                   HiddenTap tap, LogitPrefilter prefilter) {
            this(batch, memory, ropeSections, output, topK, tap, prefilter, null);
        }

        BatchSteps(ForwardBatch batch, DeviceModelMemory memory, int[][] ropeSections, Output output, int topK,
                   HiddenTap tap, LogitPrefilter prefilter, GpuTreeAcceptReadout accept) {
            this.batch = batch;
            this.memory = memory;
            this.paged = memory.pagedSequences();
            this.ropeSections = ropeSections;
            this.embeddings = batch.embeddings();
            this.output = output;
            this.calls = new GpuCallReads(head, queue, batch.outputRows().length, output, topK, prefilter);
            this.taps = new GpuTapReads(queue, tap, batch.nTokens(), 0);
            this.accept = accept;
        }

        /**
         * One multi-sequence ubatch: reserve, copy pages, commit, upload, embed, run the trunk, trim the sliding
         * windows to the next queries, read its rows.
         */
        void ubatch(UBatch u) {
            int[] tokens = u.tokens();
            int[] positions = u.positions();
            ExecutionObserver obs = ctx.observer();
            boolean observed = obs.enabled();
            // the rows store their explicit positions: a media chunk's rows share the chunk's base position
            PagedStep step = paged.prepare(u.layout(), positions);
            for (PageCopy c : step.pageCopies()) {
                memory.copyPage(c);
                if (mirror != null) {
                    mirror.copyPage(c);
                }
            }
            // the commit runs before the trunk so the trunk's queries see the copied rows' positions, as the CPU
            // backend's ubatch orders it; the copies' sources release once the queue finished (the copies executed)
            paged.commit(step);
            pendingCopies.addAll(step.pageCopies());
            int[][] sections = ubatchSections(ropeSections, u);
            memory.beginPagedStep(step, positions, sections, batch.nonCausal());
            if (mirror != null) {
                mirror.beginPagedStep(step, positions, sections, batch.nonCausal());
            }
            long[] treeMask = ubatchTreeMask(batch.treeMask(), u);
            if (treeMask != null) {
                memory.stepInputs().beginTreeStep(treeMask, sections);
                if (memory.hasRecurrentState() && recurrentRound == null) {
                    throw new IllegalStateException("the step carries a tree mask over a model with recurrent"
                            + " state, but no recurrent tree round is open: open one with openRecurrentTreeRound"
                            + " before the masked forward");
                }
            }
            AdapterRows adapterRows = ctx.projectionDeltas().rows();
            int[] slots = AdapterRows.ofRows(batch.adapterSlots(), u.batchRows());
            adapterRows.set(slots);
            try {
                if (treeMask != null && recurrentRound != null) {
                    recurrentRound.stepTreeMask(treeMask);
                    executor.recurrentTreeRound(recurrentRound);
                }
                executor.prefetchFirst();
                embedding.embedBatched(tokens, positions, null, ubatchRows(embeddings, u));
                if (observed) {
                    obs.onStepStart(step.segPos0(0), step.nTokens());
                }
                taps.batchRows(u.batchRows());
                if (batchTrunkRunner != null && batchTrunkRunner.runsBatchedTrunk(u.layout())) {
                    if (recurrentRound != null) {
                        throw new IllegalStateException("the batched trunk runner bypasses the executor's mixer"
                                + " blocks, so it cannot run a recurrent tree round");
                    }
                    batchTrunkRunner.runBatchTrunk(u.layout(), GpuLayerExecutor.StepMode.PREFILL, taps);
                } else {
                    // the masked tree verify step captures its trunk like the chain round's verify step does
                    // (stepReplay), and so does the plain paged decode step (every segment one row, the
                    // continuous-batching decode shape): the uploads around the trunk stay outside the body
                    // and its attention launches one decode kernel over the batch's sequences without
                    // per-segment uploads, so the trunk enqueues kernel dispatches alone; a commit's row moves
                    // dispatch directly between captures, so one capture serves every round of the same shape.
                    // A multi-row segment keeps running directly (its prefill uploads positions inside the
                    // trunk), and a recurrent target's round changes the mixer-block routing per round.
                    boolean capturable = chainReplay && !replayDisabled
                            && (treeMask != null || allRowsSingle(step))
                            && recurrentRound == null && !taps.wantsAny(executor.layers().size())
                            && executor.capturableDecode();
                    if (treeMask == null && allRowsSingle(step)) {
                        pagedDecodeTrunk(step, slots, batch.nonCausal(), capturable,
                                () -> executor.run(u.layout(), taps));
                    } else {
                        if (treeMask == null || pagedDecodeShape != null) {
                            invalidateStepReplay();
                        }
                        stepReplay.trunk(capturable, () -> executor.run(u.layout(), taps));
                    }
                }
                for (int seg = 0; seg < step.nSeg(); seg++) {
                    paged.appendRows(step.segSeq(seg), tokens, positions, step.segStart(seg), step.segLen(seg),
                            embeddingRowToken);
                }
                // the queue runs in order, so the trimmed pages' next writers follow this trunk's reads
                paged.finish(step);
                if (treeMask != null) {
                    memory.stepInputs().endTreeStep();
                }
                memory.endPagedStep();
                if (mirror != null) {
                    mirror.endPagedStep();
                }
                if (accept != null) {
                    stageAccept(accept, u, slots);
                } else {
                    readOutputs(u, calls, output, slots);
                }
                if (observed) {
                    obs.onStepEnd(step.segPos0(0), step.nTokens());
                }
            } catch (RuntimeException e) {
                if (treeMask != null) {
                    memory.stepInputs().endTreeStep();
                }
                memory.endPagedStep();
                if (mirror != null) {
                    mirror.endPagedStep();
                }
                throw e;
            } finally {
                executor.recurrentTreeRound(null);
                adapterRows.clear();
            }
        }
    }

    /** The batch's tree mask restricted to the ubatch's rows in ubatch order, or null when the batch has none. */
    private static long[] ubatchTreeMask(long[] mask, UBatch u) {
        if (mask == null) {
            return null;
        }
        int[] rows = u.batchRows();
        long[] out = new long[rows.length];
        for (int r = 0; r < rows.length; r++) {
            out[r] = mask[rows[r]];
        }
        return out;
    }

    /** The batch's RoPE sections restricted to the ubatch's rows in ubatch order, or null when the batch has none. */
    private static int[][] ubatchSections(int[][] sections, UBatch u) {
        if (sections == null) {
            return null;
        }
        int[] rows = u.batchRows();
        int[][] out = new int[sections.length][rows.length];
        for (int s = 0; s < sections.length; s++) {
            for (int r = 0; r < rows.length; r++) {
                out[s][r] = sections[s][rows[r]];
            }
        }
        return out;
    }

    /** The batch's embedding rows of the ubatch in ubatch order (null entries for token rows), or null. */
    private static float[][] ubatchRows(float[][] embeddings, UBatch u) {
        if (embeddings == null) {
            return null;
        }
        float[][] out = new float[u.nTokens()][];
        for (int r = 0; r < out.length; r++) {
            out[r] = embeddings[u.batchRow(r)];
        }
        return out;
    }

    /**
     * Enqueues the head reads of the ubatch's output rows, mapped to their batch-order result slots; the head's
     * products run with the output rows' adapter slots ({@code slots} of the ubatch rows, null for slot 0).
     */
    private void readOutputs(UBatch u, GpuCallReads calls, Output output, int[] slots) {
        int[] localRows = u.layout().outputRows();
        if (localRows.length == 0) {
            return;
        }
        int[] batchRows = u.batchRows();
        int[] outputs = u.batch().outputRows();
        int[] local = new int[localRows.length];
        int[] targets = new int[localRows.length];
        int k = 0;
        for (int lr : localRows) {
            int batchRow = batchRows[lr];
            local[k] = lr;
            targets[k] = indexOf(outputs, batchRow);
            k++;
        }
        ctx.projectionDeltas().rows().set(AdapterRows.ofRows(slots, local));
        ScratchSlot hidden = head.hidden(u.nTokens(), local, localRows.length);
        calls.read(output, hidden, localRows.length, targets);
    }

    /**
     * Stages the device accept walk of a single-ubatch tree verify step after its trunk: every ubatch row's
     * hidden state feeds the head's logits and greedy ids, and the {@code LLM_TreeAccept} readout enqueues the
     * walk over them; {@link GpuTreeAcceptReadout#result} assembles it after the queue finished.
     */
    private void stageAccept(GpuTreeAcceptReadout accept, UBatch u, int[] slots) {
        int n = u.nTokens();
        int[] local = new int[n];
        for (int r = 0; r < n; r++) {
            local[r] = r;
        }
        ctx.projectionDeltas().rows().set(AdapterRows.ofRows(slots, local));
        ScratchSlot hidden = head.hidden(n, local, n);
        head.logits(n, hidden);
        accept.stage(head.argmax(n));
    }

    private static int indexOf(int[] values, int v) {
        for (int i = 0; i < values.length; i++) {
            if (values[i] == v) {
                return i;
            }
        }
        throw new IllegalArgumentException("ubatch row " + v + " is not an output row of the batch");
    }

    /**
     * The trunk of a plain paged decode step (every segment one row) through {@link #stepReplay}: a capture of
     * another shape or kind is dropped first. On a single KV group the mixers' paged decode patchers
     * ({@link GpuLayerExecutor#pagedDecodeReplayPatchers()}) let the capture replay without its body, the step's
     * key count (the largest live key count of its sequences, which each decode step advances by one) as the
     * replay position; otherwise, or when a mixer cannot vouch for its launches, the capture replays through the
     * body. A step that cannot be captured runs directly and drops the capture.
     */
    private void pagedDecodeTrunk(PagedStep step, int[] slots, boolean nonCausal, boolean capturable,
                                  Runnable body) {
        PagedDecodeShape shape = capturable ? new PagedDecodeShape(step.nTokens(), slots, nonCausal) : null;
        if (shape == null || !shape.equals(pagedDecodeShape)) {
            invalidateStepReplay();
        }
        if (shape == null) {
            body.run();
            return;
        }
        pagedDecodeShape = shape;
        List<GpuStepPatcher> patchers = step.groups() == 1 ? executor.pagedDecodeReplayPatchers() : null;
        stepReplay.trunk(maxKeys(step), 1, patchers, true, body);
    }

    /** The largest key count of the step's sequences in its first KV group, the paged decode replay's position. */
    private static int maxKeys(PagedStep step) {
        int[] slots = step.seqSlots(0);
        int keys = 0;
        for (int seg = 0; seg < step.nSeg(); seg++) {
            keys = Math.max(keys, slots[step.segSeq(seg)]);
        }
        return keys;
    }

    /** Drops the step replay's capture, whichever step it holds. */
    private void invalidateStepReplay() {
        stepReplay.invalidate();
        pagedDecodeShape = null;
    }

    /**
     * Whether every segment of the paged {@code step} holds one row: the decode shape, whose attention
     * launches one paged decode kernel over the batch's sequences and uploads nothing inside the trunk,
     * so a capture of the trunk holds kernel dispatches alone.
     */
    private static boolean allRowsSingle(PagedStep step) {
        for (int seg = 0; seg < step.nSeg(); seg++) {
            if (step.segLen(seg) != 1) {
                return false;
            }
        }
        return true;
    }

    /**
     * Releases the completed page copies' sources: the queue has finished by the time this runs (the copies
     * executed), so every pending copy hands its source page back to the pool.
     */
    private static void releaseCopies(PagedSequences paged, List<PageCopy> copies) {
        if (copies.isEmpty()) {
            return;
        }
        paged.releaseCopies(copies.toArray(new PageCopy[0]));
        copies.clear();
    }

    private void step(int[] tokens, float[][] embd, int[] types, int p, int c0, int n, int[] rows, Output output,
                      int topK, GpuMemory memory, GpuCallReads calls, GpuTapReads taps) {
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStepStart(p, n);
        }
        GpuLayerExecutor.StepMode mode = memory != null && n == 1
                ? GpuLayerExecutor.StepMode.DECODE : GpuLayerExecutor.StepMode.PREFILL;
        boolean trunk = trunkRunner != null && mode == GpuLayerExecutor.StepMode.PREFILL;
        if (memory != null && !trunk) {
            memory.stepInputs().upload(p, n);
            if (mirror != null) {
                mirror.beforeStep(p, n);
            }
        }
        executor.prefetchFirst();
        positions.write(p, n);
        if (embd != null) {
            embedding.embedRows(embd, c0, n, p);
        } else {
            embedding.embed(tokens, c0, n, types, p);
        }
        taps.chunk(c0);
        if (pagedDecodeShape != null) {
            invalidateStepReplay();
        }
        if (mode == GpuLayerExecutor.StepMode.DECODE) {
            decodeReplay.trunk(p, executor.decodeReplayPatchers(),
                    !replayDisabled && !taps.wantsAny(executor.layers().size()) && executor.capturableDecode(),
                    () -> executor.run(p, n, mode, taps));
        } else if (trunk) {
            decodeReplay.distrust();
            invalidateStepReplay();
            trunkRunner.runTrunk(p, n, mode, taps);
        } else {
            decodeReplay.distrust();
            // the memory-backed multi-row step (a speculative round's verify of k + 1 rows) captures its own
            // trunk only when the speculative round's replay is on (GpuForwardBackend.CHAIN_REPLAY_PROPERTY)
            stepReplay.trunk(p, n, executor.stepReplayPatchers(),
                    chainReplay && !replayDisabled && !taps.wantsAny(executor.layers().size())
                            && executor.capturableDecode(),
                    () -> executor.run(p, n, mode, taps));
        }
        int k = 0;
        int[] local = new int[rows.length];
        int[] targets = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            int r = rows[i];
            if (r >= c0 && r < c0 + n) {
                local[k] = r - c0;
                targets[k] = i;
                k++;
            }
        }
        for (int done = 0; done < k; ) {
            int count = Math.min(n, k - done);
            ScratchSlot hidden = head.hidden(n, Arrays.copyOfRange(local, done, done + count), count);
            calls.read(output, hidden, count, Arrays.copyOfRange(targets, done, done + count));
            done += count;
        }
        if (observed) {
            obs.onStepEnd(p, n);
        }
    }

    private static int[] identity(int n) {
        int[] r = new int[n];
        for (int i = 0; i < n; i++) {
            r[i] = i;
        }
        return r;
    }

    /**
     * The shape a paged decode capture serves: the step's row count (one row per sequence), the rows' adapter slots
     * and the causal mode. Steps of equal shape enqueue the same trunk dispatches; only their attention's key-count
     * values differ.
     */
    private static final class PagedDecodeShape {
        private final int rows;
        private final int[] slots;
        private final boolean nonCausal;

        PagedDecodeShape(int rows, int[] slots, boolean nonCausal) {
            this.rows = rows;
            this.slots = slots == null ? null : slots.clone();
            this.nonCausal = nonCausal;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof PagedDecodeShape)) {
                return false;
            }
            PagedDecodeShape other = (PagedDecodeShape) o;
            return rows == other.rows && nonCausal == other.nonCausal && Arrays.equals(slots, other.slots);
        }

        @Override
        public int hashCode() {
            return 31 * (31 * rows + Arrays.hashCode(slots)) + (nonCausal ? 1 : 0);
        }
    }
}

