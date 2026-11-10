package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.DraftTree;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.LookaheadRun;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.engine.OutputKind;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SplitReadout;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;

/**
 * Evaluates the pieces one {@link BatchEngine} step scheduled: the token pieces (decodes and text-chunk rows) share
 * one forward batch through {@code ForwardBackend.forwardBatch}, a pooling piece's rows run their own batch through
 * {@code hiddenBatch} (the pooling heads' input), a media chunk piece runs its own batch, and every output row lands
 * at its piece's place in the step's output order. A sampling piece whose chain keeps its candidates inside a top-k
 * cone reads only those candidates back on the device ({@code ForwardBackend.forwardBatchTopK}, llama.cpp backend
 * sampling {@code -bs}); the pieces that need whole rows beside it (a grammar, a logprobs capture, a verify round)
 * read theirs from the same forward ({@code ForwardBackend.forwardBatchSplit}). Counts the step, reads
 * prompt-logprobs rows back, then hands the outputs to the {@link OutputRouter}; a forward that throws rolls the step
 * back through the {@link SlotLifecycle}.
 */
final class StepExecutor {

    private final ForwardBackend backend;
    private final BatchState state;
    private final SlotLifecycle lifecycle;
    private final OutputRouter router;
    /** Whether a step whose chains keep inside a device top-k samples over that readback. */
    private final boolean deviceSampling;

    StepExecutor(ForwardBackend backend, BatchState state, SlotLifecycle lifecycle, OutputRouter router,
                 boolean deviceSampling) {
        this.backend = backend;
        this.state = state;
        this.lifecycle = lifecycle;
        this.router = router;
        this.deviceSampling = deviceSampling;
    }

    /**
     * Drops the rows a step that ran out of pages committed and frees memory for the pieces that rolled back; a
     * piece whose sequence refuses the tail removal retires with the refusal, which also frees its pages.
     */
    private void rollBack(List<Piece> pieces, MemoryFullException e) {
        List<Piece> rolledBack = new ArrayList<>(pieces.size());
        for (Piece p : pieces) {
            String refusal = state.sequences.removeRefusal(p.slot.seq, p.from, -1);
            if (refusal != null) {
                IllegalStateException failure = new IllegalStateException("sequence " + p.slot.seq
                        + " ran out of pages and cannot drop the rows it committed: " + refusal, e);
                lifecycle.retire(p.slot, null, false, failure);
            } else {
                state.sequences.remove(p.slot.seq, p.from, -1);
                rolledBack.add(p);
            }
        }
        if (!rolledBack.isEmpty()) {
            lifecycle.recoverMemory(rolledBack, e);
        }
    }

    /**
     * The row counts of one step: its shared token batches' rows (the logits rows and the pooling rows run as two
     * batches), its decode rows and its output rows, and whether a token piece runs per-request adapters.
     */
    private static final class Layout {
        int tokenRows;
        int hiddenTokenRows;
        boolean adapters;
        int decodeRows;
        int outputs;
    }

    /**
     * Evaluates one step's pieces and routes their outputs: the token pieces (decodes and text-chunk rows) share
     * one forward batch, a pooling piece's rows run their own hidden-state batch beside it, a media chunk piece
     * runs its own batch (its rope sections and mask rule are batch-wide flags the shared batch cannot hold per
     * piece), and every output row lands in the step's output order. A prompt-logprobs piece reads every row's
     * logits back into its submission's capture (a prefill split over steps fills it chunk by chunk); a pooling
     * piece reads every row's hidden states into its pooling run. A piece whose row the shared batch read as top
     * candidates routes over them, every other piece over its whole rows ({@link OutputRouter#route}). A step
     * that runs out of pages rolls its rows back and frees memory through {@link SlotLifecycle#recoverMemory} (over a
     * memory with recurrent state it fails before its first ubatch, {@link BatchState#pageShortfall}, since a state
     * cannot roll back); a sequence whose committed rows cannot leave alone (the memory's {@code removeRefusal}, a
     * DeepSeek-V4 compressor beyond its rollback snapshots) fails instead. Any other failure retires every scheduled
     * sequence (the split runs ubatches in order and a throw abandons the rest, so no row can be attributed).
     */
    void run(List<Piece> pieces) {
        List<Piece> plain = new ArrayList<>(pieces.size());
        for (Piece p : pieces) {
            if (p.isLookahead()) {
                runLookahead(p);
            } else {
                plain.add(p);
            }
        }
        if (!plain.isEmpty()) {
            runPlain(plain);
        }
    }

    /**
     * Runs one lookahead step: its own forward batch over the slot's sequence group, then the walk that samples and
     * emits its tokens through the slot's run. The input token and the confirmed tokens join the fed history and the
     * cursors advance by the emitted count, the armed token staying unfed for the next step. A step that runs out of
     * pages ends the generation with {@link StopReason#CONTEXT_FULL}; any other failure retires the slot alone.
     */
    private void runLookahead(Piece p) {
        Slot slot = p.slot;
        LookaheadRun run = slot.lookahead;
        try {
            if (!run.initialized()) {
                run.initialize();
            }
            long t0 = System.nanoTime();
            float[][] rows = backend.forwardBatch(p.lookahead.batch());
            long nanos = System.nanoTime() - t0;
            state.stats.decodeNanos += nanos;
            state.stats.decodeSteps++;
            state.stats.decodeRows += p.rows;
            LookaheadRun.Result result = run.walk(p.lookahead, rows, () -> state.cancelRequested(slot));
            state.stats.lookaheadRounds++;
            state.stats.lookaheadEmitted += result.emitted();
            state.stats.lookaheadAccepted += result.accepted();
            state.stats.generatedTokens += result.emitted();
            if (result.stopped()) {
                if (result.cancelled()) {
                    lifecycle.finish(slot, StopReason.CANCELLED);
                } else {
                    // the run recorded the token-level stop: the result is complete
                    lifecycle.retire(slot, slot.run.result(), false, null);
                }
                return;
            }
            int emitted = result.emitted();
            int[] tokens = result.tokens();
            if (emitted == 0) {
                throw new IllegalStateException("a lookahead step that did not stop emitted no token");
            }
            slot.fed(slot.pendingToken);
            for (int i = 0; i < emitted - 1; i++) {
                slot.fed(tokens[i]);
            }
            slot.nextPos += emitted;
            slot.nextRow += emitted;
            slot.pendingToken = tokens[emitted - 1];
            run.commit(slot.nextPos, result.best());
        } catch (MemoryFullException e) {
            lifecycle.finish(slot, StopReason.CONTEXT_FULL);
        } catch (RuntimeException e) {
            lifecycle.retire(slot, null, false, e);
        }
    }

    /** Evaluates and routes the step's ordinary pieces (decodes, prefill chunks and media chunks). */
    private void runPlain(List<Piece> pieces) {
        Layout layout = layout(pieces);
        float[][] outputs = new float[layout.outputs][];
        state.makeWindowRoom(pieces);
        MemoryFullException shortfall = state.pageShortfall(pieces);
        if (shortfall != null) {
            // nothing ran: the recurrent states stand where the step found them
            lifecycle.recoverMemory(pieces, shortfall);
            return;
        }
        ForwardResult forward;
        try {
            forward = forward(pieces, layout, outputs);
        } catch (MemoryFullException e) {
            rollBack(pieces, e);
            return;
        } catch (RuntimeException e) {
            // the split runs ubatches in order; a throw abandons the rest of the batch, so no later row can
            // be attributed: the step's sequences fail together (rare path, see the engine's class doc)
            for (Piece p : pieces) {
                lifecycle.retire(p.slot, null, false, e);
            }
            return;
        }
        countStep(pieces, layout.decodeRows, forward.nanos);
        for (Piece p : pieces) {
            if (p.readRows) {
                capturePromptRows(p, outputs);
            }
        }
        router.route(pieces, outputs, forward.top, forward.topRows, forward.prefiltered);
    }

    /** One step's forward time and, when the shared batch read candidates back, its device top-k rows. */
    private static final class ForwardResult {
        final long nanos;
        final RowTopK top;
        /** Per piece its row in {@link #top}, or -1 when it routes over whole rows; null without {@link #top}. */
        final int[] topRows;
        /** Whether the readout applied the rows' logit prefilter before its top-k. */
        final boolean prefiltered;

        ForwardResult(long nanos, RowTopK top, int[] topRows, boolean prefiltered) {
            this.nanos = nanos;
            this.top = top;
            this.topRows = topRows;
            this.prefiltered = prefiltered;
        }
    }

    /**
     * The shared batch's cone: its width (0 for a whole-row readback), the prefilter it applies first, and per piece
     * whether the piece samples its one output row from the cone.
     */
    private static final class StepCone {
        static final StepCone NONE = new StepCone(0, null, null);
        final int width;
        final LogitPrefilter prefilter;
        /** Per piece whether it samples from the cone; null for {@link #NONE}. */
        final boolean[] coned;

        StepCone(int width, LogitPrefilter prefilter, boolean[] coned) {
            this.width = width;
            this.prefilter = prefilter;
            this.coned = coned;
        }

        boolean prefiltered() {
            return prefilter != null;
        }
    }

    /**
     * Places every piece's output rows in the step's output order ({@code outIdx}): a decode round's every row
     * samples (a verify walk reads them all), a prompt-logprobs piece and a pooling piece read every prompt row
     * back, so each of their rows asks outputs; another prefill piece asks one row when it completes the prompt.
     */
    private static Layout layout(List<Piece> pieces) {
        Layout layout = new Layout();
        for (Piece p : pieces) {
            if (!p.isMediaChunk()) {
                layout.tokenRows += p.rows;
                layout.adapters |= p.slot.submission.runsAdapterSlots();
                if (p.wantsHidden) {
                    layout.hiddenTokenRows += p.rows;
                }
            }
            if (p.decode || p.readRows || p.wantsHidden) {
                p.outIdx = layout.outputs;
                layout.outputs += p.rows;
                if (p.decode) {
                    layout.decodeRows += p.rows;
                }
            } else {
                p.outIdx = p.output ? layout.outputs++ : -1;
            }
        }
        return layout;
    }

    /**
     * Runs the step's forward batches into {@code outputs} — each media chunk piece's own batch in piece order,
     * then the shared token batch of the pieces that sample from logits, then the pooling pieces' hidden-state
     * batch — and reports their forward time in nanoseconds. Rows that carry per-request adapters run in their
     * sequences' adapter slots. When a piece samples from a device cone ({@link #logitsCone}) the shared token batch
     * reads every output row's top candidates into the returned {@link ForwardResult#top}, and whole rows only for
     * the pieces outside the cone; a cone piece's slot in {@code outputs} stays null.
     */
    private ForwardResult forward(List<Piece> pieces, Layout layout, float[][] outputs) {
        long nanos = 0;
        int logitsRows = layout.tokenRows - layout.hiddenTokenRows;
        ForwardBatch.Builder logitsBatch = ForwardBatch.builder(Math.max(1, logitsRows));
        ForwardBatch.Builder hiddenBatch = ForwardBatch.builder(Math.max(1, layout.hiddenTokenRows));
        int[] logitsSlots = layout.adapters ? new int[logitsRows] : null;
        int[] hiddenSlots = layout.adapters ? new int[layout.hiddenTokenRows] : null;
        long[] treeMask = carriesTree(pieces) ? new long[logitsRows] : null;
        int[][] rope = treeMask != null ? new int[1][logitsRows] : null;
        int logitsRow = 0;
        int hiddenRow = 0;
        for (Piece p : pieces) {
            if (p.isMediaChunk()) {
                long t0 = System.nanoTime();
                float[][] rows = backend.forwardBatch(PrefillChunks.chunkBatch(p));
                nanos += System.nanoTime() - t0;
                for (int i = 0; i < rows.length; i++) {
                    outputs[p.outIdx + i] = rows[i];
                }
            } else if (p.wantsHidden) {
                addTokenRows(hiddenBatch, p);
                fillSlots(hiddenSlots, hiddenRow, p);
                hiddenRow += p.rows;
            } else {
                addTokenRows(logitsBatch, p);
                if (treeMask != null) {
                    maskRows(treeMask, rope[0], logitsRow, p);
                }
                fillSlots(logitsSlots, logitsRow, p);
                logitsRow += p.rows;
            }
        }
        RowTopK top = null;
        int[] topRows = null;
        boolean prefiltered = false;
        if (logitsRow > 0) {
            if (logitsSlots != null) {
                logitsBatch.adapterSlots(Arrays.copyOf(logitsSlots, logitsRow));
            }
            if (treeMask != null) {
                logitsBatch.ropePositions(rope);
                logitsBatch.treeMask(treeMask);
            }
            ForwardBatch batch = logitsBatch.build();
            StepCone cone = logitsCone(pieces);
            long t0 = System.nanoTime();
            if (cone.width > 0) {
                prefiltered = cone.prefiltered();
                topRows = outputRowsOf(pieces, batch.outputRows().length);
                boolean[] fullRows = fullRows(pieces, cone.coned, topRows, batch.outputRows().length);
                if (fullRows == null) {
                    top = prefiltered
                            ? backend.forwardBatchTopK(batch, cone.width, cone.prefilter)
                            : backend.forwardBatchTopK(batch, cone.width);
                } else {
                    SplitReadout split = backend.forwardBatchSplit(batch, fullRows, cone.width, cone.prefilter);
                    top = split.top();
                    float[][] rows = new float[split.rows()][];
                    for (int r = 0; r < rows.length; r++) {
                        rows[r] = split.row(r);
                    }
                    scatterTokenRows(pieces, rows, outputs, false);
                }
                for (int i = 0; i < topRows.length; i++) {
                    if (!cone.coned[i]) {
                        topRows[i] = -1;
                    }
                }
            } else {
                float[][] rows = backend.forwardBatch(batch);
                scatterTokenRows(pieces, rows, outputs, false);
            }
            nanos += System.nanoTime() - t0;
        }
        if (hiddenRow > 0) {
            if (hiddenSlots != null) {
                hiddenBatch.adapterSlots(Arrays.copyOf(hiddenSlots, hiddenRow));
            }
            long t0 = System.nanoTime();
            float[][] rows = backend.hiddenBatch(hiddenBatch.outputKinds(EnumSet.of(OutputKind.HIDDEN)).build());
            nanos += System.nanoTime() - t0;
            scatterTokenRows(pieces, rows, outputs, true);
        }
        return new ForwardResult(nanos, top, topRows, prefiltered);
    }

    /**
     * Per piece the index of its first output row among the shared batch's output rows, or -1 when it has none there
     * (a media chunk, a pooling piece, a prompt chunk that does not complete): the batch's output rows are, in piece
     * order, every row of a decode round or a prompt-logprobs piece and the last row of a completing chunk.
     *
     * @throws IllegalStateException when the pieces' rows do not add up to the batch's {@code outputs} output rows
     */
    private static int[] outputRowsOf(List<Piece> pieces, int outputs) {
        int[] rows = new int[pieces.size()];
        int next = 0;
        for (int i = 0; i < pieces.size(); i++) {
            int n = sharedOutputRows(pieces.get(i));
            rows[i] = n > 0 ? next : -1;
            next += n;
        }
        if (next != outputs) {
            throw new IllegalStateException("the step's pieces hold " + next + " output rows but the shared batch"
                    + " reads " + outputs);
        }
        return rows;
    }

    /** The output rows a piece holds in the step's shared token batch. */
    private static int sharedOutputRows(Piece p) {
        if (p.isMediaChunk() || p.wantsHidden) {
            return 0;
        }
        return p.decode || p.readRows ? p.rows : p.output ? 1 : 0;
    }

    /**
     * The whole-row flags of the shared batch's {@code outputs} output rows: every output row of a piece outside the
     * cone; null when every output row samples from the cone.
     */
    private static boolean[] fullRows(List<Piece> pieces, boolean[] coned, int[] first, int outputs) {
        boolean[] full = new boolean[outputs];
        boolean any = false;
        for (int i = 0; i < pieces.size(); i++) {
            if (first[i] >= 0 && !coned[i]) {
                Arrays.fill(full, first[i], first[i] + sharedOutputRows(pieces.get(i)), true);
                any = true;
            }
        }
        return any ? full : null;
    }

    /**
     * The shared token batch's device cone: the pieces that sample from it, its width and the prefilter it applies
     * first; {@link StepCone#NONE} when no piece does (device sampling is off, the backend has no batched row top-k,
     * or no chain fits). A piece samples from the cone when it samples one output row (a single-row decode or a
     * completing text chunk without a prompt-logprobs read) and its chain keeps its candidates inside its top-k
     * ({@link Sampler#coneWidth}). Every other piece reads its rows whole from the same forward: a grammar or a
     * logprobs capture ({@code wantsHostSampling}), a chain that needs the whole row, a verify round, a
     * prompt-logprobs piece; a pooling piece and a media chunk run their own batches.
     *
     * <p>A chain whose cone needs the pre-top-k logit edits (penalties, DRY, a positive bias) uses
     * {@link Sampler#deviceConeWidth()} and carries its {@link Sampler#prefilterRow()} to the readout when the backend
     * applies the prefilter on the device. The width is the widest piece cone: a chain over a wider cone selects
     * exactly what it selects over its own. When the step prefilters, every cone piece's row rides the readout (a
     * plain chain with its own, possibly empty, row) and every whole-read output row carries an empty row, which
     * leaves it as the forward computed it.</p>
     */
    private StepCone logitsCone(List<Piece> pieces) {
        if (!deviceSampling || !backend.batchTopKOnDevice()) {
            return StepCone.NONE;
        }
        boolean prefilterOnDevice = backend.batchPrefilterOnDevice();
        boolean[] coned = new boolean[pieces.size()];
        boolean needsPrefilter = false;
        for (int i = 0; i < pieces.size(); i++) {
            Piece p = pieces.get(i);
            if (!samplesConeRow(p)) {
                continue;
            }
            Sampler sampler = p.slot.run.sampler();
            int cone = sampler.coneWidth();
            if (cone >= 1) {
                coned[i] = cone <= RowTopK.MAX_DEVICE_CANDIDATES;
            } else if (prefilterOnDevice && fitsDevice(sampler.deviceConeWidth())) {
                coned[i] = true;
                needsPrefilter = true;
            }
        }
        int width = 0;
        for (int i = 0; i < pieces.size(); i++) {
            if (!coned[i]) {
                continue;
            }
            Sampler sampler = pieces.get(i).slot.run.sampler();
            int cone = needsPrefilter ? sampler.deviceConeWidth() : sampler.coneWidth();
            if (fitsDevice(cone)) {
                width = Math.max(width, cone);
            } else {
                coned[i] = false;
            }
        }
        if (width < 1) {
            return StepCone.NONE;
        }
        if (!needsPrefilter) {
            return new StepCone(width, null, coned);
        }
        LogitPrefilter.Row unedited = LogitPrefilter.Row.builder().build();
        List<LogitPrefilter.Row> rows = new ArrayList<>();
        for (int i = 0; i < pieces.size(); i++) {
            Piece p = pieces.get(i);
            if (coned[i]) {
                int[] pending = !p.decode ? p.slot.submission.prompt.samplerTokens() : null;
                rows.add(p.slot.run.sampler().prefilterRow(pending));
            } else {
                for (int r = sharedOutputRows(p); r > 0; r--) {
                    rows.add(unedited);
                }
            }
        }
        return new StepCone(width, LogitPrefilter.of(rows), coned);
    }

    /** Whether a cone width fits a device readback. */
    private static boolean fitsDevice(int width) {
        return width >= 1 && width <= RowTopK.MAX_DEVICE_CANDIDATES;
    }

    /**
     * Whether a piece samples exactly one output row of the shared batch through its chain alone: a single-row
     * decode or a completing text chunk that reads no prompt logprobs, whose run captures no logprobs and runs no
     * host-only stage ({@code wantsHostSampling}).
     */
    private static boolean samplesConeRow(Piece p) {
        if (p.isMediaChunk() || p.wantsHidden || p.readRows) {
            return false;
        }
        boolean one = p.decode ? p.rows == 1 : p.output;
        return one && !p.slot.run.wantsHostSampling();
    }

    /**
     * Fills {@code slots[from .. from + p.rows)} with the piece's rows' adapter slots, when the step carries
     * adapters: the slot of each row of the sequence's token record (an activated LoRA's invocation start splits a
     * prompt's rows), counted from the piece's {@link Piece#rowBase}.
     */
    private static void fillSlots(int[] slots, int from, Piece p) {
        if (slots != null) {
            for (int i = 0; i < p.rows; i++) {
                slots[from + i] = p.slot.submission.adapterSlot(p.rowBase + i);
            }
        }
    }

    /**
     * Adds a token piece's rows to its shared batch: a decode round's every row samples (a verify walk reads them
     * all); a pooling piece's every row reads its hidden states back; a prefill piece asks for logits on the chunk
     * that completes the prompt, a prompt-logprobs piece on every row.
     */
    private static void addTokenRows(ForwardBatch.Builder fb, Piece p) {
        for (int i = 0; i < p.rows; i++) {
            fb.add(p.tokens[i], p.from + i, p.slot.seq,
                    p.decode || p.readRows || p.wantsHidden || (p.output && i == p.rows - 1));
        }
    }

    /** Whether a piece of the step's shared token batch carries a tree proposal. */
    private static boolean carriesTree(List<Piece> pieces) {
        for (Piece p : pieces) {
            if (!p.isMediaChunk() && !p.wantsHidden && p.tree != null) {
                return true;
            }
        }
        return false;
    }

    /**
     * A masked step's per-row masks and rope positions for the piece's rows at {@code rowBase}: a tree round's row 0
     * holds the armed token alone, its node rows their ancestors-or-self within the piece (bit 0 of the piece, the
     * pending row, the tree's root) and rope at the node's depth position (siblings share one); every other row of
     * the step — a plain decode row, or an admitted prefill chunk's rows — ropes at its own position and its mask
     * names exactly the piece's rows up to it plus itself, which is a chunk's causal rule (SpecPilot's segment
     * admission: the chunk rides the verify step's forward unchanged). The bits are absolute step-row indices, and
     * every piece is one sequence's fresh rows: the CPU slices each segment's rows out of the array (its bits
     * shifted past the segment's start), and the GPU uploads the relativized rows per launch.
     */
    private static void maskRows(long[] mask, int[] rope, int rowBase, Piece p) {
        if (p.tree == null) {
            for (int i = 0; i < p.rows; i++) {
                mask[rowBase + i] = (1L << (rowBase + i + 1)) - (1L << rowBase);
                rope[rowBase + i] = p.from + i;
            }
            return;
        }
        List<DraftTree.Node> nodes = p.tree.tree().nodes();
        mask[rowBase] = 1L << rowBase;
        rope[rowBase] = p.from;
        for (int i = 0; i < nodes.size(); i++) {
            int row = rowBase + i + 1;
            long bits = (1L << row) | (1L << rowBase);
            for (int parent = nodes.get(i).parent(); parent >= 0; parent = nodes.get(parent).parent()) {
                bits |= 1L << (rowBase + parent + 1);
            }
            mask[row] = bits;
            rope[row] = p.from + nodes.get(i).depth();
        }
    }

    /**
     * Places the shared batch's output rows at their pieces' {@code outIdx}, in batch order: {@code hidden} picks
     * the pooling pieces' rows, its absence the sampling pieces'.
     */
    private static void scatterTokenRows(List<Piece> pieces, float[][] rows, float[][] outputs, boolean hidden) {
        int k = 0;
        for (Piece p : pieces) {
            if (!p.isMediaChunk() && p.outIdx >= 0 && p.wantsHidden == hidden) {
                int n = p.decode || p.readRows || p.wantsHidden ? p.rows : 1;
                for (int r = 0; r < n; r++) {
                    outputs[p.outIdx + r] = rows[k++];
                }
            }
        }
    }

    /**
     * Reads a prompt-logprobs piece's logits rows into its submission's capture: row {@code i} processes prompt
     * row {@code p.promptRowBase + i} and predicts prompt token {@code p.promptRowBase + i + 1}; a recompute's
     * replayed fed decodes past the prompt rows stop the capture (the prompt rows ran first in the replay).
     */
    private static void capturePromptRows(Piece p, float[][] outputs) {
        Submission sub = p.slot.submission;
        int promptRows = sub.prompt.nTokens();
        for (int i = 0; i < p.rows && p.promptRowBase + i < promptRows; i++) {
            sub.captureRow(p.promptRowBase + i, outputs[p.outIdx + i]);
        }
    }

    /** A successful step's counters: the decode and prompt times, the rows' tokens, each chunk's forward time. */
    private void countStep(List<Piece> pieces, int decodeRows, long elapsed) {
        BatchStats.Builder stats = state.stats;
        if (decodeRows > 0) {
            stats.decodeNanos += elapsed;
            stats.decodeSteps++;
            stats.decodeRows += decodeRows;
        }
        boolean anyPrefill = false;
        for (Piece p : pieces) {
            if (p.decode) {
                continue;
            }
            anyPrefill = true;
            p.slot.prefillNanos += elapsed;
            Submission sub = p.slot.submission;
            // an encoder-decoder's prompt is its source: the piece that starts its decoder rows counts the encoding
            int encoded = sub.source != null && p.from == 0 ? sub.source.length : 0;
            if (p.slot.recompute) {
                stats.recomputedTokens += p.rows + encoded;
            } else {
                stats.promptTokens += sub.source == null ? p.rows : encoded;
            }
        }
        if (anyPrefill) {
            stats.promptNanos += elapsed;
        }
    }
}
