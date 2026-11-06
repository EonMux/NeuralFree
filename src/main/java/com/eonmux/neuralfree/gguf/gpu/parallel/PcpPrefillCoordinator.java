package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuLayerExecutor;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuPrefillDriver;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.attn.GpuAttentionMixer;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.ToIntFunction;

/**
 * The runner of every prefill trunk of a prefill context-parallel split (vLLM's {@code prefill_context_parallel_size}
 * with the DualChunkSwap query partition): rank 0's executor runs the stack layer by layer through its phase entries,
 * and between them the ranks run the layer's split. Per step the coordinator cuts the query rows into the
 * {@link PcpChunkSwap} chunk pairs, uploads every rank's packed step inputs (true positions, cache cells of the rows
 * inside the rank's block), packs the streams (every peer receives its packed rows before rank 0's far chunk moves
 * down behind its first chunk — the move's destination rows overlap the rows the peers read), and per layer runs
 * every rank's norm phase, rank 0's attention phase — whose {@link PcpAttentionMixer} exchanges the queries and the
 * cache rows and merges the attention partitions (the peers' chunk segments leave the merged buffer before its far
 * chunk moves) — then the peers' finish phases and every rank's post phase. Afterwards it restores rank 0's stream
 * to the global row order the head reads (the far chunk moves up before the peers' rows return into the packed rows
 * it leaves).
 *
 * <p>Every rank runs every layer whole over its own packed rows, so every rank holds replicated norm, projection and
 * FFN weights ({@link PcpWeightReplicator}) and its own KV cache block. The split refuses models a rank cannot
 * partition (looped schedules, recurrent mixers, routed experts, per-layer token embeddings; the group checks at
 * init) and steps whose rows a rank cannot serve (hidden taps; this runner refuses them at the step).</p>
 *
 * <p>Over a paged memory (vLLM's PCP over the paged KV of its batched steps) the coordinator also carries the whole
 * batched step through {@link GpuPrefillDriver.BatchTrunkRunner}: every multi-row segment cuts into its own
 * {@link PcpChunkSwap} chunk pairs, every single-row segment (a decode row of a mixed step) replicates onto every
 * rank's packed rows and keeps the decode-context-parallel protocol, and every rank's step inputs are the packed rows
 * of its translated {@code PagedStep} ({@code CpPageShard}'s page ownership), whose page tables, slot counts and
 * {@code kv_pos} rows stay whole for the per-rank flashes. The queries gather at the rows' global offsets, the cache
 * rows scatter by translated destination cell, and rank 0's stream restores to the global row order before the step
 * ends.</p>
 */
final class PcpPrefillCoordinator implements GpuPrefillDriver.TrunkRunner, GpuPrefillDriver.BatchTrunkRunner,
        AutoCloseable {

    private final TensorParallelGroup group;
    private final GpuForwardContext modelContext;
    private final GpuLayerExecutor mainTrunk;
    private final List<GpuLayerExecutor> peerTrunks = new ArrayList<>();
    private final int ranks;
    private final int ubatch;
    private final int streamWidth;
    private final int queryRowWidth;
    /** The step's packed true positions per rank; {@code packedPos[rank][t]} is row {@code t}'s position. */
    private final int[][] packedPos;
    private final int[] packedPos0;
    private final int[] packedRows;
    /** The step's packed global row indices per rank; {@code packedGlobal[rank][t]} is packed row {@code t}'s row. */
    private final int[][] packedGlobal;
    private final ScratchSlot[] peerInputs;
    private final ScratchSlot[] peerDeferred;
    /** Per rank the gathered query rows of the whole step (global row order) and their true positions. */
    private final GpuBuffer[] queryGather;
    private final GpuBuffer[] queryPos;
    /** Per (owner, destination) rank pair the rebound store destinations and cell indices of the shadow store. */
    private final GpuBuffer[][] shadowK;
    private final GpuBuffer[][] shadowV;
    private final GpuBuffer[][] shadowCell;
    private final ByteBuffer intStage;
    private final List<GpuBuffer> owned = new ArrayList<>();
    /** Bytes of one shadow cache-row buffer: the trunk's widest cache row count times its widest cache row. */
    private final long shadowBytes;
    /** The segments of the running batched step, and every rank's packed row runs; empty on a contiguous step. */
    private final List<SegmentPlan> batchSegments = new ArrayList<>();
    private final List<List<RowRun>> batchRuns = new ArrayList<>();
    private PcpChunkSwap swap;
    private int stepPos0;
    private int stepTokens;
    private boolean batched;
    private boolean closed;

    PcpPrefillCoordinator(TensorParallelGroup group, GpuForwardContext modelContext, GpuLayerExecutor mainTrunk) {
        this.group = Objects.requireNonNull(group, "group");
        this.modelContext = Objects.requireNonNull(modelContext, "modelContext");
        this.mainTrunk = Objects.requireNonNull(mainTrunk, "mainTrunk");
        this.ranks = group.rankCount();
        this.ubatch = group.rankContext(0).scratch().ubatch();
        this.streamWidth = widest(l -> l.residual().streamWidth());
        this.queryRowWidth = widest(l -> l.attention().qRopeDim());
        this.packedPos = new int[ranks][ubatch];
        this.packedPos0 = new int[ranks];
        this.packedRows = new int[ranks];
        this.packedGlobal = new int[ranks][ubatch];
        this.peerInputs = new ScratchSlot[ranks];
        this.peerDeferred = new ScratchSlot[ranks];
        this.queryGather = new GpuBuffer[ranks];
        this.queryPos = new GpuBuffer[ranks];
        this.shadowK = new GpuBuffer[ranks][ranks];
        this.shadowV = new GpuBuffer[ranks][ranks];
        this.shadowCell = new GpuBuffer[ranks][ranks];
        this.intStage = ByteBuffer.allocateDirect(ubatch * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        this.shadowBytes = shadowBufferSize();
        try {
            allocateExchangeBuffers();
            openPeerTrunks();
        } catch (RuntimeException failure) {
            close();
            throw failure;
        }
    }

    /** The widest value of the trunk a per-layer width getter picks; at least 1. */
    private int widest(ToIntFunction<LayerSpec> width) {
        int out = 1;
        for (LayerSpec l : mainTrunk.layers()) {
            out = Math.max(out, width.applyAsInt(l));
        }
        return out;
    }

    /**
     * Bytes of one shadow cache-row buffer: the widest K or V cache row of the trunk's split caches times the
     * largest cache row count, and at least the widest layer dimension over the packed rows. The contiguous step's
     * shadow cells are window offsets within a step, while a paged step's shadow cells are the destination cache's
     * absolute rows, so the buffers hold whole caches.
     */
    private long shadowBufferSize() {
        long bytes = (long) ubatch * widest(l -> l.attention().kDim()) * Float.BYTES;
        for (LayerSpec l : mainTrunk.layers()) {
            if (group.attentionPlan().layerOf(l) == null) {
                continue;
            }
            GpuKvLayer kv = group.pcpAttention(l).part(0).kvLayer();
            long rows = kv.layout().rows();
            bytes = Math.max(bytes, rows * Math.max(kv.layout().rowBytesK(), kv.layout().rowBytesV()));
        }
        return bytes;
    }

    /** The gather buffers of every rank and the shadow store buffers of every rank pair. */
    private void allocateExchangeBuffers() {
        for (int rank = 0; rank < ranks; rank++) {
            queryGather[rank] = own(group.rankBackend(rank), (long) ubatch * queryRowWidth * Float.BYTES);
            queryPos[rank] = own(group.rankBackend(rank), (long) ubatch * Integer.BYTES);
        }
        for (int owner = 0; owner < ranks; owner++) {
            for (int dest = 0; dest < ranks; dest++) {
                if (dest == owner) {
                    continue;
                }
                shadowK[owner][dest] = own(group.rankBackend(owner), shadowBytes);
                shadowV[owner][dest] = own(group.rankBackend(owner), shadowBytes);
                shadowCell[owner][dest] = own(group.rankBackend(owner), (long) ubatch * Integer.BYTES);
            }
        }
    }

    private GpuBuffer own(ComputeBackend backend, long bytes) {
        GpuBuffer b = backend.allocateBuffer(bytes, false);
        owned.add(b);
        return b;
    }

    /** Every peer's trunk executor over the rank context, whose attention mixer finishes the rank's own rows. */
    private void openPeerTrunks() {
        for (int rank = 1; rank < ranks; rank++) {
            int peer = rank;
            GpuBlockRegistry registry = GpuFactoryCatalog.defaults()
                    .withMixerDecorator((layer, deviceBlock) -> new PcpPeerMixer(group, layer, peer));
            peerTrunks.add(GpuLayerExecutor.forTrunk(group.rankContext(rank), registry));
        }
    }

    // ------------------------------------------------------------------ trunk runner

    /**
     * Runs the stack's layers over the step's DualChunkSwap partition. The model stream holds the step's embedding
     * rows on entry and the stack output in global row order when it returns.
     *
     * @throws UnsupportedOperationException when a hidden tap wants a packed-row stream the split cannot serve
     */
    @Override
    public void runTrunk(int pos0, int nTokens, GpuLayerExecutor.StepMode mode, GpuLayerExecutor.TapSink taps) {
        if (mode != GpuLayerExecutor.StepMode.PREFILL || nTokens < 1 || nTokens > ubatch) {
            throw new IllegalStateException("the prefill context-parallel coordinator runs prefill steps of [1, "
                    + ubatch + "] tokens, got " + nTokens + " in " + mode);
        }
        refuseHiddenTaps(taps);
        beginStep(pos0, nTokens);
        packStreams();
        try {
            for (LayerSpec layer : mainTrunk.layers()) {
                runLayer(layer);
            }
        } finally {
            restoreStreams();
            this.swap = null;
        }
    }

    /** The shared refusal of both trunk modes: a hidden tap wants the stream while the rows lie packed. */
    private void refuseHiddenTaps(GpuLayerExecutor.TapSink taps) {
        if (taps == null) {
            return;
        }
        for (int tap = 0; tap <= mainTrunk.layers().size(); tap++) {
            if (taps.wants(tap)) {
                throw new UnsupportedOperationException("a hidden tap wants the stream at tap " + tap
                        + ", whose rows lie in every rank's DualChunkSwap packed order during the trunk; run"
                        + " without hidden state streams under the prefill context-parallel size");
            }
        }
    }

    /** Whether this runner carries the batched step of {@code layout}: a paged step with a multi-row segment. */
    @Override
    public boolean runsBatchedTrunk(StepLayout layout) {
        if (layout == null || !layout.paged() || layout.nTokens() > ubatch) {
            return false;
        }
        for (int seg = 0; seg < layout.nSeg(); seg++) {
            if (layout.len(seg) > 1) {
                return true;
            }
        }
        return false;
    }

    /**
     * Runs the stack's layers over the batched step: every multi-row segment cuts into its own DualChunkSwap chunk
     * pairs, every single-row segment replicates onto the ranks, and every rank packs its rows over its translated
     * page tables. The model stream holds the step's embedding rows in global row order on entry and the stack
     * output in the same order when it returns.
     *
     * @throws UnsupportedOperationException when a hidden tap wants a packed-row stream the split cannot serve
     */
    @Override
    public void runBatchTrunk(StepLayout layout, GpuLayerExecutor.StepMode mode, GpuLayerExecutor.TapSink taps) {
        Objects.requireNonNull(layout, "layout");
        if (mode != GpuLayerExecutor.StepMode.PREFILL || layout.nTokens() < 1 || layout.nTokens() > ubatch) {
            throw new IllegalStateException("the prefill context-parallel coordinator runs batched prefill steps of"
                    + " [1, " + ubatch + "] tokens, got " + layout.nTokens() + " in " + mode);
        }
        refuseHiddenTaps(taps);
        beginBatchedStep(layout);
        try {
            packBatchedStreams();
            for (LayerSpec layer : mainTrunk.layers()) {
                runLayer(layer);
            }
        } finally {
            restoreBatchedStreams();
            clearBatchedStep();
        }
    }

    /** Clears the batched step state the next step's begin rebuilds. */
    private void clearBatchedStep() {
        this.batched = false;
        this.batchSegments.clear();
        this.batchRuns.clear();
    }

    /** Cuts the step into the ranks' chunk pairs, uploads every rank's packed step inputs and notes the window. */
    private void beginStep(int pos0, int n) {
        this.swap = PcpChunkSwap.of(n, ranks);
        this.stepPos0 = pos0;
        this.stepTokens = n;
        group.noteStepWindow(pos0, n);
        clearBatchedStep();
        for (int rank = 0; rank < ranks; rank++) {
            int rows = swap.ownedCount(rank);
            packedRows[rank] = rows;
            packedPos0[rank] = rows > 0 ? pos0 + swap.ownedStart(rank) : -1;
            if (rows == 0) {
                continue;
            }
            int[] chunkRows = swap.rows(rank);
            for (int t = 0; t < rows; t++) {
                packedPos[rank][t] = pos0 + chunkRows[t];
            }
            int[] upload = Arrays.copyOf(packedPos[rank], rows);
            /* The step rows inside the rank's block are the rows its cache receives: its own stores and the
             * scatters of the other ranks' chunk pairs. A chunk pair can lie wholly outside the rank's block
             * (every rank projects and attends its pair, but only the block owner's rows store), and the rank
             * then receives nothing and stores nothing. */
            int from = Math.max(pos0, group.contextBlockStart(rank));
            int to = Math.min(pos0 + n, group.contextBlockEnd(rank));
            group.rankMemory(rank).stepInputs().uploadPacked(upload, from < to ? from : 0,
                    from < to ? to : 0);
        }
    }

    /** Packs every rank's stream: every peer receives its rows first (the far chunk's move would overwrite
     * chunk rows the peers still read), then rank 0's far chunk moves behind its first chunk. */
    private void packStreams() {
        ScratchSlot mainX = modelContext.scratch().slot(ScratchSlots.X);
        for (int rank = 1; rank < ranks; rank++) {
            if (packedRows[rank] == 0) {
                continue;
            }
            ScratchSlot peerX = group.rankContext(rank).scratch().slot(ScratchSlots.X);
            int peerFirst = swap.chunkLen(swap.firstChunk(rank));
            copyRows(group.rankQueue(0), mainX, swap.chunkStart(swap.firstChunk(rank)),
                    group.rankQueue(rank), peerX, 0, peerFirst);
            copyRows(group.rankQueue(0), mainX, swap.chunkStart(swap.secondChunk(rank)),
                    group.rankQueue(rank), peerX, peerFirst, swap.chunkLen(swap.secondChunk(rank)));
        }
        int firstLen = swap.chunkLen(swap.firstChunk(0));
        int secondLen = swap.chunkLen(swap.secondChunk(0));
        if (secondLen > 0) {
            copyRows(group.rankQueue(0), mainX, swap.chunkStart(swap.secondChunk(0)),
                    group.rankQueue(0), mainX, firstLen, secondLen);
        }
    }

    /** Restores rank 0's stream to the global row order: the far chunk moves back up first (the peers' rows
     * return into the packed rows it leaves), then the peers' rows return. */
    private void restoreStreams() {
        ScratchSlot mainX = modelContext.scratch().slot(ScratchSlots.X);
        int firstLen = swap.chunkLen(swap.firstChunk(0));
        int secondLen = swap.chunkLen(swap.secondChunk(0));
        if (secondLen > 0) {
            copyRows(group.rankQueue(0), mainX, firstLen, group.rankQueue(0), mainX,
                    swap.chunkStart(swap.secondChunk(0)), secondLen);
        }
        for (int rank = 1; rank < ranks; rank++) {
            if (packedRows[rank] == 0) {
                continue;
            }
            ScratchSlot peerX = group.rankContext(rank).scratch().slot(ScratchSlots.X);
            int peerFirst = swap.chunkLen(swap.firstChunk(rank));
            copyRows(group.rankQueue(rank), peerX, 0, group.rankQueue(0), mainX,
                    swap.chunkStart(swap.firstChunk(rank)), peerFirst);
            copyRows(group.rankQueue(rank), peerX, peerFirst, group.rankQueue(0), mainX,
                    swap.chunkStart(swap.secondChunk(rank)), swap.chunkLen(swap.secondChunk(rank)));
        }
    }

    /** Copies {@code rows} stream rows of {@code src} row {@code srcRow} into {@code dst} row {@code dstRow}. */
    private void copyRows(ComputeQueue srcQueue, ScratchSlot src, int srcRow, ComputeQueue dstQueue, ScratchSlot dst,
                          int dstRow, int rows) {
        if (rows <= 0) {
            return;
        }
        RankExchange.replay(srcQueue, src.buffer(), src.offset() + (long) srcRow * streamWidth,
                dstQueue, dst.buffer(), dst.offset() + (long) dstRow * streamWidth, (long) rows * streamWidth);
    }

    // ------------------------------------------------------------------ batched steps (the paged memory)

    /** One segment of a batched step: its DualChunkSwap partition, or a replicated single row. */
    private static final class SegmentPlan {
        final int start;
        final int len;
        /** The segment's partition, or null when its single row replicates onto every rank. */
        final PcpChunkSwap swap;

        SegmentPlan(int start, int len, PcpChunkSwap swap) {
            this.start = start;
            this.len = len;
            this.swap = swap;
        }
    }

    /** One maximal run of a rank's packed rows that name consecutive global rows of the step. */
    private static final class RowRun {
        final int packed;
        final int global;
        final int rows;

        RowRun(int packed, int global, int rows) {
            this.packed = packed;
            this.global = global;
            this.rows = rows;
        }
    }

    /**
     * Cuts every multi-row segment of the batched step into the ranks' chunk pairs, replicates the single-row
     * segments, packs every rank's global row indices into maximal runs and uploads every rank's packed step inputs
     * over its translated page tables.
     */
    private void beginBatchedStep(StepLayout layout) {
        this.batched = true;
        this.swap = null;
        this.stepPos0 = -1;
        this.stepTokens = layout.nTokens();
        group.noteStepWindow(-1, layout.nTokens());
        batchSegments.clear();
        batchRuns.clear();
        for (int rank = 0; rank < ranks; rank++) {
            batchRuns.add(new ArrayList<>());
            packedRows[rank] = 0;
            packedPos0[rank] = -1;
        }
        int[] positions = new int[layout.nTokens()];
        for (int seg = 0; seg < layout.nSeg(); seg++) {
            int start = layout.tokStart(seg);
            int len = layout.len(seg);
            int pos0 = layout.pos0(seg);
            for (int i = 0; i < len; i++) {
                positions[start + i] = pos0 + i;
            }
            SegmentPlan plan = new SegmentPlan(start, len, len > 1 ? PcpChunkSwap.of(len, ranks) : null);
            batchSegments.add(plan);
            for (int rank = 0; rank < ranks; rank++) {
                if (plan.swap == null) {
                    appendRun(rank, start, 1);
                } else {
                    appendChunkRuns(rank, start, plan.swap.rows(rank));
                }
            }
        }
        for (int rank = 0; rank < ranks; rank++) {
            int rows = packedRows[rank];
            if (rows == 0) {
                continue;
            }
            int[] rowsOfRank = new int[rows];
            for (int t = 0; t < rows; t++) {
                rowsOfRank[t] = packedGlobal[rank][t];
                packedPos[rank][t] = positions[packedGlobal[rank][t]];
            }
            packedPos0[rank] = packedPos[rank][0];
            group.rankMemory(rank).stepInputs().packPagedStepRows(rowsOfRank);
        }
    }

    /** Appends {@code rows} consecutive global rows from {@code global} to rank {@code rank}'s packed partition. */
    private void appendRun(int rank, int global, int rows) {
        List<RowRun> runs = batchRuns.get(rank);
        int packed = packedRows[rank];
        if (!runs.isEmpty()) {
            RowRun last = runs.get(runs.size() - 1);
            if (last.packed + last.rows == packed && last.global + last.rows == global) {
                runs.set(runs.size() - 1, new RowRun(last.packed, last.global, last.rows + rows));
                for (int i = 0; i < rows; i++) {
                    packedGlobal[rank][packed + i] = global + i;
                }
                packedRows[rank] = packed + rows;
                return;
            }
        }
        runs.add(new RowRun(packed, global, rows));
        for (int i = 0; i < rows; i++) {
            packedGlobal[rank][packed + i] = global + i;
        }
        packedRows[rank] = packed + rows;
    }

    /** Appends the rows of one segment's DualChunkSwap partition (ascending offsets, in chunk order) as runs. */
    private void appendChunkRuns(int rank, int segStart, int[] rows) {
        int i = 0;
        while (i < rows.length) {
            int j = i;
            while (j + 1 < rows.length && rows[j + 1] == rows[j] + 1) {
                j++;
            }
            appendRun(rank, segStart + rows[i], j - i + 1);
            i = j + 1;
        }
    }

    /** The segment of the batched step that holds global row {@code global}. */
    private SegmentPlan segmentOf(int global) {
        for (SegmentPlan seg : batchSegments) {
            if (global < seg.start + seg.len) {
                return seg;
            }
        }
        throw new IndexOutOfBoundsException("batched step row " + global + " outside the " + stepTokens
                + " step rows");
    }

    /**
     * Packs every rank's stream: every peer receives its runs first (they read the rows rank 0's backward
     * self-moves then overwrite), then rank 0's runs move down behind their sources in ascending packed order.
     */
    private void packBatchedStreams() {
        ScratchSlot mainX = modelContext.scratch().slot(ScratchSlots.X);
        for (int rank = 1; rank < ranks; rank++) {
            ScratchSlot peerX = group.rankContext(rank).scratch().slot(ScratchSlots.X);
            for (RowRun run : batchRuns.get(rank)) {
                copyRows(group.rankQueue(0), mainX, run.global, group.rankQueue(rank), peerX, run.packed, run.rows);
            }
        }
        for (RowRun run : batchRuns.get(0)) {
            if (run.packed != run.global) {
                copyRows(group.rankQueue(0), mainX, run.global, group.rankQueue(0), mainX, run.packed, run.rows);
            }
        }
    }

    /**
     * Restores rank 0's stream to the global row order: its runs move back up in descending packed order (a later
     * source always lies at or above the destinations already written), then every peer's runs return.
     */
    private void restoreBatchedStreams() {
        ScratchSlot mainX = modelContext.scratch().slot(ScratchSlots.X);
        List<RowRun> ownRuns = batchRuns.get(0);
        for (int r = ownRuns.size() - 1; r >= 0; r--) {
            RowRun run = ownRuns.get(r);
            if (run.packed != run.global) {
                copyRows(group.rankQueue(0), mainX, run.packed, group.rankQueue(0), mainX, run.global, run.rows);
            }
        }
        for (int rank = 1; rank < ranks; rank++) {
            ScratchSlot peerX = group.rankContext(rank).scratch().slot(ScratchSlots.X);
            for (RowRun run : batchRuns.get(rank)) {
                copyRows(group.rankQueue(rank), peerX, run.packed, group.rankQueue(0), mainX, run.global, run.rows);
            }
        }
    }

    /**
     * The query all-gather of the batched step: every rank's gather buffer receives the whole step's query rows at
     * their global offsets — each row from its canonical owner's packed rows (a multi-row segment's DualChunkSwap
     * owner, a replicated row's rank 0). The launches read the rows' true positions from the translated step's
     * segment buffers, so no position buffer is written here.
     *
     * @param queryRowWidth the layer's query row width ({@code q_rope_dim})
     */
    void gatherBatchedQueries(List<GpuAttentionMixer> parts, int queryRowWidth) {
        for (int owner = 0; owner < ranks; owner++) {
            if (packedRows[owner] == 0) {
                continue;
            }
            GpuBuffer q = parts.get(owner).queryBuffer();
            for (int dest = 0; dest < ranks; dest++) {
                copyCanonicalRuns(group.rankQueue(owner), q, group.rankQueue(dest), queryGather[dest], owner, dest,
                        queryRowWidth);
            }
        }
    }

    /** Copies owner {@code owner}'s canonical query runs into {@code dst} at the rows' global offsets. */
    private void copyCanonicalRuns(ComputeQueue srcQueue, GpuBuffer src, ComputeQueue dstQueue, GpuBuffer dst,
                                   int owner, int dest, int rowWidth) {
        int t = 0;
        while (t < packedRows[owner]) {
            if (!canonicalOwner(owner, packedGlobal[owner][t])) {
                t++;
                continue;
            }
            int from = t;
            while (t + 1 < packedRows[owner] && packedGlobal[owner][t + 1] == packedGlobal[owner][t] + 1
                    && canonicalOwner(owner, packedGlobal[owner][t + 1])) {
                t++;
            }
            copyQuery(srcQueue, src, from, dstQueue, dst, packedGlobal[owner][from], t - from + 1, rowWidth);
            t++;
        }
    }

    /** Whether owner {@code owner} is the gathered copy's canonical source of global row {@code global}. */
    private boolean canonicalOwner(int owner, int global) {
        SegmentPlan seg = segmentOf(global);
        return seg.swap == null ? owner == 0 : seg.swap.ownsRow(owner, global - seg.start);
    }

    /**
     * The cache scatter of the batched step: for every rank pair, the chunk owner re-runs its stores over the rows
     * whose K/V lands in the destination rank's pages (the destination's translated cells; the replicated rows keep
     * their page owner's own store) and moves the contiguous cell runs into the destination cache.
     */
    void scatterBatchedCacheRows(List<GpuAttentionMixer> parts) {
        for (int dest = 0; dest < ranks; dest++) {
            PagedStep destStep = group.rankMemory(dest).stepInputs().currentPagedStep();
            for (int owner = 0; owner < ranks; owner++) {
                if (owner != dest && packedRows[owner] > 0) {
                    scatterToPagedCells(parts, owner, dest, destStep);
                }
            }
        }
    }

    /** Moves owner {@code owner}'s rows whose cells the destination's translated step holds in {@code destStep}. */
    private void scatterToPagedCells(List<GpuAttentionMixer> parts, int owner, int dest, PagedStep destStep) {
        GpuAttentionMixer part = parts.get(owner);
        GpuKvLayer kv = part.kvLayer();
        KvLayout layout = kv.layout();
        int destGroup = parts.get(dest).kvLayer().group();
        int[] destCells = destStep.cellIdx(destGroup);
        int[] cells = new int[packedRows[owner]];
        boolean any = false;
        for (int t = 0; t < packedRows[owner]; t++) {
            int global = packedGlobal[owner][t];
            int cell = segmentOf(global).swap != null ? destCells[global] : -1;
            cells[t] = cell;
            any |= cell >= 0;
        }
        if (!any) {
            return;
        }
        writeInts(shadowCell[owner][dest], group.rankQueue(owner), cells, packedRows[owner]);
        part.shadowStore(packedRows[owner], shadowCell[owner][dest], shadowK[owner][dest], shadowV[owner][dest]);
        moveCellRuns(owner, dest, cells, layout.rowBytesK(), shadowK[owner][dest], parts.get(dest).kvLayer().k());
        moveCellRuns(owner, dest, cells, layout.rowBytesV(), shadowV[owner][dest], parts.get(dest).kvLayer().v());
    }

    /** Moves the contiguous cell runs of {@code cells} from {@code shadow} into the destination {@code cache}. */
    private void moveCellRuns(int owner, int dest, int[] cells, long rowBytes, GpuBuffer shadow, GpuBuffer cache) {
        ComputeQueue ownerQueue = group.rankQueue(owner);
        ComputeQueue destQueue = group.rankQueue(dest);
        int t = 0;
        while (t < cells.length) {
            if (cells[t] < 0) {
                t++;
                continue;
            }
            int from = t;
            while (t + 1 < cells.length && cells[t + 1] == cells[t] + 1) {
                t++;
            }
            RankExchange.replayBytes(ownerQueue, shadow, (long) cells[from] * rowBytes, destQueue, cache,
                    (long) cells[from] * rowBytes, (long) (t - from + 1) * rowBytes);
            t++;
        }
    }

    /**
     * The attention scatter of the batched step: rank 0's merged attention buffer holds the whole step in global
     * row order, so every peer's buffer receives its packed runs first (they read the rows rank 0's backward
     * self-moves then overwrite), then rank 0's runs move in ascending packed order.
     *
     * @param rowWidth the layer's attention row width ({@code n_head * d_v})
     */
    void scatterBatchedAttention(CpAttentionMixer mixer, int rowWidth) {
        GpuBuffer merged = mixer.part(0).attentionBuffer();
        for (int rank = 1; rank < ranks; rank++) {
            GpuBuffer peer = mixer.part(rank).attentionBuffer();
            for (RowRun run : batchRuns.get(rank)) {
                if (run.rows > 0) {
                    RankExchange.replay(group.rankQueue(0), merged, (long) run.global * rowWidth,
                            group.rankQueue(rank), peer, (long) run.packed * rowWidth, (long) run.rows * rowWidth);
                }
            }
        }
        for (RowRun run : batchRuns.get(0)) {
            if (run.packed != run.global) {
                RankExchange.replay(group.rankQueue(0), merged, (long) run.global * rowWidth,
                        group.rankQueue(0), merged, (long) run.packed * rowWidth, (long) run.rows * rowWidth);
            }
        }
    }

    /** One layer over every rank: the norms, rank 0's exchange attention, the peers' finishes, the posts. */
    private void runLayer(LayerSpec layer) {
        int il = layer.il();
        ScratchSlot in0 = mainTrunk.runLayerNorm(il, packedPos0[0], packedRows[0]);
        for (int rank = 1; rank < ranks; rank++) {
            peerInputs[rank] = packedRows[rank] > 0
                    ? peerTrunks.get(rank - 1).runLayerNorm(il, packedPos0[rank], packedRows[rank])
                    : null;
        }
        ScratchSlot deferred0 = mainTrunk.runLayerAttention(il, packedPos0[0], packedRows[0],
                GpuLayerExecutor.StepMode.PREFILL, in0);
        for (int rank = 1; rank < ranks; rank++) {
            peerDeferred[rank] = packedRows[rank] > 0
                    ? peerTrunks.get(rank - 1).runLayerAttention(il, packedPos0[rank], packedRows[rank],
                    GpuLayerExecutor.StepMode.PREFILL, peerInputs[rank])
                    : null;
        }
        mainTrunk.runLayerPost(il, packedPos0[0], packedRows[0], deferred0);
        for (int rank = 1; rank < ranks; rank++) {
            if (packedRows[rank] > 0) {
                peerTrunks.get(rank - 1).runLayerPost(il, packedPos0[rank], packedRows[rank], peerDeferred[rank]);
            }
        }
    }

    // ------------------------------------------------------------------ exchange (PcpAttentionMixer)

    /** The swap partition of the running contiguous step. */
    PcpChunkSwap swap() {
        requireContiguousStep();
        return swap;
    }

    /** The running step's whole query row count. */
    int stepTokens() {
        requireStep();
        return stepTokens;
    }

    /** The running step's first position ({@code -1} on a batched step, whose segments carry their own). */
    int stepPos0() {
        requireStep();
        return stepPos0;
    }

    private void requireContiguousStep() {
        if (swap == null) {
            throw new IllegalStateException(batched
                    ? "the coordinator runs a batched step, whose segments carry their own partitions"
                    : "the coordinator runs no step");
        }
    }

    private void requireStep() {
        if (swap == null && !batched) {
            throw new IllegalStateException("the coordinator runs no step");
        }
    }

    /** Whether the coordinator runs the batched paged step (per-segment partitions over the translated tables). */
    boolean batchedStep() {
        return batched;
    }

    /** Whether a DualChunkSwap step is running: the mixers route their phases through the coordinator while it is. */
    boolean running() {
        return swap != null || batched;
    }

    /** Rows of rank {@code rank}'s packed partition of the running step. */
    int packedRows(int rank) {
        requireStep();
        return packedRows[rank];
    }

    /** First position of rank {@code rank}'s packed rows ({@code -1} when it owns none). */
    int packedPos0(int rank) {
        return packedPos0[rank];
    }

    /** The translated paged step rank {@code rank} runs the batched step over. */
    PagedStep rankStep(int rank) {
        requireStep();
        PagedStep step = group.rankMemory(rank).stepInputs().currentPagedStep();
        if (step == null) {
            throw new IllegalStateException("rank " + rank + " holds no translated paged step for the batched"
                    + " prefill context-parallel step");
        }
        return step;
    }

    /** The normed input slot of rank {@code rank}'s trunk (its phases packed it). */
    ScratchSlot normedSlot(int rank) {
        return group.rankContext(rank).scratch().slot(ScratchSlots.NORMED);
    }

    /** The gathered query rows of the whole step on rank {@code rank}'s device (global row order). */
    GpuBuffer queryGather(int rank) {
        return queryGather[rank];
    }

    /** The gathered rows' true positions on rank {@code rank}'s device. */
    GpuBuffer queryPos(int rank) {
        return queryPos[rank];
    }

    /**
     * The query all-gather: every rank's packed query rows land in every rank's gather buffer at their global row
     * offsets, and every rank's position buffer holds the whole step's true positions.
     *
     * @param queryRowWidth the layer's query row width ({@code q_rope_dim})
     */
    void gatherQueries(List<GpuAttentionMixer> parts, int queryRowWidth) {
        for (int dest = 0; dest < ranks; dest++) {
            writeInts(queryPos[dest], group.rankQueue(dest), positions(stepPos0, stepTokens), stepTokens);
            for (int owner = 0; owner < ranks; owner++) {
                if (packedRows[owner] == 0) {
                    continue;
                }
                GpuBuffer q = parts.get(owner).queryBuffer();
                int firstLen = swap.chunkLen(swap.firstChunk(owner));
                copyQuery(group.rankQueue(owner), q, 0, group.rankQueue(dest), queryGather[dest],
                        swap.chunkStart(swap.firstChunk(owner)), firstLen, queryRowWidth);
                copyQuery(group.rankQueue(owner), q, firstLen, group.rankQueue(dest), queryGather[dest],
                        swap.chunkStart(swap.secondChunk(owner)), swap.chunkLen(swap.secondChunk(owner)),
                        queryRowWidth);
            }
        }
    }

    /** Copies one packed query chunk to its global row offset in a gather buffer (skips an identity copy). */
    private void copyQuery(ComputeQueue srcQueue, GpuBuffer src, int srcRow, ComputeQueue dstQueue, GpuBuffer dst,
                           int dstRow, int rows, int rowWidth) {
        if (rows <= 0 || (src == dst && srcRow == dstRow)) {
            return;
        }
        RankExchange.replay(srcQueue, src, (long) srcRow * rowWidth, dstQueue, dst, (long) dstRow * rowWidth,
                (long) rows * rowWidth);
    }

    private int[] positions(int pos0, int n) {
        int[] out = new int[n];
        for (int t = 0; t < n; t++) {
            out[t] = pos0 + t;
        }
        return out;
    }

    /**
     * The cache scatter: for every destination block the step reaches, every chunk owner re-runs its stores over the
     * pair (the shadow store, rebound cell indices and destinations) and moves the contiguous row segments into the
     * destination cache. A destination whose whole block lies outside the step receives nothing; an owner's own rows
     * are already stored.
     */
    void scatterCacheRows(List<GpuAttentionMixer> parts) {
        for (int dest = 0; dest < ranks; dest++) {
            int lo = Math.max(0, group.contextBlockStart(dest) - stepPos0);
            int hi = Math.min(stepTokens, group.contextBlockEnd(dest) - stepPos0);
            if (lo >= hi) {
                continue;
            }
            for (int owner = 0; owner < ranks; owner++) {
                if (owner != dest && packedRows[owner] > 0) {
                    scatterToBlock(parts, owner, dest, lo, hi);
                }
            }
        }
    }

    /** Moves owner {@code owner}'s rows inside destination {@code dest}'s step-row window {@code [lo, hi)}. */
    private void scatterToBlock(List<GpuAttentionMixer> parts, int owner, int dest, int lo, int hi) {
        GpuAttentionMixer part = parts.get(owner);
        GpuKvLayer kv = part.kvLayer();
        KvLayout layout = kv.layout();
        int[] cells = shadowCells(owner, dest, lo, hi);
        if (cells.length == 0) {
            return;
        }
        writeInts(shadowCell[owner][dest], group.rankQueue(owner), cells, packedRows[owner]);
        part.shadowStore(packedRows[owner], shadowCell[owner][dest], shadowK[owner][dest], shadowV[owner][dest]);
        GpuKvLayer destKv = parts.get(dest).kvLayer();
        ComputeQueue ownerQueue = group.rankQueue(owner);
        moveChunkRows(ownerQueue, shadowK[owner][dest], destKv.k(), layout.rowBytesK(), swap.firstChunk(owner),
                dest, lo, hi);
        moveChunkRows(ownerQueue, shadowV[owner][dest], destKv.v(), layout.rowBytesV(), swap.firstChunk(owner),
                dest, lo, hi);
        moveChunkRows(ownerQueue, shadowK[owner][dest], destKv.k(), layout.rowBytesK(), swap.secondChunk(owner),
                dest, lo, hi);
        moveChunkRows(ownerQueue, shadowV[owner][dest], destKv.v(), layout.rowBytesV(), swap.secondChunk(owner),
                dest, lo, hi);
    }

    /**
     * The cell indices of owner {@code owner}'s packed rows for destination {@code dest}'s block; an empty array when
     * no row of the pair lands in the block, so the shadow store is skipped. Rows outside the block carry {@code -1},
     * which the store kernels skip.
     */
    private int[] shadowCells(int owner, int dest, int lo, int hi) {
        int[] out = new int[packedRows[owner]];
        boolean any = false;
        for (int t = 0; t < packedRows[owner]; t++) {
            int stepRow = packedPos[owner][t] - stepPos0;
            int row = stepRow >= lo && stepRow < hi ? stepRow - lo : -1;
            out[t] = row;
            any |= row >= 0;
        }
        return any ? out : new int[0];
    }

    /** Moves one chunk's rows inside {@code [lo, hi)} from the owner's shadow rows into the destination cache. */
    private void moveChunkRows(ComputeQueue ownerQueue, GpuBuffer shadow, GpuBuffer cache, long rowBytes, int chunk,
                               int dest, int lo, int hi) {
        int from = Math.max(lo, swap.chunkStart(chunk));
        int to = Math.min(hi, swap.chunkStart(chunk) + swap.chunkLen(chunk));
        if (from >= to) {
            return;
        }
        RankExchange.replayBytes(ownerQueue, shadow, (long) (from - lo) * rowBytes, group.rankQueue(dest), cache,
                (long) (stepPos0 + from - group.contextBlockStart(dest)) * rowBytes,
                (long) (to - from) * rowBytes);
    }

    /**
     * The attention scatter: rank 0's merged attention buffer holds the whole step in global row order, so every
     * peer's buffer receives its two chunk segments in its packed order first (the far chunk's move would overwrite
     * a merged row the peers still read), then the far chunk moves behind rank 0's first chunk.
     *
     * @param rowWidth the layer's attention row width ({@code n_head * d_v})
     */
    void scatterAttention(CpAttentionMixer mixer, int rowWidth) {
        GpuBuffer merged = mixer.part(0).attentionBuffer();
        for (int rank = 1; rank < ranks; rank++) {
            if (packedRows[rank] == 0) {
                continue;
            }
            GpuBuffer peer = mixer.part(rank).attentionBuffer();
            int peerFirst = swap.chunkLen(swap.firstChunk(rank));
            int peerSecond = swap.chunkLen(swap.secondChunk(rank));
            if (peerFirst > 0) {
                RankExchange.replay(group.rankQueue(0), merged,
                        (long) swap.chunkStart(swap.firstChunk(rank)) * rowWidth,
                        group.rankQueue(rank), peer, 0L, (long) peerFirst * rowWidth);
            }
            if (peerSecond > 0) {
                RankExchange.replay(group.rankQueue(0), merged,
                        (long) swap.chunkStart(swap.secondChunk(rank)) * rowWidth,
                        group.rankQueue(rank), peer, (long) peerFirst * rowWidth, (long) peerSecond * rowWidth);
            }
        }
        int firstLen = swap.chunkLen(swap.firstChunk(0));
        int secondLen = swap.chunkLen(swap.secondChunk(0));
        if (secondLen > 0) {
            RankExchange.replay(group.rankQueue(0), merged, (long) swap.chunkStart(swap.secondChunk(0)) * rowWidth,
                    group.rankQueue(0), merged, (long) firstLen * rowWidth, (long) secondLen * rowWidth);
        }
    }

    /** Finishes a peer trunk's attention phase: the merged attention output projection over the rank's rows. */
    void finishPeerRows(int rank, LayerSpec layer, int rows, ScratchSlot out) {
        if (!running() || packedRows[rank] != rows) {
            throw new IllegalStateException("the peer trunk of rank " + rank + " finishes " + rows + " rows of layer "
                    + layer.il() + ", but the coordinator packed "
                    + (running() ? String.valueOf(packedRows[rank]) : "no step"));
        }
        group.pcpAttention(layer).part(rank).finishStep(rows, out);
    }

    /** Writes {@code n} ints of {@code values} into a buffer of the given queue. */
    private void writeInts(GpuBuffer dst, ComputeQueue queue, int[] values, int n) {
        for (int t = 0; t < n; t++) {
            intStage.putInt(t * Integer.BYTES, values[t]);
        }
        GpuEvent e = queue.enqueueWriteBuffer(dst, 0L, (long) n * Integer.BYTES, intStage.duplicate());
        if (e != null) {
            e.close();
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        for (GpuLayerExecutor trunk : new ArrayList<>(peerTrunks)) {
            try {
                trunk.close();
            } catch (RuntimeException e) {
                failure = failure == null ? e : failure;
                failure.addSuppressed(e);
            }
        }
        peerTrunks.clear();
        for (GpuBuffer b : new ArrayList<>(owned)) {
            try {
                b.close();
            } catch (RuntimeException e) {
                failure = failure == null ? e : failure;
                failure.addSuppressed(e);
            }
        }
        owned.clear();
        if (failure != null) {
            throw failure;
        }
    }
}
