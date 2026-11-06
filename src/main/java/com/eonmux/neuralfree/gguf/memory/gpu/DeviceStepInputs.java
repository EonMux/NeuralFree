package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.CompressorRollback;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.memory.paged.RowReport;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Device per-step inputs ({@link GpuStepInputs}) plus the host mirror of every group's {@code kv_pos}, which is the
 * visibility bookkeeping: {@link #upload(int, int)} marks the step's rows in the mirror and writes each device
 * buffer once (the changed {@code kv_pos} row range, or the whole ring when the step wraps). Rollback, reset and
 * the context shift change the mirror and rewrite {@code kv_pos} the same way.
 *
 * <p>Also owns the position buffers of caches without per-step addressing: {@link #identityRows()} ({@code i} at
 * index {@code i}) serves as {@code cell_idx} and {@code kv_pos} of caches written once at rows {@code 0..n-1}
 * (T5 cross-attention), and {@link #shiftDelta(int)} holds the per-row deltas of the last context shift for
 * {@code LLM_KV_RopeShift_*}.
 *
 * <p>Not thread-safe; used from the thread that drives the owning memory's queue.
 */
public final class DeviceStepInputs implements GpuStepInputs, AutoCloseable {

    private final ComputeBackend backend;
    private final ComputeQueue queue;
    private final int ubatch;
    private final MropeTextPositions textPositions;
    private final SequenceState sequence;
    private final GroupPositions[] positions;
    private final GpuBuffer[] kvPos;
    private final GpuBuffer[] cellIdx;
    private final GpuBuffer[] shiftDelta;
    private final GpuBuffer qPos;
    private final GpuBuffer ropePos;
    private final GpuBuffer stateRows;
    private final GpuBuffer identityRows;
    private final GpuBuffer treeMask;
    private final Paged paged;
    private final int identityLength;
    private final int[] ints;
    private final int[] treePositions;
    /** The current tree step's mask rows, absolute over the step (the segments' uploads shift them local). */
    private final long[] treeMaskRows;
    private final byte[] staging;
    private final ByteBuffer stagingView;
    private int pos0;
    private int nTokens;
    /** Whether the current step runs under a tree mask ({@link #beginTreeStep}). */
    private boolean treeStep;
    /** Whether the current tree step's launch-local rows are already uploaded ({@link #beginTreeStep}). */
    private boolean treeSegmentUploaded;
    /** Whether the current paged step attends without the causal mask ({@link #stepNonCausal()}). */
    private boolean nonCausal;
    /** Host copy of the current paged step's row positions ({@link #rowPosition}), {@code int[ubatch]}. */
    private final int[] rowPositions;
    /** Staging copy of a packed step's true positions, which the RoPE fill would otherwise overwrite. */
    private final int[] packedTruePos;
    private CompressorRollback[] compressorSteps;
    private boolean closed;

    private DeviceStepInputs(ComputeBackend backend, ComputeQueue queue, List<MemoryPlan.KvGroup> groups, int ubatch,
                             int identityLength, MropeTextPositions textPositions, SequenceState sequence,
                             PagedSequences pages, List<GpuBuffer> owned) {
        this.backend = backend;
        this.queue = queue;
        this.ubatch = ubatch;
        this.identityLength = identityLength;
        this.textPositions = textPositions;
        this.sequence = sequence;
        this.paged = pages != null ? new Paged(backend, queue, groups.size(), ubatch, pages, owned) : null;
        int n = groups.size();
        this.positions = new GroupPositions[n];
        this.kvPos = new GpuBuffer[n];
        this.cellIdx = new GpuBuffer[n];
        this.shiftDelta = new GpuBuffer[n];
        int maxInts = Math.max(MropeTextPositions.SECTIONS * ubatch, identityLength);
        for (int g = 0; g < n; g++) {
            MemoryPlan.KvGroup group = groups.get(g);
            if (group.index() != g) {
                throw new IllegalArgumentException("KV group at position " + g + " has index " + group.index());
            }
            positions[g] = new GroupPositions(group);
            maxInts = Math.max(maxInts, group.rows());
        }
        this.ints = new int[maxInts];
        this.treePositions = new int[ubatch];
        this.treeMaskRows = new long[ubatch];
        this.rowPositions = new int[ubatch];
        this.packedTruePos = new int[ubatch];
        this.staging = new byte[maxInts * Integer.BYTES];
        this.stagingView = ByteBuffer.wrap(staging).order(ByteOrder.LITTLE_ENDIAN);
        for (int g = 0; g < n; g++) {
            kvPos[g] = track(owned, alloc((long) groups.get(g).rows() * Integer.BYTES));
            queue.enqueueFillBuffer(kvPos[g], 0, kvPos[g].sizeInBytes(), -1);
            cellIdx[g] = track(owned, alloc((long) ubatch * Integer.BYTES));
            queue.enqueueFillBuffer(cellIdx[g], 0, cellIdx[g].sizeInBytes(), 0);
        }
        this.qPos = track(owned, alloc((long) ubatch * Integer.BYTES));
        queue.enqueueFillBuffer(qPos, 0, qPos.sizeInBytes(), 0);
        this.ropePos = track(owned, alloc((long) MropeTextPositions.SECTIONS * ubatch * Integer.BYTES));
        queue.enqueueFillBuffer(ropePos, 0, ropePos.sizeInBytes(), 0);
        this.stateRows = track(owned, alloc(Integer.BYTES));
        queue.enqueueFillBuffer(stateRows, 0, Integer.BYTES, 0);
        this.identityRows = track(owned, alloc((long) identityLength * Integer.BYTES));
        this.treeMask = track(owned, alloc((long) ubatch * Long.BYTES));
        for (int i = 0; i < identityLength; i++) {
            ints[i] = i;
        }
        writeInts(identityRows, 0, ints, identityLength);
    }

    /**
     * Allocates the per-step buffers for {@code groups} and empties every {@code kv_pos}.
     *
     * @param ubatch         maximum tokens per step
     * @param identityLength length of {@link #identityRows()} (at least {@code ubatch})
     * @param textPositions  section pattern of {@link #ropePos()}
     * @param sequence       the owning memory's sequence record; marked wrapped when a ring row is overwritten
     * @param pages          the owning memory's paged bookkeeping, whose steps {@link #beginPagedStep} uploads, or
     *                       null for the contiguous form
     * @throws IllegalArgumentException for a non-positive {@code ubatch} or {@code identityLength < ubatch}
     */
    public static DeviceStepInputs allocate(ComputeBackend backend, ComputeQueue queue, List<MemoryPlan.KvGroup> groups,
                                            int ubatch, int identityLength, MropeTextPositions textPositions,
                                            SequenceState sequence, PagedSequences pages) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(queue, "queue");
        Objects.requireNonNull(groups, "groups");
        Objects.requireNonNull(textPositions, "textPositions");
        Objects.requireNonNull(sequence, "sequence");
        if (ubatch <= 0) {
            throw new IllegalArgumentException("step inputs need a positive ubatch, got " + ubatch);
        }
        if (identityLength < ubatch) {
            throw new IllegalArgumentException("identity rows (" + identityLength + ") must cover the ubatch of "
                    + ubatch);
        }
        List<GpuBuffer> owned = new ArrayList<>();
        try {
            return new DeviceStepInputs(backend, queue, groups, ubatch, identityLength, textPositions, sequence,
                    pages, owned);
        } catch (RuntimeException e) {
            for (GpuBuffer b : owned) {
                b.close();
            }
            throw e;
        }
    }

    private GpuBuffer alloc(long bytes) {
        return backend.allocateBuffer(bytes, true);
    }

    private static GpuBuffer track(List<GpuBuffer> owned, GpuBuffer b) {
        owned.add(b);
        return b;
    }

    // ------------------------------------------------------------------ GpuStepInputs

    @Override
    public int groups() {
        return positions.length;
    }

    @Override
    public GpuBuffer kvPos(int group) {
        checkOpen();
        return kvPos[checkGroup(group)];
    }

    @Override
    public GpuBuffer cellIdx(int group) {
        checkOpen();
        return cellIdx[checkGroup(group)];
    }

    @Override
    public GpuBuffer qPos() {
        checkOpen();
        return qPos;
    }

    @Override
    public GpuBuffer ropePos() {
        checkOpen();
        return ropePos;
    }

    @Override
    public GpuBuffer stateRows() {
        checkOpen();
        return stateRows;
    }

    @Override
    public int pos0() {
        return pos0;
    }

    @Override
    public int nTokens() {
        return nTokens;
    }

    /**
     * Writes {@code cell_idx}, {@code q_pos} and {@code rope_pos} for positions {@code [pos0, pos0 + nTokens)} and
     * stores those positions in their rows of every group's {@code kv_pos}. All arguments are validated against
     * every group before anything changes.
     */
    @Override
    public void upload(int pos0, int nTokens) {
        checkOpen();
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException("step of " + nTokens + " tokens outside [1, " + ubatch + "]");
        }
        if (pos0 < 0 || (long) pos0 + nTokens > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("step positions [" + pos0 + ", " + ((long) pos0 + nTokens)
                    + ") are out of range");
        }
        for (GroupPositions p : positions) {
            p.checkStep(pos0, nTokens);
        }
        boolean overwrote = false;
        for (int g = 0; g < positions.length; g++) {
            GroupPositions p = positions[g];
            int from = p.spanFrom(pos0, nTokens);
            int to = p.spanTo(pos0, nTokens);
            overwrote |= p.store(pos0, nTokens);
            for (int t = 0; t < nTokens; t++) {
                int pos = pos0 + t;
                // a context shard stores only the positions of its block; the others carry a -1 cell (skipped)
                ints[t] = p.group().holds(pos) ? p.group().rowOf(pos) : -1;
            }
            writeInts(cellIdx[g], 0, ints, nTokens);
            writeMirror(g, from, to);
        }
        for (int t = 0; t < nTokens; t++) {
            ints[t] = pos0 + t;
        }
        writeInts(qPos, 0, ints, nTokens);
        textPositions.fill(pos0, nTokens, ints);
        writeInts(ropePos, 0, ints, MropeTextPositions.SECTIONS * nTokens);
        this.pos0 = pos0;
        this.nTokens = nTokens;
        if (overwrote) {
            sequence.markWrapped();
        }
        if (compressorSteps != null) {
            compressorSteps[0].onStep(nTokens);
        }
    }

    /**
     * Writes {@code cell_idx}, {@code q_pos} and {@code rope_pos} for the packed query rows of a prefill
     * context-parallel step: {@code positions[t]} is the true position of query row {@code t} in this rank's
     * DualChunkSwap order, and the step's rows inside this rank's context block — the positions in
     * {@code [blockPosFrom, blockPosTo)} — are stored in their rows of every group's {@code kv_pos}, because
     * they are the rows whose K/V this rank's cache receives (its own and the exchanged ones). All arguments are
     * validated against every group before anything changes.
     */
    public void uploadPacked(int[] packedPos, int blockPosFrom, int blockPosTo) {
        checkOpen();
        int n = packedPos == null ? 0 : packedPos.length;
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("packed step of " + n + " rows outside [1, " + ubatch + "]");
        }
        if (blockPosFrom < 0 || blockPosTo < blockPosFrom) {
            throw new IllegalArgumentException("a packed step's context rows need 0 <= from <= to, got ["
                    + blockPosFrom + ", " + blockPosTo + ")");
        }
        int pos0 = packedPos[0];
        if (pos0 < 0 || (long) pos0 + n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("packed step positions starting at " + pos0 + " are out of range");
        }
        for (GroupPositions p : positions) {
            p.checkStep(pos0, n);
        }
        boolean overwrote = false;
        for (int g = 0; g < positions.length; g++) {
            GroupPositions p = positions[g];
            if (blockPosTo > blockPosFrom) {
                overwrote |= p.store(blockPosFrom, blockPosTo - blockPosFrom);
                writeMirror(g, p.spanFrom(blockPosFrom, blockPosTo - blockPosFrom),
                        p.spanTo(blockPosFrom, blockPosTo - blockPosFrom));
            }
            for (int t = 0; t < n; t++) {
                int pos = packedPos[t];
                // only the rows of this rank's context block carry a cache cell; the others carry a -1 (skipped)
                ints[t] = p.group().holds(pos) ? p.group().rowOf(pos) : -1;
            }
            writeInts(cellIdx[g], 0, ints, n);
        }
        System.arraycopy(packedPos, 0, ints, 0, n);
        writeInts(qPos, 0, ints, n);
        textPositions.fill(packedPos, 0, n, ints);
        writeInts(ropePos, 0, ints, MropeTextPositions.SECTIONS * n);
        this.pos0 = pos0;
        this.nTokens = n;
        if (overwrote) {
            sequence.markWrapped();
        }
        if (compressorSteps != null && blockPosTo > blockPosFrom) {
            // a packed step whose rows all lie outside this rank's context block stores nothing and leaves
            // the compressor record untouched
            compressorSteps[0].onStep(blockPosTo - blockPosFrom);
        }
    }

    /**
     * Rewrites {@code cell_idx}, {@code q_pos} and {@code rope_pos} for the packed query rows of a prefill
     * context-parallel batched step: {@code packedRows[t]} is the step's global row whose query row {@code t} of
     * this rank's DualChunkSwap order carries, and the cell of that row is taken from the current paged step's
     * translated cell indices of every group (a row whose K/V lands on another rank's page carries {@code -1},
     * which the store kernels skip). The page tables, per-sequence slot counts, query positions and
     * {@code kv_pos} rows of the translated step stay whole: the attention launches read them, not the packed
     * buffers.
     *
     * @throws IllegalStateException    when no paged step is current
     * @throws IllegalArgumentException when {@code packedRows} does not name {@code [1, ubatch]} step rows
     */
    public void packPagedStepRows(int[] packedRows) {
        checkOpen();
        PagedStep step = paged != null ? paged.current : null;
        if (step == null) {
            throw new IllegalStateException("the packed step rows rewrite needs the current paged step");
        }
        int n = packedRows == null ? 0 : packedRows.length;
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("packed step of " + n + " rows outside [1, " + ubatch + "]");
        }
        for (int t = 0; t < n; t++) {
            if (packedRows[t] < 0 || packedRows[t] >= step.nTokens()) {
                throw new IndexOutOfBoundsException("packed row " + t + " names step row " + packedRows[t]
                        + " outside [0, " + step.nTokens() + ")");
            }
        }
        for (int g = 0; g < positions.length; g++) {
            int[] cells = step.cellIdx(g);
            for (int t = 0; t < n; t++) {
                ints[t] = cells[packedRows[t]];
            }
            writeInts(cellIdx[g], 0, ints, n);
        }
        for (int t = 0; t < n; t++) {
            ints[t] = rowPositions[packedRows[t]];
            packedTruePos[t] = ints[t];
        }
        writeInts(qPos, 0, ints, n);
        textPositions.fill(packedTruePos, 0, n, ints);
        writeInts(ropePos, 0, ints, MropeTextPositions.SECTIONS * n);
        this.pos0 = packedTruePos[0];
        this.nTokens = n;
    }

    /**
     * Reports every uploaded step to the owning memory's compressor rollback records, whose snapshots each step
     * replaces: a contiguous step to sequence 0's, each segment of a paged step to its sequence's.
     *
     * @param rollbacks one record per sequence the memory holds
     */
    void trackCompressorSteps(CompressorRollback[] rollbacks) {
        this.compressorSteps = Objects.requireNonNull(rollbacks, "rollbacks");
    }

    // ------------------------------------------------------------------ paged steps

    @Override
    public PagedStep currentPagedStep() {
        checkOpen();
        return paged != null ? paged.current() : null;
    }

    /**
     * Uploads a paged step: the page-bookkeeping mirror changes queued since the last step (removals, trims,
     * partial copies), the step's {@code kv_pos} row writes, per-token {@code cell_idx}, the per-group page tables
     * of the step's segments in launch order, the per-sequence slot counts and query positions, and the per-row
     * query and RoPE positions of {@code rows}.
     *
     * @param rows         position of each ubatch row, in ubatch order (the step layout's row order)
     * @param ropeSections section-major M-RoPE positions ({@code [section][row]}), or null to derive them per
     *                     segment from its first position
     * @throws IllegalArgumentException when the step belongs to another memory or {@code rows} does not cover it
     * @throws IllegalStateException    when this memory is not paged
     */
    public void beginPagedStep(PagedStep step, int[] rows, int[][] ropeSections) {
        beginPagedStep(step, rows, ropeSections, false);
    }

    /**
     * {@link #beginPagedStep(PagedStep, int[], int[][])} with the step's mask rule: {@code nonCausal} rows attend
     * each other without the causal mask ({@link #stepNonCausal()}) until {@link #endPagedStep}.
     */
    public void beginPagedStep(PagedStep step, int[] rows, int[][] ropeSections, boolean nonCausal) {
        checkOpen();
        if (paged == null) {
            throw new IllegalStateException("step inputs of a contiguous memory hold no paged step");
        }
        if (rows == null || rows.length != step.nTokens()) {
            throw new IllegalArgumentException("a paged step carries the position of each of its " + step.nTokens()
                    + " rows");
        }
        paged.begin(step, rows, ropeSections);
        System.arraycopy(rows, 0, rowPositions, 0, rows.length);
        if (compressorSteps != null) {
            for (int seg = 0; seg < step.nSeg(); seg++) {
                compressorSteps[step.segSeq(seg)].onStep(step.segLen(seg));
            }
        }
        pos0 = rows.length > 0 ? rows[0] : 0;
        nTokens = rows.length;
        this.nonCausal = nonCausal;
    }

    /** Clears the paged step begun by the last {@link #beginPagedStep}. */
    public void endPagedStep() {
        checkOpen();
        nonCausal = false;
        if (paged != null) {
            paged.end();
        }
    }

    @Override
    public int rowPosition(int row) {
        checkOpen();
        if (row < 0 || row >= nTokens) {
            throw new IndexOutOfBoundsException("row " + row + " is outside the step's " + nTokens + " rows");
        }
        return paged != null && paged.current() != null ? rowPositions[row] : pos0 + row;
    }

    @Override
    public boolean stepNonCausal() {
        checkOpen();
        return nonCausal && paged != null && paged.current() != null;
    }

    @Override
    public GpuBuffer pageTable(int group) {
        checkOpen();
        return paged != null ? paged.pageTable(group) : null;
    }

    @Override
    public GpuBuffer seqKvLen(int group) {
        checkOpen();
        return paged != null ? paged.seqKvLen(group) : null;
    }

    @Override
    public GpuBuffer qPosSeq() {
        checkOpen();
        return paged != null ? paged.qPosSeq() : null;
    }

    @Override
    public GpuBuffer segmentPositions() {
        checkOpen();
        return paged != null ? paged.segPos() : null;
    }

    @Override
    public int pageShift() {
        checkOpen();
        return paged != null ? paged.pageShift() : -1;
    }

    @Override
    public int maxPages() {
        checkOpen();
        return paged != null ? paged.maxPages() : 0;
    }

    @Override
    public int pagedMaxSequences() {
        checkOpen();
        return paged != null ? paged.maxSequences() : 1;
    }

    @Override
    public void uploadSegmentPositions(int pos0, int len) {
        checkOpen();
        if (paged == null) {
            throw new UnsupportedOperationException("the contiguous step inputs hold no segment positions");
        }
        paged.uploadSegmentPositions(pos0, len);
    }

    @Override
    public GpuBuffer treeMask() {
        checkOpen();
        return treeStep ? treeMask : null;
    }

    @Override
    public void beginTreeStep(long[] mask, int[][] ropeSections) {
        checkOpen();
        if (mask == null || mask.length != nTokens) {
            throw new IllegalArgumentException("a tree mask holds one row of the current step, got "
                    + (mask == null ? "none" : mask.length + " rows") + " for the step's " + nTokens);
        }
        System.arraycopy(mask, 0, treeMaskRows, 0, nTokens);
        for (int r = 0; r < nTokens; r++) {
            stagingView.putLong(r * Long.BYTES, mask[r]);
        }
        queue.enqueueWriteBuffer(treeMask, 0, (long) nTokens * Long.BYTES, staging);
        if (ropeSections != null) {
            boolean multi = ropeSections.length == MropeTextPositions.SECTIONS;
            if (!multi && ropeSections.length != 1) {
                throw new IllegalArgumentException("a tree step takes one position section or the M-RoPE "
                        + MropeTextPositions.SECTIONS + " of " + nTokens + " rows, got " + ropeSections.length);
            }
            for (int s = 0; s < ropeSections.length; s++) {
                if (ropeSections[s].length != nTokens) {
                    throw new IllegalArgumentException("tree step rope section " + s + " holds "
                            + ropeSections[s].length + " positions for the step's " + nTokens + " rows");
                }
            }
            // the query positions read the temporal section; the rope kernels read every section of the step's rows
            System.arraycopy(ropeSections[0], 0, treePositions, 0, nTokens);
            writeInts(qPos, 0, treePositions, nTokens);
            if (multi) {
                for (int s = 0; s < MropeTextPositions.SECTIONS; s++) {
                    writeInts(ropePos, (long) s * nTokens, ropeSections[s], nTokens);
                }
            } else {
                writeInts(ropePos, 0, treePositions, nTokens);
            }
        } else {
            for (int r = 0; r < nTokens; r++) {
                treePositions[r] = rowPosition(r);
            }
        }
        treeStep = true;
        // the launch-local tree reads of the step's first segment (the whole single-ubatch verify step) are the
        // step's own rows, uploaded here before the step's trunk so the captured graph holds kernel dispatches
        // only; the per-segment uploads of the attention skip the redundant write
        if (paged != null) {
            paged.uploadSegmentTreePositions(0, nTokens);
            treeSegmentUploaded = true;
        }
    }

    @Override
    public void endTreeStep() {
        checkOpen();
        treeStep = false;
        treeSegmentUploaded = false;
    }

    @Override
    public void uploadSegmentTreePositions(int segStart, int len) {
        checkOpen();
        if (paged == null) {
            throw new UnsupportedOperationException("the contiguous step inputs hold no segment positions");
        }
        if (treeSegmentUploaded && segStart == 0 && len == nTokens) {
            return;
        }
        paged.uploadSegmentTreePositions(segStart, len);
    }

    /**
     * The segment's slice of the step's mask, shifted to launch-local bits (the paged prefill kernel indexes
     * {@code tree_mask} without a row offset), written over the buffer's first {@code len} rows.
     */
    @Override
    public void uploadTreeMaskSegment(int segStart, int len) {
        checkOpen();
        if (!treeStep) {
            throw new IllegalStateException("no tree step runs over the current step inputs");
        }
        if (len < 1 || segStart + len > treeMaskRows.length) {
            throw new IllegalArgumentException("segment of " + len + " rows at " + segStart + " outside the step's "
                    + treeMaskRows.length + " rows");
        }
        if (treeSegmentUploaded && segStart == 0 && len == nTokens) {
            return;
        }
        for (int t = 0; t < len; t++) {
            stagingView.putLong(t * Long.BYTES, treeMaskRows[segStart + t] >>> segStart);
        }
        queue.enqueueWriteBuffer(treeMask, 0, (long) len * Long.BYTES, staging);
    }

    @Override
    public void reserveSequenceAddressing(int group) {
        checkOpen();
        checkGroup(group);
        if (paged != null) {
            paged.reserve(group);
        }
    }

    @Override
    public GpuBuffer sequenceKvPos(int group, int seq) {
        checkOpen();
        checkGroup(group);
        return paged != null ? paged.sequenceKvPos(group, seq) : null;
    }

    // ------------------------------------------------------------------ bookkeeping used by the memory

    /** Position stored in {@code row} of KV group {@code group} (host mirror of {@code kv_pos}), -1 when empty. */
    public int positionOfRow(int group, int row) {
        return positions[checkGroup(group)].position(row);
    }

    /** Host mirror of a group's {@code kv_pos}, changed only through this class and its owning memory. */
    GroupPositions positions(int group) {
        return positions[checkGroup(group)];
    }

    /** Host mirrors of every group, indexed by group (a new array over the live mirrors). */
    GroupPositions[] allPositions() {
        return positions.clone();
    }

    /** {@code int[identityLength()]} holding {@code i} at index {@code i}. */
    public GpuBuffer identityRows() {
        checkOpen();
        return identityRows;
    }

    public int identityLength() {
        return identityLength;
    }

    /** Maximum tokens per step. */
    public int ubatch() {
        return ubatch;
    }

    /**
     * Per-row position deltas of the last context shift of a group ({@code int[rows]}, indexed by the row after
     * the shift; 0 for rows the shift did not move).
     *
     * @throws IllegalStateException when no context shift has happened since allocation
     */
    public GpuBuffer shiftDelta(int group) {
        checkOpen();
        GpuBuffer b = shiftDelta[checkGroup(group)];
        if (b == null) {
            throw new IllegalStateException("KV group " + group + " has not been shifted");
        }
        return b;
    }

    /** Empties every group: mirrors and device {@code kv_pos} to -1. */
    void clear() {
        checkOpen();
        for (int g = 0; g < positions.length; g++) {
            positions[g].clear();
            queue.enqueueFillBuffer(kvPos[g], 0, kvPos[g].sizeInBytes(), -1);
        }
        if (paged != null) {
            paged.current = null;
            paged.pages.clearReports();
        }
        pos0 = 0;
        nTokens = 0;
        nonCausal = false;
    }

    /** Marks the rows of positions {@code [pos0, pos0 + n)} empty in every group and rewrites the changed rows. */
    void drop(int pos0, int n) {
        checkOpen();
        if (n <= 0) {
            return;
        }
        for (int g = 0; g < positions.length; g++) {
            GroupPositions p = positions[g];
            int done = 0;
            while (done < n) {
                int chunk = p.group().ring() ? Math.min(n - done, p.rows()) : n - done;
                int from = p.spanFrom(pos0 + done, chunk);
                int to = p.spanTo(pos0 + done, chunk);
                p.drop(pos0 + done, chunk);
                writeMirror(g, from, to);
                done += chunk;
            }
        }
    }

    /**
     * Applies the queued page-bookkeeping mirror changes, consumes them, and returns the RoPE deltas the relative
     * reports carried: {@code out[group][row]} accumulates {@code RowReport#delta} over every named row
     * ({@code LLM_KV_RopeShift_*}); a group whose reports carried no shift holds a null row. A context shift calls
     * this at once, so its layers re-rotate before the move's row copies run; the steps' own reports are absolute
     * and {@code begin} drains them with null deltas.
     *
     * @throws UnsupportedOperationException when the memory holds no paged sequences
     */
    int[][] drainPagedShiftDeltas() {
        checkOpen();
        if (paged == null) {
            throw new UnsupportedOperationException("the memory holds no paged sequences");
        }
        int[][] deltas = new int[positions.length][];
        paged.drainReports(deltas);
        return deltas;
    }

    /**
     * Allocates the {@code shift_delta} buffer of every group that has none yet, so a context shift can install its
     * result without allocating after the KV rows have moved.
     */
    void reserveShiftDeltas() {
        checkOpen();
        for (int g = 0; g < positions.length; g++) {
            if (shiftDelta[g] == null) {
                shiftDelta[g] = alloc((long) positions[g].rows() * Integer.BYTES);
            }
        }
    }

    /**
     * Installs the result of a context shift in a group: the mirror takes the shifted positions, and the device
     * {@code kv_pos} and {@code shift_delta} buffers are rewritten whole.
     *
     * @throws IllegalStateException when {@link #reserveShiftDeltas()} has not run
     */
    void applyShift(int group, KvShift shift) {
        checkOpen();
        GroupPositions p = positions[checkGroup(group)];
        if (shiftDelta[group] == null) {
            throw new IllegalStateException("shift_delta of KV group " + group + " was not reserved");
        }
        p.apply(shift);
        writeMirror(group, 0, p.rows());
        for (int r = 0; r < p.rows(); r++) {
            ints[r] = shift.delta(r);
        }
        writeInts(shiftDelta[group], 0, ints, p.rows());
    }

    /**
     * Installs the per-row RoPE deltas of a paged context shift in a group: {@code delta[row]} for every row of the
     * group (0 for a row the shift kept in place), read whole by {@code LLM_KV_RopeShift_*}.
     *
     * @throws IllegalStateException    when {@link #reserveShiftDeltas()} has not run
     * @throws IllegalArgumentException when {@code delta} does not hold one entry per row
     */
    void installShiftDeltas(int group, int[] delta) {
        checkOpen();
        GroupPositions p = positions[checkGroup(group)];
        if (shiftDelta[group] == null) {
            throw new IllegalStateException("shift_delta of KV group " + group + " was not reserved");
        }
        if (delta.length != p.rows()) {
            throw new IllegalArgumentException("shift deltas of " + delta.length + " rows for the " + p.rows()
                    + " rows of KV group " + group);
        }
        writeInts(shiftDelta[group], 0, delta, delta.length);
    }

    /**
     * Installs {@code mirror} as the positions of a group: the host mirror takes every row's position and the
     * device {@code kv_pos} is rewritten whole (a session restore).
     *
     * @throws IllegalArgumentException when the mirror's length differs from the group's rows
     */
    void installPositions(int group, int[] mirror) {
        checkOpen();
        GroupPositions p = positions[checkGroup(group)];
        if (mirror.length != p.rows()) {
            throw new IllegalArgumentException("a mirror of " + mirror.length + " rows for KV group " + group + " of "
                    + p.rows() + " rows");
        }
        for (int r = 0; r < mirror.length; r++) {
            p.set(r, mirror[r]);
        }
        writeMirror(group, 0, p.rows());
    }

    /**
     * Rewrites every group's device {@code kv_pos} so only rows whose stored position lies in {@code [start, end)}
     * stay visible; every other row becomes -1. The host mirrors keep the sequence's true positions, so rollback,
     * reset and the context shift still see them; context-parallel attention calls this after
     * {@link #upload(int, int)} to restrict a rank's attention to its own context block.
     *
     * @throws IllegalArgumentException when {@code start} is negative or {@code end} precedes it
     */
    public void restrictKvPos(int start, int end) {
        checkOpen();
        if (start < 0 || end < start) {
            throw new IllegalArgumentException("a visible context block needs 0 <= start <= end, got ["
                    + start + ", " + end + ")");
        }
        for (int g = 0; g < positions.length; g++) {
            GroupPositions p = positions[g];
            int rows = p.rows();
            for (int r = 0; r < rows; r++) {
                int pos = p.position(r);
                ints[r] = pos >= start && pos < end ? pos : -1;
            }
            writeInts(kvPos[g], 0, ints, rows);
        }
    }

    private void writeMirror(int group, int from, int to) {
        if (to <= from) {
            return;
        }
        GroupPositions p = positions[group];
        for (int r = from; r < to; r++) {
            ints[r - from] = p.position(r);
        }
        writeInts(kvPos[group], from, ints, to - from);
    }

    private void writeInts(GpuBuffer dst, long elementOffset, int[] src, int n) {
        writeInts(dst, elementOffset, src, 0, n);
    }

    private void writeInts(GpuBuffer dst, long elementOffset, int[] src, int srcFrom, int n) {
        for (int i = 0; i < n; i++) {
            stagingView.putInt(i * Integer.BYTES, src[srcFrom + i]);
        }
        GpuEvent e = queue.enqueueWriteBuffer(dst, elementOffset * Integer.BYTES, (long) n * Integer.BYTES, staging);
        if (e != null) {
            e.close();
        }
    }

    private int checkGroup(int group) {
        if (group < 0 || group >= positions.length) {
            throw new IndexOutOfBoundsException("KV group " + group + " outside [0, " + positions.length + ")");
        }
        return group;
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("step inputs are closed");
        }
    }

    /** Releases every buffer. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        for (int g = 0; g < positions.length; g++) {
            kvPos[g].close();
            cellIdx[g].close();
            if (shiftDelta[g] != null) {
                shiftDelta[g].close();
            }
        }
        qPos.close();
        ropePos.close();
        stateRows.close();
        identityRows.close();
        treeMask.close();
        if (paged != null) {
            paged.close();
        }
    }

    // ================================================================== host bookkeeping

    /**
     * Device page tables and per-sequence addressing of a paged memory: per KV group the compacted page table of
     * the current step's segments (launch order, the row the {@code LLM_FlashDecodePaged_*} fold and the
     * {@code LLM_FlashAttnPaged_*} {@code pt_off} address), the per-sequence slot counts, and the per-launch query
     * positions. {@link #begin} uploads a step whole; the attention mixers read the buffers through
     * {@link GpuStepInputs} of the owning memory. Not thread-safe.
     */
    final class Paged {

        private final ComputeQueue queue;
        private final PagedSequences pages;
        private final int groups;
        private final int maxSequences;
        private final int maxPages;
        private final int pageShift;
        private final GpuBuffer[] pageTable;   // [groups] int[maxSequences * maxPages]
        private final GpuBuffer[] seqKvLen;    // [groups] int[maxSequences]
        private final GpuBuffer qPosSeq;       // int[maxSequences], per-launch query position of sequence s
        private final GpuBuffer segPos;        // int[ubatch], the positions of one prefill segment
        /** Cells of one sequence ({@code PagedStep.seqCells}), {@code maxPages * pageSize} long. */
        private final int[] seqCells;
        private final GpuBuffer[][] seqKvPos;  // [groups][maxSequences] int[rows], reserved per group
        private final boolean[][] kvPosBuilt;
        private PagedStep current;

        private Paged(ComputeBackend backend, ComputeQueue queue, int groups, int ubatch, PagedSequences pages,
                      List<GpuBuffer> owned) {
            this.queue = queue;
            this.pages = pages;
            this.groups = groups;
            this.maxSequences = pages.maxSequences();
            this.maxPages = pages.maxPages();
            this.pageShift = pageShiftOf(pages.pageSize());
            this.pageTable = new GpuBuffer[groups];
            this.seqKvLen = new GpuBuffer[groups];
            for (int g = 0; g < groups; g++) {
                pageTable[g] = track(owned, alloc((long) maxSequences * maxPages * Integer.BYTES));
                queue.enqueueFillBuffer(pageTable[g], 0, pageTable[g].sizeInBytes(), -1);
                seqKvLen[g] = track(owned, alloc((long) maxSequences * Integer.BYTES));
                queue.enqueueFillBuffer(seqKvLen[g], 0, seqKvLen[g].sizeInBytes(), 0);
            }
            this.qPosSeq = track(owned, alloc((long) maxSequences * Integer.BYTES));
            queue.enqueueFillBuffer(qPosSeq, 0, qPosSeq.sizeInBytes(), 0);
            this.segPos = track(owned, alloc((long) ubatch * Integer.BYTES));
            queue.enqueueFillBuffer(segPos, 0, segPos.sizeInBytes(), 0);
            this.seqCells = new int[maxPages * pages.pageSize()];
            this.seqKvPos = new GpuBuffer[groups][];
            this.kvPosBuilt = new boolean[groups][maxSequences];
        }

        /** Allocates the per-sequence {@code kv_pos} buffers of {@code group}, once. */
        void reserve(int group) {
            checkGroup(group);
            if (seqKvPos[group] != null) {
                return;
            }
            GpuBuffer[] kv = new GpuBuffer[maxSequences];
            try {
                for (int s = 0; s < maxSequences; s++) {
                    kv[s] = alloc((long) positions[group].rows() * Integer.BYTES);
                    queue.enqueueFillBuffer(kv[s], 0, kv[s].sizeInBytes(), -1);
                }
            } catch (RuntimeException e) {
                closeAll(kv);
                throw e;
            }
            seqKvPos[group] = kv;
        }

        /** {@code kv_pos} of {@code group} restricted to sequence {@code seq}'s cells (see the interface). */
        GpuBuffer sequenceKvPos(int group, int seq) {
            checkSequence(group, seq);
            if (!kvPosBuilt[group][seq]) {
                GroupPositions mirror = positions[group];
                int rows = mirror.rows();
                Arrays.fill(ints, 0, rows, -1);
                int slots = current.seqCells(group, seq, seqCells);
                for (int j = 0; j < slots; j++) {
                    int cell = seqCells[j];
                    ints[cell] = mirror.position(cell);
                }
                writeInts(seqKvPos[group][seq], 0, ints, rows);
                kvPosBuilt[group][seq] = true;
            }
            return seqKvPos[group][seq];
        }

        private void checkSequence(int group, int seq) {
            checkGroup(group);
            if (seq < 0 || seq >= maxSequences) {
                throw new IndexOutOfBoundsException("sequence " + seq + " outside [0, " + maxSequences + ")");
            }
            if (seqKvPos[group] == null) {
                throw new IllegalStateException("the per-sequence addressing of KV group " + group
                        + " was not reserved (reserveSequenceAddressing)");
            }
            if (current == null) {
                throw new IllegalStateException("no paged step is current");
            }
        }

        private void invalidateSequences() {
            for (int g = 0; g < groups; g++) {
                Arrays.fill(kvPosBuilt[g], false);
            }
        }

        private int pageShiftOf(int pageSize) {
            if (Integer.bitCount(pageSize) != 1) {
                throw new IllegalArgumentException("page size " + pageSize + " is not a power of two");
            }
            return Integer.numberOfTrailingZeros(pageSize);
        }

        /** Page table of group {@code group} (the step's segments in launch order). */
        GpuBuffer pageTable(int group) {
            checkGroup(group);
            return pageTable[group];
        }

        /** Slots each launch sequence holds in group {@code group} ({@code seq_kv_len}). */
        GpuBuffer seqKvLen(int group) {
            checkGroup(group);
            return seqKvLen[group];
        }

        /** Query position of launch sequence {@code s} ({@code q_pos_seq}). */
        GpuBuffer qPosSeq() {
            return qPosSeq;
        }

        /** Positions of the segment whose launch the calling mixer enqueues next. */
        GpuBuffer segPos() {
            return segPos;
        }

        int maxSequences() {
            return maxSequences;
        }

        int maxPages() {
            return maxPages;
        }

        int pageShift() {
            return pageShift;
        }

        PagedStep current() {
            return current;
        }

        /**
         * Applies the queued page-bookkeeping mirror changes and consumes them, accumulating the relative reports'
         * RoPE deltas into {@code deltasOut[group][row]} ({@code LLM_KV_RopeShift_*}); null collects no deltas.
         */
        void drainReports(int[][] deltasOut) {
            for (RowReport r : pages.pendingReports()) {
                int[] rows = r.rows();
                int[] values = r.values();
                GroupPositions mirror = positions[r.group()];
                for (int i = 0; i < rows.length; i++) {
                    mirror.set(rows[i], values[i]);
                }
                if (deltasOut != null && !r.absolute()) {
                    if (deltasOut[r.group()] == null) {
                        deltasOut[r.group()] = new int[mirror.rows()];
                    }
                    for (int i = 0; i < rows.length; i++) {
                        deltasOut[r.group()][rows[i]] += r.rowDelta(i);
                    }
                }
            }
            pages.clearReports();
        }

        /** Uploads {@code step}: mirrors, page tables, cell indices, per-sequence addressing and positions. */
        void begin(PagedStep step, int[] rows, int[][] ropeSections) {
            checkBelongs(step);
            drainReports(null);
            for (int g = 0; g < groups; g++) {
                applyKvPos(g, step);
                writeMirrorWhole(g);
                writeInts(cellIdx[g], 0, step.cellIdx(g), step.nTokens());
                writePageTable(g, step);
                writeInts(seqKvLen[g], 0, launchSlots(g, step), segments(step));
            }
            writeInts(qPosSeq, 0, launchPositions(step), segments(step));
            writeInts(qPos, 0, rows, rows.length);
            writeRope(step, rows, ropeSections);
            invalidateSequences();
            current = step;
        }

        /** Clears the current step (the buffers stay; the next step rewrites them). */
        void end() {
            invalidateSequences();
            current = null;
        }

        /** Writes the positions of one multi-token segment for the next paged prefill launch. */
        void uploadSegmentPositions(int pos0, int len) {
            if (len < 1 || len > ubatch) {
                throw new IllegalArgumentException("segment of " + len + " rows outside [1, " + ubatch + "]");
            }
            for (int t = 0; t < len; t++) {
                ints[t] = pos0 + t;
            }
            writeInts(segPos, 0, ints, len);
        }

        /**
         * Writes one tree step segment's depth positions ({@code beginTreeStep} laid out the step's rows) for the
         * next paged prefill launch, which reads them launch-local; a plain segment's rows take
         * {@link #uploadSegmentPositions}.
         */
        void uploadSegmentTreePositions(int segStart, int len) {
            if (len < 1 || len > ubatch) {
                throw new IllegalArgumentException("segment of " + len + " rows outside [1, " + ubatch + "]");
            }
            if (!treeStep) {
                throw new IllegalStateException("no tree step runs over the current paged step");
            }
            if (segStart < 0 || segStart + len > treePositions.length) {
                throw new IllegalArgumentException("segment of " + len + " rows at " + segStart + " outside the"
                        + " step's " + treePositions.length + " rows");
            }
            writeInts(segPos, 0, treePositions, segStart, len);
        }

        /** Applies the step's {@code kv_pos} row writes to the host mirror of {@code group}. */
        private void applyKvPos(int group, PagedStep step) {
            int[] rows = step.kvPosRows(group);
            int[] values = step.kvPosValues(group);
            GroupPositions mirror = positions[group];
            for (int i = 0; i < rows.length; i++) {
                mirror.set(rows[i], values[i]);
            }
        }

        /** One whole-buffer {@code kv_pos} write of {@code group}, carrying the mirror's every change. */
        private void writeMirrorWhole(int group) {
            GroupPositions mirror = positions[group];
            int rows = mirror.rows();
            for (int r = 0; r < rows; r++) {
                ints[r] = mirror.position(r);
            }
            writeInts(kvPos[group], 0, ints, rows);
        }

        /** The step's page table of {@code group} compacted to launch order (segment {@code s} at row {@code s}). */
        private void writePageTable(int group, PagedStep step) {
            int segs = step.nSeg();
            int[] table = step.pageTable(group);
            for (int s = 0; s < segs; s++) {
                System.arraycopy(table, step.segSeq(s) * maxPages, ints, 0, maxPages);
                writeInts(pageTable[group], (long) s * maxPages, ints, maxPages);
            }
        }

        /** Slots each launch sequence holds in {@code group} after the step. */
        private int[] launchSlots(int group, PagedStep step) {
            int segs = segments(step);
            int[] slots = new int[segs];
            for (int s = 0; s < segs; s++) {
                slots[s] = step.seqSlots(group)[step.segSeq(s)];
            }
            return slots;
        }

        /** Query position of each launch sequence (its segment's first position). */
        private int[] launchPositions(PagedStep step) {
            int segs = segments(step);
            int[] pos = new int[segs];
            for (int s = 0; s < segs; s++) {
                pos[s] = step.segPos0(s);
            }
            return pos;
        }

        /**
         * The step's per-row RoPE sections: the batch's explicit ones (one section for a single-section rope, the
         * M-RoPE four otherwise), or the pattern derived per segment.
         */
        private void writeRope(PagedStep step, int[] rows, int[][] ropeSections) {
            int n = rows.length;
            if (ropeSections != null) {
                if (ropeSections.length == 1) {
                    // a single-section rope (positionsPerToken 1): the batch's one position per row — a tree
                    // verify's depth positions — which the NORM/NEOX kernels read from the buffer's first section
                    if (ropeSections[0].length != n) {
                        throw new IllegalArgumentException("rope section 0 holds " + ropeSections[0].length
                                + " positions for the step's " + n + " rows");
                    }
                    writeInts(ropePos, 0, ropeSections[0], n);
                    return;
                }
                if (ropeSections.length != MropeTextPositions.SECTIONS) {
                    throw new IllegalArgumentException("rope sections " + ropeSections.length + " outside the "
                            + MropeTextPositions.SECTIONS + " the position buffer holds");
                }
                for (int s = 0; s < MropeTextPositions.SECTIONS; s++) {
                    System.arraycopy(ropeSections[s], 0, ints, s * n, n);
                }
                writeInts(ropePos, 0, ints, MropeTextPositions.SECTIONS * n);
                return;
            }
            int[] section = new int[ubatch];
            for (int seg = 0; seg < step.nSeg(); seg++) {
                int len = step.segLen(seg);
                int start = step.segStart(seg);
                textPositions.fill(step.segPos0(seg), len, ints);
                for (int s = 0; s < MropeTextPositions.SECTIONS; s++) {
                    for (int t = 0; t < len; t++) {
                        section[t] = ints[s * len + t];
                    }
                    writeInts(ropePos, (long) s * n + start, section, len);
                }
            }
        }

        private int segments(PagedStep step) {
            return step.nSeg();
        }

        private void checkBelongs(PagedStep step) {
            if (step.maxSequences() != maxSequences || step.groups() != groups || step.maxPages() != maxPages
                    || step.nPages() != pages.pageCount() || step.pageSize() != pages.pageSize()) {
                throw new IllegalArgumentException("the step does not belong to this paged memory");
            }
        }

        private void checkGroup(int group) {
            if (group < 0 || group >= groups) {
                throw new IndexOutOfBoundsException("KV group " + group + " outside [0, " + groups + ")");
            }
        }

        /** Releases the page-table and per-sequence buffers. Idempotent with the owning step inputs. */
        void close() {
            for (int g = 0; g < groups; g++) {
                pageTable[g].close();
                seqKvLen[g].close();
            }
            qPosSeq.close();
            segPos.close();
            for (int g = 0; g < groups; g++) {
                closeAll(seqKvPos[g]);
            }
        }
    }

    private static void closeAll(GpuBuffer[] buffers) {
        if (buffers == null) {
            return;
        }
        for (GpuBuffer b : buffers) {
            if (b != null) {
                b.close();
            }
        }
    }

    /**
     * Host record of one KV group's {@code kv_pos}: the position stored in each row, -1 when empty. Every stored
     * position {@code p} sits in row {@code group().rowOf(p)}; the operations below keep that invariant. Shared by
     * the device step inputs and the heap memory. Not thread-safe.
     */
    public static final class GroupPositions {

        private final MemoryPlan.KvGroup group;
        private int[] pos;

        public GroupPositions(MemoryPlan.KvGroup group) {
            this.group = Objects.requireNonNull(group, "group");
            this.pos = new int[group.rows()];
            Arrays.fill(pos, -1);
        }

        public MemoryPlan.KvGroup group() {
            return group;
        }

        public int rows() {
            return pos.length;
        }

        /** Position stored in {@code row}, -1 when empty. */
        public int position(int row) {
            if (row < 0 || row >= pos.length) {
                throw new IndexOutOfBoundsException("row " + row + " outside the KV group of " + pos.length + " rows");
            }
            return pos[row];
        }

        /** Copy of every row's position. */
        public int[] snapshot() {
            return pos.clone();
        }

        /**
         * Validates a step of {@code n} positions from {@code pos0}: every position must have a row, and a ring may
         * not hold fewer rows than the step (its tokens would overwrite each other).
         *
         * @throws IllegalArgumentException otherwise
         */
        public void checkStep(int pos0, int n) {
            if (n <= 0) {
                throw new IllegalArgumentException("step of " + n + " positions");
            }
            if (group.ring()) {
                if (n > pos.length) {
                    throw new IllegalArgumentException("step of " + n + " tokens exceeds the ring of " + pos.length
                            + " rows of " + group);
                }
                group.rowOf(pos0);
            } else if (group.sharded()) {
                /* The step may lie wholly or partly outside this rank's block; only the overlap is stored. */
                if (pos0 < 0) {
                    throw new IllegalArgumentException("step positions are negative: " + pos0);
                }
            } else {
                group.rowOf(pos0);
                group.rowOf(pos0 + n - 1);
            }
        }

        /**
         * Stores positions {@code [pos0, pos0 + n)} in their rows; on a context shard only the positions its block
         * holds are stored.
         *
         * @return true when a ring row holding a different stored position was overwritten
         */
        public boolean store(int pos0, int n) {
            checkStep(pos0, n);
            boolean overwrote = false;
            for (int t = 0; t < n; t++) {
                int p = pos0 + t;
                if (!group.holds(p)) {
                    continue;
                }
                int r = group.rowOf(p);
                if (pos[r] >= 0 && pos[r] != p) {
                    overwrote = true;
                }
                pos[r] = p;
            }
            return overwrote;
        }

        /** Empties the rows of positions {@code [pos0, pos0 + n)} that still hold those positions. */
        public void drop(int pos0, int n) {
            for (int t = 0; t < n; t++) {
                int p = pos0 + t;
                if (!group.holds(p)) {
                    continue;
                }
                int r = group.rowOf(p);
                if (pos[r] == p) {
                    pos[r] = -1;
                }
            }
        }

        /** Empties every row. */
        public void clear() {
            Arrays.fill(pos, -1);
        }

        /**
         * Sets the position stored in {@code row} directly (the paged step's partner of {@link #store}:
         * {@code PagedStep} addresses rows by page table, not by consecutive position, so {@code upload} cannot
         * derive the rows from the positions alone). Host mirror only; the caller uploads the range to the device
         * like every other mirror change.
         */
        public void set(int row, int position) {
            pos[row] = position;
        }

        /**
         * First row of the smallest contiguous row range covering positions {@code [pos0, pos0 + n)}: the run of
         * rows when it does not wrap, the whole group when it does.
         */
        public int spanFrom(int pos0, int n) {
            if (group.sharded()) {
                int lo = Math.max(pos0, group.rowBase());
                int hi = Math.min(pos0 + n, group.rowBase() + group.rows());
                return hi <= lo ? 0 : lo - group.rowBase();
            }
            int r0 = group.rowOf(pos0);
            return r0 + n <= pos.length ? r0 : 0;
        }

        /** End (exclusive) of the row range of {@link #spanFrom(int, int)}. */
        public int spanTo(int pos0, int n) {
            if (group.sharded()) {
                int lo = Math.max(pos0, group.rowBase());
                int hi = Math.min(pos0 + n, group.rowBase() + group.rows());
                return hi <= lo ? 0 : hi - group.rowBase();
            }
            int r0 = group.rowOf(pos0);
            return r0 + n <= pos.length ? r0 + n : pos.length;
        }

        /**
         * Plans the context shift of llama.cpp {@code seq_rm(nKeep, nKeep + nDiscard)} followed by
         * {@code seq_add(nKeep + nDiscard, inf, -nDiscard)} (src/llama-kv-cache.cpp:404, 592): positions below
         * {@code nKeep} stay, positions in {@code [nKeep, nKeep + nDiscard)} are removed, and every later position
         * {@code p} becomes {@code p - nDiscard}. Because rows are addressed by {@code rowOf}, a shifted position
         * moves to {@code rowOf(p - nDiscard)}; the returned plan lists those row moves and the per-row deltas.
         * Nothing changes until {@link #apply(KvShift)}.
         *
         * @throws IllegalArgumentException for {@code nKeep < 0} or {@code nDiscard <= 0}
         * @throws IllegalStateException    when two positions would land in one row (a broken invariant)
         */
        public KvShift planShift(int nKeep, int nDiscard) {
            if (nKeep < 0 || nDiscard <= 0) {
                throw new IllegalArgumentException("context shift needs nKeep >= 0 and nDiscard > 0, got " + nKeep
                        + "/" + nDiscard);
            }
            int rows = pos.length;
            int[] next = new int[rows];
            int[] delta = new int[rows];
            Arrays.fill(next, -1);
            List<int[]> moves = new ArrayList<>();
            long discardEnd = (long) nKeep + nDiscard;
            for (int r = 0; r < rows; r++) {
                int p = pos[r];
                if (p < 0 || (p >= nKeep && p < discardEnd)) {
                    continue;
                }
                int q = p < nKeep ? p : p - nDiscard;
                int d;
                if (p < nKeep) {
                    d = r;
                } else {
                    if (!group.holds(q)) {
                        // a context shard cannot carry a position across its block boundary
                        continue;
                    }
                    d = group.rowOf(q);
                }
                if (next[d] >= 0) {
                    throw new IllegalStateException("context shift maps positions " + next[d] + " and " + q
                            + " to row " + d + " of " + group);
                }
                next[d] = q;
                if (p >= nKeep) {
                    delta[d] = -nDiscard;
                    if (d != r) {
                        appendMove(moves, r, d);
                    }
                }
            }
            return new KvShift(next, delta, moves);
        }

        private static void appendMove(List<int[]> runs, int src, int dst) {
            if (!runs.isEmpty()) {
                int[] last = runs.get(runs.size() - 1);
                if (last[0] + last[2] == src && last[1] + last[2] == dst) {
                    last[2]++;
                    return;
                }
            }
            runs.add(new int[]{src, dst, 1});
        }

        /**
         * Whether the first query after a context shift, at position {@code nPastAfter}, still finds every kept
         * position (below {@code nKeep}) its window covers. Only a ring can fail: rows of kept positions may have been
         * overwritten, and the shift cannot bring them back.
         */
        public boolean keepsWindowAfterShift(int nKeep, int nPastAfter) {
            if (!group.ring() || nKeep <= 0) {
                return true;
            }
            for (int p = group.window().firstVisible(nPastAfter); p < nKeep; p++) {
                if (pos[group.rowOf(p)] != p) {
                    return false;
                }
            }
            return true;
        }

        /**
         * Whether the first query at {@code newPast} still finds every position of its window
         * ({@code [firstVisible(newPast), newPast)}) in its row once positions {@code [nPast, nPast + nWrite)} are
         * stored and every position from {@code newPast} on is dropped. With {@code nWrite == 0} this is the rule of
         * a plain rollback to {@code newPast}. Read from the mirror, so rows overwritten by later positions and rows
         * emptied by earlier drops both count as lost. Only a ring can fail: a full cache keeps every position below
         * {@code nPast} in its own row.
         *
         * @param nPast   positions stored now
         * @param nWrite  positions about to be stored from {@code nPast}
         * @param newPast positions left after the drop, at most {@code nPast + nWrite}
         * @throws IllegalArgumentException for negative arguments or {@code newPast > nPast + nWrite}
         */
        public boolean keepsWindowAfter(int nPast, int nWrite, int newPast) {
            if (nPast < 0 || nWrite < 0 || newPast < 0 || (long) newPast > (long) nPast + nWrite) {
                throw new IllegalArgumentException("window check needs 0 <= newPast <= nPast + nWrite, got nPast="
                        + nPast + " nWrite=" + nWrite + " newPast=" + newPast);
            }
            if (!group.ring() || newPast == 0) {
                return true;
            }
            int rows = pos.length;
            if (nWrite > rows) {
                return false;
            }
            long writeEnd = (long) nPast + nWrite;
            for (int p = group.window().firstVisible(newPast); p < newPast; p++) {
                if (p >= nPast) {
                    return true;
                }
                if (pos[group.rowOf(p)] != p) {
                    return false;
                }
                long firstOverwrite = p + (long) rows * ((nPast - p + rows - 1) / rows);
                if (firstOverwrite < writeEnd) {
                    return false;
                }
            }
            return true;
        }

        /** Installs the positions of a shift planned on this group. */
        public void apply(KvShift shift) {
            if (shift.rows() != pos.length) {
                throw new IllegalArgumentException("shift of " + shift.rows() + " rows applied to a group of "
                        + pos.length);
            }
            pos = shift.positionsAfter();
        }
    }

    /**
     * Sequence rules shared by the device and heap memories, decided from the plan, the sequence record and the host
     * mirrors of the KV groups: when the last positions can be dropped, when a snapshot can be taken or restored, and
     * when the context can shift.
     * Each check returns null when the operation is allowed, or the reason it is not.
     */
    public static final class SequenceRules {

        private SequenceRules() {
        }

        /** True when any layer has DSV4 compressed caches, whose compressor state accumulates across positions. */
        public static boolean hasCompressed(MemoryPlan plan) {
            return plan.hasCompressed();
        }

        /**
         * Why the last {@code nTokens > 0} positions cannot be dropped: too few stored, recurrent state (which rolls
         * back only through snapshot / restore), DSV4 compressor state without a usable rollback snapshot
         * ({@link CompressorRollback#blocker}), or {@link #ringBlocker}.
         *
         * @param groups     the host mirrors of every KV group of the memory
         * @param compressor the memory's compressor rollback record (consulted only when the plan has compressed
         *                   caches)
         */
        public static String rollbackBlocker(MemoryPlan plan, GroupPositions[] groups, SequenceState sequence,
                                             CompressorRollback compressor, int nTokens) {
            if (nTokens > sequence.nPast()) {
                return "only " + sequence.nPast() + " positions are stored";
            }
            if (plan.hasRecurrentState()) {
                return "the model keeps recurrent state; use snapshot() and restore()";
            }
            if (plan.hasCompressed()) {
                String why = compressor.blocker(nTokens);
                if (why != null) {
                    return why;
                }
            }
            return ringBlocker(groups, sequence, 0, nTokens);
        }

        /**
         * Why storing {@code nWrite} more positions and then dropping the last {@code nDrop} would leave a
         * sliding-window ring without keys the next query attends ({@link GroupPositions#keepsWindowAfter}); null
         * when every ring still holds the window of the query at {@code nPast + nWrite - nDrop}. This is the
         * condition llama.cpp checks through {@code seq_pos_min} before reusing a cached prefix of an SWA model
         * (tools/server/server-context.cpp:4061-4070): a ring row overwritten by a later position, or emptied by an
         * earlier drop, cannot be brought back.
         *
         * @throws IllegalArgumentException for negative counts or {@code nDrop > nPast + nWrite}
         */
        public static String ringBlocker(GroupPositions[] groups, SequenceState sequence, int nWrite, int nDrop) {
            int nPast = sequence.nPast();
            if (nWrite < 0 || nDrop < 0 || (long) nDrop > (long) nPast + nWrite) {
                throw new IllegalArgumentException("cannot store " + nWrite + " and drop " + nDrop + " of " + nPast
                        + " stored positions");
            }
            int newPast = (int) ((long) nPast + nWrite - nDrop);
            for (GroupPositions g : groups) {
                if (!g.keepsWindowAfter(nPast, nWrite, newPast)) {
                    return "the sliding-window ring of " + g.group() + " no longer holds every key the query at "
                            + "position " + newPast + " attends";
                }
            }
            return null;
        }

        /**
         * Why a snapshot cannot be taken: no recurrent state, DSV4 compressor state the snapshot would miss, or a
         * plan that reserved no shadow buffers ({@link MemoryPlan#snapshotsReserved()}).
         */
        public static String snapshotBlocker(MemoryPlan plan) {
            if (!plan.hasRecurrentState()) {
                return "memory has no recurrent state to snapshot";
            }
            if (plan.hasCompressed()) {
                return "memory with DSV4 compressed caches cannot be snapshot";
            }
            if (!plan.snapshotsReserved()) {
                return "the memory was planned without recurrent-state snapshot buffers; set "
                        + MemoryConfig.PROP_STATE_SNAPSHOTS + "=true";
            }
            return null;
        }

        /**
         * Why the context cannot shift: recurrent state, sparse side caches (DSA indexer, MSA index keys, DSV4
         * compressed rows), or layers that read a target model's cache.
         */
        public static String shiftBlocker(MemoryPlan plan) {
            if (plan.hasRecurrentState()) {
                if (!plan.hasOnlyPositionalState()) {
                    return "the model keeps recurrent state";
                }
                if (plan.config() == null || !plan.config().contextShift()) {
                    return "the Step5 region state holds only the open region; plan the memory with "
                            + MemoryConfig.PROP_CONTEXT_SHIFT + "=true so every kept position survives a shift";
                }
            }
            for (LayerMemoryPlan lp : plan.layers()) {
                if (lp.indexer() != null || lp.msa() != null || !lp.compressed().isEmpty()) {
                    return "sparse-attention caches (DSA indexer, MSA index keys, DSV4 compressed rows) cannot be "
                            + "re-positioned";
                }
                if (lp.sharedFrom() != null && lp.sharedFrom().target()) {
                    return "layer " + lp.layer() + " reads the KV cache of a target model";
                }
            }
            return null;
        }

        /**
         * Why a context shift would leave a sliding-window ring short of keys: the first query after the shift
         * ({@code nPast - nDiscard}) must still find every kept position its window covers. llama.cpp refuses any
         * shift of a cache with rings (src/llama-kv-cache-iswa.cpp:251-255); the shifts allowed here give the result
         * of the same shift on full-size caches.
         */
        public static String ringShiftBlocker(GroupPositions group, SequenceState sequence, int nKeep, int nDiscard) {
            int nPastAfter = sequence.nPast() - nDiscard;
            if (group.keepsWindowAfterShift(nKeep, nPastAfter)) {
                return null;
            }
            return "the sliding-window ring of " + group.group() + " overwrote kept positions below " + nKeep
                    + " that the query at position " + nPastAfter + " still attends";
        }

        /** Validates the range of a context shift against the stored positions. */
        public static void checkShift(SequenceState sequence, int nKeep, int nDiscard) {
            if (nKeep < 0 || nDiscard <= 0 || (long) nKeep + nDiscard > sequence.nPast()) {
                throw new IllegalArgumentException("context shift of keep=" + nKeep + " discard=" + nDiscard
                        + " is invalid for " + sequence.nPast() + " stored positions");
            }
        }

        /**
         * Removes positions {@code [nKeep, nKeep + nDiscard)} from the sequence record, keeping its wrap flag. The
         * record itself rewrites the tokens in place and caps its contiguous reuse frontier at the kept count
         * ({@link SequenceState#shift}), so a following generation can never reuse the rows
         * that were computed with the discarded context.
         */
        public static void shiftSequence(SequenceState sequence, int nKeep, int nDiscard) {
            sequence.shift(nKeep, nDiscard);
        }
    }

    /**
     * Immutable plan of one group's context shift: positions after the shift, the per-row RoPE deltas (indexed by
     * the row after the shift) and the row moves, as runs {@code (src, dst, n)} in ascending source order.
     *
     * <p>Executing the moves on a cache: when every move goes down ({@code dst < src}, always the case for a full
     * cache, where {@code dst = src - nDiscard}) the runs are copied in order, each in chunks of at most
     * {@code src - dst} rows, so no chunk overlaps itself or a row still to be read. Otherwise (a ring rotation)
     * the moved rows are first gathered into a staging area and then scattered ({@link #needsStaging()}).
     */
    public static final class KvShift {

        private final int[] positionsAfter;
        private final int[] delta;
        private final int[][] runs;
        private final int movedRows;
        private final boolean needsStaging;

        KvShift(int[] positionsAfter, int[] delta, List<int[]> runs) {
            this.positionsAfter = positionsAfter;
            this.delta = delta;
            this.runs = runs.toArray(new int[0][]);
            int moved = 0;
            boolean staging = false;
            for (int[] run : this.runs) {
                moved += run[2];
                staging |= run[1] > run[0];
            }
            this.movedRows = moved;
            this.needsStaging = staging;
        }

        public int rows() {
            return positionsAfter.length;
        }

        /** Position of a row after the shift, -1 when empty. */
        public int positionAfter(int row) {
            return positionsAfter[row];
        }

        int[] positionsAfter() {
            return positionsAfter.clone();
        }

        /** RoPE delta of a row after the shift (0 for rows not shifted). */
        public int delta(int row) {
            return delta[row];
        }

        /** Number of move runs. */
        public int runCount() {
            return runs.length;
        }

        public int runSource(int i) {
            return runs[i][0];
        }

        public int runTarget(int i) {
            return runs[i][1];
        }

        public int runRows(int i) {
            return runs[i][2];
        }

        /** Total rows moved. */
        public int movedRows() {
            return movedRows;
        }

        /** True when some row moves up, so the moves need a staging area. */
        public boolean needsStaging() {
            return needsStaging;
        }
    }
}
