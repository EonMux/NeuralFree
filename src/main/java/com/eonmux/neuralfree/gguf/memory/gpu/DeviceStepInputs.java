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
 * the context shift change the mirror and rewrite {@code kv_pos} the same way. Over a paged memory every mirror
 * change between two steps marks its row ({@link KvPosDirtyRows}), and {@link #beginPagedStep} writes only the
 * marked rows.
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
    /**
     * Per group: the rows whose device {@code kv_pos} the next paged step must rewrite; null for the contiguous form.
     */
    private final KvPosDirtyRows[] dirty;
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
        this.dirty = pages != null ? new KvPosDirtyRows[n] : null;
        int maxInts = Math.max(MropeTextPositions.SECTIONS * ubatch, identityLength);
        for (int g = 0; g < n; g++) {
            MemoryPlan.KvGroup group = groups.get(g);
            if (group.index() != g) {
                throw new IllegalArgumentException("KV group at position " + g + " has index " + group.index());
            }
            positions[g] = new GroupPositions(group);
            if (dirty != null) {
                dirty[g] = new KvPosDirtyRows(group.rows());
            }
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

    @Override
    public boolean pagedTrunkWrites() {
        return paged != null && paged.addressesSequences();
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
            if (dirty != null) {
                dirty[g].clearAll();
            }
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
     * Takes {@code mirror} as the positions of a paged group without writing the device: the rows whose position
     * changes are marked, and the next paged step's upload carries them.
     *
     * @throws IllegalArgumentException when the mirror's length differs from the group's rows
     * @throws IllegalStateException    when the memory is not paged
     */
    void installPagedMirror(int group, int[] mirror) {
        checkOpen();
        GroupPositions p = positions[checkGroup(group)];
        if (dirty == null) {
            throw new IllegalStateException("the contiguous step inputs write their mirror changes at once");
        }
        if (mirror.length != p.rows()) {
            throw new IllegalArgumentException("a mirror of " + mirror.length + " rows for KV group " + group + " of "
                    + p.rows() + " rows");
        }
        for (int r = 0; r < mirror.length; r++) {
            setPagedRow(group, r, mirror[r]);
        }
    }

    /** Sets one paged mirror row, marking it for the next step's upload when its position changes. */
    private void setPagedRow(int group, int row, int position) {
        GroupPositions p = positions[group];
        if (p.position(row) != position) {
            p.set(row, position);
            dirty[group].mark(row);
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
        if (dirty != null) {
            dirty[group].clear(from, to);
        }
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
     * positions. {@link #begin} uploads a step: the marked {@code kv_pos} rows, the page-table entries that differ
     * from what the device holds ({@link DevicePageTables}), and the per-step arrays whole; the attention mixers
     * read the buffers through {@link GpuStepInputs} of the owning memory. Not thread-safe.
     */
    final class Paged {

        private final ComputeQueue queue;
        private final PagedSequences pages;
        private final int groups;
        private final int maxSequences;
        private final int maxPages;
        private final int pageShift;
        private final DevicePageTables[] pageTable;   // [groups] int[maxSequences * maxPages]
        private final GpuBuffer[] seqKvLen;    // [groups] int[maxSequences]
        private final GpuBuffer qPosSeq;       // int[maxSequences], per-launch query position of sequence s
        private final GpuBuffer segPos;        // int[ubatch], the positions of one prefill segment
        private final SequenceKvPos[] seqKvPos;  // [groups] int[rows] per sequence, reserved per group
        private final boolean[][] kvPosBuilt;
        /** The {@code [from, to)} row ranges of one group's marked {@code kv_pos} rows. */
        private final int[] runs = new int[2 * KvPosDirtyRows.MAX_RUNS];
        /** One value per launch sequence, staged for one write. */
        private final int[] launch;
        private PagedStep current;

        private Paged(ComputeBackend backend, ComputeQueue queue, int groups, int ubatch, PagedSequences pages,
                      List<GpuBuffer> owned) {
            this.queue = queue;
            this.pages = pages;
            this.groups = groups;
            this.maxSequences = pages.maxSequences();
            this.maxPages = pages.maxPages();
            this.pageShift = pageShiftOf(pages.pageSize());
            this.pageTable = new DevicePageTables[groups];
            this.seqKvLen = new GpuBuffer[groups];
            for (int g = 0; g < groups; g++) {
                GpuBuffer table = track(owned, alloc((long) maxSequences * maxPages * Integer.BYTES));
                queue.enqueueFillBuffer(table, 0, table.sizeInBytes(), -1);
                pageTable[g] = new DevicePageTables(table, maxSequences, maxPages);
                seqKvLen[g] = track(owned, alloc((long) maxSequences * Integer.BYTES));
                queue.enqueueFillBuffer(seqKvLen[g], 0, seqKvLen[g].sizeInBytes(), 0);
            }
            this.qPosSeq = track(owned, alloc((long) maxSequences * Integer.BYTES));
            queue.enqueueFillBuffer(qPosSeq, 0, qPosSeq.sizeInBytes(), 0);
            this.segPos = track(owned, alloc((long) ubatch * Integer.BYTES));
            queue.enqueueFillBuffer(segPos, 0, segPos.sizeInBytes(), 0);
            this.launch = new int[maxSequences];
            this.seqKvPos = new SequenceKvPos[groups];
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
            seqKvPos[group] = new SequenceKvPos(kv, positions[group].rows(), maxPages * pages.pageSize());
        }

        /**
         * {@code kv_pos} of {@code group} restricted to sequence {@code seq}'s cells (see the interface): the first
         * call of a step patches the sequence's persistent buffer to the step ({@link SequenceKvPos#update}).
         */
        GpuBuffer sequenceKvPos(int group, int seq) {
            checkSequence(group, seq);
            if (!kvPosBuilt[group][seq]) {
                seqKvPos[group].update(current, group, seq, positions[group], DeviceStepInputs.this::writeInts);
                kvPosBuilt[group][seq] = true;
            }
            return seqKvPos[group].buffer(seq);
        }

        /** Whether a group reserved its per-sequence addressing, which the trunk patches ({@link #sequenceKvPos}). */
        boolean addressesSequences() {
            for (SequenceKvPos s : seqKvPos) {
                if (s != null) {
                    return true;
                }
            }
            return false;
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
            return pageTable[group].buffer();
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
                    setPagedRow(r.group(), rows[i], values[i]);
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
                writeDirtyRows(g);
                writeInts(cellIdx[g], 0, step.cellIdx(g), step.nTokens());
                pageTable[g].upload(step, g, DeviceStepInputs.this::writeInts);
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
            for (int i = 0; i < rows.length; i++) {
                setPagedRow(group, rows[i], values[i]);
            }
        }

        /** The {@code kv_pos} writes of {@code group} that carry every marked row of the mirror to the device. */
        private void writeDirtyRows(int group) {
            int n = dirty[group].runs(runs);
            for (int r = 0; r < n; r++) {
                writeMirror(group, runs[2 * r], runs[2 * r + 1]);
            }
        }

        /** Slots each launch sequence holds in {@code group} after the step, in the launch scratch. */
        private int[] launchSlots(int group, PagedStep step) {
            for (int s = 0; s < segments(step); s++) {
                launch[s] = step.seqSlot(group, step.segSeq(s));
            }
            return launch;
        }

        /** Query position of each launch sequence (its segment's first position), in the launch scratch. */
        private int[] launchPositions(PagedStep step) {
            for (int s = 0; s < segments(step); s++) {
                launch[s] = step.segPos0(s);
            }
            return launch;
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
                pageTable[g].buffer().close();
                seqKvLen[g].close();
            }
            qPosSeq.close();
            segPos.close();
            for (int g = 0; g < groups; g++) {
                if (seqKvPos[g] != null) {
                    seqKvPos[g].close();
                }
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
}
