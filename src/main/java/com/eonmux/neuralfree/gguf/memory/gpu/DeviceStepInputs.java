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
    private final int identityLength;
    private final int[] ints;
    private final byte[] staging;
    private final ByteBuffer stagingView;
    private int pos0;
    private int nTokens;
    private CompressorRollback compressorSteps;
    private boolean closed;

    private DeviceStepInputs(ComputeBackend backend, ComputeQueue queue, List<MemoryPlan.KvGroup> groups, int ubatch,
                             int identityLength, MropeTextPositions textPositions, SequenceState sequence,
                             List<GpuBuffer> owned) {
        this.backend = backend;
        this.queue = queue;
        this.ubatch = ubatch;
        this.identityLength = identityLength;
        this.textPositions = textPositions;
        this.sequence = sequence;
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
     * @throws IllegalArgumentException for a non-positive {@code ubatch} or {@code identityLength < ubatch}
     */
    public static DeviceStepInputs allocate(ComputeBackend backend, ComputeQueue queue, List<MemoryPlan.KvGroup> groups,
                                            int ubatch, int identityLength, MropeTextPositions textPositions,
                                            SequenceState sequence) {
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
            return new DeviceStepInputs(backend, queue, groups, ubatch, identityLength, textPositions, sequence, owned);
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
                ints[t] = p.group().rowOf(pos0 + t);
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
            compressorSteps.onStep(nTokens);
        }
    }

    /**
     * Reports every uploaded step to the owning memory's compressor rollback record, whose snapshots each step
     * replaces.
     */
    void trackCompressorSteps(CompressorRollback rollback) {
        this.compressorSteps = Objects.requireNonNull(rollback, "rollback");
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
        pos0 = 0;
        nTokens = 0;
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
        for (int i = 0; i < n; i++) {
            stagingView.putInt(i * Integer.BYTES, src[i]);
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
    }

    // ================================================================== host bookkeeping

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
            } else {
                group.rowOf(pos0);
                group.rowOf(pos0 + n - 1);
            }
        }

        /**
         * Stores positions {@code [pos0, pos0 + n)} in their rows.
         *
         * @return true when a ring row holding a different stored position was overwritten
         */
        public boolean store(int pos0, int n) {
            checkStep(pos0, n);
            boolean overwrote = false;
            for (int t = 0; t < n; t++) {
                int p = pos0 + t;
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
                if (p < 0 || (!group.ring() && p >= pos.length)) {
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
         * First row of the smallest contiguous row range covering positions {@code [pos0, pos0 + n)}: the run of
         * rows when it does not wrap, the whole group when it does.
         */
        public int spanFrom(int pos0, int n) {
            int r0 = group.rowOf(pos0);
            return r0 + n <= pos.length ? r0 : 0;
        }

        /** End (exclusive) of the row range of {@link #spanFrom(int, int)}. */
        public int spanTo(int pos0, int n) {
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
                int d = p < nKeep ? r : group.rowOf(q);
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
                return "the model keeps recurrent state";
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

        /** Removes positions {@code [nKeep, nKeep + nDiscard)} from the sequence record, keeping its wrap flag. */
        public static void shiftSequence(SequenceState sequence, int nKeep, int nDiscard) {
            int[] tokens = sequence.tokens();
            boolean wrapped = sequence.wrapped();
            sequence.clear();
            sequence.append(tokens, 0, nKeep);
            sequence.append(tokens, nKeep + nDiscard, tokens.length - nKeep - nDiscard);
            if (wrapped) {
                sequence.markWrapped();
            }
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
