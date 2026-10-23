package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Traces of the device memory against the host-only fake backend: per-step uploads (one write per buffer, section-major
 * RoPE positions, ring wraps), rollback on a ring of {@code nSwa + ubatch} rows (including rollbacks after earlier
 * drops and the speculative-decoding window query), reset, recurrent snapshot / restore with reserved shadows, the
 * context shift on full and ring caches, placeholders, and buffer ownership.
 */
public class DeviceStepInputsTest {

    private static final int N_SWA = 8;
    private static final int UBATCH = 8;
    private static final KvWindow SWA = KvWindow.of(SwaType.STANDARD, N_SWA, true);
    private static final int WIDTH = 4;

    private FakeComputeBackend backend;

    @Before
    public void setUp() {
        backend = FakeComputeBackend.create();
    }

    @After
    public void tearDown() {
        backend.close();
    }

    // ------------------------------------------------------------------ fixtures

    private static MemoryPlan.KvGroup ringGroup(int index, int rows) {
        return MemoryPlan.KvGroup.of(index, rows, true, SWA);
    }

    private static MemoryPlan.KvGroup fullGroup(int index, int rows) {
        return MemoryPlan.KvGroup.of(index, rows, false, KvWindow.FULL_CAUSAL);
    }

    /** One F32 KV layer on a ring of {@code rows} rows. */
    private static MemoryPlan ringPlan(int nCtx, int rows) {
        KvLayout ring = KvLayout.of(rows, true, WIDTH, WIDTH, KvDType.F32, KvDType.F32, SWA);
        return MemoryPlan.builder(MemoryKind.KV, nCtx, UBATCH)
                .group(ringGroup(0, rows))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(ring, 0).build())
                .build();
    }

    /** One F32 KV layer on a full cache of {@code nCtx} rows. */
    private static MemoryPlan fullPlan(int nCtx) {
        KvLayout full = KvLayout.of(nCtx, false, WIDTH, WIDTH, KvDType.F32, KvDType.F32, KvWindow.FULL_CAUSAL);
        return MemoryPlan.builder(MemoryKind.KV, nCtx, UBATCH)
                .group(fullGroup(0, nCtx))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(full, 0).build())
                .build();
    }

    /**
     * A full-cache attention layer followed by a recurrent layer with one conv tensor of 12 floats, planned with
     * recurrent-state snapshots.
     */
    private static MemoryPlan hybridPlan(int nCtx) {
        return hybridPlan(nCtx, true);
    }

    private static MemoryPlan hybridPlan(int nCtx, boolean snapshots) {
        KvLayout full = KvLayout.of(nCtx, false, WIDTH, WIDTH, KvDType.F32, KvDType.F32, KvWindow.FULL_CAUSAL);
        return MemoryPlan.builder(MemoryKind.HYBRID, nCtx, UBATCH)
                .config(MemoryConfig.builder().stateSnapshots(snapshots).build())
                .group(fullGroup(0, nCtx))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(full, 0).build())
                .layer(LayerMemoryPlan.builder(1, LayerMemoryPlan.Kind.RECURRENT)
                        .state(LayerMemoryPlan.StateTensor.of(LayerMemoryPlan.STATE_CONV,
                                LayerMemoryPlan.StateLayout.CONV, 12))
                        .build())
                .build();
    }

    private DeviceModelMemory allocate(MemoryPlan plan) {
        return DeviceModelMemory.allocate(backend, backend.computeQueue(), plan, MropeTextPositions.PPP0, null);
    }

    /** Uploads a step and records its tokens ({@code 100 + pos}) as the forward backend does. */
    private static void step(DeviceModelMemory mem, int pos0, int n) {
        mem.stepInputs().upload(pos0, n);
        int[] tokens = new int[n];
        for (int t = 0; t < n; t++) {
            tokens[t] = 100 + pos0 + t;
        }
        mem.sequence().append(tokens, 0, n);
    }

    private static FakeGpuBuffer fake(GpuBuffer b) {
        return (FakeGpuBuffer) b;
    }

    private static int[] ints(GpuBuffer b, int n) {
        return fake(b).getInts(0, n);
    }

    private static int[] range(int from, int n) {
        int[] r = new int[n];
        for (int i = 0; i < n; i++) {
            r[i] = from + i;
        }
        return r;
    }

    private List<FakeComputeBackend.Op> opsOf(FakeComputeBackend.OpKind kind) {
        List<FakeComputeBackend.Op> out = new ArrayList<>();
        for (FakeComputeBackend.Op op : backend.ops()) {
            if (op.kind() == kind) {
                out.add(op);
            }
        }
        return out;
    }

    /** Writes {@code pos * 10 + j} into K row {@code r} for every stored position. */
    private static void fillKWithPositions(DeviceModelMemory mem, int rows) {
        FakeGpuBuffer k = fake(mem.kv(0).k());
        DeviceStepInputs in = mem.stepInputs();
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < WIDTH; j++) {
                k.putFloat((long) r * WIDTH + j, in.positionOfRow(0, r) * 10f + j);
            }
        }
    }

    // ------------------------------------------------------------------ uploads

    @Test
    public void uploadWritesEachBufferOnceWithSectionMajorRopePositions() {
        SequenceState seq = new SequenceState(64);
        try (DeviceStepInputs in = DeviceStepInputs.allocate(backend, backend.computeQueue(),
                Arrays.asList(fullGroup(0, 64), ringGroup(1, 16)), UBATCH, UBATCH, MropeTextPositions.PPP0, seq)) {
            backend.clearLog();
            in.upload(3, 4);

            List<FakeComputeBackend.Op> writes = opsOf(FakeComputeBackend.OpKind.WRITE);
            assertEquals("kv_pos + cell_idx per group, q_pos and rope_pos", 2 * 2 + 2, writes.size());
            assertEquals(backend.ops().size(), writes.size());
            assertArrayEquals(range(3, 4), ints(in.cellIdx(0), 4));
            assertArrayEquals(range(3, 4), ints(in.cellIdx(1), 4));
            assertArrayEquals(range(3, 4), ints(in.qPos(), 4));
            assertArrayEquals(new int[]{3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 0, 0, 0, 0}, ints(in.ropePos(), 16));
            assertEquals(0, ints(in.stateRows(), 1)[0]);
            assertEquals(3, in.pos0());
            assertEquals(4, in.nTokens());
            for (FakeComputeBackend.Op w : writes) {
                if (w.buffer() == in.kvPos(0)) {
                    assertEquals(3L * Integer.BYTES, w.offset());
                    assertEquals(4L * Integer.BYTES, w.bytes());
                }
            }
            int[] kvPos = ints(in.kvPos(0), 8);
            assertArrayEquals(new int[]{-1, -1, -1, 3, 4, 5, 6, -1}, kvPos);
            assertFalse(seq.wrapped());
        }
    }

    @Test
    public void textOnlyTemporalPatternLeavesSpatialSectionsZero() {
        SequenceState seq = new SequenceState(64);
        try (DeviceStepInputs in = DeviceStepInputs.allocate(backend, backend.computeQueue(),
                Arrays.asList(fullGroup(0, 64)), UBATCH, UBATCH, MropeTextPositions.P000, seq)) {
            in.upload(10, 2);
            assertArrayEquals(new int[]{10, 11, 0, 0, 0, 0, 0, 0}, ints(in.ropePos(), 8));
        }
    }

    /**
     * Ring of 16 rows ({@code nSwa 8 + ubatch 8}): steps [0, 8), [8, 14), then [14, 19) wrap into rows 14, 15, 0, 1,
     * 2, overwriting positions 0..2. The wrapping step rewrites the whole ring with one write and marks the sequence
     * wrapped.
     */
    @Test
    public void ringWrapUploadRewritesTheWholeRingOnce() {
        SequenceState seq = new SequenceState(64);
        try (DeviceStepInputs in = DeviceStepInputs.allocate(backend, backend.computeQueue(),
                Arrays.asList(ringGroup(0, 16)), UBATCH, UBATCH, MropeTextPositions.PPP0, seq)) {
            in.upload(0, 8);
            in.upload(8, 6);
            assertFalse(seq.wrapped());
            backend.clearLog();

            in.upload(14, 5);

            assertArrayEquals(new int[]{14, 15, 0, 1, 2}, ints(in.cellIdx(0), 5));
            int kvPosWrites = 0;
            for (FakeComputeBackend.Op w : opsOf(FakeComputeBackend.OpKind.WRITE)) {
                if (w.buffer() == in.kvPos(0)) {
                    kvPosWrites++;
                    assertEquals(0L, w.offset());
                    assertEquals(16L * Integer.BYTES, w.bytes());
                }
            }
            assertEquals(1, kvPosWrites);
            assertArrayEquals(new int[]{16, 17, 18, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                    ints(in.kvPos(0), 16));
            assertTrue(seq.wrapped());
        }
    }

    @Test
    public void uploadRejectsStepsThatDoNotFitAndChangesNothing() {
        SequenceState seq = new SequenceState(16);
        try (DeviceStepInputs in = DeviceStepInputs.allocate(backend, backend.computeQueue(),
                Arrays.asList(ringGroup(0, 16), fullGroup(1, 16)), UBATCH, UBATCH, MropeTextPositions.PPP0, seq)) {
            assertThrows(IllegalArgumentException.class, () -> in.upload(0, UBATCH + 1));
            assertThrows(IllegalArgumentException.class, () -> in.upload(0, 0));
            assertThrows(IllegalArgumentException.class, () -> in.upload(12, 5));
            assertEquals(-1, in.positionOfRow(0, 12));
            assertEquals(-1, in.positionOfRow(1, 12));
        }
    }

    // ------------------------------------------------------------------ rollback, reset, snapshot

    /**
     * Ring of {@code nSwa + ubatch = 16} rows: a 14-token prompt, then a 5-token draft at positions 14..18 that wraps
     * into rows 0..2. {@code rollback(3)} is allowed (the query at 16 sees keys 9..15, all still in rows 9..15) and
     * empties rows 0..2 with one 12-byte write. {@code rollback(10)} is refused: the query at 9 needs key 2, whose
     * row now holds 18.
     */
    @Test
    public void rollbackOfThreeDraftTokensOnARingEmptiesTheirRows() {
        try (DeviceModelMemory mem = allocate(ringPlan(64, N_SWA + UBATCH))) {
            step(mem, 0, 8);
            step(mem, 8, 6);
            step(mem, 14, 5);
            assertTrue(mem.sequence().wrapped());
            assertTrue(mem.canRollback(3));
            assertFalse("row 2 holds 18, not the key 2 the query at 9 needs", mem.canRollback(10));
            backend.clearLog();

            mem.rollback(3);

            List<FakeComputeBackend.Op> writes = opsOf(FakeComputeBackend.OpKind.WRITE);
            assertEquals(1, writes.size());
            assertEquals(0L, writes.get(0).offset());
            assertEquals(3L * Integer.BYTES, writes.get(0).bytes());
            int[] expected = new int[16];
            expected[0] = -1;
            expected[1] = -1;
            expected[2] = -1;
            System.arraycopy(range(3, 13), 0, expected, 3, 13);
            assertArrayEquals(expected, ints(mem.stepInputs().kvPos(0), 16));
            assertEquals(16, mem.sequence().nPast());
            for (int k = 9; k <= 15; k++) {
                assertEquals(k, mem.stepInputs().positionOfRow(0, k));
            }
            assertThrows(IllegalStateException.class, () -> mem.rollback(17));
        }
    }

    /**
     * Two prefix-reuse rollbacks in a row on a ring of 16 rows (nSwa 8). Positions 0..39 leave rows 0..7 = 32..39 and
     * rows 8..15 = 24..31. Dropping 9 (to 31) is allowed and empties rows 15 and 0..7; 31 and 32 are written again.
     * Dropping 9 more (to 24) must be refused: the query at 24 needs keys 17..23, whose rows 1..7 were overwritten by
     * 33..39 and then emptied, although the ring is as large as the drop allows on a freshly filled ring.
     */
    @Test
    public void rollbackAfterAnEarlierDropIsRefusedWhenItsWindowWasLost() {
        try (DeviceModelMemory mem = allocate(ringPlan(64, 16))) {
            for (int pos = 0; pos < 40; pos += UBATCH) {
                step(mem, pos, UBATCH);
            }
            assertTrue(mem.canRollback(9));
            mem.rollback(9);
            assertEquals(31, mem.sequence().nPast());
            for (int r = 0; r < 8; r++) {
                assertEquals(-1, mem.stepInputs().positionOfRow(0, r));
            }
            step(mem, 31, 2);

            assertTrue(mem.canRollback(1));
            assertFalse(mem.canRollback(9));
            assertThrows(IllegalStateException.class, () -> mem.rollback(9));
            assertEquals(33, mem.sequence().nPast());
            assertEquals(32, mem.stepInputs().positionOfRow(0, 0));
            assertEquals(-1, mem.stepInputs().positionOfRow(0, 1));
        }
    }

    /**
     * Speculative window query on a ring of 16 rows holding 24..39: writing 10 positions (40..49 evict 24..33) and
     * dropping 9 leaves the query at 41 its keys 34..40; writing 12 (evicting 34 and 35) and dropping 11 does not.
     */
    @Test
    public void ringsKeepWindowPredictsDraftRollbacks() {
        try (DeviceModelMemory mem = allocate(ringPlan(64, 16))) {
            for (int pos = 0; pos < 40; pos += UBATCH) {
                step(mem, pos, UBATCH);
            }
            assertTrue(mem.ringsKeepWindow(10, 9));
            assertFalse(mem.ringsKeepWindow(12, 11));
            assertTrue(mem.ringsKeepWindow(12, 0));
            assertTrue(mem.ringsKeepWindow(0, 9));
            assertFalse(mem.ringsKeepWindow(0, 10));
            assertThrows(IllegalArgumentException.class, () -> mem.ringsKeepWindow(-1, 0));
            assertThrows(IllegalArgumentException.class, () -> mem.ringsKeepWindow(0, 41));
        }
        try (DeviceModelMemory mem = allocate(fullPlan(64))) {
            step(mem, 0, 8);
            assertTrue(mem.ringsKeepWindow(8, 15));
        }
    }

    @Test
    public void resetEmptiesPositionsZeroesStatesAndForbidsKvRollbackOfRecurrentMemory() {
        try (DeviceModelMemory mem = allocate(hybridPlan(32))) {
            step(mem, 0, 4);
            assertFalse(mem.canRollback(1));
            assertTrue(mem.canRollback(0));
            assertThrows(IllegalStateException.class, () -> mem.rollback(1));
            backend.clearLog();

            mem.reset();

            List<FakeComputeBackend.Op> fills = opsOf(FakeComputeBackend.OpKind.FILL);
            assertEquals(2, fills.size());
            boolean sawKvPos = false;
            boolean sawState = false;
            for (FakeComputeBackend.Op f : fills) {
                if (f.buffer() == mem.stepInputs().kvPos(0)) {
                    sawKvPos = f.pattern() == -1 && f.bytes() == 32L * Integer.BYTES;
                } else if (f.buffer() == mem.recurrent(1).tensor(LayerMemoryPlan.STATE_CONV)) {
                    sawState = f.pattern() == 0 && f.bytes() == 12L * Float.BYTES;
                }
            }
            assertTrue(sawKvPos);
            assertTrue(sawState);
            assertEquals(0, mem.sequence().nPast());
            assertEquals(-1, mem.stepInputs().positionOfRow(0, 0));
            assertThrows(IllegalStateException.class, mem::restore);
        }
    }

    @Test
    public void restoreBringsBackTheStateAndDropsPositionsWrittenAfterTheSnapshot() {
        try (DeviceModelMemory mem = allocate(hybridPlan(32))) {
            FakeGpuBuffer conv = fake(mem.recurrent(1).tensor(LayerMemoryPlan.STATE_CONV));
            step(mem, 0, 6);
            conv.putFloat(0, 1.5f);
            mem.snapshot();
            step(mem, 6, 4);
            conv.putFloat(0, -7f);
            backend.clearLog();

            mem.restore();

            assertEquals(1.5f, conv.getFloat(0), 0f);
            assertEquals(6, mem.sequence().nPast());
            assertEquals(5, mem.stepInputs().positionOfRow(0, 5));
            for (int r = 6; r < 10; r++) {
                assertEquals(-1, mem.stepInputs().positionOfRow(0, r));
            }
            assertEquals(1, opsOf(FakeComputeBackend.OpKind.COPY).size());
        }
    }

    @Test
    public void snapshotIsRejectedWithoutRecurrentState() {
        try (DeviceModelMemory mem = allocate(fullPlan(16))) {
            assertFalse(mem.canSnapshot());
            assertThrows(UnsupportedOperationException.class, mem::snapshot);
        }
    }

    /** Without reserved shadows the plan counts no snapshot bytes and the memory refuses to snapshot. */
    @Test
    public void snapshotIsRejectedWhenThePlanReservedNoShadows() {
        MemoryPlan plan = hybridPlan(32, false);
        assertEquals(0L, plan.snapshotBytes());
        int before = backend.liveBuffers().size();
        try (DeviceModelMemory mem = allocate(plan)) {
            assertFalse(mem.canSnapshot());
            int live = backend.liveBuffers().size();
            assertThrows(UnsupportedOperationException.class, mem::snapshot);
            assertEquals(live, backend.liveBuffers().size());
        }
        assertEquals(before, backend.liveBuffers().size());
    }

    /** Reserved shadows are allocated with the memory and counted in the plan; a snapshot only copies. */
    @Test
    public void reservedShadowsAreAllocatedWithTheMemory() {
        MemoryPlan with = hybridPlan(32, true);
        MemoryPlan without = hybridPlan(32, false);
        assertEquals(12L * Float.BYTES, with.snapshotBytes());
        assertEquals(without.totalBytes() + 12L * Float.BYTES, with.totalBytes());
        DeviceModelMemory plain = allocate(without);
        int withoutBuffers = backend.liveBuffers().size();
        plain.close();
        try (DeviceModelMemory mem = allocate(with)) {
            assertTrue(mem.canSnapshot());
            int live = backend.liveBuffers().size();
            assertEquals(withoutBuffers + 1, live);
            step(mem, 0, 4);
            backend.clearLog();
            mem.snapshot();
            assertEquals(live, backend.liveBuffers().size());
            assertEquals(1, opsOf(FakeComputeBackend.OpKind.COPY).size());
        }
    }

    // ------------------------------------------------------------------ context shift

    /**
     * Full cache: 16 positions, shift keep 4 / discard 6. Rows 10..15 move to 4..9 with one in-place copy per buffer
     * (gap 6 covers the run), rows 10..15 empty, deltas -6 on rows 4..9.
     */
    @Test
    public void contextShiftOnAFullCacheCompactsRowsInPlace() {
        try (DeviceModelMemory mem = allocate(fullPlan(32))) {
            step(mem, 0, 8);
            step(mem, 8, 8);
            fillKWithPositions(mem, 16);
            backend.clearLog();

            List<GpuBuffer> deltas = mem.shiftPositions(4, 6);

            assertEquals(2, opsOf(FakeComputeBackend.OpKind.COPY).size());
            DeviceStepInputs in = mem.stepInputs();
            FakeGpuBuffer k = fake(mem.kv(0).k());
            for (int r = 0; r < 10; r++) {
                assertEquals(r, in.positionOfRow(0, r));
                int oldPos = r < 4 ? r : r + 6;
                assertEquals(oldPos * 10f, k.getFloat((long) r * WIDTH), 0f);
            }
            for (int r = 10; r < 16; r++) {
                assertEquals(-1, in.positionOfRow(0, r));
            }
            int[] delta = ints(deltas.get(0), 16);
            for (int r = 0; r < 16; r++) {
                assertEquals(r >= 4 && r < 10 ? -6 : 0, delta[r]);
            }
            assertArrayEquals(new int[]{100, 101, 102, 103, 110, 111, 112, 113, 114, 115}, mem.sequence().tokens());
        }
    }

    /**
     * Ring of 16 rows holding positions 4..19 (16..19 in rows 0..3). Shift keep 2 / discard 5: positions 7..15 move
     * down from rows 7..15 to 2..10 and 16..19 move up from rows 0..3 to 11..14, so the rows go through a staging
     * buffer that is released afterwards.
     */
    @Test
    public void contextShiftOnARingRotatesThroughStaging() {
        try (DeviceModelMemory mem = allocate(ringPlan(64, 16))) {
            step(mem, 0, 8);
            step(mem, 8, 8);
            step(mem, 16, 4);
            fillKWithPositions(mem, 16);
            int live = backend.liveBuffers().size();

            List<GpuBuffer> deltas = mem.shiftPositions(2, 5);

            assertEquals(live + 1, backend.liveBuffers().size());
            DeviceStepInputs in = mem.stepInputs();
            FakeGpuBuffer k = fake(mem.kv(0).k());
            int[] delta = ints(deltas.get(0), 16);
            for (int r = 0; r < 16; r++) {
                if (r >= 2 && r <= 14) {
                    assertEquals(r, in.positionOfRow(0, r));
                    assertEquals((r + 5) * 10f + 3, k.getFloat((long) r * WIDTH + 3), 0f);
                    assertEquals(-5, delta[r]);
                } else {
                    assertEquals(-1, in.positionOfRow(0, r));
                    assertEquals(0, delta[r]);
                }
            }
            assertEquals(15, mem.sequence().nPast());
            assertEquals(101, mem.sequence().token(1));
            assertEquals(107, mem.sequence().token(2));
            assertTrue(mem.sequence().wrapped());
        }
    }

    /**
     * Ring of 16 rows holding positions 4..19. Shift keep 4 / discard 10 puts the next query at 10, whose window
     * (3..10) needs kept position 3, overwritten by 19: the shift is refused and nothing changes.
     */
    @Test
    public void contextShiftOnARingIsRefusedWhenTheNextWindowNeedsOverwrittenKeptPositions() {
        try (DeviceModelMemory mem = allocate(ringPlan(64, 16))) {
            step(mem, 0, 8);
            step(mem, 8, 8);
            step(mem, 16, 4);
            backend.clearLog();
            int live = backend.liveBuffers().size();

            assertThrows(UnsupportedOperationException.class, () -> mem.shiftPositions(4, 10));

            assertTrue(backend.ops().isEmpty());
            assertEquals(live, backend.liveBuffers().size());
            assertEquals(20, mem.sequence().nPast());
            assertEquals(19, mem.stepInputs().positionOfRow(0, 3));
            mem.shiftPositions(2, 5);
            assertEquals(15, mem.sequence().nPast());
        }
    }

    @Test
    public void contextShiftIsRejectedForRecurrentMemoryAndInvalidRanges() {
        try (DeviceModelMemory mem = allocate(hybridPlan(32))) {
            step(mem, 0, 8);
            assertThrows(UnsupportedOperationException.class, () -> mem.shiftPositions(2, 2));
        }
        try (DeviceModelMemory mem = allocate(fullPlan(32))) {
            step(mem, 0, 8);
            assertThrows(IllegalArgumentException.class, () -> mem.shiftPositions(4, 5));
            assertThrows(IllegalArgumentException.class, () -> mem.shiftPositions(4, 0));
            assertThrows(IllegalStateException.class, () -> mem.shiftDelta(0));
        }
    }

    // ------------------------------------------------------------------ ownership

    @Test
    public void closeReleasesEveryBufferAndEvent() {
        DeviceModelMemory hybrid = allocate(hybridPlan(32));
        step(hybrid, 0, 8);
        hybrid.snapshot();
        hybrid.close();
        DeviceModelMemory full = allocate(fullPlan(32));
        step(full, 0, 8);
        full.shiftPositions(2, 3);
        full.close();
        full.close();
        assertTrue(backend.liveBuffers().isEmpty());
        assertEquals(0, backend.openEventCount());
    }

    @Test
    public void placeholdersAreZeroedAndTyped() {
        try (GpuPlaceholders ph = GpuPlaceholders.allocate(backend, backend.computeQueue())) {
            assertTrue(fake(ph.readOnlyFloats()).isReadOnly());
            assertTrue(fake(ph.readOnlyInts()).isReadOnly());
            assertFalse(fake(ph.scratchFloats()).isReadOnly());
            assertFalse(fake(ph.scratchInts()).isReadOnly());
            assertArrayEquals(new int[GpuPlaceholders.ELEMENTS], ints(ph.readOnlyInts(), GpuPlaceholders.ELEMENTS));
        }
        assertTrue(backend.liveBuffers().isEmpty());
    }
}
