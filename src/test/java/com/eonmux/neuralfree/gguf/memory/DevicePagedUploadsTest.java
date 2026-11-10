package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import com.eonmux.neuralfree.gguf.memory.paged.PageCopy;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import com.eonmux.neuralfree.gguf.testutil.FakeOp;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The paged step upload writes only what changed ({@code DeviceStepInputs.beginPagedStep}): the device
 * {@code kv_pos} of every group receives the rows the mirror changed since the last step, the page table the
 * entries that differ from what the device holds, and each reserved per-sequence {@code kv_pos} the cells whose
 * value changed. After every operation — appends, page crossings, segment reordering, a sequence leaving and
 * another joining, tail and whole removals, a copy-on-write privatization, the sliding-window trim, a context
 * shift, a session mirror install, a context-parallel restriction and a reset — the device buffers equal what a
 * whole rewrite would leave: the device {@code kv_pos} equals the mirror, the page table equals the whole-row
 * writes of every launch row, and each per-sequence {@code kv_pos} equals the sequence's mask rebuilt from the
 * mirror.
 */
public class DevicePagedUploadsTest {

    private static final int PAGE = 4;
    private static final int CTX = 64;
    private static final int SWA_ROWS = 32;
    private static final int WINDOW = 6;
    private static final int PARALLEL = 3;
    private static final int UBATCH = 32;
    private static final int WIDTH = 4;

    private FakeComputeBackend backend;
    /** The page table a whole write of every launch row of every step would leave, per group. */
    private int[][] expectedTables;
    /** Whether the test reserved the per-sequence addressing of every group. */
    private boolean reserved;
    /** The per-sequence kv_pos buffers the checks read, by group and sequence. */
    private final GpuBuffer[][] sequenceBuffers = new GpuBuffer[2][PARALLEL + 1];

    @Before
    public void setUp() {
        backend = FakeComputeBackend.create();
    }

    @After
    public void tearDown() {
        backend.close();
    }

    /** Group 0 trims a sliding window over 32 rows, group 1 keeps every row of the 64-row context. */
    private static MemoryPlan windowPlan() {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F32).typeV(KvDType.F32)
                .nCtx(CTX).ubatch(UBATCH).parallel(PARALLEL).kvPageSize(PAGE).build();
        KvWindow swa = KvWindow.of(SwaType.STANDARD, WINDOW, true);
        KvWindow full = KvWindow.FULL_CAUSAL;
        return MemoryPlan.builder(MemoryKind.KV, CTX, UBATCH)
                .group(MemoryPlan.KvGroup.of(0, SWA_ROWS, false, swa, PAGE, SWA_ROWS / PAGE))
                .group(MemoryPlan.KvGroup.of(1, CTX, false, full, PAGE, CTX / PAGE))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV)
                        .kv(KvLayout.of(SWA_ROWS, false, WIDTH, WIDTH, KvDType.F32, KvDType.F32, swa), 0).build())
                .layer(LayerMemoryPlan.builder(1, LayerMemoryPlan.Kind.KV)
                        .kv(KvLayout.of(CTX, false, WIDTH, WIDTH, KvDType.F32, KvDType.F32, full), 1).build())
                .config(cfg)
                .build();
    }

    /** One full-attention group of {@code ctx} rows in pages of {@code page}. */
    private static MemoryPlan fullPlan(int ctx, int page, int parallel, int ubatch) {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F32).typeV(KvDType.F32)
                .nCtx(ctx).ubatch(ubatch).parallel(parallel).kvPageSize(page).build();
        MemoryPlan.KvGroup g = MemoryPlan.KvGroup.of(0, ctx, false, KvWindow.FULL_CAUSAL, page, ctx / page);
        KvLayout kv = KvLayout.of(ctx, false, WIDTH, WIDTH, KvDType.F32, KvDType.F32, KvWindow.FULL_CAUSAL);
        return MemoryPlan.builder(MemoryKind.KV, ctx, ubatch)
                .group(g)
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(kv, 0).build())
                .config(cfg)
                .build();
    }

    private DeviceModelMemory allocate(MemoryPlan plan) {
        DeviceModelMemory mem = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan,
                MropeTextPositions.PPP0, null);
        PagedSequences paged = mem.pagedSequences();
        expectedTables = new int[plan.groups().size()][paged.maxSequences() * paged.maxPages()];
        for (int[] t : expectedTables) {
            Arrays.fill(t, -1);
        }
        return mem;
    }

    /**
     * One step over {@code (seq, len, pos0)} segments in the given order, run the way the GPU driver orders it:
     * reserve, page copies, commit, upload, trunk (the checks), row record, finish.
     */
    private PagedStep run(DeviceModelMemory mem, int[]... segments) {
        PagedSequences paged = mem.pagedSequences();
        int n = 0;
        int[] seq = new int[segments.length];
        int[] start = new int[segments.length];
        int[] len = new int[segments.length];
        int[] pos0 = new int[segments.length];
        for (int s = 0; s < segments.length; s++) {
            seq[s] = segments[s][0];
            len[s] = segments[s][1];
            pos0[s] = segments[s][2];
            start[s] = n;
            n += len[s];
        }
        PagedStep step = paged.prepare(StepLayout.of(n, seq, start, len, pos0, new int[0], true));
        for (PageCopy c : step.pageCopies()) {
            mem.copyPage(c);
        }
        paged.commit(step);
        int[] rows = new int[n];
        for (int s = 0; s < segments.length; s++) {
            for (int t = 0; t < len[s]; t++) {
                rows[start[s] + t] = pos0[s] + t;
            }
        }
        mem.beginPagedStep(step, rows, null);
        try {
            expectWholeTableWrites(step);
            assertDeviceMatchesWholeWrites(mem, step);
        } finally {
            mem.endPagedStep();
        }
        for (int s = 0; s < segments.length; s++) {
            int[] tokens = new int[len[s]];
            Arrays.fill(tokens, 7);
            paged.appendTokens(seq[s], tokens);
        }
        paged.finish(step);
        paged.releaseCopies(step.pageCopies().toArray(new PageCopy[0]));
        return step;
    }

    /** The old upload's page table: every launch row rewritten whole from the step's table. */
    private void expectWholeTableWrites(PagedStep step) {
        int maxPages = step.maxPages();
        for (int g = 0; g < step.groups(); g++) {
            int[] table = step.pageTable(g);
            for (int s = 0; s < step.nSeg(); s++) {
                System.arraycopy(table, step.segSeq(s) * maxPages, expectedTables[g], s * maxPages, maxPages);
            }
        }
    }

    private static int[] ints(GpuBuffer b, int n) {
        return ((FakeGpuBuffer) b).getInts(0, n);
    }

    /** The device kv_pos, page tables and per-sequence kv_pos of the current step against the whole rewrites. */
    private void assertDeviceMatchesWholeWrites(DeviceModelMemory mem, PagedStep step) {
        DeviceStepInputs in = mem.stepInputs();
        for (int g = 0; g < in.groups(); g++) {
            int rows = mem.plan().group(g).rows();
            int[] device = ints(in.kvPos(g), rows);
            for (int r = 0; r < rows; r++) {
                assertEquals("group " + g + " kv_pos row " + r, in.positionOfRow(g, r), device[r]);
            }
            assertArrayEquals("group " + g + " page table", expectedTables[g],
                    ints(in.pageTable(g), expectedTables[g].length));
            for (int s = 0; reserved && s < step.nSeg(); s++) {
                int seq = step.segSeq(s);
                GpuBuffer buffer = in.sequenceKvPos(g, seq);
                sequenceBuffers[g][seq] = buffer;
                assertArrayEquals("group " + g + " sequence " + seq + " kv_pos", sequenceMask(in, step, g, seq, rows),
                        ints(buffer, rows));
            }
        }
    }

    /** The per-sequence kv_pos rebuilt from the mirror: the sequence's cells at their positions, -1 elsewhere. */
    private static int[] sequenceMask(DeviceStepInputs in, PagedStep step, int g, int seq, int rows) {
        int[] mask = new int[rows];
        Arrays.fill(mask, -1);
        int[] cells = new int[step.maxPages() * step.pageSize()];
        int slots = step.seqCells(g, seq, cells);
        for (int j = 0; j < slots; j++) {
            mask[cells[j]] = in.positionOfRow(g, cells[j]);
        }
        return mask;
    }

    /** WRITE operations into {@code dst} since the last log clear: {@code [offset ints, length ints]} each. */
    private List<long[]> writesInto(GpuBuffer dst) {
        List<long[]> out = new ArrayList<>();
        for (FakeOp op : backend.ops()) {
            if (op.kind() == FakeOpKind.WRITE && op.buffer() == dst) {
                out.add(new long[]{op.offset() / Integer.BYTES, op.bytes() / Integer.BYTES});
            }
        }
        return out;
    }

    private void reserveAddressing(DeviceModelMemory mem) {
        reserved = true;
        for (int g = 0; g < mem.stepInputs().groups(); g++) {
            mem.stepInputs().reserveSequenceAddressing(g);
        }
    }

    @Test
    public void everyOperationLeavesTheDeviceAsAWholeRewriteWould() {
        try (DeviceModelMemory mem = allocate(windowPlan())) {
            PagedSequences paged = mem.pagedSequences();
            reserveAddressing(mem);
            run(mem, new int[]{0, 6, 0}, new int[]{1, 5, 0}, new int[]{2, 7, 0});
            int[] next = {6, 5, 7};
            for (int step = 0; step < 6; step++) {
                run(mem, new int[]{0, 1, next[0]++}, new int[]{1, 1, next[1]++}, new int[]{2, 1, next[2]++});
            }
            // sequence 1 sits the step out: sequence 2 moves to launch row 1
            run(mem, new int[]{0, 1, next[0]++}, new int[]{2, 1, next[2]++});
            // a tail removal, then sequence 1 continues from the cut in launch row 0
            paged.remove(1, 8, -1);
            next[1] = 8;
            run(mem, new int[]{1, 1, next[1]++}, new int[]{2, 1, next[2]++});
            // sequence 2 leaves; sequence 1 is emptied and copies sequence 0's prefix (a copy-on-write page)
            paged.remove(2, 0, -1);
            paged.remove(1, 0, -1);
            paged.copy(0, 1, 0, 9);
            for (PageCopy c : paged.pendingCopies()) {
                mem.copyPage(c);
            }
            paged.releaseCopies(paged.pendingCopies().toArray(new PageCopy[0]));
            next[1] = 9;
            run(mem, new int[]{0, 1, next[0]++}, new int[]{1, 2, next[1]});
            next[1] += 2;
            // sequence 2 joins again with a prompt longer than the window
            run(mem, new int[]{0, 1, next[0]++}, new int[]{1, 1, next[1]++}, new int[]{2, 9, 0});
            // a session restore installs a mirror
            mem.installPagedMirror(mem.pagedMirrorSnapshot());
            run(mem, new int[]{0, 1, next[0]++}, new int[]{2, 1, 9});
            mem.reset();
            run(mem, new int[]{1, 3, 0});
        }
    }

    @Test
    public void aContextShiftReachesTheDeviceThroughTheMarkedRows() {
        try (DeviceModelMemory mem = allocate(fullPlan(32, 4, 2, 16))) {
            reserveAddressing(mem);
            run(mem, new int[]{0, 8, 0}, new int[]{1, 3, 0});
            run(mem, new int[]{0, 4, 8}, new int[]{1, 1, 3});
            mem.shiftSequence(0, 4, 6);
            mem.moveShiftedRows();
            run(mem, new int[]{0, 1, 6}, new int[]{1, 1, 4});
        }
    }

    @Test
    public void aDecodeStepWritesOnlyTheRowsItAppendsAndNoPageTableEntry() {
        int ctx = 1024;
        int page = 16;
        int parallel = 4;
        try (DeviceModelMemory mem = allocate(fullPlan(ctx, page, parallel, 256))) {
            int[] next = new int[parallel];
            for (int seq = 0; seq < parallel; seq++) {
                run(mem, new int[]{seq, 200, 0});
                next[seq] = 200;
            }
            // the first step over all four sequences places them in launch rows 0..3; the next one stays on
            // every sequence's last page (rows 192..207 of its list), so it writes the appended rows only
            run(mem, new int[]{0, 1, next[0]++}, new int[]{1, 1, next[1]++}, new int[]{2, 1, next[2]++},
                    new int[]{3, 1, next[3]++});
            backend.clearLog();
            run(mem, new int[]{0, 1, next[0]++}, new int[]{1, 1, next[1]++}, new int[]{2, 1, next[2]++},
                    new int[]{3, 1, next[3]++});
            List<long[]> kvWrites = writesInto(mem.stepInputs().kvPos(0));
            assertEquals("one write per sequence's appended row", parallel, kvWrites.size());
            for (long[] w : kvWrites) {
                assertEquals(1, w[1]);
            }
            assertEquals("no page-table entry changed", 0, writesInto(mem.stepInputs().pageTable(0)).size());
            // seven more rows fill the page; the step at 208: the step that crosses it writes one entry per
            // sequence
            for (int step = 0; step < 6; step++) {
                run(mem, new int[]{0, 1, next[0]++}, new int[]{1, 1, next[1]++}, new int[]{2, 1, next[2]++},
                        new int[]{3, 1, next[3]++});
            }
            backend.clearLog();
            run(mem, new int[]{0, 1, next[0]++}, new int[]{1, 1, next[1]++}, new int[]{2, 1, next[2]++},
                    new int[]{3, 1, next[3]++});
            List<long[]> tableWrites = writesInto(mem.stepInputs().pageTable(0));
            assertEquals(parallel, tableWrites.size());
            for (long[] w : tableWrites) {
                assertEquals("the new page's entry alone", 1, w[1]);
            }
        }
    }

    @Test
    public void aWideChangeWritesOneSpanInsteadOfManyRuns() {
        int ctx = 1024;
        try (DeviceModelMemory mem = allocate(fullPlan(ctx, 16, 2, 512))) {
            PagedSequences paged = mem.pagedSequences();
            run(mem, new int[]{0, 400, 0});
            run(mem, new int[]{1, 300, 0});
            paged.remove(0, 0, -1);
            backend.clearLog();
            run(mem, new int[]{1, 1, 300});
            List<long[]> kvWrites = writesInto(mem.stepInputs().kvPos(0));
            assertEquals("the cleared rows and the appended one share a write", 1, kvWrites.size());
            assertTrue("the write covers the 400 cleared rows", kvWrites.get(0)[1] >= 400);
        }
    }

    @Test
    public void anUnchangedSequenceWritesNothingAndItsTrunkIsMarkedNotReplayable() {
        try (DeviceModelMemory mem = allocate(fullPlan(1024, 16, 2, 512))) {
            DeviceStepInputs in = mem.stepInputs();
            assertFalse("no group addresses its sequences yet", in.pagedTrunkWrites());
            reserveAddressing(mem);
            assertTrue("a reserved group patches its sequences inside the trunk", in.pagedTrunkWrites());
            run(mem, new int[]{0, 30, 0}, new int[]{1, 20, 0});
            // the last row is rolled back and decoded again into the same cell at the same position
            mem.pagedSequences().remove(0, 29, -1);
            backend.clearLog();
            run(mem, new int[]{0, 1, 29});
            assertEquals("a sequence whose cells kept their positions writes nothing", 0,
                    writesInto(sequenceBuffers[0][0]).size());
        }
    }

    @Test
    public void aSequenceAddressingUpdatePatchesTheChangedCellsAlone() {
        int ctx = 1024;
        try (DeviceModelMemory mem = allocate(fullPlan(ctx, 16, 2, 512))) {
            reserveAddressing(mem);
            run(mem, new int[]{0, 300, 0}, new int[]{1, 200, 0});
            backend.clearLog();
            run(mem, new int[]{0, 1, 300}, new int[]{1, 1, 200});
            for (int seq = 0; seq < 2; seq++) {
                List<long[]> writes = writesInto(sequenceBuffers[0][seq]);
                assertEquals("sequence " + seq + " patches its appended cell alone", 1, writes.size());
                assertEquals(1, writes.get(0)[1]);
            }
        }
    }
}
