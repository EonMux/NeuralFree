package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PageCopy;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The device paged step ({@code DeviceModelMemory.beginPagedStep}): the upload applies the page bookkeeping's
 * queued {@code kv_pos} changes and the step's row writes to the mirrors, then writes the whole mirror to the
 * device {@code kv_pos}, the step's {@code cell_idx}, and the paged kernels' addressing — the per-group page table
 * compacted to launch order (segment {@code s} at table row {@code s}), the per-sequence slot counts, the
 * per-sequence query positions and the per-row query / RoPE positions. Page copies enqueue device row copies, a
 * removal's reports reach the device {@code kv_pos} with the next step, reset empties the paged state, and
 * {@code shiftPositions} refuses by name on a paged memory.
 */
public class DevicePagedStepInputsTest {

    private static final int PAGE = 4;
    private static final int N_CTX = 32;
    private static final int PARALLEL = 2;
    private static final int UBATCH = 8;
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

    /** One F32 KV group of {@code N_CTX} rows in pages of 4, two sequences, full-attention. */
    private static MemoryPlan plan() {
        MemoryConfig cfg = MemoryConfig.builder()
                .typeK(KvDType.F32).typeV(KvDType.F32)
                .nCtx(N_CTX).ubatch(UBATCH).parallel(PARALLEL).kvPageSize(PAGE)
                .build();
        MemoryPlan.KvGroup g = MemoryPlan.KvGroup.of(0, N_CTX, false, KvWindow.FULL_CAUSAL, PAGE, N_CTX / PAGE);
        KvLayout kv = KvLayout.of(N_CTX, false, WIDTH, WIDTH, KvDType.F32, KvDType.F32, KvWindow.FULL_CAUSAL);
        return MemoryPlan.builder(MemoryKind.KV, N_CTX, UBATCH)
                .group(g)
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(kv, 0).build())
                .config(cfg)
                .build();
    }

    private DeviceModelMemory allocate() {
        return DeviceModelMemory.allocate(backend, backend.computeQueue(), plan(), MropeTextPositions.PPP0, null);
    }

    /** The step of one segment per sequence: rows ordered by sequence, positions from {@code posA} / {@code posB}. */
    private static PagedStep step(DeviceModelMemory mem, int lenA, int posA, int lenB, int posB) {
        PagedSequences paged = mem.pagedSequences();
        int n = lenA + lenB;
        StepLayout layout = StepLayout.of(n, new int[]{0, 1}, new int[]{0, lenA}, new int[]{lenA, lenB},
                new int[]{posA, posB}, new int[0], true);
        return paged.prepare(layout);
    }

    /** The single-segment step of {@code seq}: {@code len} rows from {@code pos0}. */
    private static PagedStep stepOne(DeviceModelMemory mem, int seq, int len, int pos0) {
        StepLayout layout = StepLayout.of(len, new int[]{seq}, new int[]{0}, new int[]{len},
                new int[]{pos0}, new int[0], true);
        return mem.pagedSequences().prepare(layout);
    }

    /** Reserves, uploads, ends and commits one step, then records its tokens as the backend does. */
    private static void runStep(DeviceModelMemory mem, int lenA, int posA, int lenB, int posB) {
        PagedStep step = step(mem, lenA, posA, lenB, posB);
        mem.beginPagedStep(step, rowsOf(step), null);
        mem.endPagedStep();
        mem.pagedSequences().commit(step);
        record(mem, step, 0, new int[]{1, 2, 3, 4, 5, 6, 7, 8});
    }

    /** Appends the first {@code n} fixture tokens of each segment to its sequence's record. */
    private static void record(DeviceModelMemory mem, PagedStep step, int from, int[] tokens) {
        PagedSequences paged = mem.pagedSequences();
        for (int seg = 0; seg < step.nSeg(); seg++) {
            int len = Math.min(step.segLen(seg), tokens.length - from);
            int[] segTokens = new int[len];
            for (int t = 0; t < len; t++) {
                segTokens[t] = tokens[from + t];
            }
            paged.appendTokens(step.segSeq(seg), segTokens);
        }
    }

    private static int[] rowsOf(PagedStep step) {
        int[] rows = new int[step.nTokens()];
        int k = 0;
        for (int seg = 0; seg < step.nSeg(); seg++) {
            for (int t = 0; t < step.segLen(seg); t++) {
                rows[k++] = step.segPos0(seg) + t;
            }
        }
        return rows;
    }

    private static FakeGpuBuffer fake(GpuBuffer b) {
        return (FakeGpuBuffer) b;
    }

    private static int[] ints(GpuBuffer b, int n) {
        return fake(b).getInts(0, n);
    }

    private static int[] ints(GpuBuffer b, int n, int from) {
        return fake(b).getInts(from, n);
    }

    private int copyOps() {
        int copies = 0;
        for (FakeComputeBackend.Op op : backend.ops()) {
            if (op.kind() == FakeComputeBackend.OpKind.COPY) {
                copies++;
            }
        }
        return copies;
    }

    @Test
    public void aPagedPlanBuildsThePagedBookkeepingAndRefusesTheContiguousShift() {
        try (DeviceModelMemory mem = allocate()) {
            PagedSequences paged = mem.pagedSequences();
            assertNotNull(paged);
            assertEquals(PARALLEL, paged.maxSequences());
            assertEquals(PAGE, paged.pageSize());
            assertEquals(PARALLEL, mem.sequences().maxSequences());
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> mem.shiftPositions(1, 1));
            assertTrue(e.getMessage(), e.getMessage().contains("paged"));
        }
    }

    @Test
    public void beginPagedStepUploadsTheCompactedTablesAndThePerSequenceAddressing() {
        try (DeviceModelMemory mem = allocate()) {
            PagedSequences paged = mem.pagedSequences();
            PagedStep step = step(mem, 5, 0, 3, 0);
            mem.beginPagedStep(step, rowsOf(step), null);
            try {
                int[] table = step.pageTable(0);
                int maxPages = paged.maxPages();
                // launch order: segment s's table at device page-table row s
                int[] expected = new int[maxPages * PARALLEL];
                Arrays.fill(expected, -1);
                System.arraycopy(table, 0, expected, 0, maxPages);
                System.arraycopy(table, maxPages, expected, maxPages, maxPages);
                assertArrayEquals(expected, ints(mem.stepInputs().pageTable(0), maxPages * PARALLEL));
                // seq_kv_len per launch sequence, q_pos_seq = the segments' first positions
                assertArrayEquals(new int[]{5, 3}, ints(mem.stepInputs().seqKvLen(0), PARALLEL));
                assertArrayEquals(new int[]{0, 0}, ints(mem.stepInputs().qPosSeq(), PARALLEL));
                // cell_idx of the whole ubatch
                assertArrayEquals(step.cellIdx(0), ints(mem.stepInputs().cellIdx(0), step.nTokens()));
                // kv_pos: every cell the step writes carries its position, the rest -1
                int[] kvPos = ints(mem.stepInputs().kvPos(0), N_CTX);
                for (int g = 0; g < step.groups(); g++) {
                    int[] rows = step.kvPosRows(g);
                    int[] values = step.kvPosValues(g);
                    for (int i = 0; i < rows.length; i++) {
                        assertEquals(values[i], kvPos[rows[i]]);
                    }
                }
                // per-row query positions and the section-major RoPE positions of each segment
                assertArrayEquals(rowsOf(step), ints(mem.stepInputs().qPos(), step.nTokens()));
                int[] rope = ints(mem.stepInputs().ropePos(), MropeTextPositions.SECTIONS * step.nTokens());
                int[] expectedRope = new int[rope.length];
                for (int seg = 0; seg < step.nSeg(); seg++) {
                    for (int t = 0; t < step.segLen(seg); t++) {
                        int p = step.segPos0(seg) + t;
                        int row = step.segStart(seg) + t;
                        for (int s = 0; s < MropeTextPositions.SECTIONS; s++) {
                            expectedRope[s * step.nTokens() + row] = s < 3 ? p : 0;
                        }
                    }
                }
                assertArrayEquals(expectedRope, rope);
            } finally {
                mem.endPagedStep();
                paged.commit(step);
            }
        }
    }

    @Test
    public void aCopyOnWritePageCopyRunsOnTheDeviceAndRebasesTheCompactedTable() {
        try (DeviceModelMemory mem = allocate()) {
            PagedSequences paged = mem.pagedSequences();
            runStep(mem, 6, 0, 1, 0);
            // sequence 1 drops its row, then copies positions [2, posMax] of sequence 0: the leading page holds
            // rows below the range, so it privatizes (one PageCopy), the trailing page shares
            paged.remove(1, 0, -1);
            paged.copy(0, 1, 2, -1);
            assertEquals(1, paged.pendingCopies().size());
            PageCopy c = paged.pendingCopies().get(0);
            int before = copyOps();
            mem.copyPage(c);
            // K and V are separate F32 buffers: one device copy each
            assertEquals(before + 2, copyOps());
            paged.releaseCopies(c);
            // the next step of the private sequence appends into its own page table (launch row 0 = the segment)
            PagedStep next = stepOne(mem, 1, 1, 6);
            mem.beginPagedStep(next, rowsOf(next), null);
            try {
                int[] table = next.pageTable(0);
                int maxPages = paged.maxPages();
                // the device's launch row 0 carries the step's only segment: sequence 1's table
                assertArrayEquals(Arrays.copyOfRange(table, maxPages, 2 * maxPages),
                        ints(mem.stepInputs().pageTable(0), maxPages));
                assertEquals(c.dstPage(), table[maxPages]);
                // seq 1 spans its private page and the shared one: slots = the appended rows plus the copies
                // (only the launch's rows are meaningful; row 1 keeps its stale value between launches)
                assertArrayEquals(new int[]{7}, ints(mem.stepInputs().seqKvLen(0), 1, 0));
            } finally {
                mem.endPagedStep();
                paged.commit(next);
            }
        }
    }

    @Test
    public void theMirrorCarriesTheBookkeepingReportsIntoTheDeviceKvPos() {
        try (DeviceModelMemory mem = allocate()) {
            PagedSequences paged = mem.pagedSequences();
            runStep(mem, 4, 0, 1, 0);
            // dropping the whole of sequence 0 frees its pages and queues clear reports
            paged.remove(0, 0, -1);
            assertTrue(paged.pendingReports().size() > 0);
            PagedStep next = stepOne(mem, 1, 1, 1);
            mem.beginPagedStep(next, rowsOf(next), null);
            try {
                int[] kvPos = ints(mem.stepInputs().kvPos(0), N_CTX);
                int live = 0;
                for (int p : kvPos) {
                    if (p >= 0) {
                        live++;
                    }
                }
                // sequence 0's rows are empty; sequence 1 keeps its first row and the step adds the second
                assertEquals(2, live);
            } finally {
                mem.endPagedStep();
                paged.commit(next);
            }
        }
    }

    @Test
    public void resetEmptiesThePagedStateAndTheDeviceKvPos() {
        try (DeviceModelMemory mem = allocate()) {
            PagedSequences paged = mem.pagedSequences();
            runStep(mem, 4, 0, 1, 0);
            mem.reset();
            assertEquals(-1, paged.posMax(0));
            assertEquals(-1, paged.posMax(1));
            for (int p : ints(mem.stepInputs().kvPos(0), N_CTX)) {
                assertEquals(-1, p);
            }
        }
    }

    @Test
    public void aMalformedUploadRefusesByName() {
        try (DeviceModelMemory mem = allocate()) {
            PagedStep step = step(mem, 2, 0, 1, 0);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> mem.beginPagedStep(step, new int[]{0, 1}, null));
            assertTrue(e.getMessage(), e.getMessage().contains("position"));
        }
    }
}
