package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The device memory's paged context shift ({@code DeviceModelMemory.shiftSequence}, llama.cpp server's slot shift)
 * against the host-only fake backend: the discarded interior rows leave, the survivors' keys and values carry to
 * their compacted cells, every moved cell's RoPE delta lands in the group's {@code shift_delta} buffer at the cell
 * the key still sits in, the page copies release once executed, the token record and the position mirrors follow,
 * and the next step reserves and uploads over the compacted pages. The contiguous form refuses the per-sequence
 * shift and the paged form still refuses {@code shiftPositions}.
 */
public class DevicePagedShiftTest {

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

    /** One F32 KV group of {@code N_CTX} cells in pages of 4, two sequences, full-attention. */
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

    /** The single-segment step of {@code seq}: {@code len} rows from {@code pos0}, the last row an output row. */
    private static PagedStep stepOne(DeviceModelMemory mem, int seq, int len, int pos0) {
        StepLayout layout = StepLayout.of(len, new int[]{seq}, new int[]{0}, new int[]{len},
                new int[]{pos0}, new int[0], true);
        return mem.pagedSequences().prepare(layout);
    }

    /** Reserves, uploads, ends and commits one step, then records the fixture tokens of its rows. */
    private static void runStep(DeviceModelMemory mem, int seq, int len, int pos0) {
        PagedStep step = stepOne(mem, seq, len, pos0);
        mem.beginPagedStep(step, rowsOf(step), null);
        mem.endPagedStep();
        mem.pagedSequences().commit(step);
        int[] tokens = new int[len];
        for (int t = 0; t < len; t++) {
            tokens[t] = 10 + pos0 + t;
        }
        mem.pagedSequences().appendTokens(seq, tokens);
    }

    private static int[] rowsOf(PagedStep step) {
        int[] rows = new int[step.nTokens()];
        for (int t = 0; t < step.nTokens(); t++) {
            rows[t] = step.segPos0(0) + t;
        }
        return rows;
    }

    /** Distinct K and V values of row {@code cell}: the row's cell index in every component. */
    private static float[] rowValues(int cell) {
        float[] v = new float[WIDTH];
        for (int d = 0; d < WIDTH; d++) {
            v[d] = 100f + cell * 10f + d;
        }
        return v;
    }

    private static void writeRow(DeviceModelMemory mem, int cell) {
        DeviceKvLayer kv = (DeviceKvLayer) mem.kv(0);
        ((FakeGpuBuffer) kv.k()).putFloats((long) cell * WIDTH, rowValues(cell));
        ((FakeGpuBuffer) kv.v()).putFloats((long) cell * WIDTH, rowValues(cell));
    }

    private static float[] readRow(DeviceModelMemory mem, int cell, boolean v) {
        DeviceKvLayer kv = (DeviceKvLayer) mem.kv(0);
        FakeGpuBuffer buf = (FakeGpuBuffer) (v ? kv.v() : kv.k());
        return buf.getFloats((long) cell * WIDTH, WIDTH);
    }

    /** Cell of stored position {@code pos}, from the memory's mirror. */
    private static int cellOf(DeviceModelMemory mem, int pos) {
        for (int c = 0; c < N_CTX; c++) {
            if (mem.stepInputs().positionOfRow(0, c) == pos) {
                return c;
            }
        }
        return -1;
    }

    /** Cell holding each position of sequence 0, before a shift. */
    private static Map<Integer, Integer> cellsOfPositions(DeviceModelMemory mem, int nPast) {
        Map<Integer, Integer> cells = new HashMap<>();
        for (int pos = 0; pos < nPast; pos++) {
            cells.put(pos, cellOf(mem, pos));
        }
        return cells;
    }

    @Test
    public void theShiftMovesTheRowsAndRecordsTheDeltas() {
        try (DeviceModelMemory mem = allocate()) {
            runStep(mem, 0, 8, 0);
            runStep(mem, 0, 8, 8);
            runStep(mem, 1, 4, 0);
            for (int c = 0; c < N_CTX; c++) {
                if (mem.stepInputs().positionOfRow(0, c) >= 0) {
                    writeRow(mem, c);
                }
            }
            Map<Integer, Integer> before = cellsOfPositions(mem, 16);

            // keeps positions [0, 4), drops [4, 10), moves [10, 16) down by 6
            mem.shiftSequence(0, 4, 6);
            mem.moveShiftedRows();

            assertEquals("the survivors end at position 9", 9, mem.sequences().posMax(0));
            assertEquals("the kept prefix stays", 0, mem.sequences().posMin(0));
            assertEquals("the other sequence keeps its rows", 3, mem.sequences().posMax(1));
            int[] tokens = mem.sequences().sequence(0).tokens();
            assertEquals("the token record compacts with the rows", 10, tokens.length);
            assertEquals("row 9 carries the old row 15's token", 10 + 15, tokens[9]);
            assertEquals("row 3 carries the old row 3's token", 10 + 3, tokens[3]);

            // every survivor's key and value sit at the cell of its new position, carried verbatim
            for (int pos = 0; pos < 10; pos++) {
                int cell = cellOf(mem, pos);
                assertTrue("position " + pos + " lost its cell", cell >= 0);
                int src = pos < 4 ? before.get(pos) : before.get(pos + 6);
                assertArrayEquals("K of position " + pos, rowValues(src), readRow(mem, cell, false), 0f);
                assertArrayEquals("V of position " + pos, rowValues(src), readRow(mem, cell, true), 0f);
            }
            // the discarded rows' old cells are free again
            for (int pos = 10; pos < 16; pos++) {
                assertEquals("position " + pos + " left the memory", -1, cellOf(mem, pos));
            }

            // the deltas name the cells the shifted keys sat in when the shift ran
            FakeGpuBuffer delta = (FakeGpuBuffer) mem.shiftDelta(0);
            int[] deltas = delta.getInts(0, N_CTX);
            int shiftedCells = 0;
            for (int c = 0; c < N_CTX; c++) {
                shiftedCells += deltas[c] != 0 ? 1 : 0;
                if (deltas[c] != 0) {
                    assertEquals("every shifted cell carries the move's delta", -6, deltas[c]);
                }
            }
            assertEquals("exactly the six shifted rows carry a delta", 6, shiftedCells);
            for (int pos = 10; pos < 16; pos++) {
                assertEquals("delta of the shifted row " + pos, -6, deltas[before.get(pos)]);
            }
            for (int pos = 0; pos < 4; pos++) {
                assertEquals("the kept row " + pos + " carries no delta", 0, deltas[before.get(pos)]);
            }

            // the other sequence's keys and values stay where they were
            Set<Integer> ownCells = new HashSet<>(before.values());
            for (int c = 0; c < N_CTX; c++) {
                int pos = mem.stepInputs().positionOfRow(0, c);
                if (pos >= 0 && pos < 4 && !ownCells.contains(c)) {
                    assertArrayEquals("sequence B keeps its key at cell " + c, rowValues(c),
                            readRow(mem, c, false), 0f);
                    assertArrayEquals("sequence B keeps its value at cell " + c, rowValues(c),
                            readRow(mem, c, true), 0f);
                }
            }

            assertEquals("the shift's page copies released", 0, mem.pagedSequences().pendingCopies().size());
            assertEquals("the shift's row moves ran", 0, mem.pagedSequences().pendingRowMoves().size());
        }
    }

    @Test
    public void theNextStepReservesOverTheCompactedPages() {
        try (DeviceModelMemory mem = allocate()) {
            runStep(mem, 0, 8, 0);
            runStep(mem, 0, 8, 8);
            runStep(mem, 1, 4, 0);
            mem.shiftSequence(0, 4, 6);
            mem.moveShiftedRows();

            // the next decode row of sequence 0 continues at position 10
            PagedStep step = stepOne(mem, 0, 1, 10);
            mem.beginPagedStep(step, rowsOf(step), null);
            mem.endPagedStep();
            mem.pagedSequences().commit(step);
            assertEquals("the decode row stores at position 10", 10,
                    mem.stepInputs().positionOfRow(0, cellOf(mem, 10)));
            assertEquals(10, mem.sequences().posMax(0));
        }
    }

    @Test
    public void theDivisionHalvesThePositionsAndRecordsEachRowsOwnDelta() {
        try (DeviceModelMemory mem = allocate()) {
            int[] own = new int[8];
            PagedStep first = stepOne(mem, 0, 8, 0);
            System.arraycopy(first.cellIdx(0), 0, own, 0, 8);
            runStep(mem, 0, 8, 0);
            runStep(mem, 1, 4, 0);
            for (int pos = 0; pos < 8; pos++) {
                writeRow(mem, own[pos]);
            }

            mem.divSequence(0, 0, -1, 2);
            mem.moveShiftedRows();

            assertEquals("the divided positions end at 3", 3, mem.sequences().posMax(0));
            assertEquals("every row stays stored", 8, mem.sequences().sequence(0).nPast());
            for (int pos = 0; pos < 8; pos++) {
                assertEquals("row " + pos + " now stores position " + (pos / 2), pos / 2,
                        mem.sequences().sequence(0).position(pos));
            }
            assertEquals("the other sequence keeps its positions", 3, mem.sequences().posMax(1));

            // the rows keep their cells and their keys: a division re-rotates in place, it moves nothing
            for (int pos = 0; pos < 8; pos++) {
                assertArrayEquals("K of row " + pos + " stays at its cell", rowValues(own[pos]),
                        readRow(mem, own[pos], false), 0f);
                assertArrayEquals("V of row " + pos + " stays at its cell", rowValues(own[pos]),
                        readRow(mem, own[pos], true), 0f);
            }
            assertEquals("a division moves no row", 0, mem.pagedSequences().pendingRowMoves().size());

            // each moved key's cell carries its own shift, old - old / 2
            FakeGpuBuffer delta = (FakeGpuBuffer) mem.shiftDelta(0);
            int[] deltas = delta.getInts(0, N_CTX);
            for (int pos = 0; pos < 8; pos++) {
                assertEquals("delta of the row at position " + pos, pos - pos / 2, deltas[own[pos]]);
            }
        }
    }

    @Test
    public void theDivisionRefusesADivisorBelowTwoByName() {
        try (DeviceModelMemory mem = allocate()) {
            runStep(mem, 0, 4, 0);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> mem.divSequence(0, 0, -1, 1));
            assertTrue(e.getMessage(), e.getMessage().contains("divisor"));
            assertEquals(3, mem.sequences().posMax(0));
        }
    }

    @Test
    public void theShiftRefusesOutOfRangeCountsAndSequencesByName() {
        try (DeviceModelMemory mem = allocate()) {
            runStep(mem, 0, 8, 0);
            IllegalArgumentException counts = assertThrows(IllegalArgumentException.class,
                    () -> mem.shiftSequence(0, -1, 2));
            assertTrue(counts.getMessage(), counts.getMessage().contains("nKeep"));
            counts = assertThrows(IllegalArgumentException.class, () -> mem.shiftSequence(0, 2, 0));
            assertTrue(counts.getMessage(), counts.getMessage().contains("nDiscard"));
            IndexOutOfBoundsException seq = assertThrows(IndexOutOfBoundsException.class,
                    () -> mem.shiftSequence(5, 0, 1));
            assertTrue(seq.getMessage(), seq.getMessage().contains("sequence 5"));
        }
    }

    @Test
    public void thePagedMemoryRefusesTheContiguousShiftByName() {
        try (DeviceModelMemory mem = allocate()) {
            runStep(mem, 0, 8, 0);
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> mem.shiftPositions(0, 2));
            assertTrue(e.getMessage(), e.getMessage().contains("paged memory maps positions"));
        }
    }

    /** One F32 KV layer on a full contiguous cache of {@code nCtx} rows. */
    private static MemoryPlan fullPlan(int nCtx) {
        KvLayout full = KvLayout.of(nCtx, false, WIDTH, WIDTH, KvDType.F32, KvDType.F32, KvWindow.FULL_CAUSAL);
        return MemoryPlan.builder(MemoryKind.KV, nCtx, UBATCH)
                .group(MemoryPlan.KvGroup.of(0, nCtx, false, KvWindow.FULL_CAUSAL))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(full, 0).build())
                .build();
    }

    @Test
    public void theContiguousMemoryRefusesThePerSequenceShiftByName() {
        try (DeviceModelMemory mem = DeviceModelMemory.allocate(backend, backend.computeQueue(), fullPlan(16),
                MropeTextPositions.PPP0, null)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> mem.shiftSequence(0, 0, 2));
            assertTrue(e.getMessage(), e.getMessage().contains("shiftPositions"));
        }
    }
}
