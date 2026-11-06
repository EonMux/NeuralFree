package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.StepLayout;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The three split modes of {@link BatchSplitter} (llama.cpp src/llama-batch.cpp:585, 619, 790) and the equal
 * ubatch sizes a prompt's rows split into ({@link StepSizes}).
 */
public final class BatchSplitterTest {

    private static ForwardBatch oneSequence(int n, int pos0, int seq, boolean outputTail) {
        ForwardBatch.Builder b = ForwardBatch.builder(n);
        for (int i = 0; i < n; i++) {
            b.add(100 + i, pos0 + i, seq, outputTail && i == n - 1);
        }
        return b.build();
    }

    /** A media chunk's rows share the chunk's base position; they stay one segment per ubatch run. */
    private static ForwardBatch mediaRows(int n, int pos0, int seq) {
        ForwardBatch.Builder b = ForwardBatch.builder(n);
        for (int i = 0; i < n; i++) {
            b.addEmbedding(new float[]{1f + i}, pos0, seq, false);
        }
        return b.build();
    }

    @Test
    public void sharedMediaPositionsStayOneSegmentAcrossSplitUbatchRuns() {
        BatchSplitter splitter = new BatchSplitter(true);
        List<UBatch> ubatches = splitter.split(mediaRows(5, 7, 0), 2, BatchSplitter.Mode.SEQ);
        assertEquals(3, ubatches.size());
        assertArrayEquals(new int[]{0, 1}, ubatches.get(0).batchRows());
        assertArrayEquals(new int[]{7, 7}, ubatches.get(0).positions());
        assertArrayEquals(new int[]{2, 3}, ubatches.get(1).batchRows());
        assertArrayEquals(new int[]{7, 7}, ubatches.get(1).positions());
        assertArrayEquals(new int[]{4}, ubatches.get(2).batchRows());
        assertArrayEquals(new int[]{7}, ubatches.get(2).positions());
        for (UBatch u : ubatches) {
            StepLayout layout = u.layout();
            assertEquals(1, layout.nSeg());
            assertEquals(7, layout.pos0(0));
            assertFalse(layout.contiguousSingle());
        }
    }

    @Test
    public void simpleSplitsInBatchOrder() {
        BatchSplitter splitter = new BatchSplitter();
        List<UBatch> ubatches = splitter.split(oneSequence(5, 3, 0, true), 2, BatchSplitter.Mode.SIMPLE);
        assertEquals(3, ubatches.size());
        assertArrayEquals(new int[]{0, 1}, ubatches.get(0).batchRows());
        assertArrayEquals(new int[]{2, 3}, ubatches.get(1).batchRows());
        assertArrayEquals(new int[]{4}, ubatches.get(2).batchRows());
        assertArrayEquals(new int[]{100, 101}, ubatches.get(0).tokens());
        assertArrayEquals(new int[]{3, 4}, ubatches.get(0).positions());
        assertArrayEquals(new int[]{102, 103}, ubatches.get(1).tokens());
        assertArrayEquals(new int[]{104}, ubatches.get(2).tokens());
        StepLayout layout = ubatches.get(0).layout();
        assertEquals(1, layout.nSeg());
        assertEquals(0, layout.seq(0));
        assertEquals(2, layout.nTokens());
        assertEquals(3, layout.pos0(0));
        assertTrue(layout.contiguousSingle());
        assertArrayEquals(new int[0], layout.outputRows()); // the tail token runs in the third ubatch
        assertArrayEquals(new int[]{0}, ubatches.get(2).layout().outputRows());
    }

    @Test
    public void simpleMixedSequencesOrderRowsBySequenceThenPosition() {
        ForwardBatch batch = ForwardBatch.builder(4)
                .add(1, 0, 0, false) // seq 0 pos 0
                .add(2, 1, 0, false) // seq 0 pos 1
                .add(3, 0, 1, false) // seq 1 pos 0
                .add(4, 2, 0, true)  // seq 0 pos 2
                .build();
        List<UBatch> ubatches = new BatchSplitter().split(batch, 2, BatchSplitter.Mode.SIMPLE);
        assertEquals(2, ubatches.size());
        // chunk 1: rows 0,1 (one sequence)
        assertArrayEquals(new int[]{0, 1}, ubatches.get(0).batchRows());
        // chunk 2: rows 2,3 reordered to seq then position: seq 0 pos 2 first, then seq 1 pos 0
        assertArrayEquals(new int[]{3, 2}, ubatches.get(1).batchRows());
        StepLayout layout = ubatches.get(1).layout();
        assertEquals(2, layout.nSeg());
        assertEquals(0, layout.seq(0)); // segments ascend by sequence id
        assertEquals(1, layout.seq(1));
        assertEquals(1, layout.len(0));
        assertEquals(1, layout.len(1));
        assertEquals(2, layout.pos0(0));
        assertEquals(0, layout.pos0(1));
        assertArrayEquals(new int[]{0}, layout.outputRows()); // output row is the first row after the reorder
        assertFalse(layout.contiguousSingle()); // two segments
    }

    @Test
    public void simpleKeepsBothOutputRowsWhenTheReorderInvertsTheBatchOrder() {
        // seq 1's decode row is batch row 0 and seq 0's rows are batch rows 1, 2: the ubatch orders them
        // seq 0 first, so the output rows' batch order is the reverse of their ubatch order — every output
        // row must still reach the layout (the engine schedules such steps beside chunked prefills)
        ForwardBatch batch = ForwardBatch.builder(3)
                .add(9, 5, 1, true)
                .add(1, 0, 0, false)
                .add(2, 1, 0, true)
                .build();
        List<UBatch> ubatches = new BatchSplitter().split(batch, 4, BatchSplitter.Mode.SIMPLE);
        assertEquals(1, ubatches.size());
        assertArrayEquals(new int[]{1, 2, 0}, ubatches.get(0).batchRows());
        StepLayout layout = ubatches.get(0).layout();
        assertEquals(2, layout.nSeg());
        assertArrayEquals(new int[]{1, 2}, layout.outputRows()); // both output rows, in ubatch order
    }

    @Test
    public void aPromptsRowsSplitIntoEqualUbatchSizes() {
        // 20 rows of one sequence at a ubatch of 8: 7 / 7 / 6, not the greedy 8 / 8 / 4
        for (BatchSplitter.Mode mode : new BatchSplitter.Mode[]{BatchSplitter.Mode.SIMPLE,
                BatchSplitter.Mode.EQUAL}) {
            List<UBatch> ubatches = new BatchSplitter(true).split(oneSequence(20, 0, 0, true), 8, mode);
            assertEquals(mode.toString(), 3, ubatches.size());
            assertArrayEquals(new int[]{0, 1, 2, 3, 4, 5, 6}, ubatches.get(0).batchRows());
            assertArrayEquals(new int[]{7, 8, 9, 10, 11, 12, 13}, ubatches.get(1).batchRows());
            assertArrayEquals(new int[]{14, 15, 16, 17, 18, 19}, ubatches.get(2).batchRows());
            assertArrayEquals(mode + ": the output row is the tail of the last ubatch",
                    new int[]{5}, ubatches.get(2).layout().outputRows());
        }
    }

    @Test
    public void aBatchOfSeveralSequencesKeepsTheGreedyUbatchSize() {
        // two sequences of five rows at a ubatch of four: split_equal takes one token per sequence per round, so
        // the ubatches are 4 / 4 / 2 — the mode's row sets decide the boundaries, not a single sequence's split
        ForwardBatch.Builder b = ForwardBatch.builder(10);
        for (int seq = 0; seq < 2; seq++) {
            for (int i = 0; i < 5; i++) {
                b.add(10 + 5 * seq + i, i, seq, false);
            }
        }
        List<UBatch> ubatches = new BatchSplitter().split(b.build(), 4, BatchSplitter.Mode.EQUAL);
        assertEquals(3, ubatches.size());
        assertArrayEquals(new int[]{0, 1, 5, 6}, ubatches.get(0).batchRows());
        assertArrayEquals(new int[]{2, 3, 7, 8}, ubatches.get(1).batchRows());
        assertArrayEquals(new int[]{4, 9}, ubatches.get(2).batchRows());
    }

    @Test
    public void equalTakesTheSameCountEverySequence() {
        // seq 0 holds positions 0..3 (batch rows 0..3), seq 1 positions 0..1 (rows 4, 5)
        ForwardBatch batch = ForwardBatch.builder(6)
                .add(1, 0, 0, false).add(2, 1, 0, false).add(3, 2, 0, true).add(4, 3, 0, false)
                .add(5, 0, 1, false).add(6, 1, 1, true)
                .build();
        List<UBatch> ubatches = new BatchSplitter().split(batch, 4, BatchSplitter.Mode.EQUAL);
        assertEquals(2, ubatches.size());
        // first ubatch: two rounds of one token per sequence
        assertArrayEquals(new int[]{0, 1, 4, 5}, ubatches.get(0).batchRows());
        StepLayout layout = ubatches.get(0).layout();
        assertEquals(2, layout.nSeg());
        assertEquals(2, layout.len(0));
        assertEquals(2, layout.len(1));
        assertEquals(2, layout.uniformSegLen());
        assertFalse(layout.contiguousSingle());
        assertArrayEquals(new int[]{3}, layout.outputRows()); // ubatch-local index of the output row 5
        // second ubatch: sequence 1 is done, sequence 0 finishes
        assertArrayEquals(new int[]{2, 3}, ubatches.get(1).batchRows());
        StepLayout tail = ubatches.get(1).layout();
        assertEquals(1, tail.nSeg()); // one sequence left
        assertEquals(2, tail.len(0));
        assertEquals(2, tail.pos0(0));
        assertFalse(tail.paged());
        assertTrue(tail.contiguousSingle()); // single segment over un-paged memory
    }

    @Test
    public void equalChunksAtTheUbatchSize() {
        ForwardBatch batch = ForwardBatch.builder(8)
                .add(1, 0, 0, false).add(2, 1, 0, false).add(3, 2, 0, true).add(4, 3, 0, false)
                .add(5, 0, 1, false).add(6, 1, 1, false).add(7, 2, 1, false).add(8, 3, 1, true)
                .build();
        List<UBatch> ubatches = new BatchSplitter().split(batch, 2, BatchSplitter.Mode.EQUAL);
        assertEquals(4, ubatches.size());
        for (UBatch ubatch : ubatches) {
            assertEquals(1, ubatch.layout().len(0)); // one token per sequence per ubatch
            assertEquals(1, ubatch.layout().len(1));
            assertEquals(2, ubatch.nTokens());
        }
        assertArrayEquals(new int[]{0, 4}, ubatches.get(0).batchRows());
        assertArrayEquals(new int[]{1, 5}, ubatches.get(1).batchRows());
        assertArrayEquals(new int[]{2, 6}, ubatches.get(2).batchRows());
        assertArrayEquals(new int[]{3, 7}, ubatches.get(3).batchRows());
    }

    @Test
    public void seqOneSequencePerUbatch() {
        // seq 0 holds positions 0..2, seq 1 positions 0..1, interleaved claim of the batch: consecutive per seq
        ForwardBatch batch = ForwardBatch.builder(5)
                .add(1, 0, 0, false)
                .add(2, 1, 0, true)
                .add(3, 2, 0, false)
                .add(4, 0, 1, false)
                .add(5, 1, 1, true)
                .build();
        List<UBatch> ubatches = new BatchSplitter().split(batch, 2, BatchSplitter.Mode.SEQ);
        assertEquals(3, ubatches.size());
        assertArrayEquals(new int[]{0, 1}, ubatches.get(0).batchRows());
        assertArrayEquals(new int[]{2}, ubatches.get(1).batchRows());
        assertArrayEquals(new int[]{3, 4}, ubatches.get(2).batchRows());
        for (UBatch ubatch : ubatches) {
            assertEquals(1, ubatch.layout().nSeg());
            assertTrue(ubatch.layout().contiguousSingle());
        }
    }

    @Test
    public void splitRefusesBadRequests() {
        BatchSplitter splitter = new BatchSplitter();
        assertThrows(IllegalArgumentException.class, () -> splitter.split(oneSequence(1, 0, 0, false), 0,
                BatchSplitter.Mode.SIMPLE));
        assertThrows(NullPointerException.class, () -> splitter.split(oneSequence(1, 0, 0, false), 1, null));
    }

    @Test
    public void pagedFlagMarksTheLayouts() {
        ForwardBatch batch = oneSequence(2, 0, 0, true);
        List<UBatch> paged = new BatchSplitter(true).split(batch, 2, BatchSplitter.Mode.SIMPLE);
        assertEquals(1, paged.size());
        StepLayout layout = paged.get(0).layout();
        assertTrue(layout.paged());
        assertFalse(layout.contiguousSingle()); // paged memory: the contiguous delegation condition fails
        assertEquals(0, layout.pos0(0));
        assertEquals(1, layout.nSeg());
    }

    /** Three sequences in adapter slots 1, 0 and 2, their rows interleaved: row {@code i} of seq {@code i % 3}. */
    private static ForwardBatch interleavedSlots() {
        ForwardBatch.Builder b = ForwardBatch.builder(9);
        int[] slotOfSeq = {1, 0, 2};
        int[] slots = new int[9];
        for (int i = 0; i < 9; i++) {
            b.add(10 + i, i / 3, i % 3, false);
            slots[i] = slotOfSeq[i % 3];
        }
        return b.adapterSlots(slots).build();
    }

    @Test
    public void aSlotUniformSplitKeepsEveryUbatchInOneAdapterSlot() {
        ForwardBatch batch = interleavedSlots();
        int[] slots = batch.adapterSlots();
        for (BatchSplitter.Mode mode : BatchSplitter.Mode.values()) {
            List<UBatch> ubatches = new BatchSplitter(true).split(batch, 2, mode, true);
            boolean[] seen = new boolean[9];
            int[] firstUbatchOfSlot = {-1, -1, -1};
            for (int u = 0; u < ubatches.size(); u++) {
                int[] rows = ubatches.get(u).batchRows();
                int slot = slots[rows[0]];
                for (int r : rows) {
                    assertEquals(mode + ": ubatch " + u + " mixes adapter slots", slot, slots[r]);
                    assertFalse(mode + ": row " + r + " is in two ubatches", seen[r]);
                    seen[r] = true;
                }
                if (firstUbatchOfSlot[slot] < 0) {
                    firstUbatchOfSlot[slot] = u;
                }
            }
            for (int r = 0; r < 9; r++) {
                assertTrue(mode + ": row " + r + " is in no ubatch", seen[r]);
            }
            // the slots run in order of first appearance: slot 1 (row 0), then 0 (row 1), then 2 (row 2)
            assertTrue(mode.toString(), firstUbatchOfSlot[1] < firstUbatchOfSlot[0]);
            assertTrue(mode.toString(), firstUbatchOfSlot[0] < firstUbatchOfSlot[2]);
        }
    }

    @Test
    public void aMixedSplitMayHoldRowsOfSeveralSlotsAndAnUnslottedBatchSplitsAlike() {
        ForwardBatch batch = interleavedSlots();
        List<UBatch> mixed = new BatchSplitter(true).split(batch, 9, BatchSplitter.Mode.SIMPLE, false);
        assertEquals(1, mixed.size());
        assertEquals(9, mixed.get(0).batchRows().length);

        ForwardBatch plain = oneSequence(5, 0, 0, true);
        List<UBatch> uniform = new BatchSplitter().split(plain, 2, BatchSplitter.Mode.SIMPLE, true);
        List<UBatch> ordinary = new BatchSplitter().split(plain, 2, BatchSplitter.Mode.SIMPLE);
        assertEquals(ordinary.size(), uniform.size());
        for (int u = 0; u < ordinary.size(); u++) {
            assertArrayEquals(ordinary.get(u).batchRows(), uniform.get(u).batchRows());
        }
    }
}
