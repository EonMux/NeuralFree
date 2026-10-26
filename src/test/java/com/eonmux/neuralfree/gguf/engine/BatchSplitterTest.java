package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.StepLayout;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The three split modes of {@link BatchSplitter} (llama.cpp src/llama-batch.cpp:585, 619, 790). */
public final class BatchSplitterTest {

    private static ForwardBatch oneSequence(int n, int pos0, int seq, boolean outputTail) {
        ForwardBatch.Builder b = ForwardBatch.builder(n);
        for (int i = 0; i < n; i++) {
            b.add(100 + i, pos0 + i, seq, outputTail && i == n - 1);
        }
        return b.build();
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
        assertArrayEquals(new int[]{105}, ubatches.get(2).tokens());
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
        assertThrows(IllegalArgumentException.class, () -> splitter.split(oneSequence(1, 0, 0, false), 1, null));
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
}
