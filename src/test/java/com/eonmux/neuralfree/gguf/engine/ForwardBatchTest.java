package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.EnumSet;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The valid, and the invalid, batches of {@link ForwardBatch.Builder#build} (the llama.cpp batch rules). */
public final class ForwardBatchTest {

    private static ForwardBatch.Builder tokenBatch(int n) {
        ForwardBatch.Builder b = ForwardBatch.builder(n);
        for (int i = 0; i < n; i++) {
            b.add(10 + i, i, 0, i == n - 1);
        }
        return b;
    }

    @Test
    public void tokenBatchCarriesRows() {
        ForwardBatch batch = tokenBatch(4).build();
        assertEquals(4, batch.nTokens());
        assertArrayEquals(new int[]{10, 11, 12, 13}, batch.tokens());
        assertArrayEquals(new int[]{0, 1, 2, 3}, batch.positions());
        assertArrayEquals(new int[]{0, 0, 0, 0}, batch.seqIds());
        assertArrayEquals(new int[]{3}, batch.outputRows());
        assertTrue(batch.isSingleSequence());
        assertNull(batch.embeddings());
        assertNull(batch.adapterSlots());
        assertNull(batch.ropePositions());
    }

    @Test
    public void accessorsReturnCopies() {
        ForwardBatch batch = tokenBatch(2).build();
        batch.tokens()[0] = -1;
        batch.positions()[0] = -1;
        batch.seqIds()[0] = -1;
        batch.outputRows()[0] = 0;
        assertArrayEquals(new int[]{10, 11}, batch.tokens());
        assertArrayEquals(new int[]{0, 1}, batch.positions());
        assertArrayEquals(new int[]{0, 0}, batch.seqIds());
        assertArrayEquals(new int[]{1}, batch.outputRows());
    }

    @Test
    public void embeddingBatchValidatesOneWidth() {
        float[] row = {1, 2, 3};
        ForwardBatch batch = ForwardBatch.builder(2)
                .addEmbedding(row, 0, 0, true)
                .addEmbedding(new float[]{4, 5, 6}, 1, 0, false)
                .build();
        assertNotNull(batch.embeddings());
        assertNotNull(batch.embeddings()[0]);
        assertNull(batch.embeddings()[1]); // token batches and embedding batches never carry both
        assertEquals(2, batch.nTokens());
        assertArrayEquals(new float[]{1, 2, 3}, batch.embeddingRow(0), 0.0f);
        assertArrayEquals(new float[]{4, 5, 6}, batch.embeddingRow(1), 0.0f);
        assertArrayEquals(new int[]{-1, -1}, batch.tokens());
        row[0] = 99; // rows are copied at add time
        assertArrayEquals(new float[]{1, 2, 3}, batch.embeddingRow(0), 0.0f);
        assertThrows(IllegalArgumentException.class, () -> ForwardBatch.builder(2)
                .add(7, 0, 0, true)
                .addEmbedding(new float[]{1}, 1, 0, false)
                .build());
        assertThrows(IllegalArgumentException.class, () -> ForwardBatch.builder(2)
                .addEmbedding(new float[]{1}, 0, 0, true)
                .addEmbedding(new float[]{1, 2}, 1, 0, false)
                .build());
    }

    @Test
    public void buildRejectsInvalidBatches() {
        assertThrows(IllegalStateException.class, () -> ForwardBatch.builder(1).build()); // n >= 1
        assertThrows(IllegalArgumentException.class, () -> ForwardBatch.builder(0)); // capacity positive
        ForwardBatch.Builder seq = ForwardBatch.builder(1);
        seq.add(1, 0, -1, false);
        assertThrows(IllegalArgumentException.class, seq::build); // seq ids >= 0
        ForwardBatch.Builder pos = ForwardBatch.builder(1);
        pos.add(1, -1, 0, false);
        assertThrows(IllegalArgumentException.class, pos::build); // positions >= 0
        ForwardBatch.Builder jump = ForwardBatch.builder(2);
        jump.add(1, 0, 0, false).add(2, 2, 0, false);
        assertThrows(IllegalArgumentException.class, jump::build); // +1 from one row to the next
        // interleaved sequences are allowed while each keeps its own consecutive positions
        ForwardBatch.builder(3).add(1, 0, 0, false).add(2, 0, 1, false).add(3, 1, 0, false).build().nTokens();
        ForwardBatch.Builder hole = ForwardBatch.builder(3);
        hole.add(1, 0, 0, false).add(2, 0, 1, false).add(3, 2, 0, false);
        assertThrows(IllegalArgumentException.class, hole::build); // sequence 0 holed
        ForwardBatch.Builder noToken = ForwardBatch.builder(1);
        noToken.add(-1, 0, 0, true);
        assertThrows(IllegalArgumentException.class, noToken::build); // token ids on a token batch
    }

    @Test
    public void optionalRowsValidated() {
        int[][] sections = {{0, 1, 2}, {5, 6, 7}};
        ForwardBatch batch = tokenBatch(3).ropePositions(sections).build();
        assertArrayEquals(sections[0], batch.ropePositions()[0]);
        assertArrayEquals(sections[1], batch.ropePositions()[1]);
        assertThrows(IllegalArgumentException.class, () -> tokenBatch(3).ropePositions(new int[][]{{0, 1}}));
        assertThrows(IllegalArgumentException.class, () -> tokenBatch(3).ropePositions(new int[0][]));
        ForwardBatch.Builder span = ForwardBatch.builder(2);
        span.add(1, 0, 0, true).add(2, 1, 0, false);
        assertThrows(IllegalArgumentException.class, () -> span.ropePositions(sections)); // covers 3, batch holds 2
        assertThrows(IllegalArgumentException.class, () -> span.adapterSlots(new int[]{0})); // covers 1, batch 2
        assertThrows(IllegalArgumentException.class, () -> span.adapterSlots(new int[]{0, -1}));
        span.adapterSlots(new int[]{0, 1}); // valid: set once
        assertThrows(IllegalStateException.class, () -> span.adapterSlots(new int[]{0, 1})); // set once
        assertThrows(IllegalStateException.class, () -> tokenBatch(1).ropePositions(new int[][]{{0}}) // no-op set
                .ropePositions(new int[][]{{0}}));
    }

    @Test
    public void outputKindsDefaultToLogits() {
        ForwardBatch plain = tokenBatch(2).build();
        assertEquals(EnumSet.of(OutputKind.LOGITS), plain.kinds(1));
        assertTrue(plain.kinds(0).isEmpty()); // not an output row
        assertTrue(plain.produces(1, OutputKind.LOGITS));
        assertFalse(plain.produces(1, OutputKind.HIDDEN));
        ForwardBatch kinds = tokenBatch(2).outputKinds(EnumSet.of(OutputKind.HIDDEN, OutputKind.ROUTING)).build();
        assertEquals(EnumSet.of(OutputKind.HIDDEN, OutputKind.ROUTING), kinds.kinds(1));
        assertFalse(kinds.produces(1, OutputKind.LOGITS));
        assertTrue(kinds.produces(0, OutputKind.HIDDEN) == false);
        ForwardBatch.Builder empty = ForwardBatch.builder(1);
        empty.add(1, 0, 0, true);
        assertThrows(IllegalArgumentException.class, () -> empty.outputKinds(EnumSet.noneOf(OutputKind.class)));
        ForwardBatch.Builder twice = ForwardBatch.builder(1);
        twice.add(1, 0, 0, true);
        twice.outputKinds(EnumSet.of(OutputKind.LOGITS));
        assertThrows(IllegalStateException.class, () -> twice.outputKinds(EnumSet.of(OutputKind.LOGITS)));
    }

    @Test
    public void adapterSlotsRoundTrip() {
        ForwardBatch withSlots = ForwardBatch.builder(2)
                .add(1, 0, 0, true)
                .add(2, 1, 0, false)
                .adapterSlots(new int[]{0, 2})
                .build();
        assertTrue(withSlots.isSingleSequence());
        assertArrayEquals(new int[]{0, 2}, withSlots.adapterSlots());
        withSlots.adapterSlots()[0] = 9;
        assertArrayEquals(new int[]{0, 2}, withSlots.adapterSlots());
    }

    @Test
    public void multiSequenceBatchesBuiltButMarked() {
        ForwardBatch batch = ForwardBatch.builder(4)
                .add(1, 0, 0, false)
                .add(2, 1, 0, true)
                .add(3, 0, 1, false)
                .add(4, 1, 1, true)
                .build();
        assertFalse(batch.isSingleSequence());
        assertArrayEquals(new int[]{1, 3}, batch.outputRows());
    }
}
