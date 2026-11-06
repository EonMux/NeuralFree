package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The block-holder side of the paged memory (the prefix cache's block pool): a block's pages are found by their
 * positions, held past their sequence's removal, released back to the pools, and adopted by an empty sequence as
 * its leading positions — whole blocks plus the shared head of a boundary page. A sequence that owns only the head
 * of a shared page copies just those rows on its first write, and its stored range never counts the rows above its
 * cursor that belong to the page's other holders.
 */
public class PagedSequencesBlocksTest {

    private static final int PAGE = 4;
    private static final int CTX = 64;

    private static PagedSequences memory() {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(CTX).ubatch(8).parallel(2).kvPageSize(PAGE).build();
        KvWindow w = KvWindow.FULL_CAUSAL;
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, CTX, cfg.ubatch())
                .group(MemoryPlan.KvGroup.of(0, CTX, false, w, PAGE, CTX / PAGE))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV)
                        .kv(KvLayout.of(CTX, false, 8, 8, KvDType.F16, KvDType.F16, w), 0).build())
                .config(cfg)
                .build();
        return new PagedSequences(plan, 2);
    }

    /** Writes positions {@code [from, from + len)} of {@code seq} in one committed step. */
    private static PagedStep write(PagedSequences mem, int seq, int from, int len) {
        StepLayout layout = StepLayout.of(len, new int[] {seq}, new int[] {0}, new int[] {len}, new int[] {from},
                new int[] {0}, true);
        PagedStep step = mem.prepare(layout);
        mem.commit(step);
        mem.releaseCopies(mem.pendingCopies().toArray(new PageCopy[0]));
        return step;
    }

    private static int[] ids(int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 100 + i;
        }
        return t;
    }

    @Test
    public void aHeldBlockOutlivesItsSequenceAndReleasesBackToThePool() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 6);
        int[] whole = mem.blockPages(0, 0, PAGE);
        assertNotNull(whole);
        assertNull("the second block holds only two rows", mem.blockPages(0, 1, PAGE));
        int[] head = mem.blockPages(0, 1, 2);
        assertNotNull(head);
        mem.holdPages(whole);
        mem.holdPages(head);
        mem.remove(0, 0, -1);
        assertEquals("the held pages keep their six rows", CTX - 6, mem.freeCells());
        assertFalse("only the holder references them", mem.pagesShared(whole));
        mem.clearReports();
        mem.releasePages(whole);
        mem.releasePages(head);
        assertEquals(CTX, mem.freeCells());
        List<RowReport> reports = mem.pendingReports();
        assertEquals("each freed page reports its emptied rows", 2, reports.size());
    }

    @Test
    public void anAdoptedSequenceSharesTheBlocksAndContinuesAfterThem() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 6);
        int[] b0 = mem.blockPages(0, 0, PAGE);
        int[] b1 = mem.blockPages(0, 1, 2);
        mem.adoptPages(1, new int[][] {b0, b1}, 1, ids(5));
        assertEquals(0, mem.posMin(1));
        assertEquals(4, mem.posMax(1));
        assertEquals(5, mem.sequence(1).nPast());
        assertTrue("the adopted pages are shared with the first sequence", mem.pagesShared(b0));
        assertEquals("adoption writes no rows", CTX - 6, mem.freeCells());

        PagedStep next = write(mem, 1, 5, 1);
        assertEquals("the write into the shared boundary page copies it", 1, next.pageCopies().size());
        PageCopy copy = next.pageCopies().get(0);
        assertEquals(b1[0], copy.srcPage());
        assertEquals("only the adopter's own row moves to the private page", 1, copy.rows());
        assertEquals(5, mem.posMax(1));
        assertEquals("the first sequence keeps its rows", 5, mem.posMax(0));
    }

    @Test
    public void theStoredRangeIgnoresTheOtherHoldersRowsOnASharedPage() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 6);
        int[] b0 = mem.blockPages(0, 0, PAGE);
        int[] b1 = mem.blockPages(0, 1, 2);
        mem.adoptPages(1, new int[][] {b0, b1}, 1, ids(5));
        // an interior removal recomputes the range while the boundary page is still shared: position 5 on it is
        // the first sequence's, above the adopter's cursor
        mem.remove(1, 0, 1);
        assertEquals("the adopter stores up to its own position 4", 4, mem.posMax(1));
        assertEquals(1, mem.posMin(1));
    }

    @Test
    public void adoptionRefusesABusySequenceAndForeignPositions() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 8);
        int[] b0 = mem.blockPages(0, 0, PAGE);
        int[] b1 = mem.blockPages(0, 1, PAGE);
        assertThrows(IllegalStateException.class, () -> mem.adoptPages(0, new int[][] {b0}, PAGE, ids(PAGE)));
        assertThrows("block 1's page holds positions 4..7, not 0..3", IllegalStateException.class,
                () -> mem.adoptPages(1, new int[][] {b1}, PAGE, ids(PAGE)));
        assertThrows(IllegalArgumentException.class, () -> mem.adoptPages(1, new int[][] {b0}, PAGE, ids(3)));
        assertEquals("a refused adoption changes nothing", -1, mem.posMax(1));
        mem.adoptPages(1, new int[][] {b0, b1}, PAGE, ids(8));
        assertArrayEquals(ids(8), mem.sequence(1).tokens());
    }

    @Test
    public void mediaRowsSharingTheirBasePositionFindAndAdoptByExplicitPositions() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 2);
        // a media chunk of four rows that all store base position 2, then a text row at position 4
        int[] media = {2, 2, 2, 2};
        StepLayout layout = StepLayout.of(4, new int[] {0}, new int[] {0}, new int[] {4}, new int[] {2},
                new int[] {0}, true);
        mem.commit(mem.prepare(layout, media));
        write(mem, 0, 4, 1);
        int[] stored = {0, 1, 2, 2, 2, 2, 4};
        assertNull("the rows no longer store their own index", mem.blockPages(0, 1, PAGE));
        int[] b0 = mem.blockPages(0, 0, PAGE, new int[] {0, 1, 2, 2});
        int[] b1 = mem.blockPages(0, 1, 3, new int[] {2, 2, 4});
        assertNotNull(b0);
        assertNotNull(b1);
        assertThrows(IllegalArgumentException.class, () -> mem.blockPages(0, 1, 3, new int[] {2, 2}));
        int[] tokens = {100, 101, -1, -1, -1, -1, 102};
        assertThrows("a position the page does not hold refuses", IllegalStateException.class,
                () -> mem.adoptPages(1, new int[][] {b0, b1}, 3, tokens, new int[] {0, 1, 2, 3, 4, 5, 6}, -1));
        assertEquals("a refused adoption changes nothing", 0, mem.sequence(1).nPast());
        int standIn = 7;
        mem.adoptPages(1, new int[][] {b0, b1}, 3, tokens, stored, standIn);
        assertEquals(0, mem.posMin(1));
        assertEquals("the highest position is the last row's", 4, mem.posMax(1));
        SequenceState record = mem.sequence(1);
        assertEquals("one record entry per adopted row", stored.length, record.nPast());
        for (int r = 0; r < stored.length; r++) {
            assertEquals("row " + r + " records its stored position", stored[r], record.position(r));
            assertEquals("row " + r + " is an embedding row exactly when it holds no token", tokens[r] < 0,
                    record.embeddingRow(r));
        }
        assertArrayEquals("media rows record the model's stand-in", new int[] {100, 101, 7, 7, 7, 7, 102},
                record.tokens());
        assertArrayEquals("the text view leaves the media rows out", new int[] {100, 101, 102},
                record.textTokens());
        write(mem, 1, 5, 1);
        assertEquals(5, mem.posMax(1));
        mem.appendRows(1, new int[] {103}, new int[] {5}, 0, 1, standIn);
        assertEquals("the next step's row records after the adopted ones", 5, record.position(stored.length));
    }

    @Test
    public void anAdoptedTokenPrefixRecordsItsRowsAtTheirOwnIndexes() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 6);
        int[] b0 = mem.blockPages(0, 0, PAGE);
        int[] b1 = mem.blockPages(0, 1, 2);
        mem.adoptPages(1, new int[][] {b0, b1}, 2, ids(6));
        SequenceState record = mem.sequence(1);
        for (int r = 0; r < 6; r++) {
            assertEquals(r, record.position(r));
            assertFalse(record.embeddingRow(r));
        }
        assertArrayEquals(ids(6), record.textTokens());
    }

    @Test
    public void anAdoptedEmbeddingRowRecordsTheModelsStandIn() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 1);
        int[] media = {1, 1, 1};
        StepLayout layout = StepLayout.of(3, new int[] {0}, new int[] {0}, new int[] {3}, new int[] {1},
                new int[] {0}, true);
        mem.commit(mem.prepare(layout, media));
        int[] stored = {0, 1, 1, 1};
        int[] b0 = mem.blockPages(0, 0, PAGE, stored);
        assertNotNull(b0);
        assertThrows("adopted positions never descend", IllegalArgumentException.class,
                () -> mem.adoptPages(1, new int[][] {b0}, PAGE, new int[] {9, -1, -1, -1}, new int[] {0, 1, 1, 0}, 7));
        mem.adoptPages(1, new int[][] {b0}, PAGE, new int[] {9, -1, -1, -1}, stored, 7);
        assertArrayEquals(new int[] {9, 7, 7, 7}, mem.sequence(1).tokens());
        assertArrayEquals(new int[] {9}, mem.sequence(1).textTokens());
        assertEquals(1, mem.posMax(1));
    }

    @Test
    public void aClearCountsForTheHolders() {
        PagedSequences mem = memory();
        int before = mem.clears();
        mem.clear();
        assertEquals(before + 1, mem.clears());
        assertEquals(1, mem.groups());
    }
}
