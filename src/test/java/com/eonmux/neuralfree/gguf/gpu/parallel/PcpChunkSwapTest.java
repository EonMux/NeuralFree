package com.eonmux.neuralfree.gguf.gpu.parallel;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The DualChunkSwap partition of a prefill context-parallel split: every query row maps to the one rank whose
 * chunk pair holds it, the packed rows of a rank are its pair concatenated in chunk order, and the inverse of the
 * packing puts every row back in query order — over long queries, queries that cross the chunk grid unevenly, and
 * queries too short to fill every rank's pair (vLLM's DualChunkSwap over {@code prefill_context_parallel_size}).
 */
public class PcpChunkSwapTest {

    @Test
    public void theChunkPairFollowsTheVllmLayout() {
        PcpChunkSwap swap = PcpChunkSwap.of(96, 2);
        assertEquals(4, swap.chunkCount());
        assertEquals(24, swap.chunkSize());
        assertEquals(0, swap.firstChunk(0));
        assertEquals(3, swap.secondChunk(0));
        assertEquals(1, swap.firstChunk(1));
        assertEquals(2, swap.secondChunk(1));
        assertEquals(0, swap.chunkStart(0));
        assertEquals(24, swap.chunkStart(1));
        assertEquals(48, swap.chunkStart(2));
        assertEquals(72, swap.chunkStart(3));
        assertEquals(24, swap.chunkLen(2));
        assertEquals(0, swap.ownedStart(0));
        assertEquals(48, swap.ownedCount(0));
        assertEquals(24, swap.ownedStart(1));
        assertEquals(48, swap.ownedCount(1));
        assertArrayEquals(new int[]{0, 23}, new int[]{swap.rows(0)[0], swap.rows(0)[23]});
        assertArrayEquals(new int[]{24, 71}, new int[]{swap.rows(1)[0], swap.rows(1)[47]});
    }

    @Test
    public void everyRowMapsToExactlyOneRank() {
        for (int queryLen = 1; queryLen <= 40; queryLen++) {
            for (int ranks = 1; ranks <= 4; ranks++) {
                PcpChunkSwap swap = PcpChunkSwap.of(queryLen, ranks);
                int[] seenBy = new int[queryLen];
                int total = 0;
                for (int rank = 0; rank < ranks; rank++) {
                    int[] rows = swap.rows(rank);
                    total += rows.length;
                    for (int row : rows) {
                        assertTrue("row " + row + " of " + queryLen + " over " + ranks
                                        + " ranks is inside the query",
                                row >= 0 && row < queryLen);
                        seenBy[row]++;
                        assertTrue(swap.ownsRow(rank, row));
                    }
                    assertEquals(rows.length, swap.ownedCount(rank));
                    for (int i = 0; i < rows.length; i++) {
                        assertEquals("packed row " + i + " of rank " + rank + " is its own inverse index",
                                i, swap.packedIndexOfRow(rows[i]));
                    }
                }
                assertEquals("the ranks partition all " + queryLen + " rows over " + ranks + " ranks",
                        queryLen, total);
                for (int row = 0; row < queryLen; row++) {
                    assertEquals("row " + row + " of " + queryLen + " over " + ranks + " ranks has one owner",
                            1, seenBy[row]);
                    assertEquals("rankOfRow agrees with the packed rows",
                            seenBy[row] == 1 ? ownerOf(swap, row) : -1, swap.rankOfRow(row));
                }
            }
        }
    }

    /** The rank whose packed rows hold {@code row}, found by scanning. */
    private static int ownerOf(PcpChunkSwap swap, int row) {
        for (int rank = 0; rank < swap.ranks(); rank++) {
            for (int packed : swap.rows(rank)) {
                if (packed == row) {
                    return rank;
                }
            }
        }
        return -1;
    }

    @Test
    public void thePackingRoundTripsThroughTheInverse() {
        for (int queryLen = 1; queryLen <= 64; queryLen++) {
            for (int ranks = 1; ranks <= 4; ranks++) {
                PcpChunkSwap swap = PcpChunkSwap.of(queryLen, ranks);
                int[] restored = new int[queryLen];
                List<Integer> packedOrder = new ArrayList<>();
                for (int rank = 0; rank < ranks; rank++) {
                    int[] rows = swap.rows(rank);
                    for (int i = 0; i < rows.length; i++) {
                        int row = rows[i];
                        assertEquals("the inverse of the packing finds row " + row + " of rank " + rank
                                        + " at its packed index",
                                i, swap.packedIndexOfRow(row));
                        assertEquals("rankOfRow and packedIndexOfRow agree on the owner of row " + row,
                                rank, swap.rankOfRow(row));
                        restored[row] = row;
                        packedOrder.add(row);
                    }
                }
                assertEquals(queryLen, packedOrder.size());
                for (int row = 0; row < queryLen; row++) {
                    assertEquals("row " + row + " survives the round trip", row, restored[row]);
                }
            }
        }
    }

    @Test
    public void aShortQueryLeavesTrailingRanksWithoutRows() {
        PcpChunkSwap swap = PcpChunkSwap.of(3, 2);
        assertEquals(1, swap.chunkSize());
        assertEquals(1, swap.ownedCount(0));
        assertEquals(2, swap.ownedCount(1));
        assertArrayEquals(new int[]{0}, swap.rows(0));
        assertArrayEquals(new int[]{1, 2}, swap.rows(1));
        PcpChunkSwap tiny = PcpChunkSwap.of(2, 3);
        assertEquals(1, tiny.ownedCount(0));
        assertEquals(1, tiny.ownedCount(1));
        assertEquals(0, tiny.ownedCount(2));
        assertArrayEquals(new int[0], tiny.rows(2));
        assertThrows(IllegalStateException.class, () -> tiny.ownedStart(2));
    }

    @Test
    public void anEmptyQueryOrRankCountIsRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> PcpChunkSwap.of(0, 2));
        assertTrue(e.getMessage(), e.getMessage().contains("at least one query row"));
        e = assertThrows(IllegalArgumentException.class, () -> PcpChunkSwap.of(8, 0));
        assertTrue(e.getMessage(), e.getMessage().contains("at least one rank"));
        IndexOutOfBoundsException out = assertThrows(IndexOutOfBoundsException.class,
                () -> PcpChunkSwap.of(8, 2).rankOfRow(8));
        assertTrue(out.getMessage(), out.getMessage().contains("query row 8"));
    }
}
