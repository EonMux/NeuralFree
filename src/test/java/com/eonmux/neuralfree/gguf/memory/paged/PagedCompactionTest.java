package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The compaction a context shift runs ({@link PagedCompaction}): the live rows of the shifted sequence slide onto
 * a running write cursor in slot order, whatever the dead slots below them look like. A sliding-window group whose
 * trim left holes on its leading page and an interior removal give two dead runs, which must neither lose a row nor
 * write into another sequence's pages. The moves coalesce into runs contiguous in both cells, never overlap
 * themselves, run in queue order, and carry every row's data (the page copies first, as the device runs them); the
 * mirror reports are one set and one clear report per group.
 */
public class PagedCompactionTest {

    private static final int PAGE = 4;
    private static final int N_CTX = 128;
    private static final int SEQ = 0;
    private static final int OTHER = 1;

    private static PagedSequences memory(KvWindow window) {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(N_CTX).ubatch(8).parallel(2).kvPageSize(PAGE).build();
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, N_CTX, cfg.ubatch())
                .group(MemoryPlan.KvGroup.of(0, N_CTX, false, window, PAGE, N_CTX / PAGE))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV)
                        .kv(KvLayout.of(N_CTX, false, 8, 8, KvDType.F16, KvDType.F16, window), 0).build())
                .config(cfg)
                .build();
        return new PagedSequences(plan, 2);
    }

    /** Writes positions {@code [from, from + len)} of {@code seq} as one finished step (the trim runs). */
    private static void write(PagedSequences mem, int seq, int from, int len) {
        StepLayout layout = StepLayout.of(len, new int[] {seq}, new int[] {0}, new int[] {len}, new int[] {from},
                new int[] {0}, true);
        PagedStep step = mem.prepare(layout);
        mem.commit(step);
        mem.releaseCopies(mem.pendingCopies().toArray(new PageCopy[0]));
        mem.finish(step);
    }

    /** Writes {@code n0} rows of {@link #SEQ} and {@code n1} of {@link #OTHER} a page at a time, alternating. */
    private static void interleave(PagedSequences mem, int n0, int n1) {
        for (int p = 0; p < Math.max(n0, n1); p += PAGE) {
            if (p < n0) {
                write(mem, SEQ, p, Math.min(PAGE, n0 - p));
            }
            if (p < n1) {
                write(mem, OTHER, p, Math.min(PAGE, n1 - p));
            }
        }
        mem.clearReports();
    }

    /** The cells of {@code seq}'s live slots, in slot order. */
    private static int[] liveCells(PagedSequences mem, int seq) {
        PageList list = mem.lists()[seq][0];
        int cursor = mem.slots()[0][seq];
        int[] cells = new int[cursor];
        int n = 0;
        for (int s = 0; s < cursor; s++) {
            int cell = mem.cellOf(list, s);
            if (mem.kvPos()[0][cell] >= 0) {
                cells[n++] = cell;
            }
        }
        return Arrays.copyOf(cells, n);
    }

    private static int[] positionsOf(PagedSequences mem, int[] cells) {
        int[] out = new int[cells.length];
        for (int i = 0; i < cells.length; i++) {
            out[i] = mem.kvPos()[0][cells[i]];
        }
        return out;
    }

    private static Set<Integer> pagesOf(PagedSequences mem, int seq) {
        Set<Integer> pages = new HashSet<>();
        for (int p : mem.lists()[seq][0].toArray()) {
            pages.add(p);
        }
        return pages;
    }

    /**
     * A device's view of one group: a tag per cell, carried by the pending page copies, then by the pending row
     * moves in queue order (each copy of a move reads its whole source before it writes, as a non-overlapping
     * buffer copy does).
     */
    private static int[] carry(PagedSequences mem, int[] data) {
        int[] out = data.clone();
        for (PageCopy c : mem.pendingCopies()) {
            System.arraycopy(out, c.srcPage() * PAGE + c.srcRow0(), out, c.dstPage() * PAGE + c.dstRow0(), c.rows());
        }
        for (RowMove m : mem.pendingRowMoves()) {
            assertTrue("move " + m + " overlaps itself", Math.abs(m.srcRow() - m.dstRow()) >= m.rows());
            System.arraycopy(out, m.srcRow(), out, m.dstRow(), m.rows());
        }
        return out;
    }

    /**
     * Shifts {@link #SEQ} as the server's context shift does ({@code remove [keep, keep + discard)}, then
     * {@code add [keep + discard, end) -discard}) and checks the outcome against the slot-order reference.
     */
    private static void shiftAndCheck(PagedSequences mem, int keep, int discard) {
        int[] cellsBefore = liveCells(mem, SEQ);
        int[] posBefore = positionsOf(mem, cellsBefore);
        int[] otherCells = liveCells(mem, OTHER);
        int[] otherPos = positionsOf(mem, otherCells);
        Set<Integer> otherPages = pagesOf(mem, OTHER);
        int[] data = new int[mem.kvPos()[0].length];
        Arrays.fill(data, -7);
        for (int i = 0; i < cellsBefore.length; i++) {
            data[cellsBefore[i]] = 1000 + posBefore[i];
        }
        for (int i = 0; i < otherCells.length; i++) {
            data[otherCells[i]] = 2000 + otherPos[i];
        }
        Set<Integer> seqPages = pagesOf(mem, SEQ);

        mem.remove(SEQ, keep, keep + discard);
        mem.add(SEQ, keep + discard, -1, -discard);
        seqPages.addAll(pagesOf(mem, SEQ));
        for (PageCopy c : mem.pendingCopies()) {
            seqPages.add(c.dstPage());
        }

        List<Integer> wantPos = new ArrayList<>();
        List<Integer> wantTag = new ArrayList<>();
        for (int p : posBefore) {
            if (p < keep) {
                wantPos.add(p);
                wantTag.add(1000 + p);
            } else if (p >= keep + discard) {
                wantPos.add(p - discard);
                wantTag.add(1000 + p);
            }
        }
        String at = "keep " + keep + " discard " + discard;
        int[] cellsAfter = liveCells(mem, SEQ);
        assertArrayEquals(at + ": the live positions in slot order",
                wantPos.stream().mapToInt(Integer::intValue).toArray(), positionsOf(mem, cellsAfter));
        assertEquals(at + ": the cursor sits on the last live row", cellsAfter.length, mem.slots()[0][SEQ]);
        assertArrayEquals(at + ": the other sequence keeps its cells", otherCells, liveCells(mem, OTHER));
        assertArrayEquals(at + ": the other sequence keeps its positions", otherPos,
                positionsOf(mem, liveCells(mem, OTHER)));
        assertEquals(at + ": the other sequence keeps its pages", otherPages, pagesOf(mem, OTHER));
        for (RowMove m : mem.pendingRowMoves()) {
            assertTrue(at + ": move " + m + " reads the shifted sequence's pages",
                    seqPages.contains(m.srcRow() / PAGE) && seqPages.contains((m.srcRow() + m.rows() - 1) / PAGE));
            assertTrue(at + ": move " + m + " writes the shifted sequence's pages",
                    seqPages.contains(m.dstRow() / PAGE) && seqPages.contains((m.dstRow() + m.rows() - 1) / PAGE));
            assertTrue(at + ": move " + m + " never writes the other sequence's pages",
                    !otherPages.contains(m.dstRow() / PAGE)
                            && !otherPages.contains((m.dstRow() + m.rows() - 1) / PAGE));
        }
        int[] carried = carry(mem, data);
        for (int i = 0; i < cellsAfter.length; i++) {
            assertEquals(at + ": the data of position " + wantPos.get(i), (int) wantTag.get(i),
                    carried[cellsAfter[i]]);
        }
        for (int i = 0; i < otherCells.length; i++) {
            assertEquals(at + ": the other sequence's data at cell " + otherCells[i], 2000 + otherPos[i],
                    carried[otherCells[i]]);
        }
    }

    @Test
    public void windowHolesAndAnInteriorRemovalCompactWithoutTouchingTheOtherSequence() {
        for (int discard : new int[] {1, 2, 3, 5}) {
            PagedSequences mem = memory(KvWindow.of(SwaType.STANDARD, 12, true));
            interleave(mem, 27, 20);
            // committed 27 trims below 15: positions 12..14 stay as holes on the leading page
            assertEquals(15, positionsOf(mem, liveCells(mem, SEQ))[0]);
            assertTrue("the leading page keeps the trim's holes", mem.slots()[0][SEQ] > liveCells(mem, SEQ).length);
            shiftAndCheck(mem, 18, discard);
        }
    }

    @Test
    public void aPageAlignedSlideMovesWholePages() {
        PagedSequences mem = memory(KvWindow.FULL_CAUSAL);
        interleave(mem, 32, 8);
        shiftAndCheck(mem, 4, 8);
        // 20 rows slide by two pages: at most one move per page of rows, not one per row
        List<RowMove> moves = mem.pendingRowMoves();
        assertTrue("moves " + moves, moves.size() <= 5);
        assertEquals(20, moves.stream().mapToInt(RowMove::rows).sum());
    }

    @Test
    public void aSlideShorterThanARunNeverOverlapsAMove() {
        PagedSequences mem = memory(KvWindow.FULL_CAUSAL);
        interleave(mem, 24, 4);
        shiftAndCheck(mem, 5, 1);
        for (RowMove m : mem.pendingRowMoves()) {
            assertEquals("a one-row slide moves one row at a time", 1, m.rows());
        }
    }

    @Test
    public void theMirrorReportsAreOneSetAndOneClearReport() {
        PagedSequences mem = memory(KvWindow.FULL_CAUSAL);
        interleave(mem, 32, 8);
        mem.remove(SEQ, 4, 12);
        mem.clearReports();
        mem.add(SEQ, 12, -1, -8);
        List<RowReport> reports = mem.pendingReports();
        RowReport sets = reports.get(reports.size() - 2);
        RowReport clears = reports.get(reports.size() - 1);
        assertTrue(sets.absolute());
        assertEquals("every destination cell is set once", 20, sets.rows().length);
        assertEquals("the vacated sources are cleared once", 8, clears.rows().length);
        for (int v : clears.values()) {
            assertEquals(-1, v);
        }
        int[] setRows = sets.rows();
        int[] sorted = setRows.clone();
        Arrays.sort(sorted);
        assertArrayEquals("the set rows ascend", sorted, setRows);
        Set<Integer> both = new HashSet<>();
        for (int r : setRows) {
            both.add(r);
        }
        for (int r : clears.rows()) {
            assertTrue("a cell is reported once, by its final value", both.add(r));
        }
    }

    /**
     * A fork shares every page; a shift bounded below the tail leaves the tail rows unshifted on pages the fork
     * still holds. The compaction moves those rows down for the shifted sequence and leaves the shared tail page's
     * rows in place for the fork, which keeps its cells, positions and data, while the pool's used-row count stays
     * the count of the stored rows.
     */
    @Test
    public void aForkKeepsTheRowsOfASharedTailPageTheShiftVacates() {
        PagedSequences mem = memory(KvWindow.FULL_CAUSAL);
        interleave(mem, 24, 0);
        mem.copy(SEQ, OTHER, 0, -1);
        mem.clearReports();
        int[] forkCells = liveCells(mem, OTHER);
        int[] forkPos = positionsOf(mem, forkCells);
        assertArrayEquals("the fork shares every row", liveCells(mem, SEQ), forkCells);
        int tailPage = mem.lists()[SEQ][0].page(5);
        int[] data = new int[mem.kvPos()[0].length];
        Arrays.fill(data, -7);
        for (int i = 0; i < forkCells.length; i++) {
            data[forkCells[i]] = 1000 + forkPos[i];
        }

        mem.remove(SEQ, 4, 8);
        mem.add(SEQ, 8, 16, -4);

        int[] cells = liveCells(mem, SEQ);
        assertArrayEquals(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23},
                positionsOf(mem, cells));
        assertEquals(20, mem.slots()[0][SEQ]);
        assertEquals("the shifted sequence keeps the pages of its 20 rows", 5, mem.lists()[SEQ][0].size());
        assertTrue("the vacated tail page leaves the shifted sequence", !pagesOf(mem, SEQ).contains(tailPage));
        assertTrue("the fork keeps the tail page", pagesOf(mem, OTHER).contains(tailPage));
        assertEquals(1, mem.pools()[0].refCount(tailPage));
        assertArrayEquals("the fork keeps its cells", forkCells, liveCells(mem, OTHER));
        assertArrayEquals("the fork keeps its positions", forkPos, positionsOf(mem, liveCells(mem, OTHER)));
        int[] carried = carry(mem, data);
        int[] origin = {0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
        for (int i = 0; i < cells.length; i++) {
            assertEquals("the data of slot " + i, 1000 + origin[i], carried[cells[i]]);
        }
        for (int i = 0; i < forkCells.length; i++) {
            assertEquals("the fork's data at cell " + forkCells[i], 1000 + forkPos[i], carried[forkCells[i]]);
        }
        int stored = 0;
        for (int p = 0; p < mem.pools()[0].pages(); p++) {
            for (int r = 0; r < PAGE && !mem.pools()[0].isFree(p); r++) {
                stored += mem.kvPos()[0][p * PAGE + r] >= 0 ? 1 : 0;
            }
        }
        assertEquals("the used rows count the stored rows", stored, mem.usedRows()[0]);
    }

    @Test
    public void randomDeadPatternsCompactInSlotOrder() {
        Random rnd = new Random(5);
        for (int trial = 0; trial < 300; trial++) {
            boolean windowed = rnd.nextBoolean();
            int window = 6 + rnd.nextInt(20);
            PagedSequences mem = memory(windowed ? KvWindow.of(SwaType.STANDARD, window, true)
                    : KvWindow.FULL_CAUSAL);
            int n0 = 8 + rnd.nextInt(48);
            interleave(mem, n0, 1 + rnd.nextInt(40));
            int first = positionsOf(mem, liveCells(mem, SEQ))[0];
            if (n0 - first < 3) {
                continue;
            }
            int keep = first + rnd.nextInt(n0 - first - 1);
            int discard = 1 + rnd.nextInt(n0 - keep - 1);
            shiftAndCheck(mem, keep, discard);
        }
    }
}
