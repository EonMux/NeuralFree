package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Copy-on-write and reservation rules of {@link PagedSequences}: logical slot {@code j} maps to
 * {@code pages[j >> log2P] * P + (j & (P-1))}; a page with {@code refCount > 1} is never written in place, every
 * mutation of it emits exactly one whole-page {@link PageCopy} and swaps a private page into the mutating
 * sequence's list, so the other holders' page ids and reported {@code kv_pos} rows never change;
 * reservations are all-or-nothing, failing with {@link MemoryFullException} before any state changed.
 *
 * <p>Plans are built by hand over one small-page KV group (P = 4), so page ids, rows and reservation counts stay
 * concrete in the assertions; the pools are LIFO, so a fresh pool's first allocation takes the highest page id
 * (16 or 32 pages). {@link MemoryPlannerPagedTest} covers the planner side of the same rules.
 */
public class PagedSequencesTest {

    private static final int PAGE = 4;

    /** One KV group of {@code nCtx} rows in pages of 4, full-attention or (with a window) SWA-trimming. */
    private static PagedSequences memory(int nCtx, int parallel, boolean unified, boolean window, int nSwa) {
        MemoryConfig cfg = MemoryConfig.builder()
                .typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(nCtx).ubatch(8).parallel(parallel).kvPageSize(PAGE).kvUnified(unified)
                .build();
        KvWindow w = window ? KvWindow.of(SwaType.STANDARD, nSwa, true) : KvWindow.FULL_CAUSAL;
        MemoryPlan.KvGroup g = MemoryPlan.KvGroup.of(0, nCtx, false, w, PAGE, nCtx / PAGE);
        KvLayout kv = KvLayout.of(nCtx, false, 8, 8, KvDType.F16, KvDType.F16, w);
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, nCtx, cfg.ubatch())
                .group(g)
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(kv, 0).build())
                .config(cfg)
                .build();
        return new PagedSequences(plan, parallel);
    }

    /** Prepares the step over {@code tokenPositions} and applies its reservation. */
    private static PagedStep step(PagedSequences mem, int seq, int[] tokenPositions) {
        PagedStep s = reserve(mem, seq, tokenPositions);
        mem.commit(s);
        return s;
    }

    private static PagedStep reserve(PagedSequences mem, int seq, int[] tokenPositions) {
        StepLayout layout = StepLayout.of(tokenPositions.length, new int[]{seq}, new int[]{0},
                new int[]{tokenPositions.length}, new int[]{tokenPositions[0]}, new int[]{0}, true);
        return mem.prepare(layout);
    }

    private static int rowOf(int page, int slot) {
        return page * PAGE + (slot & (PAGE - 1));
    }

    private static int[] tokens(int first, int len) {
        int[] out = new int[len];
        for (int i = 0; i < len; i++) {
            out[i] = first + i;
        }
        return out;
    }

    @Test
    public void aTailRemovalInsideAPageMovesTheCursorToTheLogicalSlotOfTheFirstClearedRow() {
        PagedSequences mem = memory(64, 2, true, false, 0);
        // eleven slots fill logical pages 0..2, which the LIFO pool maps to physical pages 15, 14, 13
        PagedStep first = step(mem, 0, tokens(0, 11));
        int page2 = first.pageTableEntry(0, 0, 2);
        assertTrue("the logical and physical page ids differ", page2 != 2);

        // trimming positions 9 and 10 clears part of logical page 2: the next append resumes at slot 9 of that
        // page (the prefix cache's adoption path), not at a pool cell index
        mem.remove(0, 9, -1);
        PagedStep again = step(mem, 0, tokens(9, 3));
        for (int i = 0; i < 3; i++) {
            assertEquals("position " + (9 + i), rowOf(page2, 9 + i), again.cellIdx(0)[i]);
        }
        assertEquals(11, mem.posMax(0));
    }

    @Test
    public void anInteriorRemovalLeavesHolesAndKeepsTheCursorAboveTheLiveRows() {
        PagedSequences mem = memory(64, 2, true, false, 0);
        PagedStep first = step(mem, 0, tokens(0, 11));
        int[] liveCells = new int[11];
        System.arraycopy(first.cellIdx(0), 0, liveCells, 0, 11);
        mem.remove(0, 2, 5);                              // positions 2..4 leave; 5..10 stay live above them
        assertEquals(10, mem.posMax(0));
        PagedStep more = step(mem, 0, tokens(11, 2));
        int page2 = first.pageTableEntry(0, 0, 2);
        assertEquals("position 11 appends at slot 11", rowOf(page2, 11), more.cellIdx(0)[0]);
        assertEquals("position 12 appends at slot 12", rowOf(page3(more), 12), more.cellIdx(0)[1]);
        for (int cell : more.cellIdx(0)) {
            for (int p = 5; p <= 10; p++) {
                assertTrue("the append never overwrites live position " + p, cell != liveCells[p]);
            }
        }
        assertEquals(12, mem.posMax(0));
    }

    private static int page3(PagedStep step) {
        return step.pageTableEntry(0, 0, 3);
    }

    /**
     * The tree verify's compaction against a page boundary: each round moves the accepted rows down onto the
     * pending positions and trims the rest, so the moved-away cells die below the growing cursor. When a whole
     * back page ends up dead this way, its removal still pulls the cursor to the page's logical start — a cursor
     * above the shorter page list breaks the next append's page addressing (and any prefix reuse check) instead.
     */
    @Test
    public void aCompactionThatDeadensABackPagePullsTheCursorToItsStart() {
        PagedSequences mem = memory(64, 2, true, false, 0);
        step(mem, 0, tokens(0, 4));
        mem.appendTokens(0, tokens(0, 4));
        step(mem, 0, tokens(4, 4));                  // two full pages; the cursor sits at slot 8
        mem.appendTokens(0, tokens(4, 4));

        // the compaction gathers positions 4..7 onto the slots of positions 0..3: every own cell of logical
        // page 1 dies below the cursor, as a verify round's moved-away cells do in the aggregate
        mem.moveRows(0, new int[]{4, 5, 6, 7}, new int[]{0, 1, 2, 3});
        assertEquals(7, mem.posMax(0));

        // the trim of the moved-from positions finds page 1 dead: the page leaves and the cursor pulls to the
        // page's start (slot 4), keeping the page list exactly as long as the cursor needs
        mem.remove(0, 4, -1);
        assertEquals(3, mem.posMax(0));

        // the next append resumes at slot 4 with a fresh tail page and the addressing stays dense
        PagedStep again = step(mem, 0, tokens(4, 2));
        int page1 = again.pageTableEntry(0, 0, 1);
        assertEquals(rowOf(page1, 4), again.cellIdx(0)[0]);
        assertEquals(rowOf(page1, 5), again.cellIdx(0)[1]);
        assertEquals(5, mem.posMax(0));
        mem.appendTokens(0, tokens(4, 2));
        assertEquals(6, mem.sequence(0).nPast());
    }

    /**
     * The tree verify's compaction when a row move unmaps a row the trim then removes: the moved-from slot dies
     * under a page whose other rows stay live, so the trim's page walk clears nothing there. The cursor still has
     * to follow the live rows — a dead slot held under it puts the next append at a slot whose stored position
     * differs from the slot, which the paged attention's fresh-window rule (a fresh row indexes its tree mask bits
     * by kv_pos - (kv_end - n_fresh)) turns into wrong visibility for the GPU verify batch.
     */
    @Test
    public void aTrimOfMovedAwayRowsStillPullsTheCursorToTheLiveRows() {
        PagedSequences mem = memory(64, 2, true, false, 0);
        step(mem, 0, tokens(0, 6));
        mem.appendTokens(0, tokens(0, 6));
        PagedStep verify = step(mem, 0, tokens(6, 4));   // rows 6..9 at slots 6..9; slot 9 opens page 2
        mem.appendTokens(0, tokens(6, 4));

        // the accepted path moves positions 8 and 9 down onto 7 and 8: slots 8 and 9 die while slots 7 and 8
        // take the moved rows, so the live prefix ends at slot 8
        mem.moveRows(0, new int[] {8, 9}, new int[] {7, 8});
        assertEquals(9, mem.posMax(0));

        // the trim of position 9 finds no live row: its slot died in the move and the page's other row (position
        // 8, a moved-in row below the trim) is not in the range
        mem.remove(0, 9, -1);
        assertEquals(8, mem.posMax(0));

        // the next append resumes at slot 9 — the first slot past the live rows — not at the stale cursor 10
        PagedStep again = step(mem, 0, tokens(9, 2));
        int page2 = verify.pageTableEntry(0, 0, 2);
        assertEquals(rowOf(page2, 9), again.cellIdx(0)[0]);
        assertEquals(rowOf(page2, 10), again.cellIdx(0)[1]);
        assertEquals(10, mem.posMax(0));
        mem.appendTokens(0, tokens(9, 2));
        assertEquals(11, mem.sequence(0).nPast());
    }

    @Test
    public void slotJMapsToItsPageAndRowWithoutShiftingOrTrimming() {
        PagedSequences mem = memory(64, 2, true, false, 0);
        assertEquals(PAGE, mem.pageSize());
        assertEquals(16, mem.pageCount());
        assertEquals(16, mem.maxPages());
        assertEquals(2, mem.maxSequences());

        // eight slots = two pages of the list; the pool hands them out LIFO, highest page id first
        PagedStep eight = reserve(mem, 0, tokens(0, 8));
        int pageA = mem.pageCount() - 1;
        int pageB = mem.pageCount() - 2;
        assertEquals(pageA, eight.pageTableEntry(0, 0, 0));
        assertEquals(pageB, eight.pageTableEntry(0, 0, 1));
        for (int k = 2; k < mem.maxPages(); k++) {
            assertEquals(-1, eight.pageTableEntry(0, 0, k));
        }
        for (int j = 0; j < 8; j++) {
            int page = j / PAGE == 0 ? pageA : pageB;
            assertEquals("slot " + j, rowOf(page, j), eight.cellIdx(0)[j]);
        }
        mem.commit(eight);

        // kv_pos of the written rows carries the positions, ascending by page block of slots
        int[] writtenRows = eight.kvPosRows(0);
        int[] values = eight.kvPosValues(0);
        assertEquals(8, writtenRows.length);
        for (int i = 0; i < PAGE; i++) {
            assertEquals(rowOf(pageA, i), writtenRows[i]);
            assertEquals(i, values[i]);
            assertEquals(rowOf(pageB, i), writtenRows[PAGE + i]);
            assertEquals(PAGE + i, values[PAGE + i]);
        }

        // the next append takes fresh pages and continues the slot/row mapping
        PagedStep more = step(mem, 0, tokens(8, 8));
        int pageC = mem.pageCount() - 3;
        int pageD = mem.pageCount() - 4;
        assertEquals(pageC, more.pageTableEntry(0, 0, 2));
        assertEquals(pageD, more.pageTableEntry(0, 0, 3));
        for (int j = 8; j < 16; j++) {
            int page = j / PAGE == 2 ? pageC : pageD;
            assertEquals("slot " + j, rowOf(page, j), more.cellIdx(0)[j - 8]);
        }

        // the token record and the row accounting follow the commits
        mem.appendTokens(0, tokens(0, 16));
        assertEquals(0, mem.posMin(0));
        assertEquals(15, mem.posMax(0));
        assertEquals(16, mem.sequence(0).nPast());
        assertEquals(0, mem.sequence(1).nPast());
        assertEquals(64, mem.totalCells());
        assertEquals(64 - 16, mem.freeCells()); // four pages hold positions
    }

    @Test
    public void copyOnWriteOfASharedTailOnWrite() {
        PagedSequences mem = memory(64, 2, true, false, 0);
        PagedStep first = step(mem, 0, tokens(0, 4));
        int pageA = mem.pageCount() - 1;

        // sharing copies nothing by itself (shared prompts page-share, no multi-seq tokens)
        mem.copy(0, 1, 0, 2);
        assertEquals(0, mem.pendingCopies().size());

        // the destination continues inside the shared page: the write privatizes it, emitting one page copy
        PagedStep cow = step(mem, 1, tokens(2, 2));
        List<PageCopy> copies = cow.pageCopies();
        assertEquals(1, copies.size());
        PageCopy c = copies.get(0);
        assertEquals(0, c.group());
        assertEquals(pageA, c.srcPage());
        assertEquals(0, c.srcRow0());
        assertEquals(2, c.rows());                        // only the rows the destination owns are duplicated
        int priv = c.dstPage();

        // the appends' rows are rows of the private page, carrying the continued positions
        int[] writtenRows = cow.kvPosRows(0);
        int[] writtenValues = cow.kvPosValues(0);
        assertEquals(2, writtenValues.length);
        assertEquals(rowOf(priv, 2), writtenRows[0]);
        assertEquals(rowOf(priv, 3), writtenRows[1]);
        assertEquals(2, writtenValues[0]);
        assertEquals(3, writtenValues[1]);

        // after commit: the private page sits in the second sequence's list, the source stays with the first
        PagedStep check = reserve(mem, 0, tokens(4, 2));
        assertEquals(pageA, check.pageTableEntry(0, 0, 0));
        check = reserve(mem, 1, tokens(4, 2));
        assertEquals(priv, check.pageTableEntry(0, 1, 0));

        // the source's reference moved onto the pending copy until its device execution completed
        assertEquals(1, mem.pendingCopies().size());
        mem.releaseCopies(c);
        assertEquals(0, mem.pendingCopies().size());
        // commit mirrored the private page's copied rows into the position mirror: one absolute sets report over
        // the two rows the destination owned; the source's rows above its cursor stay behind (the step's own
        // kv_pos writes fill rows 2 and 3)
        List<RowReport> reports = mem.pendingReports();
        assertEquals(1, reports.size());
        RowReport mirror = reports.get(0);
        assertTrue(mirror.absolute());
        assertEquals(2, mirror.rows().length);
        for (int i = 0; i < 2; i++) {
            assertEquals(rowOf(priv, i), mirror.rows()[i]);
            assertEquals(i, mirror.values()[i]);
        }
    }

    @Test
    public void removeOnASharedPageEmitsOneCopyAndLeavesTheSecondSequenceAlone() {
        PagedSequences mem = memory(64, 2, true, false, 0);
        PagedStep first = step(mem, 0, tokens(0, 4));
        int pageA = mem.pageCount() - 1;
        mem.copy(0, 1, 0, -1);                            // both sequences share page A

        mem.remove(0, 1, -1);                             // positions 1..3 of the first sequence

        // exactly one whole-page copy into a fresh private page
        List<PageCopy> pending = mem.pendingCopies();
        assertEquals(1, pending.size());
        PageCopy c = pending.get(0);
        assertEquals(pageA, c.srcPage());
        assertEquals(PAGE, c.rows());
        int priv = c.dstPage();

        // the device reports the private page's mirrored rows (absolute sets) and then its emptied rows (clears),
        // never page A's rows
        List<RowReport> reports = mem.pendingReports();
        assertEquals(2, reports.size());
        RowReport mirror = reports.get(0);
        assertTrue(mirror.absolute());
        assertEquals(PAGE, mirror.rows().length);
        for (int i = 0; i < PAGE; i++) {
            assertTrue("mirrored row " + mirror.rows()[i], mirror.rows()[i] >= rowOf(priv, 0)
                    && mirror.rows()[i] < rowOf(priv, 0) + PAGE);
            assertEquals(i, mirror.values()[i]);
        }
        RowReport r = reports.get(1);
        assertTrue(r.absolute());
        assertEquals(PAGE - 1, r.rows().length);
        for (int i = 0; i < r.rows().length; i++) {
            assertTrue("reported row " + r.rows()[i], r.rows()[i] >= rowOf(priv, 0)
                    && r.rows()[i] < rowOf(priv, 0) + PAGE);
            assertEquals(-1, r.values()[i]);
        }
        mem.clearReports();

        // the mutating sequence runs on the private copy; the second still resolves page A through its list
        PagedStep check = reserve(mem, 0, tokens(1, 2));  // next position after the removal's kept row
        assertEquals(priv, check.pageTableEntry(0, 0, 0));
        check = reserve(mem, 1, tokens(4, 2));
        assertEquals(pageA, check.pageTableEntry(0, 1, 0));
        assertEquals(3, mem.posMax(1));                   // the second sequence still stores positions 0..3
        assertEquals(0, mem.posMax(0));                   // the first was truncated to its kept range
        mem.releaseCopies(c);
    }

    @Test
    public void addOnASharedPageEmitsOneCopyAndLeavesTheSecondSequenceAlone() {
        PagedSequences mem = memory(64, 2, true, false, 0);
        PagedStep first = step(mem, 0, tokens(0, 4));
        int pageA = mem.pageCount() - 1;
        mem.copy(0, 1, 0, -1);

        mem.add(0, 0, -1, 10);                            // RoPE re-rotation of the first sequence only

        // exactly one whole-page copy
        List<PageCopy> pending = mem.pendingCopies();
        assertEquals(1, pending.size());
        PageCopy c = pending.get(0);
        assertEquals(pageA, c.srcPage());
        int priv = c.dstPage();

        // one absolute mirror report for the copied rows, then one relative shift report through the private
        // page's rows, carrying the shifted positions
        List<RowReport> reports = mem.pendingReports();
        assertEquals(2, reports.size());
        RowReport mirrored = reports.get(0);
        assertTrue(mirrored.absolute());
        RowReport r = reports.get(1);
        assertEquals(10, r.delta());
        assertTrue(!r.absolute());
        int[] reportedRows = r.rows();
        int[] newPositions = r.values();
        assertEquals(PAGE, reportedRows.length);
        for (int i = 0; i < PAGE; i++) {
            assertTrue(reportedRows[i] >= rowOf(priv, 0) && reportedRows[i] < rowOf(priv, 0) + PAGE);
            assertEquals(i + 10, newPositions[i]);
        }
        mem.clearReports();

        // the second sequence still resolves page A at its untouched rows
        PagedStep check = reserve(mem, 1, tokens(4, 2));
        assertEquals(pageA, check.pageTableEntry(0, 1, 0));
        assertEquals(3, mem.posMax(1));
        mem.releaseCopies(c);
    }

    @Test
    public void swaTrimOfASharedPageWritesNothingAndLeavesTheSecondSequenceAlone() {
        PagedSequences mem = memory(128, 2, true, true, 9);
        step(mem, 0, tokens(0, 8));                       // pages A (0..3), B (4..7)
        int pageA = mem.pageCount() - 1;
        int pageB = mem.pageCount() - 2;
        mem.copy(0, 1, 0, -1);                            // fully shared; committed 8 <= 9 trims nothing
        mem.clearReports();

        // this step grows committed to 12; once its trunk ran, the trim follows the next query: 12 - 9 = 3
        mem.finish(step(mem, 0, tokens(8, 4)));

        // the straddling page A is shared: the trim copies nothing and clears none of its rows, which the
        // SWA mask hides from the first sequence's queries
        assertTrue(mem.pendingCopies().isEmpty());
        assertTrue(mem.pendingReports().isEmpty());

        // the second sequence still resolves A and B through its own list, its positions untouched
        PagedStep check = reserve(mem, 1, tokens(8, 4));
        assertEquals(pageA, check.pageTableEntry(0, 1, 0));
        assertEquals(pageB, check.pageTableEntry(0, 1, 1));
        assertEquals(7, mem.posMax(1));

        // once the first sequence holds A alone, its next trim clears the rows below the window in place
        mem.remove(1, 0, -1);
        mem.trim(0);
        assertTrue(mem.pendingCopies().isEmpty());
        assertEquals(1, mem.pendingReports().size());
        RowReport r = mem.pendingReports().get(0);
        assertArrayEquals(new int[] {rowOf(pageA, 0), rowOf(pageA, 1), rowOf(pageA, 2)}, r.rows());
    }

    @Test
    public void swaTrimOfASharedWholePageDropsOnlyTheSequencesReference() {
        PagedSequences mem = memory(128, 2, true, true, 4);
        step(mem, 0, tokens(0, 8));                       // pages A (0..3), B (4..7)
        int pageA = mem.pageCount() - 1;
        mem.copy(0, 1, 0, -1);
        int free = mem.freeCells();
        mem.clearReports();

        // committed 12, threshold 8 once the step finished: A and B lie below the window and leave the first
        // sequence
        mem.finish(step(mem, 0, tokens(8, 4)));
        assertTrue("the trim copies no page", mem.pendingCopies().isEmpty());
        assertTrue("the shared rows stay for the second sequence", mem.pendingReports().isEmpty());
        assertEquals("one new page, the shared ones stay allocated", free - PAGE, mem.freeCells());
        PagedStep check = reserve(mem, 1, tokens(8, 4));
        assertEquals(pageA, check.pageTableEntry(0, 1, 0));
        assertEquals(0, mem.posMin(1));
        assertEquals(7, mem.posMax(1));
    }

    /**
     * The trunk runs over a committed step, so the commit keeps every row the step's own queries read — the window
     * of its first query — and the rows below the next query's window leave once the step finished.
     */
    @Test
    public void aStepKeepsTheWindowOfItsFirstQueryUntilItFinishes() {
        PagedSequences mem = memory(128, 1, true, true, 4);
        mem.finish(step(mem, 0, tokens(0, 8)));          // threshold 8 - 4: page A (0..3) leaves
        int pageB = mem.pageCount() - 2;
        assertEquals("rows 4..7 stay live", 128 - 4, mem.freeCells());
        mem.clearReports();

        PagedStep s = step(mem, 0, tokens(8, 8));        // queries 8..15; the one at 8 reads from 5
        assertEquals("the commit clears only row 4, below the first query's window", 1,
                mem.pendingReports().size());
        assertArrayEquals(new int[] {rowOf(pageB, 0)}, mem.pendingReports().get(0).rows());
        assertEquals("rows 5..15 stay live through the trunk", 128 - 11, mem.freeCells());

        mem.finish(s);                                   // next query 16 reads from 13: rows 5..11 leave
        assertEquals("the window of the next query stays", 128 - 4, mem.freeCells());
        assertEquals(15, mem.posMax(0));
    }

    @Test
    public void reservationIsAllOrNothingOnExhaustion() {
        PagedSequences mem = memory(32, 2, true, false, 0);      // 8 pages, unified cap of 32 slots
        step(mem, 0, tokens(0, 16));                             // 4 pages
        step(mem, 1, tokens(0, 12));                             // 3 pages: exactly one page free
        int usedRows = mem.totalCells() - mem.freeCells();

        // a two-page plan over one free page: the reservation count fires before any pool was touched
        MemoryFullException e = assertThrows(MemoryFullException.class, () -> reserve(mem, 0, tokens(16, 8)));
        assertTrue(e.getMessage(), e.getMessage().contains("memory full"));
        assertThrows(MemoryFullException.class, () -> step(mem, 0, tokens(16, 8)));
        assertEquals(usedRows, mem.totalCells() - mem.freeCells()); // no row was claimed

        // a mixed layout fails whole as well: no page of either segment is reserved
        assertThrows(MemoryFullException.class,
                () -> mem.prepare(StepLayout.of(12, new int[]{0, 1}, new int[]{0, 8}, new int[]{8, 4},
                        new int[]{16, 12}, new int[]{0}, true)));
        assertEquals(usedRows, mem.totalCells() - mem.freeCells());

        // the one-page plan that fits then runs with the page the failed plans never reserved
        PagedStep ok = step(mem, 0, tokens(16, 4));
        assertTrue(ok.pageTableEntry(0, 0, 4) >= 0);

        // releasing a sequence opens the pool again (whole pages dropped, their rows emptied): seq 0 held
        // 20 rows, so only seq 1's 12 rows stay used
        mem.keep(1);
        assertEquals(12, mem.totalCells() - mem.freeCells());
    }

    @Test
    public void reservationRespectsThePerSequenceCap() {
        PagedSequences mem = memory(40, 2, false, false, 0);     // non-unified: cap of nCtx / parallel = 20
        assertEquals(10, mem.pageCount());
        assertEquals(5, mem.maxPages());
        step(mem, 0, tokens(0, 12));
        PagedStep full = step(mem, 1, tokens(0, 12));

        // positions 12..21 push past the cap of 20: MemoryFull before anything reserved
        int beforeUsed = mem.totalCells() - mem.freeCells();
        MemoryFullException e = assertThrows(MemoryFullException.class,
                () -> reserve(mem, 1, tokens(12, 10)));
        assertTrue(e.getMessage(), e.getMessage().contains("memory full"));
        assertEquals(beforeUsed, mem.totalCells() - mem.freeCells());

        // the tail that fits within the cap commits and fills the sequence's page list exactly
        step(mem, 1, tokens(12, 8));
        assertEquals(19, mem.posMax(1));
        assertTrue(full.pageTableEntry(0, 1, 2) >= 0);           // three entries = 12 slots of the first step
        assertTrue(full.pageTableEntry(0, 1, 3) < 0);
    }

    @Test
    public void copySharesWholePagesIncludingAPartialTail() {
        PagedSequences mem = memory(64, 2, true, false, 0);
        step(mem, 0, tokens(0, 8));                              // pages A (0..3), B (4..7)
        int pageA = mem.pageCount() - 1;
        int pageB = mem.pageCount() - 2;

        // positions [2, 6): page B shares whole, the partial page A gets a private copy of its in-range rows
        mem.copy(0, 1, 2, 6);

        List<PageCopy> pending = mem.pendingCopies();
        assertEquals(1, pending.size());
        PageCopy c = pending.get(0);
        assertEquals(pageA, c.srcPage());
        assertEquals(2, c.srcRow0());                            // the copied range starts inside A
        assertEquals(PAGE - 2, c.rows());

        // the mirror of the partial copy is reported so the device learns the copied rows' positions
        List<RowReport> reports = mem.pendingReports();
        assertEquals(1, reports.size());
        RowReport r = reports.get(0);
        assertEquals(PAGE - 2, r.rows().length);
        for (int i = 0; i < r.rows().length; i++) {
            assertEquals(rowOf(c.dstPage(), 2 + i), r.rows()[i]);
            assertEquals(2 + i, r.values()[i]);
        }
        mem.clearReports();

        // the second sequence continues INSIDE page B (positions 6, 7 are its shared rows): the write
        // privatizes the shared tail page, and prepare reports the copy on the step
        PagedStep dst = step(mem, 1, tokens(6, 2));
        assertEquals(1, dst.pageCopies().size());
        assertEquals(pageB, dst.pageCopies().get(0).srcPage());
        // ... and the first sequence's pages are untouched by it
        PagedStep check = reserve(mem, 0, tokens(8, 2));
        assertEquals(pageA, check.pageTableEntry(0, 0, 0));
        assertEquals(pageB, check.pageTableEntry(0, 0, 1));
    }

    @Test
    public void aMultiSegmentStepCarriesTheKvPosOfEverySegment() {
        PagedSequences mem = memory(64, 2, true, false, 0);
        step(mem, 0, tokens(0, 4));                              // sequence 0 stores positions 0..3

        // one step for both sequences: sequence 0 appends positions 4..5, sequence 1 starts at 0..1
        StepLayout layout = StepLayout.of(4, new int[]{0, 1}, new int[]{0, 2}, new int[]{2, 2},
                new int[]{4, 0}, new int[0], true);
        PagedStep s = mem.prepare(layout);
        assertEquals(2, s.nSeg());

        int[] cells = s.cellIdx(0);
        int[] rows = s.kvPosRows(0);
        int[] values = s.kvPosValues(0);
        assertEquals("one kv_pos write per step token", 4, rows.length);
        for (int t = 0; t < 4; t++) {
            assertEquals("row of step token " + t, cells[t], rows[t]);
        }
        assertArrayEquals(new int[]{4, 5, 0, 1}, values);
        // the two sequences write different rows (fresh pages per sequence)
        assertTrue(cells[0] != cells[2] && cells[0] != cells[3]);

        mem.commit(s);
        assertEquals(5, mem.posMax(0));
        assertEquals(1, mem.posMax(1));
        // both sequences continue from their committed positions in later steps
        step(mem, 1, tokens(2, 2));
        assertEquals(3, mem.posMax(1));
    }
}
