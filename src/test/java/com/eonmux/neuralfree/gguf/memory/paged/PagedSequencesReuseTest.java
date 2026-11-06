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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The KV chunk reuse primitive ({@link PagedSequences#reusePages}): a run of another sequence's held rows re-appends
 * at the claiming sequence's cursor — a {@code delta} 0 run whose rows meet the cursor page-aligned shares whole
 * cached pages (no allocation, no copies), everything else copies with the copied rows' positions re-rotated by the
 * run's uniform delta and reported for the layers' re-rotation. The page accounting is all-or-nothing: a run the
 * pools cannot serve fails with {@link MemoryFullException} and leaves every pool, list and cursor unchanged; a
 * shared run allocates nothing and succeeds on a full pool. Each copied source page holds one pending-copy
 * reference until {@link PagedSequences#releaseCopies} hands it back, so the cache's rows stay alive while the
 * copy is in flight.
 *
 * <p>Plans are built by hand over one small-page KV group (P = 4), as in {@link PagedSequencesTest}; the pools are
 * LIFO, so fresh allocations take the highest page ids.
 */
public class PagedSequencesReuseTest {

    private static final int PAGE = 4;

    /** One full-attention KV group of {@code nCtx} rows in pages of 4. */
    private static PagedSequences memory(int nCtx, int parallel) {
        MemoryConfig cfg = MemoryConfig.builder()
                .typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(nCtx).ubatch(8).parallel(parallel).kvPageSize(PAGE).kvUnified(true)
                .build();
        KvWindow w = KvWindow.FULL_CAUSAL;
        MemoryPlan.KvGroup g = MemoryPlan.KvGroup.of(0, nCtx, false, w, PAGE, nCtx / PAGE);
        KvLayout kv = KvLayout.of(nCtx, false, 8, 8, KvDType.F16, KvDType.F16, w);
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, nCtx, cfg.ubatch())
                .group(g)
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(kv, 0).build())
                .config(cfg)
                .build();
        return new PagedSequences(plan, parallel);
    }

    /** Commits one step of {@code seq} over the consecutive positions {@code first..first+len}. */
    private static PagedStep step(PagedSequences mem, int seq, int first, int len) {
        StepLayout layout = StepLayout.of(len, new int[]{seq}, new int[]{0}, new int[]{len},
                new int[]{first}, new int[]{0}, true);
        PagedStep s = mem.prepare(layout);
        mem.commit(s);
        return s;
    }

    private static int[] tokens(int first, int len) {
        int[] out = new int[len];
        for (int i = 0; i < len; i++) {
            out[i] = 100 + first + i;
        }
        return out;
    }

    @Test
    public void aZeroDeltaRunSharesWholePagesAndCopiesTheRest() {
        PagedSequences mem = memory(64, 2);
        PagedStep cached = step(mem, 0, 0, 12);         // the cache side: three pages
        int[] mid = mem.blockPages(0, 1, PAGE);                 // source block 1: rows [4, 8)
        int[] top = mem.blockPages(0, 2, PAGE);                 // source block 2: rows [8, 12)
        assertEquals(cached.pageTableEntry(0, 0, 2), top[0]);
        step(mem, 1, 0, 4);                             // the claimant's own first page
        int before = mem.freeCells();

        // positions [4, 10) re-append source rows [4, 10) at shift 0: one whole page that shares, then a
        // partial page's two rows that copy onto a fresh one
        mem.reusePages(1, new int[][]{mid, top}, 0, 4, 6, 0, tokens(4, 6), -1);
        assertEquals("the claimant's cursor moved past the run", 9, mem.posMax(1));
        assertEquals("the token record holds the re-appended rows", 6, mem.sequence(1).nPast());
        assertEquals("the share takes no cells, the copy its two", before - 2, mem.freeCells());

        // the run's first whole page shares the cache's page; its last two rows copy onto a fresh one
        List<PageCopy> pending = mem.pendingCopies();
        assertEquals(1, pending.size());
        PageCopy copy = pending.get(0);
        assertEquals(top[0], copy.srcPage());
        assertEquals(0, copy.srcRow0());
        assertEquals(0, copy.dstRow0());
        assertEquals(2, copy.rows());

        // the copied rows' mirror report names the fresh page's cells at the source positions
        List<RowReport> reports = mem.pendingReports();
        assertEquals(1, reports.size());
        RowReport report = reports.get(0);
        assertTrue(report.absolute());
        assertEquals(copy.dstPage() * PAGE, report.rows()[0]);
        assertArrayEquals(new int[]{8, 9}, report.values());

        // a next row fills the copied page's own free slot: the table names the shared and copied pages
        PagedStep probe = step(mem, 1, 10, 1);
        assertEquals("the shared page sits in the claimant's list", mid[0], probe.pageTableEntry(0, 1, 1));
        assertEquals("the copied page sits past it", copy.dstPage(), probe.pageTableEntry(0, 1, 2));
        assertEquals(before - 3, mem.freeCells());

        // the copy's reference hands back: the source page stays with the cache, the pools keep their count
        mem.releaseCopies(copy);
        assertEquals(0, mem.pendingCopies().size());
        assertEquals(before - 3, mem.freeCells());
    }

    @Test
    public void aShiftedRunCopiesAndReportsTheReRotatedPositions() {
        PagedSequences mem = memory(64, 2);
        step(mem, 0, 0, 12);
        int[] top = mem.blockPages(0, 2, PAGE);                 // source rows [8, 12) hold positions 8..11
        step(mem, 1, 0, 4);                             // the claimant stores positions 0..3

        // the run lands at positions [4, 8): the shift re-rotates the copied keys' positions by -4
        mem.reusePages(1, new int[][]{top}, 0, 4, 4, -4, tokens(8, 4), -1);
        assertEquals(7, mem.posMax(1));
        assertEquals(4, mem.sequence(1).nPast());

        PageCopy copy = mem.pendingCopies().get(0);
        assertArrayEquals("the copied cells carry the shifted positions", new int[]{4, 5, 6, 7},
                mem.pagePositions(0, copy.dstPage(), PAGE));

        // one relative report over the copied rows, the uniform delta the layers re-rotate by
        List<RowReport> reports = mem.pendingReports();
        assertEquals(1, reports.size());
        RowReport report = reports.get(0);
        assertFalse(report.absolute());
        assertEquals(-4, report.delta());
        assertEquals(copy.dstPage() * PAGE, report.rows()[0]);
        assertArrayEquals(new int[]{4, 5, 6, 7}, report.values());
        mem.releaseCopies(copy);
        assertEquals(0, mem.pendingCopies().size());
    }

    @Test
    public void aZeroDeltaAlignedRunSharesItsPagesEvenWhenThePoolIsFull() {
        PagedSequences mem = memory(16, 2);                     // four pages
        step(mem, 0, 0, 12);                            // three pages
        step(mem, 1, 0, 4);                             // the fourth page: the pool is full
        int[] mid = mem.blockPages(0, 1, PAGE);
        int[] top = mem.blockPages(0, 2, PAGE);

        // positions [4, 12) at shift 0, the cursor page-aligned: both pages share, none allocates
        mem.reusePages(1, new int[][]{mid, top}, 0, 4, 8, 0, tokens(4, 8), -1);
        assertEquals(11, mem.posMax(1));
        assertEquals(8, mem.sequence(1).nPast());
        assertTrue("a shared run copies nothing", mem.pendingCopies().isEmpty());
        assertTrue("a shared run moves no cell", mem.pendingReports().isEmpty());
    }

    @Test
    public void aRunThePoolCannotServeRefusesWhole() {
        PagedSequences mem = memory(16, 2);                     // four pages
        step(mem, 0, 0, 12);
        step(mem, 1, 0, 4);                             // the pool is full
        int[] top = mem.blockPages(0, 2, PAGE);
        int cells = mem.freeCells();

        // the shifted run's copy needs a page the pool lacks: refused before any state changed
        assertThrows(MemoryFullException.class,
                () -> mem.reusePages(1, new int[][]{top}, 0, 4, 4, -4, tokens(8, 4), -1));
        assertEquals(cells, mem.freeCells());
        assertEquals("the claimant's cursor stayed put", 3, mem.posMax(1));
        assertEquals("the record took no rows", 0, mem.sequence(1).nPast());
        assertTrue(mem.pendingCopies().isEmpty());
        assertTrue(mem.pendingReports().isEmpty());
    }

    @Test
    public void aRunOutsideTheClaimantOrTheSourcePagesRefusesByName() {
        PagedSequences mem = memory(64, 2);
        step(mem, 0, 0, 12);
        int[] mid = mem.blockPages(0, 1, PAGE);

        // the run's page list must cover its rows from its first page row
        assertThrows(IllegalArgumentException.class,
                () -> mem.reusePages(1, new int[][]{mid}, 0, 4, 8, 0, tokens(4, 8), -1));
        // a claimant storing nothing cannot re-append another sequence's rows
        assertThrows(IllegalArgumentException.class,
                () -> mem.reusePages(1, new int[][]{mid, mid}, 0, 0, 8, 0, tokens(4, 8), -1));
        // the run's first destination position follows the claimant's stored ones
        step(mem, 1, 0, 4);
        assertThrows(IllegalArgumentException.class,
                () -> mem.reusePages(1, new int[][]{mid, mid}, 0, 2, 8, 0, tokens(4, 8), -1));
        assertTrue(mem.pendingCopies().isEmpty());
    }
}
