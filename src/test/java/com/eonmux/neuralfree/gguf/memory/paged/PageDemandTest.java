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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The page counts {@link PageDemand} gives admission over one KV group of 16 pages of 4 rows: a fresh sequence takes
 * the pages of its rows less the whole pages it shares, a shared partial last page counting its copy-on-write copy;
 * an append counts exactly the pages the step's reservation takes; a windowed group stops at the pages its trimmed
 * window spans.
 */
public class PageDemandTest {

    private static final int PAGE = 4;
    private static final int CTX = 64;
    private static final int UBATCH = 8;

    /** One KV group of {@code CTX} rows in pages of 4, full-attention or SWA-trimming over {@code nSwa}. */
    private static PagedSequences memory(boolean window, int nSwa) {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(CTX).ubatch(UBATCH).parallel(2).kvPageSize(PAGE).build();
        KvWindow w = window ? KvWindow.of(SwaType.STANDARD, nSwa, true) : KvWindow.FULL_CAUSAL;
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, CTX, cfg.ubatch())
                .group(MemoryPlan.KvGroup.of(0, CTX, false, w, PAGE, CTX / PAGE))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV)
                        .kv(KvLayout.of(CTX, false, 8, 8, KvDType.F16, KvDType.F16, w), 0).build())
                .config(cfg)
                .build();
        return new PagedSequences(plan, 2);
    }

    /** Reserves and commits positions {@code [from, from + len)} of {@code seq}; returns the step. */
    private static PagedStep write(PagedSequences mem, int seq, int from, int len) {
        StepLayout layout = StepLayout.of(len, new int[] {seq}, new int[] {0}, new int[] {len}, new int[] {from},
                new int[] {0}, true);
        PagedStep step = mem.prepare(layout);
        mem.commit(step);
        mem.releaseCopies(mem.pendingCopies().toArray(new PageCopy[0]));
        return step;
    }

    @Test
    public void aFreshSequenceTakesItsRowsPagesLessTheWholeSharedOnes() {
        PageDemand demand = new PageDemand(memory(false, 0));
        assertEquals(1, demand.groups());
        assertEquals(CTX / PAGE, demand.freePages(0));
        assertEquals("ten rows span three pages", 3, demand.toHold(0, 10, 0));
        assertEquals("two whole shared pages cost nothing", 1, demand.toHold(0, 10, 8));
        assertEquals("a shared partial page costs its copy", 2, demand.toHold(0, 10, 7));
        assertEquals("the last row on the shared partial page still copies it", 1, demand.toHold(0, 8, 7));
        assertEquals("rows held whole on shared pages cost nothing", 0, demand.toHold(0, 8, 8));
        assertEquals(0, demand.toHold(0, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> demand.toHold(0, -1, 0));
        assertThrows(IllegalArgumentException.class, () -> demand.toHold(0, 4, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> demand.toHold(1, 4, 0));
    }

    @Test
    public void anAppendCountsThePagesItsReservationTakes() {
        PagedSequences mem = memory(false, 0);
        PageDemand demand = new PageDemand(mem);
        assertEquals("a sequence holding nothing takes its rows' pages", 2, demand.toAppend(0, 0, 6));
        write(mem, 0, 0, 6);
        assertEquals(CTX / PAGE - 2, demand.freePages(0));
        assertEquals("rows 6 and 7 fill the private partial page", 0, demand.toAppend(0, 0, 2));

        // the fork shares both pages, the second one partial: position 5 is the donor's alone
        mem.copy(0, 1, 0, 5);
        assertEquals("the fork's first row copies the shared partial page", 1, demand.toAppend(1, 0, 1));
        assertEquals("as a fresh sequence sharing five of six rows counts it", 1, demand.toHold(0, 6, 5));
        assertEquals("the donor's rows 6 and 7 would copy it too", 1, demand.toAppend(0, 0, 2));
        assertEquals("rows reaching a third page add it", 2, demand.toAppend(1, 0, 4));

        int free = demand.freePages(0);
        PagedStep forkRow = write(mem, 1, 5, 1);
        assertEquals(1, forkRow.pageCopies().size());
        assertEquals("the count is the reservation's", free - 1, demand.freePages(0));
        assertEquals("once the fork copied it, the donor writes its page in place", 0, demand.toAppend(0, 0, 2));
        assertEquals(0, demand.toAppend(1, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> demand.toAppend(2, 0, 1));
    }

    @Test
    public void aWindowedGroupStopsAtThePagesItsWindowSpans() {
        int nSwa = 8;
        PagedSequences mem = memory(true, nSwa);
        PageDemand demand = new PageDemand(mem);
        int windowPages = (nSwa + UBATCH + PAGE - 1) / PAGE + 1;
        assertEquals(windowPages, demand.windowPages(0));
        assertEquals("a short prompt takes its own pages", 3, demand.toHold(0, 12, 0));
        assertEquals("a long one no more than the window spans", windowPages, demand.toHold(0, 48, 0));
        assertEquals(windowPages, demand.toAppend(0, 0, 48));
        for (int from = 0; from < 48; from += UBATCH) {
            write(mem, 0, from, UBATCH);
            int held = CTX / PAGE - demand.freePages(0);
            assertTrue("after position " + (from + UBATCH) + " the sequence holds " + held + " pages",
                    held <= windowPages);
        }
    }
}
