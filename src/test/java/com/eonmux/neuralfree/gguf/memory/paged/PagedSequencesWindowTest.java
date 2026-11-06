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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The sliding-window side of the paged memory's holders (the windows of context checkpoints): a block holds no
 * page of a trimming group, a held window keeps the pages a sequence holds in such a group at an exact row count
 * past the sequence's removal, and an empty sequence adopts the window beside the full-attention blocks as if it had
 * trimmed the pages before it. Adoption refuses windows that do not fit the adopted rows; the page counts a step
 * would take ({@link PagedSequences#pagesToWrite}) and a window's sole-holder pages
 * ({@link PagedSequences#windowOnlyPages}) follow the references.
 */
public class PagedSequencesWindowTest {

    private static final int PAGE = 4;
    private static final int CTX = 64;
    private static final int SWA_ROWS = 32;
    private static final int WINDOW = 4;
    private static final int SWA = 0;
    private static final int FULL = 1;

    /** Group 0 trims a window of 4 over 32 rows, group 1 keeps every row of the 64-row context. */
    private static PagedSequences memory() {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(CTX).ubatch(8).parallel(2).kvPageSize(PAGE).build();
        KvWindow swa = KvWindow.of(SwaType.STANDARD, WINDOW, true);
        KvWindow full = KvWindow.FULL_CAUSAL;
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, CTX, cfg.ubatch())
                .group(MemoryPlan.KvGroup.of(SWA, SWA_ROWS, false, swa, PAGE, SWA_ROWS / PAGE))
                .group(MemoryPlan.KvGroup.of(FULL, CTX, false, full, PAGE, CTX / PAGE))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV)
                        .kv(KvLayout.of(SWA_ROWS, false, 8, 8, KvDType.F16, KvDType.F16, swa), SWA).build())
                .layer(LayerMemoryPlan.builder(1, LayerMemoryPlan.Kind.KV)
                        .kv(KvLayout.of(CTX, false, 8, 8, KvDType.F16, KvDType.F16, full), FULL).build())
                .config(cfg)
                .build();
        return new PagedSequences(plan, 2);
    }

    /** Writes positions {@code [from, from + len)} of {@code seq} as one finished step. */
    private static void write(PagedSequences mem, int seq, int from, int len) {
        StepLayout layout = StepLayout.of(len, new int[] {seq}, new int[] {0}, new int[] {len}, new int[] {from},
                new int[] {0}, true);
        PagedStep step = mem.prepare(layout);
        mem.commit(step);
        mem.releaseCopies(mem.pendingCopies().toArray(new PageCopy[0]));
        mem.finish(step);
    }

    private static int[] ids(int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 100 + i;
        }
        return t;
    }

    /** The full-attention blocks of {@code seq}'s first {@code n} whole pages, held by the caller. */
    private static int[][] holdBlocks(PagedSequences mem, int seq, int n) {
        int[][] blocks = new int[n][];
        for (int k = 0; k < n; k++) {
            blocks[k] = mem.blockPages(seq, k, PAGE);
            assertNotNull("block " + k, blocks[k]);
            mem.holdPages(blocks[k]);
        }
        return blocks;
    }

    @Test
    public void aBlockHoldsNoPageOfATrimmingGroup() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 8);
        assertTrue(mem.trims(SWA));
        assertFalse(mem.trims(FULL));
        assertEquals(WINDOW, mem.windowWidth(SWA));
        assertEquals(0, mem.windowWidth(FULL));
        int[] block = mem.blockPages(0, 0, PAGE);
        assertEquals("the trimming group gives no page", -1, block[SWA]);
        assertTrue(block[FULL] >= 0);
        int free = mem.freePages(FULL);
        mem.holdPages(block);
        assertTrue("the sequence and the holder share the full page", mem.pagesShared(block));
        mem.remove(0, 0, -1);
        assertFalse("the holder alone references it", mem.pagesShared(block));
        assertEquals("the held page outlives the sequence", free + 1, mem.freePages(FULL));
        mem.releasePages(block);
        assertEquals(free + 2, mem.freePages(FULL));
    }

    @Test
    public void aHeldWindowOutlivesItsSequenceAndAnEmptySequenceAdoptsIt() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 8);
        write(mem, 0, 8, 4);
        // next query 12 reads from 9: the window group trimmed rows 0..7, its list holds the page of 8..11
        PageWindow w = mem.holdWindow(0, SWA, 12);
        assertEquals(SWA, w.group());
        assertEquals(2, w.firstPage());
        assertEquals(1, w.size());
        assertEquals(12, w.rows());
        assertEquals("the sequence still reads the page", 0, mem.windowOnlyPages(w));
        int[][] blocks = holdBlocks(mem, 0, 3);
        mem.remove(0, 0, -1);
        assertEquals("the window alone holds its page now", 1, mem.windowOnlyPages(w));

        mem.adoptPages(1, blocks, PAGE, ids(12), null, windowsOf(w), -1);
        assertEquals(0, mem.posMin(1));
        assertEquals(11, mem.posMax(1));
        assertEquals(0, mem.windowOnlyPages(w));
        assertEquals("the adopted window's last page is whole: the next row takes a new page", 1,
                mem.pagesToWrite(1, SWA, 1));
        write(mem, 1, 12, 4);
        assertEquals(15, mem.posMax(1));
        mem.releaseWindow(w);
        for (int[] b : blocks) {
            mem.releasePages(b);
        }
        mem.remove(1, 0, -1);
        assertEquals("every page returns", SWA_ROWS / PAGE, mem.freePages(SWA));
        assertEquals(CTX / PAGE, mem.freePages(FULL));
    }

    @Test
    public void aWindowEndingInsideAPageCopiesTheSharedPageOnTheAdoptersFirstWrite() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 10);
        PageWindow w = mem.holdWindow(0, SWA, 10);
        int[][] blocks = new int[3][];
        blocks[0] = mem.blockPages(0, 0, PAGE);
        blocks[1] = mem.blockPages(0, 1, PAGE);
        blocks[2] = mem.blockPages(0, 2, 2);
        for (int[] b : blocks) {
            mem.holdPages(b);
        }
        assertEquals("the shared last page of the window is copied before the write", 1,
                mem.pagesToWrite(0, SWA, 1));
        mem.remove(0, 0, -1);
        mem.adoptPages(1, blocks, 2, ids(10), null, windowsOf(w), -1);
        assertEquals("the adopter copies the window's shared last page", 1, mem.pagesToWrite(1, SWA, 1));
        assertEquals("and the full group's shared boundary page", 1, mem.pagesToWrite(1, FULL, 1));
    }

    @Test
    public void holdWindowRefusesAFullGroupAndARowCountTheListDoesNotCover() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 8);                             // the window group keeps the page of rows 4..7
        assertThrows(IllegalArgumentException.class, () -> mem.holdWindow(0, FULL, 8));
        assertThrows("rows 0..2 would lie on a trimmed page", IllegalArgumentException.class,
                () -> mem.holdWindow(0, SWA, 7));
        assertThrows(IllegalArgumentException.class, () -> mem.holdWindow(0, SWA, 3));
        assertThrows("an empty sequence holds no window", IllegalArgumentException.class,
                () -> mem.holdWindow(1, SWA, 0));
    }

    @Test
    public void adoptionRefusesAWindowThatDoesNotFitTheAdoptedRows() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 8);
        PageWindow w = mem.holdWindow(0, SWA, 8);
        int[][] blocks = holdBlocks(mem, 0, 2);
        int[][] one = {blocks[0]};
        assertThrows("the window was taken at 8 rows", IllegalArgumentException.class,
                () -> mem.adoptPages(1, one, PAGE, ids(4), null, windowsOf(w), -1));
        assertThrows("a trimming group needs its window", IllegalArgumentException.class,
                () -> mem.adoptPages(1, blocks, PAGE, ids(8), null, new PageWindow[2], -1));
        assertThrows("the five-argument adoption has no window", IllegalArgumentException.class,
                () -> mem.adoptPages(1, blocks, PAGE, ids(8), null));
        int[][] withSwaPage = {blocks[0], {w.page(0), blocks[1][FULL]}};
        assertThrows("a block gives no page to a trimming group", IllegalArgumentException.class,
                () -> mem.adoptPages(1, withSwaPage, PAGE, ids(8), null, windowsOf(w), -1));
        assertEquals("nothing was adopted", -1, mem.posMax(1));
        mem.adoptPages(1, blocks, PAGE, ids(8), null, windowsOf(w), -1);
        assertEquals(7, mem.posMax(1));
    }

    @Test
    public void aWindowReportsItsPagesFrontFirst() {
        PagedSequences mem = memory();
        write(mem, 0, 0, 6);                             // next query 6 reads from 3: nothing leaves the list
        PageWindow w = mem.holdWindow(0, SWA, 6);
        assertEquals(0, w.firstPage());
        assertEquals(2, w.size());
        assertArrayEquals(new int[] {w.page(0), w.page(1)}, w.pages());
        assertThrows(IndexOutOfBoundsException.class, () -> w.page(2));
        assertTrue(w.toString(), w.toString().contains("rows=6"));
    }

    private static PageWindow[] windowsOf(PageWindow w) {
        PageWindow[] windows = new PageWindow[2];
        windows[SWA] = w;
        return windows;
    }
}
