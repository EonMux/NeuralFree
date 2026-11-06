package com.eonmux.neuralfree.gguf.memory.paged;

import java.util.Objects;

/**
 * The pages each KV group of a {@link PagedSequences} must give sequences as they grow: the page count an
 * admission compares to the idle pages before it lets a prompt in (vLLM's {@code KVCacheManager.allocate_slots},
 * which refuses a request whose {@code get_num_blocks_to_allocate} exceeds the free blocks). A row on a page another
 * holder keeps is written on a private copy first (copy-on-write), so a shared partial page costs a page like a new
 * one; whole shared pages cost nothing. A windowed group trims the rows its window left behind at every commit, so a
 * sequence never holds more of its pages than the window spans ({@link #windowPages}). Reads the memory's state at
 * each call; confined to the memory's thread.
 */
public final class PageDemand {

    private final PagedSequences paged;

    /** The page demand over {@code paged}'s pools. */
    public PageDemand(PagedSequences paged) {
        this.paged = Objects.requireNonNull(paged, "paged");
    }

    /** KV groups, each with its own page pool. */
    public int groups() {
        return paged.groupsCount();
    }

    /** Idle pages of group {@code g}'s pool. */
    public int freePages(int g) {
        return paged.freePages(g);
    }

    /**
     * Pages group {@code g} gives {@code seq} to append {@code rows} more rows: private copies of the shared pages
     * the writes reach and the new tail pages, the count a step's reservation takes ({@link PagedSequences#prepare}).
     *
     * @throws IndexOutOfBoundsException when {@code seq} or {@code g} is outside the memory
     */
    public int toAppend(int seq, int g, int rows) {
        Objects.checkIndex(seq, paged.maxSequences());
        checkGroup(g);
        if (rows <= 0) {
            return 0;
        }
        return capToWindow(g, paged.pagesToWrite(seq, g, rows));
    }

    /**
     * Pages group {@code g} gives a sequence that holds nothing yet to come to hold {@code rows} rows, the leading
     * {@code shared} of them on pages another holder keeps (an adopted cached prefix or a fork's copied prefix):
     * every page past the whole shared pages, so a shared partial last page counts the private copy its first write
     * takes.
     *
     * @throws IllegalArgumentException  when {@code rows} or {@code shared} is negative
     * @throws IndexOutOfBoundsException when {@code g} is outside the memory
     */
    public int toHold(int g, int rows, int shared) {
        checkGroup(g);
        if (rows < 0 || shared < 0) {
            throw new IllegalArgumentException("a sequence holds " + rows + " rows, " + shared + " shared: neither"
                    + " may be negative");
        }
        int page = paged.pageSize();
        int whole = Math.min(shared, rows) / page;
        return capToWindow(g, (rows + page - 1) / page - whole);
    }

    /**
     * Pages a sequence of windowed group {@code g} spans at most: its window plus a ubatch of rows, page-rounded,
     * and the page a trim leaves partly covered (the per-sequence share the memory planner sizes the group by).
     */
    int windowPages(int g) {
        int page = paged.pageSize();
        return (paged.windowWidth(g) + paged.plan().ubatch() + page - 1) / page + 1;
    }

    private int capToWindow(int g, int pages) {
        return paged.trims(g) ? Math.min(pages, windowPages(g)) : pages;
    }

    private int checkGroup(int g) {
        return Objects.checkIndex(g, paged.groupsCount());
    }
}
