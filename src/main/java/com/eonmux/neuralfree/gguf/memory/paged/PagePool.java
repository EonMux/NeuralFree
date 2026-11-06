package com.eonmux.neuralfree.gguf.memory.paged;

/**
 * Page pool of one KV group: {@code nPages} physical pages of {@code pageSize} rows each, with a reference count per
 * page and a LIFO free list. This is the backing store behind {@link PagedSequences} and every other holder of a
 * page (the reference-counted page rule: a page is reused for other data only
 * at reference count 0, and {@link #ref} only takes a reference on a page some holder already owns, so no holder can
 * revive a freed page).
 *
 * <p>Every argument is range-checked ({@link #alloc} and {@link #free} take none); {@link #ref} requires
 * {@code refCount(p) > 0} and throws otherwise, so only {@link #alloc} moves a page from the free list. Not
 * thread-safe; owned by the engine thread that runs the memory's steps.
 */
public final class PagePool {

    private final int pageSize;
    /** refCounts[p] >= 0; the pool owns one reference for as long as a page is allocated. */
    private final int[] refCounts;
    /** The free pages, with the most recently freed page last; freed from the end. */
    private final int[] freeStack;
    private int freeCount;

    /**
     * A pool of {@code nPages} idle pages of {@code pageSize} rows.
     *
     * @throws IllegalArgumentException for a non-positive page or page size
     */
    public PagePool(int pageSize, int nPages) {
        if (pageSize <= 0) {
            throw new IllegalArgumentException("page pool needs a positive page size, got " + pageSize);
        }
        if (nPages <= 0) {
            throw new IllegalArgumentException("page pool needs a positive page count, got " + nPages);
        }
        this.pageSize = pageSize;
        this.refCounts = new int[nPages];
        this.freeStack = new int[nPages];
        for (int p = 0; p < nPages; p++) {
            freeStack[p] = p;
        }
        this.freeCount = nPages;
    }

    /** Rows of one page (the configured {@code neuralfree.kv.page.size}). */
    public int pageSize() {
        return pageSize;
    }

    /** Pages of the pool. */
    public int pages() {
        return refCounts.length;
    }

    /** Rows of the pool: {@code pages x pageSize}. */
    public int rows() {
        return refCounts.length * pageSize;
    }

    /** Idle pages ({@link #alloc} takes one without failing as long as this is positive). */
    public int free() {
        return freeCount;
    }

    /**
     * Takes a page from the free list, most-recently-freed first (LIFO), with its reference held by the caller.
     *
     * @return the page id, or -1 when the pool is empty
     */
    public int alloc() {
        if (freeCount == 0) {
            return -1;
        }
        int p = freeStack[--freeCount];
        refCounts[p] = 1;
        return p;
    }

    /**
     * The first {@code n} page ids {@link #alloc} would hand out, in {@code alloc} order. The reservation planner of
     * {@link PagedSequences#prepare} uses it to plan an all-or-nothing step without mutating the pool: the
     * caller must {@link #alloc} in the same order before anything else touches the pool, which the single engine
     * thread guarantees.
     *
     * @throws IllegalArgumentException for a negative {@code n} or one larger than {@link #free()}
     */
    public int[] peek(int n) {
        if (n < 0 || n > freeCount) {
            throw new IllegalArgumentException("peek of " + n + " pages over " + freeCount + " free");
        }
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = freeStack[freeCount - 1 - i];
        }
        return out;
    }

    /**
     * Adds a reference to {@code page}. Only for a page a holder already owns: {@code refCount(page)} must be over 0.
     *
     * @throws IndexOutOfBoundsException when {@code page} is not in {@code [0, pages)}
     * @throws IllegalStateException     when {@code page} is free (it must be taken by {@link #alloc})
     */
    public void ref(int page) {
        checkRange(page);
        if (refCounts[page] == 0) {
            throw new IllegalStateException("page " + page + " is free; use alloc to take a page from the pool");
        }
        refCounts[page]++;
    }

    /**
     * Drops one reference of {@code page}. The page returns to the free list when the count reaches 0; the caller
     * must not touch its rows afterwards.
     *
     * @throws IndexOutOfBoundsException when {@code page} is not in {@code [0, pages)}
     * @throws IllegalStateException     when the page has no reference to drop
     */
    public void unref(int page) {
        checkRange(page);
        if (refCounts[page] == 0) {
            throw new IllegalStateException("page " + page + " is not held; unref would drop below 0");
        }
        if (--refCounts[page] == 0) {
            freeStack[freeCount++] = page;
        }
    }

    /**
     * Current references of {@code page}.
     *
     * @throws IndexOutOfBoundsException when {@code page} is not in {@code [0, pages)}
     */
    public int refCount(int page) {
        checkRange(page);
        return refCounts[page];
    }

    /**
     * True when {@code page} is in the pool's free list (reference count 0) and only waiting for reuse.
     *
     * @throws IndexOutOfBoundsException when {@code page} is not in {@code [0, pages)}
     */
    public boolean isFree(int page) {
        checkRange(page);
        return refCounts[page] == 0;
    }

    private void checkRange(int page) {
        if (page < 0 || page >= refCounts.length) {
            throw new IndexOutOfBoundsException("page " + page + " outside the pool of " + refCounts.length);
        }
    }

    @Override
    public String toString() {
        return "PagePool{" + freeCount + "/" + refCounts.length + " free, pageSize=" + pageSize + "}";
    }
}
