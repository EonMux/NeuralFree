package com.eonmux.neuralfree.gguf.memory.paged;

/**
 * Page list of one sequence in one KV group: the sequence's pages in append order. Logical slot {@code j} of the
 * sequence maps to physical row {@code page(j >> log2P) * pageSize + (j & (pageSize - 1))}, so
 * prefill and decode appends simply extend the list; trimming and context removals drop whole pages from the front,
 * leaving at most two partially holed pages per sequence.
 *
 * <p>Slots covered by the list run from {@code 0} to {@count size() * pageSize}; a slot beyond that is not reserved
 * yet. The list keeps holed (partially empty) interior pages as tombstones so the slot-to-row mapping stays stable.
 * Mutable, owned and mutated only by {@link PagedSequences} on its engine thread.
 */
public final class PageList {

    private final int pageSize;
    private int[] pages;
    private int size;

    /** An empty list for the page size {@code pageSize} (a power of two in {@code [1, 256]}). */
    public PageList(int pageSize) {
        if (pageSize < 1 || pageSize > com.eonmux.neuralfree.gguf.memory.MemoryConfig.MAX_PAGE_SIZE
                || Integer.bitCount(pageSize) != 1) {
            throw new IllegalArgumentException("page list page size must be a power of two in [1, "
                    + com.eonmux.neuralfree.gguf.memory.MemoryConfig.MAX_PAGE_SIZE + "], got " + pageSize);
        }
        this.pageSize = pageSize;
        this.pages = new int[4];
    }

    public int pageSize() {
        return pageSize;
    }

    /** Pages in the list ({@code slots + pageSize - 1} over {@code pageSize} for a fully packed list). */
    public int size() {
        return size;
    }

    /** True when the list holds no pages. */
    public boolean isEmpty() {
        return size == 0;
    }

    /** Pages the list holds for the {@code slotCount} slots of the sequence. */
    public int slotPages(int slotCount) {
        if (slotCount < 0) {
            throw new IllegalArgumentException("slot count " + slotCount + " is negative");
        }
        return (slotCount + (long) pageSize - 1) > Integer.MAX_VALUE
                ? -1
                : (slotCount + pageSize - 1) / pageSize;
    }

    /**
     * Page holding the entries of slot {@code slot}.
     *
     * @throws IndexOutOfBoundsException when {@code slot} does not map to a page of the list
     */
    public int pageOfSlot(int slot) {
        checkSlot(slot);
        return pages[slot >> log2()];
    }

    /**
     * Physical row of slot {@code slot}: the page of the slot's page index times the page size, plus the offset
     * inside the page.
     *
     * @throws IndexOutOfBoundsException when {@code slot} does not map to a page of the list
     */
    public int rowOfSlot(int slot) {
        checkSlot(slot);
        return pages[slot >> log2()] * pageSize + (slot & (pageSize - 1));
    }

    /** Page at list index {@code index} ({@code [0, size)}). */
    public int page(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("page list index " + index + " outside " + size);
        }
        return pages[index];
    }

    /** Ids of the pages, in list order; a fresh array. */
    public int[] toArray() {
        return java.util.Arrays.copyOf(pages, size);
    }

    /** Appends {@code page} at the end of the list. */
    public void append(int page) {
        if (page < 0) {
            throw new IllegalArgumentException("page " + page + " is negative");
        }
        if (size == pages.length) {
            pages = java.util.Arrays.copyOf(pages, Math.max(2 * size, 8));
        }
        pages[size++] = page;
    }

    /** Replaces the page at list index {@code index} and returns the old one. */
    public int set(int index, int page) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("page list index " + index + " outside " + size);
        }
        if (page < 0) {
            throw new IllegalArgumentException("page " + page + " is negative");
        }
        int old = pages[index];
        pages[index] = page;
        return old;
    }

    /** Removes the first {@code count} pages (whole trimmed pages), shifting the rest down. */
    public void removeFront(int count) {
        if (count < 0 || count > size) {
            throw new IllegalArgumentException("cannot remove " + count + " of " + size + " pages from the front");
        }
        System.arraycopy(pages, count, pages, 0, size - count);
        size -= count;
    }

    /** Removes the last {@code count} pages (whole trailing pages emptied of their rows). */
    public void removeBack(int count) {
        if (count < 0 || count > size) {
            throw new IllegalArgumentException("cannot remove " + count + " of " + size + " pages from the back");
        }
        size -= count;
    }

    /** Empties the list. */
    public void clear() {
        size = 0;
    }

    @Override
    public String toString() {
        return "PageList{" + size + " pages of " + pageSize + "}";
    }

    private int log2() {
        return Integer.numberOfTrailingZeros(pageSize);
    }

    private void checkSlot(int slot) {
        if (slot < 0) {
            throw new IllegalArgumentException("slot " + slot + " is negative");
        }
        if (slot >= (long) size * pageSize) {
            throw new IndexOutOfBoundsException("slot " + slot + " outside the list of " + size + " pages");
        }
    }
}
