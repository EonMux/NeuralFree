package com.eonmux.neuralfree.gguf.memory.paged;

import java.util.Arrays;

/**
 * The pages a sequence held in one sliding-window (trimming) KV group when it had stored {@link #rows()} rows,
 * front first: logical pages {@code [firstPage, firstPage + size())} of the sequence, which cover every row the
 * window still reads (the earlier pages were trimmed). A holder references each page
 * ({@link PagedSequences#holdWindow}) until it releases the window, so the rows outlive the sequence and a later
 * sequence adopts them as its window ({@link PagedSequences#adoptPages(int, int[][], int, int[], int[],
 * PageWindow[], int)}): the SWA half of a context checkpoint (llama.cpp's SWA checkpoints, here shared pages instead of
 * a copy). Immutable.
 */
public final class PageWindow {

    private final int group;
    private final int firstPage;
    private final int[] pages;
    private final int rows;

    PageWindow(int group, int firstPage, int[] pages, int rows) {
        this.group = group;
        this.firstPage = firstPage;
        this.pages = pages.clone();
        this.rows = rows;
    }

    /** The KV group of the pages. */
    public int group() {
        return group;
    }

    /** The logical page index of the first page (the pages the sequence had trimmed before it). */
    public int firstPage() {
        return firstPage;
    }

    /** Pages the window holds. */
    public int size() {
        return pages.length;
    }

    /** The physical id of page {@code i} of the window. */
    public int page(int i) {
        return pages[java.util.Objects.checkIndex(i, pages.length)];
    }

    /** The physical page ids, front first (a copy). */
    public int[] pages() {
        return pages.clone();
    }

    /** The rows the sequence had stored when the window was taken: the window ends there. */
    public int rows() {
        return rows;
    }

    @Override
    public String toString() {
        return "PageWindow{group=" + group + ", firstPage=" + firstPage + ", pages=" + Arrays.toString(pages)
                + ", rows=" + rows + '}';
    }
}
