package com.eonmux.neuralfree.gguf.memory.paged;

/**
 * The checks {@link PagedSequences} runs before it changes anything: the explicit positions of a step's segment
 * ({@link PagedSequences#prepare(StepLayout, int[])}), and per KV group the pages an adoption
 * ({@link PagedSequences#adoptPages(int, int[][], int, int[], int[], PageWindow[], int)}) gives a sequence — held, and
 * holding the positions of the rows they give it. Stateless.
 */
final class PageChecks {

    private PageChecks() {
    }

    /**
     * Checks full-attention group {@code g}'s page of every adopted block: held, and holding the positions of the
     * block's rows (every row of a whole block, {@code lastRows} of the last).
     *
     * @throws IllegalStateException when a page is missing or free, or a row holds another position
     */
    static void checkBlockPages(PagePool pool, int[] kvPos, int g, int[][] blocks, int lastRows,
                                int[] positions) {
        int pageSize = pool.pageSize();
        for (int k = 0; k < blocks.length; k++) {
            int page = blocks[k][g];
            if (page < 0 || pool.isFree(page)) {
                throw new IllegalStateException("page " + page + " of group " + g + " is free");
            }
            int rows = k == blocks.length - 1 ? lastRows : pageSize;
            for (int r = 0; r < rows; r++) {
                int row = k * pageSize + r;
                int want = positions == null ? row : positions[row];
                if (kvPos[page * pageSize + r] != want) {
                    throw new IllegalStateException("page " + page + " of group " + g + " row " + r
                            + " holds position " + kvPos[page * pageSize + r] + ", block " + k + " needs " + want);
                }
            }
        }
    }

    /**
     * Checks trimming group {@code g}'s adopted window: taken at {@code total} rows over this group and ending on
     * the adopted rows' last page, no block giving the group a page, every window page held, the rows the window
     * reads (the last {@code width} adopted rows) live at their positions, and the window's other rows at their
     * position or empty.
     *
     * @throws IllegalArgumentException when the window does not fit the adopted rows or a block gives the group a
     *                                  page
     * @throws IllegalStateException    when a window page is free or a row holds another position
     */
    static void checkWindow(PagePool pool, int[] kvPos, int width, int g, PageWindow w, int[][] blocks, int total,
                            int[] positions) {
        int pageSize = pool.pageSize();
        if (w == null || w.group() != g || w.rows() != total
                || w.firstPage() + w.size() != (total + pageSize - 1) / pageSize) {
            throw new IllegalArgumentException("trimming group " + g + " adopts a window of " + total + " rows,"
                    + " got " + w);
        }
        for (int[] block : blocks) {
            if (block[g] >= 0) {
                throw new IllegalArgumentException("a block gives page " + block[g] + " to trimming group " + g);
            }
        }
        for (int k = 0; k < w.size(); k++) {
            int page = w.page(k);
            if (pool.isFree(page)) {
                throw new IllegalStateException("window page " + page + " of group " + g + " is free");
            }
            for (int r = 0; r < pageSize; r++) {
                int row = (w.firstPage() + k) * pageSize + r;
                int pos = kvPos[page * pageSize + r];
                int want = row >= total ? -1 : positions == null ? row : positions[row];
                boolean read = row < total && row >= total - width;
                if (pos != want && (read || pos != -1)) {
                    throw new IllegalStateException("window page " + page + " of group " + g + " row " + r
                            + " holds position " + pos + ", row " + row + " needs " + want);
                }
            }
        }
    }

    /**
     * Checks the explicit positions of one segment: non-negative, non-decreasing, and starting at or above the
     * sequence's highest stored position {@code posMax} (-1 when it stores nothing; a media segment continues at the
     * shared base). Returns the segment's highest value.
     */
    static int checkRowPos(int start, int len, int[] rowPos, int seq, int posMax) {
        if (len > 0 && rowPos[start] < 0) {
            throw new IllegalArgumentException("row position " + rowPos[start] + " of sequence " + seq + " is"
                    + " negative");
        }
        if (len > 0 && posMax >= 0 && rowPos[start] < posMax) {
            throw new IllegalStateException("the rows of sequence " + seq + " start at position " + rowPos[start]
                    + ", below its highest stored position " + posMax);
        }
        int prev = -1;
        int max = -1;
        for (int t = 0; t < len; t++) {
            int p = rowPos[start + t];
            if (p < 0) {
                throw new IllegalArgumentException("row position " + p + " of sequence " + seq + " is negative");
            }
            if (p < prev) {
                throw new IllegalArgumentException("row positions of sequence " + seq + " descend: " + p + " after "
                        + prev);
            }
            prev = p;
            max = p;
        }
        return max;
    }
}
