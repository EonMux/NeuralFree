package com.eonmux.neuralfree.gguf.memory.paged;

/**
 * One device-side page copy of the copy-on-write protocol (the privatization rule, PARITY invariant 7 and T-5):
 * copy the source page's rows {@code [srcRow0, srcRow0 + rows)} of one KV group into the destination page's rows
 * {@code [srcRow0, srcRow0 + rows)}, and leave every destination row beyond that in the empty state ({@code kv_pos}
 * -1). A page with a reference count over 1 is never written in place; every mutation of a row some other holder can
 * still read first replaces the page in the mutating sequence's list by a private copy and emits one of these.
 *
 * <p>Immutable. From its emission until a non-failed completion of the submitting step has executed it, the copy
 * holds the reference its sequence held on the source page (T-4: no source page is recycled before its copy ran);
 * {@link PagedSequences} moves that reference at emission and releases it through {@link PagedSequences#releaseCopies}.
 */
public final class PageCopy {

    private final int group;
    private final int srcPage;
    private final int dstPage;
    private final int srcRow0;
    private final int rows;

    private PageCopy(int group, int srcPage, int dstPage, int srcRow0, int rows) {
        this.group = group;
        this.srcPage = srcPage;
        this.dstPage = dstPage;
        this.srcRow0 = srcRow0;
        this.rows = rows;
    }

    /** A copy of the source page's first {@code rows} rows ({@code srcRow0} 0, the T-5 shape). */
    public static PageCopy of(int group, int srcPage, int dstPage, int rows) {
        return ofRange(group, srcPage, dstPage, 0, rows);
    }

    /**
     * A copy of source rows {@code [srcRow0, srcRow0 + rows)} into the same rows of the destination, so a partial
     * page copy can also leave its destination's head rows empty or reuse them later.
     *
     * @throws IllegalArgumentException for a non-positive {@code rows} or a negative row range
     */
    public static PageCopy ofRange(int group, int srcPage, int dstPage, int srcRow0, int rows) {
        if (rows <= 0) {
            throw new IllegalArgumentException("page copy needs rows > 0, got " + rows);
        }
        if (srcRow0 < 0) {
            throw new IllegalArgumentException("page copy source row " + srcRow0 + " is negative");
        }
        if (srcPage < 0 || dstPage < 0) {
            throw new IllegalArgumentException("page copy needs non-negative page ids, got src " + srcPage + ", dst "
                    + dstPage);
        }
        return new PageCopy(group, srcPage, dstPage, srcRow0, rows);
    }

    /** KV group the copy belongs to. */
    public int group() { return group; }
    /** Source page. */
    public int srcPage() { return srcPage; }
    /** Destination page. */
    public int dstPage() { return dstPage; }
    /** First source row of the copied range. */
    public int srcRow0() { return srcRow0; }
    /** Rows the copy carries; destination rows {@code >= srcRow0 + rows} stay empty ({@code kv_pos} -1). */
    public int rows() { return rows; }

    @Override
    public String toString() {
        return "PageCopy{group=" + group + ", " + srcPage + " -> " + dstPage + ", rows=" + srcRow0 + ".."
                + (srcRow0 + rows) + "}";
    }
}
