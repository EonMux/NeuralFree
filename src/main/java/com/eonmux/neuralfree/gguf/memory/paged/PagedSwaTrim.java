package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;

import java.util.Objects;

/**
 * The sliding-window trim of a {@link PagedSequences}: whole pages whose rows all lie below a sequence's window
 * leave the sequence, and the leading page's rows below the window go empty when the sequence alone holds the
 * page. A page another holder shares (a context checkpoint's window, a fork) is never written: a whole one leaves
 * the sequence by dropping its reference, a straddling one keeps its rows below the window, which the SWA mask
 * hides from every query of the sequence. So a trim never allocates a page.
 *
 * <p>Each group's trim kind is decided from the plan ({@link #trims}): the widest window over every layer reading
 * the group, with differing window types or any window-less reader making it a full-attention group. Only windowed
 * (same-type) groups trim. Not thread-safe; used by the thread that drives the owning memory.
 */
final class PagedSwaTrim {

    private final PagedSequences owner;
    /** [group]: groups whose sequences trim (the widest reader window). */
    private final boolean[] swaGroup;
    /** [group]: W_g of a trimming group. */
    private final int[] windowWidth;

    PagedSwaTrim(PagedSequences owner, int groups) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.swaGroup = new boolean[groups];
        this.windowWidth = new int[groups];
        for (int g = 0; g < groups; g++) {
            describeGroupWindow(g);
        }
    }

    /**
     * The group's trim kind: the widest window over every layer reading the group, with differing
     * window types or any window-less reader making it a full-attention group. Only windowed (same-type) groups trim.
     */
    private void describeGroupWindow(int g) {
        swaGroup[g] = false;
        windowWidth[g] = 0;
        SwaType type = null;
        boolean sameWindowType = true;
        int width = 0;
        for (LayerMemoryPlan l : owner.plan().layers()) {
            if (l.kvGroup() != g || l.kv() == null) {
                continue;
            }
            KvWindow w = l.kv().window();
            if (w.type() == SwaType.NONE) {
                sameWindowType = false;
                break;
            }
            if (type == null) {
                type = w.type();
            } else if (type != w.type()) {
                sameWindowType = false;
                break;
            }
            width = Math.max(width, w.nSwa());
        }
        if (sameWindowType && type != null) {
            swaGroup[g] = true;
            windowWidth[g] = width;
        }
    }

    boolean[] swaGroups() {
        return swaGroup;
    }

    int[] windowWidths() {
        return windowWidth;
    }

    /**
     * Trims the SWA pages of {@code seq} to the window of its next position: the rows below
     * {@code committed - W_g} leave, committed being the sequence's committed element count.
     */
    void trim(int seq) {
        trimSequence(seq, Integer.MAX_VALUE);
    }

    /**
     * Ends a committed step once the trunk ran over it: its queries read no more, so each of its sequences trims
     * to the window of its next position ({@link #trim(int)}) and the window rows below it leave before the next
     * step reserves pages.
     */
    void finish(PagedStep step) {
        Objects.requireNonNull(step, "step");
        for (int s = 0; s < step.nSeg(); s++) {
            trim(step.segSeq(s));
        }
    }

    /**
     * Trims {@code seq}: the rows below {@code min(committed, reader) - W_g} leave.
     *
     * @param reader one past the earliest position that still reads the rows (the first row of a step the owner's
     *               {@code commit} applies), {@code Integer.MAX_VALUE} for the sequence's next position
     */
    void trimSequence(int seq, int reader) {
        int max = owner.posMaxArr()[seq];
        if (max < 0) {
            return;
        }
        int committed = Math.min(max + 1, reader);
        int groups = swaGroup.length;
        for (int g = 0; g < groups; g++) {
            if (swaGroup[g]) {
                trimGroup(g, seq, committed - windowWidth[g]);
            }
        }
    }

    private void trimGroup(int g, int seq, int threshold) {
        if (threshold <= 0) {
            return;
        }
        int pageSize = owner.pageSize();
        PageList[] seqLists = owner.lists()[seq];
        int[][] kvPos = owner.kvPos();
        PageList list = seqLists[g];
        while (!list.isEmpty()) {
            int page = list.page(0);
            int base = page * pageSize;
            int lastLive = -1;
            boolean hasBelow = false;
            for (int r = pageSize - 1; r >= 0; r--) {
                int pos = kvPos[g][base + r];
                if (pos >= 0) {
                    if (lastLive < 0) {
                        lastLive = r;
                    }
                    if (pos < threshold) {
                        hasBelow = true;
                    }
                }
            }
            boolean shared = owner.pools()[g].refCount(page) > 1;
            if (lastLive >= 0 && kvPos[g][base + lastLive] >= threshold) {
                // straddling page: the rows below the window go empty on a page the sequence alone holds
                if (hasBelow && !shared) {
                    owner.clearRowsIn(g, base, 0, threshold);
                }
                return;
            }
            // the whole page lies below the window (or has no live row): it leaves the sequence; a shared page
            // keeps its rows for the other holders
            if (!shared) {
                owner.clearRowsIn(g, base, Integer.MIN_VALUE, Integer.MAX_VALUE);
            }
            owner.sharing().unref(false, g, page);
            list.removeFront(1);
            owner.slots()[g][seq] = Math.max(0, owner.slots()[g][seq] - pageSize);
        }
    }
}
