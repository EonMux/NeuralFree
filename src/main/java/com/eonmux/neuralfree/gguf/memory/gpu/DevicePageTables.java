package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;

import java.util.Arrays;

/**
 * One KV group's device page table in launch order (segment {@code s} of the current step at table row {@code s},
 * {@code maxPages} entries per row, -1 past a sequence's pages) and a host shadow of exactly what the device holds.
 * {@link #upload} writes, per launch row, only the entry range where the step's table differs from the shadow: a row
 * whose occupant changed, whose sequence crossed into a new page, privatized a page or trimmed its front pages. The
 * device contents after every upload equal a whole write of every launch row, so the kernels' reads, the rows past
 * the step's segments included, see what they saw before; a steady decode step writes nothing. Not thread-safe.
 */
final class DevicePageTables {

    /** The device writes of int ranges, staged through the owning step inputs. */
    interface IntWrites {
        void write(GpuBuffer dst, long elementOffset, int[] src, int srcFrom, int n);
    }

    private final GpuBuffer table;
    private final int maxSequences;
    private final int maxPages;
    private final int[] shadow;
    /** Per launch row: entries from this index on are -1 in the shadow. */
    private final int[] shadowLen;
    private final int[] row;

    /** A table over {@code table}, whose device contents the caller filled with -1. */
    DevicePageTables(GpuBuffer table, int maxSequences, int maxPages) {
        this.table = table;
        this.maxSequences = maxSequences;
        this.maxPages = maxPages;
        this.shadow = new int[Math.multiplyExact(maxSequences, maxPages)];
        Arrays.fill(shadow, -1);
        this.shadowLen = new int[maxSequences];
        this.row = new int[maxPages];
    }

    GpuBuffer buffer() {
        return table;
    }

    /**
     * Brings launch row {@code s} of the device table to segment {@code s}'s table of {@code step} in group
     * {@code group}, for every segment: at most one write per launch row, over the first to the last differing
     * entry.
     *
     * @return the entries written
     */
    int upload(PagedStep step, int group, IntWrites writes) {
        int written = 0;
        for (int s = 0; s < step.nSeg(); s++) {
            int len = step.pageTableRow(group, step.segSeq(s), row);
            int base = s * maxPages;
            int old = shadowLen[s];
            int lo = -1;
            int hi = -1;
            for (int k = 0; k < len; k++) {
                if (shadow[base + k] != row[k]) {
                    if (lo < 0) {
                        lo = k;
                    }
                    hi = k + 1;
                }
            }
            if (old > len) {
                Arrays.fill(row, len, old, -1);
                for (int k = len; k < old; k++) {
                    if (shadow[base + k] != -1) {
                        if (lo < 0) {
                            lo = k;
                        }
                        hi = k + 1;
                    }
                }
            }
            if (lo >= 0) {
                writes.write(table, (long) base + lo, row, lo, hi - lo);
                System.arraycopy(row, lo, shadow, base + lo, hi - lo);
                written += hi - lo;
            }
            shadowLen[s] = trimmedLength(base, Math.max(len, old));
        }
        return written;
    }

    /** The shadow row's length once its trailing -1 entries are dropped, scanning down from {@code len}. */
    private int trimmedLength(int base, int len) {
        int n = len;
        while (n > 0 && shadow[base + n - 1] == -1) {
            n--;
        }
        return n;
    }

    /** The shadow's entry {@code k} of launch row {@code s}: the value the device holds there. */
    int shadowEntry(int s, int k) {
        if (s < 0 || s >= maxSequences || k < 0 || k >= maxPages) {
            throw new IndexOutOfBoundsException("table entry " + s + "/" + k + " outside " + maxSequences + "x"
                    + maxPages);
        }
        return shadow[s * maxPages + k];
    }
}
