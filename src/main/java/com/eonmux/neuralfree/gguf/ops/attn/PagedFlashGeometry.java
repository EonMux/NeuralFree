package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;

/**
 * The page geometry and per-sequence buffers a paged flash attention reads, taken from the step inputs of a paged
 * memory: the KV group's page table and slot counts, the per-sequence and per-segment query positions, the page
 * shift and the table width. {@link #maxLaunchSeq} bounds the largest decode launch ({@code min(parallel, ubatch)}
 * sequences). Immutable.
 */
final class PagedFlashGeometry {

    private final GpuBuffer pageTable;
    private final GpuBuffer seqKvLen;
    private final GpuBuffer qPosSeq;
    private final GpuBuffer segPos;
    private final int pageShift;
    private final int maxPages;
    private final int maxLaunchSeq;

    private PagedFlashGeometry(GpuBuffer pageTable, GpuBuffer seqKvLen, GpuBuffer qPosSeq, GpuBuffer segPos,
                               int pageShift, int maxPages, int maxLaunchSeq) {
        this.pageTable = pageTable;
        this.seqKvLen = seqKvLen;
        this.qPosSeq = qPosSeq;
        this.segPos = segPos;
        this.pageShift = pageShift;
        this.maxPages = maxPages;
        this.maxLaunchSeq = maxLaunchSeq;
    }

    /** The geometry of KV group {@code group} of a paged memory's step inputs; null when the memory is not paged. */
    static PagedFlashGeometry of(GpuStepInputs steps, int group, int ubatch) {
        if (steps.pageShift() < 0) {
            return null;
        }
        return new PagedFlashGeometry(steps.pageTable(group), steps.seqKvLen(group), steps.qPosSeq(),
                steps.segmentPositions(), steps.pageShift(), steps.maxPages(),
                Math.min(steps.pagedMaxSequences(), ubatch));
    }

    /** The launch-ordered page table, {@link #maxPages} entries per launch row. */
    GpuBuffer pageTable() {
        return pageTable;
    }

    /** The slot count of every launch row's sequence. */
    GpuBuffer seqKvLen() {
        return seqKvLen;
    }

    /** The query position of every launch row's sequence. */
    GpuBuffer qPosSeq() {
        return qPosSeq;
    }

    /** The per-row query positions of the segment a paged prefill runs. */
    GpuBuffer segPos() {
        return segPos;
    }

    /** {@code log2} of the page size. */
    int pageShift() {
        return pageShift;
    }

    /** Page-table entries per launch row. */
    int maxPages() {
        return maxPages;
    }

    /** Sequences of the largest decode launch. */
    int maxLaunchSeq() {
        return maxLaunchSeq;
    }

    /** Slots the page tables address ({@code maxPages << pageShift}), the paged plan's key capacity. */
    int slotCapacity() {
        return maxPages << pageShift;
    }
}
