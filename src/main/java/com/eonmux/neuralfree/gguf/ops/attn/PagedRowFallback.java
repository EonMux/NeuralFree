package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * The per-row fallback of a paged segment whose prefill plan has no paged prefill kernel: the segment's
 * page-table block is copied to this fallback's focus table, then the paged decode kernel runs once per row with
 * launch row 0 reading the focus table, the segment's slot count and the row's own query position, so the causal
 * mask admits the keys up to the row. Its three device buffers belong to the mixer's {@link GpuMixerResources}.
 * Not thread-safe.
 */
final class PagedRowFallback {

    private final ComputeQueue queue;
    private final PagedFlashGeometry pg;
    private final GpuBuffer table;
    private final GpuBuffer kvLen;
    private final GpuBuffer qPos;
    private final int qDim;
    private final int outDim;

    private PagedRowFallback(ComputeQueue queue, PagedFlashGeometry pg, GpuBuffer table, GpuBuffer kvLen,
                             GpuBuffer qPos, int qDim, int outDim) {
        this.queue = Objects.requireNonNull(queue, "queue");
        this.pg = Objects.requireNonNull(pg, "pg");
        this.table = table;
        this.kvLen = kvLen;
        this.qPos = qPos;
        this.qDim = qDim;
        this.outDim = outDim;
    }

    /** Allocates the focus table ({@code maxPages} ints), slot count and query position of launch row 0. */
    static PagedRowFallback allocate(GpuMixerResources res, ComputeQueue queue, PagedFlashGeometry pg, int qDim,
                                     int outDim) {
        GpuBuffer table = res.allocate((long) pg.maxPages() * Integer.BYTES);
        GpuBuffer kvLen = res.allocate(Integer.BYTES);
        GpuBuffer qPos = res.allocate(Integer.BYTES);
        return new PagedRowFallback(queue, pg, table, kvLen, qPos, qDim, outDim);
    }

    /**
     * Runs segment {@code seg} of {@code step} over its {@code keys} slots row by row through {@code paged}'s decode
     * launch, then points the launch rows back at the step's page table.
     */
    void run(GpuPagedFlashAttention paged, PagedStep step, int seg, int keys) {
        long rowBytes = (long) pg.maxPages() * Integer.BYTES;
        queue.enqueueCopyBuffer(pg.pageTable(), seg * rowBytes, table, 0, rowBytes);
        writeInt(kvLen, keys);
        paged.bindLaunchRows(table, kvLen, qPos, 0, 0);
        int tokStart = step.segStart(seg);
        for (int t = 0; t < step.segLen(seg); t++) {
            int row = tokStart + t;
            writeInt(qPos, step.segPos0(seg) + t);
            paged.launchDecodeAt(1, keys, row * qDim, row * outDim, step.segPos0(seg) + t);
        }
        paged.bindLaunchRows(pg.pageTable(), pg.seqKvLen(), pg.qPosSeq(), qDim, outDim);
    }

    /** Writes one little-endian int to element 0 of {@code dst}, ordered on the queue before later launches. */
    private void writeInt(GpuBuffer dst, int value) {
        byte[] b = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array();
        GpuEvent e = queue.enqueueWriteBuffer(dst, 0, Integer.BYTES, b);
        if (e != null) {
            e.close();
        }
    }
}
