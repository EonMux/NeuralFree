package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.gpu.head.GpuRowTopK;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The output rows one forward call reads back, in request order: the logits, hidden rows or greedy ids the
 * {@link GpuLogitsHead} prepared for the requested rows of every step, each read into a host array on the call's
 * queue and assembled by {@link #result} once the queue finished. A top-k call may also flag rows to read whole
 * (the split readout, {@code ForwardBackend.forwardBatchSplit}): those rows' logits come back beside every row's
 * candidates, from the same head pass. Shared by {@link GpuPrefillDriver} and the split forward backend.
 */
final class GpuCallReads {

    private final GpuLogitsHead head;
    private final ComputeQueue queue;
    private final GpuPrefillDriver.Output output;
    private final int topK;
    /** Per output row pre-top-k edits the device applies before the top-k readout; null for none. */
    private final LogitPrefilter prefilter;
    /** Per output row of a top-k call, whether its whole logits come back too; null for none. */
    private final boolean[] fullRows;
    private final float[][] rows;
    private final int[] ids;
    private final int[] topIds;
    private final float[] topLogits;
    private final List<float[]> floatParts = new ArrayList<>();
    private final List<byte[]> idParts = new ArrayList<>();
    /** Per top-k read part its rows' softmax denominators, when the call reads them; empty otherwise. */
    private final List<float[]> sumParts = new ArrayList<>();
    /** Whether a top-k call also reads each row's softmax denominator ({@link #readSoftmaxDenominators}). */
    private boolean denominators;
    private final List<int[]> targets = new ArrayList<>();
    private final List<GpuEvent> events = new ArrayList<>();
    /** The device slot the last greedy-id read came from, which a chained decode step reads its id from. */
    private ScratchSlot idSlot;

    GpuCallReads(GpuLogitsHead head, ComputeQueue queue, int count, GpuPrefillDriver.Output output) {
        this(head, queue, count, output, 0);
    }

    GpuCallReads(GpuLogitsHead head, ComputeQueue queue, int count, GpuPrefillDriver.Output output, int topK) {
        this(head, queue, count, output, topK, null);
    }

    GpuCallReads(GpuLogitsHead head, ComputeQueue queue, int count, GpuPrefillDriver.Output output, int topK,
                 LogitPrefilter prefilter) {
        this(head, queue, count, output, topK, prefilter, null);
    }

    /**
     * A call that reads {@code count} output rows; a top-k call reads the whole logits of the rows {@code fullRows}
     * flags beside every row's candidates (null for none).
     */
    GpuCallReads(GpuLogitsHead head, ComputeQueue queue, int count, GpuPrefillDriver.Output output, int topK,
                 LogitPrefilter prefilter, boolean[] fullRows) {
        this.head = Objects.requireNonNull(head, "head");
        this.queue = Objects.requireNonNull(queue, "queue");
        this.output = Objects.requireNonNull(output, "output");
        this.topK = topK;
        this.prefilter = prefilter;
        if (prefilter != null && prefilter.rows() != count) {
            throw new IllegalArgumentException("the prefilter holds " + prefilter.rows() + " rows, the call reads "
                    + count);
        }
        if (fullRows != null && (output != GpuPrefillDriver.Output.TOPK || fullRows.length != count)) {
            throw new IllegalArgumentException("whole-row flags belong to a top-k call, one per output row");
        }
        this.fullRows = fullRows;
        this.rows = output == GpuPrefillDriver.Output.ARGMAX
                || (output == GpuPrefillDriver.Output.TOPK && fullRows == null) ? null : new float[count][];
        this.ids = output == GpuPrefillDriver.Output.ARGMAX ? new int[count] : null;
        this.topIds = output == GpuPrefillDriver.Output.TOPK ? new int[count * topK] : null;
        this.topLogits = output == GpuPrefillDriver.Output.TOPK ? new float[count * topK] : null;
    }

    /**
     * Makes this top-k call read each row's softmax denominator beside its candidates
     * ({@link GpuPrefillDriver.Result#topSumExp()}); call it before the first {@link #read}.
     *
     * @throws IllegalStateException when the call is not a top-k call or has read already
     */
    void readSoftmaxDenominators() {
        if (output != GpuPrefillDriver.Output.TOPK || !targets.isEmpty()) {
            throw new IllegalStateException("softmax denominators belong to a top-k call, asked before its reads");
        }
        denominators = true;
    }

    /**
     * Enqueues the read of {@code count} consecutive rows the head prepared for {@code out}, placing them at the
     * {@code target} positions of the result.
     */
    void read(GpuPrefillDriver.Output out, ScratchSlot hidden, int count, int[] target) {
        ScratchSlot src;
        GpuRowTopK top = null;
        if (out == GpuPrefillDriver.Output.HIDDEN) {
            src = hidden;
        } else if (out == GpuPrefillDriver.Output.ARGMAX && head.certifiesRows(count)) {
            // the certified lane decides the ids on the device; no logits row is materialized
            src = head.argmaxCertified(count, hidden);
        } else {
            ScratchSlot logits = head.logits(count, hidden);
            if (out == GpuPrefillDriver.Output.ARGMAX) {
                src = head.argmax(count);
            } else if (out == GpuPrefillDriver.Output.TOPK) {
                if (prefilter != null) {
                    head.prefilter(prefilter.select(target));
                }
                top = head.topK(count, topK);
                src = logits;
            } else {
                src = logits;
            }
        }
        if (out == GpuPrefillDriver.Output.TOPK) {
            float[] vals = new float[count * topK];
            byte[] idx = new byte[count * topK * Integer.BYTES];
            events.add(queue.enqueueReadBuffer(top.topVal(), 0, (long) vals.length * Float.BYTES, vals));
            events.add(queue.enqueueReadBuffer(top.topIdx(), 0, idx.length, idx));
            floatParts.add(vals);
            idParts.add(idx);
            targets.add(target);
            if (denominators) {
                float[] sums = new float[count];
                events.add(queue.enqueueReadBuffer(top.sumExp(), 0, (long) count * Float.BYTES, sums));
                sumParts.add(sums);
            }
            if (fullRows != null) {
                readWhole(src, count, target);
            }
            return;
        }
        int width = width(out);
        long bytes = (long) count * width * Float.BYTES;
        if (bytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("read of " + count + " rows of " + width + " floats exceeds 2 GiB");
        }
        long srcOff = (long) src.offset() * Float.BYTES;
        if (out == GpuPrefillDriver.Output.ARGMAX) {
            byte[] dst = new byte[(int) bytes];
            idSlot = src;
            events.add(queue.enqueueReadBuffer(src.buffer(), srcOff, bytes, dst));
            idParts.add(dst);
            floatParts.add(null);
        } else {
            // every row lands in its own result array straight from the readback: no whole-block array and no
            // per-row copy out of it
            long rowBytes = (long) width * Float.BYTES;
            for (int j = 0; j < count; j++) {
                float[] dst = new float[width];
                events.add(queue.enqueueReadBuffer(src.buffer(), srcOff + j * rowBytes, rowBytes, dst));
                floatParts.add(dst);
                idParts.add(null);
                targets.add(new int[]{target[j]});
            }
            return;
        }
        targets.add(target);
    }

    /** Enqueues the whole logits of the flagged rows among {@code count} rows of {@code logits}. */
    private void readWhole(ScratchSlot logits, int count, int[] target) {
        long rowBytes = (long) head.vocab() * Float.BYTES;
        long base = (long) logits.offset() * Float.BYTES;
        for (int j = 0; j < count; j++) {
            if (fullRows[target[j]]) {
                float[] dst = new float[head.vocab()];
                events.add(queue.enqueueReadBuffer(logits.buffer(), base + j * rowBytes, rowBytes, dst));
                rows[target[j]] = dst;
            }
        }
    }

    /** Assembles the rows of every read into the result, in request order. */
    GpuPrefillDriver.Result result() {
        if (output == GpuPrefillDriver.Output.TOPK) {
            float[] sumExp = denominators ? new float[topIds.length / topK] : null;
            for (int part = 0; part < targets.size(); part++) {
                int[] target = targets.get(part);
                float[] vals = floatParts.get(part);
                ByteBuffer bb = ByteBuffer.wrap(idParts.get(part)).order(ByteOrder.LITTLE_ENDIAN);
                for (int j = 0; j < target.length; j++) {
                    int row = target[j];
                    System.arraycopy(vals, j * topK, topLogits, row * topK, topK);
                    for (int c = 0; c < topK; c++) {
                        topIds[row * topK + c] = bb.getInt((j * topK + c) * Integer.BYTES);
                    }
                    if (sumExp != null) {
                        sumExp[row] = sumParts.get(part)[j];
                    }
                }
            }
            return new GpuPrefillDriver.Result(rows, null, topK, topIds, topLogits, sumExp);
        }
        for (int part = 0; part < targets.size(); part++) {
            int[] target = targets.get(part);
            if (output == GpuPrefillDriver.Output.ARGMAX) {
                ByteBuffer bb = ByteBuffer.wrap(idParts.get(part)).order(ByteOrder.LITTLE_ENDIAN);
                for (int j = 0; j < target.length; j++) {
                    ids[target[j]] = bb.getInt(j * Integer.BYTES);
                }
                continue;
            }
            rows[target[0]] = floatParts.get(part);   /* one row per float read */
        }
        return new GpuPrefillDriver.Result(rows, ids);
    }

    /** The device slot of the greedy ids the last {@link GpuPrefillDriver.Output#ARGMAX} read reads; null before. */
    ScratchSlot idSlot() {
        return idSlot;
    }

    /**
     * Waits until every read of this call holds its data: on the last read's event, which the queue's order puts
     * after every earlier read, or on the whole queue when the backend returned no event for it.
     *
     * @throws IllegalStateException when the wait is interrupted (the interrupt status is kept)
     */
    void await() {
        GpuEvent last = events.isEmpty() ? null : events.get(events.size() - 1);
        if (last == null) {
            queue.finish();
            return;
        }
        try {
            last.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting for a forward call's readback", e);
        }
    }

    void closeEvents() {
        for (GpuEvent e : events) {
            if (e != null) {
                e.close();
            }
        }
        events.clear();
    }

    private int width(GpuPrefillDriver.Output out) {
        switch (out) {
            case HIDDEN:
                return head.hiddenWidth();
            case LOGITS:
                return head.vocab();
            default:
                return 1;
        }
    }
}
