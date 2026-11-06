package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.gpu.head.GpuRowTopK;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The output rows one forward call reads back, in request order: the logits, hidden rows or greedy ids the
 * {@link GpuLogitsHead} prepared for the requested rows of every step, each read into a host array on the call's
 * queue and assembled by {@link #result} once the queue finished. Shared by {@link GpuPrefillDriver} and the split
 * forward backend.
 */
final class GpuCallReads {

    private final GpuLogitsHead head;
    private final ComputeQueue queue;
    private final GpuPrefillDriver.Output output;
    private final int topK;
    /** Per output row pre-top-k edits the device applies before the top-k readout; null for none. */
    private final LogitPrefilter prefilter;
    private final float[][] rows;
    private final int[] ids;
    private final int[] topIds;
    private final float[] topLogits;
    private final List<float[]> floatParts = new ArrayList<>();
    private final List<byte[]> idParts = new ArrayList<>();
    private final List<int[]> targets = new ArrayList<>();
    /** Candidate slots per row of each read part: the bound readout's k, at least the requested one. */
    private final List<Integer> candidateStrides = new ArrayList<>();
    private final List<GpuEvent> events = new ArrayList<>();

    GpuCallReads(GpuLogitsHead head, ComputeQueue queue, int count, GpuPrefillDriver.Output output) {
        this(head, queue, count, output, 0);
    }

    GpuCallReads(GpuLogitsHead head, ComputeQueue queue, int count, GpuPrefillDriver.Output output, int topK) {
        this(head, queue, count, output, topK, null);
    }

    GpuCallReads(GpuLogitsHead head, ComputeQueue queue, int count, GpuPrefillDriver.Output output, int topK,
                 LogitPrefilter prefilter) {
        this.head = Objects.requireNonNull(head, "head");
        this.queue = Objects.requireNonNull(queue, "queue");
        this.output = Objects.requireNonNull(output, "output");
        this.topK = topK;
        this.prefilter = prefilter;
        if (prefilter != null && prefilter.rows() != count) {
            throw new IllegalArgumentException("the prefilter holds " + prefilter.rows() + " rows, the call reads "
                    + count);
        }
        this.rows = output == GpuPrefillDriver.Output.ARGMAX || output == GpuPrefillDriver.Output.TOPK
                ? null : new float[count][];
        this.ids = output == GpuPrefillDriver.Output.ARGMAX ? new int[count] : null;
        this.topIds = output == GpuPrefillDriver.Output.TOPK ? new int[count * topK] : null;
        this.topLogits = output == GpuPrefillDriver.Output.TOPK ? new float[count * topK] : null;
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
            // the readout keeps its bound width, which may exceed the requested candidates: read whole rows
            int stride = top.k();
            float[] vals = new float[count * stride];
            byte[] idx = new byte[count * stride * Integer.BYTES];
            events.add(queue.enqueueReadBuffer(top.topVal(), 0, (long) vals.length * Float.BYTES, vals));
            events.add(queue.enqueueReadBuffer(top.topIdx(), 0, idx.length, idx));
            floatParts.add(vals);
            idParts.add(idx);
            targets.add(target);
            candidateStrides.add(stride);
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
            events.add(queue.enqueueReadBuffer(src.buffer(), srcOff, bytes, dst));
            idParts.add(dst);
            floatParts.add(null);
        } else {
            float[] dst = new float[(int) (bytes / Float.BYTES)];
            events.add(queue.enqueueReadBuffer(src.buffer(), srcOff, bytes, dst));
            floatParts.add(dst);
            idParts.add(null);
        }
        targets.add(target);
    }

    /** Assembles the rows of every read into the result, in request order. */
    GpuPrefillDriver.Result result() {
        if (output == GpuPrefillDriver.Output.TOPK) {
            for (int part = 0; part < targets.size(); part++) {
                int[] target = targets.get(part);
                float[] vals = floatParts.get(part);
                int stride = candidateStrides.get(part);
                ByteBuffer bb = ByteBuffer.wrap(idParts.get(part)).order(ByteOrder.LITTLE_ENDIAN);
                for (int j = 0; j < target.length; j++) {
                    int row = target[j];
                    System.arraycopy(vals, j * stride, topLogits, row * topK, topK);
                    for (int c = 0; c < topK; c++) {
                        topIds[row * topK + c] = bb.getInt((j * stride + c) * Integer.BYTES);
                    }
                }
            }
            return new GpuPrefillDriver.Result(null, null, topK, topIds, topLogits);
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
            float[] data = floatParts.get(part);
            if (target.length == 1) {
                rows[target[0]] = data;
                continue;
            }
            int width = width(output);
            for (int j = 0; j < target.length; j++) {
                rows[target[j]] = Arrays.copyOfRange(data, j * width, (j + 1) * width);
            }
        }
        return new GpuPrefillDriver.Result(rows, ids);
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
