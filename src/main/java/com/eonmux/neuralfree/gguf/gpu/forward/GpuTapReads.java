package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Tap-point reads of one forward call, delivered to the {@link HiddenTap} once the queue finished: the
 * {@link GpuLayerExecutor.TapSink} the layer executors feed, with every capture read from the executor's device.
 * {@code tapOffset} maps an executor's stack-local tap point to the model's layer index, so every executor of a
 * layer split feeds the same tap. Shared by {@link GpuPrefillDriver} and the split forward backend. The rows of a
 * contiguous step land at the step's offset ({@link #chunk}); the rows of a batched ubatch, which runs its rows in
 * sequence-position order, land at their call rows ({@link #batchRows}), as the CPU backend stores them.
 *
 * <p>A {@link GpuHiddenFeed} tap takes a tap point's rows on the device: each capture enqueues one device copy
 * instead of a host read, and {@link #deliver} reports the call's device rows to the feed. A point captured on the
 * device keeps that mode for the whole call; the host path never mixes with it.
 */
final class GpuTapReads implements GpuLayerExecutor.TapSink {

    private final ComputeQueue queue;
    private final HiddenTap tap;
    private final GpuHiddenFeed deviceFeed;
    private final int tapOffset;
    private final int total;
    private final List<Integer> layers = new ArrayList<>();
    private final List<Integer> offsets = new ArrayList<>();
    private final List<Integer> counts = new ArrayList<>();
    private final List<Integer> widths = new ArrayList<>();
    private final List<float[]> parts = new ArrayList<>();
    /** Call row of each captured row, or null when the capture's rows run from its offset. */
    private final List<int[]> rowMaps = new ArrayList<>();
    /** Whether each capture went to the device feed; a capture of a device point never holds a host part. */
    private final List<Boolean> device = new ArrayList<>();
    private final List<GpuEvent> events = new ArrayList<>();
    private final Map<Integer, Boolean> devicePoints = new HashMap<>();
    private int chunkStart;
    private int[] callRows;

    GpuTapReads(ComputeQueue queue, HiddenTap tap, int total, int tapOffset) {
        this.queue = Objects.requireNonNull(queue, "queue");
        this.tap = Objects.requireNonNull(tap, "tap");
        this.deviceFeed = tap instanceof GpuHiddenFeed ? (GpuHiddenFeed) tap : null;
        this.total = total;
        this.tapOffset = tapOffset;
    }

    /** Where the rows of the next step land in the delivered arrays: consecutive rows from {@code c0}. */
    void chunk(int c0) {
        this.chunkStart = c0;
        this.callRows = null;
    }

    /** Where the rows of the next step land: row {@code t} of the step at call row {@code rows[t]}. */
    void batchRows(int[] rows) {
        this.chunkStart = 0;
        this.callRows = Objects.requireNonNull(rows, "rows").clone();
    }

    @Override
    public boolean wants(int tapPoint) {
        return tap.wants(tapPoint + tapOffset);
    }

    /** Whether the tap wants any of the stack's tap points {@code [0, points]} (the stack output included). */
    boolean wantsAny(int points) {
        for (int p = 0; p <= points; p++) {
            if (wants(p)) {
                return true;
            }
        }
        return false;
    }

    /** Whether this call's tap point {@code layer} runs on the device feed, decided at its first capture. */
    private boolean devicePoint(int layer) {
        Boolean decided = devicePoints.get(layer);
        if (decided == null) {
            decided = callRows == null && deviceFeed != null && deviceFeed.wantsDevice(layer, queue);
            devicePoints.put(layer, decided);
        }
        return decided;
    }

    @Override
    public void capture(int tapPoint, ScratchSlot x, int nTokens, int width) {
        int layer = tapPoint + tapOffset;
        if (devicePoint(layer)) {
            deviceFeed.captureDevice(layer, x.buffer(), x.offset(), nTokens, width);
            layers.add(layer);
            offsets.add(chunkStart);
            counts.add(nTokens);
            widths.add(width);
            parts.add(null);
            rowMaps.add(callRows);
            device.add(Boolean.TRUE);
            return;
        }
        float[] dst = new float[Math.multiplyExact(nTokens, width)];
        events.add(queue.enqueueReadBuffer(x.buffer(), (long) x.offset() * Float.BYTES,
                (long) dst.length * Float.BYTES, dst));
        layers.add(layer);
        offsets.add(chunkStart);
        counts.add(nTokens);
        widths.add(width);
        parts.add(dst);
        rowMaps.add(callRows);
        device.add(Boolean.FALSE);
    }

    /**
     * A capture already on the host: the first {@code nTokens} rows of {@code width} floats of {@code x}, copied, at
     * model tap point {@code layer} (no tap offset applies).
     */
    void captureHost(int layer, float[] x, int nTokens, int width) {
        layers.add(layer);
        offsets.add(chunkStart);
        counts.add(nTokens);
        widths.add(width);
        parts.add(Arrays.copyOf(x, Math.multiplyExact(nTokens, width)));
        rowMaps.add(callRows);
        device.add(Boolean.FALSE);
    }

    /** One {@link HiddenTap#onHidden} per host tap point with every row of the call, in tap order. */
    void deliver() {
        List<Integer> seen = new ArrayList<>();
        for (int i = 0; i < layers.size(); i++) {
            int layer = layers.get(i);
            if (seen.contains(layer)) {
                continue;
            }
            seen.add(layer);
            if (device.get(i)) {
                deviceFeed.onHiddenDevice(layer, total, widths.get(i));
                continue;
            }
            int width = widths.get(i);
            float[] all = new float[Math.multiplyExact(total, width)];
            for (int j = i; j < layers.size(); j++) {
                if (layers.get(j) != layer) {
                    continue;
                }
                if (device.get(j)) {
                    throw new IllegalStateException("tap point " + layer + " mixed device and host captures");
                }
                if (widths.get(j) != width) {
                    throw new IllegalStateException("tap point " + layer + " changed width from " + width
                            + " to " + widths.get(j) + " within a call");
                }
                int[] map = rowMaps.get(j);
                if (map == null) {
                    System.arraycopy(parts.get(j), 0, all, offsets.get(j) * width, counts.get(j) * width);
                    continue;
                }
                for (int t = 0; t < counts.get(j); t++) {
                    System.arraycopy(parts.get(j), t * width, all, map[t] * width, width);
                }
            }
            tap.onHidden(layer, all, total, width);
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
}
