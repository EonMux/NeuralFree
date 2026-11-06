package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuHiddenFeed;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.memory.SequenceState;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The dflash drafter's {@link HiddenTap} on the GPU target its device stack drafts on ({@link GpuHiddenFeed}):
 * every target layer's rows of a forward call are captured on the device — one device-to-device copy per ubatch
 * chunk into a slot the feed owns — and reach the drafter's encoder without a host round trip. At the call's
 * delivery the feed zips the layers' chunk sequences (every layer captures the same row spans, in call order):
 * each chunk's per-layer segments are copied into one {@code [rows][nTargets * nEmbd]} feature buffer
 * ({@code LLM_Copy2d}, the layers in feature order) and handed to the sink with the chunk's first position, one
 * injection per chunk. Rows a forward path delivers on the host anyway (a batched ubatch, or a tap point whose
 * shard runs on another queue) are uploaded into the slots, so a call mixes device and host captures freely. A
 * target whose rows carry several hyper-connection streams never captures on the device; its drafter takes the
 * host tap. Not thread-safe; the target's forward thread drives it.
 */
final class DflashDeviceFeed implements HiddenTap, GpuHiddenFeed {

    /** Receives the device features of one injection chunk. */
    interface Sink {
        /**
         * @param features row-major {@code [nRows][nTargets * nEmbd]} device rows, owned by the feed and reused
         *                 after the call returns
         */
        void onDeviceFeatures(GpuBuffer features, int nRows, int pos0);
    }

    /** One captured chunk of one tap layer: its device slot and its row count. */
    private static final class Capture {
        final GpuBuffer rows;
        final int nRows;

        Capture(GpuBuffer rows, int nRows) {
            this.rows = rows;
            this.nRows = nRows;
        }
    }

    private final int[] targetLayers;
    private final int nEmbd;
    private final int rowWidth;
    private final SequenceState seq;
    private final GpuOps ops;
    private final ComputeBackend backend;
    private final Sink sink;
    /** Captures of the current call per tap layer, in call order; every layer holds the same row spans. */
    private final Map<Integer, List<Capture>> captures = new LinkedHashMap<>();
    private final List<GpuBuffer> pool = new ArrayList<>();
    private final List<GpuBuffer> owned = new ArrayList<>();
    private int deliveredLayers;

    /**
     * @param targetLayers tap points in feature order, the layer each feature column carries
     * @param nEmbd        target hidden size, the width a captured row delivers
     * @param seq          the target's sequence, for the chunks' positions at the delivery
     * @param ops          the target's device ops, whose queue the copies ride
     * @param backend      the target's device, which allocates the slots
     */
    DflashDeviceFeed(int[] targetLayers, int nEmbd, SequenceState seq, GpuOps ops, ComputeBackend backend,
                     Sink sink) {
        if (targetLayers.length == 0 || nEmbd <= 0) {
            throw new IllegalArgumentException("dflash device features need layers and a positive width");
        }
        this.targetLayers = targetLayers.clone();
        this.nEmbd = nEmbd;
        this.rowWidth = targetLayers.length * nEmbd;
        this.seq = Objects.requireNonNull(seq, "seq");
        this.ops = Objects.requireNonNull(ops, "ops");
        this.backend = Objects.requireNonNull(backend, "backend");
        this.sink = Objects.requireNonNull(sink, "sink");
    }

    /** Values per feature row: {@code nTargets * nEmbd}. */
    int rowWidth() {
        return rowWidth;
    }

    @Override
    public boolean wants(int layer) {
        return Arrays.binarySearch(targetLayers, layer) >= 0;
    }

    @Override
    public boolean wantsDevice(int layer, ComputeQueue forwardQueue) {
        return wants(layer) && forwardQueue == ops.queue();
    }

    @Override
    public void captureDevice(int layer, GpuBuffer rows, long rowOffset, int nRows, int width) {
        check(layer, width);
        GpuBuffer slot = take((long) nRows * width);
        ops.queue().enqueueCopyBuffer(rows, rowOffset * Float.BYTES, slot, 0,
                (long) nRows * width * Float.BYTES);
        captures.computeIfAbsent(layer, l -> new ArrayList<>()).add(new Capture(slot, nRows));
    }

    @Override
    public void onHiddenDevice(int layer, int nRows, int width) {
        check(layer, width);
        deliveredLayers++;
        deliver();
    }

    /** The host delivery of rows a forward path read back anyway: uploaded into the layer's slots. */
    @Override
    public void onHidden(int layer, float[] rows, int nRows, int width) {
        check(layer, width);
        int nPast = seq.nPast();
        if (nRows > nPast) {
            throw new IllegalStateException("dflash device feed: a forward call of " + nRows + " rows left " + nPast
                    + " stored positions");
        }
        GpuBuffer slot = take((long) nRows * width);
        GpuEvent e = ops.queue().enqueueWriteBuffer(slot, 0, (long) nRows * width * Float.BYTES,
                Arrays.copyOf(rows, nRows * width));
        if (e != null) {
            e.close();
        }
        captures.computeIfAbsent(layer, l -> new ArrayList<>()).add(new Capture(slot, nRows));
        deliveredLayers++;
        deliver();
    }

    /** One injection per chunk span when every tap layer of the call has delivered. */
    private void deliver() {
        if (deliveredLayers < targetLayers.length) {
            return;
        }
        List<Capture> first = spans();
        int nPast = seq.nPast();
        int total = 0;
        for (Capture c : first) {
            total += c.nRows;
        }
        int before = 0;
        for (int chunk = 0; chunk < first.size(); chunk++) {
            int nRows = first.get(chunk).nRows;
            GpuBuffer features = take((long) nRows * rowWidth);
            for (int layer : targetLayers) {
                Capture c = captures.get(layer).get(chunk);
                if (c.nRows != nRows) {
                    throw new IllegalStateException("dflash device feed: tap layer " + layer + " delivered a chunk of "
                            + c.nRows + " rows where the call's spans hold " + nRows);
                }
                ops.copy2d(c.rows, 0, nEmbd, features, column(layer), rowWidth, nRows, nEmbd);
            }
            sink.onDeviceFeatures(features, nRows, nPast - total + before);
            before += nRows;
        }
        releaseAll();
    }

    /** The first layer's chunk sequence, the spans every layer must repeat. */
    private List<Capture> spans() {
        return captures.get(targetLayers[0]);
    }

    /** Feature column of tap layer {@code layer}. */
    private int column(int layer) {
        for (int k = 0; k < targetLayers.length; k++) {
            if (targetLayers[k] == layer) {
                return k * nEmbd;
            }
        }
        throw new IllegalStateException("dflash device feed: tap layer " + layer + " is not requested ("
                + Arrays.toString(targetLayers) + ")");
    }

    private void check(int layer, int width) {
        if (!wants(layer)) {
            throw new IllegalStateException("dflash device feed: tap layer " + layer + " is not requested ("
                    + Arrays.toString(targetLayers) + ")");
        }
        if (width != nEmbd) {
            throw new IllegalStateException("dflash device feed: tap layer " + layer + " delivers rows of " + width
                    + " floats, not one stream of " + nEmbd);
        }
    }

    /** A device buffer of at least {@code count} floats: a pooled one, or a fresh one the feed owns. */
    private GpuBuffer take(long count) {
        long bytes = Math.max(count * Float.BYTES, Integer.BYTES);
        for (int i = 0; i < pool.size(); i++) {
            GpuBuffer b = pool.get(i);
            if (b.sizeInBytes() >= bytes) {
                pool.remove(i);
                return b;
            }
        }
        GpuBuffer b = backend.allocateBuffer(bytes);
        owned.add(b);
        return b;
    }

    /** Returns every buffer of the delivered call to the pool. */
    private void releaseAll() {
        for (List<Capture> spans : captures.values()) {
            for (Capture c : spans) {
                pool.add(c.rows);
            }
        }
        captures.clear();
        deliveredLayers = 0;
    }

    /** Drops a partially delivered call: its slots return to the pool. */
    void drop() {
        releaseAll();
    }

    /** Releases every buffer. Idempotent. */
    void close() {
        for (GpuBuffer b : owned) {
            b.close();
        }
        owned.clear();
        pool.clear();
        captures.clear();
    }

    @Override
    public String toString() {
        return "DflashDeviceFeed{layers=" + Arrays.toString(targetLayers) + ", nEmbd=" + nEmbd + "}";
    }
}
