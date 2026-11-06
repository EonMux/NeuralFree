package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Objects;

/**
 * The heat-profile read-back of one device router's selected ids ({@link MoeExpertHeat}): each step's ids are read
 * asynchronously after the router kernels and counted for the router's weight layer as the reads complete. At most
 * {@link #PENDING} reads stay pending; beyond that the next step waits for the oldest instead of falling further
 * behind. {@link #drain()} counts every pending read before the blocks close. Not thread-safe: one forward pass at a
 * time.
 */
final class GpuRouterHeat {

    /** Routed-id reads kept pending while the queue runs behind. */
    static final int PENDING = 4;

    private final ComputeQueue queue;
    private final MoeExpertHeat heat;
    private final EplbLoadWindow window;
    private final int weightLayer;
    private final int il;
    private final Deque<Read> reads = new ArrayDeque<>();

    /**
     * @param heat        the routed-row recorder the counts go to, or null without one
     * @param window      the live-load window of the runtime rearrangement, or null without one
     * @param weightLayer the layer the counts belong to ({@code LayerSpec.weightLayer()})
     * @param il          the layer named in an interruption failure
     */
    GpuRouterHeat(ComputeQueue queue, MoeExpertHeat heat, EplbLoadWindow window, int weightLayer, int il) {
        this.queue = Objects.requireNonNull(queue, "queue");
        if (heat == null && window == null) {
            throw new IllegalArgumentException("a router heat reads into a profile or a load window");
        }
        this.heat = heat;
        this.window = window;
        this.weightLayer = weightLayer;
        this.il = il;
    }

    /**
     * Counts the reads that finished, then enqueues the read of this step's {@code pairs} ids from {@code ids}: the
     * read runs on the same queue after the router kernels, so it always captures this step's selection.
     */
    void note(ScratchSlot ids, int pairs) {
        while (!reads.isEmpty() && reads.peekFirst().event.isComplete()) {
            count(reads.pollFirst(), false);
        }
        while (reads.size() >= PENDING) {
            count(reads.pollFirst(), true);
        }
        byte[] bytes = new byte[pairs * Integer.BYTES];
        GpuEvent read = queue.enqueueReadBuffer(ids.buffer(), (long) ids.offset() * Float.BYTES, bytes.length, bytes);
        reads.addLast(new Read(read, bytes, pairs));
    }

    /** Counts every pending read; returns the first failure (later ones suppressed into it), or null. */
    RuntimeException drain() {
        RuntimeException failure = null;
        while (!reads.isEmpty()) {
            try {
                count(reads.pollFirst(), true);
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        return failure;
    }

    /** Counts one read into the heat profile and the load window, awaits it first when asked, releases its event. */
    private void count(Read read, boolean await) {
        try (GpuEvent event = read.event) {
            if (await) {
                event.awaitCompletion();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("the heat read of layer " + il + " was interrupted", e);
        }
        int[] routedIds = new int[read.pairs];
        ByteBuffer.wrap(read.bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(routedIds);
        if (heat != null) {
            heat.note(weightLayer, routedIds, read.pairs);
        }
        if (window != null) {
            window.note(weightLayer, routedIds, read.pairs);
        }
    }

    /** One pending routed-id read: its event, its target bytes and its pair count. */
    private static final class Read {
        final GpuEvent event;
        final byte[] bytes;
        final int pairs;

        Read(GpuEvent event, byte[] bytes, int pairs) {
            this.event = event;
            this.bytes = bytes;
            this.pairs = pairs;
        }
    }
}
