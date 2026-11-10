package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import org.lwjgl.vulkan.VkCommandBuffer;

import java.util.BitSet;
import java.util.List;

/**
 * The re-recorded form of a {@link VkGraph} whose dispatches change between submissions: every captured dispatch is
 * recorded again, with its current values, into one command buffer of a ring of {@value #RING} slots, so the next
 * step's recording can proceed while an earlier step's buffer still executes. A submission sends the one buffer.
 *
 * <p>The whole step goes into one buffer: recording only the changed runs of dispatches would submit one command
 * buffer per run, and a decode step whose flash attention, rope and cache writes change in every layer has a few
 * hundred runs, each paying a submission.</p>
 *
 * <p>Each dispatch is recorded from a graph-owned {@link VkArgState} holding the dispatch's current arguments, never
 * from its kernel handle's live state, so the step body's later bindings cannot reach a recorded buffer; only the
 * states of dispatches changed since the previous recording are refreshed. The buffer opens with a global barrier
 * and keeps every hazard barrier of the capture. Every slot binds descriptors from one frame (a pooled strategy's
 * set is reused while the dispatch's bindings stay those of the capture, which a replay requires).</p>
 */
@NotThreadSafe("Owned by one graph; used under its queue's lock.")
final class VkGraphRecording implements AutoCloseable {

    /** Ring slots: the step being submitted, the one executing and one in flight before it. */
    static final int RING = 3;

    /** The queue resources a re-recorded graph records with. */
    interface Resources {
        /** A new command batch of the queue's pool with {@code frame}, not yet begun. */
        VkCommandBatch newBatch(VkDescriptorStrategy.Frame frame, boolean simultaneousUse);

        /** A new descriptor frame of the queue's strategy. */
        VkDescriptorStrategy.Frame newFrame();

        /** Blocks until the queue's timeline reaches {@code value}. */
        void awaitTimeline(long value);

        /** Frees a batch {@link #newBatch} returned; its submissions have completed. */
        void freeBatch(VkCommandBatch batch);

        /** Destroys a frame {@link #newFrame} returned. */
        void destroyFrame(VkDescriptorStrategy.Frame frame);
    }

    private final Resources resources;
    private final VkCommandBatch[] ring;
    private final VkDescriptorStrategy.Frame frame;
    private final VkArgState[] states;
    /** The timeline value each ring slot was last submitted with, 0 when never. */
    private final long[] slotSignal = new long[RING];
    /** The slot the next recording goes to. */
    private int nextSlot;
    /** The slot recorded and not yet submitted, or -1. */
    private int recordedSlot = -1;
    private boolean closed;

    private VkGraphRecording(Resources resources, VkCommandBatch[] ring, VkDescriptorStrategy.Frame frame,
                             VkArgState[] states) {
        this.resources = resources;
        this.ring = ring;
        this.frame = frame;
        this.states = states;
    }

    /** The ring and the graph-owned argument states of {@code dispatches}; nothing is recorded yet. */
    static VkGraphRecording create(Resources resources, List<VkGraph.Dispatch> dispatches) {
        VkArgState[] states = new VkArgState[dispatches.size()];
        for (int i = 0; i < states.length; i++) {
            states[i] = ownedState(dispatches.get(i));
        }
        VkDescriptorStrategy.Frame frame = resources.newFrame();
        VkCommandBatch[] ring = new VkCommandBatch[RING];
        try {
            for (int i = 0; i < RING; i++) {
                ring[i] = resources.newBatch(frame, false);
            }
        } catch (RuntimeException e) {
            for (VkCommandBatch b : ring) {
                if (b != null) {
                    VkGraphResources.step(() -> resources.freeBatch(b), e);
                }
            }
            VkGraphResources.step(() -> resources.destroyFrame(frame), e);
            throw e;
        }
        return new VkGraphRecording(resources, ring, frame, states);
    }

    /** A graph-owned argument state holding {@code d}'s arguments, independent of the kernel handle's own. */
    private static VkArgState ownedState(VkGraph.Dispatch d) {
        VkArgState live = d.handle.argState();
        VkArgState own = new VkArgState(live.kernelName(), live.info(), live.signature());
        own.setAll(d.args);
        own.materialize();
        return own;
    }

    /**
     * The timeline value the slot the next {@link #record} takes was last submitted with, which the recording waits
     * for; 0 when that slot never ran or is the recorded slot not yet submitted.
     */
    long slotWait() {
        return recordedSlot >= 0 ? 0 : slotSignal[nextSlot];
    }

    /**
     * Records every dispatch with its current values into a ring slot, which the next {@link #submitted} hands to
     * the device: the slot recorded last when it was not submitted yet, else the next slot once its previous
     * submission completed. {@code changed} names the dispatches whose values moved since the previous recording.
     */
    void record(List<VkGraph.Dispatch> dispatches, BitSet changed) {
        requireOpen();
        int slot = recordedSlot >= 0 ? recordedSlot : nextSlot;
        if (slotSignal[slot] > 0) {
            resources.awaitTimeline(slotSignal[slot]);
        }
        for (int i = changed.nextSetBit(0); i >= 0 && i < states.length; i = changed.nextSetBit(i + 1)) {
            states[i].setAll(dispatches.get(i).args);
        }
        VkCommandBatch b = ring[slot];
        if (b.isRecording()) {
            b.end();
        }
        b.reset();
        b.begin();
        for (int i = 0; i < states.length; i++) {
            VkGraph.Dispatch d = dispatches.get(i);
            VkArgState state = states[i];
            state.materialize();
            if (d.barrier && i > 0) {
                b.barrier();
            }
            b.bindPipeline(d.variant.pipeline());
            b.bindDescriptors(d.variant.program(), state);
            d.variant.program().pushConstants(b.commandBuffer(), state);
            b.dispatch((int) d.groups[0], (int) d.groups[1], (int) d.groups[2]);
        }
        b.end();
        recordedSlot = slot;
    }

    /** The command buffer of the recorded slot, for one submission. */
    VkCommandBuffer commandBuffer() {
        requireOpen();
        if (recordedSlot < 0) {
            throw new IllegalStateException(this + " holds no recording for a submission");
        }
        return ring[recordedSlot].commandBuffer();
    }

    /** Marks the recorded slot submitted with timeline value {@code signal}; the next recording takes another. */
    void submitted(long signal) {
        if (recordedSlot < 0) {
            return;
        }
        slotSignal[recordedSlot] = signal;
        nextSlot = (recordedSlot + 1) % RING;
        recordedSlot = -1;
    }

    /** True when a recorded ring slot waits for its submission. */
    boolean recorded() {
        return recordedSlot >= 0;
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException(this + " is closed");
        }
    }

    /** Frees the ring's command buffers and the frame; the caller has waited for every submission of the ring. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        for (VkCommandBatch b : ring) {
            failure = VkGraphResources.step(() -> resources.freeBatch(b), failure);
        }
        failure = VkGraphResources.step(() -> resources.destroyFrame(frame), failure);
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "VkGraphRecording{" + states.length + " dispatches, " + RING + " slots"
                + (closed ? ", closed" : "") + "}";
    }
}
