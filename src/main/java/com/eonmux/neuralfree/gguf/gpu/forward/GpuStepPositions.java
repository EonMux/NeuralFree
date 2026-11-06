package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * The four-section RoPE positions of one step, written into a context's {@link ScratchSlots#POS} slot: the per-step
 * host staging of {@code GpuPrefillDriver} and of the split forward backend, one write per context per step.
 * Not thread-safe.
 */
final class GpuStepPositions {

    private final ComputeQueue queue;
    private final MropeTextPositions textPositions;
    private final ScratchSlot posSlot;
    private final int[] posInts;
    private final ByteBuffer posBytes;

    GpuStepPositions(GpuScratch scratch, ComputeQueue queue, MropeTextPositions textPositions) {
        this.queue = Objects.requireNonNull(queue, "queue");
        this.textPositions = Objects.requireNonNull(textPositions, "textPositions");
        this.posSlot = scratch.slot(ScratchSlots.POS).require((long) MropeTextPositions.SECTIONS * scratch.ubatch());
        this.posInts = new int[MropeTextPositions.SECTIONS * scratch.ubatch()];
        this.posBytes = ByteBuffer.allocate(posInts.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    }

    /** Writes {@code pos[t + k*n]} for a step of {@code n} tokens at {@code p} into the POS slot. */
    void write(int p, int n) {
        RopePositions.fillStep(textPositions, p, n, posInts);
        int count = MropeTextPositions.SECTIONS * n;
        posBytes.clear();
        for (int i = 0; i < count; i++) {
            posBytes.putInt(posInts[i]);
        }
        GpuEvent ev = queue.enqueueWriteBuffer(posSlot.buffer(), (long) posSlot.offset() * Float.BYTES,
                (long) count * Integer.BYTES, posBytes.array());
        if (ev != null) {
            ev.close();
        }
    }

    /** Writes the per-row positions {@code p[0 .. n)} of one step into the POS slot. */
    void write(int[] p, int n) {
        RopePositions.fillStep(textPositions, p, 0, n, posInts);
        int count = MropeTextPositions.SECTIONS * n;
        posBytes.clear();
        for (int i = 0; i < count; i++) {
            posBytes.putInt(posInts[i]);
        }
        GpuEvent ev = queue.enqueueWriteBuffer(posSlot.buffer(), (long) posSlot.offset() * Float.BYTES,
                (long) count * Integer.BYTES, posBytes.array());
        if (ev != null) {
            ev.close();
        }
    }
}
