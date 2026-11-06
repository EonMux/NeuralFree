package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.Objects;

/**
 * An FFN block that runs on another stage of a layer split: the pooled expert cache assigns the hosted MoE layers of
 * the model to the GPUs by free VRAM, and a layer whose cache lives on another GPU runs its block there. The
 * executor's normalised FFN input is read back to the host, written into the owner stage's scratch, computed by the
 * owner's block on the owner's queue, and its output copied back into the layer's branch slot on the local queue.
 *
 * <p>The read back synchronises the local queue and the owner's output read synchronises the owner queue, so the two
 * devices run in lock-step around the boundary, as {@link GpuSplitForwardBackend} stages the residual stream between
 * the spans. The block reads step results back to the host, so a decode step of a stack holding one cannot be
 * replayed ({@link #capturableDecode()}). Not thread-safe.</p>
 */
public final class PooledFfnBlock implements GpuFfnBlock {

    private static final String T_INPUT = "moe.pool.input";
    private static final String T_OUTPUT = "moe.pool.output";

    private final GpuFfnBlock delegate;
    private final GpuForwardContext owner;
    private final int ownerStage;
    private final int width;
    private GpuForwardContext local;
    private ScratchSlot ownerIn;
    private ScratchSlot ownerOut;
    private float[] inStage;
    private float[] outStage;
    private int ubatch;

    /**
     * A wrapper around {@code delegate}, whose block runs on {@code owner}.
     *
     * @param ownerStage the index of the owner stage, for diagnostics and tests
     * @param width      the FFN input width (the model's {@code n_embd})
     * @throws IllegalArgumentException when the width is not positive
     */
    public PooledFfnBlock(GpuFfnBlock delegate, int ownerStage, GpuForwardContext owner, int width) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.owner = Objects.requireNonNull(owner, "owner");
        this.ownerStage = ownerStage;
        this.width = width;
        if (width <= 0) {
            throw new IllegalArgumentException("the pooled FFN input width must be positive, got " + width);
        }
    }

    /** The stage the wrapped block runs on. */
    public int ownerStage() {
        return ownerStage;
    }

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        if (local != null) {
            throw new IllegalStateException("the pooled FFN block of layer " + layer.il() + " is already bound");
        }
        local = Objects.requireNonNull(ctx, "ctx");
        ubatch = Math.min(local.scratch().ubatch(), owner.scratch().ubatch());
        long floats = (long) ubatch * width;
        ownerIn = owner.scratch().temp(T_INPUT, floats);
        ownerOut = owner.scratch().temp(T_OUTPUT, floats);
        inStage = new float[Math.toIntExact(floats)];
        outStage = new float[Math.toIntExact(floats)];
        delegate.bind(owner, layer);
    }

    @Override
    public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
        if (local == null) {
            throw new IllegalStateException("the pooled FFN block is used before bind");
        }
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException("token count " + nTokens + " outside [1, " + ubatch + "]");
        }
        int floats = Math.multiplyExact(nTokens, width);
        in.require(floats);
        out.require(floats);
        read(local.queue(), in, floats, inStage);
        write(owner.queue(), ownerIn, floats, inStage);
        delegate.forward(nTokens, ownerIn, ownerOut);
        read(owner.queue(), ownerOut, floats, outStage);
        write(local.queue(), out, floats, outStage);
    }

    @Override
    public boolean capturableDecode() {
        return false;
    }

    @Override
    public void close() {
        delegate.close();
    }

    /** Reads the slot's rows into {@code stage}, waiting for the queue up to them. */
    private static void read(ComputeQueue queue, ScratchSlot slot, int floats, float[] stage) {
        GpuEvent evt = queue.enqueueReadBuffer(slot.buffer(), 4L * slot.offset(), 4L * floats, stage);
        if (evt == null) {
            queue.finish();
            return;
        }
        try {
            evt.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting for the pooled FFN rows", e);
        } finally {
            evt.close();
        }
    }

    /** Writes {@code stage} into the slot; the queue runs the write before the work enqueued after it. */
    private static void write(ComputeQueue queue, ScratchSlot slot, int floats, float[] stage) {
        queue.enqueueWriteBuffer(slot.buffer(), 4L * slot.offset(), 4L * floats, stage);
    }

    @Override
    public String toString() {
        return "PooledFfnBlock{owner=" + ownerStage + "}";
    }
}
