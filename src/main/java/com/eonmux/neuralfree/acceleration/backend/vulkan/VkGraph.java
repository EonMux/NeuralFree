package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
import com.eonmux.neuralfree.acceleration.backend.StickyArgs;

import java.util.Arrays;
import java.util.List;

/**
 * A captured sequence of Vulkan dispatches, replayed by resubmitting the pre-recorded command buffer:
 * the dispatches were recorded once into a simultaneous-use command buffer with their pipelines,
 * descriptors and push constants, and every replay submits that buffer again. A replay runs the step
 * body with the queue's dispatches routed here: each dispatch is matched against the next captured one
 * (the kernel handle, the group counts and, argument for argument, the bound values a replay would
 * need to re-push), so a body that differs from the capture throws
 * {@link GpuGraphMismatchException} before any device work and the caller captures the step again.
 *
 * <p>The push constants of a recorded buffer are fixed, so the replay refuses when a captured kernel's
 * bound arguments changed since the capture, naming the kernel; per-step device values must flow
 * through buffers written outside the replay. Every replay submission waits on the queue's timeline
 * for the work submitted before it, and the queue's later submissions wait on the replay in turn, so
 * the replayed buffer runs in the queue's order as if its dispatches had been enqueued one by one. A
 * graph runs only on the queue that captured it, because its command buffer belongs to that queue.
 * Buffers bound in the replay must stay open until {@link #close()}; a closed bound buffer is refused
 * by name. {@code close()} waits for the last submitted replay, then frees the command buffer and its
 * descriptor frame with the queue.</p>
 */
final class VkGraph implements GpuGraph {

    /** An owned copy of one dispatch's bound values; holder arrays are cloned, buffers kept by identity. */
    static Object[] copyArgs(VkArgState state) {
        StickyArgs sticky = state.sticky();
        Object[] out = new Object[sticky.count()];
        for (int i = 0; i < out.length; i++) {
            Object v = sticky.get(i);
            out[i] = v instanceof int[] ? ((int[]) v).clone()
                    : v instanceof float[] ? ((float[]) v).clone()
                    : v instanceof long[] ? ((long[]) v).clone()
                    : v instanceof double[] ? ((double[]) v).clone()
                    : v instanceof byte[] ? ((byte[]) v).clone()
                    : v instanceof short[] ? ((short[]) v).clone()
                    : v;
        }
        return out;
    }

    /** One captured dispatch: its kernel handle, the recorded group counts and the recorded arguments. */
    static final class Dispatch {
        final VkKernelHandle handle;
        final long[] groups;
        final Object[] args;

        Dispatch(VkKernelHandle handle, long[] groups, Object[] args) {
            this.handle = handle;
            this.groups = groups;
            this.args = args;
        }
    }

    private final VkComputeQueue queue;
    private final VkCommandBatch batch;
    private final List<Dispatch> dispatches;
    private int position;
    private boolean closed;

    VkGraph(VkComputeQueue queue, VkCommandBatch batch, List<Dispatch> dispatches) {
        this.queue = queue;
        this.batch = batch;
        this.dispatches = dispatches;
    }

    VkCommandBatch batch() {
        return batch;
    }

    @Override
    public int dispatchCount() {
        return dispatches.size();
    }

    @Override
    public GpuEvent launch(ComputeQueue target, GpuEvent... waitOnEvents) {
        return queue.launchCapture(this, target, waitOnEvents);
    }

    @Override
    public void replay(Runnable body) {
        if (body == null) {
            throw new IllegalArgumentException("replay body must not be null");
        }
        queue.beginGraphReplay(this);
        position = 0;
        boolean submitted = false;
        try {
            body.run();
            if (position != dispatches.size()) {
                throw new GpuGraphMismatchException("command replay of " + this + " enqueues " + position
                        + " dispatches, the capture holds " + dispatches.size());
            }
            queue.submitGraphReplay(this);
            submitted = true;
        } finally {
            queue.endGraphReplay(this, submitted);
        }
    }

    /**
     * One dispatch of the replay body: matches it against the next captured dispatch. The recorded push
     * constants are fixed, so a changed bound value refuses the replay; a changed group count refuses
     * it too, and so does a closed bound buffer.
     *
     * @throws IllegalStateException     when the graph is closed or a bound buffer is closed
     * @throws GpuGraphMismatchException when the body's dispatch sequence or arguments leave the capture
     */
    void dispatch(VkKernelHandle handle, long[] groups, Object[] args) {
        if (closed) {
            throw new IllegalStateException("command graph of " + handle.kernelName() + " is closed");
        }
        String name = handle.kernelName();
        if (position >= dispatches.size()) {
            throw new GpuGraphMismatchException("command replay of " + this + ": " + name
                    + " enqueues more dispatches than the " + dispatches.size() + " captured");
        }
        Dispatch d = dispatches.get(position++);
        if (d.handle != handle) {
            throw new GpuGraphMismatchException("command replay of " + this + ": dispatch " + position
                    + " is " + name + " but the capture holds " + d.handle.kernelName() + " at this point");
        }
        if (!Arrays.equals(d.groups, groups)) {
            throw new GpuGraphMismatchException("command replay of " + this + ": " + name + " changed its"
                    + " group counts from " + Arrays.toString(d.groups) + " to " + Arrays.toString(groups)
                    + "; re-capture the step");
        }
        if (d.args.length != args.length) {
            throw new GpuGraphMismatchException("command replay of " + this + ": " + name + " binds "
                    + args.length + " arguments, the capture holds " + d.args.length);
        }
        for (int i = 0; i < args.length; i++) {
            if (d.args[i] instanceof VkGpuBuffer && ((VkGpuBuffer) d.args[i]).isClosed()) {
                throw new IllegalStateException(name + ": bound buffer " + d.args[i] + " is closed while its"
                        + " command graph is open; close the graph first");
            }
            if (!KernelArgs.sameValue(d.args[i], args[i])) {
                throw new GpuGraphMismatchException("command replay of " + this + ": " + name
                        + " argument " + i + " changed since the capture; the recorded push constants are"
                        + " fixed, re-capture the step");
            }
        }
    }

    /** Waits for the last submitted replay, then releases the command buffer with the queue. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        if (queue.isClosed()) {
            // the queue's close freed the command buffer with the rest of its resources
            closed = true;
            return;
        }
        queue.releaseGraph(this);
    }

    /** Frees the command buffer and descriptor frame; the queue's submissions have drained. */
    void destroy() {
        closed = true;
    }

    /** True once {@link #close()} has run, or the owning queue closed and released the graph. */
    @Override
    public boolean isClosed() {
        return closed;
    }

    @Override
    public String toString() {
        return "VkGraph{" + dispatches.size() + " dispatches" + (closed ? ", closed" : "") + "}";
    }
}
