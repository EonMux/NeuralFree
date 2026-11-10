package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchPlan;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchSink;
import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
import com.eonmux.neuralfree.acceleration.backend.StickyArgs;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

/**
 * A captured sequence of Vulkan dispatches, replayed as one submission. The dispatches were recorded once into a
 * simultaneous-use command buffer with their pipelines, descriptors and push constants; a replay submits that
 * buffer again while no dispatch has changed since the capture.
 *
 * <p><b>Changed dispatches.</b> Push constants and group counts are baked into a recorded command buffer, so a
 * dispatch whose scalar arguments or grid change (a decode step's key count and split layout) needs a new
 * recording: from the first change on, every submission records the step again, with the current values, into a
 * command buffer of the graph's {@link VkGraphRecording} ring and submits that one buffer. Changes reach the graph
 * two ways: a body replay
 * ({@link #replay(Runnable)}) runs the step body with the queue's dispatches routed here, each matched against the
 * next captured one (the kernel handle, its local size and, argument for argument, the bound buffers) and adopted
 * when only scalars or the grid moved; a patched replay ({@link #replayPatched}) takes the plan's values through a
 * {@link GpuGraphPatchSink} without running the body. A dispatch that binds another buffer, changes an argument's
 * kind, or a body whose dispatch sequence leaves the capture throws {@link GpuGraphMismatchException} before any
 * device work, and the caller captures the step again: the hazard barriers recorded at capture time hold only for the
 * captured buffers.</p>
 *
 * <p>{@link #preparePatched} records the follow-up step while the current step runs, so the
 * follow-up's patched replay only submits. Every submission waits on the queue's timeline for the work submitted
 * before it, and the queue's later submissions wait on it in turn, so the replayed dispatches run in the queue's
 * order as if they had been enqueued one by one. A graph runs only on the queue that captured it, because its command
 * buffers belong to that queue. Buffers bound in the replay must stay open until {@link #close()}; a closed bound
 * buffer is refused by name. {@code close()} waits for the last submitted replay, then frees the command buffers and
 * descriptor frames with the queue.</p>
 */
final class VkGraph implements GpuGraph {

    /** An owned copy of one dispatch's bound values; holder arrays are cloned, buffers kept by identity. */
    static Object[] copyArgs(VkArgState state) {
        StickyArgs sticky = state.sticky();
        Object[] out = new Object[sticky.count()];
        for (int i = 0; i < out.length; i++) {
            out[i] = copyValue(sticky.get(i));
        }
        return out;
    }

    private static Object copyValue(Object v) {
        return v instanceof int[] ? ((int[]) v).clone()
                : v instanceof float[] ? ((float[]) v).clone()
                : v instanceof long[] ? ((long[]) v).clone()
                : v instanceof double[] ? ((double[]) v).clone()
                : v instanceof byte[] ? ((byte[]) v).clone()
                : v instanceof short[] ? ((short[]) v).clone()
                : v;
    }

    /**
     * One captured dispatch: its kernel handle and the pipeline variant it ran, its local size, whether a hazard
     * barrier preceded it in the capture, and its current global size, group counts and arguments. Immutable; a
     * change replaces the dispatch with {@link #with}.
     */
    static final class Dispatch {
        final VkKernelHandle handle;
        final VkKernelHandle.Variant variant;
        final int[] local3;
        final boolean barrier;
        final long[] global;
        final long[] groups;
        final Object[] args;

        Dispatch(VkKernelHandle handle, VkKernelHandle.Variant variant, int[] local3, boolean barrier, long[] global,
                 long[] groups, Object[] args) {
            this.handle = handle;
            this.variant = variant;
            this.local3 = local3;
            this.barrier = barrier;
            this.global = global;
            this.groups = groups;
            this.args = args;
        }

        /** This dispatch with another global size, group counts and arguments. */
        Dispatch with(long[] newGlobal, long[] newGroups, Object[] newArgs) {
            return new Dispatch(handle, variant, local3, barrier, newGlobal, newGroups, newArgs);
        }
    }

    private final VkComputeQueue queue;
    private final VkCommandBatch batch;
    private final List<Dispatch> dispatches;
    /** Dispatches changed by a body replay since the last {@link #takeReplayChanges()}. */
    private final BitSet replayChanges = new BitSet();
    /** Dispatches whose values differ from the last recording (or, before the first one, the capture). */
    private final BitSet stale = new BitSet();
    /** The re-recorded form, created on the first change since the capture. */
    private VkGraphRecording recording;
    /** The position the recording's next ring slot was prepared for, or -1. */
    private int preparedPos = -1;
    private int position;
    private boolean closed;

    VkGraph(VkComputeQueue queue, VkCommandBatch batch, List<Dispatch> dispatches) {
        this.queue = queue;
        this.batch = batch;
        this.dispatches = new ArrayList<>(dispatches);
    }

    /** The command buffer recorded at capture time, with the capture's values. */
    VkCommandBatch batch() {
        return batch;
    }

    /** The re-recorded form, or null while no dispatch has changed since the capture. */
    VkGraphRecording recording() {
        return recording;
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
            awaitRecordingSlot();
            queue.submitGraphReplay(this);
            submitted = true;
        } finally {
            queue.endGraphReplay(this, submitted);
        }
    }

    @Override
    public List<GpuGraphDispatch> kernelDispatches() {
        List<GpuGraphDispatch> out = new ArrayList<>(dispatches.size());
        for (int i = 0; i < dispatches.size(); i++) {
            out.add(describe(i));
        }
        return out;
    }

    @Override
    public GpuGraphDispatch kernelDispatch(int index) {
        return index >= 0 && index < dispatches.size() ? describe(index) : null;
    }

    private GpuGraphDispatch describe(int index) {
        Dispatch d = dispatches.get(index);
        long[] block = {d.local3[0], d.local3[1], d.local3[2]};
        return new GpuGraphDispatch(index, d.handle.kernelName(), d.global,
                Arrays.copyOf(block, d.global.length), d.args);
    }

    @Override
    public int[] takeReplayChanges() {
        int[] out = replayChanges.stream().toArray();
        replayChanges.clear();
        return out;
    }

    @Override
    public boolean supportsPatchedReplay() {
        return true;
    }

    @Override
    public boolean replayPatched(GpuGraphPatchPlan plan, int pos) {
        if (plan == null) {
            throw new IllegalArgumentException("patch plan must not be null");
        }
        requireOpen();
        boolean prepared = preparedPos == pos && stale.isEmpty() && recording != null && recording.recorded();
        preparedPos = -1;
        if (!prepared) {
            if (!plan.patchStep(pos, new Sink())) {
                return false;
            }
            awaitRecordingSlot();
        }
        queue.submitPatchedReplay(this, prepared);
        return true;
    }

    @Override
    public boolean preparePatched(GpuGraphPatchPlan plan, int nextPos, int lastPos) {
        if (plan == null) {
            throw new IllegalArgumentException("patch plan must not be null");
        }
        requireOpen();
        preparedPos = -1;
        if (recording == null || !plan.patchStep(nextPos, new Sink())) {
            return false;
        }
        awaitRecordingSlot();
        if (!queue.prepareGraphRecording(this)) {
            return false;
        }
        preparedPos = nextPos;
        return true;
    }

    /**
     * Blocks, outside the queue's lock, until the ring slot the next recording takes has finished its last
     * submission, so the recording under the lock never waits on the device while other threads poll the queue.
     */
    private void awaitRecordingSlot() {
        long value = recording != null ? recording.slotWait() : 0;
        if (value > 0) {
            queue.waitForValue(value);
        }
    }

    /**
     * Records the step with the dispatches' current values into the next ring slot of the re-recorded form, creating
     * the form on the first change, and returns it for a submission. Null while nothing changed since the capture
     * (the captured command buffer holds every value). A recording replaces whatever the slot held, so a prepared
     * step no longer stands once anything records. Called under the queue's lock.
     */
    VkGraphRecording recordChanges(VkGraphRecording.Resources resources) {
        preparedPos = -1;
        if (recording == null) {
            if (stale.isEmpty()) {
                return null;
            }
            recording = VkGraphRecording.create(resources, dispatches);
        }
        recording.record(dispatches, stale);
        stale.clear();
        return recording;
    }

    /**
     * One dispatch of the replay body: matches it against the next captured dispatch and adopts changed scalar
     * arguments and group counts. The bound buffers, the argument kinds and the local size must match the capture.
     *
     * @throws IllegalStateException     when the graph is closed or a bound buffer is closed
     * @throws GpuGraphMismatchException when the body's dispatch sequence, local size, buffers or argument kinds
     *                                   leave the capture
     */
    void dispatch(VkKernelHandle handle, long[] global, int[] local3, long[] groups, Object[] args) {
        requireOpen();
        String name = handle.kernelName();
        if (position >= dispatches.size()) {
            throw new GpuGraphMismatchException("command replay of " + this + ": " + name
                    + " enqueues more dispatches than the " + dispatches.size() + " captured");
        }
        int index = position++;
        Dispatch d = dispatches.get(index);
        if (d.handle != handle) {
            throw new GpuGraphMismatchException("command replay of " + this + ": dispatch " + (index + 1)
                    + " is " + name + " but the capture holds " + d.handle.kernelName() + " at this point");
        }
        if (!Arrays.equals(d.local3, local3)) {
            throw new GpuGraphMismatchException("command replay of " + this + ": " + name + " changed its"
                    + " local size from " + Arrays.toString(d.local3) + " to " + Arrays.toString(local3)
                    + "; re-capture the step");
        }
        if (d.args.length != args.length) {
            throw new GpuGraphMismatchException("command replay of " + this + ": " + name + " binds "
                    + args.length + " arguments, the capture holds " + d.args.length);
        }
        boolean recorded = !Arrays.equals(d.groups, groups);
        for (int i = 0; i < args.length; i++) {
            requireOpenBuffer(name, d.args[i]);
            if (!KernelArgs.sameValue(d.args[i], args[i])) {
                requireScalarChange(d, i, args[i]);
                recorded = true;
            }
        }
        if (recorded || !Arrays.equals(d.global, global)) {
            adopt(index, d.with(global.clone(), groups.clone(), args), recorded);
            replayChanges.set(index);
        }
    }

    /**
     * Replaces dispatch {@code index} with its changed state, and marks it for the next recording when
     * {@code recorded}: its group counts or arguments moved. A global size alone that folds to the same group counts
     * changes nothing the command buffer holds.
     */
    private void adopt(int index, Dispatch changed, boolean recorded) {
        dispatches.set(index, changed);
        if (recorded) {
            stale.set(index);
        }
    }

    /** The patch sink of one step: applies each patch that differs from the dispatch's current values. */
    private final class Sink implements GpuGraphPatchSink {
        @Override
        public void patchKernel(int recordIndex, long[] grid, int[] argIndices, Object[] values) {
            if (recordIndex < 0 || recordIndex >= dispatches.size()) {
                throw new IllegalArgumentException("graph " + VkGraph.this + " holds no kernel dispatch at "
                        + recordIndex);
            }
            if (argIndices.length != values.length) {
                throw new IllegalArgumentException("patch of dispatch " + recordIndex + " names "
                        + argIndices.length + " arguments but carries " + values.length + " values");
            }
            Dispatch d = dispatches.get(recordIndex);
            Object[] args = null;
            for (int a = 0; a < argIndices.length; a++) {
                int i = argIndices[a];
                if (i < 0 || i >= d.args.length) {
                    throw new IllegalArgumentException(d.handle.kernelName() + " has no argument " + i);
                }
                if (!KernelArgs.sameValue(d.args[i], values[a])) {
                    requireScalarChange(d, i, values[a]);
                    if (args == null) {
                        args = d.args.clone();
                    }
                    args[i] = copyValue(values[a]);
                }
            }
            long[] global = d.global;
            long[] groups = d.groups;
            if (grid != null && !Arrays.equals(grid, d.global)) {
                global = grid.clone();
                groups = queue.groupCounts(d.handle, global, d.local3);
            }
            if (args != null || global != d.global) {
                adopt(recordIndex, d.with(global, groups, args != null ? args : d.args),
                        args != null || !Arrays.equals(groups, d.groups));
            }
        }
    }

    /** Refuses a changed argument that is not a scalar of the captured kind: the barriers hold only for the buffers. */
    private void requireScalarChange(Dispatch d, int i, Object now) {
        Object then = d.args[i];
        if (then instanceof GpuBuffer || now instanceof GpuBuffer || then == null || now == null
                || then.getClass() != now.getClass()) {
            throw new GpuGraphMismatchException("command replay of " + this + ": " + d.handle.kernelName()
                    + " argument " + i + " changed from " + KernelArgs.describe(then) + " to "
                    + KernelArgs.describe(now) + "; only scalar arguments are re-recorded, re-capture the step");
        }
    }

    private static void requireOpenBuffer(String name, Object arg) {
        if (arg instanceof VkGpuBuffer && ((VkGpuBuffer) arg).isClosed()) {
            throw new IllegalStateException(name + ": bound buffer " + arg + " is closed while its"
                    + " command graph is open; close the graph first");
        }
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("command graph " + this + " is closed");
        }
    }

    /** Waits for the last submitted replay, then releases the command buffers with the queue. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        if (queue.isClosed()) {
            // the queue's close freed the command buffers with the rest of its resources
            closed = true;
            return;
        }
        queue.releaseGraph(this);
    }

    /** Marks the graph closed; the queue's submissions have drained and it freed the command buffers. */
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
        return "VkGraph{" + dispatches.size() + " dispatches"
                + (recording == null ? "" : ", re-recorded")
                + (closed ? ", closed" : "") + "}";
    }
}
