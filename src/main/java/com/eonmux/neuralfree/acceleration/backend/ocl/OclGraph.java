package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchPlan;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchSink;
import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
import com.eonmux.neuralfree.acceleration.backend.StickyArgs;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A captured sequence of OpenCL commands: kernel dispatches, device copies and fills recorded into
 * a {@code cl_command_buffer_khr} and submitted again as one {@code clEnqueueCommandBufferKHR}.
 *
 * <p>A replay runs the step body with the queue's commands routed here: each command is matched
 * against the next recorded one (the kernel, its work sizes and, argument for argument, its bound
 * values). With {@code cl_khr_command_buffer_mutable_dispatch} a changed scalar argument is written
 * into the recorded dispatch through {@code clUpdateMutableCommandsKHR}; a changed buffer binding,
 * work size or kernel is refused with {@link GpuGraphMismatchException} before any device work, so
 * the caller captures the step again. {@link #replayPatched(GpuGraphPatchPlan, int)} skips the body:
 * a plan hands each recorded dispatch's step values over, and the changed scalar arguments go through
 * the same mutable update before one enqueue of the command buffer. Copies and fills are fixed at the
 * capture. The commands of a command buffer carry no implicit order, so they are chained with sync
 * points in capture order.</p>
 *
 * <p>A graph runs only on the queue that captured it, because the command buffer was created for
 * that queue. Buffers bound in the replay must stay open until {@link #close()}; a closed bound
 * buffer is refused by name. {@code close()} drains the queue and releases the command buffer.</p>
 */
final class OclGraph implements GpuGraph {

    /** The kind of command a recorded node holds. */
    enum Kind {
        KERNEL,
        COPY,
        FILL
    }

    /** One recorded command: its work sizes or buffers, and the arguments it was captured with. */
    static final class Node {
        final Kind kind;
        final long mutableCommand;
        /** The dispatch's position among the recorded kernel commands; other kinds take {@code -1}. */
        final int ordinal;

        final OclKernelHandle handle;
        final long[] global;
        final long[] local;
        Object[] args;

        final OclGpuBuffer src;
        final OclGpuBuffer dst;
        final long srcOffset;
        final long dstOffset;
        final long size;
        final int pattern;

        private Node(Kind kind, long mutableCommand, int ordinal, OclKernelHandle handle, long[] global,
                     long[] local, Object[] args, OclGpuBuffer src, long srcOffset, OclGpuBuffer dst,
                     long dstOffset, long size, int pattern) {
            this.kind = kind;
            this.mutableCommand = mutableCommand;
            this.ordinal = ordinal;
            this.handle = handle;
            this.global = global;
            this.local = local;
            this.args = args;
            this.src = src;
            this.srcOffset = srcOffset;
            this.dst = dst;
            this.dstOffset = dstOffset;
            this.size = size;
            this.pattern = pattern;
        }

        static Node kernel(long mutableCommand, int ordinal, OclKernelHandle handle, long[] global, long[] local,
                           Object[] args) {
            return new Node(Kind.KERNEL, mutableCommand, ordinal, handle, global, local, args,
                    null, 0, null, 0, 0, 0);
        }

        static Node copy(OclGpuBuffer src, long srcOffset, OclGpuBuffer dst, long dstOffset, long size) {
            return new Node(Kind.COPY, 0, -1, null, null, null, null, src, srcOffset, dst, dstOffset, size, 0);
        }

        static Node fill(OclGpuBuffer dst, long offset, long size, int pattern) {
            return new Node(Kind.FILL, 0, -1, null, null, null, null, null, 0, dst, offset, size, pattern);
        }
    }

    /** A mutable update the replay body or a patch collected, with the argument values it writes. */
    private static final class PendingUpdate {
        final Node node;
        final OclCommandBufferApi.MutableDispatch dispatch;
        final Object[] args;

        PendingUpdate(Node node, OclCommandBufferApi.MutableDispatch dispatch, Object[] args) {
            this.node = node;
            this.dispatch = dispatch;
            this.args = args;
        }
    }

    private final OclComputeQueue queue;
    private final OclCommandBufferApi api;
    private final long commandBuffer;
    private final boolean simultaneousUse;
    private final List<Node> nodes;
    /** The recorded kernel commands by dispatch position, built once for the patched replay. */
    private final Node[] kernelNodes;
    private final List<PendingUpdate> pendingUpdates = new ArrayList<>();
    /** Kernel-dispatch positions whose recorded command the current body replay updates so far. */
    private final List<Integer> replayChanges = new ArrayList<>();
    private int position;
    private boolean closed;

    OclGraph(OclComputeQueue queue, OclCommandBufferApi api, long commandBuffer, boolean simultaneousUse,
             List<Node> nodes) {
        this.queue = queue;
        this.api = api;
        this.commandBuffer = commandBuffer;
        this.simultaneousUse = simultaneousUse;
        this.nodes = nodes;
        List<Node> kernels = new ArrayList<>();
        for (Node node : nodes) {
            if (node.kind == Kind.KERNEL) {
                kernels.add(node);
            }
        }
        this.kernelNodes = kernels.toArray(new Node[0]);
    }

    /** The queue whose command buffer this graph owns. */
    OclComputeQueue owner() {
        return queue;
    }

    long commandBuffer() {
        return commandBuffer;
    }

    boolean simultaneousUse() {
        return simultaneousUse;
    }

    @Override
    public int dispatchCount() {
        int dispatches = 0;
        for (Node node : nodes) {
            if (node.kind == Kind.KERNEL) {
                dispatches++;
            }
        }
        return dispatches;
    }

    @Override
    public GpuEvent launch(ComputeQueue target, GpuEvent... waitOnEvents) {
        return queue.launchGraph(this, target, waitOnEvents);
    }

    @Override
    public void replay(Runnable body) {
        if (body == null) {
            throw new IllegalArgumentException("replay body must not be null");
        }
        queue.beginGraphReplay(this);
        position = 0;
        pendingUpdates.clear();
        replayChanges.clear();
        boolean submitted = false;
        try {
            body.run();
            if (position != nodes.size()) {
                throw new GpuGraphMismatchException("command replay of " + this + " enqueues " + position
                        + " commands, the capture holds " + nodes.size());
            }
            queue.submitGraphReplay(this);
            submitted = true;
        } finally {
            pendingUpdates.clear();
            queue.endGraphReplay(this, submitted);
        }
    }

    @Override
    public List<GpuGraphDispatch> kernelDispatches() {
        requireOpen("report its dispatches");
        List<GpuGraphDispatch> out = new ArrayList<>(kernelNodes.length);
        for (Node node : kernelNodes) {
            out.add(dispatchOf(node));
        }
        return out;
    }

    @Override
    public GpuGraphDispatch kernelDispatch(int index) {
        requireOpen("report a dispatch");
        Node node = kernelNode(index);
        return new GpuGraphDispatch(node.ordinal, node.handle.kernelName(), node.global, node.local, node.args);
    }

    @Override
    public int[] takeReplayChanges() {
        if (replayChanges.isEmpty()) {
            return new int[0];
        }
        int[] out = new int[replayChanges.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = replayChanges.get(i);
        }
        replayChanges.clear();
        return out;
    }

    @Override
    public boolean supportsPatchedReplay() {
        return api.mutableArguments();
    }

    @Override
    public boolean replayPatched(GpuGraphPatchPlan plan, int pos) {
        if (plan == null) {
            throw new IllegalArgumentException("patch plan must not be null");
        }
        requireOpen("patch and replay");
        if (!api.mutableArguments()) {
            return false;
        }
        queue.beginGraphReplay(this);
        boolean submitted = false;
        try {
            boolean patched = plan.patchStep(pos, this::patchKernel);
            if (!patched) {
                return false;
            }
            queue.submitGraphReplay(this);
            submitted = true;
        } finally {
            pendingUpdates.clear();
            queue.endGraphReplay(this, submitted);
        }
        return true;
    }

    /** The current state of one recorded kernel command. */
    private GpuGraphDispatch dispatchOf(Node node) {
        return new GpuGraphDispatch(node.ordinal, node.handle.kernelName(), node.global, node.local, node.args);
    }

    /**
     * One patch of the plan: collects the changed scalar arguments of the recorded dispatch at
     * {@code recordIndex} as mutable updates the submission writes into the command buffer. A work-size
     * or buffer-binding change is refused like a body replay's, since a command buffer cannot update
     * either.
     */
    private void patchKernel(int recordIndex, long[] grid, int[] argIndices, Object[] values) {
        Node node = kernelNode(recordIndex);
        String name = node.handle.kernelName();
        if (grid != null && !Arrays.equals(node.global, grid)) {
            throw new GpuGraphMismatchException("command patch of " + this + ": " + name + " changed its work"
                    + " size from global " + Arrays.toString(node.global) + " to global " + Arrays.toString(grid)
                    + "; re-capture the step");
        }
        StickyArgs sticky = node.handle.args();
        if (sticky.count() != node.args.length) {
            throw new GpuGraphMismatchException("command patch of " + this + ": " + name + " binds "
                    + sticky.count() + " arguments, the capture holds " + node.args.length);
        }
        List<OclCommandBufferApi.MutableArg> updates = new ArrayList<>();
        Object[] patched = node.args.clone();
        for (int i = 0; i < argIndices.length; i++) {
            int arg = argIndices[i];
            Object value = values[i];
            if (KernelArgs.sameValue(node.args[arg], value)) {
                continue;
            }
            if (node.args[arg] instanceof OclGpuBuffer || value instanceof OclGpuBuffer) {
                throw new GpuGraphMismatchException("command patch of " + this + ": " + name + " argument " + arg
                        + " changes its buffer binding; a command buffer cannot rebind it, re-capture the step");
            }
            byte[] encoded = encode(value);
            if (node.args[arg].getClass() != value.getClass()
                    || encoded.length != encode(node.args[arg]).length) {
                throw new GpuGraphMismatchException("command patch of " + this + ": " + name + " argument " + arg
                        + " changes its type from " + KernelArgs.describe(node.args[arg]) + " to "
                        + KernelArgs.describe(value) + "; re-capture the step");
            }
            if (!api.mutableArguments() || node.mutableCommand == 0) {
                throw new GpuGraphMismatchException("command patch of " + this + ": " + name + " argument " + arg
                        + " changes and the device cannot update recorded dispatch arguments;"
                        + " re-capture the step");
            }
            updates.add(new OclCommandBufferApi.MutableArg(arg, encoded));
            patched[arg] = KernelArgs.snapshot(value, patched[arg]);
        }
        if (!updates.isEmpty()) {
            pendingUpdates.add(new PendingUpdate(node,
                    new OclCommandBufferApi.MutableDispatch(node.mutableCommand, updates), patched));
        }
    }

    /** The recorded kernel command at dispatch position {@code index}. */
    private Node kernelNode(int index) {
        if (index < 0 || index >= kernelNodes.length) {
            throw new IllegalArgumentException(this + " holds " + kernelNodes.length + " kernel dispatches,"
                    + " no dispatch " + index);
        }
        return kernelNodes[index];
    }

    /** Throws when the graph is closed; {@code what} names the refused operation. */
    private void requireOpen(String what) {
        if (closed) {
            throw new IllegalStateException("command graph cannot " + what + ": " + this + " is closed");
        }
    }

    /**
     * One kernel dispatch of the replay body: matches it against the next recorded node. The work
     * sizes and the kernel are fixed; a changed scalar argument is collected for a mutable update,
     * and any other change refuses the replay before any device work.
     *
     * @throws IllegalStateException     when the graph is closed or a bound buffer was closed
     * @throws GpuGraphMismatchException when the body's command sequence leaves the capture
     */
    void dispatch(OclKernelHandle handle, long[] global, long[] local) {
        String name = handle.kernelName();
        Node node = next(Kind.KERNEL, name);
        if (node.handle != handle) {
            throw new GpuGraphMismatchException("command replay of " + this + ": command " + position
                    + " is " + name + " but the capture holds " + node.handle.kernelName() + " at this point");
        }
        if (!Arrays.equals(node.global, global) || !Arrays.equals(node.local, local)) {
            throw new GpuGraphMismatchException("command replay of " + this + ": " + name + " changed its work"
                    + " sizes from global " + Arrays.toString(node.global) + " local " + Arrays.toString(node.local)
                    + " to global " + Arrays.toString(global) + " local " + Arrays.toString(local)
                    + "; re-capture the step");
        }
        StickyArgs sticky = handle.args();
        if (sticky.count() != node.args.length) {
            throw new GpuGraphMismatchException("command replay of " + this + ": " + name + " binds "
                    + sticky.count() + " arguments, the capture holds " + node.args.length);
        }
        List<OclCommandBufferApi.MutableArg> updates = new ArrayList<>();
        for (int i = 0; i < node.args.length; i++) {
            checkBound(node.args[i], name);
            Object current = sticky.isSet(i) ? sticky.get(i) : null;
            if (current == null) {
                throw new GpuGraphMismatchException("command replay of " + this + ": " + name + " argument "
                        + i + " is not set");
            }
            if (KernelArgs.sameValue(node.args[i], current)) {
                continue;
            }
            if (node.args[i] instanceof OclGpuBuffer || current instanceof OclGpuBuffer) {
                throw new GpuGraphMismatchException("command replay of " + this + ": " + name + " argument " + i
                        + " changed its buffer binding; a command buffer cannot rebind it, re-capture the step");
            }
            byte[] encoded = encode(current);
            if (node.args[i].getClass() != current.getClass()
                    || encoded.length != encode(node.args[i]).length) {
                throw new GpuGraphMismatchException("command replay of " + this + ": " + name + " argument " + i
                        + " changed its type from " + KernelArgs.describe(node.args[i]) + " to "
                        + KernelArgs.describe(current) + "; re-capture the step");
            }
            if (!api.mutableArguments() || node.mutableCommand == 0) {
                throw new GpuGraphMismatchException("command replay of " + this + ": " + name + " argument " + i
                        + " changed since the capture and the device cannot update recorded dispatch arguments;"
                        + " re-capture the step");
            }
            updates.add(new OclCommandBufferApi.MutableArg(i, encoded));
        }
        if (!updates.isEmpty()) {
            // the command buffer keeps the written values, so later replays compare against them
            pendingUpdates.add(new PendingUpdate(node,
                    new OclCommandBufferApi.MutableDispatch(node.mutableCommand, updates), snapshot(sticky)));
            replayChanges.add(node.ordinal);
        }
    }

    /**
     * One device copy of the replay body: matches it against the next recorded node. A command
     * buffer cannot change a recorded copy, so any difference refuses the replay.
     */
    void copy(OclGpuBuffer src, long srcOffset, OclGpuBuffer dst, long dstOffset, long size) {
        Node node = next(Kind.COPY, "a device copy");
        checkBound(node.src, "a captured device copy");
        checkBound(node.dst, "a captured device copy");
        if (node.src != src || node.dst != dst || node.srcOffset != srcOffset || node.dstOffset != dstOffset
                || node.size != size) {
            throw new GpuGraphMismatchException("command replay of " + this + ": a device copy changed its buffers"
                    + " or range; a command buffer cannot update it, re-capture the step");
        }
    }

    /** One device fill of the replay body: matched against the next recorded node. */
    void fill(OclGpuBuffer dst, long offset, long size, int pattern) {
        Node node = next(Kind.FILL, "a device fill");
        checkBound(node.dst, "a captured device fill");
        if (node.dst != dst || node.dstOffset != offset || node.size != size || node.pattern != pattern) {
            throw new GpuGraphMismatchException("command replay of " + this + ": a device fill changed its buffer,"
                    + " range or pattern; a command buffer cannot update it, re-capture the step");
        }
    }

    /** Writes the collected scalar updates into the command buffer before it is submitted. */
    void applyUpdates() {
        if (pendingUpdates.isEmpty()) {
            return;
        }
        queue.drainForGraphUpdate();
        List<OclCommandBufferApi.MutableDispatch> dispatches = new ArrayList<>(pendingUpdates.size());
        for (PendingUpdate update : pendingUpdates) {
            dispatches.add(update.dispatch);
        }
        api.updateMutableArguments(commandBuffer, dispatches);
        for (PendingUpdate update : pendingUpdates) {
            update.node.args = update.args;
        }
        pendingUpdates.clear();
    }

    /** The next recorded node, which must have the same kind as the body's command. */
    private Node next(Kind kind, String what) {
        if (closed) {
            throw new IllegalStateException("command graph of " + what + " is closed");
        }
        if (position >= nodes.size()) {
            throw new GpuGraphMismatchException("command replay of " + this + ": " + what
                    + " enqueues more commands than the " + nodes.size() + " captured");
        }
        Node node = nodes.get(position++);
        if (node.kind != kind) {
            throw new GpuGraphMismatchException("command replay of " + this + ": command " + position + " is "
                    + what + " but the capture holds a " + node.kind.name().toLowerCase(java.util.Locale.ROOT)
                    + " at this point");
        }
        return node;
    }

    private void checkBound(Object value, String what) {
        if (value instanceof OclGpuBuffer && ((OclGpuBuffer) value).getCLMem() == null) {
            throw new IllegalStateException(what + ": bound buffer " + value + " is closed while its command graph"
                    + " is open; close the graph first");
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        if (queue.isReleased()) {
            // the queue's close released the command buffer with the rest of its resources
            closed = true;
            return;
        }
        queue.releaseGraph(this);
    }

    /** Releases the command buffer; the queue has drained. */
    void destroy() {
        closed = true;
        api.releaseCommandBuffer(commandBuffer);
    }

    @Override
    public boolean isClosed() {
        return closed;
    }

    @Override
    public String toString() {
        return "OclGraph{" + nodes.size() + " commands, " + dispatchCount() + " dispatches"
                + (closed ? ", closed" : "") + "}";
    }

    /** An owned copy of every set argument; holder arrays are cloned, buffers kept by identity. */
    static Object[] snapshot(StickyArgs sticky) {
        Object[] out = new Object[sticky.count()];
        for (int i = 0; i < out.length; i++) {
            out[i] = sticky.isSet(i) ? KernelArgs.snapshot(sticky.get(i), null) : null;
        }
        return out;
    }

    /** The raw bytes of one scalar argument, in host order, as {@code clSetKernelArg} took them. */
    static byte[] encode(Object value) {
        if (value instanceof Boolean) {
            return ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.nativeOrder())
                    .putInt(0, ((Boolean) value) ? 1 : 0).array();
        }
        if (value instanceof Integer) {
            return ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.nativeOrder())
                    .putInt(0, (Integer) value).array();
        }
        if (value instanceof Float) {
            return ByteBuffer.allocate(Float.BYTES).order(ByteOrder.nativeOrder())
                    .putFloat(0, (Float) value).array();
        }
        if (value instanceof Long) {
            return ByteBuffer.allocate(Long.BYTES).order(ByteOrder.nativeOrder())
                    .putLong(0, (Long) value).array();
        }
        if (value instanceof Double) {
            return ByteBuffer.allocate(Double.BYTES).order(ByteOrder.nativeOrder())
                    .putDouble(0, (Double) value).array();
        }
        if (value instanceof byte[]) {
            return ((byte[]) value).clone();
        }
        if (value instanceof short[]) {
            short[] source = (short[]) value;
            ByteBuffer out = ByteBuffer.allocate(source.length * Short.BYTES).order(ByteOrder.nativeOrder());
            for (short v : source) {
                out.putShort(v);
            }
            return out.array();
        }
        if (value instanceof int[]) {
            int[] source = (int[]) value;
            ByteBuffer out = ByteBuffer.allocate(source.length * Integer.BYTES).order(ByteOrder.nativeOrder());
            for (int v : source) {
                out.putInt(v);
            }
            return out.array();
        }
        if (value instanceof long[]) {
            long[] source = (long[]) value;
            ByteBuffer out = ByteBuffer.allocate(source.length * Long.BYTES).order(ByteOrder.nativeOrder());
            for (long v : source) {
                out.putLong(v);
            }
            return out.array();
        }
        if (value instanceof float[]) {
            float[] source = (float[]) value;
            ByteBuffer out = ByteBuffer.allocate(source.length * Float.BYTES).order(ByteOrder.nativeOrder());
            for (float v : source) {
                out.putFloat(v);
            }
            return out.array();
        }
        if (value instanceof double[]) {
            double[] source = (double[]) value;
            ByteBuffer out = ByteBuffer.allocate(source.length * Double.BYTES).order(ByteOrder.nativeOrder());
            for (double v : source) {
                out.putDouble(v);
            }
            return out.array();
        }
        throw new IllegalArgumentException("unsupported kernel argument type " + value.getClass().getName());
    }
}
