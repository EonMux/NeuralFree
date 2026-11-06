package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchPlan;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchSink;
import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
import com.eonmux.neuralfree.acceleration.backend.StickyArgs;
import com.eonmux.neuralfree.acceleration.backend.WorkSizes;
import com.eonmux.neuralfree.acceleration.hip.HipDriver;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A captured sequence of HIP commands: kernel dispatches, device copies and fills. The commands are
 * recorded with {@code hipStreamBeginCapture}, chained by the stream's order, and instantiated once
 * as an executable graph ({@code hipGraphInstantiate}); every replay submits the whole sequence
 * with one {@code hipGraphLaunch}, the HIP counterpart of the CUDA graph.
 *
 * <p>A replay runs the step body with the queue's commands routed here: each command is matched
 * against the next captured node and, when its arguments or geometry changed, the node's parameters
 * are updated through {@code hipGraphExecKernelNodeSetParams} (or the copy and fill counterparts).
 * A command that does not match the next node throws {@link GpuGraphMismatchException} before any
 * device work, so the caller captures the step again. The replayed grid and block come from the
 * body's dispatches; the kernel of a node never changes. The captured nodes are read from the graph
 * in dependency order after the capture ends.</p>
 *
 * <p>{@link #replayPatched(GpuGraphPatchPlan, int)} skips the body: the plan hands each captured
 * dispatch's step values over, and the changed nodes' parameters are rewritten through
 * {@code hipGraphExecKernelNodeSetParams} before the one {@code hipGraphLaunch}.</p>
 *
 * <p>{@link #launch} submits the capture-time commands on any open queue of the backend. Buffers
 * referenced by the nodes must stay open until {@link #close()}, which drains the capturing stream
 * and destroys both graphs.</p>
 */
final class HipGraph implements GpuGraph {

    /** The kind of command a captured node holds. */
    enum Kind {
        KERNEL,
        COPY,
        FILL
    }

    /** One captured command: its graph node and the geometry or buffers it addresses. */
    static final class Node {
        final Kind kind;
        /** The dispatch's position among the capture's kernel nodes; other kinds take {@code -1}. */
        final int ordinal;
        final HipKernelHandle handle;
        final Object[] args;
        /** The dispatch's global work size in work items, as the body enqueued it. */
        long[] global;
        long[] block;
        long[] grid;
        long argVersion;

        HipGpuBuffer src;
        HipGpuBuffer dst;
        long srcOffset;
        long dstOffset;
        long size;
        int pattern;

        long graphNode;

        private Node(Kind kind, int ordinal, HipKernelHandle handle, long[] global, long[] grid, long[] block,
                     Object[] args, long argVersion, HipGpuBuffer src, long srcOffset, HipGpuBuffer dst,
                     long dstOffset, long size, int pattern) {
            this.kind = kind;
            this.ordinal = ordinal;
            this.handle = handle;
            this.global = global;
            this.grid = grid;
            this.block = block;
            this.args = args;
            this.argVersion = argVersion;
            this.src = src;
            this.srcOffset = srcOffset;
            this.dst = dst;
            this.dstOffset = dstOffset;
            this.size = size;
            this.pattern = pattern;
        }

        static Node kernel(HipKernelHandle handle, int ordinal, long[] global, long[] grid, long[] block,
                           Object[] args, long argVersion) {
            return new Node(Kind.KERNEL, ordinal, handle, global, grid, block, args, argVersion, null, 0, null, 0,
                    0, 0);
        }

        static Node copy(HipGpuBuffer src, long srcOffset, HipGpuBuffer dst, long dstOffset, long size) {
            return new Node(Kind.COPY, -1, null, null, null, null, null, 0, src, srcOffset, dst, dstOffset,
                    size, 0);
        }

        static Node fill(HipGpuBuffer dst, long offset, long size, int pattern) {
            return new Node(Kind.FILL, -1, null, null, null, null, null, 0, null, 0, dst, offset, size, pattern);
        }
    }

    private final HipComputeQueue queue;
    private final HipDriver driver;
    private final long graph;
    private final long exec;
    private final List<Node> nodes;
    /** The capture's kernel nodes by dispatch position, built once for the patched replay. */
    private final Node[] kernelNodes;
    /** Kernel-dispatch positions whose node the current body replay updated so far. */
    private final List<Integer> replayChanges = new ArrayList<>();
    private int position;
    private boolean closed;

    HipGraph(HipComputeQueue queue, HipDriver driver, long graph, long exec, List<Node> nodes) {
        this.queue = queue;
        this.driver = driver;
        this.graph = graph;
        this.exec = exec;
        this.nodes = nodes;
        List<Node> kernels = new ArrayList<>();
        for (Node node : nodes) {
            if (node.kind == Kind.KERNEL) {
                kernels.add(node);
            }
        }
        this.kernelNodes = kernels.toArray(new Node[0]);
    }

    /** The queue that recorded the graph and whose stream the nodes were chained on. */
    HipComputeQueue owner() {
        return queue;
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
        replayChanges.clear();
        boolean launched = false;
        try {
            body.run();
            if (position != nodes.size()) {
                throw new GpuGraphMismatchException("command replay of " + this + " enqueues " + position
                        + " commands, the capture holds " + nodes.size());
            }
            queue.launchGraphOnOwner(this);
            launched = true;
        } finally {
            queue.endGraphReplay(this, launched);
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
        if (index < 0 || index >= kernelNodes.length) {
            throw new IllegalArgumentException(this + " holds " + kernelNodes.length + " kernel dispatches,"
                    + " no dispatch " + index);
        }
        return dispatchOf(kernelNodes[index]);
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
        return true;
    }

    @Override
    public boolean replayPatched(GpuGraphPatchPlan plan, int pos) {
        if (plan == null) {
            throw new IllegalArgumentException("patch plan must not be null");
        }
        requireOpen("patch and replay");
        queue.beginGraphReplay(this);
        boolean launched = false;
        try {
            boolean patched;
            try {
                patched = plan.patchStep(pos, this::patchKernel);
            } catch (GpuGraphMismatchException e) {
                throw e;
            } catch (RuntimeException e) {
                throw new GpuGraphMismatchException("patch replay of " + this + " failed: " + e.getMessage(), e);
            }
            if (!patched) {
                return false;
            }
            queue.launchGraphOnOwner(this);
            launched = true;
        } finally {
            queue.endGraphReplay(this, launched);
        }
        return true;
    }

    /** The current state of one captured kernel node. */
    private GpuGraphDispatch dispatchOf(Node node) {
        return new GpuGraphDispatch(node.ordinal, node.handle.kernelName(), node.global, node.block, node.args);
    }

    /**
     * One patch of the plan: stores the changed step values in the handle's argument state, refreshes
     * the node's record and rewrites the node's parameters from the state, so the next launch of the
     * executable graph runs the dispatch with this step's arguments. The work size is the global work
     * size in work items, as {@link GpuGraphDispatch#grid()} reports it; the node's blocks are
     * recomputed from it over the captured block. A patch equal to what the node holds writes nothing.
     */
    private void patchKernel(int recordIndex, long[] global, int[] argIndices, Object[] values) {
        if (recordIndex < 0 || recordIndex >= kernelNodes.length) {
            throw new IllegalArgumentException(this + " holds " + kernelNodes.length + " kernel dispatches,"
                    + " no dispatch " + recordIndex);
        }
        Node node = kernelNodes[recordIndex];
        HipArgState args = node.handle.args();
        boolean changed = global != null && !sameGlobal(node.global, global);
        for (int i = 0; i < argIndices.length; i++) {
            if (!KernelArgs.sameValue(node.args[argIndices[i]], values[i])) {
                changed = true;
            }
        }
        if (!changed) {
            return;
        }
        for (int i = 0; i < argIndices.length; i++) {
            int arg = argIndices[i];
            Object value = values[i];
            if (!KernelArgs.sameValue(node.args[arg], value)) {
                args.set(arg, value);
                node.args[arg] = KernelArgs.snapshot(value, node.args[arg]);
            }
        }
        if (global != null && !sameGlobal(node.global, global)) {
            node.global = global.clone();
            node.grid = WorkSizes.groupCounts(WorkSizes.pad3(global, 1), node.block);
        }
        Object[] params = args.launchParams();
        driver.graphExecKernelNodeSetParams(exec, node.graphNode, node.handle.function(),
                (int) node.grid[0], dim(node.grid, 1), dim(node.grid, 2),
                (int) node.block[0], dim(node.block, 1), dim(node.block, 2), params);
        node.argVersion = args.sticky().version();
    }

    /** True when two global work sizes launch the same grid, so trailing dimensions of 1 do not matter. */
    private static boolean sameGlobal(long[] a, long[] b) {
        return Arrays.equals(WorkSizes.pad3(a, 1), WorkSizes.pad3(b, 1));
    }

    /** The grid or block extent of dimension {@code i}, 1 beyond the recorded dimensions. */
    private static int dim(long[] sizes, int i) {
        return (int) (sizes.length > i ? sizes[i] : 1L);
    }

    /** Throws when the graph is closed; {@code what} names the refused operation. */
    private void requireOpen(String what) {
        if (closed) {
            throw new IllegalStateException("command graph cannot " + what + ": " + this + " is closed");
        }
    }

    /**
     * One kernel dispatch of the replay body: matches it against the next captured node and updates
     * the node's parameters when the body's arguments or work size differ from what the node holds.
     *
     * @param global the dispatch's global work size in work items, as the body enqueued it
     * @param grid   the launch's blocks, {@code groupCounts} of {@code global} over {@code block}
     * @throws IllegalStateException     when the graph is closed or a bound buffer was closed
     * @throws GpuGraphMismatchException when the body's command sequence leaves the capture
     */
    void dispatch(HipKernelHandle handle, long[] global, long[] grid, long[] block, String name) {
        Node node = next(Kind.KERNEL, name);
        if (node.handle != handle) {
            throw new GpuGraphMismatchException("command replay of " + this + ": command " + position
                    + " is " + name + " but the capture holds " + node.handle.kernelName() + " at this point");
        }
        if (!Arrays.equals(node.block, block)) {
            throw new GpuGraphMismatchException("command replay of " + this + ": " + name + " changed its"
                    + " block size from " + Arrays.toString(node.block) + " to " + Arrays.toString(block));
        }
        StickyArgs sticky = handle.args().sticky();
        for (int i = 0; i < sticky.count(); i++) {
            if (!sticky.isSet(i)) {
                continue;
            }
            Object arg = sticky.get(i);
            if (arg instanceof HipGpuBuffer && ((HipGpuBuffer) arg).isClosed()) {
                throw new IllegalStateException(name + ": bound buffer " + arg + " is closed while its command"
                        + " graph is open; close the graph first");
            }
        }
        if (argsChanged(node, sticky, global)) {
            Object[] params = handle.args().launchParams();
            driver.graphExecKernelNodeSetParams(exec, node.graphNode, handle.function(),
                    (int) grid[0], (int) grid[1], (int) grid[2],
                    (int) block[0], (int) block[1], (int) block[2], params);
            node.argVersion = sticky.version();
            node.global = global.clone();
            node.grid = grid.clone();
            refreshArgs(node, sticky);
            replayChanges.add(node.ordinal);
        }
    }

    /** Refreshes a node's argument record with the values its parameters were just updated to. */
    private static void refreshArgs(Node node, StickyArgs sticky) {
        for (int i = 0; i < node.args.length && i < sticky.count(); i++) {
            if (sticky.isSet(i)) {
                node.args[i] = KernelArgs.snapshot(sticky.get(i), null);
            }
        }
    }

    /**
     * True when this node's work size or argument values differ from the handle's current state. A kernel
     * handle is shared by every dispatch site of the kernel, so a moved version only means another site
     * launched since; the node changes when its own values do.
     */
    private static boolean argsChanged(Node node, StickyArgs sticky, long[] global) {
        if (sameGlobal(node.global, global)) {
            if (sticky.version() == node.argVersion) {
                return false;
            }
            for (int i = 0; i < sticky.count() && i < node.args.length; i++) {
                if (sticky.isSet(i) && !KernelArgs.sameValue(node.args[i], sticky.get(i))) {
                    return true;
                }
            }
            return false;
        }
        return true;
    }

    /**
     * One device copy of the replay body: matches it against the next captured node and updates the
     * node's buffers and range when they changed since the capture.
     */
    void copy(HipGpuBuffer src, long srcOffset, HipGpuBuffer dst, long dstOffset, long size) {
        Node node = next(Kind.COPY, "a device copy");
        long from = devicePointer(src, "a replayed device copy's source");
        long to = devicePointer(dst, "a replayed device copy's destination");
        devicePointer(node.src, "a captured device copy's source");
        devicePointer(node.dst, "a captured device copy's destination");
        if (node.src != src || node.dst != dst || node.srcOffset != srcOffset || node.dstOffset != dstOffset
                || node.size != size) {
            driver.graphExecMemcpyNodeSetParams(exec, node.graphNode, to + dstOffset, from + srcOffset, size);
            node.src = src;
            node.dst = dst;
            node.srcOffset = srcOffset;
            node.dstOffset = dstOffset;
            node.size = size;
        }
    }

    /** One device fill of the replay body: matched against the next captured node and updated. */
    void fill(HipGpuBuffer dst, long offset, long size, int pattern) {
        Node node = next(Kind.FILL, "a device fill");
        long target = devicePointer(dst, "a replayed device fill's destination");
        devicePointer(node.dst, "a captured device fill's destination");
        if (node.dst != dst || node.dstOffset != offset || node.size != size || node.pattern != pattern) {
            driver.graphExecMemsetNodeSetParams(exec, node.graphNode, target + offset, pattern, size / 4);
            node.dst = dst;
            node.dstOffset = offset;
            node.size = size;
            node.pattern = pattern;
        }
    }

    /** The next captured node, which must have the same kind as the body's command. */
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

    /** The executable graph's launch on the queue's stream, after the replay body finished. */
    void launchOn(HipComputeQueue launchQueue) {
        driver.graphLaunch(exec, launchQueue.getHipStream());
    }

    /** Waits for the stream, then destroys both graphs. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        if (queue.isClosed()) {
            // the queue's close destroyed the executable graph with the rest of its resources
            closed = true;
            return;
        }
        queue.releaseGraph(this);
    }

    /** Destroys the executable graph and its source graph; the queue's stream has drained. */
    void destroy() {
        closed = true;
        RuntimeException failure = null;
        try {
            driver.graphExecDestroy(exec);
        } catch (RuntimeException e) {
            failure = e;
        }
        try {
            driver.graphDestroy(graph);
        } catch (RuntimeException e) {
            failure = Closeables.chain(failure, e);
        }
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public boolean isClosed() {
        return closed;
    }

    @Override
    public String toString() {
        return "HipGraph{" + nodes.size() + " commands, " + dispatchCount() + " dispatches"
                + (closed ? ", closed" : "") + "}";
    }

    private static long devicePointer(HipGpuBuffer buffer, String role) {
        Long pointer = buffer.getDevicePointer();
        if (pointer == null) {
            throw new IllegalStateException(role + " is closed while its command graph is open; close the graph"
                    + " first");
        }
        return pointer;
    }
}
