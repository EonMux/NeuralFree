package com.eonmux.neuralfree.acceleration.backend.cuda;

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
import com.eonmux.neuralfree.acceleration.backend.WorkSizes;
import jcuda.Pointer;
import jcuda.driver.CUDA_MEMCPY3D;
import jcuda.driver.CUDA_MEMSET_NODE_PARAMS;
import jcuda.driver.CUDA_KERNEL_NODE_PARAMS;
import jcuda.driver.CUcontext;
import jcuda.driver.CUfunction;
import jcuda.driver.CUgraph;
import jcuda.driver.CUgraphExec;
import jcuda.driver.CUgraphNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static jcuda.driver.CUmemorytype.CU_MEMORYTYPE_DEVICE;
import static jcuda.driver.JCudaDriver.cuGraphDestroy;
import static jcuda.driver.JCudaDriver.cuGraphExecDestroy;
import static jcuda.driver.JCudaDriver.cuGraphExecKernelNodeSetParams;
import static jcuda.driver.JCudaDriver.cuGraphExecMemcpyNodeSetParams;
import static jcuda.driver.JCudaDriver.cuGraphExecMemsetNodeSetParams;

/**
 * A captured sequence of CUDA commands: kernel dispatches and, on this backend, device copies and
 * fills. The nodes form a chain in capture order, the dependency order of the in-order stream they
 * were recorded on. The graph is replayed as one {@code cuGraphLaunch} of an instantiated executable
 * graph (llama.cpp {@code USE_CUDA_GRAPH}, vLLM's captured decode step).
 *
 * <p>A replay runs the step body with the queue's commands routed here: each command is matched
 * against the next captured node and, when its arguments or geometry changed since the node was last
 * written, the node's parameters are updated through {@code cuGraphExecKernelNodeSetParams} (or the
 * copy and fill counterparts); the body's work then reaches the device as one graph launch instead of
 * one launch per command. A command that does not match the next node (another kernel, another order,
 * more commands than captured) throws {@link GpuGraphMismatchException} before any device work, so the
 * caller can capture the step again. The replayed grid comes from the body's dispatches, so a body
 * that changes a launch's work size is picked up; the kernel of a node never changes. The source graph
 * stays alive with the executable graph, because node updates address its nodes.
 *
 * <p>{@link #replayPatched(GpuGraphPatchPlan, int)} skips the body: the plan hands each captured
 * dispatch's step values to {@code patchKernel}, which writes the changed values into the nodes'
 * parameter holders and rewrites only the changed nodes through {@code cuGraphExecKernelNodeSetParams}
 * before the one {@code cuGraphLaunch}. A node whose step values equal what the launched instantiation's
 * parameters hold is left untouched: every node update writes one of the two executable graphs, so a
 * node remembers the instantiation that lags it and the next patch of that instantiation rewrites it.
 *
 * <p>{@link #launch} submits the capture-time command sequence on any open queue of the backend.
 * Buffers referenced by the nodes must stay open until {@link #close()}, which drains the capturing
 * queue and destroys both graphs.
 */
final class CudaGraph implements GpuGraph {

    /** The kind of command a captured node holds. */
    enum Kind {
        KERNEL,
        COPY,
        FILL
    }

    /** One captured command: its graph node and the geometry or buffers it addresses. */
    static final class Node {
        final Kind kind;
        final CUgraphNode node;

        /** The dispatch's position among the capture's kernel nodes; other kinds take {@code -1}. */
        final int ordinal;
        final CudaKernelHandle handle;
        /** The dispatch's global work size in work items, as the body enqueued it. */
        long[] global;
        /** The launch's blocks, {@code groupCounts} of the global work size over the block. */
        final long[] block;
        long[] grid;
        /** The argument values the node's parameters last held; a node update keeps it current. */
        final Object[] args;
        long argVersion;
        /**
         * The executable graph whose parameters for this node lag {@link #args} and the work size (a node
         * update writes one of the two instantiations), or null when both hold them.
         */
        CUgraphExec lagging;

        final CudaBuffer src;
        final CudaBuffer dst;
        final long srcOffset;
        final long dstOffset;
        final long size;
        final int pattern;

        private Node(Kind kind, CUgraphNode node, int ordinal, CudaKernelHandle handle, long[] global, long[] grid,
                     long[] block, Object[] args, long argVersion, CudaBuffer src, long srcOffset, CudaBuffer dst,
                     long dstOffset, long size, int pattern) {
            this.kind = kind;
            this.node = node;
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

        static Node kernel(CUgraphNode node, int ordinal, CudaKernelHandle handle, long[] global, long[] grid,
                           long[] block, Object[] args, long argVersion) {
            return new Node(Kind.KERNEL, node, ordinal, handle, global, grid, block, args, argVersion,
                    null, 0, null, 0, 0, 0);
        }

        static Node copy(CUgraphNode node, CudaBuffer src, long srcOffset, CudaBuffer dst, long dstOffset,
                         long size) {
            return new Node(Kind.COPY, node, -1, null, null, null, null, null, 0, src, srcOffset, dst,
                    dstOffset, size, 0);
        }

        static Node fill(CUgraphNode node, CudaBuffer dst, long offset, long size, int pattern) {
            return new Node(Kind.FILL, node, -1, null, null, null, null, null, 0, null, 0, dst, offset, size,
                    pattern);
        }
    }

    private final CudaComputeQueue queue;
    private final CUgraph graph;
    /** The executable graph the next launch uses; {@link #preparePatched} swaps the two. */
    private CUgraphExec exec;
    /** The idle executable graph a prepared follow-up step patches while {@code exec} runs. */
    private CUgraphExec nextExec;
    private final List<Node> nodes;
    /** The capture's kernel nodes by dispatch position, built once for the patched replay. */
    private final Node[] kernelNodes;
    /** Kernel-dispatch positions whose node the current body replay updated so far. */
    private final List<Integer> replayChanges = new ArrayList<>();
    private int position;
    private boolean closed;

    CudaGraph(CudaComputeQueue queue, CUgraph graph, CUgraphExec exec, CUgraphExec nextExec, List<Node> nodes) {
        this.queue = queue;
        this.graph = graph;
        this.exec = exec;
        this.nextExec = nextExec;
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
    CudaComputeQueue owner() {
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
                patched = plan.patchStep(pos, (recordIndex, global, argIndices, values) ->
                        patchKernel(exec, recordIndex, global, argIndices, values));
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

    /**
     * Patches the idle executable graph for the follow-up step {@code nextPos} while the last launched
     * one runs, so the next patched replay launches a graph whose parameters already hold the step's
     * values and only its changed nodes need updating. The patch step is authoritative as always: a
     * follow-up step at another position than the prepared one simply rewrites what moved. On a
     * declined pattern the swap does not happen and the last launched step's patch restores the active
     * graph's records, so the caller's next replay sees a consistent state either way. The prepared
     * graph and the running one are distinct instantiations, so the driver's requirement that an
     * updated executable graph not be running is met.
     */
    @Override
    public boolean preparePatched(GpuGraphPatchPlan plan, int nextPos, int lastPos) {
        if (plan == null) {
            throw new IllegalArgumentException("patch plan must not be null");
        }
        requireOpen("prepare a patched replay");
        boolean patched;
        try {
            patched = plan.patchStep(nextPos, (recordIndex, global, argIndices, values) ->
                    patchKernel(nextExec, recordIndex, global, argIndices, values));
        } catch (GpuGraphMismatchException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new GpuGraphMismatchException("patch prepare of " + this + " failed: " + e.getMessage(), e);
        }
        if (patched) {
            CUgraphExec prepared = exec;
            exec = nextExec;
            nextExec = prepared;
            return true;
        }
        plan.patchStep(lastPos, (recordIndex, global, argIndices, values) ->
                patchKernel(exec, recordIndex, global, argIndices, values));
        return false;
    }

    /** The current state of one captured kernel node. */
    private GpuGraphDispatch dispatchOf(Node node) {
        return new GpuGraphDispatch(node.ordinal, node.handle.kernelName(), node.global, node.block, node.args);
    }

    /**
     * One patch of the plan: binds the changed values through the node's argument state, refreshes the
     * node's record and rewrites the node's parameters on {@code target}, so the next launch of that
     * executable graph runs the dispatch with this step's arguments. The work size is the global work
     * size in work items, as {@link GpuGraphDispatch#grid()} reports it; the node's blocks are
     * recomputed from it over the captured block. A patch equal to what the node holds writes nothing
     * unless {@code target} lags the node's record, which it then catches up.
     */
    private void patchKernel(CUgraphExec target, int recordIndex, long[] global, int[] argIndices,
                             Object[] values) {
        if (recordIndex < 0 || recordIndex >= kernelNodes.length) {
            throw new IllegalArgumentException(this + " holds " + kernelNodes.length + " kernel dispatches,"
                    + " no dispatch " + recordIndex);
        }
        Node node = kernelNodes[recordIndex];
        StickyArgs sticky = node.handle.args().sticky();
        boolean changed = global != null && !sameGlobal(node.global, global);
        for (int i = 0; i < argIndices.length; i++) {
            if (!KernelArgs.sameValue(node.args[argIndices[i]], values[i])) {
                changed = true;
            }
        }
        if (!changed && node.lagging != target) {
            return;
        }
        for (int i = 0; i < argIndices.length; i++) {
            int arg = argIndices[i];
            Object value = values[i];
            if (KernelArgs.sameValue(node.args[arg], value)) {
                continue;
            }
            patchArg(node.handle.args(), node.args[arg], arg, value);
            node.args[arg] = value instanceof GpuBuffer ? value : KernelArgs.snapshot(value, node.args[arg]);
        }
        if (global != null && !sameGlobal(node.global, global)) {
            node.global = global.clone();
            node.grid = WorkSizes.groupCounts(WorkSizes.pad3(global, 1), node.block);
        }
        Pointer params = node.handle.args().kernelParams();
        CUDA_KERNEL_NODE_PARAMS nodeParams = nodeParams(node.handle.function(), node.grid, node.block, params);
        CudaErrors.check("cuGraphExecKernelNodeSetParams(" + node.handle.kernelName() + ")",
                () -> cuGraphExecKernelNodeSetParams(target, node.node, nodeParams));
        node.argVersion = sticky.version();
        node.lagging = changed ? (target == exec ? nextExec : exec) : null;
    }

    /** True when two global work sizes launch the same grid, so trailing dimensions of 1 do not matter. */
    private static boolean sameGlobal(long[] a, long[] b) {
        return Arrays.equals(WorkSizes.pad3(a, 1), WorkSizes.pad3(b, 1));
    }

    /**
     * Binds one patched value through the kernel's argument state, so its sticky record and its live
     * parameter holder move together: a later direct launch that sets the argument back to its previous
     * value rewrites the holder instead of finding the record unchanged and launching the patched value.
     * The value must keep the kind the captured argument has.
     */
    private static void patchArg(CudaArgState args, Object captured, int index, Object value) {
        if (captured == null || KernelArgs.kindOf(captured) != KernelArgs.kindOf(value)) {
            throw new IllegalArgumentException("cannot patch arg " + index + " of " + args.sticky().kernelName()
                    + " with " + KernelArgs.describe(value) + ": the value's kind does not match the captured"
                    + " argument");
        }
        args.set(index, value);
    }

    /** Throws when the graph is closed; {@code what} names the refused operation. */
    private void requireOpen(String what) {
        if (closed) {
            throw new IllegalStateException("command graph cannot " + what + ": " + this + " is closed");
        }
    }

    /**
     * One kernel dispatch of the replay body: matches it against the next captured node and updates the
     * node's parameters when the body's arguments or work size differ from what the node holds.
     *
     * @param global the dispatch's global work size in work items, as the body enqueued it
     * @param grid   the launch's blocks, {@code groupCounts} of {@code global} over {@code block}
     * @throws IllegalStateException     when the graph is closed or a bound buffer was closed
     * @throws GpuGraphMismatchException when the body's command sequence leaves the capture
     */
    void dispatch(CudaKernelHandle handle, long[] global, long[] grid, long[] block, String name) {
        Node node = next(Kind.KERNEL, name);
        if (node.handle != handle) {
            throw new GpuGraphMismatchException("command replay of " + this + ": command " + position
                    + " is " + name + " but the capture holds " + node.handle.kernelName() + " at this point");
        }
        if (!Arrays.equals(node.block, block)) {
            throw new GpuGraphMismatchException("command replay of " + this + ": " + name + " changed its"
                    + " work-group size from " + Arrays.toString(node.block) + " to " + Arrays.toString(block));
        }
        StickyArgs sticky = handle.args().sticky();
        for (int i = 0; i < sticky.count(); i++) {
            if (!sticky.isSet(i)) {
                continue;
            }
            Object arg = sticky.get(i);
            if (arg instanceof CudaBuffer && ((CudaBuffer) arg).isClosed()) {
                throw new IllegalStateException(name + ": bound buffer " + arg + " is closed while its command"
                        + " graph is open; close the graph first");
            }
        }
        boolean changed = argsChanged(node, sticky, global);
        if (changed || node.lagging == exec) {
            Pointer params = handle.args().kernelParams();
            CUDA_KERNEL_NODE_PARAMS nodeParams = nodeParams(handle.function(), grid, block, params);
            CudaErrors.check("cuGraphExecKernelNodeSetParams(" + name + ")",
                    () -> cuGraphExecKernelNodeSetParams(exec, node.node, nodeParams));
            node.argVersion = sticky.version();
            node.lagging = changed ? nextExec : null;
        }
        if (changed) {
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
    void copy(CudaBuffer src, long srcOffset, CudaBuffer dst, long dstOffset, long size) {
        Node node = next(Kind.COPY, "a device copy");
        if (node.src.isClosed() || node.dst.isClosed()) {
            throw new IllegalStateException("a captured device copy of " + this + " uses a closed buffer");
        }
        if (node.src != src || node.dst != dst || node.srcOffset != srcOffset || node.dstOffset != dstOffset
                || node.size != size) {
            CUDA_MEMCPY3D params = memcpyParams(src, srcOffset, dst, dstOffset, size);
            CUcontext context = queue.backendOf().context().context();
            CudaErrors.check("cuGraphExecMemcpyNodeSetParams",
                    () -> cuGraphExecMemcpyNodeSetParams(exec, node.node, params, context));
        }
    }

    /** One device fill of the replay body: matches it against the next captured node and updates it. */
    void fill(CudaBuffer dst, long offset, long size, int pattern) {
        Node node = next(Kind.FILL, "a device fill");
        if (node.dst.isClosed()) {
            throw new IllegalStateException("a captured device fill of " + this + " uses a closed buffer");
        }
        if (node.dst != dst || node.dstOffset != offset || node.size != size || node.pattern != pattern) {
            CUDA_MEMSET_NODE_PARAMS params = memsetParams(dst, offset, size, pattern);
            CUcontext context = queue.backendOf().context().context();
            CudaErrors.check("cuGraphExecMemsetNodeSetParams",
                    () -> cuGraphExecMemsetNodeSetParams(exec, node.node, params, context));
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
    void launchOn(CudaComputeQueue launchQueue) {
        launchQueue.graphLaunch(exec, dispatchCount());
    }

    /** The kernel-node parameters of one launch. */
    static CUDA_KERNEL_NODE_PARAMS nodeParams(CUfunction function, long[] grid, long[] block, Pointer params) {
        CUDA_KERNEL_NODE_PARAMS p = new CUDA_KERNEL_NODE_PARAMS();
        p.func = function;
        p.gridDimX = (int) grid[0];
        p.gridDimY = grid.length > 1 ? (int) grid[1] : 1;
        p.gridDimZ = grid.length > 2 ? (int) grid[2] : 1;
        p.blockDimX = (int) block[0];
        p.blockDimY = block.length > 1 ? (int) block[1] : 1;
        p.blockDimZ = block.length > 2 ? (int) block[2] : 1;
        p.sharedMemBytes = 0;
        p.kernelParams = params;
        p.extra = null;
        return p;
    }

    /**
     * The memcpy-node parameters of one device-to-device copy. The source and destination addresses
     * are the buffers' base pointers plus the recorded offsets; the range is one row of bytes.
     */
    static CUDA_MEMCPY3D memcpyParams(CudaBuffer src, long srcOffset, CudaBuffer dst, long dstOffset,
                                      long size) {
        CUDA_MEMCPY3D p = new CUDA_MEMCPY3D();
        p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        p.srcDevice = src.getCUDevicePtr().withByteOffset(srcOffset);
        p.srcXInBytes = 0;
        p.srcPitch = 0;
        p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        p.dstDevice = dst.getCUDevicePtr().withByteOffset(dstOffset);
        p.dstXInBytes = 0;
        p.dstPitch = 0;
        p.WidthInBytes = size;
        p.Height = 1;
        p.Depth = 1;
        return p;
    }

    /** The memset-node parameters of one 32-bit fill: {@code size} is a multiple of four. */
    static CUDA_MEMSET_NODE_PARAMS memsetParams(CudaBuffer dst, long offset, long size, int pattern) {
        CUDA_MEMSET_NODE_PARAMS p = new CUDA_MEMSET_NODE_PARAMS();
        p.dst = dst.getCUDevicePtr().withByteOffset(offset);
        p.pitch = 0;
        p.value = pattern;
        p.elementSize = Integer.BYTES;
        p.width = size / Integer.BYTES;
        p.height = 1;
        return p;
    }

    /** Waits for the last submitted replay on the queue's stream, then destroys both graphs. */
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

    /** Destroys both executable graphs and the source graph; the queue's stream has drained. */
    void destroy() {
        closed = true;
        CudaErrors.check("cuGraphExecDestroy", () -> cuGraphExecDestroy(exec));
        CudaErrors.check("cuGraphExecDestroy", () -> cuGraphExecDestroy(nextExec));
        CudaErrors.check("cuGraphDestroy", () -> cuGraphDestroy(graph));
    }

    /** True once {@link #close()} has run, or the owning queue closed and released the graph. */
    @Override
    public boolean isClosed() {
        return closed;
    }

    @Override
    public String toString() {
        return "CudaGraph{" + nodes.size() + " commands, " + dispatchCount() + " dispatches"
                + (closed ? ", closed" : "") + "}";
    }
}
