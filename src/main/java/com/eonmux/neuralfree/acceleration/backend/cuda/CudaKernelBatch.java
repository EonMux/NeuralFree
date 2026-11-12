package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelTaskBatch;
import com.eonmux.neuralfree.acceleration.backend.LaunchGeometry;
import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.driver.CUfunction;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;

/**
 * The CUDA {@link KernelTaskBatch}: independent small kernel launches recorded as tasks and
 * submitted as one device-side submission, the Java-side form of a multi-task kernel. The tasks
 * become the kernel nodes of one captured {@link CudaGraph}, chained in add order so every task
 * starts only after its predecessor finished, and one graph launch runs the whole chain.
 *
 * <p>The graph of the previous launch is kept. When a launch has its shape (the same task count,
 * the same kernels and the same grid and block per task) the graph is replayed: its nodes' arguments
 * are updated from the tasks and it is launched again ({@link #graphReuses()}). Otherwise the graph
 * is captured again. A replay the graph refuses as a mismatch also falls back to a new capture.</p>
 *
 * <p>Every task dispatches through a private clone of its kernel handle, so recording a task never
 * changes the sticky arguments of the handle it was added with, and two tasks on one kernel keep
 * separate arguments. The clones belong to the current graph and are closed with it. Not
 * thread-safe: add, launch, clear and close come from one thread.</p>
 */
public final class CudaKernelBatch implements KernelTaskBatch {

    private static final Logger LOGGER = LoggingUtil.getLogger(CudaKernelBatch.class);

    private final CudaComputeBackend backend;
    private final List<Task> tasks = new ArrayList<>();
    private final List<CudaKernelHandle> handles = new ArrayList<>();
    /** Per captured task: the function and the grid and block dimensions the current graph holds. */
    private CUfunction[] shapeFunctions;
    private long[] shapeDims;
    private CudaGraph graph;
    private CudaComputeQueue graphQueue;
    private long launches;
    private long tasksLaunched;
    private long graphReuses;
    private boolean closed;

    /**
     * @throws IllegalArgumentException when {@code backend} is null
     */
    CudaKernelBatch(CudaComputeBackend backend) {
        if (backend == null) {
            throw new IllegalArgumentException("CUDA backend must not be null");
        }
        this.backend = backend;
    }

    /**
     * Records one kernel launch as the next task of the batch. The full argument list is required
     * here and checked against the kernel's signature now, so a wrong task is reported when it is
     * recorded and not at launch.
     *
     * @throws IllegalArgumentException when the kernel is not a CudaKernelHandle of this backend,
     *                                  an argument or work size is invalid, or a buffer argument
     *                                  is not an open CUDA buffer of this backend
     * @throws IllegalStateException    when the batch is closed, the backend is closed, or the
     *                                  kernel's clone handle was closed
     */
    @Override
    public void add(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args) {
        if (closed) {
            throw new IllegalStateException("kernel batch was already closed");
        }
        if (!(kernel instanceof CudaKernelHandle)) {
            throw new IllegalArgumentException("kernel must be a CudaKernelHandle, got "
                    + (kernel == null ? "null" : kernel.getClass().getName()));
        }
        CudaKernelHandle handle = (CudaKernelHandle) kernel;
        if (handle.backend() != backend) {
            throw new IllegalArgumentException("CUDA kernel " + handle.kernelName() + " belongs to another backend");
        }
        if (handle.isClosed()) {
            throw new IllegalStateException("CUDA kernel " + handle.kernelName() + " was used after it was closed");
        }
        LaunchGeometry geometry = CudaComputeQueue.gridBlockOf(handle, globalWorkSize, localWorkSize);
        Object[] snapshot = copyArgs(args == null ? new Object[0] : args);
        CudaArgState validation = new CudaArgState(backend, handle.kernelName(), handle.signature());
        validation.setAll(snapshot);
        validation.kernelParams();
        tasks.add(new Task(handle, globalWorkSize.clone(), localWorkSize == null ? null : localWorkSize.clone(),
                geometry.grid(), geometry.block(), snapshot));
    }

    @Override
    public int taskCount() {
        return tasks.size();
    }

    @Override
    public long launches() {
        return launches;
    }

    @Override
    public long tasksLaunched() {
        return tasksLaunched;
    }

    @Override
    public long graphReuses() {
        return graphReuses;
    }

    /**
     * Submits every recorded task as one graph launch on {@code queue}, chained in add order, and
     * clears the batch. A launch whose shape matches the previous one replays the captured graph,
     * updating its kernel nodes' arguments; any other launch captures a new graph.
     *
     * @throws IllegalArgumentException when the batch has no tasks or the queue is not a CUDA
     *                                  queue of this backend
     * @throws IllegalStateException    when the batch or its backend is closed
     * @throws RuntimeException         the translated driver failure; the tasks stay recorded
     */
    @Override
    public GpuEvent launch(ComputeQueue queue, GpuEvent... waitOnEvents) {
        if (closed) {
            throw new IllegalStateException("kernel batch was already closed");
        }
        if (tasks.isEmpty()) {
            throw new IllegalArgumentException("cannot launch a kernel batch without tasks");
        }
        CudaComputeQueue target = CudaComputeQueue.launchQueueOf(backend, queue);
        CudaGpuEvent[] waits = target.validatedWaits(waitOnEvents);
        GpuEvent result;
        if (graph != null && graphQueue == target && matchesShape()) {
            target.applyWaits(waits);
            try {
                graph.replay(replayBody());
                graphReuses++;
            } catch (GpuGraphMismatchException e) {
                LOGGER.debug("CUDA kernel batch: replay refused the tasks ({}); capturing again", e.getMessage());
                captureGraph(target);
                result = graph.launch(target, waits);
                finishLaunch();
                return result;
            }
            result = target.enqueueMarker();
        } else {
            captureGraph(target);
            result = graph.launch(target, waits);
        }
        finishLaunch();
        return result;
    }

    /** Counts the submission and clears the tasks the graph now owns. */
    private void finishLaunch() {
        launches++;
        tasksLaunched += tasks.size();
        tasks.clear();
    }

    /** Captures the recorded tasks into a fresh graph on {@code target}, closing the previous one. */
    private void captureGraph(CudaComputeQueue target) {
        closeGraph();
        graphQueue = target;
        for (Task task : tasks) {
            handles.add((CudaKernelHandle) backend.cloneKernel(task.handle));
        }
        target.beginCapture();
        try {
            replayBody().run();
        } catch (RuntimeException e) {
            try {
                target.abortCapture();
            } catch (RuntimeException abort) {
                e.addSuppressed(abort);
            }
            closeHandles();
            graphQueue = null;
            throw e;
        }
        GpuGraph captured;
        try {
            captured = target.endCapture();
        } catch (RuntimeException e) {
            closeHandles();
            graphQueue = null;
            throw e;
        }
        if (!(captured instanceof CudaGraph)) {
            closeHandles();
            graphQueue = null;
            throw new IllegalStateException("the CUDA queue returned no graph for the batch's "
                    + tasks.size() + " tasks");
        }
        graph = (CudaGraph) captured;
        rememberShape();
    }

    /** The body that records or replays the tasks through their private clone handles. */
    private Runnable replayBody() {
        return () -> {
            for (int i = 0; i < tasks.size(); i++) {
                Task task = tasks.get(i);
                graphQueue.enqueueKernelNoEvent(handles.get(i), task.global, task.local, task.args);
            }
        };
    }

    /** True when the recorded tasks have the captured graph's task count, kernels and dimensions. */
    private boolean matchesShape() {
        int n = tasks.size();
        if (shapeFunctions == null || shapeFunctions.length != n) {
            return false;
        }
        for (int i = 0; i < n; i++) {
            Task task = tasks.get(i);
            if (task.handle.function() != shapeFunctions[i]) {
                return false;
            }
            int base = 6 * i;
            if (task.grid[0] != shapeDims[base] || task.grid[1] != shapeDims[base + 1]
                    || task.grid[2] != shapeDims[base + 2] || task.block[0] != shapeDims[base + 3]
                    || task.block[1] != shapeDims[base + 4] || task.block[2] != shapeDims[base + 5]) {
                return false;
            }
        }
        return true;
    }

    /** Stores the task count, functions and dimensions the current graph holds. */
    private void rememberShape() {
        int n = tasks.size();
        shapeFunctions = new CUfunction[n];
        shapeDims = new long[6 * n];
        for (int i = 0; i < n; i++) {
            Task task = tasks.get(i);
            shapeFunctions[i] = task.handle.function();
            int base = 6 * i;
            shapeDims[base] = task.grid[0];
            shapeDims[base + 1] = task.grid[1];
            shapeDims[base + 2] = task.grid[2];
            shapeDims[base + 3] = task.block[0];
            shapeDims[base + 4] = task.block[1];
            shapeDims[base + 5] = task.block[2];
        }
    }

    @Override
    public void clear() {
        tasks.clear();
    }

    /** True once {@link #close()} has run. */
    @Override
    public boolean isClosed() {
        return closed;
    }

    /** Closes the current graph and its private kernel handles. */
    private void closeGraph() {
        if (graph != null) {
            try {
                graph.close();
            } finally {
                graph = null;
            }
        }
        graphQueue = null;
        shapeFunctions = null;
        shapeDims = null;
        closeHandles();
    }

    private void closeHandles() {
        for (CudaKernelHandle handle : handles) {
            handle.close();
        }
        handles.clear();
    }

    /**
     * Closes the batch: releases the captured graph and its private kernel handles. Tasks still
     * recorded are discarded. Idempotent.
     *
     * @throws RuntimeException the first translated driver failure
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        tasks.clear();
        if (graph != null || !handles.isEmpty()) {
            backend.enter();
            closeGraph();
        }
    }

    @Override
    public String toString() {
        return "CudaKernelBatch{tasks=" + tasks.size() + ", launches=" + launches + ", tasksLaunched="
                + tasksLaunched + ", graphReuses=" + graphReuses + (closed ? ", closed" : "") + "}";
    }

    /** One recorded kernel launch: its handle, launch sizes, computed geometry and owned arguments. */
    private static final class Task {
        private final CudaKernelHandle handle;
        private final long[] global;
        private final long[] local;
        private final long[] grid;
        private final long[] block;
        private final Object[] args;

        Task(CudaKernelHandle handle, long[] global, long[] local, long[] grid, long[] block, Object[] args) {
            this.handle = handle;
            this.global = global;
            this.local = local;
            this.grid = grid;
            this.block = block;
            this.args = args;
        }
    }

    /** An owned copy of one task's arguments; holder arrays are cloned, buffers kept by identity. */
    private static Object[] copyArgs(Object[] args) {
        Object[] out = new Object[args.length];
        for (int i = 0; i < args.length; i++) {
            Object v = args[i];
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
}
