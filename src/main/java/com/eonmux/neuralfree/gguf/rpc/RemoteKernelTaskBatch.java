package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelTaskBatch;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A batch of kernel launches one remote backend submits as one {@link RpcWire#OP_BATCH} flush: the launcher
 * records the dispatches it would launch one by one and {@link #launch} appends them to the endpoint's pipelined
 * batch in add order, so a decode step of several small kernels costs one round trip (the async dispatcher of
 * llama.cpp's RPC transport). The batch carries its own snapshot of every task's arguments — recording never
 * changes the sticky arguments of the kernel handles — and the server runs the tasks in order, with the ordering
 * of the queue's in-order execution between them. The protocol keeps no device-side batch structure, so no launch
 * reuses a graph and {@link #graphReuses()} stays 0.
 */
final class RemoteKernelTaskBatch implements KernelTaskBatch {

    /** One recorded dispatch: the handle, its work sizes and a private copy of the arguments. */
    private static final class Task {
        final RemoteKernelHandle kernel;
        final long[] globalWorkSize;
        final long[] localWorkSize;
        final Object[] args;

        Task(RemoteKernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args) {
            this.kernel = kernel;
            this.globalWorkSize = globalWorkSize;
            this.localWorkSize = localWorkSize;
            this.args = args;
        }
    }

    private final RemoteComputeBackend backend;
    private final List<Task> tasks = new ArrayList<>();
    private long launches;
    private long tasksLaunched;
    private boolean closed;

    RemoteKernelTaskBatch(RemoteComputeBackend backend) {
        this.backend = Objects.requireNonNull(backend, "backend");
    }

    @Override
    public void add(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args) {
        if (closed) {
            throw new IllegalStateException("the kernel batch is closed");
        }
        if (backend.closed()) {
            throw new IllegalStateException("the batch's remote backend is closed");
        }
        if (!(kernel instanceof RemoteKernelHandle) || !((RemoteKernelHandle) kernel).belongsTo(backend)) {
            throw new IllegalArgumentException("kernel is not a handle of this remote backend");
        }
        RemoteComputeQueue.checkWorkSizes(globalWorkSize, localWorkSize);
        Objects.requireNonNull(args, "args");
        for (Object arg : args) {
            if (arg == null) {
                throw new IllegalArgumentException(((RemoteKernelHandle) kernel).kernelName()
                        + ": kernel argument is null");
            }
        }
        // snapshot semantics per value: later mutation of a holder array changes nothing
        Object[] snapshots = new Object[args.length];
        for (int i = 0; i < args.length; i++) {
            snapshots[i] = KernelArgs.snapshot(args[i], null);
        }
        tasks.add(new Task((RemoteKernelHandle) kernel, globalWorkSize.clone(),
                localWorkSize == null ? null : localWorkSize.clone(), snapshots));
    }

    @Override
    public int taskCount() {
        return tasks.size();
    }

    @Override
    public GpuEvent launch(ComputeQueue queue, GpuEvent... waitOnEvents) {
        if (closed) {
            throw new IllegalStateException("the kernel batch is closed");
        }
        if (!(queue instanceof RemoteComputeQueue) || ((RemoteComputeQueue) queue).backend() != backend) {
            throw new IllegalArgumentException("queue is not a queue of this remote backend");
        }
        if (tasks.isEmpty()) {
            throw new IllegalArgumentException("launch of an empty kernel batch");
        }
        await(waitOnEvents);
        long seq = 0;
        for (Task task : tasks) {
            RpcWire.Writer member = new RpcWire.Writer(RpcWire.OP_ENQUEUE)
                    .i64(task.kernel.id()).longArray(task.globalWorkSize).longArray(task.localWorkSize);
            RpcWire.writeArgs(member, task.args);
            seq = backend.submit(member, true);
        }
        launches++;
        tasksLaunched += tasks.size();
        tasks.clear();
        return new RemoteGpuEvent(backend.batch(), seq);
    }

    @Override
    public void clear() {
        tasks.clear();
    }

    @Override
    public long launches() {
        return launches;
    }

    @Override
    public long tasksLaunched() {
        return tasksLaunched;
    }

    /** The protocol keeps no device-side batch structure, so no launch reuses a graph. */
    @Override
    public long graphReuses() {
        return 0;
    }

    @Override
    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        tasks.clear();
    }

    /** Awaits the events a launch was ordered behind, through the endpoint's lazy completion. */
    private static void await(GpuEvent[] events) {
        if (events == null || events.length == 0) {
            return;
        }
        for (GpuEvent event : events) {
            if (event == null) {
                continue;
            }
            if (!(event instanceof RemoteGpuEvent)) {
                throw new IllegalArgumentException("a remote batch cannot wait on an event of another backend");
            }
            try {
                event.awaitCompletion();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while waiting for a remote event", e);
            }
        }
    }

    @Override
    public String toString() {
        return "RemoteKernelTaskBatch{" + backend + ", " + tasks.size() + " task(s)}";
    }
}
