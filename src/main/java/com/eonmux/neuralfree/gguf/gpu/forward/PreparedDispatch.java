package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;

import java.util.Arrays;
import java.util.Objects;

/**
 * A kernel launch with its arguments bound once, for the decode fast path: blocks
 * prepare their dispatches at bind time and per step only re-set the arguments that change (positions, token counts)
 * before {@link #dispatch()}.
 *
 * <p>The dispatch owns a clone of the kernel handle ({@code ComputeBackend.cloneKernel}), so its sticky arguments are
 * private: {@code dispatch()} launches them pre-bound and each setter updates one argument on the handle. On a backend
 * without handle clones it shares the library handle and re-sends every argument with each launch. Argument values
 * follow the queue's snapshot semantics: a launch uses the values set before it was enqueued.</p>
 *
 * <p>Created by {@link GpuOps#prepare}. Not thread-safe. {@link #close()} releases the clone; {@code GpuOps.close()}
 * closes every dispatch still open.</p>
 */
public final class PreparedDispatch implements AutoCloseable {

    private final GpuOps owner;
    private final String kernelName;
    private final KernelHandle handle;
    private final boolean ownsHandle;
    private final ComputeQueue queue;
    private final Object[] args;
    private final long[] local;
    private long[] global;
    private boolean closed;

    PreparedDispatch(GpuOps owner, String kernelName, KernelHandle handle, boolean ownsHandle, ComputeQueue queue,
                     long[] global, long[] local, Object[] args) {
        this.owner = owner;
        this.kernelName = kernelName;
        this.handle = handle;
        this.ownsHandle = ownsHandle;
        this.queue = queue;
        this.global = global;
        this.local = local;
        this.args = args;
        if (ownsHandle) {
            for (int i = 0; i < args.length; i++) {
                queue.setKernelArg(handle, i, args[i]);
            }
        }
    }

    /** Enqueues the kernel on the context queue with the current arguments and global size. */
    public void dispatch() {
        dispatchOn(queue);
    }

    /**
     * Enqueues the kernel on another queue of the same backend.
     *
     * @throws IllegalStateException when the dispatch is closed
     */
    public void dispatchOn(ComputeQueue q) {
        Objects.requireNonNull(q, "queue");
        ensureOpen();
        if (ownsHandle) {
            q.enqueueKernelPreBound(handle, global, local);
        } else {
            q.enqueueKernelNoEvent(handle, global, local, args.clone());
        }
    }

    /**
     * Re-sets an {@code int} argument.
     *
     * @throws IllegalArgumentException when the index is out of range or the argument is not an int
     */
    public void setInt(int argIndex, int v) {
        Object current = arg(argIndex);
        if (!(current instanceof Integer || current instanceof Boolean || current instanceof int[])) {
            throw wrongKind(argIndex, "an int", current);
        }
        set(argIndex, v);
    }

    /**
     * Re-sets a {@code float} argument.
     *
     * @throws IllegalArgumentException when the index is out of range or the argument is not a float
     */
    public void setFloat(int argIndex, float v) {
        Object current = arg(argIndex);
        if (!(current instanceof Float || current instanceof float[])) {
            throw wrongKind(argIndex, "a float", current);
        }
        set(argIndex, v);
    }

    /**
     * Re-binds a buffer argument.
     *
     * @throws IllegalArgumentException when the index is out of range or the argument is not a buffer
     */
    public void setBuffer(int argIndex, GpuBuffer b) {
        Objects.requireNonNull(b, "buffer");
        Object current = arg(argIndex);
        if (!(current instanceof GpuBuffer)) {
            throw wrongKind(argIndex, "a buffer", current);
        }
        set(argIndex, b);
    }

    /**
     * Changes the global work size (for example when the token count changes); the local size stays.
     *
     * @throws IllegalArgumentException when the dimensionality differs or a size is not positive
     */
    public void setGlobal(long... globalWorkSize) {
        ensureOpen();
        Objects.requireNonNull(globalWorkSize, "globalWorkSize");
        if (globalWorkSize.length != local.length) {
            throw new IllegalArgumentException(kernelName + ": global size has " + globalWorkSize.length
                    + " dimensions, the local size " + local.length);
        }
        for (long g : globalWorkSize) {
            if (g <= 0) {
                throw new IllegalArgumentException(kernelName + ": global size must be positive: "
                        + Arrays.toString(globalWorkSize));
            }
        }
        this.global = globalWorkSize.clone();
    }

    public String kernelName() {
        return kernelName;
    }

    /** Number of kernel arguments. */
    public int argCount() {
        return args.length;
    }

    /** Current global work size (copy). */
    public long[] global() {
        return global.clone();
    }

    /** Local work size (copy). */
    public long[] local() {
        return local.clone();
    }

    private Object arg(int argIndex) {
        ensureOpen();
        if (argIndex < 0 || argIndex >= args.length) {
            throw new IllegalArgumentException(kernelName + " has no argument " + argIndex + " (" + args.length
                    + " arguments)");
        }
        return args[argIndex];
    }

    private void set(int argIndex, Object value) {
        args[argIndex] = value;
        if (ownsHandle) {
            queue.setKernelArg(handle, argIndex, value);
        }
    }

    private IllegalArgumentException wrongKind(int argIndex, String expected, Object current) {
        return new IllegalArgumentException(kernelName + " argument " + argIndex + " is "
                + current.getClass().getSimpleName() + ", not " + expected);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("prepared dispatch of " + kernelName + " is closed");
        }
    }

    /** True once closed. */
    public boolean isClosed() {
        return closed;
    }

    /** Releases the private handle. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        owner.forget(this);
        if (ownsHandle) {
            handle.close();
        }
    }

    @Override
    public String toString() {
        return "PreparedDispatch{" + kernelName + ", global=" + Arrays.toString(global) + ", local="
                + Arrays.toString(local) + (ownsHandle ? "" : ", shared handle") + (closed ? ", closed" : "") + '}';
    }
}
