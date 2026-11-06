package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;

import java.util.Map;
import java.util.TreeMap;

/**
 * A kernel handle of a remote backend: the id the server loaded or compiled it under, its sticky arguments, and
 * whether this handle is a caller-owned clone the close releases. A backend-owned handle (a plain compile or load)
 * stays cached on the server and is not released by closing the proxy.
 *
 * <p>Arguments are snapshotted when they are set, exactly as {@link KernelArgs} requires: mutating an {@code int[1]}
 * holder afterwards has no effect until the argument is set again.</p>
 */
final class RemoteKernelHandle implements KernelHandle {

    private final RemoteComputeBackend backend;
    private final long id;
    private final String kernelName;
    private final boolean owned;
    private final Map<Integer, Object> args = new TreeMap<>();
    private boolean closed;

    RemoteKernelHandle(RemoteComputeBackend backend, long id, String kernelName, boolean owned) {
        this.backend = backend;
        this.id = id;
        this.kernelName = kernelName;
        this.owned = owned;
    }

    long id() {
        return id;
    }

    boolean belongsTo(RemoteComputeBackend candidate) {
        return backend == candidate;
    }

    /** Replaces every sticky argument with a snapshot of {@code values}. */
    void arguments(Object[] values) {
        args.clear();
        for (int i = 0; i < values.length; i++) {
            args.put(i, KernelArgs.snapshot(values[i], null));
        }
    }

    /** Snapshots one sticky argument; the first call may bind any index. */
    void argument(int index, Object value) {
        if (index < 0) {
            throw new IllegalArgumentException("kernel argument index must not be negative, got " + index);
        }
        args.put(index, KernelArgs.snapshot(value, args.get(index)));
    }

    /**
     * The sticky arguments in declared order.
     *
     * @throws IllegalStateException when an index below the highest bound one was never set
     */
    Object[] arguments() {
        if (args.isEmpty()) {
            return new Object[0];
        }
        int highest = 0;
        for (Integer index : args.keySet()) {
            highest = Math.max(highest, index);
        }
        Object[] out = new Object[highest + 1];
        for (int i = 0; i <= highest; i++) {
            if (!args.containsKey(i)) {
                throw new IllegalStateException(kernelName + ": argument " + i + " was never set; a pre-bound"
                        + " launch needs every argument from 0 to " + highest);
            }
            out[i] = args.get(i);
        }
        return out;
    }

    @Override
    public String kernelName() {
        return kernelName;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (owned) {
            backend.releaseKernel(this);
        }
    }

    @Override
    public String toString() {
        return "RemoteKernelHandle{" + id + ", " + kernelName + (owned ? ", owned" : "") + '}';
    }
}
