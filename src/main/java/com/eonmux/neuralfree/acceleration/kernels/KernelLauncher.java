package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;

import java.util.Arrays;
import java.util.Objects;

/**
 * Enqueues kernel launches on one queue through {@link ComputeQueue#enqueueKernelNoEvent}. Immutable; the queue's
 * one-thread-at-a-time contract is the caller's to uphold.
 *
 * <p>A launch with a zero extent returns without enqueueing anything, because the backends reject a zero global
 * size ({@link com.eonmux.neuralfree.acceleration.backend.WorkSizes}).</p>
 */
public final class KernelLauncher {

    private final ComputeQueue queue;

    public KernelLauncher(ComputeQueue queue) {
        this.queue = Objects.requireNonNull(queue, "queue must not be null");
    }

    /**
     * The smallest multiple of {@code multiple} that is at least {@code n}.
     *
     * @throws IllegalArgumentException when {@code n} is negative, {@code multiple} is not positive, or the
     *                                  result overflows a long
     */
    public static long roundUp(long n, long multiple) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative, got " + n);
        }
        if (multiple <= 0) {
            throw new IllegalArgumentException("multiple must be positive, got " + multiple);
        }
        long rem = n % multiple;
        if (rem == 0) {
            return n;
        }
        long pad = multiple - rem;
        if (n > Long.MAX_VALUE - pad) {
            throw new IllegalArgumentException(n + " rounded up to a multiple of " + multiple + " overflows a long");
        }
        return n + pad;
    }

    /**
     * A 1-D launch of {@code roundUp(n, local)} work-items in groups of {@code local}; returns when {@code n == 0}.
     *
     * @throws IllegalArgumentException when {@code n} is negative or {@code local} is not positive
     */
    public void launch1d(KernelHandle k, long n, int local, Object... args) {
        Objects.requireNonNull(k, "kernel must not be null");
        Objects.requireNonNull(args, "args must not be null");
        if (local <= 0) {
            throw new IllegalArgumentException(k.kernelName() + ": local size must be positive, got " + local);
        }
        if (n < 0) {
            throw new IllegalArgumentException(k.kernelName() + ": work-item count must be non-negative, got " + n);
        }
        if (n == 0) {
            return;
        }
        queue.enqueueKernelNoEvent(k, new long[]{roundUp(n, local)}, new long[]{local}, args.clone());
    }

    /**
     * A launch with explicit sizes; returns when any global extent is 0.
     *
     * @throws IllegalArgumentException when the sizes do not have the same 1 to 3 dimensions, an extent is
     *                                  negative, a local extent is not positive, or a global extent is not a
     *                                  multiple of its local extent
     */
    public void launch(KernelHandle k, long[] global, long[] local, Object... args) {
        Objects.requireNonNull(k, "kernel must not be null");
        Objects.requireNonNull(global, "global must not be null");
        Objects.requireNonNull(local, "local must not be null");
        Objects.requireNonNull(args, "args must not be null");
        if (global.length < 1 || global.length > 3 || local.length != global.length) {
            throw new IllegalArgumentException(k.kernelName() + ": global " + Arrays.toString(global) + " and local "
                    + Arrays.toString(local) + " must have the same 1 to 3 dimensions");
        }
        boolean empty = false;
        for (int d = 0; d < global.length; d++) {
            if (local[d] <= 0 || global[d] < 0) {
                throw new IllegalArgumentException(k.kernelName() + ": invalid sizes global " + Arrays.toString(global)
                        + ", local " + Arrays.toString(local));
            }
            if (global[d] % local[d] != 0) {
                throw new IllegalArgumentException(k.kernelName() + ": global " + Arrays.toString(global)
                        + " is not a multiple of local " + Arrays.toString(local));
            }
            empty |= global[d] == 0;
        }
        if (empty) {
            return;
        }
        queue.enqueueKernelNoEvent(k, global.clone(), local.clone(), args.clone());
    }
}
