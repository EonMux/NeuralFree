package com.eonmux.neuralfree.acceleration.backend.ocl;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.jocl.CL.CL_COMPLETE;

/**
 * Host-side completion of one non-blocking device-to-host read.
 *
 * <p>The read's {@code CL_COMPLETE} event callback calls {@link #complete(int, Runnable)} with the
 * command's execution status. A successful read copies the staging buffer into the caller's array;
 * a read that terminated abnormally (negative status) copies nothing and records the status, so
 * stale staging bytes are never delivered as data. Either way the latch is released, and the
 * failure is reported once, by whichever of {@link OclGpuEvent#awaitCompletion()} or
 * {@link OclComputeQueue#finish()} observes it first.</p>
 */
final class OclReadCompletion {

    private final CountDownLatch latch = new CountDownLatch(1);
    private final AtomicBoolean failureReported = new AtomicBoolean();
    private final long sizeInBytes;
    private volatile int status = CL_COMPLETE;
    private volatile RuntimeException copyFailure;

    /** @param sizeInBytes bytes the read transfers, for messages */
    OclReadCompletion(long sizeInBytes) {
        this.sizeInBytes = sizeInBytes;
    }

    /**
     * Records the outcome of the read; runs on the driver's callback thread. Releases the latch in
     * every case.
     *
     * @param executionStatus {@code CL_COMPLETE}, or the negative status of an abnormal termination
     * @param copy            copies the staging buffer into the destination; run only on success
     */
    void complete(int executionStatus, Runnable copy) {
        try {
            if (executionStatus < 0) {
                status = executionStatus;
            } else {
                copy.run();
            }
        } catch (RuntimeException e) {
            copyFailure = e;
        } finally {
            latch.countDown();
        }
    }

    /** Waits until the callback has run. */
    void await() throws InterruptedException {
        latch.await();
    }

    /** True once the callback has run. */
    boolean isDone() {
        return latch.getCount() == 0;
    }

    /**
     * Throws the read's failure when it has one that was not reported yet; a failure is reported
     * at most once. Call only after {@link #await()}.
     *
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when the read terminated
     *                                                                     for lack of memory
     * @throws com.eonmux.neuralfree.acceleration.GpuBackendException     when it terminated otherwise
     * @throws IllegalStateException when copying into the destination failed
     */
    void throwIfFailed() {
        RuntimeException failure = failure();
        if (failure != null && failureReported.compareAndSet(false, true)) {
            throw failure;
        }
    }

    /** Marks the failure, if any, as reported elsewhere so {@link #throwIfFailed()} stays silent. */
    void markReported() {
        failureReported.set(true);
    }

    private RuntimeException failure() {
        int s = status;
        if (s < 0) {
            return OclErrors.translate(s, "clEnqueueReadBuffer", sizeInBytes + " bytes, completed abnormally",
                    -1, null);
        }
        RuntimeException copy = copyFailure;
        if (copy != null) {
            return new IllegalStateException("copying a completed OpenCL read of " + sizeInBytes
                    + " bytes into its destination failed: " + copy, copy);
        }
        return null;
    }
}
