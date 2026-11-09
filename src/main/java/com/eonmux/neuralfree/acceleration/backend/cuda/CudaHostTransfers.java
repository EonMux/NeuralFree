package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.utils.Closeables;
import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUevent;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.Locale;

import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;

/**
 * The staged host transfers of a {@link CudaComputeQueue}: writes and reads run through the queue's pinned staging
 * rings ({@link CudaPinnedRing}) in slices of at most a ring's slice size, each slice bracketed by a recorded event,
 * and a read's destination is filled only once its copy's stream point has been waited for or the stream drained
 * ({@link PendingRead}). The ring state and the pending reads are guarded by the transfer lock; the queue's other
 * state (guards, timing, capture) is reached through the queue.
 */
final class CudaHostTransfers {

    private static final Logger LOGGER = LoggingUtil.getLogger(CudaComputeQueue.class);

    private final CudaComputeQueue queue;
    private final Object transferLock = new Object();
    private final CudaPinnedRing upload;
    private final CudaPinnedRing readback;
    private final ArrayDeque<PendingRead> pendingReads = new ArrayDeque<>();

    /**
     * Builds the upload and readback staging rings of {@code queue} (a failure propagates to the caller, which
     * releases the stream it owns).
     */
    CudaHostTransfers(CudaComputeQueue queue) {
        this.queue = queue;
        String label = queue.getQueueType().name().toLowerCase(Locale.ROOT);
        this.upload = new CudaPinnedRing(queue.context(), label + " upload",
                CudaPinnedRing.configuredMaxBytes(CudaPinnedRing.DEFAULT_UPLOAD_MB));
        this.readback = new CudaPinnedRing(queue.context(), label + " readback",
                CudaPinnedRing.configuredMaxBytes(CudaPinnedRing.DEFAULT_READBACK_MB));
    }

    GpuEvent write(GpuBuffer dst, long dstOffset, long size, HostSource source, GpuEvent[] waits) {
        queue.enter();
        queue.requireNotCapturing("a host write");
        queue.uncapturable("a host write");
        CUdeviceptr device = queue.devicePointer(dst, "write destination");
        CudaComputeQueue.checkDeviceRange(dst, dstOffset, size, "write destination");
        queue.validateWaits(waits);
        synchronized (transferLock) {
            queue.applyWaits(waits);
            CudaGpuEvent last = null;
            long done = 0;
            try {
                while (done < size) {
                    int n = (int) Math.min(size - done, upload.maxSliceBytes());
                    CudaPinnedRing.Slice slice = upload.acquire(n);
                    CUdeviceptr target = device.withByteOffset(dstOffset + done);
                    CUevent timedStart = queue.beginTiming("memcpy HtoD");
                    try {
                        source.copyInto(slice.byteBuffer(), done, n);
                        Pointer staged = slice.pointer();
                        CudaErrors.check("cuMemcpyHtoDAsync",
                                () -> cuMemcpyHtoDAsync(target, staged, n, queue.getCUStream()));
                    } catch (RuntimeException e) {
                        queue.cancelTiming(timedStart);
                        slice.release();
                        throw e;
                    }
                    queue.endTiming(timedStart, "memcpy HtoD");
                    CudaGpuEvent event = recordAfterCopy(slice);
                    slice.releaseAfter(event.share());
                    if (last != null) {
                        last.close();
                    }
                    last = event;
                    done += n;
                }
            } catch (RuntimeException e) {
                if (last != null) {
                    last.close();
                }
                throw e;
            }
            return last;
        }
    }

    GpuEvent read(GpuBuffer src, long srcOffset, long size, HostSink sink, GpuEvent[] waits) {
        queue.enter();
        queue.requireNotCapturing("a host read");
        queue.uncapturable("a host read");
        CUdeviceptr device = queue.devicePointer(src, "read source");
        CudaComputeQueue.checkDeviceRange(src, srcOffset, size, "read source");
        queue.validateWaits(waits);
        synchronized (transferLock) {
            queue.applyWaits(waits);
            CudaGpuEvent last = null;
            long done = 0;
            try {
                while (done < size) {
                    int n = (int) Math.min(size - done, readback.maxSliceBytes());
                    CudaPinnedRing.Slice slice = acquireReadSlice(n);
                    CUdeviceptr origin = device.withByteOffset(srcOffset + done);
                    CUevent timedStart = queue.beginTiming("memcpy DtoH");
                    try {
                        Pointer staged = slice.pointer();
                        CudaErrors.check("cuMemcpyDtoHAsync",
                                () -> cuMemcpyDtoHAsync(staged, origin, n, queue.getCUStream()));
                    } catch (RuntimeException e) {
                        queue.cancelTiming(timedStart);
                        slice.release();
                        throw e;
                    }
                    queue.endTiming(timedStart, "memcpy DtoH");
                    CudaGpuEvent event = recordAfterCopy(slice);
                    pendingReads.addLast(new PendingRead(event.share(), slice, sink, done, n));
                    if (last != null) {
                        last.close();
                    }
                    last = event;
                    done += n;
                }
            } catch (RuntimeException e) {
                if (last != null) {
                    last.close();
                }
                throw e;
            }
            return last;
        }
    }

    /** A readback slice; when the ring is full of unretired reads, waits for the oldest and retires it. */
    private CudaPinnedRing.Slice acquireReadSlice(int n) {
        CudaPinnedRing.Slice slice = readback.tryAcquire(n);
        while (slice == null) {
            PendingRead oldest = pendingReads.peekFirst();
            if (oldest == null) {
                throw new IllegalStateException(
                        "CUDA " + queue.getQueueType() + " readback ring is full without pending reads");
            }
            oldest.event.synchronize();
            retireReadsLocked(oldest.sequence);
            slice = readback.tryAcquire(n);
        }
        return slice;
    }

    /**
     * Records the event that follows a staged copy. When recording fails, the slice is
     * returned to its ring only after the stream drained, so in-flight memory is never reused.
     */
    private CudaGpuEvent recordAfterCopy(CudaPinnedRing.Slice slice) {
        try {
            return queue.record();
        } catch (RuntimeException e) {
            try {
                CudaErrors.check("cuStreamSynchronize", () -> cuStreamSynchronize(queue.getCUStream()));
                slice.release();
            } catch (RuntimeException drain) {
                e.addSuppressed(drain);
            }
            throw e;
        }
    }

    /**
     * Fills the destinations of every read recorded at or before {@code upTo}. The caller has
     * already waited for a point of the stream at or after {@code upTo}.
     */
    void retire(long upTo) {
        synchronized (transferLock) {
            retireReadsLocked(upTo);
        }
    }

    private void retireReadsLocked(long upTo) {
        RuntimeException failure = null;
        while (!pendingReads.isEmpty() && pendingReads.peekFirst().sequence <= upTo) {
            try {
                pendingReads.pollFirst().complete();
            } catch (RuntimeException e) {
                failure = Closeables.chain(failure, e);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /** Discards reads whose copies did not provably finish; their destinations stay unwritten. */
    private void dropPendingReads() {
        if (!pendingReads.isEmpty()) {
            LOGGER.warn("CUDA {} queue closed with {} unfinished reads; their destinations were not written",
                    queue.getQueueType(), pendingReads.size());
        }
        while (!pendingReads.isEmpty()) {
            pendingReads.pollFirst().event.close();
        }
    }

    /**
     * Marks the queue closed under the transfer lock, so no further transfer starts.
     *
     * @return false when the queue was already closed
     */
    boolean beginClose() {
        synchronized (transferLock) {
            if (queue.isClosed()) {
                return false;
            }
            queue.markClosed();
            return true;
        }
    }

    /**
     * Completes or discards the pending reads (by whether the stream {@code drained}) and frees both staging
     * rings, chaining every failure onto {@code failure}.
     */
    RuntimeException finishClose(boolean drained, RuntimeException failure) {
        synchronized (transferLock) {
            try {
                if (drained) {
                    retireReadsLocked(Long.MAX_VALUE);
                } else {
                    dropPendingReads();
                }
            } catch (RuntimeException e) {
                failure = Closeables.chain(failure, e);
            }
            failure = closeRing(upload, failure);
            failure = closeRing(readback, failure);
            return failure;
        }
    }

    private static RuntimeException closeRing(CudaPinnedRing ring, RuntimeException failure) {
        try {
            ring.close();
            return failure;
        } catch (RuntimeException e) {
            return Closeables.chain(failure, e);
        }
    }

    /** Copies {@code n} host bytes starting at byte {@code from} into a staging view. */
    interface HostSource {
        void copyInto(ByteBuffer staging, long from, int n);
    }

    /** Copies {@code n} bytes of a staging view into the host destination at byte {@code to}. */
    interface HostSink {
        void copyFrom(ByteBuffer staging, long to, int n);
    }

    /** A device-to-host copy whose destination is filled once its stream point is reached. */
    private static final class PendingRead {
        private final CudaGpuEvent event;
        private final long sequence;
        private final CudaPinnedRing.Slice slice;
        private final HostSink sink;
        private final long hostOffset;
        private final int size;

        PendingRead(CudaGpuEvent event, CudaPinnedRing.Slice slice, HostSink sink, long hostOffset, int size) {
            this.event = event;
            this.sequence = event.sequence();
            this.slice = slice;
            this.sink = sink;
            this.hostOffset = hostOffset;
            this.size = size;
        }

        /** Copies the staged bytes to the destination and hands back the slice and event. */
        void complete() {
            try {
                sink.copyFrom(slice.byteBuffer(), hostOffset, size);
            } finally {
                slice.release();
                event.close();
            }
        }
    }
}
