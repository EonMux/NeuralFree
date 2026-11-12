package com.eonmux.neuralfree.tensor.gpu;

import com.eonmux.neuralfree.acceleration.backend.BackendResources;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelTaskBatch;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A view of a real GPU backend that counts every byte its queues move between host and device: every
 * {@code enqueueWriteBuffer} and {@code enqueueReadBuffer} on its compute queue, its transfer queue and the compute
 * queues it creates. It also logs, in order, the writes, reads, flushes and markers of its compute and transfer
 * queues ({@link #operations()}). Everything else is the delegate's: buffers, kernels, events and the device work
 * itself, so code run on this backend takes the delegate's paths. The view has its own {@link BackendResources}
 * (train context, caches), released by {@link #close()}, which leaves the delegate open.
 */
public final class TransferCountingBackend implements ComputeBackend {

    private final ComputeBackend delegate;
    private final AtomicLong written = new AtomicLong();
    private final AtomicLong read = new AtomicLong();
    private final List<String> operations = Collections.synchronizedList(new ArrayList<>());
    private final ComputeQueue compute;
    private final ComputeQueue transfer;

    public TransferCountingBackend(ComputeBackend delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate must not be null");
        this.compute = new CountingQueue(delegate.computeQueue(), "compute");
        ComputeQueue realTransfer = delegate.transferQueue();
        this.transfer = realTransfer == delegate.computeQueue() ? compute : new CountingQueue(realTransfer, "transfer");
    }

    /**
     * The writes, reads, flushes and markers of the compute and transfer queues since the last {@link #clearLog()},
     * oldest first, each as {@code "<operation> <queue>"}: {@code write}, {@code read}, {@code flush} or
     * {@code marker}, on {@code compute} or {@code transfer}.
     */
    public List<String> operations() {
        synchronized (operations) {
            return new ArrayList<>(operations);
        }
    }

    public void clearLog() {
        operations.clear();
    }

    /** Bytes written from host to device since this view was made. */
    public long bytesWritten() {
        return written.get();
    }

    /** Bytes read from device to host since this view was made. */
    public long bytesRead() {
        return read.get();
    }

    /** Bytes moved in either direction since this view was made. */
    public long bytesMoved() {
        return written.get() + read.get();
    }

    @Override
    public BackendType backendType() {
        return delegate.backendType();
    }

    @Override
    public DeviceCapabilities capabilities() {
        return delegate.capabilities();
    }

    @Override
    public ComputeQueue computeQueue() {
        return compute;
    }

    @Override
    public ComputeQueue transferQueue() {
        return transfer;
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
        return delegate.allocateBuffer(sizeInBytes, readOnly);
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes) {
        return delegate.allocateBuffer(sizeInBytes);
    }

    @Override
    public boolean supportsHostMappedBuffers() {
        return delegate.supportsHostMappedBuffers();
    }

    @Override
    public HostMappedBuffer allocateHostMappedBuffer(long sizeInBytes) {
        return delegate.allocateHostMappedBuffer(sizeInBytes);
    }

    @Override
    public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
        return delegate.compileKernel(kernelName, kernelSource, buildOptions);
    }

    @Override
    public KernelHandle loadKernel(KernelArtifact artifact) {
        return delegate.loadKernel(artifact);
    }

    @Override
    public boolean supportsArtifact(KernelArtifact.Kind kind) {
        return delegate.supportsArtifact(kind);
    }

    @Override
    public void prepareKernel(KernelHandle kernel, long[] localWorkSize) {
        delegate.prepareKernel(kernel, localWorkSize);
    }

    @Override
    public KernelHandle cloneKernel(KernelHandle template) {
        return delegate.cloneKernel(template);
    }

    @Override
    public ComputeQueue createComputeQueue() {
        return new CountingQueue(delegate.createComputeQueue(), null);
    }

    @Override
    public long getAllocatedBytes() {
        return delegate.getAllocatedBytes();
    }

    @Override
    public long getFreeVramBytes() {
        return delegate.getFreeVramBytes();
    }

    /** Releases this view's backend resources; the delegate stays open. */
    @Override
    public void close() {
        BackendResources.closeAll(this);
    }

    @Override
    public String toString() {
        return "TransferCountingBackend{" + delegate + "}";
    }

    /** A delegate queue whose host transfers add to the view's counters. */
    private final class CountingQueue implements ComputeQueue {
        private final ComputeQueue q;
        /** The name its operations are logged under, or null for a created queue, which is not logged. */
        private final String name;

        CountingQueue(ComputeQueue q, String name) {
            this.q = Objects.requireNonNull(q, "queue must not be null");
            this.name = name;
        }

        private void log(String operation) {
            if (name != null) {
                operations.add(operation + " " + name);
            }
        }

        @Override
        public QueueType getQueueType() {
            return q.getQueueType();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dst, long dstOffset, long size, byte[] src, GpuEvent... waits) {
            GpuEvent e = q.enqueueWriteBuffer(dst, dstOffset, size, src, waits);
            written.addAndGet(size);
            log("write");
            return e;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dst, long dstOffset, long size, float[] src, GpuEvent... waits) {
            GpuEvent e = q.enqueueWriteBuffer(dst, dstOffset, size, src, waits);
            written.addAndGet(size);
            log("write");
            return e;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dst, long dstOffset, long size, ByteBuffer src,
                                           GpuEvent... waits) {
            GpuEvent e = q.enqueueWriteBuffer(dst, dstOffset, size, src, waits);
            written.addAndGet(size);
            log("write");
            return e;
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer src, long srcOffset, long size, byte[] dst, GpuEvent... waits) {
            GpuEvent e = q.enqueueReadBuffer(src, srcOffset, size, dst, waits);
            read.addAndGet(size);
            log("read");
            return e;
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer src, long srcOffset, long size, float[] dst, GpuEvent... waits) {
            GpuEvent e = q.enqueueReadBuffer(src, srcOffset, size, dst, waits);
            read.addAndGet(size);
            log("read");
            return e;
        }

        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] global, long[] local, Object[] args,
                                      GpuEvent... waits) {
            return q.enqueueKernel(kernel, global, local, args, waits);
        }

        @Override
        public void enqueueKernelNoEvent(KernelHandle kernel, long[] global, long[] local, Object[] args,
                                         GpuEvent... waits) {
            q.enqueueKernelNoEvent(kernel, global, local, args, waits);
        }

        @Override
        public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset, long size,
                                      GpuEvent... waits) {
            q.enqueueCopyBuffer(src, srcOffset, dst, dstOffset, size, waits);
        }

        @Override
        public boolean supportsPeerCopy(ComputeQueue peer) {
            return peer == this || q.supportsPeerCopy(unwrap(peer));
        }

        @Override
        public void enqueuePeerCopy(ComputeQueue destination, GpuBuffer src, long srcOffset, GpuBuffer dst,
                                    long dstOffset, long size, GpuEvent... waits) {
            q.enqueuePeerCopy(unwrap(destination), src, srcOffset, dst, dstOffset, size, waits);
        }

        @Override
        public void setKernelArg(KernelHandle kernel, int argIndex, Object arg) {
            q.setKernelArg(kernel, argIndex, arg);
        }

        @Override
        public void enqueueKernelPreBound(KernelHandle kernel, long[] global, long[] local, GpuEvent... waits) {
            q.enqueueKernelPreBound(kernel, global, local, waits);
        }

        @Override
        public GpuEvent enqueueKernelPreBoundWithEvent(KernelHandle kernel, long[] global, long[] local,
                                                       GpuEvent... waits) {
            return q.enqueueKernelPreBoundWithEvent(kernel, global, local, waits);
        }

        @Override
        public void enqueueFillBuffer(GpuBuffer dst, long dstOffset, long size, int pattern, GpuEvent... waits) {
            q.enqueueFillBuffer(dst, dstOffset, size, pattern, waits);
        }

        @Override
        public GpuEvent enqueueMarker(GpuEvent... waits) {
            GpuEvent e = q.enqueueMarker(waits);
            log("marker");
            return e;
        }

        @Override
        public boolean supportsGraphCapture() {
            return q.supportsGraphCapture();
        }

        @Override
        public void beginCapture() {
            q.beginCapture();
        }

        @Override
        public GpuGraph endCapture() {
            return q.endCapture();
        }

        @Override
        public void abortCapture() {
            q.abortCapture();
        }

        @Override
        public boolean isCapturing() {
            return q.isCapturing();
        }

        @Override
        public boolean supportsKernelBatching() {
            return q.supportsKernelBatching();
        }

        @Override
        public KernelTaskBatch beginKernelBatch() {
            return q.beginKernelBatch();
        }

        @Override
        public KernelTiming kernelTiming() {
            return q.kernelTiming();
        }

        @Override
        public boolean timesGraphLaunches() {
            return q.timesGraphLaunches();
        }

        @Override
        public void flush() {
            q.flush();
            log("flush");
        }

        @Override
        public void finish() {
            q.finish();
        }

        @Override
        public void close() {
            q.close();
        }

        private ComputeQueue unwrap(ComputeQueue peer) {
            return peer instanceof CountingQueue ? ((CountingQueue) peer).q : peer;
        }
    }
}
