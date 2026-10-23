package com.eonmux.neuralfree.optim;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A host-memory backend that declines every kernel artifact, so tensor code takes its CPU paths, and counts the
 * buffers that are allocated and not yet closed ({@link #liveBuffers()}). Commands execute at enqueue time.
 */
final class LiveBufferBackend implements ComputeBackend {
    private final AtomicLong ids = new AtomicLong();
    private final AtomicInteger live = new AtomicInteger();
    private final DeviceCapabilities caps = new DeviceCapabilities.Builder()
            .deviceName("live-buffer host").maxWorkGroupSize(1024).build();
    private final HostQueue compute = new HostQueue(ComputeQueue.QueueType.COMPUTE);
    private final HostQueue transfer = new HostQueue(ComputeQueue.QueueType.TRANSFER);

    /** The number of buffers allocated on this backend and not yet closed. */
    int liveBuffers() {
        return live.get();
    }

    @Override
    public BackendType backendType() {
        return BackendType.OPENCL;
    }

    @Override
    public DeviceCapabilities capabilities() {
        return caps;
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
        if (sizeInBytes <= 0 || sizeInBytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("host buffer size must be in [1, 2^31-1], got " + sizeInBytes);
        }
        live.incrementAndGet();
        return new HostBuffer(ids.incrementAndGet(), new byte[(int) sizeInBytes], live);
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes) {
        return allocateBuffer(sizeInBytes, false);
    }

    @Override
    public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
        throw new UnsupportedKernelException("the live-buffer backend runs no kernels");
    }

    @Override
    public KernelHandle loadKernel(KernelArtifact artifact) {
        throw new UnsupportedKernelException("the live-buffer backend runs no kernels");
    }

    @Override
    public boolean supportsArtifact(KernelArtifact.Kind kind) {
        return false;
    }

    @Override
    public void close() {
    }

    /** Host bytes; closing decrements the live count once. */
    private static final class HostBuffer implements GpuBuffer {
        private final long id;
        private final byte[] data;
        private final AtomicInteger live;
        private boolean closed;

        HostBuffer(long id, byte[] data, AtomicInteger live) {
            this.id = id;
            this.data = data;
            this.live = live;
        }

        @Override
        public long sizeInBytes() {
            return data.length;
        }

        @Override
        public long bufferId() {
            return id;
        }

        @Override
        public synchronized void close() {
            if (!closed) {
                closed = true;
                live.decrementAndGet();
            }
        }

        synchronized byte[] range(long offset, long size) {
            if (closed) {
                throw new IllegalStateException("host buffer " + id + " is closed");
            }
            if (offset < 0 || size < 0 || offset + size > data.length) {
                throw new IllegalArgumentException("range [" + offset + ", " + (offset + size)
                        + ") is outside host buffer " + id + " of " + data.length + " bytes");
            }
            return data;
        }
    }

    private static final class DoneEvent implements GpuEvent {
        @Override
        public void awaitCompletion() {
        }

        @Override
        public boolean isComplete() {
            return true;
        }

        @Override
        public void close() {
        }
    }

    /** Executes every transfer at enqueue time; refuses kernels. */
    private static final class HostQueue implements ComputeQueue {
        private final QueueType type;

        HostQueue(QueueType type) {
            this.type = type;
        }

        @Override
        public QueueType getQueueType() {
            return type;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dst, long dstOffset, long size, byte[] src, GpuEvent... waits) {
            System.arraycopy(src, 0, host(dst).range(dstOffset, size), (int) dstOffset, (int) size);
            return new DoneEvent();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dst, long dstOffset, long size, float[] src, GpuEvent... waits) {
            ByteBuffer view = ByteBuffer.wrap(host(dst).range(dstOffset, size)).order(ByteOrder.nativeOrder());
            for (int i = 0; i < size / Float.BYTES; i++) {
                view.putFloat((int) dstOffset + i * Float.BYTES, src[i]);
            }
            return new DoneEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer src, long srcOffset, long size, byte[] dst, GpuEvent... waits) {
            System.arraycopy(host(src).range(srcOffset, size), (int) srcOffset, dst, 0, (int) size);
            return new DoneEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer src, long srcOffset, long size, float[] dst, GpuEvent... waits) {
            ByteBuffer view = ByteBuffer.wrap(host(src).range(srcOffset, size)).order(ByteOrder.nativeOrder());
            for (int i = 0; i < size / Float.BYTES; i++) {
                dst[i] = view.getFloat((int) srcOffset + i * Float.BYTES);
            }
            return new DoneEvent();
        }

        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] global, long[] local, Object[] args,
                                      GpuEvent... waits) {
            throw new UnsupportedOperationException("the live-buffer backend runs no kernels");
        }

        @Override
        public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset, long size,
                                      GpuEvent... waits) {
            System.arraycopy(host(src).range(srcOffset, size), (int) srcOffset, host(dst).range(dstOffset, size),
                    (int) dstOffset, (int) size);
        }

        @Override
        public GpuEvent enqueueMarker(GpuEvent... waits) {
            return new DoneEvent();
        }

        @Override
        public void flush() {
        }

        @Override
        public void finish() {
        }

        @Override
        public void close() {
        }

        private static HostBuffer host(GpuBuffer buffer) {
            if (!(buffer instanceof HostBuffer)) {
                throw new IllegalArgumentException("buffer " + buffer + " does not belong to the live-buffer backend");
            }
            return (HostBuffer) buffer;
        }
    }
}
