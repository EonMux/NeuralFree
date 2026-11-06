package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.WorkSizes;
import com.eonmux.neuralfree.acceleration.metal.MetalNative;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * One Metal command queue of a {@link MetalComputeBackend}, implementing the
 * {@link ComputeQueue} contract.
 *
 * <ul>
 *   <li><b>Batches.</b> Commands are encoded into one {@code MTLCommandBuffer} until
 *       {@link #flush()}, {@link #finish()}, a cross-queue wait, or an awaited event submits the
 *       batch. The batch signals the queue's shared event at a value of its own; an event returned
 *       by an enqueue completes when its batch does. Every dispatch of a batch is therefore
 *       ordered after every earlier command of the queue.</li>
 *   <li><b>Dispatch.</b> {@code global} counts work-items and {@code local} is the threadgroup;
 *       the grid is {@code ceil(global / local)} per dimension ({@link WorkSizes#groupCounts}), so
 *       kernels bounds-check their tail. A null {@code local} uses {@link WorkSizes#defaultLocal}.
 *       The threadgroup is checked against the pipeline's limit and the device's.</li>
 *   <li><b>Arguments</b> are sticky on the {@link MetalKernelHandle}; a null or empty
 *       {@code args} array launches the stored arguments. Buffers bind at indices {@code 0..n-1}
 *       and the scalars are packed into one constant buffer at {@code n} ({@link MetalArgState}).</li>
 *   <li><b>Writes and reads.</b> A write copies the host data into a staging buffer and records a
 *       blit before returning, so the caller may reuse its array immediately. A read records a
 *       blit into a staging buffer; the destination is filled, in issue order, when its batch
 *       retires on {@link #finish()} or {@link GpuEvent#awaitCompletion()}.</li>
 *   <li><b>Waits.</b> An event of another queue is waited for by the batch; an event of this queue
 *       is already ordered. Waiting on an event submits its producer queue.</li>
 *   <li><b>Ownership.</b> {@code close()} of a backend-owned queue only finishes it; the backend
 *       releases it with {@link #closeInternal()}. A caller-owned queue is released by
 *       {@code close()}, or by the backend's close when it is still open then.</li>
 * </ul>
 */
@NotThreadSafe("One thread encodes at a time. Events may be awaited from other threads; batch "
        + "retirement and read completion are guarded by an internal lock.")
public final class MetalComputeQueue implements ComputeQueue {

    private final MetalComputeBackend backend;
    private final long queue;
    private final QueueType queueType;
    private final boolean ownedByBackend;
    private final long event;
    private final Object lock = new Object();
    private final ArrayDeque<Batch> batches = new ArrayDeque<>();
    private long nextValue = 1;
    private Batch current;
    private volatile boolean closed;

    /**
     * Creates the queue's {@code MTLCommandQueue} and its shared event.
     *
     * @param backend        the owning backend
     * @param type           compute or transfer
     * @param ownedByBackend true for the backend's own queues, whose {@code close()} only finishes
     * @throws IllegalArgumentException when the backend is null
     * @throws RuntimeException         the framework failure of the object creation
     */
    public MetalComputeQueue(MetalComputeBackend backend, QueueType type, boolean ownedByBackend) {
        if (backend == null) {
            throw new IllegalArgumentException("Metal backend must not be null");
        }
        if (type == null) {
            throw new IllegalArgumentException("queueType must not be null");
        }
        this.backend = backend;
        this.queueType = type;
        this.ownedByBackend = ownedByBackend;
        this.queue = MetalNative.newCommandQueue(backend.device().getIndex());
        try {
            this.event = MetalNative.newSharedEvent(backend.device().getIndex());
        } catch (RuntimeException e) {
            MetalNative.releaseObject(queue);
            throw e;
        }
    }

    @Override
    public QueueType getQueueType() {
        return queueType;
    }

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                       byte[] srcData, GpuEvent... waitOnEvents) {
        ensureUsable();
        long dst = bufferHandle(dstBuffer, dstOffset, sizeInBytes);
        if (srcData == null) {
            throw new IllegalArgumentException("source byte[] must not be null");
        }
        if (sizeInBytes > srcData.length) {
            throw new IllegalArgumentException("write of " + sizeInBytes + " bytes exceeds the source array of "
                    + srcData.length + " bytes");
        }
        applyWaits(waitOnEvents);
        if (sizeInBytes > 0) {
            long staging = staging(sizeInBytes);
            ByteBuffer view = MetalNative.bufferHostView(staging, sizeInBytes);
            view.put(srcData, 0, (int) sizeInBytes);
            endEncoder();
            MetalNative.copyBuffer(batch().commandBuffer, staging, 0, dst, dstOffset, sizeInBytes);
        }
        return event();
    }

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                       float[] srcData, GpuEvent... waitOnEvents) {
        ensureUsable();
        long dst = bufferHandle(dstBuffer, dstOffset, sizeInBytes);
        if (srcData == null) {
            throw new IllegalArgumentException("source float[] must not be null");
        }
        if ((sizeInBytes & 3L) != 0) {
            throw new IllegalArgumentException("a float write of " + sizeInBytes + " bytes is not a multiple of 4");
        }
        if (sizeInBytes / 4 > srcData.length) {
            throw new IllegalArgumentException("write of " + sizeInBytes + " bytes exceeds the source array of "
                    + srcData.length + " floats");
        }
        applyWaits(waitOnEvents);
        if (sizeInBytes > 0) {
            long staging = staging(sizeInBytes);
            ByteBuffer view = MetalNative.bufferHostView(staging, sizeInBytes).order(ByteOrder.LITTLE_ENDIAN);
            view.asFloatBuffer().put(srcData, 0, (int) (sizeInBytes / 4));
            endEncoder();
            MetalNative.copyBuffer(batch().commandBuffer, staging, 0, dst, dstOffset, sizeInBytes);
        }
        return event();
    }

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                       ByteBuffer srcData, GpuEvent... waitOnEvents) {
        ensureUsable();
        long dst = bufferHandle(dstBuffer, dstOffset, sizeInBytes);
        if (srcData == null) {
            throw new IllegalArgumentException("source ByteBuffer must not be null");
        }
        if (sizeInBytes > srcData.capacity()) {
            throw new IllegalArgumentException("write of " + sizeInBytes + " bytes exceeds the source capacity of "
                    + srcData.capacity() + " bytes");
        }
        applyWaits(waitOnEvents);
        if (sizeInBytes > 0) {
            long staging = staging(sizeInBytes);
            ByteBuffer view = MetalNative.bufferHostView(staging, sizeInBytes);
            ByteBuffer source = srcData.duplicate();
            source.position(0);
            source.limit((int) sizeInBytes);
            view.put(source);
            endEncoder();
            MetalNative.copyBuffer(batch().commandBuffer, staging, 0, dst, dstOffset, sizeInBytes);
        }
        return event();
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes,
                                      byte[] dstData, GpuEvent... waitOnEvents) {
        ensureUsable();
        long src = bufferHandle(srcBuffer, srcOffset, sizeInBytes);
        if (dstData == null) {
            throw new IllegalArgumentException("destination byte[] must not be null");
        }
        if (sizeInBytes > dstData.length) {
            throw new IllegalArgumentException("read of " + sizeInBytes + " bytes exceeds the destination array of "
                    + dstData.length + " bytes");
        }
        applyWaits(waitOnEvents);
        if (sizeInBytes > 0) {
            long staging = staging(sizeInBytes);
            Batch batch = batch();
            endEncoder();
            MetalNative.copyBuffer(batch.commandBuffer, src, srcOffset, staging, 0, sizeInBytes);
            batch.reads.add(new PendingRead(staging, sizeInBytes, dstData, null));
        }
        return event();
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes,
                                      float[] dstData, GpuEvent... waitOnEvents) {
        ensureUsable();
        long src = bufferHandle(srcBuffer, srcOffset, sizeInBytes);
        if (dstData == null) {
            throw new IllegalArgumentException("destination float[] must not be null");
        }
        if ((sizeInBytes & 3L) != 0) {
            throw new IllegalArgumentException("a float read of " + sizeInBytes + " bytes is not a multiple of 4");
        }
        if (sizeInBytes / 4 > dstData.length) {
            throw new IllegalArgumentException("read of " + sizeInBytes + " bytes exceeds the destination array of "
                    + dstData.length + " floats");
        }
        applyWaits(waitOnEvents);
        if (sizeInBytes > 0) {
            long staging = staging(sizeInBytes);
            Batch batch = batch();
            endEncoder();
            MetalNative.copyBuffer(batch.commandBuffer, src, srcOffset, staging, 0, sizeInBytes);
            batch.reads.add(new PendingRead(staging, sizeInBytes, null, dstData));
        }
        return event();
    }

    @Override
    public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                  Object[] args, GpuEvent... waitOnEvents) {
        ensureUsable();
        MetalKernelHandle handle = ownHandle(kernel);
        if (args != null && args.length > 0) {
            handle.args().setAll(args);
        }
        return dispatch(handle, globalWorkSize, localWorkSize, waitOnEvents);
    }

    @Override
    public void enqueueKernelNoEvent(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                     Object[] args, GpuEvent... waitOnEvents) {
        ensureUsable();
        MetalKernelHandle handle = ownHandle(kernel);
        if (args != null && args.length > 0) {
            handle.args().setAll(args);
        }
        dispatch(handle, globalWorkSize, localWorkSize, waitOnEvents);
    }

    @Override
    public void setKernelArg(KernelHandle kernel, int argIndex, Object arg) {
        ensureUsable();
        ownHandle(kernel).args().set(argIndex, arg);
    }

    @Override
    public void enqueueKernelPreBound(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                      GpuEvent... waitOnEvents) {
        ensureUsable();
        dispatch(ownHandle(kernel), globalWorkSize, localWorkSize, waitOnEvents);
    }

    @Override
    public GpuEvent enqueueKernelPreBoundWithEvent(KernelHandle kernel, long[] globalWorkSize,
                                                   long[] localWorkSize, GpuEvent... waitOnEvents) {
        ensureUsable();
        return dispatch(ownHandle(kernel), globalWorkSize, localWorkSize, waitOnEvents);
    }

    @Override
    public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset,
                                  long sizeInBytes, GpuEvent... waitOnEvents) {
        ensureUsable();
        long source = bufferHandle(src, srcOffset, sizeInBytes);
        long destination = bufferHandle(dst, dstOffset, sizeInBytes);
        applyWaits(waitOnEvents);
        if (sizeInBytes > 0) {
            endEncoder();
            MetalNative.copyBuffer(batch().commandBuffer, source, srcOffset, destination, dstOffset, sizeInBytes);
        }
    }

    @Override
    public boolean supportsPeerCopy(ComputeQueue peer) {
        return peer instanceof MetalComputeQueue && ((MetalComputeQueue) peer).backend == backend;
    }

    @Override
    public void enqueuePeerCopy(ComputeQueue destination, GpuBuffer src, long srcOffset, GpuBuffer dst,
                                long dstOffset, long sizeInBytes, GpuEvent... waitOnEvents) {
        if (!supportsPeerCopy(destination)) {
            throw new UnsupportedOperationException("Metal peer copy requires a queue of the same backend");
        }
        enqueueCopyBuffer(src, srcOffset, dst, dstOffset, sizeInBytes, waitOnEvents);
    }

    @Override
    public void enqueueFillBuffer(GpuBuffer dst, long dstOffset, long sizeInBytes, int pattern,
                                  GpuEvent... waitOnEvents) {
        ensureUsable();
        long destination = bufferHandle(dst, dstOffset, sizeInBytes);
        if ((dstOffset & 3L) != 0 || (sizeInBytes & 3L) != 0) {
            throw new IllegalArgumentException("fill offset " + dstOffset + " and size " + sizeInBytes
                    + " must be non-negative multiples of 4");
        }
        applyWaits(waitOnEvents);
        if (sizeInBytes == 0) {
            return;
        }
        if (uniformByte(pattern) >= 0) {
            endEncoder();
            MetalNative.fillBuffer(batch().commandBuffer, destination, dstOffset, sizeInBytes, pattern);
            return;
        }
        // A Metal blit fill writes one byte; a multi-byte pattern is written from the host instead.
        byte[] patternBytes = new byte[(int) Math.min(sizeInBytes, FILL_FALLBACK_CHUNK_BYTES)];
        ByteBuffer littleEndian = ByteBuffer.wrap(patternBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < patternBytes.length; i += 4) {
            littleEndian.putInt(i, pattern);
        }
        long written = 0;
        while (written < sizeInBytes) {
            int n = (int) Math.min(patternBytes.length, sizeInBytes - written);
            byte[] chunk = n == patternBytes.length ? patternBytes : java.util.Arrays.copyOf(patternBytes, n);
            enqueueWriteBuffer(dst, dstOffset + written, n, chunk);
            written += n;
        }
    }

    /** The byte every byte of {@code pattern} equals, or -1 when the pattern is not uniform. */
    private static int uniformByte(int pattern) {
        int b0 = pattern & 0xFF;
        if (((pattern >> 8) & 0xFF) != b0 || ((pattern >> 16) & 0xFF) != b0 || ((pattern >> 24) & 0xFF) != b0) {
            return -1;
        }
        return b0;
    }

    @Override
    public GpuEvent enqueueMarker(GpuEvent... waitOnEvents) {
        ensureUsable();
        applyWaits(waitOnEvents);
        Batch batch = batch();
        return new MetalGpuEvent(this, batch);
    }

    @Override
    public void flush() {
        ensureUsable();
        commit();
    }

    @Override
    public void finish() {
        ensureUsable();
        finishInternal();
    }

    @Override
    public void close() {
        if (ownedByBackend) {
            if (backend.isClosed()) {
                return;
            }
            finishInternal();
            return;
        }
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        try {
            finishInternal();
        } catch (RuntimeException e) {
            failure = e;
        }
        MetalNative.releaseObject(event);
        MetalNative.releaseObject(queue);
        backend.unregisterQueue(this);
        if (failure != null) {
            throw failure;
        }
    }

    /** Finishes and releases a queue the backend owns; called by {@link MetalComputeBackend#close()}. */
    void closeInternal() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        try {
            finishInternal();
        } catch (RuntimeException e) {
            failure = e;
        }
        MetalNative.releaseObject(event);
        MetalNative.releaseObject(queue);
        if (failure != null) {
            throw failure;
        }
    }

    // -----------------------------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------------------------

    /** Blocks until {@code event}'s batch completes and retires the queue's batches up to it. */
    void awaitEvent(MetalGpuEvent event) {
        if (event.owner() != this) {
            throw new IllegalArgumentException("Metal event " + event + " belongs to another queue");
        }
        ensureUsable();
        commit();
        long value = event.value();
        waitFor(value);
        retireUpTo(value);
    }

    /** The highest value this queue's shared event was signaled at. */
    long signaledValue() {
        return MetalNative.eventSignaledValue(event);
    }

    /** The shared event another queue waits on. */
    long eventHandle() {
        return event;
    }

    /** The backend that owns this queue. */
    MetalComputeBackend backend() {
        return backend;
    }

    /** A point at the end of the commands encoded so far; commits nothing by itself. */
    MetalGpuEvent event() {
        return new MetalGpuEvent(this, batch());
    }

    // -----------------------------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------------------------

    private GpuEvent dispatch(MetalKernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                              GpuEvent... waitOnEvents) {
        MetalArgState.Bindings bindings = kernel.args().materialize();
        long[] global = globalWorkSize == null ? null : globalWorkSize.clone();
        if (global == null || global.length < 1 || global.length > 3) {
            throw new IllegalArgumentException(kernel.kernelName() + ": global work size must have 1 to 3"
                    + " dimensions");
        }
        long maxThreads = Math.min(backend.capabilities().getMaxWorkGroupSize(),
                kernel.maxThreadsPerThreadgroup());
        long[] local = localWorkSize == null ? WorkSizes.defaultLocal(global, maxThreads) : localWorkSize.clone();
        if (local.length != global.length) {
            throw new IllegalArgumentException(kernel.kernelName() + ": local work size has "
                    + local.length + " dimensions but global has " + global.length);
        }
        WorkSizes.validateLocal(local, maxThreads, kernel.kernelName());
        kernel.requireValidThreadgroup(local);
        // Metal dispatches on three dimensions; a 1-D or 2-D launch has a trailing size of 1.
        long[] global3 = WorkSizes.pad3(global, 1);
        long[] local3 = WorkSizes.pad3(local, 1);
        long[] groups = WorkSizes.groupCounts(global3, local3);
        long pipeline = kernel.pipeline(local);
        checkThreadgroup(pipeline, local, kernel.kernelName());
        applyWaits(waitOnEvents);
        long encoder = encoder();
        MetalNative.setPipeline(encoder, pipeline);
        int index = 0;
        for (MetalBuffer buffer : bindings.buffers()) {
            long handle = buffer.nativeHandle();
            retainOnce(handle);
            MetalNative.setBuffer(encoder, handle, 0, index++);
        }
        byte[] scalars = bindings.packedScalars();
        if (scalars.length > 0) {
            MetalNative.setBytes(encoder, scalars, scalars.length, index);
        }
        MetalNative.dispatchThreadgroups(encoder, groups[0], groups[1], groups[2],
                local3[0], local3[1], local3[2]);
        return event();
    }

    /** Rejects a threadgroup above the bound pipeline's own limit before anything is encoded. */
    private void checkThreadgroup(long pipeline, long[] local, String kernel) {
        int max = MetalNative.pipelineMaxThreads(pipeline);
        if (max > 0) {
            WorkSizes.validateLocal(local, max, kernel);
        }
    }

    private void applyWaits(GpuEvent... waitOnEvents) {
        if (waitOnEvents == null || waitOnEvents.length == 0) {
            return;
        }
        List<MetalGpuEvent> external = new ArrayList<>(waitOnEvents.length);
        for (GpuEvent wait : waitOnEvents) {
            if (!(wait instanceof MetalGpuEvent)) {
                throw new IllegalArgumentException("Metal queue cannot wait on " + wait);
            }
            MetalGpuEvent other = (MetalGpuEvent) wait;
            if (other.owner().backend != backend) {
                throw new IllegalArgumentException("Metal event " + other + " belongs to another backend");
            }
            if (other.owner() != this) {
                external.add(other);
            }
        }
        if (external.isEmpty()) {
            return;
        }
        // Submit the producers so their events have a value, publish this queue's work encoded so
        // far, then start a fresh batch whose first command is the wait.
        for (MetalGpuEvent other : external) {
            other.submit();
        }
        commit();
        Batch batch = batch();
        for (MetalGpuEvent other : external) {
            MetalNative.encodeWait(batch.commandBuffer, other.owner().eventHandle(), other.value());
        }
    }

    private Batch batch() {
        if (current == null) {
            current = new Batch(MetalNative.newCommandBuffer(queue));
        }
        return current;
    }

    private long encoder() {
        Batch batch = batch();
        if (batch.encoder == 0) {
            batch.encoder = MetalNative.computeEncoder(batch.commandBuffer);
        }
        return batch.encoder;
    }

    private void endEncoder() {
        if (current == null) {
            return;
        }
        if (current.encoder != 0) {
            MetalNative.endEncoding(current.encoder);
            current.encoder = 0;
        }
    }

    private void commit() {
        Batch batch = current;
        if (batch == null) {
            return;
        }
        endEncoder();
        synchronized (lock) {
            batch.value = nextValue++;
        }
        MetalNative.encodeSignal(batch.commandBuffer, event, batch.value);
        MetalNative.commit(batch.commandBuffer);
        current = null;
        synchronized (lock) {
            batches.addLast(batch);
        }
    }

    private void finishInternal() {
        commit();
        long last;
        synchronized (lock) {
            Batch tail = batches.peekLast();
            last = tail == null ? 0 : tail.value;
        }
        if (last > 0) {
            waitFor(last);
            retireUpTo(last);
        }
    }

    private void waitFor(long value) {
        if (value <= 0) {
            return;
        }
        if (MetalNative.eventSignaledValue(event) < value) {
            MetalNative.eventWait(event, value);
        }
    }

    /** Retires every batch at or before {@code value}: checks it, completes reads, releases memory. */
    private void retireUpTo(long value) {
        List<Batch> done = new ArrayList<>();
        synchronized (lock) {
            while (!batches.isEmpty() && batches.peekFirst().value > 0 && batches.peekFirst().value <= value) {
                done.add(batches.removeFirst());
            }
        }
        RuntimeException failure = null;
        for (Batch batch : done) {
            String error = MetalNative.commandBufferError(batch.commandBuffer);
            if (error != null && failure == null) {
                failure = new IllegalStateException("Metal command buffer failed: " + error);
            }
            if (error == null) {
                for (PendingRead read : batch.reads) {
                    read.complete();
                }
            }
            for (Long handle : batch.retained) {
                MetalNative.releaseObject(handle);
            }
            MetalNative.releaseObject(batch.commandBuffer);
        }
        if (failure != null) {
            throw failure;
        }
    }

    private long staging(long sizeInBytes) {
        long staging = MetalNative.newBuffer(backend.device().getIndex(), sizeInBytes);
        Batch batch = batch();
        batch.retained.add(staging);
        return staging;
    }

    private void retainOnce(long handle) {
        Batch batch = batch();
        if (batch.retainedOnce.add(handle)) {
            MetalNative.retainObject(handle);
            batch.retained.add(handle);
        }
    }

    private long bufferHandle(GpuBuffer buffer, long offset, long size) {
        if (!(buffer instanceof MetalBuffer) || ((MetalBuffer) buffer).backend() != backend) {
            throw new IllegalArgumentException(buffer + " is not a buffer of this Metal backend");
        }
        MetalBuffer metal = (MetalBuffer) buffer;
        if (metal.isClosed()) {
            throw new IllegalArgumentException("buffer " + buffer + " is closed");
        }
        if (offset < 0 || size < 0 || offset > metal.sizeInBytes() - size) {
            throw new IllegalArgumentException("range [" + offset + ", " + (offset + size)
                    + ") is outside the buffer of " + metal.sizeInBytes() + " bytes");
        }
        return metal.nativeHandle();
    }

    private MetalKernelHandle ownHandle(KernelHandle kernel) {
        if (!(kernel instanceof MetalKernelHandle) || ((MetalKernelHandle) kernel).backend() != backend) {
            throw new IllegalArgumentException("kernel " + kernel + " is not a handle of this Metal backend");
        }
        return (MetalKernelHandle) kernel;
    }

    private void ensureUsable() {
        if (closed) {
            throw new IllegalStateException("Metal " + queueType + " queue is closed");
        }
        backend.ensureOpen();
    }

    /** One command buffer being filled, and the bookkeeping its completion needs. */
    static final class Batch {
        final long commandBuffer;
        final List<Long> retained = new ArrayList<>();
        final Set<Long> retainedOnce = new HashSet<>();
        final List<PendingRead> reads = new ArrayList<>();
        long encoder;
        /** Shared-event value, assigned when the batch commits; 0 before that. */
        volatile long value;

        Batch(long commandBuffer) {
            this.commandBuffer = commandBuffer;
        }
    }

    /** A read whose staging buffer is copied into its destination when the batch retires. */
    private static final class PendingRead {
        private final long staging;
        private final long size;
        private final byte[] bytes;
        private final float[] floats;

        PendingRead(long staging, long size, byte[] bytes, float[] floats) {
            this.staging = staging;
            this.size = size;
            this.bytes = bytes;
            this.floats = floats;
        }

        void complete() {
            ByteBuffer view = MetalNative.bufferHostView(staging, size).order(ByteOrder.LITTLE_ENDIAN);
            if (bytes != null) {
                view.get(bytes, 0, (int) size);
            } else {
                FloatBuffer source = view.asFloatBuffer();
                source.get(floats, 0, (int) (size / 4));
            }
        }
    }
}
