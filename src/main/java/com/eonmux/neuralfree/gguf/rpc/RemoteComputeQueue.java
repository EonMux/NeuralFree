package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelTaskBatch;

import java.util.Objects;

/**
 * A command queue of a remote backend. Commands that need no answer — writes, fills, copies, dispatches — are
 * appended to the endpoint's pipelined batch and flush together, so a decode step costs one round trip instead of
 * one per command; reads and other answered commands flush the batch first, so every command reaches the server in
 * issue order and the queue stays in-order. A returned event completes lazily: awaiting it flushes the batch and
 * lets the server finish its device through the command's sequence, exactly where the caller observes. When the
 * server keeps a weight cache, writes of one chunk or more probe it by content hash first and skip the transfer on
 * a hit ({@code rpc-server --cache}). With the pipeline off
 * ({@code neuralfree.rpc.pipeline=false}), every command is one synchronous round trip that finishes the server's
 * device, as the first version of the protocol ran.
 */
final class RemoteComputeQueue implements ComputeQueue {

    private final RemoteComputeBackend backend;
    private final QueueType queueType;

    RemoteComputeQueue(RemoteComputeBackend backend, QueueType queueType) {
        this.backend = Objects.requireNonNull(backend, "backend");
        this.queueType = Objects.requireNonNull(queueType, "queueType");
    }

    @Override
    public QueueType getQueueType() {
        return queueType;
    }

    /** The remote backend this queue dispatches on. */
    RemoteComputeBackend backend() {
        return backend;
    }

    // ------------------------------------------------------------------ transfers

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, byte[] srcData,
                                       GpuEvent... waitOnEvents) {
        await(waitOnEvents);
        RemoteBuffer dst = buffer(dstBuffer, "write destination");
        if (srcData == null) {
            throw new IllegalArgumentException("write source must not be null");
        }
        checkRange(dst, dstOffset, sizeInBytes, srcData.length, "write");
        boolean cache = backend.batch().cacheEnabled();
        long seq = 0;
        long done = 0;
        while (done < sizeInBytes) {
            int n = (int) Math.min(RpcWire.CHUNK, sizeInBytes - done);
            byte[] chunk = new byte[n];
            System.arraycopy(srcData, (int) done, chunk, 0, n);
            seq = writeChunk(dst, dstOffset + done, chunk, cache);
            done += n;
        }
        return new RemoteGpuEvent(backend.batch(), seq);
    }

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, float[] srcData,
                                       GpuEvent... waitOnEvents) {
        if ((sizeInBytes & 3) != 0) {
            throw new IllegalArgumentException("a float write of " + sizeInBytes + " bytes is not a whole number"
                    + " of floats");
        }
        int floats = (int) (sizeInBytes / Float.BYTES);
        if (srcData == null || floats > srcData.length) {
            throw new IllegalArgumentException("write of " + floats + " floats exceeds the source array");
        }
        float[] slice = new float[floats];
        System.arraycopy(srcData, 0, slice, 0, floats);
        return enqueueWriteBuffer(dstBuffer, dstOffset, sizeInBytes, RpcWire.floatsToBytes(slice), waitOnEvents);
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, byte[] dstData,
                                      GpuEvent... waitOnEvents) {
        await(waitOnEvents);
        RemoteBuffer src = buffer(srcBuffer, "read source");
        if (dstData == null) {
            throw new IllegalArgumentException("read destination must not be null");
        }
        checkRange(src, srcOffset, sizeInBytes, dstData.length, "read");
        long done = 0;
        while (done < sizeInBytes) {
            int n = (int) Math.min(RpcWire.CHUNK, sizeInBytes - done);
            final long offset = srcOffset + done;
            // a read is an observation point: the server finishes the device before it answers, so every command
            // before the read counts as finished and its deferred events complete
            byte[] chunk = backend.batch()
                    .request(new RpcWire.Writer(RpcWire.OP_READ).i64(src.id()).i64(offset).i32(n), true)
                    .rawBytes();
            System.arraycopy(chunk, 0, dstData, (int) done, chunk.length);
            done += n;
        }
        return RemoteGpuEvent.COMPLETE;
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, float[] dstData,
                                      GpuEvent... waitOnEvents) {
        if ((sizeInBytes & 3) != 0) {
            throw new IllegalArgumentException("a float read of " + sizeInBytes + " bytes is not a whole number"
                    + " of floats");
        }
        int floats = (int) (sizeInBytes / Float.BYTES);
        if (dstData == null || floats > dstData.length) {
            throw new IllegalArgumentException("read of " + floats + " floats exceeds the destination array");
        }
        byte[] bytes = new byte[(int) sizeInBytes];
        enqueueReadBuffer(srcBuffer, srcOffset, sizeInBytes, bytes, waitOnEvents);
        float[] values = RpcWire.bytesToFloats(bytes);
        System.arraycopy(values, 0, dstData, 0, values.length);
        return RemoteGpuEvent.COMPLETE;
    }

    @Override
    public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset, long sizeInBytes,
                                  GpuEvent... waitOnEvents) {
        await(waitOnEvents);
        RemoteBuffer from = buffer(src, "copy source");
        RemoteBuffer to = buffer(dst, "copy destination");
        checkRange(from, srcOffset, sizeInBytes, from.sizeInBytes(), "copy source");
        checkRange(to, dstOffset, sizeInBytes, to.sizeInBytes(), "copy destination");
        backend.submit(new RpcWire.Writer(RpcWire.OP_COPY).i64(from.id()).i64(srcOffset)
                .i64(to.id()).i64(dstOffset).i64(sizeInBytes), true);
    }

    @Override
    public void enqueueFillBuffer(GpuBuffer dst, long dstOffset, long sizeInBytes, int pattern,
                                  GpuEvent... waitOnEvents) {
        await(waitOnEvents);
        RemoteBuffer target = buffer(dst, "fill destination");
        checkRange(target, dstOffset, sizeInBytes, target.sizeInBytes(), "fill");
        backend.submit(new RpcWire.Writer(RpcWire.OP_FILL).i64(target.id()).i64(dstOffset)
                .i64(sizeInBytes).i32(pattern), true);
    }

    // ------------------------------------------------------------------ dispatch

    @Override
    public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args,
                                  GpuEvent... waitOnEvents) {
        await(waitOnEvents);
        RemoteKernelHandle handle = kernel(kernel);
        checkWorkSizes(globalWorkSize, localWorkSize);
        Objects.requireNonNull(args, "args");
        for (Object arg : args) {
            if (arg == null) {
                throw new IllegalArgumentException(handle.kernelName() + ": kernel argument is null");
            }
        }
        handle.arguments(args);
        return dispatch(handle, globalWorkSize, localWorkSize);
    }

    @Override
    public void setKernelArg(KernelHandle kernel, int argIndex, Object arg) {
        kernel(kernel).argument(argIndex, Objects.requireNonNull(arg, "arg"));
    }

    @Override
    public void enqueueKernelPreBound(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                      GpuEvent... waitOnEvents) {
        enqueueKernelPreBoundWithEvent(kernel, globalWorkSize, localWorkSize, waitOnEvents);
    }

    @Override
    public GpuEvent enqueueKernelPreBoundWithEvent(KernelHandle kernel, long[] globalWorkSize,
                                                   long[] localWorkSize, GpuEvent... waitOnEvents) {
        await(waitOnEvents);
        RemoteKernelHandle handle = kernel(kernel);
        checkWorkSizes(globalWorkSize, localWorkSize);
        return dispatch(handle, globalWorkSize, localWorkSize);
    }

    /** Appends one dispatch of {@code handle}'s sticky arguments and returns its completion event. */
    private GpuEvent dispatch(RemoteKernelHandle handle, long[] globalWorkSize, long[] localWorkSize) {
        Object[] args = handle.arguments();
        RpcWire.Writer member = new RpcWire.Writer(RpcWire.OP_ENQUEUE)
                .i64(handle.id()).longArray(globalWorkSize).longArray(localWorkSize);
        RpcWire.writeArgs(member, args);
        long seq = backend.submit(member, true);
        return new RemoteGpuEvent(backend.batch(), seq);
    }

    @Override
    public GpuEvent enqueueMarker(GpuEvent... waitOnEvents) {
        await(waitOnEvents);
        return new RemoteGpuEvent(backend.batch(), backend.batch().lastSequence());
    }

    @Override
    public void flush() {
        backend.batch().flush();
    }

    /** The pipelined protocol batches dispatches into one flush, so a launch batch is one round trip. */
    @Override
    public boolean supportsKernelBatching() {
        return true;
    }

    @Override
    public KernelTaskBatch beginKernelBatch() {
        return new RemoteKernelTaskBatch(backend);
    }

    @Override
    public void finish() {
        backend.batch().finish();
    }

    /** Equivalent to {@link #finish()}, as a backend-owned queue's close is. */
    @Override
    public void close() {
        finish();
    }

    // ------------------------------------------------------------------ writes and probes

    /**
     * Appends one chunk of a write and returns its sequence. When the server keeps a weight cache and the chunk is
     * at least {@link RpcWire#HASH_THRESHOLD} bytes, a probe asks whether the cache holds the chunk by its content
     * hash; a hit writes nothing, and a miss sends the chunk with the store flag set, so a reconnect finds it. The
     * chunking of a write is a pure function of its size, so a reconnect reproduces the same chunk hashes.
     */
    private long writeChunk(RemoteBuffer dst, long offset, byte[] chunk, boolean cache) {
        if (cache && chunk.length >= RpcWire.HASH_THRESHOLD) {
            long hash = RpcWire.fnv1a(chunk);
            boolean hit = backend.batch().probeWriteHash(new RpcWire.Writer(RpcWire.OP_WRITE_HASH)
                    .i64(dst.id()).i64(offset).i64(chunk.length).rawBytes(RpcWire.hashBytes(hash)));
            if (hit) {
                return backend.batch().lastSequence();
            }
            return backend.submit(new RpcWire.Writer(RpcWire.OP_WRITE).i64(dst.id()).i64(offset)
                    .rawBytes(chunk).bool(true), true);
        }
        return backend.submit(new RpcWire.Writer(RpcWire.OP_WRITE).i64(dst.id()).i64(offset)
                .rawBytes(chunk).bool(false), true);
    }

    // ------------------------------------------------------------------ validation

    private RemoteBuffer buffer(GpuBuffer candidate, String what) {
        if (!(candidate instanceof RemoteBuffer)) {
            throw new IllegalArgumentException(what + " is not a buffer of a remote backend");
        }
        RemoteBuffer buffer = (RemoteBuffer) candidate;
        if (!backend.ownsBuffer(buffer)) {
            throw new IllegalArgumentException(what + " belongs to another remote backend");
        }
        return buffer;
    }

    private RemoteKernelHandle kernel(KernelHandle candidate) {
        if (!(candidate instanceof RemoteKernelHandle)) {
            throw new IllegalArgumentException("kernel is not a handle of a remote backend");
        }
        RemoteKernelHandle handle = (RemoteKernelHandle) candidate;
        if (!handle.belongsTo(backend)) {
            throw new IllegalArgumentException("kernel " + handle.kernelName() + " belongs to another remote"
                    + " backend");
        }
        return handle;
    }

    static void checkWorkSizes(long[] global, long[] local) {
        if (global == null || global.length < 1 || global.length > 3) {
            throw new IllegalArgumentException("a dispatch needs 1..3 global work sizes, got "
                    + (global == null ? "null" : String.valueOf(global.length)));
        }
        for (long g : global) {
            if (g <= 0) {
                throw new IllegalArgumentException("global work size " + g + " must be positive");
            }
        }
        if (local != null) {
            if (local.length != global.length) {
                throw new IllegalArgumentException(local.length + " local work sizes for " + global.length
                        + " global ones");
            }
            for (long l : local) {
                if (l <= 0) {
                    throw new IllegalArgumentException("local work size " + l + " must be positive");
                }
            }
        }
    }

    private static void checkRange(RemoteBuffer buffer, long offset, long size, long sourceBytes, String what) {
        if (offset < 0 || size < 0) {
            throw new IllegalArgumentException(what + " range [" + offset + ", " + offset + "+" + size
                    + ") must not be negative");
        }
        if (offset + size > buffer.sizeInBytes()) {
            throw new IllegalArgumentException(what + " range [" + offset + ", " + (offset + size)
                    + ") exceeds the buffer size of " + buffer.sizeInBytes() + " bytes");
        }
        if (size > sourceBytes) {
            throw new IllegalArgumentException(what + " of " + size + " bytes exceeds the " + sourceBytes
                    + "-byte array");
        }
    }

    private static void await(GpuEvent[] events) {
        if (events == null || events.length == 0) {
            return;
        }
        for (GpuEvent event : events) {
            if (event == null) {
                continue;
            }
            if (!(event instanceof RemoteGpuEvent)) {
                throw new IllegalArgumentException("a remote queue cannot wait on an event of another backend");
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
        return "RemoteComputeQueue{" + queueType + ", " + backend + '}';
    }
}
