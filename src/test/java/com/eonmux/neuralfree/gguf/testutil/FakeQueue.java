package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * One queue of a {@link FakeComputeBackend}: records host transfers and dispatches into the shared operation log,
 * emulates kernels through the backend's {@link FakeComputeBackend.KernelEmulator}s, and captures or replays dispatch
 * sequences on the backend's behalf. Instances belong to exactly one backend; the backend owns the compute and
 * transfer queues it hands out.
 */
final class FakeQueue implements ComputeQueue {
    private final FakeComputeBackend backend;
    private final String name;
    private final QueueType type;
    private final boolean ownedByBackend;
    private boolean queueClosed;

    FakeQueue(FakeComputeBackend backend, String name, QueueType type, boolean ownedByBackend) {
        this.backend = backend;
        this.name = name;
        this.type = type;
        this.ownedByBackend = ownedByBackend;
    }

    FakeComputeBackend owner() {
        return backend;
    }

    @Override
    public QueueType getQueueType() {
        return type;
    }

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                       byte[] srcData, GpuEvent... waitOnEvents) {
        Objects.requireNonNull(srcData, "srcData");
        if (sizeInBytes < 0 || sizeInBytes > srcData.length) {
            throw new IllegalArgumentException("write of " + sizeInBytes + " bytes from a " + srcData.length
                    + "-byte array");
        }
        return write(dstBuffer, dstOffset, srcData, (int) sizeInBytes, waitOnEvents);
    }

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                       float[] srcData, GpuEvent... waitOnEvents) {
        Objects.requireNonNull(srcData, "srcData");
        checkFloatTransfer(sizeInBytes, srcData.length);
        ByteBuffer bytes = ByteBuffer.allocate((int) sizeInBytes).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(srcData, 0, (int) (sizeInBytes / 4));
        return write(dstBuffer, dstOffset, bytes.array(), (int) sizeInBytes, waitOnEvents);
    }

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                       ByteBuffer srcData, GpuEvent... waitOnEvents) {
        Objects.requireNonNull(srcData, "srcData");
        ByteBuffer view = srcData.duplicate();
        view.clear();
        if (sizeInBytes < 0 || sizeInBytes > view.capacity()) {
            throw new IllegalArgumentException("write of " + sizeInBytes + " bytes from a " + view.capacity()
                    + "-byte ByteBuffer");
        }
        byte[] bytes = new byte[(int) sizeInBytes];
        view.get(bytes);
        return write(dstBuffer, dstOffset, bytes, bytes.length, waitOnEvents);
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes,
                                      byte[] dstData, GpuEvent... waitOnEvents) {
        Objects.requireNonNull(dstData, "dstData");
        if (sizeInBytes < 0 || sizeInBytes > dstData.length) {
            throw new IllegalArgumentException("read of " + sizeInBytes + " bytes into a " + dstData.length
                    + "-byte array");
        }
        return read(srcBuffer, srcOffset, dstData, (int) sizeInBytes, waitOnEvents);
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes,
                                      float[] dstData, GpuEvent... waitOnEvents) {
        Objects.requireNonNull(dstData, "dstData");
        checkFloatTransfer(sizeInBytes, dstData.length);
        byte[] bytes = new byte[(int) sizeInBytes];
        GpuEvent event = read(srcBuffer, srcOffset, bytes, bytes.length, waitOnEvents);
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(dstData, 0, bytes.length / 4);
        return event;
    }

    @Override
    public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                  Object[] args, GpuEvent... waitOnEvents) {
        return dispatch(kernel, globalWorkSize, localWorkSize, args, false, true, waitOnEvents);
    }

    @Override
    public void enqueueKernelNoEvent(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                     Object[] args, GpuEvent... waitOnEvents) {
        dispatch(kernel, globalWorkSize, localWorkSize, args, false, false, waitOnEvents);
    }

    @Override
    public void enqueueKernelPreBound(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                      GpuEvent... waitOnEvents) {
        dispatch(kernel, globalWorkSize, localWorkSize, null, true, false, waitOnEvents);
    }

    @Override
    public GpuEvent enqueueKernelPreBoundWithEvent(KernelHandle kernel, long[] globalWorkSize,
                                                   long[] localWorkSize, GpuEvent... waitOnEvents) {
        return dispatch(kernel, globalWorkSize, localWorkSize, null, true, true, waitOnEvents);
    }

    @Override
    public void setKernelArg(KernelHandle kernel, int argIndex, Object arg) {
        synchronized (backend.lock) {
            ensureUsable();
            FakeComputeBackend.Kernel k = backend.ownKernel(kernel);
            if (argIndex < 0) {
                throw new IllegalArgumentException("negative arg index " + argIndex + " for " + k.kernelName());
            }
            k.setArg(argIndex, backend.snapshotArg(k.kernelName(), argIndex, arg));
        }
    }

    @Override
    public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset,
                                  long sizeInBytes, GpuEvent... waitOnEvents) {
        synchronized (backend.lock) {
            ensureUsable();
            backend.uncapturable("a device copy");
            FakeGpuBuffer s = backend.ownBuffer(src, "copy source");
            FakeGpuBuffer d = backend.ownBuffer(dst, "copy destination");
            int waits = backend.validateWaits(waitOnEvents);
            if (sizeInBytes <= 0) {
                throw new IllegalArgumentException("copy size must be positive, got " + sizeInBytes);
            }
            if (s == d && srcOffset < dstOffset + sizeInBytes && dstOffset < srcOffset + sizeInBytes) {
                // OpenCL refuses overlapping ranges of one buffer, CUDA and Vulkan leave them undefined
                throw new IllegalArgumentException("copy ranges [" + srcOffset + ", " + (srcOffset + sizeInBytes)
                        + ") and [" + dstOffset + ", " + (dstOffset + sizeInBytes) + ") of one buffer overlap");
            }
            d.copyFrom(s, srcOffset, dstOffset, sizeInBytes);
            backend.record(FakeOp.builder(FakeOpKind.COPY, name)
                    .buffer(d, dstOffset).source(s, srcOffset).bytes(sizeInBytes).waits(waits));
        }
    }

    /**
     * Fills {@code [dstOffset, dstOffset+sizeInBytes)} with a repeating little-endian 32-bit
     * pattern. A negative, misaligned or out-of-bounds range fails with
     * {@link IllegalArgumentException}, as the queue contract specifies.
     */
    @Override
    public void enqueueFillBuffer(GpuBuffer dst, long dstOffset, long sizeInBytes, int pattern,
                                  GpuEvent... waitOnEvents) {
        synchronized (backend.lock) {
            ensureUsable();
            backend.uncapturable("a fill");
            FakeGpuBuffer d = backend.ownBuffer(dst, "fill destination");
            if (dstOffset < 0 || sizeInBytes < 0 || (dstOffset & 3) != 0 || (sizeInBytes & 3) != 0) {
                throw new IllegalArgumentException("fill offset " + dstOffset + " and size " + sizeInBytes
                        + " must be non-negative multiples of 4");
            }
            if (sizeInBytes > d.sizeInBytes() - dstOffset) {
                throw new IllegalArgumentException("fill range [" + dstOffset + ", " + dstOffset + "+"
                        + sizeInBytes + ") exceeds " + d);
            }
            int waits = backend.validateWaits(waitOnEvents);
            d.fill(dstOffset, sizeInBytes, pattern);
            backend.record(FakeOp.builder(FakeOpKind.FILL, name)
                    .buffer(d, dstOffset).bytes(sizeInBytes).pattern(pattern).waits(waits));
        }
    }

    /** Returns an event that completes after all previously enqueued work and the waits. */
    @Override
    public GpuEvent enqueueMarker(GpuEvent... waitOnEvents) {
        synchronized (backend.lock) {
            ensureUsable();
            backend.uncapturable("a marker");
            int waits = backend.validateWaits(waitOnEvents);
            backend.record(FakeOp.builder(FakeOpKind.MARKER, name).waits(waits));
            return backend.newEvent();
        }
    }

    @Override
    public boolean supportsGraphCapture() {
        return backend.graphCapture;
    }

    @Override
    public void beginCapture() {
        synchronized (backend.lock) {
            ensureUsable();
            if (!backend.graphCapture) {
                throw new UnsupportedOperationException("FakeQueue does not support graph capture");
            }
            if (backend.capturing || backend.replaying != null) {
                throw new IllegalStateException("FakeQueue is already "
                        + (backend.capturing ? "capturing" : "replaying") + " dispatches");
            }
            backend.capturing = true;
            backend.captured.clear();
        }
    }

    @Override
    public GpuGraph endCapture() {
        synchronized (backend.lock) {
            ensureUsable();
            if (!backend.graphCapture) {
                throw new UnsupportedOperationException("FakeQueue does not support graph capture");
            }
            if (backend.replaying != null) {
                throw new IllegalStateException("FakeQueue: endCapture needs a capture, a replay of "
                        + backend.replaying + " is dispatching");
            }
            if (!backend.capturing) {
                throw new IllegalStateException("FakeQueue is not capturing a graph");
            }
            backend.capturing = false;
            List<FakeComputeBackend.PendingDispatch> out = new ArrayList<>(backend.captured);
            backend.captured.clear();
            if (out.isEmpty()) {
                return null;
            }
            return new FakeComputeBackend.FakeGraph(backend, out);
        }
    }

    @Override
    public void abortCapture() {
        synchronized (backend.lock) {
            ensureUsable();
            if (!backend.graphCapture) {
                throw new UnsupportedOperationException("FakeQueue does not support graph capture");
            }
            if (backend.replaying != null) {
                throw new IllegalStateException("FakeQueue: abortCapture needs a capture, a replay of "
                        + backend.replaying + " is dispatching");
            }
            if (!backend.capturing) {
                return;
            }
            backend.discardCapturedLocked();
        }
    }

    @Override
    public boolean isCapturing() {
        synchronized (backend.lock) {
            return backend.capturing;
        }
    }

    @Override
    public void flush() {
        synchronized (backend.lock) {
            ensureUsable();
            backend.record(FakeOp.builder(FakeOpKind.FLUSH, name));
        }
    }

    @Override
    public void finish() {
        synchronized (backend.lock) {
            ensureUsable();
            backend.uncapturable("a finish");
            backend.record(FakeOp.builder(FakeOpKind.FINISH, name));
        }
        backend.checkWait();
    }

    @Override
    public void close() {
        synchronized (backend.lock) {
            if (ownedByBackend) {
                if (!backend.closed) {
                    finish();
                }
                return;
            }
            if (!queueClosed && !backend.closed) {
                finish();
            }
            queueClosed = true;
        }
    }

    @Override
    public String toString() {
        return "FakeQueue[" + name + "]";
    }

    private void ensureUsable() {
        backend.ensureOpen();
        if (queueClosed) {
            throw new IllegalStateException("queue " + name + " is closed");
        }
    }

    private void checkFloatTransfer(long sizeInBytes, int floats) {
        if (sizeInBytes < 0 || (sizeInBytes & 3) != 0 || sizeInBytes > floats * 4L) {
            throw new IllegalArgumentException("float transfer of " + sizeInBytes + " bytes with a " + floats
                    + "-element array; the size must be a non-negative multiple of 4 within the array");
        }
    }

    private GpuEvent write(GpuBuffer dstBuffer, long dstOffset, byte[] bytes, int len, GpuEvent[] waitOnEvents) {
        synchronized (backend.lock) {
            ensureUsable();
            backend.uncapturable("a host write");
            FakeGpuBuffer d = backend.ownBuffer(dstBuffer, "write destination");
            int waits = backend.validateWaits(waitOnEvents);
            d.write(dstOffset, bytes, 0, len);
            backend.record(FakeOp.builder(FakeOpKind.WRITE, name)
                    .buffer(d, dstOffset).bytes(len).waits(waits));
            return backend.newEvent();
        }
    }

    private GpuEvent read(GpuBuffer srcBuffer, long srcOffset, byte[] bytes, int len, GpuEvent[] waitOnEvents) {
        synchronized (backend.lock) {
            ensureUsable();
            backend.uncapturable("a host read");
            FakeGpuBuffer s = backend.ownBuffer(srcBuffer, "read source");
            int waits = backend.validateWaits(waitOnEvents);
            s.read(srcOffset, bytes, 0, len);
            backend.record(FakeOp.builder(FakeOpKind.READ, name)
                    .buffer(s, srcOffset).bytes(len).waits(waits));
            return backend.newEvent();
        }
    }

    private GpuEvent dispatch(KernelHandle kernel, long[] global, long[] local, Object[] args,
                              boolean preBound, boolean wantEvent, GpuEvent[] waitOnEvents) {
        synchronized (backend.lock) {
            ensureUsable();
            FakeComputeBackend.Kernel k = backend.ownKernel(kernel);
            int waits = backend.validateWaits(waitOnEvents);
            if (args != null && args.length > 0) {
                Object[] snap = new Object[args.length];
                for (int i = 0; i < args.length; i++) {
                    snap[i] = backend.snapshotArg(k.kernelName(), i, args[i]);
                }
                k.replaceArgs(snap);
            }
            Object[] bound = k.boundArgs();
            backend.validateWorkSizes(k.kernelName(), global, local);
            if (backend.strictLlmArgs) {
                backend.validateLlmArgCounts(k.kernelName(), bound);
            }
            if (backend.capturing) {
                if (wantEvent || waits > 0) {
                    throw new IllegalStateException("a " + k.kernelName() + " dispatch that "
                            + (wantEvent ? "returns an event" : "waits on events") + " cannot be captured:"
                            + " enqueue it before beginCapture, or return no event and wait on this queue's"
                            + " order");
                }
                backend.captured.add(new FakeComputeBackend.PendingDispatch(k, global.clone(),
                        local == null ? null : local.clone(), backend.copyArgs(bound), preBound));
                return null;
            } else if (backend.replaying != null) {
                if (wantEvent || waits > 0) {
                    throw new GpuGraphMismatchException("command replay of " + backend.replaying + ": "
                            + k.kernelName() + (wantEvent ? " returns an event" : " waits on another queue's"
                            + " events") + ", which the captured dispatches do not contain");
                }
                backend.verifyReplayLocked(backend.replaying, k, global, local, bound);
                return null;
            }
            FakeDispatch d = new FakeDispatch(backend.nextSequence, name,
                    k.kernelName(), global.clone(), local == null ? null : local.clone(), bound, preBound,
                    wantEvent, waits);
            backend.record(FakeOp.builder(FakeOpKind.DISPATCH, name).dispatch(d));
            FakeComputeBackend.KernelEmulator emulator = backend.emulators.get(k.kernelName());
            if (emulator != null) {
                emulator.run(d);
            }
            return wantEvent ? backend.newEvent() : null;
        }
    }
}
