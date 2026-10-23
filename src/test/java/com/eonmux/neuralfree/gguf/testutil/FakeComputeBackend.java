package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Host-only {@link ComputeBackend} for trace-style tests. Nothing runs on a device: every queue
 * operation executes synchronously against {@link FakeGpuBuffer} contents and is appended to one
 * operation log shared by all queues, in submission order.
 *
 * <p>Contract behaviour mirrored from {@link ComputeBackend} and {@link ComputeQueue}:
 * <ul>
 *   <li>arguments are snapshotted when set; sticky arguments live on the kernel handle; a
 *       non-empty {@code args} array replaces the handle's whole argument state, a null or empty one
 *       dispatches the stored state; a dispatch with an unset argument fails;</li>
 *   <li>{@code null} arguments, buffers or events of another backend, closed buffers, closed
 *       handles and closed events used as waits are rejected;</li>
 *   <li>writes copy at enqueue; reads complete at enqueue, which satisfies the "valid after
 *       finish()" rule; the {@code ByteBuffer} write reads bytes {@code [0, size)} without touching
 *       the caller's position or limit;</li>
 *   <li>queues from {@link #computeQueue()} and {@link #transferQueue()} are backend-owned and
 *       their {@code close()} is {@code finish()}; queues from {@link #createComputeQueue()} are
 *       caller-owned.</li>
 * </ul>
 *
 * <p>With {@link Builder#strictLlmArgs(boolean)} the LLM kernel rules are enforced as well: only
 * buffers, {@code Integer}, {@code Float}, {@code Boolean} and one-element {@code int[]}/{@code float[]}
 * holders; at most {@value #MAX_LLM_SCALARS} scalars and {@value #MAX_LLM_BUFFERS} buffers; an explicit
 * local size that divides the global size.
 *
 * <p>{@link KernelEmulator}s registered per kernel name run at dispatch time, so a test can give a
 * kernel observable effects (for example write an argmax result) without a device.
 */
public final class FakeComputeBackend implements ComputeBackend {

    public static final int MAX_LLM_SCALARS = 32;
    public static final int MAX_LLM_BUFFERS = 12;
    public static final String COMPUTE_QUEUE = "compute";
    public static final String TRANSFER_QUEUE = "transfer";

    /** Kind of a recorded queue operation. */
    public enum OpKind { DISPATCH, WRITE, READ, COPY, FILL, MARKER, FLUSH, FINISH }

    /** Host-side stand-in for a kernel's effect, invoked with the recorded dispatch. */
    @FunctionalInterface
    public interface KernelEmulator {
        void run(Dispatch dispatch);
    }

    private final Object lock = new Object();
    private final BackendType backendType;
    private final DeviceCapabilities capabilities;
    private final boolean strictLlmArgs;
    private final Set<String> unavailableKernels;
    private final Map<String, KernelEmulator> emulators;
    private final FakeQueue computeQueue;
    private final FakeQueue transferQueue;
    private final List<Op> ops = new ArrayList<>();
    private final List<FakeGpuBuffer> buffers = new ArrayList<>();
    private final Map<String, Kernel> kernelCache = new HashMap<>();
    private final List<String> compileCalls = new ArrayList<>();
    private final List<String> preparedKernels = new ArrayList<>();
    private long nextBufferId = 1;
    private long nextEventId = 1;
    private int nextSequence;
    private int createdQueues;
    private long allocatedBytes;
    private int openEvents;
    private boolean closed;

    private FakeComputeBackend(Builder b) {
        this.backendType = b.backendType;
        this.capabilities = b.capabilities;
        this.strictLlmArgs = b.strictLlmArgs;
        this.unavailableKernels = Collections.unmodifiableSet(new HashSet<>(b.unavailable));
        this.emulators = Collections.unmodifiableMap(new HashMap<>(b.emulators));
        this.computeQueue = new FakeQueue(COMPUTE_QUEUE, ComputeQueue.QueueType.COMPUTE, true);
        this.transferQueue = new FakeQueue(TRANSFER_QUEUE, ComputeQueue.QueueType.TRANSFER, true);
    }

    /** An OpenCL-typed fake with {@link #defaultCapabilities()} and lenient argument rules. */
    public static FakeComputeBackend create() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** 8 GiB device, 2 GiB max allocation, work groups up to 1024, subgroups and FP16 supported. */
    public static DeviceCapabilities defaultCapabilities() {
        return new DeviceCapabilities.Builder()
                .deviceName("FakeDevice")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(32)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .supportsAsyncTransfer(true)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024)
                .build();
    }

    // ================================================================== ComputeBackend

    @Override
    public BackendType backendType() {
        return backendType;
    }

    @Override
    public DeviceCapabilities capabilities() {
        return capabilities;
    }

    @Override
    public ComputeQueue computeQueue() {
        synchronized (lock) {
            ensureOpen();
            return computeQueue;
        }
    }

    @Override
    public ComputeQueue transferQueue() {
        synchronized (lock) {
            ensureOpen();
            return transferQueue;
        }
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
        synchronized (lock) {
            ensureOpen();
            if (sizeInBytes <= 0) {
                throw new IllegalArgumentException("buffer size must be positive, got " + sizeInBytes);
            }
            long maxAlloc = capabilities.getMaxAllocSizeBytes();
            if (maxAlloc > 0 && sizeInBytes > maxAlloc) {
                throw new GpuOutOfMemoryException("allocation exceeds the device max allocation size",
                        sizeInBytes, maxAlloc, backendType.name());
            }
            long total = capabilities.getTotalVramBytes();
            if (total > 0 && allocatedBytes + sizeInBytes > total) {
                throw new GpuOutOfMemoryException("allocation exceeds remaining device memory",
                        sizeInBytes, total - allocatedBytes, backendType.name());
            }
            FakeGpuBuffer buffer = new FakeGpuBuffer(nextBufferId++, sizeInBytes, readOnly, this, this::release);
            buffers.add(buffer);
            allocatedBytes += sizeInBytes;
            return buffer;
        }
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes) {
        return allocateBuffer(sizeInBytes, false);
    }

    /**
     * Returns a backend-owned handle, cached per (name, options, source). Kernels listed in
     * {@link Builder#unavailableKernels(String...)} fail with {@link UnsupportedKernelException}.
     */
    @Override
    public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
        Objects.requireNonNull(kernelName, "kernelName");
        Objects.requireNonNull(kernelSource, "kernelSource");
        synchronized (lock) {
            ensureOpen();
            compileCalls.add(kernelName);
            if (unavailableKernels.contains(kernelName)) {
                throw unavailable(kernelName);
            }
            String options = buildOptions == null ? "" : buildOptions;
            String key = kernelName + '\u0000' + options + '\u0000' + kernelSource.length() + ':'
                    + kernelSource.hashCode();
            Kernel cached = kernelCache.get(key);
            if (cached == null || cached.isClosed()) {
                cached = new Kernel(this, kernelName, options, false);
                kernelCache.put(key, cached);
            }
            return cached;
        }
    }

    /**
     * The artifact kind a real backend of this fake's type loads: OpenCL source on OpenCL, a module
     * image on CUDA, SPIR-V on Vulkan.
     */
    @Override
    public boolean supportsArtifact(KernelArtifact.Kind kind) {
        return kind == nativeArtifactKind(backendType);
    }

    /**
     * Returns a backend-owned handle for the artifact. OpenCL source goes through
     * {@link #compileKernel}; binary artifacts are never read, and their handles are cached per
     * (name, program key, specialisation constants) with the constants as the handle's build options.
     * Each call is recorded in {@link #compileCalls()}. An artifact kind this backend type cannot load,
     * or a kernel listed in {@link Builder#unavailableKernels(String...)}, fails with
     * {@link UnsupportedKernelException}.
     */
    @Override
    public KernelHandle loadKernel(KernelArtifact artifact) {
        if (artifact == null) {
            throw new IllegalArgumentException("kernel artifact must not be null");
        }
        if (!supportsArtifact(artifact.kind())) {
            throw new UnsupportedKernelException(backendType + " backend cannot load " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName());
        }
        if (artifact.kind() == KernelArtifact.Kind.OPENCL_SOURCE) {
            return compileKernel(artifact.kernelName(), artifact.sourceCode(), artifact.buildOptions());
        }
        String kernelName = artifact.kernelName();
        synchronized (lock) {
            ensureOpen();
            compileCalls.add(kernelName);
            if (unavailableKernels.contains(kernelName)) {
                throw unavailable(kernelName);
            }
            String options = artifact.specConstants().isEmpty() ? "" : "spec=" + artifact.specConstants();
            String key = artifact.kind().name() + '\u0000' + kernelName + '\u0000' + artifact.programKey()
                    + '\u0000' + options;
            Kernel cached = kernelCache.get(key);
            if (cached == null || cached.isClosed()) {
                cached = new Kernel(this, kernelName, options, false);
                kernelCache.put(key, cached);
            }
            return cached;
        }
    }

    private static KernelArtifact.Kind nativeArtifactKind(BackendType type) {
        switch (type) {
            case OPENCL:
                return KernelArtifact.Kind.OPENCL_SOURCE;
            case CUDA:
                return KernelArtifact.Kind.CUDA_IMAGE;
            case VULKAN:
                return KernelArtifact.Kind.SPIRV;
            default:
                throw new IllegalStateException("unknown backend type " + type);
        }
    }

    private UnsupportedKernelException unavailable(String kernelName) {
        return new UnsupportedKernelException("kernel '" + kernelName + "' is not available on " + backendType
                + " device " + capabilities.getDeviceName());
    }

    /** A caller-owned handle with the template's program and an empty argument state. */
    @Override
    public KernelHandle cloneKernel(KernelHandle template) {
        synchronized (lock) {
            ensureOpen();
            Kernel k = ownKernel(template);
            return new Kernel(this, k.kernelName(), k.buildOptions(), true);
        }
    }

    @Override
    public ComputeQueue createComputeQueue() {
        synchronized (lock) {
            ensureOpen();
            createdQueues++;
            return new FakeQueue(COMPUTE_QUEUE + "#" + createdQueues, ComputeQueue.QueueType.COMPUTE, false);
        }
    }

    @Override
    public long getAllocatedBytes() {
        synchronized (lock) {
            return allocatedBytes;
        }
    }

    /** Records a pipeline warm-up request for {@code kernel} at the given local size. */
    @Override
    public void prepareKernel(KernelHandle kernel, long[] localWorkSize) {
        synchronized (lock) {
            ensureOpen();
            Kernel k = ownKernel(kernel);
            if (localWorkSize != null) {
                for (long l : localWorkSize) {
                    if (l <= 0) {
                        throw new IllegalArgumentException("local size of " + k.kernelName() + " must be positive: "
                                + Arrays.toString(localWorkSize));
                    }
                }
            }
            preparedKernels.add(k.kernelName());
        }
    }

    /** Closes owned queues and cached handles. Idempotent. Caller-owned buffers stay as they are. */
    @Override
    public void close() {
        synchronized (lock) {
            if (closed) {
                return;
            }
            closed = true;
            for (Kernel k : kernelCache.values()) {
                k.close();
            }
            kernelCache.clear();
        }
    }

    // ================================================================== inspection

    public boolean isClosed() {
        synchronized (lock) {
            return closed;
        }
    }

    /** Every recorded operation since creation or the last {@link #clearLog()}. */
    public List<Op> ops() {
        synchronized (lock) {
            return Collections.unmodifiableList(new ArrayList<>(ops));
        }
    }

    public List<Dispatch> dispatches() {
        synchronized (lock) {
            List<Dispatch> out = new ArrayList<>();
            for (Op op : ops) {
                if (op.kind() == OpKind.DISPATCH) {
                    out.add(op.dispatch());
                }
            }
            return Collections.unmodifiableList(out);
        }
    }

    public List<Dispatch> dispatchesOf(String kernelName) {
        List<Dispatch> out = new ArrayList<>();
        for (Dispatch d : dispatches()) {
            if (d.kernelName().equals(kernelName)) {
                out.add(d);
            }
        }
        return Collections.unmodifiableList(out);
    }

    public List<String> dispatchedKernelNames() {
        List<String> out = new ArrayList<>();
        for (Dispatch d : dispatches()) {
            out.add(d.kernelName());
        }
        return Collections.unmodifiableList(out);
    }

    /** Drops the recorded operations; buffer contents, handles and sequence numbering are kept. */
    public void clearLog() {
        synchronized (lock) {
            ops.clear();
        }
    }

    /** One line per recorded operation, in submission order. */
    public String formatTrace() {
        StringBuilder sb = new StringBuilder();
        for (Op op : ops()) {
            sb.append(op).append('\n');
        }
        return sb.toString();
    }

    /** Every buffer allocated by this backend, closed ones included, in allocation order. */
    public List<FakeGpuBuffer> buffers() {
        synchronized (lock) {
            return Collections.unmodifiableList(new ArrayList<>(buffers));
        }
    }

    /** Buffers not yet closed; non-empty after the engine is closed indicates a leak. */
    public List<FakeGpuBuffer> liveBuffers() {
        List<FakeGpuBuffer> out = new ArrayList<>();
        for (FakeGpuBuffer b : buffers()) {
            if (!b.isClosed()) {
                out.add(b);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /** Kernel names passed to {@link #compileKernel}, one entry per call, cache hits included. */
    public List<String> compileCalls() {
        synchronized (lock) {
            return Collections.unmodifiableList(new ArrayList<>(compileCalls));
        }
    }

    public List<String> preparedKernels() {
        synchronized (lock) {
            return Collections.unmodifiableList(new ArrayList<>(preparedKernels));
        }
    }

    /** Events returned by this backend and not yet closed. */
    public int openEventCount() {
        synchronized (lock) {
            return openEvents;
        }
    }

    // ================================================================== internals

    private void release(long bytes) {
        synchronized (lock) {
            allocatedBytes -= bytes;
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("FakeComputeBackend is closed");
        }
    }

    private int record(Op.Builder op) {
        int seq = nextSequence++;
        ops.add(op.build(seq));
        return seq;
    }

    private GpuEvent newEvent() {
        openEvents++;
        return new FakeEvent(nextEventId++);
    }

    private Kernel ownKernel(KernelHandle handle) {
        if (!(handle instanceof Kernel) || ((Kernel) handle).owner() != this) {
            throw new IllegalArgumentException("kernel handle " + handle + " does not belong to this backend");
        }
        Kernel k = (Kernel) handle;
        if (k.isClosed()) {
            throw new IllegalStateException("kernel handle " + k.kernelName() + " is closed");
        }
        return k;
    }

    private FakeGpuBuffer ownBuffer(GpuBuffer buffer, String role) {
        if (!(buffer instanceof FakeGpuBuffer) || ((FakeGpuBuffer) buffer).owner() != this) {
            throw new IllegalArgumentException(role + " " + buffer + " does not belong to this backend");
        }
        FakeGpuBuffer b = (FakeGpuBuffer) buffer;
        if (b.isClosed()) {
            throw new IllegalStateException(role + " " + b + " is closed");
        }
        return b;
    }

    private int validateWaits(GpuEvent[] waits) {
        if (waits == null) {
            return 0;
        }
        for (int i = 0; i < waits.length; i++) {
            GpuEvent e = waits[i];
            if (e == null) {
                throw new IllegalArgumentException("wait event " + i + " is null");
            }
            if (!(e instanceof FakeEvent) || ((FakeEvent) e).owner() != this) {
                throw new IllegalArgumentException("wait event " + i + " belongs to another backend");
            }
            if (((FakeEvent) e).closed) {
                throw new IllegalArgumentException("wait event " + i + " was closed before use");
            }
        }
        return waits.length;
    }

    private Object snapshotArg(String kernel, int index, Object arg) {
        if (arg == null) {
            throw new IllegalArgumentException("arg " + index + " of " + kernel + " is null");
        }
        if (arg instanceof GpuBuffer) {
            return ownBuffer((GpuBuffer) arg, "arg " + index + " of " + kernel);
        }
        if (arg instanceof Integer || arg instanceof Float || arg instanceof Boolean) {
            return arg;
        }
        if (arg instanceof int[] || arg instanceof float[]) {
            int len = arg instanceof int[] ? ((int[]) arg).length : ((float[]) arg).length;
            if (strictLlmArgs && len != 1) {
                throw new IllegalArgumentException("arg " + index + " of " + kernel
                        + " is an array of length " + len + "; LLM kernels take only one-element holders");
            }
            return arg instanceof int[] ? ((int[]) arg).clone() : ((float[]) arg).clone();
        }
        boolean legacy = arg instanceof Long || arg instanceof Double || arg instanceof byte[]
                || arg instanceof short[] || arg instanceof long[] || arg instanceof double[];
        if (!legacy) {
            throw new IllegalArgumentException("arg " + index + " of " + kernel + " has unsupported type "
                    + arg.getClass().getName());
        }
        if (strictLlmArgs) {
            throw new IllegalArgumentException("arg " + index + " of " + kernel + " has legacy-only type "
                    + arg.getClass().getSimpleName() + "; LLM kernels take buffers, Integer, Float and Boolean");
        }
        if (arg instanceof byte[]) {
            return ((byte[]) arg).clone();
        }
        if (arg instanceof short[]) {
            return ((short[]) arg).clone();
        }
        if (arg instanceof long[]) {
            return ((long[]) arg).clone();
        }
        if (arg instanceof double[]) {
            return ((double[]) arg).clone();
        }
        return arg;
    }

    private void validateWorkSizes(String kernel, long[] global, long[] local) {
        if (global == null || global.length < 1 || global.length > 3) {
            throw new IllegalArgumentException("global size of " + kernel + " must have 1 to 3 dimensions: "
                    + Arrays.toString(global));
        }
        for (long g : global) {
            if (g <= 0) {
                throw new IllegalArgumentException("global size of " + kernel + " must be positive: "
                        + Arrays.toString(global));
            }
        }
        if (local == null) {
            if (strictLlmArgs) {
                throw new IllegalArgumentException("LLM dispatch of " + kernel + " must pass a local size");
            }
            return;
        }
        if (local.length != global.length) {
            throw new IllegalArgumentException("local size " + Arrays.toString(local) + " of " + kernel
                    + " does not match global size " + Arrays.toString(global));
        }
        long product = 1;
        for (int d = 0; d < local.length; d++) {
            if (local[d] <= 0) {
                throw new IllegalArgumentException("local size of " + kernel + " must be positive: "
                        + Arrays.toString(local));
            }
            if (strictLlmArgs && global[d] % local[d] != 0) {
                throw new IllegalArgumentException("global size " + Arrays.toString(global) + " of " + kernel
                        + " is not a multiple of local size " + Arrays.toString(local));
            }
            product *= local[d];
        }
        long maxWg = capabilities.getMaxWorkGroupSize();
        if (maxWg > 0 && product > maxWg) {
            throw new IllegalArgumentException("work group of " + kernel + " has " + product
                    + " items; the device maximum is " + maxWg);
        }
    }

    private void validateLlmArgCounts(String kernel, Object[] args) {
        int buffersBound = 0;
        for (Object a : args) {
            if (a instanceof FakeGpuBuffer) {
                buffersBound++;
            }
        }
        int scalars = args.length - buffersBound;
        if (scalars > MAX_LLM_SCALARS || buffersBound > MAX_LLM_BUFFERS) {
            throw new IllegalArgumentException(kernel + " binds " + scalars + " scalars and " + buffersBound
                    + " buffers; LLM kernels allow at most " + MAX_LLM_SCALARS + " and " + MAX_LLM_BUFFERS);
        }
    }

    // ================================================================== queue

    private final class FakeQueue implements ComputeQueue {
        private final String name;
        private final QueueType type;
        private final boolean ownedByBackend;
        private boolean queueClosed;

        FakeQueue(String name, QueueType type, boolean ownedByBackend) {
            this.name = name;
            this.type = type;
            this.ownedByBackend = ownedByBackend;
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
            synchronized (lock) {
                ensureUsable();
                Kernel k = ownKernel(kernel);
                if (argIndex < 0) {
                    throw new IllegalArgumentException("negative arg index " + argIndex + " for " + k.kernelName());
                }
                k.setArg(argIndex, snapshotArg(k.kernelName(), argIndex, arg));
            }
        }

        @Override
        public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset,
                                      long sizeInBytes, GpuEvent... waitOnEvents) {
            synchronized (lock) {
                ensureUsable();
                FakeGpuBuffer s = ownBuffer(src, "copy source");
                FakeGpuBuffer d = ownBuffer(dst, "copy destination");
                int waits = validateWaits(waitOnEvents);
                if (sizeInBytes <= 0) {
                    throw new IllegalArgumentException("copy size must be positive, got " + sizeInBytes);
                }
                d.copyFrom(s, srcOffset, dstOffset, sizeInBytes);
                record(Op.builder(OpKind.COPY, name).buffer(d, dstOffset).source(s, srcOffset)
                        .bytes(sizeInBytes).waits(waits));
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
            synchronized (lock) {
                ensureUsable();
                FakeGpuBuffer d = ownBuffer(dst, "fill destination");
                if (dstOffset < 0 || sizeInBytes < 0 || (dstOffset & 3) != 0 || (sizeInBytes & 3) != 0) {
                    throw new IllegalArgumentException("fill offset " + dstOffset + " and size " + sizeInBytes
                            + " must be non-negative multiples of 4");
                }
                if (sizeInBytes > d.sizeInBytes() - dstOffset) {
                    throw new IllegalArgumentException("fill range [" + dstOffset + ", " + dstOffset + "+"
                            + sizeInBytes + ") exceeds " + d);
                }
                int waits = validateWaits(waitOnEvents);
                d.fill(dstOffset, sizeInBytes, pattern);
                record(Op.builder(OpKind.FILL, name).buffer(d, dstOffset).bytes(sizeInBytes).pattern(pattern)
                        .waits(waits));
            }
        }

        /** Returns an event that completes after all previously enqueued work and the waits. */
        @Override
        public GpuEvent enqueueMarker(GpuEvent... waitOnEvents) {
            synchronized (lock) {
                ensureUsable();
                int waits = validateWaits(waitOnEvents);
                record(Op.builder(OpKind.MARKER, name).waits(waits));
                return newEvent();
            }
        }

        @Override
        public void flush() {
            synchronized (lock) {
                ensureUsable();
                record(Op.builder(OpKind.FLUSH, name));
            }
        }

        @Override
        public void finish() {
            synchronized (lock) {
                ensureUsable();
                record(Op.builder(OpKind.FINISH, name));
            }
        }

        @Override
        public void close() {
            synchronized (lock) {
                if (ownedByBackend) {
                    if (!closed) {
                        finish();
                    }
                    return;
                }
                if (!queueClosed && !closed) {
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
            ensureOpen();
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
            synchronized (lock) {
                ensureUsable();
                FakeGpuBuffer d = ownBuffer(dstBuffer, "write destination");
                int waits = validateWaits(waitOnEvents);
                d.write(dstOffset, bytes, 0, len);
                record(Op.builder(OpKind.WRITE, name).buffer(d, dstOffset).bytes(len).waits(waits));
                return newEvent();
            }
        }

        private GpuEvent read(GpuBuffer srcBuffer, long srcOffset, byte[] bytes, int len, GpuEvent[] waitOnEvents) {
            synchronized (lock) {
                ensureUsable();
                FakeGpuBuffer s = ownBuffer(srcBuffer, "read source");
                int waits = validateWaits(waitOnEvents);
                s.read(srcOffset, bytes, 0, len);
                record(Op.builder(OpKind.READ, name).buffer(s, srcOffset).bytes(len).waits(waits));
                return newEvent();
            }
        }

        private GpuEvent dispatch(KernelHandle kernel, long[] global, long[] local, Object[] args,
                                  boolean preBound, boolean wantEvent, GpuEvent[] waitOnEvents) {
            synchronized (lock) {
                ensureUsable();
                Kernel k = ownKernel(kernel);
                int waits = validateWaits(waitOnEvents);
                if (args != null && args.length > 0) {
                    Object[] snap = new Object[args.length];
                    for (int i = 0; i < args.length; i++) {
                        snap[i] = snapshotArg(k.kernelName(), i, args[i]);
                    }
                    k.replaceArgs(snap);
                }
                Object[] bound = k.boundArgs();
                validateWorkSizes(k.kernelName(), global, local);
                if (strictLlmArgs) {
                    validateLlmArgCounts(k.kernelName(), bound);
                }
                Dispatch d = new Dispatch(nextSequence, name, k.kernelName(), global.clone(),
                        local == null ? null : local.clone(), bound, preBound, wantEvent, waits);
                record(Op.builder(OpKind.DISPATCH, name).dispatch(d));
                KernelEmulator emulator = emulators.get(k.kernelName());
                if (emulator != null) {
                    emulator.run(d);
                }
                return wantEvent ? newEvent() : null;
            }
        }
    }

    // ================================================================== event

    private final class FakeEvent implements GpuEvent {
        private final long id;
        private boolean closed;

        FakeEvent(long id) {
            this.id = id;
        }

        FakeComputeBackend owner() {
            return FakeComputeBackend.this;
        }

        @Override
        public void awaitCompletion() {
            synchronized (lock) {
                if (closed) {
                    throw new IllegalStateException("event #" + id + " awaited after close");
                }
            }
        }

        @Override
        public boolean isComplete() {
            return true;
        }

        @Override
        public void close() {
            synchronized (lock) {
                if (!closed) {
                    closed = true;
                    openEvents--;
                }
            }
        }

        @Override
        public String toString() {
            return "event#" + id;
        }
    }

    // ================================================================== kernel handle

    /** Kernel handle with sticky, snapshotted argument state. */
    public static final class Kernel implements KernelHandle {
        private final FakeComputeBackend owner;
        private final String kernelName;
        private final String buildOptions;
        private final boolean clone;
        private final List<Object> args = new ArrayList<>();
        private boolean closed;

        Kernel(FakeComputeBackend owner, String kernelName, String buildOptions, boolean clone) {
            this.owner = owner;
            this.kernelName = kernelName;
            this.buildOptions = buildOptions;
            this.clone = clone;
        }

        @Override
        public String kernelName() {
            return kernelName;
        }

        public String buildOptions() {
            return buildOptions;
        }

        /** True for handles from {@code cloneKernel}, which are caller-owned. */
        public boolean isClone() {
            return clone;
        }

        public synchronized boolean isClosed() {
            return closed;
        }

        /** Number of argument slots currently set or reserved. */
        public synchronized int argSlots() {
            return args.size();
        }

        @Override
        public synchronized void close() {
            closed = true;
        }

        @Override
        public String toString() {
            return "Kernel[" + kernelName + (clone ? ",clone" : "") + "]";
        }

        FakeComputeBackend owner() {
            return owner;
        }

        synchronized void setArg(int index, Object snapshot) {
            while (args.size() <= index) {
                args.add(null);
            }
            args.set(index, snapshot);
        }

        synchronized void replaceArgs(Object[] snapshots) {
            args.clear();
            args.addAll(Arrays.asList(snapshots));
        }

        synchronized Object[] boundArgs() {
            for (int i = 0; i < args.size(); i++) {
                if (args.get(i) == null) {
                    throw new IllegalStateException(kernelName + " dispatched with arg " + i + " unset");
                }
            }
            return args.toArray();
        }
    }

    // ================================================================== records

    /** Immutable record of one kernel dispatch. Array arguments are returned as copies. */
    public static final class Dispatch {
        private final int sequence;
        private final String queue;
        private final String kernelName;
        private final long[] global;
        private final long[] local;
        private final Object[] args;
        private final boolean preBound;
        private final boolean returnedEvent;
        private final int waitCount;

        Dispatch(int sequence, String queue, String kernelName, long[] global, long[] local, Object[] args,
                 boolean preBound, boolean returnedEvent, int waitCount) {
            this.sequence = sequence;
            this.queue = queue;
            this.kernelName = kernelName;
            this.global = global;
            this.local = local;
            this.args = args;
            this.preBound = preBound;
            this.returnedEvent = returnedEvent;
            this.waitCount = waitCount;
        }

        public int sequence() {
            return sequence;
        }

        public String queue() {
            return queue;
        }

        public String kernelName() {
            return kernelName;
        }

        public long[] global() {
            return global.clone();
        }

        /** The local size passed by the caller, or null when the caller left it to the backend. */
        public long[] local() {
            return local == null ? null : local.clone();
        }

        public boolean isPreBound() {
            return preBound;
        }

        public boolean returnedEvent() {
            return returnedEvent;
        }

        public int waitCount() {
            return waitCount;
        }

        public int argCount() {
            return args.length;
        }

        /** The snapshot of argument {@code i}; arrays are copied. */
        public Object arg(int i) {
            return copyIfArray(args[i]);
        }

        public List<Object> args() {
            List<Object> out = new ArrayList<>(args.length);
            for (Object a : args) {
                out.add(copyIfArray(a));
            }
            return Collections.unmodifiableList(out);
        }

        /** Argument {@code i} as a 32-bit integer: Integer, Boolean (1/0) or a one-element int[]. */
        public int intArg(int i) {
            Object a = args[i];
            if (a instanceof Integer) {
                return (Integer) a;
            }
            if (a instanceof Boolean) {
                return (Boolean) a ? 1 : 0;
            }
            if (a instanceof int[] && ((int[]) a).length == 1) {
                return ((int[]) a)[0];
            }
            throw new IllegalStateException("arg " + i + " of " + kernelName + " is not an int: " + describe(a));
        }

        /** Argument {@code i} as a float: Float or a one-element float[]. */
        public float floatArg(int i) {
            Object a = args[i];
            if (a instanceof Float) {
                return (Float) a;
            }
            if (a instanceof float[] && ((float[]) a).length == 1) {
                return ((float[]) a)[0];
            }
            throw new IllegalStateException("arg " + i + " of " + kernelName + " is not a float: " + describe(a));
        }

        public FakeGpuBuffer bufferArg(int i) {
            Object a = args[i];
            if (a instanceof FakeGpuBuffer) {
                return (FakeGpuBuffer) a;
            }
            throw new IllegalStateException("arg " + i + " of " + kernelName + " is not a buffer: " + describe(a));
        }

        /** Buffer arguments in signature order, i.e. Vulkan binding order. */
        public List<FakeGpuBuffer> bufferArgs() {
            List<FakeGpuBuffer> out = new ArrayList<>();
            for (Object a : args) {
                if (a instanceof FakeGpuBuffer) {
                    out.add((FakeGpuBuffer) a);
                }
            }
            return Collections.unmodifiableList(out);
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append('#').append(sequence).append(' ').append(queue).append(" DISPATCH ").append(kernelName)
                    .append(" g=").append(Arrays.toString(global))
                    .append(" l=").append(local == null ? "auto" : Arrays.toString(local))
                    .append(" args=[");
            for (int i = 0; i < args.length; i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(describe(args[i]));
            }
            sb.append(']');
            if (preBound) {
                sb.append(" prebound");
            }
            if (waitCount > 0) {
                sb.append(" waits=").append(waitCount);
            }
            return sb.toString();
        }

        private static Object copyIfArray(Object a) {
            if (a instanceof int[]) {
                return ((int[]) a).clone();
            }
            if (a instanceof float[]) {
                return ((float[]) a).clone();
            }
            if (a instanceof byte[]) {
                return ((byte[]) a).clone();
            }
            if (a instanceof short[]) {
                return ((short[]) a).clone();
            }
            if (a instanceof long[]) {
                return ((long[]) a).clone();
            }
            if (a instanceof double[]) {
                return ((double[]) a).clone();
            }
            return a;
        }

        private static String describe(Object a) {
            if (a instanceof int[]) {
                return Arrays.toString((int[]) a);
            }
            if (a instanceof float[]) {
                return Arrays.toString((float[]) a);
            }
            if (a instanceof byte[]) {
                return Arrays.toString((byte[]) a);
            }
            if (a instanceof short[]) {
                return Arrays.toString((short[]) a);
            }
            if (a instanceof long[]) {
                return Arrays.toString((long[]) a);
            }
            if (a instanceof double[]) {
                return Arrays.toString((double[]) a);
            }
            return String.valueOf(a);
        }
    }

    /** Immutable record of one queue operation. Fields that do not apply to the kind are null or 0. */
    public static final class Op {
        private final int sequence;
        private final OpKind kind;
        private final String queue;
        private final Dispatch dispatch;
        private final FakeGpuBuffer buffer;
        private final long offset;
        private final FakeGpuBuffer source;
        private final long sourceOffset;
        private final long bytes;
        private final int pattern;
        private final int waitCount;

        private Op(Builder b, int sequence) {
            this.sequence = sequence;
            this.kind = b.kind;
            this.queue = b.queue;
            this.dispatch = b.dispatch;
            this.buffer = b.buffer;
            this.offset = b.offset;
            this.source = b.source;
            this.sourceOffset = b.sourceOffset;
            this.bytes = b.bytes;
            this.pattern = b.pattern;
            this.waitCount = b.dispatch != null ? b.dispatch.waitCount() : b.waitCount;
        }

        static Builder builder(OpKind kind, String queue) {
            return new Builder(kind, queue);
        }

        public int sequence() {
            return sequence;
        }

        public OpKind kind() {
            return kind;
        }

        public String queue() {
            return queue;
        }

        /** The dispatch for {@link OpKind#DISPATCH}, else null. */
        public Dispatch dispatch() {
            return dispatch;
        }

        /** Destination of WRITE/COPY/FILL, source of READ. */
        public FakeGpuBuffer buffer() {
            return buffer;
        }

        public long offset() {
            return offset;
        }

        /** Source of COPY. */
        public FakeGpuBuffer source() {
            return source;
        }

        public long sourceOffset() {
            return sourceOffset;
        }

        public long bytes() {
            return bytes;
        }

        /** Fill pattern of FILL. */
        public int pattern() {
            return pattern;
        }

        public int waitCount() {
            return waitCount;
        }

        @Override
        public String toString() {
            if (kind == OpKind.DISPATCH) {
                return dispatch.toString();
            }
            StringBuilder sb = new StringBuilder();
            sb.append('#').append(sequence).append(' ').append(queue).append(' ').append(kind);
            switch (kind) {
                case WRITE:
                case READ:
                    sb.append(' ').append(buffer).append(" @").append(offset).append(" +").append(bytes).append('B');
                    break;
                case COPY:
                    sb.append(' ').append(source).append(" @").append(sourceOffset).append(" -> ").append(buffer)
                            .append(" @").append(offset).append(" +").append(bytes).append('B');
                    break;
                case FILL:
                    sb.append(' ').append(buffer).append(" @").append(offset).append(" +").append(bytes)
                            .append("B pattern=0x").append(Integer.toHexString(pattern));
                    break;
                default:
                    break;
            }
            if (waitCount > 0) {
                sb.append(" waits=").append(waitCount);
            }
            return sb.toString();
        }

        static final class Builder {
            private final OpKind kind;
            private final String queue;
            private Dispatch dispatch;
            private FakeGpuBuffer buffer;
            private long offset;
            private FakeGpuBuffer source;
            private long sourceOffset;
            private long bytes;
            private int pattern;
            private int waitCount;

            private Builder(OpKind kind, String queue) {
                this.kind = kind;
                this.queue = queue;
            }

            Builder dispatch(Dispatch d) {
                this.dispatch = d;
                return this;
            }

            Builder buffer(FakeGpuBuffer b, long off) {
                this.buffer = b;
                this.offset = off;
                return this;
            }

            Builder source(FakeGpuBuffer b, long off) {
                this.source = b;
                this.sourceOffset = off;
                return this;
            }

            Builder bytes(long n) {
                this.bytes = n;
                return this;
            }

            Builder pattern(int p) {
                this.pattern = p;
                return this;
            }

            Builder waits(int n) {
                this.waitCount = n;
                return this;
            }

            Op build(int sequence) {
                return new Op(this, sequence);
            }
        }
    }

    // ================================================================== builder

    /** Configures a {@link FakeComputeBackend}. */
    public static final class Builder {
        private BackendType backendType = BackendType.OPENCL;
        private DeviceCapabilities capabilities = defaultCapabilities();
        private boolean strictLlmArgs;
        private final Set<String> unavailable = new LinkedHashSet<>();
        private final Map<String, KernelEmulator> emulators = new LinkedHashMap<>();

        private Builder() {
        }

        public Builder backendType(BackendType type) {
            this.backendType = Objects.requireNonNull(type, "backendType");
            return this;
        }

        public Builder capabilities(DeviceCapabilities caps) {
            this.capabilities = Objects.requireNonNull(caps, "capabilities");
            return this;
        }

        /** Enforces the LLM kernel argument and work-size rules on every dispatch. */
        public Builder strictLlmArgs(boolean strict) {
            this.strictLlmArgs = strict;
            return this;
        }

        /** Kernel names whose compilation fails, to exercise unavailable-kernel paths. */
        public Builder unavailableKernels(String... names) {
            for (String n : names) {
                unavailable.add(Objects.requireNonNull(n, "kernel name"));
            }
            return this;
        }

        public Builder emulate(String kernelName, KernelEmulator emulator) {
            Objects.requireNonNull(kernelName, "kernelName");
            Objects.requireNonNull(emulator, "emulator");
            if (emulators.putIfAbsent(kernelName, emulator) != null) {
                throw new IllegalArgumentException("an emulator for " + kernelName + " is already registered");
            }
            return this;
        }

        public FakeComputeBackend build() {
            return new FakeComputeBackend(this);
        }
    }
}
