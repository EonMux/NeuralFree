package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchPlan;
import com.eonmux.neuralfree.acceleration.backend.GpuUncapturableCommandException;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
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

    /** PCI vendor id of the vendor the default fake device claims. */
    public static final int NVIDIA_VENDOR_ID = 0x10DE;
    /** PCI vendor id of Intel, whose integrated GPU the Vulkan fakes claim. */
    public static final int INTEL_VENDOR_ID = 0x8086;

    public static final int MAX_LLM_SCALARS = 33;
    public static final int MAX_LLM_BUFFERS = 13;
    public static final String COMPUTE_QUEUE = "compute";
    public static final String TRANSFER_QUEUE = "transfer";

    /** Host-side stand-in for a kernel's effect, invoked with the recorded dispatch. */
    @FunctionalInterface
    public interface KernelEmulator {
        void run(FakeDispatch dispatch);
    }

    final Object lock = new Object();
    private final BackendType backendType;
    private final DeviceCapabilities capabilities;
    final boolean strictLlmArgs;
    final boolean graphCapture;
    final boolean graphArgUpdates;
    private final boolean parallelUploadStaging;
    private final Set<String> unavailableKernels;
    final Map<String, KernelEmulator> emulators;
    private final FakeQueue computeQueue;
    private final FakeQueue transferQueue;
    private final List<FakeOp> ops = new ArrayList<>();
    private final List<FakeGpuBuffer> buffers = new ArrayList<>();
    private final Map<String, Kernel> kernelCache = new HashMap<>();
    private final List<String> compileCalls = new ArrayList<>();
    private final List<String> preparedKernels = new ArrayList<>();
    final List<PendingDispatch> captured = new ArrayList<>();
    FakeGraph replaying;
    boolean capturing;
    private long nextBufferId = 1;
    private long nextEventId = 1;
    int nextSequence;
    private int createdQueues;
    private long allocatedBytes;
    private int openEvents;
    /** What every event wait and queue finish throws while set ({@link #failWaits}), or null. */
    private java.util.function.Supplier<? extends RuntimeException> waitFailure;
    boolean closed;

    private FakeComputeBackend(Builder b) {
        this.backendType = b.backendType;
        this.capabilities = b.capabilities;
        this.strictLlmArgs = b.strictLlmArgs;
        this.graphCapture = b.graphCapture;
        this.graphArgUpdates = b.graphArgUpdates;
        this.parallelUploadStaging = b.parallelUploadStaging;
        this.unavailableKernels = Collections.unmodifiableSet(new HashSet<>(b.unavailable));
        this.emulators = Collections.unmodifiableMap(new HashMap<>(b.emulators));
        this.computeQueue = new FakeQueue(this, COMPUTE_QUEUE, ComputeQueue.QueueType.COMPUTE, true);
        this.transferQueue = new FakeQueue(this, TRANSFER_QUEUE, ComputeQueue.QueueType.TRANSFER, true);
    }

    /** An OpenCL-typed fake with {@link #defaultCapabilities()} and lenient argument rules. */
    public static FakeComputeBackend create() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** 8 GiB NVIDIA discrete device, 2 GiB max allocation, work groups up to 1024, subgroups and FP16 supported. */
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
                .deviceClass(DeviceClass.DISCRETE)
                .identity(new DeviceIdentity("FakeDevice", NVIDIA_VENDOR_ID, null, null))
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

    /** The fake holds every buffer in host memory, so the host-mapped tier needs no device. */
    @Override
    public boolean supportsHostMappedBuffers() {
        return true;
    }

    @Override
    public boolean prefersParallelUploadStaging() {
        return parallelUploadStaging;
    }

    @Override
    public HostMappedBuffer allocateHostMappedBuffer(long sizeInBytes) {
        synchronized (lock) {
            ensureOpen();
            if (sizeInBytes <= 0) {
                throw new IllegalArgumentException("buffer size must be positive, got " + sizeInBytes);
            }
            FakeGpuBuffer buffer = new FakeGpuBuffer(nextBufferId++, sizeInBytes, true, this, null);
            buffers.add(buffer);
            return buffer;
        }
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
            return new FakeQueue(this, COMPUTE_QUEUE + "#" + createdQueues, ComputeQueue.QueueType.COMPUTE, false);
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
    public List<FakeOp> ops() {
        synchronized (lock) {
            return Collections.unmodifiableList(new ArrayList<>(ops));
        }
    }

    public List<FakeDispatch> dispatches() {
        synchronized (lock) {
            List<FakeDispatch> out = new ArrayList<>();
            for (FakeOp op : ops) {
                if (op.kind() == FakeOpKind.DISPATCH) {
                    out.add(op.dispatch());
                }
            }
            return Collections.unmodifiableList(out);
        }
    }

    public List<FakeDispatch> dispatchesOf(String kernelName) {
        List<FakeDispatch> out = new ArrayList<>();
        for (FakeDispatch d : dispatches()) {
            if (d.kernelName().equals(kernelName)) {
                out.add(d);
            }
        }
        return Collections.unmodifiableList(out);
    }

    public List<String> dispatchedKernelNames() {
        List<String> out = new ArrayList<>();
        for (FakeDispatch d : dispatches()) {
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
        for (FakeOp op : ops()) {
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

    /**
     * Makes every later event wait and queue finish throw a new exception from {@code failure}, as a lost device
     * reports its commands; null makes them succeed again. Commands still run when enqueued.
     */
    public void failWaits(java.util.function.Supplier<? extends RuntimeException> failure) {
        synchronized (lock) {
            waitFailure = failure;
        }
    }

    /** Throws the {@link #failWaits} failure while one is set. */
    void checkWait() {
        java.util.function.Supplier<? extends RuntimeException> failure;
        synchronized (lock) {
            failure = waitFailure;
        }
        if (failure != null) {
            throw failure.get();
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

    /**
     * Refuses a command the captured dispatch sequence cannot hold: a host transfer, a read, a device
     * copy or fill, a marker or a finish while capturing, and any command outside the capture during a
     * replay.
     */
    void uncapturable(String op) {
        if (!capturing && replaying == null) {
            return;
        }
        if (replaying != null) {
            throw new GpuGraphMismatchException("command replay of " + replaying + " hit " + op
                    + ", which the captured dispatches do not contain");
        }
        throw new GpuUncapturableCommandException("a " + op + " cannot be captured: perform host transfers and"
                + " reads before beginCapture, and read results only after the graph has been launched");
    }

    /** Discards the dispatches recorded so far; the failed step leaves no device work behind. */
    void discardCapturedLocked() {
        capturing = false;
        captured.clear();
    }

    /** Runs recorded dispatches against the buffers and emulators; the REPLAY op covers them in the log. */
    private void executeCapturedLocked(List<PendingDispatch> dispatches) {
        for (PendingDispatch p : dispatches) {
            FakeDispatch d = new FakeDispatch(nextSequence++, COMPUTE_QUEUE, p.kernel.kernelName(), p.global.clone(),
                    p.local == null ? null : p.local.clone(), p.args, p.preBound, false, 0);
            KernelEmulator emulator = emulators.get(p.kernel.kernelName());
            if (emulator != null) {
                emulator.run(d);
            }
        }
    }

    private void beginReplayLocked(FakeGraph graph) {
        if (capturing || replaying != null) {
            throw new IllegalStateException("FakeComputeBackend is already "
                    + (capturing ? "capturing" : "replaying") + " dispatches");
        }
        if (graph.closed) {
            throw new IllegalStateException(graph + " is closed");
        }
        replaying = graph;
        graph.position = 0;
        graph.changed.clear();
    }

    private void submitReplayLocked(FakeGraph graph) {
        if (replaying != graph) {
            throw new IllegalStateException("FakeComputeBackend: " + graph + " is not replaying");
        }
        record(FakeOp.builder(FakeOpKind.REPLAY, COMPUTE_QUEUE).bytes(graph.dispatches.size()));
        executeCapturedLocked(graph.dispatches);
    }

    /** One launch of a freshly captured graph, the step that recorded it keeping its device work. */
    private void launchCapturedGraphLocked(FakeGraph graph) {
        if (capturing || replaying != null) {
            throw new IllegalStateException("FakeComputeBackend is already "
                    + (capturing ? "capturing" : "replaying") + " dispatches");
        }
        if (graph.closed) {
            throw new IllegalStateException(graph + " is closed");
        }
        record(FakeOp.builder(FakeOpKind.LAUNCH, COMPUTE_QUEUE).bytes(graph.dispatches.size()));
        executeCapturedLocked(graph.dispatches);
    }

    private void endReplayLocked(FakeGraph graph) {
        if (replaying != graph) {
            throw new IllegalStateException("FakeComputeBackend: " + graph + " is not replaying");
        }
        replaying = null;
    }

    void verifyReplayLocked(FakeGraph graph, Kernel k, long[] global, long[] local, Object[] bound) {
        if (graph.position >= graph.dispatches.size()) {
            throw new GpuGraphMismatchException("command replay of " + graph + ": " + k.kernelName()
                    + " enqueues more dispatches than the " + graph.dispatches.size() + " captured");
        }
        PendingDispatch p = graph.dispatches.get(graph.position++);
        if (p.kernel != k) {
            throw new GpuGraphMismatchException("command replay of " + graph + ": dispatch " + graph.position
                    + " is " + k.kernelName() + " but the capture holds " + p.kernel.kernelName() + " at this point");
        }
        boolean updated = false;
        if (!Arrays.equals(p.global, global)) {
            if (!graph.argUpdates || p.global.length != global.length) {
                throw new GpuGraphMismatchException("command replay of " + graph + ": " + k.kernelName()
                        + " changed its work sizes from global " + Arrays.toString(p.global) + " local "
                        + Arrays.toString(p.local) + " to global " + Arrays.toString(global) + " local "
                        + Arrays.toString(local) + "; re-capture the step");
            }
            System.arraycopy(global, 0, p.global, 0, global.length);
            updated = true;
        }
        if (!Arrays.equals(p.local, local)) {
            throw new GpuGraphMismatchException("command replay of " + graph + ": " + k.kernelName()
                    + " changed its local size from " + Arrays.toString(p.local) + " to "
                    + Arrays.toString(local) + "; re-capture the step");
        }
        if (p.args.length != bound.length) {
            throw new GpuGraphMismatchException("command replay of " + graph + ": " + k.kernelName() + " binds "
                    + bound.length + " arguments, the capture holds " + p.args.length);
        }
        for (int i = 0; i < bound.length; i++) {
            if (p.args[i] instanceof FakeGpuBuffer && ((FakeGpuBuffer) p.args[i]).isClosed()) {
                throw new IllegalStateException(k.kernelName() + ": bound buffer " + p.args[i] + " is closed while"
                        + " its command graph is open; close the graph first");
            }
            if (KernelArgs.sameValue(p.args[i], bound[i])) {
                continue;
            }
            if (!graph.argUpdates) {
                throw new GpuGraphMismatchException("command replay of " + graph + ": " + k.kernelName()
                        + " argument " + i + " changed since the capture; the recorded arguments are fixed,"
                        + " re-capture the step");
            }
            p.args[i] = KernelArgs.snapshot(bound[i], p.args[i]);
            updated = true;
        }
        if (updated) {
            graph.changed.add(graph.position - 1);
        }
    }

    static Object[] copyArgs(Object[] args) {
        Object[] out = new Object[args.length];
        for (int i = 0; i < args.length; i++) {
            Object v = args[i];
            out[i] = v instanceof int[] ? ((int[]) v).clone()
                    : v instanceof float[] ? ((float[]) v).clone()
                    : v instanceof long[] ? ((long[]) v).clone()
                    : v instanceof double[] ? ((double[]) v).clone()
                    : v instanceof byte[] ? ((byte[]) v).clone()
                    : v instanceof short[] ? ((short[]) v).clone()
                    : v;
        }
        return out;
    }

    /** One dispatch recorded by a capture: its handle, launch sizes and owned argument snapshots. */
    static final class PendingDispatch {
        final Kernel kernel;
        final long[] global;
        final long[] local;
        final Object[] args;
        final boolean preBound;

        PendingDispatch(Kernel kernel, long[] global, long[] local, Object[] args, boolean preBound) {
            this.kernel = kernel;
            this.global = global;
            this.local = local;
            this.args = args;
            this.preBound = preBound;
        }
    }

    /**
     * A captured dispatch sequence of the fake queue, replayed by re-running the captured dispatches
     * with their emulators after the replay body's dispatches verified against the capture: the
     * host-only stand-in of a CUDA graph or a pre-recorded Vulkan command buffer. When the backend was
     * built with {@link Builder#graphArgUpdates(boolean)} the body's argument and work-size updates are
     * written into the recorded dispatches (the CUDA semantics) and a patch plan can set the recorded
     * values without a body; without it the recorded arguments are fixed (the verify-only semantics).
     * The graph launches on any open queue of the fake backend.
     */
    public static final class FakeGraph implements GpuGraph {

        private final FakeComputeBackend backend;
        private final List<PendingDispatch> dispatches;
        /** Kernel-dispatch positions whose recorded dispatch the current replay body updated so far. */
        private final List<Integer> changed = new ArrayList<>();
        private final boolean argUpdates;
        private int position;
        boolean closed;

        FakeGraph(FakeComputeBackend backend, List<PendingDispatch> dispatches) {
            this.backend = backend;
            this.dispatches = dispatches;
            this.argUpdates = backend.graphArgUpdates;
        }

        @Override
        public int dispatchCount() {
            synchronized (backend.lock) {
                return dispatches.size();
            }
        }

        @Override
        public GpuEvent launch(ComputeQueue queue, GpuEvent... waitOnEvents) {
            synchronized (backend.lock) {
                backend.ensureOpen();
                backend.requireOwnQueue(queue);
                backend.validateWaits(waitOnEvents);
                backend.launchCapturedGraphLocked(this);
                // like the device backends, the launch hands the caller an event it must close
                return backend.newEvent();
            }
        }

        @Override
        public void replay(Runnable body) {
            if (body == null) {
                throw new IllegalArgumentException("replay body must not be null");
            }
            synchronized (backend.lock) {
                backend.ensureOpen();
                backend.beginReplayLocked(this);
            }
            boolean submitted = false;
            try {
                body.run();
                synchronized (backend.lock) {
                    backend.ensureOpen();
                    if (position != dispatches.size()) {
                        throw new GpuGraphMismatchException("command replay of " + this + " enqueues "
                                + position + " dispatches, the capture holds " + dispatches.size());
                    }
                    backend.submitReplayLocked(this);
                }
                submitted = true;
            } finally {
                synchronized (backend.lock) {
                    backend.endReplayLocked(this);
                }
            }
        }

        @Override
        public List<GpuGraphDispatch> kernelDispatches() {
            synchronized (backend.lock) {
                requireOpenLocked("report its dispatches");
                List<GpuGraphDispatch> out = new ArrayList<>(dispatches.size());
                for (int i = 0; i < dispatches.size(); i++) {
                    out.add(dispatchOfLocked(dispatches.get(i), i));
                }
                return out;
            }
        }

        @Override
        public GpuGraphDispatch kernelDispatch(int index) {
            synchronized (backend.lock) {
                requireOpenLocked("report a dispatch");
                return dispatchOfLocked(dispatch(dispatches, index), index);
            }
        }

        @Override
        public int[] takeReplayChanges() {
            synchronized (backend.lock) {
                if (changed.isEmpty()) {
                    return new int[0];
                }
                int[] out = new int[changed.size()];
                for (int i = 0; i < out.length; i++) {
                    out[i] = changed.get(i);
                }
                changed.clear();
                return out;
            }
        }

        @Override
        public boolean supportsPatchedReplay() {
            synchronized (backend.lock) {
                return argUpdates;
            }
        }

        @Override
        public boolean replayPatched(GpuGraphPatchPlan plan, int pos) {
            if (plan == null) {
                throw new IllegalArgumentException("patch plan must not be null");
            }
            synchronized (backend.lock) {
                backend.ensureOpen();
                if (!argUpdates) {
                    return false;
                }
                backend.beginReplayLocked(this);
                boolean submitted = false;
                try {
                    boolean patched = plan.patchStep(pos, this::patchDispatchLocked);
                    if (!patched) {
                        return false;
                    }
                    backend.submitReplayLocked(this);
                    submitted = true;
                } finally {
                    backend.endReplayLocked(this);
                }
                return true;
            }
        }

        /** Writes one plan patch into the recorded dispatch: its changed arguments and work size. */
        private void patchDispatchLocked(int recordIndex, long[] grid, int[] argIndices, Object[] values) {
            PendingDispatch p = dispatch(dispatches, recordIndex);
            if (grid != null && !Arrays.equals(p.global, grid)) {
                if (p.global.length != grid.length) {
                    throw new GpuGraphMismatchException("command patch of " + this + ": " + p.kernel.kernelName()
                            + " changed its work-size dimensions from " + Arrays.toString(p.global) + " to "
                            + Arrays.toString(grid) + "; re-capture the step");
                }
                System.arraycopy(grid, 0, p.global, 0, grid.length);
            }
            for (int i = 0; i < argIndices.length; i++) {
                if (!KernelArgs.sameValue(p.args[argIndices[i]], values[i])) {
                    p.args[argIndices[i]] = KernelArgs.snapshot(values[i], p.args[argIndices[i]]);
                }
            }
        }

        /** The current state of one recorded dispatch. */
        private GpuGraphDispatch dispatchOfLocked(PendingDispatch p, int index) {
            return new GpuGraphDispatch(index, p.kernel.kernelName(), p.global, p.local, p.args);
        }

        private static PendingDispatch dispatch(List<PendingDispatch> dispatches, int index) {
            if (index < 0 || index >= dispatches.size()) {
                throw new IllegalArgumentException("the capture holds " + dispatches.size() + " dispatches,"
                        + " no dispatch " + index);
            }
            return dispatches.get(index);
        }

        /** Throws when the graph is closed; {@code what} names the refused operation. */
        private void requireOpenLocked(String what) {
            if (closed) {
                throw new IllegalStateException("command graph cannot " + what + ": " + this + " is closed");
            }
        }

        @Override
        public boolean isClosed() {
            synchronized (backend.lock) {
                return closed;
            }
        }

        @Override
        public void close() {
            synchronized (backend.lock) {
                if (backend.replaying == this) {
                    throw new IllegalStateException(this + " is replaying");
                }
                closed = true;
            }
        }

        @Override
        public String toString() {
            return "FakeGraph{" + dispatches.size() + " dispatches" + (closed ? ", closed" : "") + "}";
        }
    }

    /** Rejects a queue of another backend, naming it. */
    private void requireOwnQueue(ComputeQueue queue) {
        if (!(queue instanceof FakeQueue) || ((FakeQueue) queue).owner() != this) {
            throw new IllegalArgumentException("queue " + queue + " does not belong to this fake backend");
        }
    }

    void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("FakeComputeBackend is closed");
        }
    }

    int record(FakeOp.Builder op) {
        int seq = nextSequence++;
        ops.add(op.build(seq));
        return seq;
    }

    GpuEvent newEvent() {
        openEvents++;
        return new FakeEvent(nextEventId++);
    }

    Kernel ownKernel(KernelHandle handle) {
        if (!(handle instanceof Kernel) || ((Kernel) handle).owner() != this) {
            throw new IllegalArgumentException("kernel handle " + handle + " does not belong to this backend");
        }
        Kernel k = (Kernel) handle;
        if (k.isClosed()) {
            throw new IllegalStateException("kernel handle " + k.kernelName() + " is closed");
        }
        return k;
    }

    FakeGpuBuffer ownBuffer(GpuBuffer buffer, String role) {
        if (!(buffer instanceof FakeGpuBuffer) || ((FakeGpuBuffer) buffer).owner() != this) {
            throw new IllegalArgumentException(role + " " + buffer + " does not belong to this backend");
        }
        FakeGpuBuffer b = (FakeGpuBuffer) buffer;
        if (b.isClosed()) {
            throw new IllegalStateException(role + " " + b + " is closed");
        }
        return b;
    }

    int validateWaits(GpuEvent[] waits) {
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

    Object snapshotArg(String kernel, int index, Object arg) {
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

    void validateWorkSizes(String kernel, long[] global, long[] local) {
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

    void validateLlmArgCounts(String kernel, Object[] args) {
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

    // ================================================================== event

    private final class FakeEvent implements GpuEvent {
        private final long id;
        boolean closed;

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
            checkWait();
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
        boolean closed;

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

    // ================================================================== builder

    /** Configures a {@link FakeComputeBackend}. */
    public static final class Builder {
        private BackendType backendType = BackendType.OPENCL;
        private DeviceCapabilities capabilities = defaultCapabilities();
        private boolean strictLlmArgs;
        private boolean graphCapture;
        private boolean graphArgUpdates;
        private boolean parallelUploadStaging = true;
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

        /** Lets the queues capture kernel dispatches into a replayable {@link FakeGraph}. */
        public Builder graphCapture(boolean supported) {
            this.graphCapture = supported;
            return this;
        }

        /**
         * Lets captured graphs take a replay body's argument and work-size updates (the CUDA/HIP
         * semantics) and apply a patch plan's values without a body, instead of refusing every change
         * like a verify-only command buffer. Without it the recorded arguments are fixed.
         */
        public Builder graphArgUpdates(boolean supported) {
            this.graphArgUpdates = supported;
            return this;
        }

        /**
         * Whether the backend prefers uploads staged through page-locked memory filled by several threads
         * ({@link ComputeBackend#prefersParallelUploadStaging}); true by default, like CUDA.
         */
        public Builder parallelUploadStaging(boolean prefers) {
            this.parallelUploadStaging = prefers;
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
