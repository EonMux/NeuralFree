package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.WorkSizes;
import com.eonmux.neuralfree.acceleration.kernels.FaSmem;
import com.eonmux.neuralfree.acceleration.metal.MetalDevice;
import com.eonmux.neuralfree.acceleration.metal.MetalNative;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.SortedMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Metal backend on one Apple GPU.
 *
 * <ul>
 *   <li><b>Device.</b> One {@code MTLDevice}. Metal has no per-thread device state and no context
 *       to bind.</li>
 *   <li><b>Kernels.</b> {@link #compileKernel} compiles Metal Shading Language source with the
 *       Metal framework; {@link #loadKernel} serves
 *       {@link KernelArtifact.Kind#METAL_SOURCE} classpath artifacts the same way. One
 *       {@code MTLLibrary} is compiled per program key and constant set ({@link MetalModuleCache})
 *       and a compute pipeline is built per kernel and work-group size, because the generated MSL
 *       takes its threadgroup and shared-memory size from specialisation constants. Handles are
 *       cached and backend-owned; {@link #cloneKernel} gives caller-owned handles with their own
 *       arguments.</li>
 *   <li><b>Queues.</b> Two backend-owned command queues (compute and transfer);
 *       {@link #createComputeQueue()} adds caller-owned ones. A queue batches its commands into one
 *       {@code MTLCommandBuffer} per flush and tracks completion with a shared event
 *       ({@link MetalComputeQueue}).</li>
 *   <li><b>Memory.</b> Every allocation is a shared-storage {@code MTLBuffer}, rounded up to
 *       {@link #BUFFER_ALIGNMENT} bytes. Shared storage is CPU-visible on every Mac and coherent
 *       on Apple GPUs, which is what gives {@link #allocateHostMappedBuffer} its zero-copy weight
 *       path. {@link #getAllocatedBytes()} counts open buffers; {@link #getFreeVramBytes()} is the
 *       framework's recommended working set minus those bytes.</li>
 *   <li><b>Lifetime.</b> {@link #close()} finishes the queues, releases the shared event, every
 *       pipeline, every compiled library and every buffer still open. Idempotent.</li>
 * </ul>
 */
@NotThreadSafe("Each queue is used by one thread at a time. Allocation, kernel loading and the kernel "
        + "cache are safe from several threads.")
public final class MetalComputeBackend implements ComputeBackend {

    private static final Logger LOGGER = LoggingUtil.getLogger(MetalComputeBackend.class);

    /** Allocation granularity in bytes. */
    public static final long BUFFER_ALIGNMENT = 256L;

    private final MetalDevice device;
    private final DeviceCapabilities capabilities;
    private final MetalModuleCache modules;
    private final Map<String, MetalKernelHandle> handles = new ConcurrentHashMap<>();
    private final Map<Long, Long> liveBuffers = new ConcurrentHashMap<>();
    private final AtomicLong allocatedBytes = new AtomicLong();
    private final Map<String, Long> pipelines = new ConcurrentHashMap<>();
    private final Object pipelineLock = new Object();
    /** {@code NF_FA_SMEM_FLOATS} of the flash kernels on this device (function constant 3). */
    private final int faSmemFloats;
    private final Set<MetalComputeQueue> openQueues = new CopyOnWriteArraySet<>();
    private final MetalComputeQueue computeQueue;
    private final MetalComputeQueue transferQueue;
    private volatile boolean closed;
    private final Object closeLock = new Object();

    /**
     * Opens the backend on {@code device}.
     *
     * @throws IllegalArgumentException when {@code device} is null
     * @throws RuntimeException         the framework failure of a required object creation;
     *                                  nothing stays retained
     */
    public MetalComputeBackend(MetalDevice device) {
        if (device == null) {
            throw new IllegalArgumentException("MetalDevice cannot be null");
        }
        this.device = device;
        MetalComputeQueue compute = null;
        MetalComputeQueue transfer = null;
        try {
            this.capabilities = MetalCapabilitiesBuilder.build(device);
            this.faSmemFloats = FaSmem.faSmemFloats(BackendType.METAL, capabilities);
            this.modules = new MetalModuleCache(this);
            compute = new MetalComputeQueue(this, ComputeQueue.QueueType.COMPUTE, true);
            this.computeQueue = compute;
            openQueues.add(compute);
            transfer = new MetalComputeQueue(this, ComputeQueue.QueueType.TRANSFER, true);
            this.transferQueue = transfer;
            openQueues.add(transfer);
        } catch (RuntimeException | Error e) {
            if (transfer != null) {
                releaseOnFailure(transfer, e);
            }
            if (compute != null) {
                releaseOnFailure(compute, e);
            }
            throw e;
        }
        LOGGER.info("Metal backend initialized - Device: {} ({}), {}, working set: {} MB, "
                        + "threadgroup: {} threads, {} bytes",
                capabilities.getDeviceName(), device.getDeviceFamily(), capabilities.getDriverVersion(),
                capabilities.getTotalVramBytes() / (1024 * 1024), capabilities.getMaxWorkGroupSize(),
                capabilities.getMaxSharedMemoryBytes());
    }

    @Override
    public BackendType backendType() {
        return BackendType.METAL;
    }

    @Override
    public DeviceCapabilities capabilities() {
        return capabilities;
    }

    @Override
    public ComputeQueue computeQueue() {
        ensureOpen();
        return computeQueue;
    }

    @Override
    public ComputeQueue transferQueue() {
        ensureOpen();
        return transferQueue;
    }

    /**
     * Allocates a shared-storage buffer, rounded up to {@link #BUFFER_ALIGNMENT} bytes. The buffer
     * reports the requested size; its contents are undefined. {@code readOnly} is advisory on
     * Metal.
     *
     * @throws IllegalArgumentException when the size is not positive
     * @throws GpuOutOfMemoryException  when the framework refuses the allocation
     */
    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
        ensureOpen();
        long aligned = alignedSize(sizeInBytes);
        long handle = allocate(sizeInBytes, aligned);
        LOGGER.trace("Allocated Metal buffer: {} bytes ({} byte block), readOnly={}", sizeInBytes, aligned, readOnly);
        return new MetalGpuBuffer(this, handle, sizeInBytes, aligned);
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes) {
        return allocateBuffer(sizeInBytes, false);
    }

    @Override
    public boolean supportsHostMappedBuffers() {
        return true;
    }

    /**
     * {@inheritDoc}
     *
     * <p>A shared-storage Metal buffer is already host memory: the returned buffer exposes it as a
     * direct byte buffer and a kernel binds the same object.</p>
     */
    @Override
    public HostMappedBuffer allocateHostMappedBuffer(long sizeInBytes) {
        ensureOpen();
        long aligned = alignedSize(sizeInBytes);
        long handle = allocate(sizeInBytes, aligned);
        return new MetalHostMappedBuffer(this, handle, sizeInBytes, aligned);
    }

    /**
     * Compiles Metal Shading Language source into a cached handle. The program cache key is derived
     * from the source, the build options and the kernel name; the pipelines are built per
     * work-group size on the first prepare or dispatch.
     *
     * @param buildOptions whitespace-separated {@code -DNAME} or {@code -DNAME=VALUE} macros; null
     *                     means none
     * @throws IllegalStateException naming the Metal compile error when the source does not compile
     */
    @Override
    public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
        ensureOpen();
        Objects.requireNonNull(kernelName, "kernelName cannot be null");
        Objects.requireNonNull(kernelSource, "kernelSource cannot be null");
        String options = buildOptions == null ? "" : buildOptions;
        // The key holds the source itself: a hash collision must never reuse another program's pipeline.
        String programKey = "msl:" + options + "\u0000" + kernelSource;
        return handle(programKey, kernelName, () -> kernelSource, options, null);
    }

    /**
     * Loads a kernel from a {@link KernelArtifact.Kind#METAL_SOURCE} artifact. The MSL file is
     * compiled once per constant set (a kernel whose constants appear in constant expressions gets
     * the {@code SPIRV_CROSS_CONSTANT_ID_*} macros); the handle carries the artifact's
     * {@link com.eonmux.neuralfree.acceleration.backend.KernelSignature}, so launches are checked
     * against the shader index.
     *
     * @throws UnsupportedKernelException for an artifact of another kind
     * @throws IllegalStateException      naming the MSL compile error when a dispatch's pipeline cannot be built
     */
    @Override
    public KernelHandle loadKernel(KernelArtifact artifact) {
        ensureOpen();
        if (artifact == null) {
            throw new IllegalArgumentException("kernel artifact must not be null");
        }
        if (artifact.kind() != KernelArtifact.Kind.METAL_SOURCE) {
            throw new UnsupportedKernelException("Metal backend cannot load " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName());
        }
        return handle(artifact.programKey(), artifact.kernelName(), artifact::sourceCode,
                artifact.buildOptions(), artifact.signature());
    }

    @Override
    public boolean supportsArtifact(KernelArtifact.Kind kind) {
        return kind == KernelArtifact.Kind.METAL_SOURCE;
    }

    /**
     * Builds the pipeline a dispatch with this work-group size runs, so the first dispatch does
     * not pay for it. The local size is a function constant of the generated MSL, so each size has
     * its own pipeline.
     *
     * @throws IllegalArgumentException when the handle is not from this backend or the threadgroup
     *                                  is too large
     */
    @Override
    public void prepareKernel(KernelHandle kernel, long[] localWorkSize) {
        ensureOpen();
        MetalKernelHandle handle = ownHandle(kernel);
        if (localWorkSize == null) {
            return;
        }
        WorkSizes.validateLocal(localWorkSize, capabilities.getMaxWorkGroupSize(), handle.kernelName());
        handle.requireValidThreadgroup(localWorkSize);
        long pipeline = handle.prepare(localWorkSize.clone());
        int pipelineMax = MetalNative.pipelineMaxThreads(pipeline);
        if (pipelineMax > 0) {
            WorkSizes.validateLocal(localWorkSize, pipelineMax, handle.kernelName());
        }
    }

    @Override
    public KernelHandle cloneKernel(KernelHandle template) {
        ensureOpen();
        return ownHandle(template).cloneHandle();
    }

    @Override
    public ComputeQueue createComputeQueue() {
        ensureOpen();
        synchronized (this) {
            ensureOpen();
            MetalComputeQueue queue = new MetalComputeQueue(this, ComputeQueue.QueueType.COMPUTE, false);
            openQueues.add(queue);
            return queue;
        }
    }

    /** Bytes of shared memory backing the open buffers of this backend. */
    @Override
    public long getAllocatedBytes() {
        return allocatedBytes.get();
    }

    /**
     * The framework's recommended working set minus the bytes this backend holds. Metal has no
     * free-memory query, so other processes' use is not visible.
     */
    @Override
    public long getFreeVramBytes() {
        return Math.max(0L, capabilities.getTotalVramBytes() - allocatedBytes.get());
    }

    /** The device this backend runs on. */
    public MetalDevice device() {
        return device;
    }

    void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Metal backend on '" + device.getDeviceName() + "' is closed");
        }
    }

    boolean isClosed() {
        return closed;
    }

    /** Releases one allocation: the counter, the registry entry and the framework object. */
    void releaseAllocation(long handle, long bytes) {
        if (liveBuffers.remove(handle) != null) {
            allocatedBytes.addAndGet(-bytes);
        }
        MetalNative.releaseObject(handle);
    }

    /** Registers a caller-owned queue opened for this backend, so {@link #close()} releases it. */
    void registerQueue(MetalComputeQueue queue) {
        openQueues.add(queue);
    }

    /** Drops a queue that closed itself. */
    void unregisterQueue(MetalComputeQueue queue) {
        openQueues.remove(queue);
    }

    /**
     * Finishes and releases every queue, then the shared event, every pipeline, every compiled
     * library and every buffer still open. Idempotent; closing a buffer or queue afterwards does
     * nothing.
     */
    @Override
    public void close() {
        synchronized (closeLock) {
            if (closed) {
                return;
            }
            closed = true;
        }
        RuntimeException failure = null;
        for (MetalComputeQueue queue : new ArrayList<>(openQueues)) {
            try {
                queue.closeInternal();
            } catch (RuntimeException e) {
                failure = first(failure, e);
            }
        }
        openQueues.clear();
        synchronized (pipelineLock) {
            for (Long pipeline : new ArrayList<>(pipelines.values())) {
                try {
                    MetalNative.releaseObject(pipeline);
                } catch (RuntimeException e) {
                    failure = first(failure, e);
                }
            }
            pipelines.clear();
        }
        try {
            modules.close();
        } catch (RuntimeException e) {
            failure = first(failure, e);
        }
        for (Map.Entry<Long, Long> buffer : new ArrayList<>(liveBuffers.entrySet())) {
            try {
                MetalNative.releaseObject(buffer.getKey());
            } catch (RuntimeException e) {
                failure = first(failure, e);
            }
        }
        liveBuffers.clear();
        allocatedBytes.set(0);
        if (failure != null) {
            throw failure;
        }
    }

    private long allocate(long requested, long aligned) {
        long handle;
        try {
            handle = MetalNative.newBuffer(device.getIndex(), aligned);
        } catch (IllegalStateException e) {
            long free;
            try {
                free = device.getRecommendedMaxWorkingSet();
            } catch (RuntimeException unavailable) {
                free = -1;
            }
            throw new GpuOutOfMemoryException("Metal device '" + device.getDeviceName() + "' cannot allocate "
                    + aligned + " bytes: " + e.getMessage(), requested, free, "Metal");
        }
        liveBuffers.put(handle, aligned);
        allocatedBytes.addAndGet(aligned);
        return handle;
    }

    /**
     * The cached handle of a kernel. The MSL library and its pipeline are compiled on the first
     * dispatch or prepare for a work-group size.
     *
     * @throws UnsupportedKernelException when the source breaks the specialisation-constant
     *                                    contract the backend can supply
     */
    private KernelHandle handle(String programKey, String kernelName,
                                java.util.function.Supplier<String> source, String options,
                                KernelSignature signature) {
        String key = programKey + "\u0000" + kernelName;
        MetalKernelHandle cached = handles.get(key);
        if (cached != null) {
            return cached;
        }
        String text = source.get();
        SpecConstants constants = SpecConstants.parse(text, kernelName);
        MetalKernelHandle candidate = new MetalKernelHandle(this, programKey, kernelName, () -> text, options,
                constants, MetalKernelHandle.usesSubgroups(text), signature,
                (int) capabilities.getMaxWorkGroupSize());
        MetalKernelHandle raced = handles.putIfAbsent(key, candidate);
        return raced == null ? candidate : raced;
    }

    /**
     * The pipeline of {@code kernelName} for one dispatch's local size, built on first use.
     *
     * <p>The key is the program, the kernel and its constant values, so kernels without constants
     * share one pipeline for every size and a constant change never reuses another pipeline. MSL
     * that reads a constant in a constant expression takes it as the {@code SPIRV_CROSS_CONSTANT_ID_*}
     * macro instead of a function constant; such a library is compiled per constant set, which is
     * why the library cache key includes the macro values too.</p>
     */
    long pipeline(String programKey, String kernelName, java.util.function.Supplier<String> source,
                  String options, SpecConstants constants, long[] localWorkSize) {
        ensureOpen();
        SortedMap<Integer, Long> values = constants.values(WorkSizes.pad3(localWorkSize, 1), faSmemFloats);
        String macros = constants.macroOptions(values);
        String key = programKey + "\u0000" + kernelName + "\u0000" + valuesKey(values);
        Long cached = pipelines.get(key);
        if (cached != null) {
            return cached;
        }
        synchronized (pipelineLock) {
            cached = pipelines.get(key);
            if (cached != null) {
                return cached;
            }
            ensureOpen();
            String libraryKey = constants.hasMacros() ? programKey + "\u0000" + macros : programKey;
            long library = modules.library(libraryKey, source, withMacros(options, macros), kernelName);
            long pipeline = buildPipeline(library, kernelName, constants, constants.functionValues(values));
            pipelines.put(key, pipeline);
            return pipeline;
        }
    }

    private long buildPipeline(long library, String kernelName, SpecConstants constants,
                               SortedMap<Integer, Long> functionValues) {
        int[] ids = new int[functionValues.size()];
        int[] kinds = new int[functionValues.size()];
        long[] specValues = new long[functionValues.size()];
        Map<Integer, Character> declared = constants.kinds();
        int i = 0;
        for (Map.Entry<Integer, Long> e : functionValues.entrySet()) {
            ids[i] = e.getKey();
            kinds[i] = kindCode(declared.get(e.getKey()), kernelName, e.getKey());
            specValues[i] = e.getValue();
            i++;
        }
        return MetalNative.newComputePipelineWithConstants(device.getIndex(), library, kernelName,
                ids, kinds, specValues);
    }

    /** The artifact's compile options followed by the constant macros. */
    private static String withMacros(String options, String macros) {
        String base = options == null ? "" : options.trim();
        if (macros.isEmpty()) {
            return base;
        }
        return base.isEmpty() ? macros : base + " " + macros;
    }

    /** The function-constant value list, in SpecId order, as a stable pipeline-cache suffix. */
    private static String valuesKey(SortedMap<Integer, Long> values) {
        StringBuilder key = new StringBuilder();
        for (Map.Entry<Integer, Long> e : values.entrySet()) {
            key.append(e.getKey()).append('=').append(e.getValue()).append(',');
        }
        return key.toString();
    }

    private static int kindCode(Character kind, String kernelName, int specId) {
        if (kind == null) {
            throw new IllegalArgumentException("kernel " + kernelName + ": no declared function constant "
                    + specId);
        }
        switch (kind) {
            case 'u':
                return MetalNative.SPEC_CONSTANT_UINT;
            case 'i':
                return MetalNative.SPEC_CONSTANT_INT;
            case 'f':
                return MetalNative.SPEC_CONSTANT_FLOAT;
            default:
                throw new IllegalArgumentException("kernel " + kernelName + ": function constant " + specId
                        + " has unknown kind '" + kind + "'");
        }
    }

    private MetalKernelHandle ownHandle(KernelHandle kernel) {
        if (!(kernel instanceof MetalKernelHandle) || ((MetalKernelHandle) kernel).backend() != this) {
            throw new IllegalArgumentException("kernel " + kernel + " is not a handle of this Metal backend");
        }
        return (MetalKernelHandle) kernel;
    }

    private long alignedSize(long sizeInBytes) {
        if (sizeInBytes <= 0 || sizeInBytes > Long.MAX_VALUE - BUFFER_ALIGNMENT) {
            throw new IllegalArgumentException("sizeInBytes must be positive and representable, got " + sizeInBytes);
        }
        return ((sizeInBytes + BUFFER_ALIGNMENT - 1) / BUFFER_ALIGNMENT) * BUFFER_ALIGNMENT;
    }

    private static void releaseOnFailure(MetalComputeQueue queue, Throwable primary) {
        try {
            queue.closeInternal();
        } catch (RuntimeException secondary) {
            primary.addSuppressed(secondary);
        }
    }

    private static RuntimeException first(RuntimeException first, RuntimeException next) {
        if (first == null) {
            return next;
        }
        first.addSuppressed(next);
        return first;
    }
}
