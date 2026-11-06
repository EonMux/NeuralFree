package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.BackendResources;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.backend.WorkSizes;
import com.eonmux.neuralfree.acceleration.cache.CacheKey;
import com.eonmux.neuralfree.acceleration.cache.KernelCache;
import com.eonmux.neuralfree.acceleration.hip.HipDevice;
import com.eonmux.neuralfree.acceleration.hip.HipDriver;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArraySet;

/**
 * HIP runtime backend on one AMD device's implicit context.
 *
 * <ul>
 *   <li><b>Device.</b> The backend pins one device ordinal; every public entry point of the
 *       backend, its queues, buffers, events and handles makes that device current on the
 *       calling thread ({@code hipSetDevice}, {@link HipContext}).</li>
 *   <li><b>Kernels.</b> {@link #loadKernel(KernelArtifact)} loads prebuilt
 *       {@link KernelArtifact.Kind#HIP_IMAGE} code objects from the classpath, one module per
 *       resource ({@link HipModuleCache}). The objects are hipcc-built per gfx target, so the
 *       index of the kernel library pairs each kernel with the object that covers the device.
 *       {@link #compileKernel} refuses CUDA-style source text by name: the HIP runtime has no
 *       source compiler in its API, so kernels reach this backend only as code objects. Handles
 *       are cached and backend-owned; {@link #cloneKernel} gives caller-owned handles with their
 *       own arguments.</li>
 *   <li><b>Queues.</b> Two backend-owned non-blocking streams (compute and transfer);
 *       {@link #createComputeQueue()} adds caller-owned ones, which {@link #close()} releases
 *       when they are still open.</li>
 *   <li><b>Memory.</b> Allocations are rounded up to 256 bytes and served by a stream-ordered
 *       caching pool ({@link HipBufferPool}): closing a buffer records an event on every open
 *       queue's stream and keeps the block for reuse once those events complete, so it never
 *       waits for the device. A request reuses the smallest idle pooled block of at most twice
 *       its size and at most {@link HipBufferPool#MAX_REUSE_SLACK_BYTES} larger, or allocates
 *       fresh. Out of memory frees the pool and retries once, then frees the pools of the other
 *       backends sharing the device ({@link HipSharedPools}) and retries again.
 *       {@link #trimPool()} and {@link #close()} free the pool. {@link #getAllocatedBytes()}
 *       counts open buffers; {@link #getFreeVramBytes()} is the runtime's free memory
 *       ({@code hipMemGetInfo}) plus the bytes pooled by every backend of the device.</li>
 *   <li><b>Lifetime.</b> The implicit device context is shared with every other backend of the
 *       device, so it outlives this backend: {@link #close()} frees everything the backend
 *       handed out (open buffers, caller-owned queues with their streams and pinned staging)
 *       before it returns.</li>
 * </ul>
 */
@NotThreadSafe("Each queue is used by one thread at a time. Allocation, kernel loading and the kernel "
        + "cache are safe from several threads; every entry point makes the device current on "
        + "the calling thread.")
public final class HipComputeBackend implements ComputeBackend {

    private static final Logger LOGGER = LoggingUtil.getLogger(HipComputeBackend.class);

    /** Allocation granularity in bytes. */
    public static final long BUFFER_ALIGNMENT = 256L;

    private static final String IMAGE_KEY_PREFIX = "hip-image:";

    private final HipDevice hipDevice;
    private final HipDriver driver;
    private final HipContext context;
    private final DeviceCapabilities capabilities;
    private final int runtimeVersion;
    private final KernelCache kernelCache;
    private final HipModuleCache moduleCache;
    private final HipEventPool eventPool;
    private final HipComputeQueue computeQueue;
    private final HipComputeQueue transferQueue;
    private final HipBufferPool bufferPool;
    private final Set<HipComputeQueue> openQueues = new CopyOnWriteArraySet<>();
    private volatile boolean closed;

    /**
     * Opens the backend on {@code hipDevice}. The runtime must already be initialised, as device
     * enumeration does.
     *
     * @throws IllegalArgumentException when {@code hipDevice} is null
     * @throws RuntimeException         the translated driver failure; nothing stays retained
     */
    public HipComputeBackend(HipDevice hipDevice) {
        if (hipDevice == null) {
            throw new IllegalArgumentException("HipDevice cannot be null");
        }
        this.hipDevice = hipDevice;
        this.driver = hipDevice.getDriver();
        this.context = HipContext.of(hipDevice.getIndex());
        HipModuleCache modules = null;
        HipEventPool events = null;
        HipComputeQueue compute = null;
        HipComputeQueue transfer = null;
        try {
            context.makeCurrent(driver);
            this.capabilities = HipCapabilitiesBuilder.build(hipDevice);
            this.runtimeVersion = driver.driverVersion();
            this.kernelCache = new KernelCache();
            modules = new HipModuleCache(context, driver);
            this.moduleCache = modules;
            events = new HipEventPool(context, driver);
            this.eventPool = events;
            this.bufferPool = new HipBufferPool(new HipDeviceMemory(this));
            compute = new HipComputeQueue(this, newStream(), ComputeQueue.QueueType.COMPUTE, true);
            this.computeQueue = compute;
            openQueues.add(compute);
            transfer = new HipComputeQueue(this, newStream(), ComputeQueue.QueueType.TRANSFER, true);
            this.transferQueue = transfer;
            openQueues.add(transfer);
            HipSharedPools.register(hipDevice.getIndex(), bufferPool);
        } catch (RuntimeException e) {
            closeQuietly(transfer, compute, events, modules, e);
            throw e;
        }

        LOGGER.info("HIP backend initialized - Device: {} ({}), {}, VRAM: {} MB, Compute units: {}, "
                        + "Wavefront size: {}, Max Threads/Block: {}, Async copy engines: {}",
                capabilities.getDeviceName(),
                hipDevice.getGfxTarget(),
                capabilities.getDriverVersion(),
                capabilities.getTotalVramBytes() / (1024 * 1024),
                capabilities.getComputeUnits(),
                capabilities.getSubgroupSize(),
                capabilities.getMaxWorkGroupSize(),
                capabilities.supportsAsyncTransfer() ? "yes" : "no");
    }

    @Override
    public BackendType backendType() {
        return BackendType.HIP;
    }

    @Override
    public DeviceCapabilities capabilities() {
        return capabilities;
    }

    @Override
    public ComputeQueue computeQueue() {
        enter();
        return computeQueue;
    }

    @Override
    public ComputeQueue transferQueue() {
        enter();
        return transferQueue;
    }

    /**
     * Allocates device memory from the buffer pool. The request is rounded up to
     * {@link #BUFFER_ALIGNMENT} bytes; the block is an idle pooled block of at most twice that
     * size (and at most {@link HipBufferPool#MAX_REUSE_SLACK_BYTES} larger) or a fresh allocation
     * of exactly that size, and is tracked at its block size. The buffer reports the requested
     * size; its contents are undefined. The {@code readOnly} flag is advisory on HIP.
     *
     * @throws IllegalArgumentException when the size is not positive
     * @throws GpuOutOfMemoryException  when the device is out of memory after this backend's pool
     *                                  and those of the other backends of the device were freed
     */
    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
        enter();
        if (sizeInBytes <= 0 || sizeInBytes > Long.MAX_VALUE - BUFFER_ALIGNMENT) {
            throw new IllegalArgumentException("sizeInBytes must be positive and representable, got " + sizeInBytes);
        }
        long aligned = ((sizeInBytes + BUFFER_ALIGNMENT - 1) / BUFFER_ALIGNMENT) * BUFFER_ALIGNMENT;
        HipBufferPool.Allocation block = acquireBlock(aligned);
        LOGGER.trace("Allocated HIP buffer: {} bytes ({} byte block), readOnly={}", sizeInBytes, block.bytes(),
                readOnly);
        return new HipGpuBuffer(this, block.pointer(), sizeInBytes, block.bytes());
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes) {
        return allocateBuffer(sizeInBytes, false);
    }

    /**
     * The HIP runtime API has no source compiler, so there is nothing source text can be
     * compiled as: kernels reach this backend only as hipcc-built code objects
     * ({@code KernelArtifact.hipImage}).
     *
     * @throws UnsupportedKernelException always, naming the artifact route to use instead
     */
    @Override
    public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
        enter();
        Objects.requireNonNull(kernelName, "kernelName cannot be null");
        Objects.requireNonNull(kernelSource, "kernelSource cannot be null");
        throw new UnsupportedKernelException("HIP kernel " + kernelName + " was given source text; the HIP runtime"
                + " API has no source compiler. Build the dual-dialect kernels with hipcc into a code object"
                + " and ship it as KernelArtifact.hipImage.");
    }

    /**
     * Loads a kernel from a {@link KernelArtifact.Kind#HIP_IMAGE} artifact. The code object is
     * read from the classpath and loaded once per resource; the handle is cached per
     * (kernel, resource). The handle carries the signature its kernel library registered in
     * {@link HipKernelSignatures}, so its launches are checked against the kernel inventory.
     *
     * @throws IllegalArgumentException   when {@code artifact} is null
     * @throws UnsupportedKernelException for other artifact kinds, a missing resource or kernel,
     *                                    or an object without code for this device
     */
    @Override
    public KernelHandle loadKernel(KernelArtifact artifact) {
        enter();
        if (artifact == null) {
            throw new IllegalArgumentException("kernel artifact must not be null");
        }
        if (artifact.kind() != KernelArtifact.Kind.HIP_IMAGE) {
            throw new UnsupportedKernelException("HIP backend cannot load " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName());
        }
        String resource = artifact.resourcePath();
        CacheKey key = new CacheKey(artifact.kernelName(), IMAGE_KEY_PREFIX + artifact.programKey(), "");
        return kernelCache.getOrCompile(key, k -> {
            long module = moduleCache.get(resource);
            return newHandle(module, artifact.kernelName(), resource,
                    HipKernelSignatures.lookup(resource, artifact.kernelName()));
        });
    }

    @Override
    public boolean supportsArtifact(KernelArtifact.Kind kind) {
        return kind == KernelArtifact.Kind.HIP_IMAGE;
    }

    /**
     * Validates that {@code kernel} can launch with this local size, so a block that exceeds the
     * function's register-limited maximum is reported at warm-up instead of at dispatch.
     *
     * @throws IllegalArgumentException when the handle is not from this backend or the block is too large
     */
    @Override
    public void prepareKernel(KernelHandle kernel, long[] localWorkSize) {
        enter();
        HipKernelHandle handle = ownHandle(kernel);
        if (localWorkSize == null) {
            return;
        }
        WorkSizes.validateLocal(localWorkSize, handle.maxThreadsPerBlock(), handle.kernelName());
    }

    /**
     * A caller-owned handle on the same function with fresh, independent arguments.
     *
     * @throws IllegalArgumentException when the template is not a handle of this backend
     */
    @Override
    public KernelHandle cloneKernel(KernelHandle template) {
        enter();
        return ownHandle(template).cloneHandle();
    }

    /**
     * A caller-owned compute queue on a new non-blocking stream; the caller closes it, or
     * {@link #close()} releases it when it is still open. Until then, releasing a buffer of this
     * backend also fences the queue's stream. Creation is serialised with {@link #close()}, so a
     * queue created concurrently with it is either released by it or not created.
     *
     * @throws IllegalStateException after {@link #close()}
     */
    @Override
    public ComputeQueue createComputeQueue() {
        synchronized (this) {
            enter();
            HipComputeQueue queue = new HipComputeQueue(this, newStream(), ComputeQueue.QueueType.COMPUTE, false);
            openQueues.add(queue);
            return queue;
        }
    }

    /** Bytes of device memory backing the open buffers of this backend; pooled blocks are not counted. */
    @Override
    public long getAllocatedBytes() {
        return bufferPool.inUseBytes();
    }

    /**
     * Free device memory as the runtime reports it, including other processes' use
     * ({@code hipMemGetInfo}), plus the blocks pooled by every backend sharing this device,
     * which every allocation of this backend can reclaim.
     */
    @Override
    public long getFreeVramBytes() {
        enter();
        long[] info = driver.memGetInfo();
        return info[0] + HipSharedPools.pooledBytes(hipDevice.getIndex());
    }

    /**
     * Frees every block of the buffer pool ({@code hipFree}), synchronizing the device first
     * when a released buffer may still be in use. Open buffers are not affected.
     *
     * @return the bytes returned to the runtime
     * @throws IllegalStateException after {@link #close()}
     * @throws RuntimeException      the first translated driver failure, after every block was freed
     */
    public long trimPool() {
        enter();
        return bufferPool.trim();
    }

    /** Snapshot of the buffer pool's accounting: bytes in use and pooled, block counts and reuse counters. */
    public HipBufferPool.Stats bufferPoolStats() {
        return bufferPool.stats();
    }

    /**
     * Releases the caller-owned queues still open, finishes and destroys the backend-owned
     * queues, frees the buffer pool together with the memory of every buffer still open, drops
     * the kernel cache, unloads every module, destroys the event pool and unregisters the pool.
     * Idempotent. Buffers and queues still open become invalid; closing them afterwards does
     * nothing. Every step runs even when an earlier one fails; the first failure is rethrown at
     * the end.
     */
    @Override
    public void close() {
        RuntimeException failure = null;
        try {
            BackendResources.closeAll(this);
        } catch (RuntimeException e) {
            failure = e;
        }
        try {
            releaseResources();
        } catch (RuntimeException e) {
            if (failure == null) {
                failure = e;
            } else {
                failure.addSuppressed(e);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /** The device-side part of {@link #close()}, after the tensor runtime's per-backend resources are closed. */
    private void releaseResources() {
        synchronized (this) {
            if (closed) {
                return;
            }
            closed = true;
        }
        HipSharedPools.unregister(hipDevice.getIndex(), bufferPool);
        RuntimeException failure = null;
        try {
            context.makeCurrent(driver);
        } catch (RuntimeException e) {
            failure = e;
        }
        for (HipComputeQueue queue : new ArrayList<>(openQueues)) {
            if (!queue.isOwnedByBackend()) {
                failure = runStep("close a caller-owned queue", queue::closeInternal, failure);
            }
        }
        failure = runStep("close compute queue", computeQueue::closeInternal, failure);
        failure = runStep("close transfer queue", transferQueue::closeInternal, failure);
        long stillOpen = bufferPool.inUseBytes();
        failure = runStep("free buffer pool", bufferPool::close, failure);
        failure = runStep("clear kernel cache", kernelCache::clear, failure);
        failure = runStep("unload modules", moduleCache::close, failure);
        failure = runStep("destroy event pool", eventPool::close, failure);
        if (failure != null) {
            throw failure;
        }
        LOGGER.debug("HIP backend closed - freed {} bytes of buffers that were still open", stillOpen);
    }

    /** The device this backend runs on. */
    public HipDevice getHipDevice() {
        return hipDevice;
    }

    /** The runtime operations this backend is driven with. */
    public HipDriver driver() {
        return driver;
    }

    /** The device context every entry point makes current. */
    public HipContext context() {
        return context;
    }

    /** Kernel handle cache of {@link #loadKernel}. */
    public KernelCache getKernelCache() {
        return kernelCache;
    }

    /** Pool of timing-disabled events shared by every queue of this backend. */
    public HipEventPool eventPool() {
        return eventPool;
    }

    /** Module cache holding every loaded module of this backend. */
    public HipModuleCache moduleCache() {
        return moduleCache;
    }

    /** Installed runtime version as {@code 1000 * major + 10 * minor} ({@code hipDriverGetVersion}). */
    public int runtimeVersion() {
        return runtimeVersion;
    }

    /** True once {@link #close()} has started. */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Returns a closed buffer's block to the buffer pool, with this backend's device current.
     * Called by {@link HipGpuBuffer#close()}.
     *
     * @throws RuntimeException the translated failure of fencing the block, which was then freed
     */
    void releaseAllocation(Long pointer, long allocationBytes) {
        context.makeCurrent(driver);
        bufferPool.release(pointer, allocationBytes);
    }

    /** The queues whose streams a released buffer's fence covers. */
    Collection<HipComputeQueue> openQueues() {
        return Collections.unmodifiableSet(openQueues);
    }

    /** Called by {@link HipComputeQueue#closeInternal()}: the released queue's stream needs no more fences. */
    void onQueueReleased(HipComputeQueue queue) {
        openQueues.remove(queue);
    }

    /**
     * Checks the backend is open and makes its device current on this thread.
     *
     * @throws IllegalStateException after {@link #close()}
     */
    void enter() {
        if (closed) {
            throw new IllegalStateException("HipComputeBackend already closed");
        }
        context.makeCurrent(driver);
    }

    /**
     * A block of {@code bytes} from this backend's pool. When the device is still out of memory
     * after the pool freed itself, the pools of the other backends sharing the device are freed
     * and the allocation is retried once.
     *
     * @throws GpuOutOfMemoryException when the retry fails too, or nothing was pooled elsewhere;
     *                                 failures of freeing the other pools are suppressed on it
     */
    private HipBufferPool.Allocation acquireBlock(long bytes) {
        try {
            return bufferPool.acquire(bytes);
        } catch (GpuOutOfMemoryException first) {
            List<RuntimeException> trimFailures = new ArrayList<>();
            long freed = HipSharedPools.trimSiblings(hipDevice.getIndex(), bufferPool, trimFailures);
            if (freed == 0) {
                trimFailures.forEach(first::addSuppressed);
                throw first;
            }
            for (RuntimeException e : trimFailures) {
                LOGGER.warn("Freeing another HIP backend's buffer pool failed: {}", e.getMessage(), e);
            }
            LOGGER.debug("HIP allocation of {} bytes ran out of memory; freed {} bytes pooled by other backends "
                    + "of the device and retrying", bytes, freed);
            context.makeCurrent(driver);
            try {
                return bufferPool.acquire(bytes);
            } catch (RuntimeException second) {
                second.addSuppressed(first);
                throw second;
            }
        }
    }

    private long newStream() {
        return driver.streamCreate();
    }

    private HipKernelHandle newHandle(long module, String kernelName, String origin, KernelSignature signature) {
        long function;
        try {
            function = driver.moduleGetFunction(module, kernelName);
        } catch (UnsupportedKernelException e) {
            throw new UnsupportedKernelException("HIP kernel " + kernelName + " is not in " + origin, e);
        }
        return new HipKernelHandle(this, function, kernelName, signature);
    }

    private HipKernelHandle ownHandle(KernelHandle kernel) {
        if (!(kernel instanceof HipKernelHandle)) {
            throw new IllegalArgumentException("kernel must be a HipKernelHandle, got "
                    + (kernel == null ? "null" : kernel.getClass().getName()));
        }
        HipKernelHandle handle = (HipKernelHandle) kernel;
        if (handle.backend() != this) {
            throw new IllegalArgumentException("HIP kernel " + handle.kernelName() + " belongs to another backend");
        }
        return handle;
    }

    private static RuntimeException runStep(String what, Runnable step, RuntimeException failure) {
        try {
            step.run();
            return failure;
        } catch (RuntimeException e) {
            LOGGER.warn("HIP backend close: failed to {}: {}", what, e.getMessage());
            if (failure == null) {
                return e;
            }
            failure.addSuppressed(e);
            return failure;
        }
    }

    private void closeQuietly(HipComputeQueue transfer, HipComputeQueue compute, HipEventPool events,
                              HipModuleCache modules, RuntimeException cause) {
        RuntimeException failure = cause;
        if (transfer != null) {
            failure = runStep("close transfer queue", transfer::closeInternal, failure);
        }
        if (compute != null) {
            failure = runStep("close compute queue", compute::closeInternal, failure);
        }
        if (events != null) {
            failure = runStep("destroy event pool", events::close, failure);
        }
        if (modules != null) {
            failure = runStep("unload modules", modules::close, failure);
        }
    }
}
