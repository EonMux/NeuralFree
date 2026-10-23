package com.eonmux.neuralfree.acceleration.backend.cuda;

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
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArraySet;

import static jcuda.driver.JCudaDriver.cuMemGetInfo;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuStreamCreate;

/**
 * CUDA driver-API backend on one device's primary context.
 *
 * <ul>
 *   <li><b>Context.</b> The device's primary context is retained for the backend's
 *       lifetime ({@link CudaContext}) and made current on the calling thread at every
 *       public entry point of the backend, its queues, buffers, events and handles.</li>
 *   <li><b>Kernels.</b> {@link #loadKernel(KernelArtifact)} loads prebuilt
 *       {@link KernelArtifact.Kind#CUDA_IMAGE} fatbins, cubins or PTX from the classpath,
 *       one module per resource ({@link CudaModuleCache}). {@link #compileKernel} keeps its
 *       legacy meaning: the source is PTX text, loaded once per distinct text. Runtime
 *       compilation of CUDA C is not available (NVRTC is not used). Handles are cached
 *       and backend-owned; {@link #cloneKernel} gives caller-owned handles with their own
 *       arguments.</li>
 *   <li><b>Queues.</b> Two backend-owned non-blocking streams (compute and transfer);
 *       {@link #createComputeQueue()} adds caller-owned ones, which {@link #close()} releases
 *       when they are still open. CUDA does not report {@code CONCURRENT_QUEUES_PROFITABLE}.</li>
 *   <li><b>Memory.</b> Allocations are rounded up to 256 bytes and served by a stream-ordered
 *       caching pool ({@link CudaBufferPool}): closing a buffer records an event on every open
 *       queue's stream and keeps the block for reuse once those events complete, so it never
 *       waits for the device. A request reuses the smallest idle pooled block of at most twice
 *       its size and at most {@link CudaBufferPool#MAX_REUSE_SLACK_BYTES} larger, or allocates
 *       fresh. Out of memory frees the pool and retries once, then frees the pools of the other
 *       backends sharing the device's primary context ({@link CudaSharedPools}) and retries
 *       again. {@link #trimPool()} and {@link #close()} free the pool.
 *       {@link #getAllocatedBytes()} counts open buffers; {@link #getFreeVramBytes()} is the
 *       driver's free memory ({@code cuMemGetInfo}) plus the bytes pooled by every backend of
 *       the context.</li>
 *   <li><b>Lifetime.</b> The primary context is shared with every other backend of the device,
 *       so it outlives this backend: {@link #close()} frees everything the backend handed out
 *       (open buffers, caller-owned queues with their streams and pinned staging) before it
 *       releases its retain.</li>
 * </ul>
 */
@NotThreadSafe("Each queue is used by one thread at a time. Allocation, kernel loading and the kernel "
        + "cache are safe from several threads; every entry point makes the primary context current "
        + "on the calling thread.")
public final class CudaComputeBackend implements ComputeBackend {

    private static final Logger LOGGER = LoggingUtil.getLogger(CudaComputeBackend.class);

    /** Allocation granularity in bytes. */
    public static final long BUFFER_ALIGNMENT = 256L;

    private static final String IMAGE_KEY_PREFIX = "cuda-image:";
    private static final String PTX_KEY_PREFIX = "cuda-ptx:";

    private final CUDevice cuDevice;
    private final CudaContext context;
    private final DeviceCapabilities capabilities;
    private final int driverVersion;
    private final KernelCache kernelCache;
    private final CudaModuleCache moduleCache;
    private final CudaEventPool eventPool;
    private final CudaComputeQueue computeQueue;
    private final CudaComputeQueue transferQueue;
    private final CudaBufferPool bufferPool;
    private final Set<CudaComputeQueue> openQueues = new CopyOnWriteArraySet<>();
    private volatile boolean closed;

    /**
     * Opens the backend on {@code cuDevice}'s primary context. The driver must already be
     * initialised ({@code cuInit}), as device enumeration does.
     *
     * @throws IllegalArgumentException when {@code cuDevice} is null
     * @throws RuntimeException         the translated driver failure; nothing stays retained
     */
    public CudaComputeBackend(CUDevice cuDevice) {
        if (cuDevice == null) {
            throw new IllegalArgumentException("CUDevice cannot be null");
        }
        this.cuDevice = cuDevice;
        this.context = CudaContext.retainPrimary(cuDevice.getDeviceId());
        CudaModuleCache modules = null;
        CudaEventPool events = null;
        CudaComputeQueue compute = null;
        CudaComputeQueue transfer = null;
        try {
            context.makeCurrent();
            this.capabilities = CudaCapabilitiesBuilder.build(cuDevice);
            this.driverVersion = CudaCapabilitiesBuilder.driverVersion();
            this.kernelCache = new KernelCache();
            modules = new CudaModuleCache(context);
            this.moduleCache = modules;
            events = new CudaEventPool(context);
            this.eventPool = events;
            this.bufferPool = new CudaBufferPool(new CudaDeviceMemory(this));
            compute = new CudaComputeQueue(this, newStream(), ComputeQueue.QueueType.COMPUTE, true);
            this.computeQueue = compute;
            openQueues.add(compute);
            transfer = new CudaComputeQueue(this, newStream(), ComputeQueue.QueueType.TRANSFER, true);
            this.transferQueue = transfer;
            openQueues.add(transfer);
            CudaSharedPools.register(context.context(), bufferPool);
        } catch (RuntimeException e) {
            closeQuietly(transfer, compute, events, modules, e);
            throw e;
        }

        LOGGER.info("CUDA backend initialized - Device: {} (CC {}.{}), {}, VRAM: {} MB, Multiprocessors: {}, "
                        + "Max Threads/Block: {}, Async copy engines: {}",
                capabilities.getDeviceName(),
                capabilities.getComputeCapability() / 10, capabilities.getComputeCapability() % 10,
                capabilities.getDriverVersion(),
                capabilities.getTotalVramBytes() / (1024 * 1024),
                capabilities.getComputeUnits(),
                capabilities.getMaxWorkGroupSize(),
                capabilities.supportsAsyncTransfer() ? "yes" : "no");
    }

    @Override
    public BackendType backendType() {
        return BackendType.CUDA;
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
     * size (and at most {@link CudaBufferPool#MAX_REUSE_SLACK_BYTES} larger) or a fresh
     * allocation of exactly that size, and is tracked at its block size. The buffer reports the
     * requested size; its contents are undefined. The {@code readOnly} flag is advisory on CUDA.
     *
     * @throws IllegalArgumentException when the size is not positive
     * @throws GpuOutOfMemoryException  when the device is out of memory after this backend's pool
     *                                  and those of the other backends of the context were freed
     */
    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
        enter();
        if (sizeInBytes <= 0 || sizeInBytes > Long.MAX_VALUE - BUFFER_ALIGNMENT) {
            throw new IllegalArgumentException("sizeInBytes must be positive and representable, got " + sizeInBytes);
        }
        long aligned = ((sizeInBytes + BUFFER_ALIGNMENT - 1) / BUFFER_ALIGNMENT) * BUFFER_ALIGNMENT;
        CudaBufferPool.Allocation block = acquireBlock(aligned);
        LOGGER.trace("Allocated CUDA buffer: {} bytes ({} byte block), readOnly={}", sizeInBytes, block.bytes(),
                readOnly);
        return new CudaGpuBuffer(this, block.pointer(), sizeInBytes, block.bytes());
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes) {
        return allocateBuffer(sizeInBytes, false);
    }

    /**
     * Loads {@code kernelName} from PTX text ({@code .version}/{@code .target} directives
     * required). Each distinct text is loaded as one module, shared by all its kernels.
     * Build options do not apply to PTX loading and are ignored.
     *
     * @throws UnsupportedKernelException when the source is not PTX, the kernel is not in it,
     *                                    or the driver cannot JIT it for this device
     */
    @Override
    public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
        enter();
        Objects.requireNonNull(kernelName, "kernelName cannot be null");
        Objects.requireNonNull(kernelSource, "kernelSource cannot be null");
        if (!isPtx(kernelSource)) {
            throw new UnsupportedKernelException("CUDA kernel " + kernelName + " was given CUDA C source; runtime "
                    + "compilation requires NVRTC, which this backend does not use. Ship PTX or a fatbin "
                    + "(KernelArtifact.cudaImage) instead.");
        }
        CacheKey key = new CacheKey(kernelName, PTX_KEY_PREFIX + CudaModuleCache.sha256(kernelSource), "");
        return kernelCache.getOrCompile(key, k -> {
            CUmodule module = moduleCache.getPtx(kernelSource);
            return newHandle(module, kernelName, "PTX source", null);
        });
    }

    /**
     * Loads a kernel from a {@link KernelArtifact.Kind#CUDA_IMAGE} artifact. The image is read
     * from the classpath and loaded once per resource; the handle is cached per (kernel, resource).
     * The handle carries the signature its kernel library registered in {@link CudaKernelSignatures},
     * so its launches are checked against the kernel inventory.
     *
     * @throws IllegalArgumentException   when {@code artifact} is null
     * @throws UnsupportedKernelException for other artifact kinds, a missing resource or kernel,
     *                                    or an image without code for this device
     */
    @Override
    public KernelHandle loadKernel(KernelArtifact artifact) {
        enter();
        if (artifact == null) {
            throw new IllegalArgumentException("kernel artifact must not be null");
        }
        if (artifact.kind() != KernelArtifact.Kind.CUDA_IMAGE) {
            throw new UnsupportedKernelException("CUDA backend cannot load " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName());
        }
        String resource = artifact.resourcePath();
        CacheKey key = new CacheKey(artifact.kernelName(), IMAGE_KEY_PREFIX + artifact.programKey(), "");
        return kernelCache.getOrCompile(key, k -> {
            CUmodule module = moduleCache.get(resource);
            return newHandle(module, artifact.kernelName(), resource,
                    CudaKernelSignatures.lookup(resource, artifact.kernelName()));
        });
    }

    @Override
    public boolean supportsArtifact(KernelArtifact.Kind kind) {
        return kind == KernelArtifact.Kind.CUDA_IMAGE;
    }

    /**
     * Validates that {@code kernel} can launch with this local size, so a block that exceeds
     * the function's register-limited maximum is reported at warm-up instead of at dispatch.
     *
     * @throws IllegalArgumentException when the handle is not from this backend or the block is too large
     */
    @Override
    public void prepareKernel(KernelHandle kernel, long[] localWorkSize) {
        enter();
        CudaKernelHandle handle = ownHandle(kernel);
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
            CudaComputeQueue queue = new CudaComputeQueue(this, newStream(), ComputeQueue.QueueType.COMPUTE, false);
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
     * Free device memory as the driver reports it, including other processes' use
     * ({@code cuMemGetInfo}), plus the blocks pooled by every backend sharing this device's
     * primary context, which every allocation of this backend can reclaim.
     */
    @Override
    public long getFreeVramBytes() {
        enter();
        long[] free = new long[1];
        long[] total = new long[1];
        CudaErrors.check("cuMemGetInfo", () -> cuMemGetInfo(free, total));
        return free[0] + CudaSharedPools.pooledBytes(context.context());
    }

    /**
     * Frees every block of the buffer pool ({@code cuMemFree}), synchronizing the context first
     * when a released buffer may still be in use. Open buffers are not affected.
     *
     * @return the bytes returned to the driver
     * @throws IllegalStateException after {@link #close()}
     * @throws RuntimeException      the first translated driver failure, after every block was freed
     */
    public long trimPool() {
        enter();
        return bufferPool.trim();
    }

    /** Snapshot of the buffer pool's accounting: bytes in use and pooled, block counts and reuse counters. */
    public CudaBufferPool.Stats bufferPoolStats() {
        return bufferPool.stats();
    }

    /**
     * Releases the caller-owned queues still open, finishes and destroys the backend-owned queues,
     * frees the buffer pool together with the memory of every buffer still open, drops the kernel
     * cache, unloads every module, destroys the event pool and releases the primary context.
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
        CudaSharedPools.unregister(context.context(), bufferPool);
        RuntimeException failure = null;
        try {
            context.makeCurrent();
        } catch (RuntimeException e) {
            failure = e;
        }
        for (CudaComputeQueue queue : new ArrayList<>(openQueues)) {
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
        failure = runStep("release primary context", context::release, failure);
        if (failure != null) {
            throw failure;
        }
        LOGGER.debug("CUDA backend closed - freed {} bytes of buffers that were still open", stillOpen);
    }

    /** The device this backend runs on. */
    public CUDevice getCUDevice() {
        return cuDevice;
    }

    /** Kernel handle cache of {@link #compileKernel} and {@link #loadKernel}. */
    public KernelCache getKernelCache() {
        return kernelCache;
    }

    /** The retained primary context; queues, buffers, events and handles make it current. */
    public CudaContext context() {
        return context;
    }

    /** Pool of timing-disabled events shared by every queue of this backend. */
    public CudaEventPool eventPool() {
        return eventPool;
    }

    /** Module cache holding every loaded module of this backend. */
    public CudaModuleCache moduleCache() {
        return moduleCache;
    }

    /** Installed driver version as {@code 1000 * major + 10 * minor} ({@code cuDriverGetVersion}). */
    public int driverVersion() {
        return driverVersion;
    }

    /** True once {@link #close()} has started. */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Returns a closed buffer's block to the buffer pool, with this backend's context current.
     * Called by {@link CudaGpuBuffer#close()}.
     *
     * @throws RuntimeException the translated failure of fencing the block, which was then freed
     */
    void releaseAllocation(CUdeviceptr pointer, long allocationBytes) {
        context.makeCurrent();
        bufferPool.release(pointer, allocationBytes);
    }

    /** The queues whose streams a released buffer's fence covers. */
    Collection<CudaComputeQueue> openQueues() {
        return Collections.unmodifiableSet(openQueues);
    }

    /** Called by {@link CudaComputeQueue#closeInternal()}: the released queue's stream needs no more fences. */
    void onQueueReleased(CudaComputeQueue queue) {
        openQueues.remove(queue);
    }

    /**
     * Checks the backend is open and makes its context current on this thread.
     *
     * @throws IllegalStateException after {@link #close()}
     */
    void enter() {
        if (closed) {
            throw new IllegalStateException("CudaComputeBackend already closed");
        }
        context.makeCurrent();
    }

    /**
     * A block of {@code bytes} from this backend's pool. When the device is still out of memory
     * after the pool freed itself, the pools of the other backends sharing the context are freed
     * and the allocation is retried once.
     *
     * @throws GpuOutOfMemoryException when the retry fails too, or nothing was pooled elsewhere;
     *                                 failures of freeing the other pools are suppressed on it
     */
    private CudaBufferPool.Allocation acquireBlock(long bytes) {
        try {
            return bufferPool.acquire(bytes);
        } catch (GpuOutOfMemoryException first) {
            List<RuntimeException> trimFailures = new ArrayList<>();
            long freed = CudaSharedPools.trimSiblings(context.context(), bufferPool, trimFailures);
            if (freed == 0) {
                trimFailures.forEach(first::addSuppressed);
                throw first;
            }
            for (RuntimeException e : trimFailures) {
                LOGGER.warn("Freeing another CUDA backend's buffer pool failed: {}", e.getMessage(), e);
            }
            LOGGER.debug("CUDA allocation of {} bytes ran out of memory; freed {} bytes pooled by other backends "
                    + "of the context and retrying", bytes, freed);
            context.makeCurrent();
            try {
                return bufferPool.acquire(bytes);
            } catch (RuntimeException second) {
                second.addSuppressed(first);
                throw second;
            }
        }
    }

    private CUstream newStream() {
        CUstream stream = new CUstream();
        CudaErrors.check("cuStreamCreate", () -> cuStreamCreate(stream, CUstream_flags.CU_STREAM_NON_BLOCKING));
        return stream;
    }

    private CudaKernelHandle newHandle(CUmodule module, String kernelName, String origin,
                                       KernelSignature signature) {
        CUfunction function = new CUfunction();
        try {
            CudaErrors.check("cuModuleGetFunction(" + kernelName + ")",
                    () -> cuModuleGetFunction(function, module, kernelName));
        } catch (UnsupportedKernelException e) {
            throw new UnsupportedKernelException("CUDA kernel " + kernelName + " is not in " + origin, e);
        }
        return new CudaKernelHandle(this, function, kernelName, signature);
    }

    private CudaKernelHandle ownHandle(KernelHandle kernel) {
        if (!(kernel instanceof CudaKernelHandle)) {
            throw new IllegalArgumentException("kernel must be a CudaKernelHandle, got "
                    + (kernel == null ? "null" : kernel.getClass().getName()));
        }
        CudaKernelHandle handle = (CudaKernelHandle) kernel;
        if (handle.backend() != this) {
            throw new IllegalArgumentException("CUDA kernel " + handle.kernelName() + " belongs to another backend");
        }
        return handle;
    }

    /** True when the text carries the PTX {@code .version} and {@code .target} directives. */
    private static boolean isPtx(String source) {
        return source.contains(".version") && source.contains(".target");
    }

    private static RuntimeException runStep(String what, Runnable step, RuntimeException failure) {
        try {
            step.run();
            return failure;
        } catch (RuntimeException e) {
            LOGGER.warn("CUDA backend close: failed to {}: {}", what, e.getMessage());
            if (failure == null) {
                return e;
            }
            failure.addSuppressed(e);
            return failure;
        }
    }

    private void closeQuietly(CudaComputeQueue transfer, CudaComputeQueue compute, CudaEventPool events,
                              CudaModuleCache modules, RuntimeException cause) {
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
        runStep("release primary context", context::release, failure);
    }
}
