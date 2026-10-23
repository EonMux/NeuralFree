package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.BackendResources;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.cache.CacheKey;
import com.eonmux.neuralfree.acceleration.cache.KernelCache;
import com.eonmux.neuralfree.acceleration.vulkan.VkDeviceRanking;
import com.eonmux.neuralfree.acceleration.vulkan.VkDeviceSelection;
import com.eonmux.neuralfree.acceleration.vulkan.VkInstanceManager;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkDeviceCreateInfo;
import org.lwjgl.vulkan.VkDeviceQueueCreateInfo;
import org.lwjgl.vulkan.VkQueue;

import java.nio.FloatBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkCreateDevice;
import static org.lwjgl.vulkan.VK10.vkDestroyDevice;
import static org.lwjgl.vulkan.VK10.vkDeviceWaitIdle;
import static org.lwjgl.vulkan.VK10.vkGetDeviceQueue;

/**
 * Vulkan 1.2+ backend on one physical device.
 *
 * <ul>
 *   <li><b>Device.</b> Features and extensions from {@link VkDeviceFeatures#select} (timeline
 *       semaphores required; 8/16-bit storage, int8/16/64, float16, subgroup controls, push
 *       descriptors and the memory budget when supported). Up to {@value #MAX_COMPUTE_QUEUES} queues
 *       of the compute family and one queue of a dedicated transfer family, when the device has one.</li>
 *   <li><b>Queues.</b> {@link #computeQueue()} runs on compute queue 0. {@link #transferQueue()} runs
 *       on the dedicated transfer family when present (it then rejects dispatches), otherwise on
 *       compute queue 1 when the family has two or more queues, else on queue 0. Both are owned by
 *       the backend. {@link #createComputeQueue()} returns caller-owned queues spread round-robin over
 *       compute queues 1.. (queue 0 when the family has one queue); queues sharing a {@code VkQueue}
 *       serialise their submissions. {@code CONCURRENT_QUEUES_PROFITABLE} is not reported.</li>
 *   <li><b>Memory.</b> Buffers are sub-allocated by a {@link VkMemoryAllocator} in device-local
 *       memory, shared concurrently between the compute and transfer families when they differ. A
 *       buffer larger than {@link DeviceCapabilities#getMaxAllocSizeBytes()} is rejected with
 *       {@link GpuOutOfMemoryException}. Closing a buffer is safe while queues still run commands
 *       recorded with it: its memory is reclaimed once those batches retire. An allocation that
 *       runs out of memory first submits and waits for the batches such closed buffers wait for,
 *       then retries once.
 *       {@link #getFreeVramBytes()} uses {@code VK_EXT_memory_budget} when enabled.</li>
 *   <li><b>Kernels.</b> {@link #loadKernel} loads {@link KernelArtifact.Kind#SPIRV} artifacts with their
 *       specialisation constants and build-time signature; {@link #compileKernel} keeps its legacy
 *       meaning (the source is a classpath {@code .spv} path, validated by reflection only). Modules
 *       are cached per resource ({@link VkProgramCache}), pipelines per local size and constants, and the
 *       pipeline cache persists across runs ({@link VkPipelineCache}). Handles are cached and
 *       backend-owned; {@link #cloneKernel} gives caller-owned handles with fresh arguments.</li>
 * </ul>
 */
@NotThreadSafe("Each queue is recorded by one thread at a time. Allocation, kernel loading, the kernel cache and "
        + "queue creation are safe from several threads.")
public final class VkComputeBackend implements ComputeBackend {

    private static final Logger LOGGER = LoggingUtil.getLogger(VkComputeBackend.class);

    /** Queues requested from the compute family (fewer when the family has fewer). */
    public static final int MAX_COMPUTE_QUEUES = 3;

    private static final String SPIRV_KEY_PREFIX = "spirv:";
    private static final String LEGACY_KEY_PREFIX = "spv-path:";
    private static final int DEVICE_BUFFER_USAGE = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    private final VkInstanceManager instanceManager;
    private final VkPhysicalDeviceInfo info;
    private final VkDeviceFeatures.Enabled enabled;
    private final int dispatchesPerSubmit;
    private final VkDevice device;
    private final VkQueueContext[] computeContexts;
    private final VkQueueContext transferContext;
    private final VkMemoryAllocator allocator;
    private final DeviceCapabilities capabilities;
    private final VkPipelineCache pipelineCache;
    private final VkDescriptorStrategy descriptorStrategy;
    private final VkProgramCache programs;
    private final KernelCache kernelCache;
    private final VkComputeQueue computeQueue;
    private final VkComputeQueue transferQueue;
    private final Set<VkComputeQueue> createdQueues = Collections.newSetFromMap(new IdentityHashMap<>());
    private final List<VkComputeQueue> openQueues = new CopyOnWriteArrayList<>();
    private final AtomicInteger nextQueue = new AtomicInteger();
    private volatile boolean closed;

    /**
     * Opens the fastest eligible device of the shared Vulkan instance
     * ({@link VkDeviceRanking#fastest(List)}).
     *
     * @throws NoBackendAvailableException when Vulkan is unavailable or no device is eligible
     */
    public VkComputeBackend() {
        this(VkDeviceSelection.fastest());
    }

    /**
     * Opens {@code info}, one of {@code instanceManager.devices()}. The backend takes its own reference
     * on the manager and releases it in {@link #close()}.
     *
     * @throws IllegalArgumentException    when an argument is null or {@code info} belongs to another instance
     * @throws NoBackendAvailableException when the device lacks a requirement of the backend
     * @throws GpuOutOfMemoryException     when device creation runs out of host or device memory
     * @throws GpuBackendException         when device creation fails otherwise
     */
    public VkComputeBackend(VkInstanceManager instanceManager, VkPhysicalDeviceInfo info) {
        this(VkDeviceSelection.adopt(instanceManager, info));
    }

    private VkComputeBackend(VkDeviceSelection selection) {
        this.instanceManager = selection.manager();
        this.info = selection.info();
        Deque<Runnable> undo = new ArrayDeque<>();
        undo.push(instanceManager::release);
        try {
            this.dispatchesPerSubmit = VkComputeQueue.configuredDispatchesPerSubmit();
            this.enabled = VkDeviceFeatures.select(info);
            int computeFamily = info.computeQueueFamily();
            int transferFamily = info.transferQueueFamily();
            int computeCount = Math.min(MAX_COMPUTE_QUEUES, info.queueFamilies().get(computeFamily).queueCount());
            VkDevice created = createDevice(info, enabled, computeFamily, computeCount, transferFamily);
            undo.push(() -> vkDestroyDevice(created, null));
            this.device = created;

            this.computeContexts = new VkQueueContext[computeCount];
            for (int i = 0; i < computeCount; i++) {
                computeContexts[i] = new VkQueueContext(deviceQueue(created, computeFamily, i), computeFamily, i,
                        new Object());
            }
            this.transferContext = transferFamily >= 0
                    ? new VkQueueContext(deviceQueue(created, transferFamily, 0), transferFamily, 0, new Object())
                    : computeContexts[Math.min(1, computeCount - 1)];

            int[] families = transferFamily >= 0 ? new int[] {computeFamily, transferFamily} : new int[] {computeFamily};
            VkMemoryAllocator memory = new VkMemoryAllocator(created, info, enabled.memoryBudget(), families);
            undo.push(memory::close);
            this.allocator = memory;
            this.capabilities = VkCapabilitiesBuilder.build(info, enabled, memory.maxAllocSize());

            VkPipelineCache cache = new VkPipelineCache(created, info);
            undo.push(cache::close);
            this.pipelineCache = cache;
            VkDescriptorStrategy strategy = VkDescriptorStrategy.select(created, info, enabled);
            undo.push(strategy::close);
            this.descriptorStrategy = strategy;
            VkProgramCache programCache = new VkProgramCache(created, info, enabled, cache, strategy);
            undo.push(programCache::close);
            this.programs = programCache;
            this.kernelCache = new KernelCache();

            VkComputeQueue compute = new VkComputeQueue(this, computeContexts[0], ComputeQueue.QueueType.COMPUTE, true);
            undo.push(compute::closeInternal);
            this.computeQueue = compute;
            this.transferQueue = new VkComputeQueue(this, transferContext, ComputeQueue.QueueType.TRANSFER, true);
            openQueues.add(computeQueue);
            openQueues.add(transferQueue);
            memory.setReleaseGate(new QueueReleaseGate(openQueues));
        } catch (RuntimeException e) {
            closed = true;
            while (!undo.isEmpty()) {
                runStep("release partially created backend", undo.pop(), e);
            }
            throw e;
        }

        LOGGER.info("Vulkan backend initialized - Device: {} (Vulkan {}, {}), {}, VRAM: {} MB, compute queues: {}, "
                        + "transfer family: {}, descriptors: {}, max allocation: {} MB",
                capabilities.getDeviceName(), info.apiVersionString(), capabilities.getDeviceClass(),
                capabilities.getDriverVersion(), capabilities.getTotalVramBytes() / (1024 * 1024),
                computeContexts.length, transferContext.family() == computeContexts[0].family() ? "shared"
                        : String.valueOf(transferContext.family()),
                descriptorStrategy.getClass().getSimpleName(), capabilities.getMaxAllocSizeBytes() / (1024 * 1024));
    }

    @Override
    public BackendType backendType() {
        return BackendType.VULKAN;
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
     * A device-local storage buffer usable as copy source and destination. {@code readOnly} is a
     * usage hint without effect on Vulkan.
     *
     * @throws GpuOutOfMemoryException when the size exceeds the largest allocation or memory is exhausted
     */
    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
        ensureOpen();
        if (sizeInBytes <= 0) {
            throw new IllegalArgumentException("sizeInBytes must be positive, got " + sizeInBytes);
        }
        return allocator.createBuffer(sizeInBytes, DEVICE_BUFFER_USAGE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes) {
        return allocateBuffer(sizeInBytes, false);
    }

    /**
     * Loads {@code kernelName} from the classpath SPIR-V module {@code spirvPath}, without
     * specialisation constants or a build-time signature; arguments are validated against the
     * module's reflected bindings and push constants. Build options do not apply and are ignored.
     *
     * @throws UnsupportedKernelException when the module is missing, malformed, or needs a capability
     *                                    or limit the device does not provide
     */
    @Override
    public KernelHandle compileKernel(String kernelName, String spirvPath, String buildOptions) {
        ensureOpen();
        Objects.requireNonNull(kernelName, "kernelName cannot be null");
        Objects.requireNonNull(spirvPath, "spirvPath cannot be null");
        CacheKey key = new CacheKey(kernelName, LEGACY_KEY_PREFIX + spirvPath, "");
        return cached(key, () -> new VkKernelHandle(programs, programs.get(spirvPath), kernelName, null, null));
    }

    /**
     * Loads a kernel from a {@link KernelArtifact.Kind#SPIRV} artifact; the handle is cached per
     * (kernel, module, specialisation constants).
     *
     * @throws IllegalArgumentException   when {@code artifact} is null
     * @throws UnsupportedKernelException for other artifact kinds, or a module the device cannot run
     *                                    or that disagrees with the artifact's signature
     */
    @Override
    public KernelHandle loadKernel(KernelArtifact artifact) {
        ensureOpen();
        if (artifact == null) {
            throw new IllegalArgumentException("kernel artifact must not be null");
        }
        if (artifact.kind() != KernelArtifact.Kind.SPIRV) {
            throw new UnsupportedKernelException("Vulkan backend cannot load " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName());
        }
        CacheKey key = new CacheKey(artifact.kernelName(),
                SPIRV_KEY_PREFIX + artifact.programKey() + artifact.specConstants(), "");
        return cached(key, () -> new VkKernelHandle(programs, programs.get(artifact.resourcePath()),
                artifact.kernelName(), artifact.specConstants(), artifact.signature()));
    }

    @Override
    public boolean supportsArtifact(KernelArtifact.Kind kind) {
        return kind == KernelArtifact.Kind.SPIRV;
    }

    /**
     * Creates the pipeline a dispatch with this local size would use. With a null local size, the
     * pipeline of a fixed-size module is created; a module whose local size is specialised has no
     * default to prepare and is left alone.
     *
     * @throws IllegalArgumentException when the handle is not from this backend or the size is invalid
     */
    @Override
    public void prepareKernel(KernelHandle kernel, long[] localWorkSize) {
        ensureOpen();
        VkKernelHandle handle = ownHandle(kernel);
        if (localWorkSize == null) {
            if (!handle.program().info().hasAnySpecLocalSize()) {
                handle.prepare(new long[] {1}, null);
            }
            return;
        }
        handle.prepare(localWorkSize, localWorkSize);
    }

    /**
     * A caller-owned handle on the same program with fresh, independent arguments.
     *
     * @throws IllegalArgumentException when the template is not a handle of this backend
     */
    @Override
    public KernelHandle cloneKernel(KernelHandle template) {
        ensureOpen();
        return ownHandle(template).cloneHandle();
    }

    /** A caller-owned compute queue; the caller closes it, and {@link #close()} closes any left open. */
    @Override
    public ComputeQueue createComputeQueue() {
        ensureOpen();
        int n = computeContexts.length;
        VkQueueContext context = n == 1 ? computeContexts[0]
                : computeContexts[1 + Math.floorMod(nextQueue.getAndIncrement(), n - 1)];
        VkComputeQueue queue = new VkComputeQueue(this, context, ComputeQueue.QueueType.COMPUTE, false);
        synchronized (createdQueues) {
            if (!closed) {
                createdQueues.add(queue);
                openQueues.add(queue);
                return queue;
            }
        }
        queue.closeInternal();
        throw new IllegalStateException("VkComputeBackend already closed");
    }

    /** Sum of the sizes of live buffers. */
    @Override
    public long getAllocatedBytes() {
        return allocator.allocatedBytes();
    }

    /** Free device memory: the memory budget when {@code VK_EXT_memory_budget} is enabled, else VRAM minus this backend's use. */
    @Override
    public long getFreeVramBytes() {
        ensureOpen();
        return allocator.freeVramBytes();
    }

    /**
     * Waits for the device to go idle, closes every queue (draining recorded work), drops the kernel
     * cache, and destroys the programs, descriptor strategy, pipeline cache (saved to disk), memory
     * and device, then releases the instance reference. Idempotent. Every step runs even when an
     * earlier one fails; the first failure is rethrown at the end.
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
        RuntimeException failure = null;
        failure = runStep("wait for the device to go idle", () -> {
            VkErrors.check(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");
        }, failure);
        List<VkComputeQueue> leftOpen;
        synchronized (createdQueues) {
            leftOpen = new ArrayList<>(createdQueues);
            createdQueues.clear();
        }
        for (VkComputeQueue queue : leftOpen) {
            failure = runStep("close a caller-owned queue", queue::closeInternal, failure);
        }
        failure = runStep("close the compute queue", computeQueue::closeInternal, failure);
        failure = runStep("close the transfer queue", transferQueue::closeInternal, failure);
        failure = runStep("clear the kernel cache", kernelCache::clear, failure);
        failure = runStep("destroy programs", programs::close, failure);
        failure = runStep("destroy the descriptor strategy", descriptorStrategy::close, failure);
        failure = runStep("save and destroy the pipeline cache", pipelineCache::close, failure);
        failure = runStep("free device memory", allocator::close, failure);
        failure = runStep("destroy the device", () -> vkDestroyDevice(device, null), failure);
        failure = runStep("release the Vulkan instance", instanceManager::release, failure);
        if (failure != null) {
            throw failure;
        }
        LOGGER.debug("Vulkan backend closed: {}", info.name());
    }

    /** True once {@link #close()} has started. */
    public boolean isClosed() {
        return closed;
    }

    /** The logical device. */
    public VkDevice device() {
        return device;
    }

    /** The physical device snapshot. */
    public VkPhysicalDeviceInfo info() {
        return info;
    }

    /** The enabled features and extensions. */
    public VkDeviceFeatures.Enabled enabledFeatures() {
        return enabled;
    }

    /** The instance this backend holds a reference on. */
    public VkInstanceManager instanceManager() {
        return instanceManager;
    }

    /** The memory allocator of every buffer and staging ring of this backend. */
    public VkMemoryAllocator allocator() {
        return allocator;
    }

    /** The descriptor strategy every program and queue of this backend uses. */
    public VkDescriptorStrategy descriptorStrategy() {
        return descriptorStrategy;
    }

    /** Programs loaded by this backend. */
    public VkProgramCache programCache() {
        return programs;
    }

    /** The pipeline cache, persisted on {@link #close()}. */
    public VkPipelineCache getVkPipelineCache() {
        return pipelineCache;
    }

    /** Kernel handle cache of {@link #compileKernel} and {@link #loadKernel}. */
    public KernelCache getKernelCache() {
        return kernelCache;
    }

    /** Dispatches per command buffer before a queue submits it. */
    int dispatchesPerSubmit() {
        return dispatchesPerSubmit;
    }

    /** Called by a queue when it is released. */
    void onQueueClosed(VkComputeQueue queue) {
        synchronized (createdQueues) {
            createdQueues.remove(queue);
        }
        openQueues.remove(queue);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("VkComputeBackend already closed");
        }
    }

    /** The cached handle for {@code key}, created on a miss; a cached handle closed by a caller is replaced. */
    private KernelHandle cached(CacheKey key, Supplier<VkKernelHandle> factory) {
        KernelHandle handle = kernelCache.getOrCompile(key, k -> factory.get());
        if (((VkKernelHandle) handle).isClosed()) {
            kernelCache.remove(key);
            handle = kernelCache.getOrCompile(key, k -> factory.get());
        }
        return handle;
    }

    private VkKernelHandle ownHandle(KernelHandle kernel) {
        if (!(kernel instanceof VkKernelHandle)) {
            throw new IllegalArgumentException("kernel must be a VkKernelHandle, got "
                    + (kernel == null ? "null" : kernel.getClass().getName()));
        }
        VkKernelHandle handle = (VkKernelHandle) kernel;
        if (handle.program().descriptorStrategy() != descriptorStrategy) {
            throw new IllegalArgumentException("Vulkan kernel " + handle.kernelName() + " belongs to another backend");
        }
        return handle;
    }

    /**
     * Creates the logical device with {@code computeCount} queues of the compute family and one queue
     * of the transfer family (when there is one), and the selected features and extensions.
     */
    private static VkDevice createDevice(VkPhysicalDeviceInfo info, VkDeviceFeatures.Enabled enabled,
                                         int computeFamily, int computeCount, int transferFamily) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDeviceQueueCreateInfo.Buffer queues = VkDeviceQueueCreateInfo.calloc(transferFamily >= 0 ? 2 : 1, stack);
            FloatBuffer computePriorities = stack.mallocFloat(computeCount);
            for (int i = 0; i < computeCount; i++) {
                computePriorities.put(i, 1.0f);
            }
            queues.get(0).sType$Default().queueFamilyIndex(computeFamily).pQueuePriorities(computePriorities);
            if (transferFamily >= 0) {
                queues.get(1).sType$Default().queueFamilyIndex(transferFamily).pQueuePriorities(stack.floats(1.0f));
            }
            VkDeviceCreateInfo createInfo = VkDeviceCreateInfo.calloc(stack)
                    .sType$Default()
                    .pQueueCreateInfos(queues);
            enabled.applyTo(createInfo, stack);
            PointerBuffer handle = stack.mallocPointer(1);
            int result = vkCreateDevice(info.handle(), createInfo, null, handle);
            if (result != VK_SUCCESS) {
                throw VkErrors.translate(result, "vkCreateDevice on '" + info.name() + "' with " + enabled);
            }
            return new VkDevice(handle.get(0), info.handle(), createInfo);
        }
    }

    private static VkQueue deviceQueue(VkDevice device, int family, int index) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer queue = stack.mallocPointer(1);
            vkGetDeviceQueue(device, family, index, queue);
            return new VkQueue(queue.get(0), device);
        }
    }

    private static RuntimeException runStep(String what, Runnable step, RuntimeException failure) {
        try {
            step.run();
            return failure;
        } catch (RuntimeException e) {
            LOGGER.warn("Vulkan backend: failed to {}: {}", what, e.getMessage());
            if (failure == null) {
                return e;
            }
            if (failure != e) {
                failure.addSuppressed(e);
            }
            return failure;
        }
    }

    /**
     * Defers destroying a closed buffer until every open queue has retired the batch that received its
     * latest command before the close. {@link #capture()} and {@link #passed(Object)} read only volatile
     * queue state, so they never block; {@link #await(Object)} waits on the queues for an allocation
     * that ran out of memory.
     */
    private static final class QueueReleaseGate implements VkMemoryAllocator.ReleaseGate {
        private final List<VkComputeQueue> queues;

        QueueReleaseGate(List<VkComputeQueue> queues) {
            this.queues = queues;
        }

        @Override
        public Object capture() {
            List<VkComputeQueue> busy = null;
            List<Long> values = null;
            for (VkComputeQueue q : queues) {
                long last = q.lastUseValue();
                if (last > q.completedValue()) {
                    if (busy == null) {
                        busy = new ArrayList<>();
                        values = new ArrayList<>();
                    }
                    busy.add(q);
                    values.add(last);
                }
            }
            return busy == null ? null : new Outstanding(busy, values);
        }

        @Override
        public boolean passed(Object token) {
            Outstanding work = (Outstanding) token;
            for (int i = 0; i < work.queues.size(); i++) {
                if (work.queues.get(i).completedValue() < work.values.get(i)) {
                    return false;
                }
            }
            return true;
        }

        /**
         * Submits and waits for the captured batches, then retires them. A thread recording on a queue
         * (it holds that queue's monitor, for example while a staging ring grows) only waits for that
         * queue's batches already submitted: submitting would move the pending value its slice was
         * tagged with, and taking another queue's monitor would break the queue lock order. Such a
         * thread leaves the other queues' work alone.
         */
        @Override
        public void await(Object token) {
            Outstanding work = (Outstanding) token;
            VkComputeQueue recording = recordingQueue();
            for (int i = 0; i < work.queues.size(); i++) {
                VkComputeQueue q = work.queues.get(i);
                long value = work.values.get(i);
                if (q.completedValue() >= value || q.isClosed() || (recording != null && q != recording)) {
                    continue;
                }
                if (recording == null) {
                    q.flushIfPending(value);
                }
                long target = Math.min(value, q.submittedValue());
                if (target > q.completedValue()) {
                    q.waitForValue(target);
                }
            }
        }

        /** The queue whose monitor the calling thread holds, or null. */
        private VkComputeQueue recordingQueue() {
            for (VkComputeQueue q : queues) {
                if (Thread.holdsLock(q)) {
                    return q;
                }
            }
            return null;
        }

        /** Per busy queue, the timeline value its retirement must reach. */
        private static final class Outstanding {
            final List<VkComputeQueue> queues;
            final List<Long> values;

            Outstanding(List<VkComputeQueue> queues, List<Long> values) {
                this.queues = queues;
                this.values = values;
            }
        }
    }
}
