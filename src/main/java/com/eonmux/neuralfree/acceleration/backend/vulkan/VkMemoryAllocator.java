package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkBufferCreateInfo;
import org.lwjgl.vulkan.VkBufferMemoryRequirementsInfo2;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkMappedMemoryRange;
import org.lwjgl.vulkan.VkMemoryAllocateInfo;
import org.lwjgl.vulkan.VkMemoryDedicatedAllocateInfo;
import org.lwjgl.vulkan.VkMemoryDedicatedRequirements;
import org.lwjgl.vulkan.VkMemoryRequirements;
import org.lwjgl.vulkan.VkMemoryRequirements2;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryBudgetPropertiesEXT;
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties2;

import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.lwjgl.vulkan.AMDDeviceCoherentMemory.VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD;
import static org.lwjgl.vulkan.AMDDeviceCoherentMemory.VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
import static org.lwjgl.vulkan.VK10.*;
import static org.lwjgl.vulkan.VK11.VK_MEMORY_PROPERTY_PROTECTED_BIT;
import static org.lwjgl.vulkan.VK11.vkGetBufferMemoryRequirements2;
import static org.lwjgl.vulkan.VK11.vkGetPhysicalDeviceMemoryProperties2;

/**
 * Sub-allocates buffer memory from per-memory-type blocks (256 MiB by default),
 * gives large buffers dedicated allocations, maps host-visible blocks once, and
 * tracks allocated bytes and the driver's memory budget.
 *
 * <p>Sub-allocation keeps the number of {@code vkAllocateMemory} calls far below
 * {@code maxMemoryAllocationCount}. A buffer whose memory requirement is at least half a
 * block, or whose driver prefers or requires it, gets its own allocation. Offsets inside a
 * block are aligned to the buffer's requirement, {@code minStorageBufferOffsetAlignment}, 256
 * bytes and, for non-coherent host memory, {@code nonCoherentAtomSize}. An empty block is
 * freed unless it is the last block of its memory type.</p>
 *
 * <p>Memory type choice: the type must have every required flag and none of the flags
 * that need features the backend does not enable (protected, lazily allocated, AMD device
 * coherent or uncached). Among the candidates, more preferred flags win; on a discrete GPU
 * a type that carries {@code HOST_VISIBLE} or {@code DEVICE_LOCAL} without the caller
 * asking for it loses, so device buffers stay out of the BAR window and staging buffers
 * stay in system memory; remaining ties go to the lower type index. When a type runs out
 * of memory the next candidate is tried.</p>
 *
 * <p>Closing a buffer removes it from the accounting at once. Its {@code VkBuffer} and memory are
 * destroyed when the device work recorded before the close has completed, as reported by the
 * {@link ReleaseGate} the backend installs; without a gate, or when no such work is outstanding,
 * they are destroyed immediately. When an allocation runs out of memory while closed buffers
 * still wait for their work, the allocator waits for that work through the gate, destroys them
 * and retries the allocation once.</p>
 *
 * <p>Thread-safe: every mutating call holds the allocator's lock. The out-of-memory wait runs
 * without it, so queues that retire work (and reclaim closed buffers) are never blocked by it.</p>
 */
public final class VkMemoryAllocator implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(VkMemoryAllocator.class);

    /** Block size in MiB. */
    public static final String BLOCK_MB_PROPERTY = "neuralfree.vulkan.blockMb";
    /** Default block size in MiB. */
    public static final int DEFAULT_BLOCK_MB = 256;
    /** Minimum alignment of every sub-allocation in bytes. */
    public static final long MIN_ALIGNMENT = 256;

    private static final long MIB = 1024L * 1024L;
    private static final int NEVER_ALLOCATE = VK_MEMORY_PROPERTY_PROTECTED_BIT
            | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT
            | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD
            | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
    private static final int PLACEMENT_FLAGS = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    /**
     * Tells the allocator whether device work that may still use a closed buffer is outstanding.
     * {@link #capture()} and {@link #passed(Object)} are called with or without the allocator's lock
     * held, so they must not block, take locks, or call back into the allocator.
     */
    public interface ReleaseGate {
        /** A token for the device work recorded so far that may use a buffer closed now; null when there is none. */
        Object capture();

        /** True when the work described by a token from {@link #capture()} has completed. */
        boolean passed(Object token);

        /**
         * Blocks until as much of the work described by a token from {@link #capture()} as can be
         * completed safely from the calling thread has completed and been retired, so that
         * {@link #passed(Object)} reflects it. Called without the allocator's lock, when an
         * allocation ran out of memory. It may call back into the allocator.
         *
         * @throws RuntimeException the failure of waiting for the device
         */
        void await(Object token);
    }

    private final VkDevice device;
    private final VkPhysicalDevice physicalDevice;
    private final boolean memoryBudget;
    private final boolean discrete;
    private final int[] typeFlags;
    private final int[] typeHeap;
    private final long[] heapSizes;
    private final boolean[] vramHeap;
    private final long totalVram;
    private final long minStorageBufferOffsetAlignment;
    private final long nonCoherentAtomSize;
    private final long blockSize;
    private final long maxAllocSize;
    private final int sharingMode;
    private final int[] sharingFamilies;

    private final List<List<VkMemoryBlock>> blocksByType;
    private final Set<VkMemoryBlock> dedicatedBlocks = new LinkedHashSet<>();
    private final Map<Long, VkGpuBuffer> liveBuffers = new HashMap<>();
    private final List<DeferredRelease> deferred = new ArrayList<>();
    private volatile ReleaseGate releaseGate;
    private final long[] heapBoundBytes;
    private long allocatedBytes;
    private long nextTrackingId = 1;
    private boolean closed;

    /**
     * @param device               logical device
     * @param info                 physical device snapshot (memory types, limits)
     * @param memoryBudget         true when {@code VK_EXT_memory_budget} is enabled
     * @param sharingQueueFamilies queue families that access buffers concurrently; one entry means exclusive
     */
    public VkMemoryAllocator(VkDevice device, VkPhysicalDeviceInfo info, boolean memoryBudget,
                             int[] sharingQueueFamilies) {
        if (device == null) {
            throw new IllegalArgumentException("device must not be null");
        }
        if (info == null) {
            throw new IllegalArgumentException("physical device info must not be null");
        }
        if (sharingQueueFamilies == null || sharingQueueFamilies.length == 0) {
            throw new IllegalArgumentException("at least one queue family is required");
        }
        this.device = device;
        this.physicalDevice = info.handle();
        this.memoryBudget = memoryBudget;
        DeviceClass deviceClass = info.deviceClass();
        this.discrete = deviceClass == DeviceClass.DISCRETE;
        boolean integrated = deviceClass == DeviceClass.INTEGRATED;

        this.typeFlags = info.memoryTypePropertyFlags();
        this.typeHeap = info.memoryTypeHeapIndices();
        this.heapSizes = info.memoryHeapSizes();
        int[] heapFlags = info.memoryHeapFlags();
        if (typeFlags.length == 0 || typeFlags.length != typeHeap.length) {
            throw new IllegalArgumentException("inconsistent memory types: " + typeFlags.length + " flags, "
                    + typeHeap.length + " heap indices");
        }
        if (heapSizes.length == 0 || heapSizes.length != heapFlags.length) {
            throw new IllegalArgumentException("inconsistent memory heaps: " + heapSizes.length + " sizes, "
                    + heapFlags.length + " flags");
        }
        for (int heap : typeHeap) {
            if (heap < 0 || heap >= heapSizes.length) {
                throw new IllegalArgumentException("memory type refers to missing heap " + heap);
            }
        }

        this.vramHeap = new boolean[heapSizes.length];
        long vram = 0;
        long largestHeap = 0;
        for (int i = 0; i < heapSizes.length; i++) {
            boolean deviceLocal = (heapFlags[i] & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            vramHeap[i] = integrated || deviceLocal;
            if (deviceLocal) {
                vram += heapSizes[i];
            }
            largestHeap = Math.max(largestHeap, heapSizes[i]);
        }
        this.totalVram = integrated ? largestHeap : vram;

        this.minStorageBufferOffsetAlignment = positiveOrOne(info.minStorageBufferOffsetAlignment());
        this.nonCoherentAtomSize = positiveOrOne(info.nonCoherentAtomSize());
        long maxMemoryAllocation = requireLimit("maxMemoryAllocationSize", info.maxMemoryAllocationSize());
        long maxStorageRange = requireLimit("maxStorageBufferRange", info.maxStorageBufferRange());
        long maxBuffer = info.maxBufferSize();
        if (maxBuffer < 0) {
            throw new IllegalArgumentException("maxBufferSize must not be negative, got " + maxBuffer);
        }
        this.blockSize = Math.min(readBlockMb() * MIB, maxMemoryAllocation);
        long limit = Math.min(maxMemoryAllocation, maxStorageRange);
        if (maxBuffer > 0) {
            limit = Math.min(limit, maxBuffer);
        }
        this.maxAllocSize = Math.min(limit, largestVramHeapCapacity());

        int[] families = Arrays.stream(sharingQueueFamilies).distinct().toArray();
        this.sharingFamilies = families;
        this.sharingMode = families.length > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;

        List<List<VkMemoryBlock>> perType = new ArrayList<>(typeFlags.length);
        for (int i = 0; i < typeFlags.length; i++) {
            perType.add(new ArrayList<>());
        }
        this.blocksByType = Collections.unmodifiableList(perType);
        this.heapBoundBytes = new long[heapSizes.length];

        LOGGER.debug("Vulkan allocator: block {} MiB, max allocation {} MiB, VRAM {} MiB, budget={}, sharing={}",
                blockSize / MIB, maxAllocSize / MIB, totalVram / MIB, memoryBudget, Arrays.toString(families));
    }

    /**
     * Creates a buffer bound to suitable memory; dedicated memory is used for large buffers
     * or when the driver prefers it.
     *
     * @param usage          {@code VkBufferUsageFlags}
     * @param requiredFlags  memory property flags the type must have
     * @param preferredFlags memory property flags preferred when several types qualify
     * @throws GpuOutOfMemoryException when the size exceeds {@link #maxAllocSize()} or memory is exhausted
     */
    public VkGpuBuffer createBuffer(long size, int usage, int requiredFlags, int preferredFlags) {
        return createBuffer(size, usage, requiredFlags, preferredFlags, false);
    }

    /**
     * Creates a buffer bound to suitable memory.
     *
     * <p>When memory is exhausted while closed buffers still wait for device work, the allocator
     * waits for that work ({@link ReleaseGate#await(Object)}), destroys those buffers and retries
     * once; the first failure is then suppressed on a failure of the retry.</p>
     *
     * @param forceDedicated true to give the buffer its own {@code VkDeviceMemory} regardless of size
     * @throws GpuOutOfMemoryException when the size exceeds {@link #maxAllocSize()} or memory is
     *                                 exhausted after the retry
     * @throws GpuBackendException     when no memory type has the required flags or a driver call fails
     */
    public VkGpuBuffer createBuffer(long size, int usage, int requiredFlags, int preferredFlags,
                                    boolean forceDedicated) {
        validateRequest(size, usage, requiredFlags);
        try {
            return allocateBuffer(size, usage, requiredFlags, preferredFlags, forceDedicated);
        } catch (GpuOutOfMemoryException first) {
            if (!awaitDeferredReleases(first)) {
                throw first;
            }
            try {
                return allocateBuffer(size, usage, requiredFlags, preferredFlags, forceDedicated);
            } catch (RuntimeException second) {
                second.addSuppressed(first);
                throw second;
            }
        }
    }

    /** Checks the arguments of {@link #createBuffer}; a size above {@link #maxAllocSize()} is never retried. */
    private void validateRequest(long size, int usage, int requiredFlags) {
        if (size <= 0) {
            throw new IllegalArgumentException("buffer size must be positive, got " + size);
        }
        if (usage == 0) {
            throw new IllegalArgumentException("buffer usage flags must not be empty");
        }
        if ((requiredFlags & NEVER_ALLOCATE) != 0) {
            throw new IllegalArgumentException("unsupported required memory property flags 0x"
                    + Integer.toHexString(requiredFlags & NEVER_ALLOCATE));
        }
        if (size > maxAllocSize) {
            throw new GpuOutOfMemoryException("Vulkan buffer exceeds the largest single allocation of this device",
                    size, maxAllocSize, VkErrors.BACKEND);
        }
    }

    /** One allocation attempt under the allocator's lock, after destroying the closed buffers whose work completed. */
    private synchronized VkGpuBuffer allocateBuffer(long size, int usage, int requiredFlags, int preferredFlags,
                                                    boolean forceDedicated) {
        ensureOpen();
        reclaimDeferredLocked();

        long vkBuffer = createVkBuffer(size, usage);
        VkMemoryBlock.Allocation allocation = null;
        try {
            Requirements req = requirements(vkBuffer);
            int[] candidates = rankMemoryTypes(req.typeBits, requiredFlags, preferredFlags);
            if (candidates.length == 0) {
                throw new GpuBackendException(VkErrors.BACKEND, VK_ERROR_FEATURE_NOT_PRESENT,
                        "no memory type has the required property flags 0x" + Integer.toHexString(requiredFlags)
                                + " (allowed memory types 0x" + Integer.toHexString(req.typeBits) + ")");
            }
            boolean dedicated = forceDedicated || req.requiresDedicated || req.prefersDedicated
                    || req.size >= blockSize / 2;
            allocation = allocate(candidates, req, dedicated, vkBuffer, size);
            VkErrors.check(vkBindBufferMemory(device, vkBuffer, allocation.block().memory(), allocation.offset()),
                    "vkBindBufferMemory");
        } catch (RuntimeException e) {
            if (allocation != null) {
                free(allocation);
            }
            vkDestroyBuffer(device, vkBuffer, null);
            throw e;
        }

        VkGpuBuffer buffer = new VkGpuBuffer(this, vkBuffer, allocation, size, nextTrackingId++);
        liveBuffers.put(buffer.trackingId(), buffer);
        allocatedBytes += size;
        return buffer;
    }

    /**
     * Returns an allocation's memory to its block, or frees a dedicated allocation. Buffers
     * release their memory through {@link VkGpuBuffer#close()}; this method frees only the
     * memory range. It has no effect once the allocator is closed.
     */
    public synchronized void free(VkMemoryBlock.Allocation allocation) {
        if (allocation == null) {
            throw new IllegalArgumentException("allocation must not be null");
        }
        if (closed) {
            return;
        }
        VkMemoryBlock block = allocation.block();
        block.free(allocation);
        heapBoundBytes[typeHeap[block.memoryTypeIndex()]] -= allocation.size();
        if (block.isDedicated()) {
            dedicatedBlocks.remove(block);
            block.close();
            return;
        }
        if (block.isEmpty()) {
            List<VkMemoryBlock> blocks = blocksByType.get(block.memoryTypeIndex());
            if (blocks.size() > 1) {
                blocks.remove(block);
                block.close();
            }
        }
    }

    /** Sum of the requested sizes of live buffers. */
    public synchronized long allocatedBytes() {
        return allocatedBytes;
    }

    /** Number of live buffers. */
    public synchronized int liveBufferCount() {
        return liveBuffers.size();
    }

    /** Number of closed buffers whose destruction waits for device work to complete. */
    public synchronized int pendingReleaseCount() {
        return deferred.size();
    }

    /**
     * Installs the gate that defers destroying closed buffers until the device work recorded before
     * their close has completed; null destroys closed buffers immediately.
     */
    public void setReleaseGate(ReleaseGate gate) {
        this.releaseGate = gate;
    }

    /** Destroys every closed buffer whose device work has completed; the backend calls it as work retires. */
    public synchronized void reclaimDeferred() {
        reclaimDeferredLocked();
    }

    /**
     * Free device memory. With {@code VK_EXT_memory_budget}: the sum over device-local heaps
     * (every heap on an integrated GPU) of {@code min(heapBudget, heapSize) - heapUsage}, floored
     * at 0 per heap (llama.cpp ggml-vulkan.cpp:15305-15319). Without it: total VRAM minus the memory bound
     * to this allocator's buffers in those heaps.
     */
    public synchronized long freeVramBytes() {
        if (memoryBudget) {
            long[][] budgetAndUsage = queryBudget();
            long free = 0;
            for (int i = 0; i < vramHeap.length; i++) {
                if (!vramHeap[i]) {
                    continue;
                }
                long budget = Math.min(budgetAndUsage[0][i], heapSizes[i]);
                long usage = budgetAndUsage[1][i];
                free += budget > usage ? budget - usage : 0;
            }
            return free;
        }
        long bound = 0;
        for (int i = 0; i < vramHeap.length; i++) {
            if (vramHeap[i]) {
                bound += heapBoundBytes[i];
            }
        }
        return Math.max(0, totalVram - bound);
    }

    /** Device memory: the sum of device-local heaps, or the largest heap on an integrated GPU. */
    public long totalVramBytes() {
        return totalVram;
    }

    /**
     * Largest single buffer: min(maxMemoryAllocationSize, maxStorageBufferRange, maxBufferSize,
     * the largest device heap's budget at creation).
     */
    public long maxAllocSize() {
        return maxAllocSize;
    }

    public long blockSize() {
        return blockSize;
    }

    public VkDevice device() {
        return device;
    }

    /** {@code VkMemoryPropertyFlags} of a memory type. */
    public int memoryPropertyFlags(int memoryTypeIndex) {
        return typeFlags[memoryTypeIndex];
    }

    /** True when the memory type is host-coherent. */
    public boolean isHostCoherent(int memoryTypeIndex) {
        return (typeFlags[memoryTypeIndex] & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }

    /**
     * Makes host writes to {@code [offset, offset+size)} of a mapped buffer visible to the device;
     * no-op for coherent memory.
     */
    public void flush(VkGpuBuffer buffer, long offset, long size) {
        syncMappedRange(buffer, offset, size, true);
    }

    /**
     * Makes device writes to {@code [offset, offset+size)} of a mapped buffer visible to the host;
     * no-op for coherent memory.
     */
    public void invalidate(VkGpuBuffer buffer, long offset, long size) {
        syncMappedRange(buffer, offset, size, false);
    }

    /**
     * Destroys every live buffer and every closed buffer still waiting for its release, and frees
     * every block; idempotent. The caller ensures the device no longer uses any of them.
     */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (!liveBuffers.isEmpty()) {
            LOGGER.debug("Vulkan allocator closing with {} live buffers ({} bytes)", liveBuffers.size(),
                    allocatedBytes);
        }
        for (VkGpuBuffer buffer : liveBuffers.values()) {
            buffer.markClosed();
            vkDestroyBuffer(device, buffer.vkBuffer(), null);
        }
        liveBuffers.clear();
        for (DeferredRelease d : deferred) {
            vkDestroyBuffer(device, d.buffer.vkBuffer(), null);
        }
        deferred.clear();
        for (List<VkMemoryBlock> blocks : blocksByType) {
            for (VkMemoryBlock block : blocks) {
                block.close();
            }
            blocks.clear();
        }
        for (VkMemoryBlock block : dedicatedBlocks) {
            block.close();
        }
        dedicatedBlocks.clear();
        Arrays.fill(heapBoundBytes, 0);
        allocatedBytes = 0;
    }

    /**
     * Closes a buffer: it leaves the accounting at once, and its {@code VkBuffer} and memory are
     * destroyed now or, when {@code gated} and device work recorded before this call is outstanding,
     * once that work completes.
     */
    void release(VkGpuBuffer buffer, boolean gated) {
        ReleaseGate gate = gated ? releaseGate : null;
        Object outstanding = gate == null ? null : gate.capture();
        synchronized (this) {
            if (!buffer.markClosed() || closed) {
                return;
            }
            liveBuffers.remove(buffer.trackingId());
            allocatedBytes -= buffer.sizeInBytes();
            if (outstanding == null) {
                destroy(buffer);
            } else {
                deferred.add(new DeferredRelease(buffer, outstanding));
            }
        }
    }

    private void destroy(VkGpuBuffer buffer) {
        vkDestroyBuffer(device, buffer.vkBuffer(), null);
        free(buffer.allocation());
    }

    /**
     * Waits, without the allocator's lock, for the device work of every closed buffer still waiting
     * for its release, newest first so the older waits are already satisfied.
     *
     * @return true when there was such work and the wait succeeded, so a retry may succeed; a
     *         failure of the wait is suppressed on {@code failure}
     */
    private boolean awaitDeferredReleases(GpuOutOfMemoryException failure) {
        ReleaseGate gate = releaseGate;
        List<Object> tokens = new ArrayList<>();
        synchronized (this) {
            if (closed || gate == null) {
                return false;
            }
            for (DeferredRelease d : deferred) {
                tokens.add(d.outstanding);
            }
        }
        if (tokens.isEmpty()) {
            return false;
        }
        LOGGER.debug("Vulkan memory exhausted with {} closed buffers waiting for device work; waiting for it and "
                + "retrying", tokens.size());
        try {
            for (int i = tokens.size() - 1; i >= 0; i--) {
                gate.await(tokens.get(i));
            }
        } catch (RuntimeException e) {
            failure.addSuppressed(e);
            return false;
        }
        return true;
    }

    private void reclaimDeferredLocked() {
        if (closed || deferred.isEmpty()) {
            return;
        }
        ReleaseGate gate = releaseGate;
        Iterator<DeferredRelease> it = deferred.iterator();
        while (it.hasNext()) {
            DeferredRelease d = it.next();
            if (gate == null || gate.passed(d.outstanding)) {
                it.remove();
                destroy(d.buffer);
            }
        }
    }

    private VkMemoryBlock.Allocation allocate(int[] candidates, Requirements req, boolean dedicated, long vkBuffer,
                                              long size) {
        GpuOutOfMemoryException lastFailure = null;
        for (int type : candidates) {
            try {
                VkMemoryBlock.Allocation allocation = dedicated
                        ? allocateDedicated(type, req.size, vkBuffer)
                        : allocateFromBlocks(type, req.size, alignmentFor(type, req.alignment), vkBuffer);
                heapBoundBytes[typeHeap[type]] += allocation.size();
                return allocation;
            } catch (GpuOutOfMemoryException e) {
                lastFailure = e;
                LOGGER.debug("Vulkan memory type {} exhausted for {} bytes: {}", type, req.size, e.getMessage());
            }
        }
        throw new GpuOutOfMemoryException("Vulkan memory exhausted in every suitable memory type ("
                + (lastFailure != null ? lastFailure.getMessage() : "no candidate") + ")",
                size, freeVramBytes(), VkErrors.BACKEND);
    }

    private VkMemoryBlock.Allocation allocateFromBlocks(int type, long size, long alignment, long vkBuffer) {
        List<VkMemoryBlock> blocks = blocksByType.get(type);
        for (VkMemoryBlock block : blocks) {
            VkMemoryBlock.Allocation allocation = block.allocate(size, alignment);
            if (allocation != null) {
                return allocation;
            }
        }
        VkMemoryBlock block;
        try {
            block = allocateMemory(type, blockSize, VK_NULL_HANDLE, false);
        } catch (GpuOutOfMemoryException e) {
            LOGGER.debug("Vulkan block of {} bytes unavailable in memory type {}; allocating {} bytes exactly",
                    blockSize, type, size);
            return allocateDedicated(type, size, vkBuffer);
        }
        blocks.add(block);
        VkMemoryBlock.Allocation allocation = block.allocate(size, alignment);
        if (allocation == null) {
            throw new IllegalStateException("fresh block of " + blockSize + " bytes cannot hold " + size + " bytes");
        }
        return allocation;
    }

    private VkMemoryBlock.Allocation allocateDedicated(int type, long size, long vkBuffer) {
        VkMemoryBlock block = allocateMemory(type, size, vkBuffer, true);
        dedicatedBlocks.add(block);
        return block.allocate(size, 1);
    }

    private VkMemoryBlock allocateMemory(int type, long bytes, long dedicatedBuffer, boolean dedicated) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkMemoryAllocateInfo info = VkMemoryAllocateInfo.calloc(stack)
                    .sType$Default()
                    .allocationSize(bytes)
                    .memoryTypeIndex(type);
            if (dedicatedBuffer != VK_NULL_HANDLE) {
                info.pNext(VkMemoryDedicatedAllocateInfo.calloc(stack).sType$Default().buffer(dedicatedBuffer));
            }
            LongBuffer pMemory = stack.mallocLong(1);
            int result = vkAllocateMemory(device, info, null, pMemory);
            if (result != VK_SUCCESS) {
                throw VkErrors.translate(result, "vkAllocateMemory of " + bytes + " bytes from memory type " + type,
                        bytes);
            }
            long memory = pMemory.get(0);

            long hostPtr = 0;
            if ((typeFlags[type] & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
                PointerBuffer pData = stack.mallocPointer(1);
                result = vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, pData);
                if (result != VK_SUCCESS) {
                    vkFreeMemory(device, memory, null);
                    throw VkErrors.translate(result, "vkMapMemory of " + bytes + " bytes", bytes);
                }
                hostPtr = pData.get(0);
            }
            return new VkMemoryBlock(device, memory, bytes, type, dedicated, hostPtr);
        }
    }

    private long createVkBuffer(long size, int usage) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkBufferCreateInfo info = VkBufferCreateInfo.calloc(stack)
                    .sType$Default()
                    .size(size)
                    .usage(usage)
                    .sharingMode(sharingMode);
            if (sharingMode == VK_SHARING_MODE_CONCURRENT) {
                info.pQueueFamilyIndices(stack.ints(sharingFamilies));
                info.queueFamilyIndexCount(sharingFamilies.length);
            }
            LongBuffer pBuffer = stack.mallocLong(1);
            VkErrors.check(vkCreateBuffer(device, info, null, pBuffer), "vkCreateBuffer of " + size + " bytes");
            return pBuffer.get(0);
        }
    }

    private Requirements requirements(long vkBuffer) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkMemoryDedicatedRequirements dedicated = VkMemoryDedicatedRequirements.calloc(stack).sType$Default();
            VkMemoryRequirements2 req2 = VkMemoryRequirements2.calloc(stack).sType$Default().pNext(dedicated);
            VkBufferMemoryRequirementsInfo2 query = VkBufferMemoryRequirementsInfo2.calloc(stack)
                    .sType$Default()
                    .buffer(vkBuffer);
            vkGetBufferMemoryRequirements2(device, query, req2);
            VkMemoryRequirements req = req2.memoryRequirements();
            return new Requirements(req.size(), req.alignment(), req.memoryTypeBits(),
                    dedicated.prefersDedicatedAllocation(), dedicated.requiresDedicatedAllocation());
        }
    }

    /** Candidate memory types in preference order (see the class description). */
    int[] rankMemoryTypes(int typeBits, int requiredFlags, int preferredFlags) {
        int unwanted = discrete ? PLACEMENT_FLAGS & ~(requiredFlags | preferredFlags) : 0;
        List<int[]> scored = new ArrayList<>();
        for (int i = 0; i < typeFlags.length; i++) {
            int flags = typeFlags[i];
            if ((typeBits & (1 << i)) == 0 || (flags & requiredFlags) != requiredFlags
                    || (flags & NEVER_ALLOCATE) != 0) {
                continue;
            }
            scored.add(new int[]{i, Integer.bitCount(flags & preferredFlags), Integer.bitCount(flags & unwanted)});
        }
        scored.sort((a, b) -> {
            if (a[1] != b[1]) {
                return Integer.compare(b[1], a[1]);
            }
            if (a[2] != b[2]) {
                return Integer.compare(a[2], b[2]);
            }
            return Integer.compare(a[0], b[0]);
        });
        int[] order = new int[scored.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = scored.get(i)[0];
        }
        return order;
    }

    private long alignmentFor(int type, long requiredAlignment) {
        long alignment = lcm(lcm(positiveOrOne(requiredAlignment), minStorageBufferOffsetAlignment), MIN_ALIGNMENT);
        int flags = typeFlags[type];
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 && (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            alignment = lcm(alignment, nonCoherentAtomSize);
        }
        return alignment;
    }

    private void syncMappedRange(VkGpuBuffer buffer, long offset, long size, boolean flush) {
        if (buffer == null) {
            throw new IllegalArgumentException("buffer must not be null");
        }
        if (offset < 0 || size <= 0 || offset + size > buffer.sizeInBytes()) {
            throw new IllegalArgumentException("range [" + offset + ", " + (offset + size)
                    + ") is outside the buffer of " + buffer.sizeInBytes() + " bytes");
        }
        if (!buffer.isHostVisible()) {
            throw new IllegalStateException("buffer " + buffer + " is not host-visible");
        }
        if (buffer.isHostCoherent()) {
            return;
        }
        VkMemoryBlock block = buffer.allocation().block();
        long start = buffer.memoryOffset() + offset;
        long alignedStart = start - start % nonCoherentAtomSize;
        long alignedEnd = Math.min(VkMemoryBlock.alignUp(start + size, nonCoherentAtomSize), block.size());
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkMappedMemoryRange range = VkMappedMemoryRange.calloc(stack)
                    .sType$Default()
                    .memory(block.memory())
                    .offset(alignedStart)
                    .size(alignedEnd - alignedStart);
            if (flush) {
                VkErrors.check(vkFlushMappedMemoryRanges(device, range), "vkFlushMappedMemoryRanges");
            } else {
                VkErrors.check(vkInvalidateMappedMemoryRanges(device, range), "vkInvalidateMappedMemoryRanges");
            }
        }
    }

    /** Per-heap {@code heapBudget} and {@code heapUsage} from {@code VK_EXT_memory_budget}. */
    private long[][] queryBudget() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget =
                    VkPhysicalDeviceMemoryBudgetPropertiesEXT.calloc(stack).sType$Default();
            VkPhysicalDeviceMemoryProperties2 props =
                    VkPhysicalDeviceMemoryProperties2.calloc(stack).sType$Default().pNext(budget);
            vkGetPhysicalDeviceMemoryProperties2(physicalDevice, props);
            int heaps = Math.min(props.memoryProperties().memoryHeapCount(), heapSizes.length);
            long[] budgets = new long[heapSizes.length];
            long[] usages = new long[heapSizes.length];
            for (int i = 0; i < heaps; i++) {
                budgets[i] = budget.heapBudget(i);
                usages[i] = budget.heapUsage(i);
            }
            return new long[][]{budgets, usages};
        }
    }

    private long largestVramHeapCapacity() {
        long[] budgets = memoryBudget ? queryBudget()[0] : heapSizes;
        long largest = 0;
        for (int i = 0; i < heapSizes.length; i++) {
            if (vramHeap[i]) {
                largest = Math.max(largest, Math.min(budgets[i], heapSizes[i]));
            }
        }
        if (largest <= 0) {
            throw new IllegalArgumentException("device reports no usable device-local memory heap");
        }
        return largest;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("VkMemoryAllocator already closed");
        }
    }

    private static long readBlockMb() {
        String value = System.getProperty(BLOCK_MB_PROPERTY);
        if (value == null || value.trim().isEmpty()) {
            return DEFAULT_BLOCK_MB;
        }
        try {
            long mb = Long.parseLong(value.trim());
            if (mb > 0 && mb <= Integer.MAX_VALUE) {
                return mb;
            }
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("-D" + BLOCK_MB_PROPERTY + " must be a positive integer (MiB), got '"
                    + value + "'", e);
        }
        throw new IllegalArgumentException("-D" + BLOCK_MB_PROPERTY + " must be a positive integer (MiB), got '"
                + value + "'");
    }

    private static long requireLimit(String name, long value) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive (uint32 limits are widened unsigned), got "
                    + value);
        }
        return value;
    }

    private static long positiveOrOne(long value) {
        return value > 0 ? value : 1;
    }

    private static long lcm(long a, long b) {
        return a / gcd(a, b) * b;
    }

    private static long gcd(long a, long b) {
        while (b != 0) {
            long t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    /** A closed buffer waiting for the device work captured at its close. */
    private static final class DeferredRelease {
        final VkGpuBuffer buffer;
        final Object outstanding;

        DeferredRelease(VkGpuBuffer buffer, Object outstanding) {
            this.buffer = buffer;
            this.outstanding = outstanding;
        }
    }

    /** Memory requirements of one buffer. */
    private static final class Requirements {
        final long size;
        final long alignment;
        final int typeBits;
        final boolean prefersDedicated;
        final boolean requiresDedicated;

        Requirements(long size, long alignment, int typeBits, boolean prefersDedicated, boolean requiresDedicated) {
            this.size = size;
            this.alignment = alignment;
            this.typeBits = typeBits;
            this.prefersDedicated = prefersDedicated;
            this.requiresDedicated = requiresDedicated;
        }
    }
}
