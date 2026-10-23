package com.eonmux.neuralfree.acceleration.backend.vulkan;

import org.lwjgl.vulkan.VkDevice;

import java.util.Map;
import java.util.TreeMap;

import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.vkFreeMemory;
import static org.lwjgl.vulkan.VK10.vkUnmapMemory;

/**
 * One {@code VkDeviceMemory} allocation with a coalescing first-fit free list
 * and, for host-visible memory, a persistent mapping.
 *
 * <p>Not thread-safe: every call is made under the lock of the owning
 * {@link VkMemoryAllocator}.</p>
 */
public final class VkMemoryBlock implements AutoCloseable {

    private final VkDevice device;
    private final long memory;
    private final long size;
    private final int memoryTypeIndex;
    private final boolean dedicated;
    private final long hostPtr;

    /** Free ranges: start offset to length, sorted by offset, never adjacent. */
    private final TreeMap<Long, Long> freeRanges = new TreeMap<>();
    private long freeBytes;
    private int liveAllocations;
    private boolean closed;

    /**
     * @param memory          the {@code VkDeviceMemory} handle; ownership passes to this block
     * @param size            allocation size in bytes
     * @param memoryTypeIndex memory type the allocation was made from
     * @param dedicated       true when the memory backs exactly one buffer
     * @param hostPtr         address of the persistent mapping, or 0 when not mapped
     */
    public VkMemoryBlock(VkDevice device, long memory, long size, int memoryTypeIndex, boolean dedicated,
                         long hostPtr) {
        if (device == null) {
            throw new IllegalArgumentException("device must not be null");
        }
        if (memory == VK_NULL_HANDLE) {
            throw new IllegalArgumentException("memory must not be VK_NULL_HANDLE");
        }
        if (size <= 0) {
            throw new IllegalArgumentException("block size must be positive, got " + size);
        }
        if (memoryTypeIndex < 0) {
            throw new IllegalArgumentException("memoryTypeIndex must not be negative, got " + memoryTypeIndex);
        }
        this.device = device;
        this.memory = memory;
        this.size = size;
        this.memoryTypeIndex = memoryTypeIndex;
        this.dedicated = dedicated;
        this.hostPtr = hostPtr;
        this.freeRanges.put(0L, size);
        this.freeBytes = size;
    }

    /**
     * Allocates {@code size} bytes aligned to {@code alignment} from the lowest-offset
     * free range that fits. Any alignment padding stays in the free list.
     *
     * @return the allocation, or null when no free range fits
     */
    public Allocation allocate(long size, long alignment) {
        ensureOpen();
        if (size <= 0) {
            throw new IllegalArgumentException("allocation size must be positive, got " + size);
        }
        if (alignment <= 0) {
            throw new IllegalArgumentException("alignment must be positive, got " + alignment);
        }
        if (size > freeBytes) {
            return null;
        }
        for (Map.Entry<Long, Long> range : freeRanges.entrySet()) {
            long start = range.getKey();
            long end = start + range.getValue();
            long aligned = alignUp(start, alignment);
            if (aligned > end || end - aligned < size) {
                continue;
            }
            freeRanges.remove(start);
            if (aligned > start) {
                freeRanges.put(start, aligned - start);
            }
            long allocationEnd = aligned + size;
            if (allocationEnd < end) {
                freeRanges.put(allocationEnd, end - allocationEnd);
            }
            freeBytes -= size;
            liveAllocations++;
            return new Allocation(this, aligned, size);
        }
        return null;
    }

    /**
     * Returns a range to the free list, coalescing it with free neighbours.
     *
     * @throws IllegalArgumentException when the allocation belongs to another block
     * @throws IllegalStateException    when the allocation was already freed or overlaps a free range
     */
    public void free(Allocation allocation) {
        if (allocation == null) {
            throw new IllegalArgumentException("allocation must not be null");
        }
        if (allocation.block != this) {
            throw new IllegalArgumentException("allocation belongs to another memory block");
        }
        if (allocation.freed) {
            throw new IllegalStateException("allocation at offset " + allocation.offset + " was already freed");
        }
        long start = allocation.offset;
        long end = start + allocation.size;
        Map.Entry<Long, Long> below = freeRanges.floorEntry(start);
        Map.Entry<Long, Long> above = freeRanges.ceilingEntry(start);
        if (below != null && below.getKey() + below.getValue() > start) {
            throw new IllegalStateException("allocation [" + start + ", " + end + ") overlaps free range starting at "
                    + below.getKey());
        }
        if (above != null && above.getKey() < end) {
            throw new IllegalStateException("allocation [" + start + ", " + end + ") overlaps free range starting at "
                    + above.getKey());
        }
        long mergedStart = start;
        long mergedEnd = end;
        if (below != null && below.getKey() + below.getValue() == start) {
            mergedStart = below.getKey();
            freeRanges.remove(below.getKey());
        }
        if (above != null && above.getKey() == end) {
            mergedEnd = end + above.getValue();
            freeRanges.remove(above.getKey());
        }
        freeRanges.put(mergedStart, mergedEnd - mergedStart);
        freeBytes += allocation.size;
        liveAllocations--;
        allocation.freed = true;
    }

    /** The {@code VkDeviceMemory} handle. */
    public long memory() {
        return memory;
    }

    public long size() {
        return size;
    }

    public int memoryTypeIndex() {
        return memoryTypeIndex;
    }

    /** True when the memory backs exactly one buffer and is freed with it. */
    public boolean isDedicated() {
        return dedicated;
    }

    /** Address of the persistent mapping, or 0. */
    public long hostPtr() {
        return hostPtr;
    }

    public long freeBytes() {
        return freeBytes;
    }

    /** Number of free ranges; 1 for an empty block. */
    public int freeRangeCount() {
        return freeRanges.size();
    }

    /** True when no allocation is live. */
    public boolean isEmpty() {
        return liveAllocations == 0;
    }

    public boolean isClosed() {
        return closed;
    }

    /** Unmaps and frees the memory; idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (hostPtr != 0) {
            vkUnmapMemory(device, memory);
        }
        vkFreeMemory(device, memory, null);
    }

    @Override
    public String toString() {
        return "VkMemoryBlock{type=" + memoryTypeIndex + ", size=" + size + ", free=" + freeBytes
                + ", live=" + liveAllocations + (dedicated ? ", dedicated" : "") + (hostPtr != 0 ? ", mapped" : "")
                + '}';
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("memory block already closed");
        }
    }

    static long alignUp(long value, long alignment) {
        long remainder = value % alignment;
        return remainder == 0 ? value : value + (alignment - remainder);
    }

    /** One allocated range of a block. */
    public static final class Allocation {

        private final VkMemoryBlock block;
        private final long offset;
        private final long size;
        private boolean freed;

        private Allocation(VkMemoryBlock block, long offset, long size) {
            this.block = block;
            this.offset = offset;
            this.size = size;
        }

        public VkMemoryBlock block() {
            return block;
        }

        /** Byte offset of the range within the block's memory. */
        public long offset() {
            return offset;
        }

        /** Size of the range in bytes. */
        public long size() {
            return size;
        }

        /** Host address of the range, or 0 when the block is not mapped. */
        public long hostPtr() {
            return block.hostPtr == 0 ? 0 : block.hostPtr + offset;
        }

        /** True once the range has been returned to its block. */
        public boolean isFreed() {
            return freed;
        }

        @Override
        public String toString() {
            return "Allocation{offset=" + offset + ", size=" + size + ", block=" + block + '}';
        }
    }
}
