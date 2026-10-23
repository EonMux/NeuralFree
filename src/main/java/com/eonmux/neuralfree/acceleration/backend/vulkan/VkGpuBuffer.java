package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;

/**
 * A {@code VkBuffer} bound to a range of memory from a {@link VkMemoryAllocator}.
 *
 * <p>Every buffer owns its {@code VkBuffer}, so descriptors and copies address it at
 * {@link #offset()} 0 with {@link #range()} equal to its size. {@link #trackingId()} is unique
 * per allocator and identifies the buffer to the hazard tracker. Closing is idempotent: the buffer
 * becomes unusable at once, and the allocator destroys the {@code VkBuffer} and takes the memory back
 * when the device work recorded before the close has completed. It has no effect once the allocator
 * itself is closed.</p>
 */
public final class VkGpuBuffer implements GpuBuffer {

    private final VkMemoryAllocator owner;
    private final long vkBuffer;
    private final VkMemoryBlock.Allocation allocation;
    private final long size;
    private final long trackingId;
    private volatile boolean closed;

    /**
     * @param owner      allocator that created the buffer and receives its memory back
     * @param vkBuffer   the {@code VkBuffer} handle, already bound to {@code allocation}
     * @param allocation memory range bound to the buffer
     * @param size       requested size in bytes (at most the allocation size)
     * @param id         tracking id, unique per allocator
     */
    public VkGpuBuffer(VkMemoryAllocator owner, long vkBuffer, VkMemoryBlock.Allocation allocation, long size,
                       long id) {
        if (owner == null) {
            throw new IllegalArgumentException("owner must not be null");
        }
        if (vkBuffer == VK_NULL_HANDLE) {
            throw new IllegalArgumentException("vkBuffer must not be VK_NULL_HANDLE");
        }
        if (allocation == null) {
            throw new IllegalArgumentException("allocation must not be null");
        }
        if (size <= 0) {
            throw new IllegalArgumentException("size must be positive, got " + size);
        }
        if (size > allocation.size()) {
            throw new IllegalArgumentException("size " + size + " exceeds the bound allocation of "
                    + allocation.size() + " bytes");
        }
        this.owner = owner;
        this.vkBuffer = vkBuffer;
        this.allocation = allocation;
        this.size = size;
        this.trackingId = id;
    }

    @Override
    public long sizeInBytes() {
        return size;
    }

    /** The {@code VkBuffer} handle, or -1 once closed. */
    @Override
    public long bufferId() {
        return closed ? -1 : vkBuffer;
    }

    /** Monotonic id used by hazard tracking; never reused within one allocator. */
    public long trackingId() {
        return trackingId;
    }

    /** The {@code VkBuffer} handle. */
    public long vkBuffer() {
        return vkBuffer;
    }

    /** Byte offset of this buffer's range within {@link #vkBuffer()} for descriptors and copies. */
    public long offset() {
        return 0;
    }

    /** Byte length of this buffer's range for descriptors and copies. */
    public long range() {
        return size;
    }

    /** The memory range bound to the buffer. */
    public VkMemoryBlock.Allocation allocation() {
        return allocation;
    }

    /** The {@code VkDeviceMemory} the buffer is bound to. */
    public long memory() {
        return allocation.block().memory();
    }

    /** Byte offset of the buffer within {@link #memory()}. */
    public long memoryOffset() {
        return allocation.offset();
    }

    /** Memory type index of the bound memory. */
    public int memoryTypeIndex() {
        return allocation.block().memoryTypeIndex();
    }

    /** Host address of the buffer's first byte, or 0 when its memory is not host-visible. */
    public long hostPtr() {
        return allocation.hostPtr();
    }

    /** True when the buffer is persistently mapped. */
    public boolean isHostVisible() {
        return allocation.hostPtr() != 0;
    }

    /** True when host writes and device writes are visible without explicit flush or invalidate. */
    public boolean isHostCoherent() {
        return owner.isHostCoherent(allocation.block().memoryTypeIndex());
    }

    /** The allocator that owns the memory. */
    public VkMemoryAllocator owner() {
        return owner;
    }

    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (!closed) {
            owner.release(this, true);
        }
    }

    /**
     * Closes the buffer and destroys it at once, bypassing the allocator's release gate; for
     * buffers whose last device use the caller has already seen complete.
     */
    void closeRetired() {
        if (!closed) {
            owner.release(this, false);
        }
    }

    /**
     * Marks the buffer closed; called by the owner under its lock.
     *
     * @return true when this call closed the buffer, false when it was already closed
     */
    boolean markClosed() {
        if (closed) {
            return false;
        }
        closed = true;
        return true;
    }

    @Override
    public String toString() {
        return "VkGpuBuffer{id=" + trackingId + ", size=" + size + ", type=" + allocation.block().memoryTypeIndex()
                + (closed ? ", closed" : "") + '}';
    }
}
