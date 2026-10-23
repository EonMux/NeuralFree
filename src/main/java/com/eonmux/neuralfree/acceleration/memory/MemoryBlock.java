package com.eonmux.neuralfree.acceleration.memory;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

public final class MemoryBlock {
    private final GpuBuffer buffer;
    private final long sizeBytes;
    private final long offset;
    private final boolean inUse;
    private final long lastUsedTimestamp;

    public MemoryBlock(GpuBuffer buffer, long sizeBytes, long offset, boolean inUse) {
        this.buffer = buffer;
        this.sizeBytes = sizeBytes;
        this.offset = offset;
        this.inUse = inUse;
        this.lastUsedTimestamp = System.nanoTime();
    }

    public MemoryBlock(GpuBuffer buffer, long sizeBytes, long offset, boolean inUse, long lastUsedTimestamp) {
        this.buffer = buffer;
        this.sizeBytes = sizeBytes;
        this.offset = offset;
        this.inUse = inUse;
        this.lastUsedTimestamp = lastUsedTimestamp;
    }

    public static MemoryBlock create(GpuBuffer buffer) {
        return new MemoryBlock(buffer, buffer.sizeInBytes(), 0, false);
    }

    public GpuBuffer getBuffer() {
        return buffer;
    }

    public long getSizeBytes() {
        return sizeBytes;
    }

    public long getOffset() {
        return offset;
    }

    public boolean isInUse() {
        return inUse;
    }

    public long getLastUsedTimestamp() {
        return lastUsedTimestamp;
    }

    public MemoryBlock withInUse(boolean inUse) {
        return new MemoryBlock(this.buffer, this.sizeBytes, this.offset, inUse, System.nanoTime());
    }

    public MemoryBlock withUpdatedTimestamp() {
        return new MemoryBlock(this.buffer, this.sizeBytes, this.offset, this.inUse, System.nanoTime());
    }

    public long bufferId() {
        return buffer != null ? buffer.bufferId() : -1;
    }
}
