package com.eonmux.neuralfree.acceleration.pipeline;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.memory.GpuMemoryPool;
import com.eonmux.neuralfree.acceleration.memory.MemoryBlock;
import com.eonmux.neuralfree.acceleration.memory.PoolConfig;
import com.eonmux.neuralfree.acceleration.pipeline.schedule.PipelineConfig;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

@NotThreadSafe("Ring buffer operations (advance, beginPrefetch) are not synchronized. "
        + "Caller must ensure external synchronization when used from multiple threads. "
        + "Typical use case: single producer (pipeline scheduler) advances the ring buffer. "
        + "For multi-threaded use, wrap all calls in external synchronization.")
public final class RingBufferManager implements AutoCloseable {
    private static final Logger LOGGER = LoggingUtil.getLogger(RingBufferManager.class);
    
    private final ComputeBackend backend;
    private final ComputeQueue transferQueue;
    private final PipelineConfig config;
    private final int slotCount;
    private final BufferSlot[] slots;
    private int computeSlotIndex;
    private volatile boolean closed;

    public RingBufferManager(ComputeBackend backend, PipelineConfig config) {
        this.backend = Objects.requireNonNull(backend, "backend cannot be null");
        this.config = Objects.requireNonNull(config, "config cannot be null");
        this.transferQueue = backend.transferQueue();
        this.slotCount = config.ringBufferSlotCount();
        this.slots = new BufferSlot[slotCount];
        this.computeSlotIndex = 0;
        this.closed = false;
        
        for (int i = 0; i < slotCount; i++) {
            slots[i] = new BufferSlot(i);
        }
    }

    public BufferSlot activeBuffer() {
        if (closed) {
            throw new IllegalStateException("RingBufferManager is closed");
        }
        return slots[computeSlotIndex];
    }

    public BufferSlot prefetchSlot(int ahead) {
        if (closed) {
            throw new IllegalStateException("RingBufferManager is closed");
        }
        if (ahead < 1 || ahead > slotCount - 1) {
            throw new IllegalArgumentException(
                    String.format("ahead must be between 1 and %d (slotCount-1), got: %d", 
                            slotCount - 1, ahead));
        }
        int slotIndex = (computeSlotIndex + ahead) % slotCount;
        return slots[slotIndex];
    }

    public GpuEvent beginPrefetch(int ahead, float[] hostData, long dstOffset) {
        if (closed) {
            throw new IllegalStateException("RingBufferManager is closed");
        }
        BufferSlot slot = prefetchSlot(ahead);
        
        if (!slot.hasBuffer()) {
            throw new IllegalStateException(
                    String.format("Prefetch slot %d has no allocated buffer. " +
                            "Call allocateSlotBuffer() first.", slot.slotIndex));
        }
        
        GpuBuffer buffer = slot.getBuffer();
        long sizeInBytes = (long) hostData.length * Float.BYTES;
        
        LOGGER.debug("Ring buffer prefetch: slot {} (ahead {}), size {} bytes, offset {}",
                slot.slotIndex, ahead, sizeInBytes, dstOffset);
        
        GpuEvent event = transferQueue.enqueueWriteBuffer(
                buffer, dstOffset, sizeInBytes, hostData);
        
        slot.setPendingEvent(event);
        slot.markPrefetchInProgress();
        
        return event;
    }

    public GpuEvent beginPrefetch(int ahead, byte[] hostData, long dstOffset) {
        if (closed) {
            throw new IllegalStateException("RingBufferManager is closed");
        }
        BufferSlot slot = prefetchSlot(ahead);
        
        if (!slot.hasBuffer()) {
            throw new IllegalStateException(
                    String.format("Prefetch slot %d has no allocated buffer. " +
                            "Call allocateSlotBuffer() first.", slot.slotIndex));
        }
        
        GpuBuffer buffer = slot.getBuffer();
        long sizeInBytes = hostData.length;
        
        GpuEvent event = transferQueue.enqueueWriteBuffer(
                buffer, dstOffset, sizeInBytes, hostData);
        
        slot.setPendingEvent(event);
        slot.markPrefetchInProgress();
        
        return event;
    }

    public void allocateSlotBuffer(int slotIndex, long sizeInBytes) {
        if (closed) {
            throw new IllegalStateException("RingBufferManager is closed");
        }
        if (slotIndex < 0 || slotIndex >= slotCount) {
            throw new IllegalArgumentException(
                    String.format("slotIndex must be between 0 and %d, got: %d", 
                            slotCount - 1, slotIndex));
        }
        
        BufferSlot slot = slots[slotIndex];
        if (slot.hasBuffer()) {
            slot.releaseBuffer();
        }
        
        GpuBuffer buffer = backend.allocateBuffer(sizeInBytes);
        slot.setBuffer(buffer, sizeInBytes);
    }

    public void allocateSlotBufferFromPool(int slotIndex, long sizeInBytes, GpuMemoryPool pool) {
        if (closed) {
            throw new IllegalStateException("RingBufferManager is closed");
        }
        if (slotIndex < 0 || slotIndex >= slotCount) {
            throw new IllegalArgumentException(
                    String.format("slotIndex must be between 0 and %d, got: %d", 
                            slotCount - 1, slotIndex));
        }
        Objects.requireNonNull(pool, "pool cannot be null");
        
        BufferSlot slot = slots[slotIndex];
        if (slot.hasBuffer()) {
            slot.releaseBuffer();
        }
        
        MemoryBlock block = pool.acquire(sizeInBytes);
        slot.setBuffer(block.getBuffer(), block.getSizeBytes());
        slot.setMemoryBlock(block);
    }

    public void advance() {
        if (closed) {
            throw new IllegalStateException("RingBufferManager is closed");
        }
        
        int prevIndex = computeSlotIndex;
        
        BufferSlot currentSlot = slots[computeSlotIndex];
        if (currentSlot.hasPendingEvent()) {
            try {
                currentSlot.getPendingEvent().awaitCompletion();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted while waiting for slot completion", e);
            }
            currentSlot.clearPendingEvent();
        }
        
        currentSlot.markReady();
        computeSlotIndex = (computeSlotIndex + 1) % slotCount;
        
        LOGGER.debug("Ring buffer advance: {} -> {}, slotCount: {}", prevIndex, computeSlotIndex, slotCount);
    }

    public void advanceAsync() {
        if (closed) {
            throw new IllegalStateException("RingBufferManager is closed");
        }
        slots[computeSlotIndex].markReady();
        computeSlotIndex = (computeSlotIndex + 1) % slotCount;
    }

    public void waitForPrefetchCompletion(int ahead) {
        if (closed) {
            throw new IllegalStateException("RingBufferManager is closed");
        }
        BufferSlot slot = prefetchSlot(ahead);
        
        if (slot.hasPendingEvent()) {
            try {
                slot.getPendingEvent().awaitCompletion();
                slot.clearPendingEvent();
                slot.markReady();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted while waiting for prefetch completion", e);
            }
        }
    }

    public boolean isPrefetchComplete(int ahead) {
        if (closed) {
            throw new IllegalStateException("RingBufferManager is closed");
        }
        BufferSlot slot = prefetchSlot(ahead);
        
        if (!slot.hasPendingEvent()) {
            return slot.isReady();
        }
        
        boolean complete = slot.getPendingEvent().isComplete();
        if (complete) {
            slot.clearPendingEvent();
            slot.markReady();
        }
        return complete;
    }

    public int getSlotCount() {
        return slotCount;
    }

    public int getComputeSlotIndex() {
        return computeSlotIndex;
    }

    public int getPrefetchDepth() {
        return config.enablePrefetch() ? config.prefetchDepth() : 0;
    }

    public boolean isPrefetchEnabled() {
        return config.enablePrefetch();
    }

    public void reset() {
        if (closed) {
            throw new IllegalStateException("RingBufferManager is closed");
        }
        computeSlotIndex = 0;
        for (BufferSlot slot : slots) {
            slot.clearPendingEvent();
            slot.markReady();
        }
    }

    public List<BufferSlot> getAllSlots() {
        List<BufferSlot> result = new ArrayList<>(slotCount);
        for (BufferSlot slot : slots) {
            result.add(slot);
        }
        return result;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        
        for (BufferSlot slot : slots) {
            slot.close();
        }
    }

    public final class BufferSlot implements AutoCloseable {
        private final int slotIndex;
        private GpuBuffer buffer;
        private MemoryBlock memoryBlock;
        private long bufferSizeBytes;
        private GpuEvent pendingEvent;
        private volatile boolean prefetchInProgress;
        private volatile boolean ready;
        private volatile boolean closed;

        private BufferSlot(int slotIndex) {
            this.slotIndex = slotIndex;
            this.buffer = null;
            this.memoryBlock = null;
            this.bufferSizeBytes = 0;
            this.pendingEvent = null;
            this.prefetchInProgress = false;
            this.ready = true;
            this.closed = false;
        }

        public int slotIndex() {
            return slotIndex;
        }

        public boolean hasBuffer() {
            return buffer != null;
        }

        public GpuBuffer getBuffer() {
            if (closed) {
                throw new IllegalStateException("BufferSlot is closed");
            }
            return buffer;
        }

        public long getBufferSizeBytes() {
            return bufferSizeBytes;
        }

        public boolean hasPendingEvent() {
            return pendingEvent != null;
        }

        public GpuEvent getPendingEvent() {
            return pendingEvent;
        }

        private void setPendingEvent(GpuEvent event) {
            this.pendingEvent = event;
        }

        private void clearPendingEvent() {
            if (pendingEvent != null) {
                try {
                    pendingEvent.close();
                } catch (Exception ignored) {
                }
                pendingEvent = null;
            }
        }

        public boolean isPrefetchInProgress() {
            return prefetchInProgress;
        }

        public boolean isReady() {
            return ready;
        }

        private void setBuffer(GpuBuffer buffer, long sizeBytes) {
            this.buffer = buffer;
            this.bufferSizeBytes = sizeBytes;
        }

        private void setMemoryBlock(MemoryBlock block) {
            this.memoryBlock = block;
        }

        private void markPrefetchInProgress() {
            this.prefetchInProgress = true;
            this.ready = false;
        }

        private void markReady() {
            this.prefetchInProgress = false;
            this.ready = true;
        }

        private void releaseBuffer() {
            if (memoryBlock != null) {
                memoryBlock = null;
            } else if (buffer != null) {
                try {
                    buffer.close();
                } catch (Exception ignored) {
                }
            }
            buffer = null;
            bufferSizeBytes = 0;
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            
            clearPendingEvent();
            releaseBuffer();
        }

        @Override
        public String toString() {
            return String.format("BufferSlot{index=%d, hasBuffer=%s, size=%d, ready=%s, prefetching=%s}",
                    slotIndex, hasBuffer(), bufferSizeBytes, ready, prefetchInProgress);
        }
    }
}
