package com.eonmux.neuralfree.acceleration.memory;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.ThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

@ThreadSafe("Uses synchronized on acquire/release for thread-safe memory allocation. "
        + "CopyOnWriteArrayList for safe iteration. closed is volatile. "
        + "All public methods are either synchronized or thread-safe by design.")
public final class GpuMemoryPool implements AutoCloseable {
    private static final Logger LOGGER = LoggingUtil.getLogger(GpuMemoryPool.class);
    
    private final ComputeBackend backend;
    private final PoolConfig config;
    private final List<MemoryBlock> blocks;
    private long totalAllocatedBytes;
    private volatile boolean closed;

    public GpuMemoryPool(ComputeBackend backend, PoolConfig config) {
        this.backend = backend;
        this.config = config;
        this.blocks = new CopyOnWriteArrayList<>();
        this.totalAllocatedBytes = 0;
        this.closed = false;
    }

    public GpuMemoryPool(ComputeBackend backend) {
        this(backend, PoolConfig.defaultConfig(backend.capabilities().getTotalVramBytes()));
    }

    public synchronized MemoryBlock acquire(long sizeBytes) {
        if (closed) {
            throw new IllegalStateException("Memory pool is closed");
        }

        long alignedSize = alignSize(sizeBytes);
        MemoryBlock bestFit = findBestFit(alignedSize);

        if (bestFit != null) {
            int index = blocks.indexOf(bestFit);
            MemoryBlock inUseBlock = bestFit.withInUse(true);
            blocks.set(index, inUseBlock);
            LOGGER.debug("Memory pool reuse - size: {} bytes, bufferId: {}", alignedSize, inUseBlock.bufferId());
            return inUseBlock;
        }

        if (totalAllocatedBytes + alignedSize > config.getMaxPoolSizeBytes()) {
            boolean evicted = tryEvictLruBlocks(alignedSize);
            if (!evicted) {
                throw new GpuOutOfMemoryException(
                        String.format("Memory pool limit exceeded: requested=%d, totalAllocated=%d, maxPoolSize=%d",
                                alignedSize, totalAllocatedBytes, config.getMaxPoolSizeBytes()),
                        alignedSize,
                        config.getMaxPoolSizeBytes() - totalAllocatedBytes,
                        backend.backendType().name());
            }
        }

        try {
            GpuBuffer newBuffer = backend.allocateBuffer(alignedSize);
            MemoryBlock newBlock = new MemoryBlock(newBuffer, alignedSize, 0, true);
            blocks.add(newBlock);
            totalAllocatedBytes += alignedSize;
            LOGGER.debug("Memory pool allocate - size: {} bytes, totalAllocated: {} MB, blocks: {}", 
                    alignedSize, totalAllocatedBytes / (1024 * 1024), blocks.size());
            return newBlock;
        } catch (GpuOutOfMemoryException e) {
            boolean evicted = tryEvictLruBlocks(alignedSize);
            if (evicted) {
                GpuBuffer retryBuffer = backend.allocateBuffer(alignedSize);
                MemoryBlock retryBlock = new MemoryBlock(retryBuffer, alignedSize, 0, true);
                blocks.add(retryBlock);
                totalAllocatedBytes += alignedSize;
                LOGGER.debug("Memory pool allocate after eviction - size: {} bytes, totalAllocated: {} MB", 
                        alignedSize, totalAllocatedBytes / (1024 * 1024));
                return retryBlock;
            }
            throw e;
        }
    }

    public synchronized void release(MemoryBlock block) {
        if (closed || block == null) {
            return;
        }

        int index = blocks.indexOf(block);
        if (index >= 0) {
            MemoryBlock releasedBlock = block.withInUse(false);
            blocks.set(index, releasedBlock);
        }
    }

    private boolean tryEvictLruBlocks(long requiredBytes) {
        List<MemoryBlock> freeBlocks = new ArrayList<>();
        for (MemoryBlock block : blocks) {
            if (!block.isInUse()) {
                freeBlocks.add(block);
            }
        }

        freeBlocks.sort(Comparator.comparingLong(MemoryBlock::getLastUsedTimestamp));

        long freedBytes = 0;
        long targetBytes = Math.min(requiredBytes, config.getMaxPoolSizeBytes() - totalAllocatedBytes + requiredBytes);
        int evictedCount = 0;
        
        for (MemoryBlock block : freeBlocks) {
            if (freedBytes >= targetBytes) {
                break;
            }

            blocks.remove(block);
            if (block.getBuffer() != null) {
                try {
                    block.getBuffer().close();
                } catch (Exception ignored) {
                }
            }
            totalAllocatedBytes -= block.getSizeBytes();
            freedBytes += block.getSizeBytes();
            evictedCount++;
        }

        if (evictedCount > 0) {
            LOGGER.warn("Memory pool LRU eviction - {} blocks evicted, {} bytes freed, {} MB remaining",
                    evictedCount, freedBytes, totalAllocatedBytes / (1024 * 1024));
        }

        return freedBytes >= requiredBytes || 
               (config.getMaxPoolSizeBytes() - totalAllocatedBytes) >= requiredBytes;
    }

    private MemoryBlock findBestFit(long sizeBytes) {
        MemoryBlock bestFit = null;
        long bestFitSize = Long.MAX_VALUE;

        for (MemoryBlock block : blocks) {
            if (!block.isInUse() && block.getSizeBytes() >= sizeBytes) {
                if (block.getSizeBytes() < bestFitSize) {
                    bestFit = block;
                    bestFitSize = block.getSizeBytes();
                }
            }
        }

        return bestFit;
    }

    private long alignSize(long sizeBytes) {
        long minBlockSize = config.getMinBlockSizeBytes();
        if (sizeBytes <= minBlockSize) {
            return minBlockSize;
        }
        long remainder = sizeBytes % minBlockSize;
        if (remainder == 0) {
            return sizeBytes;
        }
        return sizeBytes + (minBlockSize - remainder);
    }

    public synchronized long getTotalAllocatedBytes() {
        return totalAllocatedBytes;
    }

    public synchronized int getBlockCount() {
        return blocks.size();
    }

    public synchronized int getAvailableBlockCount() {
        int count = 0;
        for (MemoryBlock block : blocks) {
            if (!block.isInUse()) {
                count++;
            }
        }
        return count;
    }

    public PoolConfig getConfig() {
        return config;
    }

    public synchronized int evictUnusedBlocks() {
        int evictedCount = 0;
        List<MemoryBlock> toRemove = new ArrayList<>();
        
        for (MemoryBlock block : blocks) {
            if (!block.isInUse()) {
                toRemove.add(block);
            }
        }

        for (MemoryBlock block : toRemove) {
            blocks.remove(block);
            if (block.getBuffer() != null) {
                try {
                    block.getBuffer().close();
                } catch (Exception ignored) {
                }
            }
            totalAllocatedBytes -= block.getSizeBytes();
            evictedCount++;
        }

        if (evictedCount > 0) {
            LOGGER.debug("Memory pool manual eviction - {} blocks evicted, {} MB remaining",
                    evictedCount, totalAllocatedBytes / (1024 * 1024));
        }

        return evictedCount;
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;

        List<MemoryBlock> blocksToClose = new ArrayList<>(blocks);
        blocks.clear();

        for (MemoryBlock block : blocksToClose) {
            if (block.getBuffer() != null) {
                try {
                    block.getBuffer().close();
                } catch (Exception ignored) {
                }
            }
        }
        totalAllocatedBytes = 0;
    }
}
