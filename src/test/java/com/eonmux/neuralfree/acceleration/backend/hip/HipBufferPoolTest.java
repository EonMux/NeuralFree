package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The stream-ordered caching of {@link HipBufferPool} against a fake device: fenced release and
 * reuse, the twice-size reuse bound, and the out-of-memory retry that frees the pool.
 */
public class HipBufferPoolTest {

    @Test
    public void freshAllocationsAreTrackedInUse() {
        FakeDriver driver = new FakeDriver();
        HipBufferPool pool = new HipBufferPool(driver);
        HipBufferPool.Allocation a = pool.acquire(1024);
        assertEquals(1024, pool.inUseBytes());
        assertEquals(0, pool.pooledBytes());
        assertTrue(driver.live.containsKey(a.pointer()));
    }

    @Test
    public void releasePoolsAndAcquireReusesWithinTheTwiceBound() {
        FakeDriver driver = new FakeDriver();
        HipBufferPool pool = new HipBufferPool(driver);
        HipBufferPool.Allocation a = pool.acquire(1024);
        pool.release(a.pointer(), a.bytes());
        driver.completeAll();
        assertEquals(1024, pool.pooledBytes());
        assertEquals(0, pool.inUseBytes());

        HipBufferPool.Allocation b = pool.acquire(2048);
        // The 1024-byte block is below the request, so the allocation is fresh and the block stays pooled.
        assertEquals(2, driver.allocations);
        assertEquals(1024, pool.pooledBytes());
        pool.release(b.pointer(), b.bytes());
        driver.completeAll();
        assertEquals(3072, pool.pooledBytes());

        HipBufferPool.Allocation c = pool.acquire(1500);
        // At most twice 1500, so the 2048-byte block is reused.
        assertEquals(2, driver.allocations);
        assertEquals(2048, c.bytes());
        assertEquals(b.pointer(), c.pointer());
    }

    @Test
    public void releasesBeyondTheSlackAreNotReused() {
        FakeDriver driver = new FakeDriver();
        HipBufferPool pool = new HipBufferPool(driver);
        HipBufferPool.Allocation a = pool.acquire(1024);
        pool.release(a.pointer(), a.bytes());
        driver.completeAll();
        HipBufferPool.Allocation huge = pool.acquire(1024 * 1024);
        assertTrue("a fresh block is allocated when the pooled block is too small", huge.pointer() != a.pointer());
    }

    @Test
    public void unfencedBlocksAreNotReused() {
        FakeDriver driver = new FakeDriver();
        HipBufferPool pool = new HipBufferPool(driver);
        HipBufferPool.Allocation a = pool.acquire(1024);
        pool.release(a.pointer(), a.bytes());
        // The fence is still in flight, so the block stays pending.
        assertEquals(0, pool.stats().readyBlocks());
        assertEquals(1, pool.stats().pendingBlocks());
        HipBufferPool.Allocation b = pool.acquire(1024);
        assertTrue("an unfenced block is never handed out again", b.pointer() != a.pointer());
    }

    @Test
    public void outOfMemoryFreesThePoolAndRetriesOnce() {
        FakeDriver driver = new FakeDriver();
        driver.capacity = 4096;
        HipBufferPool pool = new HipBufferPool(driver);
        HipBufferPool.Allocation a = pool.acquire(2048);
        pool.release(a.pointer(), a.bytes());
        // The release fence is still in flight, so the 2048 bytes are held by the pool but the
        // fresh 4096-byte allocation does not fit next to them: the retry synchronizes (which
        // completes the fence), frees the pool and succeeds.
        HipBufferPool.Allocation big = pool.acquire(4096);
        assertEquals(4096, big.bytes());
        assertEquals(1, pool.stats().oomRetries());
        assertEquals(0, pool.pooledBytes());
        assertEquals(4096, pool.inUseBytes());
    }

    @Test
    public void outOfMemoryWithoutPooledBytesPropagates() {
        FakeDriver driver = new FakeDriver();
        driver.capacity = 1024;
        HipBufferPool pool = new HipBufferPool(driver);
        try {
            pool.acquire(4096);
            fail("out of memory without pooled blocks must throw");
        } catch (GpuOutOfMemoryException expected) {
            assertEquals(0, pool.stats().oomRetries());
        }
    }

    @Test
    public void trimFreesEveryPooledBlock() {
        FakeDriver driver = new FakeDriver();
        HipBufferPool pool = new HipBufferPool(driver);
        HipBufferPool.Allocation a = pool.acquire(512);
        HipBufferPool.Allocation b = pool.acquire(512);
        pool.release(a.pointer(), a.bytes());
        pool.release(b.pointer(), b.bytes());
        driver.completeAll();
        long freed = pool.trim();
        assertEquals(1024, freed);
        assertEquals(0, pool.pooledBytes());
        assertEquals(0, driver.live.size());
    }

    @Test
    public void closeFreesTheBlocksOfOpenBuffersToo() {
        FakeDriver driver = new FakeDriver();
        HipBufferPool pool = new HipBufferPool(driver);
        pool.acquire(256);
        assertEquals(256, pool.inUseBytes());
        pool.close();
        assertEquals(0, pool.inUseBytes());
        assertEquals(0, driver.live.size());
        try {
            pool.acquire(256);
            fail("a closed pool refuses allocations");
        } catch (IllegalStateException expected) {
            // expected
        }
    }

    @Test
    public void releasingAnUnknownBlockIsRejected() {
        FakeDriver driver = new FakeDriver();
        HipBufferPool pool = new HipBufferPool(driver);
        try {
            pool.release(0x1234, 128);
            fail("a block the pool never handed out must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    /** A fake device of fixed capacity: fences complete when the test says so. */
    static final class FakeDriver implements HipMemoryDriver {
        final Map<Long, Long> live = new HashMap<>();
        final List<FakeFence> fences = new ArrayList<>();
        long capacity = Long.MAX_VALUE;
        int allocations;

        @Override
        public long allocate(long bytes) {
            allocations++;
            long used = 0;
            for (long bytes_ : live.values()) {
                used += bytes_;
            }
            if (bytes > capacity - used) {
                throw new GpuOutOfMemoryException("fake device out of memory", bytes, "HIP");
            }
            long pointer = 0x1000L + allocations;
            live.put(pointer, bytes);
            return pointer;
        }

        @Override
        public void free(long pointer) {
            if (live.remove(pointer) == null) {
                throw new IllegalStateException("free of an allocation the fake device does not hold");
            }
        }

        @Override
        public Fence recordFence() {
            FakeFence fence = new FakeFence();
            fences.add(fence);
            return fence;
        }

        @Override
        public void synchronize() {
            completeAll();
        }

        void completeAll() {
            for (FakeFence fence : fences) {
                fence.complete = true;
            }
        }
    }

    static final class FakeFence implements HipMemoryDriver.Fence {
        boolean complete;
        int releases;

        @Override
        public boolean isComplete() {
            return complete;
        }

        @Override
        public void release() {
            releases++;
            if (releases > 1) {
                throw new IllegalStateException("fence released twice");
            }
        }
    }
}
