package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import jcuda.driver.CUdeviceptr;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Bookkeeping of the CUDA buffer pool over a fake driver: exact fresh allocations, reuse only after
 * a block's fence completes, best fit within twice the request and the absolute slack, oldest-first
 * collection, the out-of-memory release-and-retry path, trim, close freeing the blocks still handed
 * out, release validation, and the in-use and pooled byte counters. No device is needed.
 */
public class CudaBufferPoolTest {

    private FakeCudaMemoryDriver driver;
    private CudaBufferPool pool;

    @Before
    public void createPool() {
        driver = new FakeCudaMemoryDriver();
        pool = new CudaBufferPool(driver);
    }

    @Test
    public void freshAllocationHasTheRequestedSizeAndIsCounted() {
        CudaBufferPool.Allocation a = pool.acquire(1024);

        assertEquals(1024, a.bytes());
        assertEquals(1, driver.allocCalls);
        assertEquals(1024, pool.inUseBytes());
        assertEquals(0, pool.pooledBytes());
        assertEquals(1, pool.stats().freshAllocations());
        assertBalanced();
    }

    @Test
    public void releasedBlockIsReusedOnlyAfterItsFenceCompletes() {
        CudaBufferPool.Allocation a = pool.acquire(2048);
        pool.release(a.pointer(), a.bytes());
        assertEquals(0, pool.inUseBytes());
        assertEquals(2048, pool.pooledBytes());
        assertEquals(1, pool.stats().pendingBlocks());

        CudaBufferPool.Allocation whileInFlight = pool.acquire(2048);
        assertNotSame(a.pointer(), whileInFlight.pointer());
        assertEquals(0, driver.syncCalls);

        driver.lastFence().complete = true;
        CudaBufferPool.Allocation reused = pool.acquire(2048);

        assertSame(a.pointer(), reused.pointer());
        assertEquals(1, driver.lastFence().releases);
        assertEquals(2, driver.allocCalls);
        assertEquals(1, pool.stats().reusedAllocations());
        assertEquals(4096, pool.inUseBytes());
        assertEquals(0, pool.pooledBytes());
        assertEquals(0, driver.syncCalls);
        assertBalanced();
    }

    @Test
    public void bestFitTakesTheSmallestSufficientBlock() {
        CudaBufferPool.Allocation big = pool.acquire(4096);
        CudaBufferPool.Allocation fit = pool.acquire(2048);
        CudaBufferPool.Allocation mid = pool.acquire(3072);
        CudaBufferPool.Allocation small = pool.acquire(1024);
        releaseCompleted(big, fit, mid, small);

        CudaBufferPool.Allocation got = pool.acquire(1792);

        assertSame(fit.pointer(), got.pointer());
        assertEquals(2048, got.bytes());
        assertEquals(3, pool.stats().readyBlocks());
        assertBalanced();
    }

    @Test
    public void blockOfMoreThanTwiceTheRequestIsNotReused() {
        CudaBufferPool.Allocation big = pool.acquire(4096);
        releaseCompleted(big);

        CudaBufferPool.Allocation tooSmall = pool.acquire(1792);
        assertNotSame(big.pointer(), tooSmall.pointer());
        assertEquals(1792, tooSmall.bytes());

        CudaBufferPool.Allocation exactlyHalf = pool.acquire(2048);
        assertSame(big.pointer(), exactlyHalf.pointer());
        assertEquals(4096, exactlyHalf.bytes());
        assertEquals(1792 + 4096, pool.inUseBytes());
        assertBalanced();
    }

    @Test
    public void largeBlockIsReusedOnlyWithinTheAbsoluteSlack() {
        long mib = 1L << 20;
        CudaBufferPool.Allocation big = pool.acquire(64 * mib);
        releaseCompleted(big);

        CudaBufferPool.Allocation tooLoose = pool.acquire(40 * mib);
        assertNotSame(big.pointer(), tooLoose.pointer());
        assertEquals(40 * mib, tooLoose.bytes());

        CudaBufferPool.Allocation withinSlack = pool.acquire(64 * mib - CudaBufferPool.MAX_REUSE_SLACK_BYTES);
        assertSame(big.pointer(), withinSlack.pointer());
        assertEquals(64 * mib, withinSlack.bytes());
        assertEquals(1, pool.stats().reusedAllocations());
        assertBalanced();
    }

    @Test
    public void pendingBlocksAreCollectedOldestFirst() {
        CudaBufferPool.Allocation older = pool.acquire(1024);
        CudaBufferPool.Allocation newer = pool.acquire(2048);
        pool.release(older.pointer(), older.bytes());
        FakeCudaMemoryDriver.FakeFence olderFence = driver.lastFence();
        pool.release(newer.pointer(), newer.bytes());
        driver.lastFence().complete = true;

        CudaBufferPool.Allocation blocked = pool.acquire(2048);
        assertNotSame(newer.pointer(), blocked.pointer());
        assertEquals(2, pool.stats().pendingBlocks());

        olderFence.complete = true;
        CudaBufferPool.Allocation reused = pool.acquire(2048);

        assertSame(newer.pointer(), reused.pointer());
        assertEquals(0, pool.stats().pendingBlocks());
        assertEquals(1, pool.stats().readyBlocks());
        assertBalanced();
    }

    @Test
    public void outOfMemoryFreesThePoolAndRetriesOnce() {
        driver.capacity = 4096;
        CudaBufferPool.Allocation a = pool.acquire(4096);
        pool.release(a.pointer(), a.bytes());
        FakeCudaMemoryDriver.FakeFence fence = driver.lastFence();

        CudaBufferPool.Allocation b = pool.acquire(2048);

        assertEquals(2048, b.bytes());
        assertEquals(1, driver.syncCalls);
        assertEquals(1, driver.freeCalls);
        assertEquals(3, driver.allocCalls);
        assertEquals(1, fence.releases);
        assertEquals(1, pool.stats().oomRetries());
        assertEquals(0, pool.pooledBytes());
        assertEquals(2048, pool.inUseBytes());
        assertBalanced();
    }

    @Test
    public void outOfMemoryWithAnEmptyPoolFailsWithoutSynchronizing() {
        driver.capacity = 1024;
        try {
            pool.acquire(2048);
            fail("expected GpuOutOfMemoryException");
        } catch (GpuOutOfMemoryException expected) {
            assertEquals(0, expected.getSuppressed().length);
        }
        assertEquals(1, driver.allocCalls);
        assertEquals(0, driver.syncCalls);
        assertEquals(0, pool.stats().oomRetries());
        assertBalanced();
    }

    @Test
    public void outOfMemoryAfterTheRetryPropagatesBothFailures() {
        driver.capacity = 4096;
        pool.acquire(2048);
        CudaBufferPool.Allocation b = pool.acquire(1024);
        pool.release(b.pointer(), b.bytes());

        try {
            pool.acquire(4096);
            fail("expected GpuOutOfMemoryException");
        } catch (GpuOutOfMemoryException expected) {
            assertEquals(1, expected.getSuppressed().length);
            assertTrue(expected.getSuppressed()[0] instanceof GpuOutOfMemoryException);
        }
        assertEquals(4, driver.allocCalls);
        assertEquals(1, driver.syncCalls);
        assertEquals(1, driver.freeCalls);
        assertEquals(2048, pool.inUseBytes());
        assertEquals(0, pool.pooledBytes());
        assertBalanced();
    }

    @Test
    public void trimFreesEveryBlockAndSynchronizesOnlyForBlocksInFlight() {
        CudaBufferPool.Allocation idle = pool.acquire(1024);
        releaseCompleted(idle);

        assertEquals(1024, pool.trim());
        assertEquals(0, driver.syncCalls);
        assertEquals(1, driver.freeCalls);

        CudaBufferPool.Allocation busy = pool.acquire(2048);
        pool.release(busy.pointer(), busy.bytes());

        assertEquals(2048, pool.trim());
        assertEquals(1, driver.syncCalls);
        assertEquals(2, driver.freeCalls);
        assertEquals(1, driver.lastFence().releases);
        assertEquals(0, pool.pooledBytes());
        assertEquals(0, driver.liveBytes());
        assertBalanced();
    }

    @Test
    public void closeFreesPooledAndHandedOutBlocksAndIgnoresLateReleases() {
        CudaBufferPool.Allocation released = pool.acquire(1024);
        CudaBufferPool.Allocation open = pool.acquire(2048);
        pool.release(released.pointer(), released.bytes());
        FakeCudaMemoryDriver.FakeFence fence = driver.lastFence();

        pool.close();
        pool.close();

        assertEquals(2, driver.freeCalls);
        assertEquals(1, driver.syncCalls);
        assertEquals(1, fence.releases);
        assertEquals(0, driver.liveBytes());
        assertEquals(0, pool.inUseBytes());
        assertEquals(0, pool.pooledBytes());
        try {
            pool.acquire(256);
            fail("expected IllegalStateException");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("closed"));
        }
        pool.release(open.pointer(), open.bytes());

        assertEquals(2, driver.freeCalls);
        assertSame(fence, driver.lastFence());
        assertEquals(0, driver.liveBytes());
        assertEquals(0, pool.trim());
    }

    @Test
    public void closeSynchronizesForHandedOutBlocksEvenWithoutPooledOnes() {
        pool.acquire(4096);

        pool.close();

        assertEquals(1, driver.syncCalls);
        assertEquals(1, driver.freeCalls);
        assertEquals(0, driver.liveBytes());
        assertEquals(0, pool.inUseBytes());
    }

    @Test
    public void closeWithOnlyIdleBlocksDoesNotSynchronize() {
        CudaBufferPool.Allocation idle = pool.acquire(1024);
        releaseCompleted(idle);

        pool.close();

        assertEquals(0, driver.syncCalls);
        assertEquals(1, driver.freeCalls);
        assertEquals(0, driver.liveBytes());
    }

    @Test
    public void reusedBlockIsFreedByCloseWhileHandedOut() {
        CudaBufferPool.Allocation first = pool.acquire(1024);
        releaseCompleted(first);
        CudaBufferPool.Allocation reused = pool.acquire(1024);
        assertSame(first.pointer(), reused.pointer());

        pool.close();

        assertEquals(1, driver.freeCalls);
        assertEquals(0, driver.liveBytes());
        assertEquals(0, pool.inUseBytes());
    }

    @Test
    public void releaseOfABlockNotHandedOutIsRejected() {
        CudaBufferPool.Allocation a = pool.acquire(1024);
        pool.release(a.pointer(), a.bytes());
        FakeCudaMemoryDriver.FakeFence fence = driver.lastFence();

        try {
            pool.release(a.pointer(), a.bytes());
            fail("expected IllegalArgumentException for a second release");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("already released"));
        }
        try {
            pool.release(new CUdeviceptr(), 1024);
            fail("expected IllegalArgumentException for a foreign pointer");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("not handed out"));
        }
        assertSame(fence, driver.lastFence());
        assertEquals(1, pool.stats().pendingBlocks());
        assertBalanced();
    }

    @Test
    public void releaseWithAnotherSizeIsRejected() {
        CudaBufferPool.Allocation a = pool.acquire(1024);

        try {
            pool.release(a.pointer(), 2048);
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("1024"));
        }
        assertEquals(1024, pool.inUseBytes());
        assertEquals(0, pool.stats().pendingBlocks());
        assertBalanced();
    }

    @Test
    public void fenceFailureFreesTheBlockAtOnce() {
        CudaBufferPool.Allocation a = pool.acquire(1024);
        driver.failFence = true;

        try {
            pool.release(a.pointer(), a.bytes());
            fail("expected the fence failure");
        } catch (IllegalStateException expected) {
            assertEquals(FakeCudaMemoryDriver.FENCE_FAILURE, expected.getMessage());
        }
        assertEquals(1, driver.freeCalls);
        assertEquals(0, pool.inUseBytes());
        assertEquals(0, pool.pooledBytes());
        assertBalanced();
    }

    @Test
    public void countersStayBalancedAcrossReuseCycles() {
        List<CudaBufferPool.Allocation> live = new ArrayList<>();
        for (int step = 0; step < 4; step++) {
            for (long size : new long[]{256, 512, 1024, 4096}) {
                live.add(pool.acquire(size));
            }
            for (CudaBufferPool.Allocation a : live) {
                pool.release(a.pointer(), a.bytes());
            }
            live.clear();
            driver.completeAll();
            assertBalanced();
        }
        CudaBufferPool.Stats stats = pool.stats();
        assertEquals(4, stats.freshAllocations());
        assertEquals(12, stats.reusedAllocations());
        assertEquals(0, stats.inUseBytes());
        assertEquals(256 + 512 + 1024 + 4096, stats.pooledBytes());
    }

    @Test(expected = IllegalArgumentException.class)
    public void nonPositiveRequestIsRejected() {
        pool.acquire(0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void nullReleaseIsRejected() {
        pool.release(null, 256);
    }

    private void releaseCompleted(CudaBufferPool.Allocation... allocations) {
        for (CudaBufferPool.Allocation a : allocations) {
            pool.release(a.pointer(), a.bytes());
            driver.lastFence().complete = true;
        }
    }

    /** Every byte the fake device holds is either in use or pooled. */
    private void assertBalanced() {
        assertEquals(driver.liveBytes(), pool.inUseBytes() + pool.pooledBytes());
    }
}
