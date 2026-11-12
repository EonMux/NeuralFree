package com.eonmux.neuralfree.acceleration.backend.ocl;

import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.*;

/**
 * Bounds and reuse policy of the OpenCL staging pool: buffers are aligned, native-order and
 * direct; buffers retained up to a ticket are reused before the cycle ends; large buffers are pooled
 * only while reused and, besides the largest, within their own byte bound, so a transfer above that
 * bound keeps its buffer; small free buffers stay within their count and byte budget, keeping the
 * smallest however many are trimmed at once.
 */
public class OclStagingPoolTest {

    @Test
    public void acquireRoundsUpToAlignmentInNativeOrder() {
        OclStagingPool pool = new OclStagingPool();
        ByteBuffer buf = pool.acquire(5);
        assertTrue(buf.isDirect());
        assertEquals(ByteOrder.nativeOrder(), buf.order());
        assertEquals(OclStagingPool.ALIGNMENT, buf.capacity());
        assertEquals(0, buf.position());
        assertEquals(0, pool.acquire(0).capacity());
    }

    @Test(expected = IllegalArgumentException.class)
    public void negativeSizeIsRejected() {
        new OclStagingPool().acquire(-1);
    }

    @Test
    public void inFlightBuffersAreReusedOnlyAfterRecycle() {
        OclStagingPool pool = new OclStagingPool();
        ByteBuffer first = pool.acquire(1024);
        pool.retain(first);
        assertNotSame(first, pool.acquire(1024));
        assertEquals(1, pool.inFlightCount());

        pool.recycleInFlight();
        assertEquals(0, pool.inFlightCount());
        assertEquals(1, pool.freeCount());
        assertEquals(1024, pool.freeBytes());
        first.position(17);
        ByteBuffer again = pool.acquire(1000);
        assertSame(first, again);
        assertEquals("a reused buffer is cleared", 0, again.position());
        assertEquals(0, pool.freeBytes());
    }

    @Test
    public void buffersRetainedUpToATicketAreReusedBeforeTheCycleEnds() {
        OclStagingPool pool = new OclStagingPool();
        ByteBuffer first = pool.acquire(64);
        long firstTicket = pool.retain(first);
        ByteBuffer second = pool.acquire(64);
        long secondTicket = pool.retain(second);
        ByteBuffer third = pool.acquire(64);
        pool.retain(third);
        assertTrue("tickets grow in retention order", secondTicket > firstTicket);

        pool.recycleThrough(secondTicket);
        assertEquals("the buffer retained after the ticket stays in flight", 1, pool.inFlightCount());
        assertEquals(2, pool.freeCount());
        ByteBuffer reused = pool.acquire(64);
        assertTrue(reused == first || reused == second);
        assertNotSame(third, pool.acquire(64));

        pool.recycleThrough(secondTicket);
        assertEquals("an older ticket frees nothing more", 1, pool.inFlightCount());
        pool.recycleInFlight();
        assertEquals(0, pool.inFlightCount());
    }

    @Test
    public void largeBuffersFreedByATicketAgeOnlyAtTheCycleEnd() {
        OclStagingPool pool = new OclStagingPool();
        int chunk = (int) OclStagingPool.MAX_SMALL_BUFFER_BYTES * 2;
        ByteBuffer large = pool.acquire(chunk);
        pool.recycleThrough(pool.retain(large));
        assertEquals("a ticket keeps the cycle open, so the buffer stays pooled", 1, pool.freeCount());
        assertSame(large, pool.acquire(chunk));
    }

    @Test
    public void trimmingManyBuffersAtOnceKeepsTheSmallest() {
        OclStagingPool pool = new OclStagingPool(4, 1 << 20, 1 << 20);
        int buffers = 20_000;
        for (int i = 0; i < buffers; i++) {
            pool.retain(pool.acquire(OclStagingPool.ALIGNMENT * (1 + (i * 7919) % 64)));
        }
        pool.recycleInFlight();
        assertEquals(4, pool.freeCount());
        assertEquals("the four smallest buffers stay", 4L * OclStagingPool.ALIGNMENT, pool.freeBytes());
    }

    @Test
    public void smallestFittingBufferIsChosen() {
        OclStagingPool pool = new OclStagingPool();
        ByteBuffer large = pool.acquire(64 * 1024);
        ByteBuffer small = pool.acquire(4 * 1024);
        pool.retain(large);
        pool.retain(small);
        pool.recycleInFlight();
        assertSame(small, pool.acquire(2 * 1024));
        assertSame(large, pool.acquire(2 * 1024));
    }

    @Test
    public void largeBuffersArePooledOnlyWhileReused() {
        OclStagingPool pool = new OclStagingPool();
        int chunk = (int) OclStagingPool.MAX_SMALL_BUFFER_BYTES * 2;

        ByteBuffer first = pool.acquire(chunk);
        ByteBuffer second = pool.acquire(chunk);
        pool.retain(first);
        pool.retain(second);
        pool.recycleInFlight();
        assertEquals(2, pool.freeCount());

        ByteBuffer reused = pool.acquire(chunk);
        assertTrue(reused == first || reused == second);
        pool.retain(reused);
        pool.recycleInFlight();
        assertEquals("the chunk buffer idle during the last cycle is dropped", 1, pool.freeCount());
        assertEquals(chunk, pool.freeBytes());

        pool.recycleInFlight();
        assertEquals("upload buffers become collectable once the upload stops reusing them", 0, pool.freeCount());
        assertEquals(0, pool.freeBytes());
    }

    @Test
    public void smallRequestsAreNotServedFromLargeBuffers() {
        OclStagingPool pool = new OclStagingPool();
        ByteBuffer large = pool.acquire((int) OclStagingPool.MAX_SMALL_BUFFER_BYTES + 1);
        pool.retain(large);
        pool.recycleInFlight();
        ByteBuffer small = pool.acquire(4);
        assertNotSame(large, small);
        assertEquals(OclStagingPool.ALIGNMENT, small.capacity());
        assertEquals(1, pool.freeCount());
    }

    @Test
    public void byteBudgetEvictsLargestBuffersFirst() {
        OclStagingPool pool = new OclStagingPool(32, 4096, 6144);
        ByteBuffer a = pool.acquire(4096);
        ByteBuffer b = pool.acquire(2048);
        ByteBuffer c = pool.acquire(1024);
        pool.retain(a);
        pool.retain(b);
        pool.retain(c);
        pool.recycleInFlight();
        assertEquals(2, pool.freeCount());
        assertEquals(3072, pool.freeBytes());
        assertSame(c, pool.acquire(1024));
        assertSame(b, pool.acquire(1024));
    }

    @Test
    public void countBoundIsKept() {
        OclStagingPool pool = new OclStagingPool(2, 1 << 20, 1 << 20);
        for (int i = 0; i < 5; i++) {
            pool.retain(pool.acquire(256));
        }
        pool.recycleInFlight();
        assertEquals(2, pool.freeCount());
        assertEquals(512, pool.freeBytes());
    }

    /**
     * Large buffers recycled before the cycle ends that no later request fits (a run of growing uploads) do not
     * accumulate: the free large buffers besides the largest stay within their byte bound, the longest-free dropped
     * first.
     */
    @Test
    public void largeFreeBuffersStayWithinTheirByteBound() {
        OclStagingPool pool = new OclStagingPool(32, 1024, 1 << 20, 4096);
        ByteBuffer a = pool.acquire(2048);
        pool.retain(a);
        ByteBuffer b = pool.acquire(2304);
        pool.retain(b);
        ByteBuffer c = pool.acquire(2560);
        pool.retain(c);
        ByteBuffer d = pool.acquire(2816);
        long last = pool.retain(d);
        pool.recycleThrough(last);
        assertEquals(2, pool.freeCount());
        assertEquals(2560 + 2816, pool.freeBytes());
        assertSame(c, pool.acquire(2048));
        assertSame(d, pool.acquire(2048));
    }

    /**
     * A transfer above the large-buffer budget repeated every cycle, as the read of a gradient and the write of the
     * stepped weights of one tensor are, reuses one buffer: freed mid-cycle together with a smaller transfer's buffer
     * (a read that completed), and at every finish.
     */
    @Test
    public void repeatedTransferAboveTheLargeBudgetReusesItsBufferAcrossFinishes() {
        OclStagingPool pool = new OclStagingPool();
        int big = (int) OclStagingPool.MAX_LARGE_FREE_BYTES + (32 << 20);
        int chunk = 12 << 20;
        ByteBuffer first = pool.acquire(big);
        pool.retain(first);
        pool.recycleInFlight();
        for (int cycle = 0; cycle < 3; cycle++) {
            assertSame("cycle " + cycle + ": the read reuses the buffer", first, pool.acquire(big));
            pool.retain(first);
            pool.recycleThrough(pool.retain(pool.acquire(chunk)));
            assertEquals("cycle " + cycle + ": both buffers freed mid-cycle stay",
                    (long) big + chunk, pool.freeBytes());
            assertSame("cycle " + cycle + ": the write reuses the buffer", first, pool.acquire(big));
            pool.retain(first);
            pool.retain(pool.acquire(chunk));
            pool.recycleInFlight();
            assertEquals("cycle " + cycle + ": the finish keeps both buffers", 2, pool.freeCount());
            assertEquals((long) big + chunk, pool.freeBytes());
        }
        pool.recycleInFlight();
        assertEquals("the idle-cycle rule still drops them once the transfers stop", 0, pool.freeCount());
    }

    @Test
    public void aLargeWriteFreedBeforeTheFinishIsReusedInTheNextCycle() {
        OclStagingPool pool = new OclStagingPool();
        int big = (int) OclStagingPool.MAX_LARGE_FREE_BYTES + (32 << 20);
        int small = 4096;
        ByteBuffer first = pool.acquire(big);
        for (int cycle = 0; cycle < 3; cycle++) {
            if (cycle > 0) {
                assertSame("cycle " + cycle + ": the write reuses the buffer", first, pool.acquire(big));
            }
            long bigTicket = pool.retain(first);
            ByteBuffer tail = pool.acquire(small);
            pool.recycleThrough(bigTicket);
            pool.retain(tail);
            pool.recycleInFlight();
            assertEquals("cycle " + cycle + ": the finish keeps the write's buffer",
                    (long) big + tail.capacity(), pool.freeBytes());
        }
        pool.recycleInFlight();
        assertEquals("an idle cycle drops the large buffer", 1, pool.freeCount());
    }

    @Test
    public void capacityRoundsUpToTheAlignment() {
        assertEquals(0, OclStagingPool.capacityFor(0));
        assertEquals(OclStagingPool.ALIGNMENT, OclStagingPool.capacityFor(1));
        assertEquals(OclStagingPool.ALIGNMENT * 2, OclStagingPool.capacityFor(OclStagingPool.ALIGNMENT + 1));
        assertEquals(Integer.MAX_VALUE, OclStagingPool.capacityFor(Integer.MAX_VALUE));
    }

    @Test
    public void clearDropsEverything() {
        OclStagingPool pool = new OclStagingPool();
        pool.retain(pool.acquire(256));
        pool.release(pool.acquire(512));
        pool.clear();
        assertEquals(0, pool.inFlightCount());
        assertEquals(0, pool.freeCount());
        assertEquals(0, pool.freeBytes());
    }
}
