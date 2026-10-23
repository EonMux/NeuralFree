package com.eonmux.neuralfree.acceleration.backend.vulkan;

import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Submission policy of the Vulkan queue: the dispatch threshold ramps from
 * {@value VkSubmitPolicy#FIRST_SUBMIT_DISPATCHES} to the cap and restarts when the queue drains, and a
 * flush defers only small dispatch-only batches while the device is busy.
 */
public class VkSubmitPolicyTest {

    @Test
    public void thresholdDoublesPerSubmissionUpToTheCap() {
        VkSubmitPolicy p = new VkSubmitPolicy(256);
        int[] expected = {16, 32, 64, 128, 256, 256};
        for (int e : expected) {
            assertEquals(e, p.threshold());
            assertFalse(p.submitAfterDispatch(e - 1));
            assertTrue(p.submitAfterDispatch(e));
            p.onSubmit();
        }
    }

    @Test
    public void drainingRestartsTheRamp() {
        VkSubmitPolicy p = new VkSubmitPolicy(256);
        for (int i = 0; i < 10; i++) {
            p.onSubmit();
        }
        assertEquals(256, p.threshold());
        p.onDrained();
        assertEquals(VkSubmitPolicy.FIRST_SUBMIT_DISPATCHES, p.threshold());
    }

    @Test
    public void capBelowTheFirstThresholdBoundsEverything() {
        VkSubmitPolicy p = new VkSubmitPolicy(4);
        assertEquals(4, p.threshold());
        p.onSubmit();
        assertEquals(4, p.threshold());
        assertTrue(p.submitOnFlush(4, false, () -> false));
        assertFalse(p.submitOnFlush(3, false, () -> false));
    }

    @Test
    public void largeCapDoesNotOverflow() {
        VkSubmitPolicy p = new VkSubmitPolicy(Integer.MAX_VALUE);
        for (int i = 0; i < 40; i++) {
            p.onSubmit();
        }
        assertEquals(Integer.MAX_VALUE, p.threshold());
    }

    @Test
    public void flushSubmitsTransfersWaitsAndReadbacksWithoutAskingTheDevice() {
        VkSubmitPolicy p = new VkSubmitPolicy(256);
        AtomicInteger asked = new AtomicInteger();
        assertTrue(p.submitOnFlush(0, true, () -> {
            asked.incrementAndGet();
            return false;
        }));
        assertEquals(0, asked.get());
    }

    @Test
    public void flushSubmitsALargeBatchWithoutAskingTheDevice() {
        VkSubmitPolicy p = new VkSubmitPolicy(256);
        AtomicInteger asked = new AtomicInteger();
        assertTrue(p.submitOnFlush(VkSubmitPolicy.MIN_FLUSH_DISPATCHES, false, () -> {
            asked.incrementAndGet();
            return false;
        }));
        assertEquals(0, asked.get());
    }

    @Test
    public void flushOfASmallBatchFollowsTheDevice() {
        VkSubmitPolicy p = new VkSubmitPolicy(256);
        assertTrue(p.submitOnFlush(3, false, () -> true));
        assertFalse(p.submitOnFlush(3, false, () -> false));
    }

    @Test
    public void nonPositiveCapIsRejected() {
        try {
            new VkSubmitPolicy(0);
            fail("a zero cap must be rejected");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("positive"));
        }
    }
}
