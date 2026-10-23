package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import org.jocl.CL;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.jocl.CL.*;
import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

/**
 * Completion of a non-blocking OpenCL read: a successful read copies, an abnormally terminated read
 * copies nothing and reports its status exactly once.
 */
public class OclReadCompletionTest {

    @BeforeClass
    public static void requireJocl() {
        boolean loaded;
        try {
            loaded = CL.stringFor_errorCode(CL_SUCCESS) != null;
        } catch (Throwable e) {
            loaded = false;
        }
        assumeTrue("JOCL native library unavailable", loaded);
    }

    @Test
    public void successfulReadCopiesAndReportsNothing() throws InterruptedException {
        OclReadCompletion read = new OclReadCompletion(64);
        AtomicInteger copies = new AtomicInteger();
        assertFalse(read.isDone());
        read.complete(CL_COMPLETE, copies::incrementAndGet);
        read.await();
        assertTrue(read.isDone());
        assertEquals(1, copies.get());
        read.throwIfFailed();
    }

    @Test
    public void abnormalTerminationSkipsTheCopyAndIsReportedOnce() throws InterruptedException {
        OclReadCompletion read = new OclReadCompletion(64);
        AtomicInteger copies = new AtomicInteger();
        read.complete(CL_INVALID_COMMAND_QUEUE, copies::incrementAndGet);
        read.await();
        assertEquals("stale staging bytes must not be delivered", 0, copies.get());
        try {
            read.throwIfFailed();
            fail("expected GpuBackendException");
        } catch (GpuBackendException e) {
            assertEquals(CL_INVALID_COMMAND_QUEUE, e.getCode());
        }
        read.throwIfFailed();
    }

    @Test
    public void outOfResourcesTerminationIsAnOutOfMemoryFailure() {
        OclReadCompletion read = new OclReadCompletion(64);
        read.complete(CL_OUT_OF_RESOURCES, () -> fail("no copy on failure"));
        try {
            read.throwIfFailed();
            fail("expected GpuOutOfMemoryException");
        } catch (GpuOutOfMemoryException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("CL_OUT_OF_RESOURCES"));
        }
    }

    @Test
    public void failureReportedThroughTheEventIsNotRepeated() {
        OclReadCompletion read = new OclReadCompletion(64);
        read.complete(CL_INVALID_EVENT, () -> fail("no copy on failure"));
        read.markReported();
        read.throwIfFailed();
    }

    @Test
    public void failedCopyIsReportedAndStillReleasesTheLatch() throws InterruptedException {
        OclReadCompletion read = new OclReadCompletion(64);
        read.complete(CL_COMPLETE, () -> {
            throw new IndexOutOfBoundsException("destination too short");
        });
        read.await();
        try {
            read.throwIfFailed();
            fail("expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getCause() instanceof IndexOutOfBoundsException);
        }
    }
}
