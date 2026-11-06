package com.eonmux.neuralfree.acceleration.backend;

import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The queue contract every backend that lacks graph capture serves: capture support is reported, and
 * the capture methods are refused with the queue named.
 */
public class GraphCaptureContractTest {

    private FakeComputeBackend backend;

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    private ComputeQueue openQueueWithoutCapture() {
        backend = FakeComputeBackend.builder().build();
        return backend.computeQueue();
    }

    @Test
    public void captureSupportIsReportedByTheQueue() {
        assertFalse(openQueueWithoutCapture().supportsGraphCapture());
        backend.close();
        backend = FakeComputeBackend.builder().graphCapture(true).build();
        assertTrue(backend.computeQueue().supportsGraphCapture());
    }

    @Test
    public void beginCaptureNamesTheQueueWithoutSupport() {
        ComputeQueue queue = openQueueWithoutCapture();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                queue::beginCapture);
        assertTrue(e.getMessage(), e.getMessage().contains("FakeQueue"));
        assertTrue(e.getMessage(), e.getMessage().contains("graph capture"));
    }

    @Test
    public void endCaptureNamesTheQueueWithoutSupport() {
        ComputeQueue queue = openQueueWithoutCapture();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                queue::endCapture);
        assertTrue(e.getMessage(), e.getMessage().contains("FakeQueue"));
    }

    @Test
    public void abortCaptureNamesTheQueueWithoutSupport() {
        ComputeQueue queue = openQueueWithoutCapture();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                queue::abortCapture);
        assertTrue(e.getMessage(), e.getMessage().contains("FakeQueue"));
    }
}
