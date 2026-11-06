package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Command-buffer capture and replay on a Vulkan device: a capture of two dispatches records into a
 * simultaneous-use command buffer without executing, a replay verifies the body's dispatches against
 * the capture and resubmits the buffer once, an argument change refuses the replay before any device
 * work, and a transfer inside the capture submits the recorded dispatches and ends it.
 */
public class VkGraphCaptureTest {

    private static final int SIZE = 256;
    private static final int LOCAL = 64;

    private VkComputeBackend backend;
    private ComputeQueue queue;
    private GpuBuffer values;
    private GpuBuffer scratch;
    private KernelHandle increment;

    @Before
    public void setUp() {
        try {
            backend = new VkComputeBackend();
        } catch (Throwable e) {
            Assume.assumeTrue("no Vulkan GPU: " + e.getMessage(), false);
            return;
        }
        queue = backend.computeQueue();
        assertTrue(queue.supportsGraphCapture());
        values = backend.allocateBuffer((long) SIZE * Float.BYTES);
        scratch = backend.allocateBuffer((long) SIZE * Float.BYTES);
        increment = backend.compileKernel("increment", "shaders/test/increment.spv", null);
        float[] data = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            data[i] = i;
        }
        queue.enqueueWriteBuffer(values, 0, (long) SIZE * Float.BYTES, data);
        queue.finish();
    }

    @After
    public void tearDown() {
        if (scratch != null) {
            scratch.close();
        }
        if (values != null) {
            values.close();
        }
        if (increment != null) {
            increment.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    /** {@code increment(buffer, n)}: element i becomes i + 1, in place. */
    private void dispatch(GpuBuffer buffer, int n) {
        queue.enqueueKernelNoEvent(increment, new long[]{roundUp(n)}, new long[]{LOCAL},
                new Object[]{buffer, n});
    }

    private static long roundUp(int n) {
        return (long) (n + LOCAL - 1) / LOCAL * LOCAL;
    }

    private float[] readValues() {
        float[] out = new float[SIZE];
        queue.enqueueReadBuffer(values, 0, (long) SIZE * Float.BYTES, out);
        queue.finish();
        return out;
    }

    private GpuGraph captureTwoDispatches() {
        queue.beginCapture();
        dispatch(values, SIZE);
        dispatch(values, SIZE);
        GpuGraph graph = queue.endCapture();
        assertNotNull(graph);
        assertEquals(2, graph.dispatchCount());
        return graph;
    }

    private void replayTwoDispatches(GpuGraph graph) {
        graph.replay(() -> {
            dispatch(values, SIZE);
            dispatch(values, SIZE);
        });
    }

    @Test
    public void captureRecordsAndReplayResubmits() {
        GpuGraph graph = captureTwoDispatches();
        assertEquals(0.0f, readValues()[0], 0.0f);

        replayTwoDispatches(graph);
        assertArrayEquals(iota(2.0f), readValues(), 1e-4f);

        replayTwoDispatches(graph);
        assertArrayEquals(iota(4.0f), readValues(), 1e-4f);
    }

    @Test
    public void theCapturingStepLaunchesItsCaptureOnce() {
        GpuGraph graph = captureTwoDispatches();
        graph.launch(queue);
        assertArrayEquals(iota(2.0f), readValues(), 1e-4f);
    }

    private static float[] iota(float addend) {
        float[] out = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            out[i] = i + addend;
        }
        return out;
    }

    @Test
    public void replayRefusesAChangedArgumentBeforeExecutingAnything() {
        GpuGraph graph = captureTwoDispatches();
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatch(values, SIZE);
            dispatch(values, SIZE / 2);
        }));
        assertEquals(0.0f, readValues()[0], 0.0f);
    }

    @Test
    public void replayRefusesAChangedWorkSizeAndExtraDispatches() {
        GpuGraph graph = captureTwoDispatches();
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() ->
                queue.enqueueKernelNoEvent(increment, new long[]{roundUp(SIZE / 2)}, new long[]{LOCAL},
                        new Object[]{values, SIZE / 2})));
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatch(values, SIZE);
            dispatch(values, SIZE);
            dispatch(values, SIZE);
        }));
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> dispatch(values, SIZE)));
        assertEquals(0.0f, readValues()[0], 0.0f);
    }

    @Test
    public void transferDuringCaptureIsRefusedAndTheCaptureAborts() {
        queue.beginCapture();
        dispatch(values, SIZE);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> queue.enqueueFillBuffer(scratch, 0, 4, 0x2A));
        assertTrue(e.getMessage(), e.getMessage().contains("cannot be captured"));
        assertTrue(queue.isCapturing());
        queue.abortCapture();
        assertFalse(queue.isCapturing());
        // neither the refused fill nor the aborted capture reached the device
        assertArrayEquals(iota(0.0f), readValues(), 1e-4f);
    }

    @Test
    public void transferDuringReplayRefusesIt() {
        GpuGraph graph = captureTwoDispatches();
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatch(values, SIZE);
            queue.enqueueFillBuffer(values, 0, Float.BYTES, 0);
            dispatch(values, SIZE);
        }));
        assertEquals(0.0f, readValues()[0], 0.0f);
    }

    @Test
    public void emptyCaptureReturnsNull() {
        queue.beginCapture();
        assertNull(queue.endCapture());
        queue.beginCapture();
        queue.abortCapture();
        assertThrows(IllegalStateException.class, queue::endCapture);
    }

    @Test
    public void aFailedStepAbortsItsCaptureWithoutDeviceWork() {
        queue.beginCapture();
        dispatch(values, SIZE);
        queue.abortCapture();
        assertArrayEquals(iota(0.0f), readValues(), 1e-4f);
        assertThrows(IllegalStateException.class, queue::endCapture);
    }

    @Test
    public void closedGraphRefusesReplayAndCaptureCallsFailOnAClosedBackend() {
        GpuGraph graph = captureTwoDispatches();
        graph.close();
        assertThrows(IllegalStateException.class, () -> graph.launch(queue));
        assertThrows(IllegalStateException.class, () -> graph.replay(() -> dispatch(values, SIZE)));

        backend.close();
        assertThrows(IllegalStateException.class, () -> queue.beginCapture());
        backend = null;
    }
}
