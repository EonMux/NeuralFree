package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.hip.FakeHipDriver;
import com.eonmux.neuralfree.acceleration.hip.HipDevice;
import org.junit.After;
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
 * HIP graph capture and replay over the simulated device of {@link FakeHipDriver}: a capture of a
 * kernel records without executing, a replay updates the node parameters with the body's values and
 * launches the executable graph once, a changed scalar argument and grid are picked up by the node
 * update, a mismatching body refuses before any device work, and copies and fills are recorded and
 * replayed too.
 */
public class HipGraphCaptureTest {

    private static final int SIZE = 8;
    private static final int LOCAL = 64;

    private FakeHipDriver driver;
    private HipComputeBackend backend;
    private ComputeQueue queue;
    private GpuBuffer data;
    private GpuBuffer other;
    private KernelHandle scale;

    @Before
    public void setUp() {
        driver = new FakeHipDriver();
        backend = new HipComputeBackend(new HipDevice(0, driver));
        queue = backend.computeQueue();
        assertTrue(queue.supportsGraphCapture());
        data = backend.allocateBuffer((long) SIZE * Float.BYTES);
        other = backend.allocateBuffer((long) SIZE * Float.BYTES);
        scale = backend.loadKernel(scaleArtifact());
        float[] initial = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            initial[i] = i + 1;
        }
        queue.enqueueWriteBuffer(data, 0, (long) SIZE * Float.BYTES, initial);
        queue.finish();
        driver.launches();
    }

    @After
    public void tearDown() {
        try {
            if (scale != null) {
                scale.close();
            }
            closeQuietly(other, data);
        } finally {
            if (backend != null) {
                backend.close();
            }
        }
    }

    private static void closeQuietly(AutoCloseable... closeables) {
        for (AutoCloseable closeable : closeables) {
            try {
                closeable.close();
            } catch (Exception ignored) {
            }
        }
    }

    private float[] read(GpuBuffer buffer) {
        float[] out = new float[SIZE];
        queue.enqueueReadBuffer(buffer, 0, (long) SIZE * Float.BYTES, out);
        queue.finish();
        return out;
    }

    private void dispatchScale(GpuBuffer buffer, float factor) {
        queue.enqueueKernelNoEvent(scale, new long[]{SIZE}, new long[]{LOCAL},
                new Object[]{buffer, factor, SIZE});
    }

    private GpuGraph captureScale(GpuBuffer buffer, float factor) {
        queue.beginCapture();
        dispatchScale(buffer, factor);
        GpuGraph graph = queue.endCapture();
        assertNotNull(graph);
        assertEquals(1, graph.dispatchCount());
        return graph;
    }

    @Test
    public void captureRecordsAndReplayLaunchesOnce() {
        GpuGraph graph = captureScale(data, 2f);
        // the capture executed nothing
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, read(data), 0.0f);
        assertTrue(driver.launches().isEmpty());

        graph.replay(() -> dispatchScale(data, 2f));

        assertArrayEquals(new float[]{2, 4, 6, 8, 10, 12, 14, 16}, read(data), 0.0f);
        assertEquals(1, driver.launches().size());
        graph.close();
    }

    @Test
    public void theCapturingStepLaunchesItsCaptureOnce() {
        GpuGraph graph = captureScale(data, 2f);
        graph.launch(queue);
        queue.finish();

        assertArrayEquals(new float[]{2, 4, 6, 8, 10, 12, 14, 16}, read(data), 0.0f);
        assertEquals(1, driver.launches().size());
        graph.close();
    }

    @Test
    public void replayPicksUpAChangedScalarArgument() {
        GpuGraph graph = captureScale(data, 2f);
        graph.replay(() -> dispatchScale(data, 3f));
        assertArrayEquals(new float[]{3, 6, 9, 12, 15, 18, 21, 24}, read(data), 0.0f);
        graph.close();
    }

    @Test
    public void replayPicksUpAChangedGrid() {
        int half = SIZE / 2;
        queue.beginCapture();
        queue.enqueueKernelNoEvent(scale, new long[]{SIZE}, new long[]{1}, new Object[]{data, 2f, SIZE});
        GpuGraph graph = queue.endCapture();
        // the replay launches four threads instead of eight: the node's grid is updated
        graph.replay(() -> queue.enqueueKernelNoEvent(scale, new long[]{half}, new long[]{1},
                new Object[]{data, 2f, half}));
        float[] out = read(data);
        assertEquals(2f, out[0], 0.0f);
        assertEquals(half * 2f, out[half - 1], 0.0f);
        assertEquals(half + 1, out[half], 0.0f);
        graph.close();
    }

    @Test
    public void aChangedBlockSizeRefusesTheReplay() {
        GpuGraph graph = captureScale(data, 2f);
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(
                () -> queue.enqueueKernelNoEvent(scale, new long[]{SIZE}, new long[]{LOCAL / 2},
                        new Object[]{data, 2f, SIZE})));
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, read(data), 0.0f);
        graph.close();
    }

    @Test
    public void repeatedReplaysKeepWorking() {
        GpuGraph graph = captureScale(data, 2f);
        for (int round = 0; round < 3; round++) {
            graph.replay(() -> dispatchScale(data, 2f));
        }
        float[] out = read(data);
        for (int i = 0; i < SIZE; i++) {
            assertEquals((i + 1) * 8f, out[i], 0.0f);
        }
        graph.close();
    }

    @Test
    public void mismatchingBodyRefusesBeforeAnyDeviceWork() {
        GpuGraph graph = captureScale(data, 2f);
        KernelHandle clone = backend.cloneKernel(scale);
        try {
            assertThrows(GpuGraphMismatchException.class, () -> graph.replay(
                    () -> queue.enqueueKernelNoEvent(clone, new long[]{SIZE}, new long[]{LOCAL},
                            new Object[]{data, 2f, SIZE})));
        } finally {
            clone.close();
        }
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatchScale(data, 2f);
            dispatchScale(data, 2f);
        }));
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
        }));
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, read(data), 0.0f);
        graph.close();
    }

    @Test
    public void copiesAndFillsAreCapturedAndReplayed() {
        queue.beginCapture();
        queue.enqueueFillBuffer(other, 0, (long) SIZE * Float.BYTES, 0);
        queue.enqueueCopyBuffer(data, 0, other, 0, (long) SIZE * Float.BYTES);
        GpuGraph graph = queue.endCapture();
        assertNotNull(graph);
        assertEquals(0, graph.dispatchCount());
        // nothing reached the device yet
        assertArrayEquals(new float[SIZE], read(other), 0.0f);

        graph.replay(() -> {
            queue.enqueueFillBuffer(other, 0, (long) SIZE * Float.BYTES, 0);
            queue.enqueueCopyBuffer(data, 0, other, 0, (long) SIZE * Float.BYTES);
        });
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, read(other), 0.0f);
        graph.close();
    }

    @Test
    public void aChangedCapturedCopyUpdatesAndRetracksTheNode() {
        float[] nines = new float[SIZE];
        float[] fives = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            nines[i] = 9f;
            fives[i] = 5f;
        }
        queue.enqueueWriteBuffer(other, 0, (long) SIZE * Float.BYTES, nines);
        queue.finish();

        queue.beginCapture();
        queue.enqueueCopyBuffer(data, 0, other, 0, (long) SIZE * Float.BYTES);
        GpuGraph graph = queue.endCapture();

        // the reversed copy reaches the executable node
        graph.replay(() -> queue.enqueueCopyBuffer(other, 0, data, 0, (long) SIZE * Float.BYTES));
        assertArrayEquals(nines, read(data), 0.0f);

        // and the next replay of the captured direction is compared against the reversed copy that ran
        queue.enqueueWriteBuffer(data, 0, (long) SIZE * Float.BYTES, fives);
        queue.finish();
        graph.replay(() -> queue.enqueueCopyBuffer(data, 0, other, 0, (long) SIZE * Float.BYTES));
        assertArrayEquals(fives, read(other), 0.0f);
        graph.close();
    }

    @Test
    public void transferDuringCaptureIsRefusedAndTheCaptureAborts() {
        queue.beginCapture();
        dispatchScale(data, 2f);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> queue.enqueueWriteBuffer(data, 0, (long) SIZE * Float.BYTES, new float[SIZE]));
        assertTrue(e.getMessage(), e.getMessage().contains("cannot be captured"));
        assertTrue(queue.isCapturing());
        queue.abortCapture();
        assertFalse(queue.isCapturing());
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, read(data), 0.0f);
    }

    @Test
    public void transferDuringReplayRefusesIt() {
        GpuGraph graph = captureScale(data, 2f);
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatchScale(data, 2f);
            queue.enqueueFillBuffer(other, 0, Float.BYTES, 0);
        }));
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, read(data), 0.0f);
        graph.close();
    }

    @Test
    public void emptyCaptureReturnsNullAndFinishDuringReplayRefuses() {
        queue.beginCapture();
        assertNull(queue.endCapture());
        GpuGraph graph = captureScale(data, 2f);
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatchScale(data, 2f);
            queue.finish();
        }));
        graph.close();
    }

    @Test
    public void aFailedStepAbortsItsCaptureWithoutDeviceWork() {
        queue.beginCapture();
        dispatchScale(data, 2f);
        queue.abortCapture();
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, read(data), 0.0f);
        assertThrows(IllegalStateException.class, queue::endCapture);
    }

    @Test
    public void closedGraphRefusesReplayAndCloseIsIdempotent() {
        GpuGraph graph = captureScale(data, 2f);
        graph.close();
        graph.close();
        assertThrows(IllegalStateException.class, () -> graph.launch(queue));
        assertThrows(IllegalStateException.class, () -> graph.replay(() -> dispatchScale(data, 2f)));
        assertTrue(graph.toString().contains("closed"));
    }

    @Test
    public void captureLifecycleIsEnforced() {
        assertThrows(IllegalStateException.class, queue::endCapture);
        queue.beginCapture();
        try {
            assertThrows(IllegalStateException.class, queue::beginCapture);
        } finally {
            queue.abortCapture();
        }
        queue.abortCapture();
        assertFalse(queue.isCapturing());
        dispatchScale(data, 2f);
        assertEquals(2f, read(data)[0], 0.0f);
    }

    @Test
    public void captureCallsFailOnAClosedBackend() {
        GpuGraph graph = captureScale(data, 2f);
        backend.close();
        assertThrows(IllegalStateException.class, queue::beginCapture);
        assertThrows(IllegalStateException.class, () -> graph.replay(() -> dispatchScale(data, 2f)));
        graph.close();
        backend = null;
        queue = null;
    }

    private static KernelArtifact scaleArtifact() {
        return KernelArtifact.hipImage(FakeHipDriver.SCALE_KERNEL,
                "hip-test-lib/src/hip/nf_test_scale.hsaco");
    }
}
