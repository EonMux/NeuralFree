package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * OpenCL command-buffer capture and replay on a real device: the graph tests are skipped by name on
 * devices without {@code cl_khr_command_buffer}, and the capability test runs on every device. A
 * capture of two kernels records without executing, its launch runs them, a replay resubmits the
 * command buffer, changed scalar arguments are updated in place only where mutable dispatch is
 * available (otherwise the replay refuses and the caller re-captures), and a mismatching body is
 * refused before any device work.
 */
public class OclGraphCaptureTest {

    private static final String ADD_KERNEL =
            "__kernel void graph_add(__global float *a, __global float *b, __global float *c, const int n) {\n" +
            "    int i = get_global_id(0);\n" +
            "    if (i < n) {\n" +
            "        c[i] = a[i] + b[i];\n" +
            "    }\n" +
            "}\n";

    private static final String SCALE_KERNEL =
            "__kernel void graph_scale(__global float *data, const float factor, const int n) {\n" +
            "    int i = get_global_id(0);\n" +
            "    if (i < n) {\n" +
            "        data[i] = data[i] * factor;\n" +
            "    }\n" +
            "}\n";

    private static final int SIZE = 64;
    private static final int LOCAL = 16;

    private HWAcceleration hwAccel;
    private OclComputeBackend backend;
    private ComputeQueue queue;
    private GpuBuffer a;
    private GpuBuffer b;
    private GpuBuffer c;
    private KernelHandle add;
    private KernelHandle scale;
    private final List<GpuEvent> events = new ArrayList<>();

    @Before
    public void setUp() {
        try {
            hwAccel = new HWAcceleration();
            if (!hwAccel.getCLDevices().isEmpty()) {
                CLDevice device = hwAccel.getCLDevices().get(0);
                backend = new OclComputeBackend(device);
                queue = backend.computeQueue();
                add = backend.compileKernel("graph_add", ADD_KERNEL, null);
                scale = backend.compileKernel("graph_scale", SCALE_KERNEL, null);
                a = backend.allocateBuffer((long) SIZE * Float.BYTES);
                b = backend.allocateBuffer((long) SIZE * Float.BYTES);
                c = backend.allocateBuffer((long) SIZE * Float.BYTES);
                float[] as = new float[SIZE];
                float[] bs = new float[SIZE];
                for (int i = 0; i < SIZE; i++) {
                    as[i] = i;
                    bs[i] = 1;
                }
                track(queue.enqueueWriteBuffer(a, 0, (long) SIZE * Float.BYTES, as));
                track(queue.enqueueWriteBuffer(b, 0, (long) SIZE * Float.BYTES, bs));
                queue.enqueueFillBuffer(c, 0, (long) SIZE * Float.BYTES, 0);
                queue.finish();
            }
        } catch (Exception e) {
            System.out.println("OpenCL graph setup failed: " + e.getMessage());
        }
        Assume.assumeTrue("no OpenCL GPU", backend != null);
    }

    @After
    public void tearDown() {
        for (GpuEvent event : events) {
            try {
                event.close();
            } catch (RuntimeException ignored) {
            }
        }
        events.clear();
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    private GpuEvent track(GpuEvent event) {
        if (event != null) {
            events.add(event);
        }
        return event;
    }

    private void requireCommandBuffer(String test) {
        Assume.assumeTrue(test + " skipped: the device does not support cl_khr_command_buffer",
                backend.capabilities().hasFeature(DeviceFeature.COMMAND_BUFFER));
        assertTrue(queue.supportsGraphCapture());
    }

    private float[] readC() {
        float[] out = new float[SIZE];
        track(queue.enqueueReadBuffer(c, 0, (long) SIZE * Float.BYTES, out));
        queue.finish();
        return out;
    }

    private void dispatchAdd() {
        queue.enqueueKernelNoEvent(add, new long[]{SIZE}, new long[]{LOCAL},
                new Object[]{a, b, c, SIZE});
    }

    private void dispatchScale(float factor) {
        queue.enqueueKernelNoEvent(scale, new long[]{SIZE}, new long[]{LOCAL},
                new Object[]{c, factor, SIZE});
    }

    private GpuGraph captureAddScale(float factor) {
        queue.beginCapture();
        dispatchAdd();
        dispatchScale(factor);
        GpuGraph graph = queue.endCapture();
        assertNotNull(graph);
        assertEquals(2, graph.dispatchCount());
        return graph;
    }

    @Test
    public void captureSupportMatchesTheCapabilityTable() {
        assertEquals(backend.capabilities().hasFeature(DeviceFeature.COMMAND_BUFFER),
                queue.supportsGraphCapture());
    }

    @Test
    public void aDeviceWithoutTheExtensionNamesItAndRefusesCapture() {
        Assume.assumeFalse(queue.supportsGraphCapture());
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class, queue::beginCapture);
        assertTrue(e.getMessage(), e.getMessage().contains("cl_khr_command_buffer"));
        assertThrows(UnsupportedOperationException.class, queue::endCapture);
        assertThrows(UnsupportedOperationException.class, queue::abortCapture);
        assertFalse(queue.isCapturing());
    }

    @Test
    public void captureRecordsAndReplayLaunchesOnce() {
        requireCommandBuffer("capture");
        GpuGraph graph = captureAddScale(2f);
        assertArrayEquals(new float[SIZE], readC(), 0.0f);

        graph.replay(() -> {
            dispatchAdd();
            dispatchScale(2f);
        });
        float[] expected = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expected[i] = (i + 1) * 2f;
        }
        assertArrayEquals(expected, readC(), 1e-5f);
        graph.close();
    }

    @Test
    public void theCapturingStepLaunchesItsCaptureOnce() {
        requireCommandBuffer("launch");
        GpuGraph graph = captureAddScale(2f);
        track(graph.launch(queue));
        queue.finish();
        float[] expected = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expected[i] = (i + 1) * 2f;
        }
        assertArrayEquals(expected, readC(), 1e-5f);
        graph.close();
    }

    @Test
    public void aChangedScalarArgumentIsUpdatedWhenMutableDispatchIsAvailable() {
        requireCommandBuffer("mutable scalar");
        GpuGraph graph = captureAddScale(2f);
        if (((OclComputeQueue) queue).supportsMutableDispatch()) {
            graph.replay(() -> {
                dispatchAdd();
                dispatchScale(3f);
            });
            float[] expected = new float[SIZE];
            for (int i = 0; i < SIZE; i++) {
                expected[i] = (i + 1) * 3f;
            }
            assertArrayEquals(expected, readC(), 1e-5f);
        } else {
            assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
                dispatchAdd();
                dispatchScale(3f);
            }));
            assertArrayEquals(new float[SIZE], readC(), 1.0f);
        }
        graph.close();
    }

    @Test
    public void repeatedMutableUpdatesFollowEveryReplay() {
        requireCommandBuffer("mutable sequence");
        GpuGraph graph = captureAddScale(2f);
        Assume.assumeTrue("the device does not support mutable dispatch",
                ((OclComputeQueue) queue).supportsMutableDispatch());
        float[] expected = new float[SIZE];
        for (int round = 0; round < 2; round++) {
            final float factor = round == 0 ? 3f : 2f;
            graph.replay(() -> {
                dispatchAdd();
                dispatchScale(factor);
            });
            for (int i = 0; i < SIZE; i++) {
                expected[i] = (i + 1) * factor;
            }
            assertArrayEquals("replay " + round + " with factor " + factor, expected, readC(), 1e-5f);
        }
        graph.close();
    }

    @Test
    public void aChangedWorkSizeNeedsANewCapture() {
        requireCommandBuffer("work size");
        GpuGraph graph = captureAddScale(2f);
        int half = SIZE / 2;
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            queue.enqueueKernelNoEvent(add, new long[]{half}, new long[]{LOCAL},
                    new Object[]{a, b, c, half});
            dispatchScale(2f);
        }));
        assertArrayEquals(new float[SIZE], readC(), 0.0f);
        graph.close();
    }

    @Test
    public void mismatchingBodyRefusesBeforeAnyDeviceWork() {
        requireCommandBuffer("mismatch");
        GpuGraph graph = captureAddScale(2f);
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatchScale(2f);
            dispatchAdd();
        }));
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatchAdd();
            dispatchScale(2f);
            dispatchAdd();
        }));
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(this::dispatchAdd));
        assertArrayEquals(new float[SIZE], readC(), 0.0f);
        graph.close();
    }

    @Test
    public void copiesAndFillsAreCapturedAndReplayed() {
        requireCommandBuffer("copy and fill");
        queue.beginCapture();
        queue.enqueueFillBuffer(c, 0, (long) SIZE * Float.BYTES, 0x3f800000);
        queue.enqueueCopyBuffer(a, 0, c, 0, (long) SIZE * Float.BYTES);
        GpuGraph graph = queue.endCapture();
        assertNotNull(graph);
        assertEquals(0, graph.dispatchCount());
        assertArrayEquals(new float[SIZE], readC(), 0.0f);

        graph.replay(() -> {
            queue.enqueueFillBuffer(c, 0, (long) SIZE * Float.BYTES, 0x3f800000);
            queue.enqueueCopyBuffer(a, 0, c, 0, (long) SIZE * Float.BYTES);
        });
        float[] expected = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expected[i] = i;
        }
        assertArrayEquals(expected, readC(), 0.0f);
        graph.close();
    }

    @Test
    public void transferDuringCaptureIsRefusedAndTheCaptureAborts() {
        requireCommandBuffer("transfer capture");
        queue.beginCapture();
        dispatchAdd();
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> queue.enqueueWriteBuffer(b, 0, (long) SIZE * Float.BYTES, new float[SIZE]));
        assertTrue(e.getMessage(), e.getMessage().contains("cannot be captured"));
        assertTrue(queue.isCapturing());
        queue.abortCapture();
        assertFalse(queue.isCapturing());
        assertArrayEquals(new float[SIZE], readC(), 0.0f);
    }

    @Test
    public void transferDuringReplayRefusesIt() {
        requireCommandBuffer("transfer replay");
        GpuGraph graph = captureAddScale(2f);
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatchAdd();
            queue.enqueueFillBuffer(c, 0, Float.BYTES, 0);
            dispatchScale(2f);
        }));
        assertArrayEquals(new float[SIZE], readC(), 0.0f);
        graph.close();
    }

    @Test
    public void emptyCaptureReturnsNullAndFinishDuringReplayRefuses() {
        requireCommandBuffer("empty capture");
        queue.beginCapture();
        assertNull(queue.endCapture());
        GpuGraph graph = captureAddScale(2f);
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatchAdd();
            queue.finish();
            dispatchScale(2f);
        }));
        graph.close();
    }

    @Test
    public void aFailedStepAbortsItsCaptureWithoutDeviceWork() {
        requireCommandBuffer("abort");
        queue.beginCapture();
        dispatchAdd();
        queue.abortCapture();
        assertArrayEquals(new float[SIZE], readC(), 0.0f);
        assertThrows(IllegalStateException.class, queue::endCapture);
    }

    @Test
    public void aGraphRunsOnlyOnTheQueueThatCapturedIt() {
        requireCommandBuffer("owner queue");
        GpuGraph graph = captureAddScale(2f);
        assertThrows(IllegalArgumentException.class, () -> graph.launch(backend.transferQueue()));
        graph.close();
    }

    @Test
    public void closedGraphRefusesReplayAndCloseIsIdempotent() {
        requireCommandBuffer("close");
        GpuGraph graph = captureAddScale(2f);
        graph.close();
        graph.close();
        assertThrows(IllegalStateException.class, () -> graph.launch(queue));
        assertThrows(IllegalStateException.class, () -> graph.replay(() -> {
            dispatchAdd();
            dispatchScale(2f);
        }));
        assertTrue(graph.toString().contains("closed"));
    }

    @Test
    public void captureLifecycleIsEnforced() {
        requireCommandBuffer("lifecycle");
        assertThrows(IllegalStateException.class, queue::endCapture);
        queue.beginCapture();
        try {
            assertThrows(IllegalStateException.class, queue::beginCapture);
        } finally {
            queue.abortCapture();
        }
        queue.abortCapture();
        assertFalse(queue.isCapturing());
        dispatchAdd();
        assertEquals(1f, readC()[0], 1e-5f);
    }

    @Test
    public void captureCallsFailOnAClosedBackend() {
        requireCommandBuffer("closed backend");
        GpuGraph graph = captureAddScale(2f);
        backend.close();
        assertThrows(IllegalStateException.class, queue::beginCapture);
        assertThrows(IllegalStateException.class, () -> graph.replay(() -> {
            dispatchAdd();
            dispatchScale(2f);
        }));
        graph.close();
        backend = null;
        queue = null;
    }
}
