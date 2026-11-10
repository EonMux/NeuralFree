package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchPlan;
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

import java.util.List;

/**
 * Command-buffer capture and replay on a Vulkan device: a capture of two dispatches records into a
 * simultaneous-use command buffer without executing, a replay verifies the body's dispatches against
 * the capture and resubmits the buffer once, a changed scalar argument or grid is re-recorded (by a body
 * replay or a patched replay, prepared ahead or not, over more steps than the recording's ring holds, with a
 * body replay between a prepared step and its patched replay), a global size that moves alone is adopted
 * without re-recording, another buffer, sequence or local size refuses the replay before any device work, and a
 * transfer inside the capture is refused.
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
    public void bodyReplayReRecordsAChangedScalarAndGrid() {
        GpuGraph graph = captureTwoDispatches();
        graph.replay(() -> {
            dispatch(values, SIZE);
            dispatch(values, SIZE / 2);
        });
        assertArrayEquals(new int[]{1}, graph.takeReplayChanges());
        assertArrayEquals(new int[0], graph.takeReplayChanges());
        float[] want = iota(1.0f);
        addFirst(want, SIZE / 2, 1.0f);
        assertArrayEquals(want, readValues(), 1e-4f);
        GpuGraphDispatch changed = graph.kernelDispatch(1);
        assertEquals(SIZE / 2, changed.arg(1));
        assertArrayEquals(new long[]{roundUp(SIZE / 2)}, changed.grid());

        replayTwoDispatches(graph);
        addFirst(want, SIZE, 2.0f);
        assertArrayEquals(want, readValues(), 1e-4f);
    }

    @Test
    public void replayRefusesAnotherBufferBeforeExecutingAnything() {
        GpuGraph graph = captureTwoDispatches();
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatch(values, SIZE);
            dispatch(scratch, SIZE);
        }));
        assertEquals(0.0f, readValues()[0], 0.0f);
    }

    /**
     * A plan that sets the second dispatch's count to the step position (and its grid to match), and from
     * step {@code firstFrom} on the first dispatch's count too, so a dispatch that first changes after the graph
     * went to re-recording reaches the later recordings.
     */
    private static GpuGraphPatchPlan countPlan(int firstFrom) {
        return (pos, sink) -> {
            sink.patchKernel(1, new long[]{roundUp(pos)}, new int[]{1}, new Object[]{pos});
            if (pos >= firstFrom) {
                sink.patchKernel(0, new long[]{roundUp(pos)}, new int[]{1}, new Object[]{pos});
            }
            return true;
        };
    }

    @Test
    public void patchedReplaysOverMoreStepsThanTheRingHoldsMatchTheDispatchedValues() {
        GpuGraph graph = captureTwoDispatches();
        assertTrue(graph.supportsPatchedReplay());
        List<GpuGraphDispatch> captured = graph.kernelDispatches();
        assertEquals(2, captured.size());
        assertEquals(SIZE, captured.get(1).arg(1));
        GpuGraphPatchPlan plan = countPlan(200);
        float[] want = iota(0.0f);
        for (int pos = 100; pos < 220; pos += 15) {
            boolean prepared = pos > 160 && graph.preparePatched(plan, pos, pos - 15);
            assertEquals(pos > 160, prepared);
            assertTrue(graph.replayPatched(plan, pos));
            addFirst(want, pos >= 200 ? pos : SIZE, 1.0f);
            addFirst(want, pos, 1.0f);
            assertArrayEquals("step " + pos, want, readValues(), 1e-4f);
        }
        assertEquals(205, graph.kernelDispatch(0).arg(1));
        assertEquals(205, graph.kernelDispatch(1).arg(1));
    }

    @Test
    public void aPreparedStepAtAnotherPositionIsPatchedAgain() {
        GpuGraph graph = captureTwoDispatches();
        GpuGraphPatchPlan plan = countPlan(Integer.MAX_VALUE);
        assertTrue(graph.replayPatched(plan, 50));
        assertTrue(graph.preparePatched(plan, 60, 50));
        assertTrue(graph.replayPatched(plan, 70));
        float[] want = iota(2.0f);
        addFirst(want, 50, 1.0f);
        addFirst(want, 70, 1.0f);
        assertArrayEquals(want, readValues(), 1e-4f);
    }

    /**
     * A body replay between {@code preparePatched(P)} and {@code replayPatched(P)} records over the prepared ring slot
     * with the body's values, so the patched replay must patch step P again instead of trusting the slot.
     */
    @Test
    public void aBodyReplayAfterThePreparedStepLeavesThePatchPlanAuthoritative() {
        GpuGraph graph = captureTwoDispatches();
        GpuGraphPatchPlan plan = countPlan(0);
        assertTrue(graph.replayPatched(plan, 50));
        assertTrue(graph.preparePatched(plan, 60, 50));
        replayTwoDispatches(graph);
        assertTrue(graph.replayPatched(plan, 60));
        float[] want = iota(0.0f);
        addFirst(want, 50, 2.0f);
        addFirst(want, SIZE, 2.0f);
        addFirst(want, 60, 2.0f);
        assertArrayEquals(want, readValues(), 1e-4f);
        assertEquals(60, graph.kernelDispatch(0).arg(1));
    }

    /**
     * A global size that changes alone (the group counts fold to the same number) is adopted: the dispatch reports
     * it, the body replay lists it as changed, and the captured command buffer still serves, as it holds nothing
     * that moved.
     */
    @Test
    public void aGlobalSizeChangeAloneIsAdoptedWithoutReRecording() {
        GpuGraph graph = captureTwoDispatches();
        long narrower = SIZE - LOCAL / 2;
        graph.replay(() -> {
            dispatch(values, SIZE);
            queue.enqueueKernelNoEvent(increment, new long[]{narrower}, new long[]{LOCAL},
                    new Object[]{values, SIZE});
        });
        assertArrayEquals(new int[]{1}, graph.takeReplayChanges());
        assertArrayEquals(new long[]{narrower}, graph.kernelDispatch(1).grid());
        assertNull("nothing the command buffer holds moved", ((VkGraph) graph).recording());
        assertArrayEquals(iota(2.0f), readValues(), 1e-4f);

        assertTrue(graph.replayPatched((pos, sink) -> {
            sink.patchKernel(0, new long[]{narrower}, new int[0], new Object[0]);
            return true;
        }, 1));
        assertArrayEquals(new long[]{narrower}, graph.kernelDispatch(0).grid());
        assertNull(((VkGraph) graph).recording());
        assertArrayEquals(iota(4.0f), readValues(), 1e-4f);
    }

    /**
     * A ring slot still executing is waited for outside the queue's lock: re-recorded steps of heavy dispatches
     * submitted faster than the device runs them reuse slots in flight, and no recording under the lock blocks on the
     * device, so other threads' polls and flushes never stall behind it.
     */
    @Test
    public void aSlotStillInFlightIsAwaitedOutsideTheQueueLock() {
        int n = 1 << 21;
        int dispatches = 64;
        try (GpuBuffer big = backend.allocateBuffer((long) n * Float.BYTES)) {
            queue.beginCapture();
            for (int i = 0; i < dispatches; i++) {
                dispatch(big, n);
            }
            GpuGraph graph = queue.endCapture();
            GpuGraphPatchPlan plan = (pos, sink) -> {
                sink.patchKernel(0, new long[]{roundUp(n - pos)}, new int[]{1}, new Object[]{n - pos});
                return true;
            };
            for (int pos = 1; pos <= 4 * VkGraphRecording.RING; pos++) {
                assertTrue(graph.replayPatched(plan, pos));
                if (pos % 2 == 0) {
                    assertTrue(graph.preparePatched(plan, pos + 1, pos));
                }
            }
            queue.finish();
            assertEquals("recordings that blocked on the device under the queue's lock", 0,
                    ((VkComputeQueue) queue).graphResources.blockedWaits());
            graph.close();
        }
    }

    @Test
    public void aRefusedPlanSubmitsNothing() {
        GpuGraph graph = captureTwoDispatches();
        assertFalse(graph.replayPatched((pos, sink) -> false, 3));
        assertEquals(0.0f, readValues()[0], 0.0f);
        assertThrows(GpuGraphMismatchException.class, () -> graph.replayPatched((pos, sink) -> {
            sink.patchKernel(0, null, new int[]{0}, new Object[]{scratch});
            return true;
        }, 3));
        assertEquals(0.0f, readValues()[0], 0.0f);
    }

    /** Adds {@code addend} to the first {@code n} elements, as one increment dispatch over n does once. */
    private static void addFirst(float[] values, int n, float addend) {
        for (int i = 0; i < n && i < values.length; i++) {
            values[i] += addend;
        }
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
