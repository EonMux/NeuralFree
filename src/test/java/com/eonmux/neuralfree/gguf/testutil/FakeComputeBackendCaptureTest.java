package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import org.junit.After;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Capture and replay of the fake queue, the host stand-in of a CUDA graph or a pre-recorded Vulkan
 * command buffer: a capture records dispatches without executing them, a replay verifies the body's
 * dispatches against the capture and executes the captured work once, an uncapturable command is
 * refused while capturing, a failed step aborts its capture without device work, and a replay that
 * leaves the capture throws before executing.
 */
public class FakeComputeBackendCaptureTest {

    private FakeComputeBackend backend;
    private FakeGpuBuffer input;
    private FakeGpuBuffer output;

    @After
    public void tearDown() {
        if (output != null) {
            output.close();
        }
        if (input != null) {
            input.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    /** Out[i] = in[i] + addend for i < n, and out[i] *= factor: the emulators the kernels run through. */
    private FakeComputeBackend open() {
        backend = FakeComputeBackend.builder()
                .graphCapture(true)
                .emulate("copy_add", d -> {
                    FakeGpuBuffer in = d.bufferArg(0);
                    FakeGpuBuffer out = d.bufferArg(1);
                    int n = d.intArg(2);
                    float addend = d.floatArg(3);
                    for (int i = 0; i < n; i++) {
                        out.putFloat(i, in.getFloat(i) + addend);
                    }
                })
                .emulate("scale_add", d -> {
                    FakeGpuBuffer out = d.bufferArg(0);
                    int n = d.intArg(1);
                    float factor = d.floatArg(2);
                    for (int i = 0; i < n; i++) {
                        out.putFloat(i, out.getFloat(i) * factor);
                    }
                })
                .build();
        input = (FakeGpuBuffer) backend.allocateBuffer(64 * Float.BYTES);
        output = (FakeGpuBuffer) backend.allocateBuffer(64 * Float.BYTES);
        for (int i = 0; i < 64; i++) {
            input.putFloat(i, i);
        }
        return backend;
    }

    private static long[] grid(int n) {
        return new long[]{n};
    }

    private void copyAdd(float addend, int n) {
        backend.computeQueue().enqueueKernelNoEvent(backend.compileKernel("copy_add", "", null), grid(n), grid(1),
                new Object[]{input, output, n, addend});
    }

    private void scaleAdd(float factor, int n) {
        backend.computeQueue().enqueueKernelNoEvent(backend.compileKernel("scale_add", "", null), grid(n), grid(1),
                new Object[]{output, n, factor});
    }

    private static float[] readFloats(FakeGpuBuffer buffer) {
        float[] out = new float[(int) (buffer.sizeInBytes() / Float.BYTES)];
        for (int i = 0; i < out.length; i++) {
            out[i] = buffer.getFloat(i);
        }
        return out;
    }

    private static float[] affine(float addend, float factor) {
        float[] out = new float[64];
        for (int i = 0; i < out.length; i++) {
            out[i] = (i + addend) * factor;
        }
        return out;
    }

    private GpuGraph captureTwoDispatches() {
        backend.computeQueue().beginCapture();
        copyAdd(10.0f, 64);
        scaleAdd(2.0f, 64);
        GpuGraph graph = backend.computeQueue().endCapture();
        assertNotNull(graph);
        assertEquals(2, graph.dispatchCount());
        return graph;
    }

    private void replayTwoDispatches(GpuGraph graph) {
        graph.replay(() -> {
            copyAdd(10.0f, 64);
            scaleAdd(2.0f, 64);
        });
    }

    @Test
    public void captureRecordsDispatchesWithoutExecutingThem() {
        open();
        captureTwoDispatches();
        assertEquals(0, backend.dispatches().size());
        assertArrayEquals(new float[64], readFloats(output), 0.0f);
    }

    @Test
    public void emptyCaptureReturnsNull() {
        open();
        backend.computeQueue().beginCapture();
        assertNull(backend.computeQueue().endCapture());
        backend.computeQueue().beginCapture();
        backend.computeQueue().abortCapture();
        assertThrows(IllegalStateException.class, () -> backend.computeQueue().endCapture());
        assertFalse(backend.computeQueue().isCapturing());
    }

    @Test
    public void replayRunsTheCapturedWorkOnce() {
        open();
        GpuGraph graph = captureTwoDispatches();
        replayTwoDispatches(graph);
        assertArrayEquals(affine(10.0f, 2.0f), readFloats(output), 0.0f);
        assertEquals(1, backend.ops().stream()
                .filter(op -> op.kind() == FakeOpKind.REPLAY).count());
        assertEquals(0, backend.dispatches().size());
    }

    @Test
    public void everyReplayRunsTheSameCapturedWork() {
        open();
        GpuGraph graph = captureTwoDispatches();
        replayTwoDispatches(graph);
        replayTwoDispatches(graph);
        // each replay rewrites out from in: both replays leave the same values, and two replay ops ran
        assertArrayEquals(affine(10.0f, 2.0f), readFloats(output), 0.0f);
        assertEquals(2, backend.ops().stream()
                .filter(op -> op.kind() == FakeOpKind.REPLAY).count());
    }

    @Test
    public void launchRunsTheCaptureOnceForTheStepThatRecordedIt() {
        open();
        GpuGraph graph = captureTwoDispatches();
        graph.launch(backend.computeQueue());
        assertArrayEquals(affine(10.0f, 2.0f), readFloats(output), 0.0f);
        assertEquals(1, backend.ops().stream()
                .filter(op -> op.kind() == FakeOpKind.LAUNCH).count());
        assertEquals(0, backend.dispatches().size());
    }

    @Test
    public void aGraphLaunchesOnAnotherQueueOfTheBackend() {
        open();
        GpuGraph graph = captureTwoDispatches();
        ComputeQueue other = backend.createComputeQueue();
        try {
            graph.launch(other);
            assertArrayEquals(affine(10.0f, 2.0f), readFloats(output), 0.0f);
        } finally {
            other.close();
        }
    }

    @Test
    public void replayRefusesFewerDispatchesThanTheCapture() {
        open();
        GpuGraph graph = captureTwoDispatches();
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> copyAdd(10.0f, 64)));
        assertArrayEquals(new float[64], readFloats(output), 0.0f);
        assertEquals(0, backend.dispatches().size());
    }

    @Test
    public void replayRefusesAChangedArgumentBeforeExecutingAnything() {
        open();
        GpuGraph graph = captureTwoDispatches();
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            copyAdd(11.0f, 64);
            scaleAdd(2.0f, 64);
        }));
        assertArrayEquals(new float[64], readFloats(output), 0.0f);
        assertEquals(0, backend.dispatches().size());
        graph.close();
    }

    @Test
    public void replayRefusesAnotherKernelAndExtraDispatches() {
        open();
        GpuGraph graph = captureTwoDispatches();
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            scaleAdd(2.0f, 64);
            scaleAdd(2.0f, 64);
        }));
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            copyAdd(10.0f, 64);
            scaleAdd(2.0f, 64);
            scaleAdd(2.0f, 64);
        }));
        assertArrayEquals(new float[64], readFloats(output), 0.0f);
    }

    @Test
    public void changedWorkSizesRefuseTheReplay() {
        open();
        backend.computeQueue().beginCapture();
        copyAdd(10.0f, 64);
        GpuGraph graph = backend.computeQueue().endCapture();
        assertNotNull(graph);
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> copyAdd(10.0f, 32)));
    }

    @Test
    public void transferDuringCaptureIsRefusedAndTheCaptureAborts() {
        open();
        backend.computeQueue().beginCapture();
        copyAdd(10.0f, 64);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> backend.computeQueue().enqueueWriteBuffer(input, 0, 8, new byte[8]));
        assertTrue(e.getMessage(), e.getMessage().contains("cannot be captured"));
        assertTrue(backend.computeQueue().isCapturing());
        backend.computeQueue().abortCapture();
        assertFalse(backend.computeQueue().isCapturing());
        // the refused write never executed and the aborted capture launched nothing
        assertEquals(63.0f, input.getFloat(63), 0.0f);
        assertArrayEquals(new float[64], readFloats(output), 0.0f);
    }

    @Test
    public void transferDuringReplayRefusesIt() {
        open();
        GpuGraph graph = captureTwoDispatches();
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            copyAdd(10.0f, 64);
            backend.computeQueue().enqueueWriteBuffer(input, 0, 4, new byte[4]);
            scaleAdd(2.0f, 64);
        }));
        assertArrayEquals(new float[64], readFloats(output), 0.0f);
    }

    @Test
    public void aFailingStepAbortsItsCaptureWithoutDeviceWork() {
        open();
        backend.computeQueue().beginCapture();
        copyAdd(10.0f, 64);
        backend.computeQueue().abortCapture();
        assertArrayEquals(new float[64], readFloats(output), 0.0f);
        assertEquals(0, backend.dispatches().size());
        backend.computeQueue().beginCapture();
        assertNull(backend.computeQueue().endCapture());
    }

    @Test
    public void closedGraphRefusesReplay() {
        open();
        GpuGraph graph = captureTwoDispatches();
        graph.close();
        assertTrue(graph.isClosed());
        assertThrows(IllegalStateException.class, () -> graph.launch(backend.computeQueue()));
        assertThrows(IllegalStateException.class, () -> graph.replay(() -> { }));
        assertTrue(graph.toString().contains("closed"));
    }

    @Test
    public void closedBoundBufferRefusesTheReplayByName() {
        open();
        GpuGraph graph = captureTwoDispatches();
        input.close();
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> graph.replay(() -> {
            copyAdd(10.0f, 64);
            scaleAdd(2.0f, 64);
        }));
        assertTrue(e.getMessage(), e.getMessage().contains("closed"));
    }

    @Test
    public void beginDuringACaptureOrReplayIsRefused() {
        open();
        backend.computeQueue().beginCapture();
        assertThrows(IllegalStateException.class, () -> backend.computeQueue().beginCapture());
        backend.computeQueue().abortCapture();
        GpuGraph graph = captureTwoDispatches();
        assertThrows(IllegalStateException.class, () -> graph.replay(() -> {
            backend.computeQueue().beginCapture();
        }));
    }

    @Test
    public void captureCallsFailOnAClosedBackend() {
        open();
        backend.computeQueue().beginCapture();
        backend.close();
        assertThrows(IllegalStateException.class, () -> backend.computeQueue().endCapture());
        backend = null;
        output = null;
        input = null;
    }
}
