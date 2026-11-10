package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.EnumSet;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * CUDA graph capture and replay on the device: a capture of two kernels records without executing, a
 * replay launches the instantiated graph once with the body's updated parameters, a changed scalar
 * argument and a changed grid are picked up by the node update, a mismatching body refuses before any
 * device work, and a host transfer inside the capture ends it with the recorded dispatches launched.
 */
public class CudaGraphCaptureTest {

    static final String ADD_PTX =
            ".version 7.5\n" +
            ".target sm_50\n" +
            ".address_size 64\n" +
            "\n" +
            ".visible .entry simple_add(\n" +
            "    .param .u64 a,\n" +
            "    .param .u64 b,\n" +
            "    .param .u64 c,\n" +
            "    .param .u32 n\n" +
            ") {\n" +
            "    .reg .f32   %f<4>;\n" +
            "    .reg .b32   %r<5>;\n" +
            "    .reg .b64   %rd<10>;\n" +
            "    .reg .pred  p0;\n" +
            "\n" +
            "    ld.param.u64    %rd1, [a];\n" +
            "    ld.param.u64    %rd2, [b];\n" +
            "    ld.param.u64    %rd3, [c];\n" +
            "    ld.param.u32    %r1, [n];\n" +
            "\n" +
            "    mov.u32         %r2, %tid.x;\n" +
            "    mov.u32         %r3, %ntid.x;\n" +
            "    mov.u32         %r4, %ctaid.x;\n" +
            "    mad.lo.u32      %r2, %r4, %r3, %r2;\n" +
            "\n" +
            "    setp.ge.u32     p0, %r2, %r1;\n" +
            "    @p0 bra         L_exit;\n" +
            "\n" +
            "    mul.wide.u32    %rd4, %r2, 4;\n" +
            "    add.s64         %rd5, %rd1, %rd4;\n" +
            "    add.s64         %rd6, %rd2, %rd4;\n" +
            "    add.s64         %rd7, %rd3, %rd4;\n" +
            "\n" +
            "    ld.global.f32   %f1, [%rd5];\n" +
            "    ld.global.f32   %f2, [%rd6];\n" +
            "    add.f32         %f3, %f1, %f2;\n" +
            "    st.global.f32   [%rd7], %f3;\n" +
            "\n" +
            "L_exit:\n" +
            "    ret;\n" +
            "}\n";

    private static final String SCALE_PTX =
            ".version 7.5\n" +
            ".target sm_50\n" +
            ".address_size 64\n" +
            "\n" +
            ".visible .entry scale(\n" +
            "    .param .u64 data,\n" +
            "    .param .f32 factor,\n" +
            "    .param .u32 n\n" +
            ") {\n" +
            "    .reg .f32   %f<4>;\n" +
            "    .reg .b32   %r<5>;\n" +
            "    .reg .b64   %rd<8>;\n" +
            "    .reg .pred  p0;\n" +
            "\n" +
            "    ld.param.u64    %rd1, [data];\n" +
            "    ld.param.f32    %f1, [factor];\n" +
            "    ld.param.u32    %r1, [n];\n" +
            "\n" +
            "    mov.u32         %r2, %tid.x;\n" +
            "    mov.u32         %r3, %ntid.x;\n" +
            "    mov.u32         %r4, %ctaid.x;\n" +
            "    mad.lo.u32      %r2, %r4, %r3, %r2;\n" +
            "\n" +
            "    setp.ge.u32     p0, %r2, %r1;\n" +
            "    @p0 bra         L_exit;\n" +
            "\n" +
            "    mul.wide.u32    %rd2, %r2, 4;\n" +
            "    add.s64         %rd3, %rd1, %rd2;\n" +
            "\n" +
            "    ld.global.f32   %f2, [%rd3];\n" +
            "    mul.f32         %f3, %f2, %f1;\n" +
            "    st.global.f32   [%rd3], %f3;\n" +
            "\n" +
            "L_exit:\n" +
            "    ret;\n" +
            "}\n";

    private static final int SIZE = 256;
    private static final int LOCAL = 64;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private ComputeQueue queue;
    private GpuBuffer a;
    private GpuBuffer b;
    private GpuBuffer c;
    private KernelHandle add;
    private KernelHandle scale;

    @Before
    public void setUp() {
        HWAcceleration accel = null;
        try {
            accel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
            Assume.assumeTrue("no CUDA GPU", !accel.getCUDevices().isEmpty());
            backend = new CudaComputeBackend(accel.getCUDevices().get(0));
            hwAccel = accel;
            accel = null;
        } finally {
            if (accel != null) {
                accel.shutdown();
            }
        }
        queue = backend.computeQueue();
        assertTrue(queue.supportsGraphCapture());
        a = backend.allocateBuffer((long) SIZE * Float.BYTES);
        b = backend.allocateBuffer((long) SIZE * Float.BYTES);
        c = backend.allocateBuffer((long) SIZE * Float.BYTES);
        add = backend.compileKernel("simple_add", ADD_PTX, null);
        scale = backend.compileKernel("scale", SCALE_PTX, null);
        float[] as = new float[SIZE];
        float[] bs = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            as[i] = i;
            bs[i] = 1;
        }
        queue.enqueueWriteBuffer(a, 0, (long) SIZE * Float.BYTES, as);
        queue.enqueueWriteBuffer(b, 0, (long) SIZE * Float.BYTES, bs);
        queue.enqueueFillBuffer(c, 0, (long) SIZE * Float.BYTES, 0);
        queue.finish();
    }

    @After
    public void tearDown() {
        closeQuietly(c, b, a, add, scale);
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    private static void closeQuietly(AutoCloseable... closeables) {
        for (AutoCloseable c : closeables) {
            try {
                c.close();
            } catch (Exception ignored) {
            }
        }
    }

    private float[] readC() {
        float[] out = new float[SIZE];
        queue.enqueueReadBuffer(c, 0, (long) SIZE * Float.BYTES, out);
        queue.finish();
        return out;
    }

    private void dispatchAdd(long n) {
        queue.enqueueKernelNoEvent(add, new long[]{roundUp(n)}, new long[]{LOCAL},
                new Object[]{a, b, c, (int) n});
    }

    private void dispatchScale(float factor, long n) {
        queue.enqueueKernelNoEvent(scale, new long[]{roundUp(n)}, new long[]{LOCAL},
                new Object[]{c, factor, (int) n});
    }

    private static long roundUp(long n) {
        return (n + LOCAL - 1) / LOCAL * LOCAL;
    }

    private GpuGraph captureAddScale() {
        queue.beginCapture();
        dispatchAdd(SIZE);
        dispatchScale(2.0f, SIZE);
        GpuGraph graph = queue.endCapture();
        assertNotNull(graph);
        assertEquals(2, graph.dispatchCount());
        return graph;
    }

    @Test
    public void captureRecordsAndReplayLaunchesOnce() {
        GpuGraph graph = captureAddScale();
        assertEquals(0, readC()[0], 0.0f);
        graph.replay(() -> {
            dispatchAdd(SIZE);
            dispatchScale(2.0f, SIZE);
        });
        float[] expected = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expected[i] = (i + 1) * 2.0f;
        }
        assertArrayEquals(expected, readC(), 1e-4f);
    }

    @Test
    public void theCapturingStepLaunchesItsCaptureOnce() {
        GpuGraph graph = captureAddScale();
        graph.launch(queue);
        float[] expected = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expected[i] = (i + 1) * 2.0f;
        }
        assertArrayEquals(expected, readC(), 1e-4f);
    }

    @Test
    public void replayPicksUpAChangedScalarArgument() {
        GpuGraph graph = captureAddScale();
        graph.replay(() -> {
            dispatchAdd(SIZE);
            dispatchScale(3.0f, SIZE);
        });
        float[] expected = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expected[i] = (i + 1) * 3.0f;
        }
        assertArrayEquals(expected, readC(), 1e-4f);
    }

    @Test
    public void replayPicksUpAChangedGrid() {
        GpuGraph graph = captureAddScale();
        int half = SIZE / 2;
        graph.replay(() -> {
            dispatchAdd(half);
            dispatchScale(2.0f, half);
        });
        float[] out = readC();
        assertEquals((0 + 1) * 2.0f, out[0], 1e-4f);
        assertEquals((half - 1 + 1) * 2.0f, out[half - 1], 1e-4f);
        assertEquals(0.0f, out[half], 0.0f);
    }

    @Test
    public void repeatedReplaysKeepWorking() {
        GpuGraph graph = captureAddScale();
        for (int round = 1; round <= 3; round++) {
            float factor = round;
            graph.replay(() -> {
                dispatchAdd(SIZE);
                dispatchScale(factor, SIZE);
            });
        }
        float[] expected = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expected[i] = (i + 1) * 3.0f;
        }
        assertArrayEquals(expected, readC(), 1e-4f);
    }

    @Test
    public void mismatchingBodyRefusesBeforeAnyDeviceWork() {
        GpuGraph graph = captureAddScale();
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatchScale(2.0f, SIZE);
            dispatchAdd(SIZE);
        }));
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatchAdd(SIZE);
            dispatchScale(2.0f, SIZE);
            dispatchAdd(SIZE);
        }));
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> dispatchAdd(SIZE)));
        assertEquals(0.0f, readC()[0], 0.0f);
    }

    @Test
    public void transferDuringCaptureIsRefusedAndTheCaptureAborts() {
        queue.beginCapture();
        dispatchAdd(SIZE);
        float[] pattern = new float[SIZE];
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> queue.enqueueWriteBuffer(b, 0, (long) SIZE * Float.BYTES, pattern));
        assertTrue(e.getMessage(), e.getMessage().contains("cannot be captured"));
        assertTrue(queue.isCapturing());
        queue.abortCapture();
        assertFalse(queue.isCapturing());
        // neither the refused write nor the aborted capture reached the device
        assertArrayEquals(new float[SIZE], readC(), 0.0f);
    }

    @Test
    public void transferDuringReplayRefusesIt() {
        GpuGraph graph = captureAddScale();
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatchAdd(SIZE);
            queue.enqueueFillBuffer(c, 0, Float.BYTES, 0);
            dispatchScale(2.0f, SIZE);
        }));
        assertEquals(0.0f, readC()[0], 0.0f);
    }

    @Test
    public void emptyCaptureReturnsNullAndFinishDuringReplayRefuses() {
        queue.beginCapture();
        assertNull(queue.endCapture());
        GpuGraph graph = captureAddScale();
        assertThrows(GpuGraphMismatchException.class, () -> graph.replay(() -> {
            dispatchAdd(SIZE);
            queue.finish();
            dispatchScale(2.0f, SIZE);
        }));
        graph.close();
    }

    @Test
    public void aFailedStepAbortsItsCaptureWithoutDeviceWork() {
        queue.beginCapture();
        dispatchAdd(SIZE);
        queue.abortCapture();
        assertArrayEquals(new float[SIZE], readC(), 0.0f);
        assertThrows(IllegalStateException.class, queue::endCapture);
    }

    @Test
    public void closedGraphRefusesReplayAndCloseIsIdempotent() {
        GpuGraph graph = captureAddScale();
        graph.close();
        graph.close();
        assertThrows(IllegalStateException.class, () -> graph.launch(queue));
        assertThrows(IllegalStateException.class, () -> graph.replay(() -> {
            dispatchAdd(SIZE);
            dispatchScale(2.0f, SIZE);
        }));
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
        dispatchAdd(SIZE);
        assertEquals(1.0f, readC()[0], 1e-4f);
    }

    @Test
    public void captureCallsFailOnAClosedBackend() {
        GpuGraph graph = captureAddScale();
        backend.close();
        assertThrows(IllegalStateException.class, () -> queue.beginCapture());
        assertThrows(IllegalStateException.class, () -> graph.replay(() -> {
            dispatchAdd(SIZE);
            dispatchScale(2.0f, SIZE);
        }));
        graph.close();
        backend = null;
        queue = null;
    }
}
