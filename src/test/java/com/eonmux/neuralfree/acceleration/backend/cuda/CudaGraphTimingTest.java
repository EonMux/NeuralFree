package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import org.junit.After;
import org.junit.Assume;
import org.junit.Test;

import java.util.EnumSet;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Graph launches under CUDA command timing: with {@code -Dneuralfree.gpu.profile} at {@code true} or {@code step},
 * the capture's first launch and every replay report once under {@link KernelTiming#graphKey(int)} with a positive
 * device time, the dispatches inside the graph have no entries of their own, and the commands around the graph keep
 * theirs.
 */
public class CudaGraphTimingTest {

    private static final int SIZE = 256;
    private static final int LOCAL = 64;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private GpuBuffer a;
    private GpuBuffer b;
    private GpuBuffer c;
    private KernelHandle add;
    private String prior;

    private ComputeQueue open(String mode) {
        prior = System.getProperty(KernelTiming.PROFILE_PROPERTY);
        System.setProperty(KernelTiming.PROFILE_PROPERTY, mode);
        HWAcceleration accel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
        if (accel.getCUDevices().isEmpty()) {
            accel.shutdown();
            Assume.assumeTrue("no CUDA GPU", false);
        }
        hwAccel = accel;
        backend = new CudaComputeBackend(accel.getCUDevices().get(0));
        ComputeQueue queue = backend.computeQueue();
        a = backend.allocateBuffer((long) SIZE * Float.BYTES);
        b = backend.allocateBuffer((long) SIZE * Float.BYTES);
        c = backend.allocateBuffer((long) SIZE * Float.BYTES);
        add = backend.compileKernel("simple_add", CudaGraphCaptureTest.ADD_PTX, null);
        float[] ones = new float[SIZE];
        java.util.Arrays.fill(ones, 1.0f);
        queue.enqueueWriteBuffer(a, 0, (long) SIZE * Float.BYTES, ones);
        queue.enqueueWriteBuffer(b, 0, (long) SIZE * Float.BYTES, ones);
        queue.finish();
        return queue;
    }

    @After
    public void tearDown() {
        for (AutoCloseable closeable : new AutoCloseable[]{add, c, b, a}) {
            if (closeable != null) {
                try {
                    closeable.close();
                } catch (Exception e) {
                    throw new AssertionError(e);
                }
            }
        }
        add = null;
        a = b = c = null;
        if (backend != null) {
            backend.close();
            backend = null;
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
            hwAccel = null;
        }
        if (prior == null) {
            System.clearProperty(KernelTiming.PROFILE_PROPERTY);
        } else {
            System.setProperty(KernelTiming.PROFILE_PROPERTY, prior);
        }
    }

    private void dispatchAdd(ComputeQueue queue) {
        queue.enqueueKernelNoEvent(add, new long[]{SIZE}, new long[]{LOCAL}, new Object[]{a, b, c, SIZE});
    }

    private void launchAndReplay(String mode) {
        ComputeQueue queue = open(mode);
        KernelTiming timing = queue.kernelTiming();
        assertNotNull(mode + ": the queue times its commands", timing);
        assertTrue(mode + ": and its graph launches", queue.timesGraphLaunches());
        queue.beginCapture();
        dispatchAdd(queue);
        dispatchAdd(queue);
        GpuGraph graph = queue.endCapture();
        assertNotNull(graph);
        try {
            timing.reset();
            graph.launch(queue).close();
            for (int i = 0; i < 2; i++) {
                graph.replay(() -> {
                    dispatchAdd(queue);
                    dispatchAdd(queue);
                });
            }
            float[] out = new float[SIZE];
            queue.enqueueReadBuffer(c, 0, (long) SIZE * Float.BYTES, out);
            queue.finish();
            float[] want = new float[SIZE];
            java.util.Arrays.fill(want, 2.0f);
            assertArrayEquals(mode + ": the graph computed", want, out, 0.0f);
            KernelTiming.Entry graphEntry = null;
            for (KernelTiming.Entry e : timing.snapshot()) {
                assertTrue(mode + ": no dispatch of the graph is timed on its own: " + e.name(),
                        !e.name().startsWith("simple_add"));
                if (e.name().equals(KernelTiming.graphKey(2))) {
                    graphEntry = e;
                }
            }
            assertNotNull(mode + ": the launches are timed: " + timing.format(), graphEntry);
            assertEquals(mode + ": the first launch and both replays", 3, graphEntry.count());
            assertTrue(mode + ": device time " + graphEntry.totalNanos(), graphEntry.totalNanos() > 0);
            assertEquals(graphEntry.totalNanos(), timing.graphNanos());
            assertTrue(mode + ": the readback around the graph keeps its entry",
                    timing.totalNanos() > timing.graphNanos());
        } finally {
            graph.close();
        }
    }

    @Test
    public void graphLaunchesAreTimedWholePerDispatchMode() {
        launchAndReplay("true");
    }

    @Test
    public void graphLaunchesAreTimedWholeInTheStepMode() {
        launchAndReplay("step");
    }

    @Test
    public void aQueueWithoutTheProfileTimesNothing() {
        ComputeQueue queue = open("false");
        assertNull(queue.kernelTiming());
        assertFalse(queue.timesGraphLaunches());
    }
}
