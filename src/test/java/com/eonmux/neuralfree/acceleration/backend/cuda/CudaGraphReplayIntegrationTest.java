package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphCache;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchPlan;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelTaskBatch;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.EnumSet;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Graph capture and replay and kernel batching of {@link CudaComputeQueue} on the device: the
 * captured commands do not run before the graph is launched, a replay recomputes from the
 * buffers' current contents, the cache stores one graph per shape, and a kernel batch matches
 * its tasks' sequential launches bit for bit.
 */
public class CudaGraphReplayIntegrationTest {

    private static final int N = 256;

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

    private static final String ADD_PTX =
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

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private ComputeQueue queue;
    private KernelHandle scale;
    private KernelHandle add;

    @Before
    public void setUp() {
        try {
            hwAccel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
            if (!hwAccel.getCUDevices().isEmpty()) {
                backend = new CudaComputeBackend(hwAccel.getCUDevices().get(0));
                queue = backend.computeQueue();
                scale = backend.compileKernel("scale", SCALE_PTX, null);
                add = backend.compileKernel("simple_add", ADD_PTX, null);
            }
        } catch (Throwable e) {
            System.out.println("CUDA GPU initialization failed: " + e.getMessage());
        }
        Assume.assumeTrue("no CUDA GPU", backend != null);
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    private GpuBuffer buffer(float[] host) {
        GpuBuffer b = backend.allocateBuffer((long) host.length * Float.BYTES);
        GpuEvent w = queue.enqueueWriteBuffer(b, 0, (long) host.length * Float.BYTES, host);
        if (w != null) {
            w.close();
        }
        return b;
    }

    private float[] readBack(GpuBuffer b) {
        return readBack(queue, b);
    }

    private static float[] readBack(ComputeQueue q, GpuBuffer b) {
        float[] out = new float[N];
        GpuEvent r = q.enqueueReadBuffer(b, 0, (long) N * Float.BYTES, out);
        if (r != null) {
            r.close();
        }
        q.finish();
        return out;
    }

    private static float[] iota(float first, float step) {
        float[] v = new float[N];
        for (int i = 0; i < N; i++) {
            v[i] = first + i * step;
        }
        return v;
    }

    private void launchScale(GpuBuffer b, float factor) {
        queue.enqueueKernelNoEvent(scale, new long[]{N}, new long[]{N}, new Object[]{b, factor, N});
    }

    private void launchAdd(GpuBuffer a, GpuBuffer b, GpuBuffer c) {
        queue.enqueueKernelNoEvent(add, new long[]{N}, new long[]{N}, new Object[]{a, b, c, N});
    }

    @Test
    public void capturedCommandsRunOnlyAtLaunchAndReplayRecomputes() {
        float[] a0 = iota(1, 1);
        float[] c0 = iota(100, 2);
        GpuBuffer a = buffer(a0);
        GpuBuffer b = buffer(new float[N]);
        GpuBuffer c = buffer(c0);
        GpuBuffer d = buffer(new float[N]);
        try {
            queue.beginCapture();
            assertTrue(queue.isCapturing());
            launchScale(a, 0.5f);
            launchAdd(a, c, d);
            GpuGraph graph = queue.endCapture();
            assertFalse(queue.isCapturing());
            try {
                assertArrayEquals("captured commands must not run before the launch", new float[N], readBack(d),
                        0.0f);
                graph.launch(queue);
                float[] expected = new float[N];
                for (int i = 0; i < N; i++) {
                    expected[i] = a0[i] * 0.5f + c0[i];
                }
                assertArrayEquals(expected, readBack(d), 0.0f);

                float[] a1 = iota(10, 3);
                GpuEvent w = queue.enqueueWriteBuffer(a, 0, (long) N * Float.BYTES, a1);
                if (w != null) {
                    w.close();
                }
                graph.launch(queue);
                for (int i = 0; i < N; i++) {
                    expected[i] = a1[i] * 0.5f + c0[i];
                }
                assertArrayEquals("a replay reads the buffers' current contents", expected, readBack(d), 0.0f);
            } finally {
                graph.close();
            }
        } finally {
            a.close();
            b.close();
            c.close();
            d.close();
        }
    }

    @Test
    public void theCacheStoresOneGraphPerShape() {
        GpuGraphCache cache = new GpuGraphCache(4);
        try {
            float[] a0 = iota(1, 1);
            GpuBuffer a = buffer(a0);
            GpuBuffer b = buffer(new float[N]);
            try {
                GpuGraph small = cache.getOrCapture("step=1", GpuGraphCache.captured(queue,
                        () -> launchScale(a, 0.5f)));
                GpuGraph large = cache.getOrCapture("step=2", GpuGraphCache.captured(queue,
                        () -> launchScale(a, 2.0f)));
                GpuGraph smallAgain = cache.getOrCapture("step=1", GpuGraphCache.captured(queue,
                        () -> {
                            throw new AssertionError("a cached shape must not capture again");
                        }));
                assertSame(small, smallAgain);
                assertNotSame(small, large);
                assertEquals(2, cache.size());

                small.launch(queue);
                assertArrayEquals(scaleOf(a0, 0.5f), readBack(a), 0.0f);
                large.launch(queue);
                assertArrayEquals(scaleOf(a0, 1.0f), readBack(a), 0.0f);
                small.launch(queue);
                assertArrayEquals(scaleOf(a0, 0.5f), readBack(a), 0.0f);
            } finally {
                a.close();
                b.close();
            }
        } finally {
            cache.close();
        }
    }

    private static float[] scaleOf(float[] input, float factor) {
        float[] out = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            out[i] = input[i] * factor;
        }
        return out;
    }

    @Test
    public void aKernelBatchMatchesItsTasksSequentialLaunches() {
        float[] x0 = iota(1, 0.5f);
        GpuBuffer x = buffer(x0);
        GpuBuffer y = buffer(iota(50, -1));
        GpuBuffer z = buffer(iota(-10, 0.25f));
        try {
            KernelTaskBatch batch = queue.beginKernelBatch();
            try {
                assertTrue(queue.supportsKernelBatching());
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 0.5f, N});
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{y, 2.0f, N});
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{z, 0.25f, N});
                assertEquals(3, batch.taskCount());
                GpuEvent done = batch.launch(queue);
                if (done != null) {
                    done.close();
                }
                assertEquals(1, batch.launches());
                assertEquals(3, batch.tasksLaunched());
                assertEquals(0, batch.taskCount());
                assertEquals(0, batch.graphReuses());
                assertArrayEquals(scaleOf(x0, 0.5f), readBack(x), 0.0f);
                assertArrayEquals(scaleOf(iota(50, -1), 2.0f), readBack(y), 0.0f);
                assertArrayEquals(scaleOf(iota(-10, 0.25f), 0.25f), readBack(z), 0.0f);

                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 0.5f, N});
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{y, 2.0f, N});
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{z, 0.25f, N});
                GpuEvent again = batch.launch(queue);
                if (again != null) {
                    again.close();
                }
                assertEquals(2, batch.launches());
                assertEquals(6, batch.tasksLaunched());
                assertEquals("the same shape reuses the previous launch's graph", 1, batch.graphReuses());
                assertArrayEquals(scaleOf(x0, 0.25f), readBack(x), 0.0f);
            } finally {
                batch.close();
            }
        } finally {
            x.close();
            y.close();
            z.close();
        }
    }

    @Test
    public void aKernelBatchKeepsInOrderSemanticsBetweenTasks() {
        float[] x0 = iota(4, 4);
        GpuBuffer x = buffer(x0);
        try {
            KernelTaskBatch batch = queue.beginKernelBatch();
            try {
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 0.5f, N});
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 0.5f, N});
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 4.0f, N});
                GpuEvent done = batch.launch(queue);
                if (done != null) {
                    done.close();
                }
                assertArrayEquals("later tasks must see earlier tasks' results", scaleOf(x0, 1.0f), readBack(x),
                        0.0f);
            } finally {
                batch.close();
            }
        } finally {
            x.close();
        }
    }

    @Test
    public void aShapeChangeRebuildsTheBatchGraph() {
        float[] x0 = iota(2, 2);
        GpuBuffer x = buffer(x0);
        GpuBuffer y = buffer(iota(1, 1));
        try {
            KernelTaskBatch batch = queue.beginKernelBatch();
            try {
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 2.0f, N});
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{y, 3.0f, N});
                GpuEvent two = batch.launch(queue);
                if (two != null) {
                    two.close();
                }
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 2.0f, N});
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{y, 3.0f, N});
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 0.5f, N});
                GpuEvent three = batch.launch(queue);
                if (three != null) {
                    three.close();
                }
                assertEquals(2, batch.launches());
                assertEquals(5, batch.tasksLaunched());
                assertEquals("a changed task count rebuilds, not reuses", 0, batch.graphReuses());
                float[] xAfterThree = scaleOf(scaleOf(scaleOf(x0, 2.0f), 2.0f), 0.5f);
                assertArrayEquals(xAfterThree, readBack(x), 0.0f);
            } finally {
                batch.close();
            }
        } finally {
            x.close();
            y.close();
        }
    }

    @Test
    public void aGraphLaunchCanWaitOnAnotherQueuesEvent() {
        float[] a0 = iota(3, 1);
        float[] c0 = iota(1, 1);
        GpuBuffer a = backend.allocateBuffer((long) N * Float.BYTES);
        GpuBuffer b = buffer(new float[N]);
        GpuBuffer c = buffer(c0);
        try {
            GpuEvent written = backend.transferQueue().enqueueWriteBuffer(a, 0, (long) N * Float.BYTES, a0);
            queue.beginCapture();
            launchAdd(a, c, b);
            GpuGraph graph = queue.endCapture();
            try {
                GpuEvent launched = graph.launch(queue, written);
                if (launched != null) {
                    launched.close();
                }
                if (written != null) {
                    written.close();
                }
                float[] expected = new float[N];
                for (int i = 0; i < N; i++) {
                    expected[i] = a0[i] + c0[i];
                }
                assertArrayEquals(expected, readBack(b), 0.0f);
            } finally {
                graph.close();
            }
        } finally {
            a.close();
            b.close();
            c.close();
        }
    }

    @Test
    public void aKernelBatchLaunchCanWaitOnAnotherQueuesEvent() {
        float[] x0 = iota(1, 2);
        GpuBuffer x = backend.allocateBuffer((long) N * Float.BYTES);
        try {
            GpuEvent written = backend.transferQueue().enqueueWriteBuffer(x, 0, (long) N * Float.BYTES, x0);
            KernelTaskBatch batch = queue.beginKernelBatch();
            try {
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 2.0f, N});
                GpuEvent done = batch.launch(queue, written);
                if (done != null) {
                    done.close();
                }
                if (written != null) {
                    written.close();
                }
                assertArrayEquals(scaleOf(x0, 2.0f), readBack(x), 0.0f);
            } finally {
                batch.close();
            }
        } finally {
            x.close();
        }
    }

    @Test
    public void captureRejectsCommandsItCannotRecord() {
        float[] a0 = iota(1, 1);
        GpuBuffer a = buffer(a0);
        GpuBuffer d = buffer(new float[N]);
        try {
            queue.beginCapture();
            try {
                try {
                    queue.enqueueWriteBuffer(a, 0, (long) N * Float.BYTES, a0);
                    fail("a write cannot be captured");
                } catch (IllegalStateException e) {
                    assertTrue(e.getMessage().contains("cannot be captured"));
                }
                try {
                    readBack(d);
                    fail("a read cannot be captured");
                } catch (IllegalStateException e) {
                    assertTrue(e.getMessage().contains("cannot be captured"));
                }
                try {
                    queue.finish();
                    fail("a synchronization cannot be captured");
                } catch (IllegalStateException e) {
                    assertTrue(e.getMessage().contains("cannot be captured"));
                }
                try {
                    queue.enqueueMarker();
                    fail("a marker cannot be captured");
                } catch (IllegalStateException e) {
                    assertTrue(e.getMessage().contains("cannot be captured"));
                }
                GpuEvent other = backend.transferQueue().enqueueMarker();
                try {
                    try {
                        queue.enqueueKernelNoEvent(scale, new long[]{N}, new long[]{N},
                                new Object[]{a, 0.5f, N}, other);
                        fail("a cross-queue wait cannot be captured");
                    } catch (IllegalStateException e) {
                        assertTrue(e.getMessage().contains("another queue"));
                    }
                    try {
                        queue.enqueueCopyBuffer(a, 0, d, 0, (long) N * Float.BYTES, other);
                        fail("a cross-queue wait on a captured copy cannot be captured");
                    } catch (IllegalStateException e) {
                        assertTrue(e.getMessage().contains("another queue"));
                    }
                    try {
                        queue.enqueueFillBuffer(d, 0, (long) N * Float.BYTES, 0, other);
                        fail("a cross-queue wait on a captured fill cannot be captured");
                    } catch (IllegalStateException e) {
                        assertTrue(e.getMessage().contains("another queue"));
                    }
                } finally {
                    if (other != null) {
                        other.close();
                    }
                }
            } finally {
                queue.abortCapture();
            }
            assertFalse(queue.isCapturing());
            launchScale(a, 2.0f);
            assertArrayEquals(scaleOf(a0, 2.0f), readBack(a), 0.0f);
        } finally {
            a.close();
            d.close();
        }
    }

    @Test
    public void capturedDeviceCopyAndFillReplay() {
        float[] src0 = iota(1, 1);
        GpuBuffer src = buffer(src0);
        GpuBuffer dst = buffer(new float[N]);
        long half = (long) (N / 2) * Float.BYTES;
        try {
            queue.beginCapture();
            queue.enqueueCopyBuffer(src, 0, dst, 0, (long) N * Float.BYTES);
            queue.enqueueFillBuffer(dst, 0, half, Float.floatToIntBits(9.0f));
            GpuGraph graph = queue.endCapture();
            try {
                float[] expected = new float[N];
                for (int i = 0; i < N; i++) {
                    expected[i] = i < N / 2 ? 9.0f : src0[i];
                }
                assertArrayEquals("a captured copy and fill run only at the launch", new float[N], readBack(dst),
                        0.0f);
                graph.launch(queue);
                assertArrayEquals(expected, readBack(dst), 0.0f);

                float[] src1 = iota(500, -1);
                GpuEvent w = queue.enqueueWriteBuffer(src, 0, (long) N * Float.BYTES, src1);
                if (w != null) {
                    w.close();
                }
                graph.launch(queue);
                for (int i = 0; i < N; i++) {
                    expected[i] = i < N / 2 ? 9.0f : src1[i];
                }
                assertArrayEquals("a replay re-copies the source's current contents", expected, readBack(dst), 0.0f);
            } finally {
                graph.close();
            }
        } finally {
            src.close();
            dst.close();
        }
    }

    @Test
    public void aGraphRunsOnAnotherQueueOfTheBackend() {
        float[] x0 = iota(2, 3);
        GpuBuffer x = buffer(x0);
        ComputeQueue other = backend.createComputeQueue();
        try {
            queue.beginCapture();
            launchScale(x, 2.0f);
            GpuGraph graph = queue.endCapture();
            try {
                graph.launch(other);
                assertArrayEquals(scaleOf(x0, 2.0f), readBack(other, x), 0.0f);
            } finally {
                graph.close();
            }
        } finally {
            other.close();
            x.close();
        }
    }

    @Test
    public void aQueueClosedWhileCapturingDiscardsItsCapture() {
        float[] x0 = iota(3, 2);
        GpuBuffer x = buffer(x0);
        ComputeQueue other = backend.createComputeQueue();
        try {
            other.beginCapture();
            other.enqueueKernelNoEvent(scale, new long[]{N}, new long[]{N}, new Object[]{x, 5.0f, N});
            other.close();
            assertArrayEquals("a discarded capture never runs", x0, readBack(x), 0.0f);
            launchScale(x, 2.0f);
            assertArrayEquals(scaleOf(x0, 2.0f), readBack(x), 0.0f);
        } finally {
            x.close();
        }
    }

    @Test
    public void aKernelBatchWithDifferentKernelsRebuilds() {
        float[] x0 = iota(1, 1);
        float[] c0 = iota(100, 1);
        GpuBuffer x = buffer(x0);
        GpuBuffer y = buffer(new float[N]);
        GpuBuffer c = buffer(c0);
        try {
            KernelTaskBatch batch = queue.beginKernelBatch();
            try {
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 2.0f, N});
                batch.add(add, new long[]{N}, new long[]{N}, new Object[]{x, c, y, N});
                GpuEvent first = batch.launch(queue);
                if (first != null) {
                    first.close();
                }
                batch.add(add, new long[]{N}, new long[]{N}, new Object[]{x, c, y, N});
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 2.0f, N});
                GpuEvent second = batch.launch(queue);
                if (second != null) {
                    second.close();
                }
                assertEquals(2, batch.launches());
                assertEquals("a different kernel order rebuilds, not reuses", 0, batch.graphReuses());
                float[] expectedX = scaleOf(x0, 4.0f);
                float[] expectedY = new float[N];
                for (int i = 0; i < N; i++) {
                    expectedY[i] = x0[i] * 2.0f + c0[i];
                }
                assertArrayEquals(expectedX, readBack(x), 0.0f);
                assertArrayEquals(expectedY, readBack(y), 0.0f);
            } finally {
                batch.close();
            }
        } finally {
            x.close();
            y.close();
            c.close();
        }
    }

    @Test
    public void kernelBatchUsageIsEnforced() {
        GpuBuffer x = buffer(iota(1, 1));
        try {
            KernelTaskBatch batch = queue.beginKernelBatch();
            try {
                try {
                    batch.launch(queue);
                    fail("an empty batch must fail");
                } catch (IllegalArgumentException e) {
                    assertTrue(e.getMessage().contains("without tasks"));
                }
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 0.5f, N});
                batch.clear();
                assertEquals(0, batch.taskCount());
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 0.5f, N});
                GpuEvent done = batch.launch(queue);
                if (done != null) {
                    done.close();
                }
                assertEquals(1, batch.launches());
            } finally {
                batch.close();
            }
            try {
                batch.add(scale, new long[]{N}, new long[]{N}, new Object[]{x, 0.5f, N});
                fail("adding to a closed batch must fail");
            } catch (IllegalStateException e) {
                assertTrue(e.getMessage().contains("closed"));
            }
            try {
                batch.launch(queue);
                fail("launching a closed batch must fail");
            } catch (IllegalStateException e) {
                assertTrue(e.getMessage().contains("closed"));
            }
        } finally {
            x.close();
        }
    }

    /**
     * A prepared follow-up step lands in the idle executable graph and the next patched replay
     * submits it: the step's factor reaches the device although its patch ran while the previous
     * launch was in flight, and a declined prepare leaves the graph the last launch made, so the
     * follow-up's own patch still drives a correct replay.
     */
    @Test
    public void aPreparedFollowupPatchesTheIdleGraphAndTheNextReplaySubmitsIt() {
        GpuBuffer x = buffer(iota(1, 1));
        try {
            queue.beginCapture();
            launchScale(x, 1.0f);
            GpuGraph graph = queue.endCapture();
            try {
                assertTrue(graph.supportsPatchedReplay());
                /* The step's patch: the kernel's factor argument is 2^pos. */
                GpuGraphPatchPlan plan = (pos, sink) -> {
                    if (pos < 0) {
                        return false;
                    }
                    sink.patchKernel(0, null, new int[]{1}, new Object[]{(float) Math.pow(2, pos)});
                    return true;
                };
                assertTrue(graph.replayPatched(plan, 0));
                assertArrayEquals(scaleOf(iota(1, 1), 1.0f), readBack(x), 0.0f);

                assertTrue("the follow-up step's graph is prepared while the last one runs",
                        graph.preparePatched(plan, 1, 0));
                assertTrue(graph.replayPatched(plan, 1));
                assertArrayEquals("the replay submits the prepared factor", scaleOf(iota(1, 1), 2.0f),
                        readBack(x), 0.0f);

                assertTrue(graph.preparePatched(plan, 2, 1));
                assertTrue(graph.replayPatched(plan, 2));
                assertArrayEquals("the launches accumulate: 1 x 2 x 4", scaleOf(iota(1, 1), 8.0f),
                        readBack(x), 0.0f);

                /* A follow-up the plan declines: the prepare restores the launched state and the
                 * follow-up's own replay patches for its real step. */
                GpuGraphPatchPlan declining = (pos, sink) -> {
                    if (pos != 3) {
                        return false;
                    }
                    sink.patchKernel(0, null, new int[]{1}, new Object[]{(float) Math.pow(2, pos)});
                    return true;
                };
                assertFalse(graph.preparePatched(declining, 4, 3));
                assertTrue(graph.replayPatched(plan, 3));
                assertArrayEquals("1 x 2 x 4 x 8", scaleOf(iota(1, 1), 64.0f), readBack(x), 0.0f);
            } finally {
                graph.close();
            }
        } finally {
            x.close();
        }
    }

    /**
     * A step value that changes once and then holds reaches both executable graphs: the patch that
     * changed it wrote one instantiation, so the next prepare of the other one must still rewrite the
     * node although the value equals the node's record. The same holds after a body replay, which
     * writes the launched instantiation only, hands the graph to the patched replay. And a patched value
     * never leaks into a direct launch: one that sets the kernel's argument back to its value before the
     * patch launches that value.
     */
    @Test
    public void aValueThatChangesOnceReachesBothExecutableGraphs() {
        GpuBuffer x = buffer(iota(1, 1));
        try {
            queue.beginCapture();
            launchScale(x, 1.0f);
            GpuGraph graph = queue.endCapture();
            try {
                /* The factor is 1 at step 0 and 2 from step 1 on. */
                GpuGraphPatchPlan plan = (pos, sink) -> {
                    sink.patchKernel(0, null, new int[]{1}, new Object[]{pos == 0 ? 1.0f : 2.0f});
                    return true;
                };
                assertTrue(graph.replayPatched(plan, 0));
                assertTrue(graph.preparePatched(plan, 1, 0));
                assertTrue(graph.replayPatched(plan, 1));
                assertTrue(graph.preparePatched(plan, 2, 1));
                assertTrue(graph.replayPatched(plan, 2));
                assertArrayEquals("1 x 1 x 2 x 2", scaleOf(iota(1, 1), 4.0f), readBack(x), 0.0f);

                /* A body replay moves the factor to 0.5 on the launched graph; the patched steps that
                 * follow keep it on both. */
                graph.replay(() -> launchScale(x, 0.5f));
                GpuGraphPatchPlan half = (pos, sink) -> {
                    sink.patchKernel(0, null, new int[]{1}, new Object[]{0.5f});
                    return true;
                };
                assertTrue(graph.replayPatched(half, 3));
                assertTrue(graph.preparePatched(half, 4, 3));
                assertTrue(graph.replayPatched(half, 4));
                assertTrue(graph.preparePatched(half, 5, 4));
                assertTrue(graph.replayPatched(half, 5));
                assertArrayEquals("4 x 0.5^4", scaleOf(iota(1, 1), 0.25f), readBack(x), 0.0f);

                /* A direct launch at 1, the patched step at 2, the direct launch at 1 again. */
                launchScale(x, 1.0f);
                assertTrue(graph.replayPatched(plan, 6));
                launchScale(x, 1.0f);
                assertArrayEquals("0.25 x 1 x 2 x 1", scaleOf(iota(1, 1), 0.5f), readBack(x), 0.0f);
            } finally {
                graph.close();
            }
        } finally {
            x.close();
        }
    }

    /**
     * After patched replays, a launch of the same kernel outside the graph and a body replay run the
     * arguments they set, also when those equal the ones the kernel had before the patches: a patch
     * sets its values as the handle's arguments, so setting the earlier value again re-binds it.
     */
    @Test
    public void launchesAfterPatchedReplaysRunTheirOwnArguments() {
        GpuBuffer x = buffer(iota(1, 1));
        try {
            queue.beginCapture();
            launchScale(x, 1.0f);
            GpuGraph graph = queue.endCapture();
            try {
                GpuGraphPatchPlan doubling = (pos, sink) -> {
                    sink.patchKernel(0, null, new int[]{1}, new Object[]{2.0f});
                    return true;
                };
                assertTrue(graph.replayPatched(doubling, 0));
                assertArrayEquals(scaleOf(iota(1, 1), 2.0f), readBack(x), 0.0f);

                launchScale(x, 1.0f);
                assertArrayEquals("the launch outside the graph runs its own factor", scaleOf(iota(1, 1), 2.0f),
                        readBack(x), 0.0f);

                graph.replay(() -> launchScale(x, 1.0f));
                assertArrayEquals("the body replay runs its own factor", scaleOf(iota(1, 1), 2.0f),
                        readBack(x), 0.0f);
            } finally {
                graph.close();
            }
        } finally {
            x.close();
        }
    }
}
