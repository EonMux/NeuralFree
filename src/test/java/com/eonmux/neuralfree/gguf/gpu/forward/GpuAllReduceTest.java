package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class GpuAllReduceTest {
    private HWAcceleration hw;
    private OclComputeBackend backend;
    private ComputeQueue second;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

    @Before
    public void open() {
        hw = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hw.getCLDevices().isEmpty());
        backend = new OclComputeBackend(hw.getCLDevices().get(0));
        second = backend.createComputeQueue();
        scratch = GpuScratch.builder(backend, 1).slot(ScratchSlots.X, 8).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
    }

    @After
    public void close() {
        if (ops != null) ops.close();
        if (placeholders != null) placeholders.close();
        if (scratch != null) scratch.close();
        if (second != null) second.close();
        if (backend != null) backend.close();
        if (hw != null) hw.shutdown();
    }

    private static void write(ComputeQueue queue, GpuBuffer b, float... values) {
        try (GpuEvent event = queue.enqueueWriteBuffer(b, 0, values.length * 4L, values)) {
            queue.finish();
        }
    }

    private static void writePending(ComputeQueue queue, GpuBuffer b, float... values) {
        try (GpuEvent event = queue.enqueueWriteBuffer(b, 0, values.length * 4L, values)) {
            // The collective waits for each queue before reading its input.
        }
    }

    private static float[] read(ComputeQueue queue, GpuBuffer b, int count) {
        float[] values = new float[count];
        try (GpuEvent event = queue.enqueueReadBuffer(b, 0, count * 4L, values)) {
            queue.finish();
        }
        return values;
    }

    @Test
    public void reducesAndBroadcastsAcrossQueuesWithHostFallback() {
        ComputeQueue root = backend.computeQueue();
        try (GpuBuffer a = backend.allocateBuffer(20); GpuBuffer b = backend.allocateBuffer(20);
             GpuBuffer c = backend.allocateBuffer(20); GpuBuffer temp = backend.allocateBuffer(12)) {
            writePending(root, a, 99, 1, 2, 3, 88);
            writePending(second, b, 77, 4, 5, 6, 66);
            writePending(root, c, 55, -2, -3, -4, 44);
            GpuAllReduce.sum(ops, temp, Arrays.asList(
                    new GpuAllReduce.Participant(root, a, 1),
                    new GpuAllReduce.Participant(second, b, 1),
                    new GpuAllReduce.Participant(root, c, 1)), 3);
            assertArrayEquals(new float[]{99, 3, 4, 5, 88}, read(root, a, 5), 0f);
            assertArrayEquals(new float[]{77, 3, 4, 5, 66}, read(second, b, 5), 0f);
            assertArrayEquals(new float[]{55, 3, 4, 5, 44}, read(root, c, 5), 0f);
            GpuAllReduce.sum(ops, temp, Arrays.asList(
                    new GpuAllReduce.Participant(root, a, 1),
                    new GpuAllReduce.Participant(second, b, 1),
                    new GpuAllReduce.Participant(root, c, 1)), 3);
            assertArrayEquals(new float[]{99, 9, 12, 15, 88}, read(root, a, 5), 0f);
            assertArrayEquals(new float[]{77, 9, 12, 15, 66}, read(second, b, 5), 0f);
            assertArrayEquals(new float[]{55, 9, 12, 15, 44}, read(root, c, 5), 0f);
        }
    }

    @Test
    public void sumsInParticipantOrder() {
        ComputeQueue root = backend.computeQueue();
        try (GpuBuffer a = backend.allocateBuffer(4); GpuBuffer b = backend.allocateBuffer(4);
             GpuBuffer c = backend.allocateBuffer(4); GpuBuffer temp = backend.allocateBuffer(4)) {
            write(root, a, 1e20f);
            write(second, b, -1e20f);
            write(root, c, 3f);
            GpuAllReduce.sum(ops, temp, Arrays.asList(new GpuAllReduce.Participant(root, a, 0),
                    new GpuAllReduce.Participant(second, b, 0), new GpuAllReduce.Participant(root, c, 0)), 1);
            assertArrayEquals(new float[]{3f}, read(root, a, 1), 0f);
            assertArrayEquals(new float[]{3f}, read(second, b, 1), 0f);
            assertArrayEquals(new float[]{3f}, read(root, c, 1), 0f);
        }
    }

    @Test
    public void reducesAcrossChunkBoundary() {
        ComputeQueue root = backend.computeQueue();
        int n = GpuAllReduce.CHUNK_FLOATS + 7;
        try (GpuBuffer a = backend.allocateBuffer((long) n * 4); GpuBuffer b = backend.allocateBuffer((long) n * 4);
             GpuBuffer temp = backend.allocateBuffer((long) GpuAllReduce.CHUNK_FLOATS * 4)) {
            float[] left = new float[n];
            float[] right = new float[n];
            for (int i = 0; i < n; i++) {
                left[i] = i % 17;
                right[i] = -(i % 11);
            }
            write(root, a, left);
            write(second, b, right);
            GpuAllReduce.sum(ops, temp, Arrays.asList(new GpuAllReduce.Participant(root, a, 0),
                    new GpuAllReduce.Participant(second, b, 0)), n);
            for (int i = 0; i < n; i++) left[i] += right[i];
            assertArrayEquals(left, read(root, a, n), 0f);
            assertArrayEquals(left, read(second, b, n), 0f);
        }
    }

    @Test
    public void reduceToRootSumsIntoRankZeroInRankOrderAndLeavesThePeersAlone() {
        ComputeQueue root = backend.computeQueue();
        int n = GpuAllReduce.CHUNK_FLOATS + 5;
        try (GpuBuffer a = backend.allocateBuffer((long) n * 4); GpuBuffer b = backend.allocateBuffer((long) n * 4);
             GpuBuffer c = backend.allocateBuffer((long) n * 4);
             GpuBuffer temp = backend.allocateBuffer((long) GpuAllReduce.CHUNK_FLOATS * 4)) {
            float[] x = new float[n];
            float[] y = new float[n];
            float[] z = new float[n];
            float[] expected = new float[n];
            for (int i = 0; i < n; i++) {
                x[i] = i % 2 == 0 ? 1e20f : i % 13;
                y[i] = i % 2 == 0 ? -1e20f : -(i % 7);
                z[i] = 0.5f * (i % 5);
                expected[i] = (x[i] + y[i]) + z[i];
            }
            writePending(root, a, x);
            writePending(second, b, y);
            writePending(root, c, z);
            GpuAllReduce.reduceToRoot(ops, temp, Arrays.asList(new GpuAllReduce.Participant(root, a, 0),
                    new GpuAllReduce.Participant(second, b, 0), new GpuAllReduce.Participant(root, c, 0)), n);
            assertArrayEquals(expected, read(root, a, n), 0f);
            assertArrayEquals("a peer's view is not overwritten with the sum", y, read(second, b, n), 0f);
            assertArrayEquals(z, read(root, c, n), 0f);
            assertThrows(IllegalArgumentException.class, () -> GpuAllReduce.reduceToRoot(ops, temp,
                    Collections.singletonList(new GpuAllReduce.Participant(second, a, 0)), 1));
        }
    }

    @Test
    public void singleRankAndValidation() {
        ComputeQueue root = backend.computeQueue();
        try (GpuBuffer a = backend.allocateBuffer(16); GpuBuffer temp = backend.allocateBuffer(16)) {
            write(root, a, 1, 2, 3, 4);
            GpuAllReduce.Participant p = new GpuAllReduce.Participant(root, a, 0);
            GpuAllReduce.sum(ops, temp, Collections.singletonList(p), 4);
            assertArrayEquals(new float[]{1, 2, 3, 4}, read(root, a, 4), 0f);
            assertThrows(IllegalArgumentException.class, () -> GpuAllReduce.sum(ops, temp, Collections.emptyList(), 1));
            assertThrows(IllegalArgumentException.class, () -> GpuAllReduce.sum(ops, temp, Arrays.asList(p, p), 1));
            assertThrows(IllegalArgumentException.class, () -> new GpuAllReduce.Participant(root, a, -1));
            assertThrows(IllegalArgumentException.class, () -> GpuAllReduce.sum(ops, temp,
                    Collections.singletonList(new GpuAllReduce.Participant(second, a, 0)), 1));
            assertThrows(IllegalArgumentException.class, () -> GpuAllReduce.sum(ops, a, Collections.singletonList(p), 1));
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> GpuAllReduce.sum(ops, temp, Collections.singletonList(p), 0)).getMessage().contains("count"));
            // a view longer than its buffer is refused; both buffers may be padded by the backend
            int over = (int) (a.sizeInBytes() / Float.BYTES) + 1;
            try (GpuBuffer bigTemp = backend.allocateBuffer((long) over * Float.BYTES)) {
                assertThrows(IllegalArgumentException.class,
                        () -> GpuAllReduce.sum(ops, bigTemp, Collections.singletonList(p), over));
            }
        }
    }
}
