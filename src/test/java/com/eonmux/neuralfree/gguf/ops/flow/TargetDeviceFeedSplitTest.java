package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.pipeline.SplitSpecs;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.flow.TargetDeviceFeed.DeviceBatch;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Test;

import java.lang.reflect.Proxy;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The device feed of a NextN drafter over a layer split: the tap rows are captured on the last trunk layer's device
 * and reach the head's device. When both are the same queue they copy device-to-device; when they differ and the
 * destination has no peer copy the feed stages them through the host.
 */
public class TargetDeviceFeedSplitTest {

    private static final int UBATCH = 4;
    private static final int N_CTX = 32;
    private static final int WIDTH = SplitSpecs.N_EMBD;
    private static final int ROWS = 3;
    private static final int[] TOKENS = {3, 7, 11};

    private final Map<String, GpuWeight> weights = new HashMap<>();
    private FakeComputeBackend device;
    private FakeComputeBackend sourceDevice;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private LlmKernels kernels;
    private DeviceModelMemory memory;
    private GpuForwardContext context;
    private MtpHead head;
    private TargetDeviceFeed feed;

    @After
    public void tearDown() {
        close(head);
        close(ops);
        close(placeholders);
        close(scratch);
        close(memory);
        if (device != null) {
            device.close();
        }
        if (sourceDevice != null) {
            sourceDevice.close();
        }
    }

    private static void close(AutoCloseable c) {
        try {
            if (c != null) {
                c.close();
            }
        } catch (Exception ignored) {
            // teardown only
        }
    }

    /** A feed whose capture queue is another device: the rows are staged through the host into the head's slots. */
    @Test
    public void rowsCapturedOnAnotherDeviceReachTheHeadThroughHostStaging() {
        open();
        float[] rows = rows(1);
        GpuBuffer source = sourceRows(rows);
        int point = point();
        feed = new TargetDeviceFeed(head, memory, point, WIDTH, sourceDevice.computeQueue());
        assertTrue(feed.wantsDevice(point, sourceDevice.computeQueue()));
        assertFalse("the head's own queue captures nothing", feed.wantsDevice(point, device.computeQueue()));

        memory.sequence().append(TOKENS, 0, ROWS);
        feed.captureDevice(point, source, 0, ROWS, WIDTH);
        feed.onHiddenDevice(point, ROWS, WIDTH);

        DeviceBatch batch = only(feed.drain());
        assertEquals(0, batch.pos0);
        assertArrayEquals(TOKENS, batch.tokens);
        assertArrayEquals(bits(rows), bits(read(batch)));
    }

    /** A feed whose capture queue is the head's own queue: no host round trip, one device-to-device copy. */
    @Test
    public void rowsCapturedOnTheHeadsDeviceCopyOnTheDevice() {
        open();
        float[] rows = rows(2);
        GpuBuffer source = deviceRows(rows);
        int point = point();
        feed = new TargetDeviceFeed(head, memory, point, WIDTH, device.computeQueue());

        memory.sequence().append(TOKENS, 0, ROWS);
        feed.captureDevice(point, source, 0, ROWS, WIDTH);
        feed.onHiddenDevice(point, ROWS, WIDTH);

        DeviceBatch batch = only(feed.drain());
        assertArrayEquals(bits(rows), bits(read(batch)));
    }

    /**
     * A feed whose capture device can reach the head's device: the source queue enqueues a peer copy (here simulated
     * on the fake backends) instead of staging through the host.
     */
    @Test
    public void rowsCapturedOnAPeerDeviceCopyDirectly() {
        open();
        float[] rows = rows(3);
        GpuBuffer source = sourceRows(rows);
        int point = point();
        boolean[] copied = {false};
        feed = new TargetDeviceFeed(head, memory, point, WIDTH, peerQueue(copied));

        memory.sequence().append(TOKENS, 0, ROWS);
        feed.captureDevice(point, source, 0, ROWS, WIDTH);
        feed.onHiddenDevice(point, ROWS, WIDTH);

        assertTrue("the source queue performed a peer copy", copied[0]);
        DeviceBatch batch = only(feed.drain());
        assertArrayEquals(bits(rows), bits(read(batch)));
    }

    private static DeviceBatch only(List<DeviceBatch> batches) {
        assertEquals(1, batches.size());
        return batches.get(0);
    }

    private float[] read(DeviceBatch batch) {
        float[] out = new float[ROWS * WIDTH];
        int off = 0;
        for (TargetDeviceFeed.Chunk chunk : batch.chunks) {
            float[] part = new float[chunk.nRows * WIDTH];
            head.read(chunk.rows, chunk.nRows, part);
            System.arraycopy(part, 0, out, off, part.length);
            off += part.length;
        }
        return out;
    }

    private int point() {
        return context.spec().nLayer();
    }

    private void open() {
        ModelSpec spec = SplitSpecs.trunk(1, com.eonmux.neuralfree.gguf.arch.MemoryKind.KV);
        device = FakeComputeBackend.builder().strictLlmArgs(true).build();
        sourceDevice = FakeComputeBackend.builder().strictLlmArgs(true).build();
        kernels = LlmKernels.open(device);
        scratch = GpuScratch.forModel(device, spec, UBATCH);
        placeholders = GpuPlaceholders.allocate(device, device.computeQueue());
        ops = new GpuOps(device.computeQueue(), kernels, scratch, placeholders);
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).nCtx(N_CTX)
                .ubatch(UBATCH).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
        memory = DeviceModelMemory.allocate(device, device.computeQueue(), plan,
                spec.hparams().mropeTextPositions(), null);
        context = GpuForwardContext.builder(kernels, ops, weights::get, spec).memory(memory).build();
        head = MtpHead.gpu(context, new GpuBlockRegistry(Collections.emptyList(), Collections.emptyList(),
                Collections.emptyList()), null);
    }

    /** Rows of {@code seed}-dependent values written to a buffer of the head's device. */
    private GpuBuffer deviceRows(float[] rows) {
        GpuBuffer buffer = device.allocateBuffer((long) rows.length * Float.BYTES);
        device.computeQueue().enqueueWriteBuffer(buffer, 0, (long) rows.length * Float.BYTES, rows);
        return buffer;
    }

    /** The same rows written to a buffer of the other device. */
    private GpuBuffer sourceRows(float[] rows) {
        GpuBuffer buffer = sourceDevice.allocateBuffer((long) rows.length * Float.BYTES);
        sourceDevice.computeQueue().enqueueWriteBuffer(buffer, 0, (long) rows.length * Float.BYTES, rows);
        return buffer;
    }

    /**
     * The other device's queue, which reports peer access to the head's device and copies through the host when the
     * feed asks for a peer copy (the fake backends share host memory, so the copy lands in the head's buffer).
     */
    private ComputeQueue peerQueue(boolean[] copied) {
        ComputeQueue base = sourceDevice.computeQueue();
        return (ComputeQueue) Proxy.newProxyInstance(ComputeQueue.class.getClassLoader(),
                new Class<?>[]{ComputeQueue.class}, (proxy, method, args) -> {
                    switch (method.getName()) {
                        case "supportsPeerCopy":
                            return args[0] == device.computeQueue();
                        case "enqueuePeerCopy":
                            copied[0] = true;
                            ComputeQueue destination = (ComputeQueue) args[0];
                            GpuBuffer src = (GpuBuffer) args[1];
                            long srcOffset = (Long) args[2];
                            GpuBuffer dst = (GpuBuffer) args[3];
                            long dstOffset = (Long) args[4];
                            long size = (Long) args[5];
                            byte[] bytes = new byte[(int) size];
                            base.enqueueReadBuffer(src, srcOffset, size, bytes);
                            destination.enqueueWriteBuffer(dst, dstOffset, size, bytes);
                            return null;
                        default:
                            try {
                                return method.invoke(base, args);
                            } catch (java.lang.reflect.InvocationTargetException e) {
                                throw e.getCause();
                            }
                    }
                });
    }

    private static float[] rows(int seed) {
        float[] out = new float[ROWS * WIDTH];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((i * 31 + seed * 17) % 101) * 0.125f - 6.0f;
        }
        return out;
    }

    private static int[] bits(float[] values) {
        int[] out = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = Float.floatToRawIntBits(values[i]);
        }
        return out;
    }
}
