package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@code LLM_MoE_Bucketize} + {@code LLM_MoE_AlignBlocks} + {@code F32_GroupedGemm} on an OpenCL device against
 * the CPU twins: the device align layout must equal {@link MoeCombine#alignBlocks}, and the grouped GEMM outputs
 * the per-pair products (0 for out-of-range pairs) of the direct reference. Skips without an OpenCL GPU.
 */
public class MoeGroupedGemmOpenClTest {

    private static final int N_EXPERT = 8;
    private static final int N_FF = 48;
    private static final int N_EMBD = 64;
    private static final int K = 2;
    private static final int N_TOKENS = 12;
    private static final int N_ROWS = N_TOKENS * K;
    private static final int TILES = MoeKernels.groupedTiles(N_ROWS, N_EXPERT);
    private static final int LOCAL = 64;

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;
    private final List<GpuBuffer> buffers = new ArrayList<>();
    private final List<GpuEvent> events = new ArrayList<>();

    @Before
    public void setUp() {
        try {
            hwAccel = new HWAcceleration();
            if (!hwAccel.getCLDevices().isEmpty()) {
                CLDevice device = hwAccel.getCLDevices().get(0);
                ComputeBackend opened = new OclComputeBackend(device);
                try {
                    kernels = LlmKernels.open(opened);
                    queue = opened.computeQueue();
                    backend = opened;
                } catch (RuntimeException e) {
                    // Close the backend before the enumeration: it was built directly on the enumerated
                    // device, so the shutdown in tearDown does not track it.
                    opened.close();
                    throw e;
                }
            }
        } catch (Throwable e) {
            backend = null;
        }
        Assume.assumeTrue("no OpenCL GPU", backend != null);
    }

    @After
    public void tearDown() {
        for (GpuBuffer b : buffers) {
            try {
                b.close();
            } catch (RuntimeException ignored) {
            }
        }
        for (GpuEvent e : events) {
            try {
                e.close();
            } catch (RuntimeException ignored) {
            }
        }
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            try {
                hwAccel.shutdown();
            } catch (Throwable ignored) {
            }
        }
    }

    private GpuBuffer buffer(long bytes, boolean readOnly) {
        GpuBuffer b = backend.allocateBuffer(bytes, readOnly);
        buffers.add(b);
        return b;
    }

    private GpuBuffer writeInts(int[] values) {
        GpuBuffer b = buffer(4L * values.length, true);
        ByteBuffer bb = ByteBuffer.allocate(4 * values.length).order(ByteOrder.LITTLE_ENDIAN);
        bb.asIntBuffer().put(values);
        events.add(queue.enqueueWriteBuffer(b, 0, bb.capacity(), bb.array()));
        return b;
    }

    private GpuBuffer writeFloats(float[] values, boolean readOnly) {
        GpuBuffer b = buffer(4L * values.length, readOnly);
        events.add(queue.enqueueWriteBuffer(b, 0, 4L * values.length, values));
        return b;
    }

    private int[] readInts(GpuBuffer b, int count) {
        byte[] raw = new byte[4 * count];
        GpuEvent e = queue.enqueueReadBuffer(b, 0, raw.length, raw);
        queue.finish();
        if (e != null) {
            e.close();
        }
        int[] out = new int[count];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(out);
        return out;
    }

    private void dispatch(String kernel, long[] global, long[] local, Object... args) {
        GpuEvent e = queue.enqueueKernel(kernels.get(kernel), global, local, args);
        if (e != null) {
            e.close();
        }
    }

    private static int[] ids() {
        int[] ids = new int[N_ROWS];
        for (int p = 0; p < N_ROWS; p++) {
            ids[p] = (p * 7) % 11 - 1;
        }
        return ids;
    }

    private static float[] values(int n, long seed) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) (Math.sin(seed * 0.017 + i * 0.29) * 2.0);
        }
        return v;
    }

    @Test
    public void deviceAlignAndGroupedGemmMatchTheCpuTwins() {
        int[] ids = ids();
        float[] weights = values(N_EXPERT * N_FF * N_EMBD, 3);
        float[] xPair = values(N_ROWS * N_EMBD, 5);
        float[] xToken = values(N_TOKENS * N_EMBD, 9);
        float[] scales = values(N_EXPERT, 7);

        GpuBuffer idsBuf = writeInts(ids);
        GpuBuffer weightBuf = writeFloats(weights, true);
        GpuBuffer xPairBuf = writeFloats(xPair, true);
        GpuBuffer xTokenBuf = writeFloats(xToken, true);
        GpuBuffer scalesBuf = writeFloats(scales, true);
        GpuBuffer countsBuf = buffer(4L * N_EXPERT, false);
        GpuBuffer offsetsBuf = buffer(4L * N_EXPERT, false);
        GpuBuffer rowTokenBuf = buffer(4L * N_ROWS, false);
        GpuBuffer rowSlotBuf = buffer(4L * N_ROWS, false);
        int aligned = (int) MoeKernels.alignedPositions(N_ROWS, N_EXPERT);
        GpuBuffer gemmTokenBuf = buffer(4L * aligned, false);
        GpuBuffer gemmSlotBuf = buffer(4L * aligned, false);
        GpuBuffer outPairBuf = buffer(4L * N_ROWS * N_FF, false);
        GpuBuffer outTokenBuf = buffer(4L * N_ROWS * N_FF, false);
        queue.finish();

        dispatch(MoeKernels.BUCKETIZE, new long[]{LOCAL}, new long[]{LOCAL},
                MoeKernels.bucketizeArgs(N_TOKENS, K, N_EXPERT, idsBuf, countsBuf, offsetsBuf, rowTokenBuf,
                        rowSlotBuf));
        dispatch(MoeKernels.ALIGN_BLOCKS, new long[]{LOCAL}, new long[]{LOCAL},
                MoeKernels.alignBlocksArgs(N_TOKENS, K, N_EXPERT, TILES, countsBuf, offsetsBuf, rowTokenBuf,
                        rowSlotBuf, gemmTokenBuf, gemmSlotBuf));
        dispatch(MoeKernels.groupedGemm(GGMLType.F32),
                new long[]{MoeKernels.GROUPED_WORKGROUP, TILES}, new long[]{MoeKernels.GROUPED_WORKGROUP, 1},
                MoeKernels.groupedGemmArgs(N_FF, N_EMBD, TILES, 1, N_EMBD, 0, 0, true, 0, N_EXPERT, gemmTokenBuf, gemmSlotBuf,
                        weightBuf, xPairBuf, scalesBuf, outPairBuf));
        dispatch(MoeKernels.groupedGemm(GGMLType.F32),
                new long[]{MoeKernels.GROUPED_WORKGROUP, TILES}, new long[]{MoeKernels.GROUPED_WORKGROUP, 1},
                MoeKernels.groupedGemmArgs(N_FF, N_EMBD, TILES, K, N_EMBD, 0, 0, true, 0, N_EXPERT, gemmTokenBuf, gemmSlotBuf,
                        weightBuf, xTokenBuf, scalesBuf, outTokenBuf));
        queue.finish();

        int[] deviceToken = readInts(gemmTokenBuf, aligned);
        int[] deviceSlot = readInts(gemmSlotBuf, aligned);
        MoeCombine.Buckets buckets = MoeCombine.bucketize(N_TOKENS, K, N_EXPERT, ids);
        MoeCombine.Aligned cpu = MoeCombine.alignBlocks(N_TOKENS, K, N_EXPERT, buckets.counts(), buckets.offsets(),
                buckets.rowToken(), buckets.rowSlot());
        assertArrayEquals("gemm_token of the device align", cpu.gemmToken(), deviceToken);
        assertArrayEquals("gemm_slot of the device align", cpu.gemmSlot(), deviceSlot);

        float[] devicePair = readFloats(outPairBuf, N_ROWS * N_FF);
        float[] deviceTokenOut = readFloats(outTokenBuf, N_ROWS * N_FF);
        for (int p = 0; p < N_ROWS; p++) {
            int expert = ids[p];
            boolean valid = expert >= 0 && expert < N_EXPERT;
            for (int r = 0; r < N_FF; r++) {
                float pairRow = 0.0f;
                float tokenRow = 0.0f;
                if (valid) {
                    float accPair = 0.0f;
                    float accToken = 0.0f;
                    for (int kk = 0; kk < N_EMBD; kk++) {
                        float w = weights[(expert * N_FF + r) * N_EMBD + kk];
                        accPair += w * xPair[p * N_EMBD + kk];
                        accToken += w * xToken[(p / K) * N_EMBD + kk];
                    }
                    pairRow = accPair * scales[expert];
                    tokenRow = accToken * scales[expert];
                }
                assertTrue("pair rows output[" + p + "][" + r + "]",
                        Math.abs(devicePair[p * N_FF + r] - pairRow) <= 1e-4 * Math.max(1.0f, Math.abs(pairRow)));
                assertTrue("token rows output[" + p + "][" + r + "]",
                        Math.abs(deviceTokenOut[p * N_FF + r] - tokenRow) <= 1e-4 * Math.max(1.0f, Math.abs(tokenRow)));
            }
        }
    }

    private float[] readFloats(GpuBuffer b, int count) {
        float[] out = new float[count];
        GpuEvent e = queue.enqueueReadBuffer(b, 0, 4L * count, out);
        queue.finish();
        if (e != null) {
            e.close();
        }
        return out;
    }
}
