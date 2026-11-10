package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelTaskBatch;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.OpenClMmqGemmParityTest;
import com.eonmux.neuralfree.gguf.ops.gemm.SplitKGemmPlan;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The OpenCL subgroup split-K twin end to end on the devices that build the subgroup forms (the Intel GPUs; other
 * devices skip): {@link GpuOps#matmul} of a 2048-deep projection through {@link GpuProjections}, whose grid of 128 x 32
 * tiles is the largest that {@link SplitKGemmPlan#subgroupSplitsFor} splits on the device, must dispatch
 * {@code {T}_MatMulSg_MMQ_KSplit} and {@code LLM_SplitK_Reduce} and neither the plain subgroup kernel nor the portable
 * kernels, and its outputs must match, within the fp32 rounding of the regrouped sums, those of the same tokens in a
 * launch of twice the token tiles, which keeps the plain {@code {T}_MatMulSg_MMQ} grid. Every token's Q8_1 records and
 * every row's fixed-order stage terms are the same in both launches, so only the grouping of the stage sums into
 * slices and the reduce differ.
 */
@RunWith(Parameterized.class)
public class SubgroupMmqSplitKGpuTest {

    private static final int K = 2048;
    /** Tokens of the split launch: one token tile. */
    private static final int SPLIT_TOKENS = 16;
    /** Tokens of the whole-grid launch: two token tiles. */
    private static final int WHOLE_TOKENS = 48;
    /** Unit roundoff of fp32. */
    private static final double FP32_U = 0x1.0p-24;

    private final GGMLType type;
    private SubgroupFormBackends devices;

    public SubgroupMmqSplitKGpuTest(GGMLType type) {
        this.type = type;
    }

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> types() {
        List<Object[]> types = new ArrayList<>();
        for (GGMLType t : SubgroupMmqGemm.TYPES) {
            types.add(new Object[]{t});
        }
        return types;
    }

    @Before
    public void open() {
        devices = SubgroupFormBackends.open(BackendType.OPENCL);
    }

    @After
    public void close() {
        if (devices != null) {
            devices.close();
        }
    }

    @Test
    public void aSplitProjectionRunsTheTwinAndTheReduceAndMatchesTheWholeGrid() {
        int served = 0;
        for (ComputeBackend backend : devices.backends()) {
            if (!LlmKernels.availability(backend).available()) {
                continue;
            }
            LlmKernels kernels = LlmKernels.open(backend);
            if (!SubgroupMmqGemm.programBuilds(kernels)
                    || !kernels.has(type.name() + GemmKernels.MATMUL_MMQ_SG_KSPLIT_SUFFIX)) {
                continue;
            }
            int units = backend.capabilities().getComputeUnits();
            int rowTiles = largestSplitRowTiles(units);
            if (rowTiles == 0) {
                continue;
            }
            served++;
            assertEquals("two token tiles keep the whole grid", 1,
                    SplitKGemmPlan.subgroupSplitsFor(units, rowTiles, 2, K));
            run(backend, kernels, rowTiles * SubgroupMmqGemm.TILE_ROWS - 40);
        }
        Assume.assumeTrue("no OpenCL device builds the " + type + " split-K twin and splits a grid", served > 0);
    }

    /** The largest number of 128-row tiles of one token tile that the device splits, or 0 when it splits none. */
    private static int largestSplitRowTiles(int units) {
        for (int tiles = SplitKGemmPlan.MAX_SPLITS; tiles >= 1; tiles--) {
            if (SplitKGemmPlan.subgroupSplitsFor(units, tiles, 1, K) > 1) {
                return tiles;
            }
        }
        return 0;
    }

    private void run(ComputeBackend backend, LlmKernels kernels, int rows) {
        String dev = backend.capabilities().getDeviceName();
        byte[] raw = OpenClMmqGemmParityTest.buildWeight(type, rows, K);
        float[] x = OpenClMmqGemmParityTest.buildX(WHOLE_TOKENS, K);
        RecordingQueue queue = new RecordingQueue(backend.computeQueue());
        GpuScratch scratch = GpuScratch.builder(backend, WHOLE_TOKENS)
                .slot(ScratchSlots.X, (long) WHOLE_TOKENS * K).build();
        GpuPlaceholders placeholders = GpuPlaceholders.allocate(backend, queue);
        GpuOps ops = new GpuOps(queue, kernels, scratch, placeholders);
        List<GpuBuffer> buffers = new ArrayList<>();
        try {
            GpuBuffer wb = upload(backend, raw, buffers);
            GpuWeight w = GpuWeight.of("blk.0.ffn_down.weight", type, new long[]{K, rows}, wb);
            GpuBuffer in = backend.allocateBuffer((long) x.length * Float.BYTES);
            buffers.add(in);
            queue.enqueueWriteBuffer(in, 0, (long) x.length * Float.BYTES, x);
            GpuBuffer split = alloc(backend, (long) rows * SPLIT_TOKENS, buffers);
            GpuBuffer whole = alloc(backend, (long) rows * WHOLE_TOKENS, buffers);

            String sg = SubgroupMmqGemm.kernel(type);
            String twin = type.name() + GemmKernels.MATMUL_MMQ_SG_KSPLIT_SUFFIX;
            List<String> splitKernels = queue.record(() -> ops.matmul(w, in, 0, split, 0, SPLIT_TOKENS));
            assertTrue(dev + " " + splitKernels, splitKernels.contains(twin));
            assertTrue(dev + " " + splitKernels, splitKernels.contains(GemmKernels.SPLIT_K_REDUCE));
            assertFalse(dev + " " + splitKernels, splitKernels.contains(sg));
            assertFalse(dev + " " + splitKernels, splitKernels.contains(type.name() + GemmKernels.MATMUL_MMQ_SUFFIX));
            assertFalse(dev + " " + splitKernels,
                    splitKernels.contains(type.name() + GemmKernels.MATMUL_MMQ_KSPLIT_SUFFIX));
            List<String> wholeKernels = queue.record(() -> ops.matmul(w, in, 0, whole, 0, WHOLE_TOKENS));
            assertTrue(dev + " " + wholeKernels, wholeKernels.contains(sg));
            assertFalse(dev + " " + wholeKernels, wholeKernels.contains(twin));

            float[] got = read(queue, split, rows * SPLIT_TOKENS);
            float[] want = read(queue, whole, rows * WHOLE_TOKENS);
            float[] wf = CpuBlockDecoders.dequantize(raw, type, (long) rows * K);
            /* the whole grid adds K / 64 stage terms in order; the split adds them in 8 slices and the reduce adds
             * the 8 planes, so both sums are within (K / 64 + 8) u of the exact one, over the terms' magnitudes
             * (a K-quant term is a scaled code dot and a minimum product, bounded here with a factor 4) */
            double growth = 4.0 * 2 * (K / SplitKGemmPlan.SUBGROUP_MMQ_SLICE_DEPTH + SplitKGemmPlan.MAX_SPLITS)
                    * FP32_U;
            for (int t = 0; t < SPLIT_TOKENS; t++) {
                for (int r = 0; r < rows; r++) {
                    double mag = 0;
                    for (int c = 0; c < K; c++) {
                        mag += Math.abs((double) wf[r * K + c] * x[t * K + c]);
                    }
                    float a = got[t * rows + r];
                    float b = want[t * rows + r];
                    assertTrue(dev + " " + type + " token " + t + " row " + r + ": split " + a + ", whole " + b,
                            Math.abs(a - b) <= growth * mag + 1e-6);
                }
            }
        } finally {
            for (GpuBuffer b : buffers) {
                b.close();
            }
            ops.close();
            placeholders.close();
            scratch.close();
        }
    }

    private static GpuBuffer upload(ComputeBackend backend, byte[] data, List<GpuBuffer> buffers) {
        GpuBuffer b = backend.allocateBuffer(data.length);
        buffers.add(b);
        backend.computeQueue().enqueueWriteBuffer(b, 0, data.length, data);
        return b;
    }

    private static GpuBuffer alloc(ComputeBackend backend, long floats, List<GpuBuffer> buffers) {
        GpuBuffer b = backend.allocateBuffer(floats * Float.BYTES);
        buffers.add(b);
        return b;
    }

    private static float[] read(ComputeQueue queue, GpuBuffer b, int n) {
        float[] out = new float[n];
        queue.enqueueReadBuffer(b, 0, (long) n * Float.BYTES, out);
        queue.finish();
        return out;
    }

    /** A queue that forwards every command to the device queue and records the names of the kernels it runs. */
    private static final class RecordingQueue implements ComputeQueue {
        private final ComputeQueue queue;
        private final List<String> names = new ArrayList<>();

        RecordingQueue(ComputeQueue queue) {
            this.queue = queue;
        }

        /** The kernels {@code action} enqueues, in order. */
        List<String> record(Runnable action) {
            names.clear();
            action.run();
            return new ArrayList<>(names);
        }

        @Override
        public QueueType getQueueType() {
            return queue.getQueueType();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dst, long off, long size, byte[] src, GpuEvent... wait) {
            return queue.enqueueWriteBuffer(dst, off, size, src, wait);
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dst, long off, long size, float[] src, GpuEvent... wait) {
            return queue.enqueueWriteBuffer(dst, off, size, src, wait);
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dst, long off, long size, java.nio.ByteBuffer src,
                                           GpuEvent... wait) {
            return queue.enqueueWriteBuffer(dst, off, size, src, wait);
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer src, long off, long size, byte[] dst, GpuEvent... wait) {
            return queue.enqueueReadBuffer(src, off, size, dst, wait);
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer src, long off, long size, float[] dst, GpuEvent... wait) {
            return queue.enqueueReadBuffer(src, off, size, dst, wait);
        }

        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] global, long[] local, Object[] args,
                                      GpuEvent... wait) {
            names.add(kernel.kernelName());
            return queue.enqueueKernel(kernel, global, local, args, wait);
        }

        @Override
        public void enqueueKernelNoEvent(KernelHandle kernel, long[] global, long[] local, Object[] args,
                                         GpuEvent... wait) {
            names.add(kernel.kernelName());
            queue.enqueueKernelNoEvent(kernel, global, local, args, wait);
        }

        @Override
        public void enqueueCopyBuffer(GpuBuffer src, long srcOff, GpuBuffer dst, long dstOff, long size,
                                      GpuEvent... wait) {
            queue.enqueueCopyBuffer(src, srcOff, dst, dstOff, size, wait);
        }

        @Override
        public void setKernelArg(KernelHandle kernel, int argIndex, Object arg) {
            queue.setKernelArg(kernel, argIndex, arg);
        }

        @Override
        public void enqueueKernelPreBound(KernelHandle kernel, long[] global, long[] local, GpuEvent... wait) {
            names.add(kernel.kernelName());
            queue.enqueueKernelPreBound(kernel, global, local, wait);
        }

        @Override
        public GpuEvent enqueueKernelPreBoundWithEvent(KernelHandle kernel, long[] global, long[] local,
                                                       GpuEvent... wait) {
            names.add(kernel.kernelName());
            return queue.enqueueKernelPreBoundWithEvent(kernel, global, local, wait);
        }

        @Override
        public void enqueueFillBuffer(GpuBuffer dst, long off, long size, int pattern, GpuEvent... wait) {
            queue.enqueueFillBuffer(dst, off, size, pattern, wait);
        }

        @Override
        public GpuEvent enqueueMarker(GpuEvent... wait) {
            return queue.enqueueMarker(wait);
        }

        /** Commands reach the device one by one, so the recorded names cover every kernel. */
        @Override
        public boolean supportsGraphCapture() {
            return false;
        }

        @Override
        public boolean supportsKernelBatching() {
            return false;
        }

        @Override
        public KernelTaskBatch beginKernelBatch() {
            throw new UnsupportedOperationException("the recording queue forwards commands one by one");
        }

        @Override
        public GpuGraph endCapture() {
            throw new UnsupportedOperationException("the recording queue does not capture");
        }

        @Override
        public KernelTiming kernelTiming() {
            return queue.kernelTiming();
        }

        @Override
        public void flush() {
            queue.flush();
        }

        @Override
        public void finish() {
            queue.finish();
        }

        /** The device queue belongs to the backend, which closes it. */
        @Override
        public void close() {
        }
    }
}
