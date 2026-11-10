package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;

import static org.junit.Assert.assertTrue;

/**
 * The stream-k form of the int8 MMQ GEMM ({@code <T>_MatMul_MMQ_StreamK} with {@code LLM_MMQ_StreamK_Fixup})
 * against the plain kernel of the same CUDA fatbin on the tensor tile: a partition of whole tiles runs every
 * tile through the plain kernel's chunk pipe, so its outputs equal the plain kernel's bit for bit; partitions
 * that split tiles across blocks add the fixup planes, which moves only the fp32 rounding of the partial
 * sums. The shapes cover partial row and token tiles and a chunk count that no block count divides.
 */
@RunWith(Parameterized.class)
public class CudaMmqStreamKParityTest {

    private static final int M = 333;  /* two full 128-row tiles and a 77-row partial */
    private static final int N = 141;  /* one full 128-token tile and a 13-token partial */

    @Parameterized.Parameters(name = "{0}")
    public static Collection<GGMLType> types() {
        return new ArrayList<>(GemmKernels.MMQ_TYPES);
    }

    private final GGMLType type;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;
    private MmqConfig config;

    public CudaMmqStreamKParityTest(GGMLType type) {
        this.type = type;
    }

    @Before
    public void setUp() {
        HWAcceleration.enableCuda = true;
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no CUDA device", !hw.getCUDevices().isEmpty());
            CUDevice device = hw.getCUDevices().get(0);
            CudaComputeBackend b = new CudaComputeBackend(device);
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            MmqConfig c = MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, b.capabilities().getComputeCapability());
            Assume.assumeTrue("the stream-k kernels run the int8 tensor tile, the device has " + c,
                    MmqConfig.INT8_TENSOR.equals(c));
            hwAccel = hw;
            backend = b;
            queue = b.computeQueue();
            kernels = LlmKernels.open(b);
            config = c;
        } catch (Throwable t) {
            if (hw != null) {
                try {
                    hw.shutdown();
                } catch (Throwable ignored) {
                }
            }
            Assume.assumeTrue("CUDA init failed: " + t.getMessage(), false);
        }
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

    /** One block per tile and one block per two tiles run whole tiles: the plain kernel's outputs bit for bit. */
    @Test
    public void wholeTilePartitionsMatchThePlainKernelBitForBit() throws InterruptedException {
        int k = 5 * type.getBlockSize();
        MmqGemmPlan plain = MmqGemmPlan.contiguous(config, type, M, k, N);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, M, k);
        float[] x = OpenClMmqGemmParityTest.buildX(N, k);
        float[] plainOut = new FastGemmParity(backend, kernels).launchMmq(plain, weight, x);
        long tiles = (long) plain.rowTiles() * plain.tokenTiles();
        for (int blocks : new int[]{(int) tiles, (int) tiles / 2}) {
            StreamKGemmPlan sk = StreamKGemmPlan.ofBlocks(plain, blocks);
            assertTrue(sk + " splits no tile", !sk.fixupNeeded());
            float[] out = launchStreamK(sk, weight, x);
            for (int i = 0; i < plainOut.length; i++) {
                assertTrue(sk + " element " + i + ": " + Float.toHexString(out[i]) + " plain "
                                + Float.toHexString(plainOut[i]),
                        Float.floatToRawIntBits(out[i]) == Float.floatToRawIntBits(plainOut[i]));
            }
        }
    }

    /**
     * Block counts that split tiles (several blocks per tile, a block spanning a tile boundary, one chunk
     * per block) match the plain kernel within the fp32 rounding of the partial sums: the split-K planes'
     * bound ({@link CudaMmqSmallTileParityTest#theSplitKPlanesSumToThePlainKernel}), since the synthetic
     * k-quant and i-quant blocks reduce to outputs far smaller than their per-chunk partial sums, whose
     * reassociation moves the result by a few ulps of the partials.
     */
    @Test
    public void splitTilePartitionsMatchThePlainKernel() throws InterruptedException {
        int k = 5 * type.getBlockSize();
        MmqGemmPlan plain = MmqGemmPlan.contiguous(config, type, M, k, N);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, M, k);
        float[] x = OpenClMmqGemmParityTest.buildX(N, k);
        float[] plainOut = new FastGemmParity(backend, kernels).launchMmq(plain, weight, x);
        long steps = (long) plain.rowTiles() * plain.tokenTiles() * (k / config.stageDepth());
        for (int blocks : new int[]{5, 7, 13, (int) steps}) {
            StreamKGemmPlan sk = StreamKGemmPlan.ofBlocks(plain, blocks);
            assertTrue(sk + " splits tiles", sk.fixupNeeded());
            float[] out = launchStreamK(sk, weight, x);
            for (int i = 0; i < plainOut.length; i++) {
                assertTrue(sk + " element " + i + ": " + out[i] + " plain " + plainOut[i],
                        Math.abs(out[i] - plainOut[i]) <= 5e-3 * (1.0 + Math.abs(plainOut[i])));
            }
        }
    }

    /**
     * The stream-k launch over freshly staged records ({@code LLM_Quant_MMQ_Q8_1}, the launch contract of
     * {@link FastGemmParity#launchMmq}): the stream-k kernel, then the fixup when a tile is split. The fixup
     * planes start as NaN so a plane the fixup reads but no block wrote shows in the outputs.
     */
    private float[] launchStreamK(StreamKGemmPlan sk, byte[] weight, float[] x) throws InterruptedException {
        MmqGemmPlan plan = sk.base();
        int k = plan.k();
        int n = plan.n();
        long records = (long) n * config.rowRecords(k);
        float[] poison = new float[(int) sk.fixupElements()];
        java.util.Arrays.fill(poison, Float.NaN);
        GpuBuffer wbuf = backend.allocateBuffer(weight.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(plan.q8ElementsRequired() * Integer.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired() * Float.BYTES);
        GpuBuffer fbuf = backend.allocateBuffer(sk.fixupElements() * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv);
                 GpuEvent fv = queue.enqueueWriteBuffer(fbuf, 0, (long) poison.length * Float.BYTES, poison, xv)) {
                Object[] quantArgs = {n, k, config.stageDepth(), k, 0, xbuf, qbuf};
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                        new long[]{(records + 63) / 64 * 64}, new long[]{64}, quantArgs, fv);
                queue.enqueueKernelNoEvent(kernels.get(sk.kernelName()), sk.globalWorkSize(), sk.localWorkSize(),
                        sk.args(wbuf, qbuf, ybuf, fbuf));
                if (sk.fixupNeeded()) {
                    queue.enqueueKernelNoEvent(kernels.get(GemmKernels.STREAM_K_FIXUP), sk.fixupGlobalWorkSize(),
                            sk.localWorkSize(), sk.fixupArgs(fbuf, ybuf));
                }
                float[] out = new float[(int) plan.yElementsRequired()];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) out.length * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
                return out;
            }
        } finally {
            wbuf.close();
            xbuf.close();
            qbuf.close();
            ybuf.close();
            fbuf.close();
        }
    }
}
