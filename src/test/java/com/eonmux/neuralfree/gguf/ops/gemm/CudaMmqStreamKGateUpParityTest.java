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
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.assertTrue;

/**
 * The fused gate/up SwiGLU stream-k kernel ({@code <T>_MatMul_MMQ_StreamK_GateUp} with
 * {@code LLM_MMQ_StreamK_Fixup_GateUp}) against the separate route of
 * the same CUDA fatbin: the plain {@code <T>_MatMul_MMQ} products of the gate and of the up weight, then
 * {@code LLM_GLU}'s SWIGLU over them. A partition of whole tiles runs each pair of products through the plain
 * kernel's chunk pipe and pairs them in shared memory, so its outputs equal the separate route bit for bit;
 * partitions that split tiles pair the summed products in the fixup after the launch, which moves only the fp32
 * rounding of the partial sums and gives the same bits on every run. The shapes
 * cover a partial 64-row output tile, a partial token tile and a chunk count no block count divides.
 */
@RunWith(Parameterized.class)
public class CudaMmqStreamKGateUpParityTest {

    private static final int M = 333;  /* five full 64-row output tiles and a 13-row partial */
    private static final int N = 141;  /* one full 128-token tile and a 13-token partial */
    private static final int GLU_LOCAL = 256;

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

    public CudaMmqStreamKGateUpParityTest(GGMLType type) {
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

    /** One block per output tile and one per two tiles run whole tiles: the separate route bit for bit. */
    @Test
    public void wholeTilePartitionsMatchTheSeparateRouteBitForBit() throws InterruptedException {
        Case c = new Case(5 * type.getBlockSize());
        long tiles = StreamKGemmPlan.ofGateUpBlocks(c.plain, 1).tiles();
        for (int blocks : new int[]{(int) tiles, (int) tiles / 2}) {
            StreamKGemmPlan sk = StreamKGemmPlan.ofGateUpBlocks(c.plain, blocks);
            assertTrue(sk + " splits no tile", !sk.splitsTiles());
            float[] out = launchGateUp(sk, c)[0];
            for (int i = 0; i < c.expected.length; i++) {
                assertTrue(sk + " element " + i + ": " + Float.toHexString(out[i]) + " separate "
                                + Float.toHexString(c.expected[i]),
                        Float.floatToRawIntBits(out[i]) == Float.floatToRawIntBits(c.expected[i]));
            }
        }
    }

    /**
     * Block counts that split tiles (several blocks per tile, a block spanning a tile boundary, one chunk per
     * block) match the separate route within the fp32 rounding of the partial sums (the bound of
     * {@link CudaMmqStreamKParityTest#splitTilePartitionsMatchThePlainKernel}, carried through the SwiGLU), and
     * give the same bits on a second run over the same buffers.
     */
    @Test
    public void splitTilePartitionsMatchTheSeparateRoute() throws InterruptedException {
        Case c = new Case(5 * type.getBlockSize());
        long steps = StreamKGemmPlan.ofGateUpBlocks(c.plain, 1).tiles() * c.plain.k() / config.stageDepth();
        for (int blocks : new int[]{5, 7, 13, (int) steps}) {
            StreamKGemmPlan sk = StreamKGemmPlan.ofGateUpBlocks(c.plain, blocks);
            assertTrue(sk + " splits tiles", sk.splitsTiles());
            float[][] runs = launchGateUp(sk, c);
            for (int i = 0; i < c.expected.length; i++) {
                float e = c.expected[i];
                float g = c.gateOut[i];
                float u = c.upOut[i];
                /* d(silu(g) u) = silu'(g) u dg + silu(g) du, silu' <= 1.1: the products' bound scaled by both */
                double bound = 5e-3 * ((1.0 + Math.abs(g)) * 1.1 * Math.abs(u) + (1.0 + Math.abs(u)) * Math.abs(g));
                assertTrue(sk + " element " + i + ": " + runs[0][i] + " separate " + e,
                        Math.abs(runs[0][i] - e) <= bound + 1e-6 * (1.0 + Math.abs(e)));
                assertTrue(sk + " element " + i + " differs between runs: " + Float.toHexString(runs[0][i])
                                + " and " + Float.toHexString(runs[1][i]),
                        Float.floatToRawIntBits(runs[0][i]) == Float.floatToRawIntBits(runs[1][i]));
            }
        }
    }

    /** The weights, input and separate-route outputs of one reduction length. */
    private final class Case {
        final MmqGemmPlan plain;
        final byte[] gate;
        final byte[] up;
        final float[] x;
        final float[] gateOut;
        final float[] upOut;
        final float[] expected;

        Case(int k) throws InterruptedException {
            plain = MmqGemmPlan.contiguous(config, type, M, k, N);
            byte[] both = OpenClMmqGemmParityTest.buildWeight(type, 2 * M, k);
            gate = Arrays.copyOfRange(both, 0, both.length / 2);
            up = Arrays.copyOfRange(both, both.length / 2, both.length);
            x = OpenClMmqGemmParityTest.buildX(N, k);
            FastGemmParity parity = new FastGemmParity(backend, kernels);
            gateOut = parity.launchMmq(plain, gate, x);
            upOut = parity.launchMmq(plain, up, x);
            expected = swiglu(gateOut, upOut);
        }
    }

    /** {@code LLM_GLU}'s SWIGLU over the N x M products. */
    private float[] swiglu(float[] g, float[] u) throws InterruptedException {
        int n = M * N;
        GpuBuffer gbuf = backend.allocateBuffer((long) n * Float.BYTES, true);
        GpuBuffer ubuf = backend.allocateBuffer((long) n * Float.BYTES, true);
        GpuBuffer obuf = backend.allocateBuffer((long) n * Float.BYTES);
        try {
            float[] out = new float[n];
            try (GpuEvent gv = queue.enqueueWriteBuffer(gbuf, 0, (long) n * Float.BYTES, g);
                 GpuEvent uv = queue.enqueueWriteBuffer(ubuf, 0, (long) n * Float.BYTES, u, gv)) {
                queue.enqueueKernelNoEvent(kernels.get(ActKernels.GLU), ActKernels.global1d(n, GLU_LOCAL),
                        new long[]{GLU_LOCAL}, ActKernels.gluArgs(GluOp.SWIGLU, ActParams.NONE, N, M, gbuf, 0, M,
                                ubuf, 0, M, obuf, 0, M), uv);
                try (GpuEvent read = queue.enqueueReadBuffer(obuf, 0, (long) n * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
            }
            return out;
        } finally {
            gbuf.close();
            ubuf.close();
            obuf.close();
        }
    }

    /**
     * The gate/up launch over freshly staged records ({@code LLM_Quant_MMQ_Q8_1}), then its fixup when a tile is
     * split, twice on the same buffers; returns both outputs. The fixup planes start as NaN so a plane the fixup
     * reads but no block wrote shows in the outputs.
     */
    private float[][] launchGateUp(StreamKGemmPlan sk, Case c) throws InterruptedException {
        MmqGemmPlan plan = sk.base();
        int k = plan.k();
        long records = (long) N * config.rowRecords(k);
        float[] poison = new float[(int) sk.fixupElements()];
        Arrays.fill(poison, Float.NaN);
        GpuBuffer gbuf = backend.allocateBuffer(c.gate.length, true);
        GpuBuffer ubuf = backend.allocateBuffer(c.up.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) c.x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(plan.q8ElementsRequired() * Integer.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired() * Float.BYTES);
        GpuBuffer fbuf = backend.allocateBuffer(sk.fixupElements() * Float.BYTES);
        try (GpuEvent gv = queue.enqueueWriteBuffer(gbuf, 0, c.gate.length, c.gate);
             GpuEvent uv = queue.enqueueWriteBuffer(ubuf, 0, c.up.length, c.up, gv);
             GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) c.x.length * Float.BYTES, c.x, uv);
             GpuEvent fv = queue.enqueueWriteBuffer(fbuf, 0, (long) poison.length * Float.BYTES, poison, xv)) {
            Object[] quantArgs = {N, k, config.stageDepth(), k, 0, xbuf, qbuf};
            queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                    new long[]{(records + 63) / 64 * 64}, new long[]{64}, quantArgs, fv);
            float[][] outs = new float[2][(int) plan.yElementsRequired()];
            for (float[] out : outs) {
                queue.enqueueKernelNoEvent(kernels.get(sk.kernelName()), sk.globalWorkSize(), sk.localWorkSize(),
                        sk.argsGateUp(gbuf, ubuf, qbuf, ybuf, fbuf));
                if (sk.splitsTiles()) {
                    queue.enqueueKernelNoEvent(kernels.get(sk.fixupKernelName()), sk.fixupGlobalWorkSize(),
                            sk.localWorkSize(), sk.fixupArgs(fbuf, ybuf));
                }
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) out.length * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
            }
            return outs;
        } finally {
            gbuf.close();
            ubuf.close();
            xbuf.close();
            qbuf.close();
            ybuf.close();
            fbuf.close();
        }
    }
}
