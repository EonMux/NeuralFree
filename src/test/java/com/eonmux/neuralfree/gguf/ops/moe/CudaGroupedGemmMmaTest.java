package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The tensor-core {@code {T}_GroupedGemmMma} on a CUDA device of sm_80 or higher against the scalar
 * {@code {T}_GroupedGemm}, the per-pair {@code {T}_BatchMatMulId} and the fp64 dot of the decoded weights, for F16,
 * BF16 and Q8_0 experts: out-of-range ids (zeroed outputs), per-expert scales, an expert tensor in two shards, and
 * the MLA strides ({@code x_stride > in_dim}, input and output offsets). Weights and activations sit on a grid that
 * is exact in the operand format, so only fp32 accumulation order separates the paths. Skips without such a device.
 */
public class CudaGroupedGemmMmaTest {

    private static final int N_EXPERT = 5;
    private static final int SHARD = 3;
    private static final int K = 2;
    private static final int N_TOKENS = 37;
    private static final int IN_DIM = 96;
    private static final int OUT_DIM = 70;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private LlmKernels kernels;
    private GroupedGemmHarness h;

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
            hwAccel = hw;
            backend = b;
            kernels = LlmKernels.open(b);
            h = new GroupedGemmHarness(b, kernels);
        } catch (Throwable t) {
            if (hw != null && hwAccel == null) {
                try {
                    hw.shutdown();
                } catch (Throwable ignored) {
                    // the device never opened
                }
            }
            Assume.assumeTrue("CUDA init failed: " + t.getMessage(), false);
        }
        Assume.assumeTrue("the tensor-core grouped GEMM needs CUDA sm_80+; device is sm_"
                + backend.capabilities().getComputeCapability(), backend.capabilities().getComputeCapability() >= 80);
    }

    @After
    public void tearDown() {
        if (h != null) {
            h.close();
        }
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    @Test
    public void groupedKernelPicksTheTensorCoreTwinExceptForF32() {
        assertEquals("F16_GroupedGemmMma", GpuGroupedRows.groupedKernel(h.ops(), GGMLType.F16, false));
        assertEquals("Q8_0_GroupedGemmMma", GpuGroupedRows.groupedKernel(h.ops(), GGMLType.Q8_0, false));
        assertEquals("F32_GroupedGemm", GpuGroupedRows.groupedKernel(h.ops(), GGMLType.F32, false));
        assertEquals("BF16_GroupedGemmMmaInd", GpuGroupedRows.groupedKernel(h.ops(), GGMLType.BF16, true));
        assertEquals("F32_GroupedGemmInd", GpuGroupedRows.groupedKernel(h.ops(), GGMLType.F32, true));
    }

    @Test
    public void f16TensorCoreGroupedGemmMatchesThePerPairKernels() {
        checkShardedExperts(GGMLType.F16, 0x1p-9);
    }

    @Test
    public void bf16TensorCoreGroupedGemmMatchesThePerPairKernels() {
        checkShardedExperts(GGMLType.BF16, 0x1p-7);
    }

    @Test
    public void q8TensorCoreGroupedGemmMatchesThePerPairKernels() {
        checkShardedExperts(GGMLType.Q8_0, 0x1p-9);
    }

    /** The MLA absorption shape on the tensor cores: heads as experts, strided pair rows, offset input and output. */
    @Test
    public void tensorCoreGroupedGemmFollowsTheMlaStrides() {
        checkMlaStrides(112, 5, 13);
    }

    /**
     * The same shape over rows whose byte phase alternates along the odd stride ({@code in_dim + 1}, odd input
     * offset): one dispatch of the tensor-core kernel and of its {@code {T}_MatVecId} reference covers pair rows
     * that start 16-byte aligned and rows that do not.
     */
    @Test
    public void tensorCoreGroupedGemmFollowsTheMisaligningMlaStrides() {
        checkMlaStrides(IN_DIM + 1, 3, 7);
    }

    /**
     * The tensor-core grouped GEMM and its per-row reference over {@code rows = 9 * 4} pairs of {@code nHead = 4}
     * experts with row starts {@code inOff + p * xStride} and output starts {@code outOff + p * OUT_DIM}, each
     * checked against the fp64 dot of the decoded weights.
     */
    private void checkMlaStrides(int xStride, int inOff, int outOff) {
        GGMLType type = GGMLType.Q8_0;
        int nHead = 4;
        int nTok = 9;
        int rows = nTok * nHead;
        int[] slots = new int[rows];
        for (int s = 0; s < rows; s++) {
            slots[s] = s % nHead;
        }
        byte[] raw = GroupedGemmHarness.weight(type, nHead * OUT_DIM, IN_DIM, 8);
        float[] x = GroupedGemmHarness.values(inOff + rows * xStride, 3, 2.0f);
        GpuBuffer slotBuf = h.ints(slots);
        GpuBuffer weight = h.bytes(raw);
        GpuBuffer input = h.floats(x);
        GpuBuffer ph = h.floats(new float[]{1.0f});
        int outLen = outOff + rows * OUT_DIM;
        GpuBuffer mma = h.out(outLen, -7.0f);
        GpuBuffer ref = h.out(outLen, -7.0f);

        GroupedGemmHarness.Sorted s = h.sort(nTok, nHead, nHead, slotBuf);
        h.grouped(MoeKernels.groupedGemmMma(type), s, OUT_DIM, IN_DIM, 1, xStride, inOff, outOff, false, 0, nHead,
                weight, input, ph, mma);
        h.perRow(type, rows, OUT_DIM, IN_DIM, xStride, inOff, outOff, nHead, slotBuf, weight, input, ph, ref);
        h.finish();

        float[] w = GroupedGemmHarness.decode(type, raw, nHead * OUT_DIM, IN_DIM);
        float[] got = h.readFloats(mma, outLen);
        float[] matVec = h.readFloats(ref, outLen);
        for (int i = 0; i < outOff; i++) {
            assertEquals("untouched prefix " + i, -7.0f, got[i], 0.0f);
        }
        for (int p = 0; p < rows; p++) {
            for (int r = 0; r < OUT_DIM; r++) {
                double[] dot = dot(w, slots[p] * OUT_DIM + r, x, inOff + p * xStride);
                int at = outOff + p * OUT_DIM + r;
                double tol = 0x1p-9 * dot[1] + 1e-3;
                assertNear("tensor-core pair " + p + " row " + r, dot[0], got[at], tol);
                assertNear("MatVecId pair " + p + " row " + r, dot[0], matVec[at], tol);
            }
        }
    }

    /**
     * Two shards ({@code [0, 3)} and {@code [3, 5)}) of a scaled expert tensor over pairs with out-of-range ids, on
     * the tensor-core, scalar grouped and per-pair kernels, each checked against the fp64 reference.
     */
    private void checkShardedExperts(GGMLType type, double rel) {
        int rows = N_TOKENS * K;
        int[] ids = GroupedGemmHarness.ids(rows, N_EXPERT, 4, true);
        byte[] raw = GroupedGemmHarness.weight(type, N_EXPERT * OUT_DIM, IN_DIM, 6);
        int split = raw.length / N_EXPERT * SHARD;
        float[] x = GroupedGemmHarness.values(N_TOKENS * IN_DIM, 2, 1.5f);
        float[] scales = {0.5f, 1.25f, 2.0f, 0.75f, 1.5f};
        GpuBuffer shard0 = h.bytes(Arrays.copyOfRange(raw, 0, split));
        GpuBuffer shard1 = h.bytes(Arrays.copyOfRange(raw, split, raw.length));
        GpuBuffer input = h.floats(x);
        GpuBuffer scaleBuf = h.floats(scales);
        GpuBuffer mma = h.out(rows * OUT_DIM, -7.0f);
        GpuBuffer scalar = h.out(rows * OUT_DIM, -7.0f);
        GpuBuffer perPair = h.out(rows * OUT_DIM, -7.0f);

        GroupedGemmHarness.Sorted s = h.sort(N_TOKENS, K, N_EXPERT, h.ints(ids));
        String mmaKernel = MoeKernels.groupedGemmMma(type);
        String scalarKernel = MoeKernels.groupedGemm(type);
        int[][] shards = {{0, SHARD}, {SHARD, N_EXPERT - SHARD}};
        GpuBuffer[] weights = {shard0, shard1};
        for (int i = 0; i < shards.length; i++) {
            int base = shards[i][0];
            int count = shards[i][1];
            h.grouped(mmaKernel, s, OUT_DIM, IN_DIM, K, IN_DIM, 0, 0, true, base, count, weights[i], input,
                    scaleBuf, mma);
            h.grouped(scalarKernel, s, OUT_DIM, IN_DIM, K, IN_DIM, 0, 0, true, base, count, weights[i], input,
                    scaleBuf, scalar);
            h.perPair(type, s, rows, OUT_DIM, IN_DIM, K, true, base, count, weights[i], input, scaleBuf, perPair);
        }
        h.finish();

        float[] w = GroupedGemmHarness.decode(type, raw, N_EXPERT * OUT_DIM, IN_DIM);
        float[] gotMma = h.readFloats(mma, rows * OUT_DIM);
        float[] gotScalar = h.readFloats(scalar, rows * OUT_DIM);
        float[] gotPerPair = h.readFloats(perPair, rows * OUT_DIM);
        for (int p = 0; p < rows; p++) {
            int e = ids[p];
            boolean valid = e >= 0 && e < N_EXPERT;
            for (int r = 0; r < OUT_DIM; r++) {
                double[] dot = valid ? dot(w, e * OUT_DIM + r, x, (p / K) * IN_DIM) : new double[2];
                double want = valid ? dot[0] * scales[e] : 0.0;
                double terms = valid ? dot[1] * scales[e] : 0.0;
                int at = p * OUT_DIM + r;
                String where = type + " pair " + p + " (expert " + e + ") row " + r;
                assertNear(where + ", tensor cores", want, gotMma[at], rel * terms + 1e-3);
                assertNear(where + ", scalar grouped", want, gotScalar[at], 1e-5 * terms + 1e-4);
                assertNear(where + ", BatchMatMulId", want, gotPerPair[at], 1e-5 * terms + 1e-4);
            }
        }
    }

    /** {@code {sum, sum of |terms|}} of weight row {@code row} against {@code x[xOff ..]}. */
    private static double[] dot(float[] w, int row, float[] x, int xOff) {
        double acc = 0.0;
        double terms = 0.0;
        for (int i = 0; i < IN_DIM; i++) {
            double t = (double) w[row * IN_DIM + i] * x[xOff + i];
            acc += t;
            terms += Math.abs(t);
        }
        return new double[]{acc, terms};
    }

    private static void assertNear(String what, double expected, float actual, double tol) {
        assertTrue(what + ": expected " + expected + ", got " + actual, Math.abs(expected - actual) <= tol);
    }
}
