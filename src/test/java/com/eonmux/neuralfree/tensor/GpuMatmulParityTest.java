package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.function.UnaryOperator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link GpuMatmul} on each GPU API against the CPU path of {@code TensorOps.matmul} and {@code TensorOps.bmm}: output
 * shapes equal, values within the GEMM tolerance class ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical
 * contract}) of a double-precision reference, and raw results that never require grad. Covers strided operands
 * (transposes, slices, expands), every rank rule, both batch-broadcast layouts, batch chunking, empty extents, and
 * {@link GpuMatmul#gemm} with direct and staged outputs and its in-place guard.
 */
@RunWith(Parameterized.class)
public class GpuMatmulParityTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();
    private long seed = 1;

    public GpuMatmulParityTest(TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        session = TrainParity.open(api);
        gpu = session.backend();
        host = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        try {
            for (Tensor t : owned) {
                t.close();
            }
        } finally {
            if (session != null) {
                session.close();
            }
        }
    }

    // ------------------------------------------------------------------ dense products

    @Test
    public void large2D() {
        checkMatmul("128x128x128", operand(128, 128), operand(128, 128));
    }

    @Test
    public void batched3D() {
        checkMatmul("batch 4 of 64^3", operand(4, 64, 64), operand(4, 64, 64));
    }

    @Test
    public void batched4D() {
        checkMatmul("batch 2x3 of 32^3", operand(2, 3, 32, 32), operand(2, 3, 32, 32));
    }

    /**
     * The device result against the CPU path's float result directly. Both lie within the GEMM class of the exact
     * sum (the CPU's ascending float sum within {@code K * 2^-24 * sum|terms|}, below the class bound for this K),
     * so they differ by at most twice the class bound.
     */
    @Test
    public void consistencyWithCpu() {
        int m = 64;
        int k = 64;
        int n = 64;
        Pair a = operand(m, k);
        Pair b = operand(k, n);
        Tensor cpu = keep(TensorOps.matmul(a.host, b.host));
        Tensor dev = keep(GpuMatmul.matmul(a.dev, b.dev));
        assertNotNull(dev);
        float[] av = a.host.toFloatArray();
        float[] bv = b.host.toFloatArray();
        float[] want = cpu.toFloatArray();
        float[] got = dev.toFloatArray();
        assertEquals(want.length, got.length);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double terms = 0.0;
                for (int kk = 0; kk < k; kk++) {
                    terms += Math.abs((double) av[i * k + kk] * bv[kk * n + j]);
                }
                double bound = 2.0 * (1e-5 * Math.sqrt(k) * terms + k * (double) TrainParity.FLT_MIN);
                int o = i * n + j;
                if (!(Math.abs((double) got[o] - want[o]) <= bound)) {
                    fail("element " + o + ": gpu " + got[o] + ", cpu " + want[o] + ", bound " + bound);
                }
            }
        }
    }

    @Test
    public void edgeTilesNotMultiplesOfTheTile() {
        checkMatmul("67x37 @ 37x45", operand(67, 37), operand(37, 45));
        checkMatmul("1x1 @ 1x1", operand(1, 1), operand(1, 1));
        checkMatmul("65x17 @ 17x129", operand(65, 17), operand(17, 129));
        checkMatmul("3x300 @ 300x2", operand(3, 300), operand(300, 2));
    }

    // ------------------------------------------------------------------ strided operands

    @Test
    public void transposedWeightAsInLinear() {
        Pair x = operand(33, 70);
        Pair w = operand(50, 70).map(Tensor::transpose);
        checkMatmul("x @ W^T", x, w);
    }

    @Test
    public void transposedOperandsAsInMatmulBackward() {
        Pair a = operand(40, 24);
        Pair b = operand(24, 31);
        Pair g = operand(40, 31);
        checkMatmul("grad @ b^T", g, b.map(Tensor::transpose));
        checkMatmul("a^T @ grad", a.map(Tensor::transpose), g);
    }

    @Test
    public void slicedOperandsCarryTheirOffsets() {
        Pair a = operand(50, 40).map(t -> t.slice(0, 7, 43));
        Pair b = operand(60, 30).map(t -> t.slice(0, 11, 51).slice(1, 3, 28));
        checkMatmul("row slice @ 2-D slice", a, b);
    }

    @Test
    public void oneBufferBoundToBothOperands() {
        Pair x = operand(29, 41);
        checkMatmul("x @ x^T", x, x.map(Tensor::transpose));
    }

    @Test
    public void expandedBatchOperand() {
        Pair a = operand(19, 23).map(t -> t.unsqueeze(0).expand(5, 19, 23));
        checkMatmul("expanded a @ batched b", a, operand(5, 23, 17));
    }

    // ------------------------------------------------------------------ rank rules

    @Test
    public void vectorOperandsFollowTheCpuRankRules() {
        checkMatmul("[K] @ [K, N]", operand(37), operand(37, 300));
        checkMatmul("[M, K] @ [K]", operand(300, 37), operand(37));
        checkMatmul("[K] @ [K]", operand(513), operand(513));
        checkMatmul("[K] @ [B, K, N]", operand(21), operand(3, 21, 18));
        checkMatmul("[B, M, K] @ [K]", operand(3, 18, 21), operand(21));
        checkMatmul("[M, K] @ [B1, B2, K, N]", operand(14, 9), operand(2, 2, 9, 11));
        checkMatmul("strided [K] @ [K, N]", operand(37, 2).map(t -> t.slice(1, 1, 2).squeeze(1)), operand(37, 66));
    }

    @Test
    public void singleColumnAndRowProducts() {
        checkMatmul("[M, K] @ [K, 1]", operand(130, 600), operand(600, 1));
        checkMatmul("[1, K] @ [K, N]", operand(1, 600), operand(600, 130));
        checkMatmul("A^T @ [K, 1]", operand(50, 30).map(Tensor::transpose), operand(50, 1));
        checkMatmul("[1, K] @ W^T", operand(1, 70), operand(45, 70).map(Tensor::transpose));
        checkMatmul("[B, T, K] @ [K, 1]", operand(4, 9, 33), operand(33, 1));
        checkMatmul("[B, M, K] @ [B, K, 1]", operand(4, 9, 33), operand(4, 33, 1));
        checkMatmul("[B, 1, K] @ [B, K, N]", operand(4, 1, 33), operand(4, 33, 70));
    }

    @Test
    public void batchSharingBFoldsIntoRows() {
        checkMatmul("[2, 3, M, K] @ [K, N]", operand(2, 3, 20, 13), operand(13, 15));
        checkMatmul("[B, T, K] @ W^T", operand(3, 7, 40), operand(25, 40).map(Tensor::transpose));
    }

    @Test
    public void batchBroadcastAcrossBothOperands() {
        checkMatmul("[2, 1, M, K] @ [1, 3, K, N]", operand(2, 1, 12, 8), operand(1, 3, 8, 10));
        checkMatmul("[3, 1, 1, M, K] @ [2, 4, K, N]", operand(3, 1, 1, 6, 5), operand(2, 4, 5, 7));
    }

    @Test
    public void batchAboveTheGroupLimitIsChunked() {
        checkMatmul("batch 70000", operand(70000, 1, 2), operand(70000, 2, 3));
    }

    @Test
    public void emptyExtents() {
        Pair a = operand(5, 0);
        Pair b = operand(0, 7);
        Tensor dev = keep(GpuMatmul.matmul(a.dev, b.dev));
        assertNotNull(dev);
        assertEquals(Shape.matrix(5, 7), dev.shape());
        assertArrayEquals(new float[35], dev.toFloatArray(), 0.0f);
        Tensor rows = keep(GpuMatmul.matmul(operand(0, 4).dev, operand(4, 6).dev));
        assertEquals(Shape.matrix(0, 6), rows.shape());
        assertEquals(0, rows.toFloatArray().length);
    }

    @Test
    public void tooManyColumnTilesIsRejected() {
        int n = GpuMatmul.MAX_GROUPS * TrainOps.GEMM_TILE + 1;
        Tensor a = keep(keep(Tensor.fromArray(gpu, new float[2])).reshape(2, 1));
        Tensor b = keep(keep(Tensor.fromArray(gpu, new float[n])).reshape(1, n));
        try {
            keep(GpuMatmul.matmul(a, b));
            fail("expected IllegalArgumentException for " + n + " columns");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("work-group limit"));
        }
    }

    @Test
    public void rawResultsNeverRequireGradOnEitherPath() {
        Pair a = operand(6, 4).map(t -> t.withRequiresGrad(true));
        Pair b = operand(6, 4, 5).map(t -> t.withRequiresGrad(true));
        Pair c = operand(6, 5, 3);
        try (ComputationGraph.EnableGradScope grad = ComputationGraph.enableGrad()) {
            int nodes = ComputationGraph.current().size();
            Tensor dev = keep(GpuMatmul.matmul(a.dev, b.dev));
            Tensor cpu = keep(CpuLinalg.matmul(a.host, b.host));
            assertNotNull("the train library is available, so the dispatcher must not decline", dev);
            assertFalse("device matmul of operands that require grad", dev.requiresGrad());
            assertFalse("CPU matmul of operands that require grad", cpu.requiresGrad());
            assertNull(dev.gradFn());
            Tensor devBmm = keep(GpuMatmul.bmm(b.dev, c.dev));
            assertFalse("device bmm with an operand that requires grad", devBmm.requiresGrad());
            Tensor cpuBmm = keep(CpuLinalg.bmm(b.host, c.host));
            assertFalse("CPU bmm with an operand that requires grad", cpuBmm.requiresGrad());
            assertEquals("raw products record nothing", nodes, ComputationGraph.current().size());
        }
    }

    // ------------------------------------------------------------------ bmm

    @Test
    public void bmmMatchesCpu() {
        checkBmm("bmm", operand(6, 31, 17), operand(6, 17, 40));
    }

    @Test
    public void bmmWithTransposedKeysAsInAttention() {
        Pair q = operand(8, 12, 16);
        Pair k = operand(8, 12, 16).map(t -> t.transpose(0, 2, 1));
        checkBmm("q @ k^T", q, k);
    }

    @Test
    public void bmmRejectsWhatTheCpuRejects() {
        try {
            GpuMatmul.bmm(operand(2, 3, 4).dev, operand(3, 4, 5).dev);
            fail("expected IllegalArgumentException for mismatched batches");
        } catch (IllegalArgumentException expected) {
            assertEquals("Batch dimensions must match: 2 vs 3", expected.getMessage());
        }
    }

    // ------------------------------------------------------------------ gemm

    @Test
    public void gemmAccumulatesWithAlphaAndBeta() {
        Pair a = operand(45, 30);
        Pair b = operand(30, 70);
        Pair c = operand(45, 70);
        float[] c0 = c.host.toFloatArray();
        assertTrue(GpuMatmul.gemm(a.dev, b.dev, 0.75f, -1.25f, c.dev));
        assertWithinGemm("alpha a@b + beta c", a.host, b.host, 0.75f, -1.25f, c0, c.dev.toFloatArray());
    }

    @Test
    public void gemmWithZeroBetaIgnoresNaNInTheOutput() {
        Pair a = operand(3, 2, 9, 11);
        Pair b = operand(11, 6);
        float[] nan = new float[3 * 2 * 9 * 6];
        Arrays.fill(nan, Float.NaN);
        Tensor c = keep(keep(Tensor.fromArray(gpu, nan)).reshape(3, 2, 9, 6));
        assertTrue(GpuMatmul.gemm(a.dev, b.dev, 2.0f, 0.0f, c));
        assertWithinGemm("beta 0 over NaN", a.host, b.host, 2.0f, 0.0f, new float[nan.length], c.toFloatArray());
    }

    @Test
    public void gemmIntoATransposedOutputIsStaged() {
        Pair a = operand(21, 13);
        Pair b = operand(13, 34);
        Pair c = operand(34, 21).map(Tensor::transpose);
        float[] c0 = c.host.toFloatArray();
        assertTrue(GpuMatmul.gemm(a.dev, b.dev, 1.0f, 0.5f, c.dev));
        assertWithinGemm("transposed output", a.host, b.host, 1.0f, 0.5f, c0, c.dev.toFloatArray());
    }

    @Test
    public void gemmWithTheOutputAliasingAnInputIsStaged() {
        Pair x = operand(37, 37);
        Pair b = operand(37, 37);
        float[] x0 = x.host.toFloatArray();
        assertTrue(GpuMatmul.gemm(x.dev, b.dev, 1.0f, 1.0f, x.dev));
        assertWithinGemm("x = x @ b + x", x.host, b.host, 1.0f, 1.0f, x0, x.dev.toFloatArray());
    }

    @Test
    public void gemmIntoAColumnSliceLeavesTheRestUntouched() {
        Pair a = operand(17, 9);
        Pair b = operand(9, 20);
        Pair big = operand(17, 31);
        float[] before = big.host.toFloatArray();
        Tensor slot = keep(big.dev.slice(1, 4, 24));
        float[] c0 = keep(big.host.slice(1, 4, 24)).toFloatArray();
        assertTrue(GpuMatmul.gemm(a.dev, b.dev, 1.0f, 2.0f, slot));
        assertWithinGemm("column slice", a.host, b.host, 1.0f, 2.0f, c0, slot.toFloatArray());
        float[] after = big.dev.toFloatArray();
        for (int r = 0; r < 17; r++) {
            for (int col = 0; col < 31; col++) {
                if (col < 4 || col >= 24) {
                    int i = r * 31 + col;
                    assertEquals("untouched element " + i, Float.floatToRawIntBits(before[i]),
                            Float.floatToRawIntBits(after[i]));
                }
            }
        }
    }

    @Test
    public void gemmIntoAStridedColumnLeavesTheRestUntouched() {
        Pair a = operand(70, 12);
        Pair b = operand(12, 1);
        Pair big = operand(70, 5);
        float[] before = big.host.toFloatArray();
        Tensor slot = keep(big.dev.slice(1, 2, 3));
        float[] c0 = keep(big.host.slice(1, 2, 3)).toFloatArray();
        assertTrue(GpuMatmul.gemm(a.dev, b.dev, -0.5f, 1.5f, slot));
        assertWithinGemm("strided column", a.host, b.host, -0.5f, 1.5f, c0, slot.toFloatArray());
        float[] after = big.dev.toFloatArray();
        for (int i = 0; i < before.length; i++) {
            if (i % 5 != 2) {
                assertEquals("untouched element " + i, Float.floatToRawIntBits(before[i]),
                        Float.floatToRawIntBits(after[i]));
            }
        }
    }

    @Test
    public void gemmRejectsABroadcastOutput() {
        Tensor c = keep(keep(keep(Tensor.fromArray(gpu, new float[4])).reshape(1, 4)).expand(3, 4));
        try {
            GpuMatmul.gemm(operand(3, 2).dev, operand(2, 4).dev, 1.0f, 0.0f, c);
            fail("expected IllegalArgumentException for a broadcast output");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("more than once"));
        }
    }

    @Test
    public void gemmRefusesAnOutputThatRequiresGradWhileGradModeIsOn() {
        Pair a = operand(9, 7);
        Pair b = operand(7, 11);
        Pair c = operand(9, 11).map(t -> t.withRequiresGrad(true));
        float[] c0 = c.host.toFloatArray();
        try (ComputationGraph.EnableGradScope grad = ComputationGraph.enableGrad()) {
            IllegalStateException e = assertThrows(IllegalStateException.class,
                    () -> GpuMatmul.gemm(a.dev, b.dev, 1.0f, 0.5f, c.dev));
            assertTrue(e.getMessage(), e.getMessage().contains("ComputationGraph.noGrad()"));
            float[] after = c.dev.toFloatArray();
            for (int i = 0; i < c0.length; i++) {
                assertEquals("untouched element " + i, Float.floatToRawIntBits(c0[i]),
                        Float.floatToRawIntBits(after[i]));
            }
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            assertTrue(GpuMatmul.gemm(a.dev, b.dev, 1.0f, 0.5f, c.dev));
        }
        assertWithinGemm("gemm under noGrad", a.host, b.host, 1.0f, 0.5f, c0, c.dev.toFloatArray());
    }

    @Test
    public void gemmDeclinesNonFloatOutput() {
        Tensor c = keep(keep(Tensor.fromArray(gpu, new int[6])).reshape(2, 3));
        assertFalse(GpuMatmul.gemm(operand(2, 5).dev, operand(5, 3).dev, 1.0f, 0.0f, c));
    }

    // ------------------------------------------------------------------ helpers

    /** The same seeded values as a host tensor and a device tensor, transformed alike into views. */
    private final class Pair {
        final Tensor host;
        final Tensor dev;

        Pair(Tensor host, Tensor dev) {
            this.host = host;
            this.dev = dev;
        }

        Pair map(UnaryOperator<Tensor> view) {
            return new Pair(keep(view.apply(host)), keep(view.apply(dev)));
        }
    }

    private Pair operand(int... dims) {
        int n = Math.toIntExact(Shape.of(dims).numElements());
        float[] data = TrainParity.uniform(seed++, n, -2.0f, 2.0f);
        Tensor h = keep(keep(Tensor.fromArray(host, data)).reshape(Shape.of(dims)));
        Tensor d = keep(keep(Tensor.fromArray(gpu, data)).reshape(Shape.of(dims)));
        return new Pair(h, d);
    }

    private Tensor keep(Tensor t) {
        if (t != null) {
            owned.add(t);
        }
        return t;
    }

    private void checkMatmul(String what, Pair a, Pair b) {
        Tensor cpu = keep(TensorOps.matmul(a.host, b.host));
        Tensor dev = keep(GpuMatmul.matmul(a.dev, b.dev));
        assertNotNull(what + ": the train library is available, so the dispatcher must not decline", dev);
        assertEquals(what + ": shape", cpu.shape(), dev.shape());
        assertTrue(what + ": contiguous result", dev.isContiguous());
        assertWithinGemm(what, a.host, b.host, 1.0f, 0.0f, new float[dev.toFloatArray().length], dev.toFloatArray());
    }

    private void checkBmm(String what, Pair a, Pair b) {
        Tensor cpu = keep(TensorOps.bmm(a.host, b.host));
        Tensor dev = keep(GpuMatmul.bmm(a.dev, b.dev));
        assertNotNull(what, dev);
        assertEquals(what + ": shape", cpu.shape(), dev.shape());
        assertWithinGemm(what, a.host, b.host, 1.0f, 0.0f, new float[cpu.toFloatArray().length], dev.toFloatArray());
    }

    /**
     * Asserts {@code got == alpha * (a @ b) + beta * c0} (beta 0 ignores {@code c0}) within the GEMM class against a
     * double reference computed from the host operands' logical values with the CPU rank and broadcast rules.
     */
    private static void assertWithinGemm(String what, Tensor a, Tensor b, float alpha, float beta, float[] c0,
                                         float[] got) {
        Shape sa = a.shape();
        Shape sb = b.shape();
        float[] av = a.toFloatArray();
        float[] bv = b.toFloatArray();
        int k = sa.dim(sa.rank() - 1);
        int m = sa.rank() == 1 ? 1 : sa.dim(sa.rank() - 2);
        int n = sb.rank() == 1 ? 1 : sb.dim(sb.rank() - 1);
        int[] batchA = sa.rank() <= 2 ? new int[0] : Arrays.copyOf(sa.dims(), sa.rank() - 2);
        int[] batchB = sb.rank() <= 2 ? new int[0] : Arrays.copyOf(sb.dims(), sb.rank() - 2);
        int rank = Math.max(batchA.length, batchB.length);
        int[] batch = new int[rank];
        for (int i = 0; i < rank; i++) {
            batch[i] = Math.max(alignedDim(batchA, rank, i), alignedDim(batchB, rank, i));
        }
        long count = 1;
        for (int d : batch) {
            count *= d;
        }
        assertEquals(what + ": element count", count * m * n, got.length);
        assertEquals(what + ": initial output length", got.length, c0.length);
        int[] idx = new int[rank];
        for (int bi = 0; bi < count; bi++) {
            int rem = bi;
            for (int i = rank - 1; i >= 0; i--) {
                idx[i] = rem % batch[i];
                rem /= batch[i];
            }
            int aBase = broadcastLinear(idx, batchA) * m * k;
            int bBase = broadcastLinear(idx, batchB) * k * n;
            for (int mi = 0; mi < m; mi++) {
                for (int ni = 0; ni < n; ni++) {
                    double s = 0.0;
                    double abs = 0.0;
                    for (int kk = 0; kk < k; kk++) {
                        double t = (double) av[aBase + mi * k + kk] * bv[bBase + kk * n + ni];
                        s += t;
                        abs += Math.abs(t);
                    }
                    int o = (bi * m + mi) * n + ni;
                    double ref = alpha * s + (beta != 0.0f ? (double) beta * c0[o] : 0.0);
                    double terms = Math.abs(alpha) * abs + (beta != 0.0f ? Math.abs((double) beta * c0[o]) : 0.0);
                    int len = Math.max(k, 1);
                    double bound = 1e-5 * Math.sqrt(len) * terms + len * (double) TrainParity.FLT_MIN;
                    if (!(Math.abs(got[o] - ref) <= bound)) {
                        fail(what + ": element " + o + ": gpu " + got[o] + ", ref64 " + ref + ", bound " + bound);
                    }
                }
            }
        }
    }

    private static int alignedDim(int[] dims, int rank, int i) {
        int j = i - (rank - dims.length);
        return j < 0 ? 1 : dims[j];
    }

    /** Row-major index of a broadcast batch index into an operand's own batch dims (size-1 dims repeat). */
    private static int broadcastLinear(int[] idx, int[] dims) {
        int off = idx.length - dims.length;
        int lin = 0;
        for (int j = 0; j < dims.length; j++) {
            lin = lin * dims[j] + (dims[j] == 1 ? 0 : idx[off + j]);
        }
        return lin;
    }
}
