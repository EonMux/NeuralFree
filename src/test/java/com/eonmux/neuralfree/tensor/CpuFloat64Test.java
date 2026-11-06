package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The FLOAT64 host path of the forward ops ({@code CpuFloat64}, {@code CpuFloat64Conv}): every op gives the shape of
 * its FLOAT32 path and the same values to float rounding, computes in double precision (results a float path cannot
 * produce), returns FLOAT64 tensors that do not require grad, and refuses FLOAT64 mixed with another dtype.
 */
public class CpuFloat64Test {

    private final List<Tensor> owned = new ArrayList<>();
    private ComputeBackend backend;

    @Before
    public void setUp() {
        backend = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        for (Tensor t : owned) {
            t.close();
        }
        owned.clear();
        backend.close();
    }

    // ------------------------------------------------------------------ values and storage

    @Test
    public void fromArrayAndToDoubleArrayRoundTripIncludingViews() {
        double[] data = {1.0 / 3, -2.5, Math.PI, 1e-300, 7.0, -0.0};
        Tensor t = keep(Tensor.fromArray(backend, data));
        assertEquals(DType.FLOAT64, t.dtype());
        assertArrayEquals(data, t.toDoubleArray(), 0.0);
        try (Tensor m = t.reshape(2, 3); Tensor tr = m.transpose()) {
            assertArrayEquals(new double[]{1.0 / 3, 1e-300, -2.5, 7.0, Math.PI, -0.0}, tr.toDoubleArray(), 0.0);
        }
        Tensor floats = keep(Tensor.fromArray(backend, new float[]{1f}));
        assertThrows(UnsupportedOperationException.class, floats::toDoubleArray);
    }

    @Test
    public void storeDoublesWritesDenseAndStridedViews() {
        Tensor t = keep(Tensor.fromArray(backend, new double[6]));
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor m = t.reshape(2, 3);
             Tensor column = m.slice(1, 1, 2)) {
            GpuOptimizerOps.storeDoubles(column, new double[]{0.1, 0.2});
            assertArrayEquals(new double[]{0, 0.1, 0, 0, 0.2, 0}, t.toDoubleArray(), 0.0);
            GpuOptimizerOps.storeDoubles(t, new double[]{1, 2, 3, 4, 5, 6});
            assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, t.toDoubleArray(), 0.0);
            assertThrows(IllegalArgumentException.class, () -> GpuOptimizerOps.storeDoubles(t, new double[5]));
            Tensor floats = keep(Tensor.fromArray(backend, new float[6]));
            assertThrows(IllegalArgumentException.class, () -> GpuOptimizerOps.storeDoubles(floats, new double[6]));
        }
    }

    // ------------------------------------------------------------------ elementwise and reductions

    @Test
    public void broadcastArithmeticMatchesFloatPath() {
        Tensor a = data(1, 2, 3, 4);
        Tensor b = data(2, 1, 4);
        same(x -> TensorOps.add(x.get(0), x.get(1)), a, b);
        same(x -> TensorOps.sub(x.get(0), x.get(1)), a, b);
        same(x -> TensorOps.mul(x.get(0), x.get(1)), a, b);
        same(x -> TensorOps.div(x.get(0), x.get(1)), a, positive(3, 1, 4));
    }

    @Test
    public void unaryFunctionsMatchFloatPath() {
        Tensor a = data(4, 3, 5);
        Tensor p = positive(6, 3, 5);
        same(x -> TensorOps.neg(x.get(0)), a);
        same(x -> TensorOps.abs(x.get(0)), a);
        same(x -> TensorOps.sqrt(x.get(0)), p);
        same(x -> TensorOps.exp(x.get(0)), a);
        same(x -> TensorOps.log(x.get(0)), p);
        same(x -> TensorOps.sin(x.get(0)), a);
        same(x -> TensorOps.cos(x.get(0)), a);
        same(x -> TensorOps.pow(x.get(0), 3.0f), a);
        same(x -> TensorOps.clamp(x.get(0), -0.3f, 0.4f), a);
        same(x -> TensorOps.add(x.get(0), 0.75f), a);
        same(x -> TensorOps.mul(x.get(0), -1.5f), a);
        same(x -> TensorOps.relu(x.get(0)), a);
        same(x -> TensorOps.gelu(x.get(0)), a);
        same(x -> TensorOps.silu(x.get(0)), a);
        same(x -> TensorOps.sigmoid(x.get(0)), a);
        same(x -> TensorOps.tanh(x.get(0)), a);
        same(x -> TensorOps.leakyRelu(x.get(0), 0.1f), a);
        same(x -> TensorOps.elu(x.get(0), 0.7f), a);
        same(x -> TensorOps.softplus(x.get(0)), a);
        same(x -> TensorOps.mish(x.get(0)), a);
    }

    @Test
    public void sumAndMeanMatchFloatPathOverAnyDims() {
        Tensor a = data(7, 2, 3, 4);
        same(x -> TensorOps.sum(x.get(0)), a);
        same(x -> TensorOps.mean(x.get(0)), a);
        same(x -> TensorOps.sum(x.get(0), 1), a);
        same(x -> TensorOps.sum(x.get(0), 0, 2), a);
        same(x -> TensorOps.mean(x.get(0), -1), a);
        same(x -> TensorOps.mean(x.get(0), 2, 0), a);
    }

    /** {@code 1 + 2^-40} and {@code 1/3} come out exact in double; a float result cannot hold them. */
    @Test
    public void computesInDoublePrecision() {
        double tiny = Math.scalb(1.0, -40);
        Tensor one = keep(Tensor.fromArray(backend, new double[]{1.0}));
        Tensor small = keep(Tensor.fromArray(backend, new double[]{tiny}));
        Tensor sum = keep(TensorOps.add(one, small));
        assertEquals(DType.FLOAT64, sum.dtype());
        assertEquals(1.0 + tiny, sum.toDoubleArray()[0], 0.0);
        assertTrue((float) (1.0 + tiny) == 1.0f);
        Tensor third = keep(Tensor.fromArray(backend, new double[]{1.0}));
        Tensor three = keep(Tensor.fromArray(backend, new double[]{3.0}));
        assertEquals(1.0 / 3.0, keep(TensorOps.div(third, three)).toDoubleArray()[0], 0.0);
    }

    // ------------------------------------------------------------------ products and shape

    @Test
    public void matmulAndBmmMatchFloatPath() {
        same(x -> TensorOps.matmul(x.get(0), x.get(1)), data(8, 3, 4), data(9, 4, 5));
        same(x -> TensorOps.matmul(x.get(0), x.get(1)), data(10, 4), data(11, 4, 5));
        same(x -> TensorOps.matmul(x.get(0), x.get(1)), data(12, 3, 4), data(13, 4));
        same(x -> TensorOps.matmul(x.get(0), x.get(1)), data(14, 2, 1, 3, 4), data(15, 3, 4, 2));
        same(x -> TensorOps.bmm(x.get(0), x.get(1)), data(16, 2, 3, 4), data(17, 2, 4, 5));
        Tensor a = data(18, 3, 4);
        try (Tensor t = a.transpose()) {
            same(x -> TensorOps.matmul(x.get(0), x.get(1)), data(19, 2, 4), t);
        }
    }

    @Test
    public void catMatchesFloatPath() {
        same(x -> TensorOps.cat(Arrays.asList(x.get(0), x.get(1)), 1), data(20, 2, 3, 2), data(21, 2, 1, 2));
        same(x -> TensorOps.cat(Arrays.asList(x.get(0), x.get(1)), 0), data(22, 1, 3), data(23, 2, 3));
    }

    // ------------------------------------------------------------------ convolution, pooling, normalization

    @Test
    public void convolutionsMatchFloatPath() {
        same(x -> ConvOps.conv2d(x.get(0), x.get(1), x.get(2), new int[]{3, 3}, new int[]{2, 1}, new int[]{1, 2},
            new int[]{1, 2}, 2), data(30, 2, 4, 6, 7), data(31, 6, 2, 3, 3), data(32, 6));
        same(x -> ConvOps.conv2d(x.get(0), x.get(1), null, new int[]{1, 1}, new int[]{1, 1}, new int[]{0, 0},
            new int[]{1, 1}, 1), data(33, 2, 3, 4, 4), data(34, 5, 3, 1, 1));
        same(x -> ConvOps.conv1d(x.get(0), x.get(1), x.get(2), 3, 2, 1, 2, 1), data(35, 2, 3, 9), data(36, 4, 3, 3),
            data(37, 4));
        same(x -> ConvOps.convTranspose2d(x.get(0), x.get(1), x.get(2), new int[]{3, 2}, new int[]{2, 2},
            new int[]{1, 0}, new int[]{1, 1}, new int[]{1, 1}, 2), data(38, 2, 4, 3, 3), data(39, 4, 3, 3, 2),
            data(40, 6));
    }

    @Test
    public void poolingMatchesFloatPath() {
        Tensor x = data(41, 2, 3, 7, 6);
        same(t -> ConvOps.maxPool2d(t.get(0), new int[]{3, 2}, new int[]{2, 2}, new int[]{1, 1}, new int[]{1, 2},
            true), x);
        same(t -> ConvOps.avgPool2d(t.get(0), new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1}, true, false), x);
        same(t -> ConvOps.avgPool2d(t.get(0), new int[]{2, 3}, new int[]{2, 1}, new int[]{1, 0}, false, true), x);
        same(t -> ConvOps.adaptiveAvgPool2d(t.get(0), 3, 4), x);
        same(t -> ConvOps.adaptiveAvgPool2d(t.get(0), 1, 1), x);
    }

    @Test
    public void batchNormMatchesFloatPathAndUpdatesRunningStatistics() {
        Tensor x = data(50, 3, 4, 2, 2);
        Tensor gamma = data(51, 4);
        Tensor beta = data(52, 4);
        Tensor mean32 = data(53, 4);
        Tensor var32 = positive(54, 4);
        Tensor mean64 = keep(mean32.to(DType.FLOAT64));
        Tensor var64 = keep(var32.to(DType.FLOAT64));
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor out32 = keep(NormOps.batchNorm(x, gamma, beta, mean32, var32, true, 0.1f, 1e-5f));
            Tensor out64 = keep(NormOps.batchNorm(keep(x.to(DType.FLOAT64)), keep(gamma.to(DType.FLOAT64)),
                keep(beta.to(DType.FLOAT64)), mean64, var64, true, 0.1f, 1e-5f));
            close("training output", out32, out64);
            close("running mean", mean32, mean64);
            close("running var", var32, var64);
        }
        same(t -> NormOps.batchNorm(t.get(0), t.get(1), t.get(2), t.get(3), t.get(4), false, 0f, 1e-5f), x, gamma,
            beta, mean32, var32);
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void refusesFloat64MixedWithAnotherDtype() {
        Tensor f = data(60, 2, 3);
        Tensor d = keep(f.to(DType.FLOAT64));
        IllegalArgumentException add = assertThrows(IllegalArgumentException.class, () -> TensorOps.add(f, d));
        assertTrue(add.getMessage(), add.getMessage().contains("add") && add.getMessage().contains("FLOAT32"));
        assertThrows(IllegalArgumentException.class, () -> TensorOps.matmul(d, keep(data(61, 3, 2))));
        Tensor w = data(62, 2, 2, 1, 1);
        Tensor x = keep(data(63, 1, 2, 3, 3).to(DType.FLOAT64));
        IllegalArgumentException conv = assertThrows(IllegalArgumentException.class, () -> ConvOps.conv2d(x, w,
            null, new int[]{1, 1}, new int[]{1, 1}, new int[]{0, 0}, new int[]{1, 1}, 1));
        assertTrue(conv.getMessage(), conv.getMessage().contains("conv2d"));
    }

    // ------------------------------------------------------------------ support

    /**
     * Applies {@code op} to {@code inputs} (FLOAT32) and to their FLOAT64 copies and checks that the FLOAT64 result
     * has the FLOAT32 result's shape and values to float rounding.
     */
    private void same(Function<List<Tensor>, Tensor> op, Tensor... inputs) {
        List<Tensor> doubles = new ArrayList<>();
        for (Tensor t : inputs) {
            doubles.add(keep(t.to(DType.FLOAT64)));
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            close("op", keep(op.apply(Arrays.asList(inputs))), keep(op.apply(doubles)));
        }
    }

    /** Checks that the FLOAT64 {@code d} has {@code f}'s shape and its values within float rounding. */
    private static void close(String what, Tensor f, Tensor d) {
        assertEquals(what + " dtype", DType.FLOAT64, d.dtype());
        assertEquals(what + " shape", f.shape(), d.shape());
        assertTrue(what + " does not require grad", !d.requiresGrad());
        float[] expected = f.toFloatArray();
        double[] actual = d.toDoubleArray();
        for (int i = 0; i < expected.length; i++) {
            double bound = 1e-5 * (1.0 + Math.abs(actual[i]));
            if (Double.isInfinite(actual[i])) {
                assertEquals(what + " element " + i, expected[i], actual[i], 0.0);
            } else {
                assertEquals(what + " element " + i, expected[i], actual[i], bound);
            }
        }
    }

    /** A FLOAT32 tensor of {@code dims} uniform in [-1, 1] from {@code seed}. */
    private Tensor data(long seed, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        try (Tensor flat = Tensor.fromArray(backend, TrainParity.uniform(seed, n, -1f, 1f))) {
            return keep(flat.reshape(dims));
        }
    }

    /** A FLOAT32 tensor of {@code dims} uniform in [0.5, 2] from {@code seed}. */
    private Tensor positive(long seed, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        try (Tensor flat = Tensor.fromArray(backend, TrainParity.uniform(seed, n, 0.5f, 2f))) {
            return keep(flat.reshape(dims));
        }
    }

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }
}
