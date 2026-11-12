package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The checks of {@link TensorOps#linear} shared by the host test and the per-API parity test: the result has the bits
 * of {@code add(matmul(x, w^T), bias)} on the same backend, the op records one node, and its gradients have the bits of
 * that composition's (the weight gradient dense in the weight's {@code [out, in]} layout): the same products summed in
 * the same order. The one exception is the weight gradient of a batched input, which {@code linear} sums over all rows
 * in one pass while the composition sums each batch matrix and then adds the batch sums; it is compared in the
 * reduction tolerance class. A weight of no rows gives zero gradients of the operands' shapes.
 */
final class LinearOpChecks {

    /** Relative bound of the batched weight gradient (reduction class, sums of at most 48 products). */
    private static final double BATCHED_WEIGHT_GRAD_TOLERANCE = 1e-5 * Math.sqrt(48);

    private final ComputeBackend backend;
    private final List<Tensor> owned = new ArrayList<>();

    LinearOpChecks(ComputeBackend backend) {
        this.backend = backend;
    }

    void close() {
        ComputationGraph.current().clear();
        for (Tensor t : owned) {
            t.close();
        }
        owned.clear();
    }

    /** The forward bits and gradients of {@code linear} for an input of {@code leading} rows over 24 features. */
    void matchesTheComposition(int[] leading, boolean withBias) {
        int in = 24;
        int out = 20;
        int rows = 1;
        for (int d : leading) {
            rows *= d;
        }
        int[] xDims = new int[leading.length + 1];
        System.arraycopy(leading, 0, xDims, 0, leading.length);
        xDims[leading.length] = in;
        float[] xv = TrainParity.uniform(31, rows * in, -1f, 1f);
        float[] wv = TrainParity.uniform(32, out * in, -1f, 1f);
        float[] bv = TrainParity.uniform(33, out, -1f, 1f);
        float[] gv = TrainParity.uniform(34, rows * out, -1f, 1f);
        String what = "linear over " + Arrays.toString(xDims) + (withBias ? " with bias" : "");

        Tensor[] fused = leaves(xv, xDims, wv, out, in, bv, withBias);
        int before = ComputationGraph.current().size();
        Tensor y = keep(TensorOps.linear(fused[0], fused[1], withBias ? fused[2] : null));
        assertEquals(what + ": one recorded node", before + 1, ComputationGraph.current().size());

        Tensor[] composed = leaves(xv, xDims, wv, out, in, bv, withBias);
        Tensor product;
        try (Tensor wT = composed[1].transpose()) {
            product = keep(TensorOps.matmul(composed[0], wT));
        }
        Tensor reference = withBias ? keep(TensorOps.add(product, composed[2])) : product;
        assertEquals(what + ": shape", reference.shape(), y.shape());
        TrainParity.assertBitIdentical(what + ": forward", reference.toFloatArray(), y.toFloatArray());

        Tensor seed = keep(Tensor.fromArray(backend, gv).reshape(y.shape()));
        ComputationGraph.current().backward(y, seed, false);
        ComputationGraph.current().backward(reference, seed, false);
        assertSameBits(what + ": input gradient", composed[0].grad(), fused[0].grad());
        Tensor gw = fused[1].grad();
        assertTrue(what + ": the weight gradient is dense", gw.isDense());
        assertEquals(what + ": weight gradient shape", Shape.of(out, in), gw.shape());
        if (leading.length <= 1) {
            assertSameBits(what + ": weight gradient", composed[1].grad(), gw);
        } else {
            assertReordered(what + ": weight gradient", composed[1].grad(), gw);
        }
        if (withBias) {
            assertSameBits(what + ": bias gradient", composed[2].grad(), fused[2].grad());
        }
    }

    /**
     * {@code linear} with a weight of no rows ({@code [0, 3]}) over a {@code [2, 3]} input: a {@code [2, 0]} output,
     * and a backward pass that gives the input a zero gradient of its shape and the weight and bias empty gradients of
     * theirs.
     */
    void anEmptyWeightGivesZeroGradients(boolean withBias) {
        int in = 3;
        String what = "linear with a [0, " + in + "] weight" + (withBias ? " and bias" : "");
        Tensor x = leaf(TrainParity.uniform(41, 2 * in, -1f, 1f), 2, in);
        Tensor w = leaf(new float[0], 0, in);
        Tensor b = withBias ? leaf(new float[0], 0) : null;
        Tensor y = keep(TensorOps.linear(x, w, b));
        assertEquals(what + ": output shape", Shape.of(2, 0), y.shape());

        Tensor seed = keep(Tensor.fromArray(backend, new float[0]).reshape(y.shape()));
        ComputationGraph.current().backward(y, seed, false);
        assertEquals(what + ": input gradient shape", x.shape(), x.grad().shape());
        assertArrayEquals(what + ": input gradient", new float[2 * in], x.grad().toFloatArray(), 0f);
        assertEquals(what + ": weight gradient shape", w.shape(), w.grad().shape());
        if (withBias) {
            assertEquals(what + ": bias gradient shape", b.shape(), b.grad().shape());
        }
    }

    private Tensor[] leaves(float[] xv, int[] xDims, float[] wv, int out, int in, float[] bv, boolean withBias) {
        Tensor x = leaf(xv, xDims);
        Tensor w = leaf(wv, out, in);
        Tensor b = withBias ? leaf(bv, out) : null;
        return new Tensor[]{x, w, b};
    }

    private Tensor leaf(float[] values, int... dims) {
        Tensor flat = keep(Tensor.fromArray(backend, values));
        Tensor shaped = keep(flat.reshape(Shape.of(dims)));
        return keep(shaped.withRequiresGrad(true));
    }

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }

    private static void assertSameBits(String what, Tensor expected, Tensor actual) {
        assertEquals(what + ": shape", expected.shape(), actual.shape());
        TrainParity.assertBitIdentical(what, expected.toFloatArray(), actual.toFloatArray());
    }

    private static void assertReordered(String what, Tensor expected, Tensor actual) {
        assertEquals(what + ": shape", expected.shape(), actual.shape());
        float[] e = expected.toFloatArray();
        float[] a = actual.toFloatArray();
        for (int i = 0; i < e.length; i++) {
            double bound = BATCHED_WEIGHT_GRAD_TOLERANCE * Math.max(1.0, Math.abs(e[i]));
            assertEquals(what + "[" + i + "]", e[i], a[i], bound);
        }
    }
}
