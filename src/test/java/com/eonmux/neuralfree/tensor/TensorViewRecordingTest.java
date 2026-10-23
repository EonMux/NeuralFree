package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradFunctions;
import com.eonmux.neuralfree.autograd.ViewGradFunctions;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Tensor views record: reshape, transpose, permute, slice, squeeze, unsqueeze, expand, contiguous and
 * the floating-point casts of {@code to(DType)} record their gradient function when grad mode is on and the operand
 * requires grad, so a gradient reaching a view flows back to the leaf; under {@code noGrad}, and for non-differentiable
 * casts, nothing records and the result does not require grad. Runs on {@link TrainParity#hostBackend()}.
 */
public class TensorViewRecordingTest {

    private ComputeBackend backend;
    private ComputationGraph graph;
    private final List<Tensor> owned = new ArrayList<>();

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
        backend = TrainParity.hostBackend();
        graph = ComputationGraph.current();
    }

    @After
    public void tearDown() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
        for (Tensor t : owned) {
            if (t.grad() != null) {
                t.grad().close();
            }
            t.close();
        }
        owned.clear();
        TrainGpu.release(backend);
        backend.close();
    }

    // ------------------------------------------------------------------ recording rule

    @Test
    public void everyViewRecordsItsGradientFunction() {
        Tensor leaf = leaf(new float[]{0f, 1f, 2f, 3f, 4f, 5f}, 2, 3);
        Tensor row = leaf(new float[]{1f, 2f, 3f}, 1, 3);
        Tensor strided = keep(keep(constant(new float[]{0f, 1f, 2f, 3f, 4f, 5f}, 2, 3).transpose())
            .withRequiresGrad(true));
        int before = graph.size();

        assertRecorded(leaf, leaf.reshape(3, 2), GradFunctions.ReshapeGrad.class);
        assertRecorded(leaf, leaf.transpose(), GradFunctions.TransposeGrad.class);
        assertRecorded(leaf, leaf.permute(1, 0), GradFunctions.TransposeGrad.class);
        assertRecorded(leaf, leaf.slice(1, 1, 3), GradFunctions.SliceGrad.class);
        assertRecorded(row, row.squeeze(), GradFunctions.ReshapeGrad.class);
        assertRecorded(row, row.squeeze(0), GradFunctions.ReshapeGrad.class);
        assertRecorded(leaf, leaf.unsqueeze(0), GradFunctions.ReshapeGrad.class);
        assertRecorded(row, row.expand(4, 3), ViewGradFunctions.ExpandGrad.class);
        assertRecorded(strided, strided.contiguous(), ViewGradFunctions.IdentityGrad.class);
        assertRecorded(leaf, leaf.to(DType.FLOAT16), ViewGradFunctions.CastGrad.class);
        assertRecorded(leaf, leaf.to(DType.BFLOAT16), ViewGradFunctions.CastGrad.class);

        assertEquals("one node per view", before + 11, graph.size());
        assertNull("the leaf stays a leaf", leaf.gradFn());
    }

    @Test
    public void noOpContiguousAndSameDtypeCastReturnTheTensorAndRecordNothing() {
        Tensor leaf = leaf(new float[]{1f, 2f, 3f, 4f}, 2, 2);
        int before = graph.size();

        assertSame(leaf, leaf.contiguous());
        assertSame(leaf, leaf.to(DType.FLOAT32));
        assertEquals(before, graph.size());
    }

    @Test
    public void viewsOfTensorsThatDoNotRequireGradRecordNothing() {
        Tensor constant = constant(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
        int before = graph.size();

        assertRaw(constant.reshape(6));
        assertRaw(constant.transpose());
        assertRaw(constant.slice(0, 0, 1));
        assertRaw(constant.unsqueeze(1));
        assertRaw(constant.to(DType.FLOAT16));
        assertEquals(before, graph.size());
    }

    @Test
    public void viewsUnderNoGradDoNotRequireGrad() {
        Tensor leaf = leaf(new float[]{0f, 1f, 2f, 3f, 4f, 5f}, 2, 3);
        Tensor row = leaf(new float[]{1f, 2f, 3f}, 1, 3);
        Tensor strided = keep(keep(constant(new float[]{0f, 1f, 2f, 3f, 4f, 5f}, 2, 3).transpose())
            .withRequiresGrad(true));
        int before = graph.size();

        try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
            assertRaw(leaf.reshape(6));
            assertRaw(leaf.transpose());
            assertRaw(leaf.permute(1, 0));
            assertRaw(leaf.slice(1, 0, 2));
            assertRaw(leaf.slice(1, 1, 1));
            assertRaw(row.squeeze());
            assertRaw(row.squeeze(0));
            assertRaw(leaf.unsqueeze(2));
            assertRaw(row.expand(5, 3));
            assertRaw(strided.contiguous());
            assertRaw(leaf.to(DType.FLOAT16));
            assertRaw(leaf.to(DType.BFLOAT16));
        }
        assertEquals("nothing recorded under noGrad", before, graph.size());
    }

    @Test
    public void nonFloatCastsNeverRecord() {
        Tensor leaf = leaf(new float[]{-1.5f, 0f, 2.5f}, 3);
        Tensor intLeaf = keep(keep(Tensor.fromArray(backend, new int[]{1, 2, 3})).withRequiresGrad(true));
        int before = graph.size();

        assertRaw(leaf.to(DType.INT32));
        assertRaw(leaf.to(DType.BOOL));
        assertRaw(leaf.to(DType.FLOAT64));
        assertRaw(intLeaf.to(DType.FLOAT32));
        assertEquals("non-differentiable casts record nothing", before, graph.size());
    }

    @Test
    public void emptySliceRecordsAndBackPropagatesZeros() {
        Tensor leaf = leaf(new float[]{1f, 2f, 3f, 4f}, 4);

        Tensor empty = keep(leaf.slice(0, 2, 2));
        assertEquals(Shape.vector(0), empty.shape());
        assertTrue(empty.requiresGrad());
        assertTrue(empty.gradFn() instanceof GradFunctions.SliceGrad);

        try (Tensor seed = Tensor.zeros(backend, Shape.vector(0))) {
            empty.backward(seed);
        }
        assertNotNull(leaf.grad());
        assertArrayEquals(new float[]{0f, 0f, 0f, 0f}, leaf.grad().toFloatArray(), 0f);
    }

    // ------------------------------------------------------------------ gradients reach the leaf

    @Test
    public void transposeRoutesTheGradientToTheLeaf() {
        Tensor leaf = leaf(new float[]{0f, 1f, 2f, 3f, 4f, 5f}, 2, 3);
        int before = graph.size();

        Tensor view = keep(leaf.transpose());
        try (Tensor seed = matrix(new float[]{10f, 40f, 20f, 50f, 30f, 60f}, 3, 2)) {
            view.backward(seed);
        }

        assertArrayEquals(new float[]{10f, 20f, 30f, 40f, 50f, 60f}, leaf.grad().toFloatArray(), 0f);
        assertEquals("backward releases the view's node", before, graph.size());
    }

    @Test
    public void sliceThenReshapeRoutesTheGradientIntoTheSlicedColumns() {
        Tensor leaf = leaf(new float[]{0f, 1f, 2f, 3f, 4f, 5f}, 2, 3);

        Tensor sliced = keep(leaf.slice(1, 1, 3));
        Tensor flat = keep(sliced.reshape(4));
        try (Tensor seed = Tensor.fromArray(backend, new float[]{1f, 2f, 3f, 4f})) {
            flat.backward(seed);
        }

        assertArrayEquals(new float[]{0f, 1f, 2f, 0f, 3f, 4f}, leaf.grad().toFloatArray(), 0f);
    }

    @Test
    public void expandSumsTheGradientOverTheBroadcastRows() {
        Tensor leaf = leaf(new float[]{1f, 2f, 3f}, 1, 3);

        Tensor expanded = keep(leaf.expand(2, 3));
        try (Tensor seed = matrix(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3)) {
            expanded.backward(seed);
        }

        assertEquals(Shape.of(1, 3), leaf.grad().shape());
        assertArrayEquals(new float[]{5f, 7f, 9f}, leaf.grad().toFloatArray(), 0f);
    }

    @Test
    public void contiguousCopyPassesTheGradientThrough() {
        Tensor strided = keep(keep(constant(new float[]{0f, 1f, 2f, 3f, 4f, 5f}, 2, 3).transpose())
            .withRequiresGrad(true));

        Tensor dense = keep(strided.contiguous());
        assertTrue(dense.isContiguous());
        try (Tensor seed = matrix(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 3, 2)) {
            dense.backward(seed);
        }

        assertEquals(Shape.of(3, 2), strided.grad().shape());
        assertArrayEquals(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, strided.grad().toFloatArray(), 0f);
    }

    @Test
    public void halfPrecisionRoundTripGivesAFloat32Gradient() {
        Tensor leaf = leaf(new float[]{0.5f, -1f, 2f}, 3);

        Tensor half = keep(leaf.to(DType.FLOAT16));
        Tensor back = keep(half.to(DType.FLOAT32));
        assertTrue(back.gradFn() instanceof ViewGradFunctions.CastGrad);
        try (Tensor seed = Tensor.fromArray(backend, new float[]{3f, 4f, 5f})) {
            back.backward(seed);
        }

        assertEquals(DType.FLOAT32, leaf.grad().dtype());
        assertArrayEquals(new float[]{3f, 4f, 5f}, leaf.grad().toFloatArray(), 0f);
    }

    @Test
    public void transposedWeightThenMatmulRoutesTheGradientToTheLeaf() {
        Tensor weight = leaf(new float[]{1f, 0f, 2f, -1f, 3f, 1f, 0f, 2f, 1f, 1f, 1f, 1f}, 4, 3);
        Tensor x = keep(matrix(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3));

        Tensor product;
        try (Tensor weightT = weight.transpose()) {
            product = keep(TensorOps.matmul(x, weightT));
        }
        try (Tensor seed = Tensor.ones(backend, Shape.of(2, 4))) {
            product.backward(seed);
        }

        assertNull("the weight stays a leaf", weight.gradFn());
        assertEquals(Shape.of(4, 3), weight.grad().shape());
        assertArrayEquals("every row is the column sums of x", new float[]{5f, 7f, 9f, 5f, 7f, 9f, 5f, 7f, 9f, 5f, 7f,
            9f}, weight.grad().toFloatArray(), 1e-6f);
    }

    @Test
    public void linearForwardDeliversTheWeightGradientToItsParameter() {
        try (Linear linear = new Linear(3, 4)) {
            linear.initialize(backend, new Random(7L));
            Tensor x = keep(matrix(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3));

            Tensor y = keep(linear.forward(x));
            assertTrue(y.requiresGrad());
            try (Tensor seed = Tensor.ones(backend, Shape.of(2, 4))) {
                y.backward(seed);
            }

            Parameter weight = linear.weight();
            assertNull(weight.tensor().gradFn());
            assertNotNull("the gradient reaches the parameter, not the transposed view", weight.grad());
            assertArrayEquals(new float[]{5f, 7f, 9f, 5f, 7f, 9f, 5f, 7f, 9f, 5f, 7f, 9f},
                weight.grad().toFloatArray(), 1e-6f);
            assertArrayEquals(new float[]{2f, 2f, 2f, 2f}, linear.bias().grad().toFloatArray(), 0f);
            linear.zeroGrad();
        }
    }

    @Test
    public void moduleToFloat16InGradModeKeepsEveryParameterALeaf() {
        try (Linear linear = new Linear(3, 2)) {
            linear.initialize(backend, new Random(11L));
            int before = graph.size();

            assertTrue(ComputationGraph.isGradEnabled());
            linear.to(DType.FLOAT16);

            for (Parameter param : linear.parameters()) {
                assertEquals(DType.FLOAT16, param.dtype());
                assertTrue(param.tensor().requiresGrad());
                assertNull("a module conversion never makes a parameter a node output", param.tensor().gradFn());
            }
            assertEquals(before, graph.size());
        }
    }

    // ------------------------------------------------------------------ in-place guard

    @Test
    public void copyIntoRefusesADestinationThatRequiresGradInGradMode() {
        Tensor leaf = leaf(new float[]{1f, 2f, 3f}, 3);
        Tensor src = constant(new float[]{4f, 5f, 6f}, 3);

        IllegalStateException e = assertThrows(IllegalStateException.class, () -> GpuCoreOps.copyInto(src, leaf));
        assertTrue(e.getMessage(), e.getMessage().startsWith("in-place write to a tensor that requires grad"));
        assertArrayEquals(new float[]{1f, 2f, 3f}, leaf.toFloatArray(), 0f);
    }

    // ------------------------------------------------------------------ helpers

    private void assertRecorded(Tensor input, Tensor view, Class<?> gradFn) {
        keep(view);
        assertNotSame(input, view);
        assertTrue(view.requiresGrad());
        assertNotNull(view.gradFn());
        assertTrue("expected " + gradFn.getSimpleName() + ", got " + view.gradFn().getClass().getSimpleName(),
            gradFn.isInstance(view.gradFn()));
    }

    private void assertRaw(Tensor view) {
        keep(view);
        assertFalse(view.requiresGrad());
        assertNull(view.gradFn());
    }

    /** A FLOAT32 leaf that requires grad with {@code values} in {@code dims}. */
    private Tensor leaf(float[] values, int... dims) {
        return keep(constant(values, dims).withRequiresGrad(true));
    }

    private Tensor constant(float[] values, int... dims) {
        Tensor flat = keep(Tensor.fromArray(backend, values));
        return keep(flat.reshape(Shape.of(dims)));
    }

    /** A new FLOAT32 tensor the caller closes. */
    private Tensor matrix(float[] values, int rows, int cols) {
        try (Tensor flat = Tensor.fromArray(backend, values)) {
            return flat.reshape(Shape.of(rows, cols));
        }
    }

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }
}
