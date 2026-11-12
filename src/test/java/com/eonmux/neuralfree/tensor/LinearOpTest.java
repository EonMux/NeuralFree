package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link TensorOps#linear} on the host backend: the bits of the matmul-and-add composition, one recorded node, the
 * composition's gradients for matrix, batched and vector inputs with and without bias, zero gradients for a weight of
 * no rows, and the shape checks.
 */
public class LinearOpTest {

    private ComputeBackend backend;
    private LinearOpChecks checks;

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        backend = TrainParity.hostBackend();
        checks = new LinearOpChecks(backend);
    }

    @After
    public void tearDown() {
        try {
            checks.close();
        } finally {
            backend.close();
        }
    }

    @Test
    public void matchesMatmulAndAdd() {
        checks.matchesTheComposition(new int[]{12}, true);
        checks.matchesTheComposition(new int[]{12}, false);
        checks.matchesTheComposition(new int[]{2, 6}, true);
        checks.matchesTheComposition(new int[]{}, true);
    }

    @Test
    public void anEmptyWeightGivesZeroGradients() {
        checks.anEmptyWeightGivesZeroGradients(true);
        checks.anEmptyWeightGivesZeroGradients(false);
    }

    @Test
    public void computesFloat64OnTheHost() {
        double[] xv = {0.1, -0.2, 0.3, 0.4, 0.5, -0.6};
        double[] wv = {1.5, -2.0, 0.25, 0.75, 1.0, -1.0};
        double[] bv = {0.125, -0.5};
        try (Tensor xFlat = Tensor.fromArray(backend, xv);
             Tensor x = xFlat.reshape(Shape.of(2, 3));
             Tensor wFlat = Tensor.fromArray(backend, wv);
             Tensor w = wFlat.reshape(Shape.of(2, 3));
             Tensor b = Tensor.fromArray(backend, bv);
             Tensor y = TensorOps.linear(x, w, b)) {
            assertEquals(DType.FLOAT64, y.dtype());
            double[] expected = new double[4];
            for (int r = 0; r < 2; r++) {
                for (int o = 0; o < 2; o++) {
                    double sum = 0;
                    for (int k = 0; k < 3; k++) {
                        sum += xv[r * 3 + k] * wv[o * 3 + k];
                    }
                    expected[r * 2 + o] = sum + bv[o];
                }
            }
            assertArrayEquals(expected, y.toDoubleArray(), 1e-15);
        }
    }

    @Test
    public void refusesMismatchedShapes() {
        try (Tensor x = Tensor.zeros(backend, Shape.of(3, 4));
             Tensor w = Tensor.zeros(backend, Shape.of(5, 4));
             Tensor wrongW = Tensor.zeros(backend, Shape.of(5, 3));
             Tensor vectorW = Tensor.zeros(backend, Shape.of(4));
             Tensor wrongBias = Tensor.zeros(backend, Shape.of(4))) {
            assertThrows(IllegalArgumentException.class, () -> TensorOps.linear(x, wrongW, null));
            assertThrows(IllegalArgumentException.class, () -> TensorOps.linear(x, vectorW, null));
            assertThrows(IllegalArgumentException.class, () -> TensorOps.linear(x, w, wrongBias));
            assertThrows(NullPointerException.class, () -> TensorOps.linear(null, w, null));
        }
    }
}
