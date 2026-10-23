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
 * {@link NormOps#l2Normalize} against {@code torch.nn.functional.normalize(x, p=2, dim, eps)}: values
 * {@code x / max(||x||, eps)} along the dimension, the gradient {@code (g - y (y . g)) / ||x||} of a row whose norm
 * exceeds {@code eps}, and the finite gradient {@code g / eps} of an all-zero row.
 */
public class NormOpsL2NormalizeTest {
    private ComputeBackend backend;

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        backend = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        ComputationGraph.current().clear();
        backend.close();
    }

    @Test
    public void normalizesAlongTheGivenDimension() {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor x = Tensor.fromArray(backend, new float[][]{{3f, 4f}, {0f, 0f}, {-1f, 0f}});
             Tensor rows = NormOps.l2Normalize(x, -1);
             Tensor columns = NormOps.l2Normalize(x, 0)) {
            assertArrayEquals(new float[]{0.6f, 0.8f, 0f, 0f, -1f, 0f}, rows.toFloatArray(), 1e-6f);
            float c0 = (float) Math.sqrt(10.0);
            assertArrayEquals(new float[]{3f / c0, 1f, 0f, 0f, -1f / c0, 0f}, columns.toFloatArray(), 1e-6f);
        }
    }

    @Test
    public void gradientMatchesTheClosedFormAndStaysFiniteOnAZeroRow() {
        ComputationGraph graph = ComputationGraph.current();
        try (Tensor values = Tensor.fromArray(backend, new float[][]{{3f, 4f}, {0f, 0f}});
             Tensor x = values.withRequiresGrad(true);
             Tensor weights = Tensor.fromArray(backend, new float[][]{{1f, 2f}, {3f, -1f}})) {
            Tensor y = NormOps.l2Normalize(x, 1);
            Tensor weighted = TensorOps.mul(y, weights);
            Tensor loss = TensorOps.sum(weighted);
            graph.backward(loss, null);
            float[] grad = x.grad().toFloatArray();
            float dot = 0.6f * 1f + 0.8f * 2f;
            assertEquals((1f - 0.6f * dot) / 5f, grad[0], 1e-6f);
            assertEquals((2f - 0.8f * dot) / 5f, grad[1], 1e-6f);
            float eps = NormOps.L2_NORMALIZE_EPS;
            assertEquals(3f / eps, grad[2], 1e-5f * (3f / eps));
            assertEquals(-1f / eps, grad[3], 1e-5f * (1f / eps));
            loss.close();
            graph.zeroGrad();
        }
    }

    @Test
    public void rejectsAnEpsWithoutAPositiveFiniteSquare() {
        try (Tensor x = Tensor.fromArray(backend, new float[]{1f, 2f})) {
            assertThrows(IllegalArgumentException.class, () -> NormOps.l2Normalize(x, 0, 0f));
            assertThrows(IllegalArgumentException.class, () -> NormOps.l2Normalize(x, 0, 1e-30f));
            assertThrows(IllegalArgumentException.class, () -> NormOps.l2Normalize(x, 0, Float.NaN));
            assertThrows(IndexOutOfBoundsException.class, () -> NormOps.l2Normalize(x, 1));
        }
    }
}
