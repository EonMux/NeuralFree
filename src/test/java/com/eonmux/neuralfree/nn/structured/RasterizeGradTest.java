package com.eonmux.neuralfree.nn.structured;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The host rasterizer node of {@link GaussianSplatting}: {@link RasterizeGrad}'s splat gradient against central
 * differences of {@link RasterizeGrad#rasterize}, a closed-form single-splat case, its argument and lifetime checks, and
 * the recorded node inside {@link GaussianSplatting#render}.
 *
 * <p>The splats sit at half-pixel centres with variances whose three-sigma radii are away from integers, so the
 * perturbations of the differences never move a box bound (the bounds are constants of the gradient). The differences
 * use the float forward with steps of 0.02 pixel for positions, 0.01 for variances and covariance and 0.002 for
 * opacities and colours; the float rounding of the summed loss divided by the step and the second-order truncation
 * stay below the bound {@code 1e-2 * max(1, |fd|)}.</p>
 */
public class RasterizeGradTest {

    private static final int WIDTH = 12;
    private static final int HEIGHT = 10;
    private static final int GAUSSIANS = 3;
    private static final double TOL = 1e-2;
    /** Rows {@code [px, py, varX, covXY, varY, opacity, r, g, b]} of three overlapping splats. */
    private static final float[] SPLATS = {
        4.5f, 4.5f, 2.0f, 0.3f, 1.5f, 0.7f, 0.9f, 0.2f, 0.4f,
        6.5f, 5.5f, 1.6f, -0.4f, 2.4f, 0.55f, 0.1f, 0.8f, 0.3f,
        5.5f, 3.5f, 3.0f, 0.0f, 1.2f, 0.8f, 0.5f, 0.5f, 0.9f
    };
    private static final float[] STEPS = {0.02f, 0.02f, 0.01f, 0.01f, 0.01f, 0.002f, 0.002f, 0.002f, 0.002f};

    private ComputeBackend host;

    @Before
    public void setUp() {
        host = TrainParity.hostBackend();
        ComputationGraph.setGradEnabled(true);
    }

    @After
    public void tearDown() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
        TrainGpu.release(host);
    }

    @Test
    public void gradientMatchesCentralDifferences() {
        float[] gy = TrainParity.uniform(7, HEIGHT * WIDTH * 4, -1, 1);
        float[] analytic = analyticGradient(SPLATS, gy);
        for (int i = 0; i < SPLATS.length; i++) {
            float[] plus = SPLATS.clone();
            float[] minus = SPLATS.clone();
            plus[i] += STEPS[i % RasterizeGrad.FIELDS];
            minus[i] -= STEPS[i % RasterizeGrad.FIELDS];
            double fd = (loss(plus, gy) - loss(minus, gy)) / ((double) plus[i] - (double) minus[i]);
            double bound = TOL * Math.max(1.0, Math.abs(fd));
            if (!(Math.abs(analytic[i] - fd) <= bound)) {
                fail("splat " + i / RasterizeGrad.FIELDS + " field " + i % RasterizeGrad.FIELDS + ": analytic "
                    + analytic[i] + ", central difference " + fd + " (bound " + bound + ")");
            }
        }
    }

    /** One splat seen by one pixel's alpha: {@code A = w = exp(-m/2) * opacity}, so only the weight terms remain. */
    @Test
    public void singleSplatAlphaGradientIsClosedForm() {
        float[] splat = {2.5f, 1.5f, 1.2f, 0.2f, 0.9f, 0.6f, 0.3f, 0.7f, 0.1f};
        int x = 3;
        int y = 2;
        float[] gy = new float[HEIGHT * WIDTH * 4];
        gy[(y * WIDTH + x) * 4 + 3] = 1.0f;
        float[] g = analyticGradient(splat, gy, 1);

        double dx = x - splat[0];
        double dy = y - splat[1];
        double det = splat[2] * splat[4] - splat[3] * splat[3];
        double m = (splat[4] * dx * dx - 2 * splat[3] * dx * dy + splat[2] * dy * dy) / det;
        double falloff = Math.exp(-0.5 * m);
        double w = falloff * splat[5];
        double dLdm = -0.5 * w;
        double[] expected = {
            dLdm * -2.0 * (splat[4] * dx - splat[3] * dy) / det,
            dLdm * -2.0 * (splat[2] * dy - splat[3] * dx) / det,
            dLdm * (dy * dy / det - m * splat[4] / det),
            dLdm * (-2.0 * dx * dy / det + 2.0 * m * splat[3] / det),
            dLdm * (dx * dx / det - m * splat[2] / det),
            falloff, 0.0, 0.0, 0.0
        };
        for (int i = 0; i < expected.length; i++) {
            assertEquals("field " + i, expected[i], g[i], 1e-5 * Math.max(1.0, Math.abs(expected[i])));
        }
    }

    @Test
    public void releasedNodeAndWrongGradientThrow() {
        RasterizeGrad fn = new RasterizeGrad(SPLATS.clone(), GAUSSIANS, WIDTH, HEIGHT, host);
        try (Tensor wrong = Tensor.zeros(host, Shape.of(HEIGHT, WIDTH, 3))) {
            fn.backward(wrong);
            fail("a gradient of the wrong shape must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("rasterizer gradient"));
        }
        assertTrue("no tensor is saved", fn.savedTensors().isEmpty());
        fn.clearSavedTensors();
        try (Tensor gy = Tensor.zeros(host, Shape.of(HEIGHT, WIDTH, 4))) {
            fn.backward(gy);
            fail("a released node must refuse its backward");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("released"));
        }
        try {
            RasterizeGrad.rasterize(new float[RasterizeGrad.FIELDS + 1], 1, WIDTH, HEIGHT);
            fail("a splat array of the wrong length must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("splat rows"));
        }
    }

    /**
     * {@link GaussianSplatting#render} in grad mode returns the output of a {@link RasterizeGrad} node, and a loss on
     * the image reaches every parameter; only the first three spherical-harmonic columns (the base colour) get a
     * non-zero gradient. The backward releases every node of the step.
     */
    @Test
    public void renderRecordsTheRasterizerNode() {
        GaussianSplatting splats = new GaussianSplatting(GAUSSIANS, WIDTH, HEIGHT);
        ComputationGraph graph = ComputationGraph.current();
        try {
            splats.initialize(host, new Random(3));
            Map<String, Object> state = new LinkedHashMap<>();
            state.put("means", new float[]{0.13f, -0.21f, 0f, 0.4f, 0.35f, -1f, -0.45f, 0.1f, 1f});
            state.put("scales", new float[]{1.1f, 1.3f, 0.7f, 0.9f, 1.2f, 0.8f, 1.4f, 0.6f, 1.0f});
            state.put("rotations", new float[]{1, 0, 0, 0, 0.9f, 0.3f, 0.2f, 0.1f, 0.8f, -0.1f, 0.4f, 0.2f});
            state.put("opacities", new float[]{0.3f, 1.0f, -0.4f});
            state.put("shCoeffs", TrainParity.uniform(5, GAUSSIANS * 48, -1, 1));
            splats.loadStateDict(state);
            int before = graph.size();
            float[] pose = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1};
            try (ComputationGraph.StepScope step = graph.step();
                 Tensor p = tensor(pose, 4, 4);
                 Tensor image = splats.render(p);
                 Tensor r = tensor(TrainParity.uniform(6, HEIGHT * WIDTH * 4, -1, 1), HEIGHT, WIDTH, 4);
                 Tensor weighted = TensorOps.mul(image, r);
                 Tensor loss = TensorOps.sum(weighted)) {
                assertTrue("the image is the rasterizer node's output", image.gradFn() instanceof RasterizeGrad);
                loss.backward();
            }
            assertEquals("the step leaves the graph as it found it", before, graph.size());
            for (Parameter param : splats.parameters()) {
                assertNotNull("gradient of " + param.name(), param.grad());
                for (float v : param.grad().toFloatArray()) {
                    assertTrue("finite gradient of " + param.name(), Float.isFinite(v));
                }
            }
            float[] sh = splats.shCoeffsParam().grad().toFloatArray();
            for (int g = 0; g < GAUSSIANS; g++) {
                for (int c = 3; c < 48; c++) {
                    assertEquals("higher-order coefficient " + c, 0.0f, sh[g * 48 + c], 0.0f);
                }
            }
        } finally {
            splats.close();
        }
    }

    private float[] analyticGradient(float[] splats, float[] gy) {
        return analyticGradient(splats, gy, GAUSSIANS);
    }

    private float[] analyticGradient(float[] splats, float[] gy, int count) {
        RasterizeGrad fn = new RasterizeGrad(splats.clone(), count, WIDTH, HEIGHT, host);
        try (Tensor g = tensor(gy, HEIGHT, WIDTH, 4)) {
            List<Tensor> grads = fn.backward(g);
            assertEquals(1, grads.size());
            try (Tensor result = grads.get(0)) {
                assertEquals(Shape.matrix(count, RasterizeGrad.FIELDS), result.shape());
                return result.toFloatArray();
            }
        }
    }

    /** {@code sum(gy * rasterize(splats))} in double. */
    private static double loss(float[] splats, float[] gy) {
        float[] image = RasterizeGrad.rasterize(splats, GAUSSIANS, WIDTH, HEIGHT);
        double sum = 0.0;
        for (int i = 0; i < image.length; i++) {
            sum += (double) gy[i] * image[i];
        }
        return sum;
    }

    private Tensor tensor(float[] data, int... dims) {
        try (Tensor flat = Tensor.fromArray(host, data)) {
            return flat.reshape(Shape.of(dims));
        }
    }
}
