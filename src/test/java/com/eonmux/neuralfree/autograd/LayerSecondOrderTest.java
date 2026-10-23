package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.activations.LeakyReLU;
import com.eonmux.neuralfree.nn.activations.Tanh;
import com.eonmux.neuralfree.nn.layers.AdaptiveAvgPool2d;
import com.eonmux.neuralfree.nn.layers.AvgPool2d;
import com.eonmux.neuralfree.nn.layers.Conv1d;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.layers.ConvTranspose2d;
import com.eonmux.neuralfree.nn.layers.Flatten;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.layers.MaxPool2d;
import com.eonmux.neuralfree.nn.norm.BatchNorm2d;
import com.eonmux.neuralfree.nn.norm.GroupNorm;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.nn.norm.RMSNorm;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Predicate;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Second-order gradients through the convolution, pooling and normalization layers, on the
 * CPU reference backend and on each GPU API of {@link TrainParity}. Each case trains a WGAN-GP critic's gradient
 * penalty {@code mean_b((sqrt(sum g_b^2 + 1e-8) - 1)^2)} with {@code g = d sum(D(x)) / dx} from a create-graph pass
 * ({@link Autograd#grad}) and compares the parameter gradient of the penalty, which runs through the recorded
 * backward of every layer, against central finite differences ({@link GradientCheckUtil#gradCheckParameters}, its
 * float tolerance). Every case also asserts that the check leaves the graph at its pre-step size.
 *
 * <p>The analytical side needs create-graph support in the backward functions of the activations, views, matmul and
 * reductions, and of conv, pool and norm. <b>Kinks.</b> LeakyReLU and max pooling have kinks; their cases draw the
 * critic input from successive seeds until every LeakyReLU input lies at least {@link #KINK_MARGIN} from zero and every
 * max-pool window's maximum exceeds its runner-up by at least {@link #KINK_MARGIN}, far above what a finite-difference
 * step of {@link #EPSILON} on one parameter element can move them, so no perturbation crosses a kink. The other cases
 * use tanh and smooth pooling and need no margin. Normalization weights start at one and biases at zero, as the modules
 * initialize them; inputs lie in {@code [-1, 1)}.</p>
 */
@RunWith(Parameterized.class)
public class LayerSecondOrderTest {

    private static final float EPSILON = 1e-3f;
    private static final float TOLERANCE = 1e-2f;
    private static final int ELEMENTS_PER_PARAMETER = 6;
    private static final float KINK_MARGIN = 0.05f;
    private static final int SEED_ATTEMPTS = 200;
    private static final long MODULE_SEED = 20261023L;

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        List<Object[]> rows = new ArrayList<>();
        rows.add(new Object[]{"host", null});
        for (TrainParity.Api api : TrainParity.Api.values()) {
            rows.add(new Object[]{api.name(), api});
        }
        return rows;
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend backend;
    private final List<Module> modules = new ArrayList<>();
    private final List<Tensor> leaves = new ArrayList<>();

    public LayerSecondOrderTest(String label, TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        if (api == null) {
            backend = TrainParity.hostBackend();
        } else {
            session = TrainParity.open(api);
            backend = session.backend();
        }
    }

    @After
    public void tearDown() {
        try {
            ComputationGraph.current().clear();
            ComputationGraph.setGradEnabled(true);
            for (Tensor leaf : leaves) {
                releaseLeaf(leaf);
            }
            for (Module m : modules) {
                m.close();
            }
        } finally {
            if (session != null) {
                session.close();
            } else if (backend != null) {
                backend.close();
            }
        }
    }

    // ------------------------------------------------------------------ cases

    /**
     * Conv2d, training-mode BatchNorm2d (batch statistics) and LeakyReLU, then a linear head: the WGAN-GP critic of a
     * convolutional GAN.
     */
    @Test
    public void wganGpConvBatchNormLeakyReluTraining() {
        Sequential front = new Sequential(conv2d(2, 3, 3, 1), batchNorm2d(3));
        Sequential critic = critic(front, new LeakyReLU(0.2f), new Flatten(), linear(48, 1));
        critic.train(true);
        checkPenalty(critic, interpolationLeaf(Shape.of(3, 2, 4, 4), front, LayerSecondOrderTest::awayFromZero));
    }

    /** The same critic with evaluation-mode BatchNorm2d: the running statistics are constants of the backward. */
    @Test
    public void wganGpConvBatchNormLeakyReluEvaluation() {
        Sequential front = new Sequential(conv2d(2, 3, 3, 1), batchNorm2d(3));
        Sequential critic = critic(front, new LeakyReLU(0.2f), new Flatten(), linear(48, 1));
        critic.train(false);
        checkPenalty(critic, interpolationLeaf(Shape.of(3, 2, 4, 4), front, LayerSecondOrderTest::awayFromZero));
    }

    /** Max pooling: the penalty's gradient runs through MaxPool2dBackwardGrad (maxPool2dSelect). */
    @Test
    public void wganGpConvMaxPool() {
        Sequential front = new Sequential(conv2d(2, 2, 3, 1), new Tanh());
        Sequential critic = critic(front, new MaxPool2d(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}),
                new Flatten(), linear(8, 1));
        checkPenalty(critic, interpolationLeaf(Shape.of(2, 2, 4, 4), front,
                LayerSecondOrderTest::distinctWindowMaxima));
    }

    /**
     * ConvTranspose2d with output padding, GroupNorm and adaptive average pooling with uneven windows: the transposed
     * weight-gradient and adaptive-backward second-order classes and the GroupNorm composition.
     */
    @Test
    public void wganGpConvTransposeGroupNormAdaptivePool() {
        ConvTranspose2d deconv = new ConvTranspose2d(2, 4, new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1},
                new int[]{1, 1}, new int[]{1, 1}, 1, true);
        deconv.initialize(backend, new Random(MODULE_SEED));
        GroupNorm norm = new GroupNorm(2, 4);
        norm.initialize(backend);
        Sequential critic = critic(new Sequential(deconv, norm), new Tanh(), new AdaptiveAvgPool2d(4, 3),
                new Flatten(), linear(48, 1));
        checkPenalty(critic, interpolationLeaf(Shape.of(2, 2, 3, 3), null, null));
    }

    /** Conv1d and LayerNorm over the sequence: the unsqueezed weight-gradient path and the LayerNorm composition. */
    @Test
    public void wganGpConv1dLayerNorm() {
        Conv1d conv = new Conv1d(3, 4, 3, 1, 1);
        conv.initialize(backend, new Random(MODULE_SEED));
        LayerNorm norm = new LayerNorm(5);
        norm.initialize(backend);
        Sequential critic = critic(new Sequential(conv, norm), new Tanh(), new Flatten(), linear(20, 1));
        checkPenalty(critic, interpolationLeaf(Shape.of(2, 3, 5), null, null));
    }

    /** Conv2d, RMSNorm over the width and padded, overlapping average pooling: the RMSNorm composition. */
    @Test
    public void wganGpConvRmsNormAvgPool() {
        RMSNorm norm = new RMSNorm(5);
        norm.initialize(backend);
        Sequential critic = critic(new Sequential(conv2d(2, 3, 3, 1), norm), new Tanh(),
                new AvgPool2d(new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1}, false, false), new Flatten(),
                linear(27, 1));
        checkPenalty(critic, interpolationLeaf(Shape.of(2, 2, 5, 5), null, null));
    }

    /**
     * One WGAN-GP critic step as a training loop runs it: inside a {@link ComputationGraph.StepScope} the
     * critic loss {@code mean(D(x)) + gp} (the Wasserstein term keeps every parameter on the loss's path; the linear
     * bias does not reach the penalty) is back-propagated through the create-graph penalty, every parameter receives
     * a finite gradient, and closing the scope returns the graph to its size before the step, with the create-graph
     * flag off.
     */
    @Test
    public void createGraphStepRestoresGraphSize() {
        Sequential front = new Sequential(conv2d(2, 3, 3, 1), batchNorm2d(3));
        Sequential critic = critic(front, new LeakyReLU(0.2f), new Flatten(), linear(48, 1));
        critic.train(true);
        Tensor x = interpolationLeaf(Shape.of(3, 2, 4, 4), front, LayerSecondOrderTest::awayFromZero);
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        try (ComputationGraph.StepScope step = graph.step();
             Tensor penalty = penalty(critic, x).get();
             Tensor score = critic.forward(x);
             Tensor wasserstein = TensorOps.mean(score);
             Tensor loss = TensorOps.add(wasserstein, penalty)) {
            assertTrue("the penalty is recorded with respect to the parameters", penalty.requiresGrad());
            loss.backward();
        }
        assertEquals("the step leaves nothing recorded", before, graph.size());
        assertFalse(Autograd.isCreatingGraph());
        for (Parameter p : critic.parameters()) {
            Tensor grad = p.grad();
            assertNotNull("gradient of " + p.name(), grad);
            for (float v : grad.toFloatArray()) {
                assertTrue("finite gradient of " + p.name(), Float.isFinite(v));
            }
        }
        critic.zeroGrad();
    }

    // ------------------------------------------------------------------ the check

    /** The penalty's parameter gradient against finite differences, leaving the graph at its pre-check size. */
    private void checkPenalty(Module critic, Tensor x) {
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParameters(penalty(critic, x),
                critic, critic.parameters(), EPSILON, TOLERANCE, ELEMENTS_PER_PARAMETER);
        assertTrue(result.toString(), result.passed());
        assertEquals("the check leaves nothing recorded", before, graph.size());
        assertFalse(Autograd.isCreatingGraph());
    }

    /**
     * The gradient penalty of {@code critic} at the leaf {@code x}, as {@code GANTrainer} computes it. Called in grad
     * mode (the analytical side), the input gradient comes from a create-graph pass, so the penalty is recorded with
     * respect to the parameters; called under {@code noGrad} (the finite-difference side), the critic still records
     * its forward for the first-order input gradient, which is released, and the penalty is a plain value.
     */
    private static Supplier<Tensor> penalty(Module critic, Tensor x) {
        return () -> {
            boolean createGraph = ComputationGraph.isGradEnabled();
            Tensor gradient;
            try (ComputationGraph.EnableGradScope grad = ComputationGraph.enableGrad();
                 Tensor score = critic.forward(x);
                 Tensor total = TensorOps.sum(score)) {
                gradient = Autograd.grad(total, Collections.singletonList(x), null, createGraph, createGraph).get(0);
            }
            if (gradient == null) {
                fail("the critic output does not depend on its input");
            }
            int batch = x.shape().dim(0);
            int perSample = (int) (x.numElements() / batch);
            try (Tensor g = gradient;
                 Tensor rows = g.reshape(batch, perSample);
                 Tensor squares = TensorOps.mul(rows, rows);
                 Tensor sumSquares = TensorOps.sum(squares, 1);
                 Tensor shifted = TensorOps.add(sumSquares, 1e-8f);
                 Tensor norm = TensorOps.sqrt(shifted);
                 Tensor deviation = TensorOps.add(norm, -1.0f);
                 Tensor squared = TensorOps.mul(deviation, deviation)) {
                return TensorOps.mean(squared);
            }
        };
    }

    // ------------------------------------------------------------------ inputs and kink margins

    /**
     * The critic input: a leaf of {@code shape} with values uniform in {@code [-1, 1)}. When {@code kinkFree} is
     * given, successive seeds are tried until {@code front}'s output on the input (computed under {@code noGrad})
     * satisfies it.
     */
    private Tensor interpolationLeaf(Shape shape, Module front, Predicate<Tensor> kinkFree) {
        int n = Math.toIntExact(shape.numElements());
        for (int attempt = 0; attempt < SEED_ATTEMPTS; attempt++) {
            float[] values = TrainParity.uniform(MODULE_SEED + 1 + attempt, n, -1.0f, 1.0f);
            if (kinkFree == null || clearOfKinks(values, shape, front, kinkFree)) {
                Tensor raw = upload(values, shape);
                Tensor leaf = raw.withRequiresGrad(true);
                if (leaf != raw) {
                    raw.close();
                }
                leaves.add(leaf);
                return leaf;
            }
        }
        fail("no input among " + SEED_ATTEMPTS + " seeds keeps the critic " + KINK_MARGIN + " away from its kinks");
        return null;
    }

    private boolean clearOfKinks(float[] values, Shape shape, Module front, Predicate<Tensor> kinkFree) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor input = upload(values, shape);
             Tensor out = front.forward(input)) {
            return kinkFree.test(out);
        }
    }

    /** Every element at least {@link #KINK_MARGIN} from zero (the LeakyReLU kink). */
    private static boolean awayFromZero(Tensor t) {
        for (float v : t.toFloatArray()) {
            if (!(Math.abs(v) >= KINK_MARGIN)) {
                return false;
            }
        }
        return true;
    }

    /**
     * In every 2x2 window of stride 2 of an NCHW tensor, the maximum exceeds the runner-up by at least
     * {@link #KINK_MARGIN} (no max-pool routing switch).
     */
    private static boolean distinctWindowMaxima(Tensor t) {
        float[] v = t.toFloatArray();
        int planes = t.shape().dim(0) * t.shape().dim(1);
        int h = t.shape().dim(2);
        int w = t.shape().dim(3);
        for (int p = 0; p < planes; p++) {
            for (int oh = 0; oh + 1 < h; oh += 2) {
                for (int ow = 0; ow + 1 < w; ow += 2) {
                    float first = Float.NEGATIVE_INFINITY;
                    float second = Float.NEGATIVE_INFINITY;
                    for (int k = 0; k < 4; k++) {
                        float value = v[p * h * w + (oh + k / 2) * w + ow + k % 2];
                        if (value > first) {
                            second = first;
                            first = value;
                        } else if (value > second) {
                            second = value;
                        }
                    }
                    if (!(first - second >= KINK_MARGIN)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    // ------------------------------------------------------------------ modules

    private Sequential critic(Module... layers) {
        Sequential critic = new Sequential(layers);
        modules.add(critic);
        return critic;
    }

    /** {@code Conv2d(in, out, k x k)} with stride 1 and padding {@code pad}, initialized from the fixed seed. */
    private Conv2d conv2d(int in, int out, int k, int pad) {
        Conv2d conv = new Conv2d(in, out, new int[]{k, k}, new int[]{1, 1}, new int[]{pad, pad});
        conv.initialize(backend, new Random(MODULE_SEED));
        return conv;
    }

    private BatchNorm2d batchNorm2d(int channels) {
        BatchNorm2d norm = new BatchNorm2d(channels);
        norm.initialize(backend);
        return norm;
    }

    private Linear linear(int in, int out) {
        Linear linear = new Linear(in, out);
        linear.initialize(backend, new Random(MODULE_SEED + 7));
        return linear;
    }

    private Tensor upload(float[] values, Shape shape) {
        return Tensor.fromStorage(backend, shape, DType.FLOAT32, TensorStorage.fromHostFloat(backend, values));
    }

    /** Clears and closes the gradient a backward accumulated into the leaf, then closes the leaf. */
    private static void releaseLeaf(Tensor leaf) {
        Tensor grad = leaf.grad();
        if (grad != null) {
            leaf.setGrad(null);
            grad.close();
        }
        leaf.close();
    }
}
