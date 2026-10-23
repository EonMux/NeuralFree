package com.eonmux.neuralfree.nn.activations;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.UnaryOperator;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The activation modules on each GPU API against the same modules on the CPU reference backend, under the tolerance
 * classes of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}: ReLU bit-identical, LeakyReLU
 * single-rounding, every other activation transcendental. Inputs include transposed views, so the strided device paths
 * run, and the modules' autograd recording is checked on both backends.
 */
@RunWith(Parameterized.class)
public class ActivationsParityTest {

    private static final float[] SPECIAL = {
            0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
            Float.MIN_VALUE, -Float.MIN_VALUE, Float.MIN_NORMAL, 3e9f, -3e9f, 1e-3f
    };

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private enum Tolerance { BIT, ROUNDING, TRANSCENDENTAL }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();

    public ActivationsParityTest(TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        ComputationGraph.setGradEnabled(true);
        session = TrainParity.open(api);
        gpu = session.backend();
        host = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        try {
            ComputationGraph.current().clear();
            for (int i = owned.size() - 1; i >= 0; i--) {
                owned.get(i).close();
            }
        } finally {
            if (session != null) {
                session.close();
            }
        }
    }

    // ------------------------------------------------------------------ elementwise modules

    @Test
    public void reluIsBitIdentical() {
        float[] x = withSpecials(TrainParity.uniform(11, 24 * 40, -10f, 10f));
        assertElementwise("ReLU", new ReLU(), x, Tolerance.BIT);
    }

    @Test
    public void leakyReluIsSingleRounding() {
        float[] x = withSpecials(TrainParity.uniform(12, 24 * 40, -10f, 10f));
        assertElementwise("LeakyReLU(0.01)", new LeakyReLU(), x, Tolerance.ROUNDING);
        assertElementwise("LeakyReLU(0.2)", new LeakyReLU(0.2f), x, Tolerance.ROUNDING);
    }

    @Test
    public void transcendentalActivationsMatch() {
        float[] x = TrainParity.uniform(13, 24 * 40, -10f, 10f);
        assertElementwise("ELU(1)", new ELU(), x, Tolerance.TRANSCENDENTAL);
        assertElementwise("ELU(0.5)", new ELU(0.5f), x, Tolerance.TRANSCENDENTAL);
        assertElementwise("GELU", new GELU(), x, Tolerance.TRANSCENDENTAL);
        assertElementwise("SiLU", new SiLU(), x, Tolerance.TRANSCENDENTAL);
        assertElementwise("Sigmoid", new Sigmoid(), x, Tolerance.TRANSCENDENTAL);
        assertElementwise("Tanh", new Tanh(), x, Tolerance.TRANSCENDENTAL);
        assertElementwise("Mish", new Mish(), x, Tolerance.TRANSCENDENTAL);
    }

    // ------------------------------------------------------------------ dim-based modules

    @Test
    public void softmaxAndLogSoftmaxMatchOnEveryDim() {
        int[] dims = {4, 5, 6};
        float[] x = TrainParity.uniform(14, 4 * 5 * 6, -10f, 10f);
        for (int dim : new int[]{0, 1, 2, -1, -3}) {
            assertAlongDim("Softmax(" + dim + ")", new Softmax(dim), x, dims, Tolerance.TRANSCENDENTAL);
            assertAlongDim("LogSoftmax(" + dim + ")", new LogSoftmax(dim), x, dims, Tolerance.TRANSCENDENTAL);
        }
        assertAlongDim("Softmax()", new Softmax(), x, dims, Tolerance.TRANSCENDENTAL);
        assertAlongDim("LogSoftmax()", new LogSoftmax(), x, dims, Tolerance.TRANSCENDENTAL);
    }

    @Test
    public void gatedUnitsMatchOnEveryEvenDim() {
        int[] dims = {4, 6, 8};
        float[] x = TrainParity.uniform(15, 4 * 6 * 8, -10f, 10f);
        for (int dim : new int[]{0, 1, 2, -1, -2}) {
            assertAlongDim("GLU(" + dim + ")", new GLU(dim), x, dims, Tolerance.TRANSCENDENTAL);
            assertAlongDim("GeGLU(" + dim + ")", new GeGLU(dim), x, dims, Tolerance.TRANSCENDENTAL);
        }
        assertAlongDim("GLU()", new GLU(), x, dims, Tolerance.TRANSCENDENTAL);
        assertAlongDim("GeGLU()", new GeGLU(), x, dims, Tolerance.TRANSCENDENTAL);
    }

    @Test
    public void gatedUnitsRejectAnOddDimOnBothBackends() {
        float[] x = TrainParity.uniform(16, 3 * 4, -1f, 1f);
        for (ComputeBackend b : new ComputeBackend[]{host, gpu}) {
            Tensor t = floats(b, x, 3, 4);
            assertThrows(IllegalArgumentException.class, () -> keep(new GLU(0).forward(t)));
            assertThrows(IllegalArgumentException.class, () -> keep(new GeGLU(0).forward(t)));
        }
    }

    // ------------------------------------------------------------------ recording

    /**
     * Each module's forward records exactly one node when grad mode is on and its input requires grad, and nothing for
     * an input without grad or under {@link ComputationGraph#noGrad()}; the recorded output has the untracked values,
     * and a backward from its sum delivers an input-shaped gradient to the leaf and releases the node.
     */
    @Test
    public void forwardsRecordOneNodeExactlyWhenTheInputRequiresGrad() {
        float[] x = TrainParity.uniform(17, 4 * 6, -3f, 3f);
        Module[] modules = {
                new ReLU(), new LeakyReLU(), new ELU(), new GELU(), new SiLU(), new Sigmoid(), new Tanh(), new Mish(),
                new Softmax(), new LogSoftmax(), new GLU(), new GeGLU()
        };
        ComputationGraph graph = ComputationGraph.current();
        for (ComputeBackend b : new ComputeBackend[]{host, gpu}) {
            Tensor plain = floats(b, x, 4, 6);
            Tensor leaf = keep(plain.withRequiresGrad(true));
            for (Module m : modules) {
                String what = m + " on " + b.backendType();
                int before = graph.size();
                Tensor withoutGrad = keep(m.forward(plain));
                assertFalse(what + ": output of a plain input requires grad", withoutGrad.requiresGrad());
                assertNull(what + ": output of a plain input has a gradient function", withoutGrad.gradFn());
                Tensor untracked;
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                    untracked = keep(m.forward(leaf));
                }
                assertFalse(what + ": output under noGrad requires grad", untracked.requiresGrad());
                assertNull(what + ": output under noGrad has a gradient function", untracked.gradFn());
                assertEquals(what + ": nodes recorded without grad", before, graph.size());

                Tensor withGrad = m.forward(leaf);
                try (Tensor loss = TensorOps.sum(withGrad)) {
                    assertTrue(what + ": output of a leaf does not require grad", withGrad.requiresGrad());
                    assertNotNull(what + ": output of a leaf has no gradient function", withGrad.gradFn());
                    assertEquals(what + ": nodes recorded by the forward and the sum", before + 2, graph.size());
                    TrainParity.assertBitIdentical(what + ": recorded output", withoutGrad.toFloatArray(),
                            withGrad.toFloatArray());
                    loss.backward();
                }
                assertEquals(what + ": nodes left after backward", before, graph.size());
                Tensor grad = leaf.grad();
                assertNotNull(what + ": leaf got no gradient", grad);
                leaf.setGrad(null);
                try (Tensor g = grad) {
                    assertEquals(what + ": gradient shape", leaf.shape(), g.shape());
                }
            }
        }
    }

    // ------------------------------------------------------------------ helpers

    /** The module on a {@code [24, 40]} transposed view of {@code x}, on both backends. */
    private void assertElementwise(String what, Module module, float[] x, Tolerance tol) {
        UnaryOperator<Tensor> view = t -> keep(t.transpose());
        Tensor ref = keep(module.forward(view.apply(floats(host, x, 40, 24))));
        Tensor got = keep(module.forward(view.apply(floats(gpu, x, 40, 24))));
        assertMatches(what, ref, got, tol);
    }

    /**
     * The module on a contiguous {@code dims} tensor and on a strided view of the same logical shape (a permutation of
     * a differently laid out tensor), on both backends.
     */
    private void assertAlongDim(String what, Module module, float[] x, int[] dims, Tolerance tol) {
        Tensor ref = keep(module.forward(floats(host, x, dims)));
        Tensor got = keep(module.forward(floats(gpu, x, dims)));
        assertMatches(what, ref, got, tol);

        UnaryOperator<Tensor> view = t -> keep(t.permute(1, 2, 0));
        Tensor refView = keep(module.forward(view.apply(floats(host, x, dims[2], dims[0], dims[1]))));
        Tensor gotView = keep(module.forward(view.apply(floats(gpu, x, dims[2], dims[0], dims[1]))));
        assertMatches(what + " on a permuted view", refView, gotView, tol);
    }

    private void assertMatches(String what, Tensor ref, Tensor got, Tolerance tol) {
        assertEquals(what + ": shape", ref.shape(), got.shape());
        float[] r = ref.toFloatArray();
        float[] g = got.toFloatArray();
        switch (tol) {
            case BIT:
                TrainParity.assertBitIdentical(what, r, g);
                break;
            case ROUNDING:
                TrainParity.assertSingleRounding(what, r, g);
                break;
            default:
                TrainParity.assertTranscendental(what, r, g);
                break;
        }
    }

    private static float[] withSpecials(float[] values) {
        System.arraycopy(SPECIAL, 0, values, 0, Math.min(SPECIAL.length, values.length));
        return values;
    }

    private Tensor keep(Tensor t) {
        if (t != null) {
            owned.add(t);
        }
        return t;
    }

    private Tensor floats(ComputeBackend b, float[] data, int... dims) {
        return keep(keep(Tensor.fromArray(b, data)).reshape(Shape.of(dims)));
    }
}
