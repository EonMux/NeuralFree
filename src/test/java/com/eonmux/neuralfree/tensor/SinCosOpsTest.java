package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradFunctions;
import com.eonmux.neuralfree.autograd.GradientCheckUtil;
import com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
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
import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;

import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.in;
import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.uniform;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link TensorOps#sin} and {@link TensorOps#cos} on the host reference backend and on every available
 * {@link TrainParity} GPU API: the forward against {@code (float) Math.sin}/{@code Math.cos} (bit-identical on the CPU
 * path, the transcendental class of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract} on the device,
 * over specials, strided views and arguments up to {@code Float.MAX_VALUE}, so every range-reduction path runs), the
 * recording rule, the first derivatives {@code cos(x)} and {@code -sin(x)}, central finite differences ({@link
 * GradCheck}), and the second derivatives of a create-graph pass against {@code -sin(x)} and {@code -cos(x)} and
 * {@link GradientCheckUtil#gradGradCheck}.
 */
@RunWith(Parameterized.class)
public class SinCosOpsTest {

    /** Signed zeros, non-finite values, denormals, quadrant boundaries and the edges of the range reductions. */
    private static final float[] SPECIAL = {
            0.0f, -0.0f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.MIN_VALUE,
            -Float.MIN_VALUE, Float.MIN_NORMAL, 1e-30f, 0.5f, -0.5f, (float) (Math.PI / 4), (float) (Math.PI / 2),
            (float) Math.PI, -(float) (3 * Math.PI / 2), 65535.996f, 65536.0f, -65536.0f, 102943.7f, 16777216.0f,
            1e10f, -3e9f, 2.5e27f, Float.MAX_VALUE, -Float.MAX_VALUE
    };
    private static final float[] SMOOTH_POINTS = {-2.3f, -1.3f, -0.7f, -0.2f, 0.15f, 0.6f, 1.1f, 1.7f};
    private static final float[] WIDE_POINTS = {-5.5f, 3.9f, 12.25f, -40.5f, 100.75f, 1234.5f, -9876.25f, 70000.5f};
    private static final float GRAD_GRAD_EPSILON = 1e-3f;
    private static final float GRAD_GRAD_TOLERANCE = 1e-2f;
    private static final float FORMULA_TOLERANCE = 1e-4f;

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return GradCheck.targets();
    }

    private final GradCheck.Target target;
    private GradCheck fx;
    private ComputationGraph graph;
    private final List<Tensor> owned = new ArrayList<>();

    public SinCosOpsTest(GradCheck.Target target) {
        this.target = target;
    }

    @Before
    public void setUp() {
        ComputationGraph.setGradEnabled(true);
        graph = ComputationGraph.current();
        graph.clear();
        fx = GradCheck.open(target);
    }

    @After
    public void tearDown() {
        try {
            graph.clear();
            ComputationGraph.setGradEnabled(true);
            closeAll(owned);
        } finally {
            if (fx != null) {
                fx.close();
            }
        }
    }

    // ------------------------------------------------------------------ forward

    @Test
    public void forwardFollowsJavaMathOverTheWholeFloatRange() {
        float[] small = withSpecials(TrainParity.uniform(1, 24 * 40, -10f, 10f));
        float[] wide = TrainParity.uniform(2, 24 * 40, -1e5f, 1e5f);
        float[] huge = TrainParity.uniform(3, 24 * 40, -1e30f, 1e30f);
        for (float[] x : new float[][]{small, wide, huge}) {
            assertForward("sin", x, TensorOps::sin, GpuTensorOps::sin, Math::sin);
            assertForward("cos", x, TensorOps::cos, GpuTensorOps::cos, Math::cos);
        }
    }

    @Test
    public void sinKeepsTheSignOfZeroAndNonFiniteArgumentsGiveNaN() {
        Tensor x = fx.constant(new float[]{0.0f, -0.0f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY},
                5);
        float[] sin = keep(TensorOps.sin(x)).toFloatArray();
        float[] cos = keep(TensorOps.cos(x)).toFloatArray();
        assertEquals(Float.floatToRawIntBits(0.0f), Float.floatToRawIntBits(sin[0]));
        assertEquals(Float.floatToRawIntBits(-0.0f), Float.floatToRawIntBits(sin[1]));
        assertEquals(1.0f, cos[0], 1e-5f);
        assertEquals(1.0f, cos[1], 1e-5f);
        for (int i = 2; i < 5; i++) {
            assertTrue("sin of " + x.toFloatArray()[i], Float.isNaN(sin[i]));
            assertTrue("cos of " + x.toFloatArray()[i], Float.isNaN(cos[i]));
        }
    }

    // ------------------------------------------------------------------ recording

    @Test
    public void recordsSinGradAndCosGradOnlyForAnInputThatRequiresGrad() {
        Tensor constant = fx.constant(SMOOTH_POINTS, 2, 4);
        Tensor plainSin = keep(TensorOps.sin(constant));
        Tensor plainCos = keep(TensorOps.cos(constant));
        assertFalse(plainSin.requiresGrad());
        assertFalse(plainCos.requiresGrad());
        assertNull(plainSin.gradFn());
        assertNull(plainCos.gradFn());

        Tensor leaf = keep(constant.withRequiresGrad(true));
        int before = graph.size();
        try (ComputationGraph.StepScope step = graph.step()) {
            Tensor sin = keep(TensorOps.sin(leaf));
            Tensor cos = keep(TensorOps.cos(leaf));
            assertTrue(sin.requiresGrad());
            assertTrue(cos.requiresGrad());
            assertTrue("sin records SinGrad", sin.gradFn() instanceof GradFunctions.SinGrad);
            assertTrue("cos records CosGrad", cos.gradFn() instanceof GradFunctions.CosGrad);
            assertTrue(((GradFunctions.SinGrad) sin.gradFn()).supportsCreateGraph());
            assertTrue(((GradFunctions.CosGrad) cos.gradFn()).supportsCreateGraph());
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Tensor untracked = keep(TensorOps.sin(leaf));
                assertFalse("no recording under noGrad", untracked.requiresGrad());
            }
        }
        assertEquals("the step leaves the graph at its size", before, graph.size());
    }

    // ------------------------------------------------------------------ first order

    @Test
    public void firstDerivativesAreCosAndMinusSin() {
        for (float[] points : new float[][]{SMOOTH_POINTS, WIDE_POINTS}) {
            float[] dSin = gradient(TensorOps::sin, points);
            float[] dCos = gradient(TensorOps::cos, points);
            float[] cos = new float[points.length];
            float[] minusSin = new float[points.length];
            for (int i = 0; i < points.length; i++) {
                cos[i] = (float) Math.cos(points[i]);
                minusSin[i] = -(float) Math.sin(points[i]);
            }
            assertClass("d sin / dx", cos, dSin);
            assertClass("d cos / dx", minusSin, dCos);
        }
    }

    @Test
    public void gradientsMatchFiniteDifferences() {
        GradCheck.Input x = in(uniform(41, 12, -1f, 1f), 3, 4);
        fx.inputs("sin", a -> TensorOps.sin(a.get(0)), x);
        fx.inputs("cos", a -> TensorOps.cos(a.get(0)), x);
        fx.inputs("sin(3x)", a -> GradCheck.then(TensorOps.mul(a.get(0), 3.0f), a, TensorOps::sin), x);
        fx.inputs("cos(3x)", a -> GradCheck.then(TensorOps.mul(a.get(0), 3.0f), a, TensorOps::cos), x);
        fx.inputs("sin(x) * cos(y)", a -> {
            try (Tensor s = TensorOps.sin(a.get(0));
                 Tensor c = TensorOps.cos(a.get(1))) {
                return TensorOps.mul(s, c);
            }
        }, x, in(uniform(42, 12, -1f, 1f), 3, 4));
    }

    // ------------------------------------------------------------------ second order

    @Test
    public void secondDerivativesAreMinusSinAndMinusCos() {
        float[] d2Sin = secondDerivative(TensorOps::sin, SMOOTH_POINTS);
        float[] d2Cos = secondDerivative(TensorOps::cos, SMOOTH_POINTS);
        assertFormula("sin''", x -> -Math.sin(x), d2Sin);
        assertFormula("cos''", x -> -Math.cos(x), d2Cos);
    }

    @Test
    public void gradGradChecksPass() {
        assertGradGrad("sin", in -> TensorOps.sin(in.get(0)), fx.constant(SMOOTH_POINTS, 2, 4));
        assertGradGrad("cos", in -> TensorOps.cos(in.get(0)), fx.constant(SMOOTH_POINTS, 2, 4));
        assertGradGrad("sin(x) * cos(x)", in -> {
            try (Tensor s = TensorOps.sin(in.get(0));
                 Tensor c = TensorOps.cos(in.get(0))) {
                return TensorOps.mul(s, c);
            }
        }, fx.constant(SMOOTH_POINTS, 2, 4));
    }

    // ------------------------------------------------------------------ helpers

    private boolean onDevice() {
        return target != GradCheck.Target.HOST;
    }

    /**
     * {@code op} on a transposed view of {@code x} through the facade and the dispatcher: the dispatcher declines
     * exactly on the host, and the facade's result is bit-identical to {@code (float) ref(v)} on the host and in the
     * transcendental class on a device.
     */
    private void assertForward(String what, float[] x, UnaryOperator<Tensor> facade, UnaryOperator<Tensor> dispatcher,
                               DoubleUnaryOperator ref) {
        Tensor view = keep(fx.constant(x, 40, 24).transpose());
        float[] values = view.toFloatArray();
        float[] expected = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            expected[i] = (float) ref.applyAsDouble(values[i]);
        }
        Tensor direct = dispatcher.apply(view);
        if (onDevice()) {
            assertNotNull(target + " " + what + ": the device path declined", keep(direct));
            TrainParity.assertTranscendental(target + " " + what + " (dispatcher)", expected, direct.toFloatArray());
        } else {
            assertNull("the host backend has no train library", direct);
        }
        Tensor result = keep(facade.apply(view));
        assertEquals(view.shape(), result.shape());
        assertClass(what, expected, result.toFloatArray());
    }

    /** Bit-identical on the host, the transcendental class on a device. */
    private void assertClass(String what, float[] expected, float[] actual) {
        String label = target + " " + what;
        if (onDevice()) {
            TrainParity.assertTranscendental(label, expected, actual);
        } else {
            TrainParity.assertBitIdentical(label, expected, actual);
        }
    }

    /** {@code d sum(f(x)) / dx} at {@code points} from one first-order pass inside a step. */
    private float[] gradient(UnaryOperator<Tensor> f, float[] points) {
        int before = graph.size();
        List<Tensor> created = new ArrayList<>();
        Tensor leaf = leaf(points);
        float[] result;
        try (ComputationGraph.StepScope step = graph.step()) {
            Tensor y = track(created, f.apply(leaf));
            Tensor loss = track(created, TensorOps.sum(y));
            List<Tensor> grads = Autograd.grad(loss, Collections.singletonList(leaf), null, false, false);
            created.addAll(grads);
            assertNotNull("the input receives a gradient", grads.get(0));
            result = grads.get(0).toFloatArray();
        } finally {
            closeAll(created);
            leaf.close();
        }
        assertEquals("the step leaves the graph at its size", before, graph.size());
        return result;
    }

    /**
     * {@code f''(x)} at {@code points}: the gradient of {@code sum(d sum(f(x)) / dx)} from a create-graph pass, which
     * is the diagonal of the (diagonal) Hessian of {@code sum(f(x))}.
     */
    private float[] secondDerivative(UnaryOperator<Tensor> f, float[] points) {
        int before = graph.size();
        List<Tensor> created = new ArrayList<>();
        Tensor leaf = leaf(points);
        float[] result;
        try (ComputationGraph.StepScope step = graph.step()) {
            Tensor y = track(created, f.apply(leaf));
            Tensor loss = track(created, TensorOps.sum(y));
            List<Tensor> first = Autograd.grad(loss, Collections.singletonList(leaf), null, false, true);
            created.addAll(first);
            assertNotNull("the input receives a first-order gradient", first.get(0));
            assertTrue("the first-order gradient is recorded", first.get(0).requiresGrad());
            Tensor total = track(created, TensorOps.sum(first.get(0)));
            List<Tensor> second = Autograd.grad(total, Collections.singletonList(leaf), null, false, false);
            created.addAll(second);
            assertNotNull("the input receives a second-order gradient", second.get(0));
            result = second.get(0).toFloatArray();
        } finally {
            closeAll(created);
            leaf.close();
        }
        assertEquals("the create-graph step leaves the graph at its size", before, graph.size());
        return result;
    }

    private void assertFormula(String what, DoubleUnaryOperator closedForm, float[] actual) {
        for (int i = 0; i < SMOOTH_POINTS.length; i++) {
            double expected = closedForm.applyAsDouble(SMOOTH_POINTS[i]);
            double diff = Math.abs(expected - actual[i]);
            if (!(diff <= FORMULA_TOLERANCE || diff <= FORMULA_TOLERANCE * Math.abs(expected))) {
                fail(target + " " + what + " at " + SMOOTH_POINTS[i] + ": expected " + expected + " but was "
                        + actual[i]);
            }
        }
    }

    private void assertGradGrad(String what, GradientCheckUtil.TensorFunction fn, Tensor input) {
        int before = graph.size();
        GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradGradCheck(fn,
                Collections.singletonList(input), GRAD_GRAD_EPSILON, GRAD_GRAD_TOLERANCE);
        assertTrue(target + " " + what + ": " + result, result.passed());
        assertEquals(target + " " + what + " leaves the graph at its size", before, graph.size());
    }

    /** A new leaf of shape {@code [2, n/2]} on the check's backend; the caller closes it. */
    private Tensor leaf(float[] points) {
        try (Tensor values = Tensor.fromStorage(fx.backend(), Shape.of(2, points.length / 2), DType.FLOAT32,
                TensorStorage.fromHostFloat(fx.backend(), points))) {
            return values.withRequiresGrad(true);
        }
    }

    private Tensor keep(Tensor t) {
        if (t != null) {
            owned.add(t);
        }
        return t;
    }

    private static Tensor track(List<Tensor> created, Tensor t) {
        created.add(t);
        return t;
    }

    private static float[] withSpecials(float[] values) {
        System.arraycopy(SPECIAL, 0, values, 0, Math.min(SPECIAL.length, values.length));
        return values;
    }

    /** Closes every non-null tensor, last first (closing is idempotent). */
    private static void closeAll(List<Tensor> tensors) {
        for (int i = tensors.size() - 1; i >= 0; i--) {
            Tensor t = tensors.get(i);
            if (t != null) {
                t.close();
            }
        }
    }
}
