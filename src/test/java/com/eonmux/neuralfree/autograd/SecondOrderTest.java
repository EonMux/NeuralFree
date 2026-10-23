package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.GpuReduceOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.ElementwiseOp;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Second-order gradients of the core gradient functions: Hessian-vector products of an MLP loss
 * against finite differences of its gradient, exact second derivatives through {@code contiguous}, a same-shape add
 * and {@code sum}, {@link GradientCheckUtil#gradGradCheck} for softmax, log-softmax, every activation of the partials
 * table and the routing primitives ({@code max}/{@code min}, {@code indexSelectRows}, {@code segmentReduce}), the
 * closed-form {@code f''} of the smooth activations, the create-graph refusal of a checkpointed segment, the recording
 * rule of the backward primitives, the dense gradient a transposed leaf receives and the {@code sgn} derivative of
 * {@code abs} at {@code +-0}. Every case asserts that {@link ComputationGraph#size()} is back at its value before the
 * case once its {@link ComputationGraph.StepScope} closed. Runs on the CPU reference backend and on each GPU API of
 * {@link TrainParity}.
 *
 * <p><b>Kinks.</b> The inputs of piecewise functions keep at least 0.09 from every kink, far above the
 * finite-difference step {@link #EPSILON}; the extremum and segment-maximum inputs differ by at least 0.05 within a
 * row or segment.</p>
 */
@RunWith(Parameterized.class)
public class SecondOrderTest {

    private static final float EPSILON = 1e-3f;
    private static final float TOLERANCE = 1e-2f;
    private static final float FORMULA_TOLERANCE = 1e-4f;
    private static final float[] AWAY_FROM_KINKS = {-0.93f, -0.61f, -0.37f, -0.18f, 0.21f, 0.44f, 0.67f, 0.89f};
    private static final float[] SMOOTH_POINTS = {-2.3f, -1.3f, -0.7f, -0.2f, 0.15f, 0.6f, 1.1f, 1.7f};
    private static final float ELU_ALPHA = 0.7f;
    private static final float LEAKY_SLOPE = 0.1f;
    private static final float CLAMP_BOUND = 0.8f;
    private static final double SQRT_TWO_OVER_PI = Math.sqrt(2.0 / Math.PI);
    private static final double GELU_CUBIC = 0.044715;

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
    private ComputationGraph graph;
    private final List<Tensor> owned = new ArrayList<>();

    public SecondOrderTest(String label, TrainParity.Api api) {
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
        graph = ComputationGraph.current();
        graph.clear();
    }

    @After
    public void tearDown() {
        try {
            graph.clear();
            ComputationGraph.setGradEnabled(true);
            for (Tensor t : owned) {
                Tensor grad = t.requiresGrad() ? t.grad() : null;
                if (grad != null) {
                    t.setGrad(null);
                    grad.close();
                }
                t.close();
            }
        } finally {
            if (session != null) {
                session.close();
            } else if (backend != null) {
                backend.close();
            }
        }
    }

    // ------------------------------------------------------------------ Hessian-vector products

    /** {@code mean((tanh(x W1 + b1) W2 - target)^2)} over the parameters {@code (W1, b1, W2)}. */
    private Function<List<Tensor>, Tensor> mlpLoss() {
        Tensor x = keep(tensor(TrainParity.uniform(11, 15, -1f, 1f), 5, 3));
        Tensor target = keep(tensor(TrainParity.uniform(12, 10, -1f, 1f), 5, 2));
        return p -> {
            Tensor h = TensorOps.tanh(TensorOps.add(TensorOps.matmul(x, p.get(0)), p.get(1)));
            Tensor d = TensorOps.sub(TensorOps.matmul(h, p.get(2)), target);
            return TensorOps.mean(TensorOps.mul(d, d));
        };
    }

    private static final Shape[] MLP_SHAPES = {Shape.of(3, 4), Shape.of(4), Shape.of(4, 2)};

    private static float[][] mlpParameters() {
        return new float[][]{TrainParity.uniform(13, 12, -0.8f, 0.8f), TrainParity.uniform(14, 4, -0.5f, 0.5f),
            TrainParity.uniform(15, 8, -0.8f, 0.8f)};
    }

    @Test
    public void mlpHessianVectorProductMatchesFiniteDifferences() {
        Function<List<Tensor>, Tensor> loss = mlpLoss();
        float[][] theta = mlpParameters();
        float[][] v = {TrainParity.uniform(21, 12, -1f, 1f), TrainParity.uniform(22, 4, -1f, 1f),
            TrainParity.uniform(23, 8, -1f, 1f)};
        float[][] analytical = hessianVector(loss, theta, MLP_SHAPES, v);
        float[][] plus = gradientAt(loss, shifted(theta, v, EPSILON), MLP_SHAPES);
        float[][] minus = gradientAt(loss, shifted(theta, v, -EPSILON), MLP_SHAPES);
        for (int i = 0; i < theta.length; i++) {
            for (int j = 0; j < theta[i].length; j++) {
                float numerical = (plus[i][j] - minus[i][j]) / (2.0f * EPSILON);
                assertClose("H v of parameter " + i + " element " + j, numerical, analytical[i][j], TOLERANCE);
            }
        }
    }

    @Test
    public void mlpGradGradCheck() {
        Function<List<Tensor>, Tensor> loss = mlpLoss();
        float[][] theta = mlpParameters();
        assertGradGrad("mlp", loss::apply, tensor(theta[0], 3, 4), tensor(theta[1], 4), tensor(theta[2], 4, 2));
    }

    // ------------------------------------------------------------------ exact second derivatives

    /** {@code sum(c * c)} with {@code c = x^T.contiguous()} (a recorded copy): {@code H w = 2 w}. */
    @Test
    public void secondDerivativeThroughContiguous() {
        float[] x = TrainParity.uniform(31, 6, -1f, 1f);
        float[] w = TrainParity.uniform(32, 6, 0.5f, 1.5f);
        float[][] hv = hessianVector(in -> {
            Tensor c = in.get(0).transpose().contiguous();
            return TensorOps.mul(c, c);
        }, new float[][]{x}, new Shape[]{Shape.of(2, 3)}, new float[][]{w});
        for (int j = 0; j < w.length; j++) {
            assertClose("contiguous element " + j, 2.0f * w[j], hv[0][j], 1e-6f);
        }
    }

    /** {@code sum((x + b)^2)} over one shape: {@code H w = 2 (w_x + w_b)} for both operands. */
    @Test
    public void secondDerivativeThroughSameShapeAdd() {
        float[] x = TrainParity.uniform(33, 6, -1f, 1f);
        float[] b = TrainParity.uniform(34, 6, -1f, 1f);
        float[] wx = TrainParity.uniform(35, 6, 0.5f, 1.5f);
        float[] wb = TrainParity.uniform(36, 6, 0.5f, 1.5f);
        Shape shape = Shape.of(2, 3);
        float[][] hv = hessianVector(in -> {
            Tensor s = TensorOps.add(in.get(0), in.get(1));
            return TensorOps.mul(s, s);
        }, new float[][]{x, b}, new Shape[]{shape, shape}, new float[][]{wx, wb});
        for (int j = 0; j < x.length; j++) {
            float expected = 2.0f * (wx[j] + wb[j]);
            assertClose("add x element " + j, expected, hv[0][j], 1e-5f);
            assertClose("add b element " + j, expected, hv[1][j], 1e-5f);
        }
    }

    /** {@code sum(x)^2}: {@code H w = 2 sum(w)} in every element. */
    @Test
    public void secondDerivativeThroughSum() {
        float[] x = TrainParity.uniform(37, 6, -1f, 1f);
        float[] w = TrainParity.uniform(38, 6, 0.5f, 1.5f);
        float[][] hv = hessianVector(in -> {
            Tensor s = TensorOps.sum(in.get(0));
            return TensorOps.mul(s, s);
        }, new float[][]{x}, new Shape[]{Shape.of(2, 3)}, new float[][]{w});
        double total = 0.0;
        for (float wj : w) {
            total += wj;
        }
        for (int j = 0; j < w.length; j++) {
            assertClose("sum element " + j, (float) (2.0 * total), hv[0][j], 1e-4f);
        }
    }

    // ------------------------------------------------------------------ softmax and activations

    @Test
    public void softmaxGradGrad() {
        assertGradGrad("softmax", in -> TensorOps.mul(TensorOps.softmax(in.get(0), 1), in.get(0)),
            tensor(AWAY_FROM_KINKS, 2, 4));
    }

    @Test
    public void logSoftmaxGradGrad() {
        assertGradGrad("logSoftmax", in -> TensorOps.mul(TensorOps.logSoftmax(in.get(0), 1), in.get(0)),
            tensor(AWAY_FROM_KINKS, 2, 4));
    }

    /** {@code act(x) * x} for every activation of the partials table, so {@code gy} depends on {@code x} too. */
    @Test
    public void activationGradGrad() {
        for (ElementwiseOp.Backward op : ElementwiseOp.Backward.values()) {
            Function<Tensor, Tensor> act = activation(op);
            assertGradGrad(op.name(), in -> TensorOps.mul(act.apply(in.get(0)), in.get(0)),
                tensor(AWAY_FROM_KINKS, 2, 4));
        }
    }

    /**
     * {@code d^2/dx^2 sum(act(x))} against the closed forms of the partials table of
     * {@link SecondOrderGradFunctions.ActivationBackwardGrad}.
     */
    @Test
    public void smoothActivationSecondDerivativesMatchTheClosedForms() {
        checkSecondDerivative(ElementwiseOp.Backward.SIGMOID, x -> {
            double s = sigmoid(x);
            return s * (1 - s) * (1 - 2 * s);
        });
        checkSecondDerivative(ElementwiseOp.Backward.TANH, x -> {
            double t = Math.tanh(x);
            return -2 * t * (1 - t * t);
        });
        checkSecondDerivative(ElementwiseOp.Backward.SOFTPLUS, x -> sigmoid(x) * (1 - sigmoid(x)));
        checkSecondDerivative(ElementwiseOp.Backward.SILU, x -> {
            double s = sigmoid(x);
            return s * (1 - s) * (2 + x * (1 - 2 * s));
        });
        checkSecondDerivative(ElementwiseOp.Backward.GELU, x -> {
            double du = SQRT_TWO_OVER_PI * (1 + 3 * GELU_CUBIC * x * x);
            double t = Math.tanh(SQRT_TWO_OVER_PI * (x + GELU_CUBIC * x * x * x));
            return (1 - t * t) * (du + 3 * SQRT_TWO_OVER_PI * GELU_CUBIC * x * x - x * t * du * du);
        });
        checkSecondDerivative(ElementwiseOp.Backward.MISH, x -> {
            double s = sigmoid(x);
            double tt = Math.tanh(Math.log1p(Math.exp(x)));
            return (1 - tt * tt) * s * (2 + x * (1 - s - 2 * tt * s));
        });
        checkSecondDerivative(ElementwiseOp.Backward.ELU, x -> x > 0 ? 0.0 : ELU_ALPHA * Math.exp(x));
    }

    private void checkSecondDerivative(ElementwiseOp.Backward op, DoubleUnaryOperator closedForm) {
        float[] ones = new float[SMOOTH_POINTS.length];
        Arrays.fill(ones, 1.0f);
        Function<Tensor, Tensor> act = activation(op);
        float[][] hv = hessianVector(in -> act.apply(in.get(0)), new float[][]{SMOOTH_POINTS},
            new Shape[]{Shape.of(2, 4)}, new float[][]{ones});
        for (int j = 0; j < SMOOTH_POINTS.length; j++) {
            assertClose(op + " f'' at " + SMOOTH_POINTS[j], (float) closedForm.applyAsDouble(SMOOTH_POINTS[j]),
                hv[0][j], FORMULA_TOLERANCE);
        }
    }

    // ------------------------------------------------------------------ routing primitives

    @Test
    public void maxAndMinGradGrad() {
        float[] x = {0.3f, -0.4f, 0.9f, 0.1f, -0.7f, 0.55f, 0.2f, -0.15f, 0.65f, -0.9f, 0.05f, 0.4f};
        assertGradGrad("max", in -> square(TensorOps.max(in.get(0), 1)), tensor(x, 3, 4));
        assertGradGrad("min", in -> square(TensorOps.min(in.get(0), 1)), tensor(x, 3, 4));
    }

    @Test
    public void indexSelectRowsGradGrad() {
        float[] table = TrainParity.uniform(51, 12, -1f, 1f);
        Tensor indices = keep(Tensor.fromArray(backend, new int[]{0, 2, 2, 3, 1}));
        assertGradGrad("indexSelectRows with a padding row",
            in -> square(TensorOps.indexSelectRows(in.get(0), indices, false, 2)), tensor(table, 4, 3));
        Tensor outOfRange = keep(Tensor.fromArray(backend, new int[]{0, 5, -1, 3}));
        assertGradGrad("indexSelectRows clamped",
            in -> square(TensorOps.indexSelectRows(in.get(0), outOfRange, true)), tensor(table, 4, 3));
    }

    /** Three segments over four source rows, the middle one empty: {@code ptr {0, 2, 2, 4}}, edges {3, 0, 1, 2}. */
    @Test
    public void segmentReduceGradGrad() {
        Tensor ptr = keep(Tensor.fromArray(backend, new int[]{0, 2, 2, 4}));
        Tensor edges = keep(Tensor.fromArray(backend, new int[]{3, 0, 1, 2}));
        float[] src = {0.8f, -0.3f, 0.1f, 0.6f, -0.5f, 0.25f, 0.35f, -0.75f};
        for (GpuReduceOps.SegmentOp op : GpuReduceOps.SegmentOp.values()) {
            assertGradGrad("segmentReduce " + op,
                in -> square(TensorOps.segmentReduce(in.get(0), ptr, edges, 3, op, 0.0f)), tensor(src, 4, 2));
        }
    }

    // ------------------------------------------------------------------ recording rules

    @Test
    public void createGraphThroughACheckpointThrows() {
        Tensor x = keep(leaf(new float[]{1f, 2f, 3f}, 3));
        int before = graph.size();
        try (ComputationGraph.StepScope step = graph.step()) {
            Tensor y = CheckpointFunction.checkpoint(t -> TensorOps.mul(t, t), x);
            Tensor loss = TensorOps.sum(y);
            int recorded = graph.size();
            try {
                Autograd.grad(loss, Collections.singletonList(x), null, false, true);
                fail("Autograd.grad with createGraph must refuse a checkpointed segment");
            } catch (UnsupportedOperationException expected) {
                assertEquals("nothing was recorded", recorded, graph.size());
            }
            try {
                graph.backward(loss, null, true, true);
                fail("backward with createGraph must refuse a checkpointed segment");
            } catch (UnsupportedOperationException expected) {
                assertNull("no gradient was delivered", x.grad());
            }
        }
        assertEquals(before, graph.size());
    }

    @Test
    public void backwardPrimitivesDoNotRecordOutsideACreateGraphPass() {
        Tensor x = keep(leaf(AWAY_FROM_KINKS, 2, 4));
        int before = graph.size();
        try (ComputationGraph.StepScope step = graph.step()) {
            Tensor y = TensorOps.tanh(x);
            assertTrue(y.requiresGrad());
            List<Tensor> results = new ArrayList<>();
            results.add(TensorOps.activationBackward(ElementwiseOp.Backward.TANH, null, y, x, 0f, 0f));
            results.add(TensorOps.softmaxBackward(y, x, 1));
            results.add(TensorOps.logSoftmaxBackward(y, x, 1));
            int size = graph.size();
            for (Tensor r : results) {
                try {
                    assertFalse("a backward primitive outside create-graph does not record", r.requiresGrad());
                    assertNull(r.gradFn());
                } finally {
                    r.close();
                }
            }
            assertEquals(size, graph.size());
        }
        assertEquals(before, graph.size());
    }

    @Test
    public void backwardPrimitivesRecordInsideACreateGraphPass() {
        Tensor x = keep(leaf(SMOOTH_POINTS, 2, 4));
        int before = graph.size();
        try (ComputationGraph.StepScope step = graph.step()) {
            Tensor loss = TensorOps.sum(TensorOps.sigmoid(x));
            Tensor g = Autograd.grad(loss, Collections.singletonList(x), null, false, true).get(0);
            assertNotNull(g);
            assertTrue("the gradient of a create-graph pass requires grad", g.requiresGrad());
            assertTrue("activationBackward recorded its second-order function",
                g.gradFn() instanceof SecondOrderGradFunctions.ActivationBackwardGrad);
        }
        assertEquals(before, graph.size());
    }

    @Test
    public void ownedGradientFunctionsSupportCreateGraph() {
        Shape s = Shape.of(2, 3);
        List<GradientFunction> fns = Arrays.asList(new GradFunctions.AddGrad(s, s), new GradFunctions.SubGrad(s, s),
            new GradFunctions.MulGrad(s, s), new GradFunctions.DivGrad(s, s), new GradFunctions.PowGrad(3f),
            new GradFunctions.NegGrad(), new GradFunctions.SqrtGrad(), new GradFunctions.ExpGrad(),
            new GradFunctions.LogGrad(), new GradFunctions.AddScalarGrad(s), new GradFunctions.MulScalarGrad(2f),
            new GradFunctions.AbsGrad(), new GradFunctions.ClampGrad(0f, 1f), new GradFunctions.ReluGrad(),
            new GradFunctions.GeluGrad(), new GradFunctions.SiluGrad(), new GradFunctions.SigmoidGrad(),
            new GradFunctions.TanhGrad(), new GradFunctions.LeakyReluGrad(0.1f), new GradFunctions.EluGrad(1f),
            new GradFunctions.SoftplusGrad(), new GradFunctions.MishGrad(), new GradFunctions.SoftmaxGrad(1),
            new GradFunctions.LogSoftmaxGrad(1), new GradFunctions.GluGrad(1), new GradFunctions.GegluGrad(1),
            new GradFunctions.MatmulGrad(), new GradFunctions.BmmGrad(), new GradFunctions.SumGrad(s, null),
            new GradFunctions.MeanGrad(s, null), new GradFunctions.MaxGrad(1, s), new GradFunctions.ReshapeGrad(s),
            new GradFunctions.TransposeGrad(null), new GradFunctions.SliceGrad(s, 0, 0, 1),
            new GradFunctions.CatGrad(0, new int[]{1, 2}), new GradFunctions.StackGrad(0, 2),
            new GradFunctions.GatherGrad(1, s), new GradFunctions.ScatterGrad(1),
            new SelectionGradFunctions.MinGrad(1, s), new IndexGradFunctions.IndexSelectRowsGrad(s, false, -1),
            new IndexGradFunctions.SegmentReduceGrad(GpuReduceOps.SegmentOp.MAX),
            new SecondOrderGradFunctions.ActivationBackwardGrad(ElementwiseOp.Backward.TANH, 0f, 0f, false, true,
                true),
            new SecondOrderGradFunctions.SoftmaxBackwardGrad(1, true, true),
            new SecondOrderGradFunctions.LogSoftmaxBackwardGrad(1, true, true),
            new SecondOrderGradFunctions.MaxBackwardGrad(1, Shape.of(2)),
            new SecondOrderGradFunctions.IndexAddRowsGrad(s, 4, false, -1),
            new SecondOrderGradFunctions.SegmentReduceBackwardGrad(GpuReduceOps.SegmentOp.SUM, s, Shape.of(4, 3)));
        for (GradientFunction fn : fns) {
            assertTrue(fn.getClass().getSimpleName() + " supports create-graph", fn.supportsCreateGraph());
        }
    }

    // ------------------------------------------------------------------ first-order contracts

    /** {@code x @ W^T} (Linear's weight layout): the leaf {@code W} receives a contiguous gradient of its shape. */
    @Test
    public void transposedLeafReceivesAContiguousGradient() {
        float[] xs = TrainParity.uniform(41, 12, -1f, 1f);
        Tensor x = keep(tensor(xs, 4, 3));
        Tensor w = keep(leaf(TrainParity.uniform(42, 6, -1f, 1f), 2, 3));
        int before = graph.size();
        try (ComputationGraph.StepScope step = graph.step()) {
            Tensor loss = keep(TensorOps.sum(TensorOps.matmul(x, w.transpose())));
            loss.backward();
        }
        assertEquals(before, graph.size());
        Tensor grad = w.grad();
        assertNotNull(grad);
        assertEquals(Shape.of(2, 3), grad.shape());
        assertTrue("the weight gradient is row-major from offset 0", grad.isContiguous());
        float[] values = grad.toFloatArray();
        for (int o = 0; o < 2; o++) {
            for (int i = 0; i < 3; i++) {
                double column = 0.0;
                for (int r = 0; r < 4; r++) {
                    column += xs[r * 3 + i];
                }
                assertClose("dW[" + o + ", " + i + "]", (float) column, values[o * 3 + i], 1e-5f);
            }
        }
    }

    /** {@code d|x|/dx = sgn(x)}: {@code +0} at {@code -0}, {@code +0} and NaN. */
    @Test
    public void absDerivativeIsSgn() {
        Tensor x = keep(leaf(new float[]{-0.0f, 0.0f, Float.NaN, 1.5f, -2.0f}, 5));
        int before = graph.size();
        try (ComputationGraph.StepScope step = graph.step()) {
            Tensor loss = keep(TensorOps.sum(TensorOps.abs(x)));
            loss.backward();
        }
        assertEquals(before, graph.size());
        float[] expected = {0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
        float[] actual = x.grad().toFloatArray();
        for (int j = 0; j < expected.length; j++) {
            assertEquals("abs' element " + j, Float.floatToRawIntBits(expected[j]),
                Float.floatToRawIntBits(actual[j]));
        }
    }

    // ------------------------------------------------------------------ helpers

    private Function<Tensor, Tensor> activation(ElementwiseOp.Backward op) {
        switch (op) {
            case RELU:
                return TensorOps::relu;
            case GELU:
                return TensorOps::gelu;
            case SILU:
                return TensorOps::silu;
            case SIGMOID:
                return TensorOps::sigmoid;
            case TANH:
                return TensorOps::tanh;
            case LEAKY_RELU:
                return t -> TensorOps.leakyRelu(t, LEAKY_SLOPE);
            case ELU:
                return t -> TensorOps.elu(t, ELU_ALPHA);
            case SOFTPLUS:
                return TensorOps::softplus;
            case MISH:
                return TensorOps::mish;
            case ABS:
                return TensorOps::abs;
            case CLAMP:
                return t -> TensorOps.clamp(t, -CLAMP_BOUND, CLAMP_BOUND);
            default:
                throw new IllegalArgumentException("no forward for " + op);
        }
    }

    private static Tensor square(Tensor t) {
        return TensorOps.mul(t, t);
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /** {@link GradientCheckUtil#gradGradCheck} passes and leaves the graph at its size. */
    private void assertGradGrad(String what, GradientCheckUtil.TensorFunction fn, Tensor... inputs) {
        for (Tensor input : inputs) {
            keep(input);
        }
        int before = graph.size();
        GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradGradCheck(fn, Arrays.asList(inputs),
            EPSILON, TOLERANCE);
        assertTrue(what + ": " + result, result.passed());
        assertEquals(what + " leaves the graph at its size", before, graph.size());
    }

    /**
     * {@code d/dx sum_i sum(G_i * w_i)} with {@code G = d sum(fn(x)) / dx} from a create-graph pass over leaves of
     * {@code values}, inside one step; asserts the graph is back at its size afterwards.
     */
    private float[][] hessianVector(Function<List<Tensor>, Tensor> fn, float[][] values, Shape[] shapes,
                                    float[][] w) {
        int before = graph.size();
        List<Tensor> leaves = new ArrayList<>();
        List<Tensor> created = new ArrayList<>();
        float[][] result = new float[values.length][];
        try {
            for (int i = 0; i < values.length; i++) {
                leaves.add(leaf(values[i], shapes[i].dims()));
            }
            try (ComputationGraph.StepScope step = graph.step()) {
                Tensor loss = track(created, TensorOps.sum(track(created, fn.apply(leaves))));
                List<Tensor> first = Autograd.grad(loss, leaves, null, false, true);
                created.addAll(first);
                Tensor dot = null;
                for (int i = 0; i < first.size(); i++) {
                    assertNotNull("input " + i + " receives a first-order gradient", first.get(i));
                    Tensor wi = track(created, tensor(w[i], shapes[i].dims()));
                    Tensor term = track(created, TensorOps.sum(track(created, TensorOps.mul(first.get(i), wi))));
                    dot = dot == null ? term : track(created, TensorOps.add(dot, term));
                }
                List<Tensor> second = Autograd.grad(dot, leaves, null, false, false);
                created.addAll(second);
                for (int i = 0; i < second.size(); i++) {
                    result[i] = second.get(i) == null ? new float[values[i].length] : second.get(i).toFloatArray();
                }
            }
        } finally {
            closeAll(created);
            closeAll(leaves);
        }
        assertEquals("the create-graph step leaves the graph at its size", before, graph.size());
        return result;
    }

    /** {@code d sum(fn(x)) / dx} from a first-order pass over leaves of {@code values}, inside one step. */
    private float[][] gradientAt(Function<List<Tensor>, Tensor> fn, float[][] values, Shape[] shapes) {
        int before = graph.size();
        List<Tensor> leaves = new ArrayList<>();
        List<Tensor> created = new ArrayList<>();
        float[][] result = new float[values.length][];
        try {
            for (int i = 0; i < values.length; i++) {
                leaves.add(leaf(values[i], shapes[i].dims()));
            }
            try (ComputationGraph.StepScope step = graph.step()) {
                Tensor loss = track(created, TensorOps.sum(track(created, fn.apply(leaves))));
                List<Tensor> grads = Autograd.grad(loss, leaves, null, false, false);
                created.addAll(grads);
                for (int i = 0; i < grads.size(); i++) {
                    assertNotNull(grads.get(i));
                    result[i] = grads.get(i).toFloatArray();
                }
            }
        } finally {
            closeAll(created);
            closeAll(leaves);
        }
        assertEquals(before, graph.size());
        return result;
    }

    private static float[][] shifted(float[][] values, float[][] direction, float step) {
        float[][] out = new float[values.length][];
        for (int i = 0; i < values.length; i++) {
            out[i] = new float[values[i].length];
            for (int j = 0; j < values[i].length; j++) {
                out[i][j] = values[i][j] + step * direction[i][j];
            }
        }
        return out;
    }

    private static void assertClose(String what, float expected, float actual, float tolerance) {
        float diff = Math.abs(expected - actual);
        float scale = Math.max(Math.abs(expected), Math.abs(actual));
        if (!(diff <= tolerance || diff <= tolerance * scale)) {
            fail(what + ": expected " + expected + " but was " + actual);
        }
    }

    private Tensor tensor(float[] data, int... dims) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(Shape.of(dims));
        }
    }

    private Tensor leaf(float[] data, int... dims) {
        try (Tensor values = tensor(data, dims)) {
            return values.withRequiresGrad(true);
        }
    }

    /** {@code t}, closed (with its gradient) when the test ends. */
    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }

    private static Tensor track(List<Tensor> created, Tensor t) {
        created.add(t);
        return t;
    }

    /** Closes every non-null tensor (closing is idempotent, so tensors the step closed are unaffected). */
    private static void closeAll(List<Tensor> tensors) {
        for (Tensor t : tensors) {
            if (t != null) {
                t.close();
            }
        }
    }
}
