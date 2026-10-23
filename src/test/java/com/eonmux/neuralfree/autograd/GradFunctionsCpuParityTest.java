package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntToDoubleFunction;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link GradFunctions} as {@code TensorOps} compositions. {@link CpuPath} runs every backward on the host reference
 * backend against host reference formulas in the compositions' operation order (bit for bit), and against independent
 * definitions for batched and vector matmul, bmm with {@code K != N}, sums over non-leading dims and negative dims,
 * and checks the ownership contract of saved tensors and
 * returned gradients. {@link GpuParity} runs each composed backward on every available GPU API against the host
 * backend under the tolerance classes of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract};
 * integer-valued operands make every sum and product exact, so data movement, reductions and products compare bit for
 * bit. Backwards that are one device op (the activations, Max, Softmax, LogSoftmax) have their GPU parity in
 * {@code TensorOpsParityTest} and {@code GpuReduceOpsParityTest}.
 */
@RunWith(Enclosed.class)
public class GradFunctionsCpuParityTest {

    private static final float SQRT_TWO_OVER_PI = (float) Math.sqrt(2.0 / Math.PI);

    /**
     * A tolerance class of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}: compares a GPU result
     * with the host one.
     */
    private interface Tol {
        void check(String what, float[] ref, float[] gpu);
    }

    private static final Tol BIT = TrainParity::assertBitIdentical;
    private static final Tol SINGLE = TrainParity::assertSingleRounding;
    private static final Tol DIV = TrainParity::assertDivSqrt;
    private static final Tol TRANSCENDENTAL = TrainParity::assertTranscendental;

    public static class CpuPath {
        private final Fixture fx = new Fixture();
        private final ComputeBackend host = TrainParity.hostBackend();

        @After
        public void tearDown() {
            ComputationGraph.current().clear();
            fx.close();
        }

        @Test
        public void activationGradientsMatchTheHostLoops() {
            float[] x = TrainParity.uniform(1, 64, -6f, 6f);
            x[0] = 0.0f;
            x[1] = -0.0f;
            x[2] = -1.5f;
            x[3] = 1.5f;
            float[] g = TrainParity.uniform(2, 64, -2f, 2f);
            Tensor tx = fx.floats(host, x, 8, 8);
            Tensor tg = fx.floats(host, g, 8, 8);
            float[] sig = fx.keep(TensorOps.sigmoid(tx)).toFloatArray();
            float[] th = fx.keep(TensorOps.tanh(tx)).toFloatArray();
            float[] el = fx.keep(TensorOps.elu(tx, 0.7f)).toFloatArray();
            expect("relu", saving(new GradFunctions.ReluGrad(), tx), tg, i -> x[i] > 0 ? g[i] : 0.0f);
            expect("leakyRelu", saving(new GradFunctions.LeakyReluGrad(0.1f), tx), tg,
                    i -> x[i] > 0 ? g[i] : 0.1f * g[i]);
            expect("elu", saving(new GradFunctions.EluGrad(0.7f), tx, fx.floats(host, el, 8, 8)), tg,
                    i -> x[i] > 0 ? g[i] : g[i] * (el[i] + 0.7f));
            expect("sigmoid", saving(new GradFunctions.SigmoidGrad(), fx.floats(host, sig, 8, 8)), tg,
                    i -> g[i] * sig[i] * (1.0f - sig[i]));
            expect("tanh", saving(new GradFunctions.TanhGrad(), fx.floats(host, th, 8, 8)), tg,
                    i -> g[i] * (1.0f - th[i] * th[i]));
            expect("gelu", saving(new GradFunctions.GeluGrad(), tx), tg, i -> g[i] * geluDerivative(x[i]));
            expect("silu", saving(new GradFunctions.SiluGrad(), tx), tg,
                    i -> g[i] * (sigmoidOf(x[i]) * (1.0f + x[i] * (1.0f - sigmoidOf(x[i])))));
            expect("softplus", saving(new GradFunctions.SoftplusGrad(), tx), tg, i -> g[i] * sigmoidOf(x[i]));
            expect("mish", saving(new GradFunctions.MishGrad(), tx), tg, i -> {
                float t = (float) Math.tanh((float) Math.log(1.0 + Math.exp(x[i])));
                return g[i] * (t + x[i] * (1.0f - t * t) * sigmoidOf(x[i]));
            });
            expect("abs", saving(new GradFunctions.AbsGrad(), tx), tg,
                    i -> x[i] > 0 ? g[i] : x[i] < 0 ? -g[i] : 0.0f);
            expect("clamp", saving(new GradFunctions.ClampGrad(-1.5f, 1.5f), tx), tg,
                    i -> x[i] < -1.5f || x[i] > 1.5f ? 0.0f : g[i]);
        }

        @Test
        public void arithmeticGradientsReduceBroadcastOperands() {
            float[] a = {1, -2, 3, 4, 2, -1};
            float[] b = {2, -1, 4};
            float[] g = {3, 1, -2, 2, -3, 1};
            Tensor ta = fx.floats(host, a, 2, 3);
            Tensor tb = fx.floats(host, b, 3);
            Tensor tg = fx.floats(host, g, 2, 3);
            Shape sa = Shape.of(2, 3);
            Shape sb = Shape.of(3);
            List<Tensor> add = fx.backward(new GradFunctions.AddGrad(sa, sb), tg);
            assertArrayEquals(g, add.get(0).toFloatArray(), 0f);
            assertEquals(sb, add.get(1).shape());
            assertArrayEquals(columnSums(i -> g[i]), add.get(1).toFloatArray(), 0f);
            List<Tensor> sub = fx.backward(new GradFunctions.SubGrad(sa, sb), tg);
            assertArrayEquals(columnSums(i -> -g[i]), sub.get(1).toFloatArray(), 0f);
            List<Tensor> mg = fx.backward(saving(new GradFunctions.MulGrad(sa, sb), ta, tb), tg);
            assertArrayEquals(map(6, i -> g[i] * b[i % 3]), mg.get(0).toFloatArray(), 0f);
            assertArrayEquals(columnSums(i -> g[i] * a[i]), mg.get(1).toFloatArray(), 0f);
            List<Tensor> dg = fx.backward(saving(new GradFunctions.DivGrad(sa, sb), ta, tb), tg);
            assertArrayEquals(map(6, i -> g[i] / b[i % 3]), dg.get(0).toFloatArray(), 0f);
            assertArrayEquals(columnSums(i -> g[i] * -(a[i] / (b[i % 3] * b[i % 3]))), dg.get(1).toFloatArray(),
                    0f);
            List<Tensor> partial = fx.backward(saving(new GradFunctions.MulGrad(sa, sb, true, false), ta, tb), tg);
            assertNotNull(partial.get(0));
            assertNull("an operand that needs no gradient gets none", partial.get(1));
        }

        @Test
        public void powerAndTranscendentalGradients() {
            float[] x = TrainParity.uniform(3, 12, 0.5f, 4f);
            float[] g = TrainParity.uniform(4, 12, -2f, 2f);
            Tensor tx = fx.floats(host, x, 3, 4);
            Tensor tg = fx.floats(host, g, 3, 4);
            float[] y = fx.keep(TensorOps.sqrt(tx)).toFloatArray();
            float[] e = fx.keep(TensorOps.exp(tx)).toFloatArray();
            assertArrayEquals(new float[12], fx.single(saving(new GradFunctions.PowGrad(0.0f), tx), tg), 0f);
            assertArrayEquals(map(12, i -> g[i] * (x[i] * 2.0f)),
                    fx.single(saving(new GradFunctions.PowGrad(2.0f), tx), tg), 0f);
            assertArrayEquals(map(12, i -> g[i] * ((float) Math.pow(x[i], 2.0f) * 3.0f)),
                    fx.single(saving(new GradFunctions.PowGrad(3.0f), tx), tg), 0f);
            TrainParity.assertDivSqrt("pow 0.5", map(12, i -> g[i] * (0.5f / (float) Math.sqrt(x[i]))),
                    fx.single(saving(new GradFunctions.PowGrad(0.5f), tx), tg));
            assertArrayEquals(map(12, i -> g[i] / (y[i] * 2.0f)),
                    fx.single(saving(new GradFunctions.SqrtGrad(), fx.floats(host, y, 3, 4)), tg), 0f);
            assertArrayEquals(map(12, i -> g[i] * e[i]),
                    fx.single(saving(new GradFunctions.ExpGrad(), fx.floats(host, e, 3, 4)), tg), 0f);
            assertArrayEquals(map(12, i -> g[i] / x[i]), fx.single(saving(new GradFunctions.LogGrad(), tx), tg), 0f);
        }

        @Test
        public void matrixProductGradientsFollowEveryRankRule() {
            int[][][] cases = {{{2, 3}, {3, 4}}, {{3}, {3, 4}}, {{2, 3}, {3}}, {{3}, {3}}, {{2, 2, 3}, {3, 4}},
                    {{3}, {2, 3, 4}}, {{2, 1, 2, 3}, {3, 3, 2}}, {{2, 2, 3}, {3}}};
            long seed = 10;
            for (int[][] c : cases) {
                MatmulCase mc = new MatmulCase(c[0], c[1], seed++);
                List<Tensor> grads = mc.backward(fx, host, new GradFunctions.MatmulGrad());
                String what = Arrays.toString(c[0]) + " @ " + Arrays.toString(c[1]);
                assertEquals(what + " ga shape", Shape.of(mc.shapeA), grads.get(0).shape());
                assertEquals(what + " gb shape", Shape.of(mc.shapeB), grads.get(1).shape());
                assertArrayEquals(what + " ga", mc.grad(true), grads.get(0).toFloatArray(), 0f);
                assertArrayEquals(what + " gb", mc.grad(false), grads.get(1).toFloatArray(), 0f);
            }
            MatmulCase bmm = new MatmulCase(new int[]{2, 2, 3}, new int[]{2, 3, 4}, 30);
            List<Tensor> bmmGrads = bmm.backward(fx, host, new GradFunctions.BmmGrad());
            assertArrayEquals("bmm ga", bmm.grad(true), bmmGrads.get(0).toFloatArray(), 0f);
            assertArrayEquals("bmm gb", bmm.grad(false), bmmGrads.get(1).toFloatArray(), 0f);
        }

        @Test
        public void reductionGradientsBroadcastBackOverAnyDims() {
            Shape in = Shape.of(2, 3, 4);
            float[] g8 = TrainParity.uniform(5, 8, -2f, 2f);
            assertArrayEquals(map(24, i -> g8[(i / 12) * 4 + i % 4]),
                    fx.single(new GradFunctions.SumGrad(in, new int[]{1}), fx.floats(host, g8, 2, 4)), 0f);
            float[] g3 = {1.5f, -2f, 4f};
            assertArrayEquals(map(24, i -> g3[(i / 4) % 3]),
                    fx.single(new GradFunctions.SumGrad(in, new int[]{0, -1}), fx.floats(host, g3, 3)), 0f);
            assertArrayEquals(map(24, i -> 2.5f),
                    fx.single(new GradFunctions.SumGrad(in, null), fx.floats(host, new float[]{2.5f})), 0f);
            float[] g6 = TrainParity.uniform(6, 6, -2f, 2f);
            assertArrayEquals(map(24, i -> g6[i / 4] * 0.25f),
                    fx.single(new GradFunctions.MeanGrad(in, new int[]{-1}), fx.floats(host, g6, 2, 3)), 0f);
            assertArrayEquals(map(24, i -> 3f * (1.0f / 24)),
                    fx.single(new GradFunctions.MeanGrad(in, new int[0]), fx.floats(host, new float[]{3f})), 0f);
            Tensor tx = fx.floats(host, new float[]{1, 3, 3, 2, 5, 5, 0, -1}, 2, 4);
            Tensor tg = fx.floats(host, new float[]{7, -2}, 2);
            for (int dim : new int[]{1, -1}) {
                GradFunctions.MaxGrad fn = new GradFunctions.MaxGrad(dim, tx.shape());
                fn.saveForBackward(tx, fx.keep(TensorOps.max(tx, dim)));
                assertArrayEquals("max dim " + dim, new float[]{0, 7, 0, 0, -2, 0, 0, 0}, fx.single(fn, tg), 0f);
            }
        }

        @Test
        public void viewAndSliceGradientsInvertTheView() {
            float[] g = map(24, i -> i);
            assertArrayEquals(g, fx.single(new GradFunctions.ReshapeGrad(Shape.of(6, 4)), fx.floats(host, g, 4, 6)),
                    0f);
            assertArrayEquals(new float[]{0, 2, 4, 1, 3, 5},
                    fx.single(new GradFunctions.TransposeGrad(null), fx.floats(host, map(6, i -> i), 3, 2)), 0f);
            float[] permuted = fx.single(new GradFunctions.TransposeGrad(new int[]{2, 0, 1}),
                    fx.floats(host, g, 4, 2, 3));
            assertArrayEquals(map(24, i -> g[(i % 4) * 6 + (i / 12) * 3 + (i / 4) % 3]), permuted, 0f);
            float[] g4 = {1, 2, 3, 4};
            assertArrayEquals(new float[]{0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0}, fx.single(
                    new GradFunctions.SliceGrad(Shape.of(4, 3), 0, 1, 3), fx.floats(host, map(6, i -> i + 1), 2, 3)),
                    0f);
            assertArrayEquals(new float[]{0, 0, 1, 2, 0, 0, 0, 3, 4, 0},
                    fx.single(new GradFunctions.SliceGrad(Shape.of(2, 5), 1, -3, -1), fx.floats(host, g4, 2, 2)), 0f);
            assertArrayEquals(g4,
                    fx.single(new GradFunctions.SliceGrad(Shape.of(2, 2), 1, 0, 2), fx.floats(host, g4, 2, 2)), 0f);
        }

        @Test
        public void catAndStackSplitTheGradient() {
            Tensor tg = fx.floats(host, map(6, i -> i + 1), 2, 3);
            List<Tensor> cat = fx.backward(new GradFunctions.CatGrad(-1, new int[]{1, 3}), tg);
            assertEquals(Shape.of(2, 1), cat.get(0).shape());
            assertArrayEquals(new float[]{1, 4}, cat.get(0).toFloatArray(), 0f);
            assertArrayEquals(new float[]{2, 3, 5, 6}, cat.get(1).toFloatArray(), 0f);
            List<Tensor> rows = fx.backward(new GradFunctions.CatGrad(0, new int[]{1, 2}, new boolean[]{false, true}),
                    tg);
            assertNull(rows.get(0));
            assertArrayEquals(new float[]{4, 5, 6}, rows.get(1).toFloatArray(), 0f);
            List<Tensor> stackLast = fx.backward(new GradFunctions.StackGrad(-1, 3), tg);
            assertEquals(Shape.of(2), stackLast.get(1).shape());
            assertArrayEquals(new float[]{2, 5}, stackLast.get(1).toFloatArray(), 0f);
            List<Tensor> stackFirst = fx.backward(new GradFunctions.StackGrad(0, 2), tg);
            assertArrayEquals(new float[]{4, 5, 6}, stackFirst.get(1).toFloatArray(), 0f);
        }

        @Test
        public void gatherAccumulatesAndScatterSplitsTheGradient() {
            GradFunctions.GatherGrad gather = new GradFunctions.GatherGrad(1, Shape.of(2, 3));
            gather.saveForBackward(fx.ints(host, new int[]{0, 0, 2, 1}, 2, 2));
            assertArrayEquals(new float[]{3, 0, 0, 0, 4, 3},
                    fx.single(gather, fx.floats(host, new float[]{1, 2, 3, 4}, 2, 2)), 0f);

            GradFunctions.ScatterGrad scatter = new GradFunctions.ScatterGrad(1);
            scatter.saveForBackward(fx.ints(host, new int[]{1, 0}, 2, 1), fx.floats(host, new float[]{9, 9}, 2, 1));
            List<Tensor> grads = fx.backward(scatter, fx.floats(host, map(6, i -> i + 1), 2, 3));
            assertEquals(3, grads.size());
            assertArrayEquals(new float[]{1, 0, 3, 0, 5, 6}, grads.get(0).toFloatArray(), 0f);
            assertNull("the index gets no gradient", grads.get(1));
            assertArrayEquals(new float[]{2, 4}, grads.get(2).toFloatArray(), 0f);
        }

        @Test
        public void softmaxGradientsAcceptNegativeDims() {
            float[] g = TrainParity.uniform(8, 12, -1f, 1f);
            Tensor tx = fx.floats(host, TrainParity.uniform(7, 12, -3f, 3f), 3, 4);
            Tensor tg = fx.floats(host, g, 3, 4);
            float[] y = fx.keep(TensorOps.softmax(tx, 1)).toFloatArray();
            float[] ly = fx.keep(TensorOps.logSoftmax(tx, 1)).toFloatArray();
            float[] softRef = new float[12];
            float[] logRef = new float[12];
            for (int r = 0; r < 12; r += 4) {
                float dot = 0.0f;
                float sum = 0.0f;
                for (int i = r; i < r + 4; i++) {
                    dot += y[i] * g[i];
                    sum += g[i];
                }
                for (int i = r; i < r + 4; i++) {
                    softRef[i] = y[i] * (g[i] - dot);
                    logRef[i] = g[i] - (float) Math.exp(ly[i]) * sum;
                }
            }
            for (int dim : new int[]{1, -1}) {
                assertArrayEquals("softmax dim " + dim, softRef,
                        fx.single(saving(new GradFunctions.SoftmaxGrad(dim), fx.floats(host, y, 3, 4)), tg), 0f);
                assertArrayEquals("logSoftmax dim " + dim, logRef,
                        fx.single(saving(new GradFunctions.LogSoftmaxGrad(dim), fx.floats(host, ly, 3, 4)), tg), 0f);
            }
        }

        @Test
        public void gatedUnitGradientsMatchTheHostLoops() {
            float[] x = TrainParity.uniform(9, 8, -3f, 3f);
            float[] g = TrainParity.uniform(10, 4, -1f, 1f);
            Tensor tx = fx.floats(host, x, 2, 4);
            Tensor tg = fx.floats(host, g, 2, 2);
            float[] gluRef = new float[8];
            float[] gegluRef = new float[8];
            for (int r = 0; r < 2; r++) {
                for (int d = 0; d < 2; d++) {
                    int ia = r * 4 + d;
                    int ib = ia + 2;
                    float gy = g[r * 2 + d];
                    float xb = x[ib];
                    float s = sigmoidOf(xb);
                    gluRef[ia] = gy * s;
                    gluRef[ib] = gy * x[ia] * s * (1.0f - s);
                    gegluRef[ia] = gy * (0.5f * xb * (1.0f + (float) Math.tanh(SQRT_TWO_OVER_PI
                            * (xb + 0.044715f * xb * xb * xb))));
                    gegluRef[ib] = gy * x[ia] * geluDerivative(xb);
                }
            }
            for (int dim : new int[]{1, -1}) {
                assertArrayEquals("glu dim " + dim, gluRef,
                        fx.single(saving(new GradFunctions.GluGrad(dim), tx), tg), 0f);
                assertArrayEquals("geglu dim " + dim, gegluRef,
                        fx.single(saving(new GradFunctions.GegluGrad(dim), tx), tg), 0f);
            }
        }

        @Test
        public void returnedGradientsAreGradOutputOrNewHandlesAndGradOutputStaysOpen() {
            float[] g = {1, 2, 3, 4};
            Tensor tg = fx.floats(host, g, 2, 2);
            Shape s = Shape.of(2, 2);
            List<GradientFunction> fns = Arrays.asList(new GradFunctions.AddGrad(s, s),
                    new GradFunctions.AddScalarGrad(s), new GradFunctions.ReshapeGrad(Shape.of(4)),
                    new GradFunctions.TransposeGrad(new int[]{0, 1}), new GradFunctions.SliceGrad(s, 0, 0, 2),
                    new GradFunctions.CatGrad(0, new int[]{2}));
            for (GradientFunction fn : fns) {
                String name = fn.getClass().getSimpleName();
                List<Tensor> grads = fn.backward(tg);
                // Rule G4: a pass-through gradient is gradOutput itself, which stays the caller's; every other
                // gradient is a distinct new handle owned by the caller.
                if (grads.size() == 2 && grads.get(0) != tg && grads.get(1) != tg) {
                    assertNotSame(name + " must return one new handle per input", grads.get(0), grads.get(1));
                }
                for (Tensor grad : grads) {
                    assertArrayEquals(name + " gradient values", g, grad.toFloatArray(), 0f);
                }
                for (Tensor grad : grads) {
                    if (grad != tg) {
                        grad.close();
                    }
                }
                assertArrayEquals(name + " left gradOutput readable", g, tg.toFloatArray(), 0f);
            }
        }

        @Test
        public void savedTensorsAreOwnedHandles() {
            Tensor a = Tensor.fromArray(host, new float[]{1, 2, 3});
            Tensor b = Tensor.fromArray(host, new float[]{4, 5, 6});
            GradFunctions.MulGrad mul = new GradFunctions.MulGrad(a.shape(), b.shape());
            mul.saveForBackward(a, b);
            a.close();
            b.close();
            List<Tensor> grads = fx.backward(mul, fx.floats(host, new float[]{1, 1, 1}, 3));
            assertArrayEquals("the handles outlive the caller's tensors", new float[]{4, 5, 6},
                    grads.get(0).toFloatArray(), 0f);

            Tensor y = fx.floats(host, new float[]{0.5f, 1.5f}, 2);
            GradFunctions.ExpGrad exp = new GradFunctions.ExpGrad();
            exp.saveForBackward(y);
            assertEquals(1, exp.savedTensors().size());
            exp.clearSavedTensors();
            assertTrue(exp.savedTensors().isEmpty());
            assertArrayEquals("clearing never closes the caller's tensor", new float[]{0.5f, 1.5f},
                    y.toFloatArray(), 0f);
            Tensor tg = fx.floats(host, new float[]{1, 1}, 2);
            assertThrows(IllegalStateException.class, () -> exp.backward(tg));
        }

        @Test
        public void recordingUsesNeedFlagsAndNoGradRecordsNothing() {
            ComputationGraph.current().clear();
            Tensor a = fx.keep(fx.floats(host, new float[]{1, 2}, 2).withRequiresGrad(true));
            Tensor b = fx.floats(host, new float[]{3, 4}, 2);
            try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
                Tensor c = fx.keep(TensorOps.mul(a, b));
                assertFalse("a result computed under noGrad does not require grad", c.requiresGrad());
                assertNull("no backward is built outside recording", c.gradFn());
                assertTrue(ComputationGraph.current().isEmpty());
            }
            Tensor c = fx.keep(TensorOps.mul(a, b));
            assertTrue(c.gradFn() instanceof GradFunctions.MulGrad);
            GradientFunction fn = (GradientFunction) c.gradFn();
            List<Tensor> grads = fx.backward(fn, fx.floats(host, new float[]{1, 1}, 2));
            assertArrayEquals(new float[]{3, 4}, grads.get(0).toFloatArray(), 0f);
            assertNull("b requires no gradient", grads.get(1));
            assertFalse(fn.savedTensors().isEmpty());
        }

        private void expect(String what, GradientFunction fn, Tensor gy, IntToDoubleFunction ref) {
            float[] got = fx.single(fn, gy);
            TrainParity.assertBitIdentical(what, map(got.length, ref), got);
        }

        private static float[] columnSums(IntToDoubleFunction term) {
            return map(3, c -> (float) term.applyAsDouble(c) + (float) term.applyAsDouble(3 + c));
        }
    }

    @RunWith(Parameterized.class)
    public static class GpuParity {
        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private final Fixture fx = new Fixture();
        private final ComputeBackend host = TrainParity.hostBackend();
        private TrainParity.Session session;

        public GpuParity(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            session = TrainParity.open(api);
        }

        @After
        public void tearDown() {
            try {
                fx.close();
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }

        @Test
        public void arithmeticGradients() {
            float[] a = integers(31, 12);
            float[] divisor = {1, -2, 2, -1};
            float[] g = integers(33, 12);
            float[] pos = TrainParity.uniform(34, 12, 0.5f, 50f);
            Shape sa = Shape.of(3, 4);
            Shape sb = Shape.of(4);
            compare("add", BIT, b -> run(b, new GradFunctions.AddGrad(sa, sb), g, 3, 4));
            compare("sub", BIT, b -> run(b, new GradFunctions.SubGrad(sa, sb), g, 3, 4));
            compare("mul", BIT, b -> fx.backward(saving(new GradFunctions.MulGrad(sa, sb), fx.floats(b, a, 3, 4),
                    fx.floats(b, divisor, 4)), fx.floats(b, g, 3, 4)));
            compare("div", TRANSCENDENTAL, b -> fx.backward(saving(new GradFunctions.DivGrad(sa, sb),
                    fx.floats(b, a, 3, 4), fx.floats(b, divisor, 4)), fx.floats(b, g, 3, 4)));
            compare("neg", BIT, b -> run(b, new GradFunctions.NegGrad(), g, 12));
            compare("mulScalar", SINGLE, b -> run(b, new GradFunctions.MulScalarGrad(0.3f), g, 12));
            compare("pow 2", SINGLE, b -> unary(b, new GradFunctions.PowGrad(2.0f), pos, g));
            compare("pow 0.5", TRANSCENDENTAL, b -> unary(b, new GradFunctions.PowGrad(0.5f), pos, g));
            compare("pow 2.5", TRANSCENDENTAL, b -> unary(b, new GradFunctions.PowGrad(2.5f), pos, g));
            compare("sqrt", DIV, b -> unary(b, new GradFunctions.SqrtGrad(), pos, g));
            compare("exp", SINGLE, b -> unary(b, new GradFunctions.ExpGrad(), pos, g));
            compare("log", DIV, b -> unary(b, new GradFunctions.LogGrad(), pos, g));
        }

        @Test
        public void matrixProductGradients() {
            int[][][] cases = {{{5, 7}, {7, 3}}, {{7}, {7, 3}}, {{5, 7}, {7}}, {{2, 5, 7}, {7, 3}}, {{7}, {2, 7, 3}},
                    {{3, 1, 4, 6}, {2, 6, 5}}};
            long seed = 40;
            for (int[][] c : cases) {
                MatmulCase mc = new MatmulCase(c[0], c[1], seed++);
                compare("matmul " + Arrays.toString(c[0]) + " @ " + Arrays.toString(c[1]), BIT,
                        b -> mc.backward(fx, b, new GradFunctions.MatmulGrad()));
            }
            MatmulCase bmm = new MatmulCase(new int[]{3, 4, 6}, new int[]{3, 6, 5}, 50);
            compare("bmm", BIT, b -> bmm.backward(fx, b, new GradFunctions.BmmGrad()));
        }

        @Test
        public void broadcastAndViewGradients() {
            Shape in = Shape.of(3, 4, 5);
            float[] g15 = integers(61, 15);
            float[] g60 = integers(62, 60);
            compare("sum dim 1", BIT, b -> run(b, new GradFunctions.SumGrad(in, new int[]{1}), g15, 3, 5));
            compare("sum all", BIT, b -> run(b, new GradFunctions.SumGrad(in, null), new float[]{3f}));
            compare("mean dims 0,2", SINGLE, b -> run(b, new GradFunctions.MeanGrad(in, new int[]{0, 2}),
                    Arrays.copyOf(g15, 4), 4));
            compare("reshape", BIT, b -> run(b, new GradFunctions.ReshapeGrad(in), g60, 12, 5));
            compare("permute", BIT, b -> run(b, new GradFunctions.TransposeGrad(new int[]{2, 0, 1}), g60, 5, 3, 4));
            compare("slice", BIT, b -> run(b, new GradFunctions.SliceGrad(in, 1, 1, 3), Arrays.copyOf(g60, 30), 3,
                    2, 5));
            compare("cat", BIT, b -> run(b, new GradFunctions.CatGrad(2, new int[]{2, 5}), g60, 3, 4, 5));
            compare("stack", BIT, b -> run(b, new GradFunctions.StackGrad(1, 4), g60, 3, 4, 5));
        }

        @Test
        public void indexAndGatedGradients() {
            float[] g9 = integers(71, 9);
            float[] g12 = integers(72, 12);
            compare("gather", BIT, b -> fx.backward(saving(new GradFunctions.GatherGrad(1, Shape.of(3, 4)),
                    fx.ints(b, new int[]{0, 3, 3, 1, 2, 0, 3, 3, 1}, 3, 3)), fx.floats(b, g9, 3, 3)));
            compare("scatter", BIT, b -> {
                GradFunctions.ScatterGrad fn = new GradFunctions.ScatterGrad(1);
                fn.saveForBackward(fx.ints(b, new int[]{2, 0, 1, 3, 1, 0}, 3, 2), fx.floats(b, new float[6], 3, 2));
                return fx.backward(fn, fx.floats(b, g12, 3, 4));
            });
            float[] gx = TrainParity.uniform(75, 48, -5f, 5f);
            float[] gg = TrainParity.uniform(76, 24, -1f, 1f);
            compare("glu", TRANSCENDENTAL, b -> fx.backward(saving(new GradFunctions.GluGrad(-1), fx.floats(b, gx, 4,
                    12)), fx.floats(b, gg, 4, 6)));
            compare("geglu", TRANSCENDENTAL, b -> fx.backward(saving(new GradFunctions.GegluGrad(0), fx.floats(b, gx,
                    8, 6)), fx.floats(b, gg, 4, 6)));
        }

        /** Runs {@code c} on the host and on the GPU and compares every gradient under {@code tol}. */
        private void compare(String what, Tol tol, Function<ComputeBackend, List<Tensor>> c) {
            List<Tensor> ref = c.apply(host);
            List<Tensor> dev = c.apply(session.backend());
            assertEquals(what + ": gradient count", ref.size(), dev.size());
            for (int i = 0; i < ref.size(); i++) {
                assertEquals(what + ": gradient " + i + " present", ref.get(i) != null, dev.get(i) != null);
                if (ref.get(i) != null) {
                    assertEquals(what + ": shape " + i, ref.get(i).shape(), dev.get(i).shape());
                    tol.check(what + ": gradient " + i, ref.get(i).toFloatArray(), dev.get(i).toFloatArray());
                }
            }
        }

        /** {@code fn} with the vector {@code saved} as its one saved tensor, run on the gradient {@code g}. */
        private List<Tensor> unary(ComputeBackend b, GradFunctions.SavingGradFn fn, float[] saved, float[] g) {
            return fx.backward(saving(fn, fx.floats(b, saved, saved.length)), fx.floats(b, g, g.length));
        }

        /** {@code fn.backward} of the gradient {@code g} shaped {@code dims} on backend {@code b}. */
        private List<Tensor> run(ComputeBackend b, GradientFunction fn, float[] g, int... dims) {
            return fx.backward(fn, fx.floats(b, g, dims));
        }
    }

    /**
     * {@code fn} after saving {@code saved} in the order its {@code saveForBackward} takes them (the input before
     * the output); for the grad functions whose {@code saveForBackward} only stores its arguments.
     */
    private static <F extends GradFunctions.SavingGradFn> F saving(F fn, Tensor... saved) {
        fn.save(saved);
        return fn;
    }

    /** Tensors and backward functions a test created, released at the end of the test. */
    private static final class Fixture implements AutoCloseable {
        private final List<Tensor> owned = new ArrayList<>();
        private final List<GradientFunction> functions = new ArrayList<>();

        Tensor keep(Tensor t) {
            assertNotNull(t);
            owned.add(t);
            return t;
        }

        Tensor floats(ComputeBackend b, float[] data, int... dims) {
            return keep(keep(Tensor.fromArray(b, data)).reshape(Shape.of(dims)));
        }

        Tensor ints(ComputeBackend b, int[] data, int... dims) {
            return keep(keep(Tensor.fromArray(b, data)).reshape(Shape.of(dims)));
        }

        /** {@code fn.backward(gy)}, keeping every returned gradient and the function for release. */
        List<Tensor> backward(GradientFunction fn, Tensor gy) {
            functions.add(fn);
            List<Tensor> grads = fn.backward(gy);
            grads.stream().filter(Objects::nonNull).forEach(owned::add);
            return grads;
        }

        /** The values of the first gradient. */
        float[] single(GradientFunction fn, Tensor gy) {
            Tensor grad = backward(fn, gy).get(0);
            assertNotNull(grad);
            return grad.toFloatArray();
        }

        @Override
        public void close() {
            functions.forEach(GradientFunction::clearSavedTensors);
            for (int i = owned.size() - 1; i >= 0; i--) {
                owned.get(i).close();
            }
            owned.clear();
            functions.clear();
        }
    }

    /**
     * A matmul case with integer operands in {@code [-3, 3]}, so every product and sum is exact, and its gradients
     * from the definition {@code out[bt, i, j] = sum_k A[bt, i, k] * B[bt, k, j]} over the broadcast batch, with a
     * rank-1 {@code a} as {@code [1, K]} and a rank-1 {@code b} as {@code [K, 1]}.
     */
    private static final class MatmulCase {
        final int[] shapeA;
        final int[] shapeB;
        final float[] a;
        final float[] b;
        final float[] g;
        private final int m;
        private final int k;
        private final int n;
        private final int[] batchA;
        private final int[] batchB;
        private final int[] batch;

        MatmulCase(int[] shapeA, int[] shapeB, long seed) {
            this.shapeA = shapeA;
            this.shapeB = shapeB;
            m = shapeA.length > 1 ? shapeA[shapeA.length - 2] : 1;
            k = shapeA[shapeA.length - 1];
            n = shapeB.length > 1 ? shapeB[shapeB.length - 1] : 1;
            batchA = Arrays.copyOf(shapeA, Math.max(0, shapeA.length - 2));
            batchB = Arrays.copyOf(shapeB, Math.max(0, shapeB.length - 2));
            batch = new int[Math.max(batchA.length, batchB.length)];
            for (int i = 0; i < batch.length; i++) {
                int ia = i - batch.length + batchA.length;
                int ib = i - batch.length + batchB.length;
                batch[i] = Math.max(ia < 0 ? 1 : batchA[ia], ib < 0 ? 1 : batchB[ib]);
            }
            a = integers(seed, count(shapeA));
            b = integers(seed + 100, count(shapeB));
            g = integers(seed + 200, count(batch) * m * n);
        }

        /** {@code fn} (a matmul or bmm backward) on this case's operands and gradient on backend {@code b}. */
        List<Tensor> backward(Fixture fx, ComputeBackend b, GradFunctions.SavingGradFn fn) {
            return fx.backward(saving(fn, fx.floats(b, a, shapeA), fx.floats(b, this.b, shapeB)),
                    fx.floats(b, g, outShape()));
        }

        /** The output shape {@code TensorOps.matmul} produces. */
        int[] outShape() {
            if (shapeA.length <= 2 && shapeB.length <= 2) {
                if (shapeA.length == 1 || shapeB.length == 1) {
                    return new int[]{shapeA.length == 1 ? n : m};
                }
                return new int[]{m, n};
            }
            int[] out = Arrays.copyOf(batch, batch.length + 2);
            out[batch.length] = m;
            out[batch.length + 1] = n;
            return out;
        }

        /** The gradient of {@code a} ({@code left}) or {@code b}, summed over the batch dims it was broadcast along. */
        float[] grad(boolean left) {
            double[] out = new double[left ? a.length : b.length];
            int[] idx = new int[batch.length];
            for (int bt = 0; bt < count(batch); bt++) {
                for (int i = batch.length - 1, rem = bt; i >= 0; rem /= batch[i], i--) {
                    idx[i] = rem % batch[i];
                }
                int aOff = offset(idx, batchA) * m * k;
                int bOff = offset(idx, batchB) * k * n;
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        double gy = g[bt * m * n + i * n + j];
                        for (int kk = 0; kk < k; kk++) {
                            if (left) {
                                out[aOff + i * k + kk] += gy * b[bOff + kk * n + j];
                            } else {
                                out[bOff + kk * n + j] += a[aOff + i * k + kk] * gy;
                            }
                        }
                    }
                }
            }
            return map(out.length, i -> (float) out[i]);
        }

        /** The matrix index into an operand's own batch dims for broadcast batch index {@code idx}. */
        private static int offset(int[] idx, int[] own) {
            int lead = idx.length - own.length;
            int o = 0;
            for (int i = 0; i < own.length; i++) {
                o = o * own[i] + (own[i] == 1 ? 0 : idx[lead + i]);
            }
            return o;
        }
    }

    private static int count(int[] dims) {
        return Math.toIntExact(Shape.of(dims).numElements());
    }

    /** {@code n} integers in {@code [-3, 3]} as floats. */
    private static float[] integers(long seed, int n) {
        float[] u = TrainParity.uniform(seed, n, -3.5f, 3.5f);
        return map(n, i -> Math.round(u[i]));
    }

    private static float[] map(int n, IntToDoubleFunction f) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) f.applyAsDouble(i);
        }
        return out;
    }

    private static float sigmoidOf(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    /** The tanh-form GELU derivative in the composition's float operation order. */
    private static float geluDerivative(float x) {
        float x3 = x * x * x;
        float inner = SQRT_TWO_OVER_PI * (x + 0.044715f * x3);
        float tanhInner = (float) Math.tanh(inner);
        float sech2 = 1.0f - tanhInner * tanhInner;
        float dInner = SQRT_TWO_OVER_PI * (1.0f + 3.0f * 0.044715f * x * x);
        return 0.5f * (1.0f + tanhInner) + 0.5f * x * sech2 * dInner;
    }
}
