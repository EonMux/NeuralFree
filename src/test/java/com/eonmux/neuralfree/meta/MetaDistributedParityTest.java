package com.eonmux.neuralfree.meta;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.CheckpointFunction;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.distributed.FederatedTrainer;
import com.eonmux.neuralfree.distributed.PipelineParallel;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.precision.QATModule;
import com.eonmux.neuralfree.precision.Quantizer;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
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
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The meta, distributed and precision utilities and the parameter and checkpoint helpers on each GPU API against
 * their CPU path (a backend without the train library). Data movement, fake quantization on exactly representable
 * quotients, its straight-through gradient and pipeline gradients on small integers are compared bit for bit;
 * training runs, which chain reductions, GEMMs and transcendentals over a few steps, are compared with a relative
 * tolerance of 1e-4. Each scenario also checks its own expected values on the CPU path, that device results stay on
 * the device backend, and that every training run leaves the caller's autograd graph at its size before the run.
 */
@RunWith(Enclosed.class)
public class MetaDistributedParityTest {

    static final float TRAINING_TOLERANCE = 1e-4f;

    /** The CPU path, on a backend without the train library. */
    public static class HostPath {
        private ComputeBackend host;

        @Before
        public void setUp() {
            ComputationGraph.current().clear();
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }

        @Test
        public void parameterCopiesAreIndependent() {
            Scenarios.parameterCopy(host);
        }

        @Test
        public void checkpointGradientMatchesDirectGradient() {
            float[] grad = Scenarios.checkpointGradient(host);
            float[] x = Scenarios.CHECKPOINT_INPUT;
            for (int i = 0; i < x.length; i++) {
                assertEquals("d(sum x^2)/dx at " + i, 2 * x[i], grad[i], 0.0f);
            }
        }

        @Test
        public void fakeQuantizationKeepsGridValues() {
            float[] out = Scenarios.fakeQuantize(host);
            float[] expected = Scenarios.fakeQuantizeExpected();
            TrainParity.assertBitIdentical("cpu fake quantization", expected, out);
        }

        @Test
        public void quantizerDequantizesOnTheSourceBackend() {
            Scenarios.quantize(host);
        }

        @Test
        public void pipelineKeepsMicroBatchOrder() {
            float[] out = Scenarios.pipeline(host, host, 3);
            TrainParity.assertBitIdentical("pipeline", Scenarios.pipelineExpected(), out);
        }

        @Test
        public void trainingRunsChangeTheWeights() {
            float[] initial = Scenarios.initialWeights(3, 1, 11);
            assertFalse("MAML moved the weights", Arrays.equals(initial, Scenarios.maml(host)));
            assertFalse("federated meta moved the weights", Arrays.equals(initial, Scenarios.federatedMeta(host)));
            assertFalse("FedProx moved the weights", Arrays.equals(initial, Scenarios.federated(host)));
            float[] student = Scenarios.distill(host);
            assertFalse("distillation moved the student", Arrays.equals(Scenarios.initialWeights(3, 2, 21),
                    Arrays.copyOf(student, 6)));
        }

        @Test
        public void distillationTermAloneDrivesTheStudent() {
            float[] student = Scenarios.distillOnly(host);
            assertFalse("the divergence term moved the student", Arrays.equals(Scenarios.initialWeights(3, 2, 21),
                    Arrays.copyOf(student, 6)));
            assertTrue("the divergence decreased", student[6] > student[student.length - 1]);
        }

        @Test
        public void pipelineStageFailureIsRethrownAndThePipelineStaysUsable() {
            Scenarios.pipelineFailure(host);
        }

        @Test
        public void pipelineBackwardMatchesTheUnsplitGradients() {
            TrainParity.assertBitIdentical("pipeline backward", Scenarios.pipelineBackwardExpected(),
                    Scenarios.pipelineBackward(host, host, 3));
        }

        @Test
        public void fakeQuantizationGradientIsTheClippedStraightThroughEstimator() {
            TrainParity.assertBitIdentical("fake quantization gradient", Scenarios.fakeQuantizeGradientExpected(),
                    Scenarios.fakeQuantizeGradient(host));
        }

        /**
         * One second-order MAML iteration (one task, one support and one query pair, one inner step) against the
         * closed form {@code W - beta * (I - 2 alpha xs^T xs) * 2 xq^T (xq W' - yq)}, which differs from the
         * first-order update by the Hessian term.
         */
        @Test
        public void secondOrderMamlMatchesTheClosedForm() {
            float[] got = Scenarios.secondOrderMamlSingleStep(host);
            double[] expected = Scenarios.secondOrderMamlSingleStepExpected(false);
            double[] firstOrder = Scenarios.secondOrderMamlSingleStepExpected(true);
            boolean differs = false;
            for (int i = 0; i < expected.length; i++) {
                assertEquals("element " + i, expected[i], got[i], 1e-5 * Math.max(1.0, Math.abs(expected[i])));
                differs |= Math.abs(expected[i] - firstOrder[i]) > 1e-3;
            }
            assertTrue("the scenario exercises the Hessian term", differs);
        }

        @Test
        public void secondOrderMamlTrainsMultipleStepsAndTasks() {
            float[] initial = Scenarios.initialWeights(3, 1, 11);
            assertFalse("second-order MAML moved the weights",
                    Arrays.equals(initial, Arrays.copyOf(Scenarios.secondOrderMaml(host), 3)));
        }

        @Test(expected = IllegalArgumentException.class)
        public void distillerRejectsATeacherSharingTheStudentsParameters() {
            Linear model = new Linear(host, Scenarios.initialWeights(3, 2, 21), 3, 2);
            try (AutogradMse loss = new AutogradMse()) {
                new KnowledgeDistiller(model, model, 2.0f, 0.5f, loss).close();
            } finally {
                model.close();
            }
        }
    }

    /** The device path on each GPU API, against the CPU path. */
    @RunWith(Parameterized.class)
    public static class Device {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;
        private ComputeBackend gpu;
        private ComputeBackend host;

        public Device(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            ComputationGraph.current().clear();
            session = TrainParity.open(api);
            gpu = session.backend();
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            try {
                ComputationGraph.current().clear();
                TrainGpu.release(host);
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }

        @Test
        public void parameterCopiesAreIndependent() {
            Scenarios.parameterCopy(gpu);
        }

        @Test
        public void checkpointGradientMatchesCpuPath() {
            TrainParity.assertSingleRounding(api + " checkpoint gradient", Scenarios.checkpointGradient(host),
                    Scenarios.checkpointGradient(gpu));
        }

        @Test
        public void fakeQuantizationMatchesCpuPath() {
            TrainParity.assertSingleRounding(api + " fake quantization", Scenarios.fakeQuantize(host),
                    Scenarios.fakeQuantize(gpu));
        }

        @Test
        public void quantizerDequantizesOnTheSourceBackend() {
            assertEquals(api + " quantization params", Scenarios.quantize(host), Scenarios.quantize(gpu));
        }

        @Test
        public void pipelineMatchesCpuPath() {
            TrainParity.assertBitIdentical(api + " pipeline", Scenarios.pipelineExpected(),
                    Scenarios.pipeline(gpu, gpu, 3));
            TrainParity.assertBitIdentical(api + " pipeline from host stage", Scenarios.pipelineExpected(),
                    Scenarios.pipeline(host, gpu, 2));
        }

        @Test
        public void pipelineBackwardMatchesCpuPath() {
            TrainParity.assertBitIdentical(api + " pipeline backward", Scenarios.pipelineBackwardExpected(),
                    Scenarios.pipelineBackward(gpu, gpu, 3));
            TrainParity.assertBitIdentical(api + " pipeline backward from host stage",
                    Scenarios.pipelineBackwardExpected(), Scenarios.pipelineBackward(host, gpu, 2));
        }

        @Test
        public void fakeQuantizationGradientMatchesCpuPath() {
            TrainParity.assertBitIdentical(api + " fake quantization gradient", Scenarios.fakeQuantizeGradient(host),
                    Scenarios.fakeQuantizeGradient(gpu));
        }

        @Test
        public void mamlMatchesCpuPath() {
            Scenarios.assertClose(api + " MAML", Scenarios.maml(host), Scenarios.maml(gpu));
        }

        @Test
        public void secondOrderMamlMatchesCpuPath() {
            Scenarios.assertClose(api + " second-order MAML", Scenarios.secondOrderMaml(host),
                    Scenarios.secondOrderMaml(gpu));
            Scenarios.assertClose(api + " second-order MAML step", Scenarios.secondOrderMamlSingleStep(host),
                    Scenarios.secondOrderMamlSingleStep(gpu));
        }

        @Test
        public void federatedMetaLearnerMatchesCpuPath() {
            Scenarios.assertClose(api + " federated meta", Scenarios.federatedMeta(host), Scenarios.federatedMeta(gpu));
        }

        @Test
        public void federatedTrainerMatchesCpuPath() {
            Scenarios.assertClose(api + " FedProx", Scenarios.federated(host), Scenarios.federated(gpu));
        }

        @Test
        public void distillationMatchesCpuPath() {
            Scenarios.assertClose(api + " distillation", Scenarios.distill(host), Scenarios.distill(gpu));
        }
    }

    /** Scenario bodies shared by both paths; each returns the values the paths must agree on. */
    static final class Scenarios {
        static final float[] CHECKPOINT_INPUT = {0.5f, -1.25f, 2.0f, 3.5f};

        private Scenarios() {
        }

        /** copy() and copyTensor() hold their own values; copyFrom() writes in place and keeps the gradient. */
        static void parameterCopy(ComputeBackend b) {
            float[] values = {1f, -2f, 3.5f, -0.0f, 7f, 8.25f};
            float[] other = {9f, 10f, -11f, 12f, 13f, 14.5f};
            Parameter p = new Parameter(Tensor.fromArray(b, new float[][]{{1f, -2f, 3.5f}, {-0.0f, 7f, 8.25f}}), "w");
            Parameter copy = p.copy();
            try (Tensor snapshot = p.copyTensor();
                 Tensor source = Tensor.fromArray(b, new float[][]{{9f, 10f, -11f}, {12f, 13f, 14.5f}});
                 Tensor grad = Tensor.zeros(b, p.shape())) {
                assertTrue("copy requires grad", copy.tensor().requiresGrad());
                assertFalse("snapshot does not require grad", snapshot.requiresGrad());
                assertSame("copy backend", b, copy.backend());
                assertNotSame("copy storage", p.tensor().storage().gpuBuffer(), copy.tensor().storage().gpuBuffer());
                p.setGrad(grad);
                p.copyFrom(source);
                TrainParity.assertBitIdentical("copyFrom", other, p.tensor().toFloatArray());
                TrainParity.assertBitIdentical("copy", values, copy.tensor().toFloatArray());
                TrainParity.assertBitIdentical("copyTensor", values, snapshot.toFloatArray());
                assertSame("gradient kept", grad, p.grad());
            } finally {
                copy.close();
                p.close();
            }
        }

        /** d(sum(checkpoint(x -> x * x)))/dx through the recomputing backward. */
        static float[] checkpointGradient(ComputeBackend b) {
            try (Tensor base = Tensor.fromArray(b, CHECKPOINT_INPUT);
                 Tensor x = base.withRequiresGrad(true)) {
                Tensor out = CheckpointFunction.checkpoint(t -> TensorOps.mul(t, t), x, true);
                Tensor loss = TensorOps.sum(out);
                loss.backward();
                assertSame("output backend", b, out.backend());
                Tensor grad = x.grad();
                assertSame("gradient backend", b, grad.backend());
                float[] values = grad.toFloatArray();
                loss.close();
                out.close();
                return values;
            }
        }

        /** Grid values, quarter steps off the grid and NaN through an identity QAT module (INT8 symmetric). */
        static float[] fakeQuantize(ComputeBackend b) {
            QATModule qat = QATModule.builder().baseModule(new Identity()).quantizeWeights(false).build();
            try (Tensor x = Tensor.fromArray(b, fakeQuantizeInput());
                 Tensor y = qat.forward(x)) {
                assertSame("output backend", b, y.backend());
                return y.toFloatArray();
            } finally {
                qat.close();
            }
        }

        /** Values whose range is +-127/64, so the scale is exactly 1/64 and every quotient is exact. */
        static float[] fakeQuantizeInput() {
            return new float[]{-127f / 64, 127f / 64, 0.25f / 64, 5.25f / 64, -9.75f / 64, 100.75f / 64, Float.NaN,
                -0.0f, 63.25f / 64};
        }

        static float[] fakeQuantizeExpected() {
            return new float[]{-127f / 64, 127f / 64, 0f, 5f / 64, -10f / 64, 101f / 64, 0f, 0f, 63f / 64};
        }

        /**
         * {@code d(sum y)/dx} through an identity QAT module whose observers were calibrated on
         * {@link #fakeQuantizeInput()} without recording and then frozen: every finite value has an in-range code.
         */
        static float[] fakeQuantizeGradient(ComputeBackend b) {
            QATModule qat = QATModule.builder().baseModule(new Identity()).quantizeWeights(false).build();
            try (Tensor base = Tensor.fromArray(b, fakeQuantizeInput());
                 Tensor x = base.withRequiresGrad(true)) {
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                    qat.forward(base).close();
                }
                qat.freezeObservers();
                int graphBefore = ComputationGraph.current().size();
                Tensor y = qat.forward(x);
                Tensor loss = TensorOps.sum(y);
                loss.backward();
                assertEquals("graph size after the step", graphBefore, ComputationGraph.current().size());
                Tensor grad = x.grad();
                assertSame("gradient backend", b, grad.backend());
                float[] values = grad.toFloatArray();
                x.setGrad(null);
                grad.close();
                loss.close();
                y.close();
                return values;
            } finally {
                qat.close();
            }
        }

        /** The straight-through gradient: 1 for every finite input of {@link #fakeQuantizeInput()}, 0 for NaN. */
        static float[] fakeQuantizeGradientExpected() {
            return new float[]{1f, 1f, 1f, 1f, 1f, 1f, 0f, 1f, 1f};
        }

        /** Calibration on a parameter with a NaN, quantization, and dequantization onto the parameter's backend. */
        static String quantize(ComputeBackend b) {
            Module model = new Linear(b, new float[]{-2.54f, 1.0f, Float.NaN, 0.5f, 2.0f, -1.5f}, 3, 2);
            try {
                Quantizer quantizer = Quantizer.builder().quantType(Quantizer.QuantizationType.INT8_ASYMMETRIC).build();
                Map<String, Quantizer.QuantizedTensor> codes = quantizer.quantizeModel(model);
                Quantizer.QuantizedTensor weight = codes.get("weight");
                assertSame("codes remember the backend", b, weight.backend());
                try (Tensor restored = quantizer.dequantizeTensor(weight)) {
                    assertSame("dequantized backend", b, restored.backend());
                    assertEquals("dequantized shape", Shape.of(3, 2), restored.shape());
                }
                return quantizer.getQuantParams("weight").toString() + Arrays.toString(weight.data());
            } finally {
                model.close();
            }
        }

        /** Rows {@code 2 * x + 1} of a 5-row batch through two stages; stage 0 on {@code first}, stage 1 on {@code second}. */
        static float[] pipeline(ComputeBackend first, ComputeBackend second, int microBatches) {
            PipelineParallel pp = new PipelineParallel(Arrays.asList(new Affine(2f, 0f), new Affine(1f, 1f)),
                    Arrays.asList(first, second), microBatches);
            try (Tensor input = Tensor.fromArray(first, new float[][]{{0f, 1f}, {2f, 3f}, {4f, 5f}, {6f, 7f}, {8f, 9f}});
                 Tensor out = pp.forward(input)) {
                assertSame("output backend", second, out.backend());
                assertEquals("output shape", Shape.of(5, 2), out.shape());
                return out.toFloatArray();
            } finally {
                pp.close();
            }
        }

        static final float[][] PIPE_INPUT = {{0f, 1f}, {2f, 3f}, {4f, 5f}, {6f, 7f}, {8f, 9f}};
        static final float[] PIPE_W0 = {1f, -1f, 2f, 1f};
        static final float[] PIPE_W1 = {3f, -2f};
        static final float[] PIPE_GRAD = {1f, -2f, 3f, 0f, 2f};

        /**
         * {@code y = x W0 W1} through two recording stages (stage 0 on {@code first}, stage 1 on {@code second}), then
         * {@link PipelineParallel#backward} with a non-uniform gradient: returns {@code dW0, dW1, dx} and {@code y}.
         * The caller's graph is back at its size before the step.
         */
        static float[] pipelineBackward(ComputeBackend first, ComputeBackend second, int microBatches) {
            Linear stage0 = new Linear(first, PIPE_W0, 2, 2);
            Linear stage1 = new Linear(second, PIPE_W1, 2, 1);
            PipelineParallel pp = new PipelineParallel(Arrays.asList(stage0, stage1), Arrays.asList(first, second),
                    microBatches);
            int graphBefore = ComputationGraph.current().size();
            try (Tensor base = Tensor.fromArray(first, PIPE_INPUT);
                 Tensor x = base.withRequiresGrad(true);
                 Tensor g = Tensor.fromArray(second, new float[][]{{PIPE_GRAD[0]}, {PIPE_GRAD[1]}, {PIPE_GRAD[2]},
                     {PIPE_GRAD[3]}, {PIPE_GRAD[4]}});
                 Tensor y = pp.forward(x);
                 Tensor dx = pp.backward(g)) {
                assertTrue("recorded result requires grad", y.requiresGrad());
                assertSame("input gradient backend", first, dx.backend());
                assertSame("stage 0 gradient backend", first, stage0.weight.grad().backend());
                assertSame("stage 1 gradient backend", second, stage1.weight.grad().backend());
                assertEquals("graph size after the step", graphBefore, ComputationGraph.current().size());
                float[] out = new float[0];
                out = append(out, stage0.weight.grad().toFloatArray());
                out = append(out, stage1.weight.grad().toFloatArray());
                out = append(out, dx.toFloatArray());
                return append(out, y.toFloatArray());
            } finally {
                pp.close();
                stage0.close();
                stage1.close();
            }
        }

        /** {@code dW0 = x^T (g W1^T)}, {@code dW1 = (x W0)^T g}, {@code dx = g W1^T W0^T}, {@code y = x W0 W1}. */
        static float[] pipelineBackwardExpected() {
            int rows = PIPE_INPUT.length;
            float[][] h = new float[rows][2];
            float[] y = new float[rows];
            float[][] gh = new float[rows][2];
            float[] dx = new float[rows * 2];
            for (int r = 0; r < rows; r++) {
                for (int j = 0; j < 2; j++) {
                    h[r][j] = PIPE_INPUT[r][0] * PIPE_W0[j] + PIPE_INPUT[r][1] * PIPE_W0[2 + j];
                    gh[r][j] = PIPE_GRAD[r] * PIPE_W1[j];
                }
                y[r] = h[r][0] * PIPE_W1[0] + h[r][1] * PIPE_W1[1];
                for (int i = 0; i < 2; i++) {
                    dx[r * 2 + i] = gh[r][0] * PIPE_W0[i * 2] + gh[r][1] * PIPE_W0[i * 2 + 1];
                }
            }
            float[] dW0 = new float[4];
            float[] dW1 = new float[2];
            for (int r = 0; r < rows; r++) {
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        dW0[i * 2 + j] += PIPE_INPUT[r][i] * gh[r][j];
                    }
                    dW1[i] += h[r][i] * PIPE_GRAD[r];
                }
            }
            return append(append(append(dW0, dW1), dx), y);
        }

        private static float[] append(float[] a, float[] b) {
            float[] out = Arrays.copyOf(a, a.length + b.length);
            System.arraycopy(b, 0, out, a.length, b.length);
            return out;
        }

        /**
         * A failing middle stage: forward rethrows the stage's exception, every stage stops, and the pipeline then
         * runs a healthy batch.
         */
        static void pipelineFailure(ComputeBackend b) {
            FailOnce failing = new FailOnce();
            PipelineParallel pp = new PipelineParallel(Arrays.asList(new Affine(2f, 0f), failing, new Affine(1f, 1f)),
                    Arrays.asList(b, b, b), 3);
            try (Tensor input = Tensor.fromArray(b, new float[][]{{0f, 1f}, {2f, 3f}, {4f, 5f}, {6f, 7f}, {8f, 9f}})) {
                try {
                    pp.forward(input).close();
                    fail("the stage failure was not rethrown");
                } catch (IllegalStateException e) {
                    assertEquals(FailOnce.MESSAGE, e.getMessage());
                }
                try (Tensor out = pp.forward(input)) {
                    TrainParity.assertBitIdentical("pipeline after a failure", pipelineExpected(), out.toFloatArray());
                }
            } finally {
                pp.close();
            }
        }

        static float[] pipelineExpected() {
            return new float[]{1f, 3f, 5f, 7f, 9f, 11f, 13f, 15f, 17f, 19f};
        }

        /** Final weights and losses of two first-order MAML iterations over two regression tasks. */
        static float[] maml(ComputeBackend b) {
            Linear model = new Linear(b, initialWeights(3, 1, 11), 3, 1);
            List<MAML.Task> tasks = Arrays.asList(task(b, 1), task(b, 2));
            try (AutogradMse loss = new AutogradMse();
                 MAML maml = new MAML(model, 0.05f, 0.1f, 2, loss)) {
                float[] before = model.weights();
                Map<String, Tensor> adapted = maml.adapt(tasks.get(0));
                try {
                    assertSame("adapted backend", b, adapted.get("weight").backend());
                    TrainParity.assertBitIdentical("adapt leaves the weights", before, model.weights());
                } finally {
                    for (Tensor t : adapted.values()) {
                        t.close();
                    }
                }
                int graphBefore = ComputationGraph.current().size();
                MAML.MAMLResult result = maml.metaTrain(tasks, 2);
                assertEquals("graph size after meta-training", graphBefore, ComputationGraph.current().size());
                return concat(model.weights(), result.metaLosses(), result.validationLosses());
            } finally {
                closeTasks(tasks);
                model.close();
            }
        }

        /** Final weights and losses of two second-order MAML iterations over two regression tasks, two inner steps. */
        static float[] secondOrderMaml(ComputeBackend b) {
            Linear model = new Linear(b, initialWeights(3, 1, 11), 3, 1);
            List<MAML.Task> tasks = Arrays.asList(task(b, 1), task(b, 2));
            try (AutogradMse loss = new AutogradMse();
                 MAML maml = new MAML(model, 0.05f, 0.1f, 2, loss, false)) {
                int graphBefore = ComputationGraph.current().size();
                MAML.MAMLResult result = maml.metaTrain(tasks, 2);
                assertEquals("graph size after meta-training", graphBefore, ComputationGraph.current().size());
                assertSame("weights stay on the backend", b, model.weight.tensor().backend());
                assertEquals("gradients released", null, model.weight.grad());
                return concat(model.weights(), result.metaLosses(), result.validationLosses());
            } finally {
                closeTasks(tasks);
                model.close();
            }
        }

        static final float SECOND_ORDER_ALPHA = 0.25f;
        static final float SECOND_ORDER_BETA = 0.5f;

        /** Support input, support target, query input, query target and initial weights of the closed-form task. */
        private static float[][] secondOrderData() {
            return new float[][]{TrainParity.uniform(41, 3, -1f, 1f), TrainParity.uniform(42, 1, -1f, 1f),
                TrainParity.uniform(43, 3, -1f, 1f), TrainParity.uniform(44, 1, -1f, 1f), initialWeights(3, 1, 11)};
        }

        /** Weights, meta loss and validation loss after one second-order iteration on the closed-form task. */
        static float[] secondOrderMamlSingleStep(ComputeBackend b) {
            float[][] d = secondOrderData();
            Linear model = new Linear(b, d[4], 3, 1);
            MAML.Task task = new MAML.Task(Collections.singletonList(Tensor.fromArray(b, new float[][]{d[0]})),
                    Collections.singletonList(Tensor.fromArray(b, new float[][]{d[1]})),
                    Collections.singletonList(Tensor.fromArray(b, new float[][]{d[2]})),
                    Collections.singletonList(Tensor.fromArray(b, new float[][]{d[3]})));
            try (AutogradMse loss = new AutogradMse();
                 MAML maml = new MAML(model, SECOND_ORDER_ALPHA, SECOND_ORDER_BETA, 1, loss, false)) {
                int graphBefore = ComputationGraph.current().size();
                MAML.MAMLResult result = maml.metaTrain(Collections.singletonList(task), 1);
                assertEquals("graph size after meta-training", graphBefore, ComputationGraph.current().size());
                return concat(model.weights(), result.metaLosses(), result.validationLosses());
            } finally {
                closeTasks(Collections.singletonList(task));
                model.close();
            }
        }

        /**
         * The closed form of {@link #secondOrderMamlSingleStep} in double precision: {@code W' = W - 2 alpha rs xs},
         * {@code h = 2 rq xq} with {@code rq = xq W' - yq}, meta-gradient {@code h - 2 alpha xs (xs . h)} (or
         * {@code h} with {@code firstOrder}), {@code W - beta * gradient}, then the query loss {@code rq^2} and the
         * validation loss at the new weights.
         */
        static double[] secondOrderMamlSingleStepExpected(boolean firstOrder) {
            float[][] d = secondOrderData();
            double alpha = SECOND_ORDER_ALPHA;
            double beta = SECOND_ORDER_BETA;
            double rs = dot(d[0], d[4]) - d[1][0];
            double[] adapted = new double[3];
            for (int i = 0; i < 3; i++) {
                adapted[i] = d[4][i] - alpha * 2 * rs * d[0][i];
            }
            double rq = dot(d[2], adapted) - d[3][0];
            double[] h = new double[3];
            for (int i = 0; i < 3; i++) {
                h[i] = 2 * rq * d[2][i];
            }
            double xsh = dot(d[0], h);
            double[] out = new double[5];
            double rv = -d[3][0];
            for (int i = 0; i < 3; i++) {
                double grad = firstOrder ? h[i] : h[i] - 2 * alpha * d[0][i] * xsh;
                out[i] = d[4][i] - beta * grad;
                rv += d[2][i] * out[i];
            }
            out[3] = rq * rq;
            out[4] = rv * rv;
            return out;
        }

        private static double dot(float[] a, float[] w) {
            double sum = 0;
            for (int i = 0; i < a.length; i++) {
                sum += (double) a[i] * w[i];
            }
            return sum;
        }

        private static double dot(float[] a, double[] w) {
            double sum = 0;
            for (int i = 0; i < a.length; i++) {
                sum += a[i] * w[i];
            }
            return sum;
        }

        /** Final weights and round losses of two federated meta-learning rounds over two clients. */
        static float[] federatedMeta(ComputeBackend b) {
            Linear model = new Linear(b, initialWeights(3, 1, 11), 3, 1);
            List<MAML.Task> tasks = Arrays.asList(task(b, 3), task(b, 4));
            try (AutogradMse loss = new AutogradMse();
                 FederatedMetaLearner learner = new FederatedMetaLearner(model, loss, 2, 2, 1, 0.05f, 0.1f, 1.0f)) {
                List<List<MAML.Task>> clients = Arrays.asList(Collections.singletonList(tasks.get(0)),
                        Collections.singletonList(tasks.get(1)));
                int graphBefore = ComputationGraph.current().size();
                FederatedMetaLearner.FederatedMetaResult result = learner.train(clients);
                assertEquals("graph size after training", graphBefore, ComputationGraph.current().size());
                return concat(model.weights(), result.roundLosses(), result.roundAccuracies());
            } finally {
                closeTasks(tasks);
                model.close();
            }
        }

        /** Final weights and evaluation losses of two FedProx rounds over two clients with SGD. */
        static float[] federated(ComputeBackend b) {
            Linear model = new Linear(b, initialWeights(3, 1, 11), 3, 1);
            List<Tensor> data = Arrays.asList(matrix(b, 5, 4, 3), matrix(b, 6, 4, 3));
            List<Tensor> labels = Arrays.asList(matrix(b, 7, 4, 1), matrix(b, 8, 4, 1));
            try (FederatedTrainer trainer = new FederatedTrainer(model, new SGD(model.parameters(), 0.1f), 2, 2, 2,
                    0.05f, FederatedTrainer.FederatedStrategy.FED_PROX, 0.1f)) {
                int graphBefore = ComputationGraph.current().size();
                FederatedTrainer.FederatedResult result = trainer.train(data, labels);
                assertEquals("graph size after training", graphBefore, ComputationGraph.current().size());
                return concat(model.weights(), result.roundLosses, Collections.emptyList());
            } finally {
                closeAll(data);
                closeAll(labels);
                model.close();
            }
        }

        /** Final student weights and losses of three distillation steps at temperature 2. */
        static float[] distill(ComputeBackend b) {
            Linear teacher = new Linear(b, initialWeights(3, 2, 31), 3, 2);
            Linear student = new Linear(b, initialWeights(3, 2, 21), 3, 2);
            try (Tensor input = matrix(b, 9, 2, 3);
                 Tensor target = matrix(b, 10, 2, 2);
                 AutogradMse loss = new AutogradMse();
                 KnowledgeDistiller distiller = new KnowledgeDistiller(teacher, student, 2.0f, 0.5f, loss)) {
                assertFalse("the teacher is frozen", teacher.weight.requiresGrad());
                assertFalse("the teacher is in evaluation mode", teacher.isTraining());
                int graphBefore = ComputationGraph.current().size();
                KnowledgeDistiller.DistillationResult result = distiller.train(input, target, 0.1f, 3);
                assertEquals("graph size after training", graphBefore, ComputationGraph.current().size());
                TrainParity.assertBitIdentical("the teacher is unchanged", initialWeights(3, 2, 31), teacher.weights());
                assertEquals("the teacher has no gradient", null, teacher.weight.grad());
                return concat(student.weights(), result.totalLosses(), result.distillationLosses());
            } finally {
                teacher.close();
                student.close();
            }
        }

        /**
         * Student weights, then the distillation losses, of three steps at temperature 2 with {@code alpha = 1}, so
         * only the divergence term drives the student.
         */
        static float[] distillOnly(ComputeBackend b) {
            Linear teacher = new Linear(b, initialWeights(3, 2, 31), 3, 2);
            Linear student = new Linear(b, initialWeights(3, 2, 21), 3, 2);
            try (Tensor input = matrix(b, 9, 2, 3);
                 Tensor target = matrix(b, 10, 2, 2);
                 AutogradMse loss = new AutogradMse();
                 KnowledgeDistiller distiller = new KnowledgeDistiller(teacher, student, 2.0f, 1.0f, loss)) {
                KnowledgeDistiller.DistillationResult result = distiller.train(input, target, 0.1f, 3);
                return concat(student.weights(), result.distillationLosses(), Collections.emptyList());
            } finally {
                teacher.close();
                student.close();
            }
        }

        static float[] initialWeights(int rows, int cols, long seed) {
            return TrainParity.uniform(seed, rows * cols, -0.5f, 0.5f);
        }

        static void assertClose(String what, float[] ref, float[] got) {
            assertEquals(what + ": length", ref.length, got.length);
            for (int i = 0; i < ref.length; i++) {
                if (Float.isNaN(ref[i]) || Float.isNaN(got[i])) {
                    fail(what + ": element " + i + " is NaN: ref " + ref[i] + ", got " + got[i]);
                }
                if (Math.abs(got[i] - ref[i]) > TRAINING_TOLERANCE * Math.max(1f, Math.abs(ref[i]))) {
                    fail(what + ": element " + i + ": ref " + ref[i] + ", got " + got[i]);
                }
            }
        }

        private static MAML.Task task(ComputeBackend b, long seed) {
            List<Tensor> supportX = Arrays.asList(matrix(b, seed * 10, 1, 3), matrix(b, seed * 10 + 1, 1, 3));
            List<Tensor> supportY = Arrays.asList(matrix(b, seed * 10 + 2, 1, 1), matrix(b, seed * 10 + 3, 1, 1));
            List<Tensor> queryX = Arrays.asList(matrix(b, seed * 10 + 4, 1, 3), matrix(b, seed * 10 + 5, 1, 3));
            List<Tensor> queryY = Arrays.asList(matrix(b, seed * 10 + 6, 1, 1), matrix(b, seed * 10 + 7, 1, 1));
            return new MAML.Task(supportX, supportY, queryX, queryY);
        }

        private static Tensor matrix(ComputeBackend b, long seed, int rows, int cols) {
            float[] values = TrainParity.uniform(seed, rows * cols, -1f, 1f);
            float[][] m = new float[rows][cols];
            for (int r = 0; r < rows; r++) {
                System.arraycopy(values, r * cols, m[r], 0, cols);
            }
            return Tensor.fromArray(b, m);
        }

        private static float[] concat(float[] weights, List<Float> first, List<Float> second) {
            float[] out = Arrays.copyOf(weights, weights.length + first.size() + second.size());
            int k = weights.length;
            for (Float v : first) {
                out[k++] = v;
            }
            for (Float v : second) {
                out[k++] = v;
            }
            return out;
        }

        private static void closeTasks(List<MAML.Task> tasks) {
            for (MAML.Task task : tasks) {
                closeAll(task.supportInputs());
                closeAll(task.supportTargets());
                closeAll(task.queryInputs());
                closeAll(task.queryTargets());
            }
        }

        private static void closeAll(List<Tensor> tensors) {
            for (Tensor t : tensors) {
                t.close();
            }
        }
    }

    /** {@code x @ W} recorded by TensorOps, with one parameter named {@code weight} of shape {@code [in, out]}. */
    static final class Linear extends Module {
        final Parameter weight;

        Linear(ComputeBackend b, float[] values, int in, int out) {
            float[][] m = new float[in][out];
            for (int r = 0; r < in; r++) {
                System.arraycopy(values, r * out, m[r], 0, out);
            }
            weight = registerParameter("weight", new Parameter(Tensor.fromArray(b, m), "weight"));
        }

        float[] weights() {
            return weight.tensor().toFloatArray();
        }

        @Override
        public Tensor forward(Tensor input) {
            return TensorOps.matmul(input, weight.tensor());
        }
    }

    /** {@code scale * x + shift} with tensor ops (exact for the test values). */
    static final class Affine extends Module {
        private final float scale;
        private final float shift;

        Affine(float scale, float shift) {
            this.scale = scale;
            this.shift = shift;
        }

        @Override
        public Tensor forward(Tensor input) {
            try (Tensor scaled = TensorOps.mul(input, scale)) {
                return TensorOps.add(scaled, shift);
            }
        }
    }

    /** The identity on every call except the second, which throws {@link #MESSAGE}. */
    static final class FailOnce extends Module {
        static final String MESSAGE = "stage failed on purpose";
        private int calls;

        @Override
        public Tensor forward(Tensor input) {
            if (++calls == 2) {
                throw new IllegalStateException(MESSAGE);
            }
            return input;
        }
    }

    static final class Identity extends Module {
        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    /**
     * Mean squared error recorded by TensorOps, so gradients reach the model's parameters. The intermediate
     * tensors stay open for backward and are closed with this loss.
     */
    static final class AutogradMse implements LossFunction, AutoCloseable {
        private final List<Tensor> intermediates = new ArrayList<>();

        @Override
        public Tensor compute(Tensor prediction, Tensor target) {
            Tensor diff = TensorOps.sub(prediction, target);
            Tensor squared = TensorOps.mul(diff, diff);
            intermediates.add(diff);
            intermediates.add(squared);
            return TensorOps.mean(squared);
        }

        @Override
        public String name() {
            return "AutogradMse";
        }

        @Override
        public void close() {
            for (Tensor t : intermediates) {
                t.close();
            }
            intermediates.clear();
        }
    }
}
