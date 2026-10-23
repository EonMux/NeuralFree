package com.eonmux.neuralfree.nn.norm;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
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
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The nn/norm modules on each GPU API and on their CPU path, both against double-precision references of the module
 * formulas in the reduction class {@code 1e-5 * sqrt(K) * sum|terms| + K * FLT_MIN} (the absolute sizes of the
 * terms that form each output as the term sum). Holding both paths to one reference bounds their difference by twice that tolerance. The running
 * statistics of the batch norms are checked after a training step on both paths, and evaluation mode is checked
 * against the statistics each module then holds. Every forward runs in grad mode with trainable parameters, so its
 * output must be a recorded node; each step is scoped so the graph is left as it was. The backward of {@code sum(y)}
 * must reach the input, the weight and the bias (whose gradient is exactly the element count per position or
 * channel) and must leave the batch norms' running statistics and batch counters as the forward left them. Each
 * training forward of a batch norm counts one batch in its persistent {@code num_batches_tracked} buffer and an
 * evaluation forward counts none; with a null momentum the running statistics are the cumulative average of every
 * training batch. Parameter and buffer setup writes run under no-grad.
 */
@RunWith(Enclosed.class)
public class NormModulesParityTest {

    static final float EPS = 1e-5f;
    static final float MOMENTUM = 0.1f;
    static final int C_FEATURES = 4;

    /** The CPU path, on a backend without the train library. */
    public static class HostPath {
        private ComputeBackend host;

        @Before
        public void setUp() {
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            TrainGpu.release(host);
        }

        @Test
        public void batchNorm1dWritesRunningStatsBack() {
            Scenarios.expectBatchNorm1d("cpu path rank 3", Scenarios.batchNorm1d(host, 5), 5);
            Scenarios.expectBatchNorm1d("cpu path rank 2", Scenarios.batchNorm1d(host, 1), 1);
        }

        @Test
        public void batchNorm2dWritesRunningStatsBack() {
            Scenarios.expectBatchNorm2d("cpu path", Scenarios.batchNorm2d(host));
        }

        @Test
        public void nullMomentumKeepsTheCumulativeAverage() {
            Scenarios.expectCumulativeBatchNorm("cpu path", Scenarios.cumulativeBatchNorm(host));
        }

        @Test
        public void momentumIsFixedOrCumulativeAndValidated() {
            try (BatchNorm1d fixed = new BatchNorm1d(C_FEATURES, EPS, MOMENTUM, true);
                 BatchNorm2d cumulative = new BatchNorm2d(C_FEATURES, EPS, (Float) null, true)) {
                assertFalse(fixed.cumulativeAverage());
                assertEquals(MOMENTUM, fixed.momentum(), 0f);
                assertTrue(cumulative.cumulativeAverage());
                assertThrows(IllegalStateException.class, cumulative::momentum);
            }
            assertThrows(IllegalArgumentException.class, () -> new BatchNorm1d(C_FEATURES, EPS, Float.NaN, true));
            assertThrows(IllegalArgumentException.class, () -> new BatchNorm2d(C_FEATURES, EPS, 1.5f, true));
            assertThrows(IllegalArgumentException.class,
                    () -> new BatchNorm2d(C_FEATURES, EPS, Float.valueOf(-0.5f), true));
            assertThrows(IllegalArgumentException.class, () -> new BatchNorm1d(C_FEATURES, Float.NaN, MOMENTUM, true));
        }

        @Test
        public void groupNormNormalizesEverySpatialElement() {
            Scenarios.expectGroupNorm("cpu path", Scenarios.groupNorm(host));
        }

        @Test
        public void layerNormMatchesReference() {
            Scenarios.expectLayerNorm("cpu path", Scenarios.layerNorm(host, host));
        }

        @Test
        public void rmsNormMatchesReference() {
            Scenarios.expectRmsNorm("cpu path", Scenarios.rmsNorm(host));
        }

        @Test
        public void backwardReachesInputAndParameters() {
            Scenarios.expectBackward("cpu path", host);
        }
    }

    /** The device path on each GPU API, against the same references as the CPU path. */
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
            session = TrainParity.open(api);
            gpu = session.backend();
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            try {
                TrainGpu.release(host);
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }

        @Test
        public void batchNorm1dMatchesCpuPath() {
            Scenarios.expectBatchNorm1d("cpu path", Scenarios.batchNorm1d(host, 7), 7);
            Scenarios.expectBatchNorm1d(api + " rank 3", Scenarios.batchNorm1d(gpu, 7), 7);
            Scenarios.expectBatchNorm1d(api + " rank 2", Scenarios.batchNorm1d(gpu, 1), 1);
        }

        @Test
        public void batchNorm2dMatchesCpuPath() {
            Scenarios.expectBatchNorm2d("cpu path", Scenarios.batchNorm2d(host));
            Scenarios.expectBatchNorm2d(api.toString(), Scenarios.batchNorm2d(gpu));
        }

        @Test
        public void nullMomentumMatchesCpuPath() {
            Scenarios.expectCumulativeBatchNorm("cpu path", Scenarios.cumulativeBatchNorm(host));
            Scenarios.expectCumulativeBatchNorm(api.toString(), Scenarios.cumulativeBatchNorm(gpu));
        }

        @Test
        public void batchNorm2dKeepsRunningStatsOnTheDevice() {
            BatchNorm2d m = new BatchNorm2d(2);
            m.initialize(gpu);
            ComputationGraph graph = ComputationGraph.current();
            int start = graph.size();
            try (Tensor x = Tensor.fromArray(gpu, TrainParity.uniform(201, 2 * 2 * 9, -1f, 1f));
                 Tensor in = x.reshape(Shape.of(2, 2, 3, 3));
                 ComputationGraph.StepScope step = graph.step();
                 Tensor out = m.forward(in)) {
                assertSame("output backend", gpu, out.backend());
                assertSame("running mean backend", gpu, m.runningMean().backend());
                assertSame("running var backend", gpu, m.runningVar().backend());
                assertSame("batch counter backend", gpu, m.namedBuffers().get("num_batches_tracked").backend());
                assertEquals("batches trained", 1, m.numBatchesTrained());
            } finally {
                m.close();
            }
            assertEquals("graph size after the step", start, graph.size());
        }

        @Test
        public void groupNormMatchesCpuPath() {
            Scenarios.expectGroupNorm("cpu path", Scenarios.groupNorm(host));
            Scenarios.expectGroupNorm(api.toString(), Scenarios.groupNorm(gpu));
        }

        @Test
        public void layerNormMatchesCpuPath() {
            Scenarios.expectLayerNorm("cpu path", Scenarios.layerNorm(host, host));
            Scenarios.expectLayerNorm(api.toString(), Scenarios.layerNorm(gpu, gpu));
        }

        @Test
        public void rmsNormMatchesCpuPath() {
            Scenarios.expectRmsNorm("cpu path", Scenarios.rmsNorm(host));
            Scenarios.expectRmsNorm(api.toString(), Scenarios.rmsNorm(gpu));
        }

        @Test
        public void parametersOnAnotherBackendTakeTheCpuPath() {
            Scenarios.expectLayerNorm(api + " input, host parameters", Scenarios.layerNorm(host, gpu));
        }

        @Test
        public void backwardReachesInputAndParameters() {
            Scenarios.expectBackward(api.toString(), gpu);
        }
    }

    // ------------------------------------------------------------------ scenarios

    /** Module runs on one backend, returning the host copies of their results, and the reference checks. */
    static final class Scenarios {
        private static final int B = 3;
        private static final int C = C_FEATURES;

        private Scenarios() {
        }

        /** BatchNorm1d on {@code [B, C, l]} ({@code l == 1} uses a rank-2 input): train step, then eval. */
        static float[][] batchNorm1d(ComputeBackend be, int l) {
            BatchNorm1d m = new BatchNorm1d(C, EPS, MOMENTUM, true);
            m.initialize(be);
            try {
                setBatchNormState(m.weight(), m.bias(), m.runningMean(), m.runningVar());
                int[] dims = l == 1 ? new int[]{B, C} : new int[]{B, C, l};
                float[] x = TrainParity.uniform(10 + l, B * C * l, -2f, 3f);
                float[] train = forward(m, be, x, dims);
                float[] mean = m.runningMean().toFloatArray();
                float[] var = m.runningVar().toFloatArray();
                assertEquals("batches trained", 1, m.numBatchesTrained());
                Map<String, Object> state = m.stateDict();
                assertEquals("state dict keys", Arrays.asList("weight", "bias", "running_mean", "running_var",
                        "num_batches_tracked"), new ArrayList<>(state.keySet()));
                assertArrayEquals("saved batch count", new float[]{1f}, (float[]) state.get("num_batches_tracked"),
                        0f);
                m.eval();
                float[] eval = forward(m, be, x, dims);
                assertEquals("eval does not count a batch", 1, m.numBatchesTrained());
                return new float[][]{x, train, mean, var, eval};
            } finally {
                m.close();
            }
        }

        static void expectBatchNorm1d(String what, float[][] r, int l) {
            expectBatchNorm(what + " BatchNorm1d", r, l);
        }

        /** BatchNorm2d on {@code [B, C, 5, 6]}: train step, then eval. */
        static float[][] batchNorm2d(ComputeBackend be) {
            BatchNorm2d m = new BatchNorm2d(C, EPS, MOMENTUM, true);
            m.initialize(be);
            try {
                setBatchNormState(m.weight(), m.bias(), m.runningMean(), m.runningVar());
                int[] dims = {B, C, 5, 6};
                float[] x = TrainParity.uniform(20, B * C * 30, -2f, 3f);
                float[] train = forward(m, be, x, dims);
                float[] mean = m.runningMean().toFloatArray();
                float[] var = m.runningVar().toFloatArray();
                assertEquals("batches trained", 1, m.numBatchesTrained());
                m.eval();
                float[] eval = forward(m, be, x, dims);
                assertEquals("eval does not count a batch", 1, m.numBatchesTrained());
                return new float[][]{x, train, mean, var, eval};
            } finally {
                m.close();
            }
        }

        static void expectBatchNorm2d(String what, float[][] r) {
            expectBatchNorm(what + " BatchNorm2d", r, 30);
        }

        /**
         * BatchNorm2d with a null momentum on {@code [B, C, 5, 6]}: two training steps on different inputs, then the
         * counter is reloaded through the state dict. Returns {x1, x2, mean and var after the first step, mean and
         * var after the second}.
         */
        static float[][] cumulativeBatchNorm(ComputeBackend be) {
            BatchNorm2d m = new BatchNorm2d(C, EPS, (Float) null, true);
            m.initialize(be);
            try {
                setBatchNormState(m.weight(), m.bias(), m.runningMean(), m.runningVar());
                int[] dims = {B, C, 5, 6};
                float[] x1 = TrainParity.uniform(21, B * C * 30, -2f, 3f);
                float[] x2 = TrainParity.uniform(22, B * C * 30, -1f, 4f);
                forward(m, be, x1, dims);
                assertEquals("batches trained after one step", 1, m.numBatchesTrained());
                float[] mean1 = m.runningMean().toFloatArray();
                float[] var1 = m.runningVar().toFloatArray();
                forward(m, be, x2, dims);
                assertEquals("batches trained after two steps", 2, m.numBatchesTrained());
                float[] mean2 = m.runningMean().toFloatArray();
                float[] var2 = m.runningVar().toFloatArray();

                Map<String, Object> state = new LinkedHashMap<>();
                state.put("num_batches_tracked", new float[]{7f});
                m.loadStateDict(state);
                assertEquals("loaded batch count", 7, m.numBatchesTrained());
                assertSame("batch counter backend", be, m.namedBuffers().get("num_batches_tracked").backend());
                return new float[][]{x1, x2, mean1, var1, mean2, var2};
            } finally {
                m.close();
            }
        }

        /**
         * The first step's factor {@code 1 / 1} replaces the running statistics by the batch statistics; the second
         * step's factor {@code 1 / 2} averages them with the second batch's.
         */
        static void expectCumulativeBatchNorm(String what, float[][] r) {
            String name = what + " cumulative BatchNorm2d";
            int s = 30;
            Stats first = Stats.of(r[0], B, C, s, 0);
            Stats second = Stats.of(r[1], B, C, s, 0);
            double bessel = unbiasedScale(B * s);
            for (int k = 0; k < C; k++) {
                TrainParity.assertReduction(name + " running mean after one step " + k, first.mean[k], r[2][k],
                        B * s, first.meanTerms[k]);
                TrainParity.assertReduction(name + " running var after one step " + k, first.var[k] * bessel,
                        r[3][k], B * s, first.varTerms[k] * bessel);
                TrainParity.assertReduction(name + " running mean after two steps " + k,
                        0.5 * r[2][k] + 0.5 * second.mean[k], r[4][k], B * s,
                        0.5 * Math.abs(r[2][k]) + 0.5 * second.meanTerms[k]);
                TrainParity.assertReduction(name + " running var after two steps " + k,
                        0.5 * r[3][k] + 0.5 * second.var[k] * bessel, r[5][k], B * s,
                        0.5 * Math.abs(r[3][k]) + 0.5 * second.varTerms[k] * bessel);
            }
        }

        /**
         * GroupNorm(3 groups, 6 channels) on a rank-4 view {@code [2, 6, 3, 4]} that starts one batch into its
         * buffer, so the device path stages an offset input.
         */
        static float[][] groupNorm(ComputeBackend be) {
            int b = 2;
            int c = 6;
            int s = 12;
            GroupNorm m = new GroupNorm(3, c, EPS, true);
            m.initialize(be);
            float[] all = TrainParity.uniform(30, (b + 1) * c * s, -3f, 3f);
            float[] x = new float[b * c * s];
            System.arraycopy(all, c * s, x, 0, x.length);
            try (Tensor flat = Tensor.fromArray(be, all);
                 Tensor shaped = flat.reshape(Shape.of(b + 1, c, 3, 4));
                 Tensor view = shaped.slice(0, 1, b + 1)) {
                setAffine(m.weight(), gamma(c), m.bias(), beta(c));
                return new float[][]{x, output(m, view)};
            } finally {
                m.close();
            }
        }

        static void expectGroupNorm(String what, float[][] r) {
            Stats st = Stats.of(r[0], 2, 6, 12, 3);
            assertChannels(what + " GroupNorm", r[1], r[0], 2, 6, 12, st, gamma(6), beta(6));
        }

        /**
         * LayerNorm over {@code [4, 5]} of an input {@code [3, 4, 5]}, with parameters on {@code paramBackend} and
         * the input on {@code inputBackend}.
         */
        static float[][] layerNorm(ComputeBackend paramBackend, ComputeBackend inputBackend) {
            LayerNorm m = new LayerNorm(new int[]{4, 5}, EPS, true);
            m.initialize(paramBackend);
            try {
                setAffine(m.weight(), gamma(20), m.bias(), beta(20));
                float[] x = TrainParity.uniform(40, 3 * 20, -3f, 5f);
                return new float[][]{x, forward(m, inputBackend, x, new int[]{3, 4, 5})};
            } finally {
                m.close();
            }
        }

        static void expectLayerNorm(String what, float[][] r) {
            assertRows(what + " LayerNorm", r[1], r[0], 3, 20, gamma(20), beta(20), false, EPS);
        }

        /** RMSNorm over the last dim 40 of an input {@code [3, 3, 40]}. */
        static float[][] rmsNorm(ComputeBackend be) {
            float eps = 1e-8f;
            RMSNorm m = new RMSNorm(40, eps, true);
            m.initialize(be);
            try {
                setAffine(m.weight(), gamma(40), null, null);
                float[] x = TrainParity.uniform(50, 9 * 40, -4f, 4f);
                return new float[][]{x, forward(m, be, x, new int[]{3, 3, 40})};
            } finally {
                m.close();
            }
        }

        static void expectRmsNorm(String what, float[][] r) {
            assertRows(what + " RMSNorm", r[1], r[0], 9, 40, gamma(40), null, true, 1e-8f);
        }

        /** {x, train output, running mean, running var, eval output} of a batch norm on {@code [B, C, s]}. */
        private static void expectBatchNorm(String what, float[][] r, int s) {
            float[] x = r[0];
            Stats st = Stats.of(x, B, C, s, 0);
            assertChannels(what + " train", r[1], x, B, C, s, st, gamma(C), beta(C));
            float[] runMean = runMean();
            float[] runVar = runVar();
            double bessel = unbiasedScale(B * s);
            for (int k = 0; k < C; k++) {
                double expMean = (1 - MOMENTUM) * (double) runMean[k] + MOMENTUM * st.mean[k];
                double expVar = (1 - MOMENTUM) * (double) runVar[k] + MOMENTUM * st.var[k] * bessel;
                TrainParity.assertReduction(what + " running mean " + k, expMean, r[2][k], B * s,
                        (1 - MOMENTUM) * Math.abs(runMean[k]) + MOMENTUM * st.meanTerms[k]);
                TrainParity.assertReduction(what + " running var " + k, expVar, r[3][k], B * s,
                        (1 - MOMENTUM) * Math.abs(runVar[k]) + MOMENTUM * st.varTerms[k] * bessel);
            }
            assertChannels(what + " eval", r[4], x, B, C, s, Stats.given(r[2], r[3]), gamma(C), beta(C));
        }

        /**
         * {@code n / (n - 1)}: the running variance blends in the unbiased batch variance, PyTorch's {@code _BatchNorm}
         * update, while the batch is normalized with the biased one.
         */
        private static double unbiasedScale(int n) {
            return (double) n / (n - 1);
        }

        /** Writes the affine parameters and the running statistics; parameter writes need no-grad. */
        private static void setBatchNormState(Parameter w, Parameter b, Tensor mean, Tensor var) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                GpuOptimizerOps.storeFloats(w.tensor(), gamma(C));
                GpuOptimizerOps.storeFloats(b.tensor(), beta(C));
                GpuOptimizerOps.storeFloats(mean, runMean());
                GpuOptimizerOps.storeFloats(var, runVar());
            }
        }

        /** Writes the scale and, when given, the shift under no-grad, as parameter writes must be. */
        private static void setAffine(Parameter weight, float[] scale, Parameter bias, float[] shift) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                GpuOptimizerOps.storeFloats(weight.tensor(), scale);
                if (bias != null) {
                    GpuOptimizerOps.storeFloats(bias.tensor(), shift);
                }
            }
        }

        private static float[] forward(Module m, ComputeBackend be, float[] x, int[] dims) {
            try (Tensor flat = Tensor.fromArray(be, x);
                 Tensor in = flat.reshape(Shape.of(dims))) {
                return output(m, in);
            }
        }

        /**
         * The module's output for {@code in} as host floats; the output lives on the input's backend and, since the
         * module's parameters require grad, is a recorded node. The step is scoped, so the graph is left as it was.
         */
        private static float[] output(Module m, Tensor in) {
            ComputationGraph graph = ComputationGraph.current();
            int start = graph.size();
            float[] result;
            try (ComputationGraph.StepScope step = graph.step();
                 Tensor out = m.forward(in)) {
                assertSame("output backend", in.backend(), out.backend());
                assertEquals("output shape", in.shape(), out.shape());
                assertTrue("output requires grad", out.requiresGrad());
                assertNotNull("output is a recorded node", out.gradFn());
                result = out.toFloatArray();
            }
            assertEquals("graph size after the step", start, graph.size());
            return result;
        }

        // -------------------------------------------------------------- backward

        /**
         * {@code sum(module(x))} back-propagated through each norm module: LayerNorm over {@code [4, 5]} of
         * {@code [3, 4, 5]}, RMSNorm over 5, GroupNorm(3, 6) on {@code [2, 6, 3, 4]}, BatchNorm1d on {@code [3, 4]}
         * and BatchNorm2d on {@code [3, 4, 5, 6]} in training and in evaluation mode.
         */
        static void expectBackward(String what, ComputeBackend be) {
            LayerNorm ln = new LayerNorm(new int[]{4, 5}, EPS, true);
            ln.initialize(be);
            try {
                setAffine(ln.weight(), gamma(20), ln.bias(), beta(20));
                expectGradients(what + " LayerNorm", ln, be, new int[]{3, 4, 5}, ln.weight(), ln.bias(), 3);
            } finally {
                ln.close();
            }
            RMSNorm rn = new RMSNorm(5, 1e-8f, true);
            rn.initialize(be);
            try {
                setAffine(rn.weight(), gamma(5), null, null);
                expectGradients(what + " RMSNorm", rn, be, new int[]{3, 4, 5}, rn.weight(), null, 0);
            } finally {
                rn.close();
            }
            GroupNorm gn = new GroupNorm(3, 6, EPS, true);
            gn.initialize(be);
            try {
                setAffine(gn.weight(), gamma(6), gn.bias(), beta(6));
                expectGradients(what + " GroupNorm", gn, be, new int[]{2, 6, 3, 4}, gn.weight(), gn.bias(), 24);
            } finally {
                gn.close();
            }
            BatchNorm1d bn1 = new BatchNorm1d(C, EPS, MOMENTUM, true);
            bn1.initialize(be);
            try {
                setBatchNormState(bn1.weight(), bn1.bias(), bn1.runningMean(), bn1.runningVar());
                expectGradients(what + " BatchNorm1d train", bn1, be, new int[]{B, C}, bn1.weight(), bn1.bias(), B);
            } finally {
                bn1.close();
            }
            BatchNorm2d bn2 = new BatchNorm2d(C, EPS, MOMENTUM, true);
            bn2.initialize(be);
            try {
                setBatchNormState(bn2.weight(), bn2.bias(), bn2.runningMean(), bn2.runningVar());
                int[] dims = {B, C, 5, 6};
                expectGradients(what + " BatchNorm2d train", bn2, be, dims, bn2.weight(), bn2.bias(), B * 30);
                bn2.eval();
                expectGradients(what + " BatchNorm2d eval", bn2, be, dims, bn2.weight(), bn2.bias(), B * 30);
            } finally {
                bn2.close();
            }
        }

        /**
         * Runs {@code sum(m(x))} for an input leaf of {@code dims}, back-propagates it and checks: every gradient has
         * its target's shape and backend and is finite; the bias gradient is exactly {@code biasCount} (the number of
         * outputs each bias element adds to, each with upstream gradient 1); and the module's buffers after the
         * backward equal those after the forward bit for bit. Gradients are released afterwards and the graph is
         * left as it was.
         */
        private static void expectGradients(String what, Module m, ComputeBackend be, int[] dims, Parameter weight,
                                            Parameter bias, int biasCount) {
            ComputationGraph graph = ComputationGraph.current();
            int start = graph.size();
            int n = 1;
            for (int d : dims) {
                n *= d;
            }
            try (Tensor flat = Tensor.fromArray(be, TrainParity.uniform(70 + n, n, -2f, 3f));
                 Tensor base = flat.reshape(Shape.of(dims));
                 Tensor in = base.withRequiresGrad(true)) {
                Map<String, float[]> afterForward;
                try (ComputationGraph.StepScope step = graph.step();
                     Tensor out = m.forward(in);
                     Tensor loss = TensorOps.sum(out)) {
                    assertNotNull(what + ": output is a recorded node", out.gradFn());
                    afterForward = buffers(m);
                    loss.backward();
                }
                assertEquals(what + ": graph size after the step", start, graph.size());
                try {
                    Map<String, float[]> afterBackward = buffers(m);
                    for (Map.Entry<String, float[]> e : afterForward.entrySet()) {
                        assertArrayEquals(what + ": buffer " + e.getKey() + " unchanged by the backward",
                                e.getValue(), afterBackward.get(e.getKey()), 0.0f);
                    }
                    expectFinite(what + " input", in.grad(), in.shape(), be);
                    expectFinite(what + " weight", weight.grad(), weight.tensor().shape(), be);
                    if (bias != null) {
                        float[] gb = expectFinite(what + " bias", bias.grad(), bias.tensor().shape(), be);
                        for (int k = 0; k < gb.length; k++) {
                            assertEquals(what + ": bias gradient [" + k + "]", biasCount, gb[k], 0.0f);
                        }
                    }
                } finally {
                    graph.zeroGrad();
                }
            }
        }

        private static Map<String, float[]> buffers(Module m) {
            Map<String, float[]> values = new LinkedHashMap<>();
            for (Map.Entry<String, Tensor> e : m.namedBuffers().entrySet()) {
                values.put(e.getKey(), e.getValue().toFloatArray());
            }
            return values;
        }

        private static float[] expectFinite(String what, Tensor grad, Shape shape, ComputeBackend be) {
            assertNotNull(what + ": gradient delivered", grad);
            assertEquals(what + ": gradient shape", shape, grad.shape());
            assertSame(what + ": gradient backend", be, grad.backend());
            float[] values = grad.toFloatArray();
            for (int i = 0; i < values.length; i++) {
                assertTrue(what + ": finite gradient [" + i + "] " + values[i], Float.isFinite(values[i]));
            }
            return values;
        }

        private static float[] gamma(int n) {
            return TrainParity.uniform(1000 + n, n, 0.5f, 1.5f);
        }

        private static float[] beta(int n) {
            return TrainParity.uniform(2000 + n, n, -0.5f, 0.5f);
        }

        private static float[] runMean() {
            return TrainParity.uniform(3000, C, -0.1f, 0.1f);
        }

        private static float[] runVar() {
            return TrainParity.uniform(3001, C, 0.9f, 1.1f);
        }
    }

    // ------------------------------------------------------------------ references

    /** Row-wise LayerNorm ({@code rms == false}) or RMSNorm reference, asserted element by element. */
    static void assertRows(String what, float[] got, float[] x, int rows, int cols, float[] scale, float[] shift,
                           boolean rms, float eps) {
        assertEquals(what + ": length", x.length, got.length);
        for (int r = 0; r < rows; r++) {
            int base = r * cols;
            double mean = 0;
            double sq = 0;
            for (int j = 0; j < cols; j++) {
                mean += x[base + j];
                sq += (double) x[base + j] * x[base + j];
            }
            mean /= cols;
            double var = 0;
            for (int j = 0; j < cols; j++) {
                var += (x[base + j] - mean) * (x[base + j] - mean);
            }
            var /= cols;
            double denom = rms ? Math.sqrt(sq / cols + eps) : Math.sqrt(var + eps);
            double center = rms ? 0 : mean;
            for (int j = 0; j < cols; j++) {
                double g = scale == null ? 1 : scale[j];
                double b = shift == null ? 0 : shift[j];
                double ref = (x[base + j] - center) / denom * g + b;
                double terms = (Math.abs(x[base + j]) + Math.abs(center)) / denom * Math.abs(g) + Math.abs(b);
                TrainParity.assertReduction(what + " [" + r + "," + j + "]", ref, got[base + j], cols, terms);
            }
        }
    }

    /** Asserts a channel normalization of {@code x [b, c, s]} against {@code stats} (per channel or per group). */
    static void assertChannels(String what, float[] got, float[] x, int b, int c, int s, Stats stats, float[] gamma,
                               float[] beta) {
        assertEquals(what + ": length", x.length, got.length);
        for (int bi = 0; bi < b; bi++) {
            for (int ci = 0; ci < c; ci++) {
                int k = stats.index(bi, ci, c);
                double sd = Math.sqrt(stats.var[k] + EPS);
                double g = gamma == null ? 1 : gamma[ci];
                double sh = beta == null ? 0 : beta[ci];
                for (int si = 0; si < s; si++) {
                    int e = (bi * c + ci) * s + si;
                    double ref = (x[e] - stats.mean[k]) / sd * g + sh;
                    double terms = (Math.abs(x[e]) + Math.abs(stats.mean[k])) / sd * Math.abs(g) + Math.abs(sh);
                    TrainParity.assertReduction(what + " [" + e + "]", ref, got[e], stats.count, terms);
                }
            }
        }
    }

    /** Double-precision mean and biased variance per channel ({@code groups == 0}) or per (batch, group). */
    static final class Stats {
        final double[] mean;
        final double[] var;
        final double[] meanTerms;
        final double[] varTerms;
        final int groups;
        final int count;

        private Stats(int sets, int groups, int count) {
            this.mean = new double[sets];
            this.var = new double[sets];
            this.meanTerms = new double[sets];
            this.varTerms = new double[sets];
            this.groups = groups;
            this.count = count;
        }

        int index(int b, int c, int channels) {
            return groups == 0 ? c : b * groups + c / (channels / groups);
        }

        /** Given per-channel statistics (evaluation mode), each used as an exact single term. */
        static Stats given(float[] mean, float[] var) {
            Stats st = new Stats(mean.length, 0, 1);
            for (int k = 0; k < mean.length; k++) {
                st.mean[k] = mean[k];
                st.var[k] = var[k];
            }
            return st;
        }

        static Stats of(float[] x, int b, int c, int s, int groups) {
            int sets = groups == 0 ? c : b * groups;
            int count = groups == 0 ? b * s : (c / groups) * s;
            Stats st = new Stats(sets, groups, count);
            double[] abs = new double[sets];
            for (int bi = 0; bi < b; bi++) {
                for (int ci = 0; ci < c; ci++) {
                    int k = st.index(bi, ci, c);
                    for (int si = 0; si < s; si++) {
                        double v = x[(bi * c + ci) * s + si];
                        st.mean[k] += v;
                        abs[k] += Math.abs(v);
                    }
                }
            }
            for (int k = 0; k < sets; k++) {
                st.mean[k] /= count;
                st.meanTerms[k] = abs[k] / count;
            }
            for (int bi = 0; bi < b; bi++) {
                for (int ci = 0; ci < c; ci++) {
                    int k = st.index(bi, ci, c);
                    for (int si = 0; si < s; si++) {
                        double v = x[(bi * c + ci) * s + si];
                        double m = Math.abs(v) + Math.abs(st.mean[k]);
                        st.var[k] += (v - st.mean[k]) * (v - st.mean[k]);
                        st.varTerms[k] += m * m;
                    }
                }
            }
            for (int k = 0; k < sets; k++) {
                st.var[k] /= count;
                st.varTerms[k] /= count;
            }
            return st;
        }
    }
}
