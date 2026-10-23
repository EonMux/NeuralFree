package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.loss.LossFunction.Reduction;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The loss functions on their CPU path (a backend without the train library) and on each GPU API, all against
 * double-precision references of the loss formulas in the reduction class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}. Each element's bound uses the absolute sizes of the
 * terms that form it, with every logarithm, exponential or softplus counted at least 1 (the transcendental class's
 * absolute floor). Holding both paths to one reference bounds their difference by twice that tolerance. Every loss is
 * checked with each reduction; the device results must stay on the device. Every loss records an autograd node exactly
 * when grad mode is on and an operand requires grad, and the gradients of the where-based and logit losses are checked
 * against their analytic derivatives (division class), including the points where a composition of kinked ops could get
 * them wrong (a zero logit, identical triplet rows).
 */
@RunWith(Enclosed.class)
public class LossFunctionsTest {

    /** Names, configuration and constructor validation, which need no backend. */
    public static class Configuration {

        @Test
        public void lossFunctionInterfaceAndReductionsExist() {
            assertTrue(LossFunction.class.isInterface());
            assertEquals(3, Reduction.values().length);
        }

        @Test
        public void lossNames() {
            assertEquals("MSELoss", new MSELoss().name());
            assertEquals("L1Loss", new L1Loss().name());
            assertEquals("SmoothL1Loss", new SmoothL1Loss().name());
            assertEquals("HuberLoss", new HuberLoss().name());
            assertEquals("CrossEntropyLoss", new CrossEntropyLoss().name());
            assertEquals("BCELoss", new BCELoss().name());
            assertEquals("BCEWithLogitsLoss", new BCEWithLogitsLoss().name());
            assertEquals("KLDivLoss", new KLDivLoss().name());
            assertEquals("TripletMarginLoss", new TripletMarginLoss().name());
            assertEquals("ContrastiveLoss", new ContrastiveLoss().name());
            assertEquals("FocalLoss", new FocalLoss().name());
        }

        @Test
        public void reductionsAndProperties() {
            assertEquals(Reduction.MEAN, new MSELoss().reduction());
            assertEquals(Reduction.SUM, new MSELoss(Reduction.SUM).reduction());
            assertEquals(Reduction.NONE, new MSELoss(Reduction.NONE).reduction());
            SmoothL1Loss smooth = new SmoothL1Loss(Reduction.SUM, 0.5f);
            assertEquals(0.5f, smooth.beta(), 0.0f);
            assertEquals(Reduction.SUM, smooth.reduction());
            assertEquals(2.0f, new HuberLoss(Reduction.SUM, 2.0f).delta(), 0.0f);
            CrossEntropyLoss ce = new CrossEntropyLoss(Reduction.SUM, 0, 0.1f);
            assertEquals(0, ce.dim());
            assertEquals(0.1f, ce.labelSmoothing(), 0.0f);
            FocalLoss focal = new FocalLoss(0.5f, 3.0f, Reduction.SUM, 1);
            assertEquals(0.5f, focal.alpha(), 0.0f);
            assertEquals(3.0f, focal.gamma(), 0.0f);
            assertEquals(1, focal.dim());
            TripletMarginLoss triplet = new TripletMarginLoss(2.0f, 1.0f, Reduction.SUM);
            assertEquals(2.0f, triplet.margin(), 0.0f);
            assertEquals(1.0f, triplet.p(), 0.0f);
            assertEquals(2.0f, new ContrastiveLoss(2.0f, Reduction.SUM).margin(), 0.0f);
            assertEquals(2.0f, new BCEWithLogitsLoss(Reduction.SUM, 2.0f).posWeight(), 0.0f);
            assertTrue(new KLDivLoss(Reduction.SUM, true).logTarget());
            KLDivLoss kl = new KLDivLoss();
            assertTrue("batchmean is the default", kl.isBatchMean());
            assertFalse(kl.logTarget());
            assertEquals(Reduction.MEAN, kl.reduction());
            assertTrue(KLDivLoss.batchMean(true).isBatchMean());
            assertTrue(KLDivLoss.batchMean(true).logTarget());
            assertFalse(new KLDivLoss(Reduction.MEAN).isBatchMean());
            assertFalse(new KLDivLoss(Reduction.SUM, true).isBatchMean());
        }

        @Test(expected = IllegalArgumentException.class)
        public void smoothL1RejectsNonPositiveBeta() {
            new SmoothL1Loss(Reduction.MEAN, 0.0f);
        }

        @Test(expected = IllegalArgumentException.class)
        public void huberRejectsNegativeDelta() {
            new HuberLoss(Reduction.MEAN, -1.0f);
        }

        @Test(expected = IllegalArgumentException.class)
        public void focalRejectsNegativeGamma() {
            new FocalLoss(0.25f, -1.0f);
        }

        @Test(expected = IllegalArgumentException.class)
        public void tripletRejectsNegativeMargin() {
            new TripletMarginLoss(-1.0f);
        }

        @Test(expected = IllegalArgumentException.class)
        public void crossEntropyRejectsSmoothingOfOne() {
            new CrossEntropyLoss(Reduction.MEAN, -1, 1.0f);
        }
    }

    /** Every loss on the CPU path, and the argument checks the compositions add. */
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
        public void elementwiseLossesMatchReference() {
            Scenarios.elementwise(host, "cpu path");
        }

        @Test
        public void binaryAndDivergenceLossesMatchReference() {
            Scenarios.binary(host, "cpu path");
        }

        @Test
        public void crossEntropyMatchesReference() {
            Scenarios.crossEntropy(host, "cpu path");
        }

        @Test
        public void focalMatchesReference() {
            Scenarios.focal(host, "cpu path");
        }

        @Test
        public void embeddingLossesMatchReference() {
            Scenarios.embedding(host, "cpu path");
        }

        @Test
        public void everyLossRecordsExactlyWhenAnOperandRequiresGrad() {
            Scenarios.recording(host, "cpu path");
        }

        @Test
        public void gradientsMatchTheLossDerivatives() {
            Scenarios.gradients(host, "cpu path");
        }

        @Test
        public void anEmptyClassAxisGivesARecordedZeroLoss() {
            Scenarios.emptyClassAxis(host, "cpu path");
        }

        @Test(expected = IllegalArgumentException.class)
        public void elementwiseTargetWithAnotherElementCountIsRejected() {
            try (Tensor p = Scenarios.tensor(host, new float[6], 2, 3);
                 Tensor t = Scenarios.tensor(host, new float[5], 5)) {
                new HuberLoss().compute(p, t);
            }
        }

        @Test(expected = IllegalArgumentException.class)
        public void classTargetWithAnotherSampleCountIsRejected() {
            try (Tensor p = Scenarios.tensor(host, new float[12], 4, 3);
                 Tensor t = Scenarios.tensor(host, new float[3], 3)) {
                new CrossEntropyLoss().compute(p, t);
            }
        }

        @Test(expected = IllegalArgumentException.class)
        public void classDimOutsideTheRankIsRejected() {
            try (Tensor p = Scenarios.tensor(host, new float[12], 4, 3);
                 Tensor t = Scenarios.tensor(host, new float[4], 4)) {
                new FocalLoss(0.25f, 2.0f, Reduction.MEAN, 2).compute(p, t);
            }
        }

        @Test(expected = IllegalArgumentException.class)
        public void tripletWithMismatchedInputsIsRejected() {
            try (Tensor a = Scenarios.tensor(host, new float[6], 2, 3);
                 Tensor b = Scenarios.tensor(host, new float[6], 2, 3);
                 Tensor c = Scenarios.tensor(host, new float[4], 2, 2)) {
                new TripletMarginLoss().compute(a, b, c);
            }
        }

        @Test(expected = IllegalArgumentException.class)
        public void contrastiveLabelWithAnotherSampleCountIsRejected() {
            try (Tensor a = Scenarios.tensor(host, new float[6], 2, 3);
                 Tensor b = Scenarios.tensor(host, new float[6], 2, 3);
                 Tensor y = Scenarios.tensor(host, new float[3], 3)) {
                new ContrastiveLoss().compute(a, b, y);
            }
        }

        @Test(expected = UnsupportedOperationException.class)
        public void pairLossesRejectTheTwoArgumentForm() {
            try (Tensor a = Scenarios.tensor(host, new float[6], 2, 3)) {
                new TripletMarginLoss().compute(a, a);
            }
        }
    }

    /** Every loss on each GPU API, against the same references as the CPU path. */
    @RunWith(Parameterized.class)
    public static class Device {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;
        private ComputeBackend gpu;

        public Device(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            session = TrainParity.open(api);
            gpu = session.backend();
        }

        @After
        public void tearDown() {
            if (session != null) {
                session.close();
            }
        }

        @Test
        public void elementwiseLossesMatchReference() {
            Scenarios.elementwise(gpu, api.toString());
        }

        @Test
        public void binaryAndDivergenceLossesMatchReference() {
            Scenarios.binary(gpu, api.toString());
        }

        @Test
        public void crossEntropyMatchesReference() {
            Scenarios.crossEntropy(gpu, api.toString());
        }

        @Test
        public void focalMatchesReference() {
            Scenarios.focal(gpu, api.toString());
        }

        @Test
        public void embeddingLossesMatchReference() {
            Scenarios.embedding(gpu, api.toString());
        }

        @Test
        public void everyLossRecordsExactlyWhenAnOperandRequiresGrad() {
            Scenarios.recording(gpu, api.toString());
        }

        @Test
        public void gradientsMatchTheLossDerivatives() {
            Scenarios.gradients(gpu, api.toString());
        }

        @Test
        public void anEmptyClassAxisGivesARecordedZeroLoss() {
            Scenarios.emptyClassAxis(gpu, api.toString());
        }
    }

    // ------------------------------------------------------------------ scenarios

    /** Loss runs on one backend, checked against {@link Ref} references for every reduction. */
    static final class Scenarios {
        private static final float BCE_EPS = 1e-7f;
        private static final double TRIPLET_EPS = 1e-6;

        private Scenarios() {
        }

        static Tensor tensor(ComputeBackend b, float[] data, int... dims) {
            try (Tensor flat = Tensor.fromArray(b, data)) {
                return flat.reshape(dims);
            }
        }

        /** MSE, L1, SmoothL1 and Huber on {@code [3, 4]}; Huber takes its target as {@code [12]}. */
        static void elementwise(ComputeBackend b, String what) {
            float[] p = TrainParity.uniform(1, 12, -3f, 3f);
            float[] t = TrainParity.uniform(2, 12, -3f, 3f);
            Ref mse = new Ref(Shape.of(3, 4), 1);
            Ref l1 = new Ref(Shape.of(3, 4), 1);
            Ref smooth = new Ref(Shape.of(3, 4), 1);
            Ref huber = new Ref(Shape.of(3, 4), 1);
            float beta = 0.8f;
            float delta = 0.9f;
            for (int i = 0; i < 12; i++) {
                float d = p[i] - t[i];
                float a = Math.abs(d);
                mse.set(i, (double) d * d, (double) d * d);
                l1.set(i, a, a);
                if (a < beta) {
                    double v = 0.5 * a * a / ((double) beta * beta);
                    smooth.set(i, v, v);
                } else {
                    smooth.set(i, a / (double) beta - 0.5, a / (double) beta + 0.5);
                }
                if (a <= delta) {
                    huber.set(i, 0.5 * a * a, 0.5 * a * a);
                } else {
                    huber.set(i, delta * (a - 0.5 * delta), delta * (a + 0.5 * delta));
                }
            }
            try (Tensor pt = tensor(b, p, 3, 4);
                 Tensor tt = tensor(b, t, 3, 4);
                 Tensor flatTarget = tensor(b, t, 12)) {
                expect(what + " MSELoss", b, mse, r -> new MSELoss(r).compute(pt, tt));
                expect(what + " L1Loss", b, l1, r -> new L1Loss(r).compute(pt, tt));
                expect(what + " SmoothL1Loss", b, smooth, r -> new SmoothL1Loss(r, beta).compute(pt, tt));
                expect(what + " HuberLoss", b, huber, r -> new HuberLoss(r, delta).compute(pt, flatTarget));
            }
        }

        /**
         * BCE (with clamped probabilities), BCE with logits (with and without a positive weight) and the KL divergence
         * of PyTorch's {@code kl_div} on log-probability predictions (a zero target contributes exactly 0), with every
         * reduction and with {@code batchmean}.
         */
        static void binary(ComputeBackend b, String what) {
            float[] prob = TrainParity.uniform(3, 12, 0f, 1f);
            prob[0] = 0.0f;
            prob[1] = 1.0f;
            prob[2] = 1e-9f;
            float[] t = TrainParity.uniform(4, 12, 0f, 1f);
            t[0] = 1.0f;
            t[1] = 0.0f;
            float[] logits = TrainParity.uniform(5, 12, -8f, 8f);
            logits[0] = 120f;
            logits[1] = -120f;
            float[] logT = TrainParity.uniform(6, 12, -5f, 0f);
            float[] logP = TrainParity.uniform(7, 12, -6f, 0f);
            logP[2] = 0.0f;
            Ref bce = new Ref(Shape.of(2, 6), 1);
            Ref logitsRef = new Ref(Shape.of(2, 6), 1);
            Ref weighted = new Ref(Shape.of(2, 6), 1);
            Ref kl = new Ref(Shape.of(2, 6), 1);
            Ref klLog = new Ref(Shape.of(2, 6), 1);
            double w = 2.5;
            for (int i = 0; i < 12; i++) {
                float pc = Math.max(BCE_EPS, Math.min(1.0f - BCE_EPS, prob[i]));
                double lp = Math.log(pc);
                double lq = Math.log(1.0f - pc);
                bce.set(i, -(t[i] * lp + (1.0 - t[i]) * lq), Math.abs(t[i]) * floor1(lp)
                        + Math.abs(1.0 - t[i]) * floor1(lq));

                double x = logits[i];
                double s = Math.log1p(Math.exp(-Math.abs(x)));
                double pos = Math.max(x, 0.0);
                double neg = Math.max(-x, 0.0);
                logitsRef.set(i, pos - t[i] * x + s, pos + Math.abs(t[i] * x) + floor1(s));
                weighted.set(i, w * t[i] * (neg + s) + (1.0 - t[i]) * (pos + s),
                        Math.abs(w * t[i]) * (neg + floor1(s)) + Math.abs(1.0 - t[i]) * (pos + floor1(s)));

                double xi = logP[i];
                if (t[i] == 0.0f) {
                    kl.set(i, 0.0, 0.0);
                } else {
                    double logQ = Math.log(t[i]);
                    kl.set(i, t[i] * logQ - t[i] * xi, t[i] * floor1(logQ) + Math.abs(t[i] * xi));
                }
                double qe = Math.exp(logT[i]);
                klLog.set(i, qe * (logT[i] - xi), floor1(qe) * (Math.abs(logT[i]) + Math.abs(xi)));
            }
            try (Tensor probs = tensor(b, prob, 2, 6);
                 Tensor targets = tensor(b, t, 2, 6);
                 Tensor x = tensor(b, logits, 2, 6);
                 Tensor logProbs = tensor(b, logP, 2, 6);
                 Tensor logTargets = tensor(b, logT, 12)) {
                expect(what + " BCELoss", b, bce, r -> new BCELoss(r).compute(probs, targets));
                expect(what + " BCEWithLogitsLoss", b, logitsRef, r -> new BCEWithLogitsLoss(r).compute(x, targets));
                expect(what + " BCEWithLogitsLoss posWeight", b, weighted,
                        r -> new BCEWithLogitsLoss(r, (float) w).compute(x, targets));
                expect(what + " KLDivLoss", b, kl, r -> new KLDivLoss(r).compute(logProbs, targets));
                expect(what + " KLDivLoss logTarget", b, klLog,
                        r -> new KLDivLoss(r, true).compute(logProbs, logTargets));
                expectBatchMean(what + " KLDivLoss batchmean", b, kl, 2,
                        () -> new KLDivLoss().compute(logProbs, targets));
                expectBatchMean(what + " KLDivLoss batchmean logTarget", b, klLog, 2,
                        () -> KLDivLoss.batchMean(true).compute(logProbs, logTargets));
            }
        }

        /**
         * Class distributions along an inner dim, label smoothing, and FLOAT32 class values read as
         * {@code Math.round} (NaN as class 0).
         */
        static void crossEntropy(ComputeBackend b, String what) {
            float[] x3 = TrainParity.uniform(10, 24, -4f, 4f);
            float[] t3 = TrainParity.uniform(11, 24, -0.2f, 1f);
            Ref dist = new Ref(Shape.of(2, 4), 3);
            for (int o = 0; o < 2; o++) {
                for (int i = 0; i < 4; i++) {
                    double[] lp = logSoftmax(x3, o * 12 + i, 4, 3);
                    double v = 0;
                    double m = 0;
                    for (int c = 0; c < 3; c++) {
                        float tc = t3[o * 12 + c * 4 + i];
                        if (tc > 0) {
                            v -= tc * lp[c];
                            m += tc * floor1(lp[c]);
                        }
                    }
                    dist.set(o * 4 + i, v, m);
                }
            }

            float[] x2 = TrainParity.uniform(12, 15, -3f, 3f);
            float[] oneHot = new float[15];
            for (int r = 0; r < 3; r++) {
                oneHot[r * 5 + (2 * r + 1) % 5] = 1.0f;
            }
            float ls = 0.1f;
            Ref smooth = new Ref(Shape.of(3), 5);
            for (int r = 0; r < 3; r++) {
                double[] lp = logSoftmax(x2, r * 5, 1, 5);
                double v = 0;
                double m = 0;
                for (int c = 0; c < 5; c++) {
                    double wc = oneHot[r * 5 + c] * (1.0 - ls) + ls / 5.0;
                    v -= wc * lp[c];
                    m += wc * floor1(lp[c]);
                }
                smooth.set(r, v, m);
            }

            float[] xc = TrainParity.uniform(13, 12, -3f, 3f);
            float[] classes = {1.0f, 1.6f, -0.4f, Float.NaN};
            int[] expectedClass = {1, 2, 0, 0};
            Ref index = classRef(xc, 4, 3, expectedClass, lp -> -lp, lp -> floor1(lp));

            float[] xr = TrainParity.uniform(14, 12, -3f, 3f);
            float[] classes3 = {0.0f, 2.0f, 1.4f, 0.5f};
            Ref index3 = new Ref(Shape.of(2, 2), 3);
            int[] expected3 = {0, 2, 1, 1};
            for (int o = 0; o < 2; o++) {
                for (int i = 0; i < 2; i++) {
                    double lp = logSoftmax(xr, o * 6 + i, 2, 3)[expected3[o * 2 + i]];
                    index3.set(o * 2 + i, -lp, floor1(lp));
                }
            }

            try (Tensor p3 = tensor(b, x3, 2, 3, 4);
                 Tensor tt3 = tensor(b, t3, 2, 3, 4);
                 Tensor p2 = tensor(b, x2, 3, 5);
                 Tensor hot = tensor(b, oneHot, 3, 5);
                 Tensor pc = tensor(b, xc, 4, 3);
                 Tensor tc = tensor(b, classes, 4);
                 Tensor pr = tensor(b, xr, 2, 3, 2);
                 Tensor tr = tensor(b, classes3, 2, 2)) {
                expect(what + " CrossEntropyLoss distribution dim 1", b, dist,
                        r -> new CrossEntropyLoss(r, 1, 0.0f).compute(p3, tt3));
                expect(what + " CrossEntropyLoss smoothing", b, smooth,
                        r -> new CrossEntropyLoss(r, -1, ls).compute(p2, hot));
                expect(what + " CrossEntropyLoss class values", b, index, r -> new CrossEntropyLoss(r).compute(pc, tc));
                expect(what + " CrossEntropyLoss class values dim 1", b, index3,
                        r -> new CrossEntropyLoss(r, 1, 0.0f).compute(pr, tr));
                ComputationGraph graph = ComputationGraph.current();
                int start = graph.size();
                try (Tensor withGrad = pc.withRequiresGrad(true);
                     ComputationGraph.StepScope step = graph.step();
                     Tensor perSample = new CrossEntropyLoss(Reduction.NONE).compute(withGrad, tc);
                     Tensor mean = new CrossEntropyLoss(Reduction.MEAN).compute(withGrad, tc)) {
                    assertTrue(what + " per-sample loss requires grad", perSample.requiresGrad());
                    assertNotNull(what + " per-sample loss is a recorded node", perSample.gradFn());
                    assertTrue(what + " mean loss requires grad", mean.requiresGrad());
                    assertNotNull(what + " mean loss is a recorded node", mean.gradFn());
                }
                assertEquals(what + " the step leaves the graph as it was", start, graph.size());
                try (Tensor plain = new CrossEntropyLoss(Reduction.MEAN).compute(pc, tc)) {
                    assertFalse(what + " loss without grad", plain.requiresGrad());
                    assertNull(what + " loss without grad records nothing", plain.gradFn());
                }
            }
        }

        /** Focal loss on a class distribution and on FLOAT32 class values read as {@code Math.round}. */
        static void focal(ComputeBackend b, String what) {
            float alpha = 0.25f;
            float gamma = 2.0f;
            float[] x = TrainParity.uniform(20, 12, -3f, 3f);
            float[] t = TrainParity.uniform(21, 12, -0.2f, 1f);
            Ref dist = new Ref(Shape.of(3), 4);
            for (int r = 0; r < 3; r++) {
                double[] lp = logSoftmax(x, r * 4, 1, 4);
                double v = 0;
                double m = 0;
                for (int c = 0; c < 4; c++) {
                    float tc = t[r * 4 + c];
                    if (tc > 0) {
                        double a = tc > 0.5f ? alpha : 1.0 - alpha;
                        double f = Math.pow(1.0 - Math.exp(lp[c]), gamma);
                        v -= a * f * tc * lp[c];
                        m += a * tc * floor1(lp[c]) * (2.0 + gamma);
                    }
                }
                dist.set(r, v, m);
            }

            float alpha2 = 0.4f;
            float gamma2 = 1.5f;
            float[] xc = TrainParity.uniform(22, 12, -3f, 3f);
            float[] classes = {2.0f, 0.2f, 1.4f, -0.3f};
            int[] expectedClass = {2, 0, 1, 0};
            Ref index = classRef(xc, 4, 3, expectedClass,
                    lp -> -alpha2 * Math.pow(1.0 - Math.exp(lp), gamma2) * lp,
                    lp -> alpha2 * floor1(lp) * (2.0 + gamma2));

            try (Tensor p = tensor(b, x, 3, 4);
                 Tensor tt = tensor(b, t, 12);
                 Tensor pc = tensor(b, xc, 4, 3);
                 Tensor tc = tensor(b, classes, 4)) {
                expect(what + " FocalLoss distribution", b, dist,
                        r -> new FocalLoss(alpha, gamma, r, -1).compute(p, tt));
                expect(what + " FocalLoss class values", b, index,
                        r -> new FocalLoss(alpha2, gamma2, r, 1).compute(pc, tc));
            }
        }

        /**
         * Triplet margin (p = 2, and p = 3 with partial weights; distances {@code ||a - b + 1e-6||_p}) and
         * contrastive loss on {@code [4, 5]}.
         */
        static void embedding(ComputeBackend b, String what) {
            float[] anchor = TrainParity.uniform(30, 20, -1f, 1f);
            float[] noise = TrainParity.uniform(31, 20, -0.3f, 0.3f);
            float[] positive = new float[20];
            for (int i = 0; i < 20; i++) {
                positive[i] = anchor[i] + noise[i];
            }
            float[] negative = TrainParity.uniform(32, 20, -1f, 1f);
            float[] weights = {2.0f, 0.5f};
            Ref triplet2 = tripletRef(anchor, positive, negative, 2.0f, 1.0f, null);
            Ref triplet3 = tripletRef(anchor, positive, negative, 3.0f, 0.5f, weights);

            float[] labels = {0.0f, 1.0f, 1.0f, 0.5f};
            float margin = 1.5f;
            Ref contrastive = new Ref(Shape.of(4), 5);
            for (int s = 0; s < 4; s++) {
                double ds = 0;
                for (int f = 0; f < 5; f++) {
                    float d = anchor[s * 5 + f] - negative[s * 5 + f];
                    ds += (double) d * d;
                }
                double dist = Math.sqrt(ds + 1e-8);
                double hinge = Math.max(0.0, margin - dist);
                double y = labels[s];
                contrastive.set(s, (1.0 - y) * ds + y * hinge * hinge,
                        Math.abs(1.0 - y) * ds + Math.abs(y) * (margin + dist) * (margin + dist));
            }

            try (Tensor a = tensor(b, anchor, 4, 5);
                 Tensor pos = tensor(b, positive, 4, 5);
                 Tensor neg = tensor(b, negative, 20);
                 Tensor y = tensor(b, labels, 4)) {
                expect(what + " TripletMarginLoss p=2", b, triplet2,
                        r -> new TripletMarginLoss(1.0f, 2.0f, r).compute(a, pos, neg));
                expect(what + " TripletMarginLoss p=3 weighted", b, triplet3,
                        r -> new TripletMarginLoss(0.5f, 3.0f, r).compute(a, pos, neg, weights));
                expect(what + " ContrastiveLoss", b, contrastive,
                        r -> new ContrastiveLoss(margin, r).compute(a, neg, y));
            }
        }

        // -------------------------------------------------------------- autograd

        /**
         * Every loss on a {@code [3, 4]} prediction: with a prediction that requires grad in grad mode the NONE and
         * MEAN results are recorded nodes; under no-grad, and for a prediction that does not require grad, they
         * neither require grad nor carry a node. The graph is left as it was.
         */
        static void recording(ComputeBackend b, String what) {
            float[] p = TrainParity.uniform(40, 12, 0.05f, 0.95f);
            float[] t = TrainParity.uniform(41, 12, 0f, 1f);
            float[] other = TrainParity.uniform(42, 12, -1f, 1f);
            ComputationGraph graph = ComputationGraph.current();
            int start = graph.size();
            try (Tensor base = tensor(b, p, 3, 4);
                 Tensor leaf = base.withRequiresGrad(true);
                 Tensor target = tensor(b, t, 3, 4);
                 Tensor classes = tensor(b, new float[]{0f, 3f, 1f}, 3);
                 Tensor second = tensor(b, other, 3, 4);
                 Tensor labels = tensor(b, new float[]{0f, 1f, 1f}, 3)) {
                for (Map.Entry<String, BiFunction<Tensor, Reduction, Tensor>> e
                        : losses(target, classes, second, labels).entrySet()) {
                    String name = what + " " + e.getKey();
                    for (Reduction r : new Reduction[]{Reduction.NONE, Reduction.MEAN}) {
                        try (ComputationGraph.StepScope step = graph.step();
                             Tensor out = e.getValue().apply(leaf, r)) {
                            assertTrue(name + " " + r + " requires grad", out.requiresGrad());
                            assertNotNull(name + " " + r + " is a recorded node", out.gradFn());
                        }
                        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                             Tensor out = e.getValue().apply(leaf, r)) {
                            assertFalse(name + " " + r + " under no-grad", out.requiresGrad());
                            assertNull(name + " " + r + " under no-grad records nothing", out.gradFn());
                        }
                        try (Tensor out = e.getValue().apply(base, r)) {
                            assertFalse(name + " " + r + " without a grad operand", out.requiresGrad());
                            assertNull(name + " " + r + " without a grad operand records nothing", out.gradFn());
                        }
                        assertEquals(name + " " + r + ": graph size", start, graph.size());
                    }
                }
            }
        }

        /**
         * Cross-entropy and focal loss on a {@code [3, 0]} prediction (an empty class axis): the per-sample loss is
         * the empty sum, 0 for each of the 3 samples, on the prediction's backend, and it is a recorded node when the
         * prediction requires grad. The graph is left as it was.
         */
        static void emptyClassAxis(ComputeBackend b, String what) {
            ComputationGraph graph = ComputationGraph.current();
            int start = graph.size();
            try (Tensor base = Tensor.zeros(b, Shape.of(3, 0));
                 Tensor leaf = base.withRequiresGrad(true);
                 Tensor classes = tensor(b, new float[]{0f, 1f, 2f}, 3)) {
                Map<String, Function<Tensor, Tensor>> losses = new LinkedHashMap<>();
                losses.put("CrossEntropyLoss", x -> new CrossEntropyLoss(Reduction.NONE).compute(x, classes));
                losses.put("FocalLoss", x -> new FocalLoss(0.25f, 2.0f, Reduction.NONE, -1).compute(x, classes));
                for (Map.Entry<String, Function<Tensor, Tensor>> e : losses.entrySet()) {
                    String name = what + " " + e.getKey() + " with no classes";
                    try (ComputationGraph.StepScope step = graph.step();
                         Tensor out = e.getValue().apply(leaf)) {
                        assertEquals(name + ": shape", Shape.of(3), out.shape());
                        assertSame(name + ": backend", b, out.backend());
                        assertNotNull(name + ": recorded node", out.gradFn());
                        float[] values = out.toFloatArray();
                        for (int i = 0; i < values.length; i++) {
                            assertEquals(name + ": loss [" + i + "]", 0.0f, values[i], 0.0f);
                        }
                    }
                    assertEquals(name + ": graph size", start, graph.size());
                }
            }
        }

        /** The losses of {@link #recording}, keyed by name, as functions of the prediction and the reduction. */
        private static Map<String, BiFunction<Tensor, Reduction, Tensor>> losses(Tensor target, Tensor classes,
                                                                                 Tensor second, Tensor labels) {
            Map<String, BiFunction<Tensor, Reduction, Tensor>> m = new LinkedHashMap<>();
            m.put("MSELoss", (x, r) -> new MSELoss(r).compute(x, target));
            m.put("L1Loss", (x, r) -> new L1Loss(r).compute(x, target));
            m.put("SmoothL1Loss", (x, r) -> new SmoothL1Loss(r, 0.3f).compute(x, target));
            m.put("HuberLoss", (x, r) -> new HuberLoss(r, 0.3f).compute(x, target));
            m.put("BCELoss", (x, r) -> new BCELoss(r).compute(x, target));
            m.put("BCEWithLogitsLoss", (x, r) -> new BCEWithLogitsLoss(r).compute(x, target));
            m.put("BCEWithLogitsLoss posWeight", (x, r) -> new BCEWithLogitsLoss(r, 2.0f).compute(x, target));
            m.put("KLDivLoss", (x, r) -> new KLDivLoss(r).compute(x, target));
            m.put("CrossEntropyLoss class values", (x, r) -> new CrossEntropyLoss(r).compute(x, classes));
            m.put("CrossEntropyLoss distribution", (x, r) -> new CrossEntropyLoss(r).compute(x, target));
            m.put("FocalLoss class values", (x, r) -> new FocalLoss(0.25f, 2.0f, r, -1).compute(x, classes));
            m.put("FocalLoss distribution", (x, r) -> new FocalLoss(0.25f, 2.0f, r, -1).compute(x, target));
            m.put("TripletMarginLoss", (x, r) -> new TripletMarginLoss(1.0f, 2.0f, r).compute(x, second, target));
            m.put("ContrastiveLoss", (x, r) -> new ContrastiveLoss(1.0f, r).compute(x, second, labels));
            return m;
        }

        /**
         * Gradients of MEAN losses with respect to the prediction against their derivatives: BCE with logits
         * ({@code (sigmoid(x) - t) / n}, and {@code ((1 - t) * sigmoid(x) - w * t * sigmoid(-x)) / n} with a positive
         * weight) at logits including 0; cross-entropy on class values ({@code (softmax - onehot) / n}); Huber
         * ({@code d / n} inside delta, {@code delta * sign(d) / n} beyond); a triplet whose positive equals the
         * anchor, whose gradient must stay finite; and the {@code batchmean} KL divergence
         * ({@code -t / batch}, including zero targets, and {@code -exp(lt) / batch} with log targets).
         */
        static void gradients(ComputeBackend b, String what) {
            float[] x = {0.0f, 0.0f, 2.0f, -3.0f};
            float[] t = {0.0f, 1.0f, 0.25f, 1.0f};
            float w = 2.0f;
            double[] plain = new double[4];
            double[] weighted = new double[4];
            for (int i = 0; i < 4; i++) {
                double s = 1.0 / (1.0 + Math.exp(-x[i]));
                plain[i] = (s - t[i]) / 4.0;
                weighted[i] = ((1.0 - t[i]) * s - w * t[i] * (1.0 - s)) / 4.0;
            }
            try (Tensor tt = tensor(b, t, 4)) {
                expectGradient(what + " BCEWithLogitsLoss", plain, 2.0 / 4.0,
                        gradient(b, x, new int[]{4}, p -> new BCEWithLogitsLoss(Reduction.MEAN).compute(p, tt)));
                expectGradient(what + " BCEWithLogitsLoss posWeight", weighted, (1.0 + w) / 4.0,
                        gradient(b, x, new int[]{4}, p -> new BCEWithLogitsLoss(Reduction.MEAN, w).compute(p, tt)));
            }

            float[] xc = TrainParity.uniform(13, 12, -3f, 3f);
            float[] classes = {1.0f, 1.6f, -0.4f, Float.NaN};
            int[] expectedClass = {1, 2, 0, 0};
            double[] ce = new double[12];
            for (int r = 0; r < 4; r++) {
                double[] lp = logSoftmax(xc, r * 3, 1, 3);
                for (int c = 0; c < 3; c++) {
                    ce[r * 3 + c] = (Math.exp(lp[c]) - (c == expectedClass[r] ? 1.0 : 0.0)) / 4.0;
                }
            }
            try (Tensor tc = tensor(b, classes, 4)) {
                expectGradient(what + " CrossEntropyLoss class values", ce, 2.0 / 4.0,
                        gradient(b, xc, new int[]{4, 3}, p -> new CrossEntropyLoss(Reduction.MEAN).compute(p, tc)));
            }

            float[] p = TrainParity.uniform(1, 12, -3f, 3f);
            float[] target = TrainParity.uniform(2, 12, -3f, 3f);
            float delta = 0.9f;
            double[] huber = new double[12];
            for (int i = 0; i < 12; i++) {
                float d = p[i] - target[i];
                huber[i] = (Math.abs(d) > delta ? delta * Math.signum(d) : d) / 12.0;
            }
            try (Tensor tt = tensor(b, target, 3, 4)) {
                expectGradient(what + " HuberLoss", huber, (6.0 + delta) / 12.0,
                        gradient(b, p, new int[]{3, 4}, q -> new HuberLoss(Reduction.MEAN, delta).compute(q, tt)));
            }

            float[] logP = TrainParity.uniform(15, 8, -4f, 0f);
            float[] tk = {0.0f, 0.25f, 0.5f, 0.25f, 1.0f, 0.0f, 0.0f, 0.0f};
            float[] ltk = TrainParity.uniform(16, 8, -3f, 0f);
            double[] klGrad = new double[8];
            double[] klLogGrad = new double[8];
            for (int i = 0; i < 8; i++) {
                klGrad[i] = -tk[i] / 2.0;
                klLogGrad[i] = -Math.exp(ltk[i]) / 2.0;
            }
            try (Tensor tt = tensor(b, tk, 2, 4); Tensor lt = tensor(b, ltk, 2, 4)) {
                expectGradient(what + " KLDivLoss batchmean", klGrad, 1.0 / 2.0,
                        gradient(b, logP, new int[]{2, 4}, q -> new KLDivLoss().compute(q, tt)));
                expectGradient(what + " KLDivLoss batchmean logTarget", klLogGrad, 1.0 / 2.0,
                        gradient(b, logP, new int[]{2, 4}, q -> KLDivLoss.batchMean(true).compute(q, lt)));
            }

            float[] anchor = TrainParity.uniform(30, 20, -1f, 1f);
            float[] negative = TrainParity.uniform(32, 20, -1f, 1f);
            try (Tensor same = tensor(b, anchor, 4, 5);
                 Tensor neg = tensor(b, negative, 4, 5)) {
                float[] g = gradient(b, anchor, new int[]{4, 5},
                        a -> new TripletMarginLoss(1.0f, 2.0f, Reduction.MEAN).compute(a, same, neg));
                for (int i = 0; i < g.length; i++) {
                    assertTrue(what + " TripletMarginLoss identical rows: finite gradient [" + i + "] " + g[i],
                            Float.isFinite(g[i]));
                }
            }
        }

        /**
         * The gradient that back-propagating {@code loss} of a leaf holding {@code data} delivers to the leaf, as
         * host floats. The loss must be a recorded scalar; the step and the leaf gradient are released afterwards, so
         * the graph is left as it was.
         */
        static float[] gradient(ComputeBackend b, float[] data, int[] dims, Function<Tensor, Tensor> loss) {
            ComputationGraph graph = ComputationGraph.current();
            int start = graph.size();
            try (Tensor base = tensor(b, data, dims);
                 Tensor leaf = base.withRequiresGrad(true)) {
                try (ComputationGraph.StepScope step = graph.step();
                     Tensor result = loss.apply(leaf)) {
                    assertNotNull("the loss is a recorded node", result.gradFn());
                    result.backward();
                }
                assertEquals("graph size after the step", start, graph.size());
                Tensor g = leaf.grad();
                assertNotNull("the leaf received a gradient", g);
                try {
                    assertEquals("gradient shape", leaf.shape(), g.shape());
                    assertSame("gradient backend", b, g.backend());
                    return g.toFloatArray();
                } finally {
                    graph.zeroGrad();
                }
            }
        }

        /**
         * Gradients against a double reference within the transcendental class of their largest term:
         * {@code |got - ref| <= 1e-5 * terms}, with {@code terms} bounding the absolute size of the terms that form
         * each element (the derivative's terms divided by the element count of the mean).
         */
        static void expectGradient(String what, double[] ref, double terms, float[] got) {
            assertEquals(what + ": gradient length", ref.length, got.length);
            for (int i = 0; i < ref.length; i++) {
                assertEquals(what + ": gradient [" + i + "]", ref[i], got[i], 1e-5 * terms);
            }
        }

        // -------------------------------------------------------------- references

        /** Log-softmax in double of {@code n} values of {@code x} starting at {@code base} with stride {@code step}. */
        static double[] logSoftmax(float[] x, int base, int step, int n) {
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < n; c++) {
                max = Math.max(max, x[base + c * step]);
            }
            double sum = 0;
            for (int c = 0; c < n; c++) {
                sum += Math.exp(x[base + c * step] - max);
            }
            double lse = max + Math.log(sum);
            double[] lp = new double[n];
            for (int c = 0; c < n; c++) {
                lp[c] = x[base + c * step] - lse;
            }
            return lp;
        }

        /** Per-sample class-value loss of {@code [rows, classes]} logits for the classes {@code cls}. */
        static Ref classRef(float[] x, int rows, int classes, int[] cls, DoubleUnaryOperator value,
                            DoubleUnaryOperator magnitude) {
            Ref ref = new Ref(Shape.of(rows), classes);
            for (int r = 0; r < rows; r++) {
                double lp = logSoftmax(x, r * classes, 1, classes)[cls[r]];
                ref.set(r, value.applyAsDouble(lp), magnitude.applyAsDouble(lp));
            }
            return ref;
        }

        static Ref tripletRef(float[] a, float[] pos, float[] neg, float p, float margin, float[] weights) {
            Ref ref = new Ref(Shape.of(4), 5);
            for (int s = 0; s < 4; s++) {
                double dp = 0;
                double dn = 0;
                double powers = 0;
                for (int f = 0; f < 5; f++) {
                    int i = s * 5 + f;
                    double tp = Math.pow(Math.abs((double) (a[i] - pos[i]) + TRIPLET_EPS), p);
                    double tn = Math.pow(Math.abs((double) (a[i] - neg[i]) + TRIPLET_EPS), p);
                    dp += tp;
                    dn += tn;
                    powers += floor1(tp) + floor1(tn);
                }
                dp = Math.pow(dp, 1.0 / p);
                dn = Math.pow(dn, 1.0 / p);
                double w = weights != null && s < weights.length ? weights[s] : 1.0;
                ref.set(s, w * Math.max(0.0, dp - dn + margin), Math.abs(w) * (powers + dp + dn + margin));
            }
            return ref;
        }

        /** {@code max(1, |v|)}: the size of a transcendental term in the tolerance bound. */
        static double floor1(double v) {
            return Math.max(1.0, Math.abs(v));
        }

        /** Runs {@code loss} with every reduction and checks each result against {@code ref}. */
        static void expect(String what, ComputeBackend b, Ref ref, Function<Reduction, Tensor> loss) {
            for (Reduction r : Reduction.values()) {
                try (Tensor result = loss.apply(r)) {
                    assertNotNull(what + " " + r, result);
                    assertSame(what + " " + r + ": result backend", b, result.backend());
                    float[] got = result.toFloatArray();
                    if (r == Reduction.NONE) {
                        assertEquals(what + ": per-sample shape", ref.shape, result.shape());
                        for (int i = 0; i < got.length; i++) {
                            TrainParity.assertReduction(what + " [" + i + "]", ref.value[i], got[i], ref.terms,
                                    ref.magnitude[i]);
                        }
                    } else {
                        assertEquals(what + " " + r + ": shape", Shape.scalar(), result.shape());
                        int n = ref.value.length;
                        double v = 0;
                        double m = 0;
                        for (int i = 0; i < n; i++) {
                            v += ref.value[i];
                            m += ref.magnitude[i];
                        }
                        if (r == Reduction.MEAN) {
                            v /= n;
                            m /= n;
                        }
                        TrainParity.assertReduction(what + " " + r, v, got[0], n * ref.terms, m);
                    }
                }
            }
        }

        /**
         * Runs {@code loss}, a {@code batchmean} loss of {@code ref}'s elements, and checks it against the sum of the
         * reference values divided by {@code batch}.
         */
        static void expectBatchMean(String what, ComputeBackend b, Ref ref, int batch, Supplier<Tensor> loss) {
            try (Tensor result = loss.get()) {
                assertSame(what + ": result backend", b, result.backend());
                assertEquals(what + ": shape", Shape.scalar(), result.shape());
                double v = 0;
                double m = 0;
                for (int i = 0; i < ref.value.length; i++) {
                    v += ref.value[i];
                    m += ref.magnitude[i];
                }
                TrainParity.assertReduction(what, v / batch, result.toFloatArray()[0], ref.value.length * ref.terms,
                        m / batch);
            }
        }
    }

    /**
     * Double-precision per-sample reference: values, the absolute term sizes bounding them, and the terms per
     * sample.
     */
    static final class Ref {
        final Shape shape;
        final int terms;
        final double[] value;
        final double[] magnitude;

        Ref(Shape shape, int terms) {
            this.shape = shape;
            this.terms = terms;
            this.value = new double[(int) shape.numElements()];
            this.magnitude = new double[value.length];
        }

        void set(int i, double v, double m) {
            value[i] = v;
            magnitude[i] = m;
        }
    }
}
