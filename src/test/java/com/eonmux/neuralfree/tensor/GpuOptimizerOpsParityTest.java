package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps.AdagradParams;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps.AdamParams;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps.LambParams;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps.RmspropParams;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps.SgdParams;
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
import java.util.OptionalInt;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link GpuOptimizerOps} against double-precision references of the CPU optimizer loops (optimizer-step and reduction
 * tolerance class), clipping and unscaling against the CPU rules, view handling, and the Adam CPU path's offset-correct
 * write-back.
 */
@RunWith(Enclosed.class)
public class GpuOptimizerOpsParityTest {

    private static final float FLT_MIN = TrainParity.FLT_MIN;

    /**
     * {@code |gpu - ref| <= 1e-5 * sqrt(k) * mag + k * FLT_MIN} per element, the reduction class of the
     * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.
     */
    static void assertNear(String what, double[] ref, double[] mag, float[] gpu, int k) {
        assertEquals(what + ": length", ref.length, gpu.length);
        for (int i = 0; i < ref.length; i++) {
            double bound = 1e-5 * Math.sqrt(k) * mag[i] + k * (double) FLT_MIN;
            if (!(Math.abs(gpu[i] - ref[i]) <= bound)) {
                TrainParity.assertReduction(what + " element " + i, ref[i], gpu[i], k, mag[i]);
            }
        }
    }

    /** Paths that need no GPU: the CPU-path signal of every op and the CPU write-back. */
    public static class HostPath {
        private ComputeBackend host;

        @Before
        public void setUp() {
            host = TrainParity.hostBackend();
        }

        @Test
        public void everyOpDeclinesWithoutTrainKernels() {
            float[] pData = TrainParity.uniform(1, 8, -1f, 1f);
            Tensor p = Tensor.fromArray(host, pData);
            Tensor g = Tensor.fromArray(host, TrainParity.uniform(2, 8, -1f, 1f));
            Tensor m = Tensor.fromArray(host, new float[8]);
            Tensor v = Tensor.fromArray(host, new float[8]);
            assertFalse(GpuOptimizerOps.sgdStep(p, g, null, new SgdParams(0.1f, 0f, 0f, false)));
            assertFalse(GpuOptimizerOps.adamStep(p, g, m, v, AdamParams.adam(0.1f, 0.9f, 0.999f, 1e-8f, 0f, 0.001f)));
            assertFalse(GpuOptimizerOps.rmspropStep(p, g, v, null, null,
                    new RmspropParams(0.01f, 0.9f, 1e-8f, 0f, 0f, false)));
            assertFalse(GpuOptimizerOps.adagradStep(p, g, v, new AdagradParams(0.01f, 1e-10f, 0f)));
            assertFalse(GpuOptimizerOps.lambStep(p, g, m, v, new LambParams(0.01f, 0.9f, 0.999f, 1e-6f, 0f, 1f, 1f)));
            assertFalse(GpuOptimizerOps.clipGradNorm(Collections.singletonList(g), 0.5f));
            assertFalse(GpuOptimizerOps.clipGradValue(Collections.singletonList(g), 0.5f));
            assertFalse(GpuOptimizerOps.unscaleAndCheck(Collections.singletonList(g), 0.5f).isPresent());
            try (Parameter view = new Parameter(p.detach())) {
                assertFalse(GpuOptimizerOps.shouldUseGpu(view));
            }
            TrainParity.assertBitIdentical("parameter untouched", pData, p.toFloatArray());

            Tensor base = Tensor.fromArray(host, new float[20]);
            List<Tensor> overlapping = Arrays.asList(base.slice(0, 0, 12), base.slice(0, 6, 18));
            assertFalse("the CPU path takes overlapping views", GpuOptimizerOps.clipGradNorm(overlapping, 1f));
        }

        @Test
        public void emptyGradientListsNeedNoWork() {
            assertTrue(GpuOptimizerOps.clipGradNorm(Collections.emptyList(), 1f));
            assertTrue(GpuOptimizerOps.clipGradValue(Collections.emptyList(), 1f));
            assertEquals(OptionalInt.of(0), GpuOptimizerOps.unscaleAndCheck(Collections.emptyList(), 1f));
        }

        @Test
        public void adamCpuPathStoresIntoAnOffsetView() {
            float[] base = TrainParity.uniform(3, 40, -1f, 1f);
            float[] grad = TrainParity.uniform(4, 20, -1f, 1f);
            Tensor all = Tensor.fromArray(host, base).reshape(Shape.of(4, 10));
            Parameter param = new Parameter(all.slice(0, 1, 3));
            param.setGrad(Tensor.fromArray(host, grad).reshape(Shape.of(2, 10)));
            float lr = 0.01f;
            float beta1 = 0.9f;
            float beta2 = 0.999f;
            float eps = 1e-8f;
            float wd = 0.01f;
            Adam adam = new Adam(Collections.singletonList(param), lr, beta1, beta2, eps, wd);
            float[] stateM;
            try {
                adam.step();
                Map<String, Object> state = adam.stateDict();
                try {
                    stateM = ((Tensor) state.get("moment_0")).toFloatArray();
                } finally {
                    Optimizer.closeStateDict(state);
                }
            } finally {
                adam.close();
            }
            float bc1 = 1.0f - (float) Math.pow(beta1, 1);
            float bc2 = 1.0f - (float) Math.pow(beta2, 1);
            float stepSize = lr / bc1;
            float[] expected = base.clone();
            float[] expectedM = new float[20];
            for (int i = 0; i < 20; i++) {
                float p = base[10 + i];
                float g = grad[i] + wd * p;
                float m = beta1 * 0.0f + (1.0f - beta1) * g;
                float v = beta2 * 0.0f + (1.0f - beta2) * g * g;
                expectedM[i] = m;
                expected[10 + i] = p - stepSize * m / ((float) Math.sqrt(v / bc2) + eps);
            }
            TrainParity.assertBitIdentical("parameter rows and their neighbours", expected, all.toFloatArray());
            TrainParity.assertBitIdentical("first moment", expectedM, stateM);
        }

        @Test
        public void storeFloatsWritesAStridedView() {
            float[] base = TrainParity.uniform(5, 24, -1f, 1f);
            Tensor all = Tensor.fromArray(host, base).reshape(Shape.of(4, 6));
            Tensor view = all.transpose().slice(0, 1, 4);
            float[] values = TrainParity.uniform(6, 12, 2f, 3f);
            GpuOptimizerOps.storeFloats(view, values);
            float[] expected = base.clone();
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    expected[j * 6 + 1 + i] = values[i * 4 + j];
                }
            }
            TrainParity.assertBitIdentical("strided store", expected, all.toFloatArray());
        }

        @Test
        public void viewIdentityAndSharedElements() {
            Tensor all = Tensor.fromArray(host, new float[24]).reshape(Shape.of(4, 6));
            assertTrue(GpuOptimizerOps.sameView(all, all.reshape(Shape.of(24))));
            assertTrue(GpuOptimizerOps.sameView(all, all.slice(0, 0, 4)));
            assertFalse(GpuOptimizerOps.sameView(all, all.transpose()));
            assertTrue(GpuOptimizerOps.sharesElements(all, all.transpose()));
            assertTrue(GpuOptimizerOps.sharesElements(all.slice(0, 0, 2), all.slice(0, 1, 3)));
            assertFalse(GpuOptimizerOps.sharesElements(all.slice(0, 0, 2), all.slice(0, 2, 4)));
            assertFalse(GpuOptimizerOps.sharesElements(all.slice(1, 0, 3), all.slice(1, 3, 6)));
            assertTrue(GpuOptimizerOps.sharesElements(all.slice(1, 0, 4), all.slice(1, 3, 6)));
            assertFalse(GpuOptimizerOps.sharesElements(all, Tensor.fromArray(host, new float[24])));
        }

        @Test(expected = IllegalArgumentException.class)
        public void storeFloatsRejectsBroadcastViews() {
            Tensor row = Tensor.fromArray(host, new float[]{1f, 2f, 3f}).reshape(Shape.of(1, 3));
            GpuOptimizerOps.storeFloats(row.expand(Shape.of(4, 3)), new float[12]);
        }

        @Test(expected = IllegalArgumentException.class)
        public void storeFloatsRejectsAWrongLength() {
            GpuOptimizerOps.storeFloats(Tensor.fromArray(host, new float[5]), new float[4]);
        }
    }

    /** Every op on each GPU API with the train kernels available. */
    @RunWith(Parameterized.class)
    public static class Device {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;
        private ComputeBackend gpu;
        private final List<Tensor> owned = new ArrayList<>();

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
            try {
                for (Tensor t : owned) {
                    t.close();
                }
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }

        private Tensor dev(float[] data, int... dims) {
            Tensor t = Tensor.fromArray(gpu, data);
            owned.add(t);
            return dims.length == 0 ? t : t.reshape(Shape.of(dims));
        }

        @Test
        public void sgdStepMatchesTheCpuLoop() {
            int n = 1531;
            float[][] configs = {{0f, 0f, 0f}, {0.9f, 0.01f, 0f}, {0.9f, 0.01f, 1f}};
            for (float[] c : configs) {
                float lr = 0.05f;
                float mom = c[0];
                float wd = c[1];
                boolean nesterov = c[2] != 0f;
                float[] p = TrainParity.uniform(10, n, -1f, 1f);
                float[] g = TrainParity.uniform(11, n, -1f, 1f);
                float[] v = TrainParity.uniform(12, n, -0.5f, 0.5f);
                Tensor pt = dev(p);
                Tensor vt = dev(v);
                assertTrue(GpuOptimizerOps.sgdStep(pt, dev(g), mom > 0 ? vt : null,
                        new SgdParams(lr, mom, wd, nesterov)));
                double[] pRef = new double[n];
                double[] pMag = new double[n];
                double[] vRef = new double[n];
                double[] vMag = new double[n];
                for (int i = 0; i < n; i++) {
                    double gi = g[i];
                    double magG = Math.abs(gi);
                    if (wd != 0f) {
                        gi += (double) wd * p[i];
                        magG += Math.abs((double) wd * p[i]);
                    }
                    double upd = gi;
                    double magU = magG;
                    vRef[i] = v[i];
                    vMag[i] = Math.abs(v[i]);
                    if (mom > 0) {
                        vRef[i] = mom * (double) v[i] + gi;
                        vMag[i] = Math.abs(mom * (double) v[i]) + magG;
                        upd = nesterov ? mom * vRef[i] + gi : vRef[i];
                        magU = nesterov ? mom * vMag[i] + magG : vMag[i];
                    }
                    pRef[i] = p[i] - lr * upd;
                    pMag[i] = Math.abs(p[i]) + lr * magU + Math.abs(lr * upd);
                }
                String what = "sgd " + Arrays.toString(c);
                assertNear(what + " p", pRef, pMag, pt.toFloatArray(), 1);
                assertNear(what + " v", vRef, vMag, vt.toFloatArray(), 1);
            }
        }

        @Test
        public void adamAndAdamWStepsMatchTheCpuLoops() {
            int n = 777;
            float lr = 0.01f;
            float b1 = 0.9f;
            float b2 = 0.999f;
            float eps = 1e-8f;
            float wd = 0.02f;
            float bc1 = 1.0f - (float) Math.pow(b1, 3);
            float bc2 = 1.0f - (float) Math.pow(b2, 3);
            for (boolean decoupled : new boolean[]{false, true}) {
                float[] p = TrainParity.uniform(20, n, -1f, 1f);
                float[] g = TrainParity.uniform(21, n, -1f, 1f);
                float[] m = TrainParity.uniform(22, n, -0.1f, 0.1f);
                float[] v = TrainParity.uniform(23, n, 0.001f, 0.01f);
                Tensor pt = dev(p);
                Tensor mt = dev(m);
                Tensor vt = dev(v);
                float stepLr = decoupled ? lr : lr / bc1;
                float stepBc1 = decoupled ? bc1 : 1.0f;
                AdamParams h = decoupled ? AdamParams.adamW(lr, b1, b2, eps, wd, bc1, bc2)
                        : AdamParams.adam(stepLr, b1, b2, eps, wd, bc2);
                assertTrue(GpuOptimizerOps.adamStep(pt, dev(g), mt, vt, h));
                double[][] ref = new double[6][n];
                for (int i = 0; i < n; i++) {
                    double pi = p[i];
                    double magP = Math.abs(pi);
                    double gi = g[i];
                    double magG = Math.abs(gi);
                    if (decoupled) {
                        pi -= (double) lr * wd * p[i];
                        magP += Math.abs((double) lr * wd * p[i]);
                    } else {
                        gi += (double) wd * p[i];
                        magG += Math.abs((double) wd * p[i]);
                    }
                    double mi = b1 * (double) m[i] + (1.0 - b1) * gi;
                    double magM = Math.abs(b1 * (double) m[i]) + (1.0 - b1) * magG;
                    double vi = b2 * (double) v[i] + (1.0 - b2) * gi * gi;
                    double magV = b2 * (double) v[i] + (1.0 - b2) * magG * magG;
                    double step = (double) stepLr / stepBc1;
                    double den = Math.sqrt(vi / bc2) + eps;
                    double q = step * mi / den;
                    ref[0][i] = pi - q;
                    ref[1][i] = magP + step * magM / den + Math.abs(q);
                    ref[2][i] = mi;
                    ref[3][i] = magM;
                    ref[4][i] = vi;
                    ref[5][i] = magV;
                }
                String what = decoupled ? "adamw" : "adam";
                assertNear(what + " p", ref[0], ref[1], pt.toFloatArray(), 1);
                assertNear(what + " m", ref[2], ref[3], mt.toFloatArray(), 1);
                assertNear(what + " v", ref[4], ref[5], vt.toFloatArray(), 1);
            }
        }

        @Test
        public void adamOptimizerOnAnOffsetParameterMatchesItsCpuPath() {
            ComputeBackend host = TrainParity.hostBackend();
            float[] base = TrainParity.uniform(30, 60, -1f, 1f);
            Tensor hostAll = Tensor.fromArray(host, base).reshape(Shape.of(6, 10));
            Tensor devAll = dev(base, 6, 10);
            Parameter hostParam = new Parameter(hostAll.slice(0, 2, 5));
            Parameter devParam = new Parameter(devAll.slice(0, 2, 5));
            Adam hostAdam = new Adam(Collections.singletonList(hostParam), 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);
            Adam devAdam = new Adam(Collections.singletonList(devParam), 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);
            double[] travelled = new double[30];
            try {
                for (int step = 0; step < 3; step++) {
                    float[] grad = TrainParity.uniform(31 + step, 30, -1f, 1f);
                    hostParam.setGrad(Tensor.fromArray(host, grad).reshape(Shape.of(3, 10)));
                    devParam.setGrad(dev(grad, 3, 10));
                    float[] before = hostParam.tensor().toFloatArray();
                    hostAdam.step();
                    devAdam.step();
                    float[] after = hostParam.tensor().toFloatArray();
                    for (int i = 0; i < 30; i++) {
                        travelled[i] += Math.abs((double) after[i] - before[i]);
                    }
                }
            } finally {
                hostAdam.close();
                devAdam.close();
                hostParam.close();
                devParam.close();
            }
            float[] hostOut = hostAll.toFloatArray();
            float[] devOut = devAll.toFloatArray();
            for (int i = 0; i < 60; i++) {
                boolean inParam = i >= 20 && i < 50;
                if (!inParam) {
                    assertEquals("element " + i + " outside the parameter", Float.floatToRawIntBits(base[i]),
                            Float.floatToRawIntBits(devOut[i]));
                    continue;
                }
                double bound = 1e-5 * (Math.abs(hostOut[i]) + travelled[i - 20]) + FLT_MIN;
                if (!(Math.abs((double) devOut[i] - hostOut[i]) <= bound)) {
                    fail("element " + i + ": cpu " + hostOut[i] + ", gpu " + devOut[i] + ", bound " + bound);
                }
            }
        }

        @Test
        public void rmspropStepMatchesTheCpuLoops() {
            int n = 1029;
            float lr = 0.01f;
            float alpha = 0.9f;
            float eps = 1e-8f;
            for (boolean centered : new boolean[]{false, true}) {
                float mom = centered ? 0.9f : 0f;
                float wd = centered ? 0.01f : 0f;
                float[] p = TrainParity.uniform(40, n, -1f, 1f);
                float[] g = TrainParity.uniform(41, n, -1f, 1f);
                float[] sq = TrainParity.uniform(42, n, 0.5f, 1f);
                float[] buf = TrainParity.uniform(43, n, -0.1f, 0.1f);
                float[] gavg = TrainParity.uniform(44, n, -0.1f, 0.1f);
                Tensor pt = dev(p);
                Tensor sqt = dev(sq);
                Tensor buft = dev(buf);
                Tensor gavgt = dev(gavg);
                assertTrue(GpuOptimizerOps.rmspropStep(pt, dev(g), sqt, centered ? buft : null,
                        centered ? gavgt : null, new RmspropParams(lr, alpha, eps, wd, mom, centered)));
                double[][] ref = new double[8][n];
                for (int i = 0; i < n; i++) {
                    double gi = g[i] + (double) wd * p[i];
                    double magG = Math.abs(g[i]) + Math.abs((double) wd * p[i]);
                    double si = alpha * (double) sq[i] + (1.0 - alpha) * gi * gi;
                    double magS = alpha * (double) sq[i] + (1.0 - alpha) * magG * magG;
                    double den = si;
                    double magDen = magS;
                    ref[6][i] = gavg[i];
                    ref[7][i] = Math.abs(gavg[i]);
                    if (centered) {
                        double ai = alpha * (double) gavg[i] + (1.0 - alpha) * gi;
                        double magA = Math.abs(alpha * (double) gavg[i]) + (1.0 - alpha) * magG;
                        den = si - ai * ai;
                        magDen = magS + 2 * Math.abs(ai) * magA + ai * ai;
                        ref[6][i] = ai;
                        ref[7][i] = magA;
                    }
                    double root = Math.sqrt(den) + eps;
                    double upd = gi / root;
                    double magU = magG / root + Math.abs(upd) * (1 + magDen / (2 * den));
                    double bi = buf[i];
                    double magB = Math.abs(bi);
                    if (mom > 0) {
                        bi = mom * (double) buf[i] + upd;
                        magB = Math.abs(mom * (double) buf[i]) + magU;
                    }
                    double delta = mom > 0 ? bi : upd;
                    ref[0][i] = p[i] - lr * delta;
                    ref[1][i] = Math.abs(p[i]) + lr * (mom > 0 ? magB : magU) + Math.abs(lr * delta);
                    ref[2][i] = si;
                    ref[3][i] = magS;
                    ref[4][i] = bi;
                    ref[5][i] = magB;
                }
                String what = centered ? "rmsprop centered momentum" : "rmsprop";
                assertNear(what + " p", ref[0], ref[1], pt.toFloatArray(), 1);
                assertNear(what + " sq", ref[2], ref[3], sqt.toFloatArray(), 1);
                assertNear(what + " buf", ref[4], ref[5], buft.toFloatArray(), 1);
                assertNear(what + " gavg", ref[6], ref[7], gavgt.toFloatArray(), 1);
            }
        }

        @Test
        public void adagradStepMatchesTheCpuLoop() {
            int n = 999;
            float lr = 0.1f;
            float eps = 1e-10f;
            float wd = 0.01f;
            float[] p = TrainParity.uniform(50, n, -1f, 1f);
            float[] g = TrainParity.uniform(51, n, -1f, 1f);
            float[] acc = TrainParity.uniform(52, n, 0.1f, 1f);
            Tensor pt = dev(p);
            Tensor acct = dev(acc);
            assertTrue(GpuOptimizerOps.adagradStep(pt, dev(g), acct, new AdagradParams(lr, eps, wd)));
            double[][] ref = new double[4][n];
            for (int i = 0; i < n; i++) {
                double gi = g[i] + (double) wd * p[i];
                double magG = Math.abs(g[i]) + Math.abs((double) wd * p[i]);
                double ai = acc[i] + gi * gi;
                double den = Math.sqrt(ai) + eps;
                double q = lr * gi / den;
                ref[0][i] = p[i] - q;
                ref[1][i] = Math.abs(p[i]) + lr * magG / den + Math.abs(q);
                ref[2][i] = ai;
                ref[3][i] = acc[i] + magG * magG;
            }
            assertNear("adagrad p", ref[0], ref[1], pt.toFloatArray(), 1);
            assertNear("adagrad acc", ref[2], ref[3], acct.toFloatArray(), 1);
        }

        @Test
        public void lambStepMatchesTheCpuLoopForEveryReductionShape() {
            float lr = 0.01f;
            float b1 = 0.9f;
            float b2 = 0.999f;
            float eps = 1e-6f;
            float wd = 0.01f;
            float bc1 = 1.0f - (float) Math.pow(b1, 2);
            float bc2 = 1.0f - (float) Math.pow(b2, 2);
            for (int n : new int[]{50, 5000, 100_000, 600_000}) {
                float[] p = TrainParity.uniform(60, n, -1f, 1f);
                float[] g = TrainParity.uniform(61, n, -1f, 1f);
                float[] m = TrainParity.uniform(62, n, -0.1f, 0.1f);
                float[] v = TrainParity.uniform(63, n, 0.001f, 0.01f);
                Tensor pt = dev(p);
                Tensor mt = dev(m);
                Tensor vt = dev(v);
                assertTrue(GpuOptimizerOps.lambStep(pt, dev(g), mt, vt, new LambParams(lr, b1, b2, eps, wd, bc1, bc2)));
                double[] r = new double[n];
                double[] magR = new double[n];
                double[][] ref = new double[6][n];
                double sumP = 0;
                double sumR = 0;
                for (int i = 0; i < n; i++) {
                    double mi = b1 * (double) m[i] + (1.0 - b1) * g[i];
                    double magM = Math.abs(b1 * (double) m[i]) + (1.0 - b1) * Math.abs(g[i]);
                    double vi = b2 * (double) v[i] + (1.0 - b2) * g[i] * (double) g[i];
                    double den = Math.sqrt(vi / bc2) + eps;
                    r[i] = mi / bc1 / den + (double) wd * p[i];
                    magR[i] = magM / bc1 / den + Math.abs(mi / bc1 / den) + Math.abs((double) wd * p[i]);
                    sumP += (double) p[i] * p[i];
                    sumR += r[i] * r[i];
                    ref[2][i] = mi;
                    ref[3][i] = magM;
                    ref[4][i] = vi;
                    ref[5][i] = b2 * (double) v[i] + (1.0 - b2) * g[i] * (double) g[i];
                }
                double wn = Math.sqrt(sumP);
                double un = Math.sqrt(sumR);
                double trust = wn > 0 && un > 0 ? wn / (un + eps) : 1.0;
                for (int i = 0; i < n; i++) {
                    ref[0][i] = p[i] - lr * trust * r[i];
                    ref[1][i] = Math.abs(p[i]) + lr * trust * magR[i];
                }
                assertNear("lamb n=" + n + " p", ref[0], ref[1], pt.toFloatArray(), n);
                assertNear("lamb n=" + n + " m", ref[2], ref[3], mt.toFloatArray(), 1);
                assertNear("lamb n=" + n + " v", ref[4], ref[5], vt.toFloatArray(), 1);
            }
        }

        @Test
        public void clipGradNormCountsListedDuplicatesAndScalesEachViewOnce() {
            float[] a = TrainParity.uniform(70, 300, -1f, 1f);
            float[] b = TrainParity.uniform(71, 70, -1f, 1f);
            float[] base = TrainParity.uniform(72, 600, -1f, 1f);
            Tensor at = dev(a);
            Tensor bt = dev(b);
            Tensor baseT = dev(base, 20, 30);
            Tensor view = baseT.transpose().slice(0, 2, 25);
            float[] viewData = view.toFloatArray();
            double sum = 0;
            for (float x : a) {
                sum += 2.0 * x * x;
            }
            for (float x : b) {
                sum += (double) x * x;
            }
            for (float x : viewData) {
                sum += (double) x * x;
            }
            double total = Math.sqrt(sum);
            float maxNorm = (float) (total * 0.5);
            double coef = maxNorm / (total + 1e-6);
            int k = 2 * a.length + b.length + viewData.length;
            assertTrue(GpuOptimizerOps.clipGradNorm(Arrays.asList(at, bt, at, view), maxNorm));
            assertScaled("a", a, coef, at.toFloatArray(), k);
            assertScaled("b", b, coef, bt.toFloatArray(), k);
            float[] out = baseT.toFloatArray();
            double[] ref = new double[600];
            double[] mag = new double[600];
            for (int r = 0; r < 20; r++) {
                for (int c = 0; c < 30; c++) {
                    int i = r * 30 + c;
                    boolean inView = c >= 2 && c < 25;
                    ref[i] = inView ? base[i] * coef : base[i];
                    mag[i] = inView ? Math.abs(ref[i]) : 0;
                    if (!inView) {
                        assertEquals("untouched element " + i, Float.floatToRawIntBits(base[i]),
                                Float.floatToRawIntBits(out[i]));
                    }
                }
            }
            assertNear("strided view", ref, mag, out, k);
        }

        private static void assertScaled(String what, float[] in, double coef, float[] gpu, int k) {
            double[] ref = new double[in.length];
            double[] mag = new double[in.length];
            for (int i = 0; i < in.length; i++) {
                ref[i] = in[i] * coef;
                mag[i] = Math.abs(ref[i]);
            }
            assertNear(what, ref, mag, gpu, k);
        }

        @Test
        public void clipGradNormBelowTheThresholdLeavesGradientsUntouched() {
            float[] a = TrainParity.uniform(73, 5000, -1f, 1f);
            Tensor at = dev(a);
            assertTrue(GpuOptimizerOps.clipGradNorm(Collections.singletonList(at), 1e6f));
            TrainParity.assertBitIdentical("unclipped", a, at.toFloatArray());
        }

        @Test
        public void clipGradValueIsJavaClampOncePerView() {
            float[] edge = {Float.NaN, -0.0f, 0.0f, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 1e-40f,
                    -1e-40f, 0.5f, -0.5f, 0.75f, -3e38f, 0.25f};
            float[] x = TrainParity.uniform(80, 400, -2f, 2f);
            System.arraycopy(edge, 0, x, 0, edge.length);
            float[] y = TrainParity.uniform(81, 48, -2f, 2f);
            Tensor xt = dev(x);
            Tensor yt = dev(y, 6, 8);
            float max = 0.5f;
            assertTrue(GpuOptimizerOps.clipGradValue(Arrays.asList(xt, xt, yt.transpose()), max));
            TrainParity.assertBitIdentical("clamp", clamp(x, max), xt.toFloatArray());
            TrainParity.assertBitIdentical("clamp strided", clamp(y, max), yt.toFloatArray());
        }

        private static float[] clamp(float[] in, float max) {
            float[] out = new float[in.length];
            for (int i = 0; i < in.length; i++) {
                out[i] = Math.max(-max, Math.min(max, in[i]));
            }
            return out;
        }

        @Test
        public void unscaleAndCheckScalesEachViewOnceAndFlagsNonFiniteResults() {
            float inv = 1.0f / 65536f;
            float[] a = TrainParity.uniform(90, 1000, -1e4f, 1e4f);
            float[] b = TrainParity.uniform(91, 60, -1e4f, 1e4f);
            Tensor at = dev(a);
            Tensor bt = dev(b, 6, 10);
            OptionalInt flag = GpuOptimizerOps.unscaleAndCheck(Arrays.asList(at, at, bt.transpose()), inv);
            assertEquals(OptionalInt.of(0), flag);
            TrainParity.assertSingleRounding("unscaled", scale(a, inv), at.toFloatArray());
            TrainParity.assertSingleRounding("unscaled strided", scale(b, inv), bt.toFloatArray());

            float[] big = {1f, 3e38f, -2f};
            Tensor bigT = dev(big);
            assertEquals(OptionalInt.of(1), GpuOptimizerOps.unscaleAndCheck(Collections.singletonList(bigT), 4f));
            TrainParity.assertBitIdentical("overflowed", scale(big, 4f), bigT.toFloatArray());

            Tensor nan = dev(new float[]{0.5f, Float.NaN});
            assertEquals(OptionalInt.of(1), GpuOptimizerOps.unscaleAndCheck(Collections.singletonList(nan), 0.5f));
        }

        private static float[] scale(float[] in, float s) {
            float[] out = new float[in.length];
            for (int i = 0; i < in.length; i++) {
                out[i] = in[i] * s;
            }
            return out;
        }

        @Test
        public void overlappingUnequalGradientViewsAreRejected() {
            Tensor base = dev(TrainParity.uniform(100, 20, -1f, 1f));
            List<Tensor> overlapping = Arrays.asList(base.slice(0, 0, 12), base.slice(0, 6, 18));
            try {
                GpuOptimizerOps.clipGradNorm(overlapping, 1f);
                fail("clipGradNorm accepted overlapping views");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains("shares elements"));
            }
            try {
                GpuOptimizerOps.unscaleAndCheck(overlapping, 0.5f);
                fail("unscaleAndCheck accepted overlapping views");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains("shares elements"));
            }
        }

        @Test
        public void interleavedDisjointViewsOfOneBufferAreDistinct() {
            float[] data = TrainParity.uniform(101, 24, -2f, 2f);
            Tensor all = dev(data, 4, 6);
            List<Tensor> halves = Arrays.asList(all.slice(1, 0, 3), all.slice(1, 3, 6));
            assertTrue(GpuOptimizerOps.clipGradValue(halves, 0.5f));
            TrainParity.assertBitIdentical("both halves clamped", clamp(data, 0.5f), all.toFloatArray());
        }

        @Test
        public void stepsOnStridedAndAliasedOperandsWriteOnlyTheirElements() {
            float lr = 0.25f;
            float[] base = TrainParity.uniform(110, 72, -1f, 1f);
            float[] g = TrainParity.uniform(111, 48, -1f, 1f);
            Tensor all = dev(base, 8, 9);
            Tensor param = all.transpose().slice(0, 1, 7);
            assertTrue(GpuOptimizerOps.sgdStep(param, dev(g, 6, 8), null, new SgdParams(lr, 0f, 0f, false)));
            float[] out = all.toFloatArray();
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 9; c++) {
                    int i = r * 9 + c;
                    float expected = c >= 1 && c < 7 ? base[i] - lr * g[(c - 1) * 8 + r] : base[i];
                    assertEquals("element " + i, Float.floatToRawIntBits(expected), Float.floatToRawIntBits(out[i]));
                }
            }

            float[] p = TrainParity.uniform(112, 300, -1f, 1f);
            Tensor pt = dev(p);
            assertTrue(GpuOptimizerOps.sgdStep(pt, pt, null, new SgdParams(lr, 0f, 0f, false)));
            float[] expected = new float[p.length];
            for (int i = 0; i < p.length; i++) {
                expected[i] = p[i] - lr * p[i];
            }
            TrainParity.assertSingleRounding("gradient aliasing the parameter", expected, pt.toFloatArray());
        }

        @Test(expected = IllegalArgumentException.class)
        public void overlappingStateIsRejected() {
            Tensor p = dev(new float[16]);
            Tensor g = dev(new float[16]);
            Tensor state = dev(new float[16]);
            GpuOptimizerOps.adamStep(p, g, state, state, AdamParams.adam(0.1f, 0.9f, 0.999f, 1e-8f, 0f, 0.001f));
        }
    }
}
