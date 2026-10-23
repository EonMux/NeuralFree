package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.ml.GaussianProcess.GPPrediction;
import com.eonmux.neuralfree.ml.GaussianProcess.KernelType;
import com.eonmux.neuralfree.ml.gpu.MlGpu;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link GaussianProcess} on its CPU reference and on each GPU API. The device kernel matrices ({@code ML_GP_Kernel},
 * cross form without noise and training form with noise on the diagonal, offset views) are compared with the host twin
 * {@link MlGpu#gpKernel(float[], int, float[], int, int, int, float, float)}: LINEAR in the reduction class, the others
 * in the transcendental class ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}). On integer data the
 * LINEAR kernel is exact on every path, so both paths factor the same matrix: variances are bit-identical and means
 * agree within twice the GEMM-class bound. For RBF the means agree within the propagation of the kernel tolerance
 * through {@code (K + noise I)^-1}.
 */
@RunWith(Enclosed.class)
public class GaussianProcessGpuParityTest {

    private static final float LS = 0.8f;
    private static final float VAR = 1.5f;
    private static final float NOISE = 0.25f;

    static float[][] uniformRows(long seed, int rows, int cols, float lo, float hi) {
        float[] flat = TrainParity.uniform(seed, rows * cols, lo, hi);
        float[][] x = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * cols, x[i], 0, cols);
        }
        return x;
    }

    static float[][] integerRows(long seed, int rows, int cols) {
        Random rng = new Random(seed);
        float[][] x = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                x[i][j] = rng.nextInt(7) - 3;
            }
        }
        return x;
    }

    /** {@code A^-1 y} for a symmetric positive definite row-major {@code n x n} matrix, in double. */
    static double[] solve(double[] a, double[] y, int n) {
        double[] l = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double s = a[i * n + j];
                for (int p = 0; p < j; p++) {
                    s -= l[i * n + p] * l[j * n + p];
                }
                l[i * n + j] = i == j ? Math.sqrt(s) : s / l[j * n + j];
            }
        }
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            double s = y[i];
            for (int j = 0; j < i; j++) {
                s -= l[i * n + j] * z[j];
            }
            z[i] = s / l[i * n + i];
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = z[i];
            for (int j = i + 1; j < n; j++) {
                s -= l[j * n + i] * x[j];
            }
            x[i] = s / l[i * n + i];
        }
        return x;
    }

    /** Paths that need no GPU: the CPU reference of the class. */
    public static class HostPath {
        private ComputeBackend host;

        @Before
        public void setUp() {
            host = TrainParity.hostBackend();
        }

        @Test
        public void stationaryKernelsInterpolateAndRevertToThePrior() {
            float[][] x = new float[10][1];
            float[] y = new float[10];
            for (int i = 0; i < 10; i++) {
                x[i][0] = i;
                y[i] = (float) Math.sin(i);
            }
            for (KernelType type : new KernelType[]{KernelType.RBF, KernelType.MATERN32, KernelType.MATERN52}) {
                try (GaussianProcess gp = new GaussianProcess(host, type, 1f, 1f, 1e-6f)) {
                    gp.fit(x, y);
                    GPPrediction p = gp.predict(x);
                    for (int i = 0; i < 10; i++) {
                        assertEquals(type + " mean at training point " + i, y[i], p.mean()[i], 1e-2);
                        assertTrue(type + " variance at training point " + i, p.variance()[i] < 1e-2);
                    }
                    GPPrediction far = gp.predict(new float[][]{{100f}});
                    assertEquals(type + " mean far away", 0f, far.mean()[0], 1e-6);
                    assertEquals(type + " variance far away", 1f, far.variance()[0], 1e-6);
                    assertEquals(type + " std dev far away", 1f, far.stdDev()[0], 1e-6);
                }
            }
        }

        @Test
        public void linearKernelExtrapolatesALine() {
            float[][] x = {{1f}, {2f}, {3f}, {4f}};
            float[] y = {2f, 4f, 6f, 8f};
            try (GaussianProcess gp = new GaussianProcess(host, KernelType.LINEAR, 1f, 1f, 1e-3f)) {
                gp.fit(x, y);
                assertEquals(20f, gp.predict(new float[][]{{10f}}).mean()[0], 1e-2);
            }
        }

        @Test
        public void nonPositiveDefiniteCovarianceThrows() {
            try (GaussianProcess gp = new GaussianProcess(host, KernelType.LINEAR, 1f, 1f, 0f)) {
                try {
                    gp.fit(new float[][]{{1f, 0f}, {2f, 0f}}, new float[]{1f, 2f});
                    fail("a singular covariance must throw");
                } catch (IllegalArgumentException expected) {
                    assertTrue(expected.getMessage().contains("not positive definite"));
                }
                try {
                    gp.predict(new float[][]{{1f, 0f}});
                    fail("a failed fit leaves the model unfitted");
                } catch (IllegalStateException expected) {
                    assertTrue(expected.getMessage().contains("not fitted"));
                }
            }
        }

        @Test
        public void edgeCasesAndValidation() {
            try (GaussianProcess gp = new GaussianProcess(host)) {
                gp.fit(new float[][]{{0f, 1f}, {1f, 0f}}, new float[]{1f, -1f});
                GPPrediction empty = gp.predict(new float[0][]);
                assertEquals(0, empty.mean().length);
                assertEquals(0, empty.variance().length);
                expectIllegalArgument(() -> gp.predict(new float[][]{{0f}}));
                expectIllegalArgument(() -> gp.predict(new float[][]{{0f, 1f}, {0f}}));
                expectIllegalArgument(() -> gp.predict(null));
                expectIllegalArgument(() -> gp.fit(new float[][]{{0f}}, new float[2]));
                expectIllegalArgument(() -> gp.fit(new float[0][], new float[0]));
            }
            expectIllegalArgument(() -> new GaussianProcess(host, KernelType.RBF, 0f, 1f, 0f));
            expectIllegalArgument(() -> new GaussianProcess(host, KernelType.RBF, 1f, 0f, 0f));
            expectIllegalArgument(() -> new GaussianProcess(host, KernelType.RBF, 1f, 1f, -1f));
            expectIllegalArgument(() -> new GaussianProcess(host, null, 1f, 1f, 0f));
            expectIllegalArgument(() -> new GaussianProcess(null));
        }

        @Test
        public void closedModelRejectsCalls() {
            GaussianProcess gp = new GaussianProcess(host);
            gp.fit(new float[][]{{0f}}, new float[]{1f});
            gp.close();
            gp.close();
            try {
                gp.predict(new float[][]{{0f}});
                fail("a closed model must throw");
            } catch (IllegalStateException expected) {
                assertTrue(expected.getMessage().contains("closed"));
            }
        }
    }

    /** The device path on each GPU API against the host twins and the CPU reference. */
    @RunWith(Parameterized.class)
    public static class Parity {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> apis() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;
        private ComputeBackend host;

        public Parity(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            session = TrainParity.open(api);
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            if (session != null) {
                session.close();
            }
        }

        @Test
        public void kernelMatricesMatchTheHostTwin() {
            int f = 5;
            float[] big = TrainParity.uniform(51, 12 * f, -1f, 1f);
            float[] other = TrainParity.uniform(52, 6 * f, -1f, 1f);
            TrainGpu g = session.gpu();
            assertNotNull(g);
            int[] types = {TrainOps.ML_GP_RBF, TrainOps.ML_GP_MATERN32, TrainOps.ML_GP_MATERN52, TrainOps.ML_GP_LINEAR};
            try (MlGpu.Scope scope = MlGpu.scope()) {
                Tensor all = scope.add(MlGpu.upload(session.backend(), big, 12, f));
                Tensor a = scope.add(all.slice(0, 3, 10));
                Tensor b = scope.add(MlGpu.upload(session.backend(), other, 6, f));
                for (int type : types) {
                    Tensor cross = scope.add(MlGpu.gpKernel(g, a, b, type, LS, VAR, NOISE, true));
                    compare(api + " type " + type + " cross", type, big, 3, 7, other, 0, 6, f, false,
                            MlGpu.readFloats(cross));
                    Tensor train = scope.add(MlGpu.gpKernel(g, a, a, type, LS, VAR, NOISE, true));
                    compare(api + " type " + type + " training", type, big, 3, 7, big, 3, 7, f, true,
                            MlGpu.readFloats(train));
                    Tensor plain = scope.add(MlGpu.gpKernel(g, a, a, type, LS, VAR, NOISE, false));
                    compare(api + " type " + type + " square without noise", type, big, 3, 7, big, 3, 7, f, false,
                            MlGpu.readFloats(plain));
                }
            }
        }

        @Test
        public void linearModelMatchesTheCpuPath() {
            int n = 20;
            float[][] x = integerRows(61, n, 2);
            float[] y = new float[n];
            for (int i = 0; i < n; i++) {
                y[i] = 2f * x[i][0] - x[i][1] + (i % 3) - 1f;
            }
            float[][] test = integerRows(62, 8, 2);
            double[] k64 = new double[n * n];
            double[] y64 = new double[n];
            for (int i = 0; i < n; i++) {
                y64[i] = y[i];
                for (int j = 0; j < n; j++) {
                    k64[i * n + j] = x[i][0] * (double) x[j][0] + x[i][1] * (double) x[j][1] + (i == j ? 0.5 : 0.0);
                }
            }
            double[] alpha64 = solve(k64, y64, n);
            try (GaussianProcess cpu = new GaussianProcess(host, KernelType.LINEAR, 1f, 1f, 0.5f);
                 GaussianProcess gpu = new GaussianProcess(session.backend(), KernelType.LINEAR, 1f, 1f, 0.5f)) {
                cpu.fit(x, y);
                gpu.fit(x, y);
                GPPrediction pc = cpu.predict(test);
                GPPrediction pg = gpu.predict(test);
                TrainParity.assertBitIdentical(api + " variance", pc.variance(), pg.variance());
                for (int i = 0; i < test.length; i++) {
                    double sumAbs = 0;
                    for (int j = 0; j < n; j++) {
                        double kv = test[i][0] * (double) x[j][0] + test[i][1] * (double) x[j][1];
                        sumAbs += Math.abs(kv * alpha64[j]);
                    }
                    double bound = 2.0 * (1e-5 * Math.sqrt(n) * 1.001 * sumAbs + n * (double) TrainParity.FLT_MIN)
                            + 2.0 * Math.ulp(Math.abs(pc.mean()[i]));
                    assertEquals(api + " mean " + i, pc.mean()[i], pg.mean()[i], bound);
                }
            }
        }

        @Test
        public void rbfModelMatchesTheCpuPathWithinTheKernelTolerance() {
            int n = 15;
            float[][] x = uniformRows(71, n, 3, -2f, 2f);
            float[] y = TrainParity.uniform(72, n, -1f, 1f);
            float[][] test = uniformRows(73, 6, 3, -2f, 2f);
            float noise = 0.5f;
            double[] k64 = new double[n * n];
            double[] y64 = new double[n];
            for (int i = 0; i < n; i++) {
                y64[i] = y[i];
                for (int j = 0; j < n; j++) {
                    k64[i * n + j] = rbf64(x[i], x[j]) + (i == j ? noise : 0.0);
                }
            }
            double[] alpha64 = solve(k64, y64, n);
            double alphaNorm = 0;
            for (double a : alpha64) {
                alphaNorm += a * a;
            }
            alphaNorm = Math.sqrt(alphaNorm);
            try (GaussianProcess cpu = new GaussianProcess(host, KernelType.RBF, 1f, 1f, noise);
                 GaussianProcess gpu = new GaussianProcess(session.backend(), KernelType.RBF, 1f, 1f, noise)) {
                cpu.fit(x, y);
                gpu.fit(x, y);
                GPPrediction pc = cpu.predict(test);
                GPPrediction pg = gpu.predict(test);
                // Kernel values differ by at most 2e-5 per element (both paths within 1e-5 of the true value):
                // |dK|_2 <= 2e-5 n, |(K + noise I)^-1|_2 <= 1 / noise, |k*|_2 <= sqrt(n), |dk*|_2 <= 2e-5 sqrt(n).
                double dk = 2e-5;
                double meanBound = 2.0 * (dk * Math.sqrt(n) * alphaNorm
                        + Math.sqrt(n) * (dk * n / noise) * alphaNorm) + 1e-5;
                double varBound = 2.0 * (dk + 2.0 * Math.sqrt(n) * dk * Math.sqrt(n) / noise
                        + n * (dk * n / noise) / noise) + 1e-5;
                for (int i = 0; i < test.length; i++) {
                    assertEquals(api + " mean " + i, pc.mean()[i], pg.mean()[i], meanBound);
                    assertEquals(api + " variance " + i, pc.variance()[i], pg.variance()[i], varBound);
                }
            }
        }

        private static double rbf64(float[] a, float[] b) {
            double sq = 0;
            for (int j = 0; j < a.length; j++) {
                double d = (double) a[j] - b[j];
                sq += d * d;
            }
            return Math.exp(-sq / 2.0);
        }

        /** Device {@code k[i*n + j]} against the host twin for rows from {@code ai0} of a and {@code bi0} of b. */
        private static void compare(String what, int type, float[] a, int ai0, int m, float[] b, int bi0, int n, int f,
                                    boolean noiseOnDiagonal, float[] k) {
            assertEquals(what + ": length", m * n, k.length);
            float[] ref = new float[m * n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    float v = MlGpu.gpKernel(a, (ai0 + i) * f, b, (bi0 + j) * f, f, type, LS, VAR);
                    ref[i * n + j] = noiseOnDiagonal && m == n && i == j ? v + NOISE : v;
                }
            }
            if (type != TrainOps.ML_GP_LINEAR) {
                TrainParity.assertTranscendental(what, ref, k);
                return;
            }
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double dot = 0;
                    double abs = 0;
                    for (int q = 0; q < f; q++) {
                        double p = (double) a[(ai0 + i) * f + q] * b[(bi0 + j) * f + q];
                        dot += p;
                        abs += Math.abs(p);
                    }
                    boolean diag = noiseOnDiagonal && m == n && i == j;
                    double ref64 = VAR * dot + (diag ? NOISE : 0.0);
                    TrainParity.assertReduction(what + " " + i + "," + j, ref64, k[i * n + j], f,
                            VAR * abs + (diag ? NOISE : 0.0));
                }
            }
        }
    }

    static void expectIllegalArgument(Runnable r) {
        try {
            r.run();
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage() != null);
        }
    }
}
