package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.ml.PCA.PCAResult;
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
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link PCA} on its CPU reference and on each GPU API. Both fits are checked against a double-precision reference of
 * the data: the mean in the reduction class, every fitted component as an eigenpair of the reference covariance within
 * the covariance's reduction-class error, and {@link PCA#transform} in the GEMM class ({@linkplain
 * com.eonmux.neuralfree.tensor.gpu numerical contract}). The two paths agree on the explained variance within twice
 * that error (Weyl) and on the components within the Davis-Kahan bound.
 */
@RunWith(Enclosed.class)
public class PcaGpuParityTest {

    static final int N = 300;
    static final int F = 5;
    private static final double[] SCALES = {4.0, 2.5, 1.5, 0.8, 0.3};
    private static final double[] OFFSET = {10.0, -3.0, 0.5, 2.0, 7.0};

    /** {@code N x F} samples: independent normal latents scaled by {@link #SCALES}, rotated, then offset. */
    static float[] data() {
        Random rng = new Random(5);
        double[][] r = new double[F][F];
        for (int i = 0; i < F; i++) {
            for (int j = 0; j < F; j++) {
                r[i][j] = rng.nextGaussian();
            }
            for (int p = 0; p < i; p++) {
                double d = 0;
                for (int j = 0; j < F; j++) {
                    d += r[i][j] * r[p][j];
                }
                for (int j = 0; j < F; j++) {
                    r[i][j] -= d * r[p][j];
                }
            }
            double norm = 0;
            for (int j = 0; j < F; j++) {
                norm += r[i][j] * r[i][j];
            }
            for (int j = 0; j < F; j++) {
                r[i][j] /= Math.sqrt(norm);
            }
        }
        float[] x = new float[N * F];
        for (int s = 0; s < N; s++) {
            double[] z = new double[F];
            for (int i = 0; i < F; i++) {
                z[i] = rng.nextGaussian() * SCALES[i];
            }
            for (int j = 0; j < F; j++) {
                double v = OFFSET[j];
                for (int i = 0; i < F; i++) {
                    v += z[i] * r[i][j];
                }
                x[s * F + j] = (float) v;
            }
        }
        return x;
    }

    /** Double-precision moments of row-major data and the reduction-class error bound of the float covariance. */
    static final class Reference {
        final int n;
        final int f;
        final double[] mean;
        final double[] meanAbs;
        final double[] cov;
        /** Frobenius norm of the per-entry covariance error bounds. */
        final double covError;
        final double covNorm;

        Reference(float[] x, int n, int f) {
            this.n = n;
            this.f = f;
            mean = new double[f];
            meanAbs = new double[f];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < f; j++) {
                    mean[j] += x[i * f + j];
                    meanAbs[j] += Math.abs(x[i * f + j]);
                }
            }
            for (int j = 0; j < f; j++) {
                mean[j] /= n;
                meanAbs[j] /= n;
            }
            cov = new double[f * f];
            double err2 = 0;
            double norm2 = 0;
            double alpha = 1.0 / (n - 1);
            for (int a = 0; a < f; a++) {
                for (int b = 0; b < f; b++) {
                    double s = 0;
                    double abs = 0;
                    for (int i = 0; i < n; i++) {
                        double p = (x[i * f + a] - mean[a]) * (x[i * f + b] - mean[b]);
                        s += p;
                        abs += Math.abs(p);
                    }
                    cov[a * f + b] = alpha * s;
                    double e = 2.0 * (1e-5 * Math.sqrt(n) * alpha * abs + n * (double) TrainParity.FLT_MIN);
                    err2 += e * e;
                    norm2 += cov[a * f + b] * cov[a * f + b];
                }
            }
            covError = Math.sqrt(err2);
            covNorm = Math.sqrt(norm2);
        }

        void checkMean(String what, float[] fitted) {
            for (int j = 0; j < f; j++) {
                TrainParity.assertReduction(what + " mean " + j, mean[j], fitted[j], n, meanAbs[j]);
            }
        }

        /** Each component is a unit eigenvector of the reference covariance with its explained variance. */
        void checkEigenpairs(String what, PCAResult r) {
            float[] comps = r.components();
            float[] lambda = r.explainedVariance();
            double bound = covError + 1e-5 * covNorm;
            for (int c = 0; c < r.nComponents(); c++) {
                double residual = 0;
                double norm = 0;
                int pivot = 0;
                for (int a = 0; a < f; a++) {
                    double cv = 0;
                    for (int b = 0; b < f; b++) {
                        cv += cov[a * f + b] * comps[c * f + b];
                    }
                    double d = cv - lambda[c] * (double) comps[c * f + a];
                    residual += d * d;
                    norm += comps[c * f + a] * (double) comps[c * f + a];
                    if (Math.abs(comps[c * f + a]) > Math.abs(comps[c * f + pivot])) {
                        pivot = a;
                    }
                }
                assertEquals(what + " component " + c + " norm", 1.0, Math.sqrt(norm), 1e-6);
                assertTrue(what + " component " + c + " is signed by its largest entry", comps[c * f + pivot] > 0);
                assertTrue(what + " component " + c + " residual " + Math.sqrt(residual) + " beyond " + bound,
                        Math.sqrt(residual) <= bound);
                if (c > 0) {
                    assertTrue(what + " variances descend", lambda[c] <= lambda[c - 1]);
                }
            }
        }
    }

    /** {@code transform} against a double evaluation with the model's own mean and components (GEMM class). */
    static void checkTransform(String what, PCA pca, float[] x, int rows) {
        float[] mean = pca.getMean();
        float[] comps = pca.getComponents();
        int c = comps.length / F;
        float[] out = pca.transform(x);
        assertEquals(what + " transform length", rows * c, out.length);
        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < c; k++) {
                double ref = 0;
                double abs = 0;
                for (int j = 0; j < F; j++) {
                    double p = ((double) x[i * F + j] - mean[j]) * comps[k * F + j];
                    ref += p;
                    abs += Math.abs(p);
                }
                TrainParity.assertReduction(what + " transform " + i + "," + k, ref, out[i * c + k], F,
                        abs + Math.abs(ref));
            }
        }
    }

    /** Paths that need no GPU: the CPU reference of the class. */
    public static class HostPath {
        private ComputeBackend host;

        @Before
        public void setUp() {
            host = TrainParity.hostBackend();
        }

        @Test
        public void fitMatchesTheDoubleReference() {
            float[] x = data();
            Reference ref = new Reference(x, N, F);
            try (PCA pca = new PCA(host, 3)) {
                PCAResult r = pca.fit(x, N, F);
                assertEquals(3, r.nComponents());
                ref.checkMean("cpu", r.mean());
                ref.checkEigenpairs("cpu", r);
                checkTransform("cpu", pca, x, N);
            }
        }

        @Test
        public void componentsAreOrthonormal() {
            float[] comps;
            try (PCA pca = new PCA(host, F)) {
                comps = pca.fit(data(), N, F).components();
            }
            for (int a = 0; a < F; a++) {
                for (int b = 0; b < F; b++) {
                    double d = 0;
                    for (int j = 0; j < F; j++) {
                        d += comps[a * F + j] * (double) comps[b * F + j];
                    }
                    assertEquals("component " + a + " . " + b, a == b ? 1.0 : 0.0, d, 1e-6);
                }
            }
        }

        @Test
        public void fewerSamplesThanComponentsKeepsTheSampleCount() {
            float[] x = {1f, 2f, 3f, 4f, 0f, -1f};
            try (PCA pca = new PCA(host, 3)) {
                PCAResult r = pca.fit(x, 2, 3);
                assertEquals(2, r.nComponents());
                assertEquals(2, pca.getExplainedVariance().length);
                assertEquals("transform uses the kept components", 4, pca.transform(x).length);
                assertEquals(0, pca.transform(new float[0]).length);
            }
        }

        @Test
        public void constantDataHasZeroVariance() {
            float[] x = {2f, 3f, 2f, 3f, 2f, 3f};
            try (PCA pca = new PCA(host, 2)) {
                PCAResult r = pca.fit(x, 3, 2);
                assertEquals(0f, r.explainedVariance()[0], 0f);
                assertEquals(0f, r.explainedVariance()[1], 0f);
                float[] out = pca.transform(x);
                for (float v : out) {
                    assertEquals(0f, v, 0f);
                }
            }
        }

        @Test
        public void invalidInputsThrow() {
            try (PCA pca = new PCA(host, 2)) {
                expectIllegalArgument(() -> pca.fit(new float[]{1f, 2f}, 1, 2));
                expectIllegalArgument(() -> pca.fit(new float[]{1f, 2f, 3f}, 1, 3));
                expectIllegalArgument(() -> pca.fit(new float[]{1f, 2f, 3f, 4f}, 4, 1));
                expectIllegalArgument(() -> pca.fit(new float[5], 2, 3));
                expectIllegalArgument(() -> pca.fit(null, 2, 2));
                expectIllegalArgument(() -> pca.fit(new float[]{Float.NaN, 1f, 2f, 3f}, 2, 2));
                try {
                    pca.transform(new float[2]);
                    fail("transform before fit must throw");
                } catch (IllegalStateException expected) {
                    assertTrue(expected.getMessage().contains("not fitted"));
                }
                pca.fit(new float[]{1f, 2f, 3f, 5f, 0f, 1f}, 3, 2);
                expectIllegalArgument(() -> pca.transform(new float[3]));
            }
            expectIllegalArgument(() -> new PCA(host, 0));
            expectIllegalArgument(() -> new PCA(null, 1));
        }
    }

    /** The device path on each GPU API against the double reference and the CPU path. */
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
        public void fitMatchesTheReferenceAndTheCpuPath() {
            float[] x = data();
            Reference ref = new Reference(x, N, F);
            PCAResult cpu;
            try (PCA cpuModel = new PCA(host, F)) {
                cpu = cpuModel.fit(x, N, F);
            }
            try (PCA pca = new PCA(session.backend(), F)) {
                PCAResult gpu = pca.fit(x, N, F);
                assertEquals(F, gpu.nComponents());
                ref.checkMean(api + " gpu", gpu.mean());
                ref.checkEigenpairs(api + " gpu", gpu);
                checkTransform(api + " gpu", pca, x, N);

                float[] lc = cpu.explainedVariance();
                float[] lg = gpu.explainedVariance();
                double weyl = 2.0 * ref.covError + 2e-5 * ref.covNorm;
                float[] vc = cpu.components();
                float[] vg = gpu.components();
                for (int c = 0; c < F; c++) {
                    assertEquals(api + " explained variance " + c, lc[c], lg[c], weyl);
                    double gap = Double.MAX_VALUE;
                    for (int o = 0; o < F; o++) {
                        if (o != c) {
                            gap = Math.min(gap, Math.abs(lc[c] - lc[o]));
                        }
                    }
                    double sin = 2.0 * weyl / gap;
                    double dot = 0;
                    for (int j = 0; j < F; j++) {
                        dot += vc[c * F + j] * (double) vg[c * F + j];
                    }
                    assertTrue(api + " component " + c + " agrees with the CPU path in direction and sign: dot "
                            + dot, dot >= 1.0 - 2.0 * sin * sin - 1e-5);
                }
            }
        }

        @Test
        public void refitAndTransformOnTheDevice() {
            float[] x = data();
            try (PCA pca = new PCA(session.backend(), 2)) {
                pca.fit(x, N, F);
                float[] first = pca.transform(x);
                assertEquals(N * 2, first.length);
                float[] half = new float[(N / 2) * F];
                System.arraycopy(x, 0, half, 0, half.length);
                PCAResult r = pca.fit(half, N / 2, F);
                new Reference(half, N / 2, F).checkEigenpairs(api + " refit", r);
                checkTransform(api + " refit", pca, x, N);
            }
        }

        @Test
        public void fewerSamplesThanComponents() {
            float[] x = {1f, 2f, 3f, 4f, 0f, -1f};
            try (PCA cpu = new PCA(host, 3); PCA gpu = new PCA(session.backend(), 3)) {
                PCAResult rc = cpu.fit(x, 2, 3);
                PCAResult rg = gpu.fit(x, 2, 3);
                assertEquals(rc.nComponents(), rg.nComponents());
                assertEquals(4, gpu.transform(x).length);
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
