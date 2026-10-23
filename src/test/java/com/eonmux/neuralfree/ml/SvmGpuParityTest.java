package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.ml.SVM.KernelType;
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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link SVM} on its CPU reference and on each GPU API. The device kernel matrices ({@code ML_SVM_Kernel}, training and
 * cross form, offset views) are compared with the host twin {@link MlGpu#svmKernel(float[], int, float[], int, int,
 * int, float, int, float)}: LINEAR in the reduction class, RBF and POLY in the transcendental class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}. On integer data the LINEAR and POLY kernels are
 * exact on every path, so SMO sees the same matrix: the support vectors are bit-identical and the decision values agree
 * within twice the GEMM-class bound, using {@code |a_j y_j| <= C}.
 */
@RunWith(Enclosed.class)
public class SvmGpuParityTest {

    /** Two Gaussian clusters around (-3, -3) (label -1) and (3, 3) (label 1), {@code perClass} points each. */
    static float[][] clusters(int perClass, long seed) {
        Random rng = new Random(seed);
        float[][] x = new float[2 * perClass][2];
        for (int i = 0; i < 2 * perClass; i++) {
            float center = i < perClass ? -3f : 3f;
            x[i][0] = center + (float) rng.nextGaussian() * 0.5f;
            x[i][1] = center + (float) rng.nextGaussian() * 0.5f;
        }
        return x;
    }

    static int[] clusterLabels(int perClass) {
        int[] y = new int[2 * perClass];
        for (int i = 0; i < y.length; i++) {
            y[i] = i < perClass ? -1 : 1;
        }
        return y;
    }

    /** Integer points of 4 features around (-2, ...) (label -1) and (2, ...) (label 1), jitter in {-1, 0, 1}. */
    static float[][] integerClusters(int perClass, long seed) {
        Random rng = new Random(seed);
        float[][] x = new float[2 * perClass][4];
        for (int i = 0; i < 2 * perClass; i++) {
            int center = i < perClass ? -2 : 2;
            for (int j = 0; j < 4; j++) {
                x[i][j] = center + rng.nextInt(3) - 1;
            }
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
        public void separableClustersAreClassifiedByEveryKernel() {
            float[][] train = clusters(40, 1);
            int[] y = clusterLabels(40);
            float[][] test = clusters(20, 2);
            int[] yTest = clusterLabels(20);
            SVM[] models = {
                new SVM(host, KernelType.LINEAR),
                new SVM(host, KernelType.RBF),
                new SVM(host, KernelType.POLY, 1f, 0.1f, 2, 1f, 1e-3f, 1000)
            };
            for (SVM svm : models) {
                try {
                    svm.fit(train, y);
                    assertArrayEquals("training points", y, svm.predict(train));
                    assertArrayEquals("held-out points", yTest, svm.predict(test));
                    assertTrue("support vectors exist", svm.getSupportVectors().length > 0);
                } finally {
                    svm.close();
                }
            }
        }

        @Test
        public void supportVectorsAreTrainingRows() {
            float[][] train = clusters(30, 3);
            try (SVM svm = new SVM(host, KernelType.LINEAR)) {
                svm.fit(train, clusterLabels(30));
                float[] sv = svm.getSupportVectors();
                assertEquals(0, sv.length % 2);
                for (int s = 0; s < sv.length / 2; s++) {
                    boolean found = false;
                    for (float[] row : train) {
                        found |= row[0] == sv[2 * s] && row[1] == sv[2 * s + 1];
                    }
                    assertTrue("support vector " + s + " is a training row", found);
                }
            }
        }

        @Test
        public void labelsAtOrAboveZeroArePositive() {
            float[][] train = clusters(20, 4);
            int[] y = new int[40];
            for (int i = 0; i < 40; i++) {
                y[i] = i < 20 ? -5 : 0;
            }
            try (SVM svm = new SVM(host, KernelType.LINEAR)) {
                svm.fit(train, y);
                assertArrayEquals(clusterLabels(20), svm.predict(train));
            }
        }

        @Test
        public void edgeCasesAndValidation() {
            try (SVM svm = new SVM(host, KernelType.LINEAR)) {
                try {
                    svm.predict(new float[][]{{0f, 0f}});
                    fail("predict before fit must throw");
                } catch (IllegalStateException expected) {
                    assertTrue(expected.getMessage().contains("not fitted"));
                }
                svm.fit(clusters(10, 5), clusterLabels(10));
                assertEquals(0, svm.decisionFunction(new float[0][]).length);
                assertEquals(0, svm.predict(new float[0][]).length);
                expectIllegalArgument(() -> svm.predict(new float[][]{{0f, 0f, 0f}}));
                expectIllegalArgument(() -> svm.predict(new float[][]{{0f, 0f}, {0f}}));
                expectIllegalArgument(() -> svm.fit(new float[][]{{0f}}, new int[]{1, 1}));
                expectIllegalArgument(() -> svm.fit(new float[0][], new int[0]));
                expectIllegalArgument(() -> svm.fit(null, new int[0]));
            }
            expectIllegalArgument(() -> new SVM(host, KernelType.LINEAR, 0f, 1f, 3, 0f, 1e-3f, 10));
            expectIllegalArgument(() -> new SVM(host, KernelType.RBF, 1f, 0f, 3, 0f, 1e-3f, 10));
            expectIllegalArgument(() -> new SVM(host, null, 1f, 1f, 3, 0f, 1e-3f, 10));
            expectIllegalArgument(() -> new SVM(null));
        }

        @Test
        public void closedModelRejectsCalls() {
            SVM svm = new SVM(host, KernelType.LINEAR);
            svm.fit(clusters(5, 6), clusterLabels(5));
            svm.close();
            svm.close();
            try {
                svm.getSupportVectors();
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
            float[] big = TrainParity.uniform(21, 12 * f, -1f, 1f);
            float[] other = TrainParity.uniform(22, 6 * f, -1f, 1f);
            TrainGpu g = session.gpu();
            assertNotNull(g);
            int[] types = {TrainOps.ML_SVM_LINEAR, TrainOps.ML_SVM_RBF, TrainOps.ML_SVM_POLY};
            try (MlGpu.Scope scope = MlGpu.scope()) {
                Tensor all = scope.add(MlGpu.upload(session.backend(), big, 12, f));
                Tensor a = scope.add(all.slice(0, 2, 9));
                Tensor b = scope.add(MlGpu.upload(session.backend(), other, 6, f));
                for (int type : types) {
                    float[] cross = MlGpu.readFloats(scope.add(MlGpu.svmKernel(g, a, b, type, 0.5f, 3, 1f)));
                    compare(api + " type " + type + " cross", type, big, 2, 7, other, 0, 6, f, cross);
                    float[] train = MlGpu.readFloats(scope.add(MlGpu.svmKernel(g, a, a, type, 0.5f, 3, 1f)));
                    compare(api + " type " + type + " training", type, big, 2, 7, big, 2, 7, f, train);
                }
            }
        }

        @Test
        public void linearModelMatchesTheCpuPath() {
            checkModel(KernelType.LINEAR, 1f, 0f, 3, 0f);
        }

        @Test
        public void polynomialModelMatchesTheCpuPath() {
            checkModel(KernelType.POLY, 1f, 1f, 2, 1f);
        }

        @Test
        public void rbfModelClassifiesLikeTheCpuPath() {
            float[][] train = clusters(40, 31);
            float[][] test = clusters(20, 32);
            try (SVM cpu = new SVM(host, KernelType.RBF); SVM gpu = new SVM(session.backend(), KernelType.RBF)) {
                cpu.fit(train, clusterLabels(40));
                gpu.fit(train, clusterLabels(40));
                assertArrayEquals(api + " held-out labels", clusterLabels(20), gpu.predict(test));
                assertArrayEquals(api + " agrees with the CPU path", cpu.predict(test), gpu.predict(test));
            }
        }

        private void checkModel(KernelType type, float c, float gamma, int degree, float coef0) {
            float[][] train = integerClusters(25, 41);
            int[] y = clusterLabels(25);
            float[][] test = integerClusters(15, 42);
            int kernel = type == KernelType.LINEAR ? TrainOps.ML_SVM_LINEAR : TrainOps.ML_SVM_POLY;
            try (SVM cpu = new SVM(host, type, c, gamma, degree, coef0, 1e-3f, 1000);
                 SVM gpu = new SVM(session.backend(), type, c, gamma, degree, coef0, 1e-3f, 1000)) {
                cpu.fit(train, y);
                gpu.fit(train, y);
                float[] sv = cpu.getSupportVectors();
                TrainParity.assertBitIdentical(api + " " + type + " support vectors", sv, gpu.getSupportVectors());
                float[] dc = cpu.decisionFunction(test);
                float[] dg = gpu.decisionFunction(test);
                int nsv = sv.length / 4;
                for (int i = 0; i < test.length; i++) {
                    double sumAbs = 0;
                    for (int s = 0; s < nsv; s++) {
                        sumAbs += Math.abs(MlGpu.svmKernel(test[i], 0, sv, s * 4, 4, kernel, gamma, degree, coef0));
                    }
                    double bound = 2.0 * (1e-5 * Math.sqrt(nsv) * 1.001 * c * sumAbs
                            + nsv * (double) TrainParity.FLT_MIN)
                            + 2.0 * Math.ulp(Math.abs(dc[i]));
                    assertEquals(api + " " + type + " decision " + i, dc[i], dg[i], bound);
                    if (Math.abs(dc[i]) > bound) {
                        assertEquals(api + " " + type + " label " + i, dc[i] >= 0, dg[i] >= 0);
                    }
                }
            }
        }

        /** Device {@code k[i*n + j]} against the host twin for rows from {@code ai0} of a and {@code bi0} of b. */
        private static void compare(String what, int type, float[] a, int ai0, int m, float[] b, int bi0, int n, int f,
                                    float[] k) {
            assertEquals(what + ": length", m * n, k.length);
            float[] ref = new float[m * n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    ref[i * n + j] = MlGpu.svmKernel(a, (ai0 + i) * f, b, (bi0 + j) * f, f, type, 0.5f, 3, 1f);
                }
            }
            if (type != TrainOps.ML_SVM_LINEAR) {
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
                    TrainParity.assertReduction(what + " " + i + "," + j, dot, k[i * n + j], f, abs);
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
