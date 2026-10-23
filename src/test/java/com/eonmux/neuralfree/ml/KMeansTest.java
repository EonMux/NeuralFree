package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.ml.KMeans.KMeansResult;
import com.eonmux.neuralfree.ml.gpu.MlGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link KMeans} through the class itself: its CPU reference on a backend without train kernels (clustering quality,
 * the empty-cluster and tie rules, the chunked update, validation and lifecycle), and its device path on each GPU API
 * against the CPU reference (equal assignments and iteration counts, centroids within the division class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}).
 */
@RunWith(Enclosed.class)
public class KMeansTest {

    /** 150 points around (0, 0), (5, 5) and (10, 0), 50 each, standard deviation 0.5. */
    static float[] threeClusters() {
        Random rng = new Random(42);
        float[][] centers = {{0f, 0f}, {5f, 5f}, {10f, 0f}};
        float[] data = new float[150 * 2];
        for (int i = 0; i < 150; i++) {
            float[] c = centers[i / 50];
            data[2 * i] = c[0] + (float) rng.nextGaussian() * 0.5f;
            data[2 * i + 1] = c[1] + (float) rng.nextGaussian() * 0.5f;
        }
        return data;
    }

    /** {@code n} points of 3 features around 4 well separated centers. */
    static float[] fourClusters(int n, long seed) {
        Random rng = new Random(seed);
        float[][] centers = {{-20f, 0f, 5f}, {20f, 0f, -5f}, {0f, 20f, 0f}, {0f, -20f, 10f}};
        float[] data = new float[n * 3];
        for (int i = 0; i < n; i++) {
            float[] c = centers[i % 4];
            for (int j = 0; j < 3; j++) {
                data[3 * i + j] = c[j] + (float) rng.nextGaussian();
            }
        }
        return data;
    }

    /** Paths that need no GPU: the CPU reference of the class. */
    public static class HostPath {
        private ComputeBackend host;

        @Before
        public void setUp() {
            host = TrainParity.hostBackend();
        }

        @Test
        public void convergesOnSimple2DClustering() {
            KMeansResult result;
            try (KMeans km = new KMeans(host, 3, 100, 1e-4f)) {
                result = km.fit(threeClusters(), 150, 2);
            }
            assertTrue("KMeans should converge on simple 2D clustering", result.converged());
            assertTrue("Should converge in reasonable iterations", result.iterations() < 50);
            assertEquals(6, result.centroids().length);
            assertEquals(150, result.assignments().length);
            int[] counts = new int[3];
            for (int a : result.assignments()) {
                counts[a]++;
            }
            assertArrayEquals("each true cluster becomes one cluster", new int[]{50, 50, 50}, sorted(counts));
        }

        @Test
        public void centroidsNearTrueCenters() {
            float[] centroids;
            try (KMeans km = new KMeans(host, 3)) {
                centroids = km.fit(threeClusters(), 150, 2).centroids();
            }
            float[][] truth = {{0f, 0f}, {5f, 5f}, {10f, 0f}};
            for (float[] t : truth) {
                boolean found = false;
                for (int c = 0; c < 3; c++) {
                    double dx = centroids[2 * c] - t[0];
                    double dy = centroids[2 * c + 1] - t[1];
                    found |= Math.sqrt(dx * dx + dy * dy) < 0.5;
                }
                assertTrue("a centroid near " + Arrays.toString(t), found);
            }
        }

        @Test
        public void centroidsAreMeansOfTheirPoints() {
            float[] data = threeClusters();
            KMeansResult r;
            try (KMeans km = new KMeans(host, 3)) {
                r = km.fit(data, 150, 2);
            }
            assertTrue(r.converged());
            double[] sums = new double[6];
            int[] counts = new int[3];
            int[] assign = r.assignments();
            for (int i = 0; i < 150; i++) {
                counts[assign[i]]++;
                sums[2 * assign[i]] += data[2 * i];
                sums[2 * assign[i] + 1] += data[2 * i + 1];
            }
            float[] centroids = r.centroids();
            for (int t = 0; t < 6; t++) {
                double mean = sums[t] / counts[t / 2];
                assertEquals("centroid element " + t, mean, centroids[t], 1e-5 * Math.max(1.0, Math.abs(mean)));
            }
        }

        @Test
        public void predictUsesTheAssignmentRule() {
            float[] data = threeClusters();
            try (KMeans km = new KMeans(host, 3)) {
                KMeansResult r = km.fit(data, 150, 2);
                float[] centroids = km.getCentroids();
                assertArrayEquals(r.centroids(), centroids, 0f);
                for (int c = 0; c < 3; c++) {
                    float[] point = {centroids[2 * c], centroids[2 * c + 1]};
                    assertEquals("a centroid belongs to its own cluster", c, km.predict(point));
                }
            }
        }

        @Test
        public void emptyClusterKeepsItsCentroidAndTiesGoToTheLowestCluster() {
            float[] data = {1f, 1f, 5f};
            KMeansResult r;
            try (KMeans km = new KMeans(host, 3)) {
                r = km.fit(data, 3, 1);
            }
            assertTrue(r.converged());
            assertEquals("no centroid moves", 1, r.iterations());
            float[] centroids = r.centroids();
            float[] s = centroids.clone();
            Arrays.sort(s);
            assertArrayEquals("the empty cluster keeps its centroid", new float[]{1f, 1f, 5f}, s, 0f);
            int first = -1;
            for (int c = 0; c < 3; c++) {
                if (centroids[c] == 1f) {
                    first = c;
                    break;
                }
            }
            int[] assign = r.assignments();
            assertEquals("tie goes to the lowest cluster", first, assign[0]);
            assertEquals("tie goes to the lowest cluster", first, assign[1]);
        }

        @Test
        public void chunkedUpdateMatchesTheMeans() {
            int n = 5000;
            assertTrue("the update is chunked", MlGpu.kmeansChunks(n) > 1);
            float[] data = fourClusters(n, 7);
            KMeansResult r;
            try (KMeans km = new KMeans(host, 4)) {
                r = km.fit(data, n, 3);
            }
            assertTrue(r.converged());
            double[] sums = new double[12];
            int[] counts = new int[4];
            int[] assign = r.assignments();
            for (int i = 0; i < n; i++) {
                counts[assign[i]]++;
                for (int j = 0; j < 3; j++) {
                    sums[3 * assign[i] + j] += data[3 * i + j];
                }
            }
            float[] centroids = r.centroids();
            for (int t = 0; t < 12; t++) {
                if (counts[t / 3] == 0) {
                    continue;
                }
                double mean = sums[t] / counts[t / 3];
                assertEquals("centroid element " + t, mean, centroids[t], 1e-4 * Math.max(1.0, Math.abs(mean)));
            }
        }

        @Test
        public void chunkRuleFollowsTheShape() {
            assertEquals(1, MlGpu.kmeansChunks(1));
            assertEquals(1, MlGpu.kmeansChunks(4096));
            assertEquals(5, MlGpu.kmeansChunks(4097));
            assertEquals(256, MlGpu.kmeansChunks(1 << 20));
        }

        @Test
        public void refitReplacesTheModel() {
            try (KMeans km = new KMeans(host, 2)) {
                km.fit(new float[]{0f, 0f, 1f, 1f, 9f, 9f, 10f, 10f}, 4, 2);
                assertEquals(4, km.getCentroids().length);
                km.fit(new float[]{0f, 1f, 9f, 10f}, 4, 1);
                assertEquals(2, km.getCentroids().length);
                assertEquals(km.predict(new float[]{0.5f}), km.predict(new float[]{0f}));
            }
        }

        @Test
        public void invalidInputsThrow() {
            try (KMeans km = new KMeans(host, 3)) {
                expectIllegalArgument(() -> km.fit(null, 1, 1));
                expectIllegalArgument(() -> km.fit(new float[4], 2, 3));
                expectIllegalArgument(() -> km.fit(new float[2], 2, 1));
                expectIllegalArgument(() -> km.fit(new float[0], 0, 1));
                try {
                    km.predict(new float[]{0f});
                    fail("predict before fit must throw");
                } catch (IllegalStateException expected) {
                    assertTrue(expected.getMessage().contains("not fitted"));
                }
                km.fit(new float[]{0f, 1f, 2f}, 3, 1);
                expectIllegalArgument(() -> km.predict(new float[]{0f, 1f}));
            }
            expectIllegalArgument(() -> new KMeans(host, 0));
            expectIllegalArgument(() -> new KMeans(host, 2, 0, 1e-4f));
            expectIllegalArgument(() -> new KMeans(host, 2, 10, 0f));
            expectIllegalArgument(() -> new KMeans(null, 2));
        }

        @Test
        public void closedModelRejectsCalls() {
            KMeans km = new KMeans(host, 1);
            km.fit(new float[]{1f, 2f}, 2, 1);
            km.close();
            km.close();
            try {
                km.getCentroids();
                fail("a closed model must throw");
            } catch (IllegalStateException expected) {
                assertTrue(expected.getMessage().contains("closed"));
            }
        }
    }

    /** The device path on each GPU API against the CPU reference of the class. */
    @RunWith(Parameterized.class)
    public static class Parity {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
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
        public void threeClustersMatchTheCpuPath() {
            check("three clusters", threeClusters(), 150, 2, 3);
        }

        @Test
        public void chunkedUpdateMatchesTheCpuPath() {
            check("5000 points, chunked update", fourClusters(5000, 11), 5000, 3, 4);
        }

        @Test
        public void emptyClusterMatchesTheCpuPath() {
            check("empty cluster", new float[]{1f, 1f, 5f}, 3, 1, 3);
        }

        @Test
        public void singleIterationLimit() {
            KMeansResult ref;
            KMeansResult gpu;
            try (KMeans cpu = new KMeans(host, 3, 1, 1e-4f); KMeans dev = new KMeans(session.backend(), 3, 1, 1e-4f)) {
                ref = cpu.fit(threeClusters(), 150, 2);
                gpu = dev.fit(threeClusters(), 150, 2);
            }
            assertEquals(1, gpu.iterations());
            assertEquals(ref.converged(), gpu.converged());
            assertArrayEquals(ref.assignments(), gpu.assignments());
            TrainParity.assertDivSqrt("centroids after one iteration", ref.centroids(), gpu.centroids());
        }

        private void check(String what, float[] data, int n, int d, int k) {
            KMeansResult ref;
            KMeansResult gpu;
            try (KMeans cpu = new KMeans(host, k); KMeans dev = new KMeans(session.backend(), k)) {
                ref = cpu.fit(data, n, d);
                gpu = dev.fit(data, n, d);
                float[] centroids = gpu.centroids();
                for (int c = 0; c < k; c++) {
                    float[] centroid = Arrays.copyOfRange(centroids, c * d, (c + 1) * d);
                    int p = dev.predict(centroid);
                    assertArrayEquals(what + ": predict on the device model finds the centroid itself", centroid,
                            Arrays.copyOfRange(centroids, p * d, (p + 1) * d), 0f);
                }
            }
            assertEquals(what + ": converged", ref.converged(), gpu.converged());
            assertEquals(what + ": iterations", ref.iterations(), gpu.iterations());
            assertArrayEquals(what + ": assignments", ref.assignments(), gpu.assignments());
            TrainParity.assertDivSqrt(what + ": centroids", ref.centroids(), gpu.centroids());
            assertFalse(what + ": no NaN centroid", Float.isNaN(gpu.centroids()[0]));
        }
    }

    private static int[] sorted(int[] values) {
        int[] s = values.clone();
        Arrays.sort(s);
        return s;
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
