package com.eonmux.neuralfree.training;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * {@link GradientClipping}: the host path against the trainers' rules (clipping and scaling, once per gradient slot),
 * and the device path on each GPU API with gradients that share one buffer as equal views, as overlapping views and as
 * a broadcast view.
 */
@RunWith(Enclosed.class)
public class GradientClippingTest {

    /** Trainer's clip-by-norm rule in float, applied to copies: the scaled gradients in list order. */
    static float[][] clipByNormReference(List<float[]> grads, float maxNorm) {
        float total = 0.0f;
        for (float[] g : grads) {
            for (float x : g) {
                total += x * x;
            }
        }
        total = (float) Math.sqrt(total);
        float[][] out = new float[grads.size()][];
        for (int i = 0; i < grads.size(); i++) {
            out[i] = grads.get(i).clone();
            if (total > maxNorm) {
                float scale = maxNorm / (total + 1e-6f);
                for (int j = 0; j < out[i].length; j++) {
                    out[i][j] *= scale;
                }
            }
        }
        return out;
    }

    /** Trainer's clip-by-value rule in float, applied to a copy. */
    static float[] clipByValueReference(float[] g, float maxValue) {
        float[] out = g.clone();
        for (int i = 0; i < out.length; i++) {
            if (out[i] > maxValue) {
                out[i] = maxValue;
            } else if (out[i] < -maxValue) {
                out[i] = -maxValue;
            }
        }
        return out;
    }

    /** The coefficient of the norm rule in double precision over the listed gradients. */
    static double normCoefficient64(List<float[]> grads, float maxNorm) {
        double sum = 0.0;
        for (float[] g : grads) {
            for (float x : g) {
                sum += (double) x * x;
            }
        }
        double total = Math.sqrt(sum);
        return total > maxNorm ? maxNorm / (total + 1e-6) : 1.0;
    }

    static float[] slice(float[] data, int from, int to) {
        return Arrays.copyOfRange(data, from, to);
    }

    /** The host path: a backend without train kernels. */
    public static class HostPath {
        private ComputeBackend host;
        private final List<AutoCloseable> owned = new ArrayList<>();

        @Before
        public void setUp() {
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() throws Exception {
            for (int i = owned.size() - 1; i >= 0; i--) {
                owned.get(i).close();
            }
            owned.clear();
        }

        private Parameter parameter(float[] grad, Shape shape) {
            Tensor flat = track(Tensor.fromArray(host, new float[grad.length]));
            Parameter p = track(new Parameter(track(flat.reshape(shape))));
            Tensor g = track(Tensor.fromArray(host, grad));
            p.setGrad(track(g.reshape(shape)));
            return p;
        }

        private <T extends AutoCloseable> T track(T t) {
            owned.add(t);
            return t;
        }

        @Test
        public void clipByNormFollowsTheTrainerRuleWithSharedGradientsCountedPerParameter() {
            float[] a = TrainParity.uniform(1, 6, -2f, 2f);
            float[] b = TrainParity.uniform(2, 4, -2f, 2f);
            Parameter pa = parameter(a, Shape.of(2, 3));
            Parameter pb = parameter(b, Shape.of(4));
            Tensor sharedFlat = track(Tensor.fromArray(host, new float[6]));
            Parameter shared = track(new Parameter(track(sharedFlat.reshape(2, 3))));
            Tensor original = pa.grad();
            Tensor originalValues = track(original.detach());
            shared.setGrad(original);
            List<Parameter> params = Arrays.asList(pa, pb, shared);
            float[][] expected = clipByNormReference(Arrays.asList(a, b, a), 0.5f);

            GradientClipping.clipByNorm(params, 0.5f);

            TrainParity.assertBitIdentical("first", expected[0], pa.grad().toFloatArray());
            TrainParity.assertBitIdentical("second", expected[1], pb.grad().toFloatArray());
            TrainParity.assertBitIdentical("shared", expected[2], shared.grad().toFloatArray());
            assertNotSame("each slot gets its own clipped gradient", pa.grad(), shared.grad());
            TrainParity.assertBitIdentical("the original gradient's buffer is not modified", a,
                    originalValues.toFloatArray());
            assertEquals(Shape.of(2, 3), pa.grad().shape());
        }

        @Test
        public void clipByNormBelowTheLimitKeepsTheGradients() {
            float[] a = {0.1f, -0.2f};
            Parameter pa = parameter(a, Shape.of(2));
            Tensor before = pa.grad();
            GradientClipping.clipByNorm(Collections.singletonList(pa), 10f);
            assertSame(before, pa.grad());
        }

        @Test
        public void clipByValueClampsAndKeepsNaN() {
            float[] a = {3f, -3f, 0.25f, Float.NaN, -0.0f, Float.NEGATIVE_INFINITY};
            Parameter pa = parameter(a, Shape.of(6));
            GradientClipping.clipByValue(Collections.singletonList(pa), 1f);
            TrainParity.assertBitIdentical("clamped", clipByValueReference(a, 1f), pa.grad().toFloatArray());
        }

        @Test
        public void clipUsesNormBeforeValue() {
            float[] a = {3f, 4f};
            Parameter norm = parameter(a, Shape.of(2));
            Parameter value = parameter(a, Shape.of(2));
            GradientClipping.clip(Collections.singletonList(norm),
                    TrainerConfig.builder().gradientClipNorm(1f).gradientClipValue(0.5f).build());
            GradientClipping.clip(Collections.singletonList(value),
                    TrainerConfig.builder().gradientClipValue(0.5f).build());
            TrainParity.assertBitIdentical("by norm", clipByNormReference(Collections.singletonList(a), 1f)[0],
                    norm.grad().toFloatArray());
            TrainParity.assertBitIdentical("by value", clipByValueReference(a, 0.5f), value.grad().toFloatArray());
        }

        @Test
        public void parametersWithoutGradientsAreSkipped() {
            Parameter none = track(new Parameter(track(Tensor.fromArray(host, new float[3]))));
            Parameter pa = parameter(new float[]{5f, 0f, 0f}, Shape.of(3));
            GradientClipping.clipByValue(Arrays.asList(none, pa), 1f);
            float[] byValue = {1f, 0f, 0f};
            TrainParity.assertBitIdentical("clipped by value", byValue, pa.grad().toFloatArray());
            GradientClipping.clipByNorm(Arrays.asList(none, pa), 0.5f);
            assertNull(none.grad());
            TrainParity.assertBitIdentical("clipped by norm",
                    clipByNormReference(Collections.singletonList(byValue), 0.5f)[0], pa.grad().toFloatArray());
        }

        @Test
        public void parametersOverOneTensorShareOneSlotThatIsClippedOnce() {
            float[] a = {3f, -4f, 0.5f};
            Parameter pa = parameter(a, Shape.of(3));
            Parameter tied = track(new Parameter(pa.tensor()));
            assertSame("both parameters address one gradient slot", pa.grad(), tied.grad());
            List<Parameter> params = Arrays.asList(pa, tied, pa);

            GradientClipping.clipByNorm(params, 1f);

            float[] byNorm = clipByNormReference(Arrays.asList(a, a, a), 1f)[0];
            TrainParity.assertBitIdentical("counted per listing, scaled once", byNorm, pa.grad().toFloatArray());
            assertSame(pa.grad(), tied.grad());

            GradientClipping.clipByValue(params, 0.1f);
            TrainParity.assertBitIdentical("clamped once", clipByValueReference(byNorm, 0.1f),
                    tied.grad().toFloatArray());
        }

        @Test
        public void scaleMultipliesEveryGradientSlotOnce() {
            float[] a = {1.5f, -3f, 0.25f};
            float[] b = {-2f, 6f};
            Parameter pa = parameter(a, Shape.of(3));
            Parameter pb = parameter(b, Shape.of(2));
            Parameter tied = track(new Parameter(pa.tensor()));
            Parameter none = track(new Parameter(track(Tensor.fromArray(host, new float[2]))));

            GradientClipping.scale(Arrays.asList(pa, pb, tied, pa, none), 1.0f / 3);

            float third = 1.0f / 3;
            float[] expectedA = new float[a.length];
            for (int i = 0; i < a.length; i++) {
                expectedA[i] = a[i] * third;
            }
            float[] expectedB = {b[0] * third, b[1] * third};
            TrainParity.assertBitIdentical("first", expectedA, pa.grad().toFloatArray());
            TrainParity.assertBitIdentical("second", expectedB, pb.grad().toFloatArray());
            assertSame("the tied parameter shares the scaled slot", pa.grad(), tied.grad());
            assertNull("a parameter without a gradient keeps none", none.grad());
        }

        @Test(expected = IllegalArgumentException.class)
        public void nonFiniteScaleIsRejected() {
            GradientClipping.scale(Collections.emptyList(), Float.POSITIVE_INFINITY);
        }

        @Test
        public void deviceGradientsSignalTheHostPath() {
            Parameter pa = parameter(new float[]{1f}, Shape.of(1));
            assertNull(GradientClipping.deviceGradients(Collections.singletonList(pa)));
            Parameter none = track(new Parameter(track(Tensor.fromArray(host, new float[1]))));
            assertTrue(GradientClipping.deviceGradients(Collections.singletonList(none)).isEmpty());
        }

        @Test(expected = IllegalArgumentException.class)
        public void negativeMaxNormIsRejected() {
            GradientClipping.clipByNorm(Collections.emptyList(), -1f);
        }

        @Test(expected = IllegalArgumentException.class)
        public void nanMaxValueIsRejected() {
            GradientClipping.clipByValue(Collections.emptyList(), Float.NaN);
        }
    }

    /** The device path on each GPU API with the train kernels available. */
    @RunWith(Parameterized.class)
    public static class Device {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;
        private ComputeBackend gpu;
        private final List<AutoCloseable> owned = new ArrayList<>();

        public Device(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            session = TrainParity.open(api);
            gpu = session.backend();
        }

        @After
        public void tearDown() throws Exception {
            try {
                for (int i = owned.size() - 1; i >= 0; i--) {
                    owned.get(i).close();
                }
                owned.clear();
            } finally {
                session.close();
            }
        }

        private <T extends AutoCloseable> T track(T t) {
            owned.add(t);
            return t;
        }

        private Parameter parameter(Shape shape) {
            Tensor flat = track(Tensor.fromArray(gpu, new float[(int) shape.numElements()]));
            return track(new Parameter(track(flat.reshape(shape))));
        }

        /**
         * Three parameters over one buffer of 20 gradients: elements 0-11, elements 6-17 (overlapping the first) and
         * the first parameter's gradient tensor itself.
         */
        private Parameter[] overlapping(Tensor buffer) {
            Parameter p1 = parameter(Shape.of(12));
            Parameter p2 = parameter(Shape.of(12));
            Parameter p3 = parameter(Shape.of(12));
            Tensor first = track(buffer.slice(0, 0, 12));
            p1.setGrad(first);
            p2.setGrad(track(buffer.slice(0, 6, 18)));
            p3.setGrad(first);
            return new Parameter[]{p1, p2, p3};
        }

        @Test
        public void clipByNormGivesOverlappingGradientsTheirOwnCopy() {
            float[] data = TrainParity.uniform(3, 20, -2f, 2f);
            Tensor buffer = track(Tensor.fromArray(gpu, data));
            Parameter[] p = overlapping(buffer);
            List<float[]> listed = Arrays.asList(slice(data, 0, 12), slice(data, 6, 18), slice(data, 0, 12));
            double coef = normCoefficient64(listed, 0.5f);
            Tensor firstBefore = p[0].grad();

            GradientClipping.clipByNorm(Arrays.asList(p), 0.5f);
            track(p[1].grad());

            assertSame("the first gradient is clipped in place", firstBefore, p[0].grad());
            assertSame("an equal view stays shared", p[0].grad(), p[2].grad());
            assertFalse("the overlapping gradient got its own copy",
                    GpuOptimizerOps.sharesElements(p[1].grad(), buffer));
            int k = 36;
            for (int i = 0; i < 2; i++) {
                float[] original = listed.get(i);
                float[] got = p[i].grad().toFloatArray();
                for (int j = 0; j < got.length; j++) {
                    TrainParity.assertReduction(api + " gradient " + i + " element " + j, original[j] * coef, got[j],
                            k, Math.abs(original[j]));
                }
            }
            float[] after = buffer.toFloatArray();
            for (int j = 12; j < 20; j++) {
                assertEquals("elements outside the first view keep their values", data[j], after[j], 0.0f);
            }
        }

        @Test
        public void clipByValueMatchesTheHostPathForOverlappingAndBroadcastGradients() {
            float[] data = TrainParity.uniform(4, 20, -2f, 2f);
            Tensor buffer = track(Tensor.fromArray(gpu, data));
            Parameter[] p = overlapping(buffer);
            float[] rowData = {-2f, 0.5f, 3f};
            Tensor row = track(track(Tensor.fromArray(gpu, rowData)).reshape(Shape.of(1, 3)));
            Parameter broadcast = parameter(Shape.of(4, 3));
            broadcast.setGrad(track(row.expand(Shape.of(4, 3))));
            List<Parameter> params = Arrays.asList(p[0], p[1], p[2], broadcast);

            GradientClipping.clipByValue(params, 0.5f);
            track(p[1].grad());
            track(broadcast.grad());

            TrainParity.assertBitIdentical(api + " first", clipByValueReference(slice(data, 0, 12), 0.5f),
                    p[0].grad().toFloatArray());
            TrainParity.assertBitIdentical(api + " overlapping", clipByValueReference(slice(data, 6, 18), 0.5f),
                    p[1].grad().toFloatArray());
            TrainParity.assertBitIdentical(api + " shared", p[0].grad().toFloatArray(), p[2].grad().toFloatArray());
            float[] clippedRow = clipByValueReference(rowData, 0.5f);
            float[] expectedBroadcast = new float[12];
            for (int r = 0; r < 4; r++) {
                System.arraycopy(clippedRow, 0, expectedBroadcast, 3 * r, 3);
            }
            TrainParity.assertBitIdentical(api + " broadcast", expectedBroadcast, broadcast.grad().toFloatArray());
            TrainParity.assertBitIdentical(api + " broadcast source untouched", rowData, row.toFloatArray());
        }

        @Test
        public void scaleReachesEachDistinctViewOnce() {
            float[] data = TrainParity.uniform(6, 20, -2f, 2f);
            Tensor buffer = track(Tensor.fromArray(gpu, data));
            Parameter[] p = overlapping(buffer);
            Tensor firstBefore = p[0].grad();

            GradientClipping.scale(Arrays.asList(p), 0.5f);
            track(p[1].grad());

            assertSame("the first gradient is scaled in place", firstBefore, p[0].grad());
            assertSame("an equal view stays shared", p[0].grad(), p[2].grad());
            assertFalse("the overlapping gradient got its own copy",
                    GpuOptimizerOps.sharesElements(p[1].grad(), buffer));
            float[] first = slice(data, 0, 12);
            float[] second = slice(data, 6, 18);
            for (int j = 0; j < first.length; j++) {
                first[j] *= 0.5f;
                second[j] *= 0.5f;
            }
            TrainParity.assertSingleRounding(api + " first, scaled once", first, p[0].grad().toFloatArray());
            TrainParity.assertSingleRounding(api + " overlapping copy", second, p[1].grad().toFloatArray());
            float[] after = buffer.toFloatArray();
            for (int j = 12; j < 20; j++) {
                assertEquals("elements outside the first view keep their values", data[j], after[j], 0.0f);
            }
        }

        @Test
        public void deviceGradientsAreDisjointOrEqualViews() {
            Tensor buffer = track(Tensor.fromArray(gpu, TrainParity.uniform(5, 20, -1f, 1f)));
            Parameter[] p = overlapping(buffer);
            List<Tensor> grads = GradientClipping.deviceGradients(Arrays.asList(p));
            assertNotNull(grads);
            track(p[1].grad());
            assertEquals(3, grads.size());
            assertTrue(GpuOptimizerOps.sameView(grads.get(0), grads.get(2)));
            assertFalse(GpuOptimizerOps.sharesElements(grads.get(0), grads.get(1)));
            assertNotSame(grads.get(0), grads.get(1));
            assertSame(p[1].grad(), grads.get(1));
        }
    }
}
