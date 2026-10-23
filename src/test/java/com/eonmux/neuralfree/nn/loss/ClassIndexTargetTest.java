package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.loss.LossFunction.Reduction;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collection;
import java.util.function.Function;
import java.util.function.Supplier;

import static com.eonmux.neuralfree.nn.loss.LossFunctionsTest.Scenarios.floor1;
import static com.eonmux.neuralfree.nn.loss.LossFunctionsTest.Scenarios.gradient;
import static com.eonmux.neuralfree.nn.loss.LossFunctionsTest.Scenarios.logSoftmax;
import static com.eonmux.neuralfree.nn.loss.LossFunctionsTest.Scenarios.tensor;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Class-index targets of {@link CrossEntropyLoss} and {@link FocalLoss}, on the CPU path and on each GPU API: INT32,
 * INT64 (narrowed with Java {@code (int)}) and integral FLOAT32 targets give bit-identical losses and gradients;
 * samples whose class is the ignore index contribute 0 and are left out of the MEAN denominator; and a class outside
 * {@code [0, C)} that is not ignored throws {@link IndexOutOfBoundsException}, at once on the CPU path and at the next
 * host read on the device, with the gather's message on both.
 */
@RunWith(Enclosed.class)
public class ClassIndexTargetTest {

    /** The ignore index's default, accessors and validation, which need no backend. */
    public static class Configuration {

        @Test
        public void ignoreIndexDefaultsToPyTorchs() {
            assertEquals(-100, CrossEntropyLoss.DEFAULT_IGNORE_INDEX);
            assertEquals(-100, FocalLoss.DEFAULT_IGNORE_INDEX);
            assertEquals(-100, new CrossEntropyLoss().ignoreIndex());
            assertEquals(-100, new FocalLoss().ignoreIndex());
            assertEquals(3, new CrossEntropyLoss(Reduction.SUM, 1, 0.1f, 3).ignoreIndex());
            assertEquals(-7, new FocalLoss(0.5f, 1.0f, Reduction.SUM, -1, -7).ignoreIndex());
            assertTrue(new CrossEntropyLoss(Reduction.MEAN, -1, 0.0f, 5).toString().contains("ignoreIndex=5"));
            assertTrue(new FocalLoss(0.5f, 1.0f, Reduction.MEAN, -1, 5).toString().contains("ignoreIndex=5"));
        }

        @Test
        public void ignoreIndexMustBeExactInFloat() {
            int bound = 1 << 24;
            assertEquals(bound - 1, new CrossEntropyLoss(Reduction.MEAN, -1, 0.0f, bound - 1).ignoreIndex());
            assertEquals(1 - bound, new FocalLoss(0.25f, 2.0f, Reduction.MEAN, -1, 1 - bound).ignoreIndex());
            assertThrows(IllegalArgumentException.class, () -> new CrossEntropyLoss(Reduction.MEAN, -1, 0.0f, bound));
            assertThrows(IllegalArgumentException.class,
                    () -> new CrossEntropyLoss(Reduction.MEAN, -1, 0.0f, -bound));
            assertThrows(IllegalArgumentException.class, () -> new FocalLoss(0.25f, 2.0f, Reduction.MEAN, -1, bound));
        }
    }

    /** The scenarios on the CPU path. */
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
        public void integerAndFloatTargetsAgree() {
            Scenarios.dtypesAgree(host, "cpu path");
        }

        @Test
        public void ignoredSamplesLeaveTheMean() {
            Scenarios.ignored(host, "cpu path");
        }

        @Test
        public void ignoredRowsAreInert() {
            Scenarios.ignoredRowsAreInert(host, "cpu path");
        }

        @Test
        public void outOfRangeClassesThrow() {
            Scenarios.outOfRange(host, "cpu path", false);
        }

        @Test
        public void otherTargetDtypesAreRejected() {
            Scenarios.unsupportedDtype(host);
        }
    }

    /** The scenarios on each GPU API. */
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
        public void integerAndFloatTargetsAgree() {
            Scenarios.dtypesAgree(gpu, api.toString());
        }

        @Test
        public void ignoredSamplesLeaveTheMean() {
            Scenarios.ignored(gpu, api.toString());
        }

        @Test
        public void ignoredRowsAreInert() {
            Scenarios.ignoredRowsAreInert(gpu, api.toString());
        }

        @Test
        public void outOfRangeClassesRaiseAtTheNextRead() {
            Scenarios.outOfRange(gpu, api.toString(), true);
        }
    }

    // ------------------------------------------------------------------ scenarios

    static final class Scenarios {
        private static final int ROWS = 5;
        private static final int CLASSES = 4;
        private static final int IGNORE = CrossEntropyLoss.DEFAULT_IGNORE_INDEX;
        /** The classes of the five rows; row 2 is ignored. */
        private static final int[] CLASS_OF_ROW = {3, 0, IGNORE, 2, 1};
        private static final float LABEL_SMOOTHING = 0.1f;
        private static final float ALPHA = 0.4f;
        private static final float GAMMA = 1.5f;

        private Scenarios() {
        }

        /**
         * Cross-entropy (plain and label-smoothed) and focal loss of {@code [5, 4]} logits with the same classes as
         * INT32, as INT64 (its last value {@code 2^32 + 1} narrows to 1) and as FLOAT32 (its last value 1.2 rounds to
         * 1): every reduction gives bit-identical results, which match the double reference with the ignored row
         * left out, and the MEAN gradients are bit-identical. A {@code [2, 3, 2]} prediction with the class dim 1
         * checks the same agreement for an inner class axis.
         */
        static void dtypesAgree(ComputeBackend b, String what) {
            float[] x = TrainParity.uniform(70, ROWS * CLASSES, -3f, 3f);
            float[] xInner = TrainParity.uniform(71, 12, -3f, 3f);
            try (Tensor p = tensor(b, x, ROWS, CLASSES);
                 Tensor i32 = int32(b, CLASS_OF_ROW, ROWS);
                 Tensor i64 = int64(b, new long[]{3, 0, IGNORE, 2, (1L << 32) + 1}, ROWS);
                 Tensor f32 = tensor(b, new float[]{3f, 0f, IGNORE, 2f, 1.2f}, ROWS);
                 Tensor pInner = tensor(b, xInner, 2, 3, 2);
                 Tensor innerI32 = int32(b, new int[]{2, IGNORE, 0, 1}, 2, 2);
                 Tensor innerI64 = int64(b, new long[]{2, IGNORE, 0, 1}, 2, 2);
                 Tensor innerF32 = tensor(b, new float[]{2f, IGNORE, 0f, 1f}, 2, 2)) {
                Function<Reduction, LossFunction> plain = r -> new CrossEntropyLoss(r);
                Function<Reduction, LossFunction> smooth = r -> new CrossEntropyLoss(r, -1, LABEL_SMOOTHING);
                Function<Reduction, LossFunction> focal = r -> new FocalLoss(ALPHA, GAMMA, r, -1);
                Function<Reduction, LossFunction> inner = r -> new CrossEntropyLoss(r, 1, LABEL_SMOOTHING);
                agree(what + " CrossEntropyLoss", b, x, plain, p, i32, i64, f32);
                agree(what + " CrossEntropyLoss label smoothing", b, x, smooth, p, i32, i64, f32);
                agree(what + " FocalLoss", b, x, focal, p, i32, i64, f32);
                agree(what + " CrossEntropyLoss class dim 1", b, xInner, inner, pInner, innerI32, innerI64, innerF32);

                double[][] cePlain = rowReference(x, (lp, c) -> -lp[c], (lp, c) -> floor1(lp[c]));
                double[][] ceSmooth = rowReference(x, Scenarios::smoothedValue, Scenarios::smoothedMagnitude);
                double[][] focalRef = rowReference(x,
                        (lp, c) -> -ALPHA * Math.pow(1.0 - Math.exp(lp[c]), GAMMA) * lp[c],
                        (lp, c) -> ALPHA * floor1(lp[c]) * (2.0 + GAMMA));
                for (Reduction r : Reduction.values()) {
                    expectRows(what + " CrossEntropyLoss " + r, b, r, cePlain, CLASSES, plain.apply(r), p, i32);
                    expectRows(what + " CrossEntropyLoss label smoothing " + r, b, r, ceSmooth, CLASSES + 2,
                            smooth.apply(r), p, i32);
                    expectRows(what + " FocalLoss " + r, b, r, focalRef, CLASSES, focal.apply(r), p, i32);
                }

                double[] ceGrad = new double[ROWS * CLASSES];
                for (int row = 0; row < ROWS; row++) {
                    if (CLASS_OF_ROW[row] == IGNORE) {
                        continue;
                    }
                    double[] lp = logSoftmax(x, row * CLASSES, 1, CLASSES);
                    for (int c = 0; c < CLASSES; c++) {
                        ceGrad[row * CLASSES + c] = (Math.exp(lp[c]) - (c == CLASS_OF_ROW[row] ? 1.0 : 0.0)) / 4.0;
                    }
                }
                LossFunctionsTest.Scenarios.expectGradient(what + " CrossEntropyLoss MEAN over the 4 kept rows",
                        ceGrad, 2.0 / 4.0, gradient(b, x, new int[]{ROWS, CLASSES},
                                q -> new CrossEntropyLoss(Reduction.MEAN).compute(q, i32)));
            }
        }

        /**
         * The ignore index set to a valid class (1) leaves its rows out of every reduction, and a batch whose rows are
         * all ignored has a MEAN of NaN (0/0, as PyTorch), a SUM of 0 and per-sample losses of 0.
         */
        static void ignored(ComputeBackend b, String what) {
            float[] x = TrainParity.uniform(72, ROWS * CLASSES, -3f, 3f);
            int[] classes = {1, 0, 1, 3, 2};
            try (Tensor p = tensor(b, x, ROWS, CLASSES);
                 Tensor t = int32(b, classes, ROWS);
                 Tensor none = int32(b, new int[]{IGNORE, IGNORE, IGNORE, IGNORE, IGNORE}, ROWS)) {
                double[][] ref = new double[3][ROWS];
                for (int row = 0; row < ROWS; row++) {
                    double[] lp = logSoftmax(x, row * CLASSES, 1, CLASSES);
                    boolean kept = classes[row] != 1;
                    ref[0][row] = kept ? 1.0 : 0.0;
                    ref[1][row] = kept ? -lp[classes[row]] : 0.0;
                    ref[2][row] = kept ? floor1(lp[classes[row]]) : 0.0;
                }
                for (Reduction r : Reduction.values()) {
                    expectRows(what + " CrossEntropyLoss ignoring class 1 " + r, b, r, ref, CLASSES,
                            new CrossEntropyLoss(r, -1, 0.0f, 1), p, t);
                    for (LossFunction loss : new LossFunction[]{new CrossEntropyLoss(r), new FocalLoss(ALPHA, GAMMA,
                            r, -1)}) {
                        String name = what + " " + loss.name() + " with every row ignored " + r;
                        try (Tensor out = loss.compute(p, none)) {
                            float[] got = out.toFloatArray();
                            for (float v : got) {
                                if (r == Reduction.MEAN) {
                                    assertTrue(name + ": NaN, got " + v, Float.isNaN(v));
                                } else {
                                    assertEquals(name, 0.0f, v, 0.0f);
                                }
                            }
                        }
                    }
                }
            }
        }

        /**
         * Ignored rows whose class-0 log-probability is 0 (a dominant logit) or {@code -inf} (a {@code -inf} logit)
         * leave the MEAN finite and get a gradient of exactly 0, for focal loss with gamma 0.5 (whose focal weight has
         * an infinite derivative at {@code p = 1}) and label-smoothed cross-entropy; the kept row's gradient is finite.
         */
        static void ignoredRowsAreInert(ComputeBackend b, String what) {
            float[] x = {0.3f, -0.7f, 1.1f, 0.2f, 30f, 0f, 0f, 0f, Float.NEGATIVE_INFINITY, 0f, 1f, 2f};
            int[] dims = {3, CLASSES};
            LossFunction[] losses = {new FocalLoss(ALPHA, 0.5f, Reduction.MEAN, -1),
                    new CrossEntropyLoss(Reduction.MEAN, -1, LABEL_SMOOTHING)};
            try (Tensor t = int32(b, new int[]{2, IGNORE, IGNORE}, 3)) {
                for (LossFunction loss : losses) {
                    String name = what + " " + loss.name() + " with inert ignored rows";
                    try (Tensor p = tensor(b, x, dims);
                         Tensor out = loss.compute(p, t)) {
                        float v = out.toFloatArray()[0];
                        assertTrue(name + ": finite loss, got " + v, Float.isFinite(v));
                    }
                    float[] g = gradient(b, x, dims, q -> loss.compute(q, t));
                    for (int i = 0; i < g.length; i++) {
                        if (i < CLASSES) {
                            assertTrue(name + ": finite kept gradient [" + i + "], got " + g[i],
                                    Float.isFinite(g[i]));
                        } else {
                            assertEquals(name + ": ignored gradient [" + i + "]", 0.0f, g[i], 0.0f);
                        }
                    }
                }
            }
        }

        /**
         * Classes outside {@code [0, 4)} that are not ignored: INT32 4, INT64 -1, FLOAT32 3.6 (rounds to 4) and -100
         * under an ignore index of 0. On the device the loss itself is enqueued without error and the exception is
         * raised by the read; afterwards the backend computes a valid loss again.
         */
        static void outOfRange(ComputeBackend b, String what, boolean deferred) {
            float[] x = TrainParity.uniform(73, 3 * CLASSES, -3f, 3f);
            try (Tensor p = tensor(b, x, 3, CLASSES);
                 Tensor tooLarge = int32(b, new int[]{0, 4, 1}, 3);
                 Tensor negative = int64(b, new long[]{0, 2, -1}, 3);
                 Tensor rounded = tensor(b, new float[]{3.6f, 0f, 1f}, 3);
                 Tensor notIgnored = int32(b, new int[]{IGNORE, 1, 2}, 3);
                 Tensor valid = int32(b, new int[]{0, 3, 1}, 3)) {
                expectOutOfRange(what + " CrossEntropyLoss INT32 4", deferred,
                        () -> new CrossEntropyLoss(Reduction.MEAN).compute(p, tooLarge));
                expectOutOfRange(what + " CrossEntropyLoss INT64 -1", deferred,
                        () -> new CrossEntropyLoss(Reduction.NONE).compute(p, negative));
                expectOutOfRange(what + " FocalLoss FLOAT32 3.6", deferred,
                        () -> new FocalLoss(ALPHA, GAMMA, Reduction.SUM, -1).compute(p, rounded));
                expectOutOfRange(what + " CrossEntropyLoss -100 with ignore index 0", deferred,
                        () -> new CrossEntropyLoss(Reduction.MEAN, -1, 0.0f, 0).compute(p, notIgnored));
                try (Tensor out = new CrossEntropyLoss(Reduction.SUM).compute(p, valid)) {
                    assertTrue(what + " a valid loss after the errors", Float.isFinite(out.toFloatArray()[0]));
                }
            }
        }

        /** A BOOL class-index target is rejected before anything runs. */
        static void unsupportedDtype(ComputeBackend b) {
            try (Tensor p = tensor(b, new float[12], 3, CLASSES);
                 Tensor t = Tensor.zeros(b, Shape.of(3), DType.BOOL)) {
                assertThrows(IllegalArgumentException.class, () -> new CrossEntropyLoss().compute(p, t));
                assertThrows(IllegalArgumentException.class, () -> new FocalLoss().compute(p, t));
            }
        }

        // -------------------------------------------------------------- checks

        /**
         * {@code loss} of {@code p} with each of the three targets gives bit-identical results for every reduction
         * and bit-identical MEAN gradients.
         */
        private static void agree(String what, ComputeBackend b, float[] x, Function<Reduction, LossFunction> loss,
                                  Tensor p, Tensor i32, Tensor i64, Tensor f32) {
            for (Reduction r : Reduction.values()) {
                LossFunction fn = loss.apply(r);
                float[] ref = values(fn, p, i32);
                TrainParity.assertBitIdentical(what + " " + r + " INT64 vs INT32", ref, values(fn, p, i64));
                TrainParity.assertBitIdentical(what + " " + r + " FLOAT32 vs INT32", ref, values(fn, p, f32));
            }
            LossFunction mean = loss.apply(Reduction.MEAN);
            int[] dims = p.shape().dims();
            float[] g32 = gradient(b, x, dims, q -> mean.compute(q, i32));
            TrainParity.assertBitIdentical(what + " gradient INT64 vs INT32", g32,
                    gradient(b, x, dims, q -> mean.compute(q, i64)));
            TrainParity.assertBitIdentical(what + " gradient FLOAT32 vs INT32", g32,
                    gradient(b, x, dims, q -> mean.compute(q, f32)));
        }

        private static float[] values(LossFunction fn, Tensor p, Tensor t) {
            try (Tensor out = fn.compute(p, t)) {
                return out.toFloatArray();
            }
        }

        /**
         * Checks {@code loss(p, t)} against per-row references {@code ref = {kept, value, magnitude}}: per sample
         * for NONE (0 for a row that is not kept), the sum of the kept rows for SUM, and that sum over the number of
         * kept rows for MEAN.
         */
        private static void expectRows(String what, ComputeBackend b, Reduction r, double[][] ref, int terms,
                                       LossFunction loss, Tensor p, Tensor t) {
            try (Tensor out = loss.compute(p, t)) {
                assertSame(what + ": backend", b, out.backend());
                float[] got = out.toFloatArray();
                if (r == Reduction.NONE) {
                    assertEquals(what + ": per-sample shape", Shape.of(ref[0].length), out.shape());
                    for (int i = 0; i < got.length; i++) {
                        if (ref[0][i] == 0.0) {
                            assertEquals(what + " ignored [" + i + "]", 0.0f, got[i], 0.0f);
                        } else {
                            TrainParity.assertReduction(what + " [" + i + "]", ref[1][i], got[i], terms, ref[2][i]);
                        }
                    }
                    return;
                }
                assertEquals(what + ": shape", Shape.scalar(), out.shape());
                double kept = 0;
                double v = 0;
                double m = 0;
                for (int i = 0; i < ref[0].length; i++) {
                    kept += ref[0][i];
                    v += ref[1][i];
                    m += ref[2][i];
                }
                if (r == Reduction.MEAN) {
                    v /= kept;
                    m /= kept;
                }
                TrainParity.assertReduction(what, v, got[0], ref[0].length * terms, m);
            }
        }

        /**
         * Runs {@code loss} and reads its result, expecting the gather's {@link IndexOutOfBoundsException}: from the
         * read when {@code deferred}, from the computation itself otherwise.
         */
        private static void expectOutOfRange(String what, boolean deferred, Supplier<Tensor> loss) {
            IndexOutOfBoundsException e;
            if (deferred) {
                try (Tensor out = loss.get()) {
                    e = assertThrows(what, IndexOutOfBoundsException.class, out::toFloatArray);
                }
            } else {
                e = assertThrows(what, IndexOutOfBoundsException.class, () -> loss.get().close());
            }
            assertTrue(what + ": message " + e.getMessage(), e.getMessage().startsWith("Gather index"));
        }

        // -------------------------------------------------------------- references

        private interface RowTerm {
            double of(double[] lp, int c);
        }

        /** {@code {kept, value, magnitude}} of each row of the {@code [5, 4]} logits {@code x}. */
        private static double[][] rowReference(float[] x, RowTerm value, RowTerm magnitude) {
            double[][] ref = new double[3][ROWS];
            for (int row = 0; row < ROWS; row++) {
                int c = CLASS_OF_ROW[row];
                if (c == IGNORE) {
                    continue;
                }
                double[] lp = logSoftmax(x, row * CLASSES, 1, CLASSES);
                ref[0][row] = 1.0;
                ref[1][row] = value.of(lp, c);
                ref[2][row] = magnitude.of(lp, c);
            }
            return ref;
        }

        /** PyTorch's smoothed class-index cross-entropy: {@code (1 - ls) * -lp[c] + ls * -sum(lp) / C}. */
        private static double smoothedValue(double[] lp, int c) {
            double sum = 0;
            for (double v : lp) {
                sum += v;
            }
            return (1.0 - LABEL_SMOOTHING) * -lp[c] - LABEL_SMOOTHING * sum / lp.length;
        }

        private static double smoothedMagnitude(double[] lp, int c) {
            double sum = 0;
            for (double v : lp) {
                sum += floor1(v);
            }
            return (1.0 - LABEL_SMOOTHING) * floor1(lp[c]) + LABEL_SMOOTHING * sum / lp.length;
        }

        // -------------------------------------------------------------- tensors

        static Tensor int32(ComputeBackend b, int[] values, int... dims) {
            ByteBuffer bytes = ByteBuffer.allocate(values.length * Integer.BYTES).order(ByteOrder.nativeOrder());
            bytes.asIntBuffer().put(values);
            return Tensor.fromStorage(b, Shape.of(dims), DType.INT32, TensorStorage.fromHost(b, bytes, DType.INT32));
        }

        static Tensor int64(ComputeBackend b, long[] values, int... dims) {
            ByteBuffer bytes = ByteBuffer.allocate(values.length * Long.BYTES).order(ByteOrder.nativeOrder());
            bytes.asLongBuffer().put(values);
            return Tensor.fromStorage(b, Shape.of(dims), DType.INT64, TensorStorage.fromHost(b, bytes, DType.INT64));
        }
    }
}
