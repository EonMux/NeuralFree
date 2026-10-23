package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.IntUnaryOperator;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link ConvOps}, {@link NormOps} and {@link TensorOps#maskScale} on each GPU API against the same facades on the
 * CPU reference backend, under the tolerance classes of the dispatchers. The device path is the dispatcher's (the
 * facade result is bit-identical to a direct dispatcher call, whose kernels are deterministic). For the reduction
 * class both paths lie within {@code 1e-5 * sqrt(K) * sum|terms| + K * FLT_MIN} of the exact result, so their
 * difference is held to twice that bound. Operands on another backend take the CPU path.
 */
@RunWith(Parameterized.class)
public class LayerOpsParityTest {

    private static final float EPS = 1e-5f;

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();

    public LayerOpsParityTest(TrainParity.Api api) {
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
            for (int i = owned.size() - 1; i >= 0; i--) {
                owned.get(i).close();
            }
        } finally {
            try {
                TrainGpu.release(host);
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }
    }

    // ------------------------------------------------------------------ convolutions

    @Test
    public void conv2dDeviceMatchesCpu() {
        int b = 2, c = 6, h = 11, w = 9, oc = 4, groups = 2;
        int[] k = {3, 2}, s = {2, 1}, p = {1, 1}, d = {1, 2};
        float[] x = TrainParity.uniform(11, b * c * h * w, -3f, 3f);
        float[] wt = TrainParity.uniform(12, oc * (c / groups) * k[0] * k[1], -1f, 1f);
        float[] bias = TrainParity.uniform(13, oc, -2f, 2f);
        int[] xd = {b, c, h, w}, wd = {oc, c / groups, k[0], k[1]};
        Tensor dev = device(ConvOps.conv2d(upload(gpu, x, xd), upload(gpu, wt, wd), upload(gpu, bias, oc),
                k, s, p, d, groups));
        Tensor cpu = keep(ConvOps.conv2d(upload(host, x, xd), upload(host, wt, wd), upload(host, bias, oc),
                k, s, p, d, groups));
        Tensor terms = keep(ConvOps.conv2d(upload(host, abs(x), xd), upload(host, abs(wt), wd),
                upload(host, abs(bias), oc), k, s, p, d, groups));
        assertEquals(cpu.shape(), dev.shape());
        assertReductionPair("conv2d", cpu, dev, (c / groups) * k[0] * k[1] + 1, terms.toFloatArray());
        TrainParity.assertBitIdentical("conv2d facade vs dispatcher", keep(GpuConvOps.conv2dForward(
                upload(gpu, x, xd), upload(gpu, wt, wd), upload(gpu, bias, oc), k, s, p, d, groups)).toFloatArray(),
                dev.toFloatArray());
    }

    @Test
    public void conv1dDeviceMatchesCpu() {
        int b = 3, c = 4, l = 17, oc = 6, kernel = 3, stride = 2, pad = 1, dil = 2, groups = 2;
        float[] x = TrainParity.uniform(21, b * c * l, -3f, 3f);
        float[] wt = TrainParity.uniform(22, oc * (c / groups) * kernel, -1f, 1f);
        int[] xd = {b, c, l}, wd = {oc, c / groups, kernel};
        Tensor dev = device(ConvOps.conv1d(upload(gpu, x, xd), upload(gpu, wt, wd), null, kernel, stride, pad, dil,
                groups));
        Tensor cpu = keep(ConvOps.conv1d(upload(host, x, xd), upload(host, wt, wd), null, kernel, stride, pad, dil,
                groups));
        Tensor terms = keep(ConvOps.conv1d(upload(host, abs(x), xd), upload(host, abs(wt), wd), null, kernel,
                stride, pad, dil, groups));
        assertEquals(cpu.shape(), dev.shape());
        assertReductionPair("conv1d", cpu, dev, (c / groups) * kernel, terms.toFloatArray());
    }

    @Test
    public void convTranspose2dDeviceMatchesCpu() {
        int b = 2, c = 4, h = 5, w = 6, groups = 2, ocg = 3, oc = ocg * groups;
        int[] k = {3, 4}, s = {2, 3}, p = {1, 2}, op = {1, 2}, d = {1, 2};
        float[] x = TrainParity.uniform(31, b * c * h * w, -3f, 3f);
        float[] wt = TrainParity.uniform(32, c * ocg * k[0] * k[1], -1f, 1f);
        float[] bias = TrainParity.uniform(33, oc, -1f, 1f);
        int[] xd = {b, c, h, w}, wd = {c, ocg, k[0], k[1]};
        Tensor dev = device(ConvOps.convTranspose2d(upload(gpu, x, xd), upload(gpu, wt, wd), upload(gpu, bias, oc),
                k, s, p, op, d, groups));
        Tensor cpu = keep(ConvOps.convTranspose2d(upload(host, x, xd), upload(host, wt, wd),
                upload(host, bias, oc), k, s, p, op, d, groups));
        Tensor terms = keep(ConvOps.convTranspose2d(upload(host, abs(x), xd), upload(host, abs(wt), wd),
                upload(host, abs(bias), oc), k, s, p, op, d, groups));
        assertEquals(cpu.shape(), dev.shape());
        assertReductionPair("convTranspose2d", cpu, dev, (c / groups) * k[0] * k[1] + 1, terms.toFloatArray());
    }

    // ------------------------------------------------------------------ pooling

    @Test
    public void maxPool2dDeviceIsBitIdentical() {
        int h = 10, w = 9;
        int[] k = {3, 2}, s = {2, 2}, p = {1, 1}, d = {2, 1};
        float[] x = TrainParity.uniform(41, 2 * 3 * h * w, -10f, 10f);
        x[5] = Float.NaN;
        x[100] = Float.POSITIVE_INFINITY;
        x[w + 1] = -0.0f;
        x[w + 2] = 0.0f;
        for (boolean ceil : new boolean[]{false, true}) {
            Tensor dev = device(ConvOps.maxPool2d(upload(gpu, x, 2, 3, h, w), k, s, p, d, ceil));
            Tensor cpu = keep(ConvOps.maxPool2d(upload(host, x, 2, 3, h, w), k, s, p, d, ceil));
            assertEquals(cpu.shape(), dev.shape());
            TrainParity.assertBitIdentical("maxPool2d ceil=" + ceil, cpu.toFloatArray(), dev.toFloatArray());
        }
    }

    @Test
    public void averagePoolingDeviceIsWithinTheDivisionClass() {
        int[] k = {3, 3}, s = {2, 2}, p = {1, 1};
        float[] x = TrainParity.uniform(51, 2 * 3 * 8 * 7, -10f, 10f);
        for (boolean ceil : new boolean[]{false, true}) {
            for (boolean includePad : new boolean[]{false, true}) {
                Tensor dev = device(ConvOps.avgPool2d(upload(gpu, x, 2, 3, 8, 7), k, s, p, ceil, includePad));
                Tensor cpu = keep(ConvOps.avgPool2d(upload(host, x, 2, 3, 8, 7), k, s, p, ceil, includePad));
                assertEquals(cpu.shape(), dev.shape());
                TrainParity.assertDivSqrt("avgPool2d ceil=" + ceil + " includePad=" + includePad,
                        cpu.toFloatArray(), dev.toFloatArray());
            }
        }
        for (int[] size : new int[][]{{3, 4}, {4, 9}, {8, 7}}) {
            Tensor dev = device(ConvOps.adaptiveAvgPool2d(upload(gpu, x, 2, 3, 8, 7), size[0], size[1]));
            Tensor cpu = keep(ConvOps.adaptiveAvgPool2d(upload(host, x, 2, 3, 8, 7), size[0], size[1]));
            assertEquals(cpu.shape(), dev.shape());
            TrainParity.assertDivSqrt("adaptiveAvgPool2d " + size[0] + "x" + size[1], cpu.toFloatArray(),
                    dev.toFloatArray());
        }
    }

    // ------------------------------------------------------------------ normalization

    @Test
    public void rowNormsDeviceMatchCpu() {
        int rows = 7, cols = 300;
        float[] x = TrainParity.uniform(61, rows * cols, -3f, 5f);
        float[] gamma = TrainParity.uniform(62, cols, 0.5f, 1.5f);
        float[] beta = TrainParity.uniform(63, cols, -1f, 1f);
        Tensor dev = device(NormOps.layerNorm(upload(gpu, x, rows, cols), cols, upload(gpu, gamma, cols),
                upload(gpu, beta, cols), EPS));
        Tensor cpu = keep(NormOps.layerNorm(upload(host, x, rows, cols), cols, upload(host, gamma, cols),
                upload(host, beta, cols), EPS));
        IntUnaryOperator row = i -> i / cols;
        IntUnaryOperator col = i -> i % cols;
        assertReductionPair("layerNorm", cpu, dev, cols,
                normTerms(x, row, rows, false, null, null, EPS, gamma, beta, col));

        Tensor devGammaOnly = device(NormOps.layerNorm(upload(gpu, x, rows, cols), cols, upload(gpu, gamma, cols),
                null, EPS));
        Tensor cpuGammaOnly = keep(NormOps.layerNorm(upload(host, x, rows, cols), cols, upload(host, gamma, cols),
                null, EPS));
        assertReductionPair("layerNorm gamma only", cpuGammaOnly, devGammaOnly, cols,
                normTerms(x, row, rows, false, null, null, EPS, gamma, null, col));

        float[] w = TrainParity.uniform(64, cols, 0.25f, 2f);
        Tensor devRms = device(NormOps.rmsNorm(upload(gpu, x, rows, cols), upload(gpu, w, cols), 1e-8f));
        Tensor cpuRms = keep(NormOps.rmsNorm(upload(host, x, rows, cols), upload(host, w, cols), 1e-8f));
        assertReductionPair("rmsNorm", cpuRms, devRms, cols,
                normTerms(x, row, rows, true, null, null, 1e-8f, w, null, col));
    }

    @Test
    public void batchNormDeviceMatchesCpuInBothModes() {
        int b = 3, c = 4, sp = 6 * 5;
        float momentum = 0.1f;
        float[] x = TrainParity.uniform(71, b * c * sp, -2f, 3f);
        float[] gamma = TrainParity.uniform(72, c, 0.5f, 1.5f);
        float[] beta = TrainParity.uniform(73, c, -0.5f, 0.5f);
        float[] mean0 = TrainParity.uniform(74, c, -0.1f, 0.1f);
        float[] var0 = TrainParity.uniform(75, c, 0.9f, 1.1f);
        Tensor devMean = upload(gpu, mean0, c);
        Tensor devVar = upload(gpu, var0, c);
        Tensor cpuMean = upload(host, mean0, c);
        Tensor cpuVar = upload(host, var0, c);
        IntUnaryOperator channel = i -> (i / sp) % c;

        Tensor dev = device(NormOps.batchNorm(upload(gpu, x, b, c, 6, 5), upload(gpu, gamma, c),
                upload(gpu, beta, c), devMean, devVar, true, momentum, EPS));
        assertTrue("grad mode is restored after the running-statistic scope", ComputationGraph.isGradEnabled());
        Tensor cpu = keep(NormOps.batchNorm(upload(host, x, b, c, 6, 5), upload(host, gamma, c),
                upload(host, beta, c), cpuMean, cpuVar, true, momentum, EPS));
        assertReductionPair("batchNorm training", cpu, dev, b * sp,
                normTerms(x, channel, c, false, null, null, EPS, gamma, beta, channel));
        assertRunningPair(x, channel, c, b * sp, momentum, mean0, var0, cpuMean.toFloatArray(),
                devMean.toFloatArray(), cpuVar.toFloatArray(), devVar.toFloatArray());

        float[] evalMean = cpuMean.toFloatArray();
        float[] evalVar = cpuVar.toFloatArray();
        Tensor devEval = device(NormOps.batchNorm(upload(gpu, x, b, c, 6, 5), upload(gpu, gamma, c),
                upload(gpu, beta, c), upload(gpu, evalMean, c), upload(gpu, evalVar, c), false, momentum, EPS));
        Tensor cpuEval = keep(NormOps.batchNorm(upload(host, x, b, c, 6, 5), upload(host, gamma, c),
                upload(host, beta, c), upload(host, evalMean, c), upload(host, evalVar, c), false, momentum, EPS));
        assertReductionPair("batchNorm evaluation", cpuEval, devEval, 1,
                normTerms(x, channel, c, false, toDouble(evalMean), toDouble(evalVar), EPS, gamma, beta, channel));
    }

    @Test
    public void groupNormDeviceMatchesCpu() {
        int b = 2, c = 6, sp = 4 * 3, groups = 3;
        float[] x = TrainParity.uniform(81, b * c * sp, -3f, 3f);
        float[] gamma = TrainParity.uniform(82, c, 0.5f, 1.5f);
        float[] beta = TrainParity.uniform(83, c, -0.5f, 0.5f);
        Tensor dev = device(NormOps.groupNorm(upload(gpu, x, b, c, 4, 3), groups, upload(gpu, gamma, c),
                upload(gpu, beta, c), EPS));
        Tensor cpu = keep(NormOps.groupNorm(upload(host, x, b, c, 4, 3), groups, upload(host, gamma, c),
                upload(host, beta, c), EPS));
        IntUnaryOperator channel = i -> (i / sp) % c;
        IntUnaryOperator group = i -> (i / sp) / c * groups + ((i / sp) % c) / (c / groups);
        assertReductionPair("groupNorm", cpu, dev, (c / groups) * sp,
                normTerms(x, group, b * groups, false, null, null, EPS, gamma, beta, channel));
    }

    @Test
    public void operandsOnAnotherBackendTakeTheCpuPath() {
        int rows = 4, cols = 24;
        float[] x = TrainParity.uniform(91, rows * cols, -2f, 2f);
        float[] gamma = TrainParity.uniform(92, cols, 0.5f, 1.5f);
        float[] beta = TrainParity.uniform(93, cols, -1f, 1f);
        Tensor mixed = device(NormOps.layerNorm(upload(gpu, x, rows, cols), cols, upload(host, gamma, cols),
                upload(host, beta, cols), EPS));
        Tensor cpu = keep(NormOps.layerNorm(upload(host, x, rows, cols), cols, upload(host, gamma, cols),
                upload(host, beta, cols), EPS));
        TrainParity.assertBitIdentical("layerNorm with host parameters", cpu.toFloatArray(), mixed.toFloatArray());
    }

    // ------------------------------------------------------------------ dropout

    @Test
    public void maskScaleDeviceMatchesCpu() {
        float keepProb = 0.6f;
        float scale = 1.0f / keepProb;
        float[] x = TrainParity.uniform(101, 300, -5f, 5f);
        x[7] = Float.NaN;
        x[8] = Float.POSITIVE_INFINITY;
        float[] draws = TrainParity.uniform(102, 300, 0f, 1f);
        float[] scaled = new float[x.length];
        float[] ones = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            scaled[i] = draws[i] < keepProb ? scale : 0.0f;
            ones[i] = draws[i] < keepProb ? 1.0f : 0.0f;
        }
        scaled[7] = 0.0f;
        ones[7] = -0.0f;
        Tensor dev = device(TensorOps.maskScale(upload(gpu, x, 10, 30), upload(gpu, scaled, 10, 30), 1.0f));
        Tensor cpu = keep(TensorOps.maskScale(upload(host, x, 10, 30), upload(host, scaled, 10, 30), 1.0f));
        TrainParity.assertSingleRounding("maskScale", cpu.toFloatArray(), dev.toFloatArray());
        Tensor devScalar = device(TensorOps.maskScale(upload(gpu, x, 10, 30), upload(gpu, ones, 10, 30), scale));
        TrainParity.assertBitIdentical("device mask 1/keep, scale 1 vs mask 1, scale 1/keep", dev.toFloatArray(),
                devScalar.toFloatArray());
    }

    // ------------------------------------------------------------------ assertions and references

    /** {@code |gpu - cpu| <= 2 * (1e-5 * sqrt(K) * terms + K * FLT_MIN)} per element; NaN only against NaN. */
    private static void assertReductionPair(String what, Tensor cpu, Tensor gpu, int k, float[] terms) {
        double[] t = new double[terms.length];
        for (int i = 0; i < terms.length; i++) {
            t[i] = terms[i];
        }
        assertReductionPair(what, cpu, gpu, k, t);
    }

    private static void assertReductionPair(String what, Tensor cpu, Tensor gpu, int k, double[] terms) {
        float[] ref = cpu.toFloatArray();
        float[] got = gpu.toFloatArray();
        assertEquals(what + ": length", ref.length, got.length);
        assertEquals(what + ": term count", ref.length, terms.length);
        for (int i = 0; i < ref.length; i++) {
            pair(what + " element " + i, ref[i], got[i], k, terms[i]);
        }
    }

    private static void pair(String what, float ref, float got, int k, double terms) {
        if (Float.isNaN(ref) || Float.isNaN(got)) {
            if (!(Float.isNaN(ref) && Float.isNaN(got))) {
                fail(what + ": cpu " + ref + ", gpu " + got);
            }
            return;
        }
        double bound = 2 * (1e-5 * Math.sqrt(k) * terms + k * (double) TrainParity.FLT_MIN);
        if (!(Math.abs((double) got - ref) <= bound)) {
            fail(what + ": cpu " + ref + ", gpu " + got + ", bound " + bound);
        }
    }

    /**
     * The running statistics after one training step on both paths: {@code (1 - m) * running + m * batch}, the batch
     * variance being the unbiased one ({@code / (count - 1)}), with the absolute sizes of the running value and of
     * the batch statistic's terms as the term sum.
     */
    private static void assertRunningPair(float[] x, IntUnaryOperator channelOf, int channels, int count,
                                          float momentum, float[] mean0, float[] var0, float[] cpuMean,
                                          float[] gpuMean, float[] cpuVar, float[] gpuVar) {
        double[] mean = new double[channels];
        double[] abs = new double[channels];
        for (int i = 0; i < x.length; i++) {
            mean[channelOf.applyAsInt(i)] += x[i];
            abs[channelOf.applyAsInt(i)] += Math.abs(x[i]);
        }
        double[] sq = new double[channels];
        for (int i = 0; i < x.length; i++) {
            int ch = channelOf.applyAsInt(i);
            double span = Math.abs(x[i]) + Math.abs(mean[ch] / count);
            sq[ch] += span * span;
        }
        for (int ch = 0; ch < channels; ch++) {
            pair("running mean " + ch, cpuMean[ch], gpuMean[ch], count,
                    (1 - momentum) * Math.abs(mean0[ch]) + momentum * abs[ch] / count);
            pair("running var " + ch, cpuVar[ch], gpuVar[ch], count,
                    (1 - momentum) * Math.abs(var0[ch]) + momentum * sq[ch] / (count - 1));
        }
    }

    /**
     * Per-element term sums {@code (|x| + |center|) / denom * |g| + |b|} of a normalization whose element {@code i}
     * belongs to statistics set {@code setOf(i)} and reads affine index {@code paramOf(i)}: the double-precision
     * mean and {@code sqrt(var + eps)} of each set ({@code givenMean}/{@code givenVar} when not null), or for
     * {@code rms} the centre 0 and {@code sqrt(mean(x^2) + eps)}.
     */
    private static double[] normTerms(float[] x, IntUnaryOperator setOf, int sets, boolean rms, double[] givenMean,
                                      double[] givenVar, float eps, float[] g, float[] b, IntUnaryOperator paramOf) {
        double[] mean = new double[sets];
        double[] var = new double[sets];
        double[] sq = new double[sets];
        int[] count = new int[sets];
        for (int i = 0; i < x.length; i++) {
            int k = setOf.applyAsInt(i);
            mean[k] += x[i];
            sq[k] += (double) x[i] * x[i];
            count[k]++;
        }
        for (int k = 0; k < sets; k++) {
            mean[k] /= Math.max(1, count[k]);
        }
        for (int i = 0; i < x.length; i++) {
            int k = setOf.applyAsInt(i);
            var[k] += (x[i] - mean[k]) * (x[i] - mean[k]);
        }
        double[] terms = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            int k = setOf.applyAsInt(i);
            int n = Math.max(1, count[k]);
            double center = rms ? 0 : givenMean != null ? givenMean[k] : mean[k];
            double denom = rms ? Math.sqrt(sq[k] / n + eps)
                    : Math.sqrt((givenVar != null ? givenVar[k] : var[k] / n) + eps);
            int j = paramOf.applyAsInt(i);
            double scale = g == null ? 1 : Math.abs(g[j]);
            double shift = b == null ? 0 : Math.abs(b[j]);
            terms[i] = (Math.abs(x[i]) + Math.abs(center)) / denom * scale + shift;
        }
        return terms;
    }

    private static double[] toDouble(float[] v) {
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i];
        }
        return out;
    }

    private static float[] abs(float[] v) {
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = Math.abs(v[i]);
        }
        return out;
    }

    // ------------------------------------------------------------------ tensors

    /** A facade result computed on the device backend: kept for closing, on {@code gpu}, without requiresGrad. */
    private Tensor device(Tensor t) {
        assertNotNull(t);
        keep(t);
        assertSame("result backend", gpu, t.backend());
        assertEquals(DType.FLOAT32, t.dtype());
        assertFalse("raw results do not require grad", t.requiresGrad());
        return t;
    }

    private Tensor upload(ComputeBackend b, float[] data, int... dims) {
        return keep(keep(Tensor.fromArray(b, data)).reshape(Shape.of(dims)));
    }

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }
}
