package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.precision.Quantizer.CalibrationMethod;
import com.eonmux.neuralfree.precision.Quantizer.QuantizationParams;
import com.eonmux.neuralfree.precision.Quantizer.QuantizationType;
import com.eonmux.neuralfree.precision.QuantizerTest.Weight;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link Quantizer} calibration methods on hand-built distributions whose clip ranges are known exactly or
 * bracketed analytically.
 */
public class QuantizerCalibrationTest {
    private ComputeBackend host;

    @Before
    public void setUp() {
        host = TrainParity.hostBackend();
    }

    /** PERCENTILE at 100 clips nowhere, so it derives the MIN_MAX parameters exactly. */
    @Test
    public void percentile100MatchesMinMax() {
        float[] values = new float[999];
        Random random = new Random(11);
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) random.nextGaussian() * 3f + 0.5f;
        }
        for (QuantizationType type : new QuantizationType[]{QuantizationType.INT8_SYMMETRIC,
                QuantizationType.INT8_ASYMMETRIC, QuantizationType.INT4_ASYMMETRIC, QuantizationType.FP8_E4M3}) {
            QuantizationParams minMax = params(builder(type, CalibrationMethod.MIN_MAX), values);
            QuantizationParams percentile = params(builder(type, CalibrationMethod.PERCENTILE).percentile(100f), values);
            assertSameParams(type.name(), minMax, percentile);
        }
    }

    /**
     * Magnitudes 1..1000 (bins of width 0.5): the 99th percentile of {@code |x|} lies in the bin of 990, interpolated
     * to 990.5.
     */
    @Test
    public void symmetricPercentileClipsMagnitudes() {
        float[] values = new float[1000];
        for (int k = 1; k <= 1000; k++) {
            values[k - 1] = (k % 2 == 0 ? 1f : -1f) * k;
        }
        QuantizationParams p = params(builder(QuantizationType.INT8_SYMMETRIC, CalibrationMethod.PERCENTILE)
            .percentile(99f), values);
        float threshold = p.scale() * 127;
        assertTrue("threshold " + threshold, threshold >= 990f && threshold <= 991f);
        assertEquals(0, p.zeroPoint());
    }

    /** Dense uniform values on {@code [-1, 3)}: the 1st and 99th percentiles are -0.96 and 2.96. */
    @Test
    public void asymmetricPercentileClipsBothTails() {
        QuantizationParams p = params(builder(QuantizationType.INT8_ASYMMETRIC, CalibrationMethod.PERCENTILE)
            .percentile(99f), uniform(-1.0, 4.0, 20480));
        float range = p.scale() * 255;
        assertTrue("range " + range, range >= 3.91f && range <= 3.93f);
        assertEquals(-66, p.zeroPoint());
    }

    /**
     * Uniform magnitudes on {@code (0, 1)} with INT4 symmetric codes: the expected error
     * {@code t^3 / (12 * 49) + (1 - t)^3 / 3} is smallest at {@code t = 14 / 15} and at least 3% larger outside
     * {@code [0.90, 0.96]}; the clipped codes have a smaller error on the data than the min-max codes.
     */
    @Test
    public void symmetricMseClipsUniformInt4() {
        float[] values = signedUniform(8192);
        QuantizationParams p = params(builder(QuantizationType.INT4_SYMMETRIC, CalibrationMethod.MSE), values);
        float threshold = p.scale() * 7;
        assertTrue("threshold " + threshold, threshold >= 0.90f && threshold <= 0.96f);
        assertTrue(error(builder(QuantizationType.INT4_SYMMETRIC, CalibrationMethod.MSE), values)
            < error(builder(QuantizationType.INT4_SYMMETRIC, CalibrationMethod.MIN_MAX), values));
    }

    /**
     * FP8 MSE uses the FP8 codec in the error: on uniform magnitudes the rounding error grows as {@code t^3} while the
     * clipping error starts cubic in {@code 1 - t}, so the best threshold lies below amax and lowers the error.
     */
    @Test
    public void fp8MseClipsBelowAmax() {
        float[] values = signedUniform(8192);
        QuantizationParams minMax = params(builder(QuantizationType.FP8_E4M3, CalibrationMethod.MIN_MAX), values);
        QuantizationParams mse = params(builder(QuantizationType.FP8_E4M3, CalibrationMethod.MSE), values);
        assertEquals(0, mse.zeroPoint());
        assertTrue(mse.scale() < minMax.scale());
        assertTrue(error(builder(QuantizationType.FP8_E4M3, CalibrationMethod.MSE), values)
            < error(builder(QuantizationType.FP8_E4M3, CalibrationMethod.MIN_MAX), values));
    }

    /** Asymmetric MSE narrows the uniform range {@code [-1, 3)} for INT4 codes and lowers the error on the data. */
    @Test
    public void asymmetricMseNarrowsUniformInt4() {
        float[] values = uniform(-1.0, 4.0, 20480);
        QuantizationParams minMax = params(builder(QuantizationType.INT4_ASYMMETRIC, CalibrationMethod.MIN_MAX), values);
        QuantizationParams mse = params(builder(QuantizationType.INT4_ASYMMETRIC, CalibrationMethod.MSE), values);
        assertTrue(mse.scale() < minMax.scale());
        assertTrue(error(builder(QuantizationType.INT4_ASYMMETRIC, CalibrationMethod.MSE), values)
            < error(builder(QuantizationType.INT4_ASYMMETRIC, CalibrationMethod.MIN_MAX), values));
    }

    /**
     * Exactly 4 values per bin over all 2048 bins: the full range is the only candidate whose expanded distribution
     * equals the reference (divergence 0); every narrower one folds outliers into its last bin. ENTROPY keeps amax.
     */
    @Test
    public void symmetricEntropyKeepsADenseUniformRange() {
        float[] values = signedUniform(8192);
        assertSameParams("INT8_SYMMETRIC",
            params(builder(QuantizationType.INT8_SYMMETRIC, CalibrationMethod.MIN_MAX), values),
            params(builder(QuantizationType.INT8_SYMMETRIC, CalibrationMethod.ENTROPY), values));
    }

    /**
     * Bins of width {@code 2^-11}: bins 0..127 hold {@code 128 - j} values and one outlier sits in bin 2047. Every
     * candidate between 128 and 2048 bins folds the outlier into an empty bin (infinite divergence); keeping 128 bins
     * (one bin per level) diverges by about {@code 4.7e-5}, keeping all 2048 by more than {@code 2.7e-3} (the linear
     * counts 16..1 of bins 112..127 against their group mean). The threshold is {@code 128 * 2^-11 = 0.0625}.
     */
    @Test
    public void symmetricEntropyClipsASparseTail() {
        QuantizationParams p = params(builder(QuantizationType.INT8_SYMMETRIC, CalibrationMethod.ENTROPY),
            linearBulkWithOutlier(true));
        assertEquals(0.0625f / 127, p.scale(), 0f);
        assertEquals(0, p.zeroPoint());
    }

    /** As the symmetric case with 256 levels over the signed histogram: dense uniform data keeps the min-max range. */
    @Test
    public void asymmetricEntropyKeepsADenseUniformRange() {
        float[] values = uniform(-1.0, 4.0, 20480);
        assertSameParams("INT8_ASYMMETRIC",
            params(builder(QuantizationType.INT8_ASYMMETRIC, CalibrationMethod.MIN_MAX), values),
            params(builder(QuantizationType.INT8_ASYMMETRIC, CalibrationMethod.ENTROPY), values));
    }

    /**
     * The sparse-tail distribution with non-negative values: the range must start at the zero edge, and with 256
     * levels 128 kept bins are again one bin per group, so the range is {@code [0, 0.0625]}.
     */
    @Test
    public void asymmetricEntropyClipsASparseTail() {
        QuantizationParams p = params(builder(QuantizationType.INT8_ASYMMETRIC, CalibrationMethod.ENTROPY),
            linearBulkWithOutlier(false));
        assertEquals(0.0625f / 255, p.scale(), 0f);
        assertEquals(-128, p.zeroPoint());
    }

    /** Histograms merge exactly, so any order of calibration calls gives the parameters of all the data at once. */
    @Test
    public void repeatedCalibrationDoesNotDependOnOrder() {
        float[] narrow = uniform(-0.5, 1.0, 3000);
        float[] wide = uniform(-2.0, 7.0, 5000);
        float[] both = new float[narrow.length + wide.length];
        System.arraycopy(narrow, 0, both, 0, narrow.length);
        System.arraycopy(wide, 0, both, narrow.length, wide.length);
        List<Quantizer.Builder> builders = new ArrayList<>();
        builders.add(builder(QuantizationType.INT8_SYMMETRIC, CalibrationMethod.PERCENTILE).percentile(99f));
        builders.add(builder(QuantizationType.INT4_ASYMMETRIC, CalibrationMethod.MSE));
        builders.add(builder(QuantizationType.INT8_SYMMETRIC, CalibrationMethod.ENTROPY));
        builders.add(builder(QuantizationType.FP8_E5M2, CalibrationMethod.PERCENTILE).percentile(99.5f));
        for (Quantizer.Builder b : builders) {
            QuantizationParams forward = calibrated(b, narrow, wide);
            QuantizationParams backward = calibrated(b, wide, narrow);
            QuantizationParams atOnce = calibrated(b, both);
            assertSameParams("narrow then wide", atOnce, forward);
            assertSameParams("wide then narrow", atOnce, backward);
        }
    }

    /**
     * A parameter of 1000 values {@code 0..999} with at most 10 samples contributes the values at indices
     * {@code floor(i * 1000 / 10)}: 0, 100, ..., 900, so PERCENTILE at 100 clips at 900.
     */
    @Test
    public void numCalibrationSamplesBoundsTheHistogrammedValues() {
        float[] values = new float[1000];
        for (int k = 0; k < values.length; k++) {
            values[k] = k;
        }
        Quantizer.Builder sampled = builder(QuantizationType.INT8_SYMMETRIC, CalibrationMethod.PERCENTILE)
            .percentile(100f).numCalibrationSamples(10);
        assertEquals(900f / 127, params(sampled, values).scale(), 0f);
        Quantizer.Builder all = builder(QuantizationType.INT8_SYMMETRIC, CalibrationMethod.PERCENTILE).percentile(100f);
        assertEquals(999f / 127, params(all, values).scale(), 0f);
        assertArrayEquals(new int[]{0, 100, 200, 300, 400, 500, 600, 700, 800, 900}, PtqCalibrator.sampleIndices(1000, 10));
        assertArrayEquals(new int[]{0, 3, 6}, PtqCalibrator.sampleIndices(10, 3));
    }

    /**
     * Per-channel calibration of a {@code [2, n]} parameter whose rows have different ranges: with every method, each
     * channel gets exactly the parameters of a per-tensor calibration of its row alone.
     */
    @Test
    public void perChannelCalibratesEachRowOnItsOwn() {
        float[][] rows = {uniform(-0.5, 1.0, 2048), uniform(-3.0, 10.0, 2048)};
        for (QuantizationType type : QuantizationType.values()) {
            for (CalibrationMethod method : CalibrationMethod.values()) {
                if (Quantizer.isFp8(type) && method == CalibrationMethod.ENTROPY) {
                    continue;
                }
                QuantizationParams perChannel = channelParams(builder(type, method).percentile(99f)
                    .perChannel(true), rows);
                assertTrue(perChannel.isPerChannel());
                assertEquals(2, perChannel.channels());
                for (int c = 0; c < rows.length; c++) {
                    QuantizationParams row = params(builder(type, method).percentile(99f), rows[c]);
                    String label = type + " " + method + " channel " + c;
                    assertEquals(label + " scale", row.scale(), perChannel.scale(c), 0f);
                    assertEquals(label + " zero point", row.zeroPoint(), perChannel.zeroPoint(c));
                    assertEquals(label + " code min", row.quantMin(), perChannel.quantMin());
                    assertEquals(label + " code max", row.quantMax(), perChannel.quantMax());
                }
            }
        }
    }

    /**
     * With at most 20 samples, each of the 2 rows of 1000 values contributes the 10 values at
     * {@code floor(i * 1000 / 10)}: row 0 holds {@code 0..999} and clips at 900, row 1 holds {@code -2k} and clips at
     * 1800 (PERCENTILE at 100).
     */
    @Test
    public void perChannelSamplesEveryRow() {
        float[][] rows = new float[2][1000];
        for (int k = 0; k < 1000; k++) {
            rows[0][k] = k;
            rows[1][k] = -2f * k;
        }
        QuantizationParams p = channelParams(builder(QuantizationType.INT8_SYMMETRIC, CalibrationMethod.PERCENTILE)
            .percentile(100f).numCalibrationSamples(20).perChannel(true), rows);
        assertEquals(900f / 127, p.scale(0), 0f);
        assertEquals(1800f / 127, p.scale(1), 0f);
        assertArrayEquals(new int[]{0, 100, 200, 300, 400, 500, 600, 700, 800, 900,
            1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900}, PtqCalibrator.rowSampleIndices(2000, 2, 20));
        assertArrayEquals(PtqCalibrator.sampleIndices(1000, 10), PtqCalibrator.rowSampleIndices(1000, 1, 10));
    }

    /** A parameter calibrated per channel rejects an observation with another channel count, naming it. */
    @Test
    public void perChannelRejectsAChangedChannelCount() {
        for (CalibrationMethod method : new CalibrationMethod[]{CalibrationMethod.MIN_MAX, CalibrationMethod.MSE}) {
            Quantizer quantizer = builder(QuantizationType.INT8_SYMMETRIC, method).perChannel(true).build();
            Module two = new Rows(host, new float[][]{{1f, 2f}, {3f, 4f}});
            Module three = new Rows(host, new float[][]{{1f}, {2f}, {3f}});
            try {
                quantizer.calibrate(two);
                quantizer.calibrate(three);
                throw new AssertionError("a changed channel count was accepted");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("'weight'"));
            } finally {
                two.close();
                three.close();
            }
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void fp8RejectsEntropy() {
        builder(QuantizationType.FP8_E4M3, CalibrationMethod.ENTROPY).build();
    }

    @Test(expected = IllegalArgumentException.class)
    public void asymmetricPercentileNeedsAtLeastFifty() {
        builder(QuantizationType.INT8_ASYMMETRIC, CalibrationMethod.PERCENTILE).percentile(40f).build();
    }

    /** A histogram method rejects an infinite value, naming the parameter, and leaves the quantizer uncalibrated. */
    @Test
    public void histogramCalibrationRejectsInfiniteValues() {
        Quantizer quantizer = builder(QuantizationType.INT8_SYMMETRIC, CalibrationMethod.MSE).build();
        Module model = new Weight(host, "weight", "weight", new float[]{1f, Float.POSITIVE_INFINITY});
        try {
            quantizer.calibrate(model);
            throw new AssertionError("an infinite value was accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("'weight'"));
        } finally {
            model.close();
        }
        assertFalse(quantizer.isCalibrated());
    }

    @Test
    public void reportsItsSettings() {
        Quantizer quantizer = builder(QuantizationType.INT4_ASYMMETRIC, CalibrationMethod.MSE)
            .percentile(97f).numCalibrationSamples(64).build();
        assertEquals(CalibrationMethod.MSE, quantizer.calibrationMethod());
        assertEquals(97f, quantizer.percentile(), 0f);
        assertEquals(64, quantizer.numCalibrationSamples());
        assertEquals(Integer.MAX_VALUE, Quantizer.builder().build().numCalibrationSamples());
        assertFalse(Quantizer.builder().build().perChannel());
        assertTrue(Quantizer.builder().perChannel(true).build().perChannel());
    }

    /** A module with one parameter {@code "weight"} of shape {@code [rows, cols]}. */
    private static final class Rows extends Module {
        Rows(ComputeBackend backend, float[][] values) {
            registerParameter("weight", new Parameter(Tensor.fromArray(backend, values), "weight"));
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    /** The parameters a fresh quantizer from {@code builder} derives for one {@code [rows, cols]} parameter. */
    private QuantizationParams channelParams(Quantizer.Builder builder, float[][] values) {
        Quantizer quantizer = builder.build();
        Module model = new Rows(host, values);
        try {
            quantizer.quantizeModel(model);
            return quantizer.getQuantParams("weight");
        } finally {
            model.close();
        }
    }

    private static Quantizer.Builder builder(QuantizationType type, CalibrationMethod method) {
        return Quantizer.builder().quantType(type).calibrationMethod(method);
    }

    /** The parameters a fresh quantizer from {@code builder} derives for a single parameter holding {@code values}. */
    private QuantizationParams params(Quantizer.Builder builder, float[] values) {
        return calibrated(builder, values);
    }

    /** Calibrates a fresh quantizer on each value set in turn (one parameter path), then quantizes the first set. */
    private QuantizationParams calibrated(Quantizer.Builder builder, float[]... observations) {
        Quantizer quantizer = builder.build();
        for (float[] values : observations) {
            Module model = new Weight(host, "weight", "weight", values);
            quantizer.calibrate(model);
            model.close();
        }
        Module first = new Weight(host, "weight", "weight", observations[0]);
        try {
            quantizer.quantizeModel(first);
            return quantizer.getQuantParams("weight");
        } finally {
            first.close();
        }
    }

    /** {@code sum((x - dequantize(quantize(x)))^2)} over {@code values} with a fresh quantizer from {@code builder}. */
    private double error(Quantizer.Builder builder, float[] values) {
        Quantizer quantizer = builder.build();
        Module model = new Weight(host, "weight", "weight", values);
        try (Tensor restored = quantizer.dequantizeTensor(quantizer.quantizeModel(model).get("weight"))) {
            float[] reconstructed = restored.toFloatArray();
            double sum = 0.0;
            for (int i = 0; i < values.length; i++) {
                double d = (double) values[i] - reconstructed[i];
                sum += d * d;
            }
            return sum;
        } finally {
            model.close();
        }
    }

    private static void assertSameParams(String label, QuantizationParams expected, QuantizationParams actual) {
        assertEquals(label + " scale", expected.scale(), actual.scale(), 0f);
        assertEquals(label + " zero point", expected.zeroPoint(), actual.zeroPoint());
        assertEquals(label + " code min", expected.quantMin(), actual.quantMin());
        assertEquals(label + " code max", expected.quantMax(), actual.quantMax());
    }

    /** {@code n} evenly spaced values {@code low + (k + 0.5) * span / n}, none on a power-of-two grid edge. */
    private static float[] uniform(double low, double span, int n) {
        float[] values = new float[n];
        for (int k = 0; k < n; k++) {
            values[k] = (float) (low + (k + 0.5) * span / n);
        }
        return values;
    }

    /** Magnitudes {@code (k + 0.5) / n} of alternating sign: {@code n / 2048} values in each of 2048 bins. */
    private static float[] signedUniform(int n) {
        float[] values = new float[n];
        for (int k = 0; k < n; k++) {
            values[k] = (float) ((k % 2 == 0 ? 1.0 : -1.0) * (k + 0.5) / n);
        }
        return values;
    }

    /**
     * {@code 128 - j} values at the center {@code (j + 0.5) * 2^-11} of bin {@code j} for {@code j < 128}
     * (alternating in sign when {@code signed}) and one value at the center of bin 2047.
     */
    private static float[] linearBulkWithOutlier(boolean signed) {
        List<Float> values = new ArrayList<>();
        for (int j = 0; j < 128; j++) {
            for (int copy = 0; copy < 128 - j; copy++) {
                float magnitude = (j + 0.5f) / 2048f;
                values.add(signed && copy % 2 == 1 ? -magnitude : magnitude);
            }
        }
        values.add(2047.5f / 2048f);
        float[] array = new float[values.size()];
        for (int i = 0; i < array.length; i++) {
            array[i] = values.get(i);
        }
        return array;
    }
}
