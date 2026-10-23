package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.AdaptiveAvgPool2d;
import com.eonmux.neuralfree.nn.layers.AvgPool2d;
import com.eonmux.neuralfree.nn.layers.Conv1d;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.layers.ConvTranspose2d;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.nn.layers.MaxPool2d;
import com.eonmux.neuralfree.nn.norm.BatchNorm1d;
import com.eonmux.neuralfree.nn.norm.BatchNorm2d;
import com.eonmux.neuralfree.nn.norm.GroupNorm;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.nn.norm.RMSNorm;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The CPU references of {@link ConvOps}, {@link NormOps} and {@link TensorOps#maskScale}, on a host backend where
 * the train library is unavailable.
 * <ul>
 *   <li>Pooling against independent definitions of PyTorch's windows: output sizes ({@code pooling_output_shape},
 *       ceil mode dropping a last window that would start inside the right padding), max pooling bit-identical (a
 *       NaN tap propagates, +0 ranks above -0, a window without input elements yields {@code -inf}), average pooling
 *       with the padded-extent divisor ({@code count_include_pad}) or the in-bounds count, adaptive windows
 *       {@code [floor(o*L/O), ceil((o+1)*L/O))} (overlapping when {@code L % O != 0}, and for {@code O > L}); the
 *       forwards and the average and adaptive backward in the reduction class against double-precision sums.</li>
 *   <li>The nn modules route through these facades with their own configuration: each module forward is
 *       bit-identical to the facade called with the module's parameters and settings, including the running
 *       statistics BatchNorm updates and Dropout's mask. Convolution and normalization values are checked against
 *       double-precision definitions by {@code LayersParityTest.CpuPath} and {@code NormModulesParityTest}.</li>
 *   <li>The kernel formula of {@code maskScale}, and the argument checks both paths share.</li>
 * </ul>
 * Comparisons that pass module parameters run under {@link ComputationGraph#noGrad()}, so they compare values only.
 */
public class LayerOpsCpuReferenceTest {

    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();
    private final List<Module> modules = new ArrayList<>();

    @Before
    public void setUp() {
        host = TrainParity.hostBackend();
        assertNull("the host backend has no train library", TrainGpu.forBackend(host));
    }

    @After
    public void tearDown() {
        try {
            for (int i = owned.size() - 1; i >= 0; i--) {
                owned.get(i).close();
            }
            for (Module m : modules) {
                m.close();
            }
        } finally {
            TrainGpu.release(host);
        }
    }

    // ------------------------------------------------------------------ convolutions

    @Test
    public void conv2dModuleForwardsThroughTheFacade() {
        untracked(() -> {
            Conv2d grouped = module(new Conv2d(6, 4, new int[]{3, 2}, new int[]{2, 1}, new int[]{1, 1},
                    new int[]{1, 2}, 2, true));
            grouped.initialize(host, new Random(5));
            store(grouped.bias().tensor(), TrainParity.uniform(6, 4, -1f, 1f));
            Tensor x = floats(TrainParity.uniform(7, 2 * 6 * 9 * 8, -3f, 3f), 2, 6, 9, 8);
            expectSame("conv2d grouped", keep(grouped.forward(x)), keep(ConvOps.conv2d(x, grouped.weight().tensor(),
                    grouped.bias().tensor(), grouped.kernelSize(), grouped.stride(), grouped.padding(),
                    grouped.dilation(), grouped.groups())));

            Conv2d plain = module(new Conv2d(3, 5, new int[]{2, 3}, new int[]{1, 2}, new int[]{0, 2},
                    new int[]{2, 1}, 1, false));
            plain.initialize(host, new Random(8));
            Tensor y = floats(TrainParity.uniform(9, 3 * 7 * 6, -2f, 2f), 1, 3, 7, 6);
            expectSame("conv2d without bias", keep(plain.forward(y)), keep(ConvOps.conv2d(y, plain.weight().tensor(),
                    null, plain.kernelSize(), plain.stride(), plain.padding(), plain.dilation(), plain.groups())));
        });
    }

    @Test
    public void conv1dModuleForwardsThroughTheFacade() {
        untracked(() -> {
            Conv1d conv = module(new Conv1d(4, 6, 3, 2, 1, 2, 2, true));
            conv.initialize(host, new Random(11));
            store(conv.bias().tensor(), TrainParity.uniform(12, 6, -1f, 1f));
            Tensor x = floats(TrainParity.uniform(13, 3 * 4 * 17, -3f, 3f), 3, 4, 17);
            expectSame("conv1d", keep(conv.forward(x)), keep(ConvOps.conv1d(x, conv.weight().tensor(),
                    conv.bias().tensor(), conv.kernelSize(), conv.stride(), conv.padding(), conv.dilation(),
                    conv.groups())));
        });
    }

    @Test
    public void convTranspose2dModuleForwardsThroughTheFacade() {
        untracked(() -> {
            ConvTranspose2d conv = module(new ConvTranspose2d(4, 6, new int[]{3, 4}, new int[]{2, 3}, new int[]{1, 2},
                    new int[]{1, 2}, new int[]{1, 2}, 2, true));
            conv.initialize(host, new Random(21));
            store(conv.bias().tensor(), TrainParity.uniform(22, 6, -1f, 1f));
            Tensor x = floats(TrainParity.uniform(23, 2 * 4 * 5 * 6, -3f, 3f), 2, 4, 5, 6);
            expectSame("convTranspose2d", keep(conv.forward(x)), keep(ConvOps.convTranspose2d(x,
                    conv.weight().tensor(), conv.bias().tensor(), conv.kernelSize(), conv.stride(), conv.padding(),
                    conv.outputPadding(), conv.dilation(), conv.groups())));
        });
    }

    /**
     * The device entry applies PyTorch's transposed-convolution rule itself, before it looks for the train library:
     * an output padding at least both the stride and the dilation is rejected, one below the dilation is accepted
     * (and declined on this host backend).
     */
    @Test
    public void convTranspose2dDeviceEntryRejectsOutputPaddingBeyondStrideAndDilation() {
        Tensor x = floats(new float[4], 1, 1, 2, 2);
        Tensor w = floats(new float[4], 1, 1, 2, 2);
        int[] k = {2, 2};
        int[] one = {1, 1};
        int[] zero = {0, 0};
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> GpuConvOps.convTranspose2dForward(x, w, null, k, one, zero, new int[]{1, 0}, one, 1));
        assertTrue(e.getMessage(), e.getMessage().contains("output padding must be smaller than either stride or "
                + "dilation"));
        assertNull(GpuConvOps.convTranspose2dForward(x, w, null, k, one, zero, new int[]{1, 0}, new int[]{2, 1}, 1));
    }

    @Test
    public void convolutionArgumentsAreCheckedOnTheCpuPath() {
        int[] k = {1, 1}, one = {1, 1}, zero = {0, 0};
        Tensor x = floats(new float[12], 1, 3, 2, 2);
        assertThrows(IllegalArgumentException.class, () -> ConvOps.conv2d(x, floats(new float[4], 2, 2, 1, 1),
                null, k, one, zero, one, 1));
        assertThrows(IllegalArgumentException.class, () -> ConvOps.conv2d(x, floats(new float[3], 1, 3, 1, 1),
                null, new int[]{2, 2}, one, zero, one, 1));
        assertThrows(IllegalArgumentException.class, () -> ConvOps.maxPool2d(x, new int[]{5, 5}, one, zero, one,
                false));
    }

    /**
     * PyTorch's pooling argument checks on the CPU path: padding at most half the (effective) kernel and an output
     * size of at least 1, for the forwards and the backward primitives alike.
     */
    @Test
    public void poolingArgumentsAreCheckedOnTheCpuPath() {
        int[] one = {1, 1}, zero = {0, 0}, two = {2, 2};
        Tensor x = floats(new float[2 * 5 * 5], 1, 2, 5, 5);
        assertThrows(IllegalArgumentException.class, () -> ConvOps.avgPool2d(x, two, one, two, false, true));
        assertThrows(IllegalArgumentException.class, () -> ConvOps.maxPool2d(x, new int[]{3, 3}, one, two, one,
                false));
        assertEquals("padding 2 <= effective kernel 5 / 2", Shape.of(1, 2, 5, 5),
                keep(ConvOps.maxPool2d(x, new int[]{3, 3}, one, two, two, false)).shape());
        Tensor row = floats(new float[4], 1, 1, 1, 4);
        assertThrows(IllegalArgumentException.class, () -> ConvOps.maxPool2d(row, new int[]{2, 1}, one, zero, one,
                false));
        assertThrows(IllegalArgumentException.class, () -> ConvOps.avgPool2d(row, new int[]{3, 1}, new int[]{2, 1},
                zero, true, true));
        assertThrows(IllegalArgumentException.class, () -> ConvOps.avgPool2dBackward(floats(new float[9], 1, 1, 3, 3),
                x.shape(), two, two, two, true, true));
        assertThrows(IllegalArgumentException.class, () -> ConvOps.poolOutputSize(4, 2, 0, 0, 1, false));
        assertThrows(IllegalArgumentException.class, () -> ConvOps.poolOutputSize(-1, 1, 1, 0, 1, false));
    }

    /** The output-size rule against PyTorch's documented examples, including the ceil-mode drop. */
    @Test
    public void poolOutputSizeFollowsPyTorch() {
        assertEquals(3, ConvOps.poolOutputSize(5, 2, 2, 1, 1, false));
        assertEquals(3, ConvOps.poolOutputSize(5, 2, 2, 1, 1, true));
        assertEquals(2, ConvOps.poolOutputSize(5, 1, 3, 0, 1, true));
        assertEquals(4, ConvOps.poolOutputSize(8, 3, 2, 1, 1, false));
        assertEquals(5, ConvOps.poolOutputSize(8, 3, 2, 1, 1, true));
        assertEquals(5, ConvOps.poolOutputSize(9, 2, 2, 1, 1, true));
        assertEquals(1, ConvOps.poolOutputSize(0, 2, 1, 1, 1, false));
        assertEquals(56, ConvOps.poolOutputSize(112, 3, 2, 1, 1, false));
        for (int in = 1; in <= 12; in++) {
            for (int k = 1; k <= 4; k++) {
                for (int st = 1; st <= 4; st++) {
                    for (int d = 1; d <= 2; d++) {
                        for (int p = 0; p <= (d * (k - 1) + 1) / 2; p++) {
                            for (boolean ceil : new boolean[]{false, true}) {
                                int expected = pooledSize(in, k, st, p, d, ceil);
                                if (expected < 1) {
                                    continue;
                                }
                                String what = "in " + in + " k " + k + " s " + st + " p " + p + " d " + d
                                        + " ceil " + ceil;
                                assertEquals(what, expected, ConvOps.poolOutputSize(in, k, st, p, d, ceil));
                                assertTrue(what + ": the last window starts inside the input or the left padding",
                                        (expected - 1) * st - p < in);
                            }
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ pooling

    /**
     * Max pooling over a 10 x 9 input with a dilated 3 x 2 window, stride 2 and padding 1: in ceil mode the height
     * gains a row (5 instead of 4) while the width keeps 5 columns, because its sixth window would start at column
     * 9, inside the right padding, and is dropped.
     */
    @Test
    public void maxPool2dMatchesTheWindowDefinition() {
        int b = 2, c = 3, h = 10, w = 9;
        int[] k = {3, 2}, s = {2, 2}, p = {1, 1}, d = {2, 1};
        float[] data = TrainParity.uniform(31, b * c * h * w, -10f, 10f);
        data[5] = Float.NaN;
        data[100] = Float.POSITIVE_INFINITY;
        data[w + 1] = -0.0f;
        data[w + 2] = 0.0f;
        Tensor x = floats(data, b, c, h, w);
        assertEquals(Shape.of(b, c, 5, 5), keep(ConvOps.maxPool2d(x, k, s, p, d, true)).shape());
        for (boolean ceil : new boolean[]{false, true}) {
            String what = "maxPool2d ceil=" + ceil;
            int oh = pooledSize(h, k[0], s[0], p[0], d[0], ceil);
            int ow = pooledSize(w, k[1], s[1], p[1], d[1], ceil);
            Tensor facade = keep(ConvOps.maxPool2d(x, k, s, p, d, ceil));
            assertEquals(what + ": shape", Shape.of(b, c, oh, ow), facade.shape());
            TrainParity.assertBitIdentical(what, maxPoolDefinition(data, b * c, h, w, oh, ow, k, s, p, d),
                    facade.toFloatArray());
            MaxPool2d pool = module(new MaxPool2d(k, s, p, d, ceil));
            expectSame(what + " module", keep(pool.forward(x)), facade);
        }
    }

    /**
     * Average pooling against PyTorch's windows over two geometries: 8 x 7 with a 3 x 3 window, stride 2 and
     * padding 1, whose last ceil-mode row window reaches past the padded edge (count_include_pad divides by 2 x 3,
     * not 3 x 3); and 5 x 7 with a 2 x 2 window, stride 2 and padding 1, where ceil mode drops the last row and
     * column window.
     */
    @Test
    public void avgPool2dMatchesTheWindowDefinition() {
        int b = 2, c = 3;
        int[][] sizes = {{8, 7}, {5, 7}};
        int[][][] geometry = {{{3, 3}, {2, 2}, {1, 1}}, {{2, 2}, {2, 2}, {1, 1}}};
        for (int g = 0; g < sizes.length; g++) {
            int h = sizes[g][0], w = sizes[g][1];
            int[] k = geometry[g][0], s = geometry[g][1], p = geometry[g][2];
            float[] data = TrainParity.uniform(41 + g, b * c * h * w, -10f, 10f);
            Tensor x = floats(data, b, c, h, w);
            for (boolean ceil : new boolean[]{false, true}) {
                for (boolean includePad : new boolean[]{false, true}) {
                    String what = "avgPool2d " + h + "x" + w + " ceil=" + ceil + " includePad=" + includePad;
                    int oh = pooledSize(h, k[0], s[0], p[0], 1, ceil);
                    int ow = pooledSize(w, k[1], s[1], p[1], 1, ceil);
                    Tensor facade = keep(ConvOps.avgPool2d(x, k, s, p, ceil, includePad));
                    assertEquals(what + ": shape", Shape.of(b, c, oh, ow), facade.shape());
                    float[] got = facade.toFloatArray();
                    int i = 0;
                    for (int plane = 0; plane < b * c; plane++) {
                        for (int r = 0; r < oh; r++) {
                            for (int col = 0; col < ow; col++, i++) {
                                int r0 = r * s[0] - p[0];
                                int c0 = col * s[1] - p[1];
                                int count = includePad
                                        ? paddedOverlap(r0, k[0], h, p[0]) * paddedOverlap(c0, k[1], w, p[1])
                                        : overlap(r0, k[0], h) * overlap(c0, k[1], w);
                                expectWindowMean(what + " [" + i + "]", data, plane, h, w, r0, r0 + k[0], c0,
                                        c0 + k[1], count, got[i]);
                            }
                        }
                    }
                    AvgPool2d pool = module(new AvgPool2d(k, s, p, ceil, includePad));
                    assertEquals(what + ": module shape", facade.shape(), pool.computeOutputShape(x.shape()));
                    expectSame(what + " module", keep(pool.forward(x)), facade);

                    float[] gy = TrainParity.uniform(45 + g, b * c * oh * ow, -1f, 1f);
                    Tensor gx = keep(ConvOps.avgPool2dBackward(floats(gy, b, c, oh, ow), x.shape(), k, s, p, ceil,
                            includePad));
                    expectScatter(what + " backward", gx.toFloatArray(), gy, b * c, h, w, oh, ow,
                            (r, col) -> new int[]{r * s[0] - p[0], r * s[0] - p[0] + k[0],
                                    col * s[1] - p[1], col * s[1] - p[1] + k[1]},
                            (r, col) -> includePad
                                    ? paddedOverlap(r * s[0] - p[0], k[0], h, p[0])
                                    * paddedOverlap(col * s[1] - p[1], k[1], w, p[1])
                                    : overlap(r * s[0] - p[0], k[0], h) * overlap(col * s[1] - p[1], k[1], w));
                }
            }
        }
    }

    /**
     * Adaptive average pooling over PyTorch's windows: output row {@code o} of {@code O} averages input rows
     * {@code [floor(o*H/O), ceil((o+1)*H/O))} (columns likewise). Covers {@code H % O != 0} (3 of 10 rows, 4 of 7
     * columns: neighbouring windows share a row or column), {@code O > L} (9 of 7 columns: each column lies in two
     * or three windows), the identity size and global pooling, for the forward and the backward.
     */
    @Test
    public void adaptiveAvgPool2dMatchesTheWindowDefinition() {
        int b = 2, c = 2, h = 10, w = 7;
        float[] data = TrainParity.uniform(51, b * c * h * w, -10f, 10f);
        Tensor x = floats(data, b, c, h, w);
        for (int[] size : new int[][]{{3, 4}, {4, 9}, {10, 7}, {1, 1}}) {
            String what = "adaptiveAvgPool2d " + size[0] + "x" + size[1];
            Tensor facade = keep(ConvOps.adaptiveAvgPool2d(x, size[0], size[1]));
            assertEquals(what + ": shape", Shape.of(b, c, size[0], size[1]), facade.shape());
            float[] got = facade.toFloatArray();
            int i = 0;
            for (int plane = 0; plane < b * c; plane++) {
                for (int r = 0; r < size[0]; r++) {
                    int r0 = floorDiv(r * h, size[0]);
                    int r1 = ceilDiv((r + 1) * h, size[0]);
                    for (int col = 0; col < size[1]; col++, i++) {
                        int c0 = floorDiv(col * w, size[1]);
                        int c1 = ceilDiv((col + 1) * w, size[1]);
                        expectWindowMean(what + " [" + i + "]", data, plane, h, w, r0, r1, c0, c1,
                                (r1 - r0) * (c1 - c0), got[i]);
                    }
                }
            }
            AdaptiveAvgPool2d pool = module(new AdaptiveAvgPool2d(size));
            expectSame(what + " module", keep(pool.forward(x)), facade);

            int oh = size[0], ow = size[1];
            float[] gy = TrainParity.uniform(52, b * c * oh * ow, -1f, 1f);
            Tensor gx = keep(ConvOps.adaptiveAvgPool2dBackward(floats(gy, b, c, oh, ow), x.shape()));
            expectScatter(what + " backward", gx.toFloatArray(), gy, b * c, h, w, oh, ow,
                    (r, col) -> new int[]{floorDiv(r * h, oh), ceilDiv((r + 1) * h, oh), floorDiv(col * w, ow),
                            ceilDiv((col + 1) * w, ow)},
                    (r, col) -> (ceilDiv((r + 1) * h, oh) - floorDiv(r * h, oh))
                            * (ceilDiv((col + 1) * w, ow) - floorDiv(col * w, ow)));
        }
    }

    // ------------------------------------------------------------------ normalization

    @Test
    public void layerNormModuleForwardsThroughTheFacade() {
        untracked(() -> {
            LayerNorm affine = module(new LayerNorm(new int[]{4, 5}, 1e-4f, true));
            affine.initialize(host);
            store(affine.weight().tensor(), TrainParity.uniform(61, 20, 0.5f, 1.5f));
            store(affine.bias().tensor(), TrainParity.uniform(62, 20, -1f, 1f));
            Tensor x = floats(TrainParity.uniform(63, 3 * 4 * 5, -3f, 5f), 3, 4, 5);
            expectSame("layerNorm", keep(affine.forward(x)), keep(NormOps.layerNorm(x, 20, affine.weight().tensor(),
                    affine.bias().tensor(), affine.epsilon())));

            LayerNorm plain = module(new LayerNorm(5, 1e-5f, false));
            plain.initialize(host);
            expectSame("layerNorm without affine", keep(plain.forward(x)),
                    keep(NormOps.layerNorm(x, 5, null, null, plain.epsilon())));
        });
    }

    @Test
    public void rmsNormModuleForwardsThroughTheFacade() {
        untracked(() -> {
            RMSNorm rms = module(new RMSNorm(12, 1e-6f, true));
            rms.initialize(host);
            store(rms.weight().tensor(), TrainParity.uniform(71, 12, 0.25f, 2f));
            Tensor x = floats(TrainParity.uniform(72, 2 * 3 * 12, -4f, 4f), 2, 3, 12);
            expectSame("rmsNorm", keep(rms.forward(x)), keep(NormOps.rmsNorm(x, rms.weight().tensor(), rms.epsilon())));

            RMSNorm plain = module(new RMSNorm(12, 1e-8f, false));
            plain.initialize(host);
            expectSame("rmsNorm without weight", keep(plain.forward(x)),
                    keep(NormOps.rmsNorm(x, null, plain.epsilon())));
        });
    }

    @Test
    public void batchNorm1dModuleForwardsThroughTheFacadeInBothModes() {
        untracked(() -> {
            BatchNorm1d bn = module(new BatchNorm1d(5, 1e-5f, 0.2f, true));
            bn.initialize(host);
            expectBatchNorm(bn, bn.weight().tensor(), bn.bias().tensor(), bn.runningMean(), bn.runningVar(),
                    bn.momentum(), bn.epsilon(), 81, 3, 5, 7);
            BatchNorm1d flat = module(new BatchNorm1d(4, 1e-3f, 1.0f, false));
            flat.initialize(host);
            expectBatchNorm(flat, null, null, flat.runningMean(), flat.runningVar(), flat.momentum(), flat.epsilon(),
                    82, 6, 4);
        });
    }

    @Test
    public void batchNorm2dModuleForwardsThroughTheFacadeInBothModes() {
        untracked(() -> {
            BatchNorm2d bn = module(new BatchNorm2d(3));
            bn.initialize(host);
            expectBatchNorm(bn, bn.weight().tensor(), bn.bias().tensor(), bn.runningMean(), bn.runningVar(),
                    bn.momentum(), bn.epsilon(), 91, 2, 3, 4, 5);
        });
    }

    /**
     * PyTorch's {@code _BatchNorm} running update on the CPU path: the one-channel batch {@code [0, 2]} is normalized
     * with its biased variance 1, while {@code running_var} blends in its unbiased variance 2, giving
     * {@code 0.9 * 1 + 0.1 * 2 = 1.1}. A training batch with one value per channel has no unbiased variance and is
     * rejected before the running statistics change.
     */
    @Test
    public void batchNormRunningVarianceIsUnbiasedAndSingleValueChannelsAreRejected() {
        Tensor mean = floats(new float[]{0f}, 1);
        Tensor var = floats(new float[]{1f}, 1);
        Tensor out = keep(NormOps.batchNorm(floats(new float[]{0f, 2f}, 2, 1), null, null, mean, var, true, 0.1f,
                1e-5f));

        float scale = (float) (1 / Math.sqrt(1 + 1e-5));
        assertEquals(-scale, out.toFloatArray()[0], 1e-6f);
        assertEquals(scale, out.toFloatArray()[1], 1e-6f);
        assertEquals(0.1f, mean.toFloatArray()[0], 1e-7f);
        assertEquals(1.1f, var.toFloatArray()[0], 1e-6f);

        Tensor single = floats(new float[]{1f, 2f, 3f}, 1, 3);
        Tensor singleMean = floats(new float[]{0.5f, 0.5f, 0.5f}, 3);
        Tensor singleVar = floats(new float[]{2f, 2f, 2f}, 3);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> NormOps.batchNorm(single, null, null, singleMean, singleVar, true, 0.1f, 1e-5f));
        assertTrue(e.getMessage(), e.getMessage().contains("more than 1 value per channel"));
        TrainParity.assertBitIdentical("running mean untouched", new float[]{0.5f, 0.5f, 0.5f},
                singleMean.toFloatArray());
        TrainParity.assertBitIdentical("running var untouched", new float[]{2f, 2f, 2f}, singleVar.toFloatArray());
        keep(NormOps.batchNorm(single, null, null, singleMean, singleVar, false, 0.1f, 1e-5f));
    }

    @Test
    public void groupNormModuleForwardsThroughTheFacade() {
        untracked(() -> {
            GroupNorm gn = module(new GroupNorm(3, 6, 1e-5f, true));
            gn.initialize(host);
            store(gn.weight().tensor(), TrainParity.uniform(101, 6, 0.5f, 1.5f));
            store(gn.bias().tensor(), TrainParity.uniform(102, 6, -0.5f, 0.5f));
            Tensor x = floats(TrainParity.uniform(103, 2 * 6 * 4 * 3, -3f, 3f), 2, 6, 4, 3);
            expectSame("groupNorm", keep(gn.forward(x)), keep(NormOps.groupNorm(x, 3, gn.weight().tensor(),
                    gn.bias().tensor(), gn.epsilon())));

            GroupNorm plain = module(new GroupNorm(2, 6, 1e-5f, false));
            plain.initialize(host);
            expectSame("groupNorm without affine", keep(plain.forward(x)),
                    keep(NormOps.groupNorm(x, 2, null, null, plain.epsilon())));
        });
    }

    @Test
    public void normArgumentsAreCheckedOnTheCpuPath() {
        Tensor x = floats(new float[24], 2, 4, 3);
        Tensor four = floats(new float[4], 4);
        assertThrows(IllegalArgumentException.class, () -> NormOps.layerNorm(x, 5, null, null, 1e-5f));
        assertThrows(IllegalArgumentException.class, () -> NormOps.layerNorm(x, 3, four, null, 1e-5f));
        assertThrows(IllegalArgumentException.class, () -> NormOps.rmsNorm(x, four, 1e-5f));
        assertThrows(IllegalArgumentException.class, () -> NormOps.groupNorm(x, 3, null, null, 1e-5f));
        assertThrows(IllegalArgumentException.class, () -> NormOps.groupNorm(x, 2, four, null, 1e-5f));
        assertThrows(IllegalArgumentException.class, () -> NormOps.batchNorm(x, null, null, four, null, true,
                0.1f, 1e-5f));
        assertThrows(NullPointerException.class, () -> NormOps.batchNorm(x, null, null, null, null, false,
                0.1f, 1e-5f));
        Tensor broadcast = keep(floats(new float[1], 1, 1).expand(Shape.of(1, 4)));
        assertThrows(IllegalArgumentException.class, () -> NormOps.batchNorm(x, null, null, broadcast, four, true,
                0.1f, 1e-5f));
    }

    @Test
    public void batchNormWithoutRunningStatsInTrainingModeUpdatesNothing() {
        float[] data = TrainParity.uniform(111, 3 * 2 * 4, -1f, 1f);
        Tensor x = floats(data, 3, 2, 4);
        Tensor out = keep(NormOps.batchNorm(x, null, null, null, null, true, 0.1f, 1e-5f));
        assertEquals(x.shape(), out.shape());
        assertTrue("grad mode is restored after the running-statistic scope", ComputationGraph.isGradEnabled());
    }

    // ------------------------------------------------------------------ dropout

    @Test
    public void dropoutModuleForwardsThroughMaskScale() {
        Dropout dropout = module(new Dropout(0.3f));
        dropout.setSeed(7);
        dropout.train();
        float[] data = TrainParity.uniform(121, 4 * 25, -5f, 5f);
        data[3] = Float.NaN;
        data[17] = Float.NEGATIVE_INFINITY;
        data[40] = -0.0f;
        Tensor x = floats(data, 4, 25);
        Tensor out = keep(dropout.forward(x));
        expectSame("dropout", out, keep(TensorOps.maskScale(x, dropout.getLastMask(), 1.0f)));
    }

    @Test
    public void maskScaleUsesTheKernelFormula() {
        float nan = Float.NaN;
        float inf = Float.POSITIVE_INFINITY;
        float[] xs = {1.5f, nan, inf, -2f, 3f, -0.0f, 7f, 1e-30f};
        float[] ms = {0.0f, -0.0f, 0.0f, 2f, nan, 4f, 0.5f, 1e-30f};
        float scale = 1.25f;
        float[] expected = new float[xs.length];
        for (int i = 0; i < xs.length; i++) {
            expected[i] = ms[i] == 0.0f ? 0.0f : (xs[i] * ms[i]) * scale;
        }
        Tensor out = keep(TensorOps.maskScale(floats(xs, 2, 4), floats(ms, 2, 4), scale));
        assertEquals(Shape.of(2, 4), out.shape());
        assertFalse(out.requiresGrad());
        TrainParity.assertBitIdentical("maskScale", expected, out.toFloatArray());
        assertEquals("a dropped NaN is +0", 0, Float.floatToRawIntBits(out.toFloatArray()[1]));
    }

    @Test
    public void maskScaleTakesTheKeepScaleFromTheMaskOrTheScalar() {
        float keep = 0.7f;
        float scale = 1.0f / keep;
        float[] data = TrainParity.uniform(131, 64, -4f, 4f);
        float[] scaled = new float[data.length];
        float[] ones = new float[data.length];
        Random r = new Random(132);
        for (int i = 0; i < data.length; i++) {
            boolean kept = r.nextFloat() < keep;
            scaled[i] = kept ? scale : 0.0f;
            ones[i] = kept ? 1.0f : 0.0f;
        }
        Tensor x = floats(data, 8, 8);
        Tensor inMask = keep(TensorOps.maskScale(x, floats(scaled, 8, 8), 1.0f));
        Tensor inScalar = keep(TensorOps.maskScale(x, floats(ones, 8, 8), scale));
        TrainParity.assertBitIdentical("mask 1/keep, scale 1 vs mask 1, scale 1/keep", inMask.toFloatArray(),
                inScalar.toFloatArray());
    }

    @Test
    public void maskScaleRejectsMismatchedShapes() {
        assertThrows(IllegalArgumentException.class,
                () -> TensorOps.maskScale(floats(new float[6], 2, 3), floats(new float[6], 3, 2), 1.0f));
    }

    // ------------------------------------------------------------------ helpers

    /**
     * PyTorch's pooled extent of {@code in} for a window of {@code k} taps spaced {@code d} apart
     * ({@code pooling_output_shape}): floor or ceil division of {@code in + 2p - d(k-1) - 1} by {@code s}, plus one,
     * minus one more in ceil mode when the last window would start inside the right padding.
     */
    private static int pooledSize(int in, int k, int s, int p, int d, boolean ceil) {
        int span = in + 2 * p - d * (k - 1) - 1;
        int out = (ceil ? floorDiv(span + s - 1, s) : floorDiv(span, s)) + 1;
        if (ceil && (out - 1) * s >= in + p) {
            out--;
        }
        return out;
    }

    /** Division rounding toward negative infinity, for any sign of {@code a} and {@code b > 0}. */
    private static int floorDiv(int a, int b) {
        int q = a / b;
        return (a % b != 0 && a < 0) ? q - 1 : q;
    }

    /** Division rounding toward positive infinity, for {@code a >= 0} and {@code b > 0}. */
    private static int ceilDiv(int a, int b) {
        return (a + b - 1) / b;
    }

    /** Number of indices of {@code [start, start + k)} inside {@code [0, n)}. */
    private static int overlap(int start, int k, int n) {
        return Math.max(0, Math.min(start + k, n) - Math.max(start, 0));
    }

    /**
     * Number of indices of {@code [start, start + k)} below the padded edge {@code n + p} (PyTorch's
     * count_include_pad window size; {@code start >= -p} always holds for a pooling window).
     */
    private static int paddedOverlap(int start, int k, int n, int p) {
        return Math.min(start + k, n + p) - start;
    }

    /** The window {@code {r0, r1, c0, c1}} (in-bounds part taken by the caller) of output {@code (r, col)}. */
    private interface Window {
        int[] bounds(int r, int col);
    }

    /** The divisor of output {@code (r, col)}. */
    private interface Divisor {
        int of(int r, int col);
    }

    /**
     * Checks a pooling input gradient against its scatter definition in double: every output {@code (r, col)} of
     * every plane adds {@code gy / divisor} to each in-bounds element of its window, and each element must match
     * the sum of its terms in the reduction class.
     */
    private static void expectScatter(String what, float[] got, float[] gy, int planes, int h, int w, int oh, int ow,
                                      Window window, Divisor divisor) {
        assertEquals(what + ": length", planes * h * w, got.length);
        double[] sum = new double[got.length];
        double[] sumAbs = new double[got.length];
        int[] terms = new int[got.length];
        for (int plane = 0; plane < planes; plane++) {
            for (int r = 0; r < oh; r++) {
                for (int col = 0; col < ow; col++) {
                    int[] b = window.bounds(r, col);
                    int count = divisor.of(r, col);
                    for (int ih = Math.max(b[0], 0); ih < Math.min(b[1], h); ih++) {
                        for (int iw = Math.max(b[2], 0); iw < Math.min(b[3], w); iw++) {
                            double term = gy[(plane * oh + r) * ow + col] / (double) count;
                            int e = (plane * h + ih) * w + iw;
                            sum[e] += term;
                            sumAbs[e] += Math.abs(term);
                            terms[e]++;
                        }
                    }
                }
            }
        }
        for (int e = 0; e < got.length; e++) {
            TrainParity.assertReduction(what + " element " + e, sum[e], got[e], Math.max(1, terms[e]), sumAbs[e]);
        }
    }

    /**
     * Max pooling by its definition over {@code planes} planes of {@code h x w}: the largest in-bounds tap of each
     * window, NaN when any tap is NaN, +0 ranked above -0, and {@code -inf} for a window without input elements.
     */
    private static float[] maxPoolDefinition(float[] x, int planes, int h, int w, int oh, int ow, int[] k, int[] s,
                                             int[] p, int[] d) {
        float[] out = new float[planes * oh * ow];
        int i = 0;
        for (int plane = 0; plane < planes; plane++) {
            for (int r = 0; r < oh; r++) {
                for (int col = 0; col < ow; col++, i++) {
                    float best = Float.NEGATIVE_INFINITY;
                    boolean nan = false;
                    for (int kh = 0; kh < k[0]; kh++) {
                        int ih = r * s[0] + kh * d[0] - p[0];
                        for (int kw = 0; kw < k[1]; kw++) {
                            int iw = col * s[1] + kw * d[1] - p[1];
                            if (ih < 0 || ih >= h || iw < 0 || iw >= w) {
                                continue;
                            }
                            float v = x[(plane * h + ih) * w + iw];
                            if (Float.isNaN(v)) {
                                nan = true;
                            } else if (v > best || (v == 0f && best == 0f && Float.floatToRawIntBits(v) == 0)) {
                                best = v;
                            }
                        }
                    }
                    out[i] = nan ? Float.NaN : best;
                }
            }
        }
        return out;
    }

    /**
     * Checks one pooled mean: the in-bounds elements of rows {@code [r0, r1)} and columns {@code [c0, c1)} of
     * {@code plane}, summed in double and divided by {@code count}, in the reduction class; a window whose count is
     * 0 must yield +0.
     */
    private static void expectWindowMean(String what, float[] x, int plane, int h, int w, int r0, int r1, int c0,
                                         int c1, int count, float got) {
        if (count == 0) {
            assertEquals(what + ": an empty window yields +0", 0, Float.floatToRawIntBits(got));
            return;
        }
        double sum = 0;
        double sumAbs = 0;
        int taps = 0;
        for (int r = Math.max(r0, 0); r < Math.min(r1, h); r++) {
            for (int col = Math.max(c0, 0); col < Math.min(c1, w); col++) {
                double v = x[(plane * h + r) * w + col];
                sum += v;
                sumAbs += Math.abs(v);
                taps++;
            }
        }
        TrainParity.assertReduction(what, sum / count, got, Math.max(1, taps), sumAbs / count);
    }

    /**
     * Trains {@code bn} on one batch and compares the output and the updated running statistics with
     * {@link NormOps#batchNorm} on separate copies of the initial statistics, then does the same in evaluation mode.
     */
    private void expectBatchNorm(Module bn, Tensor gamma, Tensor beta, Tensor runningMean, Tensor runningVar,
                                 float momentum, float eps, long seed, int... dims) {
        int c = dims[1];
        if (gamma != null) {
            store(gamma, TrainParity.uniform(seed + 1, c, 0.5f, 1.5f));
            store(beta, TrainParity.uniform(seed + 2, c, -0.5f, 0.5f));
        }
        float[] mean0 = TrainParity.uniform(seed + 3, c, -0.1f, 0.1f);
        float[] var0 = TrainParity.uniform(seed + 4, c, 0.9f, 1.1f);
        store(runningMean, mean0);
        store(runningVar, var0);
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        Tensor x = floats(TrainParity.uniform(seed, n, -2f, 3f), dims);
        Tensor facadeMean = floats(mean0, c);
        Tensor facadeVar = floats(var0, c);

        bn.train();
        Tensor trained = keep(bn.forward(x));
        Tensor facade = keep(NormOps.batchNorm(x, gamma, beta, facadeMean, facadeVar, true, momentum, eps));
        expectSame("batchNorm training output", trained, facade);
        TrainParity.assertBitIdentical("running mean", runningMean.toFloatArray(), facadeMean.toFloatArray());
        TrainParity.assertBitIdentical("running var", runningVar.toFloatArray(), facadeVar.toFloatArray());

        bn.eval();
        Tensor evaluated = keep(bn.forward(x));
        expectSame("batchNorm evaluation output", evaluated,
                keep(NormOps.batchNorm(x, gamma, beta, facadeMean, facadeVar, false, momentum, eps)));
        TrainParity.assertBitIdentical("evaluation leaves the running mean", runningMean.toFloatArray(),
                facadeMean.toFloatArray());
    }

    private void expectSame(String what, Tensor module, Tensor facade) {
        assertEquals(what + ": shape", module.shape(), facade.shape());
        assertEquals(what + ": dtype", DType.FLOAT32, facade.dtype());
        assertSame(what + ": backend", host, facade.backend());
        assertFalse(what + ": requiresGrad", facade.requiresGrad());
        TrainParity.assertBitIdentical(what, module.toFloatArray(), facade.toFloatArray());
    }

    /**
     * Runs a comparison that hands module parameters, which require grad, to the facades under
     * {@link ComputationGraph#noGrad()}: the facades then return results that do not require grad,
     * and neither the modules nor the facades record a node that this test would never release.
     */
    private static void untracked(Runnable comparison) {
        try (ComputationGraph.NoGradScope ignored = ComputationGraph.noGrad()) {
            comparison.run();
        }
    }

    /** Writes {@code values} into a parameter or buffer outside grad mode, as initialization does. */
    private static void store(Tensor target, float[] values) {
        try (ComputationGraph.NoGradScope ignored = ComputationGraph.noGrad()) {
            GpuOptimizerOps.storeFloats(target, values);
        }
    }

    private Tensor floats(float[] data, int... dims) {
        return keep(keep(Tensor.fromArray(host, data)).reshape(Shape.of(dims)));
    }

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }

    private <M extends Module> M module(M m) {
        modules.add(m);
        return m;
    }
}
