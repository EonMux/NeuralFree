package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.ConvGradFunctions;
import com.eonmux.neuralfree.autograd.GradientFunction;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link GpuConvGradOps} and the {@link ConvGradFunctions} gradients on each GPU API, and the CPU references through
 * {@link ConvOps} on the host reference backend: convolution weight and input gradients against double-precision
 * scatter references of the forward definitions (reduction class), the max-pool backward and select against the CPU
 * reference (single-rounding and bit-identical), the average-pool and adaptive average-pool backward against double
 * references of PyTorch's windows (reduction class). Covers stride below the kernel, dilation, groups including
 * depthwise, ceil mode (including a dropped last window), count_include_pad at the padded edge, overlapping adaptive
 * windows ({@code L % O != 0} and {@code O > L}), NaN and signed-zero windows, PyTorch's pooling argument checks, the
 * chunk rule's boundaries, transposed convolutions with dilation above stride and output padding at least the stride,
 * and recording through {@link ConvOps}. The convolution weight gradients and input gradients (the transposed
 * convolution for a convolution, the convolution for a transposed one) also run on each algorithm choice (the direct
 * kernels, im2col + GEMM, im2col + GEMM in batch chunks, and the shape-based selection) and on the CPU path, each
 * against the same double reference.
 */
@RunWith(Parameterized.class)
public class GpuConvGradOpsParityTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();

    public GpuConvGradOpsParityTest(TrainParity.Api api) {
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
            ComputationGraph.current().clear();
            ComputationGraph.setGradEnabled(true);
            for (int i = owned.size() - 1; i >= 0; i--) {
                owned.get(i).close();
            }
        } finally {
            if (session != null) {
                session.close();
            }
        }
    }

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }

    private Tensor upload(ComputeBackend b, float[] data, int... dims) {
        return keep(keep(Tensor.fromArray(b, data)).reshape(Shape.of(dims)));
    }

    private ComputeBackend[] backends() {
        return new ComputeBackend[]{gpu, host};
    }

    // ------------------------------------------------------------------ convolution weight gradients

    @Test
    public void conv2dWeightGradGroupedOverlappingDilated() {
        checkConv2dWeightGrad(2, 6, 11, 9, 4, new int[]{3, 2}, new int[]{1, 1}, new int[]{1, 1}, new int[]{2, 1}, 2,
                101);
    }

    @Test
    public void conv2dWeightGradDepthwiseStrided() {
        checkConv2dWeightGrad(3, 4, 8, 7, 4, new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 0}, new int[]{1, 1}, 4,
                102);
    }

    @Test
    public void weightGradChunkRuleDependsOnlyOnTheShape() {
        assertEquals(1, GpuConvGradOps.weightGradChunks(1, 32768));
        assertEquals(2, GpuConvGradOps.weightGradChunks(1, 32769));
        assertEquals(2, GpuConvGradOps.weightGradChunks(4095, 32769));
        assertEquals(1, GpuConvGradOps.weightGradChunks(4096, 32769));
        assertEquals(64, GpuConvGradOps.weightGradChunks(1, 32768L * 64 + 1));
        assertEquals(64, GpuConvGradOps.weightGradChunks(1, 32768L * 100));
    }

    @Test
    public void conv2dWeightGradAtTheChunkBoundaries() {
        int[] one = {1, 1};
        int[] zero = {0, 0};
        GpuIm2col.Strategy direct = GpuIm2col.Strategy.DIRECT;
        checkConv2dWeightGrad(2, 1, 128, 128, 1, one, one, zero, one, 1, 111, backends(), direct);
        checkConv2dWeightGrad(1, 1, 1, 32769, 1, one, one, zero, one, 1, 112, backends(), direct);
        checkConv2dWeightGrad(1, 1, 1, 32768 * 5 + 3, 1, one, one, zero, one, 1, 113, backends(), direct);
        checkConv2dWeightGrad(1, 63, 1, 32769, 65, one, one, zero, one, 1, 114, new ComputeBackend[]{gpu}, direct);
        checkConv2dWeightGrad(1, 64, 1, 32769, 64, one, one, zero, one, 1, 115, new ComputeBackend[]{gpu}, direct);
    }

    @Test
    public void conv2dWeightGradWithoutTermsIsZero() {
        int[] s = {2, 1};
        int[] zero = {0, 0};
        int[] one = {1, 1};
        for (ComputeBackend b : backends()) {
            Tensor x = upload(b, TrainParity.uniform(121, 2 * 2 * 5, -1f, 1f), 2, 2, 1, 5);
            Tensor gy = keep(Tensor.zeros(b, Shape.of(2, 3, 0, 5)));
            Tensor gw = keep(ConvOps.conv2dWeightGrad(x, gy, Shape.of(3, 2, 3, 1), s, zero, one, 1));
            assertEquals(Shape.of(3, 2, 3, 1), gw.shape());
            for (float v : gw.toFloatArray()) {
                assertEquals(0.0f, v, 0.0f);
            }
        }
    }

    @Test
    public void convTranspose2dWeightGradWithDilationAboveStride() {
        int b = 2, c = 4, h = 5, w = 4, ocg = 3, groups = 2;
        int[] k = {2, 3}, s = {2, 1}, p = {1, 0}, op = {2, 1}, d = {3, 2};
        ConvT t = new ConvT(b, c, h, w, ocg * groups, k, s, p, op, d, groups, 131);
        for (ComputeBackend be : backends()) {
            Tensor x = upload(be, t.x, b, c, h, w);
            Tensor gy = upload(be, t.gy, b, t.oc, t.oh, t.ow);
            Tensor gw = keep(ConvOps.convTranspose2dWeightGrad(x, gy, Shape.of(c, ocg, k[0], k[1]), s, p, op, d,
                    groups));
            assertEquals(Shape.of(c, ocg, k[0], k[1]), gw.shape());
            t.gw.check("convTranspose2d gw on " + be.backendType(), gw.toFloatArray());
        }
    }

    // ------------------------------------------------------------------ convolution gradient functions

    @Test
    public void conv2dGradInputGradientKeepsTheInputExtent() {
        // span mod S != 0 in both dims
        checkConv2dGrad(2, 4, 10, 7, 6, new int[]{3, 2}, new int[]{2, 3}, new int[]{1, 0}, new int[]{1, 2}, 2, 141);
        // negative span in height (OH = 1 from truncation), output padding 1 in width
        checkConv2dGrad(1, 2, 1, 5, 3, new int[]{3, 2}, new int[]{3, 2}, new int[]{0, 1}, new int[]{1, 3}, 1, 142);
        // depthwise, stride below the kernel, dilation
        checkConv2dGrad(2, 3, 9, 8, 3, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 2}, new int[]{2, 1}, 3, 143);
        // empty forward output: the input gradient is zero
        checkConv2dGrad(1, 2, 1, 4, 2, new int[]{3, 1}, new int[]{2, 1}, new int[]{0, 0}, new int[]{1, 1}, 1, 144);
    }

    @Test
    public void convTranspose2dGradCutsTheOutputPaddingRows() {
        // S <= op < D in both dims
        checkConvTranspose2dGrad(2, 4, 5, 4, 3, new int[]{2, 3}, new int[]{2, 1}, new int[]{1, 0}, new int[]{2, 1},
                new int[]{3, 2}, 2, 151);
        // op >= S and op >= D (rejected by the forward; the backward still cuts the rows), op = 0 in width
        checkConvTranspose2dGrad(1, 2, 3, 6, 4, new int[]{3, 2}, new int[]{1, 2}, new int[]{0, 1}, new int[]{3, 0},
                new int[]{1, 1}, 1, 152);
    }

    @Test
    public void conv1dGradMatchesTheUnsqueezedConv2dFormulas() {
        int b = 2, c = 4, l = 13, oc = 6, kernel = 3, stride = 2, pad = 1, dil = 2, groups = 2;
        float[] x = TrainParity.uniform(161, b * c * l, -1f, 1f);
        float[] wt = TrainParity.uniform(162, oc * (c / groups) * kernel, -1f, 1f);
        int ol = (l + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
        float[] gy = TrainParity.uniform(163, b * oc * ol, -1f, 1f);
        int[] k = {1, kernel}, s = {1, stride}, p = {0, pad}, d = {1, dil};
        Ref[] ref = conv2dRefs(x, wt, gy, b, c, 1, l, oc, 1, ol, k, s, p, d, groups);
        for (ComputeBackend be : backends()) {
            ConvGradFunctions.Conv1dGrad f = new ConvGradFunctions.Conv1dGrad(Shape.of(b, c, l),
                    Shape.of(oc, c / groups, kernel), Shape.of(oc), stride, pad, dil, groups, true, true, true);
            f.saveForBackward(upload(be, x, b, c, l), upload(be, wt, oc, c / groups, kernel));
            List<Tensor> g = backward(f, upload(be, gy, b, oc, ol));
            assertEquals(Shape.of(b, c, l), g.get(0).shape());
            assertEquals(Shape.of(oc, c / groups, kernel), g.get(1).shape());
            assertEquals(Shape.of(oc), g.get(2).shape());
            ref[0].check("conv1d gx", g.get(0).toFloatArray());
            ref[1].check("conv1d gw", g.get(1).toFloatArray());
            ref[2].check("conv1d gb", g.get(2).toFloatArray());
        }
    }

    @Test
    public void convOpsRecordAndBackpropagateToLeaves() {
        int b = 2, c = 2, h = 6, w = 5, oc = 3;
        int[] k = {3, 3}, s = {2, 1}, p = {1, 1}, d = {1, 1};
        float[] x = TrainParity.uniform(171, b * c * h * w, -1f, 1f);
        float[] wt = TrainParity.uniform(172, oc * c * 9, -1f, 1f);
        float[] bias = TrainParity.uniform(173, oc, -1f, 1f);
        int oh = (h + 2 - 2 - 1) / 2 + 1;
        float[] gy = TrainParity.uniform(174, b * oc * oh * w, -1f, 1f);
        Tensor xl = keep(upload(gpu, x, b, c, h, w).withRequiresGrad(true));
        Tensor wl = keep(upload(gpu, wt, oc, c, 3, 3).withRequiresGrad(true));
        Tensor bl = keep(upload(gpu, bias, oc).withRequiresGrad(true));
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            Tensor plain = keep(ConvOps.conv2d(xl, wl, bl, k, s, p, d, 1));
            assertFalse(plain.requiresGrad());
            assertNull(plain.gradFn());
        }
        Tensor y = keep(ConvOps.conv2d(xl, wl, bl, k, s, p, d, 1));
        assertTrue(y.requiresGrad());
        assertTrue(y.gradFn() instanceof ConvGradFunctions.Conv2dGrad);
        y.backward(upload(gpu, gy, b, oc, oh, w));
        Ref[] ref = conv2dRefs(x, wt, gy, b, c, h, w, oc, oh, w, k, s, p, d, 1);
        ref[0].check("recorded conv2d gx", xl.grad().toFloatArray());
        ref[1].check("recorded conv2d gw", wl.grad().toFloatArray());
        ref[2].check("recorded conv2d gb", bl.grad().toFloatArray());
        ComputationGraph.current().zeroGrad();

        Tensor pooled = keep(ConvOps.maxPool2d(xl, new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0},
                new int[]{1, 1}, true));
        assertTrue(pooled.gradFn() instanceof ConvGradFunctions.MaxPool2dGrad);
        Tensor averaged = keep(ConvOps.avgPool2d(xl, new int[]{2, 2}, new int[]{1, 1}, new int[]{1, 1}, false,
                true));
        assertTrue(averaged.gradFn() instanceof ConvGradFunctions.AvgPool2dGrad);
        Tensor adaptive = keep(ConvOps.adaptiveAvgPool2d(xl, 4, 3));
        assertTrue(adaptive.gradFn() instanceof ConvGradFunctions.AdaptiveAvgPool2dGrad);
    }

    // ------------------------------------------------------------------ direct and im2col + GEMM algorithms

    @Test
    public void conv2dGradientsOnEveryAlgorithm() {
        int[][] geometries = {
                // b, c, h, w, oc, kh, kw, sh, sw, ph, pw, dh, dw, groups
                {5, 6, 11, 9, 4, 3, 3, 2, 1, 1, 1, 1, 2, 2},
                {3, 8, 8, 7, 8, 3, 3, 2, 2, 1, 0, 1, 1, 8},
                {4, 16, 6, 5, 12, 1, 1, 1, 1, 0, 0, 1, 1, 1},
                {3, 4, 10, 7, 6, 3, 2, 2, 3, 1, 0, 1, 2, 2}};
        long seed = 401;
        for (int[] g : geometries) {
            int b = g[0], c = g[1], h = g[2], w = g[3], oc = g[4], groups = g[13];
            int[] k = {g[5], g[6]}, s = {g[7], g[8]}, p = {g[9], g[10]}, d = {g[11], g[12]};
            int oh = (h + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1;
            int ow = (w + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1;
            int[] op = {inputGradOutputPadding(h, k[0], s[0], p[0], d[0], oh),
                    inputGradOutputPadding(w, k[1], s[1], p[1], d[1], ow)};
            float[] x = TrainParity.uniform(seed++, b * c * h * w, -1f, 1f);
            float[] wt = TrainParity.uniform(seed++, oc * (c / groups) * k[0] * k[1], -1f, 1f);
            float[] gy = TrainParity.uniform(seed++, b * oc * oh * ow, -1f, 1f);
            Ref[] ref = conv2dRefs(x, wt, gy, b, c, h, w, oc, oh, ow, k, s, p, d, groups);
            Shape ws = Shape.of(oc, c / groups, k[0], k[1]);
            String geo = " " + Arrays.toString(g);
            // The weight gradient's columns (over x) and the input gradient's (the transposed convolution of gy
            // into C channels over OH x OW positions) both hold C*KH*KW*OH*OW elements per image.
            for (GpuIm2col.Strategy strategy : GpuConvOpsParityTest.strategies((long) c * k[0] * k[1] * oh * ow)) {
                Tensor xt = upload(gpu, x, b, c, h, w);
                Tensor gyt = upload(gpu, gy, b, oc, oh, ow);
                Tensor wtt = upload(gpu, wt, ws.dims());
                Tensor gw = keep(GpuConvGradOps.conv2dWeightGrad(xt, gyt, ws, s, p, d, groups, strategy));
                assertNotNull(gw);
                assertEquals(ws, gw.shape());
                ref[1].check("conv2d gw" + geo + " " + strategy, gw.toFloatArray());
                Tensor gx = keep(GpuConvOps.convTranspose2dForward(gyt, wtt, null, k, s, p, op, d, groups, strategy));
                assertNotNull(gx);
                assertEquals(Shape.of(b, c, h, w), gx.shape());
                ref[0].check("conv2d gx" + geo + " " + strategy, gx.toFloatArray());
            }
            Tensor gyHost = upload(host, gy, b, oc, oh, ow);
            Tensor gw = keep(ConvOps.conv2dWeightGrad(upload(host, x, b, c, h, w), gyHost, ws, s, p, d, groups));
            ref[1].check("conv2d gw" + geo + " CPU path", gw.toFloatArray());
            Tensor gx = keep(ConvOps.convTranspose2d(gyHost, upload(host, wt, ws.dims()), null, k, s, p, op, d,
                    groups));
            ref[0].check("conv2d gx" + geo + " CPU path", gx.toFloatArray());
        }
    }

    @Test
    public void convTranspose2dGradientsOnEveryAlgorithm() {
        int[][] geometries = {
                // b, c, h, w, ocg, kh, kw, sh, sw, ph, pw, oph, opw, dh, dw, groups (op < stride)
                {5, 4, 5, 6, 3, 3, 4, 2, 3, 1, 2, 1, 2, 1, 2, 2},
                {3, 6, 4, 4, 1, 4, 4, 2, 2, 1, 1, 0, 1, 1, 1, 6},
                {4, 16, 3, 5, 5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1}};
        long seed = 421;
        for (int[] g : geometries) {
            int b = g[0], c = g[1], h = g[2], w = g[3], ocg = g[4], groups = g[15];
            int[] k = {g[5], g[6]}, s = {g[7], g[8]}, p = {g[9], g[10]}, op = {g[11], g[12]}, d = {g[13], g[14]};
            ConvT t = new ConvT(b, c, h, w, ocg * groups, k, s, p, op, d, groups, seed);
            seed += 3;
            Shape ws = Shape.of(c, ocg, k[0], k[1]);
            String geo = " " + Arrays.toString(g);
            // The weight gradient's columns (over gy, the kernel's x) and the input gradient's (the convolution of
            // gy) both hold OC*KH*KW*H*W elements per image.
            for (GpuIm2col.Strategy strategy : GpuConvOpsParityTest.strategies((long) t.oc * k[0] * k[1] * h * w)) {
                Tensor xt = upload(gpu, t.x, b, c, h, w);
                Tensor gyt = upload(gpu, t.gy, b, t.oc, t.oh, t.ow);
                Tensor gw = keep(GpuConvGradOps.convTranspose2dWeightGrad(xt, gyt, ws, s, p, op, d, groups,
                        strategy));
                assertNotNull(gw);
                assertEquals(ws, gw.shape());
                t.gw.check("convTranspose2d gw" + geo + " " + strategy, gw.toFloatArray());
                Tensor gx = keep(GpuConvOps.conv2dForward(gyt, upload(gpu, t.w, ws.dims()), null, k, s, p, d, groups,
                        strategy));
                assertNotNull(gx);
                assertEquals(Shape.of(b, c, h, w), gx.shape());
                t.gx.check("convTranspose2d gx" + geo + " " + strategy, gx.toFloatArray());
            }
            Tensor gyHost = upload(host, t.gy, b, t.oc, t.oh, t.ow);
            Tensor gw = keep(ConvOps.convTranspose2dWeightGrad(upload(host, t.x, b, c, h, w), gyHost, ws, s, p, op,
                    d, groups));
            t.gw.check("convTranspose2d gw" + geo + " CPU path", gw.toFloatArray());
            Tensor gx = keep(ConvOps.conv2d(gyHost, upload(host, t.w, ws.dims()), null, k, s, p, d, groups));
            t.gx.check("convTranspose2d gx" + geo + " CPU path", gx.toFloatArray());
        }
    }

    // ------------------------------------------------------------------ pooling

    @Test
    public void maxPool2dBackwardRoutesTiesNaNAndSignedZerosLikeTheCpu() {
        int b = 2, c = 3, h = 10, w = 8;
        int[] k = {3, 2}, s = {2, 1}, p = {1, 1}, d = {1, 2};
        float[] x = specialValues(181, b * c * h * w);
        for (boolean ceil : new boolean[]{false, true}) {
            Tensor[] y = new Tensor[2];
            Tensor[] gx = new Tensor[2];
            Tensor[] sel = new Tensor[2];
            ComputeBackend[] bs = backends();
            for (int i = 0; i < 2; i++) {
                Tensor xt = upload(bs[i], x, b, c, h, w);
                y[i] = keep(ConvOps.maxPool2d(xt, k, s, p, d, ceil));
                Shape ys = y[i].shape();
                Tensor gy = upload(bs[i], TrainParity.uniform(182, (int) ys.numElements(), -1f, 1f), ys.dims());
                gx[i] = keep(i == 0 ? GpuConvGradOps.maxPool2dBackward(xt, y[i], gy, k, s, p, d, ceil)
                        : ConvOps.maxPool2dBackward(xt, y[i], gy, k, s, p, d, ceil));
                Tensor g = upload(bs[i], TrainParity.uniform(183, b * c * h * w, -1f, 1f), b, c, h, w);
                sel[i] = keep(i == 0 ? GpuConvGradOps.maxPool2dSelect(xt, y[i], g, k, s, p, d, ceil)
                        : ConvOps.maxPool2dSelect(xt, y[i], g, k, s, p, d, ceil));
            }
            assertEquals(Shape.of(b, c, h, w), gx[0].shape());
            TrainParity.assertSingleRounding("maxPool2dBackward ceil=" + ceil, gx[1].toFloatArray(),
                    gx[0].toFloatArray());
            TrainParity.assertBitIdentical("maxPool2dSelect ceil=" + ceil, sel[1].toFloatArray(),
                    sel[0].toFloatArray());
        }
    }

    @Test
    public void maxPool2dBackwardFirstMatchSemantics() {
        int[] k = {2, 2}, one = {1, 1}, zero = {0, 0};
        for (ComputeBackend be : backends()) {
            // ties route to the first tap; -0 == +0 routes to the -0 tap; a NaN window routes nowhere
            Tensor x = upload(be, new float[]{1f, 2f, 2f, 0f, -0f, 0f, 0f, -0f, 5f, Float.NaN, 5f, 1f}, 3, 1, 2, 2);
            Tensor y = keep(ConvOps.maxPool2d(x, k, one, zero, one, false));
            Tensor gy = upload(be, new float[]{7f, 8f, 9f}, 3, 1, 1, 1);
            Tensor gx = keep(ConvOps.maxPool2dBackward(x, y, gy, k, one, zero, one, false));
            TrainParity.assertBitIdentical("first-match routing on " + be.backendType(),
                    new float[]{0f, 7f, 0f, 0f, 8f, 0f, 0f, 0f, 0f, 0f, 0f, 0f}, gx.toFloatArray());
        }
    }

    /**
     * PyTorch's average-pool backward over two geometries: 8 x 6 with a 3 x 3 window, stride (2, 1) and padding 1,
     * where the last ceil-mode row window reaches past the padded edge (its divisor with count_include_pad is 2 rows,
     * not 3); and 5 x 7 with a 2 x 2 window, stride 2 and padding 1, where ceil mode drops the last row and column
     * window (it would start inside the right padding).
     */
    @Test
    public void avgPool2dBackwardMatchesTheDoubleReference() {
        int b = 2, c = 2;
        int[][] sizes = {{8, 6}, {5, 7}};
        int[][][] geometry = {{{3, 3}, {2, 1}, {1, 1}}, {{2, 2}, {2, 2}, {1, 1}}};
        int[][] ceilSizes = {{5, 6}, {3, 4}};
        for (int g = 0; g < sizes.length; g++) {
            int h = sizes[g][0], w = sizes[g][1];
            int[] k = geometry[g][0], s = geometry[g][1], p = geometry[g][2];
            for (boolean ceil : new boolean[]{false, true}) {
                int oh = pooledSize(h, k[0], s[0], p[0], 1, ceil);
                int ow = pooledSize(w, k[1], s[1], p[1], 1, ceil);
                if (ceil) {
                    assertEquals("ceil-mode output of " + h + "x" + w, Shape.of(1, 1, ceilSizes[g][0],
                            ceilSizes[g][1]), Shape.of(1, 1, oh, ow));
                }
                for (boolean cip : new boolean[]{false, true}) {
                    float[] gy = TrainParity.uniform(191 + g, b * c * oh * ow, -1f, 1f);
                    Ref ref = avgPoolRef(gy, b * c, h, w, oh, ow, k, s, p, cip);
                    for (ComputeBackend be : backends()) {
                        Tensor gx = keep(ConvOps.avgPool2dBackward(upload(be, gy, b, c, oh, ow),
                                Shape.of(b, c, h, w), k, s, p, ceil, cip));
                        ref.check("avgPool2dBackward " + h + "x" + w + " ceil=" + ceil + " cip=" + cip + " on "
                                + be.backendType(), gx.toFloatArray());
                    }
                }
            }
        }
    }

    /**
     * Adaptive average-pool backward over PyTorch's overlapping windows: 7 rows into 3 windows ({@code 7 % 3 != 0},
     * rows 2 and 4 lie in two windows each) and 2 columns into 5 windows ({@code O > L}, every column lies in three
     * windows), against a double reference on both backends.
     */
    @Test
    public void adaptiveAvgPool2dBackwardWithOverlappingWindows() {
        int b = 2, c = 3, h = 7, w = 2, oh = 3, ow = 5;
        float[] gy = TrainParity.uniform(201, b * c * oh * ow, -1f, 1f);
        Ref ref = adaptivePoolRef(gy, b * c, h, w, oh, ow);
        Tensor device = keep(GpuConvGradOps.adaptiveAvgPool2dBackward(upload(gpu, gy, b, c, oh, ow),
                Shape.of(b, c, h, w)));
        assertNotNull(device);
        ref.check("adaptiveAvgPool2dBackward on " + gpu.backendType(), device.toFloatArray());
        Tensor cpu = keep(ConvOps.adaptiveAvgPool2dBackward(upload(host, gy, b, c, oh, ow), Shape.of(b, c, h, w)));
        ref.check("adaptiveAvgPool2dBackward on " + host.backendType(), cpu.toFloatArray());
    }

    /**
     * PyTorch's pooling argument checks on both paths: an output size below 1, a padding above half the (effective)
     * kernel, and operands shaped with an extra ceil-mode window that would start inside the right padding (5 rows,
     * window 2, stride 2, padding 1: 3 windows, not 4).
     */
    @Test
    public void poolingArgumentChecksFollowPyTorch() {
        int[] one = {1, 1}, zero = {0, 0};
        for (ComputeBackend be : backends()) {
            Tensor x = upload(be, TrainParity.uniform(211, 2 * 4, -1f, 1f), 1, 2, 1, 4);
            assertThrows(IllegalArgumentException.class, () -> ConvOps.maxPool2d(x, new int[]{3, 1},
                    new int[]{2, 1}, zero, one, false));
            assertThrows(IllegalArgumentException.class, () -> ConvOps.avgPool2d(x, new int[]{1, 2}, one,
                    new int[]{0, 2}, false, true));
            assertThrows(IllegalArgumentException.class, () -> ConvOps.maxPool2d(x, new int[]{1, 3}, one,
                    new int[]{0, 2}, one, false));
            Tensor dilatedPad = keep(ConvOps.maxPool2d(x, new int[]{1, 2}, one, new int[]{0, 1}, new int[]{1, 2},
                    false));
            assertEquals("padding 1 <= effective kernel 3 / 2", Shape.of(1, 2, 1, 4), dilatedPad.shape());
            Tensor wide = upload(be, TrainParity.uniform(212, 5 * 5, -1f, 1f), 1, 1, 5, 5);
            Tensor paddedWindowGrad = upload(be, new float[16], 1, 1, 4, 4);
            assertThrows(IllegalArgumentException.class, () -> ConvOps.avgPool2dBackward(paddedWindowGrad,
                    wide.shape(), new int[]{2, 2}, new int[]{2, 2}, one, true, true));
            Tensor y = keep(ConvOps.maxPool2d(wide, new int[]{2, 2}, new int[]{2, 2}, one, one, true));
            assertEquals(Shape.of(1, 1, 3, 3), y.shape());
            assertThrows(IllegalArgumentException.class, () -> ConvOps.maxPool2dBackward(wide, paddedWindowGrad,
                    paddedWindowGrad, new int[]{2, 2}, new int[]{2, 2}, one, one, true));
        }
    }

    @Test
    public void hostOperandsTakeTheCpuPath() {
        int[] k = {2, 2}, one = {1, 1}, zero = {0, 0};
        Tensor x = upload(host, new float[16], 1, 1, 4, 4);
        Tensor y = upload(host, new float[9], 1, 1, 3, 3);
        assertNull(GpuConvGradOps.maxPool2dBackward(x, y, y, k, one, zero, one, false));
        assertNull(GpuConvGradOps.maxPool2dSelect(x, y, x, k, one, zero, one, false));
        assertNull(GpuConvGradOps.avgPool2dBackward(y, x.shape(), k, one, zero, false, true));
        assertNull(GpuConvGradOps.adaptiveAvgPool2dBackward(y, x.shape()));
        assertNull(GpuConvGradOps.conv2dWeightGrad(x, y, Shape.of(1, 1, 2, 2), one, zero, one, 1));
        assertNull(GpuConvGradOps.convTranspose2dWeightGrad(y, x, Shape.of(1, 1, 2, 2), one, zero, zero, one, 1));
    }

    @Test(expected = IllegalArgumentException.class)
    public void gradientsOfTheWrongShapeAreRejected() {
        int[] k = {2, 2}, one = {1, 1}, zero = {0, 0};
        GpuConvGradOps.conv2dWeightGrad(upload(gpu, new float[16], 1, 1, 4, 4), upload(gpu, new float[4], 1, 1, 2, 2),
                Shape.of(1, 1, 2, 2), one, zero, one, 1);
    }

    // ------------------------------------------------------------------ checks

    private void checkConv2dWeightGrad(int b, int c, int h, int w, int oc, int[] k, int[] s, int[] p, int[] d,
                                       int groups, long seed) {
        checkConv2dWeightGrad(b, c, h, w, oc, k, s, p, d, groups, seed, backends(), GpuIm2col.Strategy.AUTO);
    }

    /** The weight gradient on each of {@code on}: the dispatcher with {@code strategy} on the GPU, the CPU path. */
    private void checkConv2dWeightGrad(int b, int c, int h, int w, int oc, int[] k, int[] s, int[] p, int[] d,
                                       int groups, long seed, ComputeBackend[] on, GpuIm2col.Strategy strategy) {
        int oh = (h + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1;
        int ow = (w + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1;
        float[] x = TrainParity.uniform(seed, b * c * h * w, -1f, 1f);
        float[] gy = TrainParity.uniform(seed + 1, b * oc * oh * ow, -1f, 1f);
        Ref ref = conv2dRefs(x, new float[oc * (c / groups) * k[0] * k[1]], gy, b, c, h, w, oc, oh, ow, k, s, p, d,
                groups)[1];
        Shape ws = Shape.of(oc, c / groups, k[0], k[1]);
        for (ComputeBackend be : on) {
            Tensor xt = upload(be, x, b, c, h, w);
            Tensor gyt = upload(be, gy, b, oc, oh, ow);
            Tensor gw = keep(be == gpu ? GpuConvGradOps.conv2dWeightGrad(xt, gyt, ws, s, p, d, groups, strategy)
                    : ConvOps.conv2dWeightGrad(xt, gyt, ws, s, p, d, groups));
            assertNotNull(gw);
            assertEquals(ws, gw.shape());
            ref.check("conv2d gw on " + be.backendType(), gw.toFloatArray());
        }
    }

    private void checkConv2dGrad(int b, int c, int h, int w, int oc, int[] k, int[] s, int[] p, int[] d, int groups,
                                 long seed) {
        int oh = (h + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1;
        int ow = (w + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1;
        float[] x = TrainParity.uniform(seed, b * c * h * w, -1f, 1f);
        float[] wt = TrainParity.uniform(seed + 1, oc * (c / groups) * k[0] * k[1], -1f, 1f);
        float[] gy = TrainParity.uniform(seed + 2, b * oc * oh * ow, -1f, 1f);
        Ref[] ref = conv2dRefs(x, wt, gy, b, c, h, w, oc, oh, ow, k, s, p, d, groups);
        Shape ws = Shape.of(oc, c / groups, k[0], k[1]);
        for (ComputeBackend be : backends()) {
            ConvGradFunctions.Conv2dGrad f = new ConvGradFunctions.Conv2dGrad(Shape.of(b, c, h, w), ws, Shape.of(oc),
                    s, p, d, groups, true, true, true);
            f.saveForBackward(upload(be, x, b, c, h, w), upload(be, wt, ws.dims()));
            List<Tensor> g = backward(f, upload(be, gy, b, oc, oh, ow));
            String on = " on " + be.backendType() + " for H=" + h + " W=" + w;
            assertEquals("conv2d gx shape" + on, Shape.of(b, c, h, w), g.get(0).shape());
            ref[0].check("conv2d gx" + on, g.get(0).toFloatArray());
            ref[1].check("conv2d gw" + on, g.get(1).toFloatArray());
            ref[2].check("conv2d gb" + on, g.get(2).toFloatArray());
        }
    }

    private void checkConvTranspose2dGrad(int b, int c, int h, int w, int ocg, int[] k, int[] s, int[] p, int[] op,
                                          int[] d, int groups, long seed) {
        ConvT t = new ConvT(b, c, h, w, ocg * groups, k, s, p, op, d, groups, seed);
        Shape ws = Shape.of(c, ocg, k[0], k[1]);
        for (ComputeBackend be : backends()) {
            ConvGradFunctions.ConvTranspose2dGrad f = new ConvGradFunctions.ConvTranspose2dGrad(Shape.of(b, c, h, w),
                    ws, Shape.of(t.oc), s, p, op, d, groups, true, true, true);
            f.saveForBackward(upload(be, t.x, b, c, h, w), upload(be, t.w, ws.dims()));
            List<Tensor> g = backward(f, upload(be, t.gy, b, t.oc, t.oh, t.ow));
            String on = " on " + be.backendType();
            assertEquals("convTranspose2d gx shape" + on, Shape.of(b, c, h, w), g.get(0).shape());
            t.gx.check("convTranspose2d gx" + on, g.get(0).toFloatArray());
            t.gw.check("convTranspose2d gw" + on, g.get(1).toFloatArray());
            t.gb.check("convTranspose2d gb" + on, g.get(2).toFloatArray());
        }
    }

    /** Runs {@code f.backward}, keeps every gradient, then releases the saved tensors. */
    private List<Tensor> backward(GradientFunction f, Tensor gy) {
        try {
            List<Tensor> grads = f.backward(gy);
            for (Tensor g : grads) {
                if (g != null) {
                    keep(g);
                }
            }
            return grads;
        } finally {
            f.clearSavedTensors();
        }
    }

    // ------------------------------------------------------------------ references

    /**
     * The output padding that makes the transposed convolution over {@code out} positions restore {@code size}
     * positions (the input gradient of a convolution whose span {@code size + 2p - d(k-1) - 1} is non-negative).
     */
    private static int inputGradOutputPadding(int size, int k, int s, int p, int d, int out) {
        int span = size + 2 * p - d * (k - 1) - 1;
        assertTrue("non-negative span", span >= 0);
        return span - (out - 1) * s;
    }

    /** Values from a small set with ties, signed zeros and occasional NaN, so windows hit every routing rule. */
    private static float[] specialValues(long seed, int n) {
        float[] pick = {-1f, 0f, -0f, 1f, 2f, 2f, 3f, -2f, 0.5f, 2f, -0f, 1f, 3f, Float.NaN, 0f, 2f};
        Random r = new Random(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = pick[r.nextInt(pick.length)];
        }
        return out;
    }

    /**
     * Double references of a 2-D convolution's gradients by scattering every forward term
     * {@code x[b, cin, ih, iw] * w[oc, ic, kh, kw]} into gx (times gy), gw (times gy) and gb: index 0 gx, 1 gw, 2 gb.
     */
    private static Ref[] conv2dRefs(float[] x, float[] wt, float[] gy, int b, int c, int h, int w, int oc, int oh,
                                    int ow, int[] k, int[] s, int[] p, int[] d, int groups) {
        Ref gx = new Ref(b * c * h * w);
        Ref gw = new Ref(oc * (c / groups) * k[0] * k[1]);
        Ref gb = new Ref(oc);
        int cg = c / groups, ocg = oc / groups;
        for (int n = 0; n < b; n++) {
            for (int o = 0; o < oc; o++) {
                for (int y = 0; y < oh; y++) {
                    for (int z = 0; z < ow; z++) {
                        double g = gy[((n * oc + o) * oh + y) * ow + z];
                        gb.add(o, g);
                        for (int ic = 0; ic < cg; ic++) {
                            int cin = (o / ocg) * cg + ic;
                            for (int kh = 0; kh < k[0]; kh++) {
                                for (int kw = 0; kw < k[1]; kw++) {
                                    int ih = y * s[0] + kh * d[0] - p[0];
                                    int iw = z * s[1] + kw * d[1] - p[1];
                                    if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                        int xi = ((n * c + cin) * h + ih) * w + iw;
                                        int wi = ((o * cg + ic) * k[0] + kh) * k[1] + kw;
                                        gx.add(xi, g * wt[wi]);
                                        gw.add(wi, g * x[xi]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return new Ref[]{gx, gw, gb};
    }

    /** A transposed convolution's random operands and its double gradient references (scatter definition). */
    private static final class ConvT {
        final int oc, oh, ow;
        final float[] x, w, gy;
        final Ref gx, gw, gb;

        ConvT(int b, int c, int h, int wd, int oc, int[] k, int[] s, int[] p, int[] op, int[] d, int groups,
              long seed) {
            this.oc = oc;
            this.oh = (h - 1) * s[0] - 2 * p[0] + d[0] * (k[0] - 1) + op[0] + 1;
            this.ow = (wd - 1) * s[1] - 2 * p[1] + d[1] * (k[1] - 1) + op[1] + 1;
            int cg = c / groups, ocg = oc / groups;
            this.x = TrainParity.uniform(seed, b * c * h * wd, -1f, 1f);
            this.w = TrainParity.uniform(seed + 1, c * ocg * k[0] * k[1], -1f, 1f);
            this.gy = TrainParity.uniform(seed + 2, b * oc * oh * ow, -1f, 1f);
            this.gx = new Ref(x.length);
            this.gw = new Ref(w.length);
            this.gb = new Ref(oc);
            for (int n = 0; n < b; n++) {
                for (int o = 0; o < oc; o++) {
                    for (int i = 0; i < oh * ow; i++) {
                        gb.add(o, gy[(n * oc + o) * oh * ow + i]);
                    }
                }
                for (int cin = 0; cin < c; cin++) {
                    for (int ih = 0; ih < h; ih++) {
                        for (int iw = 0; iw < wd; iw++) {
                            int xi = ((n * c + cin) * h + ih) * wd + iw;
                            for (int q = 0; q < ocg; q++) {
                                int o = (cin / cg) * ocg + q;
                                for (int kh = 0; kh < k[0]; kh++) {
                                    for (int kw = 0; kw < k[1]; kw++) {
                                        int y = ih * s[0] + kh * d[0] - p[0];
                                        int z = iw * s[1] + kw * d[1] - p[1];
                                        if (y >= 0 && y < oh && z >= 0 && z < ow) {
                                            double g = gy[((n * oc + o) * oh + y) * ow + z];
                                            int wi = ((cin * ocg + q) * k[0] + kh) * k[1] + kw;
                                            gx.add(xi, g * w[wi]);
                                            gw.add(wi, g * x[xi]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * PyTorch's pooled extent ({@code pooling_output_shape}), written out independently: floor or ceil division of
     * {@code in + 2p - d(k-1) - 1} by {@code s}, plus one, minus one more in ceil mode when the last window would
     * start inside the right padding.
     */
    private static int pooledSize(int in, int k, int s, int p, int d, boolean ceil) {
        int span = in + 2 * p - d * (k - 1) - 1;
        int out = (ceil ? Math.floorDiv(span + s - 1, s) : Math.floorDiv(span, s)) + 1;
        if (ceil && (out - 1) * s >= in + p) {
            out--;
        }
        return out;
    }

    /**
     * Average-pool backward in double over PyTorch's windows: window {@code (y, z)} covers the taps
     * {@code [y*s - p, y*s - p + k)} clipped to the padded extent {@code [-p, h + p)}; it adds {@code gy / divisor}
     * to its in-bounds taps, the divisor being the clipped tap count with count_include_pad and the in-bounds count
     * without.
     */
    private static Ref avgPoolRef(float[] gy, int planes, int h, int w, int oh, int ow, int[] k, int[] s, int[] p,
                                  boolean cip) {
        Ref ref = new Ref(planes * h * w);
        for (int pl = 0; pl < planes; pl++) {
            for (int y = 0; y < oh; y++) {
                for (int z = 0; z < ow; z++) {
                    int padded = 0;
                    int inside = 0;
                    for (int kh = 0; kh < k[0]; kh++) {
                        for (int kw = 0; kw < k[1]; kw++) {
                            int ih = y * s[0] + kh - p[0];
                            int iw = z * s[1] + kw - p[1];
                            if (ih < h + p[0] && iw < w + p[1]) {
                                padded++;
                            }
                            if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                inside++;
                            }
                        }
                    }
                    double divisor = cip ? padded : inside;
                    for (int kh = 0; kh < k[0]; kh++) {
                        for (int kw = 0; kw < k[1]; kw++) {
                            int ih = y * s[0] + kh - p[0];
                            int iw = z * s[1] + kw - p[1];
                            if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                ref.add((pl * h + ih) * w + iw, gy[(pl * oh + y) * ow + z] / divisor);
                            }
                        }
                    }
                }
            }
        }
        return ref;
    }

    /**
     * Adaptive average-pool backward in double over PyTorch's windows: output row {@code y} of {@code oh} covers the
     * input rows {@code r} with {@code floor(y*h/oh) <= r < ceil((y+1)*h/oh)}, tested exactly in integer arithmetic
     * ({@link #inAdaptiveWindow}), columns likewise; each window adds {@code gy / size} to its elements.
     */
    private static Ref adaptivePoolRef(float[] gy, int planes, int h, int w, int oh, int ow) {
        Ref ref = new Ref(planes * h * w);
        for (int pl = 0; pl < planes; pl++) {
            for (int y = 0; y < oh; y++) {
                for (int z = 0; z < ow; z++) {
                    List<Integer> cells = new ArrayList<>();
                    for (int r = 0; r < h; r++) {
                        for (int col = 0; col < w; col++) {
                            if (inAdaptiveWindow(r, y, h, oh) && inAdaptiveWindow(col, z, w, ow)) {
                                cells.add(r * w + col);
                            }
                        }
                    }
                    for (int cell : cells) {
                        ref.add(pl * h * w + cell, gy[(pl * oh + y) * ow + z] / (double) cells.size());
                    }
                }
            }
        }
        return ref;
    }

    /**
     * {@code floor(o*in/out) <= i < ceil((o+1)*in/out)}, i.e. {@code (i+1)*out > o*in} and
     * {@code i*out < (o+1)*in}.
     */
    private static boolean inAdaptiveWindow(int i, int o, int in, int out) {
        return (long) (i + 1) * out > (long) o * in && (long) i * out < (long) (o + 1) * in;
    }

    /** Double-precision sums with their absolute-term sums and term counts, checked in the reduction class. */
    private static final class Ref {
        final double[] value;
        final double[] sumAbs;
        final int[] terms;

        Ref(int n) {
            this.value = new double[n];
            this.sumAbs = new double[n];
            this.terms = new int[n];
        }

        void add(int i, double term) {
            value[i] += term;
            sumAbs[i] += Math.abs(term);
            terms[i]++;
        }

        void check(String what, float[] actual) {
            assertEquals(what + ": length", value.length, actual.length);
            for (int i = 0; i < value.length; i++) {
                TrainParity.assertReduction(what + " element " + i, value[i], actual[i], Math.max(1, terms[i]),
                        sumAbs[i]);
            }
        }
    }
}
