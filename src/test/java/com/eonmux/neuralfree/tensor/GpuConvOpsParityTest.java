package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.layers.AdaptiveAvgPool2d;
import com.eonmux.neuralfree.nn.layers.AvgPool2d;
import com.eonmux.neuralfree.nn.layers.MaxPool2d;
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link GpuConvOps} on each GPU API against the CPU path: convolutions against double-precision references
 * (reduction class; the transposed convolution reference is the scatter definition, independent of the kernel's
 * gather form), pooling against the layers' CPU paths on the host reference backend (max pooling bit-identical,
 * average pooling in the division class), including strided and offset operand views, ceil mode and inputs
 * without elements. The convolutions run on both algorithms (the direct kernels and im2col + GEMM, also with a
 * column budget that splits the batch into chunks) and on the CPU path, each against the same reference, across
 * groups (including depthwise), stride, padding and dilation; the algorithm selection rule is checked on its own.
 */
@RunWith(Parameterized.class)
public class GpuConvOpsParityTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();

    public GpuConvOpsParityTest(TrainParity.Api api) {
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
            for (Tensor t : owned) {
                t.close();
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

    // ------------------------------------------------------------------ convolutions

    @Test
    public void conv2dGroupedStridedDilatedWithBias() {
        int b = 2, c = 6, h = 11, w = 9, oc = 4, groups = 2;
        int[] k = {3, 2}, s = {2, 1}, p = {1, 1}, d = {1, 2};
        float[] x = TrainParity.uniform(11, b * c * h * w, -3f, 3f);
        float[] wt = TrainParity.uniform(12, oc * (c / groups) * k[0] * k[1], -1f, 1f);
        float[] bias = TrainParity.uniform(13, oc, -2f, 2f);
        Tensor out = keep(GpuConvOps.conv2dForward(upload(gpu, x, b, c, h, w), upload(gpu, wt, oc, c / groups, k[0], k[1]),
                upload(gpu, bias, oc), k, s, p, d, groups));
        assertNotNull(out);
        Ref ref = conv2dRef(x, b, c, h, w, wt, oc, k, s, p, d, groups, bias);
        assertEquals(Shape.of(b, oc, ref.oh, ref.ow), out.shape());
        ref.check("conv2d", out.toFloatArray());
    }

    @Test
    public void conv2dReadsTransposedInputAndOffsetWeightWithoutBias() {
        int b = 3, c = 4, h = 7, w = 10, oc = 5;
        int[] k = {3, 3}, s = {1, 2}, p = {2, 0}, d = {2, 1};
        float[] stored = TrainParity.uniform(21, b * c * w * h, -2f, 2f);
        float[] x = new float[stored.length];
        for (int i = 0; i < b * c; i++) {
            for (int y = 0; y < h; y++) {
                for (int z = 0; z < w; z++) {
                    x[(i * h + y) * w + z] = stored[(i * w + z) * h + y];
                }
            }
        }
        int wCount = oc * c * k[0] * k[1];
        float[] padded = TrainParity.uniform(22, wCount + c * k[0] * k[1], -1f, 1f);
        float[] wt = Arrays.copyOfRange(padded, c * k[0] * k[1], padded.length);
        Tensor input = keep(upload(gpu, stored, b, c, w, h).transpose(0, 1, 3, 2));
        Tensor weight = keep(upload(gpu, padded, oc + 1, c, k[0], k[1]).slice(0, 1, oc + 1));
        assertFalse(input.isDense());
        assertTrue(weight.isDense() && !weight.isContiguous());
        Ref ref = conv2dRef(x, b, c, h, w, wt, oc, k, s, p, d, 1, null);
        for (GpuIm2col.Strategy strategy : new GpuIm2col.Strategy[]{GpuIm2col.Strategy.DIRECT,
                GpuIm2col.Strategy.im2col(GpuIm2col.COLUMN_BUDGET_BYTES)}) {
            Tensor out = keep(GpuConvOps.conv2dForward(input, weight, null, k, s, p, d, 1, strategy));
            assertNotNull(out);
            ref.check("conv2d views " + strategy, out.toFloatArray());
        }
    }

    @Test
    public void conv1dMatchesTheDirectDefinition() {
        int b = 3, c = 4, l = 17, oc = 6, kernel = 3, stride = 2, pad = 1, dil = 2, groups = 2;
        float[] x = TrainParity.uniform(31, b * c * l, -3f, 3f);
        float[] wt = TrainParity.uniform(32, oc * (c / groups) * kernel, -1f, 1f);
        float[] bias = TrainParity.uniform(33, oc, -1f, 1f);
        Tensor out = keep(GpuConvOps.conv1dForward(upload(gpu, x, b, c, l), upload(gpu, wt, oc, c / groups, kernel),
                upload(gpu, bias, oc), kernel, stride, pad, dil, groups));
        assertNotNull(out);
        int ol = (l + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
        assertEquals(Shape.of(b, oc, ol), out.shape());
        Ref ref = new Ref(b * oc * ol, 1, ol);
        int cg = c / groups, ocg = oc / groups;
        for (int n = 0; n < b; n++) {
            for (int o = 0; o < oc; o++) {
                for (int t = 0; t < ol; t++) {
                    int i = (n * oc + o) * ol + t;
                    ref.add(i, bias[o]);
                    for (int ic = 0; ic < cg; ic++) {
                        int cin = (o / ocg) * cg + ic;
                        for (int q = 0; q < kernel; q++) {
                            int pos = t * stride + q * dil - pad;
                            if (pos >= 0 && pos < l) {
                                ref.add(i, (double) x[(n * c + cin) * l + pos] * wt[(o * cg + ic) * kernel + q]);
                            }
                        }
                    }
                }
            }
        }
        ref.check("conv1d", out.toFloatArray());
    }

    @Test
    public void convTranspose2dMatchesTheScatterDefinition() {
        int b = 2, c = 4, h = 5, w = 6, groups = 2, ocg = 3, oc = ocg * groups;
        int[] k = {3, 4}, s = {2, 3}, p = {1, 2}, op = {1, 2}, d = {1, 2};
        float[] x = TrainParity.uniform(41, b * c * h * w, -3f, 3f);
        float[] wt = TrainParity.uniform(42, c * ocg * k[0] * k[1], -1f, 1f);
        float[] bias = TrainParity.uniform(43, oc, -1f, 1f);
        Tensor out = keep(GpuConvOps.convTranspose2dForward(upload(gpu, x, b, c, h, w),
                upload(gpu, wt, c, ocg, k[0], k[1]), upload(gpu, bias, oc), k, s, p, op, d, groups));
        assertNotNull(out);
        Ref ref = convTranspose2dRef(x, b, c, h, w, wt, ocg, k, s, p, op, d, groups, bias);
        assertEquals(Shape.of(b, oc, ref.oh, ref.ow), out.shape());
        ref.check("convTranspose2d", out.toFloatArray());
    }

    @Test
    public void convolutionOverAnInputWithoutElementsYieldsTheBias() {
        float[] bias = {1.5f, -2.25f, 0.0f};
        int[] k = {1, 1}, one = {1, 1};
        Tensor out = keep(GpuConvOps.conv2dForward(upload(gpu, new float[0], 2, 4, 0, 3),
                upload(gpu, TrainParity.uniform(51, 3 * 4, -1f, 1f), 3, 4, 1, 1), upload(gpu, bias, 3),
                k, one, new int[]{1, 1}, one, 1));
        assertNotNull(out);
        assertEquals(Shape.of(2, 3, 2, 5), out.shape());
        float[] expected = new float[2 * 3 * 2 * 5];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = bias[(i / 10) % 3];
        }
        TrainParity.assertBitIdentical("empty-input conv2d", expected, out.toFloatArray());
    }

    // ------------------------------------------------------------------ direct and im2col + GEMM algorithms

    @Test
    public void conv2dAlgorithmsMatchTheReference() {
        int[][] geometries = {
                // b, c, h, w, oc, kh, kw, sh, sw, ph, pw, dh, dw, groups
                {5, 6, 11, 9, 4, 3, 3, 2, 1, 1, 1, 1, 2, 2},
                {3, 8, 7, 10, 8, 3, 3, 1, 2, 2, 0, 2, 1, 8},
                {4, 16, 6, 5, 12, 1, 1, 1, 1, 0, 0, 1, 1, 1},
                {2, 3, 9, 9, 6, 4, 2, 3, 2, 2, 1, 1, 1, 3}};
        long seed = 301;
        for (int[] g : geometries) {
            int b = g[0], c = g[1], h = g[2], w = g[3], oc = g[4], groups = g[13];
            int[] k = {g[5], g[6]}, s = {g[7], g[8]}, p = {g[9], g[10]}, d = {g[11], g[12]};
            float[] x = TrainParity.uniform(seed++, b * c * h * w, -3f, 3f);
            float[] wt = TrainParity.uniform(seed++, oc * (c / groups) * k[0] * k[1], -1f, 1f);
            float[] bias = (seed & 1) == 0 ? TrainParity.uniform(seed, oc, -2f, 2f) : null;
            seed++;
            Ref ref = conv2dRef(x, b, c, h, w, wt, oc, k, s, p, d, groups, bias);
            Shape outShape = Shape.of(b, oc, ref.oh, ref.ow);
            long columns = (long) c * k[0] * k[1] * ref.oh * ref.ow;
            for (GpuIm2col.Strategy strategy : strategies(columns)) {
                Tensor out = keep(GpuConvOps.conv2dForward(upload(gpu, x, b, c, h, w),
                        upload(gpu, wt, oc, c / groups, k[0], k[1]), bias == null ? null : upload(gpu, bias, oc),
                        k, s, p, d, groups, strategy));
                assertNotNull(out);
                assertEquals(outShape, out.shape());
                ref.check("conv2d " + Arrays.toString(g) + " " + strategy, out.toFloatArray());
            }
            Tensor cpu = keep(ConvOps.conv2d(upload(host, x, b, c, h, w), upload(host, wt, oc, c / groups, k[0], k[1]),
                    bias == null ? null : upload(host, bias, oc), k, s, p, d, groups));
            assertEquals(outShape, cpu.shape());
            ref.check("conv2d " + Arrays.toString(g) + " CPU path", cpu.toFloatArray());
        }
    }

    @Test
    public void conv1dAlgorithmsMatchTheReference() {
        int b = 5, c = 8, l = 23, oc = 6, kernel = 4, stride = 2, pad = 3, dil = 2, groups = 2;
        float[] x = TrainParity.uniform(311, b * c * l, -3f, 3f);
        float[] wt = TrainParity.uniform(312, oc * (c / groups) * kernel, -1f, 1f);
        float[] bias = TrainParity.uniform(313, oc, -1f, 1f);
        int ol = (l + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
        Ref ref = conv2dRef(x, b, c, 1, l, wt, oc, new int[]{1, kernel}, new int[]{1, stride}, new int[]{0, pad},
                new int[]{1, dil}, groups, bias);
        for (GpuIm2col.Strategy strategy : strategies((long) c * kernel * ol)) {
            Tensor out = keep(GpuConvOps.conv1dForward(upload(gpu, x, b, c, l),
                    upload(gpu, wt, oc, c / groups, kernel), upload(gpu, bias, oc), kernel, stride, pad, dil, groups,
                    strategy));
            assertNotNull(out);
            assertEquals(Shape.of(b, oc, ol), out.shape());
            ref.check("conv1d " + strategy, out.toFloatArray());
        }
    }

    @Test
    public void convTranspose2dAlgorithmsMatchTheScatterDefinition() {
        int[][] geometries = {
                // b, c, h, w, ocg, kh, kw, sh, sw, ph, pw, oph, opw, dh, dw, groups
                {5, 4, 5, 6, 3, 3, 4, 2, 3, 1, 2, 1, 2, 1, 2, 2},
                {3, 6, 4, 4, 1, 4, 4, 2, 2, 1, 1, 0, 1, 1, 1, 6},
                {4, 16, 3, 5, 5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1},
                {2, 3, 6, 3, 2, 3, 3, 1, 1, 2, 0, 0, 0, 2, 3, 1}};
        long seed = 321;
        for (int[] g : geometries) {
            int b = g[0], c = g[1], h = g[2], w = g[3], ocg = g[4], groups = g[15], oc = ocg * groups;
            int[] k = {g[5], g[6]}, s = {g[7], g[8]}, p = {g[9], g[10]}, op = {g[11], g[12]}, d = {g[13], g[14]};
            float[] x = TrainParity.uniform(seed++, b * c * h * w, -3f, 3f);
            float[] wt = TrainParity.uniform(seed++, c * ocg * k[0] * k[1], -1f, 1f);
            float[] bias = (seed & 1) == 0 ? TrainParity.uniform(seed, oc, -2f, 2f) : null;
            seed++;
            Ref ref = convTranspose2dRef(x, b, c, h, w, wt, ocg, k, s, p, op, d, groups, bias);
            Shape outShape = Shape.of(b, oc, ref.oh, ref.ow);
            long columns = (long) oc * k[0] * k[1] * h * w;
            for (GpuIm2col.Strategy strategy : strategies(columns)) {
                Tensor out = keep(GpuConvOps.convTranspose2dForward(upload(gpu, x, b, c, h, w),
                        upload(gpu, wt, c, ocg, k[0], k[1]), bias == null ? null : upload(gpu, bias, oc),
                        k, s, p, op, d, groups, strategy));
                assertNotNull(out);
                assertEquals(outShape, out.shape());
                ref.check("convTranspose2d " + Arrays.toString(g) + " " + strategy, out.toFloatArray());
            }
            Tensor cpu = keep(ConvOps.convTranspose2d(upload(host, x, b, c, h, w),
                    upload(host, wt, c, ocg, k[0], k[1]), bias == null ? null : upload(host, bias, oc),
                    k, s, p, op, d, groups));
            assertEquals(outShape, cpu.shape());
            ref.check("convTranspose2d " + Arrays.toString(g) + " CPU path", cpu.toFloatArray());
        }
    }

    @Test
    public void algorithmSelectionDependsOnlyOnTheShape() {
        // 16 taps (C/groups = 4, 2x2) select im2col; 12 taps (3, 2x2) and depthwise 3x3 stay direct.
        GpuConvOps.ConvGeometry wide = geometry(9, 8, 10, 10, 6, 2, 2, 2);
        GpuIm2col.Plan plan = GpuIm2col.planForward(wide, GpuIm2col.Strategy.AUTO);
        assertNotNull(plan);
        assertEquals(8 * 2 * 2 * 9 * 9, plan.columnsPerImage);
        assertEquals(9, plan.imagesPerChunk);
        assertNotNull(GpuIm2col.planWeightGrad(wide, GpuIm2col.Strategy.AUTO));
        assertNull(GpuIm2col.planForward(geometry(9, 3, 10, 10, 6, 2, 2, 1), GpuIm2col.Strategy.AUTO));
        assertNull(GpuIm2col.planForward(geometry(9, 8, 10, 10, 8, 3, 3, 8), GpuIm2col.Strategy.AUTO));
        assertNull(GpuIm2col.planForward(wide, GpuIm2col.Strategy.DIRECT));
        // A budget of 4 images' columns splits 9 images into chunks of 4.
        GpuIm2col.Plan chunked = GpuIm2col.planForward(wide,
                GpuIm2col.Strategy.im2col(4L * plan.columnsPerImage * Float.BYTES + 3));
        assertEquals(4, chunked.imagesPerChunk);
        // One image whose columns exceed the budget: direct under AUTO, rejected when im2col is forced.
        GpuConvOps.ConvGeometry huge = geometry(1, 64, 300, 300, 8, 3, 3, 1);
        assertNull(GpuIm2col.planForward(huge, GpuIm2col.Strategy.AUTO));
        assertThrows(IllegalArgumentException.class,
                () -> GpuIm2col.planForward(huge, GpuIm2col.Strategy.im2col(GpuIm2col.COLUMN_BUDGET_BYTES)));
        assertThrows(IllegalArgumentException.class, () -> GpuIm2col.Strategy.im2col(3));
        // The transposed plan counts the output channels' columns over the input positions.
        GpuIm2col.Plan transposed = GpuIm2col.planTransposed(geometry(3, 8, 5, 5, 6, 2, 2, 1),
                GpuIm2col.Strategy.AUTO);
        assertNotNull(transposed);
        assertEquals(6 * 2 * 2 * 5 * 5, transposed.columnsPerImage);
        assertEquals(3, transposed.imagesPerChunk);
    }

    // ------------------------------------------------------------------ pooling

    @Test
    public void maxPool2dCeilModeDilatedIsBitIdentical() {
        int h = 10, w = 9;
        int[] k = {3, 2}, s = {2, 2}, p = {1, 1}, d = {2, 1};
        float[] x = TrainParity.uniform(61, 2 * 3 * h * w, -10f, 10f);
        x[5] = Float.NaN;
        x[100] = Float.POSITIVE_INFINITY;
        // Window (oh 1, ow 1) of plane 0 covers rows {1, 3, 5} and columns {1, 2}: -0 then +0 above negatives.
        // Window (oh 1, ow 2) covers columns {3, 4}: +0 then -0. Math.max yields +0 for both orders.
        for (int row : new int[]{1, 3, 5}) {
            for (int col = 1; col <= 4; col++) {
                x[row * w + col] = -1.0f;
            }
        }
        x[w + 1] = -0.0f;
        x[w + 2] = 0.0f;
        x[w + 3] = 0.0f;
        x[w + 4] = -0.0f;
        for (boolean ceil : new boolean[]{false, true}) {
            MaxPool2d layer = new MaxPool2d(k, s, p, d, ceil);
            Tensor ref = keep(layer.forward(upload(host, x, 2, 3, h, w)));
            Tensor out = keep(GpuConvOps.maxPool2dForward(upload(gpu, x, 2, 3, h, w), k, s, p, d, ceil));
            assertNotNull(out);
            assertEquals(layer.computeOutputShape(Shape.of(2, 3, h, w)), out.shape());
            assertEquals(ref.shape(), out.shape());
            TrainParity.assertBitIdentical("maxPool2d ceil=" + ceil, ref.toFloatArray(), out.toFloatArray());
        }
    }

    @Test
    public void avgPool2dHonoursCeilModeAndCountIncludePad() {
        int[] k = {3, 3}, s = {2, 2}, p = {1, 1};
        float[] x = TrainParity.uniform(71, 2 * 3 * 8 * 7, -10f, 10f);
        for (boolean ceil : new boolean[]{false, true}) {
            for (boolean includePad : new boolean[]{false, true}) {
                AvgPool2d layer = new AvgPool2d(k, s, p, ceil, includePad);
                Tensor ref = keep(layer.forward(upload(host, x, 2, 3, 8, 7)));
                Tensor out = keep(GpuConvOps.avgPool2dForward(upload(gpu, x, 2, 3, 8, 7), k, s, p, ceil, includePad));
                assertNotNull(out);
                assertEquals(layer.computeOutputShape(Shape.of(2, 3, 8, 7)), out.shape());
                assertEquals(ref.shape(), out.shape());
                TrainParity.assertDivSqrt("avgPool2d ceil=" + ceil + " includePad=" + includePad,
                        ref.toFloatArray(), out.toFloatArray());
            }
        }
    }

    @Test
    public void adaptiveAvgPool2dOnAnOffsetViewWithOverlappingWindows() {
        float[] x = TrainParity.uniform(81, 3 * 2 * 10 * 7, -10f, 10f);
        float[] tail = Arrays.copyOfRange(x, 2 * 10 * 7, x.length);
        Tensor view = keep(upload(gpu, x, 3, 2, 10, 7).slice(0, 1, 3));
        for (int[] size : new int[][]{{3, 4}, {4, 9}, {10, 7}}) {
            Tensor ref = keep(new AdaptiveAvgPool2d(size).forward(upload(host, tail, 2, 2, 10, 7)));
            Tensor out = keep(GpuConvOps.adaptiveAvgPool2d(view, size[0], size[1]));
            assertNotNull(out);
            assertEquals(ref.shape(), out.shape());
            TrainParity.assertDivSqrt("adaptiveAvgPool2d " + size[0] + "x" + size[1], ref.toFloatArray(),
                    out.toFloatArray());
        }
    }

    @Test
    public void poolingOverAnInputWithoutElements() {
        int[] k = {2, 2}, s = {1, 1}, p = {1, 1}, d = {1, 1};
        Tensor max = keep(GpuConvOps.maxPool2dForward(upload(gpu, new float[0], 1, 2, 0, 2), k, s, p, d, false));
        assertNotNull(max);
        assertEquals(Shape.of(1, 2, 1, 3), max.shape());
        float[] inf = new float[6];
        Arrays.fill(inf, Float.NEGATIVE_INFINITY);
        TrainParity.assertBitIdentical("empty-input maxPool2d", inf, max.toFloatArray());
        Tensor avg = keep(GpuConvOps.avgPool2dForward(upload(gpu, new float[0], 1, 2, 0, 2), k, s, p, false, true));
        assertNotNull(avg);
        assertEquals(Shape.of(1, 2, 1, 3), avg.shape());
        TrainParity.assertBitIdentical("empty-input avgPool2d", new float[6], avg.toFloatArray());
    }

    // ------------------------------------------------------------------ contract

    @Test
    public void cpuBackendAndOtherDtypesTakeTheCpuPath() {
        int[] k = {2, 2}, one = {1, 1}, zero = {0, 0};
        Tensor hostInput = upload(host, new float[16], 1, 1, 4, 4);
        assertNull(GpuConvOps.maxPool2dForward(hostInput, k, one, zero, one, false));
        assertNull(GpuConvOps.avgPool2dForward(hostInput, k, one, zero, false, true));
        assertNull(GpuConvOps.adaptiveAvgPool2d(hostInput, 2, 2));
        assertNull(GpuConvOps.conv2dForward(hostInput, upload(host, new float[4], 1, 1, 2, 2), null, k, one, zero,
                one, 1));
        Tensor ints = keep(keep(Tensor.fromArray(gpu, new int[16])).reshape(Shape.of(1, 1, 4, 4)));
        assertNull(GpuConvOps.maxPool2dForward(ints, k, one, zero, one, false));
    }

    @Test(expected = IllegalArgumentException.class)
    public void mismatchedChannelsAreRejected() {
        int[] k = {1, 1}, one = {1, 1}, zero = {0, 0};
        GpuConvOps.conv2dForward(upload(gpu, new float[12], 1, 3, 2, 2), upload(gpu, new float[4], 2, 2, 1, 1),
                null, k, one, zero, one, 1);
    }

    // ------------------------------------------------------------------ references

    /** Direct 2-D convolution in double ({@code nn.layers.Conv2d} definition), with the per-output term bounds. */
    private static Ref conv2dRef(float[] x, int b, int c, int h, int w, float[] wt, int oc, int[] k, int[] s,
                                 int[] p, int[] d, int groups, float[] bias) {
        int oh = (h + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1;
        int ow = (w + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1;
        Ref ref = new Ref(b * oc * oh * ow, oh, ow);
        int cg = c / groups, ocg = oc / groups;
        for (int n = 0; n < b; n++) {
            for (int o = 0; o < oc; o++) {
                for (int y = 0; y < oh; y++) {
                    for (int z = 0; z < ow; z++) {
                        int i = ((n * oc + o) * oh + y) * ow + z;
                        if (bias != null) {
                            ref.add(i, bias[o]);
                        }
                        for (int ic = 0; ic < cg; ic++) {
                            int cin = (o / ocg) * cg + ic;
                            for (int kh = 0; kh < k[0]; kh++) {
                                for (int kw = 0; kw < k[1]; kw++) {
                                    int ih = y * s[0] + kh * d[0] - p[0];
                                    int iw = z * s[1] + kw * d[1] - p[1];
                                    if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                        ref.add(i, (double) x[((n * c + cin) * h + ih) * w + iw]
                                                * wt[((o * cg + ic) * k[0] + kh) * k[1] + kw]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return ref;
    }

    /**
     * Transposed 2-D convolution in double by its scatter definition ({@code nn.layers.ConvTranspose2d}): every input
     * element times every weight tap lands on output row {@code ih*SH - PH + kh*DH} (columns likewise), plus the
     * bias when given.
     */
    private static Ref convTranspose2dRef(float[] x, int b, int c, int h, int w, float[] wt, int ocg, int[] k,
                                          int[] s, int[] p, int[] op, int[] d, int groups, float[] bias) {
        int oc = ocg * groups;
        int oh = (h - 1) * s[0] - 2 * p[0] + d[0] * (k[0] - 1) + op[0] + 1;
        int ow = (w - 1) * s[1] - 2 * p[1] + d[1] * (k[1] - 1) + op[1] + 1;
        Ref ref = new Ref(b * oc * oh * ow, oh, ow);
        if (bias != null) {
            for (int i = 0; i < b * oc * oh * ow; i++) {
                ref.add(i, bias[(i / (oh * ow)) % oc]);
            }
        }
        int cg = c / groups;
        for (int n = 0; n < b; n++) {
            for (int cin = 0; cin < c; cin++) {
                int g = cin / cg;
                for (int ih = 0; ih < h; ih++) {
                    for (int iw = 0; iw < w; iw++) {
                        double v = x[((n * c + cin) * h + ih) * w + iw];
                        for (int o = 0; o < ocg; o++) {
                            for (int kh = 0; kh < k[0]; kh++) {
                                for (int kw = 0; kw < k[1]; kw++) {
                                    int y = ih * s[0] - p[0] + kh * d[0];
                                    int z = iw * s[1] - p[1] + kw * d[1];
                                    if (y >= 0 && y < oh && z >= 0 && z < ow) {
                                        int i = ((n * oc + g * ocg + o) * oh + y) * ow + z;
                                        ref.add(i, v * wt[((cin * ocg + o) * k[0] + kh) * k[1] + kw]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return ref;
    }

    /**
     * Every algorithm choice: the direct kernels, im2col + GEMM with the default budget, im2col + GEMM with a budget
     * of two images' columns (so a batch above two runs in chunks), and the shape-based selection.
     */
    static GpuIm2col.Strategy[] strategies(long columnsPerImage) {
        return new GpuIm2col.Strategy[]{
                GpuIm2col.Strategy.DIRECT,
                GpuIm2col.Strategy.im2col(GpuIm2col.COLUMN_BUDGET_BYTES),
                GpuIm2col.Strategy.im2col(2L * columnsPerImage * Float.BYTES),
                GpuIm2col.Strategy.AUTO};
    }

    /** A convolution geometry with stride 1, no padding and dilation 1 (OH = H - KH + 1). */
    private static GpuConvOps.ConvGeometry geometry(int b, int c, int h, int w, int oc, int kh, int kw, int groups) {
        return new GpuConvOps.ConvGeometry(b, c, h, w, oc, h - kh + 1, w - kw + 1, kh, kw, 1, 1, 0, 0, 1, 1, groups);
    }

    /** Double-precision sums with their absolute-term sums and term counts, checked in the reduction class. */
    private static final class Ref {
        final int oh;
        final int ow;
        final double[] value;
        final double[] sumAbs;
        final int[] terms;

        Ref(int n, int oh, int ow) {
            this.oh = oh;
            this.ow = ow;
            this.value = new double[n];
            this.sumAbs = new double[n];
            this.terms = new int[n];
        }

        void add(int i, double term) {
            value[i] += term;
            sumAbs[i] += Math.abs(term);
            terms[i]++;
        }

        void check(String what, float[] gpu) {
            assertEquals(what + ": length", value.length, gpu.length);
            for (int i = 0; i < value.length; i++) {
                TrainParity.assertReduction(what + " element " + i, value[i], gpu[i], Math.max(1, terms[i]),
                        sumAbs[i]);
            }
        }
    }
}
