package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.gpu.ElementwiseOp.Backward;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.function.UnaryOperator;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link GpuTensorOps} on each GPU API against the CPU path ({@link TensorOps} on the host reference backend, or the
 * cited CPU formula where TensorOps has no such op), under the tolerance classes of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}. Inputs are strided views (transposes, offset
 * slices, broadcasts) and rank-7 shapes, so the coalescing and split paths run.
 */
@RunWith(Parameterized.class)
public class GpuTensorOpsParityTest {

    private static final float[] SPECIAL = {
            0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.5f, -2.5f, -1.5f, Float.NaN, Float.POSITIVE_INFINITY,
            Float.NEGATIVE_INFINITY, Float.MIN_VALUE, -Float.MIN_VALUE, Float.MIN_NORMAL, 3e9f, -3e9f, 1e-3f
    };

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();

    public GpuTensorOpsParityTest(TrainParity.Api api) {
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
            if (session != null) {
                session.close();
            }
        }
    }

    // ------------------------------------------------------------------ unary

    @Test
    public void bitIdenticalUnaryOps() {
        float[] x = withSpecials(TrainParity.uniform(1, 24 * 40, -10f, 10f));
        assertUnary("neg", x, TensorOps::neg, GpuTensorOps::neg, Tolerance.BIT);
        assertUnary("abs", x, TensorOps::abs, GpuTensorOps::abs, Tolerance.BIT);
        assertUnary("relu", x, TensorOps::relu, GpuTensorOps::relu, Tolerance.BIT);
        assertUnary("clamp", x, t -> TensorOps.clamp(t, -1.5f, 2.5f), t -> GpuTensorOps.clamp(t, -1.5f, 2.5f),
                Tolerance.BIT);
        float[] normal = x.clone();
        float[] floors = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            if (normal[i] != 0.0f && Math.abs(normal[i]) < Float.MIN_NORMAL) {
                normal[i] = Math.copySign(0.25f, normal[i]);
            }
            floors[i] = (float) Math.floor(normal[i]);
        }
        assertBitIdenticalAnyNaN("floor", floors, keep(GpuTensorOps.floor(floats(gpu, normal, x.length)))
                .toFloatArray());
    }

    @Test
    public void singleRoundingUnaryOps() {
        float[] x = withSpecials(TrainParity.uniform(2, 24 * 40, -10f, 10f));
        assertUnary("addScalar", x, t -> TensorOps.add(t, 0.75f), t -> GpuTensorOps.addScalar(t, 0.75f),
                Tolerance.ROUNDING);
        assertUnary("mulScalar", x, t -> TensorOps.mul(t, -1.25f), t -> GpuTensorOps.mulScalar(t, -1.25f),
                Tolerance.ROUNDING);
        assertUnary("leakyRelu", x, t -> TensorOps.leakyRelu(t, 0.01f), t -> GpuTensorOps.leakyRelu(t, 0.01f),
                Tolerance.ROUNDING);
    }

    @Test
    public void transcendentalUnaryOps() {
        float[] x = TrainParity.uniform(3, 24 * 40, -10f, 10f);
        assertUnary("exp", x, TensorOps::exp, GpuTensorOps::exp, Tolerance.TRANSCENDENTAL);
        assertUnary("sigmoid", x, TensorOps::sigmoid, GpuTensorOps::sigmoid, Tolerance.TRANSCENDENTAL);
        assertUnary("tanh", x, TensorOps::tanh, GpuTensorOps::tanh, Tolerance.TRANSCENDENTAL);
        assertUnary("gelu", x, TensorOps::gelu, GpuTensorOps::gelu, Tolerance.TRANSCENDENTAL);
        assertUnary("silu", x, TensorOps::silu, GpuTensorOps::silu, Tolerance.TRANSCENDENTAL);
        assertUnary("elu", x, t -> TensorOps.elu(t, 1.0f), t -> GpuTensorOps.elu(t, 1.0f), Tolerance.TRANSCENDENTAL);
        assertUnary("softplus", x, TensorOps::softplus, GpuTensorOps::softplus, Tolerance.TRANSCENDENTAL);
        assertUnary("mish", x, TensorOps::mish, GpuTensorOps::mish, Tolerance.TRANSCENDENTAL);
        assertUnary("pow3", x, t -> TensorOps.pow(t, 3.0f), t -> GpuTensorOps.pow(t, 3.0f), Tolerance.TRANSCENDENTAL);
        assertUnary("pow0.5", x, t -> TensorOps.pow(t, 0.5f), t -> GpuTensorOps.pow(t, 0.5f),
                Tolerance.TRANSCENDENTAL);
        float[] positive = TrainParity.uniform(4, 24 * 40, 1e-3f, 100f);
        assertUnary("log", positive, TensorOps::log, GpuTensorOps::log, Tolerance.TRANSCENDENTAL);
        assertUnary("sqrt", positive, TensorOps::sqrt, GpuTensorOps::sqrt, Tolerance.DIV_SQRT);
    }

    @Test
    public void roundCastsToJavaRound() {
        float[] x = withSpecials(TrainParity.uniform(5, 1000, -1e4f, 1e4f));
        x[100] = 0.49999997f;
        x[101] = 8388609.0f;
        x[102] = -8388609.5f;
        x[103] = 2.5e9f;
        Tensor rounded = keep(GpuTensorOps.round(floats(gpu, x, x.length)));
        int[] asInt = keep(rounded.to(DType.INT32)).toIntArray();
        for (int i = 0; i < x.length; i++) {
            assertEquals("round of " + x[i], Math.round(x[i]), asInt[i]);
        }
    }

    // ------------------------------------------------------------------ binary, compare, where

    @Test
    public void binaryOpsBroadcastStridedViews() {
        float[] a = TrainParity.uniform(6, 4 * 7, -10f, 10f);
        float[] b = TrainParity.uniform(7, 3, 0.5f, 10f);
        UnaryOperator<Tensor> view = t -> keep(keep(t.reshape(Shape.of(7, 1, 4))).transpose(2, 1, 0));
        Tensor ha = view.apply(floats(host, a, a.length));
        Tensor ga = view.apply(floats(gpu, a, a.length));
        Tensor hb = floats(host, b, 3, 1);
        Tensor gb = floats(gpu, b, 3, 1);
        assertSame("add", keep(TensorOps.add(ha, hb)), keep(GpuTensorOps.add(ga, gb)), Tolerance.ROUNDING);
        assertSame("sub", keep(TensorOps.sub(ha, hb)), keep(GpuTensorOps.sub(ga, gb)), Tolerance.ROUNDING);
        assertSame("mul", keep(TensorOps.mul(ha, hb)), keep(GpuTensorOps.mul(ga, gb)), Tolerance.ROUNDING);
        assertSame("div", keep(TensorOps.div(ha, hb)), keep(GpuTensorOps.div(ga, gb)), Tolerance.DIV_SQRT);
    }

    @Test
    public void binaryAndCompareAtRankSeven() {
        int[] dims = {2, 3, 2, 3, 2, 3, 2};
        int[] perm = {6, 0, 5, 1, 4, 2, 3};
        float[] a = withSpecials(TrainParity.uniform(8, 432, -4f, 4f));
        float[] b = withSpecials(TrainParity.uniform(9, 12, -4f, 4f));
        Tensor ha = keep(floats(host, a, dims).transpose(perm));
        Tensor ga = keep(floats(gpu, a, dims).transpose(perm));
        Tensor hb = floats(host, b, 2, 1, 3, 1, 2, 1);
        Tensor gb = floats(gpu, b, 2, 1, 3, 1, 2, 1);
        assertSame("rank 7 mul", keep(TensorOps.mul(ha, hb)), keep(GpuTensorOps.mul(ga, gb)), Tolerance.ROUNDING);
        assertSameBytes("rank 7 gt", keep(TensorOps.gt(ha, hb)), keep(GpuTensorOps.gt(ga, gb)));
    }

    @Test
    public void compareOpsFollowJavaFloatSemantics() {
        float[] a = withSpecials(TrainParity.uniform(10, 30 * 17, -3f, 3f));
        float[] b = withSpecials(TrainParity.uniform(11, 17, -3f, 3f));
        a[40] = 1.0f;
        b[40 % 17] = 1.0f;
        Tensor ha = keep(floats(host, a, 17, 30).transpose());
        Tensor ga = keep(floats(gpu, a, 17, 30).transpose());
        Tensor hb = floats(host, b, 17);
        Tensor gb = floats(gpu, b, 17);
        assertSameBytes("eq", keep(TensorOps.eq(ha, hb)), keep(GpuTensorOps.eq(ga, gb)));
        assertSameBytes("gt", keep(TensorOps.gt(ha, hb)), keep(GpuTensorOps.gt(ga, gb)));
        assertSameBytes("lt", keep(TensorOps.lt(ha, hb)), keep(GpuTensorOps.lt(ga, gb)));
    }

    @Test
    public void whereWithFloatCondition() {
        float[] c = withSpecials(TrainParity.uniform(12, 19 * 23, -1f, 1f));
        float[] a = TrainParity.uniform(13, 19 * 23, -5f, 5f);
        float[] b = TrainParity.uniform(14, 19 * 23, -5f, 5f);
        Tensor hc = keep(floats(host, c, 19, 23).transpose());
        Tensor gc = keep(floats(gpu, c, 19, 23).transpose());
        Tensor ha = keep(floats(host, a, 19, 23).transpose());
        Tensor ga = keep(floats(gpu, a, 19, 23).transpose());
        Tensor hb = floats(host, b, 23, 19);
        Tensor gb = floats(gpu, b, 23, 19);
        assertSame("where", keep(TensorOps.where(hc, ha, hb)), keep(GpuTensorOps.where(gc, ga, gb)), Tolerance.BIT);
    }

    @Test
    public void whereWithBoolConditionAtByteOffsets() {
        byte[] mask = new byte[9 * 13];
        Random r = new Random(15);
        for (int i = 0; i < mask.length; i++) {
            mask[i] = (byte) (r.nextInt(3) == 0 ? 0 : 1 + r.nextInt(200));
        }
        float[] a = TrainParity.uniform(16, 9 * 8, -5f, 5f);
        float[] b = TrainParity.uniform(17, 9 * 8, -5f, 5f);
        Tensor cond = keep(bools(gpu, mask, 9, 13).slice(1, 3, 11));
        Tensor out = keep(GpuTensorOps.where(cond, floats(gpu, a, 9, 8), floats(gpu, b, 9, 8)));
        float[] expected = new float[9 * 8];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 8; j++) {
                expected[i * 8 + j] = mask[i * 13 + 3 + j] != 0 ? a[i * 8 + j] : b[i * 8 + j];
            }
        }
        TrainParity.assertBitIdentical("bool where", expected, out.toFloatArray());
    }

    @Test
    public void whereAtRankSevenStagesEachPart() {
        int[] dims = {2, 3, 2, 3, 2, 3, 2};
        int[] perm = {1, 0, 3, 2, 5, 4, 6};
        float[] c = withSpecials(TrainParity.uniform(18, 432, -1f, 1f));
        float[] a = TrainParity.uniform(19, 432, -5f, 5f);
        float[] b = TrainParity.uniform(20, 432, -5f, 5f);
        Tensor hc = keep(floats(host, c, dims).transpose(perm));
        Tensor gc = keep(floats(gpu, c, dims).transpose(perm));
        Tensor ha = keep(floats(host, a, dims).transpose(perm));
        Tensor ga = keep(floats(gpu, a, dims).transpose(perm));
        Tensor hb = keep(floats(host, b, dims).transpose(perm));
        Tensor gb = keep(floats(gpu, b, dims).transpose(perm));
        assertSame("rank 7 where", keep(TensorOps.where(hc, ha, hb)), keep(GpuTensorOps.where(gc, ga, gb)),
                Tolerance.BIT);
    }

    // ------------------------------------------------------------------ gather and scatter

    @Test
    public void gatherMatchesCpu() {
        float[] in = TrainParity.uniform(21, 5 * 6 * 4, -5f, 5f);
        int[] idx = randomIndices(22, 3 * 4 * 2, 6);
        Tensor hin = keep(floats(host, in, 4, 6, 5).transpose(2, 1, 0));
        Tensor gin = keep(floats(gpu, in, 4, 6, 5).transpose(2, 1, 0));
        assertSame("gather dim 1", keep(TensorOps.gather(hin, 1, ints(host, idx, 3, 4, 2))),
                keep(GpuTensorOps.gather(gin, 1, ints(gpu, idx, 3, 4, 2))), Tolerance.BIT);
        int[] last = randomIndices(23, 5 * 6 * 7, 4);
        assertSame("gather dim -1", keep(TensorOps.gather(hin, -1, ints(host, last, 5, 6, 7))),
                keep(GpuTensorOps.gather(gin, -1, ints(gpu, last, 5, 6, 7))), Tolerance.BIT);
    }

    @Test
    public void gatherAtRankSevenCropsTheInput() {
        int[] inDims = {2, 3, 2, 3, 2, 3, 4};
        float[] in = TrainParity.uniform(24, 864, -5f, 5f);
        int[][] cases = {{2, 2, 2, 3, 1, 3, 5}, {2, 3, 5, 3, 2, 3, 4}};
        int[] dims = {6, 2};
        for (int c = 0; c < cases.length; c++) {
            int[] idxDims = cases[c];
            int[] idx = randomIndices(25 + c, (int) Shape.of(idxDims).numElements(), inDims[dims[c]]);
            assertSame("rank 7 gather dim " + dims[c],
                    keep(TensorOps.gather(floats(host, in, inDims), dims[c], ints(host, idx, idxDims))),
                    keep(GpuTensorOps.gather(floats(gpu, in, inDims), dims[c], ints(gpu, idx, idxDims))),
                    Tolerance.BIT);
        }
    }

    @Test
    public void outOfRangeGatherIndexThrowsAtReadback() {
        Tensor out = keep(GpuTensorOps.gather(floats(gpu, new float[]{1f, 2f, 3f}, 3), 0,
                ints(gpu, new int[]{0, 3, 1}, 3)));
        assertNotNull(out);
        assertThrows(IndexOutOfBoundsException.class, out::toFloatArray);
        TrainParity.assertBitIdentical("after the error", new float[]{1f, 0f, 2f}, out.toFloatArray());
    }

    @Test
    public void scatterMatchesCpuLastWriteWins() {
        float[] in = TrainParity.uniform(27, 5 * 6 * 4, -5f, 5f);
        int[] idx = randomIndices(28, 3 * 7 * 4, 6);
        float[] src = TrainParity.uniform(29, 3 * 7 * 4, -5f, 5f);
        Tensor hsrc = keep(floats(host, src, 4, 7, 3).transpose(2, 1, 0));
        Tensor gsrc = keep(floats(gpu, src, 4, 7, 3).transpose(2, 1, 0));
        assertSame("scatter", keep(TensorOps.scatter(floats(host, in, 5, 6, 4), 1, ints(host, idx, 3, 7, 4), hsrc)),
                keep(GpuTensorOps.scatter(floats(gpu, in, 5, 6, 4), 1, ints(gpu, idx, 3, 7, 4), gsrc)),
                Tolerance.BIT);
    }

    @Test
    public void scatterAtRankSeven() {
        int[][] inDims = {{2, 3, 2, 3, 2, 3, 4}, {3, 2, 2, 2, 2, 2, 5}};
        int[][] idxDims = {{2, 2, 2, 3, 1, 3, 6}, {2, 2, 2, 2, 2, 2, 7}};
        for (int c = 0; c < inDims.length; c++) {
            int count = (int) Shape.of(inDims[c]).numElements();
            int idxCount = (int) Shape.of(idxDims[c]).numElements();
            float[] in = TrainParity.uniform(30 + c, count, -5f, 5f);
            int[] idx = randomIndices(32 + c, idxCount, inDims[c][6]);
            float[] src = TrainParity.uniform(34 + c, idxCount, -5f, 5f);
            assertSame("rank 7 scatter " + c, keep(TensorOps.scatter(floats(host, in, inDims[c]), 6,
                            ints(host, idx, idxDims[c]), floats(host, src, idxDims[c]))),
                    keep(GpuTensorOps.scatter(floats(gpu, in, inDims[c]), 6, ints(gpu, idx, idxDims[c]),
                            floats(gpu, src, idxDims[c]))), Tolerance.BIT);
            float[] expected = scatterAddReference(in, inDims[c], 6, idx, idxDims[c], src);
            TrainParity.assertSingleRounding("rank 7 scatterAdd " + c, expected, keep(GpuTensorOps.scatterAdd(
                    floats(gpu, in, inDims[c]), 6, ints(gpu, idx, idxDims[c]), floats(gpu, src, idxDims[c])))
                    .toFloatArray());
        }
    }

    @Test
    public void scatterAddAccumulatesInIndexOrder() {
        float[] in = TrainParity.uniform(36, 6 * 5, -5f, 5f);
        int[] idx = randomIndices(37, 9 * 5, 6);
        float[] src = TrainParity.uniform(38, 9 * 5, -5f, 5f);
        float[] expected = scatterAddReference(in, new int[]{6, 5}, 0, idx, new int[]{9, 5}, src);
        Tensor out = keep(GpuTensorOps.scatterAdd(floats(gpu, in, 6, 5), 0, ints(gpu, idx, 9, 5),
                floats(gpu, src, 9, 5)));
        TrainParity.assertSingleRounding("scatterAdd", expected, out.toFloatArray());
    }

    // ------------------------------------------------------------------ in place

    @Test
    public void axpbyAndScaleInPlaceOnViews() {
        float[] base = TrainParity.uniform(39, 12 * 10, -5f, 5f);
        float[] xs = TrainParity.uniform(40, 10 * 6, -5f, 5f);
        Tensor y = floats(gpu, base, 12, 10);
        Tensor slice = keep(y.slice(0, 3, 9));
        assertTrue(GpuTensorOps.axpbyInPlace(slice, 0.5f, floats(gpu, xs, 6, 10), 1.0f));
        Tensor columns = keep(keep(y.transpose()).slice(0, 1, 7));
        assertTrue(GpuTensorOps.scaleInPlace(columns, -2.0f));
        Tensor alias = keep(y.slice(0, 0, 6));
        assertTrue(GpuTensorOps.axpbyInPlace(alias, 3.0f, keep(y.slice(0, 6, 12)), 0.0f));
        float[] expected = base.clone();
        for (int i = 3; i < 9; i++) {
            for (int j = 0; j < 10; j++) {
                expected[i * 10 + j] = 0.5f * xs[(i - 3) * 10 + j] + 1.0f * expected[i * 10 + j];
            }
        }
        for (int i = 0; i < 12; i++) {
            for (int j = 1; j < 7; j++) {
                expected[i * 10 + j] = expected[i * 10 + j] * -2.0f;
            }
        }
        for (int k = 0; k < 60; k++) {
            expected[k] = 3.0f * expected[60 + k];
        }
        TrainParity.assertSingleRounding("axpby/scale", expected, y.toFloatArray());
    }

    // ------------------------------------------------------------------ activation backward and dropout

    @Test
    public void unaryBackwardMatchesGradFunctions() {
        int n = 31 * 33;
        float[] x = TrainParity.uniform(41, n, -10f, 10f);
        float[] gy = TrainParity.uniform(42, n, -2f, 2f);
        x[0] = 0.0f;
        x[1] = -0.0f;
        x[2] = Float.NaN;
        x[3] = -1.5f;
        x[4] = 2.5f;
        Tensor hx = floats(host, x, n);
        Tensor devX = floats(gpu, x, n);
        Tensor devGy = floats(gpu, gy, n);
        float alpha = 0.7f;
        float[] sig = keep(TensorOps.sigmoid(hx)).toFloatArray();
        float[] tanh = keep(TensorOps.tanh(hx)).toFloatArray();
        float[] elu = keep(TensorOps.elu(hx, alpha)).toFloatArray();
        for (Backward op : Backward.values()) {
            float p0 = op == Backward.CLAMP ? -1.5f : op == Backward.LEAKY_RELU ? 0.01f : alpha;
            float p1 = op == Backward.CLAMP ? 2.5f : 0.0f;
            float[] y = op == Backward.SIGMOID ? sig : op == Backward.TANH ? tanh : op == Backward.ELU ? elu : null;
            Tensor devY = y == null ? null : floats(gpu, y, n);
            float[] expected = new float[n];
            for (int i = 0; i < n; i++) {
                expected[i] = backwardReference(op, x[i], y == null ? 0.0f : y[i], gy[i], p0, p1);
            }
            float[] got = keep(GpuTensorOps.unaryBackward(op, op.readsX() ? devX : null, devY, devGy, p0, p1))
                    .toFloatArray();
            boolean transcendental = op == Backward.GELU || op == Backward.SILU || op == Backward.SOFTPLUS
                    || op == Backward.MISH;
            if (transcendental) {
                TrainParity.assertTranscendental("backward " + op, expected, got);
            } else {
                TrainParity.assertSingleRounding("backward " + op, expected, got);
            }
        }
    }

    @Test
    public void maskScaleMatchesDropout() {
        int n = 1000;
        float[] x = withSpecials(TrainParity.uniform(43, n, -5f, 5f));
        float[] m = new float[n];
        Random r = new Random(44);
        for (int i = 0; i < n; i++) {
            m[i] = r.nextFloat() < 0.3f ? (i % 2 == 0 ? 0.0f : -0.0f) : 1.0f;
        }
        float scale = 1.0f / 0.7f;
        float[] expected = new float[n];
        for (int i = 0; i < n; i++) {
            expected[i] = m[i] == 0.0f ? 0.0f : (x[i] * m[i]) * scale;
        }
        Tensor out = keep(GpuTensorOps.maskScale(floats(gpu, x, n), floats(gpu, m, n), scale));
        TrainParity.assertSingleRounding("maskScale", expected, out.toFloatArray());
    }

    // ------------------------------------------------------------------ dispatch contract

    @Test
    public void cpuPathAndUnsupportedDtypesReturnNull() {
        Tensor hostTensor = floats(host, new float[]{1f, 2f}, 2);
        assertFalse(GpuTensorOps.shouldUseGpu(hostTensor));
        assertNull(GpuTensorOps.relu(hostTensor));
        assertFalse(GpuTensorOps.scaleInPlace(hostTensor, 2f));
        Tensor intTensor = ints(gpu, new int[]{1, 2}, 2);
        assertFalse(GpuTensorOps.shouldUseGpu(intTensor));
        assertNull(GpuTensorOps.neg(intTensor));
        assertTrue(GpuTensorOps.shouldUseGpu(floats(gpu, new float[]{1f}, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> GpuTensorOps.add(floats(gpu, new float[3], 3), floats(gpu, new float[2], 2)));
    }

    // ------------------------------------------------------------------ references and helpers

    /** The GradFunctions formulas (ReluGrad through MishGrad, AbsGrad, ClampGrad) in their float operation order. */
    private static float backwardReference(Backward op, float x, float y, float g, float p0, float p1) {
        float s = (float) Math.sqrt(2.0 / Math.PI);
        switch (op) {
            case RELU:
                return x > 0 ? g : 0.0f;
            case GELU: {
                float x3 = x * x * x;
                float t = (float) Math.tanh(s * (x + 0.044715f * x3));
                float d = s * (1.0f + 3.0f * 0.044715f * x * x);
                return g * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * d);
            }
            case SILU: {
                float sg = (float) (1.0 / (1.0 + Math.exp(-x)));
                return g * (sg * (1.0f + x * (1.0f - sg)));
            }
            case SIGMOID:
                return g * y * (1.0f - y);
            case TANH:
                return g * (1.0f - y * y);
            case LEAKY_RELU:
                return x > 0 ? g : p0 * g;
            case ELU:
                return x > 0 ? g : g * (y + p0);
            case SOFTPLUS:
                return g * (float) (1.0 / (1.0 + Math.exp(-x)));
            case MISH: {
                float t = (float) Math.tanh((float) Math.log(1.0 + Math.exp(x)));
                float sg = (float) (1.0 / (1.0 + Math.exp(-x)));
                return g * (t + x * (1.0f - t * t) * sg);
            }
            case ABS:
                return x > 0 ? g : x < 0 ? -g : 0.0f;
            case CLAMP:
                return x < p0 || x > p1 ? 0.0f : g;
            default:
                throw new IllegalArgumentException("no reference for " + op);
        }
    }

    /** GradFunctions.GatherGrad's loop, starting from {@code in} instead of zeros. */
    private static float[] scatterAddReference(float[] in, int[] inDims, int dim, int[] idx, int[] idxDims,
                                               float[] src) {
        float[] out = in.clone();
        int rank = inDims.length;
        int[] coord = new int[rank];
        for (int i = 0; i < idx.length; i++) {
            int rem = i;
            for (int k = rank - 1; k >= 0; k--) {
                coord[k] = rem % idxDims[k];
                rem /= idxDims[k];
            }
            int lin = 0;
            for (int k = 0; k < rank; k++) {
                lin = lin * inDims[k] + (k == dim ? idx[i] : coord[k]);
            }
            out[lin] += src[i];
        }
        return out;
    }

    private enum Tolerance { BIT, ROUNDING, DIV_SQRT, TRANSCENDENTAL }

    private void assertUnary(String what, float[] x, UnaryOperator<Tensor> cpu, UnaryOperator<Tensor> dev,
                             Tolerance tol) {
        UnaryOperator<Tensor> view = t -> keep(keep(t.reshape(Shape.of(40, 24))).transpose());
        Tensor ref = keep(cpu.apply(view.apply(floats(host, x, x.length))));
        Tensor got = keep(dev.apply(view.apply(floats(gpu, x, x.length))));
        assertSame(what, ref, got, tol);
    }

    private void assertSame(String what, Tensor ref, Tensor got, Tolerance tol) {
        assertNotNull(what + ": the GPU path declined", got);
        assertEquals(what + ": shape", ref.shape(), got.shape());
        float[] r = ref.toFloatArray();
        float[] g = got.toFloatArray();
        switch (tol) {
            case BIT:
                TrainParity.assertBitIdentical(what, r, g);
                break;
            case ROUNDING:
                TrainParity.assertSingleRounding(what, r, g);
                break;
            case DIV_SQRT:
                TrainParity.assertDivSqrt(what, r, g);
                break;
            default:
                TrainParity.assertTranscendental(what, r, g);
                break;
        }
    }

    private void assertSameBytes(String what, Tensor ref, Tensor got) {
        assertNotNull(what + ": the GPU path declined", got);
        assertEquals(what + ": shape", ref.shape(), got.shape());
        assertEquals(what + ": dtype", DType.BOOL, got.dtype());
        TrainParity.assertBitIdentical(what, bytes(ref), bytes(got));
    }

    /**
     * Bit-identical on raw bits, except that a NaN reference accepts any NaN: arithmetic such as floor may return a
     * canonical NaN on the device (NVIDIA returns {@code 0x7fffffff}) where Java keeps the input's payload.
     */
    private static void assertBitIdenticalAnyNaN(String what, float[] ref, float[] got) {
        assertEquals(what + ": length", ref.length, got.length);
        for (int i = 0; i < ref.length; i++) {
            boolean same = Float.isNaN(ref[i]) ? Float.isNaN(got[i])
                    : Float.floatToRawIntBits(ref[i]) == Float.floatToRawIntBits(got[i]);
            assertTrue(what + ": element " + i + " differs: ref " + ref[i] + ", gpu " + got[i], same);
        }
    }

    private static byte[] bytes(Tensor t) {
        ByteBuffer bb = t.toByteBuffer();
        byte[] out = new byte[bb.remaining()];
        bb.get(out);
        return out;
    }

    private static float[] withSpecials(float[] values) {
        System.arraycopy(SPECIAL, 0, values, 0, Math.min(SPECIAL.length, values.length));
        return values;
    }

    private static int[] randomIndices(long seed, int n, int bound) {
        Random r = new Random(seed);
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = r.nextInt(bound);
        }
        return out;
    }

    private Tensor keep(Tensor t) {
        if (t != null) {
            owned.add(t);
        }
        return t;
    }

    private Tensor floats(ComputeBackend b, float[] data, int... dims) {
        return keep(keep(Tensor.fromArray(b, data)).reshape(Shape.of(dims)));
    }

    private Tensor ints(ComputeBackend b, int[] data, int... dims) {
        return keep(keep(Tensor.fromArray(b, data)).reshape(Shape.of(dims)));
    }

    private Tensor bools(ComputeBackend b, byte[] data, int... dims) {
        TensorStorage st = TensorStorage.fromHost(b, ByteBuffer.wrap(data), DType.BOOL);
        return keep(Tensor.fromStorage(b, Shape.of(dims), DType.BOOL, st));
    }
}
