package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradientCheckUtil;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.awayFrom;
import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.distinct;
import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.grid;
import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.in;
import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.uniform;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Gradient checks of every differentiable {@link TensorOps} operation, the recorded {@link Tensor} views and casts, and
 * the {@link ConvOps} and {@link NormOps} forwards: the gradient one recorded backward pass gives each input is
 * compared with central finite differences ({@link GradientCheckUtil#gradCheck}), on the host reference backend and on
 * every available {@link TrainParity} GPU API. Each checked scalar is {@code sum(op(x) * w)} with fixed pseudo-random
 * weights {@code w} ({@link GradCheck#weighted}), so a gradient routed to the wrong element, or a gradient that does
 * not depend on the upstream gradient, fails.
 *
 * <p>Every input lies in [-1, 1]. Kinks are avoided by the input generators of {@link GradCheck}:
 * inputs of relu, leaky relu, elu, abs and clamp
 * keep a distance of at least {@link GradCheck#KINK_MARGIN} from every kink; inputs of max, min, max pooling and
 * segment MAX are distinct values on a grid whose spacing exceeds twice the finite-difference step, so no
 * perturbation changes which element is selected; casts use values on a grid of 1/64 and a step of 1/16, so every
 * perturbed value is exact in FLOAT16 and BFLOAT16.</p>
 */
@RunWith(Parameterized.class)
public class TensorOpsGradCheckTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return GradCheck.targets();
    }

    private final GradCheck.Target target;
    private GradCheck fx;

    public TensorOpsGradCheckTest(GradCheck.Target target) {
        this.target = target;
    }

    @Before
    public void setUp() {
        fx = GradCheck.open(target);
    }

    @After
    public void tearDown() {
        if (fx != null) {
            fx.close();
        }
    }

    @Test
    public void broadcastArithmetic() {
        fx.inputs("add", a -> TensorOps.add(a.get(0), a.get(1)),
                in(uniform(1, 24, -1f, 1f), 2, 3, 4), in(uniform(2, 3, -1f, 1f), 3, 1));
        fx.inputs("sub", a -> TensorOps.sub(a.get(0), a.get(1)),
                in(uniform(3, 24, -1f, 1f), 2, 3, 4), in(uniform(4, 4, -1f, 1f), 4));
        fx.inputs("mul", a -> TensorOps.mul(a.get(0), a.get(1)),
                in(uniform(5, 12, -1f, 1f), 3, 4), in(uniform(6, 8, -1f, 1f), 2, 1, 4));
        fx.inputs("div", a -> TensorOps.div(a.get(0), a.get(1)),
                in(uniform(7, 6, -1f, 1f), 2, 3), in(uniform(8, 3, 0.5f, 1f), 3));
    }

    @Test
    public void scalarAndUnaryArithmetic() {
        GradCheck.Input x = in(uniform(11, 12, -1f, 1f), 3, 4);
        GradCheck.Input positive = in(uniform(12, 12, 0.25f, 1f), 3, 4);
        fx.inputs("add scalar", a -> TensorOps.add(a.get(0), 0.75f), x);
        fx.inputs("mul scalar", a -> TensorOps.mul(a.get(0), -1.5f), x);
        fx.inputs("neg", a -> TensorOps.neg(a.get(0)), x);
        fx.inputs("pow 3", a -> TensorOps.pow(a.get(0), 3.0f), x);
        fx.inputs("pow 2.5", a -> TensorOps.pow(a.get(0), 2.5f), positive);
        fx.inputs("sqrt", a -> TensorOps.sqrt(a.get(0)), positive);
        fx.inputs("exp", a -> TensorOps.exp(a.get(0)), x);
        fx.inputs("log", a -> TensorOps.log(a.get(0)), positive);
    }

    @Test
    public void piecewiseActivations() {
        GradCheck.Input x = in(awayFrom(21, 12, -1f, 1f, 0f), 3, 4);
        fx.inputs("relu", a -> TensorOps.relu(a.get(0)), x);
        fx.inputs("leaky relu", a -> TensorOps.leakyRelu(a.get(0), 0.1f), x);
        fx.inputs("elu", a -> TensorOps.elu(a.get(0), 0.7f), x);
        fx.inputs("abs", a -> TensorOps.abs(a.get(0)), x);
        fx.inputs("clamp", a -> TensorOps.clamp(a.get(0), -0.5f, 0.5f),
                in(awayFrom(22, 12, -1f, 1f, -0.5f, 0.5f), 3, 4));
    }

    @Test
    public void smoothActivations() {
        GradCheck.Input x = in(uniform(31, 12, -1f, 1f), 3, 4);
        fx.inputs("gelu", a -> TensorOps.gelu(a.get(0)), x);
        fx.inputs("silu", a -> TensorOps.silu(a.get(0)), x);
        fx.inputs("sigmoid", a -> TensorOps.sigmoid(a.get(0)), x);
        fx.inputs("tanh", a -> TensorOps.tanh(a.get(0)), x);
        fx.inputs("softplus", a -> TensorOps.softplus(a.get(0)), x);
        fx.inputs("mish", a -> TensorOps.mish(a.get(0)), x);
    }

    @Test
    public void gatedActivations() {
        fx.inputs("glu", a -> TensorOps.glu(a.get(0), 1), in(uniform(41, 12, -1f, 1f), 2, 6));
        fx.inputs("geglu", a -> TensorOps.geglu(a.get(0), -1), in(uniform(42, 24, -1f, 1f), 2, 3, 4));
    }

    @Test
    public void softmaxFamily() {
        GradCheck.Input x = in(uniform(51, 15, -1f, 1f), 3, 5);
        fx.inputs("softmax dim 1", a -> TensorOps.softmax(a.get(0), 1), x);
        fx.inputs("softmax dim 0", a -> TensorOps.softmax(a.get(0), 0), x);
        fx.inputs("log softmax", a -> TensorOps.logSoftmax(a.get(0), -1), x);
    }

    /** Masked ({@code -inf}) entries and an all-masked row get exactly 0, and so does their gradient. */
    @Test
    public void maskedSoftmax() {
        float inf = Float.NEGATIVE_INFINITY;
        float[] values = {0.3f, inf, -0.7f, 1.1f, -0.2f, 0.9f, 0.4f, -1.3f, inf, inf, inf, inf};
        fx.inputs("masked softmax", a -> TensorOps.maskedSoftmax(a.get(0), 1), in(values, 3, 4));
    }

    @Test
    public void matrixProducts() {
        fx.inputs("matmul", a -> TensorOps.matmul(a.get(0), a.get(1)),
                in(uniform(61, 12, -1f, 1f), 3, 4), in(uniform(62, 8, -1f, 1f), 4, 2));
        fx.inputs("matmul broadcast batch", a -> TensorOps.matmul(a.get(0), a.get(1)),
                in(uniform(63, 24, -1f, 1f), 2, 3, 4), in(uniform(64, 20, -1f, 1f), 4, 5));
        fx.inputs("matmul vector-matrix", a -> TensorOps.matmul(a.get(0), a.get(1)),
                in(uniform(65, 4, -1f, 1f), 4), in(uniform(66, 12, -1f, 1f), 4, 3));
        fx.inputs("matmul matrix-vector", a -> TensorOps.matmul(a.get(0), a.get(1)),
                in(uniform(67, 12, -1f, 1f), 3, 4), in(uniform(68, 4, -1f, 1f), 4));
        fx.inputs("bmm", a -> TensorOps.bmm(a.get(0), a.get(1)),
                in(uniform(69, 24, -1f, 1f), 2, 3, 4), in(uniform(70, 16, -1f, 1f), 2, 4, 2));
    }

    @Test
    public void reductions() {
        GradCheck.Input x = in(uniform(71, 24, -1f, 1f), 2, 3, 4);
        fx.inputs("sum dims 0 2", a -> TensorOps.sum(a.get(0), 0, 2), x);
        fx.inputs("sum all", a -> TensorOps.sum(a.get(0)), x);
        fx.inputs("mean dim 1", a -> TensorOps.mean(a.get(0), 1), x);
        fx.inputs("mean all", a -> TensorOps.mean(a.get(0)), x);
        fx.inputs("var dim 1", a -> TensorOps.var(a.get(0), 1), x);
        fx.inputs("var dims 0 2", a -> TensorOps.var(a.get(0), 0, 2), x);
        fx.inputs("var all", a -> TensorOps.var(a.get(0)), x);
    }

    @Test
    public void selectionReductions() {
        GradCheck.Input x = in(distinct(81, 20), 4, 5);
        fx.inputs("max dim 1", a -> TensorOps.max(a.get(0), 1), x);
        fx.inputs("min dim 0", a -> TensorOps.min(a.get(0), 0), x);
    }

    @Test
    public void sumToShape() {
        GradCheck.Input x = in(uniform(91, 24, -1f, 1f), 2, 3, 4);
        fx.inputs("sumToShape [3,1]", a -> TensorOps.sumToShape(a.get(0), Shape.of(3, 1)), x);
        fx.inputs("sumToShape [4]", a -> TensorOps.sumToShape(a.get(0), Shape.of(4)), x);
        fx.inputs("sumToShape rearranging", a -> TensorOps.sumToShape(a.get(0), Shape.of(4, 6)), x);
    }

    @Test
    public void views() {
        GradCheck.Input matrix = in(uniform(101, 12, -1f, 1f), 3, 4);
        GradCheck.Input cube = in(uniform(102, 24, -1f, 1f), 2, 3, 4);
        fx.inputs("reshape", a -> a.get(0).reshape(4, 3), matrix);
        fx.inputs("transpose", a -> a.get(0).transpose(), matrix);
        fx.inputs("permute", a -> a.get(0).permute(2, 0, 1), cube);
        fx.inputs("slice", a -> a.get(0).slice(1, 1, 3), cube);
        fx.inputs("squeeze", a -> a.get(0).squeeze(1), in(uniform(103, 12, -1f, 1f), 3, 1, 4));
        fx.inputs("unsqueeze", a -> a.get(0).unsqueeze(0), matrix);
        fx.inputs("expand", a -> a.get(0).expand(2, 3, 4), in(uniform(104, 3, -1f, 1f), 3, 1));
        fx.inputs("contiguous of a transpose",
                a -> GradCheck.then(a.get(0).transpose(), a, Tensor::contiguous), matrix);
        fx.inputs("contiguous of an expand",
                a -> GradCheck.then(a.get(0).expand(2, 3, 4), a, Tensor::contiguous),
                in(uniform(105, 4, -1f, 1f), 1, 1, 4));
        fx.inputs("reshape of a transpose (copying)",
                a -> GradCheck.then(a.get(0).transpose(), a, t -> t.reshape(2, 6)), matrix);
        fx.inputs("squeeze every unit axis", a -> a.get(0).squeeze(), in(uniform(106, 12, -1f, 1f), 1, 3, 1, 4));
    }

    /** Values on a grid of 1/64 perturbed by 1/16 stay exact in FLOAT16 and BFLOAT16, so the cast is the identity. */
    @Test
    public void halfPrecisionCasts() {
        GradCheck.Input x = in(grid(111, 12, 64), 3, 4);
        float step = 1.0f / 16;
        fx.inputs("cast FLOAT16", step, GradCheck.TOLERANCE,
                a -> GradCheck.then(a.get(0).to(DType.FLOAT16), a, h -> h.to(DType.FLOAT32)), x);
        fx.inputs("cast BFLOAT16", step, GradCheck.TOLERANCE,
                a -> GradCheck.then(a.get(0).to(DType.BFLOAT16), a, h -> h.to(DType.FLOAT32)), x);
    }

    @Test
    public void catStackSplit() {
        fx.inputs("cat", a -> TensorOps.cat(a, 1),
                in(uniform(121, 6, -1f, 1f), 2, 1, 3), in(uniform(122, 12, -1f, 1f), 2, 2, 3),
                in(uniform(123, 18, -1f, 1f), 2, 3, 3));
        fx.inputs("stack", a -> TensorOps.stack(a, 1),
                in(uniform(124, 6, -1f, 1f), 2, 3), in(uniform(125, 6, -1f, 1f), 2, 3));
        fx.inputs("split, middle part unused", a -> {
            List<Tensor> parts = TensorOps.split(a.get(0), 3, 1);
            parts.get(1).close();
            try (Tensor first = GradCheck.weighted(parts.get(0), a, 1);
                 Tensor last = GradCheck.weighted(parts.get(2), a, 2)) {
                return TensorOps.add(first, last);
            }
        }, in(uniform(126, 12, -1f, 1f), 2, 6));
    }

    @Test
    public void gatherAndScatter() {
        Tensor gatherIndex = fx.int32(new int[]{0, 3, 3, 1, 1, 2, 2, 0, 0}, 3, 3);
        Tensor uniqueIndex = fx.int32(new int[]{0, 2, 3, 1, 2, 0}, 3, 2);
        Tensor repeatedIndex = fx.int32(new int[]{0, 0, 2, 1, 3, 1, 2, 2, 2}, 3, 3);
        fx.inputs("gather", a -> TensorOps.gather(a.get(0), 1, gatherIndex), in(uniform(131, 12, -1f, 1f), 3, 4));
        fx.inputs("scatter", a -> TensorOps.scatter(a.get(0), 1, uniqueIndex, a.get(1)),
                in(uniform(132, 12, -1f, 1f), 3, 4), in(uniform(133, 6, -1f, 1f), 3, 2));
        fx.inputs("scatterAdd", a -> TensorOps.scatterAdd(a.get(0), 1, repeatedIndex, a.get(1)),
                in(uniform(134, 12, -1f, 1f), 3, 4), in(uniform(135, 9, -1f, 1f), 3, 3));
    }

    @Test
    public void where() {
        Tensor floatCondition = fx.constant(new float[]{1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0}, 3, 4);
        Tensor boolCondition = fx.keep(TensorOps.gt(floatCondition, fx.constant(new float[12], 3, 4)));
        GradCheck.Input a = in(uniform(141, 12, -1f, 1f), 3, 4);
        GradCheck.Input b = in(uniform(142, 12, -1f, 1f), 3, 4);
        fx.inputs("where FLOAT32 condition", t -> TensorOps.where(floatCondition, t.get(0), t.get(1)), a, b);
        fx.inputs("where BOOL condition", t -> TensorOps.where(boolCondition, t.get(0), t.get(1)), a, b);
    }

    @Test
    public void indexSelectRows() {
        Tensor repeated = fx.int32(new int[]{0, 4, 4, 2, 0, 1}, 2, 3);
        Tensor outOfRange = fx.int32(new int[]{-1, 2, 9, 4}, 4);
        GradCheck.Input table = in(uniform(151, 15, -1f, 1f), 5, 3);
        fx.inputs("indexSelectRows", a -> TensorOps.indexSelectRows(a.get(0), repeated, false), table);
        fx.inputs("indexSelectRows clamped", a -> TensorOps.indexSelectRows(a.get(0), outOfRange, true), table);
    }

    /** Three segments over five source rows in edge order {3, 0 | - | 4, 1, 2}; the middle one is empty. */
    @Test
    public void segmentReduce() {
        Tensor ptr = fx.int32(new int[]{0, 2, 2, 5}, 4);
        Tensor edges = fx.int32(new int[]{3, 0, 4, 1, 2}, 5);
        GradCheck.Input src = in(uniform(161, 15, -1f, 1f), 5, 3);
        fx.inputs("segment SUM",
                a -> TensorOps.segmentReduce(a.get(0), ptr, edges, 3, GpuReduceOps.SegmentOp.SUM, 0f), src);
        fx.inputs("segment MEAN",
                a -> TensorOps.segmentReduce(a.get(0), ptr, edges, 3, GpuReduceOps.SegmentOp.MEAN, 0f), src);
        fx.inputs("segment MAX",
                a -> TensorOps.segmentReduce(a.get(0), ptr, edges, 3, GpuReduceOps.SegmentOp.MAX, 0f),
                in(distinct(162, 15), 5, 3));
    }

    @Test
    public void maskScale() {
        Tensor mask = fx.constant(new float[]{0, 2, 2, 0, 2, 0, 2, 2, 0, 0, 2, 2}, 3, 4);
        fx.inputs("maskScale", a -> TensorOps.maskScale(a.get(0), mask, 0.5f), in(uniform(171, 12, -1f, 1f), 3, 4));
    }

    @Test
    public void convolutions() {
        fx.inputs("conv2d grouped, strided, padded", a -> ConvOps.conv2d(a.get(0), a.get(1), a.get(2),
                        new int[]{3, 3}, new int[]{2, 1}, new int[]{1, 0}, new int[]{1, 1}, 2),
                in(uniform(181, 128, -1f, 1f), 2, 4, 4, 4), in(uniform(182, 72, -0.5f, 0.5f), 4, 2, 3, 3),
                in(uniform(183, 4, -0.5f, 0.5f), 4));
        fx.inputs("conv2d dilated without bias", a -> ConvOps.conv2d(a.get(0), a.get(1), null,
                        new int[]{2, 2}, new int[]{1, 1}, new int[]{1, 1}, new int[]{2, 2}, 1),
                in(uniform(184, 50, -1f, 1f), 1, 2, 5, 5), in(uniform(185, 24, -0.5f, 0.5f), 3, 2, 2, 2));
        fx.inputs("conv1d", a -> ConvOps.conv1d(a.get(0), a.get(1), a.get(2), 3, 2, 1, 1, 1),
                in(uniform(186, 42, -1f, 1f), 2, 3, 7), in(uniform(187, 36, -0.5f, 0.5f), 4, 3, 3),
                in(uniform(188, 4, -0.5f, 0.5f), 4));
        fx.inputs("conv1d depthwise dilated", a -> ConvOps.conv1d(a.get(0), a.get(1), a.get(2), 2, 1, 0, 2, 3),
                in(uniform(189, 18, -1f, 1f), 1, 3, 6), in(uniform(190, 6, -0.5f, 0.5f), 3, 1, 2),
                in(uniform(191, 3, -0.5f, 0.5f), 3));
    }

    /** The second case has {@code stride <= outputPadding < dilation}, where the input gradient is cut to H x W. */
    @Test
    public void transposedConvolutions() {
        fx.inputs("convTranspose2d strided", a -> ConvOps.convTranspose2d(a.get(0), a.get(1), a.get(2),
                        new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1}, new int[]{1, 0}, new int[]{1, 1}, 1),
                in(uniform(201, 18, -1f, 1f), 1, 2, 3, 3), in(uniform(202, 54, -0.5f, 0.5f), 2, 3, 3, 3),
                in(uniform(203, 3, -0.5f, 0.5f), 3));
        fx.inputs("convTranspose2d output padding beyond stride", a -> ConvOps.convTranspose2d(a.get(0), a.get(1),
                        null, new int[]{2, 2}, new int[]{1, 1}, new int[]{0, 0}, new int[]{1, 1}, new int[]{2, 2}, 1),
                in(uniform(204, 18, -1f, 1f), 1, 2, 3, 3), in(uniform(205, 16, -0.5f, 0.5f), 2, 2, 2, 2));
    }

    @Test
    public void pooling() {
        GradCheck.Input distinctValues = in(distinct(211, 50), 1, 2, 5, 5);
        GradCheck.Input smooth = in(uniform(212, 50, -1f, 1f), 1, 2, 5, 5);
        fx.inputs("maxPool2d ceil mode", a -> ConvOps.maxPool2d(a.get(0), new int[]{2, 2}, new int[]{2, 2},
                new int[]{0, 0}, new int[]{1, 1}, true), distinctValues);
        fx.inputs("maxPool2d overlapping padded", a -> ConvOps.maxPool2d(a.get(0), new int[]{3, 3},
                new int[]{2, 2}, new int[]{1, 1}, new int[]{1, 1}, false), distinctValues);
        fx.inputs("maxPool2d dilated", a -> ConvOps.maxPool2d(a.get(0), new int[]{2, 2}, new int[]{1, 1},
                new int[]{0, 0}, new int[]{2, 2}, false), distinctValues);
        fx.inputs("avgPool2d padded", a -> ConvOps.avgPool2d(a.get(0), new int[]{3, 3}, new int[]{2, 2},
                new int[]{1, 1}, false, false), smooth);
        fx.inputs("avgPool2d counting padding", a -> ConvOps.avgPool2d(a.get(0), new int[]{3, 3},
                new int[]{2, 2}, new int[]{1, 1}, false, true), smooth);
        fx.inputs("avgPool2d ceil mode", a -> ConvOps.avgPool2d(a.get(0), new int[]{2, 2}, new int[]{2, 2},
                new int[]{0, 0}, true, true), smooth);
        fx.inputs("adaptiveAvgPool2d uneven", a -> ConvOps.adaptiveAvgPool2d(a.get(0), 3, 3),
                in(uniform(213, 40, -1f, 1f), 1, 2, 5, 4));
        fx.inputs("adaptiveAvgPool2d upsampling", a -> ConvOps.adaptiveAvgPool2d(a.get(0), 3, 3),
                in(uniform(214, 4, -1f, 1f), 1, 1, 2, 2));
    }

    @Test
    public void layerAndRmsNorm() {
        fx.inputs("layerNorm affine", a -> NormOps.layerNorm(a.get(0), 4, a.get(1), a.get(2), 1e-5f),
                in(uniform(221, 12, -1f, 1f), 3, 4), in(uniform(222, 4, 0.5f, 1f), 4),
                in(uniform(223, 4, -0.5f, 0.5f), 4));
        fx.inputs("layerNorm plain", a -> NormOps.layerNorm(a.get(0), 6, null, null, 1e-5f),
                in(uniform(224, 12, -1f, 1f), 2, 2, 3));
        fx.inputs("rmsNorm weighted", a -> NormOps.rmsNorm(a.get(0), a.get(1), 1e-5f),
                in(uniform(225, 15, -1f, 1f), 3, 5), in(uniform(226, 5, 0.5f, 1f), 5));
        fx.inputs("rmsNorm plain", a -> NormOps.rmsNorm(a.get(0), null, 1e-5f), in(uniform(227, 8, -1f, 1f), 2, 4));
    }

    @Test
    public void batchNorm() {
        Tensor runningMean = fx.constant(new float[]{0.2f, -0.3f}, 2);
        Tensor runningVar = fx.constant(new float[]{0.8f, 1.4f}, 2);
        fx.inputs("batchNorm training", a -> NormOps.batchNorm(a.get(0), a.get(1), a.get(2), null, null, true,
                        0.1f, 1e-5f),
                in(uniform(231, 24, -1f, 1f), 4, 3, 2), in(uniform(232, 3, 0.5f, 1f), 3),
                in(uniform(233, 3, -0.5f, 0.5f), 3));
        fx.inputs("batchNorm evaluation", a -> NormOps.batchNorm(a.get(0), a.get(1), a.get(2), runningMean,
                        runningVar, false, 0.1f, 1e-5f),
                in(uniform(234, 24, -1f, 1f), 3, 2, 2, 2), in(uniform(235, 2, 0.5f, 1f), 2),
                in(uniform(236, 2, -0.5f, 0.5f), 2));
    }

    @Test
    public void groupNorm() {
        fx.inputs("groupNorm affine", a -> NormOps.groupNorm(a.get(0), 2, a.get(1), a.get(2), 1e-5f),
                in(uniform(241, 24, -1f, 1f), 2, 4, 3), in(uniform(242, 4, 0.5f, 1f), 4),
                in(uniform(243, 4, -0.5f, 0.5f), 4));
        fx.inputs("groupNorm per channel", a -> NormOps.groupNorm(a.get(0), 4, null, null, 1e-5f),
                in(uniform(244, 32, -1f, 1f), 2, 4, 2, 2));
    }

    /**
     * Support shared by the gradient-check tests of tensor ops, layers, activations, norms and losses: the backends
     * they run on, the inputs they check and the comparison itself.
     *
     * <p>A check runs on the host reference backend ({@link Target#HOST}) or on one GPU API (the test is skipped when
     * the API or its train kernels are unavailable, {@link TrainParity#open}). {@link #inputs} compares the gradients
     * of the inputs of a function ({@link GradientCheckUtil#gradCheck}); {@link #parameters} those of a module's
     * parameters ({@link GradientCheckUtil#gradCheckParameters}, which restores the module's buffers and gradient
     * slots). Both check {@code sum(f * w)} ({@link #weighted}) with the finite-difference step {@link #EPSILON} and
     * the tolerance {@link #TOLERANCE} of {@code GradientCheckUtil}, and assert that the check
     * left nothing recorded in the calling thread's graph. Every tensor and module handed to {@link #keep} is closed
     * by {@link #close}, then the GPU session.</p>
     */
    public static final class GradCheck implements AutoCloseable {

        /** The finite-difference step. */
        public static final float EPSILON = 1e-3f;
        /** The absolute and relative tolerance. */
        public static final float TOLERANCE = 1e-2f;
        /** Elements checked per parameter at most (evenly spaced over larger parameters). */
        public static final int MAX_ELEMENTS_PER_PARAMETER = 48;
        /** The smallest distance of a generated value from a kink: 50 finite-difference steps. */
        public static final float KINK_MARGIN = 0.05f;

        private static final long WEIGHT_SEED = 0x6C1A3L;

        /** A backend a check runs on. */
        public enum Target {
            HOST(null), OPENCL(TrainParity.Api.OPENCL), CUDA(TrainParity.Api.CUDA), VULKAN(TrainParity.Api.VULKAN);

            private final TrainParity.Api api;

            Target(TrainParity.Api api) {
                this.api = api;
            }
        }

        /** A function input: its values in row-major order and its shape. */
        public static final class Input {
            private final float[] values;
            private final int[] dims;

            private Input(float[] values, int[] dims) {
                if (Shape.of(dims).numElements() != values.length) {
                    throw new IllegalArgumentException(values.length + " values for shape " + Arrays.toString(dims));
                }
                this.values = values.clone();
                this.dims = dims.clone();
            }
        }

        private final Target target;
        private final ComputeBackend backend;
        private final TrainParity.Session session;
        private final List<AutoCloseable> owned = new ArrayList<>();

        private GradCheck(Target target, ComputeBackend backend, TrainParity.Session session) {
            this.target = target;
            this.backend = backend;
            this.session = session;
        }

        /** JUnit {@code Parameterized} rows: one per {@link Target}. */
        public static Collection<Object[]> targets() {
            return Arrays.stream(Target.values()).map(t -> new Object[]{t}).collect(Collectors.toList());
        }

        /** Opens {@code target}; a GPU target skips the calling test when it is unavailable. */
        public static GradCheck open(Target target) {
            if (target.api == null) {
                return new GradCheck(target, TrainParity.hostBackend(), null);
            }
            TrainParity.Session session = TrainParity.open(target.api);
            return new GradCheck(target, session.backend(), session);
        }

        public ComputeBackend backend() {
            return backend;
        }

        public static Input in(float[] values, int... dims) {
            return new Input(values, dims);
        }

        // -------------------------------------------------------------- checks

        /** Checks the gradients of {@code fn}'s inputs with the default step and tolerance. */
        public void inputs(String what, GradientCheckUtil.TensorFunction fn, Input... inputs) {
            inputs(what, EPSILON, TOLERANCE, fn, inputs);
        }

        /**
         * Checks the gradients of {@code fn}'s inputs: {@code fn} receives leaf copies of {@code inputs} and returns
         * a new tensor or one of its arguments; it closes its own intermediate results.
         */
        public void inputs(String what, float epsilon, float tolerance, GradientCheckUtil.TensorFunction fn,
                           Input... inputs) {
            int nodes = ComputationGraph.current().size();
            List<Tensor> values = new ArrayList<>(inputs.length);
            try {
                for (Input in : inputs) {
                    values.add(upload(in.values, in.dims));
                }
                GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheck(
                        args -> weighted(fn.apply(args), args, 0), values, epsilon, tolerance);
                assertTrue(target + " " + what + ": " + result, result.passed());
            } finally {
                closeAll(values);
            }
            assertEquals(target + " " + what + ": nodes left recorded", nodes, ComputationGraph.current().size());
        }

        /** Checks the gradients of every parameter of {@code module}. */
        public void parameters(String what, Module module, Supplier<Tensor> forward) {
            parameters(what, module, module.parameters(), forward);
        }

        /**
         * Checks the gradients of {@code params}: {@code forward} evaluates the module on fixed inputs that do not
         * require grad and returns a new tensor each time.
         */
        public void parameters(String what, Module module, List<Parameter> params, Supplier<Tensor> forward) {
            int nodes = ComputationGraph.current().size();
            GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParameters(
                    () -> weighted(forward.get(), Collections.emptyList(), 0), module, params, EPSILON, TOLERANCE,
                    MAX_ELEMENTS_PER_PARAMETER);
            assertTrue(target + " " + what + " parameters: " + result, result.passed());
            assertEquals(target + " " + what + ": nodes left recorded", nodes, ComputationGraph.current().size());
        }

        // -------------------------------------------------------------- composition helpers

        /**
         * {@code y * w}, with {@code w} the {@link #weights} of {@code y}'s element count and {@code salt}; closes
         * {@code y} unless it is one of {@code args}. The weights give every element of {@code y} its own factor in
         * the checked sum.
         */
        public static Tensor weighted(Tensor y, List<Tensor> args, int salt) {
            Objects.requireNonNull(y, "the checked function returned null");
            float[] w = weights(Math.toIntExact(y.numElements()), salt);
            try (Tensor factors = Tensor.fromStorage(y.backend(), y.shape(), DType.FLOAT32,
                    TensorStorage.fromHostFloat(y.backend(), w))) {
                return TensorOps.mul(y, factors);
            } finally {
                closeUnlessArgument(y, args);
            }
        }

        /** The factors {@link #weighted} uses for {@code n} elements and {@code salt}: uniform in [-1, 1). */
        public static float[] weights(int n, int salt) {
            return uniform(WEIGHT_SEED + 31L * salt + n, n, -1f, 1f);
        }

        /** {@code op(t)}; {@code t} is closed afterwards unless it is one of {@code args} or the result itself. */
        public static Tensor then(Tensor t, List<Tensor> args, UnaryOperator<Tensor> op) {
            Tensor result = null;
            try {
                result = op.apply(t);
                return result;
            } finally {
                if (result != t) {
                    closeUnlessArgument(t, args);
                }
            }
        }

        private static void closeUnlessArgument(Tensor t, List<Tensor> args) {
            if (args.stream().noneMatch(a -> a == t)) {
                t.close();
            }
        }

        // -------------------------------------------------------------- generators

        /** {@code n} values uniform in {@code [lo, hi)} from a seeded stream. */
        public static float[] uniform(long seed, int n, float lo, float hi) {
            return TrainParity.uniform(seed, n, lo, hi);
        }

        /**
         * {@code n} values uniform in {@code [lo, hi)}, each redrawn until it is at least {@link #KINK_MARGIN} from
         * every one of {@code kinks}.
         */
        public static float[] awayFrom(long seed, int n, float lo, float hi, float... kinks) {
            Random r = new Random(seed);
            float[] out = new float[n];
            for (int i = 0; i < n; i++) {
                float v;
                do {
                    v = lo + (hi - lo) * r.nextFloat();
                } while (nearKink(v, kinks));
                out[i] = v;
            }
            return out;
        }

        private static boolean nearKink(float v, float[] kinks) {
            for (float k : kinks) {
                if (Math.abs(v - k) < KINK_MARGIN) {
                    return true;
                }
            }
            return false;
        }

        /**
         * The {@code n} grid values {@code -1 + (2i + 1) / n} in a seeded random order: pairwise at least
         * {@code 2 / n} apart, more than twice {@link #EPSILON} for {@code n < 1000}, so no finite-difference
         * perturbation changes an ordering between them.
         */
        public static float[] distinct(long seed, int n) {
            if (n >= 1000) {
                throw new IllegalArgumentException("distinct values need n < 1000, got " + n);
            }
            float[] out = new float[n];
            for (int i = 0; i < n; i++) {
                out[i] = -1f + (2f * i + 1f) / n;
            }
            Random r = new Random(seed);
            for (int i = n - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                float v = out[i];
                out[i] = out[j];
                out[j] = v;
            }
            return out;
        }

        /** {@code n} random multiples of {@code 1 / denominator} in {@code [-1, 1]}. */
        public static float[] grid(long seed, int n, int denominator) {
            Random r = new Random(seed);
            float[] out = new float[n];
            for (int i = 0; i < n; i++) {
                out[i] = (r.nextInt(2 * denominator + 1) - denominator) / (float) denominator;
            }
            return out;
        }

        // -------------------------------------------------------------- tensors

        /**
         * {@code resource}, closed by {@link #close}; a {@link Module} first gets its gradients cleared
         * ({@link Module#zeroGrad}), which input checks accumulate into its parameters, so the calling thread's graph
         * keeps no closed leaf among the leaves it gave gradients to.
         */
        public <T extends AutoCloseable> T keep(T resource) {
            owned.add(resource);
            return resource;
        }

        /** A kept FLOAT32 tensor on the check's backend that does not require grad. */
        public Tensor constant(float[] values, int... dims) {
            return keep(upload(values, dims));
        }

        /** A kept INT32 tensor on the check's backend. */
        public Tensor int32(int[] values, int... dims) {
            Shape shape = Shape.of(dims);
            if (shape.numElements() != values.length) {
                throw new IllegalArgumentException(values.length + " indices for shape " + shape);
            }
            ByteBuffer bytes = ByteBuffer.allocate(values.length * Integer.BYTES).order(ByteOrder.nativeOrder());
            bytes.asIntBuffer().put(values);
            return keep(Tensor.fromStorage(backend, shape, DType.INT32,
                    TensorStorage.fromHost(backend, bytes, DType.INT32)));
        }

        /**
         * Overwrites a dense FLOAT32 tensor at storage offset 0 (a parameter or a buffer) under
         * {@link ComputationGraph#noGrad()}, as every parameter write runs.
         */
        public void write(Tensor t, float[] values) {
            if (t.numElements() != values.length || !t.isContiguous() || t.strides().offset() != 0) {
                throw new IllegalArgumentException("write needs a dense tensor of " + values.length + " elements at"
                        + " offset 0, got " + t);
            }
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                GpuEvent done = t.storage().copyFromHostFloat(values, t.backend().transferQueue());
                if (done != null) {
                    done.close();
                }
            }
        }

        private Tensor upload(float[] values, int[] dims) {
            return Tensor.fromStorage(backend, Shape.of(dims), DType.FLOAT32,
                    TensorStorage.fromHostFloat(backend, values));
        }

        private static void closeAll(List<? extends AutoCloseable> resources) {
            RuntimeException failure = null;
            for (int i = resources.size() - 1; i >= 0; i--) {
                try (AutoCloseable resource = resources.get(i)) {
                    if (resource instanceof Module) {
                        ((Module) resource).zeroGrad();
                    }
                } catch (Exception e) {
                    if (failure == null) {
                        failure = new IllegalStateException("closing a gradient-check resource failed", e);
                    } else {
                        failure.addSuppressed(e);
                    }
                }
            }
            if (failure != null) {
                throw failure;
            }
        }

        @Override
        public void close() {
            try {
                closeAll(owned);
            } finally {
                owned.clear();
                if (session != null) {
                    session.close();
                }
            }
        }
    }
}
