package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.GpuReduceOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.ElementwiseOp;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The recording rule of {@link TensorOps}: every differentiable public operation records exactly
 * one node with its gradient function and inputs when grad mode is on and a differentiable operand requires grad,
 * nothing under {@link ComputationGraph#noGrad()}, and never for the non-differentiable operations, the backward
 * primitives or a constant operand (index, condition, mask, CSR). Also the backward of the recorded index, segment,
 * dropout and split operations through {@link Tensor#backward()}. Runs on {@link TrainParity#hostBackend()} (the CPU
 * reference path).
 */
public class TensorOpsRecordingTest {

    private ComputeBackend host;
    private ComputationGraph graph;
    private final List<Tensor> owned = new ArrayList<>();

    private Tensor x;
    private Tensor c;
    private Tensor positive;

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
        host = TrainParity.hostBackend();
        graph = ComputationGraph.current();
        x = leaf(new float[]{0.5f, 1f, 1.5f, 2f, 2.5f, 3f, 3.5f, 4f}, 2, 4);
        c = constant(new float[]{1f, -2f, 3f, -4f, 5f, -6f, 7f, -8f}, 2, 4);
        positive = constant(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 2, 4);
    }

    @After
    public void tearDown() {
        try {
            graph.zeroGrad();
            graph.clear();
            for (int i = owned.size() - 1; i >= 0; i--) {
                owned.get(i).close();
            }
        } finally {
            ComputationGraph.setGradEnabled(true);
            host.close();
        }
    }

    // ------------------------------------------------------------------ recording truth table

    @Test
    public void arithmeticAndActivationsRecordOneNode() {
        records("add", GradFunctions.AddGrad.class, list(x, c), () -> TensorOps.add(x, c));
        records("add reversed", GradFunctions.AddGrad.class, list(c, x), () -> TensorOps.add(c, x));
        records("sub", GradFunctions.SubGrad.class, list(x, c), () -> TensorOps.sub(x, c));
        records("mul", GradFunctions.MulGrad.class, list(x, c), () -> TensorOps.mul(x, c));
        records("div", GradFunctions.DivGrad.class, list(x, positive), () -> TensorOps.div(x, positive));
        records("pow", GradFunctions.PowGrad.class, list(x), () -> TensorOps.pow(x, 2f));
        records("neg", GradFunctions.NegGrad.class, list(x), () -> TensorOps.neg(x));
        records("abs", GradFunctions.AbsGrad.class, list(x), () -> TensorOps.abs(x));
        records("sqrt", GradFunctions.SqrtGrad.class, list(x), () -> TensorOps.sqrt(x));
        records("exp", GradFunctions.ExpGrad.class, list(x), () -> TensorOps.exp(x));
        records("log", GradFunctions.LogGrad.class, list(x), () -> TensorOps.log(x));
        records("clamp", GradFunctions.ClampGrad.class, list(x), () -> TensorOps.clamp(x, 1f, 3f));
        records("add scalar", GradFunctions.AddScalarGrad.class, list(x), () -> TensorOps.add(x, 2f));
        records("mul scalar", GradFunctions.MulScalarGrad.class, list(x), () -> TensorOps.mul(x, 2f));
        records("relu", GradFunctions.ReluGrad.class, list(x), () -> TensorOps.relu(x));
        records("gelu", GradFunctions.GeluGrad.class, list(x), () -> TensorOps.gelu(x));
        records("silu", GradFunctions.SiluGrad.class, list(x), () -> TensorOps.silu(x));
        records("sigmoid", GradFunctions.SigmoidGrad.class, list(x), () -> TensorOps.sigmoid(x));
        records("tanh", GradFunctions.TanhGrad.class, list(x), () -> TensorOps.tanh(x));
        records("leakyRelu", GradFunctions.LeakyReluGrad.class, list(x), () -> TensorOps.leakyRelu(x, 0.1f));
        records("elu", GradFunctions.EluGrad.class, list(x), () -> TensorOps.elu(x, 1f));
        records("softplus", GradFunctions.SoftplusGrad.class, list(x), () -> TensorOps.softplus(x));
        records("mish", GradFunctions.MishGrad.class, list(x), () -> TensorOps.mish(x));
        records("glu", GradFunctions.GluGrad.class, list(x), () -> TensorOps.glu(x, 1));
        records("geglu", GradFunctions.GegluGrad.class, list(x), () -> TensorOps.geglu(x, -1));
        Tensor mask = constant(new float[]{0f, 2f, 2f, 0f, 2f, 2f, 0f, 2f}, 2, 4);
        records("maskScale", IndexGradFunctions.MaskScaleGrad.class, list(x, mask),
                () -> TensorOps.maskScale(x, mask, 1f));
    }

    @Test
    public void softmaxProductsAndReductionsRecordOneNode() {
        records("softmax", GradFunctions.SoftmaxGrad.class, list(x), () -> TensorOps.softmax(x, 1));
        records("logSoftmax", GradFunctions.LogSoftmaxGrad.class, list(x), () -> TensorOps.logSoftmax(x, 1));
        records("maskedSoftmax", GradFunctions.SoftmaxGrad.class, list(x), () -> TensorOps.maskedSoftmax(x, 1));
        Tensor right = constant(new float[12], 4, 3);
        records("matmul", GradFunctions.MatmulGrad.class, list(x, right), () -> TensorOps.matmul(x, right));
        Tensor batched = leaf(new float[8], 1, 2, 4);
        Tensor batchedRight = constant(new float[12], 1, 4, 3);
        records("bmm", GradFunctions.BmmGrad.class, list(batched, batchedRight),
                () -> TensorOps.bmm(batched, batchedRight));
        records("sum", GradFunctions.SumGrad.class, list(x), () -> TensorOps.sum(x, 1));
        records("sum all", GradFunctions.SumGrad.class, list(x), () -> TensorOps.sum(x));
        records("mean", GradFunctions.MeanGrad.class, list(x), () -> TensorOps.mean(x, 0));
        records("max", GradFunctions.MaxGrad.class, list(x), () -> TensorOps.max(x, 1));
        records("min", SelectionGradFunctions.MinGrad.class, list(x), () -> TensorOps.min(x, 1));
        records("var", SelectionGradFunctions.VarGrad.class, list(x), () -> TensorOps.var(x, 1));
        records("sumToShape", SelectionGradFunctions.SumToShapeGrad.class, list(x),
                () -> TensorOps.sumToShape(x, Shape.of(4)));
    }

    @Test
    public void selectionIndexAndShapeOpsRecordOneNode() {
        Tensor condition = constant(new float[]{1f, 0f, 0f, 1f, 1f, 1f, 0f, 0f}, 2, 4);
        records("where a", SelectionGradFunctions.WhereGrad.class, list(condition, x, c),
                () -> TensorOps.where(condition, x, c));
        records("where b", SelectionGradFunctions.WhereGrad.class, list(condition, c, x),
                () -> TensorOps.where(condition, c, x));
        Tensor index = ints(new int[]{0, 3, 1, 2}, 2, 2);
        Tensor values = constant(new float[]{9f, 8f, 7f, 6f}, 2, 2);
        Tensor valuesLeaf = leaf(new float[]{9f, 8f, 7f, 6f}, 2, 2);
        records("gather", GradFunctions.GatherGrad.class, list(x, index), () -> TensorOps.gather(x, 1, index));
        records("scatter", GradFunctions.ScatterGrad.class, list(x, index, values),
                () -> TensorOps.scatter(x, 1, index, values));
        records("scatterAdd by source", SelectionGradFunctions.ScatterAddGrad.class, list(c, index, valuesLeaf),
                () -> TensorOps.scatterAdd(c, 1, index, valuesLeaf));
        Tensor rows = ints(new int[]{1, 0, 1}, 3);
        records("indexSelectRows", IndexGradFunctions.IndexSelectRowsGrad.class, list(x, rows),
                () -> TensorOps.indexSelectRows(x, rows, false));
        records("indexSelectRows padded", IndexGradFunctions.IndexSelectRowsGrad.class, list(x, rows),
                () -> TensorOps.indexSelectRows(x, rows, true, 0));
        Tensor ptr = ints(new int[]{0, 1, 2}, 3);
        Tensor edges = ints(new int[]{1, 0}, 2);
        for (GpuReduceOps.SegmentOp op : GpuReduceOps.SegmentOp.values()) {
            records("segmentReduce " + op, IndexGradFunctions.SegmentReduceGrad.class, list(x, ptr, edges),
                    () -> TensorOps.segmentReduce(x, ptr, edges, 2, op, 0f));
        }
        records("cat", GradFunctions.CatGrad.class, list(c, x), () -> TensorOps.cat(list(c, x), 0));
        records("stack", GradFunctions.StackGrad.class, list(x, c), () -> TensorOps.stack(list(x, c), -1));
    }

    @Test
    public void splitRecordsOneSliceNodePerPart() {
        int before = graph.size();
        try (ComputationGraph.StepScope step = graph.step()) {
            List<Tensor> parts = TensorOps.split(x, 2, 1);
            assertEquals(2, parts.size());
            for (Tensor part : parts) {
                assertTrue(part.requiresGrad());
                assertTrue(part.gradFn() instanceof GradFunctions.SliceGrad);
                assertSame(x, graph.getNode(part).inputs().get(0));
            }
            assertEquals("one node per part", before + 2, graph.size());
        }
        assertEquals(before, graph.size());
        try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
            for (Tensor part : TensorOps.split(x, 4, 1)) {
                keep(part);
                assertFalse(part.requiresGrad());
            }
        }
        assertEquals(before, graph.size());
    }

    @Test
    public void constantOperandsDoNotRecord() {
        Tensor conditionLeaf = leaf(new float[]{1f, 0f, 0f, 1f, 1f, 1f, 0f, 0f}, 2, 4);
        neverRecords("where on a requires-grad condition", () -> TensorOps.where(conditionLeaf, c, positive));
        Tensor maskLeaf = leaf(new float[]{0f, 2f, 2f, 0f, 2f, 2f, 0f, 2f}, 2, 4);
        neverRecords("maskScale with a requires-grad mask", () -> TensorOps.maskScale(c, maskLeaf, 1f));
        neverRecords("constants only", () -> TensorOps.add(c, positive));
        try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
            neverRecords("noGrad", () -> TensorOps.add(x, c));
        }
    }

    @Test
    public void nonDifferentiableOpsAndBackwardPrimitivesNeverRecord() {
        neverRecords("round", () -> TensorOps.round(x));
        neverRecords("floor", () -> TensorOps.floor(x));
        neverRecords("argmax", () -> TensorOps.argmax(x, 1));
        neverRecords("argmin", () -> TensorOps.argmin(x, 1));
        neverRecords("eq", () -> TensorOps.eq(x, c));
        neverRecords("gt", () -> TensorOps.gt(x, c));
        neverRecords("lt", () -> TensorOps.lt(x, c));
        neverRecords("activationBackward", () -> TensorOps.activationBackward(ElementwiseOp.Backward.RELU, x, null,
                x, 0f, 0f));
        neverRecords("softmaxBackward", () -> TensorOps.softmaxBackward(x, x, 1));
        neverRecords("logSoftmaxBackward", () -> TensorOps.logSoftmaxBackward(x, x, 1));
        Tensor rowGrad = leaf(new float[]{1f, 2f}, 2);
        Tensor rowMax = constant(new float[]{2f, 4f}, 2);
        neverRecords("maxBackward", () -> TensorOps.maxBackward(x, rowMax, rowGrad, 1));
        Tensor rows = ints(new int[]{1, 0}, 2);
        neverRecords("indexAddRows", () -> TensorOps.indexAddRows(x, rows, 3, false, -1));
        Tensor ptr = ints(new int[]{0, 1, 2}, 3);
        Tensor edges = ints(new int[]{1, 0}, 2);
        neverRecords("segmentReduceBackward", () -> TensorOps.segmentReduceBackward(c, ptr, edges, null, x,
                GpuReduceOps.SegmentOp.SUM));
    }

    @Test
    public void inPlaceOpsRefuseALeafInGradMode() {
        assertThrows(IllegalStateException.class, () -> TensorOps.addInPlace(x, c, 1f));
        assertThrows(IllegalStateException.class, () -> TensorOps.mulInPlace(x, 2f));
        assertThrows(IllegalStateException.class, () -> TensorOps.zeroInPlace(x));
        TensorOps.addInPlace(c, positive, 1f);
        assertArrayEquals(new float[]{2f, 0f, 6f, 0f, 10f, 0f, 14f, 0f}, c.toFloatArray(), 0f);
    }

    // ------------------------------------------------------------------ recorded backward (CPU path)

    @Test
    public void indexSelectRowsGradientSkipsThePaddingRow() {
        Tensor table = leaf(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 3, 2);
        Tensor idx = ints(new int[]{2, 0, 2, 1}, 2, 2);
        Tensor weights = constant(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 2, 2, 2);
        backward(() -> TensorOps.sum(TensorOps.mul(TensorOps.indexSelectRows(table, idx, false, 1), weights)));
        assertArrayEquals("row 0 from selection 1; row 2 from selections 0 and 2; padding row 1 nothing",
                new float[]{3f, 4f, 0f, 0f, 6f, 8f}, table.grad().toFloatArray(), 0f);
    }

    @Test
    public void segmentReduceGradientRoutesPerOp() {
        Tensor src = leaf(new float[]{1f, 5f, 3f, 2f, 3f, 0f}, 3, 2);
        Tensor ptr = ints(new int[]{0, 2, 2, 3}, 4);
        Tensor edges = ints(new int[]{0, 2, 1}, 3);
        Tensor up = constant(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 3, 2);
        backward(() -> TensorOps.sum(TensorOps.mul(TensorOps.segmentReduce(src, ptr, edges, 3,
                GpuReduceOps.SegmentOp.MAX, -1f), up)));
        assertArrayEquals("MAX: segment 0 routes column 0 to row 2 and column 1 to row 0; segment 1 is empty",
                new float[]{0f, 2f, 5f, 6f, 1f, 0f}, src.grad().toFloatArray(), 0f);
        graph.zeroGrad();
        backward(() -> TensorOps.sum(TensorOps.mul(TensorOps.segmentReduce(src, ptr, edges, 3,
                GpuReduceOps.SegmentOp.MEAN, 0f), up)));
        assertArrayEquals("MEAN: segment 0 divides by 2", new float[]{0.5f, 1f, 5f, 6f, 0.5f, 1f},
                src.grad().toFloatArray(), 0f);
    }

    @Test
    public void maskScaleAndWhereGradients() {
        Tensor mask = constant(new float[]{0f, 2f, 2f, 0f, 2f, 2f, 0f, 2f}, 2, 4);
        backward(() -> TensorOps.sum(TensorOps.maskScale(x, mask, 0.5f)));
        assertArrayEquals(new float[]{0f, 1f, 1f, 0f, 1f, 1f, 0f, 1f}, x.grad().toFloatArray(), 0f);
        graph.zeroGrad();
        Tensor condition = constant(new float[]{1f, 0f, 0f, 1f, 1f, 1f, 0f, 0f}, 2, 4);
        Tensor other = leaf(new float[8], 2, 4);
        backward(() -> TensorOps.sum(TensorOps.where(condition, x, other)));
        assertArrayEquals(new float[]{1f, 0f, 0f, 1f, 1f, 1f, 0f, 0f}, x.grad().toFloatArray(), 0f);
        assertArrayEquals(new float[]{0f, 1f, 1f, 0f, 0f, 0f, 1f, 1f}, other.grad().toFloatArray(), 0f);
    }

    @Test
    public void splitBackPropagatesIncludingZeroSizeParts() {
        Tensor weights = constant(new float[]{1f, 2f, 3f, 4f}, 2, 2);
        int before = graph.size();
        try (ComputationGraph.StepScope step = graph.step()) {
            List<Tensor> parts = TensorOps.split(x, 2, 1);
            try (Tensor loss = TensorOps.sum(TensorOps.mul(parts.get(1), weights))) {
                loss.backward();
            }
        }
        assertEquals("the step closes the unused part and its node", before, graph.size());
        assertArrayEquals(new float[]{0f, 0f, 1f, 2f, 0f, 0f, 3f, 4f}, x.grad().toFloatArray(), 0f);

        Tensor empty = keep(keep(Tensor.zeros(host, Shape.of(2, 0))).withRequiresGrad(true));
        try (ComputationGraph.StepScope step = graph.step()) {
            List<Tensor> parts = TensorOps.split(empty, 3, 1);
            assertEquals("zero-size parts are recorded", before + 3, graph.size());
            for (Tensor part : parts) {
                assertEquals(Shape.of(2, 0), part.shape());
                assertTrue(part.requiresGrad());
                assertTrue(part.gradFn() instanceof GradFunctions.SliceGrad);
            }
            try (Tensor loss = TensorOps.sum(TensorOps.cat(list(parts.get(1), weights), 1))) {
                loss.backward();
            }
        }
        assertEquals(before, graph.size());
        assertNotNull("the empty input receives its (empty) gradient", empty.grad());
        assertEquals(Shape.of(2, 0), empty.grad().shape());
    }

    // ------------------------------------------------------------------ helpers

    /**
     * {@code call} records exactly one node with a {@code fn} gradient function over {@code inputs} (by identity)
     * in grad mode, and nothing under noGrad, where its result does not require grad.
     */
    private void records(String op, Class<?> fn, List<Tensor> inputs, Supplier<Tensor> call) {
        int before = graph.size();
        try (ComputationGraph.StepScope step = graph.step()) {
            Tensor out = call.get();
            assertTrue(op + ": the result requires grad", out.requiresGrad());
            assertTrue(op + ": gradient function " + out.gradFn(), fn.isInstance(out.gradFn()));
            assertEquals(op + ": one node per call", before + 1, graph.size());
            ComputationGraph.GraphNode node = graph.getNode(out);
            assertNotNull(op + ": the result is the node's output", node);
            assertEquals(op + ": inputs", inputs.size(), node.inputs().size());
            for (int i = 0; i < inputs.size(); i++) {
                assertSame(op + ": input " + i, inputs.get(i), node.inputs().get(i));
            }
        }
        assertEquals(op + ": the step leaves no node", before, graph.size());
        try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
            Tensor out = keep(call.get());
            assertFalse(op + ": no grad under noGrad", out.requiresGrad());
            assertNull(op + ": no node under noGrad", out.gradFn());
        }
        assertEquals(op + ": nothing recorded under noGrad", before, graph.size());
    }

    /** {@code call} records nothing and returns a result that does not require grad. */
    private void neverRecords(String op, Supplier<Tensor> call) {
        int before = graph.size();
        Tensor out = keep(call.get());
        assertFalse(op + ": the result does not require grad", out.requiresGrad());
        assertNull(op + ": no gradient function", out.gradFn());
        assertEquals(op + ": nothing recorded", before, graph.size());
    }

    /** Back-propagates the scalar {@code loss} and checks that the backward released every node it recorded. */
    private void backward(Supplier<Tensor> loss) {
        int before = graph.size();
        try (Tensor l = loss.get()) {
            l.backward();
        }
        assertEquals("backward releases the step's nodes", before, graph.size());
    }

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }

    private Tensor constant(float[] data, int... dims) {
        return keep(keep(Tensor.fromArray(host, data)).reshape(Shape.of(dims)));
    }

    private Tensor leaf(float[] data, int... dims) {
        return keep(constant(data, dims).withRequiresGrad(true));
    }

    private Tensor ints(int[] data, int... dims) {
        return keep(keep(Tensor.fromArray(host, data)).reshape(Shape.of(dims)));
    }

    private static List<Tensor> list(Tensor... tensors) {
        return Collections.unmodifiableList(Arrays.asList(tensors));
    }
}
