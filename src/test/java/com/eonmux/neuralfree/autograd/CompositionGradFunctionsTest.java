package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SelectionGradFunctions} and {@link ViewGradFunctions} on the host reference backend, where every op takes its
 * CPU path, against hand-computed gradients. Operands are small integers or binary fractions, so every expected value
 * is exact and compared bit for bit.
 */
public class CompositionGradFunctionsTest {

    private final ComputeBackend host = TrainParity.hostBackend();
    private final List<Tensor> owned = new ArrayList<>();
    private final List<GradientFunction> functions = new ArrayList<>();

    @After
    public void tearDown() {
        functions.forEach(GradientFunction::clearSavedTensors);
        for (int i = owned.size() - 1; i >= 0; i--) {
            owned.get(i).close();
        }
        owned.clear();
        functions.clear();
    }

    // ------------------------------------------------------------------ SelectionGradFunctions

    @Test
    public void whereRoutesTheGradientByTheCondition() {
        Tensor gy = floats(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor floatCondition = floats(new float[]{1, 0, 0, 1, 0, 1}, 2, 3);
        Tensor x = floats(new float[]{1, -1, -2, 3, -4, 5}, 2, 3);
        Tensor boolCondition = keep(TensorOps.gt(x, floats(new float[6], 2, 3)));
        assertEquals(DType.BOOL, boolCondition.dtype());
        for (Tensor condition : new Tensor[]{floatCondition, boolCondition}) {
            SelectionGradFunctions.WhereGrad fn = new SelectionGradFunctions.WhereGrad();
            fn.saveForBackward(condition);
            List<Tensor> grads = backward(fn, gy);
            assertEquals(3, grads.size());
            assertNull("the condition gets no gradient", grads.get(0));
            assertEquals(Shape.of(2, 3), grads.get(1).shape());
            assertArrayEquals(new float[]{1, 0, 0, 4, 0, 6}, grads.get(1).toFloatArray(), 0f);
            assertArrayEquals(new float[]{0, 2, 3, 0, 5, 0}, grads.get(2).toFloatArray(), 0f);
        }
        SelectionGradFunctions.WhereGrad onlyB = new SelectionGradFunctions.WhereGrad(false, true);
        onlyB.saveForBackward(floatCondition);
        List<Tensor> partial = backward(onlyB, gy);
        assertNull("an operand that needs no gradient gets none", partial.get(1));
        assertArrayEquals(new float[]{0, 2, 3, 0, 5, 0}, partial.get(2).toFloatArray(), 0f);
        assertTrue(onlyB.supportsCreateGraph());
    }

    @Test
    public void minRoutesTiesToTheFirstIndex() {
        Tensor x = floats(new float[]{3, 1, 1, 2, 2, 5}, 2, 3);
        Tensor rowMin = keep(TensorOps.min(x, 1));
        assertArrayEquals(new float[]{1, 2}, rowMin.toFloatArray(), 0f);
        SelectionGradFunctions.MinGrad rows = new SelectionGradFunctions.MinGrad(1, x.shape());
        rows.saveForBackward(x, rowMin);
        assertArrayEquals(new float[]{0, 10, 0, 20, 0, 0},
                single(rows, floats(new float[]{10, 20}, 2)), 0f);

        Tensor colMin = keep(TensorOps.min(x, 0));
        SelectionGradFunctions.MinGrad cols = new SelectionGradFunctions.MinGrad(0, x.shape());
        cols.saveForBackward(x, colMin);
        assertArrayEquals(new float[]{0, 2, 3, 1, 0, 0},
                single(cols, floats(new float[]{1, 2, 3}, 3)), 0f);
        assertTrue("maxBackward records its second-order function", cols.supportsCreateGraph());
        assertThrows(IllegalArgumentException.class,
                () -> new SelectionGradFunctions.MinGrad(0, Shape.of(3, 2)).saveForBackward(x, colMin));
    }

    @Test
    public void varGradientIsTwiceTheCenteredInputOverN() {
        float[] xs = {1, 2, 6, 4, 4, 7};
        float[] means = {3, 5};
        float[] g = {3, -1.5f};
        Tensor x = floats(xs, 2, 3);
        SelectionGradFunctions.VarGrad rows = new SelectionGradFunctions.VarGrad(x.shape(), new int[]{-1});
        rows.saveForBackward(x);
        float[] expected = new float[6];
        for (int i = 0; i < 6; i++) {
            expected[i] = ((xs[i] - means[i / 3]) * (2.0f / 3)) * g[i / 3];
        }
        Tensor gx = backward(rows, floats(g, 2)).get(0);
        assertEquals(x.shape(), gx.shape());
        assertArrayEquals(expected, gx.toFloatArray(), 0f);
        assertTrue(rows.supportsCreateGraph());

        Tensor square = floats(new float[]{1, 2, 3, 6}, 2, 2);
        SelectionGradFunctions.VarGrad all = new SelectionGradFunctions.VarGrad(square.shape(), null);
        all.saveForBackward(square);
        assertArrayEquals(new float[]{-2, -1, 0, 3}, single(all, keep(Tensor.scalar(host, 2.0f))), 0f);
        assertThrows(IllegalArgumentException.class,
                () -> new SelectionGradFunctions.VarGrad(x.shape(), new int[]{1, -1}));
    }

    @Test
    public void sumToShapeGradientExpandsOrRearranges() {
        Shape source = Shape.of(2, 3);
        SelectionGradFunctions.SumToShapeGrad rows = new SelectionGradFunctions.SumToShapeGrad(source, Shape.of(3));
        Tensor fromRow = backward(rows, floats(new float[]{1, 2, 3}, 3)).get(0);
        assertEquals(source, fromRow.shape());
        assertTrue("the broadcast gradient is dense", fromRow.isContiguous());
        assertArrayEquals(new float[]{1, 2, 3, 1, 2, 3}, fromRow.toFloatArray(), 0f);

        SelectionGradFunctions.SumToShapeGrad cols = new SelectionGradFunctions.SumToShapeGrad(source,
                Shape.of(2, 1));
        assertArrayEquals(new float[]{5, 5, 5, 7, 7, 7}, single(cols, floats(new float[]{5, 7}, 2, 1)), 0f);

        Tensor row = floats(new float[]{1, 2, 3}, 1, 3);
        Tensor column = keep(TensorOps.sumToShape(row, Shape.of(3, 1)));
        assertArrayEquals("the forward copies when the target only rearranges", new float[]{1, 2, 3},
                column.toFloatArray(), 0f);
        SelectionGradFunctions.SumToShapeGrad rearranged = new SelectionGradFunctions.SumToShapeGrad(row.shape(),
                Shape.of(3, 1));
        Tensor back = backward(rearranged, floats(new float[]{4, 5, 6}, 3, 1)).get(0);
        assertEquals(row.shape(), back.shape());
        assertArrayEquals(new float[]{4, 5, 6}, back.toFloatArray(), 0f);

        SelectionGradFunctions.SumToShapeGrad droppedOne = new SelectionGradFunctions.SumToShapeGrad(Shape.of(1, 3),
                Shape.of(3));
        Tensor restored = backward(droppedOne, floats(new float[]{7, 8, 9}, 3)).get(0);
        assertEquals(Shape.of(1, 3), restored.shape());
        assertArrayEquals(new float[]{7, 8, 9}, restored.toFloatArray(), 0f);

        SelectionGradFunctions.SumToShapeGrad lifted = new SelectionGradFunctions.SumToShapeGrad(Shape.of(3),
                Shape.of(1, 3));
        Tensor flat = backward(lifted, floats(new float[]{4, 5, 6}, 1, 3)).get(0);
        assertEquals(Shape.of(3), flat.shape());
        assertArrayEquals(new float[]{4, 5, 6}, flat.toFloatArray(), 0f);

        assertTrue(rows.supportsCreateGraph());
        assertThrows(IllegalArgumentException.class,
                () -> new SelectionGradFunctions.SumToShapeGrad(source, Shape.of(4)));
    }

    @Test
    public void scatterAddPassesTheGradientToTheInputAndGathersItForTheSource() {
        Tensor gy = floats(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor index = keep(keep(Tensor.fromArray(host, new int[]{0, 2, 1, 1})).reshape(Shape.of(2, 2)));
        Tensor src = floats(new float[4], 2, 2);
        SelectionGradFunctions.ScatterAddGrad fn = new SelectionGradFunctions.ScatterAddGrad(1);
        fn.saveForBackward(index, src);
        List<Tensor> grads = backward(fn, gy);
        assertEquals(3, grads.size());
        assertSame("the input gradient is gy itself", gy, grads.get(0));
        assertNull("the index gets no gradient", grads.get(1));
        assertEquals(Shape.of(2, 2), grads.get(2).shape());
        assertArrayEquals(new float[]{1, 3, 5, 5}, grads.get(2).toFloatArray(), 0f);

        SelectionGradFunctions.ScatterAddGrad srcOnly = new SelectionGradFunctions.ScatterAddGrad(1, false, true);
        srcOnly.saveForBackward(index, src);
        assertNull(backward(srcOnly, gy).get(0));
        assertTrue(fn.supportsCreateGraph());
        assertThrows(IllegalArgumentException.class,
                () -> new SelectionGradFunctions.ScatterAddGrad(1).saveForBackward(index, floats(new float[2], 2)));
    }

    @Test
    public void releasedSavedTensorsMakeBackwardThrow() {
        SelectionGradFunctions.WhereGrad fn = new SelectionGradFunctions.WhereGrad();
        fn.saveForBackward(floats(new float[]{1, 0}, 2));
        fn.clearSavedTensors();
        assertTrue(fn.savedTensors().isEmpty());
        Tensor gy = floats(new float[]{1, 2}, 2);
        assertThrows(IllegalStateException.class, () -> fn.backward(gy));
    }

    // ------------------------------------------------------------------ ViewGradFunctions

    @Test
    public void expandGradientSumsTheBroadcastDims() {
        Tensor gy = floats(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        ViewGradFunctions.ExpandGrad fromRow = new ViewGradFunctions.ExpandGrad(Shape.of(3));
        Tensor rowGrad = backward(fromRow, gy).get(0);
        assertEquals(Shape.of(3), rowGrad.shape());
        assertArrayEquals(new float[]{5, 7, 9}, rowGrad.toFloatArray(), 0f);

        ViewGradFunctions.ExpandGrad fromCol = new ViewGradFunctions.ExpandGrad(Shape.of(2, 1));
        Tensor colGrad = backward(fromCol, gy).get(0);
        assertEquals(Shape.of(2, 1), colGrad.shape());
        assertArrayEquals(new float[]{6, 15}, colGrad.toFloatArray(), 0f);

        ViewGradFunctions.ExpandGrad unchanged = new ViewGradFunctions.ExpandGrad(Shape.of(2, 3));
        assertSame(gy, backward(unchanged, gy).get(0));

        ViewGradFunctions.ExpandGrad toEmpty = new ViewGradFunctions.ExpandGrad(Shape.of(1, 3));
        Tensor emptyGrad = backward(toEmpty, keep(Tensor.zeros(host, Shape.of(0, 3)))).get(0);
        assertEquals("a size-1 dim expanded to 0 gets a zero gradient", Shape.of(1, 3), emptyGrad.shape());
        assertArrayEquals(new float[3], emptyGrad.toFloatArray(), 0f);
        assertTrue(fromRow.supportsCreateGraph());
    }

    @Test
    public void castGradientsAreFloat32() {
        float[] values = {0.5f, -2, 3, 1.25f};
        Tensor gy = floats(values, 2, 2);
        ViewGradFunctions.CastGrad widen = new ViewGradFunctions.CastGrad(DType.FLOAT16, DType.FLOAT32);
        assertSame("a FLOAT32 gradient passes through", gy, backward(widen, gy).get(0));

        Tensor half = keep(gy.to(DType.FLOAT16));
        assertEquals(DType.FLOAT16, half.dtype());
        ViewGradFunctions.CastGrad narrow = new ViewGradFunctions.CastGrad(DType.FLOAT32, DType.FLOAT16);
        Tensor fromHalf = backward(narrow, half).get(0);
        assertEquals(DType.FLOAT32, fromHalf.dtype());
        assertEquals(Shape.of(2, 2), fromHalf.shape());
        assertTrue(fromHalf.isContiguous());
        assertArrayEquals(values, fromHalf.toFloatArray(), 0f);

        Tensor bf16 = keep(gy.to(DType.BFLOAT16));
        ViewGradFunctions.CastGrad toBf16 = new ViewGradFunctions.CastGrad(DType.FLOAT32, DType.BFLOAT16);
        Tensor fromBf16 = backward(toBf16, bf16).get(0);
        assertEquals(DType.FLOAT32, fromBf16.dtype());
        assertArrayEquals(values, fromBf16.toFloatArray(), 0f);

        assertTrue(narrow.supportsCreateGraph());
        assertThrows(IllegalArgumentException.class,
                () -> new ViewGradFunctions.CastGrad(DType.FLOAT32, DType.INT32));
        assertThrows(IllegalArgumentException.class,
                () -> new ViewGradFunctions.CastGrad(DType.FLOAT16, DType.FLOAT16));
    }

    @Test
    public void identityGradientIsGradOutputItself() {
        Tensor gy = floats(new float[]{1, 2, 3}, 3);
        ViewGradFunctions.IdentityGrad fn = new ViewGradFunctions.IdentityGrad();
        List<Tensor> grads = fn.backward(gy);
        assertEquals(1, grads.size());
        assertSame(gy, grads.get(0));
        assertTrue(fn.savedTensors().isEmpty());
        assertTrue(fn.supportsCreateGraph());
    }

    // ------------------------------------------------------------------ fixture

    private Tensor keep(Tensor t) {
        assertNotNull(t);
        owned.add(t);
        return t;
    }

    private Tensor floats(float[] data, int... dims) {
        return keep(keep(Tensor.fromArray(host, data)).reshape(Shape.of(dims)));
    }

    /**
     * {@code fn.backward(gy)}, keeping the function for release and every returned gradient that is not {@code gy}
     * itself for closing.
     */
    private List<Tensor> backward(GradientFunction fn, Tensor gy) {
        functions.add(fn);
        List<Tensor> grads = fn.backward(gy);
        for (Tensor g : grads) {
            if (g != null && g != gy) {
                owned.add(Objects.requireNonNull(g));
            }
        }
        return grads;
    }

    private float[] single(GradientFunction fn, Tensor gy) {
        Tensor grad = backward(fn, gy).get(0);
        assertNotNull(grad);
        return grad.toFloatArray();
    }
}
