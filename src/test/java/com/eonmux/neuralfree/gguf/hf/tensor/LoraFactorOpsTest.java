package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.function.UnaryOperator;

import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link LoraFactorOps}: every tensor transform a LoRA factor pair supports gives the same delta as the transform
 * applied to the dense {@code B @ A}, acting on the factor the port of {@code LoraTorchTensor} names; the others throw
 * the low-rank error.
 */
public class LoraFactorOpsTest {

    /** A {@code [2, 3]}, B {@code [4, 2]}: the delta is {@code [4, 3]}. */
    private static final RowSource A = RopeTensors.fromFloats(new long[]{2, 3}, new float[]{1, 0, 2, 0, 1, -1});
    private static final RowSource B = RopeTensors.fromFloats(new long[]{4, 2}, new float[]{1, 2, 3, 4, -1, 0, 0, 2});
    private static final RowSource A2 = RopeTensors.fromFloats(new long[]{2, 3}, new float[]{2, 1, 0, 1, 1, 1});
    private static final RowSource B2 = RopeTensors.fromFloats(new long[]{4, 2}, new float[]{0, 1, 1, 0, 2, 2, 1, -1});

    private static LoraFactorSource lora(RowSource a, RowSource b) {
        return new LoraFactorSource("lora.q", a, b);
    }

    private static RowSource dense(RowSource s) throws IOException {
        return RopeTensors.fromFloats(s.shape(), values(s));
    }

    /** Applies {@code op} to the pair and to its dense delta and compares the results. */
    private static LoraFactorSource check(RowSource pair, UnaryOperator<RowSource> op) throws IOException {
        RowSource got = op.apply(pair);
        assertTrue(got.getClass().getSimpleName(), got instanceof LoraFactorSource);
        RowSource want = op.apply(dense(pair));
        assertArrayEquals(want.shape(), got.shape());
        assertArrayEquals(values(want), values(got), 0f);
        return (LoraFactorSource) got;
    }

    @Test
    public void theDeltaIsBTimesA() throws IOException {
        LoraFactorSource l = lora(A, B);
        assertArrayEquals(new long[]{4, 3}, l.shape());
        assertEquals(2, l.rank());
        assertArrayEquals(new float[]{1, 2, 0, 3, 4, 2, -1, 0, -2, 0, 2, -2}, values(l), 0f);
        assertThrows(IllegalArgumentException.class, () -> new LoraFactorSource("x", B, A));
    }

    @Test
    public void rowOpsActOnB() throws IOException {
        LoraFactorSource l = lora(A, B);
        assertSame(A, check(l, s -> RowTransforms.permuteQk(s, 1, 1)).a());
        assertSame(A, check(l, s -> RowTransforms.sliceRows(s, 1, 2)).a());
        check(l, s -> RowTransforms.interleaveRows(s, 2, 1));
        check(l, s -> RowTransforms.normalToNeoxPartial(s, 1, 4, 4));
        check(l, s -> RowTransforms.negateSecondHalfRows(s, 2));
        check(l, s -> RowTransforms.padRows(s, 6, 0f));
        check(l, s -> ElementTransforms.mulRowsBy(s, new float[]{1, -1, 2, 0.5f}));
        LoraFactorSource scaled = check(l, s -> ElementTransforms.mulScalar(s, 3f));
        assertSame(A, scaled.a());
    }

    @Test
    public void columnOpsActOnA() throws IOException {
        LoraFactorSource l = lora(A, B);
        assertSame(B, check(l, s -> ColumnTransforms.sliceCols(s, 1, 2)).b());
        check(l, s -> ColumnTransforms.interleaveCols(s, 2, 0));
        check(l, s -> ColumnTransforms.permuteColGroups(s, 1, new int[]{2, 0, 1}));
        check(l, s -> ElementTransforms.gatherInner(s, new int[]{1, 1, 0}));
        check(l, s -> ElementTransforms.mulColsBy(s, new float[]{2, -1, 0.5f}));
    }

    @Test
    public void concatenationsKeepTheCommonFactorOrGoBlockDiagonal() throws IOException {
        LoraFactorSource sameA = (LoraFactorSource) RowTransforms.concatRows(Arrays.asList(lora(A, B), lora(A, B2)));
        assertSame(A, sameA.a());
        assertEquals(2, sameA.rank());
        RowSource rowsDense = RowTransforms.concatRows(Arrays.asList(dense(lora(A, B)), dense(lora(A, B2))));
        assertArrayEquals(values(rowsDense), values(sameA), 0f);

        LoraFactorSource diag = (LoraFactorSource) RowTransforms.concatRows(Arrays.asList(lora(A, B), lora(A2, B2)));
        assertEquals(4, diag.rank());
        assertArrayEquals(values(RowTransforms.concatRows(Arrays.asList(dense(lora(A, B)), dense(lora(A2, B2))))),
                values(diag), 0f);

        LoraFactorSource sameB = (LoraFactorSource) ColumnTransforms.concatCols(Arrays.asList(lora(A, B), lora(A2, B)));
        assertSame(B, sameB.b());
        assertArrayEquals(values(ColumnTransforms.concatCols(Arrays.asList(dense(lora(A, B)), dense(lora(A2, B))))),
                values(sameB), 0f);
        LoraFactorSource colDiag = (LoraFactorSource) ColumnTransforms.concatCols(
                Arrays.asList(lora(A, B), lora(A2, B2)));
        assertEquals(4, colDiag.rank());
        assertArrayEquals(values(ColumnTransforms.concatCols(Arrays.asList(dense(lora(A, B)), dense(lora(A2, B2))))),
                values(colDiag), 0f);
    }

    @Test
    public void stackStacksBothFactors() throws IOException {
        LoraFactorSource shared = (LoraFactorSource) ExpertTransforms.stack(Arrays.asList(lora(A, B), lora(A, B2)));
        assertArrayEquals(new long[]{1, 2, 3}, shared.a().shape());
        assertArrayEquals(new long[]{2, 4, 3}, shared.shape());
        RowSource stackedDense = ExpertTransforms.stack(Arrays.asList(dense(lora(A, B)), dense(lora(A, B2))));
        assertArrayEquals(values(stackedDense), values(shared), 0f);

        LoraFactorSource both = (LoraFactorSource) ExpertTransforms.stack(Arrays.asList(lora(A, B), lora(A2, B2)));
        assertArrayEquals(new long[]{2, 2, 3}, both.a().shape());
        RowSource bothDense = ExpertTransforms.stack(Arrays.asList(dense(lora(A, B)), dense(lora(A2, B2))));
        assertArrayEquals(values(bothDense), values(both), 0f);
        check(both, s -> RowTransforms.sliceRows(s, 1, 1));
    }

    @Test
    public void transposeSwapsRolesAndConvertAppliesToBoth() throws IOException {
        LoraFactorSource t = check(lora(A, B), ExpertTransforms::transposeLast2);
        assertArrayEquals(new long[]{2, 4}, t.a().shape());
        assertArrayEquals(new long[]{3, 2}, t.b().shape());
        LoraFactorSource h = (LoraFactorSource) ElementTransforms.convert(lora(A, B), GGMLType.F16);
        assertEquals(ElementFormat.of(GGMLType.F16), h.a().format());
        assertEquals(ElementFormat.of(GGMLType.F16), h.b().format());
        assertArrayEquals(values(lora(A, B)), values(h), 0f);
        LoraFactorSource r = check(lora(A, B), s -> ElementTransforms.reshape(s, 2, 2, 3));
        assertArrayEquals(new long[]{1, 2, 3}, r.a().shape());
        assertArrayEquals(new long[]{2, 2, 2}, r.b().shape());
    }

    @Test
    public void otherOpsThrowTheLowRankError() {
        LoraFactorSource l = lora(A, B);
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> ElementTransforms.exp(l));
        assertEquals("LoRA delta of lora.q needs exp, which a low-rank factorization cannot express", e.getMessage());
        assertThrows(UnsupportedOperationException.class, () -> ElementTransforms.addScalar(l, 1f));
        assertThrows(UnsupportedOperationException.class, () -> ElementTransforms.negExp(l));
        assertThrows(UnsupportedOperationException.class, () -> RowTransforms.padRows(l, 6, 1f));
        assertThrows(UnsupportedOperationException.class, () -> ElementTransforms.reshape(l, 2, 6));
        assertThrows(UnsupportedOperationException.class, () -> ElementTransforms.ternaryRequant(l));
        assertThrows(UnsupportedOperationException.class, () -> QkvLayouts.mlaKvB(l, 2, 1, 1, 3));
    }
}
