package com.eonmux.neuralfree.tensor;

import org.junit.Test;

import static org.junit.Assert.*;

public class BroadcastEngineTest {

    @Test
    public void testBroadcastShapeSameShape() {
        Shape a = Shape.matrix(3, 4);
        Shape b = Shape.matrix(3, 4);
        
        Shape result = BroadcastEngine.broadcastShape(a, b);
        
        assertEquals(Shape.matrix(3, 4), result);
    }

    @Test
    public void testBroadcastShapeScalar() {
        Shape scalar = Shape.scalar();
        Shape matrix = Shape.matrix(3, 4);
        
        Shape result1 = BroadcastEngine.broadcastShape(scalar, matrix);
        Shape result2 = BroadcastEngine.broadcastShape(matrix, scalar);
        
        assertEquals(Shape.matrix(3, 4), result1);
        assertEquals(Shape.matrix(3, 4), result2);
    }

    @Test
    public void testBroadcastShapeWithOnes() {
        Shape a = Shape.matrix(3, 4);
        Shape b = Shape.of(1, 4);
        
        Shape result = BroadcastEngine.broadcastShape(a, b);
        
        assertEquals(Shape.matrix(3, 4), result);
    }

    @Test
    public void testBroadcastShapeRightAligned() {
        Shape a = Shape.matrix(3, 4);
        Shape b = Shape.vector(4);
        
        Shape result = BroadcastEngine.broadcastShape(a, b);
        
        assertEquals(Shape.matrix(3, 4), result);
    }

    @Test
    public void testBroadcastShape3D() {
        Shape a = Shape.of(8, 1, 6, 1);
        Shape b = Shape.of(7, 1, 5);
        
        Shape result = BroadcastEngine.broadcastShape(a, b);
        
        assertEquals(Shape.of(8, 7, 6, 5), result);
    }

    @Test
    public void testBroadcastShapeBothHaveOnes() {
        Shape a = Shape.of(1, 3, 1);
        Shape b = Shape.of(2, 1, 4);
        
        Shape result = BroadcastEngine.broadcastShape(a, b);
        
        assertEquals(Shape.of(2, 3, 4), result);
    }

    @Test
    public void testBroadcastShapeMultipleLeadingOnes() {
        Shape a = Shape.vector(5);
        Shape b = Shape.of(1, 1, 5);
        
        Shape result = BroadcastEngine.broadcastShape(a, b);
        
        assertEquals(Shape.of(1, 1, 5), result);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBroadcastShapeIncompatible() {
        Shape a = Shape.vector(3);
        Shape b = Shape.vector(4);
        
        BroadcastEngine.broadcastShape(a, b);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBroadcastShapeIncompatible2D() {
        Shape a = Shape.matrix(3, 5);
        Shape b = Shape.matrix(3, 4);
        
        BroadcastEngine.broadcastShape(a, b);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBroadcastShapeIncompatibleMiddleDimension() {
        Shape a = Shape.of(2, 3, 4);
        Shape b = Shape.of(2, 5, 4);
        
        BroadcastEngine.broadcastShape(a, b);
    }

    @Test
    public void testIsBroadcastableToTrue() {
        assertTrue(BroadcastEngine.isBroadcastableTo(Shape.vector(3), Shape.matrix(2, 3)));
        assertTrue(BroadcastEngine.isBroadcastableTo(Shape.of(1, 3), Shape.matrix(2, 3)));
        assertTrue(BroadcastEngine.isBroadcastableTo(Shape.scalar(), Shape.matrix(5, 5)));
        assertTrue(BroadcastEngine.isBroadcastableTo(Shape.matrix(2, 3), Shape.matrix(2, 3)));
    }

    @Test
    public void testIsBroadcastableToFalse() {
        assertFalse(BroadcastEngine.isBroadcastableTo(Shape.vector(4), Shape.matrix(2, 3)));
        assertFalse(BroadcastEngine.isBroadcastableTo(Shape.of(1, 4), Shape.matrix(2, 3)));
        assertFalse(BroadcastEngine.isBroadcastableTo(Shape.of(2, 3), Shape.vector(3)));
        assertFalse(BroadcastEngine.isBroadcastableTo(Shape.matrix(3, 3), Shape.matrix(2, 2)));
    }

    @Test
    public void testIsBroadcastableToHigherRank() {
        assertFalse(BroadcastEngine.isBroadcastableTo(Shape.of(3, 4, 5), Shape.matrix(4, 5)));
    }

    @Test
    public void testBroadcastStridesNoBroadcast() {
        Shape sourceShape = Shape.matrix(3, 4);
        Shape targetShape = Shape.matrix(3, 4);
        Strides strides = Strides.contiguous(sourceShape);
        
        long[] result = BroadcastEngine.broadcastStrides(sourceShape, targetShape, strides);
        
        assertEquals(2, result.length);
        assertEquals(4, result[0]);
        assertEquals(1, result[1]);
    }

    @Test
    public void testBroadcastStridesWithLeadingOne() {
        Shape sourceShape = Shape.of(1, 4);
        Shape targetShape = Shape.matrix(3, 4);
        Strides strides = Strides.contiguous(sourceShape);
        
        long[] result = BroadcastEngine.broadcastStrides(sourceShape, targetShape, strides);
        
        assertEquals(2, result.length);
        assertEquals(0, result[0]);
        assertEquals(1, result[1]);
    }

    @Test
    public void testBroadcastStridesWithTrailingOne() {
        Shape sourceShape = Shape.of(3, 1);
        Shape targetShape = Shape.matrix(3, 4);
        Strides strides = Strides.contiguous(sourceShape);
        
        long[] result = BroadcastEngine.broadcastStrides(sourceShape, targetShape, strides);
        
        assertEquals(2, result.length);
        assertEquals(1, result[0]);
        assertEquals(0, result[1]);
    }

    @Test
    public void testBroadcastStridesWithScalar() {
        Shape sourceShape = Shape.scalar();
        Shape targetShape = Shape.matrix(3, 4);
        Strides strides = Strides.scalar();
        
        long[] result = BroadcastEngine.broadcastStrides(sourceShape, targetShape, strides);
        
        assertEquals(2, result.length);
        assertEquals(0, result[0]);
        assertEquals(0, result[1]);
    }

    @Test
    public void testBroadcastStridesWithVector() {
        Shape sourceShape = Shape.vector(4);
        Shape targetShape = Shape.matrix(3, 4);
        Strides strides = Strides.contiguous(sourceShape);
        
        long[] result = BroadcastEngine.broadcastStrides(sourceShape, targetShape, strides);
        
        assertEquals(2, result.length);
        assertEquals(0, result[0]);
        assertEquals(1, result[1]);
    }

    @Test
    public void testComputeBroadcast() {
        Shape a = Shape.matrix(3, 1);
        Shape b = Shape.of(1, 4);
        
        BroadcastEngine.BroadcastInfo info = BroadcastEngine.computeBroadcast(a, b);
        
        assertEquals(Shape.matrix(3, 4), info.resultShape());
        
        long[] stridesA = info.stridesA();
        long[] stridesB = info.stridesB();
        
        assertEquals(2, stridesA.length);
        assertEquals(2, stridesB.length);
        
        assertEquals(1, stridesA[0]);
        assertEquals(0, stridesA[1]);
        
        assertEquals(0, stridesB[0]);
        assertEquals(1, stridesB[1]);
    }

    @Test
    public void testComputeBroadcastSameShape() {
        Shape a = Shape.matrix(2, 3);
        Shape b = Shape.matrix(2, 3);
        
        BroadcastEngine.BroadcastInfo info = BroadcastEngine.computeBroadcast(a, b);
        
        assertEquals(Shape.matrix(2, 3), info.resultShape());
        
        long[] stridesA = info.stridesA();
        long[] stridesB = info.stridesB();
        
        assertEquals(2, stridesA.length);
        assertEquals(2, stridesB.length);
        
        assertEquals(1, stridesA[1]);
        assertEquals(1, stridesB[1]);
    }

    @Test
    public void testBroadcastInfoStrideAccessors() {
        Shape a = Shape.matrix(2, 3);
        Shape b = Shape.matrix(2, 3);
        
        BroadcastEngine.BroadcastInfo info = BroadcastEngine.computeBroadcast(a, b);
        
        assertEquals(3, info.strideA(0));
        assertEquals(1, info.strideA(1));
        assertEquals(3, info.strideB(0));
        assertEquals(1, info.strideB(1));
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testBroadcastInfoStrideAOutOfBounds() {
        Shape a = Shape.matrix(2, 3);
        Shape b = Shape.matrix(2, 3);
        
        BroadcastEngine.BroadcastInfo info = BroadcastEngine.computeBroadcast(a, b);
        info.strideA(5);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testBroadcastInfoStrideBOutOfBounds() {
        Shape a = Shape.matrix(2, 3);
        Shape b = Shape.matrix(2, 3);
        
        BroadcastEngine.BroadcastInfo info = BroadcastEngine.computeBroadcast(a, b);
        info.strideB(-1);
    }

    @Test
    public void testReduceShapeNoKeepDim() {
        Shape shape = Shape.matrix(3, 4);
        
        Shape result0 = BroadcastEngine.reduceShape(shape, new int[]{0}, false);
        Shape result1 = BroadcastEngine.reduceShape(shape, new int[]{1}, false);
        Shape resultAll = BroadcastEngine.reduceShape(shape, new int[]{0, 1}, false);
        
        assertEquals(Shape.vector(4), result0);
        assertEquals(Shape.vector(3), result1);
        assertEquals(Shape.scalar(), resultAll);
    }

    @Test
    public void testReduceShapeWithKeepDim() {
        Shape shape = Shape.matrix(3, 4);
        
        Shape result0 = BroadcastEngine.reduceShape(shape, new int[]{0}, true);
        Shape result1 = BroadcastEngine.reduceShape(shape, new int[]{1}, true);
        Shape resultAll = BroadcastEngine.reduceShape(shape, new int[]{0, 1}, true);
        
        assertEquals(Shape.of(1, 4), result0);
        assertEquals(Shape.of(3, 1), result1);
        assertEquals(Shape.of(1, 1), resultAll);
    }

    @Test
    public void testReduceShapeNegativeAxis() {
        Shape shape = Shape.of(2, 3, 4);
        
        Shape result = BroadcastEngine.reduceShape(shape, new int[]{-1}, false);
        
        assertEquals(Shape.of(2, 3), result);
    }

    @Test
    public void testReduceShapeEmptyDimsNoKeepDim() {
        Shape shape = Shape.matrix(3, 4);
        
        Shape result = BroadcastEngine.reduceShape(shape, new int[]{}, false);
        
        assertEquals(Shape.scalar(), result);
    }

    @Test
    public void testReduceShapeEmptyDimsWithKeepDim() {
        Shape shape = Shape.matrix(3, 4);
        
        Shape result = BroadcastEngine.reduceShape(shape, new int[]{}, true);
        
        assertEquals(shape, result);
    }

    @Test
    public void testReduceShape3D() {
        Shape shape = Shape.of(2, 3, 4);
        
        Shape result0 = BroadcastEngine.reduceShape(shape, new int[]{0}, false);
        Shape result1 = BroadcastEngine.reduceShape(shape, new int[]{1}, false);
        Shape result2 = BroadcastEngine.reduceShape(shape, new int[]{2}, false);
        
        assertEquals(Shape.of(3, 4), result0);
        assertEquals(Shape.of(2, 4), result1);
        assertEquals(Shape.of(2, 3), result2);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testReduceShapeAxisOutOfBounds() {
        Shape shape = Shape.matrix(3, 4);
        BroadcastEngine.reduceShape(shape, new int[]{5}, false);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testReduceShapeNegativeAxisOutOfBounds() {
        Shape shape = Shape.matrix(3, 4);
        BroadcastEngine.reduceShape(shape, new int[]{-3}, false);
    }

    @Test
    public void testNormalizeAxesPositive() {
        int[] axes = {0, 1, 2};
        int[] normalized = BroadcastEngine.normalizeAxes(axes, 3);
        
        assertArrayEquals(new int[]{0, 1, 2}, normalized);
    }

    @Test
    public void testNormalizeAxesNegative() {
        int[] axes = {-1, -2, -3};
        int[] normalized = BroadcastEngine.normalizeAxes(axes, 3);
        
        assertArrayEquals(new int[]{2, 1, 0}, normalized);
    }

    @Test
    public void testNormalizeAxesMixed() {
        int[] axes = {0, -1};
        int[] normalized = BroadcastEngine.normalizeAxes(axes, 3);
        
        assertArrayEquals(new int[]{0, 2}, normalized);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testNormalizeAxesOutOfBounds() {
        int[] axes = {5};
        BroadcastEngine.normalizeAxes(axes, 3);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testNormalizeAxesNegativeOutOfBounds() {
        int[] axes = {-5};
        BroadcastEngine.normalizeAxes(axes, 3);
    }

    @Test
    public void testSqueezeShape() {
        Shape shape = Shape.of(1, 3, 1, 4, 1);
        
        Shape result = BroadcastEngine.squeezeShape(shape, new int[]{});
        
        assertEquals(Shape.of(3, 4), result);
    }

    @Test
    public void testSqueezeShapeSpecificDims() {
        Shape shape = Shape.of(1, 3, 1, 4);
        
        Shape result = BroadcastEngine.squeezeShape(shape, new int[]{0, 2});
        
        assertEquals(Shape.of(3, 4), result);
    }

    @Test
    public void testSqueezeShapeToScalar() {
        Shape shape = Shape.of(1, 1, 1);
        
        Shape result = BroadcastEngine.squeezeShape(shape, new int[]{});
        
        assertEquals(Shape.scalar(), result);
    }

    @Test
    public void testSqueezeShapeNegativeAxis() {
        Shape shape = Shape.of(1, 3, 4);
        
        Shape result = BroadcastEngine.squeezeShape(shape, new int[]{0});
        
        assertEquals(Shape.of(3, 4), result);
    }

    @Test
    public void testBroadcastShapeBothScalars() {
        Shape a = Shape.scalar();
        Shape b = Shape.scalar();
        
        Shape result = BroadcastEngine.broadcastShape(a, b);
        
        assertEquals(Shape.scalar(), result);
    }

    @Test
    public void testBroadcastStridesDefensiveCopy() {
        Shape sourceShape = Shape.matrix(3, 4);
        Shape targetShape = Shape.matrix(3, 4);
        Strides strides = Strides.contiguous(sourceShape);
        
        long[] result1 = BroadcastEngine.broadcastStrides(sourceShape, targetShape, strides);
        long[] result2 = BroadcastEngine.broadcastStrides(sourceShape, targetShape, strides);
        
        assertNotSame(result1, result2);
    }

    @Test
    public void testBroadcastInfoStridesDefensiveCopy() {
        Shape a = Shape.matrix(2, 3);
        Shape b = Shape.matrix(2, 3);
        
        BroadcastEngine.BroadcastInfo info = BroadcastEngine.computeBroadcast(a, b);
        
        long[] strides1 = info.stridesA();
        long[] strides2 = info.stridesA();
        
        assertNotSame(strides1, strides2);
        
        strides1 = info.stridesB();
        strides2 = info.stridesB();
        
        assertNotSame(strides1, strides2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBroadcastShapeNumPyIncompatibleCase1() {
        Shape a = Shape.of(1, 3, 2);
        Shape b = Shape.of(2, 1);
        
        BroadcastEngine.broadcastShape(a, b);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBroadcastShapeNumPyIncompatibleCase2() {
        Shape a = Shape.of(1, 3, 2);
        Shape b = Shape.of(2, 2);
        
        BroadcastEngine.broadcastShape(a, b);
    }

    @Test
    public void testBroadcastShapeNumPyEdgeCase2() {
        Shape a = Shape.of(4, 1, 6);
        Shape b = Shape.of(3, 1);
        
        Shape result = BroadcastEngine.broadcastShape(a, b);
        
        assertEquals(Shape.of(4, 3, 6), result);
    }

    @Test
    public void testBroadcastShapeNumPyEdgeCase3() {
        Shape a = Shape.of(1, 3);
        Shape b = Shape.of(8, 4, 3);
        
        Shape result = BroadcastEngine.broadcastShape(a, b);
        
        assertEquals(Shape.of(8, 4, 3), result);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBroadcastShapeNumPyIncompatibleCase3() {
        Shape a = Shape.of(2, 1);
        Shape b = Shape.of(8, 4, 3);
        
        BroadcastEngine.broadcastShape(a, b);
    }

    @Test(expected = NullPointerException.class)
    public void testBroadcastShapeNullA() {
        BroadcastEngine.broadcastShape(null, Shape.scalar());
    }

    @Test(expected = NullPointerException.class)
    public void testBroadcastShapeNullB() {
        BroadcastEngine.broadcastShape(Shape.scalar(), null);
    }

    @Test(expected = NullPointerException.class)
    public void testIsBroadcastableToNullSource() {
        BroadcastEngine.isBroadcastableTo(null, Shape.scalar());
    }

    @Test(expected = NullPointerException.class)
    public void testIsBroadcastableToNullTarget() {
        BroadcastEngine.isBroadcastableTo(Shape.scalar(), null);
    }

    @Test(expected = NullPointerException.class)
    public void testBroadcastStridesNullSourceShape() {
        BroadcastEngine.broadcastStrides(null, Shape.scalar(), Strides.scalar());
    }

    @Test(expected = NullPointerException.class)
    public void testBroadcastStridesNullTargetShape() {
        BroadcastEngine.broadcastStrides(Shape.scalar(), null, Strides.scalar());
    }

    @Test(expected = NullPointerException.class)
    public void testBroadcastStridesNullStrides() {
        BroadcastEngine.broadcastStrides(Shape.scalar(), Shape.scalar(), null);
    }

    @Test(expected = NullPointerException.class)
    public void testComputeBroadcastNullA() {
        BroadcastEngine.computeBroadcast(null, Shape.scalar());
    }

    @Test(expected = NullPointerException.class)
    public void testComputeBroadcastNullB() {
        BroadcastEngine.computeBroadcast(Shape.scalar(), null);
    }

    @Test(expected = NullPointerException.class)
    public void testReduceShapeNullShape() {
        BroadcastEngine.reduceShape(null, new int[]{0}, false);
    }

    @Test(expected = NullPointerException.class)
    public void testReduceShapeNullDims() {
        BroadcastEngine.reduceShape(Shape.scalar(), null, false);
    }

    @Test(expected = NullPointerException.class)
    public void testSqueezeShapeNullShape() {
        BroadcastEngine.squeezeShape(null, new int[]{});
    }

    @Test(expected = NullPointerException.class)
    public void testSqueezeShapeNullDims() {
        BroadcastEngine.squeezeShape(Shape.scalar(), null);
    }

    @Test
    public void testToString() {
        Shape a = Shape.matrix(2, 3);
        Shape b = Shape.matrix(2, 3);
        
        BroadcastEngine.BroadcastInfo info = BroadcastEngine.computeBroadcast(a, b);
        String str = info.toString();
        
        assertTrue(str.contains("BroadcastInfo"));
        assertTrue(str.contains("resultShape"));
        assertTrue(str.contains("stridesA"));
        assertTrue(str.contains("stridesB"));
    }
}
