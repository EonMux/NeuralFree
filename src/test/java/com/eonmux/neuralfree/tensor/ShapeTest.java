package com.eonmux.neuralfree.tensor;

import org.junit.Test;
import static org.junit.Assert.*;

public class ShapeTest {

    @Test
    public void testScalar() {
        Shape s = Shape.scalar();
        assertEquals(0, s.rank());
        assertEquals(1, s.numElements());
        assertTrue(s.isScalar());
        assertFalse(s.isVector());
        assertFalse(s.isMatrix());
    }

    @Test
    public void testVector() {
        Shape s = Shape.vector(10);
        assertEquals(1, s.rank());
        assertEquals(10, s.dim(0));
        assertEquals(10, s.numElements());
        assertFalse(s.isScalar());
        assertTrue(s.isVector());
        assertFalse(s.isMatrix());
    }

    @Test
    public void testMatrix() {
        Shape s = Shape.matrix(3, 4);
        assertEquals(2, s.rank());
        assertEquals(3, s.dim(0));
        assertEquals(4, s.dim(1));
        assertEquals(12, s.numElements());
        assertFalse(s.isScalar());
        assertFalse(s.isVector());
        assertTrue(s.isMatrix());
    }

    @Test
    public void testOf() {
        Shape s = Shape.of(2, 3, 4);
        assertEquals(3, s.rank());
        assertEquals(2, s.dim(0));
        assertEquals(3, s.dim(1));
        assertEquals(4, s.dim(2));
        assertEquals(24, s.numElements());
    }

    @Test
    public void testByteSize() {
        Shape s = Shape.vector(100);
        assertEquals(400, s.byteSize(DType.FLOAT32));
        assertEquals(800, s.byteSize(DType.FLOAT64));
        assertEquals(100, s.byteSize(DType.INT8));
    }

    @Test
    public void testReshape() {
        Shape s = Shape.matrix(2, 3);
        Shape r = s.reshape(3, 2);
        assertEquals(2, r.rank());
        assertEquals(3, r.dim(0));
        assertEquals(2, r.dim(1));
        assertEquals(6, r.numElements());
    }

    @Test
    public void testReshapeWithInference() {
        Shape s = Shape.matrix(2, 3);
        Shape r = s.reshape(3, -1);
        assertEquals(2, r.rank());
        assertEquals(3, r.dim(0));
        assertEquals(2, r.dim(1));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testReshapeInvalidElementCount() {
        Shape s = Shape.vector(5);
        s.reshape(2, 3);
    }

    @Test
    public void testTranspose2D() {
        Shape s = Shape.matrix(3, 5);
        Shape t = s.transpose();
        assertEquals(2, t.rank());
        assertEquals(5, t.dim(0));
        assertEquals(3, t.dim(1));
    }

    @Test
    public void testTransposeWithAxes() {
        Shape s = Shape.of(2, 3, 4);
        Shape t = s.transpose(2, 0, 1);
        assertEquals(3, t.rank());
        assertEquals(4, t.dim(0));
        assertEquals(2, t.dim(1));
        assertEquals(3, t.dim(2));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTransposeInvalidAxesCount() {
        Shape s = Shape.matrix(2, 3);
        s.transpose(0);
    }

    @Test
    public void testBroadcastSameShape() {
        Shape a = Shape.matrix(2, 3);
        Shape b = Shape.matrix(2, 3);
        Shape result = a.broadcast(b);
        assertEquals(2, result.rank());
        assertEquals(2, result.dim(0));
        assertEquals(3, result.dim(1));
    }

    @Test
    public void testBroadcastWithOnes() {
        Shape a = Shape.matrix(2, 3);
        Shape b = Shape.of(1, 3);
        Shape result = a.broadcast(b);
        assertEquals(2, result.dim(0));
        assertEquals(3, result.dim(1));
    }

    @Test
    public void testBroadcastRightAligned() {
        Shape a = Shape.matrix(2, 3);
        Shape b = Shape.vector(3);
        Shape result = a.broadcast(b);
        assertEquals(2, result.dim(0));
        assertEquals(3, result.dim(1));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBroadcastIncompatible() {
        Shape a = Shape.vector(3);
        Shape b = Shape.vector(4);
        a.broadcast(b);
    }

    @Test
    public void testIsBroadcastCompatible() {
        assertTrue(Shape.isBroadcastCompatible(Shape.matrix(2, 3), Shape.of(1, 3)));
        assertTrue(Shape.isBroadcastCompatible(Shape.matrix(2, 3), Shape.vector(3)));
        assertFalse(Shape.isBroadcastCompatible(Shape.vector(3), Shape.vector(4)));
    }

    @Test
    public void testSqueeze() {
        Shape s = Shape.of(1, 3, 1, 4);
        Shape squeezed = s.squeeze();
        assertEquals(2, squeezed.rank());
        assertEquals(3, squeezed.dim(0));
        assertEquals(4, squeezed.dim(1));
    }

    @Test
    public void testSqueezeAxis() {
        Shape s = Shape.of(1, 3, 4);
        Shape squeezed = s.squeeze(0);
        assertEquals(2, squeezed.rank());
        assertEquals(3, squeezed.dim(0));
        assertEquals(4, squeezed.dim(1));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSqueezeNonOneAxis() {
        Shape s = Shape.matrix(2, 3);
        s.squeeze(0);
    }

    @Test
    public void testUnsqueeze() {
        Shape s = Shape.vector(3);
        Shape u = s.unsqueeze(0);
        assertEquals(2, u.rank());
        assertEquals(1, u.dim(0));
        assertEquals(3, u.dim(1));
    }

    @Test
    public void testUnsqueezeAtEnd() {
        Shape s = Shape.vector(3);
        Shape u = s.unsqueeze(1);
        assertEquals(2, u.rank());
        assertEquals(3, u.dim(0));
        assertEquals(1, u.dim(1));
    }

    @Test
    public void testDimsReturnsCopy() {
        Shape s = Shape.of(2, 3);
        int[] dims = s.dims();
        dims[0] = 100;
        assertEquals(2, s.dim(0));
    }

    @Test
    public void testEqualsAndHashCode() {
        Shape a = Shape.matrix(2, 3);
        Shape b = Shape.matrix(2, 3);
        Shape c = Shape.matrix(3, 2);
        
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotEquals(a, c);
    }

    @Test
    public void testToString() {
        Shape s = Shape.matrix(2, 3);
        assertTrue(s.toString().contains("2"));
        assertTrue(s.toString().contains("3"));
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testDimOutOfBounds() {
        Shape s = Shape.vector(5);
        s.dim(1);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testDimNegativeIndex() {
        Shape s = Shape.vector(5);
        s.dim(-1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testVectorNegativeDimension() {
        Shape.vector(-1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMatrixNegativeRow() {
        Shape.matrix(-1, 3);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMatrixNegativeCol() {
        Shape.matrix(2, -1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testOfNegativeDimension() {
        Shape.of(2, -1, 3);
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testTransposeScalar() {
        Shape s = Shape.scalar();
        s.transpose();
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testTransposeVector() {
        Shape s = Shape.vector(5);
        s.transpose();
    }
}
