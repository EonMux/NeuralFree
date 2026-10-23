package com.eonmux.neuralfree.tensor;

import org.junit.Test;
import static org.junit.Assert.*;

public class StridesTest {

    @Test
    public void testScalar() {
        Strides s = Strides.scalar();
        assertEquals(0, s.rank());
        assertEquals(0, s.offset());
    }

    @Test
    public void testContiguousVector() {
        Shape shape = Shape.vector(10);
        Strides s = Strides.contiguous(shape);
        assertEquals(1, s.rank());
        assertEquals(1, s.stride(0));
        assertEquals(0, s.offset());
        assertTrue(s.isContiguous(shape));
    }

    @Test
    public void testContiguousMatrix() {
        Shape shape = Shape.matrix(3, 4);
        Strides s = Strides.contiguous(shape);
        assertEquals(2, s.rank());
        assertEquals(4, s.stride(0));
        assertEquals(1, s.stride(1));
        assertEquals(0, s.offset());
        assertTrue(s.isContiguous(shape));
    }

    @Test
    public void testContiguous3D() {
        Shape shape = Shape.of(2, 3, 4);
        Strides s = Strides.contiguous(shape);
        assertEquals(3, s.rank());
        assertEquals(12, s.stride(0));
        assertEquals(4, s.stride(1));
        assertEquals(1, s.stride(2));
        assertEquals(0, s.offset());
        assertTrue(s.isContiguous(shape));
    }

    @Test
    public void testFortranVector() {
        Shape shape = Shape.vector(10);
        Strides s = Strides.fortran(shape);
        assertEquals(1, s.rank());
        assertEquals(1, s.stride(0));
        assertTrue(s.isFortran(shape));
        assertTrue(s.isContiguous(shape));
    }

    @Test
    public void testFortranMatrix() {
        Shape shape = Shape.matrix(3, 4);
        Strides s = Strides.fortran(shape);
        assertEquals(2, s.rank());
        assertEquals(1, s.stride(0));
        assertEquals(3, s.stride(1));
        assertTrue(s.isFortran(shape));
        assertFalse(s.isContiguous(shape));
    }

    @Test
    public void testFortran3D() {
        Shape shape = Shape.of(2, 3, 4);
        Strides s = Strides.fortran(shape);
        assertEquals(3, s.rank());
        assertEquals(1, s.stride(0));
        assertEquals(2, s.stride(1));
        assertEquals(6, s.stride(2));
        assertTrue(s.isFortran(shape));
        assertFalse(s.isContiguous(shape));
    }

    @Test
    public void testTranspose() {
        Shape shape = Shape.matrix(3, 4);
        Strides s = Strides.contiguous(shape);
        Strides t = s.transpose(1, 0);
        
        assertEquals(2, t.rank());
        assertEquals(1, t.stride(0));
        assertEquals(4, t.stride(1));
        
        Shape transposedShape = shape.transpose();
        assertFalse(t.isContiguous(transposedShape));
    }

    @Test
    public void testTranspose3D() {
        Shape shape = Shape.of(2, 3, 4);
        Strides s = Strides.contiguous(shape);
        Strides t = s.transpose(2, 0, 1);
        
        assertEquals(3, t.rank());
        assertEquals(1, t.stride(0));
        assertEquals(12, t.stride(1));
        assertEquals(4, t.stride(2));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTransposeDuplicateAxis() {
        Strides s = Strides.contiguous(Shape.matrix(3, 4));
        s.transpose(0, 0);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testTransposeOutOfBounds() {
        Strides s = Strides.contiguous(Shape.matrix(3, 4));
        s.transpose(0, 2);
    }

    @Test
    public void testLinearIndexContiguous() {
        Shape shape = Shape.matrix(3, 4);
        Strides s = Strides.contiguous(shape);
        
        assertEquals(0, s.linearIndex(0, 0));
        assertEquals(1, s.linearIndex(0, 1));
        assertEquals(4, s.linearIndex(1, 0));
        assertEquals(5, s.linearIndex(1, 1));
        assertEquals(11, s.linearIndex(2, 3));
    }

    @Test
    public void testLinearIndexFortran() {
        Shape shape = Shape.matrix(3, 4);
        Strides s = Strides.fortran(shape);
        
        assertEquals(0, s.linearIndex(0, 0));
        assertEquals(1, s.linearIndex(1, 0));
        assertEquals(3, s.linearIndex(0, 1));
        assertEquals(4, s.linearIndex(1, 1));
        assertEquals(11, s.linearIndex(2, 3));
    }

    @Test
    public void testLinearIndexLong() {
        Shape shape = Shape.of(2, 3, 4);
        Strides s = Strides.contiguous(shape);
        
        assertEquals(0, s.linearIndex(0L, 0L, 0L));
        assertEquals(1, s.linearIndex(0L, 0L, 1L));
        assertEquals(4, s.linearIndex(0L, 1L, 0L));
        assertEquals(12, s.linearIndex(1L, 0L, 0L));
        assertEquals(23, s.linearIndex(1L, 2L, 3L));
    }

    @Test
    public void testWithOffset() {
        Strides s = Strides.contiguous(Shape.vector(10));
        Strides withOffset = s.withOffset(5);
        
        assertEquals(5, withOffset.offset());
        assertEquals(5, withOffset.linearIndex(0));
        assertEquals(6, withOffset.linearIndex(1));
    }

    @Test
    public void testSlice() {
        Shape shape = Shape.matrix(3, 4);
        Strides s = Strides.contiguous(shape);
        Strides sliced = s.slice(0, 1, 3);
        
        assertEquals(4, sliced.offset());
    }

    @Test
    public void testSqueeze() {
        Shape shape = Shape.of(2, 1, 4);
        Strides s = Strides.contiguous(shape);
        Strides squeezed = s.squeeze(1);
        
        assertEquals(2, squeezed.rank());
        assertEquals(4, squeezed.stride(0));
        assertEquals(1, squeezed.stride(1));
    }

    @Test
    public void testUnsqueeze() {
        Shape shape = Shape.matrix(3, 4);
        Strides s = Strides.contiguous(shape);
        Strides unsqueezed = s.unsqueeze(0);
        
        assertEquals(3, unsqueezed.rank());
        assertEquals(1, unsqueezed.stride(0));
        assertEquals(4, unsqueezed.stride(1));
        assertEquals(1, unsqueezed.stride(2));
    }

    @Test
    public void testStridesReturnsCopy() {
        Strides s = Strides.contiguous(Shape.vector(10));
        long[] strides = s.strides();
        strides[0] = 999;
        assertEquals(1, s.stride(0));
    }

    @Test
    public void testEqualsAndHashCode() {
        Strides s1 = Strides.contiguous(Shape.vector(10));
        Strides s2 = Strides.contiguous(Shape.vector(10));
        Strides s3 = Strides.fortran(Shape.matrix(3, 4));
        
        assertEquals(s1, s2);
        assertEquals(s1.hashCode(), s2.hashCode());
        assertNotEquals(s1, s3);
    }

    @Test
    public void testToString() {
        Strides s = Strides.contiguous(Shape.vector(10));
        String str = s.toString();
        assertTrue(str.contains("Strides"));
        assertTrue(str.contains("offset=0"));
    }

    @Test
    public void testIsContiguousWithUnitDim() {
        Shape shape = Shape.of(1, 4, 1);
        Strides s = Strides.contiguous(shape);
        assertTrue(s.isContiguous(shape));
    }

    @Test
    public void testIsContiguousWithOffset() {
        Shape shape = Shape.vector(10);
        Strides s = Strides.contiguous(shape).withOffset(5);
        assertFalse(s.isContiguous(shape));
    }

    @Test
    public void testContiguousMatchesCOrder() {
        Shape shape = Shape.of(2, 3, 4);
        Strides s = Strides.contiguous(shape);
        
        assertEquals(12, s.stride(0));
        assertEquals(4, s.stride(1));
        assertEquals(1, s.stride(2));
    }

    @Test(expected = NullPointerException.class)
    public void testContiguousNullShape() {
        Strides.contiguous(null);
    }

    @Test(expected = NullPointerException.class)
    public void testFortranNullShape() {
        Strides.fortran(null);
    }

    @Test(expected = NullPointerException.class)
    public void testIsContiguousNullShape() {
        Strides s = Strides.contiguous(Shape.vector(10));
        s.isContiguous(null);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testStrideOutOfBounds() {
        Strides s = Strides.contiguous(Shape.vector(10));
        s.stride(1);
    }

    @Test
    public void testIsFortranWithUnitDim() {
        Shape shape = Shape.of(1, 4, 1);
        Strides s = Strides.fortran(shape);
        assertTrue(s.isFortran(shape));
    }
}
