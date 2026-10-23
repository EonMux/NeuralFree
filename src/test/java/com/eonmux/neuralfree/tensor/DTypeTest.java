package com.eonmux.neuralfree.tensor;

import org.junit.Test;
import static org.junit.Assert.*;

public class DTypeTest {

    @Test
    public void testByteSizes() {
        assertEquals(2, DType.FLOAT16.byteSize());
        assertEquals(4, DType.FLOAT32.byteSize());
        assertEquals(8, DType.FLOAT64.byteSize());
        assertEquals(1, DType.INT8.byteSize());
        assertEquals(2, DType.INT16.byteSize());
        assertEquals(4, DType.INT32.byteSize());
        assertEquals(8, DType.INT64.byteSize());
        assertEquals(2, DType.BFLOAT16.byteSize());
        assertEquals(1, DType.BOOL.byteSize());
    }

    @Test
    public void testIsFloatingPoint() {
        assertTrue(DType.FLOAT16.isFloatingPoint());
        assertTrue(DType.FLOAT32.isFloatingPoint());
        assertTrue(DType.FLOAT64.isFloatingPoint());
        assertTrue(DType.BFLOAT16.isFloatingPoint());
        
        assertFalse(DType.INT8.isFloatingPoint());
        assertFalse(DType.INT16.isFloatingPoint());
        assertFalse(DType.INT32.isFloatingPoint());
        assertFalse(DType.INT64.isFloatingPoint());
        assertFalse(DType.BOOL.isFloatingPoint());
    }

    @Test
    public void testIsInteger() {
        assertTrue(DType.INT8.isInteger());
        assertTrue(DType.INT16.isInteger());
        assertTrue(DType.INT32.isInteger());
        assertTrue(DType.INT64.isInteger());
        
        assertFalse(DType.FLOAT16.isInteger());
        assertFalse(DType.FLOAT32.isInteger());
        assertFalse(DType.FLOAT64.isInteger());
        assertFalse(DType.BFLOAT16.isInteger());
        assertFalse(DType.BOOL.isInteger());
    }

    @Test
    public void testIsBoolean() {
        assertTrue(DType.BOOL.isBoolean());
        
        assertFalse(DType.FLOAT32.isBoolean());
        assertFalse(DType.INT32.isBoolean());
    }

    @Test
    public void testAllDTypesExist() {
        DType[] types = DType.values();
        assertEquals(9, types.length);
    }
}
