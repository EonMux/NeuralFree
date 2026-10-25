package com.eonmux.neuralfree.gguf;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link DataSegment}: validation, overflow refusal, containment and file-offset arithmetic. */
public class DataSegmentTest {

    @Test
    public void mapsVirtualOffsetsToFileOffsets() {
        DataSegment s = new DataSegment("b.gguf", 96, 1000, 64);
        assertEquals(1064, s.virtualEnd());
        assertEquals(96, s.fileOffset(1000));
        assertEquals(159, s.fileOffset(1063));
    }

    @Test
    public void containsOnlyRangesInsideTheSegment() {
        DataSegment s = new DataSegment("b.gguf", 96, 1000, 64);
        assertTrue(s.contains(1000, 64));
        assertTrue(s.contains(1063, 1));
        assertTrue(s.contains(1064, 0));
        assertFalse(s.contains(999, 1));
        assertFalse(s.contains(1060, 8));
        assertFalse(s.contains(1064, 1));
        assertFalse(s.contains(1000, -1));
        assertFalse(s.contains(1000, Long.MAX_VALUE));
        assertFalse(s.contains(Long.MAX_VALUE, Long.MAX_VALUE));
    }

    @Test
    public void emptySegmentHoldsOnlyEmptyRanges() {
        DataSegment s = new DataSegment("a.gguf", 32, 0, 0);
        assertEquals(0, s.virtualEnd());
        assertTrue(s.contains(0, 0));
        assertFalse(s.contains(0, 1));
    }

    @Test
    public void refusesNegativeValues() {
        assertThrows(IllegalArgumentException.class, () -> new DataSegment("a", -1, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> new DataSegment("a", 0, -1, 1));
        assertThrows(IllegalArgumentException.class, () -> new DataSegment("a", 0, 0, -1));
        assertThrows(IllegalArgumentException.class, () -> new DataSegment("", 0, 0, 1));
        assertThrows(NullPointerException.class, () -> new DataSegment(null, 0, 0, 1));
    }

    @Test
    public void refusesSegmentsWhoseEndOverflows() {
        IllegalArgumentException file = assertThrows(IllegalArgumentException.class,
                () -> new DataSegment("a", Long.MAX_VALUE - 7, 0, 8));
        assertTrue(file.getMessage(), file.getMessage().contains("overflows a 64-bit offset"));
        IllegalArgumentException virtual = assertThrows(IllegalArgumentException.class,
                () -> new DataSegment("a", 0, Long.MAX_VALUE - 7, 8));
        assertTrue(virtual.getMessage(), virtual.getMessage().contains("overflows a 64-bit offset"));
        assertEquals(Long.MAX_VALUE, new DataSegment("a", 0, Long.MAX_VALUE - 8, 8).virtualEnd());
    }

    @Test
    public void equalityCoversEveryField() {
        DataSegment s = new DataSegment("a", 1, 2, 3);
        assertEquals(s, new DataSegment("a", 1, 2, 3));
        assertEquals(s.hashCode(), new DataSegment("a", 1, 2, 3).hashCode());
        assertNotEquals(s, new DataSegment("b", 1, 2, 3));
        assertNotEquals(s, new DataSegment("a", 0, 2, 3));
        assertNotEquals(s, new DataSegment("a", 1, 0, 3));
        assertNotEquals(s, new DataSegment("a", 1, 2, 0));
    }
}
