package com.eonmux.neuralfree.distributed;

import org.junit.Test;
import static org.junit.Assert.*;

public class SharedMemoryBackendTest {

    @Test
    public void testConstructionWithValidParams() {
        SharedMemoryBackend backend = new SharedMemoryBackend(0, 2, "test_shm_" + System.currentTimeMillis());
        
        assertEquals(0, backend.rank());
        assertEquals(2, backend.worldSize());
        assertTrue(backend.isInitialized());
        
        backend.close();
        assertFalse(backend.isInitialized());
    }

    @Test
    public void testConstructionWithDefaultShmPrefix() {
        SharedMemoryBackend backend = new SharedMemoryBackend(1, 3);
        
        assertEquals(1, backend.rank());
        assertEquals(3, backend.worldSize());
        assertTrue(backend.isInitialized());
        
        backend.close();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidRankThrowsNegative() {
        new SharedMemoryBackend(-1, 2, "test");
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidRankThrowsTooHigh() {
        new SharedMemoryBackend(2, 2, "test");
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNullShmPrefixThrows() {
        new SharedMemoryBackend(0, 2, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEmptyShmPrefixThrows() {
        new SharedMemoryBackend(0, 2, "");
    }

    @Test
    public void testOperationsAfterCloseThrow() {
        SharedMemoryBackend backend = new SharedMemoryBackend(0, 2);
        backend.close();
        
        try {
            backend.rank();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException expected) {}
        
        try {
            backend.worldSize();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException expected) {}
        
        try {
            backend.barrier();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException expected) {}
    }

    @Test
    public void testCloseIsIdempotent() {
        SharedMemoryBackend backend = new SharedMemoryBackend(0, 2);
        
        backend.close();
        backend.close();
        backend.close();
    }

    @Test
    public void testSingleProcessAllReduce() {
        SharedMemoryBackend backend = new SharedMemoryBackend(0, 1, "test_allreduce_" + System.currentTimeMillis());
        
        backend.barrier();
        
        backend.close();
    }

    @Test
    public void testMultipleBackendsDifferentRanks() {
        String prefix = "test_multi_" + System.currentTimeMillis();
        SharedMemoryBackend backend0 = new SharedMemoryBackend(0, 2, prefix);
        SharedMemoryBackend backend1 = new SharedMemoryBackend(1, 2, prefix);
        
        assertEquals(0, backend0.rank());
        assertEquals(1, backend1.rank());
        assertEquals(2, backend0.worldSize());
        assertEquals(2, backend1.worldSize());
        
        backend0.close();
        backend1.close();
    }
}
