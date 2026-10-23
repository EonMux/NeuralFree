package com.eonmux.neuralfree.acceleration.cache;

import org.junit.Test;

import static org.junit.Assert.*;

public class CacheKeyTest {

    @Test
    public void testEqualsAndHashCode() {
        CacheKey key1 = new CacheKey("kernel", "hash", "-O3");
        CacheKey key2 = new CacheKey("kernel", "hash", "-O3");
        CacheKey key3 = new CacheKey("kernel", "hash", "-O2");
        
        assertEquals(key1, key2);
        assertEquals(key1.hashCode(), key2.hashCode());
        assertNotEquals(key1, key3);
    }

    @Test
    public void testNullBuildOptions() {
        CacheKey key1 = new CacheKey("kernel", "hash", null);
        CacheKey key2 = new CacheKey("kernel", "hash", "");
        
        assertEquals(key1, key2);
    }

    @Test
    public void testDifferentSourceHash() {
        CacheKey key1 = new CacheKey("kernel", "hash1", "");
        CacheKey key2 = new CacheKey("kernel", "hash2", "");
        
        assertNotEquals(key1, key2);
    }
}
