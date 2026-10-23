package com.eonmux.neuralfree.acceleration.pipeline.schedule;

import org.junit.Test;

import static org.junit.Assert.*;

public class PipelineConfigTest {

    @Test
    public void testDefaultConfig() {
        PipelineConfig config = PipelineConfig.defaultConfig();
        
        assertTrue("Default should enable prefetch", config.enablePrefetch());
        assertEquals("Default prefetch depth should be 1", 1, config.prefetchDepth());
        assertTrue("Default should enable kernel cache", config.enableKernelCache());
        assertTrue("Default should enable memory pool", config.enableMemoryPool());
    }

    @Test
    public void testBuilderCustomConfig() {
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(false)
                .prefetchDepth(2)
                .enableKernelCache(false)
                .enableMemoryPool(false)
                .build();
        
        assertFalse("Prefetch should be disabled", config.enablePrefetch());
        assertEquals("Prefetch depth should be 2", 2, config.prefetchDepth());
        assertFalse("Kernel cache should be disabled", config.enableKernelCache());
        assertFalse("Memory pool should be disabled", config.enableMemoryPool());
    }

    @Test
    public void testRingBufferSlotCountWithPrefetchDisabled() {
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(false)
                .build();
        
        assertEquals("Disabled prefetch should have 1 slot", 1, config.ringBufferSlotCount());
    }

    @Test
    public void testRingBufferSlotCountDepth1() {
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        assertEquals("Depth 1 should have 2 slots (double buffer)", 
                2, config.ringBufferSlotCount());
    }

    @Test
    public void testRingBufferSlotCountDepth2() {
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(2)
                .build();
        
        assertEquals("Depth 2 should have 3 slots (triple buffer)", 
                3, config.ringBufferSlotCount());
    }

    @Test
    public void testRingBufferSlotCountDepth3() {
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(3)
                .build();
        
        assertEquals("Depth 3 should have 4 slots", 4, config.ringBufferSlotCount());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testPrefetchDepthTooLow() {
        PipelineConfig.builder().prefetchDepth(0).build();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testPrefetchDepthTooHigh() {
        PipelineConfig.builder().prefetchDepth(4).build();
    }

    @Test
    public void testEqualsAndHashCode() {
        PipelineConfig config1 = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        PipelineConfig config2 = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        PipelineConfig config3 = PipelineConfig.builder()
                .enablePrefetch(false)
                .prefetchDepth(1)
                .build();
        
        assertEquals("Equal configs should be equal", config1, config2);
        assertEquals("Equal configs should have same hash", config1.hashCode(), config2.hashCode());
        assertNotEquals("Different configs should not be equal", config1, config3);
    }

    @Test
    public void testToString() {
        PipelineConfig config = PipelineConfig.defaultConfig();
        String str = config.toString();
        
        assertTrue("toString should contain enablePrefetch", str.contains("enablePrefetch"));
        assertTrue("toString should contain prefetchDepth", str.contains("prefetchDepth"));
        assertTrue("toString should contain enableKernelCache", str.contains("enableKernelCache"));
        assertTrue("toString should contain enableMemoryPool", str.contains("enableMemoryPool"));
    }
}
