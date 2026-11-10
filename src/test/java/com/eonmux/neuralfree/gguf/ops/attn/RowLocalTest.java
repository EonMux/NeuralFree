package com.eonmux.neuralfree.gguf.ops.attn;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * The workgroup size of the per-row norm+rotation steps ({@link GpuNormRopeStages#rowLocal}): one warp for rows of up
 * to 128 values, doubling per further 4 values per work-item, capped at the mixer's workgroup size.
 */
public class RowLocalTest {

    @Test
    public void rowsTakeFourValuesPerWorkItemFromOneWarp() {
        assertEquals(32, GpuNormRopeStages.rowLocal(64, 256));
        assertEquals(32, GpuNormRopeStages.rowLocal(128, 256));
        assertEquals(64, GpuNormRopeStages.rowLocal(129, 256));
        assertEquals(64, GpuNormRopeStages.rowLocal(256, 256));
        assertEquals(128, GpuNormRopeStages.rowLocal(512, 256));
    }

    @Test
    public void theMixerWorkgroupCapsTheRowWorkgroup() {
        assertEquals(256, GpuNormRopeStages.rowLocal(4096, 256));
        assertEquals(64, GpuNormRopeStages.rowLocal(512, 64));
        assertEquals(16, GpuNormRopeStages.rowLocal(128, 16));
    }
}
