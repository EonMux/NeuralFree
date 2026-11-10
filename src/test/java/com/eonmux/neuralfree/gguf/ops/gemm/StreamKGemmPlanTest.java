package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The stream-k plan of the CUDA tensor tile: which launches leave the plain grid (llama.cpp's 90% wave
 * fill), the even partition of the (tile, chunk) sequence the kernels compute, and the launch geometry.
 */
public class StreamKGemmPlanTest {

    private static final int UNITS = 60;  /* the RTX 5080 Laptop's SMs */

    private static MmqGemmPlan plain(GGMLType type, int m, int k, int n) {
        return MmqGemmPlan.contiguous(MmqConfig.INT8_TENSOR, type, m, k, n);
    }

    /** The 4B and 0.6B prefill grids whose last wave runs part-empty take stream-k over one block per unit. */
    @Test
    public void partEmptyWavesTakeStreamK() {
        StreamKGemmPlan down = StreamKGemmPlan.of(plain(GGMLType.Q8_0, 2560, 9728, 512), UNITS);
        assertNotNull("80 tiles fill 67% of two waves", down);
        assertEquals(80, down.tiles());
        assertEquals(UNITS, down.blocks());
        assertEquals(9728 / 32, down.chunksPerTile());
        assertEquals("Q8_0_MatMul_MMQ_StreamK", down.kernelName());
        assertNotNull("304 tiles fill 84% of six waves",
                StreamKGemmPlan.of(plain(GGMLType.Q4_K, 9728, 2560, 512), UNITS));
        assertNotNull("32 tiles fill half a wave", StreamKGemmPlan.of(plain(GGMLType.Q8_0, 1024, 1024, 512), UNITS));
    }

    /** Full waves keep the plain grid, as do tiles other than the tensor tile, the FP8 types and a device
     *  without a compute-unit count. */
    @Test
    public void fullWavesAndOtherTilesKeepThePlainGrid() {
        assertNull("120 tiles are two full waves", StreamKGemmPlan.of(plain(GGMLType.Q8_0, 3840, 2560, 512), UNITS));
        assertNull("55 tiles fill 91% of one wave", StreamKGemmPlan.of(plain(GGMLType.Q8_0, 128 * 55, 2560, 128),
                UNITS));
        assertNull(StreamKGemmPlan.of(plain(GGMLType.Q8_0, 2560, 9728, 512), DeviceCapabilities.UNKNOWN_COMPUTE_UNITS));
        assertNull(StreamKGemmPlan.of(MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, GGMLType.Q8_0, 2560, 9728, 512),
                UNITS));
        assertNull(StreamKGemmPlan.of(plain(GGMLType.F8_E4M3, 2560, 9728, 512), UNITS));
        try {
            StreamKGemmPlan.of(plain(GGMLType.Q8_0, 2560, 9728, 512), -1);
            fail("a negative compute-unit count must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("compute-unit"));
        }
    }

    @Test
    public void waveFillIsTheBusyShareOfTheWaves() {
        assertEquals(66, StreamKGemmPlan.waveFillPercent(80, 60));
        assertEquals(84, StreamKGemmPlan.waveFillPercent(304, 60));
        assertEquals(100, StreamKGemmPlan.waveFillPercent(120, 60));
        assertEquals(53, StreamKGemmPlan.waveFillPercent(32, 60));
    }

    /** The blocks cover the (tile, chunk) sequence exactly once in equal runs, as the kernel's start does. */
    @Test
    public void theBlocksPartitionTheSequenceEvenly() {
        StreamKGemmPlan p = StreamKGemmPlan.of(plain(GGMLType.Q8_0, 2560, 9728, 512), UNITS);
        long total = p.tiles() * p.chunksPerTile();
        assertEquals(0, p.blockStart(0));
        assertEquals(total, p.blockStart(p.blocks()));
        for (int b = 0; b < p.blocks(); b++) {
            long run = p.blockStart(b + 1) - p.blockStart(b);
            assertTrue("block " + b + " runs " + run, run == total / p.blocks() || run == total / p.blocks() + 1);
        }
        assertTrue("80 tiles over 60 blocks split tiles", p.fixupNeeded());
        assertEquals((long) UNITS * 128 * 128, p.fixupElements());
        assertEquals(UNITS * MmqConfig.INT8_TENSOR.workGroupSize(), p.globalWorkSize()[0]);
        assertEquals(MmqConfig.INT8_TENSOR.workGroupSize(), p.localWorkSize()[0]);
        assertEquals(UNITS * MmqConfig.INT8_TENSOR.workGroupSize(), p.fixupGlobalWorkSize()[0]);
        assertEquals("a tile's 128 x 128 outputs over 256-work-item groups of four", 16, p.fixupGlobalWorkSize()[1]);
    }

    /** A partition of whole tiles needs no fixup; one block per chunk splits every tile. */
    @Test
    public void wholeTilePartitionsNeedNoFixup() {
        MmqGemmPlan base = plain(GGMLType.Q4_K, 512, 512, 256);
        assertFalse(StreamKGemmPlan.ofBlocks(base, 8).fixupNeeded());
        assertFalse(StreamKGemmPlan.ofBlocks(base, 4).fixupNeeded());
        assertTrue(StreamKGemmPlan.ofBlocks(base, 5).fixupNeeded());
        assertTrue(StreamKGemmPlan.ofBlocks(base, 8 * 16).fixupNeeded());
        try {
            StreamKGemmPlan.ofBlocks(base, 8 * 16 + 1);
            fail("more blocks than (tile, chunk) steps must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("blocks"));
        }
        try {
            StreamKGemmPlan.ofBlocks(MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 512, 512, 256), 4);
            fail("the packed-dot tile has no stream-k kernel");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("tensor tile"));
        }
    }
}
