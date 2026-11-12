package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The stream-k plan of the CUDA tensor tiles: which launches leave the plain grid (llama.cpp's 90% wave
 * fill), the even partition of the (tile, chunk) sequence the kernels compute, the contributors of a split
 * tile whose planes the fixup adds, and the geometry and arguments of both launches of each form.
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
        assertTrue("80 tiles over 60 blocks split tiles", p.splitsTiles());
        assertEquals("one plane per block", 1, p.planesPerBlock());
        assertEquals((long) UNITS * 128 * 128, p.fixupElements());
        assertEquals(GemmKernels.STREAM_K_FIXUP, p.fixupKernelName());
        assertEquals(UNITS * MmqConfig.INT8_TENSOR.workGroupSize(), p.globalWorkSize()[0]);
        assertEquals(MmqConfig.INT8_TENSOR.workGroupSize(), p.localWorkSize()[0]);
        assertEquals("the fixup runs every block's work-groups", p.globalWorkSize()[0], p.fixupGlobalWorkSize()[0]);
        assertEquals("128 x 128 outputs in slices of 4 x 256", 16, p.fixupGlobalWorkSize()[1]);
    }

    /**
     * The owner formula the kernel uses to count a split tile's contributors agrees with a scan of the block
     * starts for every step, and a tile's contributors are the blocks whose runs meet it.
     */
    @Test
    public void ownersAndContributorsFollowTheBlockStarts() {
        int[][] shapes = {{2560, 9728, 512}, {9728, 2560, 512}, {1024, 1024, 512}, {333, 160, 141}};
        for (int[] shape : shapes) {
            MmqGemmPlan base = plain(GGMLType.Q8_0, shape[0], shape[1], shape[2]);
            long steps = (long) base.rowTiles() * base.tokenTiles() * (base.k() / 32);
            for (int blocks : new int[]{1, 5, 7, 13, UNITS, 997, (int) steps}) {
                if (blocks > steps) {
                    continue;
                }
                StreamKGemmPlan p = StreamKGemmPlan.ofBlocks(base, blocks);
                int b = 0;
                for (long step = 0; step < steps; step++) {
                    while (p.blockStart(b + 1) <= step) {
                        b++;
                    }
                    assertEquals(p + " step " + step, b, p.ownerOf(step));
                }
                for (long tile = 0; tile < p.tiles(); tile++) {
                    long first = tile * p.chunksPerTile();
                    long last = first + p.chunksPerTile();
                    int meet = 0;
                    for (int q = 0; q < blocks; q++) {
                        if (p.blockStart(q) < last && p.blockStart(q + 1) > first) {
                            meet++;
                        }
                    }
                    assertEquals(p + " tile " + tile, meet, p.contributors(tile));
                }
            }
        }
    }

    /**
     * The gate/up form tiles the output in half-height (64-row) tiles of the 128-row tensor tile, always runs over
     * the resident blocks (no plain grid to keep, even at full waves), and keeps a head and a tail plane per block
     * for its own fixup, whose slices cover the 64 x 128 outputs of a tile.
     */
    @Test
    public void theGateUpFormRunsHalfHeightTilesOverTheResidentBlocks() {
        StreamKGemmPlan p = StreamKGemmPlan.ofGateUp(plain(GGMLType.Q8_0, 9728, 2560, 512), UNITS);
        assertNotNull(p);
        assertTrue(p.gateUp());
        assertEquals("Q8_0_MatMul_MMQ_StreamK_GateUp", p.kernelName());
        assertEquals("152 output row tiles of 64 rows by 4 token tiles", 608, p.tiles());
        assertEquals(UNITS, p.blocks());
        assertEquals(2, p.planesPerBlock());
        assertEquals((long) UNITS * 2 * 128 * 128, p.fixupElements());
        assertEquals(GemmKernels.STREAM_K_FIXUP_GATE_UP, p.fixupKernelName());
        assertEquals(UNITS * MmqConfig.INT8_TENSOR.workGroupSize(), p.fixupGlobalWorkSize()[0]);
        assertEquals("64 x 128 outputs in slices of 4 x 256", 8, p.fixupGlobalWorkSize()[1]);
        assertTrue("608 tiles over 60 blocks split tiles", p.splitsTiles());
        StreamKGemmPlan full = StreamKGemmPlan.ofGateUp(plain(GGMLType.Q4_K, 64 * 15, 1024, 512), UNITS);
        assertNotNull("60 tiles, one full wave, still take the gate/up form", full);
        assertFalse(full.splitsTiles());
        assertEquals(333 / 64 + 1, StreamKGemmPlan.ofGateUpBlocks(plain(GGMLType.Q8_0, 333, 160, 100), 1).tiles());
        assertNull(StreamKGemmPlan.ofGateUp(plain(GGMLType.Q8_0, 9728, 2560, 512),
                DeviceCapabilities.UNKNOWN_COMPUTE_UNITS));
        assertNull(StreamKGemmPlan.ofGateUp(MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, GGMLType.Q8_0, 9728, 2560,
                512), UNITS));
        assertFalse(StreamKGemmPlan.of(plain(GGMLType.Q8_0, 2560, 9728, 512), UNITS).gateUp());
    }

    /** Each form takes its own argument list: the gate/up form appends the up weight, in its own buffer; both
     *  fixups take the launch's shape and output view, the planes and the output. */
    @Test
    public void eachFormTakesItsOwnArguments() {
        StreamKGemmPlan gateUp = StreamKGemmPlan.ofGateUpBlocks(plain(GGMLType.Q8_0, 256, 64, 128), 4);
        StreamKGemmPlan plainSk = StreamKGemmPlan.ofBlocks(plain(GGMLType.Q8_0, 256, 64, 128), 2);
        MmqGemmPlan base = gateUp.base();
        FakeGpuBuffer weight = new FakeGpuBuffer(1, base.weightBytes(), true);
        FakeGpuBuffer up = new FakeGpuBuffer(2, base.weightBytes(), true);
        FakeGpuBuffer q8 = new FakeGpuBuffer(3, base.q8ElementsRequired() * Integer.BYTES, false);
        FakeGpuBuffer y = new FakeGpuBuffer(4, base.yElementsRequired() * Float.BYTES, false);
        FakeGpuBuffer fixup = new FakeGpuBuffer(5, gateUp.fixupElements() * Float.BYTES, false);
        Object[] args = gateUp.argsGateUp(weight, up, q8, y, fixup);
        assertEquals(StreamKGemmPlan.GATE_UP_ARG_COUNT, args.length);
        assertTrue(args[StreamKGemmPlan.ARG_WEIGHT_UP] == up);
        assertTrue(args[GemmKernels.ARG_WEIGHT] == weight);
        assertTrue(args[StreamKGemmPlan.ARG_FIXUP] == fixup);
        Object[] plainArgs = plainSk.args(weight, q8, y, fixup);
        assertEquals(StreamKGemmPlan.ARG_COUNT, plainArgs.length);
        assertTrue(plainArgs[StreamKGemmPlan.ARG_FIXUP] == fixup);
        for (StreamKGemmPlan p : new StreamKGemmPlan[]{gateUp, plainSk}) {
            Object[] fix = p.fixupArgs(fixup, y);
            assertEquals(StreamKGemmPlan.FIXUP_ARG_COUNT, fix.length);
            assertEquals(256, fix[StreamKGemmPlan.FIXUP_ARG_M]);
            assertEquals(64, fix[StreamKGemmPlan.FIXUP_ARG_K]);
            assertEquals(128, fix[StreamKGemmPlan.FIXUP_ARG_N]);
            assertEquals(256, fix[StreamKGemmPlan.FIXUP_ARG_Y_STRIDE]);
            assertEquals(0, fix[StreamKGemmPlan.FIXUP_ARG_Y_OFF]);
            assertTrue(fix[StreamKGemmPlan.FIXUP_ARG_FIXUP] == fixup);
            assertTrue(fix[StreamKGemmPlan.FIXUP_ARG_Y] == y);
        }
        try {
            gateUp.fixupArgs(new FakeGpuBuffer(8, plainSk.fixupElements() * Float.BYTES, false), y);
            fail("the gate/up fixup reads two planes per block");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("fixup buffer holds"));
        }
        try {
            plainSk.args(weight, q8, y, y);
            fail("the planes need their own buffer");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("own buffer"));
        }
        try {
            gateUp.args(weight, q8, y, fixup);
            fail("the gate/up form needs the up weight");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("argsGateUp"));
        }
        try {
            plainSk.argsGateUp(weight, up, q8, y, fixup);
            fail("the plain form has no up weight");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("no up weight"));
        }
        try {
            gateUp.argsGateUp(weight, weight, q8, y, fixup);
            fail("the up weight needs its own buffer");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("own buffer"));
        }
        try {
            gateUp.argsGateUp(weight, new FakeGpuBuffer(7, 64, true), q8, y, fixup);
            fail("an up weight smaller than the gate must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("less than the gate"));
        }
    }

    /**
     * The small-token tensor tile (128 rows x 16 tokens) plans its own stream-k kernel and fixup over one resident
     * block per unit ({@link StreamKGemmPlan#SMALL_TILE_BLOCKS_PER_UNIT}: a second block of its kernels does not fit
     * a multiprocessor's registers), takes one block per tile when the tile grid's waves are full, keeps the twin for
     * a grid of fewer tiles than a quarter of the units (up to 14 tiles on 60), has no gate/up form, and is never
     * planned by the device tile's factories.
     */
    @Test
    public void theSmallTokenTilePlansItsOwnKernelsOverItsResidentBlocks() {
        int resident = UNITS * StreamKGemmPlan.SMALL_TILE_BLOCKS_PER_UNIT;
        assertEquals("one resident block per unit", UNITS, resident);
        MmqGemmPlan gate = MmqGemmPlan.tensorSmallOf(GGMLType.Q8_0, 9728, 2560, 16, 80, 0, 9728, 0);
        assertEquals(MmqConfig.INT8_TENSOR_SMALL, gate.config());
        assertEquals("Q8_0_MatMul_MMQ_StreamK_S", gate.kernelName());
        assertEquals(76, gate.rowTiles());
        assertEquals(1, gate.tokenTiles());
        StreamKGemmPlan sk = StreamKGemmPlan.ofSmall(gate, UNITS);
        assertNotNull(sk);
        assertEquals("Q8_0_MatMul_MMQ_StreamK_S", sk.kernelName());
        assertEquals(GemmKernels.STREAM_K_FIXUP_SMALL, sk.fixupKernelName());
        assertEquals(UNITS, sk.blocks());
        assertTrue(sk.splitsTiles());
        assertEquals((long) sk.blocks() * 128 * 16, sk.fixupElements());
        assertEquals("128 x 16 outputs in slices of 4 x 256", 2, sk.fixupGlobalWorkSize()[1]);
        assertEquals(256, sk.localWorkSize()[0]);
        MmqGemmPlan wide = MmqGemmPlan.tensorSmallOf(GGMLType.Q4_K, 128 * resident, 1024, 16, 32, 0,
                128 * resident, 0);
        StreamKGemmPlan full = StreamKGemmPlan.ofSmall(wide, UNITS);
        assertEquals("a full wave runs one block per tile", resident, full.blocks());
        assertFalse(full.splitsTiles());
        assertNull(StreamKGemmPlan.ofSmall(gate, DeviceCapabilities.UNKNOWN_COMPUTE_UNITS));
        assertNull("8 tiles on 60 units keep the twin",
                StreamKGemmPlan.ofSmall(MmqGemmPlan.tensorSmallOf(GGMLType.Q8_0, 1024, 2560, 16, 80, 0, 1024, 0),
                        UNITS));
        assertNull("14 tiles on 60 units keep the twin",
                StreamKGemmPlan.ofSmall(MmqGemmPlan.tensorSmallOf(GGMLType.Q8_0, 1792, 2560, 16, 80, 0, 1792, 0),
                        UNITS));
        assertNotNull("15 tiles on 60 units take the stream-k form",
                StreamKGemmPlan.ofSmall(MmqGemmPlan.tensorSmallOf(GGMLType.Q8_0, 1920, 2560, 16, 80, 0, 1920, 0),
                        UNITS));
        assertNull("the device tile's factory plans the device tile only", StreamKGemmPlan.of(gate, UNITS));
        assertNull("the small tile has no gate/up form", StreamKGemmPlan.ofGateUp(gate, UNITS));
        try {
            StreamKGemmPlan.ofGateUpBlocks(gate, 4);
            fail("the small tile has no gate/up kernel");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("gate/up"));
        }
        try {
            StreamKGemmPlan.ofSmall(plain(GGMLType.Q8_0, 9728, 2560, 16), UNITS);
            fail("ofSmall plans the small tile only");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("small-token"));
        }
        try {
            MmqGemmPlan.tensorSmallOf(GGMLType.F8_E4M3, 128, 256, 16, 8, 0, 128, 0);
            fail("the small tensor tile runs the int8 types");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("int8"));
        }
    }

    /** A partition of whole tiles splits none; one block per chunk splits every tile. */
    @Test
    public void wholeTilePartitionsSplitNoTile() {
        MmqGemmPlan base = plain(GGMLType.Q4_K, 512, 512, 256);
        assertFalse(StreamKGemmPlan.ofBlocks(base, 8).splitsTiles());
        assertFalse(StreamKGemmPlan.ofBlocks(base, 4).splitsTiles());
        assertTrue(StreamKGemmPlan.ofBlocks(base, 5).splitsTiles());
        assertTrue(StreamKGemmPlan.ofBlocks(base, 8 * 16).splitsTiles());
        assertEquals(16, StreamKGemmPlan.ofBlocks(base, 8 * 16).contributors(3));
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
