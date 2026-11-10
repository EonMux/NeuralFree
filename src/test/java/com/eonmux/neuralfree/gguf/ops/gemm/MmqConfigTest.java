package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.hip.HipGfxTarget;
import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.After;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The MMQ tile configs: the device selection (CUDA compute capability, the HIP gfx families, the
 * OpenCL override), the tile geometry the kernel ABI derives from them, and the kernel header's HIP
 * family blocks against {@link HipGfxTarget}. No GPU is used.
 */
public class MmqConfigTest {

    private static final String CONFIG_HEADER = "kernels/llm/common/matmul_mmq_config.h";

    @After
    public void clearOverride() {
        System.clearProperty(MmqConfig.TILE_PROPERTY);
    }

    @Test
    public void cudaOpenClAndVulkanKeepTheirExistingTiles() {
        assertEquals(MmqConfig.INT8_TENSOR, MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, 80));
        assertEquals(MmqConfig.INT8_TENSOR, MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, 120));
        assertEquals(MmqConfig.PACKED_DOT, MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, 75));
        assertEquals(MmqConfig.PACKED_DOT, MmqConfig.forDevice(ComputeBackend.BackendType.OPENCL, 120));
        assertEquals(MmqConfig.PACKED_DOT, MmqConfig.forDevice(ComputeBackend.BackendType.VULKAN, 0));
    }

    @Test
    public void hipSelectsTheTileFromTheGfxFamily() {
        assertTile("gfx1100", MmqConfig.RDNA3);
        assertTile("gfx1101", MmqConfig.RDNA3);
        assertTile("gfx1102", MmqConfig.RDNA3);
        assertTile("gfx1103", MmqConfig.RDNA3);
        assertTile("gfx908", MmqConfig.CDNA);
        assertTile("gfx90a", MmqConfig.CDNA);
        assertTile("gfx942", MmqConfig.CDNA);
        assertTile("gfx950", MmqConfig.CDNA);
        assertTile("gfx1250", MmqConfig.CDNA);
        /* The families whose llama.cpp tables are not ported keep the packed-dot tile. */
        assertTile("gfx906", MmqConfig.PACKED_DOT);
        assertTile("gfx909", MmqConfig.PACKED_DOT);
        assertTile("gfx90c", MmqConfig.PACKED_DOT);
        assertTile("gfx1010", MmqConfig.PACKED_DOT);
        assertTile("gfx1030", MmqConfig.PACKED_DOT);
        assertTile("gfx1150", MmqConfig.PACKED_DOT);
        assertTile("gfx1170", MmqConfig.PACKED_DOT);
        assertTile("gfx1200", MmqConfig.PACKED_DOT);
        assertTile("gfx1201", MmqConfig.PACKED_DOT);
    }

    private static void assertTile(String gfx, MmqConfig expected) {
        HipGfxTarget target = HipGfxTarget.parse(gfx);
        assertEquals(gfx + " target", expected, MmqConfig.forHipTarget(target));
        assertEquals(gfx + " computed capability", expected,
                MmqConfig.forDevice(ComputeBackend.BackendType.HIP, target.amdComputeCapability()));
    }

    @Test(expected = IllegalArgumentException.class)
    public void hipComputeCapabilityWithoutTheAmdRangeIsRejected() {
        MmqConfig.forDevice(ComputeBackend.BackendType.HIP, 120);
    }

    @Test
    public void rdna3AndCdnaPortTheLlamaCppTableTiles() {
        /* llama.cpp mmq-config-rdna3.cuh: I 64, J 128, nthreads 128, K_vram 256, non-fallback rows. */
        assertEquals(16, MmqConfig.RDNA3.stageDepth());
        assertEquals(64, MmqConfig.RDNA3.tileRows());
        assertEquals(128, MmqConfig.RDNA3.tileTokens());
        assertEquals(256, MmqConfig.RDNA3.workGroupSize());
        /* llama.cpp mmq-config-cdna.cuh: I 128, J 64, K_vram 256; eight rows per work-item. */
        assertEquals(16, MmqConfig.CDNA.stageDepth());
        assertEquals(128, MmqConfig.CDNA.tileRows());
        assertEquals(64, MmqConfig.CDNA.tileTokens());
        assertEquals(8, MmqConfig.CDNA.rowsPerWorkItem());
        assertEquals(256, MmqConfig.CDNA.workGroupSize());
        /* Every family max is the widest tile of its table, so the ported config is a table row. */
        assertEquals(MmqConfig.MAX_TILE_TOKENS, MmqConfig.RDNA3.tileTokens());
        assertEquals(MmqConfig.MAX_TILE_ROWS, MmqConfig.CDNA.tileRows());
        assertEquals(MmqConfig.MAX_STAGE_DEPTH, MmqConfig.INT8_TENSOR.stageDepth());
    }

    @Test
    public void theExistingConfigsKeepTheirGeometry() {
        assertEquals(256, MmqConfig.PACKED_DOT.workGroupSize());
        assertEquals(64, MmqConfig.PACKED_DOT.tileRows());
        assertEquals(16, MmqConfig.PACKED_DOT.tileTokens());
        /* llama.cpp's widest table row: I 128, J 128 over its 256-thread block, eight rows per
         * work-item (eight warps of two m16 minitiles); the twins keep four rows per work-item. */
        assertEquals(256, MmqConfig.INT8_TENSOR.workGroupSize());
        assertEquals(128, MmqConfig.INT8_TENSOR.tileRows());
        assertEquals(128, MmqConfig.INT8_TENSOR.tileTokens());
        assertEquals(8, MmqConfig.INT8_TENSOR.rowsPerWorkItem());
        assertEquals(4, MmqConfig.INT8_TENSOR.twinRowsPerWorkItem());
        assertEquals("a tile's twins take its rows per work-item unless it pins them",
                MmqConfig.CDNA.rowsPerWorkItem(), MmqConfig.CDNA.twinRowsPerWorkItem());
        assertEquals(4 + 2 + 1, MmqConfig.PACKED_DOT.recordUnits());
        assertEquals(8 + 2 + 2, MmqConfig.INT8_TENSOR.recordUnits());
        assertEquals(4, MmqConfig.PACKED_DOT.rowRecords(64));
        assertEquals(256, MmqConfig.PACKED_DOT.rowRecords(4096));
        assertEquals(128, MmqConfig.INT8_TENSOR.rowRecords(4096));
        assertEquals(4L * (3 * 128 * 16 + 9 * 128 + 64 * (4 + 2 + 1)), MmqConfig.CDNA.sharedBytes());
    }

    @Test
    public void kernelDefinesLeaveTheDefaultTilesAlone() {
        assertEquals("", MmqConfig.PACKED_DOT.kernelDefines(false));
        assertEquals("", MmqConfig.INT8_TENSOR.kernelDefines(true));
        assertEquals("", MmqConfig.PACKED_DOT.kernelKey(false));
        assertEquals("-DNF_MMQ_TM=64 -DNF_MMQ_TN=128 -DNF_MMQ_TK=16", MmqConfig.RDNA3.kernelDefines(false));
        assertEquals("-DNF_MMQ_TM=128 -DNF_MMQ_TN=64 -DNF_MMQ_TK=16 -DNF_MMQ_RM=8",
                MmqConfig.CDNA.kernelDefines(false));
        assertEquals("-DNF_MMQ_TM=128 -DNF_MMQ_TN=128 -DNF_MMQ_TK=32 -DNF_MMQ_RM=8 -DNF_MMQ_TWIN_RM=4",
                MmqConfig.INT8_TENSOR.kernelDefines(false));
        assertEquals("@int8-tensor", MmqConfig.INT8_TENSOR.kernelKey(false));
        assertEquals("@cdna", MmqConfig.CDNA.kernelKey(false));
    }

    @Test
    public void aForcedTileAppliesToTheRuntimeCompiledOpenClKernels() {
        assertNull(MmqConfig.tileOverride());
        System.setProperty(MmqConfig.TILE_PROPERTY, "CDNA");
        assertEquals(MmqConfig.CDNA, MmqConfig.forDevice(ComputeBackend.BackendType.OPENCL, 0));
        System.setProperty(MmqConfig.TILE_PROPERTY, "packed_dot");
        assertEquals(MmqConfig.PACKED_DOT, MmqConfig.forDevice(ComputeBackend.BackendType.OPENCL, 0));
        System.setProperty(MmqConfig.TILE_PROPERTY, "rdna3");
        assertEquals(MmqConfig.RDNA3, MmqConfig.forDevice(ComputeBackend.BackendType.OPENCL, 0));
        /* Build-time tiles of CUDA and HIP are not affected by the runtime override. */
        assertEquals(MmqConfig.INT8_TENSOR, MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, 120));
        assertEquals(MmqConfig.RDNA3, MmqConfig.forDevice(ComputeBackend.BackendType.HIP,
                HipGfxTarget.parse("gfx1100").amdComputeCapability()));
    }

    @Test
    public void anUnknownForcedTileIsRefusedByName() {
        System.setProperty(MmqConfig.TILE_PROPERTY, "ampere");
        try {
            MmqConfig.forDevice(ComputeBackend.BackendType.OPENCL, 0);
            fail("an unknown tile must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.mmq.tile"));
            assertTrue(e.getMessage(), e.getMessage().contains("ampere"));
            assertTrue(e.getMessage(), e.getMessage().contains("cdna"));
        }
    }

    @Test
    public void tileSizesAreValidatedAtTheBoundary() {
        assertEquals(MmqConfig.CDNA, MmqConfig.of(16, 128, 64, 8));
        assertEquals(4, MmqConfig.of(16, 128, 64).rowsPerWorkItem());
        expectInvalid("stage depth", () -> MmqConfig.of(8, 64, 16));
        expectInvalid("rows per work-item", () -> MmqConfig.of(16, 64, 16, 0));
        expectInvalid("rows per work-item", () -> MmqConfig.of(16, 64, 16, 12));
        expectInvalid("row tile", () -> MmqConfig.of(16, 63, 16));
        expectInvalid("row tile", () -> MmqConfig.of(16, 256, 16));
        expectInvalid("row tile", () -> MmqConfig.of(16, 130, 16, 8));
        expectInvalid("token tile", () -> MmqConfig.of(16, 64, 17));
        expectInvalid("token tile", () -> MmqConfig.of(16, 64, 256));
        expectInvalid("rows per work-item", () -> MmqConfig.CDNA.kernelDefines(true));
        expectInvalid("rows per work-item", () -> MmqConfig.of(32, 128, 128, 8).kernelDefines(true));
    }

    /** The small-token-tile twin keeps the device tile's stage depth over 16 tokens, with a row tile
     * whose workgroup stays at the twin's 256 items (the 128-row tensor tile's twin runs 64 rows);
     * the 64-token twin shares the row tile and workgroup. */
    @Test
    public void theSmallTileTwinKeepsTheDeviceRowsAndStageDepthOver16Tokens() {
        assertEquals("the 128-row tensor tile's twin keeps the 64-row, four-rows-per-work-item tile",
                MmqConfig.of(32, 64, 16, 4), MmqConfig.smallTile(MmqConfig.INT8_TENSOR));
        assertEquals("the CDNA tile's 8-row work items already fit the twin workgroup",
                MmqConfig.of(16, 128, 16, 8), MmqConfig.smallTile(MmqConfig.CDNA));
        assertEquals(256, MmqConfig.smallTile(MmqConfig.INT8_TENSOR).workGroupSize());
        assertEquals(64, MmqConfig.smallTile(MmqConfig.RDNA3).tileRows());
        expectInvalid("no 16-token twin", () -> MmqConfig.smallTile(MmqConfig.PACKED_DOT));
        try {
            MmqConfig.smallTile(null);
            fail("a null device tile must be refused");
        } catch (NullPointerException e) {
            assertEquals("device", e.getMessage());
        }
    }

    /** The twin ladder: 16 tokens below the counted-matvec bound, the 64-token twin over 17..64
     *  (the measured mid batches whose 64-token tile beats the 128-token device tile: n=48 head
     *  2063 vs 2923us), the plain tile above — except a small-K reduction (the Qwen3 0.6B layers,
     *  K 1024/2048: pp24 measured 2701 t/s on the 16-token twin against 2171 on the 64-token one),
     *  whose launches up to 48 tokens keep the 16-token twin. */
    @Test
    public void theTwinTokensFollowTheLadderOfTheLaunch() {
        assertEquals(16, MmqConfig.twinTokens(1, 2560));
        assertEquals(16, MmqConfig.twinTokens(16, 1024));
        assertEquals(64, MmqConfig.twinTokens(17, 2560));
        assertEquals(64, MmqConfig.twinTokens(48, 2560));
        assertEquals(64, MmqConfig.twinTokens(64, 2560));
        assertEquals("above 64 the plain tile serves", 64, MmqConfig.twinTokens(65, 2560));
        assertEquals("a small-K launch keeps the 16-token twin up to 48 tokens",
                16, MmqConfig.twinTokens(17, SplitKGemmPlan.MIN_SPLIT_K - 1));
        assertEquals(16, MmqConfig.twinTokens(48, 1024));
        assertEquals("a small-K launch past 48 tokens takes the 64-token twin",
                64, MmqConfig.twinTokens(49, 1024));
        assertEquals("the K threshold itself takes the wide twin",
                64, MmqConfig.twinTokens(48, SplitKGemmPlan.MIN_SPLIT_K));
        assertEquals("the 64-token twin is the device tile's stage depth over 64 tokens",
                MmqConfig.of(32, 64, 64, 4),
                MmqConfig.smallTile(MmqConfig.INT8_TENSOR, MmqConfig.TWIN_MAX_TOKENS));
        assertEquals("the twin widths share the 256-item workgroup",
                MmqConfig.smallTile(MmqConfig.INT8_TENSOR).workGroupSize(),
                MmqConfig.smallTile(MmqConfig.INT8_TENSOR, MmqConfig.TWIN_MAX_TOKENS).workGroupSize());
        expectInvalid("no 64-token twin",
                () -> MmqConfig.smallTile(MmqConfig.CDNA, MmqConfig.TWIN_MAX_TOKENS));
        expectInvalid("twin tiles come in", () -> MmqConfig.smallTile(MmqConfig.INT8_TENSOR, 32));
    }

    @Test
    public void mmqPlansFollowTheTileOfTheirConfig() {
        MmqGemmPlan rdna3 = MmqGemmPlan.contiguous(MmqConfig.RDNA3, GGMLType.Q4_K, 130, 256, 130);
        assertEquals(3, rdna3.rowTiles());
        assertEquals(2, rdna3.tokenTiles());
        assertEquals(256, rdna3.localWorkSize()[0]);
        assertEquals(3L * 256, rdna3.globalWorkSize()[0]);
        MmqGemmPlan cdna = MmqGemmPlan.contiguous(MmqConfig.CDNA, GGMLType.Q4_K, 130, 256, 130);
        assertEquals(2, cdna.rowTiles());
        assertEquals(3, cdna.tokenTiles());
        assertEquals(256, cdna.localWorkSize()[0]);
        assertEquals(2L * 256, cdna.globalWorkSize()[0]);
    }

    /**
     * The HIP family blocks of the kernel header select the same gfx targets, and the same tile
     * numbers, as {@link HipGfxTarget} and this class: the host and the prebuilt code object must
     * agree on who gets which tile.
     */
    @Test
    public void kernelHeaderHipFamiliesMirrorTheHostSelection() throws IOException {
        String header = read(CONFIG_HEADER);
        Set<String> cdna = gfxMacrosIn(header, "128", "64");
        Set<String> rdna3 = gfxMacrosIn(header, "64", "128");
        assertFalse("the header lists CDNA gfx macros", cdna.isEmpty());
        assertFalse("the header lists RDNA3 gfx macros", rdna3.isEmpty());
        for (String gfx : new String[]{"gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950", "gfx1250",
                "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx906", "gfx909", "gfx90c", "gfx1010",
                "gfx1030", "gfx1150", "gfx1170", "gfx1200", "gfx1201"}) {
            boolean headerFamily = cdna.contains("__" + gfx + "__") || rdna3.contains("__" + gfx + "__");
            MmqConfig host = MmqConfig.forHipTarget(HipGfxTarget.parse(gfx));
            boolean hostTiled = host != MmqConfig.PACKED_DOT;
            assertEquals(gfx + ": the kernel header and the host agree on the family tile",
                    hostTiled, headerFamily);
            if (headerFamily) {
                assertEquals(gfx + " tile", cdna.contains("__" + gfx + "__") ? MmqConfig.CDNA : MmqConfig.RDNA3,
                        host);
            }
        }
    }

    /**
     * The gfx macros of the header's HIP block that selects a {@code TM} x {@code TN} tile: the macro
     * tokens between the block's {@code #if} and its defines.
     */
    private static Set<String> gfxMacrosIn(String header, String rows, String tokens) {
        Matcher m = Pattern.compile("(?s)#(?:if|elif) defined\\(__gfx[^\\n]*\\n(?:[^\\n]*\\n)*?"
                + "#define NF_MMQ_TM " + rows + "\\n#define NF_MMQ_TN " + tokens + "\\n").matcher(header);
        Set<String> found = new LinkedHashSet<>();
        while (m.find()) {
            Matcher gfx = Pattern.compile("__gfx[0-9a-f]+__").matcher(m.group());
            while (gfx.find()) {
                found.add(gfx.group());
            }
        }
        return found;
    }

    private static void expectInvalid(String message, Runnable action) {
        try {
            action.run();
            fail("expected IllegalArgumentException mentioning " + message);
        } catch (IllegalArgumentException e) {
            assertNotNull(e.getMessage());
            assertTrue(e.getMessage(), e.getMessage().contains(message));
        }
    }

    private static String read(String resource) throws IOException {
        ClassLoader cl = MmqConfigTest.class.getClassLoader();
        try (InputStream in = cl.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IOException("missing classpath resource " + resource);
            }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int n;
            while ((n = in.read(buffer)) > 0) {
                out.write(buffer, 0, n);
            }
            return new String(out.toByteArray(), StandardCharsets.UTF_8);
        }
    }
}
