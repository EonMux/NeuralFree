package com.eonmux.neuralfree.gguf.kernels;

import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Guards the committed IQ table sources: {@code kernels/llm/common/iq_tables.h} and
 * {@code shaders/llm/common/iq_tables.glsl} must equal {@link IqTableSourceGenerator} output for
 * {@code gguf.quant.QuantTables}, and the packed GLSL forms must decode back to the source tables.
 * Runs from the project directory (the Gradle test working directory).
 */
public class IqTableSourcesTest {

    private static final Path RESOURCES = Paths.get("src", "main", "resources");

    /** First words of llama.cpp vulkan-shaders/types.glsl {@code iq1s_grid_const}, an independent anchor. */
    private static final int[] LLAMA_CPP_IQ1S_PACKED_PREFIX = {
        0xfffdffff, 0xfff7fff0, 0xffccfff5, 0xffdfffc0, 0xffd7ffdd, 0xff30ffd5, 0xff03ff0c, 0xff10ff01
    };

    @Test
    public void committedOpenClTablesMatchGenerator() throws IOException {
        assertEquals("kernels/llm/common/iq_tables.h is stale; rerun IqTableSourceGenerator",
                IqTableSourceGenerator.openCl(), readCommitted(IqTableSourceGenerator.OPENCL_RELATIVE_PATH));
    }

    @Test
    public void committedGlslTablesMatchGenerator() throws IOException {
        assertEquals("shaders/llm/common/iq_tables.glsl is stale; rerun IqTableSourceGenerator",
                IqTableSourceGenerator.glsl(), readCommitted(IqTableSourceGenerator.GLSL_RELATIVE_PATH));
    }

    @Test
    public void tablesHaveGgmlLengths() {
        IqTableSourceGenerator.IqGridData t = IqTableSourceGenerator.IqGridData.fromQuantTables();
        assertEquals(128, t.ksignsIq2xs().length);
        assertEquals(256, t.iq3xxsGrid().length);
        assertEquals(512, t.iq3sGrid().length);
        assertEquals(256, t.iq2xxsGrid().length);
        assertEquals(512, t.iq2xsGrid().length);
        assertEquals(1024, t.iq2sGrid().length);
        assertEquals(2048, t.iq1sGrid().length);
    }

    @Test
    public void ksignsAreIndexPlusParityBit() {
        int[] ksigns = IqTableSourceGenerator.IqGridData.fromQuantTables().ksignsIq2xs();
        for (int i = 0; i < ksigns.length; i++) {
            int expected = i | ((Integer.bitCount(i) & 1) << 7);
            assertEquals("ksigns[" + i + "]", expected, ksigns[i]);
        }
        IqTableSourceGenerator.requireParityKsigns(ksigns);
    }

    @Test
    public void nonParityKsignsAreRejected() {
        int[] ksigns = IqTableSourceGenerator.IqGridData.fromQuantTables().ksignsIq2xs();
        ksigns[1] = 1;
        try {
            IqTableSourceGenerator.requireParityKsigns(ksigns);
            fail("a ksigns entry without its parity bit must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("KSIGNS_IQ2XS[1]"));
        }
    }

    @Test
    public void iq1sPackingRoundTrips() {
        long[] grid = IqTableSourceGenerator.IqGridData.fromQuantTables().iq1sGrid();
        int[] packed = IqTableSourceGenerator.packIq1s(grid);
        assertEquals(1024, packed.length);
        for (int e = 0; e < grid.length; e++) {
            int entry = (packed[e >> 1] >>> (16 * (e & 1))) & 0xFFFF;
            for (int j = 0; j < 8; j++) {
                int field = (entry >>> (2 * j)) & 3;
                int decoded = (field << 30) >> 30;
                int expected = (byte) (grid[e] >>> (8 * j));
                assertEquals("iq1s entry " + e + " coordinate " + j, expected, decoded);
            }
        }
    }

    @Test
    public void iq1sPackingMatchesLlamaCppVulkanTable() {
        int[] packed = IqTableSourceGenerator.packIq1s(
                IqTableSourceGenerator.IqGridData.fromQuantTables().iq1sGrid());
        int[] prefix = new int[LLAMA_CPP_IQ1S_PACKED_PREFIX.length];
        System.arraycopy(packed, 0, prefix, 0, prefix.length);
        assertArrayEquals(LLAMA_CPP_IQ1S_PACKED_PREFIX, prefix);
    }

    @Test
    public void openClSourceDeclaresKernelTableNames() {
        String src = IqTableSourceGenerator.openCl();
        assertTrue(src.contains("NF_TABLE(uchar, KSIGNS_IQ2XS_CL, 128) = {"));
        assertTrue(src.contains("NF_TABLE(uint, IQ3XXS_GRID_CL, 256) = {"));
        assertTrue(src.contains("NF_TABLE(uint, IQ3S_GRID_CL, 512) = {"));
        assertTrue(src.contains("NF_TABLE(nf_u64, IQ2XXS_GRID_CL, 256) = {"));
        assertTrue(src.contains("NF_TABLE(nf_u64, IQ2XS_GRID_CL, 512) = {"));
        assertTrue(src.contains("NF_TABLE(nf_u64, IQ2S_GRID_CL, 1024) = {"));
        assertTrue(src.contains("NF_TABLE(nf_u64, IQ1S_GRID_CL, 2048) = {"));
    }

    @Test
    public void constructorRejectsWrongLength() {
        IqTableSourceGenerator.IqGridData t = IqTableSourceGenerator.IqGridData.fromQuantTables();
        try {
            new IqTableSourceGenerator.IqGridData(new int[127], t.iq3xxsGrid(), t.iq3sGrid(), t.iq2xxsGrid(),
                    t.iq2xsGrid(), t.iq2sGrid(), t.iq1sGrid());
            fail("a 127-entry ksigns table must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("KSIGNS_IQ2XS"));
        }
    }

    @Test
    public void constructorRejectsNonTernaryIq1sGrid() {
        IqTableSourceGenerator.IqGridData t = IqTableSourceGenerator.IqGridData.fromQuantTables();
        long[] iq1s = t.iq1sGrid();
        iq1s[5] = 0x0000000000000002L;
        try {
            new IqTableSourceGenerator.IqGridData(t.ksignsIq2xs(), t.iq3xxsGrid(), t.iq3sGrid(), t.iq2xxsGrid(),
                    t.iq2xsGrid(), t.iq2sGrid(), iq1s);
            fail("an IQ1_S coordinate of 2 must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("iq1s grid entry 5"));
        }
    }

    private static String readCommitted(String relativePath) throws IOException {
        Path path = RESOURCES.resolve(relativePath);
        if (!Files.isRegularFile(path)) {
            fail(path.toAbsolutePath() + " is missing; run IqTableSourceGenerator");
        }
        return new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
    }
}
