package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The large MMA tile's kernel names, and the alignment its shared stages declare. */
public class MmaGemmPlanTest {

    private static final String LARGE_TILE_SOURCE = "kernels/llm/matmul_mma_large.cl";

    @Test
    public void largeTileKernelsAreNamedPerDenseType() {
        assertEquals("F16_MatMul_MMA_L", GemmKernels.matmulMmaLarge(GGMLType.F16));
        assertEquals("BF16_MatMul_MMA_L", GemmKernels.matmulMmaLarge(GGMLType.BF16));
        assertThrows(IllegalArgumentException.class, () -> GemmKernels.matmulMmaLarge(GGMLType.Q4_K));
    }

    /**
     * The large tile writes its shared stages with 16-byte stores and reads them with ldmatrix, whose row addresses
     * must be 16-byte aligned, so both stage arrays declare that alignment rather than inherit the 2 bytes of their
     * element type from wherever the compiler places them.
     */
    @Test
    public void theLargeTilesSharedStagesAreSixteenByteAligned() throws IOException {
        String source = resource(LARGE_TILE_SOURCE);
        Matcher m = Pattern.compile("NF_LOCAL\\s+([^;\\\\]*?)\\b(\\w+)\\s*\\[").matcher(source);
        List<String> arrays = new ArrayList<>();
        while (m.find()) {
            arrays.add(m.group(2));
            assertTrue(m.group(2) + " is declared without __align__(16): " + m.group(),
                    m.group(1).contains("__align__(16)"));
        }
        assertEquals(List.of("a_s", "b_s"), arrays);
    }

    private static String resource(String path) throws IOException {
        try (InputStream in = MmaGemmPlanTest.class.getClassLoader().getResourceAsStream(path)) {
            assertNotNull("missing resource " + path, in);
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
