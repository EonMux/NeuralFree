package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The warp-row matvec plan ({@link WarpRowMatvec}): the twin names of the three single-token integer-dot forms, the
 * backend that builds them, the warps per work-group it picks per type and shape (and the shapes it leaves to the
 * family), the launch geometry of one warp per row, and the forms' packed dots going through the dialect's
 * {@code nf_dp4a}, whose scalar fallback keeps the source building for every CUDA architecture the build may name.
 */
public class WarpRowMatvecTest {

    @Test
    public void twinNamesInsertRowBeforeTheQ8SuffixOfTheThreeSingleTokenForms() {
        assertEquals("Q8_0_MatVecRow_Q8", WarpRowMatvec.twinName("Q8_0_MatVec_Q8", GGMLType.Q8_0));
        assertEquals("Q4_K_MatVecAddRow_Q8", WarpRowMatvec.twinName("Q4_K_MatVecAdd_Q8", GGMLType.Q4_K));
        assertEquals("Q6_K_MatVecFusedGateUpRow_Q8",
                WarpRowMatvec.twinName("Q6_K_MatVecFusedGateUp_Q8", GGMLType.Q6_K));
        assertNull("the paired form has no twin", WarpRowMatvec.twinName("Q4_K_MatVec2_Q8", GGMLType.Q4_K));
        assertNull("the float form has no twin", WarpRowMatvec.twinName("Q4_K_MatVec", GGMLType.Q4_K));
        assertNull("another type's kernel", WarpRowMatvec.twinName("Q8_0_MatVec_Q8", GGMLType.Q4_K));
        assertNull("a type without forms", WarpRowMatvec.twinName("Q5_K_MatVec_Q8", GGMLType.Q5_K));
        assertNull(WarpRowMatvec.twinName(null, GGMLType.Q8_0));
    }

    @Test
    public void onlyTheCudaBackendBuildsTheForms() {
        for (ComputeBackend.BackendType type : ComputeBackend.BackendType.values()) {
            assertEquals(type.toString(), type == ComputeBackend.BackendType.CUDA, WarpRowMatvec.served(type));
        }
    }

    /** The measured choices: Q4_K two warps, Q6_K one (the family on a wide head), Q8_0 narrow rows only. */
    @Test
    public void warpsFollowTheMeasuredTypeAndShape() {
        int head = WarpRowMatvec.HEAD_ROWS;
        assertEquals(2, WarpRowMatvec.warps(GGMLType.Q4_K, 9728, 2560));
        assertEquals(2, WarpRowMatvec.warps(GGMLType.Q4_K, 151669, 2560));
        assertEquals(1, WarpRowMatvec.warps(GGMLType.Q6_K, 2560, 9728));
        assertEquals(1, WarpRowMatvec.warps(GGMLType.Q6_K, 151669, 1024));
        assertEquals(0, WarpRowMatvec.warps(GGMLType.Q6_K, 151669, 2560));
        assertEquals(0, WarpRowMatvec.warps(GGMLType.Q6_K, head, 1280));
        assertEquals(1, WarpRowMatvec.warps(GGMLType.Q6_K, head - 1, 1280));
        assertEquals(1, WarpRowMatvec.warps(GGMLType.Q8_0, 3072, 1024));
        assertEquals(4, WarpRowMatvec.warps(GGMLType.Q8_0, 151669, 1024));
        assertEquals(0, WarpRowMatvec.warps(GGMLType.Q8_0, 9728, 2560));
        assertEquals(0, WarpRowMatvec.warps(GGMLType.Q8_0, 151669, 2560));
    }

    @Test
    public void shapesWithoutWholeBlocksOrFormsKeepTheFamily() {
        assertEquals(0, WarpRowMatvec.warps(GGMLType.Q4_K, 1024, 1000));
        assertEquals(0, WarpRowMatvec.warps(GGMLType.Q8_0, 1024, 48));
        assertEquals(0, WarpRowMatvec.warps(GGMLType.Q5_K, 1024, 1024));
        assertEquals(0, WarpRowMatvec.warps(GGMLType.F16, 1024, 1024));
        assertEquals(0, WarpRowMatvec.warps(GGMLType.Q4_K, 0, 1024));
        assertFalse(WarpRowMatvec.TYPES.contains(GGMLType.Q5_K));
    }

    @Test
    public void aLaunchRunsOneWarpPerRowInWholeWorkGroups() {
        assertEquals(32, WarpRowMatvec.localSize(1));
        assertEquals(128, WarpRowMatvec.localSize(4));
        assertEquals(13 * 32, WarpRowMatvec.globalSize(13, 1));
        assertEquals(7 * 64, WarpRowMatvec.globalSize(13, 2));
        assertEquals(4 * 128, WarpRowMatvec.globalSize(13, 4));
        assertEquals(151669L / 4 * 128 + 128, WarpRowMatvec.globalSize(151669, 4));
        assertTrue(WarpRowMatvec.globalSize(1, 2) == 64);
        assertThrows(IllegalArgumentException.class, () -> WarpRowMatvec.globalSize(0, 1));
        assertThrows(IllegalArgumentException.class, () -> WarpRowMatvec.localSize(3));
        assertThrows(IllegalArgumentException.class, () -> WarpRowMatvec.localSize(8));
    }

    /** {@code __dp4a} exists only from sm_61, so the forms call the dialect's {@code nf_dp4a}, which falls back below. */
    @Test
    public void theFormsTakeTheirPackedDotFromTheDialect() throws IOException {
        String source;
        try (InputStream in = WarpRowMatvecTest.class.getResourceAsStream("/kernels/llm/matvec_row.cl")) {
            assertNotNull("kernels/llm/matvec_row.cl is on the classpath", in);
            source = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
        assertFalse("no direct __dp4a", source.contains("__dp4a"));
        assertTrue(source.contains("nf_dp4a("));
    }
}
