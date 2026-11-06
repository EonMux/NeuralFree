package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The pooled expert cache's assignment of the routed layers to the GPUs: in layer order, each GPU takes the run its
 * share of the free VRAM earns (Avalon {@code ggml_backend_sched_prime_moe_cache}), and the plan reports the owner per
 * layer and the runs for the load log.
 */
public class MoeCachePoolTest {

    @Test
    public void equalSharesDivideTheLayersInOrder() {
        List<LayerSpec> layers = layers(0, 1, 2, 3);
        MoeCachePool pool = MoeCachePool.plan(2, layers, bytes(1, 1, 1, 1), new long[]{1, 1});
        assertEquals(0, pool.ownerOf(0));
        assertEquals(0, pool.ownerOf(1));
        assertEquals(1, pool.ownerOf(2));
        assertEquals(1, pool.ownerOf(3));
        assertEquals(Arrays.asList(layers.get(0), layers.get(1)), pool.layersOf(0));
        assertEquals(Arrays.asList(layers.get(2), layers.get(3)), pool.layersOf(1));
        assertEquals(2L, pool.bytesOf(0));
        assertEquals(2L, pool.bytesOf(1));
        assertEquals(2, pool.stageCount());
        assertTrue(pool.owns(0, 1));
        assertFalse(pool.owns(1, 1));
        assertEquals(-1, pool.ownerOf(9));
        assertEquals("MoeCachePool{0:0-1, 1:2-3}", pool.toString());
    }

    @Test
    public void aThreeToOneShareGivesTheFirstGpuThreeQuartersOfTheBytes() {
        MoeCachePool pool = MoeCachePool.plan(2, layers(0, 1, 2, 3, 4, 5, 6, 7), bytes(1, 1, 1, 1, 1, 1, 1, 1),
                new long[]{3, 1});
        for (int il = 0; il <= 5; il++) {
            assertEquals("layer " + il, 0, pool.ownerOf(il));
        }
        assertEquals(1, pool.ownerOf(6));
        assertEquals(1, pool.ownerOf(7));
        assertEquals("MoeCachePool{0:0-5, 1:6-7}", pool.toString());
    }

    @Test
    public void unequalLayerBytesChooseTheOwnerByTheLayersMiddle() {
        // total 30, bound 15: layers 0 and 1 end their middles at 5 and 15 (inside), layer 2 at 25 (beyond).
        MoeCachePool pool = MoeCachePool.plan(2, layers(0, 1, 2), bytes(10, 10, 10), new long[]{1, 1});
        assertEquals(0, pool.ownerOf(0));
        assertEquals(0, pool.ownerOf(1));
        assertEquals(1, pool.ownerOf(2));
    }

    @Test
    public void threeGpusTakeThreeRuns() {
        MoeCachePool pool = MoeCachePool.plan(3, layers(0, 1, 2, 3, 4, 5), bytes(1, 1, 1, 1, 1, 1),
                new long[]{1, 1, 1});
        assertEquals("MoeCachePool{0:0-1, 1:2-3, 2:4-5}", pool.toString());
    }

    @Test
    public void aSingleStageOrNoBytesIsNoPool() {
        assertNull(MoeCachePool.plan(1, layers(0, 1), bytes(1, 1), new long[]{1}));
        assertNull(MoeCachePool.plan(2, layers(0, 1), bytes(0, 0), new long[]{1, 1}));
        assertNull(MoeCachePool.plan(2, new ArrayList<LayerSpec>(), bytes(), new long[]{1, 1}));
        assertNull(MoeCachePool.plan(2, layers(0, 1), bytes(1, 1), new long[]{0, 0}));
    }

    @Test
    public void thePlanRefusesInconsistentInputs() {
        assertThrows(IllegalArgumentException.class,
                () -> MoeCachePool.plan(2, layers(0, 1), bytes(1), new long[]{1, 1}));
        assertThrows(IllegalArgumentException.class,
                () -> MoeCachePool.plan(2, layers(0, 1), bytes(1, 1), new long[]{1}));
        assertThrows(IllegalArgumentException.class,
                () -> MoeCachePool.plan(2, layers(0, 0), bytes(1, 1), new long[]{1, 1}));
        assertThrows(IllegalArgumentException.class,
                () -> MoeCachePool.plan(2, layers(0, 1), bytes(-1, 1), new long[]{1, 1}));
    }

    // ------------------------------------------------------------------ fixtures

    /** {@code n} routed MoE layers, index equal to weight layer. */
    private static List<LayerSpec> layers(int... ils) {
        List<LayerSpec> out = new ArrayList<>(ils.length);
        for (int il : ils) {
            out.add(moeLayer(il));
        }
        return out;
    }

    private static long[] bytes(long... bytes) {
        return bytes;
    }

    private static LayerSpec moeLayer(int il) {
        FfnSpec shared = FfnSpec.builder(FfnSpec.Role.SHARED_EXPERT).nFf(32)
                .glu(FfnGating.PAR, GluOp.SWIGLU).build();
        MoeSpec moe = MoeSpec.builder().experts(4, 2, 16).gating(GatingFunc.SOFTMAX)
                .glu(GluOp.SWIGLU).sharedExpert(shared, true).build();
        return LayerSpec.builder(il, MixerKind.NONE, FfnKind.MOE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, 64).build())
                .moe(moe)
                .build();
    }
}
