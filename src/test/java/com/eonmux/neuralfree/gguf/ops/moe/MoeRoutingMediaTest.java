package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The embedding-row routing parameters ({@link MoeRouting.Params#forMedia}, deepseek4.cpp:1290-1301) and the merge of
 * two row sets' selections into one step ({@link MoeRouting#merge}).
 */
public class MoeRoutingMediaTest {

    private static MoeSpec.Builder spec() {
        return MoeSpec.builder().experts(8, 2, 16).gating(GatingFunc.SQRT_SOFTPLUS).normWeights(true)
                .weightsScale(1.5f).groups(4, 2).glu(GluOp.SWIGLU);
    }

    @Test
    public void aHashLayerRoutesEmbeddingRowsThroughTheRouterWithoutABias() {
        MoeSpec hash = spec().hashRouting(true).build();
        assertTrue(hash.mediaRouting());
        assertEquals("token rows take the hash ids", MoeKernels.SEL_PRESELECTED,
                MoeRouting.Params.forExperts(hash).selMode());
        MoeRouting.Params media = MoeRouting.Params.forMedia(hash);
        assertEquals(hash.selectionMode().code(), media.selMode());
        assertFalse(media.hasBias());
        assertEquals(4, media.nGroups());
        assertEquals(2, media.nGroupUsed());
        assertEquals(1.5f, media.weightsScale(), 0f);
        assertTrue(media.normWeights());
    }

    @Test
    public void theVisionBiasBecomesTheSelectionBias() {
        MoeSpec hash = spec().hashRouting(true).mediaProbBias(true).build();
        assertTrue(MoeRouting.Params.forMedia(hash).hasBias());
        MoeSpec biased = spec().probBias(true).mediaProbBias(true).build();
        assertTrue(biased.mediaRouting());
        assertTrue(MoeRouting.Params.forMedia(biased).hasBias());
    }

    @Test
    public void aLayerWithoutMediaRoutingRefuses() {
        MoeSpec plain = spec().probBias(true).build();
        assertFalse(plain.mediaRouting());
        assertThrows(IllegalStateException.class, () -> MoeRouting.Params.forMedia(plain));
    }

    @Test
    public void aMergePlacesEachSelectionAtItsRows() {
        MoeRouting.Params p = new MoeRouting.Params(4, 2, GatingFunc.SOFTMAX, 1, 1, false, 1f, false,
                MoeKernels.SEL_PROBS, 0, false);
        float[] a = {4, 3, 0, 0, 0, 0, 3, 4};
        float[] b = {0, 4, 3, 0};
        MoeRouting.Selection sa = MoeRouting.route(p, a, 2, null);
        MoeRouting.Selection sb = MoeRouting.route(p, b, 1, null);
        MoeRouting.Selection merged = MoeRouting.merge(3, new int[] {0, 2}, sa, new int[] {1}, sb);
        assertArrayEquals(new int[] {0, 1, 1, 2, 3, 2}, merged.ids());
        for (int j = 0; j < 2; j++) {
            assertEquals(sa.weight(0, j), merged.weight(0, j), 0f);
            assertEquals(sb.weight(0, j), merged.weight(1, j), 0f);
            assertEquals(sa.weight(1, j), merged.weight(2, j), 0f);
        }
    }

    @Test
    public void aMergeRefusesRowsThatDoNotCoverTheStepOnce() {
        MoeRouting.Params p = new MoeRouting.Params(4, 2, GatingFunc.SOFTMAX, 1, 1, false, 1f, false,
                MoeKernels.SEL_PROBS, 0, false);
        MoeRouting.Selection one = MoeRouting.route(p, new float[] {1, 2, 3, 4}, 1, null);
        assertThrows(IllegalArgumentException.class,
                () -> MoeRouting.merge(2, new int[] {0}, one, new int[] {0}, one));
        assertThrows(IllegalArgumentException.class,
                () -> MoeRouting.merge(2, new int[] {0}, one, new int[] {2}, one));
        assertThrows(IllegalArgumentException.class,
                () -> MoeRouting.merge(3, new int[] {0}, one, new int[] {1}, one));
        MoeRouting.Params k1 = new MoeRouting.Params(4, 1, GatingFunc.SOFTMAX, 1, 1, false, 1f, false,
                MoeKernels.SEL_PROBS, 0, false);
        MoeRouting.Selection other = MoeRouting.route(k1, new float[] {1, 2, 3, 4}, 1, null);
        assertThrows(IllegalArgumentException.class,
                () -> MoeRouting.merge(2, new int[] {0}, one, new int[] {1}, other));
    }
}
