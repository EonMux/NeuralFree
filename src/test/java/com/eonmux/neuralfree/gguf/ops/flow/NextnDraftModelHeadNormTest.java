package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.arch.Arch;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Predicate;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The head-norm tensor each architecture's NextN/MTP head normalises its block output with. */
public class NextnDraftModelHeadNormTest {

    private static final int WL = 3;
    private static final String OUTPUT_NORM = "output_norm.weight";
    private static final String SHARED = "blk.3.nextn.shared_head_norm.weight";
    private static final String LAYER_OUT = "blk.3.layer_output_norm.weight";

    private static Predicate<String> file(String... tensors) {
        Set<String> names = new HashSet<>(Arrays.asList(tensors));
        return names::contains;
    }

    /** bailingmoe3.cpp:152 loads the head norm from layer_output_norm; output_norm is never the head norm. */
    @Test
    public void bailingmoe3UsesTheBlockLayerOutputNorm() {
        assertEquals(LAYER_OUT, NextnDraftModel.headNormWeight(Arch.BAILINGMOE3, WL,
                file(OUTPUT_NORM, LAYER_OUT), OUTPUT_NORM));
        assertEquals(LAYER_OUT, NextnDraftModel.headNormWeight(Arch.BAILINGMOE3, WL,
                file(OUTPUT_NORM, SHARED, LAYER_OUT), OUTPUT_NORM));
    }

    /** bailingmoe3.cpp:430 asserts the head norm exists; there is no fallback. */
    @Test
    public void bailingmoe3RequiresTheBlockLayerOutputNorm() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () ->
                NextnDraftModel.headNormWeight(Arch.BAILINGMOE3, WL, file(OUTPUT_NORM, SHARED), OUTPUT_NORM));
        assertTrue(e.getMessage(), e.getMessage().contains(LAYER_OUT));
    }

    /** mimo2.cpp:384-386: shared_head_norm, else layer_output_norm, else output_norm. */
    @Test
    public void mimo2PrefersSharedThenLayerOutputThenOutputNorm() {
        assertEquals(SHARED, NextnDraftModel.headNormWeight(Arch.MIMO2, WL,
                file(OUTPUT_NORM, SHARED, LAYER_OUT), OUTPUT_NORM));
        assertEquals(LAYER_OUT, NextnDraftModel.headNormWeight(Arch.MIMO2, WL,
                file(OUTPUT_NORM, LAYER_OUT), OUTPUT_NORM));
        assertEquals(OUTPUT_NORM, NextnDraftModel.headNormWeight(Arch.MIMO2, WL, file(OUTPUT_NORM), OUTPUT_NORM));
    }

    @Test
    public void otherHeadsUseSharedHeadNormElseOutputNorm() {
        assertEquals(SHARED, NextnDraftModel.headNormWeight(Arch.DEEPSEEK2, WL,
                file(OUTPUT_NORM, SHARED), OUTPUT_NORM));
        assertEquals(OUTPUT_NORM, NextnDraftModel.headNormWeight(Arch.DEEPSEEK2, WL,
                file(OUTPUT_NORM, LAYER_OUT), OUTPUT_NORM));
    }

    @Test
    public void nemotronHMoeRequiresSharedHeadNorm() {
        assertEquals(SHARED, NextnDraftModel.headNormWeight(Arch.NEMOTRON_H_MOE, WL,
                file(OUTPUT_NORM, SHARED), OUTPUT_NORM));
        assertThrows(IllegalArgumentException.class, () ->
                NextnDraftModel.headNormWeight(Arch.NEMOTRON_H_MOE, WL, file(OUTPUT_NORM), OUTPUT_NORM));
    }

    @Test
    public void aHeadWithoutAnyNormIsRejected() {
        assertThrows(IllegalArgumentException.class, () ->
                NextnDraftModel.headNormWeight(Arch.DEEPSEEK2, WL, file(), null));
        assertThrows(IllegalArgumentException.class, () ->
                NextnDraftModel.headNormWeight(Arch.DEEPSEEK2, WL, file(), OUTPUT_NORM));
    }
}
