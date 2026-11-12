package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.spec.AttentionSpec;

/**
 * Which attention layers run under a tree verify step's mask, shared by the GPU and CPU attention mixers: the mask
 * replaces the position rules of a plain softmax attention, so layers whose attention reads more than those rules
 * refuse it by name.
 */
final class TreeMaskRules {

    private TreeMaskRules() {
    }

    /**
     * Why layer {@code il} cannot attend under a tree mask, or null when it can.
     *
     * @param dualChunk true when the layer runs dual-chunk attention
     * @param dual      true when the layer runs dual-attention (loop) layers
     * @param mode      the layer's position-bias mode
     */
    static String refusal(int il, boolean dualChunk, boolean dual, AttentionSpec.BiasMode mode) {
        if (dualChunk) {
            return "layer " + il + " runs dual-chunk attention, which takes no tree mask";
        }
        if (dual) {
            return "layer " + il + " runs dual-attention layers, which take no tree mask";
        }
        if (mode == AttentionSpec.BiasMode.T5_CAUSAL || mode == AttentionSpec.BiasMode.T5_BIDIR
                || mode == AttentionSpec.BiasMode.INKLING) {
            return "layer " + il + " runs " + mode + " position bias, which reads a relative table over the"
                    + " position rules a tree mask replaces";
        }
        return null;
    }
}
