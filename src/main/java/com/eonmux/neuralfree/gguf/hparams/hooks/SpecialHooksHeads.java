package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

/**
 * Keys of the decision heads that clef and vllm-decision2 read after their backbone's hook (combine with
 * {@link SpecialHooks#extend}). Line references are to llama.cpp src/models/.
 */
public final class SpecialHooksHeads {

    /** vllm-decision2 head keys, applied after the qwen3 hook. */
    public static final ArchHparamsHook VLLM_DECISION2_HEAD = SpecialHooksHeads::vllmDecision2Head;
    /** clef head keys, applied after the qwen35 hook. */
    public static final ArchHparamsHook CLEF_HEAD = SpecialHooksHeads::clefHead;

    private SpecialHooksHeads() {
    }

    /** vllm-decision2.cpp:10-14: layer-norm epsilon of the head and one score per token. */
    private static void vllmDecision2Head(ModelHparamsBuilder b, GgufKv kv) {
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));             // vllm-decision2.cpp:11
        b.nEmbdOutImpl(1);                                                         // vllm-decision2.cpp:14
    }

    /**
     * clef.cpp:10-22: routing and joint head blocks, head count, layer-norm epsilon of the head and one score
     * per token.
     */
    private static void clefHead(ModelHparamsBuilder b, GgufKv kv) {
        b.decisionRoutingLayers(kv.u32Required(GgufKeys.DECISION_ROUTING_BLOCK_COUNT));  // clef.cpp:10
        b.decisionJointLayers(kv.u32Required(GgufKeys.DECISION_BLOCK_COUNT));            // clef.cpp:11
        b.decisionHeadCount(kv.u32Required(GgufKeys.DECISION_HEAD_COUNT));               // clef.cpp:12
        if (b.decisionHeadCount() == 0 || b.decisionRoutingLayers() > ModelHparams.MAX_LAYERS
                || b.decisionJointLayers() > ModelHparams.MAX_LAYERS) {                  // clef.cpp:14-16
            throw SpecialHooks.invalid(b, "invalid size of the decision head: " + b.decisionRoutingLayers()
                    + " routing blocks, " + b.decisionJointLayers() + " joint blocks, " + b.decisionHeadCount()
                    + " heads");
        }
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));             // clef.cpp:19
        b.nEmbdOutImpl(1);                                                         // clef.cpp:22
    }
}
