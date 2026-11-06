package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.function.UnaryOperator;

/**
 * Hugging Face Solar conversion (vLLM's {@code solar.py}, a Llama decoder): the family's plain checkpoints are the
 * Llama graph, so the entry reuses {@link HfLlamaHooks} and only refuses the additions the llama layout cannot carry,
 * the block-skip connections between layers ({@code bskcn_1} to {@code bskcn_4}) and a non-neutral logit multiplier.
 */
public final class HfSolarHooks {
    private static final String[] BSKCN = {"bskcn_1", "bskcn_2", "bskcn_3", "bskcn_4"};

    private HfSolarHooks() {
    }

    /**
     * Validates the configuration and applies the Llama fixup.
     *
     * @throws GgufFormatException naming the unsupported field
     */
    public static HfConfig fixup(HfConfig cfg) {
        for (String key : BSKCN) {
            if (!cfg.findIntList(key).orElse(List.of()).isEmpty()) {
                throw new GgufFormatException(cfg.source() + ": '" + key + "': Solar block-skip connections are"
                        + " unsupported");
            }
        }
        OptionalDouble logitScale = cfg.findFloat("logit_scale");
        if (logitScale.isPresent() && logitScale.getAsDouble() != 1.0) {
            throw new GgufFormatException(cfg.source() + ": 'logit_scale': a Solar logit multiplier other than 1 is"
                    + " unsupported");
        }
        return HfLlamaHooks.fixup(cfg);
    }

    /** The Llama metadata after {@link #fixup}. */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfLlamaHooks.meta(ctx, kv, plan);
    }

    /** The Llama tensor mapping (with the Q/K permutation its NORM rotary needs). */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        HfLlamaHooks.tensors(ctx, hfName, src, out);
    }

    /** The Llama name filter. */
    public static String filter(String name, UnaryOperator<String> base) {
        return HfLlamaHooks.filter(name, base);
    }

    /** The Llama vocabulary. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfLlamaHooks.vocab(ctx, pre);
    }
}
