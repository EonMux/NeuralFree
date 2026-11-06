package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.OptionalInt;

/**
 * The hooks of vLLM's Step1 decoder ({@code vllm/model_executor/models/step1.py}). Step1 has no rotary position:
 * its attention adds vLLM's square-root ALiBi bias ({@code use_alibi_sqrt}) with the per-head slopes of
 * {@code _get_step_alibi_slopes}, which are the standard ALiBi slopes for a maximum bias of 8. The feed-forward is
 * a gated SwiGLU, the norms are RMS norms, and the q/k/v, output and MLP projections carry biases when the
 * configuration enables them.
 */
public final class HfStep1Hooks {

    /** {@code _get_step_alibi_slopes}: 2**(-8 / 2**floor(log2(heads))). */
    private static final double ALIBI_MAX_BIAS = 8.0;

    private HfStep1Hooks() {
    }

    /**
     * The key/value head count: {@code num_attention_groups}, else {@code num_key_value_heads}, else one (vLLM's
     * fallback), and the SwiGLU activation check.
     */
    public static HfConfig fixup(HfConfig cfg) {
        Map<String, Object> derived = new LinkedHashMap<>();
        OptionalInt groups = cfg.findInt("num_attention_groups");
        if (groups.isPresent() && groups.getAsInt() > 0) {
            derived.put("num_key_value_heads", (long) groups.getAsInt());
        } else if (cfg.findInt(0, "num_key_value_heads") <= 0) {
            derived.put("num_key_value_heads", 1L);
        }
        String act = cfg.findString("hidden_act", "activation_function").orElse("silu");
        if (!"silu".equalsIgnoreCase(act)) {
            throw new GgufFormatException(cfg.source() + ": 'hidden_act' = '" + act
                    + "' is not supported: Step1 uses the gated SwiGLU feed-forward");
        }
        return cfg.withFlat(derived);
    }

    /** The entry of the Step1 checkpoints. */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.STEP1;
    }

    /** {@code StepDecoderModel}: the generic keys, the exact vocabulary and the fixed ALiBi maximum bias. */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.VOCAB_SIZE, ctx.config().requireInt("vocab_size"));
        kv.set(GgufKeys.ATTENTION_MAX_ALIBI_BIAS, ALIBI_MAX_BIAS);
    }

    /** A tied {@code lm_head} is dropped; every other tensor maps with its biases. */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String bare = hfName.startsWith("model.") ? hfName.substring("model.".length()) : hfName;
        if ("lm_head.weight".equals(bare) && ctx.config().findBool(true, "tie_word_embeddings")) {
            out.drop(hfName, "tied output layer; token_embd.weight serves as the output");
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }
}
