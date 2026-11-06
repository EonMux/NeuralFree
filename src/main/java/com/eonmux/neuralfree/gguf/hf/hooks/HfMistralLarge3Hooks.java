package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hooks of vLLM's {@code MistralLarge3ForCausalLM} (V/model_executor/models/mistral_large_3.py), the
 * {@code DeepseekV3ForCausalLM} subclass for a Mistral Large 3 checkpoint whose {@code config.json} names it: the
 * DeepSeek-V3 configuration, tensors under the Hugging Face DeepSeek names or the native Mistral names, which its
 * {@code hf_to_vllm_mapper} renames ({@link #index}), and the {@code llama_4_scaling} attention temperature
 * ({@link #meta}). The rest is {@link HfDeepseekHooks}. A native checkpoint described by {@code params.json} is the
 * synthesized Mistral MoE format of {@link HfMistralMoeHooks} instead.
 */
public final class HfMistralLarge3Hooks {

    /** The suffix renames of {@code orig_to_new_suffix}, applied after the regex renames. */
    private static final String[][] SUFFIXES = {{".qscale_act", ".input_scale"}, {".qscale_weight", ".weight_scale"}};

    /**
     * The anchored native-name renames of {@code orig_to_new_regex}, in mapper order; every result starts with
     * {@code model.} or {@code lm_head.}, so no later rule matches a renamed name.
     */
    private static final List<Rename> RENAMES = Collections.unmodifiableList(Arrays.asList(
            new Rename("layers\\.(\\d+)\\.attention_norm\\.weight", "model.layers.$1.input_layernorm.weight"),
            new Rename("layers\\.(\\d+)\\.attention\\.wq_a\\.(\\w+)", "model.layers.$1.self_attn.q_a_proj.$2"),
            new Rename("layers\\.(\\d+)\\.attention\\.q_a_norm\\.weight",
                    "model.layers.$1.self_attn.q_a_layernorm.weight"),
            new Rename("layers\\.(\\d+)\\.attention\\.wq_b\\.(\\w+)", "model.layers.$1.self_attn.q_b_proj.$2"),
            new Rename("layers\\.(\\d+)\\.attention\\.wkv_a_with_mqa\\.(\\w+)",
                    "model.layers.$1.self_attn.kv_a_proj_with_mqa.$2"),
            new Rename("layers\\.(\\d+)\\.attention\\.kv_a_norm\\.weight",
                    "model.layers.$1.self_attn.kv_a_layernorm.weight"),
            new Rename("layers\\.(\\d+)\\.attention\\.wkv_b\\.(\\w+)", "model.layers.$1.self_attn.kv_b_proj.$2"),
            new Rename("layers\\.(\\d+)\\.attention\\.wo\\.(\\w+)", "model.layers.$1.self_attn.o_proj.$2"),
            new Rename("layers\\.(\\d+)\\.ffn_norm\\.weight", "model.layers.$1.post_attention_layernorm.weight"),
            new Rename("layers\\.(\\d+)\\.feed_forward\\.w1\\.(\\w+)", "model.layers.$1.mlp.gate_proj.$2"),
            new Rename("layers\\.(\\d+)\\.feed_forward\\.w2\\.(\\w+)", "model.layers.$1.mlp.down_proj.$2"),
            new Rename("layers\\.(\\d+)\\.feed_forward\\.w3\\.(\\w+)", "model.layers.$1.mlp.up_proj.$2"),
            new Rename("layers\\.(\\d+)\\.gate\\.weight", "model.layers.$1.mlp.gate.weight"),
            new Rename("layers\\.(\\d+)\\.shared_experts\\.w1\\.(\\w+)",
                    "model.layers.$1.mlp.shared_experts.gate_proj.$2"),
            new Rename("layers\\.(\\d+)\\.shared_experts\\.w2\\.(\\w+)",
                    "model.layers.$1.mlp.shared_experts.down_proj.$2"),
            new Rename("layers\\.(\\d+)\\.shared_experts\\.w3\\.(\\w+)",
                    "model.layers.$1.mlp.shared_experts.up_proj.$2"),
            new Rename("layers\\.(\\d+)\\.experts\\.(\\d+)\\.w1\\.(\\w+)",
                    "model.layers.$1.mlp.experts.$2.gate_proj.$3"),
            new Rename("layers\\.(\\d+)\\.experts\\.(\\d+)\\.w2\\.(\\w+)",
                    "model.layers.$1.mlp.experts.$2.down_proj.$3"),
            new Rename("layers\\.(\\d+)\\.experts\\.(\\d+)\\.w3\\.(\\w+)",
                    "model.layers.$1.mlp.experts.$2.up_proj.$3"),
            new Rename("norm\\.weight", "model.norm.weight"),
            new Rename("tok_embeddings\\.weight", "model.embed_tokens.weight"),
            new Rename("output\\.weight", "lm_head.weight")));

    private HfMistralLarge3Hooks() {
    }

    /** One anchored rename of the mapper. */
    private static final class Rename {
        final Pattern pattern;
        final String replacement;

        Rename(String regex, String replacement) {
            this.pattern = Pattern.compile(regex);
            this.replacement = replacement;
        }
    }

    /**
     * The renames of {@code hf_to_vllm_mapper}, raised into the index step so the quantization rewrite sees the
     * renamed FP8 scales: every native name matching a rule of the regex table takes its Hugging Face DeepSeek name,
     * then {@code .qscale_act} becomes {@code .input_scale} and {@code .qscale_weight} {@code .weight_scale}. Names
     * already in the Hugging Face layout pass unchanged.
     *
     * @throws GgufFormatException when two checkpoint tensors rename onto one name
     */
    public static HfTensorIndex index(HfLoadContext ctx, HfTensorIndex index) {
        Map<String, RowSource> out = new LinkedHashMap<>();
        Map<String, String> origin = new LinkedHashMap<>();
        boolean renamed = false;
        for (String name : index.names()) {
            String n = rename(name);
            String first = origin.put(n, name);
            if (first != null) {
                throw new GgufFormatException("checkpoint tensors '" + first + "' and '" + name
                        + "' both become '" + n + "' after MistralLarge3ForCausalLM's renames");
            }
            renamed |= !n.equals(name);
            out.put(n, index.get(name));
        }
        return renamed ? HfTensorIndex.of(out) : index;
    }

    /** {@code name} through the regex table, then the suffix renames. */
    static String rename(String name) {
        String n = name;
        for (Rename r : RENAMES) {
            Matcher m = r.pattern.matcher(n);
            if (m.matches()) {
                n = m.replaceFirst(r.replacement);
            }
        }
        for (String[] s : SUFFIXES) {
            if (n.endsWith(s[0])) {
                n = HfHookSupport.renameTail(n, s[0], s[1]);
            }
        }
        return n;
    }

    /**
     * The DeepSeek-V2 metadata ({@link HfDeepseekHooks#meta}), then the {@code llama_4_scaling} block the model
     * applies to its queries (deepseek_v2.py {@code _get_llama_4_scaling},
     * {@code 1 + beta * ln(1 + floor(pos / original_max_position_embeddings))}) as the attention temperature scale
     * and length.
     *
     * @throws GgufFormatException when {@code llama_4_scaling} is not an object with a number {@code beta} and a
     *                             positive integer {@code original_max_position_embeddings}
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfDeepseekHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        Optional<Object> scaling = cfg.find("llama_4_scaling");
        if (!scaling.isPresent()) {
            return;
        }
        if (!(scaling.get() instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": 'llama_4_scaling' must be an object");
        }
        Map<?, ?> s = (Map<?, ?>) scaling.get();
        Object beta = s.get("beta");
        if (!(beta instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": 'llama_4_scaling.beta' must be a number, got " + beta);
        }
        Object orig = s.get("original_max_position_embeddings");
        if (!(orig instanceof Long || orig instanceof Integer) || ((Number) orig).longValue() <= 0
                || ((Number) orig).longValue() > Integer.MAX_VALUE) {
            throw new GgufFormatException(cfg.source() + ": 'llama_4_scaling.original_max_position_embeddings'"
                    + " must be a positive integer, got " + orig);
        }
        kv.set(GgufKeys.ATTENTION_TEMPERATURE_SCALE, ((Number) beta).doubleValue());
        kv.set(GgufKeys.ATTENTION_TEMPERATURE_LENGTH, ((Number) orig).intValue());
    }

    /**
     * The vocabulary: the mistral_common {@code tekken.json} when the checkpoint has no {@code tokenizer.json}
     * (vLLM's {@code tokenizer_mode mistral}), else the DeepSeek-V2 vocabulary ({@link HfDeepseekHooks#vocab}).
     *
     * @throws IOException when a tokenizer file cannot be read
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (!files.hasGgufTokenizer() && files.tekkenJson() != null && files.tokenizerJson() == null) {
            return HfVocabWriters.mistral(HfHookSupport.vocabInput(ctx), pre);
        }
        return HfDeepseekHooks.vocab(ctx, pre);
    }
}
