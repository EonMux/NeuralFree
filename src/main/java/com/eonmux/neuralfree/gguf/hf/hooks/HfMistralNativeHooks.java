package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Hooks of llama.cpp's {@code MistralModel} (LC/conversion/mistral.py:24-110), the unregistered class that converts
 * Mistral native checkpoints ({@code params.json} with {@code consolidated*.safetensors}): a {@code LlamaModel} with
 * {@code is_mistral_format} set and {@code undo_permute} off, whose native tensor names
 * ({@code tok_embeddings}, {@code layers.<b>.attention.wq}, {@code feed_forward.w1}, {@code attention_norm},
 * {@code ffn_norm}, {@code norm}, {@code output}) map through the name map as they are.
 *
 * <p>Every method is a public static hook reference; the entry registers them with {@code withUndoPermute(false)}.</p>
 */
public final class HfMistralNativeHooks {

    /** The FP8 scale suffixes of Mistral native checkpoints and the Hugging Face names they are renamed to. */
    private static final String QSCALE_WEIGHT = ".qscale_weight";
    private static final String QSCALE_ACT = ".qscale_act";
    private static final String WEIGHT_SCALE = ".weight_scale";
    private static final String INPUT_SCALE = ".input_scale";

    private HfMistralNativeHooks() {
    }

    /**
     * {@link Arch#MISTRAL3} when {@code params.json} has {@code llama_4_scaling}, else {@link Arch#LLAMA}
     * ({@code MistralModel.__init__}, mistral.py:31-40).
     */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return cfg.flat().containsKey("llama_4_scaling") ? Arch.MISTRAL3 : Arch.LLAMA;
    }

    /**
     * The FP8 scales of a native checkpoint under the names the quantization rewrite reads: {@code *.qscale_weight}
     * becomes {@code *.weight_scale} and {@code *.qscale_act} becomes {@code *.input_scale}, the tensors
     * {@code dequant_model} pairs with each weight (mistral.py:41-52, LC/conversion/base.py:486-495). Every tensor
     * keeps its position.
     *
     * @throws GgufFormatException when a renamed scale collides with a tensor of the checkpoint
     */
    public static HfTensorIndex index(HfLoadContext ctx, HfTensorIndex index) {
        Map<String, RowSource> out = new LinkedHashMap<>();
        boolean renamed = false;
        for (String name : index.names()) {
            String target = name;
            if (name.endsWith(QSCALE_WEIGHT)) {
                target = HfHookSupport.renameTail(name, QSCALE_WEIGHT, WEIGHT_SCALE);
            } else if (name.endsWith(QSCALE_ACT)) {
                target = HfHookSupport.renameTail(name, QSCALE_ACT, INPUT_SCALE);
            }
            if (!target.equals(name)) {
                if (index.has(target)) {
                    throw new GgufFormatException("checkpoint tensors '" + name + "' and '" + target
                            + "' are both the scale of one weight");
                }
                renamed = true;
            }
            out.put(target, index.get(name));
        }
        return renamed ? HfTensorIndex.of(out) : index;
    }

    /**
     * {@code MistralModel.set_gguf_parameters} (mistral.py:92-94): the Llama metadata without {@code vocab_size}
     * ({@link HfLlamaHooks#llamaKv}, which skips it for Mistral-format checkpoints), then {@link #mistralConfig}.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfLlamaHooks.llamaKv(ctx, kv, true);
        mistralConfig(ctx.config(), kv);
    }

    /**
     * {@code MistralModel.set_mistral_config} (mistral.py:96-110), shared with the Mistral classes that call it:
     * a {@code yarn} block gives scaling type {@code yarn}, its factor, {@code yarn_beta_fast = beta},
     * {@code yarn_beta_slow = alpha}, {@code yarn_log_multiplier = apply_scale ? 0 : 1} and the original context
     * length; {@code llama_4_scaling.beta} gives {@code attention.temperature_scale}.
     *
     * @throws GgufFormatException when a block lacks a value or has one of the wrong type
     */
    public static void mistralConfig(HfConfig cfg, HfMetadataBuilder kv) {
        JsonView root = cfg.raw();
        if (root.raw("yarn") != null) {
            JsonView yarn = root.obj("yarn");
            kv.set(GgufKeys.ROPE_SCALING_TYPE, "yarn");
            kv.set(GgufKeys.ROPE_SCALING_FACTOR, yarn.f64("factor"));
            kv.set(GgufKeys.ROPE_SCALING_YARN_BETA_FAST, yarn.f64("beta"));
            kv.set(GgufKeys.ROPE_SCALING_YARN_BETA_SLOW, yarn.f64("alpha"));
            kv.set(GgufKeys.ROPE_SCALING_YARN_LOG_MUL, yarn.bool("apply_scale") ? 0.0 : 1.0);
            kv.set(GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, yarn.i64("original_max_position_embeddings"));
        }
        if (root.raw("llama_4_scaling") != null) {
            kv.set(GgufKeys.ATTENTION_TEMPERATURE_SCALE, root.obj("llama_4_scaling").f64("beta"));
        }
    }

    /**
     * {@code LlamaModel.modify_tensors} with the native names: {@link HfLlamaHooks#tensors}, which leaves Q and K as
     * stored because the entry does not undo the permutation (mistral.py:29).
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        HfLlamaHooks.tensors(ctx, hfName, src, out);
    }

    /** {@code LlamaModel.generate_extra_tensors} ({@link HfLlamaHooks#extra}). */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        HfLlamaHooks.extra(ctx, out);
    }

    /**
     * {@code LlamaModel.set_vocab} of a Mistral-format checkpoint (llama.py:118-119): the mistral_common vocabulary
     * ({@code tekken.json}, else the versioned SentencePiece model) with the checkpoint's {@code chat_template.jinja}
     * or the community template ({@link HfVocabWriters#mistral}).
     *
     * @throws IOException when a tokenizer file cannot be read
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.mistral(HfHookSupport.vocabInput(ctx), pre);
    }
}
