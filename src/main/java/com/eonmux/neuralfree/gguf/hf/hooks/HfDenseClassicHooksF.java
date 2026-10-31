package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Hooks of the llama.cpp classic-dense converter classes of Appendix A table HK-D1, part F: the tail of the former
 * part E over {@code Spark2_5Model} (spark2_5.py:14-64), {@code Phi2Model} (phi.py:18-39), {@code StarCoder2Model}
 * (LC/conversion/starcoder.py:24-27), {@code TalkieModel} (talkie.py:15-53), {@code XverseModel} (xverse.py:15-86)
 * and {@code GraniteSwitchModel} (granite.py:232-359), which composes the HK-0 {@link HfGraniteHooks} GraniteMoe
 * hooks read-only. The leading classes of part E (from {@code InternLM2Model} to {@code PanguEmbeddedModel}) stay in
 * {@link HfDenseClassicHooksE}. <p>Every method is a public static hook reference that the table composes and
 * subclasses reuse where the converter calls {@code super()}.</p>
 */
public final class HfDenseClassicHooksF {

    private HfDenseClassicHooksF() {
    }

    /** {@code self.hparams[key]} as a double, required, with a clear message. */
    static double requireDouble(HfConfig cfg, String key) {
        return cfg.findFloat(key).orElseThrow(() -> new GgufFormatException(cfg.source() + ": required key '" + key
                + "' is missing"));
    }

    /** {@code Spark2_5Model.set_gguf_parameters} (spark2_5.py:17-51): the inherited keys, then the layer-type checks ({@code layer_types} over {@code sliding_attention | full_attention}, the sigmoid head-wise attention gates and GELU, spark2_5.py:19-33), the exact {@code vocab_size}, the sliding window and per-layer pattern, and the full-attention and sliding-attention RoPE widths of their partial factors (spark2_5.py:42-50). @throws GgufFormatException when the layer types, the gates or the activation fall outside the supported set, or a partial rotary factor is missing */
    public static void spark25Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        List<?> layerTypes = layerTypes(ctx, "layer_types");
        if (layerTypes.size() != ctx.blockCount()) {
            throw new GgufFormatException(cfg.source() + ": layer_types length " + layerTypes.size()
                    + " != num_hidden_layers " + ctx.blockCount());
        }
        List<Boolean> pattern = new ArrayList<>();
        for (Object t : layerTypes) {
            String type = String.valueOf(t);
            if (!"sliding_attention".equals(type) && !"full_attention".equals(type)) {
                throw new GgufFormatException(cfg.source() + ": Spark2_5 has unsupported layer_types: " + type);
            }
            pattern.add("sliding_attention".equals(type));
        }
        if (!"sigmoid".equals(cfg.findString("gate_attn_act_mode").orElse(""))
                || !Boolean.TRUE.equals(cfg.findBool("headwise_attn_output_gate").orElse(Boolean.FALSE))) {
            throw new GgufFormatException(cfg.source()
                    + ": Spark2_5 conversion requires head-wise sigmoid attention gates");
        }
        if (!"gelu".equals(cfg.findStringOr("", "hidden_act"))) {
            throw new GgufFormatException(cfg.source() + ": Spark2_5 conversion requires GELU, got "
                    + cfg.findString("hidden_act").orElse(""));
        }
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, Collections.unmodifiableList(pattern));
        int headDim = cfg.requireInt("head_dim");
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) (headDim * partial(ctx, "full_attention")));
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT_SWA, (int) (headDim * partial(ctx, "sliding_attention")));
    }

    /** The partial rotary factor of one rope layer type, as spark2_5.py:43-50 reads it. */
    private static double partial(HfLoadContext ctx, String layerType) {
        return ctx.config().rope().layer(layerType)
                .orElseThrow(() -> missing(ctx, "partial_rotary_factor", layerType))
                .partialRotaryFactor()
                .orElseThrow(() -> missing(ctx, "partial_rotary_factor", layerType));
    }

    private static GgufFormatException missing(HfLoadContext ctx, String key, String layerType) {
        return new GgufFormatException(ctx.config().source() + ": required key '" + key + "' is missing for rope"
                + " layer type '" + layerType + "'");
    }

    /** The list value of a configuration key. */
    private static List<?> layerTypes(HfLoadContext ctx, String key) {
        Object flat = ctx.config().flat().get(key);
        if (!(flat instanceof List<?>)) {
            throw new GgufFormatException(ctx.config().source() + ": required key '" + key + "' is missing");
        }
        return (List<?>) flat;
    }

    /** {@code Spark2_5Model.modify_tensors} (spark2_5.py:53-63): the fused {@code self_attn.q_k_v_proj} weight maps to the ATTN_QKV tensor, requiring a block id (spark2_5.py:55-59); the attention gate width is checked against the head count (spark2_5.py:61-69); every other tensor passes on. @throws GgufFormatException when a fused tensor has no block id, or a gate width is off */
    public static void spark25Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith(".self_attn.q_k_v_proj.weight")) {
            int bid = HfLoadContext.bidOf(hfName);
            if (bid < 0) {
                throw new GgufFormatException("Spark2_5 fused QKV tensor has no block id: " + hfName);
            }
            out.add(ctx.names().ggufName("ATTN_QKV", bid, ".weight"), src);
            return;
        }
        if (hfName.endsWith(".self_attn.g_proj.weight")) {
            int bid = HfLoadContext.bidOf(hfName);
            if (bid < 0) {
                throw new GgufFormatException("Spark2_5 attention gate tensor has no block id: " + hfName);
            }
            if (src.rows() != ctx.config().requireInt("num_attention_heads")) {
                throw new GgufFormatException("Spark2_5 layer " + bid + " attention gate width " + src.rows()
                        + " != head count " + ctx.config().requireInt("num_attention_heads"));
            }
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code Phi2Model.set_gguf_parameters} (phi.py:22-38), hand-set: the {@code n_positions | max_position_embeddings} context, the {@code hidden_size | n_embd} width and 4x FFN, the heads over the same count as key/value heads, the {@code layer_norm_epsilon | layer_norm_eps} epsilon, the {@code partial_rotary_factor}-scaled RoPE width, and no BOS token. */
    public static void phi2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int hidden = cfg.findInt("hidden_size", "n_embd").orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key 'hidden_size' or 'n_embd' is missing"));
        int heads = cfg.findInt("num_attention_heads", "n_head").orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key 'num_attention_heads' or 'n_head' is missing"));
        double partial = cfg.rope().global().partialRotaryFactor().orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key 'partial_rotary_factor' is missing"));
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.findInt("n_positions", "max_position_embeddings")
                .orElseThrow(() -> new GgufFormatException(cfg.source()
                        + ": required key 'n_positions' or 'max_position_embeddings' is missing")));
        kv.set(GgufKeys.EMBEDDING_LENGTH, hidden);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, 4 * hidden);
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, heads);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, heads);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, cfg.findFloat("layer_norm_epsilon", "layer_norm_eps")
                .orElseThrow(() -> new GgufFormatException(cfg.source()
                        + ": required key 'layer_norm_epsilon' or 'layer_norm_eps' is missing")));
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) (partial * hidden) / heads);
        kv.set(GgufKeys.TOKENIZER_ADD_BOS, false);
    }

    /** {@code StarCoder2Model.set_gguf_parameters}: the class overrides no parameter, so the inherited keys. */
    public static void starcoder2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
    }

    /** {@code TalkieModel.set_gguf_parameters} (talkie.py:20-24): the inherited keys, then the RoPE epsilon of a float32 ({@code torch.finfo(torch.float32).eps}). */
    public static void talkieMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, 1.1920929E-7);
    }

    /** {@code TalkieModel.modify_tensors} (talkie.py:26-53) over the {@code model.blocks.<bid>.} names: the {@code attn_gain.a_g} gain becomes the ATTN_OUT scale and {@code mlp_gain.a_g} the FFN_DOWN scale (talkie.py:33-38), the {@code lm_head_gain.w_g} scalar becomes the logit scale (talkie.py:39-41), the {@code attn_query} and {@code attn_key} weights absorb the inverse RoPE by negating the second half of every head ({@link RowTransforms#negateSecondHalfRows}, talkie.py:43-51), the {@code attn.head_gain.head_g} gains get a broadcast trailing dim (talkie.py:52-54), and a tensor whose name lacks {@code .weight} gains it (talkie.py:56-58). */
    public static void talkieTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        int bid = HfLoadContext.bidOf(hfName);
        String suffix = hfName;
        if (bid >= 0 && hfName.startsWith("model.blocks." + bid + ".")) {
            suffix = hfName.substring(("model.blocks." + bid + ".").length());
        }
        HfConfig cfg = ctx.config();
        RowSource data = src;
        String name = hfName;
        if (suffix.equals("attn_gain.a_g")) {
            out.add(ctx.names().ggufName("ATTN_OUT", bid, ".scale"), src);
            return;
        } else if (suffix.equals("mlp_gain.a_g")) {
            out.add(ctx.names().ggufName("FFN_DOWN", bid, ".scale"), src);
            return;
        } else if (suffix.equals("lm_head_gain.w_g")) {
            out.addKv(GgufKeys.LOGIT_SCALE, GGUFValueType.FLOAT32, firstFloat(src));
            return;
        } else if (suffix.equals("attn.attn_query.weight") || suffix.equals("attn.attn_key.weight")) {
            data = RowTransforms.negateSecondHalfRows(src, cfg.requireInt("head_dim"));
        } else if (suffix.equals("attn.head_gain.head_g")) {
            long[] shape = src.shape();
            if (shape.length != 1) {
                throw new GgufFormatException("talkie head gain '" + hfName + "' must be 1-D, got "
                        + Arrays.toString(shape));
            }
            data = ElementTransforms.reshape(src, shape[0], 1);
        }
        if (!name.endsWith(".weight")) {
            name = name + ".weight";
        }
        HfTensorHook.DEFAULT.apply(ctx, name, data, out);
    }

    /** The first value of a 1-D float source, as the logit scale needs. */
    private static float firstFloat(RowSource src) throws IOException {
        if (src.rows() < 1) {
            throw new GgufFormatException(src.describe() + ": the logit gain is empty");
        }
        ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        src.readRows(0, 1, b);
        return b.getFloat(0);
    }

    /** {@code XverseModel.set_gguf_parameters} (xverse.py:55-59): the inherited keys, then the {@code Meta AI original pth} tensor data layout and the RoPE width {@code hidden_size / num_attention_heads}. */
    public static void xverseMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.TENSOR_DATA_LAYOUT, "Meta AI original pth");
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT,
                cfg.requireInt("hidden_size") / cfg.requireInt("num_attention_heads"));
    }

    /** {@code XverseModel.set_vocab} (xverse.py:19-54): the Llama-HF vocabulary of {@code tokenizer.json} — the byte tokens stay in the token list under their {@code <0x00>} spelling (xverse.py:31-36). */
    public static Map<String, Object> xverseVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.llamaHf(HfHookSupport.vocabInput(ctx), pre);
    }

    /** {@code XverseModel.modify_tensors} (xverse.py:61-72): the {@code q_proj} and {@code k_proj} weights reverse the HF permutation with the {@code n_head //= n_kv_head} rule ({@code _reverse_hf_permute}, {@link RowTransforms#reversePermute}, xverse.py:74-86); every other tensor passes on. */
    public static void xverseTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        int heads = cfg.requireInt("num_attention_heads");
        int kvHeads = cfg.findInt(heads, "num_key_value_heads");
        RowSource data = src;
        if (hfName.endsWith("q_proj.weight")) {
            data = RowTransforms.reversePermute(src, heads, heads);
        } else if (hfName.endsWith("k_proj.weight")) {
            data = RowTransforms.reversePermute(src, heads, kvHeads);
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, data, out);
    }

    /** The block count of {@code GraniteSwitchModel.__init__} (granite.py:242-259): the highest {@code (model.)layers.<id>.} tensor block id plus one, the weightless switch layer(s) of {@code num_hidden_layers} left out. @throws GgufFormatException when the checkpoint holds no {@code layers.N} tensor name */
    public static int graniteSwitchBlocks(HfLoadContext ctx) {
        return graniteSwitchWeightLayers(ctx.index().names(), ctx);
    }

    /** The weight-layer count of the tensor names, per granite.py:242-259. */
    static int graniteSwitchWeightLayers(List<String> names, HfLoadContext ctx) {
        int max = -1;
        for (String name : names) {
            int i = 0;
            String n = name;
            if (n.startsWith("model.")) {
                i = 6;
                n = n.substring(i);
            }
            if (!n.startsWith("layers.")) {
                continue;
            }
            int from = 7;
            int end = n.indexOf('.', from);
            if (end < 0) {
                continue;
            }
            try {
                max = Math.max(max, Integer.parseInt(n.substring(from, end)));
            } catch (NumberFormatException ignore) {
                // not a block id
            }
        }
        if (max < 0) {
            // the names list is the input; the load context is not part of this walk
            throw new GgufFormatException("graniteswitch: no 'model.layers.N' tensors found in checkpoint");
        }
        return max + 1;
    }

    /** {@code GraniteSwitchModel.set_gguf_parameters} (granite.py:268-286): the GraniteMoe keys, then the pinned zero expert count of the dense checkpoints, the adapter count and LoRA rank, the activate and substitute token ids, the {@code granite-switch} decision type when an adapter is named {@code decision}, and the control-token router gain (15, by default at granite.py:283). */
    public static void graniteSwitchMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGraniteHooks.moeMeta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        int adapters = cfg.requireInt("num_adapters");
        if (cfg.findInt(0, "num_local_experts") == 0) {
            kv.set(GgufKeys.EXPERT_USED_COUNT, 0);
        }
        kv.set(GgufKeys.ADAPTER_COUNT, adapters);
        kv.set(GgufKeys.ADAPTER_LORA_RANK, cfg.requireInt("max_lora_rank"));
        cfg.findIntList("adapter_token_ids").ifPresent(v -> kv.set(GgufKeys.ADAPTER_TOKEN_IDS_ACTIVATE, v));
        cfg.findIntList("adapter_substitute_token_ids").ifPresent(v ->
                kv.set(GgufKeys.ADAPTER_TOKEN_IDS_SUBSTITUTE, v));
        if (decisionAdapter(ctx)) {
            kv.set(GgufKeys.DECISION_TYPE, "granite-switch");
        }
        kv.set(GgufKeys.ADAPTER_ROUTER_GAIN, cfg.findFloat(15.0, "control_token_gain"));
    }

    /** True when the checkpoint's {@code adapter_names} includes {@code decision}, per granite.py:287-289. */
    private static boolean decisionAdapter(HfLoadContext ctx) {
        Object flat = ctx.config().flat().get("adapter_names");
        if (!(flat instanceof List<?>)) {
            return false;
        }
        for (Object t : (List<?>) flat) {
            if ("decision".equals(String.valueOf(t))) {
                return true;
            }
        }
        return false;
    }

    /** {@code GraniteSwitchModel.modify_tensors} (granite.py:332-423): the weightless switch and control-token buffers (the {@code model.switch.} / {@code switch.} prefixes and the {@code adapter_token_ids} / {@code control_to_substitute_lut} tensors) are dropped (granite.py:341-347), the fused {@code self_attn.qkv_proj} base weight splits into permuted query / key / value rows over ATTN_QKV (granite.py:349-362), the fused {@code shared_mlp.input_linear} splits into gate and up (granite.py:386-399), the {@code o_proj} and {@code shared_mlp.output_linear} have their LoRA adapters, base tensors and splits map, the layer norms map over ATTN_NORM and FFN_NORM (granite.py:415-421), tied embeddings map, and the tied {@code lm_head.weight} is dropped; every other tensor is refused. @throws GgufFormatException for tensors the converter itself refuses, and when a LoRA slot ACL index or a fused split width does not fit the source */
    public static void graniteSwitchTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        int bid = HfLoadContext.bidOf(hfName);
        String bare = hfName.substring(hfName.lastIndexOf('.') + 1);
        if (hfName.startsWith("model.switch.") || hfName.startsWith("switch.")
                || bare.equals("adapter_token_ids") || bare.equals("control_to_substitute_lut")) {
            out.drop(hfName, "the weightless switch buffers are rebuilt at load time (granite.py:341-347)");
            return;
        }
        if (hfName.contains("self_attn.qkv_proj")) {
            if (hfName.endsWith("base_layer.weight")) {
                out.add(ctx.names().ggufName("ATTN_QKV", bid, ".weight"),
                        fusedQkv(ctx, src, cfg.requireInt("num_attention_heads"),
                                cfg.requireInt("num_key_value_heads")));
                return;
            }
            String slot = lastSegment(hfName);
            if (hfName.contains("lora_A_slices.") && slot.matches("[012]")) {
                out.add(ctx.names().ggufName(keyOf("ATTN_Q", "ATTN_K", "ATTN_V", Integer.parseInt(slot)), bid,
                        ".lora_a"), loraA(src));
                return;
            }
            if (hfName.contains("lora_B_slices.") && slot.matches("[012]")) {
                boolean qkv = Integer.parseInt(slot) < 2;
                String tensor = keyOf("ATTN_Q", "ATTN_K", "ATTN_V", Integer.parseInt(slot));
                int heads = Integer.parseInt(slot) == 0 ? cfg.requireInt("num_attention_heads")
                        : cfg.requireInt("num_key_value_heads");
                out.add(ctx.names().ggufName(tensor, bid, ".lora_b"),
                        loraB(src, qkv ? heads : null));
                return;
            }
            throw refused(bid);
        }
        if (hfName.contains("self_attn.o_proj")) {
            if (hfName.endsWith("base_layer.weight")) {
                out.add(ctx.names().ggufName("ATTN_OUT", bid, ".weight"), src);
                return;
            }
            if (hfName.endsWith("lora_A")) {
                out.add(ctx.names().ggufName("ATTN_OUT", bid, ".lora_a"), loraA(src));
                return;
            }
            if (hfName.endsWith("lora_B")) {
                out.add(ctx.names().ggufName("ATTN_OUT", bid, ".lora_b"), loraB(src, null));
                return;
            }
            throw refused(bid);
        }
        if (hfName.contains("shared_mlp.input_linear")) {
            int ffn = cfg.requireInt("shared_intermediate_size");
            if (hfName.endsWith("base_layer.weight")) {
                out.add(ctx.names().ggufName("FFN_GATE", bid, ".weight"), RowTransforms.sliceRows(src, 0, ffn));
                out.add(ctx.names().ggufName("FFN_UP", bid, ".weight"),
                        RowTransforms.sliceRows(src, ffn, src.rows() - ffn));
                return;
            }
            String slot = lastSegment(hfName);
            if (hfName.contains("lora_A_slices.") && slot.matches("[01]")) {
                String tensor = Integer.parseInt(slot) == 0 ? "FFN_GATE" : "FFN_UP";
                out.add(ctx.names().ggufName(tensor, bid, ".lora_a"), loraA(src));
                return;
            }
            if (hfName.contains("lora_B_slices.") && slot.matches("[01]")) {
                String tensor = Integer.parseInt(slot) == 0 ? "FFN_GATE" : "FFN_UP";
                out.add(ctx.names().ggufName(tensor, bid, ".lora_b"), loraB(src, null));
                return;
            }
            throw refused(bid);
        }
        if (hfName.contains("shared_mlp.output_linear")) {
            if (hfName.endsWith("base_layer.weight")) {
                out.add(ctx.names().ggufName("FFN_DOWN", bid, ".weight"), src);
                return;
            }
            if (hfName.endsWith("lora_A")) {
                out.add(ctx.names().ggufName("FFN_DOWN", bid, ".lora_a"), loraA(src));
                return;
            }
            if (hfName.endsWith("lora_B")) {
                out.add(ctx.names().ggufName("FFN_DOWN", bid, ".lora_b"), loraB(src, null));
                return;
            }
            throw refused(bid);
        }
        if (bid >= 0 && hfName.contains(".layers.") && hfName.contains("input_layernorm")) {
            out.add(ctx.names().ggufName("ATTN_NORM", bid, ".weight"), src);
            return;
        }
        if (bid >= 0 && hfName.contains(".layers.") && hfName.contains("post_attention_layernorm")) {
            out.add(ctx.names().ggufName("FFN_NORM", bid, ".weight"), src);
            return;
        }
        if ("model.embed_tokens.weight".equals(hfName) || "embed_tokens.weight".equals(hfName)) {
            out.add(ctx.names().mapOrThrow(hfName), src);
            return;
        }
        if ("model.norm.weight".equals(hfName) || "norm.weight".equals(hfName)) {
            out.add(ctx.names().mapOrThrow(hfName), src);
            return;
        }
        if ("lm_head.weight".equals(hfName)) {
            out.drop(hfName, "tied to token_embd (granite.py:422)");
            return;
        }
        throw refused(bid);
    }

    /** The unhandled-tensor refusal of granite.py:423. */
    private static GgufFormatException refused(int bid) {
        return new GgufFormatException("graniteswitch: unhandled tensor (bid=" + bid + ")");
    }

    /** The {@code lora_a} tensor of {@code _lora_a} (granite.py:310-314): the unit adapter dim of {@code [n_adapters, 1, max_rank, in]} squeezed away and the zero slot 0 prepended. */
    private static RowSource loraA(RowSource src) {
        long[] shape = src.shape();
        if (shape.length != 4 || shape[1] != 1) {
            throw new GgufFormatException(src.describe() + ": graniteswitch lora A must be [n_adapters, 1, max_rank,"
                    + " in], got " + Arrays.toString(shape));
        }
        int rank = (int) shape[2];
        int in = (int) shape[3];
        long nAdapters = shape[0];
        RowSource a = ElementTransforms.reshape(src, nAdapters, rank * (long) in);
        RowSource slot = zeroSlot(a, 1);
        return ElementTransforms.reshape(slot, nAdapters + 1, rank, in);
    }

    /** The {@code lora_b} tensor of {@code _lora_b} (granite.py:316-326): the unit dim squeezed, each adapter's output rows permuted like the permuted {@code permuteHeads} heads when given, and the zero slot 0 prepended. */
    private static RowSource loraB(RowSource src, Integer permuteHeads) {
        long[] shape = src.shape();
        if (shape.length != 4 || shape[1] != 1) {
            throw new GgufFormatException(src.describe() + ": graniteswitch lora B must be [n_adapters, 1, out,"
                    + " max_rank], got " + Arrays.toString(shape));
        }
        int outPerAdapter = (int) shape[2];
        int rank = (int) shape[3];
        long nAdapters = shape[0];
        RowSource b = ElementTransforms.reshape(src, nAdapters * outPerAdapter, rank);
        RowSource padded = RowTransforms.padRows(b, (nAdapters + 1) * outPerAdapter, 0.0f);
        int[] idx = new int[(int) ((nAdapters + 1) * outPerAdapter)];
        for (int slot = 0; slot <= (int) nAdapters; slot++) {
            int base = slot == 0 ? (int) nAdapters * outPerAdapter : (slot - 1) * outPerAdapter;
            for (int r = 0; r < outPerAdapter; r++) {
                if (permuteHeads != null && outPerAdapter % (2 * permuteHeads) == 0) {
                    int half = outPerAdapter / (2 * permuteHeads);
                    int h = r / (2 * half);
                    int rem = r % (2 * half);
                    idx[slot * outPerAdapter + r] = base + h * 2 * half + (rem % 2) * half + rem / 2;
                } else {
                    idx[slot * outPerAdapter + r] = base + r;
                }
            }
        }
        RowSource gathered = RowTransforms.gatherRows(padded, 1, idx);
        return ElementTransforms.reshape(gathered, nAdapters + 1, outPerAdapter, rank);
    }

    /** Pads dim 0 by one row of zero and gathers the zero slot in first (granite.py:312-313). */
    private static RowSource zeroSlot(RowSource src, int extraRows) {
        long rows = src.rows();
        int[] idx = new int[(int) rows + extraRows];
        for (int i = 0; i < extraRows; i++) {
            idx[i] = (int) rows + i;
        }
        for (int i = 0; i < rows; i++) {
            idx[extraRows + i] = i;
        }
        return RowTransforms.gatherRows(RowTransforms.padRows(src, rows + extraRows, 0.0f), 1, idx);
    }

    /** The q/k/v fused base of granite.py:349-362: the three row blocks of the {@code qSize | kvSize | kvSize} split, with the q and k blocks permuted for the ggml RoPE layout. */
    private static RowSource fusedQkv(HfLoadContext ctx, RowSource src, int heads, int kvHeads) {
        HfConfig cfg = ctx.config();
        int headDim = headDim(ctx, heads);
        RowSource[] parts = RowTransforms.splitRows(src, heads * headDim, kvHeads * headDim, kvHeads * headDim);
        return RowTransforms.concatRows(Arrays.asList(
                RowTransforms.permuteQk(parts[0], heads, heads),
                RowTransforms.permuteQk(parts[1], kvHeads, kvHeads),
                parts[2]));
    }

    /** {@code _head_dim} (granite.py:264-269): the projection or head dim, else hidden / heads. */
    private static int headDim(HfLoadContext ctx, int heads) {
        HfConfig cfg = ctx.config();
        return cfg.findInt("projection_head_dim", "head_dim")
                .orElseGet(() -> cfg.requireInt("hidden_size") / heads);
    }

    /** The one of {@code first, second, third} at {@code which}. */
    private static String keyOf(String first, String second, String third, int which) {
        if (which == 0) {
            return first;
        }
        return which == 1 ? second : third;
    }

    /** The last dot-separated segment of {@code name}. */
    private static String lastSegment(String name) {
        return name.substring(name.lastIndexOf('.') + 1);
    }
}
