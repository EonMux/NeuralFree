package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.UnaryOperator;

/**
 * Hooks of the llama.cpp classic-dense converter classes of Appendix A table HK-D1, part A: {@code BaichuanModel}
 * (LC/conversion/baichuan.py:13-60), {@code BitnetModel} (bitnet.py:13-50), {@code BloomModel} (bloom.py:17-68),
 * {@code ChameleonModel} (chameleon.py:17-60), {@code ChatGLMModel} (chatglm.py:13-168) and {@code CodeShellModel}
 * (codeshell.py:8-22). {@code CogVLMModel} (cogvlm.py:34) overrides no method and the table registers it on the
 * {@link HfLlamaHooks} references with the fixed {@link Arch#COGVLM}.
 *
 * <p>Every method is a public static hook reference that the table composes and subclasses reuse where the converter
 * calls {@code super()}.</p>
 */
public final class HfDenseClassicHooksA {

    private HfDenseClassicHooksA() {
    }

    // ------------------------------------------------------------------ helpers

    /**
     * {@code TextModel.set_vocab} (LC/conversion/base.py): the GPT-2 vocabulary of the checkpoint's own tokenizer
     * files. Every classic-dense class without a {@code set_vocab} override takes this.
     */
    public static Map<String, Object> gpt2Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        if (ctx.tokenizerFiles().hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(ctx.tokenizerFiles());
        }
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    /** {@code self.hparams[key]} as a double, required, with a clear message. */
    static double requireDouble(HfConfig cfg, String key) {
        return cfg.findFloat(key)
                .orElseThrow(() -> new GgufFormatException(cfg.source() + ": required key '" + key
                        + "' is missing"));
    }

    static GgufFormatException miss(HfConfig cfg, String key, String wanted) {
        return new GgufFormatException(cfg.source() + ": '" + key + "' must hold " + wanted);
    }

    // ------------------------------------------------------------------ Baichuan

    /** The {@code W_pack} fused projection tail of {@code BaichuanModel.modify_tensors} (baichuan.py:27-41). */
    static final String W_PACK_TAIL = "self_attn.W_pack.weight";

    /**
     * {@code BaichuanModel.set_gguf_parameters} (baichuan.py:23-26): the inherited keys, then the Meta layout marker
     * and the RoPE width {@code hidden_size // num_attention_heads}.
     */
    public static void baichuanMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.TENSOR_DATA_LAYOUT, "Meta AI original pth");
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT,
                cfg.requireInt("hidden_size") / cfg.requireInt("num_attention_heads"));
    }

    /**
     * {@code BaichuanModel.modify_tensors} (baichuan.py:27-41): the first third of {@code W_pack}, the query part,
     * and the second third, the key part, go through the reverse permutation with the {@code n_head //= n_kv_head}
     * rule (baichuan.py:42-51); the last third passes on; everything else maps directly.
     */
    public static void baichuanTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith(W_PACK_TAIL)) {
            int bid = HfHookSupport.layerOf(hfName);
            HfConfig cfg = ctx.config();
            int heads = cfg.requireInt("num_attention_heads");
            int kvHeads = cfg.findInt(heads, "num_key_value_heads");
            RowSource[] thirds = QkvLayouts.splitQkv(src, QkvLayouts.Layout.THIRDS, heads, heads, 1);
            out.add(ctx.names().ggufName("ATTN_Q", bid, ".weight"),
                    RowTransforms.reversePermute(thirds[0], heads, kvHeads));
            out.add(ctx.names().ggufName("ATTN_K", bid, ".weight"),
                    RowTransforms.reversePermute(thirds[1], heads, kvHeads));
            out.add(ctx.names().ggufName("ATTN_V", bid, ".weight"), thirds[2]);
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ BitNet

    /** The projection parts {@code BitnetModel.modify_tensors} re-quantizes (bitnet.py:35-40). */
    private static final String[] BITNET_PROJ_PARTS =
            {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "up_proj.weight",
                    "down_proj.weight", "gate_proj.weight"};

    /**
     * {@code BitnetModel.set_gguf_parameters} (bitnet.py:23-25): the inherited keys, then the linear RoPE with a
     * factor of 1 (unchanged positions).
     */
    public static void bitnetMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "linear");
        kv.set(GgufKeys.ROPE_SCALING_FACTOR, 1.0);
    }

    /**
     * {@code BitnetModel.modify_tensors} (bitnet.py:35-50): every listed projection of a float checkpoint goes
     * through the ternary re-quantization {@code weight_quant} (bitnet.py:24-33), as
     * {@link ElementTransforms#ternaryRequant}; a quantized checkpoint built with {@code quant_method: bitnet} is
     * left to the quantization rewrite (QR-1's BITNET source).
     */
    public static void bitnetTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        for (String part : BITNET_PROJ_PARTS) {
            if (hfName.endsWith(part)) {
                if (src.format().isFloat()) {
                    HfTensorHook.DEFAULT.apply(ctx, hfName, ElementTransforms.ternaryRequant(src), out);
                } else {
                    HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
                }
                return;
            }
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ Bloom

    /** The fused QKV tail of Bloom and GPT-NeoX checkpoints (bloom.py:51; gptneox.py:43). */
    public static final String QKV_TAIL = "query_key_value";

    /**
     * {@code BloomModel.modify_tensors}' QKV parts (bloom.py:51-71), shared with {@code GPTNeoXModel}
     * (gptneox.py:43-63): a fused {@code query_key_value} weight or bias of the Bloom layout
     * {@code [nHead, 3, headDim, ...]} (LC/conversion/bloom.py:54-81) is re-ordered to {@code [Q; K; V]} and emitted
     * under the original name, which keeps its GGUF mapping; everything else goes through the inherited filters.
     *
     * @throws GgufFormatException when the fused tensor does not hold {@code 3 * n_head} heads of the head width
     */
    public static void interleavedQkvTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName) || !hfName.contains("." + QKV_TAIL + ".")) {
            return;
        }
        HfConfig cfg = ctx.config();
        int heads = cfg.findInt(cfg.requireInt("num_attention_heads"), "n_head", "num_attention_heads");
        int hidden = cfg.findInt(cfg.requireInt("hidden_size"), "n_embed", "hidden_size");
        int headDim = hidden / heads;
        RowSource regrouped = QkvLayouts.regroupQkv(src, QkvLayouts.Layout.PER_HEAD_INTERLEAVED, heads, heads,
                headDim);
        if (!out.isConsumed(hfName)) {
            out.add(ctx.names().mapOrThrow(hfName), regrouped);
        }
    }

    // ------------------------------------------------------------------ Chameleon

    /** The image-tokenizer prefix {@code ChameleonModel.filter_tensors} drops (chameleon.py:29-31). */
    public static final String VQ_PREFIX = "model.vqmodel";

    /** {@code ChameleonModel.filter_tensors} (chameleon.py:29-31): the {@code model.vqmodel} prefix is dropped. */
    public static String chameleonFilter(String name, UnaryOperator<String> base) {
        if (name.startsWith(VQ_PREFIX)) {
            return null;
        }
        return base.apply(name);
    }

    /**
     * {@code ChameleonModel.set_gguf_parameters} (chameleon.py:22-23): the inherited keys, then {@code swin_norm}
     * (default false).
     */
    public static void chameleonMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.SWIN_NORM, ctx.config().findBool(false, "swin_norm"));
    }

    /**
     * {@code ChameleonModel.modify_tensors} (chameleon.py:35-44): q/k weights and biases get the Llama permutation,
     * and the q/k norm weights the reverse head permute of chameleon.py:47-54 expanded over the head count.
     */
    public static void chameleonTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        int heads = cfg.requireInt("num_attention_heads");
        int kvHeads = cfg.findInt(heads, "num_key_value_heads");
        RowSource data = src;
        if (hfName.endsWith("q_proj.weight") || hfName.endsWith("q_proj.bias")) {
            data = RowTransforms.permuteQk(data, heads, heads);
        } else if (hfName.endsWith("k_proj.weight") || hfName.endsWith("k_proj.bias")) {
            data = RowTransforms.permuteQk(data, heads, kvHeads);
        } else if (hfName.endsWith("q_norm.weight")) {
            data = expandedNorm(cfg, data, heads);
        } else if (hfName.endsWith("k_norm.weight")) {
            data = expandedNorm(cfg, data, kvHeads);
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, data, out);
    }

    /**
     * {@code ChameleonModel._reverse_hf_permute} (chameleon.py:47-54) over {@code nHeads}: row 0 acts as a
     * {@code [2, headDim / 2]} table transposed into the GGUF interleaved order and repeated once per head, since a
     * Chameleon Q/K norm covers every head.
     */
    private static RowSource expandedNorm(HfConfig cfg, RowSource src, int nHeads) {
        int headDim = cfg.requireInt("hidden_size") / cfg.requireInt("num_attention_heads");
        long[] shape = src.shape();
        if (shape.length != 2 || shape[0] != 1 || shape[1] != headDim) {
            throw new GgufFormatException("chameleon q/k norm '" + src.describe() + "' must be [1, " + headDim
                    + "], got " + Arrays.toString(shape));
        }
        int half = headDim / 2;
        int[] order = new int[headDim];
        for (int i = 0; i < half; i++) {
            order[2 * i] = i;
            order[2 * i + 1] = half + i;
        }
        RowSource one = ElementTransforms.reshape(ElementTransforms.gatherInner(src, order), 1, headDim);
        return ExpertTransforms.stack(Collections.nCopies(nHeads, one));
    }

    // ------------------------------------------------------------------ ChatGLM

    /** The class filter of {@code ChatGLMModel} (chatglm.py:163-168). */
    public static final HfNameFilter CHATGLM_FILTER = HfDenseClassicHooksA::chatglmFilter;

    /**
     * {@code ChatGLMModel.filter_tensors} (chatglm.py:163-168): the rotated position embedding table is dropped and
     * the {@code transformer.} prefix removed before the inherited filters.
     */
    public static String chatglmFilter(String name, UnaryOperator<String> base) {
        if (name.endsWith(".rotary_pos_emb.inv_freq")) {
            return null;
        }
        return base.apply(HfHookSupport.stripPrefixes(name, "transformer."));
    }

    /**
     * {@code ChatGLMModel.set_gguf_parameters} (chatglm.py:128-152), the hand-set keys: the alias chains
     * {@code hidden_size|n_embed}, {@code n_head|num_attention_heads},
     * {@code multi_query_group_num|num_key_value_heads|n_head}, the context {@code seq_length} (or the width), the
     * FFN width {@code ffn_hidden_size|intermediate_size|4 * width}, the RMS epsilon
     * {@code layernorm_epsilon|1e-5}, the RoPE width {@code attention_dim|hidden // n_head} times the partial rotary
     * factor (0.5), no BOS token, and the base scaled by {@code rope_ratio}.
     *
     * @throws GgufFormatException when a required value is missing or has the wrong type
     */
    public static void chatglmMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int nEmbed = cfg.requireInt("hidden_size", "n_embed");
        int heads = cfg.requireInt("n_head", "num_attention_heads");
        int kvHeads = cfg.findInt(heads, "multi_query_group_num", "num_key_value_heads", "n_head");
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.findInt(nEmbed, "seq_length"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, nEmbed);
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.findInt(4 * nEmbed, "ffn_hidden_size", "intermediate_size"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, heads);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, kvHeads);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, cfg.findFloat(1e-5, "layernorm_epsilon"));
        int ropeDim = cfg.findInt(nEmbed / heads, "attention_dim");
        double partial = cfg.rope().global().partialRotaryFactor().orElse(0.5);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) (ropeDim * partial));
        kv.set(GgufKeys.TOKENIZER_ADD_BOS, false);
        kv.set(GgufKeys.ROPE_FREQ_BASE, 10000.0 * cfg.findFloat(1.0, "rope_ratio"));
    }

    /**
     * {@code ChatGLMModel.set_vocab} (chatglm.py:63-97) for a {@code chatglm3} checkpoint: the SentencePiece
     * vocabulary with the {@code chatglm-spm} pre-tokenizer and the forced ids 0, 1 and 2 of the unknown, bos and
     * eos entries.
     */
    public static Map<String, Object> chatglm3Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.sentencePiece(
                HfHookSupport.vocabInput(ctx), pre));
        md.put(GgufKeys.TOKENIZER_PRE, "chatglm-spm");
        md.put(GgufKeys.TOKENIZER_UNK_ID, 0);
        md.put(GgufKeys.TOKENIZER_BOS_ID, 1);
        md.put(GgufKeys.TOKENIZER_EOS_ID, 2);
        return md;
    }

    // ------------------------------------------------------------------ CodeShell

    /**
     * {@code CodeShellModel.set_gguf_parameters} (codeshell.py:10-22), hand-set: the {@code n_positions} context,
     * the {@code n_embd} width and 4x FFN, the {@code n_head} heads over {@code num_query_groups} KV heads, the
     * {@code layer_norm_epsilon} epsilon, and the fixed 10000.0 linear scaling.
     */
    public static void codeshellMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int nEmb = cfg.requireInt("n_embd");
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("n_positions"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, nEmb);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, 4 * nEmb);
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, cfg.requireInt("n_head"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, cfg.requireInt("num_query_groups"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, requireDouble(cfg, "layer_norm_epsilon"));
        kv.set(GgufKeys.ROPE_FREQ_BASE, 10000.0);
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "linear");
        kv.set(GgufKeys.ROPE_SCALING_FACTOR, 1.0);
    }
}
