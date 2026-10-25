package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfSpecialVocab;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.TiktokenReader;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;

/**
 * The KDA (Kimi Delta Attention) hybrid converter classes (LC/conversion): {@code KimiLinearModel}
 * (kimi_linear.py:17-224) and {@code KimiK3Model} (kimi_k3.py:19-345). Both share the hybrid MLA + KDA skeleton:
 * the KDA conv reshape to {@code (1, d_inner, 1, d_conv)}; {@code A_log} to {@code -exp(A_log)}, K3 keeping only the
 * first {@code num_attention_heads} entries; {@code dt_bias} renamed to {@code dt_proj.bias}; the MLA
 * {@code kv_b_proj} split into the per-head transposed {@code k_b} and {@code v_b}; and the per-expert
 * {@code w1}/{@code w2}/{@code w3} stacking. The vocabulary is GPT-2 BPE when the checkpoint ships one, else
 * Kimi's tiktoken vocabulary with pre {@code kimi-k2} (kimi_linear.py:34-76). The Kimi-K3 part — the
 * attention-residual-score fusions (kimi_k3.py:40-44, :242-267), the {@code g_proj} layer-type resolution to
 * {@code ssm_g} on KDA layers and to {@code attn_gate} on full-attention layers over the 1-based
 * {@code full_attn_layers} list, the situ activation and the bundled Kimi-K3.jinja over the compressed-tensors
 * {@code mxfp4-pack-quantized} experts — is {@link HfKdaHooksB}.
 */
public final class HfKdaHooks {

    /** A resolver to {@link Arch#KIMI_LINEAR}. */
    public static HfArchResolver linearArch() {
        return HfArchResolver.of(Arch.KIMI_LINEAR);
    }

    /** The per-expert parts in the merge order the converter writes them (kimi_k3.py:321-323). */
    static final List<String> EXPERT_PARTS = Collections.unmodifiableList(Arrays.asList("w1", "w2", "w3"));

    private static final String WEIGHT = ".weight";
    private static final String EXPERTS_STATE = "kda.experts";

    private HfKdaHooks() {
    }

    // ------------------------------------------------------------------ Kimi-Linear

    /**
     * {@code KimiLinearModel.set_gguf_parameters} (kimi_linear.py:78-142): the KV head count forced to 1 (MLA is
     * served as MQA with one large head), the vocabulary size, the per-layer KV head array (1 on every
     * {@code full_attn_layers} layer, 0 on the KDA layers), the optional KDA conv kernel and head dim, the MLA ranks
     * and lengths over both Hugging Face and internal alias names, and the MoE sizes.
     */
    public static void linearMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        oneKvHead(kv);
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kvHeadArray(ctx, kv);
        Map<String, ?> lin = nested(cfg, "linear_attn_config");
        memberInt(lin, "short_conv_kernel_size").ifPresent(v -> kv.set(GgufKeys.SSM_CONV_KERNEL, v));
        memberInt(lin, "head_dim").ifPresent(v -> kv.set(GgufKeys.KDA_HEAD_DIM, v));
        OptionalInt qLora = cfg.findInt("q_lora_rank", "n_lora_q");
        if (qLora.isPresent()) {
            kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, qLora.getAsInt());
        }
        int kvLora = cfg.requireInt("kv_lora_rank", "n_lora_kv");
        kv.set(GgufKeys.ATTENTION_KV_LORA_RANK, kvLora);
        int rope = cfg.requireInt("qk_rope_head_dim", "n_rot");
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, rope);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, kvLora + rope);
        int nope = cfg.findInt(0, "qk_nope_head_dim");
        if (nope > 0) {
            kv.set(GgufKeys.ATTENTION_KEY_LENGTH_MLA, nope + rope);
        }
        int v = Math.max(cfg.findInt(0, "n_embd_head_v_mla"), cfg.findInt(0, "v_head_dim"));
        if (v > 0) {
            kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_MLA, v);
        }
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, cfg.requireInt("num_shared_experts"));
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, cfg.requireInt("first_k_dense_replace"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, cfg.findFloat(2.446, "routed_scaling_factor"));
    }

    /** {@code KimiLinearModel.modify_tensors} (kimi_linear.py:151-224) without K3's fusions. */
    public static void linearTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("block_sparse_moe.experts")) {
            experts(ctx, ctx.config().requireInt("num_local_experts", "num_experts")).offer(ctx, hfName, out);
            return;
        }
        kdaCommon(ctx, hfName, src, out);
    }

    /** {@code KimiLinearModel.prepare_tensors} (kimi_linear.py:144-149): no leftover expert group. */
    public static void linearExtra(HfLoadContext ctx, TensorSink out) {
        completeExperts(ctx);
    }

    /**
     * {@code KimiLinearModel.set_vocab} (kimi_linear.py:23-76): GPT-2 BPE when the checkpoint has a tokenizer, else
     * the Kimi tiktoken vocabulary.
     */
    public static Map<String, Object> linearVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return kimiVocab(ctx, HfHookSupport.vocabInput(ctx), pre);
    }

    // ------------------------------------------------------------------ shared KDA tensor parts

    /** The tensor antecedents both classes perform before their MoE or split work. */
    static void kdaCommon(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        String name = hfName;
        RowSource s = src;
        if (name.endsWith(".q_conv1d.weight") || name.endsWith(".k_conv1d.weight")
                || name.endsWith(".v_conv1d.weight")) {
            s = convWeight(s, name);
        } else if (name.endsWith(".A_log")) {
            s = aLog(ctx, s);
        } else if (name.endsWith(".dt_bias")) {
            name = HfHookSupport.renameTail(name, ".dt_bias", ".dt_proj.bias");
        } else if (name.endsWith("kv_b_proj.weight")) {
            splitKvB(ctx, name, s, out);
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, name, s, out);
    }

    static RowSource convWeight(RowSource src, String hfName) {
        long[] shape = src.shape();
        long dInner;
        long dConv;
        if (shape.length == 3) {
            dInner = shape[0];
            dConv = shape[2];
        } else if (shape.length == 2) {
            dInner = shape[0];
            dConv = shape[1];
        } else {
            throw new GgufFormatException("unexpected conv1d rank " + shape.length + " for '" + hfName + "'");
        }
        return ElementTransforms.reshape(src, 1, dInner, 1, dConv);
    }

    static RowSource aLog(HfLoadContext ctx, RowSource src) {
        int nHead = ctx.config().requireInt("num_attention_heads");
        long[] shape = src.shape();
        if (shape.length > 0 && shape[0] > nHead) {
            return ElementTransforms.negExp(RowTransforms.sliceRows(src, 0, nHead));
        }
        return ElementTransforms.negExp(src);
    }

    static void splitKvB(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        HfConfig cfg = ctx.config();
        long[] shape = src.shape();
        if (shape.length != 2) {
            throw new GgufFormatException("tensor '" + hfName + "' has shape " + Arrays.toString(shape)
                    + "; kv_b_proj must have rank 2");
        }
        int nHead = cfg.requireInt("num_key_value_heads");
        int vHead = cfg.requireInt("v_head_dim");
        int qkNope = cfg.requireInt("qk_nope_head_dim");
        RowSource[] kvB;
        try {
            kvB = QkvLayouts.mlaKvB(src, nHead, qkNope, vHead, Math.toIntExact(shape[1]));
        } catch (IllegalArgumentException | ArithmeticException e) {
            throw new GgufFormatException("tensor '" + hfName + "' does not split into " + nHead + " heads of "
                    + qkNope + " key and " + vHead + " value rows: " + e.getMessage(), e);
        }
        out.add(ctx.names().mapOrThrow(hfName.replace("kv_b_proj", "k_b_proj")), kvB[0]);
        out.add(ctx.names().mapOrThrow(hfName.replace("kv_b_proj", "v_b_proj")), kvB[1]);
    }

    /** The per-layer KV head array: the latent MQA head count on full-attention layers, 0 on KDA layers. */
    static void kvHeadArray(HfLoadContext ctx, HfMetadataBuilder kv) {
        HfConfig cfg = ctx.config();
        int nLayer = cfg.requireInt("num_hidden_layers");
        Set<Integer> full = fullAttnLayers(cfg);
        int perLayer = cfg.findInt(1, "num_key_value_heads");
        List<Integer> arr = new ArrayList<>(nLayer);
        for (int il = 0; il < nLayer; il++) {
            arr.add(full.contains(il + 1) ? perLayer : 0);
        }
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, arr);
    }

    /** The 1-based {@code linear_attn_config.full_attn_layers} set. */
    static Set<Integer> fullAttnLayers(HfConfig cfg) {
        Object v = nested(cfg, "linear_attn_config").get("full_attn_layers");
        if (!(v instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": 'linear_attn_config.full_attn_layers' must be an"
                    + " array of 1-based full-attention layer numbers");
        }
        Set<Integer> out = new HashSet<>();
        for (Object o : (List<?>) v) {
            if (o instanceof Integer || o instanceof Long) {
                int i = ((Number) o).intValue();
                if (i >= 0) {
                    out.add(i);
                    continue;
                }
            }
            throw new GgufFormatException(cfg.source() + ": 'linear_attn_config.full_attn_layers' holds the"
                    + " non-1-based layer number " + o);
        }
        return out;
    }

    /** The expert groups complete when no per-part collector is pending. */
    static void completeExperts(HfLoadContext ctx) {
        Experts e = ctx.state(EXPERTS_STATE, Experts.class, Experts::new);
        synchronized (e) {
            if (e.collector != null) {
                e.collector.requireComplete();
            }
        }
    }

    static HfHookSupport.ExpertCollector experts(HfLoadContext ctx, int nExperts) {
        Experts e = ctx.state(EXPERTS_STATE, Experts.class, Experts::new);
        synchronized (e) {
            if (e.collector == null) {
                e.collector = new HfHookSupport.ExpertCollector(nExperts, EXPERT_PARTS,
                        HfHookSupport.MergedName.SAME_PREFIX);
            }
            return e.collector;
        }
    }

    /** The per-load expert collector of the KDA classes. */
    private static final class Experts {
        HfHookSupport.ExpertCollector collector;
    }

    // ------------------------------------------------------------------ shared small helpers

    /** The one-KV-head MQA metadata of the MLA-served KDA architectures. */
    static void oneKvHead(HfMetadataBuilder kv) {
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, 1);
    }

    @SuppressWarnings("unchecked")
    static Map<String, ?> nested(HfConfig cfg, String key) {
        Object v = cfg.find(key).orElse(null);
        if (!(v instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": required key '" + key + "' is missing"
                    + (v == null ? "" : " or is not an object"));
        }
        return (Map<String, ?>) v;
    }

    private static OptionalInt memberInt(Map<String, ?> map, String key) {
        Object v = map.get(key);
        if (v instanceof Integer || v instanceof Long) {
            return OptionalInt.of(((Number) v).intValue());
        }
        return OptionalInt.empty();
    }

    // ------------------------------------------------------------------ the Kimi tiktoken vocabulary

    /**
     * Kimi's vocabulary (kimi_linear.py:23-76): GPT-2 BPE when a tokenizer.json / vocab.json / a GGUF source is
     * readable or the checkpoint has no tiktoken file, else the tiktoken vocabulary; a tiktoken checkpoint whose
     * pre-tokenizer differs from {@code kimi-k2} raises a named error.
     */
    static Map<String, Object> kimiVocab(HfLoadContext ctx, HfVocabWriters.Input in, PreNameResolver pre)
            throws IOException {
        HfTokenizerFiles files = in.files();
        if (files.hasGgufTokenizer() || files.tokenizerJson() != null || files.vocabJson() != null
                || files.tiktokenFile() == null) {
            return HfVocabWriters.gpt2(in, pre);
        }
        return KimiTiktoken.build(ctx, in, pre);
    }

    /** The implementation of kimi_linear.py:34-76 over {@code kimi-k2} tiktoken vocabularies. */
    private static final class KimiTiktoken {
        private static final String PRE = "kimi-k2";
        private static final String[] BYTE_TEXT = byteText();

        private KimiTiktoken() {
        }

        static Map<String, Object> build(HfLoadContext ctx, HfVocabWriters.Input in, PreNameResolver pre)
                throws IOException {
            HfTokenizerFiles files = in.files();
            TiktokenReader ranks = TiktokenReader.read(files.tiktokenFile());
            int vocabSize = in.config().requireInt("vocab_size");
            Map<String, Integer> vocab = new LinkedHashMap<>();
            List<String> merges = new ArrayList<>();
            for (int i = 0; i < ranks.size(); i++) {
                byte[] token = ranks.token(i);
                int rank = ranks.rank(i);
                vocab.put(text(token), rank);
                if (token.length == 1) {
                    continue;
                }
                List<byte[]> merged = bpe(ranks, token, rank);
                if (merged.size() == 2) {
                    merges.add(text(merged.get(0)) + " " + text(merged.get(1)));
                }
            }
            Map<String, Integer> specials = specialsOf(files, ranks.size());
            Map<Integer, String> reverse = new HashMap<>();
            for (Map.Entry<String, Integer> e : vocab.entrySet()) {
                reverse.put(e.getValue(), e.getKey());
            }
            for (Map.Entry<String, Integer> e : specials.entrySet()) {
                reverse.put(e.getValue(), e.getKey());
            }
            Set<Integer> specialIds = new HashSet<>(specials.values());
            List<String> tokens = new ArrayList<>(vocabSize);
            List<Integer> types = new ArrayList<>(vocabSize);
            for (int i = 0; i < vocabSize; i++) {
                String token = reverse.get(i);
                if (token == null) {
                    tokens.add("[PAD" + i + "]");
                    types.add(5);
                } else {
                    tokens.add(token);
                    types.add(specialIds.contains(i) ? 3 : 1);
                }
            }
            Map<String, Object> md = new LinkedHashMap<>();
            md.put(GgufKeys.TOKENIZER_MODEL, "gpt2");
            md.put(GgufKeys.TOKENIZER_PRE, PRE);
            md.put(GgufKeys.TOKENIZER_LIST, Collections.unmodifiableList(tokens));
            md.put(GgufKeys.TOKENIZER_TOKEN_TYPE, Collections.unmodifiableList(types));
            md.put(GgufKeys.TOKENIZER_MERGES, Collections.unmodifiableList(merges));
            HfSpecialVocab.load(files, null, in.config().raw(), false, vocabSize).writeTo(md);
            md.putAll(HfChatTemplates.resolve(files, in.fallbackChatTemplate()));
            return Collections.unmodifiableMap(md);
        }

        /**
         * Kimi's special tokens (deepseek.py:280-287 analogue in the Kimi k2 tokenizer): for each id in
         * {@code [nBase, nBase + 256)}, the {@code added_tokens_decoder} text of the id, else
         * {@code <|reserved_token_<id>|>}.
         */
        private static Map<String, Integer> specialsOf(HfTokenizerFiles files, int nBase) throws IOException {
            JsonView tc = files.readTokenizerConfig();
            Object raw = tc.raw("added_tokens_decoder");
            Map<Integer, String> decoder = new HashMap<>();
            if (raw instanceof Map) {
                for (Map.Entry<?, ?> e : ((Map<?, ?>) raw).entrySet()) {
                    int id;
                    try {
                        id = Integer.parseInt(String.valueOf(e.getKey()));
                    } catch (NumberFormatException ex) {
                        throw new GgufFormatException(tc.where("added_tokens_decoder") + ": key '" + e.getKey()
                                + "' is not a token id");
                    }
                    if (id < 0) {
                        throw new GgufFormatException(tc.where("added_tokens_decoder") + ": key '" + e.getKey()
                                + "' is not a token id");
                    }
                    Object entry = e.getValue();
                    Object content = entry instanceof Map ? ((Map<?, ?>) entry).get("content") : null;
                    if (!(content instanceof String)) {
                        throw new GgufFormatException(tc.where("added_tokens_decoder") + "." + id
                                + " has no string 'content'");
                    }
                    decoder.put(id, (String) content);
                }
            } else if (raw != null) {
                throw new GgufFormatException(tc.where("added_tokens_decoder") + " must be an object");
            }
            Map<String, Integer> out = new LinkedHashMap<>();
            for (int id = nBase; id < nBase + 256; id++) {
                String textValue = decoder.get(id);
                out.put(textValue != null ? textValue : "<|reserved_token_" + id + "|>", id);
            }
            return out;
        }

        /**
         * {@code QwenModel.bpe(ranks, token, max_rank)} (qwen.py:38-43): merges the lowest-ranked adjacent pair
         * below {@code maxRank} until none remains.
         */
        static List<byte[]> bpe(TiktokenReader ranks, byte[] token, int maxRank) {
            List<byte[]> parts = new ArrayList<>(token.length);
            for (byte b : token) {
                parts.add(new byte[]{b});
            }
            while (true) {
                int minIdx = -1;
                int minRank = -1;
                for (int i = 0; i + 1 < parts.size(); i++) {
                    int rank = ranks.rankOf(concat(parts.get(i), parts.get(i + 1)));
                    if (rank >= 0 && (minRank < 0 || rank < minRank)) {
                        minIdx = i;
                        minRank = rank;
                    }
                }
                if (minRank < 0 || minRank >= maxRank) {
                    return parts;
                }
                parts.set(minIdx, concat(parts.get(minIdx), parts.get(minIdx + 1)));
                parts.remove(minIdx + 1);
            }
        }

        private static byte[] concat(byte[] a, byte[] b) {
            byte[] out = new byte[a.length + b.length];
            System.arraycopy(a, 0, out, 0, a.length);
            System.arraycopy(b, 0, out, a.length, b.length);
            return out;
        }

        /** {@code QwenModel.token_bytes_to_string} (qwen.py:22-26): GPT-2 {@code bytes_to_unicode}. */
        static String text(byte[] token) {
            StringBuilder sb = new StringBuilder(token.length);
            for (byte b : token) {
                sb.append(BYTE_TEXT[b & 0xFF]);
            }
            return sb.toString();
        }

        private static String[] byteText() {
            String[] table = new String[256];
            int n = 0;
            for (int b = 0; b < 256; b++) {
                boolean printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
                table[b] = new String(Character.toChars(printable ? b : 256 + n++));
            }
            return table;
        }
    }
}
