package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfHookSupport.ExpertCollector;
import com.eonmux.neuralfree.gguf.hf.hooks.HfHookSupport.MergedName;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.names.HfTensorNameMap;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfSpecialVocab;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.TiktokenReader;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.Set;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The hooks of {@code DeepseekV2Model} (LC/conversion/deepseek.py:236-462), the class DeepSeek-V2/V3 and the
 * DeepSeek-OCR family convert with, and the base other MLA classes compose: the DEEPSEEK2 / DEEPSEEK2OCR resolver, the
 * vision tower filter, the MLA metadata, the per-layer expert stacking, the {@code kv_b_proj} split and the
 * vocabulary (GPT-2 BPE, or Kimi-K2's tiktoken vocabulary).
 */
public final class HfDeepseekHooks {

    /** The chat template the converter adds for the OCR models when the checkpoint has none (deepseek.py:255). */
    public static final String OCR_CHAT_TEMPLATE = "{% for m in messages %}{{m['content']}}{% endfor %}";

    /** The architectures converted as DEEPSEEK2OCR (deepseek.py:251). */
    private static final Set<String> OCR_ARCHITECTURES = Set.of("DeepseekOCRForCausalLM", "DeepseekOCR2ForCausalLM",
            "UnlimitedOCRForCausalLM");

    /** The merged expert parts, in the order the converter writes them (deepseek.py:416). */
    private static final List<String> EXPERT_PARTS = List.of("down_proj", "gate_proj", "up_proj");

    private static final String EXPERTS_STATE = "deepseek2.experts";

    /** {@code re.match(r"model.layers.(\d+)", name)} (deepseek.py:400), dots unescaped as in the converter. */
    private static final Pattern MODEL_LAYER = Pattern.compile("model.layers.(\\d+)");

    private static final double DEFAULT_RMS_EPS = 1e-6;
    private static final int DEFAULT_KV_LORA_RANK = 512;
    private static final int MAX_LAYER_DIGITS = 9;

    private HfDeepseekHooks() {
    }

    // ------------------------------------------------------------------ resolver and filter

    /**
     * {@code DeepseekV2Model.__init__} (deepseek.py:244-255): DEEPSEEK2OCR for the DeepSeek-OCR, DeepSeek-OCR-2 and
     * Unlimited-OCR architectures, DEEPSEEK2 otherwise.
     */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return OCR_ARCHITECTURES.contains(cfg.architecture()) ? Arch.DEEPSEEK2OCR : Arch.DEEPSEEK2;
    }

    /**
     * {@code DeepseekV2Model.filter_tensors} (deepseek.py:257-263): drops the DeepSeek-OCR vision encoders (the SAM
     * tower and DeepSeek-OCR-2's Qwen2 tower), then applies the inherited filter.
     */
    public static String filter(String name, UnaryOperator<String> base) {
        if (name.contains("sam_model") || name.contains("qwen2_model")) {
            return null;
        }
        return base.apply(name);
    }

    // ------------------------------------------------------------------ metadata

    /**
     * {@code DeepseekV2Model.set_gguf_parameters} (deepseek.py:319-388): the generic keys with the KV head count forced
     * to 1 (MLA stored as MQA; OCR keeps its own) and the RMS epsilon defaulting to 1e-6, then the leading dense block
     * count, vocabulary size, LoRA ranks, MLA key and value lengths (not for OCR), expert keys, RoPE dimension count,
     * OCR sliding window and the YaRN log multiplier {@code 0.1 * mscale_all_dim}.
     *
     * <p>The converter's OCR {@code rope_theta} default of 10000 (deepseek.py:323) is set after its RoPE parameters
     * were captured (base.py:1527-1538), so it never reaches {@code rope.freq_base}; this hook writes what the
     * converter writes.</p>
     *
     * @throws GgufFormatException when a required hyperparameter is missing
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        boolean ocr = ctx.arch() == Arch.DEEPSEEK2OCR;
        HfGenericMetadata.apply(ctx, kv);
        if (!ocr) {
            kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, 1);
        }
        if (!cfg.flat().containsKey("rms_norm_eps")) {
            kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, DEFAULT_RMS_EPS);
        }

        boolean hasMoe = cfg.find("n_routed_experts").isPresent();
        OptionalInt firstDense = cfg.findInt("first_k_dense_replace");
        int leadingDense = firstDense.isPresent() ? firstDense.getAsInt()
                : hasMoe ? 0 : cfg.requireInt("num_hidden_layers");
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, leadingDense);
        int kvLoraRank = cfg.findInt(DEFAULT_KV_LORA_RANK, "kv_lora_rank");
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        OptionalInt qLoraRank = cfg.findInt("q_lora_rank");
        if (qLoraRank.isPresent()) {
            kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, qLoraRank.getAsInt());
        }

        int ropeDim = cfg.requireInt("qk_rope_head_dim");
        if (!ocr) {
            kv.set(GgufKeys.ATTENTION_KV_LORA_RANK, kvLoraRank);
            kv.set(GgufKeys.ATTENTION_KEY_LENGTH, kvLoraRank + ropeDim);
            kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, kvLoraRank);
            kv.set(GgufKeys.ATTENTION_KEY_LENGTH_MLA, cfg.requireInt("qk_nope_head_dim") + ropeDim);
            kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_MLA, cfg.requireInt("v_head_dim"));
        }

        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size", "intermediate_size"));
        OptionalInt nRouted = cfg.findInt("n_routed_experts");
        if (nRouted.isPresent()) {
            kv.set(GgufKeys.EXPERT_COUNT, nRouted.getAsInt());
        }
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, cfg.findInt(0, "n_shared_experts"));
        OptionalDouble routedScale = cfg.findFloat("routed_scaling_factor");
        if (routedScale.isPresent()) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, routedScale.getAsDouble());
        }
        if (truthy(cfg.find("norm_topk_prob"))) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, true);
        }
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, ropeDim);

        if (ocr) {
            int window = cfg.findInt(0, "sliding_window_size");
            if (window == 0) {
                window = cfg.findInt(0, "sliding_window");
            }
            if (window != 0) {
                kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, window);
            }
        }

        Object mscaleAllDim = cfg.rope().raw().get("mscale_all_dim");
        if (mscaleAllDim != null) {
            kv.set(GgufKeys.ROPE_SCALING_YARN_LOG_MUL, 0.1 * number(cfg, "rope_parameters.mscale_all_dim",
                    mscaleAllDim));
        }
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code DeepseekV2Model.modify_tensors} (deepseek.py:390-452) with the class defaults {@code skip_mtp = True} and
     * {@code merge_expert = True} (deepseek.py:240-242): see {@link #tensors(HfLoadContext, String, RowSource,
     * TensorSink, boolean, boolean)}.
     *
     * @throws GgufFormatException as the full form does
     * @throws IOException         as the default mapping does
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        tensors(ctx, hfName, src, out, true, true);
    }

    /**
     * {@code DeepseekV2Model.modify_tensors} (deepseek.py:390-452) for a class with the given {@code skip_mtp} and
     * {@code merge_expert} attributes, the form the subclasses compose (DeepseekV32, Glm4MoeLite, GlmMoeDsa and Dots3
     * keep their MTP layers; HYV4 and Mistral 4 store their experts already stacked): drops a tied {@code lm_head};
     * with {@code skipMtp}, drops the multi-token-prediction layers ({@code model.layers.<b>} with
     * {@code b >= num_hidden_layers}); with {@code mergeExperts}, stacks the routed experts of each layer into
     * {@code down_proj}, {@code gate_proj} and {@code up_proj} tensors; splits {@code kv_b_proj} into the per-head
     * transposed {@code k_b_proj} and {@code v_b_proj} with the checkpoint's KV head count (the converter splits before
     * the head count is overridden for the metadata); maps every other tensor.
     *
     * @param skipMtp      {@code skip_mtp}: drop the layers past {@code num_hidden_layers}
     * @param mergeExperts {@code merge_expert}: stack the per-expert {@code mlp.experts} tensors
     * @throws GgufFormatException when a hyperparameter is missing, an expert tensor is not a member of a complete
     *                             expert set, or {@code kv_b_proj} does not have the MLA shape
     * @throws IOException         as the default mapping does
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out, boolean skipMtp,
                               boolean mergeExperts) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        if (cfg.findBool(false, "tie_word_embeddings")
                && (hfName.equals("lm_head.weight") || hfName.equals("model.lm_head.weight"))) {
            out.drop(hfName, "tied output layer; token_embd.weight serves as the output");
            return;
        }
        Matcher layer = MODEL_LAYER.matcher(hfName);
        if (skipMtp && layer.lookingAt() && atOrAbove(layer.group(1), cfg.requireInt("num_hidden_layers"))) {
            out.drop(hfName, "multi-token prediction layer; the converter skips MTP for this architecture");
            return;
        }
        if (mergeExperts && hfName.contains("mlp.experts")) {
            if (!experts(ctx).offer(ctx, hfName, out)) {
                throw new GgufFormatException("Unprocessed experts: tensor '" + hfName + "' is not an expert member"
                        + " <prefix>.experts.<x>.<part> with a part in " + EXPERT_PARTS);
            }
            return;
        }
        if (hfName.endsWith("kv_b_proj.weight")) {
            splitKvB(ctx, hfName, src, out);
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /**
     * {@code DeepseekV2Model.prepare_tensors} (deepseek.py:454-462): every expert set that was started must have been
     * stacked. A model without routed experts has none to check.
     *
     * @throws GgufFormatException naming the first incomplete expert set
     */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        if (ctx.config().findInt(0, "n_routed_experts") > 0) {
            experts(ctx).requireComplete();
        }
    }

    private static ExpertCollector experts(HfLoadContext ctx) {
        return ctx.state(EXPERTS_STATE, ExpertCollector.class, () -> new ExpertCollector(
                ctx.config().requireInt("n_routed_experts"), EXPERT_PARTS, MergedName.SAME_PREFIX));
    }

    /**
     * deepseek.py:434-450: {@code kv_b [n_head*(nope+v), kv_lora]} to {@code k_b [n_head, kv_lora, nope]} and
     * {@code v_b [n_head, v, kv_lora]}.
     */
    private static void splitKvB(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
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
        HfTensorNameMap names = ctx.names();
        out.add(names.mapOrThrow(hfName.replace("kv_b_proj", "k_b_proj")), kvB[0]);
        out.add(names.mapOrThrow(hfName.replace("kv_b_proj", "v_b_proj")), kvB[1]);
    }

    // ------------------------------------------------------------------ vocabulary

    /**
     * {@code DeepseekV2Model.set_vocab} (deepseek.py:265-317): {@code _set_vocab_gpt2} when the checkpoint has a
     * tokenizer it can read (a GGUF tokenizer source, tokenizer.json or vocab.json, or no tiktoken vocabulary at all),
     * else Kimi-K2's tiktoken vocabulary. The OCR models fall back to {@link #OCR_CHAT_TEMPLATE}.
     *
     * @throws IOException         when a tokenizer file cannot be read
     * @throws GgufFormatException when the tiktoken vocabulary's pre-tokenizer is not {@code kimi-k2}
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        String fallback = fallbackChatTemplate(ctx.arch());
        if (fallback != null) {
            in = in.withFallbackChatTemplate(fallback);
        }
        HfTokenizerFiles files = in.files();
        if (files.hasGgufTokenizer() || files.tokenizerJson() != null || files.vocabJson() != null
                || files.tiktokenFile() == null) {
            return HfVocabWriters.gpt2(in, pre);
        }
        return KimiVocab.build(in, pre);
    }

    /**
     * The chat template used when the checkpoint has none: {@link #OCR_CHAT_TEMPLATE} for DEEPSEEK2OCR
     * (deepseek.py:253-255), null otherwise.
     */
    public static String fallbackChatTemplate(Arch arch) {
        return arch == Arch.DEEPSEEK2OCR ? OCR_CHAT_TEMPLATE : null;
    }

    // ------------------------------------------------------------------ helpers

    private static boolean atOrAbove(String digits, int limit) {
        return digits.length() > MAX_LAYER_DIGITS || Integer.parseInt(digits) >= limit;
    }

    /** Python truthiness of a configuration value. */
    private static boolean truthy(Optional<Object> v) {
        if (!v.isPresent()) {
            return false;
        }
        Object o = v.get();
        if (o instanceof Boolean) {
            return (Boolean) o;
        }
        if (o instanceof Number) {
            return ((Number) o).doubleValue() != 0;
        }
        if (o instanceof String) {
            return !((String) o).isEmpty();
        }
        if (o instanceof Collection) {
            return !((Collection<?>) o).isEmpty();
        }
        return !(o instanceof Map) || !((Map<?, ?>) o).isEmpty();
    }

    private static double number(HfConfig cfg, String key, Object value) {
        if (!(value instanceof Number) || value instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must be a number, got " + value);
        }
        return ((Number) value).doubleValue();
    }

    /**
     * The Kimi-K2 tiktoken vocabulary of {@code set_vocab} (deepseek.py:276-315): each rank's byte-level text, the
     * merges that split a token into exactly two lower-ranked parts ({@code QwenModel.bpe}, qwen.py:28-43), and the
     * special tokens of Kimi's tokenizer code: the {@code added_tokens_decoder} texts, else
     * {@code <|reserved_token_<i>|>}, for the 256 ids after the mergeable ranks.
     */
    private static final class KimiVocab {

        private static final String PRE = "kimi-k2";
        private static final int RESERVED_SPECIAL_TOKENS = 256;
        private static final int NORMAL = 1;
        private static final int CONTROL = 3;
        private static final int UNUSED = 5;
        private static final String[] BYTE_TEXT = byteText();

        private KimiVocab() {
        }

        static Map<String, Object> build(HfVocabWriters.Input in, PreNameResolver pre) throws IOException {
            HfTokenizerFiles files = in.files();
            Path file = files.tiktokenFile();
            TiktokenReader ranks = TiktokenReader.read(file);
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
            Map<String, Integer> specials = specials(files.readTokenizerConfig(), ranks.size());
            Map<String, Integer> all = new LinkedHashMap<>(vocab);
            all.putAll(specials);
            Map<Integer, String> reverse = new HashMap<>();
            for (Map.Entry<String, Integer> e : all.entrySet()) {
                reverse.put(e.getValue(), e.getKey());
            }
            Set<Integer> specialIds = new HashSet<>(specials.values());
            List<String> tokens = new ArrayList<>(vocabSize);
            List<Integer> types = new ArrayList<>(vocabSize);
            for (int i = 0; i < vocabSize; i++) {
                String token = reverse.get(i);
                if (token == null) {
                    tokens.add("[PAD" + i + "]");
                    types.add(UNUSED);
                } else {
                    tokens.add(token);
                    types.add(specialIds.contains(i) ? CONTROL : NORMAL);
                }
            }

            Map<String, Object> md = new LinkedHashMap<>();
            md.put(GgufKeys.TOKENIZER_MODEL, "gpt2");
            md.put(GgufKeys.TOKENIZER_LIST, Collections.unmodifiableList(tokens));
            md.put(GgufKeys.TOKENIZER_TOKEN_TYPE, Collections.unmodifiableList(types));
            md.put(GgufKeys.TOKENIZER_MERGES, Collections.unmodifiableList(merges));
            HfSpecialVocab.load(files, null, in.config().raw(), false, vocabSize).writeTo(md);
            md.put(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES, true);
            md.put(GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK, false);
            md.put(GgufKeys.NF_TOKENIZER_CLEAN_SPACES, false);

            Map<String, Object> candidate = new LinkedHashMap<>(md);
            candidate.put(GgufKeys.TOKENIZER_PRE, PRE);
            GGUFTokenizer tokenizer = new GGUFTokenizer(TokenizerVocab.fromMetadata(candidate));
            Function<String, int[]> encoder = text -> tokenizer.encode(text, false, true);
            String name = pre.resolve(null, encoder);
            if (!PRE.equals(name)) {
                throw new GgufFormatException(file + ": Deepseek pre-tokenizer '" + name
                        + "' is not supported yet (only kimi-k2 tiktoken vocabularies are)");
            }
            md.put(GgufKeys.TOKENIZER_PRE, PRE);
            md.putAll(HfChatTemplates.resolve(files, in.fallbackChatTemplate()));
            return Collections.unmodifiableMap(md);
        }

        /**
         * The special tokens of Kimi's tokenizer: for each id in {@code [nBase, nBase + 256)}, the
         * {@code added_tokens_decoder} text of the id, else {@code <|reserved_token_<id>|>}; a repeated text keeps the
         * last id.
         */
        static Map<String, Integer> specials(JsonView tokenizerConfig, int nBase) {
            Map<Integer, String> decoder = new HashMap<>();
            Object raw = tokenizerConfig.raw("added_tokens_decoder");
            if (raw != null && !(raw instanceof Map)) {
                throw new GgufFormatException(tokenizerConfig.where("added_tokens_decoder") + " must be an object");
            }
            if (raw != null) {
                for (Map.Entry<?, ?> e : ((Map<?, ?>) raw).entrySet()) {
                    int id = decoderId(tokenizerConfig, String.valueOf(e.getKey()));
                    Object entry = e.getValue();
                    Object content = entry instanceof Map ? ((Map<?, ?>) entry).get("content") : null;
                    if (!(content instanceof String)) {
                        throw new GgufFormatException(tokenizerConfig.where("added_tokens_decoder") + "." + id
                                + " has no string 'content'");
                    }
                    decoder.put(id, (String) content);
                }
            }
            Map<String, Integer> out = new LinkedHashMap<>();
            for (int id = nBase; id < nBase + RESERVED_SPECIAL_TOKENS; id++) {
                String text = decoder.get(id);
                out.put(text != null ? text : "<|reserved_token_" + id + "|>", id);
            }
            return out;
        }

        private static int decoderId(JsonView tokenizerConfig, String key) {
            try {
                int id = Integer.parseInt(key);
                if (id >= 0) {
                    return id;
                }
            } catch (NumberFormatException e) {
                throw new GgufFormatException(tokenizerConfig.where("added_tokens_decoder") + ": key '" + key
                        + "' is not a token id", e);
            }
            throw new GgufFormatException(tokenizerConfig.where("added_tokens_decoder") + ": key '" + key
                    + "' is not a token id");
        }

        /**
         * {@code QwenModel.bpe(ranks, token, max_rank)} (qwen.py:28-43): merges the lowest-ranked adjacent pair below
         * {@code maxRank} until none remains.
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

        /** {@code QwenModel.token_bytes_to_string} (qwen.py:22-26): each byte through {@code bytes_to_unicode}. */
        static String text(byte[] token) {
            StringBuilder sb = new StringBuilder(token.length);
            for (byte b : token) {
                sb.append(BYTE_TEXT[b & 0xFF]);
            }
            return sb.toString();
        }

        /**
         * GPT-2 {@code bytes_to_unicode}: the printable bytes {@code !..~}, {@code ¡..¬} and {@code ®..ÿ} map to
         * themselves, the others to U+0100 onward in byte order.
         */
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
