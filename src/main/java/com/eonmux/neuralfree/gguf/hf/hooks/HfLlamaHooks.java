package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.function.UnaryOperator;

/**
 * Hooks of llama.cpp's {@code LlamaModel} (LC/conversion/llama.py:33-360), the base of most text converter classes:
 * the SmolVLM name filter, the EAGLE3 architecture switch, the Llama metadata, the Q/K permutation and Mixtral expert
 * stacking of {@code modify_tensors}, the llama3 {@code rope_freqs} tensor and the {@code set_vocab} dispatch. The
 * EAGLE3 parts the class implements inline live in {@link HfEagle3Hooks}; these hooks call them when the checkpoint
 * resolves to {@link Arch#EAGLE3}.
 *
 * <p>Every method is a public static hook reference that subclasses compose where the converter calls
 * {@code super()}. {@link #vocabInput} is the vocabulary input every hook pack builds.</p>
 */
public final class HfLlamaHooks {

    /** The headless Llama class, whose tensors lack the {@code model.} prefix (llama.py:258-259). */
    public static final String LLAMA_HEADLESS = "LlamaModel";
    /** The SmolVLM2 text class whose configuration may omit {@code num_attention_heads} (llama.py:39-41). */
    public static final String VLLAMA3 = "VLlama3ForCausalLM";
    /** The root architecture whose text model takes the GLM-Edge vocabulary (llama.py:115-116). */
    public static final String GLMASR = "GlmasrModel";
    /** The {@code tokenizer_class} of the HybridDNA tokenizer (LC/conversion/base.py:2176-2181). */
    public static final String HYBRID_DNA_TOKENIZER = "HybridDNATokenizer";
    /** The {@code num_attention_heads} VLlama3ForCausalLM defaults to (llama.py:41). */
    public static final int VLLAMA3_DEFAULT_HEADS = 32;

    /** The SmolVLM text-tower prefix the class filter removes (llama.py:204-205). */
    static final String SMOLVLM_TEXT_PREFIX = "text_model.";
    /** The vocabulary size {@code set_vocab} assumes when the configuration has none (llama.py:146, 158). */
    static final int DEFAULT_VOCAB_SIZE = 32000;
    /** The CodeLlama vocabulary size whose infill tokens are set (llama.py:145-155). */
    static final int CODELLAMA_VOCAB_SIZE = 32016;
    /** The {@code eot} id {@code set_vocab} falls back to for CodeLlama (llama.py:154). */
    static final int CODELLAMA_EOT_ID = 32010;
    /** The Granite small-model vocabulary size that turns {@code add_bos_token} off (llama.py:157-159). */
    static final int GRANITE_SMALL_VOCAB_SIZE = 49152;

    private static final String MODEL_PREFIX = "model.";
    private static final String EXPERTS_MARKER = "block_sparse_moe.experts";
    private static final List<String> MIXTRAL_PARTS = Collections.unmodifiableList(Arrays.asList("w1", "w2", "w3"));
    private static final String MIXTRAL_STATE = HfLlamaHooks.class.getName() + ".mixtralExperts";
    private static final String[] N_HEAD = {"n_heads", "num_attention_heads"};
    private static final String[] N_HEAD_KV = {"n_kv_heads", "num_key_value_heads"};

    /** {@link #filter} as an {@link HfNameFilter}, for entries and subclass filters that wrap it. */
    public static final HfNameFilter FILTER = HfLlamaHooks::filter;

    private HfLlamaHooks() {
    }

    // ------------------------------------------------------------------ configuration and architecture

    /**
     * {@code LlamaModel.__init__}'s configuration changes: the EAGLE3 {@code transformer_layer_config} flattening
     * ({@link HfEagle3Hooks#fixup}, llama.py:212-214), then the SmolVLM2 head-count default (llama.py:39-41).
     */
    public static HfConfig fixup(HfConfig cfg) {
        HfConfig out = HfEagle3Hooks.fixup(cfg);
        if (VLLAMA3.equals(out.architecture()) && !out.flat().containsKey("num_attention_heads")) {
            out = out.withFlat(Collections.singletonMap("num_attention_heads", VLLAMA3_DEFAULT_HEADS));
        }
        return out;
    }

    /**
     * {@link Arch#EAGLE3} for an EAGLE-3 draft (the configuration test of llama.py:49-50, on the flattened
     * configuration, or one of vLLM's EAGLE-3 layout names, {@link HfEagle3Hooks#VLLM_ALIASES}); else
     * {@link Arch#LLAMA}.
     *
     * @throws GgufFormatException when a configuration with {@code draft_vocab_size} has no layer count
     */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return HfEagle3Hooks.isEagle3(cfg) ? Arch.EAGLE3 : Arch.LLAMA;
    }

    /**
     * {@code LlamaModel.filter_tensors} (llama.py:200-206): SmolVLM's {@code text_model.} removed, then the inherited
     * filters.
     */
    public static String filter(String name, UnaryOperator<String> base) {
        return base.apply(name.contains(SMOLVLM_TEXT_PREFIX) ? name.replace(SMOLVLM_TEXT_PREFIX, "") : name);
    }

    // ------------------------------------------------------------------ metadata

    /**
     * {@code LlamaModel.set_gguf_parameters} (llama.py:165-174) after {@code TextModel.set_gguf_parameters}, and for
     * an EAGLE3 draft the keys its {@code __init__} writes ({@link HfEagle3Hooks#meta}, llama.py:76-104).
     *
     * @throws IOException when the EAGLE3 target cannot be opened
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        llamaKv(ctx, kv, true);
        if (ctx.arch() == Arch.EAGLE3) {
            HfEagle3Hooks.meta(ctx, kv, plan);
        }
    }

    /**
     * The keys {@code LlamaModel.set_gguf_parameters} adds (llama.py:169-174): {@code vocab_size}, except for a
     * Mistral-format checkpoint, and {@code rope.dimension_count}, which is {@code head_dim} when
     * {@code honorHeadDim} and the configuration has one, else {@code hidden_size / num_attention_heads}.
     *
     * @param honorHeadDim false for the classes that pop {@code head_dim} first (GraniteModel, granite.py:33-34)
     * @throws GgufFormatException when a required value is missing or the head count is not positive
     */
    public static void llamaKv(HfLoadContext ctx, HfMetadataBuilder kv, boolean honorHeadDim) {
        HfConfig cfg = ctx.config();
        if (!cfg.mistralNative()) {
            kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        }
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, ropeDim(cfg, honorHeadDim));
    }

    /** {@code head_dim} when honored and present, else {@code hidden_size // num_attention_heads}. */
    static int ropeDim(HfConfig cfg, boolean honorHeadDim) {
        OptionalInt headDim = honorHeadDim ? cfg.findInt("head_dim") : OptionalInt.empty();
        if (headDim.isPresent()) {
            return headDim.getAsInt();
        }
        int heads = cfg.requireInt("num_attention_heads");
        if (heads <= 0) {
            throw new GgufFormatException(cfg.source() + ": num_attention_heads must be positive, got " + heads);
        }
        return Math.floorDiv(cfg.requireInt("hidden_size"), heads);
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code LlamaModel.modify_tensors} (llama.py:233-297): for an EAGLE3 draft the renamed and special tensors of
     * {@link HfEagle3Hooks#tensor} first; then {@link #llamaTensor}.
     *
     * @throws IOException when the EAGLE3 {@code d2t} tensor or its target cannot be read
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName;
        if (ctx.arch() == Arch.EAGLE3) {
            name = HfEagle3Hooks.rename(hfName);
            if (HfEagle3Hooks.tensor(ctx, hfName, name, src, out)) {
                return;
            }
        }
        llamaTensor(ctx, hfName, name, src, out);
    }

    /**
     * The common path of {@code LlamaModel.modify_tensors} (llama.py:255-297): the {@code model.} prefix of the
     * headless {@code LlamaModel} (:258-259), the Q/K permutation when the entry undoes it (:261-265, the
     * {@link RowTransforms#permuteQk} of {@code LlamaModel.permute}, :176-182, which also covers the NVFP4 path of
     * {@code _repack_nvfp4}, :184-196), the Mixtral expert stacking into {@code layers.<b>.feed_forward.experts.w<k>}
     * (:268-295), and the name map for everything else.
     *
     * @param hfName the index name of the tensor
     * @param name   the name the converter sees (the index name after any class-level renames)
     * @throws GgufFormatException when a {@code block_sparse_moe.experts} tensor is not a Mixtral expert part, the
     *                             expert count is missing, or the name has no mapping
     */
    public static void llamaTensor(HfLoadContext ctx, String hfName, String name, RowSource src, TensorSink out) {
        String n = LLAMA_HEADLESS.equals(ctx.config().architecture()) ? MODEL_PREFIX + name : name;
        RowSource data = ctx.entry().undoPermute() ? permuteQk(ctx.config(), n, src) : src;
        if (n.contains(EXPERTS_MARKER)) {
            if (!mixtralExperts(ctx).offer(ctx, hfName, out)) {
                throw new GgufFormatException("tensor '" + hfName + "' is not a Mixtral expert weight"
                        + " (block_sparse_moe.experts.<x>.w1|w2|w3)");
            }
            return;
        }
        out.add(ctx.names().mapOrThrow(n), data);
    }

    /**
     * {@code src} through {@code LlamaModel.permute} when {@code name} is a Q or K projection weight or bias
     * (llama.py:261-265): Q with {@code n_head} groups, K with {@code n_kv_head}. {@code n_kv_head} defaults to
     * {@code n_head}, the value transformers' {@code AutoConfig} fills in for Llama configurations.
     *
     * @throws GgufFormatException when the head count is missing
     */
    public static RowSource permuteQk(HfConfig cfg, String name, RowSource src) {
        boolean q = name.endsWith("q_proj.weight") || name.endsWith("q_proj.bias");
        boolean k = name.endsWith("k_proj.weight") || name.endsWith("k_proj.bias");
        if (!q && !k) {
            return src;
        }
        int nHead = cfg.requireInt(N_HEAD);
        if (q) {
            return RowTransforms.permuteQk(src, nHead, nHead);
        }
        return RowTransforms.permuteQk(src, nHead, cfg.findInt(nHead, N_HEAD_KV));
    }

    /** The per-load Mixtral expert collector: parts {@code w1, w2, w3} of {@code num_local_experts} experts. */
    private static HfHookSupport.ExpertCollector mixtralExperts(HfLoadContext ctx) {
        int nExperts = ctx.config().requireInt("num_local_experts");
        return ctx.state(MIXTRAL_STATE, HfHookSupport.ExpertCollector.class,
                () -> new HfHookSupport.ExpertCollector(nExperts, MIXTRAL_PARTS,
                        (prefix, part, tail) -> "layers." + HfHookSupport.layerOf(prefix) + ".feed_forward.experts."
                                + part + tail));
    }

    /**
     * {@code LlamaModel.generate_extra_tensors} (llama.py:299-327): the llama3 {@code rope_freqs} factors
     * ({@link RopeTensors#llama3Freqs}). Also checks that every Mixtral layer whose experts were seen was stacked
     * ({@code prepare_tensors}, llama.py:356-360).
     *
     * @throws GgufFormatException when a layer's experts are incomplete
     */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        Optional<RowSource> freqs = RopeTensors.llama3Freqs(ctx.config());
        if (freqs.isPresent()) {
            out.add(ctx.names().ggufName("ROPE_FREQS", -1, ".weight"), freqs.get());
        }
        if (ctx.config().findInt(0, "num_local_experts") > 0) {
            mixtralExperts(ctx).requireComplete();
        }
    }

    // ------------------------------------------------------------------ vocabulary

    /**
     * {@code LlamaModel.set_vocab} (llama.py:106-163) on the checkpoint's own tokenizer files; an EAGLE3 draft takes
     * its target's tokenizer ({@link HfEagle3Hooks#vocab}, llama.py:107-113).
     *
     * @throws IOException when a tokenizer file or the EAGLE3 target cannot be read
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        if (ctx.arch() == Arch.EAGLE3) {
            return HfEagle3Hooks.vocab(ctx, pre);
        }
        return llamaVocab(ctx, pre, ctx.tokenizerFiles(), ctx.config());
    }

    /**
     * The dispatch of {@code LlamaModel.set_vocab} (llama.py:115-159) over {@code files}:
     * <ol>
     *   <li>a root architecture {@code GlmasrModel}: the GLM-Edge vocabulary;</li>
     *   <li>a Mistral-format checkpoint, or a {@code tekken.json} without {@code tokenizer.json}: the mistral_common
     *       vocabulary;</li>
     *   <li>otherwise tokenizer_config's {@code add_prefix_space} as {@code add_space_prefix}, then the HybridDNA,
     *       SentencePiece, Llama-HF or GPT-2 vocabulary, whose keys take precedence;</li>
     *   <li>for the SentencePiece, Llama-HF and GPT-2 vocabularies, the CodeLlama {@code eot} id and the Granite
     *       small-model {@code add_bos_token}.</li>
     * </ol>
     *
     * @param files       the tokenizer files the converter reads (the target's for an EAGLE3 draft)
     * @param vocabConfig the configuration the writers read: its root for the special-token ids of
     *                    {@code config.json} (the files' own), its flattened values for the vocabulary size (the
     *                    checkpoint's {@code hparams})
     * @throws IOException         when a tokenizer file cannot be read
     * @throws GgufFormatException when {@code add_prefix_space} is not a boolean
     */
    public static Map<String, Object> llamaVocab(HfLoadContext ctx, PreNameResolver pre, HfTokenizerFiles files,
                                                 HfConfig vocabConfig) throws IOException {
        HfConfig cfg = ctx.config();
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx, files, vocabConfig);
        if (GLMASR.equals(rootArchitecture(cfg))) {
            return HfVocabWriters.glmEdge(in, pre);
        }
        if (cfg.mistralNative() || (files.tekkenJson() != null && files.tokenizerJson() == null)) {
            return HfVocabWriters.mistral(in, pre);
        }
        JsonView tc = files.readTokenizerConfig();
        Map<String, Object> md = new LinkedHashMap<>();
        Object addPrefix = tc.raw("add_prefix_space");
        if (addPrefix != null) {
            if (!(addPrefix instanceof Boolean)) {
                throw new GgufFormatException(tc.where("add_prefix_space") + " must be a boolean");
            }
            md.put(GgufKeys.TOKENIZER_ADD_PREFIX, addPrefix);
        }
        if (HYBRID_DNA_TOKENIZER.equals(tc.raw("tokenizer_class"))) {
            md.putAll(HfVocabWriters.hybridDna(in, pre));
            return Collections.unmodifiableMap(md);
        }
        if (HfVocabWriters.canSentencePiece(in)) {
            md.putAll(HfVocabWriters.sentencePiece(in, pre));
        } else if (HfVocabWriters.canLlamaHf(in)) {
            md.putAll(HfVocabWriters.llamaHf(in, pre));
        } else {
            md.putAll(HfVocabWriters.gpt2(in, pre));
        }
        int vocabSize = cfg.findInt(DEFAULT_VOCAB_SIZE, "vocab_size");
        if (vocabSize == CODELLAMA_VOCAB_SIZE) {
            md.put(GgufKeys.TOKENIZER_EOT_ID, codeLlamaEot(files, vocabConfig.raw()));
        }
        if (vocabSize == GRANITE_SMALL_VOCAB_SIZE) {
            md.put(GgufKeys.TOKENIZER_ADD_BOS, false);
        }
        return Collections.unmodifiableMap(md);
    }

    /**
     * The {@code eot} id of llama.py:146-155's {@code SpecialVocab(special_token_types=['prefix', 'suffix',
     * 'middle', 'eot'])} (LC/gguf-py/gguf/vocab.py:155-165, 355-386): tokenizer_config's {@code eot_token} (a string
     * or {@code {content}}) found among tokenizer.json's added tokens, else {@code config.json}'s
     * {@code eot_token_id} (root, then {@code text_config}), else 32010. The prefix, suffix and middle ids that call
     * also resolves are never written: {@code GGUFWriter} has no handler for them and {@code add_to_gguf} skips them
     * (vocab.py:91-97).
     *
     * @throws GgufFormatException when the resolved id is negative
     */
    static int codeLlamaEot(HfTokenizerFiles files, JsonView configRoot) throws IOException {
        JsonView tc = files.readTokenizerConfig();
        Object entry = tc.raw("eot_token");
        String content = entry instanceof String ? (String) entry
                : entry instanceof Map && ((Map<?, ?>) entry).get("content") instanceof String
                ? (String) ((Map<?, ?>) entry).get("content") : null;
        HfTokenizerJson tj = content == null ? null : files.readTokenizerJson();
        if (tj != null) {
            for (HfTokenizerJson.AddedToken t : tj.addedTokens()) {
                if (content.equals(t.content())) {
                    return checkTokenId(tc.where("eot_token"), t.id());
                }
            }
        }
        Object id = configRoot.raw("eot_token_id");
        String where = configRoot.where("eot_token_id");
        if (id == null) {
            Object text = configRoot.raw("text_config");
            if (text instanceof Map) {
                id = ((Map<?, ?>) text).get("eot_token_id");
                where = configRoot.where("text_config") + ".eot_token_id";
            }
        }
        if (id instanceof BigInteger) {
            throw new GgufFormatException(where + " " + id + " is out of range");
        }
        if (id instanceof Long || id instanceof Integer || id instanceof Short || id instanceof Byte) {
            long v = ((Number) id).longValue();
            if (v > Integer.MAX_VALUE) {
                throw new GgufFormatException(where + " " + v + " is out of range");
            }
            return checkTokenId(where, (int) Math.max(v, Integer.MIN_VALUE));
        }
        return CODELLAMA_EOT_ID;
    }

    private static int checkTokenId(String where, int id) {
        if (id < 0) {
            throw new GgufFormatException("invalid value for special token type eot: " + id + " (" + where + ")");
        }
        return id;
    }

    /** The root {@code architectures[0]} of {@code cfg} ({@code origin_hf_arch}, llama.py:42-47), or null. */
    static String rootArchitecture(HfConfig cfg) {
        if (cfg.mistralNative()) {
            return null;
        }
        Object a = cfg.raw().raw("architectures");
        if (a instanceof List && !((List<?>) a).isEmpty() && ((List<?>) a).get(0) instanceof String) {
            return (String) ((List<?>) a).get(0);
        }
        return null;
    }
}
