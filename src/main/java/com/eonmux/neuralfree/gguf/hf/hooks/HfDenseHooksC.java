package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfConfigFixup;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfSpecialVocab;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.tensor.HalfFloats;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * The hooks of the dense converter classes without the shared parents of the other hook files of this family:
 * {@code OlmoModel} and {@code Olmo2Model} (LC/conversion/olmo.py:15-69), the {@code ArceeModel} pooling
 * (LC/conversion/llama.py:366-370), the Apertus xIELU parameter arrays of {@code ApertusModel} (llama.py:434-469),
 * and the HunYuan dense model and its Hunyuan-VL text variant with their NTK-alpha and XD-RoPE parameters and
 * their two tokenizer paths (LC/conversion/hunyuan.py:157-360). SeedOss and SmolLM3 need no hooks of their own
 * ({@code olmo.py:42-45}, {@code llama.py:428-431}); their table rows compose the generic hooks and the GPT-2
 * vocabulary.
 */
public final class HfDenseHooksC {

    private static final Logger LOG = LogManager.getLogger(HfDenseHooksC.class);

    private static final String LAYER_TYPES = "layer_types";
    private static final float OLMO_LAYER_NORM_EPS = 1e-5f;
    private static final double HUNYUAN_DEFAULT_ALPHA = 50.0;
    private static final double HUNYUAN_DEFAULT_THETA = 10_000.0;
    private static final int HUNYUAN_DYNAMIC_CONTEXT = 256 * 1024;
    private static final int HUNYUAN_FIXED_BOS_HIDDEN = 4096;
    private static final int HUNYUAN_FIXED_BOS_ID = 127958;
    private static final String XIELU_TAIL_ALPHA_N = ".act_fn.alpha_n";
    private static final String XIELU_TAIL_ALPHA_P = ".act_fn.alpha_p";
    private static final String XIELU_TAIL_BETA = ".act_fn.beta";
    private static final String XIELU_TAIL_EPS = ".act_fn.eps";
    private static final String XIELU_STATE = HfDenseHooksC.class.getName() + ".xielu";

    private HfDenseHooksC() {
    }

    // ------------------------------------------------------------------ shared hooks

    /**
     * {@code _set_vocab_gpt2} (LC/conversion/base.py:2301-2350): GPT-2 BPE, the vocabulary of SeedOss, Olmo and
     * Olmo2 and of every modern class without SentencePiece or tiktoken.
     */
    public static Map<String, Object> gpt2Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    // ------------------------------------------------------------------ OlmoModel

    /**
     * {@code OlmoModel.set_gguf_parameters} (olmo.py:21-26): the generic metadata, the fixed LayerNorm epsilon
     * 1e-5 and the optional {@code attention.clamp_kqv} of {@code clip_qkv}.
     */
    public static void olmoMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, OLMO_LAYER_NORM_EPS);
        HfConfig cfg = ctx.config();
        if (cfg.find("clip_qkv").isPresent()) {
            kv.set(GgufKeys.ATTENTION_CLAMP_KQV, cfg.findFloat(0.0, "clip_qkv"));
        }
    }

    /**
     * {@code OlmoModel.modify_tensors} (olmo.py:30-39): the Q and K projections through the Llama permutation,
     * with the head counts {@code num_attention_heads} and {@code num_key_value_heads} of the checkpoint, then
     * the mapped tensor.
     */
    public static void olmoTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, HfLlamaHooks.permuteQk(ctx.config(), hfName, src), out);
    }

    // ------------------------------------------------------------------ Olmo2Model

    /**
     * {@code Olmo2Model.set_gguf_parameters} (olmo.py:54-69), which Olmo3 inherits: with a sliding window, the
     * window and its per-layer pattern from {@code layer_types}, else the Olmo-2 period (every layer except every
     * 4th); without one, neither key.
     */
    public static void olmo2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        if (!cfg.find("sliding_window").isPresent()) {
            return;
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));
        Map<String, Object> flat = cfg.flat();
        List<Boolean> pattern;
        if (flat.containsKey(LAYER_TYPES)) {
            pattern = HfBaseDenseHooks.slidingFlags(cfg);
        } else {
            int layers = cfg.requireInt("num_hidden_layers");
            pattern = new ArrayList<>(layers);
            for (int i = 0; i < layers; i++) {
                pattern.add((i + 1L) % 4 != 0);
            }
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    }

    // ------------------------------------------------------------------ ArceeModel

    /**
     * {@code ArceeModel.set_gguf_parameters} (llama.py:368-370): the Llama metadata of {@link HfLlamaHooks#meta},
     * then the sentence-transformers pooling type.
     */
    public static void arceeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfLlamaHooks.meta(ctx, kv, plan);
        HfQwenHooks.pooling(ctx, kv);
    }

    // ------------------------------------------------------------------ ApertusModel

    /**
     * {@code ApertusModel.modify_tensors} (llama.py:448-469): every {@code .act_fn.<param>} scalar, one per layer,
     * is collected into the {@code xielu.*} key of its parameter, written through
     * {@link TensorSink#addKv} once the layer count is reached; the scalar tensors themselves are dropped. Every
     * other tensor follows the Llama tensors of {@link HfLlamaHooks#tensors}.
     *
     * @throws IOException            as the Llama tensors do
     * @throws GgufFormatException when the scalar cannot be read, a layer id is missing, or a layer appears twice
     */
    public static void apertusTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String kind = null;
        if (hfName.endsWith(XIELU_TAIL_ALPHA_N)) {
            kind = "alpha_n";
        } else if (hfName.endsWith(XIELU_TAIL_ALPHA_P)) {
            kind = "alpha_p";
        } else if (hfName.endsWith(XIELU_TAIL_BETA)) {
            kind = "beta";
        } else if (hfName.endsWith(XIELU_TAIL_EPS)) {
            kind = "eps";
        }
        if (kind == null) {
            HfLlamaHooks.tensors(ctx, hfName, src, out);
            return;
        }
        Apertus state = ctx.state(XIELU_STATE, Apertus.class, Apertus::new);
        if (state.offer(kind, hfName, scalar(src, hfName), ctx.config().requireInt("num_hidden_layers"))) {
            out.addKv(xieluKey(kind), GGUFValueType.ARRAY, GGUFValueType.FLOAT32, state.completedOf(kind));
        } else {
            out.drop(hfName, "collected into " + xieluKey(kind));
        }
    }

    /** The {@code xielu.*} key of a parameter kind. */
    private static String xieluKey(String kind) {
        switch (kind) {
            case "alpha_n": return GgufKeys.XIELU_ALPHA_N;
            case "alpha_p": return GgufKeys.XIELU_ALPHA_P;
            case "beta": return GgufKeys.XIELU_BETA;
            case "eps": return GgufKeys.XIELU_EPS;
            default:
                throw new IllegalStateException("unknown xielu kind " + kind);
        }
    }

    /** The single value of a scalar checkpoint tensor ({@code data_torch.float().item()}, llama.py:450-464). */
    static float scalar(RowSource src, String hfName) throws IOException {
        long[] shape = src.shape();
        long count = 1;
        for (long d : shape) {
            count = Math.multiplyExact(count, d);
        }
        if (count != 1) {
            throw new GgufFormatException("tensor '" + hfName + "' has the shape " + Arrays.toString(shape)
                    + ", not a scalar activation parameter");
        }
        switch (src.format().ggmlType()) {
            case F32:
            case F64:
            case F16:
            case BF16:
            case I8:
            case I16:
            case I32:
            case I64:
                return readScalar(src);
            default:
                return readScalar(ElementTransforms.dequant(src, GGMLType.F32));
        }
    }

    private static float readScalar(RowSource src) throws IOException {
        if (src.rows() != 1 || src.shape().length > 1 && src.shape()[0] != 1) {
            throw new GgufFormatException("tensor '" + src.describe() + "' is not a scalar");
        }
        ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(src.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        src.readRows(0, 1, b);
        b.flip();
        switch (src.format().ggmlType()) {
            case F32: return b.getFloat();
            case F64: return (float) b.getDouble();
            case F16: return HalfFloats.halfToFloat(b.getShort());
            case BF16: return Float.intBitsToFloat((b.getShort() & 0xFFFF) << 16);
            case I8: return b.get();
            case I16: return b.getShort();
            case I32: return b.getInt();
            case I64: return (float) b.getLong();
            default:
                throw new GgufFormatException("tensor '" + src.describe() + "' has the unsupported format "
                        + src.format());
        }
    }

    /**
     * The per-load xIELU parameters of {@code ApertusModel} (llama.py:436-441): a float per layer per kind,
     * complete and taken in block order when the layer count was reached. The tensors of a kind that was visited
     * more than once for one layer are refused, as a repeated layer id can only be a malformed checkpoint.
     */
    private static final class Apertus {
        private final Map<String, TreeMap<Integer, Float>> values = new LinkedHashMap<>();
        private final Map<String, Set<String>> names = new LinkedHashMap<>();
        private final Map<String, List<Float>> complete = new LinkedHashMap<>();

        /** Collects {@code value} for {@code kind}; true when the kind's layer set reached {@code nLayers}. */
        boolean offer(String kind, String hfName, float value, int nLayers) {
            values.putIfAbsent(kind, new TreeMap<>());
            names.putIfAbsent(kind, new LinkedHashSet<>());
            int bid = HfLoadContext.bidOf(hfName);
            if (bid < 0) {
                throw new GgufFormatException("tensor '" + hfName + "' has no layer number");
            }
            if (!names.get(kind).add(hfName)) {
                throw new GgufFormatException("tensor '" + hfName + "' is an already collected " + kind
                        + " of layer " + bid);
            }
            values.get(kind).put(bid, value);
            if (!complete.containsKey(kind) && values.get(kind).size() == nLayers) {
                List<Float> out = new ArrayList<>(nLayers);
                for (Integer b : values.get(kind).keySet()) {
                    out.add(values.get(kind).get(b));
                }
                complete.put(kind, Collections.unmodifiableList(out));
                LOG.info("Apertus xIELU {}: {} layers, from {} to {}", kind, nLayers, out.get(0),
                        out.get(out.size() - 1));
            }
            return complete.containsKey(kind);
        }

        /** The completed value list of {@code kind}, or null while it is incomplete. */
        List<Float> completedOf(String kind) {
            return complete.get(kind);
        }
    }

    // ------------------------------------------------------------------ HunYuanModel and HunyuanVLTextModel

    /**
     * The {@code __init__} normalization of {@code HunyuanVLTextModel} (hunyuan.py:339-342): rope parameters of
     * type {@code dynamic} with an {@code mrope_section} become XD-RoPE with the same sections, so the dense
     * path's dynamic-RoPE context override never applies and the loader reads the XDRoPE keys.
     */
    public static final HfConfigFixup HUNYUAN_VL_FIXUP = HfDenseHooksC::hunyuanVlFixupConfig;

    /** The configuration change of {@link #HUNYUAN_VL_FIXUP}. */
    static HfConfig hunyuanVlFixupConfig(HfConfig cfg) {
        Map<String, Object> ropeRaw = cfg.rope().raw();
        if (!"dynamic".equals(String.valueOf(ropeRaw.get("rope_type")))
                || !(ropeRaw.get("mrope_section") instanceof List)) {
            return cfg;
        }
        Map<String, Object> rope = new LinkedHashMap<>(ropeRaw);
        rope.put("rope_type", "xdrope");
        rope.put("type", "xdrope");
        rope.put("xdrope_section", new ArrayList<>((List<?>) ropeRaw.get("mrope_section")));
        return cfg.withFlat(Collections.singletonMap("rope_parameters", rope));
    }

    /**
     * {@code HunYuanModel.set_gguf_parameters} (hunyuan.py:248-275), which {@code HunyuanVLTextModel} calls for
     * its {@code super()}: the generic metadata without an {@code expert_count} key when the configuration states
     * the unreal {@code num_experts = 1} (hunyuan.py:251-253 pops it, :253 restores it only above 1), then the
     * NTK-alpha RoPE of the dynamic and xdrope types — the scaled base {@code theta * alpha^(dim / (dim - 2))} of
     * the head size, an alpha of {@value #HUNYUAN_DEFAULT_ALPHA} (the dense default), a none scaling type with a
     * factor of 1, and for {@code dynamic} a 256k original-context and context length.
     *
     * @throws GgufFormatException with the class's assertion when a dynamic configuration's values changed
     */
    public static void hunyuanMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        Object experts = cfg.flat().get("num_experts");
        boolean unrealExperts = experts instanceof Number && !(experts instanceof Boolean)
                && ((Number) experts).doubleValue() <= 1;
        HfGenericMetadata.apply(ctx, kv);
        if (unrealExperts) {
            kv.remove(GgufKeys.EXPERT_COUNT);
        }
        String type = ropeType(cfg);
        if (!"dynamic".equals(type) && !"xdrope".equals(type)) {
            return;
        }
        double alpha = ropeRawNumber(cfg, "alpha", HUNYUAN_DEFAULT_ALPHA);
        double base = ropeRawNumber(cfg, "rope_theta", HUNYUAN_DEFAULT_THETA);
        int dim = cfg.requireInt("head_dim");
        if (dim <= 2) {
            throw new GgufFormatException(cfg.source() + ": the NTK-alpha scaling needs a head size above 2");
        }
        double scaled = base * Math.pow(alpha, dim / (double) (dim - 2));
        kv.set(GgufKeys.ROPE_FREQ_BASE, scaled);
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");
        kv.set(GgufKeys.ROPE_SCALING_FACTOR, 1);
        if ("dynamic".equals(type)) {
            kv.set(GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, HUNYUAN_DYNAMIC_CONTEXT);
            kv.set(GgufKeys.CONTEXT_LENGTH, HUNYUAN_DYNAMIC_CONTEXT);
            double maxPos = cfg.requireInt("max_position_embeddings");
            if (base != HUNYUAN_DEFAULT_THETA
                    || (maxPos != 32 * 1024 && maxPos != HUNYUAN_DYNAMIC_CONTEXT)) {
                throw new GgufFormatException(cfg.source() + ": HunYuan dynamic RoPE scaling assumptions changed,"
                        + " please update the logic or context length manually");
            }
        }
    }

    /**
     * {@code HunyuanVLTextModel.set_gguf_parameters} (hunyuan.py:344-360): the dense metadata, then the XDRoPE
     * metadata when the (normalized) rope type is {@code xdrope}: the raw base {@code rope_theta}, the
     * {@code rope.scaling.alpha} (required; the loader rescales the base from it), a none scaling type, the
     * scaling factor ({@code factor}, default 1), the checkpoint context as the origin and context lengths, and
     * the {@code rope.dimension_sections} of {@code xdrope_section}.
     *
     * @throws GgufFormatException when the configuration omits an XDRoPE parameter
     */
    public static void hunyuanVlMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        hunyuanMeta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        Map<String, Object> rope = cfg.rope().raw();
        if (!"xdrope".equals(String.valueOf(rope.get("rope_type")))) {
            return;
        }
        kv.set(GgufKeys.ROPE_FREQ_BASE, ropeRawNumber(cfg, "rope_theta", Double.NaN));
        kv.set(GgufKeys.ROPE_SCALING_ALPHA, ropeRawNumber(cfg, "alpha", Double.NaN));
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");
        kv.set(GgufKeys.ROPE_SCALING_FACTOR, ropeRawNumber(cfg, "factor", 1.0));
        int ctxLen = cfg.requireInt("max_position_embeddings");
        kv.set(GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, ctxLen);
        kv.set(GgufKeys.CONTEXT_LENGTH, ctxLen);
        Object sections = rope.get("xdrope_section");
        if (!(sections instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": required key rope_scaling.xdrope_section is missing");
        }
        kv.set(GgufKeys.ROPE_DIMENSION_SECTIONS, new ArrayList<>((List<?>) sections));
    }

    /** The rope type of {@code cfg} ({@code rope_type}, never null). */
    private static String ropeType(HfConfig cfg) {
        Object t = cfg.rope().raw().get("rope_type");
        return t == null ? null : String.valueOf(t);
    }

    /** A number of the global RoPE parameters, or {@code NaN} when required and a fallback when absent. */
    private static double ropeRawNumber(HfConfig cfg, String key, double dflt) {
        Object v = cfg.rope().raw().get(key);
        if (v == null) {
            if (Double.isNaN(dflt)) {
                throw new GgufFormatException(cfg.source() + ": required key rope_scaling." + key + " is missing");
            }
            return dflt;
        }
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": rope_scaling." + key + " must be a number, got " + v);
        }
        return ((Number) v).doubleValue();
    }

    /**
     * {@code HunYuanModel.modify_tensors} (hunyuan.py:277-283, also :114-118 through the MoE class): a tied
     * {@code lm_head.weight} is dropped; everything else is mapped.
     */
    public static void hunyuanTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if ("lm_head.weight".equals(hfName) && ctx.config().findBool(false, "tie_word_embeddings")) {
            out.drop(hfName, "tied output layer; token_embd.weight serves as the output");
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /**
     * {@code HunYuanModel.set_vocab} (hunyuan.py:162-246): the pad and eod ids read from the checkpoint's own
     * configuration; then the GPT-2 BPE of {@code tokenizer.json} with its merges, or the tiktoken vocabulary
     * {@link HfVocabWriters#qwen} rebuilds; the special tokens of {@code SpecialVocab}, skipped for the pad when
     * the checkpoint states a negative id (hunyuan.py:179-182); the tiktoken branch's fixed
     * {@value #HUNYUAN_FIXED_BOS_ID} bos of the 7B-dense word size (hunyuan.py:233-234); and the
     * {@code eod_token_id} eos plus the {@code generation_config.json} eot when its eos list has at least two
     * entries (hunyuan.py:237-246).
     *
     * @throws IOException when a tokenizer file cannot be read
     */
    public static Map<String, Object> hunyuanVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfConfig cfg = ctx.config();
        HfTokenizerFiles files = ctx.tokenizerFiles();
        Map<String, Object> md = new LinkedHashMap<>();
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        boolean hasTokenizerJson = files.readTokenizerJson() != null;
        if (hasTokenizerJson) {
            md.putAll(HfVocabWriters.gpt2(in, pre));
        } else {
            md.putAll(HfVocabWriters.qwen(in, pre));
        }
        int tokens = tokenCount(md);
        if (hasTokenizerJson) {
            HfTokenizerJson tj = files.readTokenizerJson();
            JsonView config = withoutNegativePad(cfg);
            HfSpecialVocab.load(files, tj, config, true, tokens).writeTo(md);
        } else {
            if (cfg.findInt(0, "hidden_size") == HUNYUAN_FIXED_BOS_HIDDEN) {
                md.put(GgufKeys.TOKENIZER_BOS_ID, HUNYUAN_FIXED_BOS_ID);
            }
        }
        Integer eod = eodId(cfg);
        if (eod != null) {
            md.put(GgufKeys.TOKENIZER_EOS_ID, eod);
        }
        if (cfg.generation().isPresent()) {
            Object eos = cfg.generation().get().raw("eos_token_id");
            if (eos instanceof List && ((List<?>) eos).size() >= 2
                    && ((List<?>) eos).get(0) instanceof Number) {
                md.put(GgufKeys.TOKENIZER_EOT_ID, ((Number) ((List<?>) eos).get(0)).intValue());
            }
        }
        return Collections.unmodifiableMap(md);
    }

    /** The token count of the tokenizer metadata {@code md}, which every writer filled. */
    private static int tokenCount(Map<String, Object> md) {
        Object tokens = md.get(GgufKeys.TOKENIZER_LIST);
        if (!(tokens instanceof List)) {
            throw new GgufFormatException("the vocabulary writer produced no token list");
        }
        return ((List<?>) tokens).size();
    }

    /** The {@code eod_token_id} of the configuration, or null when it has none or is not an integer. */
    private static Integer eodId(HfConfig cfg) {
        Object v = cfg.find("eod_token_id").orElse(null);
        if (!(v instanceof Number) || v instanceof Boolean) {
            return null;
        }
        return ((Number) v).intValue();
    }

    /**
     * The configuration without a negative {@code pad_token_id}, whose {@code SpecialVocab} the class skips pad
     * handling for (hunyuan.py:179-182).
     */
    private static JsonView withoutNegativePad(HfConfig cfg) {
        Object pad = firstNumber(cfg, "pad_token_id");
        if (pad == null || ((Number) pad).intValue() >= 0) {
            return cfg.raw();
        }
        return JsonView.of(stripPad(cfg.flat()), cfg.source());
    }

    /** A first {@code pad_token_id} (root, then {@code text_config}) that is a negative number, else null. */
    private static Number firstNumber(HfConfig cfg, String key) {
        Map<String, Object> flat = cfg.flat();
        Number root = numberOrMinus(flat.get(key));
        if (root != null && root.intValue() < 0) {
            return root;
        }
        return numberOrMinus(textConfig(flat).get(key));
    }

    private static Number numberOrMinus(Object v) {
        return v instanceof Number && !(v instanceof Boolean) ? (Number) v : null;
    }

    private static Map<String, Object> textConfig(Map<String, Object> flat) {
        Object t = flat.get("text_config");
        return t instanceof Map ? (Map<String, Object>) t : Collections.emptyMap();
    }

    /** A copy of {@code flat} without the {@code pad_token_id} keys of the root and the {@code text_config}. */
    private static Map<String, Object> stripPad(Map<String, Object> flat) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : flat.entrySet()) {
            if ("pad_token_id".equals(e.getKey())) {
                continue;
            }
            if ("text_config".equals(e.getKey()) && e.getValue() instanceof Map) {
                Map<String, Object> text = new LinkedHashMap<>();
                for (Map.Entry<String, Object> te : ((Map<String, Object>) e.getValue()).entrySet()) {
                    if (!"pad_token_id".equals(te.getKey())) {
                        text.put(te.getKey(), te.getValue());
                    }
                }
                copy.put(e.getKey(), text);
                continue;
            }
            copy.put(e.getKey(), e.getValue());
        }
        return copy;
    }
}
