package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.names.HfTensorNameMap;
import com.eonmux.neuralfree.gguf.hf.tensor.ColumnTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreTokenizerCanonicalizer;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.Set;
import java.util.function.UnaryOperator;
import java.util.regex.Pattern;

/**
 * The hooks of {@code DeepseekV4Model} (LC/conversion/deepseek.py:527-926): the RoPE parameter fixup, the native name
 * table ({@link HfDeepseekV4Names}), multi-token prediction kept in the same view as extra blocks, the hash-routing
 * tables, the stacked routed experts, the MTP {@code eh_proj} concatenation, the forced types and the hyper-connection,
 * compressor and indexer metadata.
 *
 * <p>FP8 weights with E8M0 128x128 block scales arrive as F8_E4M3_B128 and routed MXFP4 experts as MXFP4 from the
 * quantization rewrite; they keep their exact types (the converter dequantizes FP8 weights and stores them as
 * Q8_0).</p>
 */
public final class HfDeepseekV4Hooks {

    /** The bundled default chat templates (deepseek.py:563-570). */
    public static final String TEMPLATE = "deepseek-ai-DeepSeek-V4.jinja";
    public static final String TEMPLATE_0731 = "deepseek-ai-DeepSeek-V4-Flash-0731.jinja";

    /** The routed expert members the extra tensors stack (deepseek.py:881). */
    private static final Pattern EXPERT_MEMBER =
            Pattern.compile("layers\\.\\d+\\.ffn\\.experts\\.\\d+\\.w[123]\\.(weight|scale)");

    /** The MTP projections the extra tensors concatenate into {@code nextn.eh_proj}. */
    private static final Pattern NEXTN_EH_PART = Pattern.compile("layers\\.\\d+\\.nextn\\.[eh]_proj\\.weight");

    /** An MTP layer's own output hyper-connection head, renamed by {@link #filter(HfLoadContext)}. */
    private static final Pattern NEXTN_HC_HEAD = Pattern.compile("layers\\.\\d+\\.nextn\\.hc_head_(fn|base|scale)");

    private static final Set<String> MTP_HC_HEAD = Set.of("hc_head_fn", "hc_head_base", "hc_head_scale");
    private static final Set<String> MTP_NEXTN = Set.of("e_proj.weight", "e_proj.scale", "h_proj.weight",
            "h_proj.scale");

    /** Expert projection to converter tensor, in the converter's write order (deepseek.py:783-785). */
    private static final Map<String, String> EXPERT_TENSORS = expertTensors();

    private static final Set<StDType> INTEGER_DTYPES = EnumSet.of(StDType.BOOL, StDType.U8, StDType.I8, StDType.I16,
            StDType.U16, StDType.I32, StDType.U32, StDType.I64, StDType.U64);

    private static final int MAX_BID_DIGITS = 9;

    /** The native name of the token embedding (deepseek.py:811). */
    private static final String EMBED = "embed.weight";

    private HfDeepseekV4Hooks() {
    }

    // ------------------------------------------------------------------ configuration, filter and block count

    /**
     * {@code DeepseekV4Model.__init__} (deepseek.py:538-549). The {@code setdefault} merge of the raw config.json keys
     * adds nothing here: the flattened configuration already holds every config.json key. When the global RoPE
     * parameters have no {@code rope_type}, the root {@code rope_scaling} object (with {@code rope_type} taken from a
     * legacy {@code type}) is merged into them, the transformers 5.x {@code main}/{@code compress} layout workaround.
     *
     * @throws GgufFormatException when {@code rope_scaling} is present but not an object
     */
    public static HfConfig configFixup(HfConfig cfg) {
        if (cfg.rope().global().type().isPresent()) {
            return cfg;
        }
        Object scaling = cfg.raw().raw("rope_scaling");
        if (scaling == null) {
            return cfg;
        }
        if (!(scaling instanceof Map)) {
            throw new GgufFormatException(cfg.raw().where("rope_scaling") + " must be an object");
        }
        Map<String, Object> s = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) scaling).entrySet()) {
            s.put(String.valueOf(e.getKey()), e.getValue());
        }
        if (!s.containsKey("rope_type") && s.get("type") != null) {
            s.put("rope_type", s.get("type"));
        }
        Map<String, Object> merged = new LinkedHashMap<>(cfg.rope().raw());
        merged.putAll(s);
        return cfg.withFlat(Collections.singletonMap("rope_parameters", merged));
    }

    /**
     * {@code DeepseekV4Model.filter_tensors} (deepseek.py:578-631) for a view that keeps multi-token prediction in the
     * model's file: drops the vision aligner and image tensors; renames {@code mtp.<i>.<suffix>} into block
     * {@code num_hidden_layers + i} as the converter's MTP export does ({@code e_proj}/{@code h_proj} and the norms
     * under {@code nextn.}, the rest under {@code layers.<b>.}), and the MTP layer's own {@code hc_head_*} under
     * {@code layers.<b>.nextn.} because the main model's head keeps the model-level names; drops {@code mtp.*} when
     * {@link ModelSourceOptions#includeMtp()} is false; applies the inherited filter to every other name.
     *
     * @throws GgufFormatException for an {@code mtp.} name without a layer index or with one at or above
     *                             {@code num_nextn_predict_layers}
     */
    public static HfNameFilter filter(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        int mainLayers = cfg.requireInt("num_hidden_layers");
        int nextnLayers = cfg.findInt(0, "num_nextn_predict_layers");
        boolean includeMtp = ctx.options().includeMtp();
        return (name, base) -> {
            if (isVision(name)) {
                return null;
            }
            if (name.startsWith("mtp.")) {
                return includeMtp ? mtpName(name, mainLayers, nextnLayers) : null;
            }
            return base.apply(name);
        };
    }

    /**
     * {@link #filter(HfLoadContext)} without multi-token prediction: the vision tensors and every {@code mtp.*} tensor
     * are dropped, as the converter does for the main model file.
     */
    public static String filterWithoutMtp(String name, UnaryOperator<String> base) {
        if (isVision(name) || name.startsWith("mtp.")) {
            return null;
        }
        return base.apply(name);
    }

    /**
     * The block count: {@code num_hidden_layers}, plus {@code num_nextn_predict_layers} when multi-token prediction is
     * included and the checkpoint has MTP tensors ({@code mtp.*}, or blocks past the main layers once renamed).
     */
    public static int blockCount(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        int mainLayers = cfg.requireInt("num_hidden_layers");
        if (!ctx.options().includeMtp()) {
            return mainLayers;
        }
        for (String name : ctx.index().names()) {
            if (name.startsWith("mtp.") || HfLoadContext.bidOf(name) >= mainLayers) {
                return mainLayers + cfg.findInt(0, "num_nextn_predict_layers");
            }
        }
        return mainLayers;
    }

    private static boolean isVision(String name) {
        return name.startsWith("aligner.") || name.startsWith("image_");
    }

    /** deepseek.py:589-622. */
    private static String mtpName(String name, int mainLayers, int nextnLayers) {
        String[] parts = name.split("\\.", 3);
        if (parts.length < 3 || parts[1].isEmpty() || !parts[1].chars().allMatch(c -> c >= '0' && c <= '9')) {
            throw new GgufFormatException("Unexpected DeepSeek-V4 MTP tensor '" + name + "'");
        }
        if (parts[1].length() > MAX_BID_DIGITS || Integer.parseInt(parts[1]) >= nextnLayers) {
            throw new GgufFormatException("Unexpected DeepSeek-V4 MTP layer " + parts[1] + " in '" + name + "' ("
                    + nextnLayers + " MTP layers)");
        }
        int bid = mainLayers + Integer.parseInt(parts[1]);
        String suffix = parts[2];
        if (MTP_HC_HEAD.contains(suffix) || MTP_NEXTN.contains(suffix)) {
            return "layers." + bid + ".nextn." + suffix;
        }
        switch (suffix) {
            case "enorm.weight":
                return "layers." + bid + ".nextn.enorm.weight";
            case "hnorm.weight":
                return "layers." + bid + ".nextn.hnorm.weight";
            case "norm.weight":
                return "layers." + bid + ".nextn.shared_head_norm.weight";
            default:
                return "layers." + bid + "." + suffix;
        }
    }

    // ------------------------------------------------------------------ metadata

    /**
     * {@code DeepseekV4Model.set_gguf_parameters} (deepseek.py:659-689): the generic keys, then the RoPE dimension
     * count, Q LoRA rank, sliding window, expert keys, per-block SwiGLU clamps, indexer, grouped output projection,
     * compression, hyper-connection and hash-routing keys, the output embedding length {@code hidden_size * hc_mult},
     * and the MTP layer count when the view has blocks past {@code num_hidden_layers}.
     *
     * @throws GgufFormatException when a required hyperparameter is missing
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, cfg.requireInt("qk_rope_head_dim"));
        kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, cfg.requireInt("q_lora_rank"));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));

        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, cfg.requireInt("n_shared_experts"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, requireFloat(cfg, "routed_scaling_factor"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, require(cfg, "norm_topk_prob"));
        List<Double> clamp = Collections.nCopies(ctx.blockCount(), requireFloat(cfg, "swiglu_limit"));
        kv.set(GgufKeys.SWIGLU_CLAMP_EXP, clamp);
        kv.set(GgufKeys.SWIGLU_CLAMP_SHEXP, clamp);

        kv.set(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT, cfg.requireInt("index_n_heads"));
        kv.set(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH, cfg.requireInt("index_head_dim"));
        kv.set(GgufKeys.ATTENTION_INDEXER_TOP_K, cfg.requireInt("index_topk"));

        kv.set(GgufKeys.ATTENTION_OUTPUT_GROUP_COUNT, cfg.requireInt("o_groups"));
        kv.set(GgufKeys.ATTENTION_OUTPUT_LORA_RANK, cfg.requireInt("o_lora_rank"));
        kv.set(GgufKeys.ATTENTION_COMPRESS_RATIOS, require(cfg, "compress_ratios"));
        kv.set(GgufKeys.ATTENTION_COMPRESS_ROPE_FREQ_BASE, requireFloat(cfg, "compress_rope_theta"));
        int hcMult = cfg.requireInt("hc_mult");
        kv.set(GgufKeys.HYPER_CONNECTION_COUNT, hcMult);
        kv.set(GgufKeys.HYPER_CONNECTION_SINKHORN_ITERATIONS, cfg.requireInt("hc_sinkhorn_iters"));
        kv.set(GgufKeys.HYPER_CONNECTION_EPSILON, requireFloat(cfg, "hc_eps"));
        kv.set(GgufKeys.HASH_LAYER_COUNT, cfg.requireInt("num_hash_layers"));
        if (ctx.arch() == Arch.DEEPSEEK4) {
            kv.set(GgufKeys.EMBEDDING_LENGTH_OUT, Math.multiplyExact((long) cfg.requireInt("hidden_size"), hcMult));
        }
        int nextn = ctx.blockCount() - cfg.requireInt("num_hidden_layers");
        if (nextn > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextn);
        }
    }

    private static Object require(HfConfig cfg, String key) {
        Optional<Object> v = cfg.find(key);
        if (!v.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": required key '" + key + "' is missing");
        }
        return v.get();
    }

    private static double requireFloat(HfConfig cfg, String key) {
        OptionalDouble v = cfg.findFloat(key);
        if (!v.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": required key '" + key + "' is missing");
        }
        return v.getAsDouble();
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code DeepseekV4Model.modify_tensors} (deepseek.py:880-891): the routed expert members, the MTP
     * {@code e_proj}/{@code h_proj} pair and the hash-layer {@code tid2eid} tables are planned by {@link #extra};
     * the gate bias of hash-routed layers, {@code tid2eid} of other layers and an MTP layer's own output head are
     * dropped; every other tensor is mapped through {@link HfDeepseekV4Names}.
     *
     * @throws GgufFormatException {@code "Unsupported DeepSeek-V4 tensor"} for a name the table does not map, including
     *                             a block scale the quantization rewrite left behind
     */
    public static void tensors(HfLoadContext ctx, String name, RowSource src, TensorSink out) {
        tensors(ctx, name, src, out, HfDeepseekV4Names.Table.DEFAULT);
    }

    /**
     * {@link #tensors(HfLoadContext, String, RowSource, TensorSink)} with the native names mapped by {@code table}, the
     * form a subclass that extends {@code _map_dsv4_tensor_name} composes (DeepseekV4DSpark, deepseek.py:1005-1008).
     *
     * @throws GgufFormatException as {@code table} does for a name it does not map
     */
    public static void tensors(HfLoadContext ctx, String name, RowSource src, TensorSink out,
                               HfDeepseekV4Names.Table table) {
        Objects.requireNonNull(table, "table");
        if (out.isConsumed(name)) {
            return;
        }
        if (EXPERT_MEMBER.matcher(name).matches()) {
            out.consume(name);
            return;
        }
        HfConfig cfg = ctx.config();
        int bid = ctx.bid(name);
        int hashLayers = cfg.requireInt("num_hash_layers");
        if (name.endsWith(".ffn.gate.bias") && bid >= 0 && bid < hashLayers) {
            out.drop(name, "hash-routed layer: text tokens route through ffn_gate_tid2eid, so gate.bias is unused");
            return;
        }
        if (NEXTN_HC_HEAD.matcher(name).matches()) {
            out.drop(name, "MTP layer output head: the MTP draft collapses its streams with the model's output_hc_*");
            return;
        }
        if (bid >= cfg.requireInt("num_hidden_layers") && NEXTN_EH_PART.matcher(name).matches()) {
            out.consume(name);
            return;
        }
        HfDeepseekV4Names.Mapped m = table.map(name, bid);
        if (m.tensor().equals("FFN_GATE_TID2EID")) {
            if (bid < hashLayers) {
                out.consume(name);
            } else {
                out.drop(name, "tid2eid of a layer past num_hash_layers (" + hashLayers
                        + "), which is not hash-routed");
            }
            return;
        }
        out.add(ctx.names().ggufName(m.tensor(), m.root() ? -1 : bid, m.suffix()), src);
    }

    /**
     * {@code DeepseekV4Model.generate_extra_tensors} (deepseek.py:765-804): the hash-routing tables of the first
     * {@code num_hash_layers} blocks as I32, the routed experts of every block stacked {@code [E, rows, cols]} per
     * projection ({@code w1} gate, {@code w2} down, {@code w3} up) in the format the quantization rewrite produced,
     * and for each MTP block the concatenation of {@code e_proj} and {@code h_proj} along the input dimension as
     * {@code nextn.eh_proj}. With {@link ModelSourceOptions#mtpOnly()} the main-model blocks are absent from the
     * index, so only the MTP blocks' experts and projections are generated (deepseek.py:777-787).
     *
     * @throws GgufFormatException when a hash table, an expert or one of an MTP projection pair is missing, an expert
     *                             block scale was not consumed by the quantization rewrite, or the parts do not fit
     *                             together
     */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        HfConfig cfg = ctx.config();
        HfTensorIndex index = ctx.index();
        HfTensorNameMap names = ctx.names();

        // --mtp keeps no main-model block, so neither the hash tables nor the main experts are written
        int firstBlock = ctx.options().mtpOnly() ? cfg.requireInt("num_hidden_layers") : 0;
        int hashLayers = cfg.requireInt("num_hash_layers");
        for (int bid = 0; bid < hashLayers && firstBlock == 0; bid++) {
            String name = "layers." + bid + ".ffn.gate.tid2eid";
            if (!index.has(name)) {
                throw new GgufFormatException("Missing hash routing tensor " + name);
            }
            out.add(names.ggufName("FFN_GATE_TID2EID", bid, ".weight"), toInt32(name, index.get(name)));
        }

        int nExperts = cfg.requireInt("n_routed_experts");
        for (int bid = firstBlock; bid < ctx.blockCount(); bid++) {
            for (Map.Entry<String, String> e : EXPERT_TENSORS.entrySet()) {
                out.add(names.ggufName(e.getValue(), bid, ".weight"), stackExperts(index, bid, e.getKey(), nExperts));
            }
        }

        for (int bid = cfg.requireInt("num_hidden_layers"); bid < ctx.blockCount(); bid++) {
            String eName = "layers." + bid + ".nextn.e_proj.weight";
            String hName = "layers." + bid + ".nextn.h_proj.weight";
            if (!index.has(eName) && !index.has(hName)) {
                continue;
            }
            if (!index.has(eName) || !index.has(hName)) {
                throw new GgufFormatException("Missing DeepSeek-V4 MTP e/h projection pair for block " + bid);
            }
            RowSource eh;
            try {
                eh = ColumnTransforms.concatCols(List.of(index.get(eName), index.get(hName)));
            } catch (IllegalArgumentException ex) {
                throw new GgufFormatException("MTP projections of block " + bid + " do not concatenate: "
                        + ex.getMessage(), ex);
            }
            out.add(names.ggufName("NEXTN_EH_PROJ", bid, ".weight"), eh);
        }
    }

    /** deepseek.py:721-746: the routed experts of {@code proj} in block {@code bid}, stacked in expert order. */
    private static RowSource stackExperts(HfTensorIndex index, int bid, String proj, int nExperts) {
        List<RowSource> parts = new ArrayList<>(nExperts);
        for (int x = 0; x < nExperts; x++) {
            String base = "layers." + bid + ".ffn.experts." + x + "." + proj;
            if (!index.has(base + ".weight")) {
                throw new GgufFormatException("Missing routed expert tensors for " + base + ".weight");
            }
            if (index.has(base + ".scale")) {
                throw new GgufFormatException(base + ".scale was not consumed by the quantization rewrite; the"
                        + " routed experts need the checkpoint's quantization_config (MXFP4 with E8M0 scales)");
            }
            parts.add(index.get(base + ".weight"));
        }
        try {
            return ExpertTransforms.stack(parts);
        } catch (IllegalArgumentException ex) {
            throw new GgufFormatException("routed experts " + proj + " of block " + bid + " do not stack: "
                    + ex.getMessage(), ex);
        }
    }

    /** {@code data_torch.to(torch.int32)} (deepseek.py:757), kept as an integer tensor. */
    private static RowSource toInt32(String name, RowSource src) {
        ElementFormat f = src.format();
        if (f.ggmlType() == GGMLType.I32) {
            return ElementTransforms.keepInteger(src);
        }
        if (!INTEGER_DTYPES.contains(f.stDType())) {
            throw new GgufFormatException(name + ": the hash routing table has dtype " + f
                    + "; an integer tensor is required");
        }
        return ElementTransforms.keepInteger(new Int32Rows(src));
    }

    // ------------------------------------------------------------------ types and vocabulary

    /**
     * {@code DeepseekV4Model.tensor_force_quant} (deepseek.py:894-906): the MTP {@code eh_proj} is Q8_0 unless the
     * output type keeps source types; a stored F32 tensor stays F32 and a stored BF16 tensor of rank 2 or more stays
     * BF16; every other tensor follows the dtype policy (the FP8 block and MXFP4 sources keep their exact types).
     */
    public static GGMLType forceType(String ggufName, int nDims, ElementFormat source,
                                     ModelSourceOptions.OutType outType) {
        if (ggufName.endsWith(".nextn.eh_proj.weight") && outType != ModelSourceOptions.OutType.SOURCE) {
            return GGMLType.Q8_0;
        }
        GGMLType t = source.ggmlType();
        if (t == GGMLType.F32) {
            return GGMLType.F32;
        }
        if (t == GGMLType.BF16 && nDims >= 2) {
            return GGMLType.BF16;
        }
        return null;
    }

    /**
     * {@code TextModel.set_vocab} ({@code _set_vocab_gpt2}) with the bundled DeepSeek-V4 chat template as the
     * fallback: the Flash 0731 template when the model directory, or its Hugging Face cache repository folder, names
     * {@code 0731} (deepseek.py:563-570).
     *
     * @throws IOException when a tokenizer file cannot be read
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfVocabWriters.Input in = HfVocabWriters.Input.of(ctx.config(), ctx.tokenizerFiles(),
                        PreTokenizerCanonicalizer::toSpec)
                .withGgufArch(ctx.arch().ggufName())
                .withFallbackChatTemplate(HfChatTemplates.bundled(defaultTemplate(ctx.config().dir())));
        if (ctx.index().has(EMBED)) {
            in = in.withEmbeddingRows(HfHookSupport.embeddingRows(EMBED, ctx.index().get(EMBED)));
        }
        return HfVocabWriters.gpt2(in, pre);
    }

    /** The bundled template file for a model in {@code dir}. */
    public static String defaultTemplate(Path dir) {
        return modelIdHint(dir).contains("0731") ? TEMPLATE_0731 : TEMPLATE;
    }

    /**
     * The model id hint of {@code dir}: its name, preceded by the repository folder when it is a Hugging Face cache
     * snapshot ({@code models--<org>--<name>/snapshots/<revision>}).
     */
    static String modelIdHint(Path dir) {
        if (dir == null) {
            return "";
        }
        Path abs = dir.toAbsolutePath().normalize();
        String hint = abs.getFileName() == null ? "" : abs.getFileName().toString();
        Path parent = abs.getParent();
        if (parent != null && parent.getFileName() != null && parent.getFileName().toString().equals("snapshots")) {
            Path repo = parent.getParent();
            if (repo != null && repo.getFileName() != null && repo.getFileName().toString().startsWith("models--")) {
                hint = repo.getFileName() + "/" + hint;
            }
        }
        return hint;
    }

    private static Map<String, String> expertTensors() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("w1", "FFN_GATE_EXP");
        m.put("w2", "FFN_DOWN_EXP");
        m.put("w3", "FFN_UP_EXP");
        return Collections.unmodifiableMap(m);
    }

    /**
     * An integer tensor narrowed to I32 element by element ({@code .to(torch.int32)}); a value outside the I32 range
     * is an error instead of the cast's wrap-around. Row-local; immutable.
     */
    private static final class Int32Rows implements RowSource {

        private static final ElementFormat I32 = ElementFormat.of(GGMLType.I32);

        private final RowSource src;
        private final StDType dtype;
        private final long[] shape;
        private final int inner;

        Int32Rows(RowSource src) {
            this.src = Objects.requireNonNull(src, "src");
            this.dtype = src.format().stDType();
            this.shape = src.shape();
            this.inner = Math.toIntExact(shape.length == 0 ? 1 : shape[shape.length - 1]);
        }

        @Override
        public String describe() {
            return src.describe() + " as I32";
        }

        @Override
        public ElementFormat format() {
            return I32;
        }

        @Override
        public long[] shape() {
            return shape.clone();
        }

        @Override
        public long rows() {
            return src.rows();
        }

        @Override
        public long rowBytes() {
            return 4L * inner;
        }

        @Override
        public boolean rowLocal() {
            return src.rowLocal();
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            if (count < 0) {
                throw new IllegalArgumentException(describe() + ": negative row count " + count);
            }
            long outBytes = Math.multiplyExact((long) count, rowBytes());
            if (dst.remaining() < outBytes) {
                throw new IllegalArgumentException(describe() + ": " + outBytes + " bytes do not fit in "
                        + dst.remaining() + " remaining");
            }
            ByteBuffer in = ByteBuffer.allocate(Math.toIntExact(Math.multiplyExact((long) count, src.rowBytes())))
                    .order(ByteOrder.LITTLE_ENDIAN);
            src.readRows(first, count, in);
            in.flip();
            ByteBuffer o = dst.slice().order(ByteOrder.LITTLE_ENDIAN);
            long n = (long) count * inner;
            for (long i = 0; i < n; i++) {
                long v = next(in);
                if (v < Integer.MIN_VALUE || v > Integer.MAX_VALUE) {
                    throw new IOException(src.describe() + ": value " + v + " at row "
                            + (first + i / inner) + " does not fit I32");
                }
                o.putInt((int) v);
            }
            dst.position(dst.position() + (int) outBytes);
        }

        private long next(ByteBuffer in) throws IOException {
            switch (dtype) {
                case BOOL:
                case U8:
                    return in.get() & 0xFFL;
                case I8:
                    return in.get();
                case I16:
                    return in.getShort();
                case U16:
                    return in.getShort() & 0xFFFFL;
                case I32:
                    return in.getInt();
                case U32:
                    return in.getInt() & 0xFFFF_FFFFL;
                case I64:
                    return in.getLong();
                case U64: {
                    long v = in.getLong();
                    if (v < 0) {
                        throw new IOException(src.describe() + ": value " + Long.toUnsignedString(v)
                                + " does not fit I32");
                    }
                    return v;
                }
                default:
                    throw new IllegalStateException("not an integer dtype: " + dtype);
            }
        }
    }
}
