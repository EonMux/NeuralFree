package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDeepseekV4Names.Mapped;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.EngramRows;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreTokenizerCanonicalizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hooks of {@code DeepseekV41Model} (LC/conversion/deepseek41.py:117-292), the lossless DeepSeek-V4.1 format: the
 * DeepSeek-V4 metadata with the compressed-cache source layers and the two-level candidate selection, the V4 tensor
 * names extended with the indexer key and key-norm tensors, the main layers only (no MTP export) and the F32
 * first-layer gate.
 *
 * <p>The Engram n-gram embedding layers are converted: the {@code ple.*} and {@code engram.*} keys carry the bucket
 * moduli, the hash multipliers reproduced from numpy's PCG64 stream ({@link HfEngramHash}) and the compressed token
 * map reproduced from the tokenizer normalizers, and all layers' tables are written as one {@code per_layer_token_embd}
 * with the rows concatenated in layer order.</p>
 */
public final class HfDeepseekV41Hooks {

    /** {@code layers.<b>.<tail>} (deepseek41.py:271-275). */
    private static final Pattern LAYER = Pattern.compile("layers\\.(\\d+)\\.(.+)");

    private static final String WEIGHT = ".weight";

    /** The converter's {@code _ENGRAM_EMBED_RE}: the tables {@code per_layer_token_embd} concatenates. */
    private static final Pattern ENGRAM_EMBED = Pattern.compile("layers\\.\\d+\\.engram\\.embed\\.(weight|scale)");

    /** {@code DeepseekV41Model._LAYER_MAP}: the V4.1 layer tails and their converter tensor. */
    private static final Map<String, String> LAYER_TENSORS = layerTensors();

    private HfDeepseekV41Hooks() {
    }

    /** {@link Arch#DEEPSEEK41}. */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.DEEPSEEK41;
    }

    // ------------------------------------------------------------------ configuration

    /**
     * {@code DeepseekV41Model.__init__} (deepseek41.py:130-142) over {@link HfDeepseekV4Hooks#configFixup}: the hash
     * layer count defaults to zero, the FP8 scale format must be E8M0 and {@code gate_temp} must be 1.
     *
     * @throws GgufFormatException naming the unsupported key
     */
    public static HfConfig configFixup(HfConfig cfg) {
        HfConfig fixed = HfDeepseekV4Hooks.configFixup(cfg);
        if (!fixed.flat().containsKey("num_hash_layers")) {
            fixed = fixed.withFlat(Collections.singletonMap("num_hash_layers", 0));
        }
        Object quantization = fixed.flat().get("quantization_config");
        if (quantization instanceof Map) {
            Object format = ((Map<?, ?>) quantization).get("scale_fmt");
            if (format != null && !"ue8m0".equals(format)) {
                throw new GgufFormatException(fixed.source() + ": DeepSeek-V4.1: unsupported quantization_config.scale_fmt"
                        + " '" + format + "', only 'ue8m0' is supported (deepseek41.py:136)");
            }
        }
        OptionalDouble gateTemp = fixed.findFloat("gate_temp");
        if (gateTemp.isPresent() && gateTemp.getAsDouble() != 1.0) {
            throw new GgufFormatException(fixed.source() + ": DeepSeek-V4.1: gate_temp " + gateTemp.getAsDouble()
                    + " is not supported, only 1 (deepseek41.py:138)");
        }
        return fixed;
    }

    /**
     * {@code DeepseekV41Model} without a multi-token prediction export ({@code supports_mtp_export = False}): the vision
     * tensors and every {@code mtp.*} tensor drop; the rest maps through the inherited filter.
     */
    public static HfNameFilter filter() {
        return HfDeepseekV4Hooks::filterWithoutMtp;
    }

    /** The block count is {@code num_hidden_layers}: the DSpark draft is a separate export. */
    public static int blockCount(HfConfig cfg) {
        return cfg.requireInt("num_hidden_layers");
    }

    // ------------------------------------------------------------------ metadata

    /**
     * {@code DeepseekV41Model.set_gguf_parameters} (deepseek41.py:148-175): the DeepSeek-V4 metadata, then the
     * compressed-cache source layers of the KV cache and the indexer, when the checkpoint selects candidates in a
     * second level their source layer, top-k block count and block size, and, for a checkpoint with Engram layers, the
     * {@code ple.*} layout and the {@code engram.*} token keys.
     *
     * @throws GgufFormatException when a required layer list or Engram key is missing, an Engram layer's moduli do not
     *                             sum to its row count, or the token map does not match the configuration
     * @throws IOException         when the tokenizer file cannot be read
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfDeepseekV4Hooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ATTENTION_KV_SOURCE_LAYERS, layers(cfg, "kv_source_layer_ids"));
        kv.set(GgufKeys.ATTENTION_INDEXER_SOURCE_LAYERS, layers(cfg, "index_source_layer_ids"));
        int candidateSource = candidateSource(cfg);
        if (candidateSource >= 0) {
            kv.set(GgufKeys.ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, candidateSource);
            kv.set(GgufKeys.ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, cfg.requireInt("candidate_topk_blocks"));
            kv.set(GgufKeys.ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE, cfg.requireInt("candidate_block_size"));
        }
        if (!engramLayers(cfg).isEmpty()) {
            engramMeta(ctx, kv);
        }
    }

    /** deepseek41.py:158-175: the Engram layout keys and the compressed token map. */
    private static void engramMeta(HfLoadContext ctx, HfMetadataBuilder kv) throws IOException {
        HfConfig cfg = ctx.config();
        HfEngramLayout layout = HfEngramLayout.of(cfg);
        int compressedVocab = cfg.requireInt("engram_compressed_vocab_size");
        HfEngramTokenMap.Result tokens = HfEngramTokenMap.build(ctx.tokenizerFiles(),
                cfg.requireInt("vocab_size"));
        if (tokens.compressedVocab() != compressedVocab) {
            throw new GgufFormatException(cfg.source() + ": DeepSeek-V4.1 Engram: the compressed token map of "
                    + tokens.tokens() + " tokens has " + tokens.compressedVocab()
                    + " distinct keys, engram_compressed_vocab_size says " + compressedVocab);
        }
        kv.setTyped(kv.key(GgufKeys.PLE_LAYERS), GGUFValueType.ARRAY, GGUFValueType.UINT32, layout.layers());
        kv.set(GgufKeys.PLE_NGRAM_SIZE, layout.maxNgram());
        kv.set(GgufKeys.PLE_HEADS_PER_NGRAM, layout.heads());
        kv.set(GgufKeys.EMBEDDING_LENGTH_PER_LAYER, cfg.requireInt("engram_head_dim"));
        kv.setTyped(kv.key(GgufKeys.PLE_LAYER_MULTIPLIERS), GGUFValueType.ARRAY, GGUFValueType.UINT64,
                boxed(layout.multipliers(compressedVocab)));
        kv.setTyped(kv.key(GgufKeys.PLE_HEAD_OFFSETS), GGUFValueType.ARRAY, GGUFValueType.UINT64,
                boxed(layout.headOffsets()));
        kv.setTyped(kv.key(GgufKeys.PLE_HEAD_VOCAB_SIZES), GGUFValueType.ARRAY, GGUFValueType.UINT64,
                boxed(layout.headSizes()));
        kv.set(GgufKeys.ENGRAM_PAD_TOKEN_ID, cfg.requireInt("engram_pad_token_id"));
        kv.setTyped(kv.key(GgufKeys.ENGRAM_TOKEN_MAP), GGUFValueType.ARRAY, GGUFValueType.INT32,
                tokens.toList());
    }

    /** The Engram layer ids; an absent key is no Engram layers (deepseek41.py:151). */
    private static List<Integer> engramLayers(HfConfig cfg) {
        return cfg.findIntList("engram_layer_ids").orElse(Collections.emptyList());
    }

    private static List<Long> boxed(long[] values) {
        List<Long> out = new ArrayList<>(values.length);
        for (long value : values) {
            out.add(value);
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * {@code self.hparams.get("candidate_source_layer_id", -1)} (deepseek41.py:154): the second-level source layer, or
     * -1 when the checkpoint selects no candidates in a second level.
     *
     * @throws GgufFormatException when the value is not an integer
     */
    private static int candidateSource(HfConfig cfg) {
        Object raw = cfg.flat().get("candidate_source_layer_id");
        if (raw == null) {
            return -1;
        }
        if (!(raw instanceof Number) || raw instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": 'candidate_source_layer_id' must be an integer, got "
                    + raw);
        }
        return ((Number) raw).intValue();
    }

    private static List<Integer> layers(HfConfig cfg, String key) {
        Optional<List<Integer>> list = cfg.findIntList(key);
        if (!list.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": required key '" + key + "' is missing");
        }
        return list.get();
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code DeepseekV41Model._map_dsv4_tensor_name} (deepseek41.py:271-275): the V4.1 indexer key, key norm and Engram
     * KV tails, else the DeepSeek-V4 names.
     */
    public static final HfDeepseekV4Names.Table TABLE = (name, bid) -> {
        Objects.requireNonNull(name, "name");
        Matcher m = LAYER.matcher(name);
        if (m.matches()) {
            int layer = Integer.parseInt(m.group(1));
            String tensor = LAYER_TENSORS.get(m.group(2));
            if (tensor != null) {
                if (bid != layer) {
                    throw new GgufFormatException("Tensor '" + name + "' parsed bid " + bid + " but layer name has "
                            + layer);
                }
                return new Mapped(tensor, WEIGHT, false);
            }
        }
        return HfDeepseekV4Names.map(name, bid);
    };

    /**
     * {@code DeepseekV41Model.modify_tensors} (deepseek41.py:277-285) through {@link #TABLE}: an Engram key table is
     * folded into {@code engram_gate} with the query table, its query table and both Engram embedding tensors are
     * consumed by {@link #extra}, and the rest maps through the DeepSeek-V4 names.
     *
     * @throws GgufFormatException when a query table has no key table, or the name is not a DeepSeek-V4.1 tensor
     */
    public static void tensors(HfLoadContext ctx, String name, RowSource src, TensorSink out) {
        Objects.requireNonNull(name, "name");
        if (out.isConsumed(name)) {
            return;
        }
        if (name.endsWith(".engram.embed.weight") || name.endsWith(".engram.embed.scale")) {
            out.consume(name);
            return;
        }
        if (name.endsWith(".engram.k_weight")) {
            out.consume(name);
            return;
        }
        if (name.endsWith(".engram.q_weight")) {
            String key = name.substring(0, name.length() - "q_weight".length()) + "k_weight";
            if (!ctx.index().has(key)) {
                throw new GgufFormatException("Missing Engram key table " + key + " for " + name);
            }
            out.consume(key);
            out.add(ctx.names().ggufName("ENGRAM_GATE", ctx.bid(name), WEIGHT),
                    EngramRows.product(src, ctx.index().get(key)));
            return;
        }
        HfDeepseekV4Hooks.tensors(ctx, name, src, out, TABLE);
    }

    /**
     * {@code DeepseekV41Model.generate_extra_tensors} (deepseek41.py:254-263) over the DeepSeek-V4 experts: the
     * Engram embedding tables of all layers as one {@code per_layer_token_embd}, rows concatenated in layer order, as
     * BF16 (F32 with {@code --outtype f32}).
     *
     * @throws GgufFormatException when an Engram table or scale is missing or has the wrong width or row count
     * @throws IOException         when a table cannot be read
     */
    public static void extra(HfLoadContext ctx, TensorSink out) throws IOException {
        HfDeepseekV4Hooks.extra(ctx, out);
        List<Integer> layers = engramLayers(ctx.config());
        if (layers.isEmpty()) {
            return;
        }
        HfEngramLayout layout = HfEngramLayout.of(ctx.config());
        long[] rows = layout.rows();
        ElementFormat scaleFormat = null;
        List<EngramRows.Part> parts = new ArrayList<>(layers.size());
        for (int i = 0; i < layers.size(); i++) {
            int layer = layers.get(i);
            String weight = "layers." + layer + ".engram.embed.weight";
            String scale = "layers." + layer + ".engram.embed.scale";
            if (!ctx.index().has(weight) || !ctx.index().has(scale)) {
                throw new GgufFormatException("Missing Engram table tensors " + weight + " and " + scale);
            }
            RowSource scaleSource = ctx.index().get(scale);
            if (scaleFormat == null) {
                scaleFormat = scaleSource.format();
            } else if (!scaleFormat.equals(scaleSource.format())) {
                throw new GgufFormatException("Engram scales have mixed dtypes: " + scaleFormat + " and "
                        + scaleSource.format());
            }
            RowSource weightSource = ctx.index().get(weight);
            long[] shape = weightSource.shape();
            if (shape.length != 2 || shape[0] != rows[i] || shape[1] % 32 != 0) {
                throw new GgufFormatException(weight + ": expected [" + rows[i] + ", d] with a width that is a"
                        + " multiple of 32, got " + java.util.Arrays.toString(shape));
            }
            parts.add(new EngramRows.Part(weightSource, scaleSource, rows[i], (int) shape[1]));
        }
        GGMLType output = ctx.options().outType() == ModelSourceOptions.OutType.F32 ? GGMLType.F32 : GGMLType.BF16;
        out.add(TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD), EngramRows.table(parts, output));
    }

    /**
     * {@code DeepseekV41Model.tensor_force_quant} (deepseek41.py:287-292): the first-layer gate stays F32 in the
     * checkpoint, so it is dequantized losslessly, and the Engram table is BF16, or F32 with {@code --outtype f32};
     * every other tensor follows the DeepSeek-V4 rules.
     */
    public static GGMLType forceType(String ggufName, int nDims, ElementFormat source,
                                     ModelSourceOptions.OutType outType) {
        if (ggufName.endsWith(".ffn_gate_inp.weight") || ggufName.endsWith(".engram_gate.weight")) {
            return GGMLType.F32;
        }
        if (ggufName.equals(TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD))) {
            return outType == ModelSourceOptions.OutType.F32 ? GGMLType.F32 : GGMLType.BF16;
        }
        return HfDeepseekV4Hooks.forceType(ggufName, nDims, source, outType);
    }

    // ------------------------------------------------------------------ vocabulary

    /**
     * {@code DeepseekV41Model.__init__}'s template removal with {@code TextModel.set_vocab}: the GPT-2 vocabulary
     * without the DeepSeek-V4 fallback template, which the model does not use (deepseek41.py:141-142); a template
     * shipped with the checkpoint still lands in the file.
     *
     * @throws IOException when a tokenizer file cannot be read
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfVocabWriters.Input in = HfVocabWriters.Input.of(ctx.config(), ctx.tokenizerFiles(),
                        PreTokenizerCanonicalizer::toSpec)
                .withGgufArch(ctx.arch().ggufName());
        if (ctx.index().has("embed.weight")) {
            in = in.withEmbeddingRows(HfHookSupport.embeddingRows("embed.weight", ctx.index().get("embed.weight")));
        }
        return HfVocabWriters.gpt2(in, pre);
    }

    private static Map<String, String> layerTensors() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("attn.indexer.wk.weight", "INDEXER_ATTN_K");
        m.put("attn.indexer.k_norm.weight", "INDEXER_K_NORM");
        m.put("engram.wkv.weight", "ENGRAM_KV");
        return Collections.unmodifiableMap(m);
    }
}
