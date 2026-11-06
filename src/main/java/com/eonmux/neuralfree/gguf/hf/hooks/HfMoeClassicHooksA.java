package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ColumnTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerRegistry;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfSpecialVocab;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.SentencePieceModel;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * Hooks of the classic MoE converter classes, part A: {@code Llama4Model} (LC/conversion/llama.py:380-417),
 * {@code DbrxModel} (dbrx.py:13-76), {@code GrokModel} (grok.py:17-117) and {@code ArcticModel} (arctic.py:20-163).
 * {@link #gpt2Vocab} serves the default {@code TextModel.set_vocab} the other classic-MoE classes keep; the Bailing MoE
 * classes and Deepseek v1 are in {@link HfMoeClassicHooksB}, the long tail in {@link HfMoeClassicHooksC}.
 */
public final class HfMoeClassicHooksA {

    /** The merged expert parts in the order Bailing, Deepseek v1 and the per-projection stackers use. */
    static final List<String> GATE_UP_DOWN =
            Collections.unmodifiableList(Arrays.asList("down_proj", "gate_proj", "up_proj"));

    /** The Mixtral order Arctic, Grok and MiniMax-M2 stack (arctic.py:137, grok.py:99, minimax.py:142). */
    static final List<String> MIXTRAL_PARTS = Collections.unmodifiableList(Arrays.asList("w1", "w2", "w3"));

    private static final Logger LOG = LogManager.getLogger(HfMoeClassicHooksA.class);
    private static final String WEIGHT = ".weight";
    private static final String EXPERTS_STATE = HfMoeClassicHooksA.class.getName() + ".experts";
    private static final String ARCTIC_STATE = HfMoeClassicHooksA.class.getName() + ".arctic";
    private static final String GROK_STATE = HfMoeClassicHooksA.class.getName() + ".grok";
    private static final float PAD_SCORE = -10000f;
    private static final int SPM_NORMAL = 1;
    private static final int SPM_UNKNOWN = 2;
    private static final int SPM_CONTROL = 3;
    private static final int SPM_USER_DEFINED = 4;
    private static final int SPM_UNUSED = 5;
    private static final int SPM_BYTE = 6;

    private HfMoeClassicHooksA() {
    }

    /**
     * The default vocabulary of {@code TextModel.set_vocab} (LC/conversion/base.py:1575-1576), for the classic-MoE
     * classes that leave {@code set_vocab} on the default.
     */
    public static Map<String, Object> gpt2Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    /** The per-load expert collector under {@code key}, one per load (converter class instance buffer). */
    static HfHookSupport.ExpertCollector stackedExperts(HfLoadContext ctx, String key, int nExperts,
                                                        List<String> parts) {
        Holder holder = ctx.state(key, Holder.class, Holder::new);
        synchronized (holder) {
            if (holder.collector == null) {
                holder.collector = new HfHookSupport.ExpertCollector(nExperts, parts,
                        HfHookSupport.MergedName.SAME_PREFIX);
            }
            return holder.collector;
        }
    }

    /**
     * The leftover check of {@code prepare_tensors} for a class-level expert buffer: incomplete when the buffer was
     * opened (an expert tensor arrived) but its layer is missing a member (the converter's {@code Unprocessed
     * experts}). When no expert tensor arrived, the state is null and the check passes.
     */
    static void requireOrphans(HfLoadContext ctx, String key, String missingName) {
        Holder holder = ctx.state(key, Holder.class, Holder::new);
        if (holder.collector != null) {
            holder.collector.requireComplete();
        }
    }

    /** The single collector holder of one load. */
    static final class Holder {
        HfHookSupport.ExpertCollector collector;
    }

    /** {@code head_dim} when present, else {@code hidden_size // num_attention_heads} (the rope-width default). */
    static int ropeWidth(HfConfig cfg) {
        OptionalInt hd = cfg.findInt("head_dim");
        if (hd.isPresent()) {
            return hd.getAsInt();
        }
        int heads = cfg.requireInt("num_attention_heads");
        if (heads <= 0) {
            throw new GgufFormatException(cfg.source() + ": num_attention_heads must be positive, got " + heads);
        }
        return Math.floorDiv(cfg.requireInt("hidden_size"), heads);
    }

    // ------------------------------------------------------------------ Llama4Model

    /**
     * {@code Llama4Model.__init__} (llama.py:384-388): the MoE experts' width moves to {@code intermediate_size_moe}
     * and the dense layers' {@code intermediate_size_mlp} replaces {@code intermediate_size}, undoing the rename so
     * the generic feed-forward length stays the dense width.
     *
     * @throws GgufFormatException when only one of the two sizes is present
     */
    public static HfConfig llama4Fixup(HfConfig cfg) {
        Optional<Object> mlp = cfg.find("intermediate_size_mlp");
        if (!mlp.isPresent()) {
            if (cfg.find("intermediate_size_moe").isPresent()) {
                throw new GgufFormatException(cfg.source() + ": 'intermediate_size_moe' without"
                        + " 'intermediate_size_mlp' is not a Llama-4 configuration");
            }
            return cfg;
        }
        Optional<Object> ff = cfg.find("intermediate_size");
        if (!ff.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": 'intermediate_size_mlp' without 'intermediate_size'"
                    + " is not a Llama-4 configuration");
        }
        Map<String, Object> swap = new LinkedHashMap<>();
        swap.put("intermediate_size_moe", ff.get());
        swap.put("intermediate_size", mlp.get());
        return cfg.withFlat(swap);
    }

    /**
     * {@code Llama4Model.set_gguf_parameters} (llama.py:393-400) after {@code LlamaModel.set_gguf_parameters}: the
     * interleave step, the MoE experts' width, and a sliding window of 0 when every layer type is full attention
     * (MobileLLM; models/llama4.cpp:8-12 reads it).
     */
    public static void llama4Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfLlamaHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.INTERLEAVE_MOE_LAYER_STEP, cfg.requireInt("interleave_moe_layer_step"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("intermediate_size_moe"));
        if (allFullAttention(cfg)) {
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, 0);
        }
    }

    private static boolean allFullAttention(HfConfig cfg) {
        Object types = cfg.flat().get("layer_types");
        if (!(types instanceof List) || ((List<?>) types).isEmpty()) {
            return false;
        }
        for (Object t : (List<?>) types) {
            if (!"full_attention".equals(t)) {
                return false;
            }
        }
        return true;
    }

    /**
     * {@code Llama4Model.modify_tensors} (llama.py:402-417): a fused {@code gate_up_proj} expert stack
     * {@code [E, D, 2F]} is transposed and split ({@code transpose(-1, -2).split(F, dim=-2)}) into gate and up
     * halves that the Llama path maps; an expert {@code down_proj} stack {@code [E, F, D]} is transposed and gains
     * {@code .weight}; everything else goes to {@link HfLlamaHooks#tensors}.
     */
    public static void llama4Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("gate_up_proj")) {
            long last = src.shape()[src.shape().length - 1];
            if (last <= 0 || last % 2 != 0) {
                throw new GgufFormatException("tensor '" + hfName + "': the fused gate/up stack's last dimension "
                        + last + " is not an even split");
            }
            RowSource transposed = ExpertTransforms.transposeLast2(src);
            RowSource[] gateUp = ExpertTransforms.splitExperts(transposed, -2, 2);
            HfLlamaHooks.tensors(ctx, hfName.replace("gate_up_proj", "gate_proj" + WEIGHT), gateUp[0], out);
            HfLlamaHooks.tensors(ctx, hfName.replace("gate_up_proj", "up_proj" + WEIGHT), gateUp[1], out);
            return;
        }
        if (hfName.endsWith("down_proj")) {
            HfLlamaHooks.tensors(ctx, hfName + WEIGHT, ExpertTransforms.transposeLast2(src), out);
            return;
        }
        HfLlamaHooks.tensors(ctx, hfName, src, out);
    }

    /** {@code Llama4Model.set_vocab} (llama.py:390-391): the GPT-2 BPE vocabulary, bypassing the Llama dispatch. */
    public static Map<String, Object> llama4Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return gpt2Vocab(ctx, pre);
    }

    // ------------------------------------------------------------------ DbrxModel

    private static Map<String, Object> subConfig(HfConfig cfg, String key) {
        Object v = cfg.flat().get(key);
        if (!(v instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must be an object, got "
                    + typeName(v));
        }
        return Collections.unmodifiableMap((Map<String, Object>) v);
    }

    private static String typeName(Object v) {
        if (v == null) return "nothing";
        if (v instanceof Boolean) return "boolean";
        if (v instanceof Long || v instanceof Integer || v instanceof BigInteger) return "integer";
        if (v instanceof List) return "array";
        if (v instanceof Map) return "object";
        if (v instanceof Number) return "number";
        return v.getClass().getSimpleName();
    }

    private static int subInt(HfConfig cfg, Map<String, Object> sub, String key) {
        Object v = sub.get(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must be an integer, got " + typeName(v));
        }
        double d = ((Number) v).doubleValue();
        if (d != Math.rint(d) || d < 0 || d > Integer.MAX_VALUE) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must be a non-negative int32, got " + v);
        }
        return (int) d;
    }

    private static double subDouble(HfConfig cfg, Map<String, Object> sub, String key) {
        Object v = sub.get(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must be a number, got " + typeName(v));
        }
        return ((Number) v).doubleValue();
    }

    /**
     * {@code DbrxModel.set_gguf_parameters} (dbrx.py:16-38): the hand-written DBRX metadata from the nested
     * {@code ffn_config} and {@code attn_config} (no generic metadata is written first, as in the converter), the
     * constant 1e-5 layer-norm epsilon and the clamp on the attention logits.
     */
    public static void dbrxMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        Map<String, Object> ffn = subConfig(cfg, "ffn_config");
        Map<String, Object> attn = subConfig(cfg, "attn_config");
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("max_seq_len"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, cfg.requireInt("d_model"));
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, subInt(cfg, ffn, "ffn_hidden_size"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, cfg.requireInt("n_heads"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, subInt(cfg, attn, "kv_n_heads"));
        kv.set(GgufKeys.ROPE_FREQ_BASE, subDouble(cfg, attn, "rope_theta"));
        kv.set(GgufKeys.ATTENTION_CLAMP_KQV, subDouble(cfg, attn, "clip_qkv"));
        kv.set(GgufKeys.EXPERT_COUNT, subInt(cfg, ffn, "moe_num_experts"));
        kv.set(GgufKeys.EXPERT_USED_COUNT, subInt(cfg, ffn, "moe_top_k"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, 1e-5);
    }

    /**
     * {@code DbrxModel.modify_tensors} (dbrx.py:40-71): an expert tensor — one containing
     * {@code ffn.experts.mlp.w1|w2|v1} and no {@code .weight} — is reshaped as the converter's
     * {@code view(n_expert, n_ff, n_embd)} (the three expert matrices concatenated along rows), {@code w2} gets the
     * additional {@code permute(0, 2, 1)}; the mapped name gains {@code .weight}. Everything else maps unchanged.
     */
    public static void dbrxTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        Map<String, Object> ffn = subConfig(cfg, "ffn_config");
        int nExpert = subInt(cfg, ffn, "moe_num_experts");
        int nFf = subInt(cfg, ffn, "ffn_hidden_size");
        int nEmbd = cfg.requireInt("d_model");
        for (String marker : Arrays.asList("ffn.experts.mlp.w1", "ffn.experts.mlp.w2", "ffn.experts.mlp.v1")) {
            if (!hfName.contains(marker) || hfName.contains(WEIGHT)) {
                continue;
            }
            if (nExpert < 1 || nFf < 1 || nEmbd < 1) {
                throw new GgufFormatException("tensor '" + hfName + "': DBRX expert dimensions must be positive,"
                        + " got " + nExpert + "/" + nFf + "/" + nEmbd);
            }
            if (product(src.shape()) != (long) nExpert * nFf * nEmbd) {
                throw new GgufFormatException("tensor '" + hfName + "' holds " + product(src.shape())
                        + " elements, not the DBRX expert view " + nExpert + " * " + nFf + " * " + nEmbd);
            }
            RowSource viewed = ElementTransforms.reshape(src, nExpert, nFf, nEmbd);
            RowSource data = marker.endsWith("w2") ? ExpertTransforms.transposeLast2(viewed) : viewed;
            out.add(ctx.names().mapOrThrow(hfName + WEIGHT), data);
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /**
     * {@code DbrxModel.tensor_force_quant} (dbrx.py:73-76): the converter's truthy {@code n_dims > 1} means every
     * multi-dimensional tensor is forced to F32, whatever the output type is.
     */
    public static GGMLType dbrxForceType(String ggufName, int nDims, ElementFormat source,
                                         ModelSourceOptions.OutType outType) {
        if (nDims <= 0) {
            throw new IllegalArgumentException("'" + ggufName + "': a planned tensor has at least one dimension,"
                    + " got " + nDims);
        }
        return nDims > 1 ? GGMLType.F32 : null;
    }

    // ------------------------------------------------------------------ GrokModel

    /**
     * {@code GrokModel.set_gguf_parameters} (grok.py:34-63) after the generic metadata: the softcaps with their 30.0
     * defaults, the MoE width when configured, YaRN (or the "original" spelling treated as YaRN) from the top-level
     * keys, the attention temperature length when truthy, the attention output scale with the
     * {@code rope_dim ** -0.5} default, and the MuP embedding and logit scales.
     */
    public static void grokMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ATTN_LOGIT_SOFTCAPPING, valueOr(cfg, "attn_logit_softcapping", 30.0));
        kv.set(GgufKeys.ROUTER_LOGIT_SOFTCAPPING, valueOr(cfg, "router_logit_softcapping", 30.0));
        Optional<Double> finalCap = cfg.find("final_logit_softcapping")
                .filter(v -> v instanceof Number && ((Number) v).doubleValue() != 0.0)
                .map(v -> ((Number) v).doubleValue());
        if (finalCap.isPresent()) {
            kv.set(GgufKeys.FINAL_LOGIT_SOFTCAPPING, finalCap.get());
        }
        OptionalInt moeFf = cfg.findInt("moe_intermediate_size");
        if (moeFf.isPresent()) {
            kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, moeFf.getAsInt());
        }
        String ropeType = cfg.rope().global().type().orElse(null);
        if ("yarn".equals(ropeType) || "original".equals(ropeType)) {
            kv.set(GgufKeys.ROPE_SCALING_TYPE, "yarn");
            kv.set(GgufKeys.ROPE_SCALING_FACTOR, requireNumber(cfg, "scaling_factor"));
            kv.set(GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, cfg.requireInt("original_max_position_embeddings"));
            kv.set(GgufKeys.ROPE_SCALING_YARN_EXT_FACTOR, requireNumber(cfg, "extrapolation_factor"));
            kv.set(GgufKeys.ROPE_SCALING_YARN_ATTN_FACTOR, requireNumber(cfg, "attn_factor"));
            kv.set(GgufKeys.ROPE_SCALING_YARN_BETA_FAST, requireNumber(cfg, "beta_fast"));
            kv.set(GgufKeys.ROPE_SCALING_YARN_BETA_SLOW, requireNumber(cfg, "beta_slow"));
        }
        OptionalInt tempLen = cfg.findInt("attn_temperature_len");
        if (tempLen.isPresent() && tempLen.getAsInt() != 0) {
            kv.set(GgufKeys.ATTENTION_TEMPERATURE_LENGTH, tempLen.getAsInt());
        }
        kv.set(GgufKeys.ATTENTION_OUTPUT_SCALE, valueOr(cfg, "attn_output_multiplier", Math.pow(ropeWidth(cfg), -0.5)));
        kv.set(GgufKeys.EMBEDDING_SCALE, flatDouble(cfg, "embedding_multiplier_scale"));
        kv.set(GgufKeys.LOGIT_SCALE, flatDouble(cfg, "output_multiplier_scale"));
    }

    private static double valueOr(HfConfig cfg, String name, double dflt) {
        return cfg.findFloat(dflt, name);
    }

    /** A required number-valued key, for the plain hparam reads the converter does with {@code hparams[...]}. */
    static double requireNumber(HfConfig cfg, String name) {
        Object v = cfg.find(name).orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": '" + name + "' must be present"));
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + name + "' must be a number, got " + typeName(v));
        }
        return ((Number) v).doubleValue();
    }

    private static double flatDouble(HfConfig cfg, String name) {
        Object v = cfg.find(name).orElse(null);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + name + "' must be a number, got " + typeName(v));
        }
        return ((Number) v).doubleValue();
    }

    /** The total element count of {@code shape}. */
    private static long product(long[] shape) {
        long n = 1;
        for (long d : shape) {
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    /**
     * {@code GrokModel.modify_tensors} (grok.py:68-117): expert pieces buffer per chunk name; when a layer holds all
     * {@code 3 * num_local_experts} names ({@code transformer.decoder_layer.<n>.moe.<x>.linear|linear_1|linear_v} or
     * its {@code model.layers..<block_sparse_moe>} spelling), the three merged tensors are planned with the TP pieces
     * concatenated along dim 0 ({@code linear}/{@code linear_v}) or the row dimension ({@code linear_1}); a
     * non-expert tensor flushes a complete layer first.
     */
    public static void grokTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if (out.isConsumed(hfName)) {
            return;
        }
        GrokExperts experts = ctx.state(GROK_STATE, GrokExperts.class, () -> new GrokExperts(ctx));
        experts.buffer(ctx, hfName, src, out);
    }

    /** {@code GrokModel.prepare_tensors}' final check (grok.py:117): incomplete expert buffers are invalid. */
    public static void grokExtra(HfLoadContext ctx, TensorSink out) {
        ctx.state(GROK_STATE, GrokExperts.class, () -> new GrokExperts(ctx)).requireComplete();
    }

    /** The Grok instance buffer of the converter's {@code _experts} list-of-dicts. */
    private static final class GrokExperts {
        private final int nExperts;
        private final Map<Long, Map<String, List<RowSource>>> parts = new LinkedHashMap<>();
        private boolean expertPending;

        GrokExperts(HfLoadContext ctx) {
            this.nExperts = ctx.config().requireInt("num_local_experts");
        }

        void buffer(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
            boolean expert = hfName.contains(".moe.") || hfName.contains(".block_sparse_moe.experts.");
            // only the expert cache is keyed by layer; the other tensors plan with bid outside
            // the expert branch (grok.py:77-79 asserts the bid there)
            long bid = expert ? HfLoadContext.bidOf(hfName) : -1;
            if (expert && bid < 0) {
                throw new GgufFormatException("tensor '" + hfName + "' has no layer number");
            }
            Map<String, List<RowSource>> layer = parts.get(bid);
            if (expert && layer != null && layer.containsKey(hfName)) {
                layer.get(hfName).add(src);
                flush(ctx, out);
                return;
            }
            if (expert) {
                List<RowSource> list = new ArrayList<>();
                list.add(src);
                parts.computeIfAbsent(bid, k -> new LinkedHashMap<>()).put(hfName, list);
                expertPending = true;
            }
            if (expert || expertPending) {
                flush(ctx, out);
            }
            if (!expert) {
                expertPending = false;
                out.add(ctx.names().mapOrThrow(hfName), src);
            }
        }

        private void flush(HfLoadContext ctx, TensorSink out) {
            for (Long bid : new ArrayList<>(parts.keySet())) {
                Map<String, List<RowSource>> layer = parts.get(bid);
                if (layer == null || layer.size() < nExperts * 3L) {
                    continue;
                }
                for (String marker : PART_MARKERS) {
                    List<RowSource> stacked = new ArrayList<>(nExperts);
                    for (int x = 0; x < nExperts; x++) {
                        String canonical = "transformer.decoder_layer." + bid + ".moe." + x + "." + marker + WEIGHT;
                        List<RowSource> pieces = layer.remove(canonical);
                        if (pieces == null) {
                            pieces = layer.remove("model.layers." + bid + ".block_sparse_moe.experts." + x + "."
                                    + wName(marker) + WEIGHT);
                        }
                        if (pieces == null || pieces.isEmpty()) {
                            throw new GgufFormatException("Grok experts of layer " + bid
                                    + " lost the '" + canonical + "' pieces");
                        }
                        stacked.add("linear_1".equals(marker) ? ColumnTransforms.concatCols(pieces)
                                : RowTransforms.concatRows(pieces));
                    }
                    out.add(ctx.names().mapOrThrow("transformer.decoder_layer." + bid + ".moe." + marker + WEIGHT),
                            ExpertTransforms.stack(stacked));
                }
                if (layer.isEmpty()) {
                    parts.remove(bid);
                }
            }
        }

        void requireComplete() {
            if (!parts.isEmpty()) {
                throw new GgufFormatException("Unprocessed experts: " + parts.keySet());
            }
        }
    }

    private static final List<String> PART_MARKERS =
            Collections.unmodifiableList(Arrays.asList("linear", "linear_1", "linear_v"));

    private static String wName(String marker) {
        switch (marker) {
            case "linear": return "w1";
            case "linear_1": return "w2";
            default: return "w3";
        }
    }

    /** {@code GrokModel.set_vocab} (grok.py:20-27): tokenizer.model → SentencePiece; else tokenizer.json with
     * chat_template.jinja → GPT-2 BPE; the converter aborts with its download message when neither exists. */
    public static Map<String, Object> grokVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        if (files.file(HfTokenizerFiles.TOKENIZER_MODEL) != null) {
            return HfVocabWriters.sentencePiece(HfHookSupport.vocabInput(ctx), pre);
        }
        if (files.tokenizerJson() != null && files.chatTemplateJinja() != null) {
            return gpt2Vocab(ctx, pre);
        }
        throw new GgufFormatException("Error: Missing vocab and chat template: a Grok checkpoint needs"
                + " tokenizer.model, or tokenizer.json with chat_template.jinja");
    }

    // ------------------------------------------------------------------ ArcticModel

    /**
     * {@code ArcticModel.set_gguf_parameters} (arctic.py:107-111) after the generic metadata: the vocabulary size
     * and the RoPE dimension count {@code hidden_size // num_attention_heads}.
     */
    public static void arcticMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT,
                Math.floorDiv(cfg.requireInt("hidden_size"), cfg.requireInt("num_attention_heads")));
    }

    /**
     * {@code ArcticModel.modify_tensors} (arctic.py:115-154): the Llama Q/K permutation on q/k weights and biases,
     * then the Mixtral stacking of {@code block_sparse_moe.experts.<x>.w{1,2,3}} into the spelled
     * {@code layers.<n>.feed_forward.experts.<w>} names ({@link #arcticExtra} fails on leftovers).
     */
    public static void arcticTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource data = HfLlamaHooks.permuteQk(ctx.config(), hfName, src);
        if (hfName.contains("block_sparse_moe.experts")) {
            if (!arcticCollector(ctx).offer(ctx, hfName, out)) {
                throw new GgufFormatException("tensor '" + hfName + "' is not an Arctic expert weight"
                        + " (block_sparse_moe.experts.<x>.w1|w2|w3)");
            }
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), data);
    }

    /** The collector of arctic.py:126-152, built from the configured expert count. */
    private static HfHookSupport.ExpertCollector arcticCollector(HfLoadContext ctx) {
        Holder holder = ctx.state(ARCTIC_STATE, Holder.class, Holder::new);
        if (holder.collector == null) {
            holder.collector = new HfHookSupport.ExpertCollector(ctx.config().requireInt("num_local_experts"),
                    MIXTRAL_PARTS, HfMoeClassicHooksA::arcticMerged);
        }
        return holder.collector;
    }

    /** The merged Arctic expert name the converter spells explicitly (arctic.py:147). */
    private static String arcticMerged(String prefix, String part, String tail) {
        return "layers." + HfHookSupport.layerOf(prefix) + ".feed_forward.experts." + part + tail;
    }

    /** {@code ArcticModel.prepare_tensors} (arctic.py:156-163): leftover experts are invalid; without experts not. */
    public static void arcticExtra(HfLoadContext ctx, TensorSink out) {
        requireOrphans(ctx, ARCTIC_STATE, "block_sparse_moe.experts");
    }

    /**
     * {@code ArcticModel.set_vocab} (arctic.py:23-105): the SentencePiece vocabulary padded to {@code vocab_size}
     * (default the piece count) with {@code [PAD<i>]} UNUSED entries; the pieces overwrite their slots; the
     * tokenizer_config {@code added_tokens_decoder} entries become USER_DEFINED (score -10000) or the special
     * CONTROL/UNKNOWN override with score 0.0; then the special vocabulary of the checkpoint.
     */
    public static Map<String, Object> arcticVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        Path model = files.file(HfTokenizerFiles.TOKENIZER_MODEL);
        if (model == null) {
            throw new GgufFormatException("Error: Missing " + HfTokenizerFiles.TOKENIZER_MODEL
                    + " for an Arctic checkpoint");
        }
        SentencePieceModel sp = SentencePieceModel.read(model);
        HfConfig cfg = ctx.config();
        int size = cfg.findInt(sp.pieceCount(), "vocab_size");
        if (sp.pieceCount() > size) {
            throw new GgufFormatException(model + ": " + sp.pieceCount() + " SentencePiece pieces exceed the"
                    + " configured vocab_size " + size);
        }
        List<String> tokens = new ArrayList<>(size);
        List<Float> scores = new ArrayList<>(size);
        List<Integer> types = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            tokens.add("[PAD" + i + "]");
            scores.add(PAD_SCORE);
            types.add(SPM_UNUSED);
        }
        for (int i = 0; i < sp.pieceCount(); i++) {
            tokens.set(i, sp.piece(i));
            scores.set(i, sp.score(i));
            types.set(i, spmType(sp.type(i).name()));
        }
        JsonView tc = files.readTokenizerConfig();
        String unkContent = tokenContent(tc, "unk_token");
        for (HfTokenizerJson.AddedToken t : addedDecoder(tc)) {
            int id = t.id();
            if (id < 0 || id >= size) {
                ctx.log("ignore token " + id + ": id is out of range, max=" + (size - 1));
                continue;
            }
            boolean special = t.special();
            types.set(id, special ? (t.content().equals(unkContent) ? SPM_UNKNOWN : SPM_CONTROL)
                    : SPM_USER_DEFINED);
            scores.set(id, special ? 0.0f : PAD_SCORE);
            tokens.set(id, t.content());
        }
        Map<String, Object> md = new LinkedHashMap<>();
        md.put(GgufKeys.TOKENIZER_MODEL, "llama");
        md.put(GgufKeys.TOKENIZER_PRE, PreTokenizerRegistry.DEFAULT_NAME);
        md.put(GgufKeys.TOKENIZER_LIST, Collections.unmodifiableList(tokens));
        md.put(GgufKeys.TOKENIZER_SCORES, Collections.unmodifiableList(scores));
        md.put(GgufKeys.TOKENIZER_TOKEN_TYPE, Collections.unmodifiableList(types));
        HfSpecialVocab.load(files, files.readTokenizerJson(), cfg.raw(), false, size).writeTo(md);
        md.putAll(HfChatTemplates.resolve(files, null));
        return Collections.unmodifiableMap(md);
    }

    /** tokenizer_config {@code added_tokens_decoder} entries, key order preserved, ids validated. */
    private static List<HfTokenizerJson.AddedToken> addedDecoder(JsonView tc) {
        if (tc == null) {
            return Collections.emptyList();
        }
        Object raw = tc.raw("added_tokens_decoder");
        if (!(raw instanceof Map)) {
            return Collections.emptyList();
        }
        List<HfTokenizerJson.AddedToken> out = new ArrayList<>();
        for (Map.Entry<String, Object> e : ((Map<String, Object>) raw).entrySet()) {
            int id;
            try {
                id = Integer.parseInt(e.getKey().trim());
            } catch (NumberFormatException ex) {
                throw new GgufFormatException(tc.where("added_tokens_decoder") + ": key '" + e.getKey()
                        + "' must be an integer token id");
            }
            if (!(e.getValue() instanceof Map)) {
                throw new GgufFormatException(tc.where("added_tokens_decoder") + ": entry '" + e.getKey()
                        + "' must be an object");
            }
            Map<?, ?> entry = (Map<?, ?>) e.getValue();
            Object content = entry.get("content");
            if (!(content instanceof String)) {
                throw new GgufFormatException(tc.where("added_tokens_decoder") + ": entry '" + e.getKey()
                        + "' has no string 'content'");
            }
            out.add(new HfTokenizerJson.AddedToken(id, (String) content, Boolean.TRUE.equals(entry.get("special")),
                    false, false, false, false));
        }
        return out;
    }

    private static String tokenContent(JsonView tc, String key) {
        Object v = tc == null ? null : tc.raw(key);
        if (v instanceof String) {
            return (String) v;
        }
        if (v instanceof Map) {
            Object content = ((Map<?, ?>) v).get("content");
            return content instanceof String ? (String) content : null;
        }
        return null;
    }

    private static int spmType(String name) {
        switch (name) {
            case "UNKNOWN": return SPM_UNKNOWN;
            case "CONTROL": return SPM_CONTROL;
            case "UNUSED": return SPM_UNUSED;
            case "BYTE": return SPM_BYTE;
            default: return SPM_NORMAL;
        }
    }
}
