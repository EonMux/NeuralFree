package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.Set;

/**
 * Hooks of {@code HYV4Model} (LC/conversion/hy_v4.py:26-244), part C of the DeepseekV2-composing MLA classes:
 * it composes the {@link HfDeepseekHooks} and adds the fused stacked gate/up experts split, the iHC metadata, the DSA
 * indexer key types of the marked layers, and the F32 forcing of the iHC {@code *_fn} matrices and the indexer key
 * norm bias. The other DeepseekV2-deriving classes live in {@link HfMoeMlaHooksA}, the standalone text classes in
 * {@link HfMoeMlaHooksB}; every method is a public static hook reference the entries register.
 */
public final class HfMoeMlaHooksC {

    private static final Logger LOG = LogManager.getLogger(HfMoeMlaHooksC.class);

    /** The constant F32 forcing of the HY_V4 indexer key norm bias (hy_v4.py:213-215). */
    public static final GGMLType INDEXER_F32 = GGMLType.F32;
    /** {@code gguf.ExpertGatingFuncType.SIGMOID} (LC/gguf-py/gguf/constants.py:6285-6288). */
    static final int GATING_SIGMOID = 2;

    /** The tensors a "full" indexer layer must carry (hy_v4.py:46-54). */
    public static final Set<String> INDEXER_TENSOR_SUFFIXES = Set.of("wq_b.weight", "wk.weight", "k_norm.weight",
            "k_norm.bias", "weights_proj.weight");

    /** The class filter of {@code HYV4Model.filter_tensors} (hy_v4.py:55-60): the MTP layers are never read. */
    public static final HfNameFilter HYV4_FILTER = (name, base) -> name.startsWith("model.mtp_layers.") ? null
            : HfDeepseekHooks.filter(name, base);

    private HfMoeMlaHooksC() {
    }

    /**
     * {@code HYV4Model.__init__} (hy_v4.py:105-136) and {@code prepare_tensors} (hy_v4.py:178-180): the ungrouped
     * routing keys dropped for the base metadata, the dense leading block count derived from {@code mlp_layer_types}
     * when the configuration has none, and {@code num_key_value_heads := num_attention_heads}.
     */
    public static HfConfig hyV4Fixup(HfConfig cfg) {
        Map<String, Object> fix = new LinkedHashMap<>();
        if (cfg.findInt(1, "n_group") == 1 && cfg.findInt(1, "topk_group") == 1) {
            fix.put("n_group", null);
            fix.put("topk_group", null);
        }
        Object mlpTypes = cfg.flat().get("mlp_layer_types");
        OptionalInt explicit = cfg.findInt("first_k_dense_replace");
        Integer derived = null;
        if (mlpTypes instanceof List) {
            List<?> types = (List<?>) mlpTypes;
            int lead = 0;
            while (lead < types.size() && "dense".equals(types.get(lead))) {
                lead++;
            }
            for (int i = lead; i < types.size(); i++) {
                if ("dense".equals(types.get(i))) {
                    throw new GgufFormatException(cfg.source() + ": HY_V4 converter expects a contiguous leading"
                            + " dense block");
                }
            }
            derived = lead;
        }
        if (explicit.isPresent() && derived != null && explicit.getAsInt() != derived) {
            throw new GgufFormatException(cfg.source() + ": HY_V4 first_k_dense_replace (" + explicit.getAsInt()
                    + ") disagrees with mlp_layer_types leading-dense count (" + derived + ")");
        }
        if (!explicit.isPresent()) {
            if (derived == null) {
                throw new GgufFormatException(cfg.source() + ": HY_V4 needs first_k_dense_replace or"
                        + " mlp_layer_types to place dense layers");
            }
            fix.put("first_k_dense_replace", (long) derived);
        }
        fix.put("num_key_value_heads", (long) cfg.requireInt("num_attention_heads"));
        return cfg.withFlat(fix);
    }

    /**
     * {@code HYV4Model.set_gguf_parameters} (hy_v4.py:102-175) after the base metadata: sigmoid routing, the routed
     * SwiGLU clamp over the block count, the iHC metadata and the DSA indexer keys with the per-layer
     * {@code indexer_types}. Multi-token prediction is dropped, so no NextN count is written.
     */
    public static void hyV4Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfDeepseekHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.EXPERT_GATING_FUNC, GATING_SIGMOID);
        OptionalDouble swiglu = cfg.findFloat("swiglu_limit");
        if (swiglu.isPresent() && swiglu.getAsDouble() > 0.0) {
            kv.set(GgufKeys.SWIGLU_CLAMP_EXP, Collections.nCopies(ctx.blockCount(), swiglu.getAsDouble()));
        }
        kv.set(GgufKeys.HYPER_CONNECTION_COUNT, cfg.requireInt("hc_mult"));
        kv.set(GgufKeys.HYPER_CONNECTION_EPSILON, requireFloat(cfg, "hc_eps"));
        kv.set(GgufKeys.HYPER_CONNECTION_MAGNITUDE, requireFloat(cfg, "hc_magnitude"));
        List<Boolean> isFull = hyV4IndexerFull(cfg);
        if (isFull != null) {
            kv.set(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT, cfg.requireInt("index_n_heads"));
            kv.set(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH, cfg.requireInt("index_head_dim"));
            kv.set(GgufKeys.ATTENTION_INDEXER_TOP_K, cfg.requireInt("index_topk"));
            kv.set(GgufKeys.ATTENTION_INDEXER_TYPES, isFull);
            LOG.info("HY_V4 DSA: {}/{} layers own an indexer", isFull.stream().filter(Boolean::booleanValue).count(),
                    isFull.size());
        }
    }

    /** The per-layer indexer ownership ({@code indexer_is_full}, hy_v4.py:67-100), or null without DSA. */
    public static List<Boolean> hyV4IndexerFull(HfConfig cfg) {
        Object indexerTypes = cfg.flat().get("indexer_types");
        int nLayer = cfg.requireInt("num_hidden_layers");
        if (indexerTypes == null) {
            Object layerTypes = cfg.flat().get("layer_types");
            if (!(layerTypes instanceof List)) {
                return null;
            }
            List<?> types = (List<?>) layerTypes;
            if (types.size() < nLayer) {
                throw new GgufFormatException(cfg.source() + ": HY_V4 layer_types has " + types.size()
                        + " entries, need " + nLayer);
            }
            List<Boolean> out = new ArrayList<>();
            boolean anySparse = false;
            for (int i = 0; i < nLayer; i++) {
                boolean sparse = "sparse_attention".equals(types.get(i))
                        || "deepseek_sparse_attention".equals(types.get(i));
                anySparse |= sparse;
                out.add(sparse);
            }
            if (!anySparse) {
                return null;
            }
            checkIndexerHparams(cfg);
            return out;
        }
        checkIndexerHparams(cfg);
        List<?> types = (List<?>) indexerTypes;
        if (types.size() < nLayer) {
            throw new GgufFormatException(cfg.source() + ": HY_V4 indexer_types has " + types.size()
                    + " entries, need " + nLayer);
        }
        List<Boolean> out = new ArrayList<>();
        for (int i = 0; i < nLayer; i++) {
            Object t = types.get(i);
            if (!"full".equals(t) && !"shared".equals(t)) {
                throw new GgufFormatException(cfg.source() + ": HY_V4 unknown indexer_types value: " + t);
            }
            out.add("full".equals(t));
        }
        if (!out.isEmpty() && !out.get(0)) {
            throw new GgufFormatException(cfg.source() + ": HY_V4 layer 0 must be indexer_types 'full' (nothing"
                    + " precedes it to share)");
        }
        return out;
    }

    /** The required float of {@code cfg} over {@code aliases}. */
    private static double requireFloat(HfConfig cfg, String... aliases) {
        return cfg.findFloat(aliases).orElseThrow(() -> new GgufFormatException(
                cfg.source() + ": required float '" + aliases[0] + "' is missing"));
    }

    /** {@code _check_indexer_hparams} (hy_v4.py:62-65). */
    private static void checkIndexerHparams(HfConfig cfg) {
        for (String key : Arrays.asList("index_n_heads", "index_head_dim", "index_topk")) {
            if (!cfg.flat().containsKey(key)) {
                throw new GgufFormatException(cfg.source() + ": HY_V4 has DSA layers but no " + key);
            }
        }
    }

    /** The block count: the main layers only ({@link ModelSourceOptions#includeMtp} excluded, hy_v4.py:37-39). */
    public static int hyV4BlockCount(HfConfig cfg) {
        return cfg.requireInt("num_hidden_layers");
    }

    /**
     * {@code HYV4Model.modify_tensors} (hy_v4.py:221-244): the fused stacked gate/up experts split into gate and up
     * stacks ({@code split_gate_up}, hy_v4.py:12-21), the un-suffixed members gain {@code .weight}, and everything
     * else goes to the base path with the fixed key head count and no expert merging.
     */
    public static void hyV4Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName;
        HfConfig cfg = ctx.config();
        int moeInter = cfg.requireInt("moe_intermediate_size");
        if (name.endsWith("mlp.experts.gate_up_proj")) {
            long[] shape = src.shape();
            long rows = shape.length >= 2 ? shape[shape.length - 2] : 0;
            if (rows != 2L * moeInter) {
                throw new GgufFormatException("tensor '" + name + "': the fused stacked gate_up of row count " + rows
                        + " is not 2 * " + moeInter);
            }
            int bid = ctx.bid(name);
            if (bid < 0) {
                throw new GgufFormatException("tensor '" + name + "' has no layer number");
            }
            out.add(ctx.names().ggufName("FFN_GATE_EXP", bid, ".weight"), RowTransforms.sliceDim(src, 1, 0, moeInter));
            out.add(ctx.names().ggufName("FFN_UP_EXP", bid, ".weight"),
                    RowTransforms.sliceDim(src, 1, moeInter, moeInter));
            return;
        }
        if (name.endsWith("mlp.experts.down_proj") || name.endsWith(".self_attn.learnable_sink_param")) {
            name += ".weight";
        }
        if (name.matches(".*\\.hc_head\\.hc_head_(fn|base|scale)$")
                || name.matches(".*\\.hc_(attn|mlp)_layer\\.hc_pre\\.hc_(fn|base|scale)$")) {
            name += ".weight";
        }
        HfDeepseekHooks.tensors(ctx, name, src, out, true, false);
    }

    /** {@code HYV4Model.prepare_tensors} (hy_v4.py:182-202): the indexer presence of the marked layers. */
    public static void hyV4Extra(HfLoadContext ctx, TensorSink out) {
        List<Boolean> isFull = hyV4IndexerFull(ctx.config());
        if (isFull == null) {
            return;
        }
        for (int il = 0; il < isFull.size(); il++) {
            String prefix = "model.layers." + il + ".self_attn.indexer.";
            Map<String, Boolean> seen = new LinkedHashMap<>();
            for (String suffix : INDEXER_TENSOR_SUFFIXES) {
                seen.put(suffix, ctx.index().has(prefix + suffix));
            }
            if (isFull.get(il) && seen.containsValue(true) && seen.containsValue(false)) {
                throw new GgufFormatException("HY_V4 layer " + il + " is indexer_types 'full' but is missing indexer"
                        + " tensors: " + missing(seen));
            }
            if (isFull.get(il) && !seen.containsValue(true)) {
                throw new GgufFormatException("HY_V4 layer " + il + " is indexer_types 'full' but is missing indexer"
                        + " tensors: " + INDEXER_TENSOR_SUFFIXES);
            }
            if (!isFull.get(il) && seen.containsValue(true)) {
                throw new GgufFormatException("HY_V4 layer " + il + " is indexer_types 'shared' but carries indexer"
                        + " tensors: " + present(seen));
            }
        }
    }

    private static List<String> present(Map<String, Boolean> seen) {
        List<String> out = new ArrayList<>();
        for (Map.Entry<String, Boolean> e : seen.entrySet()) {
            if (e.getValue()) {
                out.add(e.getKey());
            }
        }
        return out;
    }

    private static List<String> missing(Map<String, Boolean> seen) {
        List<String> used = present(seen);
        List<String> out = new ArrayList<>(INDEXER_TENSOR_SUFFIXES);
        out.removeAll(used);
        return out;
    }

    /**
     * {@code HYV4Model.tensor_force_quant} (hy_v4.py:205-219): the iHC {@code *_fn} matrices, the indexer key norm
     * bias and (with {@code enable_lm_head_fp32}) the output head are F32.
     */
    public static GGMLType hyV4ForceType(String ggufName, int nDims, ElementFormat source,
                                         ModelSourceOptions.OutType outType) {
        if (ggufName.endsWith("hc_attn_fn.weight") || ggufName.endsWith("hc_ffn_fn.weight")
                || ggufName.endsWith("output_hc_fn.weight")) {
            return GGMLType.F32;
        }
        if (ggufName.matches(".+\\.indexer\\.k_norm\\.bias")) {
            return GGMLType.F32;
        }
        return null;
    }
}
