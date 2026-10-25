package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Hooks of the sliding-window / sink MoE converter classes, part A: {@code FlintModel}
 * (LC/conversion/flint.py:9-37, on {@code Qwen2MoeModel}), {@code AfmoeModel} (afmoe.py:17-80, on
 * {@code LlamaModel}) and {@code Kolibri1Model} (kolibri1.py:9-22, on {@code AfmoeModel}), plus the Cohere-2 MoE
 * (command_r.py:65-183) and Granite SWA (granite.py:78-200) rows. The MimoV2/Step35/Beam classes are in
 * {@link HfMoeSwaHooksB}, the long tail in {@link HfMoeSwaHooksC}.
 *
 * <p>Every method is a public static hook reference that subclasses compose where the converter calls
 * {@code super()}.</p>
 */
public final class HfMoeSwaHooksA {

    /** The expert parts of the classes whose checkpoint names are per-expert projections. */
    static final List<String> GATE_UP_DOWN =
            Collections.unmodifiableList(java.util.Arrays.asList("gate_proj", "up_proj", "down_proj"));

    /** The shared per-load expert collector holder key of the A classes. */
    private static final String EXPERTS_STATE = HfMoeSwaHooksA.class.getName() + ".experts";

    private HfMoeSwaHooksA() {
    }

    // ------------------------------------------------------------------ shared helpers

    /**
     * The default vocabulary of a class whose {@code set_vocab} stays on the default {@code TextModel._set_vocab_gpt2}
     * (LC/conversion/base.py:1575-1576).
     */
    public static Map<String, Object> gpt2Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    /** {@code EXPERT_GATING_FUNC} = 2, the {@code gguf.ExpertGatingFuncType.SIGMOID} code (constants.py:6285-6288). */
    static void gatingSigmoid(HfMetadataBuilder kv) {
        kv.set(GgufKeys.EXPERT_GATING_FUNC, 2);
    }

    /** {@code EXPERT_GATING_FUNC} = 1, the {@code gguf.ExpertGatingFuncType.SOFTMAX} code. */
    static void gatingSoftmax(HfMetadataBuilder kv) {
        kv.set(GgufKeys.EXPERT_GATING_FUNC, 1);
    }

    /**
     * The routed count of a class whose configuration names it one of the converter's
     * {@code ["num_local_experts", "num_experts"]} aliases or one of {@code extra}, required.
     */
    static int expertCount(HfConfig cfg, String... extra) {
        String[] aliases = new String[2 + extra.length];
        aliases[0] = "num_local_experts";
        aliases[1] = "num_experts";
        java.lang.System.arraycopy(extra, 0, aliases, 2, extra.length);
        return cfg.requireInt(aliases);
    }

    /** Writes {@code key} from {@code name} when present and truthy, as a walrus-guarded writer call does. */
    static void truthyInt(HfConfig cfg, HfMetadataBuilder kv, String name, String key) {
        Object v = cfg.find(name).orElse(null);
        if (v != null && pythonTruthy(v) && v instanceof Number) {
            kv.set(key, ((Number) v).intValue());
        }
    }

    static void truthyBool(HfConfig cfg, HfMetadataBuilder kv, String name, String key) {
        Object v = cfg.find(name).orElse(null);
        if (v != null && pythonTruthy(v) && v instanceof Boolean) {
            kv.set(key, v);
        }
    }

    static void truthyScale(HfConfig cfg, HfMetadataBuilder kv, String name, String key) {
        Object v = cfg.find(name).orElse(null);
        if (v != null && pythonTruthy(v) && v instanceof Number) {
            kv.set(key, ((Number) v).doubleValue());
        }
    }

    /** Python truthiness of a JSON value. */
    static boolean pythonTruthy(Object v) {
        if (v == null) return false;
        if (v instanceof Boolean) return (Boolean) v;
        if (v instanceof Number) return ((Number) v).doubleValue() != 0.0;
        if (v instanceof String) return !((String) v).isEmpty();
        if (v instanceof List) return !((List<?>) v).isEmpty();
        if (v instanceof Map) return !((Map<?, ?>) v).isEmpty();
        return true;
    }

    /**
     * The expert collector of a class whose per-expert tensors arrive at
     * {@code model.layers.<b>.mlp.experts.<x>.<part>.weight}, stacked into one tensor per part under the
     * {@code model.layers.<b>.mlp.experts.<part>.weight} names the tensor map reads.
     */
    static HfHookSupport.ExpertCollector experts(HfLoadContext ctx, String key, int nExperts,
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

    /** The leftover check of the class's {@code prepare_tensors}: incomplete buffers are invalid. */
    static void requireOrphans(HfLoadContext ctx, String key) {
        Holder holder = ctx.state(key, Holder.class, Holder::new);
        if (holder.collector != null) {
            holder.collector.requireComplete();
        }
    }

    /** The single collector holder of one load. */
    static final class Holder {
        HfHookSupport.ExpertCollector collector;
    }

    /** The {@code boolean} list of {@code name} when present, or null. */
    static List<Boolean> flagList(HfConfig cfg, String name) {
        Object v = cfg.flat().get(name);
        if (!(v instanceof List) || ((List<?>) v).isEmpty()) {
            return null;
        }
        List<Boolean> out = new ArrayList<>();
        for (Object t : (List<?>) v) {
            if (!(t instanceof String)) {
                return null;
            }
            out.add("sliding_attention".equals(t));
        }
        return out;
    }

    /**
     * {@code layer_types} as a boolean sliding-window list; every entry must be a layer-type string
     * (the converter's map raises for unknown kinds).
     */
    static List<Boolean> slidingPattern(HfConfig cfg, String... names) {
        for (String name : names) {
            List<Boolean> flags = flagList(cfg, name);
            if (flags != null) {
                return flags;
            }
        }
        throw new GgufFormatException(cfg.source() + ": no layer_types / mtp_layer_types entry names a layer type");
    }

    /** The pattern of {@code kernel} as {@line "i % kernel != 0"}, the transformers default period. */
    static List<Boolean> periodPattern(int blocks, int kernel) {
        List<Boolean> out = new ArrayList<>(blocks);
        for (int i = 0; i < blocks; i++) {
            out.add(i % kernel != 0);
        }
        return out;
    }

    /** The aligned {@code extra} appended to {@code base}; both trimmed/padded to {@code blocks}. */
    static List<Boolean> concatPattern(List<Boolean> base, List<Boolean> extra, int blocks) {
        List<Boolean> out = new ArrayList<>(base);
        out.addAll(extra);
        while (out.size() < blocks) {
            out.add(false);
        }
        return Collections.unmodifiableList(out.subList(0, blocks));
    }

    // ------------------------------------------------------------------ FlintModel

    /**
     * {@code FlintModel.set_gguf_parameters} (flint.py:12-28) after {@code Qwen2MoeModel}'s keys: the dense lead, the
     * shared expert count derived from the two FFN widths, the routing norm, the sliding window and its
     * {@code layer_types} pattern and the RoPE pattern of {@code no_rope_layer_indices}.
     *
     * @throws GgufFormatException when the dense layers do not precede the MoE layers (flint.py:14-16)
     */
    public static void flintMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfQwenMoeHooks.qwen2MoeMeta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        List<Integer> dense = cfg.findIntList(java.util.Collections.emptyList(), "mlp_only_layers");
        if (dense.isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": mlp_only_layers is required");
        }
        for (int i = 0; i < dense.size(); i++) {
            if (dense.get(i) != i) {
                throw new GgufFormatException(cfg.source() + ": Flint dense layers must precede the MoE layers");
            }
        }
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, dense.size());
        int sharedFf = cfg.requireInt("shared_expert_intermediate_size");
        int moeFf = cfg.requireInt("moe_intermediate_size");
        if (moeFf <= 0 || sharedFf % moeFf != 0) {
            throw new GgufFormatException(cfg.source() + ": shared expert width " + sharedFf + " does not split"
                    + " into " + moeFf + "-wide experts");
        }
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, sharedFf / moeFf);
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(true, "norm_topk_prob"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, 1.0);
        int swa = cfg.requireInt("sliding_window");
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, swa);
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, slidingPattern(cfg, "layer_types"));
        List<Integer> indices = cfg.findIntList(java.util.Collections.emptyList(), "no_rope_layer_indices");
        List<Boolean> rope = new ArrayList<>();
        for (int i = 0; i < ctx.blockCount(); i++) {
            rope.add(!indices.contains(i));
        }
        kv.set(GgufKeys.ATTENTION_ROPE_PATTERN, rope);
    }

    /**
     * {@code FlintModel.modify_tensors} (flint.py:30-37): the {@code softmax_offset} gains {@code .weight}; the
     * router's {@code expert_bias} is dropped when {@code router_expert_bias} is off and named
     * {@code mlp.gate.expert_bias.bias} otherwise.
     */
    public static void flintTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName;
        if (name.endsWith(".softmax_offset")) {
            name += ".weight";
        }
        if (name.endsWith(".gate.expert_bias")) {
            if (!ctx.config().findBool(false, "router_expert_bias")) {
                out.drop(name, "flint router_expert_bias is off");
                return;
            }
            name += ".bias";
        }
        HfQwenMoeHooks.qwen2MoeTensors(ctx, name, src, out);
    }

    // ------------------------------------------------------------------ AfmoeModel

    /** {@code AfmoeModel.filter_tensors} (afmoe.py:49-55): {@code .expert_bias} names the router bias tensor. */
    public static final HfNameFilter AFMOE_FILTER = (name, base) ->
            base.apply(name.endsWith(".expert_bias") ? name.replace(".expert_bias", ".expert_bias.bias") : name);

    /**
     * {@code AfmoeModel.set_gguf_parameters} (afmoe.py:24-45) after {@code LlamaModel}'s keys: the MoE counts, the
     * route norm/scale and the sliding window, each only when the configuration has it.
     */
    public static void afmoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfLlamaHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        truthyInt(cfg, kv, "num_shared_experts", GgufKeys.EXPERT_SHARED_COUNT);
        truthyInt(cfg, kv, "moe_intermediate_size", GgufKeys.EXPERT_FEED_FORWARD_LENGTH);
        truthyInt(cfg, kv, "num_dense_layers", GgufKeys.LEADING_DENSE_BLOCK_COUNT);
        truthyBool(cfg, kv, "route_norm", GgufKeys.EXPERT_WEIGHTS_NORM);
        truthyScale(cfg, kv, "route_scale", GgufKeys.EXPERT_WEIGHTS_SCALE);
        truthyInt(cfg, kv, "sliding_window", GgufKeys.ATTENTION_SLIDING_WINDOW);
    }

    /**
     * {@code AfmoeModel.modify_tensors} (afmoe.py:57-79): per-expert {@code mlp.experts.<x>.<part>.weight} tensors
     * stack into {@code mlp.experts.<part>.weight} (the merged names {@code ModelBase.modify_tensors} maps),
     * everything else goes through the Llama path with its Q/K permutation.
     */
    public static void afmoeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("mlp.experts")) {
            int nExperts = expertCount(ctx.config());
            if (!experts(ctx, EXPERTS_STATE, nExperts, GATE_UP_DOWN).offer(ctx, hfName, out)) {
                throw new GgufFormatException("Unprocessed experts: ['" + hfName + "']");
            }
            return;
        }
        HfLlamaHooks.tensors(ctx, hfName, src, out);
    }

    /** {@code AfmoeModel.prepare_tensors} (afmoe.py:79): leftover experts are invalid. */
    public static void afmoeExtra(HfLoadContext ctx, TensorSink out) {
        requireOrphans(ctx, EXPERTS_STATE);
    }

    // ------------------------------------------------------------------ Kolibri1Model

    /**
     * {@code Kolibri1Model.__init__} (kolibri1.py:14-19): the Afmoe keys the configuration stores under other names —
     * one shared expert, {@code route_norm} from {@code norm_topk_prob} and
     * {@code intermediate_size} from {@code moe_intermediate_size}.
     */
    public static com.eonmux.neuralfree.gguf.hf.config.HfConfig kolibri1Fixup(
            com.eonmux.neuralfree.gguf.hf.config.HfConfig cfg) {
        java.util.Map<String, Object> props = new java.util.LinkedHashMap<>();
        props.put("num_shared_experts", 1);
        Object normTopk = cfg.find("norm_topk_prob").orElse(null);
        if (normTopk != null) {
            props.put("route_norm", normTopk);
        }
        Object moeFf = cfg.find("moe_intermediate_size").orElse(null);
        if (moeFf != null) {
            props.put("intermediate_size", moeFf);
        }
        return cfg.withFlat(props);
    }

    /**
     * {@code Kolibri1Model.set_gguf_parameters} (kolibri1.py:21-22) after AfmoeModel's keys: the shared expert width,
     * the {@code layer_types} sliding pattern and the sigmoid gating function.
     */
    public static void kolibri1Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        afmoeMeta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, cfg.requireInt("shared_expert_intermediate_size"));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, slidingPattern(cfg, "layer_types"));
        gatingSigmoid(kv);
    }

    // ------------------------------------------------------------------ Cohere2MoeModel

    /** The NextN block count of Cohere2MoeModel (command_r.py:74-79): trunk + {@code num_nextn_predict_layers}. */
    public static int cohere2BlockCount(com.eonmux.neuralfree.gguf.hf.config.HfConfig cfg) {
        return cfg.blockCount() + cfg.findInt(0, "num_nextn_predict_layers");
    }

    /**
     * {@code Cohere2MoeModel.__init__}'s appended MTP block count (command_r.py:74-79) — the block count function of
     * the command_r.py row; the draft-only MTP filtering itself is the load's block-count option, and the filter keeps
     * every name.
     */
    public static HfNameFilter cohere2Filter(HfLoadContext ctx) {
        int n = ctx.config().blockCount();
        return (name, base) -> {
            String t = base.apply(name);
            if (t == null) {
                return null;
            }
            java.util.regex.Matcher m = Pattern.compile("^model\\.layers\\.(\\d{1,9})\\.").matcher(t);
            if (m.find() && Integer.parseInt(m.group(1)) >= n && !ctx.options().includeMtp()) {
                // the appended MTP block tensors mtp-only loads carry is out of scope for a whole-model conversion
                return t;
            }
            return t;
        };
    }

    /**
     * {@code Cohere2MoeModel.set_gguf_parameters} (command_r.py:91-116) after the generic metadata: the logit scale,
     * the sliding window and its {@code layer_types} pattern, the fused expert FFN width, the dense lead, the routing
     * norm, optional shared experts, NextN, and the full {@code head_dim} RoPE with no scaling type.
     */
    public static void cohere2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.LOGIT_SCALE, cfg.findFloat(1.0, "logit_scale"));
        int swa = cfg.requireInt("sliding_window");
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, swa);
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, slidingPattern(cfg, "layer_types"));
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        int ffn = cfg.requireInt("intermediate_size");
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, ffn);
        List<Object> mlpTypes = typesOrEmpty(cfg, "mlp_layer_types");
        int lead = cfg.findInt(0, "first_k_dense_replace");
        for (int i = 0; i < mlpTypes.size(); i++) {
            if (!"dense".equals(mlpTypes.get(i))) {
                lead = i;
                break;
            }
            lead = i + 1;
        }
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, lead);
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(false, "norm_topk_prob"));
        int shared = cfg.findInt(0, "num_shared_experts");
        if (shared > 0) {
            String strategy = cfg.findStringOr("average", "shared_expert_combination_strategy");
            if (!"average".equals(strategy)) {
                throw new GgufFormatException(cfg.source() + ": Cohere2 MoE only supports average shared expert"
                        + " combination, got '" + strategy + "'");
            }
            kv.set(GgufKeys.EXPERT_SHARED_COUNT, shared);
            kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, (long) ffn * shared);
        }
        if (ctx.options().includeMtp() && cfg.findInt(0, "num_nextn_predict_layers") > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, cfg.findInt(0, "num_nextn_predict_layers"));
        }
        Object headDim = cfg.find("head_dim").orElse(null);
        if (!(headDim instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": head_dim must be a number, got " + headDim);
        }
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, ((Number) headDim).intValue());
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");
    }

    /**
     * {@code Cohere2MoeModel.modify_tensors} (command_r.py:141-175): every {@code .bias} tensor that stores nonzero
     * values is an error, the zero ones are dropped; per-expert tensors stack per layer into the classic
     * {@code mlp.experts.<part>.weight}; everything else is mapped by name.
     *
     * @throws GgufFormatException for a nonzero bias (command_r.py:143-145)
     */
    public static void cohere2Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith(".bias")) {
            out.drop(hfName, "cohere2 bias tensors are zero and dropped");
            return;
        }
        if (hfName.contains(".mlp.experts.")) {
            HfHookSupport.layerOf(hfName);
            if (!experts(ctx, EXPERTS_STATE, ctx.config().requireInt("num_experts"), GATE_UP_DOWN)
                    .offer(ctx, hfName, out)) {
                throw new GgufFormatException("Unprocessed experts: ['" + hfName + "']");
            }
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** {@code Cohere2MoeModel.prepare_tensors} (command_r.py:177-183): leftover experts are invalid. */
    public static void cohere2Extra(HfLoadContext ctx, TensorSink out) {
        requireOrphans(ctx, EXPERTS_STATE);
    }

    // ------------------------------------------------------------------ Granite SWA rows

    /**
     * {@code Gemma4UnifiedModel} (gemma.py:936-...), the HK-B2 {@code Gemma4Model} hooks plus the
     * {@code suppress_tokens} of the checkpoint's {@code generation_config.json}.
     */
    public static void gemma4UnifiedMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGemma4Hooks.meta(ctx, kv, plan);
        Object suppress = ctx.config().generation().map(g -> g.raw("suppress_tokens")).orElse(null);
        if (suppress instanceof List && !((List<?>) suppress).isEmpty()) {
            List<Integer> ids = new ArrayList<>();
            for (Object v : (List<?>) suppress) {
                if (v instanceof Number) {
                    ids.add(((Number) v).intValue());
                }
            }
            kv.set(GgufKeys.TOKENIZER_SUPPRESS_TOKENS, ids);
        }
    }

    /**
     * {@code GraniteSWAModel.filter_tensors} (granite.py:83-89) on the Granite filter: {@code sinks} gains
     * {@code .weight} before {@code super()}'s filter runs.
     */
    public static final HfNameFilter GRANITE_SWA_FILTER = (name, base) ->
            HfGraniteHooks.filter(name.endsWith("sinks") ? name + ".weight" : name, base);

    /**
     * {@code GraniteSWAModel.set_gguf_parameters} (granite.py:91-118) applied over {@code swaMeta}: a window default
     * of 128, the {@code layer_types} pattern or the transformers period-4 default, and the RoPE pattern of
     * {@code no_rope_layers} (1 = use RoPE).
     */
    public static void graniteSwaMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGraniteHooks.meta(ctx, kv, plan);
        swaKv(ctx, kv, plan);
    }

    /**
     * {@code GraniteMoeSWAModel.set_gguf_parameters} (granite.py:125-143) over the MoE granite keys plus
     * {@link #graniteSwaMeta}'s keys and the shared expert width.
     */
    public static void graniteMoeSwaMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGraniteHooks.moeMeta(ctx, kv, plan);
        swaKv(ctx, kv, plan);
        Object shared = ctx.config().find("shared_intermediate_size").orElse(null);
        if (shared instanceof Number) {
            kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, ((Number) shared).intValue());
        }
    }

    private static void swaKv(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int swa = cfg.findInt(128, "sliding_window");
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, swa);
        List<Boolean> pattern = flagList(cfg, "layer_types");
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                pattern == null ? periodPattern(ctx.blockCount(), 4) : pattern);
        List<Object> noRope = typesOrEmpty(cfg, "no_rope_layers");
        if (!noRope.isEmpty()) {
            List<Boolean> rope = new ArrayList<>();
            for (Object v : noRope) {
                rope.add(!Boolean.FALSE.equals(v) && !Integer.valueOf(0).equals(v));
            }
            kv.set(GgufKeys.ATTENTION_ROPE_PATTERN, rope);
        }
    }

    private static List<Object> typesOrEmpty(HfConfig cfg, String name) {
        Object v = cfg.flat().get(name);
        return v instanceof List ? (List<Object>) v : java.util.Collections.emptyList();
    }
}
