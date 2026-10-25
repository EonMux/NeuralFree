package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hooks of the sliding-window / sink MoE converter classes, part C, the long tail: {@code LagunaModel}
 * (LC/conversion/laguna.py:17-208), {@code MapleModel} (maple.py:15-129), {@code MellumModel}
 * (mellum.py:15-131), {@code ExaoneMoEModel} (LC/conversion/exaone.py:132-229, composed on the HK-B2 Exaone4 hooks)
 * and {@code SmallThinkerModel} (smallthinker.py:15-83). The Flint/Afmoe/Kolibri-1 and Cohere-2/Granite rows are
 * in {@link HfMoeSwaHooksA}, the MimoV2/Beam classes in {@link HfMoeSwaHooksB} and the Step3.5/MuseSpark classes
 * in {@link HfMoeSwaHooksD}.
 *
 * <p>Every method is a public static hook reference that subclasses compose where the converter calls
 * {@code super()}.</p>
 */
public final class HfMoeSwaHooksC {

    /** The experts of {@code model.layers.<b>.mlp.experts.<x>.<proj>.weight} (laguna.py:169-176, maple, mellum). */
    private static final Pattern EXPERT = Pattern.compile("\\.experts\\.(\\d{1,9})\\.");

    /** The NextN prefix tensors of exaone.py:181-199. */
    private static final Pattern EXAONE_MTP_LAYERS = Pattern.compile("^mtp\\.layers\\.(\\d{1,9})\\.(.+)$");

    /** The shared MTP head tensor remapper of exaone.py:184-196. */
    private static final Map<String, String> EXAONE_MTP_HEAD_RENAMES;

    static {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("mtp.fc", "eh_proj");
        m.put("mtp.pre_fc_norm_embedding", "enorm");
        m.put("mtp.pre_fc_norm_hidden", "hnorm");
        m.put("mtp.norm", "shared_head.norm");
        EXAONE_MTP_HEAD_RENAMES = Collections.unmodifiableMap(m);
    }

    /** The per-load expert collector holder key of the C classes. */
    private static final String EXPERTS_STATE = HfMoeSwaHooksC.class.getName() + ".experts";

    private HfMoeSwaHooksC() {
    }

    // ------------------------------------------------------------------ shared helpers

    /** The reference {@code HfVocabWriters.gpt2} vocabulary of the classes that keep the default vocabulary. */
    private static Map<String, Object> gpt2(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    /** The per-load expert collector shared with part A's holders. */
    private static HfHookSupport.ExpertCollector experts(HfLoadContext ctx, int nExperts, List<String> parts) {
        HfMoeSwaHooksA.Holder holder = ctx.state(EXPERTS_STATE, HfMoeSwaHooksA.Holder.class,
                HfMoeSwaHooksA.Holder::new);
        synchronized (holder) {
            if (holder.collector == null) {
                holder.collector = new HfHookSupport.ExpertCollector(nExperts, parts,
                        HfHookSupport.MergedName.SAME_PREFIX);
            }
            return holder.collector;
        }
    }

    /** A layer's expert tensor under {@code prefix} that stacks when all experts arrived. */
    private static boolean offer(HfLoadContext ctx, String hfName, TensorSink out, int nExperts,
                                 List<String> parts) {
        Matcher m = EXPERT.matcher(hfName);
        return m.find() && experts(ctx, nExperts, parts).offer(ctx, hfName, out);
    }

    /** Writes {@code key} from a required {@code cfg} number. */
    private static void required(HfMetadataBuilder kv, HfConfig cfg, String name, String key) {
        Object v = cfg.find(name).orElse(null);
        if (!(v instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": '" + name + "' must be a number, got " + v);
        }
        kv.set(key, ((Number) v).intValue());
    }

    private static void requiredDouble(HfMetadataBuilder kv, HfConfig cfg, String name, String key) {
        Object v = cfg.find(name).orElse(null);
        if (!(v instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": '" + name + "' must be a number, got " + v);
        }
        kv.set(key, ((Number) v).doubleValue());
    }

    private static void optionalInt(HfMetadataBuilder kv, HfConfig cfg, String name, String key) {
        Object v = cfg.find(name).orElse(null);
        if (v instanceof Number) {
            kv.set(key, ((Number) v).intValue());
        }
    }

    private static void optionalDouble(HfMetadataBuilder kv, HfConfig cfg, String name, String key) {
        Object v = cfg.find(name).orElse(null);
        if (v instanceof Number) {
            kv.set(key, ((Number) v).doubleValue());
        }
    }

    private static void optionalBoolean(HfMetadataBuilder kv, HfConfig cfg, String name, String key) {
        Object v = cfg.find(name).orElse(null);
        if (v instanceof Boolean) {
            kv.set(key, v);
        }
    }

    // ------------------------------------------------------------------ LagunaModel

    /**
     * The per-layer attention gate types of {@code LagunaModel._attn_gate_types} (laguna.py:124-162): the
     * {@code gating_types} array when present, else the scalar {@code gating} ("per-element"/"per-head", a legacy
     * boolean True = per-head); fails loudly for a per-element gate whose {@code gating} field does not declare it.
     */
    static List<String> gateTypes(HfConfig cfg) {
        int n = cfg.blockCount();
        Object gate = cfg.find("gating").orElse(null);
        Object gateTypes = cfg.find("gating_types").orElse(null);
        List<String> norm;
        if (gateTypes instanceof List) {
            if (((List<?>) gateTypes).size() != n) {
                throw new GgufFormatException(cfg.source() + ": gating_types length != num_hidden_layers");
            }
            norm = new ArrayList<>();
            for (Object t : (List<?>) gateTypes) {
                norm.add(normGate(cfg, t));
            }
        } else if (gate instanceof String) {
            norm = Collections.nCopies(n, normGate(cfg, gate));
        } else if (Boolean.TRUE.equals(gate)) {
            norm = Collections.nCopies(n, "per_head");
        } else {
            throw new GgufFormatException(cfg.source() + ": cannot determine attention gate type (gating=" + gate
                    + ", gating_types=" + gateTypes + ")");
        }
        if (norm.contains("per_element") && !(gate instanceof String
                && "per_element".equals(normGate(cfg, gate)))) {
            throw new GgufFormatException(cfg.source() + ": per-element attention gate without `gating ="
                    + " \"per-element\"` would mis-handle the checkpoint");
        }
        return norm;
    }

    /** The hyphen-insensitive normalized gate type. */
    private static String normGate(HfConfig cfg, Object t) {
        String s = String.valueOf(t).replace('-', '_');
        if ("per_element".equals(s) || "per_head".equals(s)) {
            return s;
        }
        throw new GgufFormatException(cfg.source() + ": unrecognised attention gate type " + t);
    }

    /** The leading dense count of {@code mlp_layer_types} (laguna.py:100-107, maple uses layer_types). */
    private static int leadingDense(HfConfig cfg, String name) {
        List<String> types = new ArrayList<>();
        Object v = cfg.find(name).orElse(null);
        if (v instanceof List) {
            for (Object o : (List<?>) v) {
                types.add(String.valueOf(o));
            }
        }
        int dense = 0;
        while (dense < types.size() && "dense".equals(types.get(dense))) {
            dense++;
        }
        return dense;
    }

    /**
     * {@code LagunaModel.set_gguf_parameters} (laguna.py:70-123) after the generic metadata: the vocabulary size, the
     * per-layer head counts, the gate validation, the sliding window when present, the routed and shared expert
     * widths with the sigmoid routing, the dense lead and the partial-rotary RoPE widths of the full and sliding
     * layers.
     */
    public static void lagunaMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        required(kv, cfg, "vocab_size", GgufKeys.VOCAB_SIZE);
        List<Integer> perLayer = cfg.findIntList(Collections.emptyList(), "num_attention_heads_per_layer");
        int fallback = cfg.requireInt("num_attention_heads");
        List<Integer> byLayer = new ArrayList<>();
        if (perLayer.isEmpty()) {
            for (int i = 0; i < cfg.blockCount(); i++) {
                byLayer.add(fallback);
            }
        } else {
            if (perLayer.size() != cfg.blockCount()) {
                throw new GgufFormatException(cfg.source() + ": num_attention_heads_per_layer length "
                        + perLayer.size() + " != num_hidden_layers " + cfg.blockCount());
            }
            byLayer.addAll(perLayer);
        }
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, byLayer);
        gateTypes(cfg);
        optionalSlidingWindow(kv, cfg, "sliding_window");
        required(kv, cfg, "moe_intermediate_size", GgufKeys.EXPERT_FEED_FORWARD_LENGTH);
        required(kv, cfg, "shared_expert_intermediate_size", GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH);
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, true);
        requiredDouble(kv, cfg, "moe_routed_scaling_factor", GgufKeys.EXPERT_WEIGHTS_SCALE);
        HfMoeSwaHooksA.gatingSigmoid(kv);
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, leadingDense(cfg, "mlp_layer_types"));
        double fullFactor = cfg.findFloat(1.0, "partial_rotary_factor");
        int headDim = cfg.requireInt("head_dim");
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) (headDim * fullFactor));
        Object swaFactor = cfg.find("sliding_attention.partial_rotary_factor").orElse(null);
        if (swaFactor instanceof Number) {
            kv.set(GgufKeys.ROPE_DIMENSION_COUNT_SWA, (int) (headDim * ((Number) swaFactor).doubleValue()));
        }
    }

    /** {@code add_sliding_window} when the window is present and positive (laguna.py:95-98). */
    private static void optionalSlidingWindow(HfMetadataBuilder kv, HfConfig cfg, String name) {
        Object v = cfg.find(name).orElse(null);
        if (v instanceof Number && ((Number) v).intValue() > 0) {
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, ((Number) v).intValue());
        }
    }

    /**
     * {@code LagunaModel.modify_tensors} (laguna.py:165-207): the numbered per-expert weights stack per layer; the
     * gate projection width is cross-checked against the declared gate type; everything else maps.
     */
    public static void lagunaTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith("self_attn.g_proj.weight") && !EXPERT.matcher(hfName).find()) {
            int bid = HfHookSupport.layerOf(hfName);
            List<Integer> perLayer = ctx.config().findIntList(Collections.emptyList(),
                    "num_attention_heads_per_layer");
            int nHead = perLayer.size() > bid ? perLayer.get(bid) : ctx.config().requireInt("num_attention_heads");
            int headDim = ctx.config().requireInt("head_dim");
            String gateType = gateTypes(ctx.config()).get(bid);
            long expected = "per_element".equals(gateType) ? (long) nHead * headDim : nHead;
            if (src.shape()[0] != expected) {
                throw new GgufFormatException("Laguna layer " + bid + ": g_proj output width " + src.shape()[0]
                        + " contradicts the declared " + gateType + " gate (expected " + expected + ")");
            }
        }
        if (offer(ctx, hfName, out, ctx.config().requireInt("num_experts"), HfMoeSwaHooksA.GATE_UP_DOWN)) {
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code LagunaModel.prepare_tensors}: leftover experts are invalid. */
    public static void lagunaExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeSwaHooksA.requireOrphans(ctx, EXPERTS_STATE);
    }

    /**
     * {@code LagunaModel.set_vocab} (laguna.py:31-56): the GPT-2 vocabulary with the resolved
     * {@code chat_template.jinja} preferred over the tokenizer_config include directive and the extra turn-end of an
     * eos list registered as eot.
     */
    public static Map<String, Object> lagunaVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(gpt2(ctx, pre));
        Map<String, Object> template = HfChatTemplates.resolve(ctx.tokenizerFiles(), null);
        String resolved = (String) template.get(GgufKeys.TOKENIZER_CHAT_TEMPLATE);
        if (resolved != null) {
            md.put(GgufKeys.TOKENIZER_CHAT_TEMPLATE, resolved);
        }
        Object eosIds = ctx.config().raw().raw("eos_token_id");
        if (eosIds instanceof List && !((List<?>) eosIds).isEmpty()) {
            int bos = ctx.config().findInt(0, "bos_token_id");
            for (Object id : (List<?>) eosIds) {
                if (id instanceof Number && ((Number) id).intValue() != bos) {
                    md.put(GgufKeys.TOKENIZER_EOT_ID, ((Number) id).intValue());
                    break;
                }
            }
        }
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ MapleModel

    /**
     * {@code MapleModel.set_gguf_parameters} (maple.py:20-41) after the generic metadata: the vocabulary size, the
     * partial-rotary rope width, the sliding window and its {@code layer_types} pattern, the routed expert width and
     * the reference SwiGLU gate clamp of 7.0.
     */
    public static void mapleMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        if (!"silu".equals(cfg.findStringOr("silu", "hidden_act"))) {
            throw new GgufFormatException(cfg.source() + ": the Maple reference is silu-only, got "
                    + cfg.findStringOr("silu", "hidden_act"));
        }
        required(kv, cfg, "vocab_size", GgufKeys.VOCAB_SIZE);
        int headDim = cfg.findInt(Math.floorDiv(cfg.requireInt("hidden_size"),
                cfg.requireInt("num_attention_heads")), "head_dim");
        double partial = cfg.findFloat(1.0, "partial_rotary_factor");
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) (headDim * partial));
        required(kv, cfg, "sliding_window", GgufKeys.ATTENTION_SLIDING_WINDOW);
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                HfMoeSwaHooksA.slidingPattern(cfg, "layer_types"));
        required(kv, cfg, "moe_intermediate_size", GgufKeys.EXPERT_FEED_FORWARD_LENGTH);
        kv.set(GgufKeys.SWIGLU_CLAMP_EXP, clampList(ctx.blockCount(), 7.0));
    }

    private static List<Double> clampList(int blocks, double value) {
        List<Double> out = new ArrayList<>(blocks);
        for (int i = 0; i < blocks; i++) {
            out.add(value);
        }
        return out;
    }

    /**
     * {@code MapleModel.modify_tensors} (maple.py:56-96, mixed_models): the per-expert weights stack per layer,
     * everything else maps.
     */
    public static void mapleTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (offer(ctx, hfName, out, ctx.config().requireInt("num_experts"), HfMoeSwaHooksA.GATE_UP_DOWN)) {
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code MapleModel.prepare_tensors} (maple.py:99-108): leftover experts are invalid. */
    public static void mapleExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeSwaHooksA.requireOrphans(ctx, EXPERTS_STATE);
    }

    /** {@code MapleModel.set_vocab} (maple.py:111-118): the GPT-2 BPE vocabulary. */
    public static Map<String, Object> mapleVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return gpt2(ctx, pre);
    }

    // ------------------------------------------------------------------ MellumModel

    /**
     * {@code MellumModel.__init__}'s block count (mellum.py:23-37): the trunk plus
     * {@code num_nextn_predict_layers}/{@code num_mtp_layers} MTP blocks; MTP-only loads demand them.
     */
    public static int mellumBlocks(HfLoadContext ctx) {
        return HfQwenHooks.mtpBlockCount(ctx);
    }

    /**
     * {@code MellumModel.set_gguf_parameters} (mellum.py:66-88) after the generic metadata: the routed expert width,
     * the optional sliding window with its {@code layer_types} + MTP-layer pattern, the routing norm and NextN.
     */
    public static void mellumMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        optionalInt(kv, cfg, "moe_intermediate_size", GgufKeys.EXPERT_FEED_FORWARD_LENGTH);
        Object useSwa = cfg.find("use_sliding_window").orElse(null);
        Object window = cfg.find("sliding_window").orElse(null);
        boolean useSliding = useSwa == null || Boolean.TRUE.equals(useSwa);
        int nextn = Math.max(0, ctx.blockCount() - cfg.blockCount());
        if (useSliding && window instanceof Number) {
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, ((Number) window).intValue());
            List<String> types = HfMoeSwaHooksB.stringList(cfg, "layer_types");
            List<String> mtpTypes = HfMoeSwaHooksB.stringList(cfg, "mtp_layer_types");
            List<Boolean> pattern = new ArrayList<>();
            for (String t : types) {
                pattern.add("sliding_attention".equals(t));
            }
            for (int i = 0; i < nextn; i++) {
                pattern.add(i < mtpTypes.size() ? "sliding_attention".equals(mtpTypes.get(i)) : true);
            }
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        }
        optionalBoolean(kv, cfg, "norm_topk_prob", GgufKeys.EXPERT_WEIGHTS_NORM);
        if (nextn > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextn);
        }
    }

    /** {@code MellumModel.modify_tensors} (mellum.py:103-131): per-expert stacking; everything else maps. */
    public static void mellumTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("experts") && offer(ctx, hfName, out,
                ctx.config().requireInt("num_local_experts", "num_experts"), HfMoeSwaHooksA.GATE_UP_DOWN)) {
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code MellumModel.prepare_tensors}: leftover experts are invalid. */
    public static void mellumExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeSwaHooksA.requireOrphans(ctx, EXPERTS_STATE);
    }

    // ------------------------------------------------------------------ ExaoneMoEModel

    /**
     * {@code ExaoneMoEModel.__init__}'s block count (exaone.py:136-140): the trunk plus
     * {@code num_nextn_predict_layers}.
     */
    public static int exaoneMoeBlocks(HfLoadContext ctx) {
        return ctx.config().blockCount() + nextn(ctx);
    }

    /** the NextN count of a load: the configured layer count dropped with the no-MTP option. */
    private static int nextn(HfLoadContext ctx) {
        return ctx.options().includeMtp() ? ctx.config().findInt(0, "num_nextn_predict_layers") : 0;
    }

    /**
     * {@code ExaoneMoEModel.set_gguf_parameters} (exaone.py:142-157) after the HK-B2 Exaone4 metadata: the shared and
     * routed expert widths, the routing scale/norm, the dense lead, NextN and plain RoPE scaling.
     */
    public static void exaoneMoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfBaseDenseHooks.exaone4Meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        required(kv, cfg, "moe_intermediate_size", GgufKeys.EXPERT_FEED_FORWARD_LENGTH);
        required(kv, cfg, "num_shared_experts", GgufKeys.EXPERT_SHARED_COUNT);
        int moeFf = cfg.requireInt("moe_intermediate_size");
        int shared = cfg.requireInt("num_shared_experts");
        if (shared <= 0 || (long) moeFf * shared > Integer.MAX_VALUE) {
            throw new GgufFormatException(cfg.source() + ": MoE width " + moeFf + " does not split into " + shared
                    + " shared experts");
        }
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, moeFf * shared);
        requiredDouble(kv, cfg, "routed_scaling_factor", GgufKeys.EXPERT_WEIGHTS_SCALE);
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(true, "norm_topk_prob"));
        int lead = cfg.findInt(cfg.findInt(0, "first_last_k_dense_replace"), "first_k_dense_replace");
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, lead);
        int nextnN = nextn(ctx);
        if (nextnN > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextnN);
        }
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");
    }

    /**
     * {@code ExaoneMoEModel.modify_tensors} (exaone.py:159-229, on the HK-B2 Exaone4 hooks): the MTP tensors are
     * renamed into the appended layer slots and the shared fc/norm/head tensors are replicated into
     * <em>every</em> NextN block (having the shared head so a draft has a head per block, exaone.py:181-199).
     */
    public static void exaoneMoeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        int trunk = ctx.config().blockCount();
        int nextnN = nextn(ctx);
        if (hfName.startsWith("mtp.")) {
            Matcher layerM = EXAONE_MTP_LAYERS.matcher(hfName);
            if (layerM.matches()) {
                String renamed = "model.layers." + (trunk + Integer.parseInt(layerM.group(1))) + "."
                        + layerM.group(2);
                HfTensorHook.DEFAULT.apply(ctx, renamed, src, out);
                return;
            }
            String stem = hfName.endsWith(".weight") ? hfName.substring(0, hfName.length() - ".weight".length())
                    : hfName;
            String rename = EXAONE_MTP_HEAD_RENAMES.get(stem);
            if (rename != null) {
                String suffix = hfName.endsWith(".weight") ? ".weight" : "";
                for (int bidMtp = 0; bidMtp < nextnN; bidMtp++) {
                    HfTensorHook.DEFAULT.apply(ctx, "model.layers." + (trunk + bidMtp) + "." + rename + suffix,
                            src, out);
                }
                return;
            }
        }
        if (offer(ctx, hfName, out, ctx.config().requireInt("num_local_experts", "num_experts"),
                HfMoeSwaHooksA.GATE_UP_DOWN)) {
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code ExaoneMoEModel.prepare_tensors} (exaone.py:227-229): leftover experts are invalid. */
    public static void exaoneMoeExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeSwaHooksA.requireOrphans(ctx, EXPERTS_STATE);
    }

    // ------------------------------------------------------------------ SmallThinkerModel

    /**
     * {@code SmallThinkerModel.set_gguf_parameters} (smallthinker.py:17-49) after the generic metadata: the routed
     * counts, the shared MoE FFN width as the plain feed-forward length too, the router's gate function (softmax when
     * the primary router applies one, sigmoid otherwise) and the sliding window when a layout entry is nonzero and a
     * size is set.
     */
    public static void smallthinkerMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        optionalInt(kv, cfg, "moe_num_primary_experts", GgufKeys.EXPERT_COUNT);
        optionalInt(kv, cfg, "moe_num_active_primary_experts", GgufKeys.EXPERT_USED_COUNT);
        optionalInt(kv, cfg, "moe_ffn_hidden_size", GgufKeys.EXPERT_FEED_FORWARD_LENGTH);
        optionalInt(kv, cfg, "moe_ffn_hidden_size", GgufKeys.FEED_FORWARD_LENGTH);
        HfMoeSwaHooksA.gatingSoftmax(kv);
        optionalListWindow(kv, cfg);
    }

    private static void optionalListWindow(HfMetadataBuilder kv, HfConfig cfg) {
        List<String> layout = HfMoeSwaHooksB.stringList(cfg, "sliding_window_layout");
        for (String t : layout) {
            boolean nonzero = !Boolean.FALSE.toString().equals(t) && !"0".equals(t) && !"false".equals(t);
            boolean enabled;
            try {
                enabled = !nonzero || Integer.parseInt(t) != 0;
            } catch (NumberFormatException e) {
                continue;
            }
            if (enabled) {
                Object window = cfg.find("sliding_window_size").orElse(null);
                if (window instanceof Number && ((Number) window).intValue() > 0) {
                    kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, ((Number) window).intValue());
                }
                break;
            }
        }
    }

    /**
     * {@code SmallThinkerModel.modify_tensors} (smallthinker.py:51-83): the {@code block_sparse_moe.experts.<x>.<w>.}
     * tensors (parts down/gate/up) stack per layer, everything else maps.
     */
    public static void smallthinkerTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("experts")) {
            int nExperts = ctx.config().findInt(0, "moe_num_primary_experts");
            if (nExperts <= 0) {
                nExperts = ctx.config().requireInt("num_local_experts", "num_experts");
            }
            if (!experts(ctx, nExperts, java.util.Arrays.asList("down", "gate", "up")).offer(ctx, hfName, out)) {
                throw new GgufFormatException("Unprocessed experts: ['" + hfName + "']");
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code SmallThinkerModel.prepare_tensors} (smallthinker.py:74-83): leftover experts are invalid. */
    public static void smallthinkerExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeSwaHooksA.requireOrphans(ctx, EXPERTS_STATE);
    }
}
