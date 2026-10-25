package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfRope;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfSpecialVocab;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The hooks of {@code Gemma4Model} (LC/conversion/gemma.py:634-810), which extends {@code Gemma3Model}
 * ({@link HfGemmaHooks}): no norm shift, the {@code gemma4} vocabulary, per-layer key/value widths and KV heads for
 * full and sliding-window layers, proportional RoPE frequency factors, the router and per-expert scales, and the
 * NVFP4 experts of ModelOpt checkpoints stacked per layer ({@code _generate_nvfp4_tensors},
 * LC/conversion/base.py:1032-1127).
 */
public final class HfGemma4Hooks {

    /** The tokens the chat parser reads, always rendered (LC/conversion/gemma.py:647). */
    static final Set<String> VISIBLE_TOKENS = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
            "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>",
            "<|\"|>")));

    private static final int USER_DEFINED = 4;
    private static final String FFN_GATE_INP = "FFN_GATE_INP";
    private static final String FFN_DOWN_EXP = "FFN_DOWN_EXP";
    private static final String ROPE_FREQS = "ROPE_FREQS";
    private static final String WEIGHT = ".weight";
    private static final String SCALE = ".scale";
    private static final String FULL_ATTENTION = "full_attention";
    private static final String PER_EXPERT_SCALE = ".per_expert_scale";
    private static final String ROUTER_PER_EXPERT_SCALE = "router.per_expert_scale";
    private static final String WEIGHT_SCALE_2 = ".weight_scale_2";
    private static final List<String> EXPERT_SUFFIXES = Arrays.asList(WEIGHT, ".weight_scale", WEIGHT_SCALE_2,
            ".input_scale");
    private static final Pattern LAYER = Pattern.compile("\\.layers\\.(\\d{1,9})\\.");
    private static final Pattern PER_EXPERT = Pattern.compile(
            "\\.experts\\.\\d{1,9}\\.(gate_proj|up_proj|down_proj)\\.weight$");
    private static final String EXPERT_STATE = HfGemma4Hooks.class.getName() + ".experts";

    private HfGemma4Hooks() {
    }

    // ------------------------------------------------------------------ vocabulary

    /**
     * {@code Gemma4Model.set_vocab} (LC/conversion/gemma.py:641-669): the {@code LlamaHfVocab} tokens with tokenizer
     * model {@code gemma4} and no pre-tokenizer name, the chat-parser tokens USER_DEFINED, the merges, no space
     * prefix and a BOS token.
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        Map<String, Object> base = HfVocabWriters.llamaHf(HfHookSupport.vocabInput(ctx), pre);
        List<?> tokens = (List<?>) base.get(GgufKeys.TOKENIZER_LIST);
        List<?> types = (List<?>) base.get(GgufKeys.TOKENIZER_TOKEN_TYPE);
        List<Integer> visibleTypes = new ArrayList<>(types.size());
        for (int i = 0; i < types.size(); i++) {
            boolean visible = VISIBLE_TOKENS.contains(tokens.get(i));
            visibleTypes.add(visible ? USER_DEFINED : ((Number) types.get(i)).intValue());
        }
        List<String> merges = HfSpecialVocab.load(files, files.readTokenizerJson(), ctx.config().raw(), true,
                tokens.size()).merges();
        Map<String, Object> md = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : base.entrySet()) {
            String key = e.getKey();
            if (key.equals(GgufKeys.TOKENIZER_PRE)) {
                continue;
            }
            if (key.equals(GgufKeys.TOKENIZER_MODEL)) {
                md.put(key, "gemma4");
            } else if (key.equals(GgufKeys.TOKENIZER_TOKEN_TYPE)) {
                md.put(key, Collections.unmodifiableList(visibleTypes));
                if (!merges.isEmpty()) {
                    md.put(GgufKeys.TOKENIZER_MERGES, merges);
                }
            } else {
                md.put(key, e.getValue());
            }
        }
        md.put(GgufKeys.TOKENIZER_ADD_PREFIX, false);
        md.put(GgufKeys.TOKENIZER_ADD_BOS, true);
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ metadata

    /**
     * {@code Gemma4Model.set_gguf_parameters} (LC/conversion/gemma.py:671-733) after the Gemma 3 metadata: shared-KV
     * layers, the per-layer input width, the sliding-window pattern of {@code layer_types}, the full
     * ({@code global_head_dim} or {@code per_layer_config}) and SWA ({@code head_dim}) key/value widths, the expert
     * FFN width, the FFN array doubled on KV-shared layers with {@code use_double_wide_mlp}, the per-layer KV head
     * counts, and the full/SWA RoPE dimensions.
     *
     * @throws GgufFormatException when a required value is missing
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        HfGemmaHooks.meta(ctx, kv, plan);
        int sharedKv = cfg.requireInt("num_kv_shared_layers");
        kv.set(GgufKeys.ATTENTION_SHARED_KV_LAYERS, sharedKv);
        kv.set(GgufKeys.EMBEDDING_LENGTH_PER_LAYER, cfg.findInt(0, "hidden_size_per_layer_input"));
        List<Boolean> swaLayers = HfBaseDenseHooks.slidingFlags(cfg);
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, swaLayers);
        int headDimFull = headDimFull(cfg);
        int headDimSwa = cfg.requireInt("head_dim");
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDimFull);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, headDimFull);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH_SWA, headDimSwa);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_SWA, headDimSwa);
        Optional<Object> expertFf = cfg.find("expert_intermediate_size", "moe_intermediate_size");
        if (expertFf.isPresent()) {
            kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, expertFf.get() instanceof List ? expertFf.get()
                    : (Object) cfg.findLong("expert_intermediate_size", "moe_intermediate_size").getAsLong());
        }
        if (cfg.findBool(false, "use_double_wide_mlp")) {
            int nFf = cfg.requireInt("intermediate_size");
            int firstShared = ctx.blockCount() - sharedKv;
            List<Integer> ff = new ArrayList<>(ctx.blockCount());
            for (int il = 0; il < ctx.blockCount(); il++) {
                ff.add(il < firstShared ? nFf : Math.multiplyExact(nFf, 2));
            }
            kv.set(GgufKeys.FEED_FORWARD_LENGTH, ff);
        }
        OptionalInt kvFull = cfg.findInt("num_global_key_value_heads");
        if (!kvFull.isPresent()) {
            kvFull = perLayerFull(cfg, "num_key_value_heads");
        }
        OptionalInt kvSwa = cfg.findInt("num_key_value_heads");
        if (kvFull.isPresent() && kvSwa.isPresent()) {
            List<Integer> heads = new ArrayList<>(swaLayers.size());
            for (boolean swa : swaLayers) {
                heads.add(swa ? kvSwa.getAsInt() : kvFull.getAsInt());
            }
            kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, heads);
        }
        double rotarySwa = HfBaseDenseHooks.numberOr(cfg.rope().raw().get("partial_rotary_factor"), 1.0,
                cfg.source() + ": rope_parameters.partial_rotary_factor");
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, headDimFull);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT_SWA, (int) (headDimSwa * rotarySwa));
    }

    /**
     * The key/value width of full-attention layers: {@code global_head_dim}, else the {@code head_dim} of the first
     * {@code per_layer_config} entry of a full-attention layer.
     *
     * @throws GgufFormatException when neither gives one, or the width is not positive
     */
    static int headDimFull(HfConfig cfg) {
        OptionalInt v = cfg.findInt("global_head_dim");
        if (!v.isPresent()) {
            v = perLayerFull(cfg, "head_dim");
        }
        if (!v.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": Gemma 4 needs global_head_dim or a per_layer_config"
                    + " head_dim for a full-attention layer");
        }
        if (v.getAsInt() <= 0) {
            throw new GgufFormatException(cfg.source() + ": the full-attention head width must be positive, got "
                    + v.getAsInt());
        }
        return v.getAsInt();
    }

    /**
     * {@code key} of the first {@code per_layer_config} entry, in file order, whose layer is a full-attention layer
     * of {@code layer_types} and that sets it.
     */
    private static OptionalInt perLayerFull(HfConfig cfg, String key) {
        Object plc = cfg.flat().get("per_layer_config");
        if (!(plc instanceof Map)) {
            return OptionalInt.empty();
        }
        Object lt = cfg.flat().get("layer_types");
        List<?> layerTypes = lt instanceof List ? (List<?>) lt : Collections.emptyList();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) plc).entrySet()) {
            String where = cfg.source() + ": per_layer_config." + e.getKey();
            int idx;
            try {
                idx = Integer.parseInt(String.valueOf(e.getKey()).trim());
            } catch (NumberFormatException ex) {
                throw new GgufFormatException(where + ": the key is not a layer index", ex);
            }
            if (idx < 0 || idx >= layerTypes.size() || !FULL_ATTENTION.equals(layerTypes.get(idx))
                    || !(e.getValue() instanceof Map) || !((Map<?, ?>) e.getValue()).containsKey(key)) {
                continue;
            }
            return OptionalInt.of(HfBaseDenseHooks.toInt(((Map<?, ?>) e.getValue()).get(key), where + "." + key));
        }
        return OptionalInt.empty();
    }

    // ------------------------------------------------------------------ extra tensors

    /**
     * {@code Gemma4Model.generate_extra_tensors} (LC/conversion/gemma.py:735-762): the {@code rope_freqs} of the
     * full-attention layers' proportional RoPE, {@code [1.0] * n_rot + [1e30] * n_unrot}, so that the unrotated
     * dimensions stay fixed under NeoX RoPE. Also checks that every layer's per-expert tensors were stacked and every
     * per-expert scale left to the down experts was planned.
     *
     * @throws GgufFormatException when the full-attention RoPE is not {@code proportional}, has no
     *                             {@code partial_rotary_factor} in [0, 1], or a deferred per-expert scale or a layer's
     *                             experts were not planned
     */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        expertState(ctx).requireComplete();
        HfConfig cfg = ctx.config();
        Optional<HfRope.Params> full = cfg.rope().layer(FULL_ATTENTION);
        if (!full.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": Gemma 4 needs rope_parameters.full_attention");
        }
        HfRope.Params p = full.get();
        if (!"proportional".equals(p.type().orElse(null))) {
            throw new GgufFormatException(cfg.source() + ": rope_parameters.full_attention.rope_type must be"
                    + " 'proportional', got " + p.type().orElse(null));
        }
        if (!p.partialRotaryFactor().isPresent()) {
            throw new GgufFormatException(cfg.source() + ": rope_parameters.full_attention.partial_rotary_factor is"
                    + " required");
        }
        double rotary = p.partialRotaryFactor().getAsDouble();
        if (!(rotary >= 0 && rotary <= 1)) {
            throw new GgufFormatException(cfg.source() + ": rope_parameters.full_attention.partial_rotary_factor must"
                    + " be in [0, 1], got " + rotary);
        }
        out.add(ctx.names().ggufName(ROPE_FREQS, -1, WEIGHT),
                RopeTensors.proportionalFreqs(headDimFull(cfg), rotary));
    }

    // ------------------------------------------------------------------ names and index

    /**
     * {@code Gemma4Model.filter_tensors} (LC/conversion/gemma.py:789-797): {@code per_dim_scale} and
     * {@code layer_scalar} get {@code .weight}, and so does an {@code .experts.} tensor without a weight or scale
     * suffix.
     */
    public static String filter(String name, UnaryOperator<String> base) {
        String n = name;
        if (n.endsWith("per_dim_scale") || n.endsWith("layer_scalar")) {
            n = n + WEIGHT;
        }
        if (n.contains(".experts.") && EXPERT_SUFFIXES.stream().noneMatch(n::endsWith)) {
            n = n + WEIGHT;
        }
        return base.apply(n);
    }

    /**
     * {@code Gemma4Model._generate_nvfp4_tensors} (LC/conversion/gemma.py:764-787), on the checkpoint names before
     * the quantization rewrite: a layer's {@code router.per_expert_scale} scales each expert's output, which is where
     * the NVFP4 expert scale applies, so when the layer's experts are NVFP4 ({@code down_proj.weight_scale_2}
     * present) each expert's {@code weight_scale_2} becomes {@code weight_scale_2 * per_expert_scale[e]} (float32)
     * and the per-expert scale is removed.
     *
     * @throws GgufFormatException when only some of a layer's experts are NVFP4, or the per-expert scale has fewer
     *                             values than experts
     */
    public static HfTensorIndex index(HfLoadContext ctx, HfTensorIndex index) throws IOException {
        int nExperts = ctx.config().findInt(0, "num_local_experts", "num_experts");
        Map<String, RowSource> replace = new LinkedHashMap<>();
        Set<String> remove = new LinkedHashSet<>();
        for (String name : index.names()) {
            if (!name.endsWith(".router" + PER_EXPERT_SCALE)) {
                continue;
            }
            Matcher m = LAYER.matcher(name);
            if (!m.find()) {
                continue;
            }
            String bid = m.group(1);
            String prefix = name.substring(0, m.end());
            List<String> targets = new ArrayList<>(Math.max(nExperts, 0));
            int present = 0;
            for (int e = 0; e < nExperts; e++) {
                String t = prefix + "experts." + e + ".down_proj" + WEIGHT_SCALE_2;
                targets.add(t);
                if (index.has(t)) {
                    present++;
                }
            }
            if (present == 0) {
                continue;
            }
            if (present != targets.size()) {
                throw new GgufFormatException("layer " + bid + ": partial NVFP4 quantization across experts");
            }
            float[] r = readFloats(index.get(name), name, HfConfig.MAX_EXPERTS);
            if (r.length < nExperts) {
                throw new GgufFormatException("tensor '" + name + "' has " + r.length + " values for " + nExperts
                        + " experts");
            }
            remove.add(name);
            for (int e = 0; e < nExperts; e++) {
                String t = targets.get(e);
                RowSource s = index.get(t);
                float[] v = readFloats(s, t, HfConfig.MAX_EXPERTS);
                float[] scaled = new float[v.length];
                for (int i = 0; i < v.length; i++) {
                    scaled[i] = v[i] * r[e];
                }
                replace.put(t, RopeTensors.fromFloats(s.shape(), scaled));
            }
        }
        return remove.isEmpty() ? index : index.withReplaced(replace, remove);
    }

    /** The values of a small tensor as float32, row-major. */
    static float[] readFloats(RowSource src, String name, long maxElements) throws IOException {
        long n = 1;
        for (long d : src.shape()) {
            n = Math.multiplyExact(n, d);
        }
        if (n > maxElements) {
            throw new GgufFormatException("tensor '" + name + "' has " + n + " values; at most " + maxElements
                    + " are expected");
        }
        RowSource f = ElementTransforms.convert(src, GGMLType.F32);
        long bytes = Math.multiplyExact(f.rows(), f.rowBytes());
        if (bytes != 4 * n) {
            throw new GgufFormatException("tensor '" + name + "' does not decode to " + n + " float32 values");
        }
        ByteBuffer b = ByteBuffer.allocate((int) bytes).order(ByteOrder.LITTLE_ENDIAN);
        f.readRows(0, Math.toIntExact(f.rows()), b);
        float[] out = new float[(int) n];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getFloat(4 * i);
        }
        return out;
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code Gemma4Model.modify_tensors} (LC/conversion/gemma.py:799-810): {@code router.scale} becomes
     * {@code ffn_gate_inp.scale}, a remaining {@code per_expert_scale} becomes {@code ffn_down_exps.scale}, and the
     * rest goes through the Gemma 3 step without the norm shift. Per-expert {@code gate_proj}/{@code up_proj}/
     * {@code down_proj} weights (the NVFP4 experts of ModelOpt checkpoints) are stacked per layer and projection under
     * {@code model.layers.<b>.mlp.experts.<proj>.weight} ({@code _flush_nvfp4_experts},
     * LC/conversion/base.py:1105-1127); their per-expert scales become the stacked tensor's {@code .scale} and
     * {@code .input_scale}.
     *
     * <p>A layer's {@code router.per_expert_scale} that the index hook did not fold is planned with the layer's down
     * experts: as {@code ffn_down_exps.scale} when they carry no output scale, else multiplied into that scale per
     * expert in float32 ({@code weight_scale_2[e] * per_expert_scale[e]}, as the converter folds it for NVFP4
     * experts), so the one {@code ffn_down_exps.scale} slot holds both factors and the expert weights stay in their
     * quantized format.</p>
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        try {
            plan(ctx, hfName, src, new PerExpertScaleSink(ctx, hfName, out));
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
    }

    private static void plan(HfLoadContext ctx, String hfName, RowSource src, PerExpertScaleSink sink)
            throws IOException {
        if (hfName.endsWith("router.scale")) {
            sink.add(ctx.names().ggufName(FFN_GATE_INP, HfHookSupport.layerOf(hfName), SCALE), src);
            return;
        }
        if (hfName.contains(PER_EXPERT_SCALE)) {
            if (hasDownExperts(ctx, hfName)) {
                expertState(ctx).defer(hfName);
                sink.consume(hfName);
            } else {
                sink.add(ctx.names().ggufName(FFN_DOWN_EXP, HfHookSupport.layerOf(hfName), SCALE), src);
            }
            return;
        }
        Matcher m = PER_EXPERT.matcher(hfName);
        if (m.find()) {
            if (!expertState(ctx).collector(ctx, m.group(1)).offer(ctx, hfName, sink)) {
                throw new IllegalStateException("expert tensor '" + hfName + "' was not taken by its collector");
            }
            return;
        }
        HfGemmaHooks.plan(ctx, hfName, src, sink, false);
    }

    /**
     * True when {@code hfName} is a layer's {@code router.per_expert_scale} and the index holds that layer's down
     * experts, fused ({@code experts.down_proj.weight}) or per expert ({@code experts.0.down_proj.weight}).
     */
    private static boolean hasDownExperts(HfLoadContext ctx, String hfName) {
        String prefix = layerPrefix(hfName);
        if (prefix == null || !hfName.equals(prefix + ROUTER_PER_EXPERT_SCALE)) {
            return false;
        }
        HfTensorIndex index = ctx.index();
        return index.has(prefix + "experts.down_proj" + WEIGHT) || index.has(prefix + "experts.0.down_proj" + WEIGHT);
    }

    /** The name up to and including its first {@code .layers.<b>.}, or null when it has none. */
    private static String layerPrefix(String hfName) {
        Matcher m = LAYER.matcher(hfName);
        return m.find() ? hfName.substring(0, m.end()) : null;
    }

    private static ExpertState expertState(HfLoadContext ctx) {
        return ctx.state(EXPERT_STATE, ExpertState.class, ExpertState::new);
    }

    /** The checkpoint name {@code model.layers.<b>.mlp.experts.<part><tail>} the stacked experts are mapped under. */
    private static String mergedName(String prefix, String part, String tail) {
        return "model.layers." + HfHookSupport.layerOf(prefix) + ".mlp.experts." + part + tail;
    }

    /**
     * The per-load expert state: one expert collector per projection, as the converter flushes each (layer,
     * projection) on its own, and the per-expert scales left to the down experts.
     */
    private static final class ExpertState {
        private final Map<String, HfHookSupport.ExpertCollector> byPart = new LinkedHashMap<>();
        private final Set<String> deferred = new LinkedHashSet<>();
        private final Set<String> applied = new LinkedHashSet<>();

        synchronized HfHookSupport.ExpertCollector collector(HfLoadContext ctx, String part) {
            HfHookSupport.ExpertCollector c = byPart.get(part);
            if (c == null) {
                int nExperts = ctx.config().requireInt("num_local_experts", "num_experts");
                c = new HfHookSupport.ExpertCollector(nExperts, Collections.singletonList(part),
                        HfGemma4Hooks::mergedName);
                byPart.put(part, c);
            }
            return c;
        }

        synchronized void defer(String perExpertScale) {
            deferred.add(perExpertScale);
        }

        synchronized void applied(String perExpertScale) {
            applied.add(perExpertScale);
        }

        /**
         * Checks that every layer's experts were stacked and every deferred per-expert scale was planned.
         *
         * @throws GgufFormatException naming the first incomplete layer or unplanned per-expert scale
         */
        synchronized void requireComplete() {
            for (HfHookSupport.ExpertCollector c : byPart.values()) {
                c.requireComplete();
            }
            for (String name : deferred) {
                if (!applied.contains(name)) {
                    throw new GgufFormatException("tensor '" + name + "' was left to its layer's down experts, which"
                            + " were never planned");
                }
            }
        }
    }

    /**
     * A sink that plans a layer's {@code router.per_expert_scale} together with the layer's down experts
     * ({@code ffn_down_exps.weight}): their output scale times the per-expert scale when they carry one, else the
     * per-expert scale as {@code ffn_down_exps.scale}. Other tensors pass through.
     */
    private static final class PerExpertScaleSink implements TensorSink {
        private final HfLoadContext ctx;
        private final String hfName;
        private final TensorSink out;

        PerExpertScaleSink(HfLoadContext ctx, String hfName, TensorSink out) {
            this.ctx = ctx;
            this.hfName = hfName;
            this.out = out;
        }

        @Override
        public void add(String ggufName, RowSource src) {
            String prefix = layerPrefix(hfName);
            if (prefix == null) {
                out.add(ggufName, src);
                return;
            }
            int bid = HfHookSupport.layerOf(prefix);
            String scaleName = prefix + ROUTER_PER_EXPERT_SCALE;
            if (!ggufName.equals(ctx.names().ggufName(FFN_DOWN_EXP, bid, WEIGHT)) || !ctx.index().has(scaleName)) {
                out.add(ggufName, src);
                return;
            }
            RowSource perExpert = ctx.index().get(scaleName);
            if (src.outputScale() == null) {
                out.add(ggufName, src);
                out.add(ctx.names().ggufName(FFN_DOWN_EXP, bid, SCALE), perExpert);
            } else {
                float[] factors;
                try {
                    factors = readFloats(perExpert, scaleName, HfConfig.MAX_EXPERTS);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                out.add(ggufName, new PerExpertScaled(src, factors, scaleName));
            }
            if (!out.isConsumed(scaleName)) {
                out.consume(scaleName);
            }
            expertState(ctx).applied(scaleName);
        }

        @Override
        public void addKv(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
            out.addKv(key, type, elementType, value);
        }

        @Override
        public void drop(String name, String reason) {
            out.drop(name, reason);
        }

        @Override
        public boolean isConsumed(String name) {
            return out.isConsumed(name);
        }

        @Override
        public void consume(String name) {
            out.consume(name);
        }
    }

    /**
     * Stacked experts whose output scale is multiplied per expert by {@code factors} in float32; every other aspect is
     * the source's, so a quantized or directly mapped source keeps its format and bytes.
     */
    private static final class PerExpertScaled implements RowSource {
        private final RowSource src;
        private final float[] factors;
        private final String factorName;

        /**
         * @throws GgufFormatException when the factors do not match the source's experts or output scale
         */
        PerExpertScaled(RowSource src, float[] factors, String factorName) {
            long[] shape = src.shape();
            float[] os = src.outputScale();
            if (shape.length < 2 || shape[0] != factors.length || (os.length != 1 && os.length != factors.length)) {
                throw new GgufFormatException("tensor '" + factorName + "' has " + factors.length + " values for "
                        + src.describe() + " " + Arrays.toString(shape) + " with " + os.length + " output scales");
            }
            this.src = src;
            this.factors = factors.clone();
            this.factorName = factorName;
        }

        @Override
        public String describe() {
            return src.describe() + " with its per-expert scale";
        }

        @Override
        public ElementFormat format() {
            return src.format();
        }

        @Override
        public long[] shape() {
            return src.shape();
        }

        @Override
        public long rows() {
            return src.rows();
        }

        @Override
        public long rowBytes() {
            return src.rowBytes();
        }

        @Override
        public boolean rowLocal() {
            return src.rowLocal();
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            src.readRows(first, count, dst);
        }

        @Override
        public long directVirtualOffset() {
            return src.directVirtualOffset();
        }

        @Override
        public float[] outputScale() {
            float[] os = src.outputScale();
            float[] out = new float[factors.length];
            for (int e = 0; e < out.length; e++) {
                out[e] = (os.length == 1 ? os[0] : os[e]) * factors[e];
            }
            return out;
        }

        @Override
        public float[] inputScale() {
            return src.inputScale();
        }

        @Override
        public int[] inputZeroPoint() {
            return src.inputZeroPoint();
        }

        @Override
        public int[] inputPerm() {
            return src.inputPerm();
        }

        @Override
        public ActQuantScheme actQuant() {
            return src.actQuant();
        }

        @Override
        public RowSource foldInputPerm() {
            return new PerExpertScaled(src.foldInputPerm(), factors, factorName);
        }

        @Override
        public RowSource foldOutputScale() {
            return ElementTransforms.foldScale(this);
        }
    }
}
