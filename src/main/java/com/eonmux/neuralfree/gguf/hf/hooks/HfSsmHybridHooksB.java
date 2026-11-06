package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.StTensor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;

/**
 * Hooks of the LFM2 converter family (LC/conversion/lfm2.py): {@code LFM2Model} (lfm2.py:20-51), the decision
 * {@code D1Model} (lfm2.py:89-131), the bidirectional {@code LFM2ColBertModel} (lfm2.py:135-185), the decision
 * {@code D1OmniModel} (lfm2.py:188-276), the MoE {@code LFM2MoeModel} (lfm2.py:279-345) and the audio
 * {@code LFM25AudioTokenizer} (lfm2.py:495-519). Multiple-choice head blocks fold into the same LFM2 trunk as one
 * continuation of its layer arrays.
 */
public final class HfSsmHybridHooksB {

    /**
     * {@code LFM2Model.filter_tensors} (lfm2.py:40-45): the Conformer audio tensors skip and every {@code lfm.}
     * prefix renames to {@code model.} (the audio training tensors), around the plain base filter.
     */
    public static final HfNameFilter LFM2_FILTER = (name, base) -> {
        if (isAudioTensor(name)) {
            return null;
        }
        return base.apply(name.replace("lfm.", "model."));
    };

    /**
     * {@code LFM2MoeModel.filter_tensors} (lfm2.py:320-324): the router bias holds no weights and renames to a
     * {@code .bias} tail; the audio and {@code lfm.} prefixes stay skipped and renamed.
     */
    public static final HfNameFilter LFM2_MOE_FILTER = (name, base) -> {
        if (name.endsWith(".expert_bias")) {
            name = name.substring(0, name.length() - ".expert_bias".length()) + ".expert_bias.bias";
        }
        return LFM2_FILTER.filter(name, base);
    };

    /**
     * {@code LFM2ColBertModel.filter_tensors}: no class filter; the {@code lfm2.} prefix surgery of
     * {@code modify_tensors} lives with {@link #colBertTensor}.
     */
    public static final HfNameFilter LFM2_DENSE_FILTER = LFM2_FILTER;

    /**
     * {@code D1OmniModel.filter_tensors} (lfm2.py:250-261): the vision and audio tower tensors skip, the
     * {@code encoder.} trunk renames to {@code model.}, the doubled head stem renames and the underscored
     * in-projections become dotted, and the type/scoring head tensors lose their {@code head.} prefix.
     */
    public static final HfNameFilter D1_OMNI_FILTER = (name, base) -> {
        if (name.startsWith("vision.") || name.startsWith("audio.")) {
            return null;
        }
        String out = name;
        if (out.startsWith("encoder.")) {
            out = replaceOnce(out, "encoder.", "model.");
        }
        out = replaceOnce(replaceOnce(out, "head.head.layers.", "head.layers."), "in_proj_", "in_proj.");
        if (out.startsWith("head.type_emb") || out.startsWith("head.scorer")) {
            out = HfHookSupport.stripPrefixes(out, "head.");
        }
        if (isAudioTensor(out)) {
            return null;
        }
        return base.apply(out.replace("lfm.", "model."));
    };

    /**
     * {@code LFM25AudioTokenizer.filter_tensors} (lfm2.py:508-514): the ISTFT window and doubled embedding
     * tensors skip and every {@code lin} becomes {@code dense_2_out}.
     */
    public static final HfNameFilter AUDIO_TOKENIZER_FILTER = (name, base) -> {
        if (name.equals("istft.window") || name.startsWith("emb.emb")) {
            return null;
        }
        return LFM2_FILTER.filter(name.replace("lin", "dense_2_out"), base);
    };

    /** {@code ConformerAudioModel.is_audio_tensor} (gemma.py:319-321), the audio tensors of the multimodal builds. */
    static boolean isAudioTensor(String name) {
        return name.contains("audio") || name.contains("codebook") || name.contains("conformer")
                || name.contains("depth_embedding") || name.contains("depthformer")
                || name.contains("depth_linear");
    }

    /** Python's {@code str.replace(old, new, 1)}: the first occurrence only. */
    private static String replaceOnce(String name, String oldPart, String newPart) {
        int at = name.indexOf(oldPart);
        return at < 0 ? name : name.substring(0, at) + newPart + name.substring(at + oldPart.length());
    }

    /** The bidirectional checkpoints take {@code lfm2.}-prefixed names and an extra dense head, lfm2.py:152-157. */
    static final String DENSE_TENSOR_NAME = "dense_2";

    /** The decision checkpoints' systemone chat template fields (the D1 trunk prompt, lfm2.py:97). */
    private static final String D1_SYSTEMONE = yesNoTemplate();

    /** {@code D1Model._systemone_template} (lfm2.py:99-129), reused by the LFM3 d1 vocabulary. */
    static String d1Systemone() {
        return D1_SYSTEMONE;
    }

    /** The D1 Alexandre head blocks come after the trunk; their per-block share is added, lfm2.py:172-179. */
    private static final long D1_MAX_LENGTH = 16384;
    private static final long D1_IMAGE_TEXT_LENGTH = 896;
    private static final long D1_AUDIO_TEXT_LENGTH = 15360;

    private HfSsmHybridHooksB() {
    }

    // ------------------------------------------------------------------ LFM2Model (lfm2.py:20)

    /** {@link Arch#LFM2} for every LFM2 checkpoint, including the decision and bidirectional heads. */
    public static Arch lfm2Arch(HfConfig cfg, com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex index) {
        return Arch.LFM2;
    }

    /** {@link Arch#LFM2MOE} for the MoE checkpoints. */
    public static Arch lfm2MoeArch(HfConfig cfg, com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex index) {
        return Arch.LFM2MOE;
    }

    /**
     * {@code LFM2Model.set_gguf_parameters}' key/value patch (lfm2.py:33-37): {@code num_key_value_heads} becomes
     * the per-layer vector, zero on the conv layers, so the generic metadata writes the vector.
     */
    public static HfConfig lfm2Fixup(HfConfig cfg) {
        Object types = cfg.flat().get("layer_types");
        if (!(types instanceof List)) {
            return cfg;
        }
        Object heads = cfg.flat().get("num_key_value_heads");
        if (!(heads instanceof Number)) {
            return cfg;
        }
        return cfg.withFlat(Map.of("num_key_value_heads", perLayerKv((List<?>) types, "conv", heads)));
    }

    /** The per-layer vector: the scalar on every typed layer except {@code skipType}, 0 there (lfm2.py:34-36). */
    private static List<Number> perLayerKv(List<?> types, String skipType, Object heads) {
        long value = ((Number) heads).longValue();
        List<Number> out = new ArrayList<>(types.size());
        for (Object t : types) {
            out.add(skipType.equals(t) ? 0L : value);
        }
        return out;
    }

    /**
     * {@code LFM2Model.set_gguf_parameters} (lfm2.py:33-49): the generic text-model metadata over the fixed-up
     * configuration, then {@code vocab_size}, the {@code conv_L_cache} of the short convolutions, the RMS
     * epsilon of {@code norm_eps}, and the auto-adjusted feed-forward width.
     */
    public static void lfm2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.VOCAB_SIZE, (long) cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.SHORTCONV_L_CACHE, (long) cfg.requireInt("conv_L_cache"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, cfg.findFloat("norm_eps").getAsDouble());
        addFeedForwardLength(ctx, kv);
    }

    /**
     * {@code LFM2Model._add_feed_forward_length} (lfm2.py:23-31): {@code block_ff_dim} (alias
     * {@code intermediate_size}) rounded by {@code block_auto_adjust_ff_dim}, the optional
     * {@code block_ffn_dim_multiplier} and {@code block_multiple_of}.
     */
    private static void addFeedForwardLength(HfLoadContext ctx, HfMetadataBuilder kv) {
        HfConfig cfg = ctx.config();
        long ffDim = cfg.requireInt("block_ff_dim", "intermediate_size");
        java.util.Optional<Boolean> auto = cfg.findBool("block_auto_adjust_ff_dim");
        if (!auto.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": required key 'block_auto_adjust_ff_dim' is missing"
                    + " (lfm2.py:24)");
        }
        if (auto.get()) {
            ffDim = 2L * ffDim / 3;
            OptionalDouble multiplier = cfg.findFloat("block_ffn_dim_multiplier");
            if (multiplier.isPresent()) {
                ffDim = (long) (multiplier.getAsDouble() * ffDim);
            }
            long multipleOf = cfg.requireInt("block_multiple_of");
            ffDim = multipleOf * ((ffDim + multipleOf - 1) / multipleOf);
        }
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, ffDim);
    }

    /**
     * {@code LFM2Model.modify_tensors} (lfm2.py:47-50): the convolution weights squeeze to two dimensions, the
     * rest map plainly.
     */
    public static void lfm2Tensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource s = src;
        if (hfName.contains("conv.conv")) {
            s = ElementTransforms.squeezeConv(s);
        }
        out.add(ctx.names().mapOrThrow(hfName), s);
    }

    /** The plain {@code TextModel} vocabulary (a HF tokenizer over the checkpoint files). */
    public static Map<String, Object> lfm2Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        return HfVocabWriters.canLlamaHf(in) ? HfVocabWriters.llamaHf(in, pre) : HfVocabWriters.gpt2(in, pre);
    }

    // ------------------------------------------------------------------ D1Model (lfm2.py:89)

    /**
     * {@code D1Model.set_gguf_parameters} (lfm2.py:128-130): the LFM2 parameters plus the {@code lfm2-d1}
     * decision type.
     */
    public static void d1Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        lfm2Meta(ctx, kv, plan);
        kv.set(GgufKeys.DECISION_TYPE, "lfm2-d1");
    }

    /** {@code D1Model.set_vocab} (lfm2.py:91-93): the LFM2 vocabulary plus the systemone chat template. */
    public static Map<String, Object> d1Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(lfm2Vocab(ctx, pre));
        md.putAll(HfChatTemplates.toGguf(List.of(Map.of("name", "systemone", "template", D1_SYSTEMONE))));
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ LFM2ColBertModel (lfm2.py:135)

    /**
     * {@code LFM2ColBertModel.set_gguf_parameters} (lfm2.py:140-145): the LFM2 parameters, non-causal attention on
     * the bidirectional checkpoints, then the sentence-transformers pooling.
     */
    public static void colBertMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        lfm2Meta(ctx, kv, plan);
        String arch = ctx.config().architecture();
        if ("Lfm2BidirectionalModel".equals(arch) || "Lfm2BidirectionalForMaskedLM".equals(arch)) {
            kv.set(GgufKeys.ATTENTION_CAUSAL, false);
        }
        HfQwenHooks.pooling(ctx, kv);
    }

    /**
     * {@code LFM2ColBertModel.modify_tensors} (lfm2.py:147-157): the masked-LM {@code lfm2.} prefix strips and
     * every tensor outside the dense head takes {@code model.}, then the LFM2 path applies.
     */
    public static void colBertTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        String name = HfHookSupport.stripPrefixes(hfName, "lfm2.");
        if (!name.startsWith(DENSE_TENSOR_NAME)) {
            name = "model." + name;
        }
        lfm2Tensor(ctx, name, src, out);
    }

    /** {@code lfm2Fixup} and the dense filter are inherited-identical; the vocabulary is the plain one. */
    public static Map<String, Object> colBertVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return lfm2Vocab(ctx, pre);
    }

    /**
     * {@code LFM2ColBertModel.generate_extra_tensors} (lfm2.py:159-184): the optional dense projection of
     * {@code 1_Dense/model.safetensors} as {@code dense_2.weight}, with its outer width as
     * {@code embedding_length_out}; absent when the side file does not exist.
     */
    public static void colBertExtra(HfLoadContext ctx, TensorSink out) throws IOException {
        java.nio.file.Path file = ctx.config().sideFile("1_Dense/model.safetensors");
        if (!java.nio.file.Files.isRegularFile(file)) {
            return;
        }
        SafetensorsFile side = ctx.checkpoint().openSideFile(
                "1_Dense/model.safetensors");
        StTensor tensor = side.tensor("linear.weight");
        if (tensor == null) {
            throw new GgufFormatException(side.path()
                    + ": the dense head file has no 'linear.weight' (lfm2.py:161-164)");
        }
        out.addKv(GgufKeys.EMBEDDING_LENGTH_OUT, com.eonmux.neuralfree.gguf.GGUFValueType.UINT32,
                tensor.shape()[0]);
        out.add(ctx.names().mapOrThrow("dense_2_out"),
                com.eonmux.neuralfree.gguf.hf.st.StRowSource.of(side, tensor));
    }

    // ------------------------------------------------------------------ D1OmniModel (lfm2.py:188)

    /**
     * {@code _load_d1_omni_hparams} (lfm2.py:171-179) folded: the head layers append to the trunk with a plain
     * {@code 4 d_model} MLP, the retired auto-adjust flag and the per-layer key/value vector patched, all into the
     * flat configuration the generic metadata reads.
     */
    public static HfConfig d1OmniFixup(HfConfig cfg) {
        if (cfg.flat().get("intermediate_size") instanceof List) {
            return cfg; // the patch already ran; the reference loads the hparams once (lfm2.py:174-183)
        }
        long nLayer = cfg.requireInt("num_hidden_layers");
        long nLayerHead = cfg.requireInt("head_layers");
        OptionalDouble multiplier = cfg.findFloat("block_ffn_dim_multiplier");
        if (!multiplier.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": required key 'block_ffn_dim_multiplier' is missing"
                    + " (lfm2.py:174)");
        }
        long ffn = (long) (multiplier.getAsDouble() * (2L * cfg.requireInt("intermediate_size") / 3));
        long multipleOf = cfg.requireInt("block_multiple_of");
        ffn = multipleOf * ((ffn + multipleOf - 1) / multipleOf);
        long hiddenSize = cfg.requireInt("hidden_size");
        List<Number> ffnArray = new ArrayList<>();
        for (long i = 0; i < nLayer; i++) {
            ffnArray.add(ffn);
        }
        for (long i = 0; i < nLayerHead; i++) {
            ffnArray.add(4L * hiddenSize);
        }
        Map<String, Object> patch = new LinkedHashMap<>();
        patch.put("num_hidden_layers", nLayer + nLayerHead);
        patch.put("intermediate_size", ffnArray);
        patch.put("block_auto_adjust_ff_dim", Boolean.FALSE);
        Object types = cfg.flat().get("layer_types");
        Object heads = cfg.flat().get("num_key_value_heads");
        if (types instanceof List && heads instanceof Number) {
            List<Number> kv = perLayerKv((List<?>) types, "conv", heads);
            long nHead = cfg.requireInt("num_attention_heads");
            for (long i = 0; i < nLayerHead; i++) {
                kv.add(nHead);
            }
            patch.put("num_key_value_heads", kv);
        }
        return cfg.withFlat(patch);
    }

    /**
     * {@code D1OmniModel.set_gguf_parameters} (lfm2.py:219-243): the fixed text lengths of the server's cuts, the
     * per-layer key/value vector that also covers the head blocks, the generic metadata (no LFM2 auto-adjust, the
     * patch set {@intermediate_size}), then the head constants: the head RMS epsilon of 1e-5, the bidirectional
     * attention, the {@code lfm2-d1-omni} decision head count and its per-option temperatures.
     */
    public static void d1OmniMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        if (cfg.requireInt("max_length") != D1_MAX_LENGTH || cfg.requireInt("image_text_length")
                != D1_IMAGE_TEXT_LENGTH || cfg.requireInt("audio_text_length") != D1_AUDIO_TEXT_LENGTH) {
            throw new GgufFormatException(cfg.source() + ": unexpected d1-omni text lengths (lfm2.py:221-223)");
        }
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.VOCAB_SIZE, (long) cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.SHORTCONV_L_CACHE, (long) cfg.requireInt("conv_L_cache"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, 1e-5);
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);
        kv.set(GgufKeys.DECISION_TYPE, "lfm2-d1-omni");
        long nLayerHead = cfg.requireInt("head_layers");
        kv.set(GgufKeys.DECISION_BLOCK_COUNT, nLayerHead);
        Object temperatures = cfg.flat().get("temperatures");
        if (temperatures instanceof Map) {
            for (Map.Entry<?, ?> e : ((Map<?, ?>) temperatures).entrySet()) {
                String name = String.valueOf(e.getKey()).replace(":", ".").replace("-", "_");
                while (name.endsWith("+")) {
                    name = name.substring(0, name.length() - 1);
                }
                if (!(e.getValue() instanceof Number)) {
                    throw new GgufFormatException(cfg.source() + ": '" + e.getKey() + "' temperature must be a"
                            + " number, got " + e.getValue());
                }
                kv.setAt(GgufKeys.DECISION_TEMPERATURE, ((Number) e.getValue()).doubleValue(), name);
            }
        }
    }

    /**
     * {@code D1OmniModel.modify_tensors} (lfm2.py:263-270): the head blocks fold into the block sequence after the
     * trunk, spanning the total layer count of the fixed-up configuration, then the LFM2 path applies.
     */
    public static void d1OmniTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        String name = hfName;
        int bid = HfLoadContext.bidOf(name);
        long nLayerHead = ctx.config().requireInt("head_layers");
        if (name.startsWith("head.layers.") && bid >= 0) {
            String[] parts = name.split("\\.", 4);
            bid += ctx.blockCount() - (int) nLayerHead;
            name = "head.layers." + bid + "." + parts[3];
        }
        lfm2Tensor(ctx, name, src, out);
    }

    /**
     * {@code D1OmniModel.set_vocab} (lfm2.py:196-203): the LFM2 vocabulary without a BOS token and with the three
     * decision token types, plus the omni systemone template.
     */
    public static Map<String, Object> d1OmniVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(lfm2Vocab(ctx, pre));
        md.remove(GgufKeys.TOKENIZER_ADD_BOS);
        md.put(GgufKeys.TOKENIZER_ADD_BOS, false);
        md.put(GgufKeys.TOKENIZER_TOKEN_TYPE_COUNT, 3L);
        md.putAll(HfChatTemplates.toGguf(List.of(Map.of("name", "systemone", "template", D1_OMNI_SYSTEMONE))));
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ LFM2MoeModel (lfm2.py:279)

    /**
     * {@code LFM2MoeModel.set_gguf_parameters}' key/value patch (lfm2.py:290-293): the per-layer key/value vector
     * patches, the scalar on the {@code full_attention} layers and 0 elsewhere.
     */
    public static HfConfig lfm2MoeFixup(HfConfig cfg) {
        Object types = cfg.flat().get("layer_types");
        if (!(types instanceof List)) {
            return cfg;
        }
        Object heads = cfg.flat().get("num_key_value_heads");
        if (!(heads instanceof Number)) {
            return cfg;
        }
        return cfg.withFlat(Map.of("num_key_value_heads", perLayerKeepKv((List<?>) types, "full_attention", heads)));
    }

    /** The per-layer vector: the scalar on the {@code keepType} layers, 0 elsewhere (lfm2.py:290-293). */
    private static List<Number> perLayerKeepKv(List<?> types, String keepType, Object heads) {
        long value = ((Number) heads).longValue();
        List<Number> out = new ArrayList<>(types.size());
        for (Object t : types) {
            out.add(keepType.equals(t) ? value : 0L);
        }
        return out;
    }

    /**
     * {@code LFM2MoeModel.set_gguf_parameters} (lfm2.py:288-303): the generic metadata over the fixed-up vector,
     * then the expert feed-forward width, the leading dense blocks, the sigmoid gating, and the vocabulary size
     * and short-conv cache of the LFM2 trunk. The RMS epsilon is the generic one.
     */
    public static void lfm2MoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, (long) cfg.requireInt("moe_intermediate_size"));
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, (long) cfg.requireInt("num_dense_layers"));
        kv.set(GgufKeys.EXPERT_GATING_FUNC, 2);
        kv.set(GgufKeys.VOCAB_SIZE, (long) cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.SHORTCONV_L_CACHE, (long) cfg.requireInt("conv_L_cache"));
    }

    /** {@code LFM2MoeModel.modify_tensors} (lfm2.py:327-345): the LFM2 squeeze, the experts stacked as
     * {@code w1}/{@code w2}/{@code w3}, and the plain mapping otherwise. */
    public static void lfm2MoeTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("conv.conv")) {
            lfm2Tensor(ctx, hfName, src, out);
            return;
        }
        if (hfName.contains("experts")) {
            HfConfig cfg = ctx.config();
            lfm2MoeExperts(ctx, cfg.requireInt("num_local_experts", "num_experts")).offer(ctx, hfName, out);
            return;
        }
        lfm2Tensor(ctx, hfName, src, out);
    }

    /** The per-load LFM2 MoE expert collector (lfm2.py:339-359); the merged name drops the {@code model.} prefix
     * (lfm2.py:352 {@code layers.<b>.feed_forward.experts.<part>.weight}). */
    private static HfHookSupport.ExpertCollector lfm2MoeExperts(HfLoadContext ctx, int nExperts) {
        return ctx.state(HfSsmHybridHooksB.class.getName() + ".experts", HfHookSupport.ExpertCollector.class,
                () -> new HfHookSupport.ExpertCollector(nExperts, Arrays.asList("w1", "w2", "w3"),
                        (prefix, part, tail) -> prefix.replaceFirst("^model\\.", "") + "." + part + tail));
    }

    /** {@code LFM2MoeModel.prepare_tensors}' unprocessed-experts assertion (lfm2.py:364-366). */
    public static void lfm2MoeExtra(HfLoadContext ctx, TensorSink out) {
        HfHookSupport.ExpertCollector experts = ctx.state(HfSsmHybridHooksB.class.getName() + ".experts",
                HfHookSupport.ExpertCollector.class,
                () -> new HfHookSupport.ExpertCollector(
                        ctx.config().requireInt("num_local_experts", "num_experts"),
                        Arrays.asList("w1", "w2", "w3"), HfHookSupport.MergedName.SAME_PREFIX));
        experts.requireComplete();
    }

    // ------------------------------------------------------------------ LFM25AudioTokenizer (lfm2.py:495)

    /**
     * {@code LFM25AudioTokenizer.set_gguf_parameters} (lfm2.py:500-504): the LFM2 parameters plus the sliding
     * window and the embedding width the downstream projectors read.
     */
    public static void audioTokenizerMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        lfm2Meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, (long) cfg.requireInt("sliding_window"));
        kv.set(GgufKeys.EMBEDDING_LENGTH_OUT, (long) cfg.requireInt("output_size"));
    }

    /** {@link #lfm2Tensor}: the audio tokenizer has no tensor surgery beyond the LFM2 squeeze. */
    public static void audioTokenizerTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        lfm2Tensor(ctx, hfName, src, out);
    }

    /** {@code LFM25AudioTokenizer.set_vocab} (lfm2.py:497-499): a {@value GgufKeys#TOKENIZER_MODEL}-none vocabulary. */
    public static Map<String, Object> audioTokenizerVocab(HfLoadContext ctx, PreNameResolver pre) {
        return HfVocabWriters.none(HfHookSupport.vocabInput(ctx), pre);
    }

    // ------------------------------------------------------------------ the systemone templates (lfm2.py:97-183)

    /** The Jinja expression {@code jinja_str_or_json(name)} renders: as-is for strings, JSON-encoded otherwise. */
    private static String jinja(String name) {
        return "{{ " + name + " if " + name + " is string else " + name + " | tojson }}";
    }

    /**
     * {@code D1Model._systemone_template} (lfm2.py:99-129): the D1 prompt, whose options render as their labels
     * (choice), as yes/no prefixes (noul when criteria-determined) or as score lists.
     */
    private static String yesNoTemplate() {
        String description = jinja("o.description");
        String choice = "{{ '\\n\\nOptions:\\n' }}"
                + "{% for o in options %}{{ o.label }} {% if o.description %}" + description
                + "{% else %}{{ o.key | replace('_', ' ') }}{% endif %}"
                + "{% if not loop.last %}{{ '\\n' }}{% endif %}{% endfor %}"
                + "{{ '\\n\\nReply with the option code only.' }}";
        String noul = "{% set ns = namespace(criteria=false) %}{% for o in options %}"
                + "{% if o.description is not none %}{% set ns.criteria = true %}{% endif %}{% endfor %}"
                + "{% if ns.criteria %}"
                + "{% for o in options %}{{ '\\nYes: ' if o.key == 'true' else '\\nNo: ' }}"
                + "{% if o.description is none %}None{% else %}" + description + "{% endif %}{% endfor %}{% endif %}"
                + "{{ '\\n\\nReply with yes or no only.' }}";
        String score = "{{ '\\n\\n' }}{% for o in options %}{{ o.key }} " + description + "{{ '\\n' }}{% endfor %}"
                + "{{ '\\nReply with a single digit 0-' }}{{ options | length - 1 }}{{ ' only.' }}";
        return "<|startoftext|><|im_start|>user\n"
                + "{% for image in images %}{{ image }}{% endfor %}"
                + "{% if state is not none %}{% if state is string %}{{ state }}"
                + "{% else %}{{ state | tojson(indent=2) }}{% endif %}"
                + "{{ '\\n\\n\\nQUESTION:\\n' }}{% endif %}"
                + jinja("instructions")
                + "{% if type == 'choice' %}" + choice + "{% elif type == 'noul' %}" + noul
                + "{% else %}" + score + "{% endif %}"
                + "{{ '<|im_end|>\\n<|im_start|>assistant\\n' }}";
    }

    /** The type- and audio-sensitive option line of {@code D1OmniModel._systemone_template} (lfm2.py:210-216). */
    private static String omniOption() {
        String description = jinja("o.description");
        String hasDescription = "o.description is not none and o.description != ''";
        String yesNo = "{{ 'yes' if o.key == 'true' else 'no' }}";
        String optionCode = "{% if loop.index0 < 10 %}00{% elif loop.index0 < 100 %}0{% endif %}{{ loop.index0 }}";
        return "{% if type == 'choice' and audio %}option_" + optionCode + ": "
                + "{% if " + hasDescription + " %}" + description + "{% else %}{{ o.key }}{% endif %}"
                + "{% elif type == 'choice' %}{{ o.key }}{% if " + hasDescription + " %}: " + description
                + "{% endif %}"
                + "{% elif type == 'score' %}level {{ o.key }}: " + description
                + "{% elif audio %}{{ o.key }}: " + yesNo
                + "{% else %}{{ o.key }}: {% if " + hasDescription + " %}" + description
                + "{% elif images and not ns.criteria %}" + yesNo
                + "{% elif o.key == 'true' %}yes, the statement holds"
                + "{% else %}no, the statement does not hold{% endif %}{% endif %}";
    }

    /**
     * {@code D1OmniModel._systemone_template} (lfm2.py:206-241): the omni prompt, whose reserved markers and
     * state/question marks lay out the BOS-after-media prompt the server cut per piece.
     */
    private static final String D1_OMNI_SYSTEMONE = buildOmniSystemone();

    private static String buildOmniSystemone() {
        String state = "{% if state is string %}{{ state }}{% elif state is not none %}{{ state | tojson }}"
                + "{% elif audio %}{}{% endif %}";
        return "{% set ns = namespace(criteria=false) %}"
                + "{% for o in options %}{% if o.description is not none %}{% set ns.criteria = true %}{% endif %}{% endfor %}"
                + "{% for image in images %}{{ image }}{% endfor %}{{ sep }}"
                + "<|startoftext|><|reserved_7|>{{ sep }}{{ mark_state }}" + state
                + "{{ sep }}{{ mark_question }}<|reserved_8|>" + jinja("instructions")
                + "{% for o in options %}{{ sep }}<|reserved_9|><|mask|>{{ sep }}{{ mark_option }} " + omniOption()
                + "{{ sep }}<|reserved_10|>{% endfor %}{{ sep }}<|reserved_11|>";
    }
}
