package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The single audio towers and their projectors through the Hugging Face loader, ports of llama.cpp's
 * {@code GlmASRWhisperEncoderModel} (conversion/ultravox.py:24-86), {@code VoxtralRealtimeEncoderModel}
 * (ultravox.py:216-237), {@code GraniteSpeechMmprojModel} / {@code GraniteSpeechPlusMmprojModel}
 * (conversion/granite.py:579-690), {@code LFM2AudioModel} (conversion/lfm2.py:465-509) and the audio half of
 * {@code D1OmniMmprojModel} (lfm2.py:413-462): the audio configuration aliases, the projector keys, the conformer
 * batch-norm folding, the conv-stem reshapes, the fused K/V split and the GLM-ASR adapter and boundary rows. Every
 * method is a public static hook reference.
 */
public final class HfAudioTowerHooks {

    /** The projector of GLM-ASR ({@code VisionProjectorType.GLMA}). */
    public static final String GLMA = "glma";
    /** The realtime Voxtral projector ({@code VisionProjectorType.VOXTRAL_RT}). */
    public static final String VOXTRAL_RT = "voxtral_rt";
    /** The Granite Speech projector ({@code VisionProjectorType.GRANITE_SPEECH}). */
    public static final String GRANITE_SPEECH = "granite_speech";
    /** The LFM2 audio projector ({@code VisionProjectorType.LFM2A}). */
    public static final String LFM2A = "lfm2a";
    /** The d1-omni audio projector ({@code VisionProjectorType.D1OMNI_A}). */
    public static final String D1OMNI_A = "d1omni_a";

    /** The default layer-norm epsilon of the Whisper and conformer towers. */
    private static final double DEFAULT_EPS = 1e-5;
    /** The batch-norm folding epsilon of the conformer and Granite Speech towers. */
    private static final float BATCH_NORM_EPS = 1e-5f;

    private HfAudioTowerHooks() {
    }

    // ------------------------------------------------------------------ GLM-ASR

    /**
     * {@code GlmASRWhisperEncoderModel.filter_tensors} (ultravox.py:67-86): the language model drops, the Whisper
     * tower renames to {@code audio_tower.}, the adapter's layer norm and projection move under
     * {@code audio.multi_modal_projector.} with their sub-module renames.
     */
    public static final HfNameFilter GLMA_FILTER = (name, base) -> {
        if (name.startsWith("model.") || name.startsWith("lm_head.")) {
            return null;
        }
        if (name.startsWith("audio_encoder.whisper.")) {
            name = "audio_tower." + name.substring("audio_encoder.whisper.".length());
        }
        if (name.contains("audio_encoder.layer_norm.") || name.contains("audio_encoder.proj.")) {
            name = name.replace("audio_encoder.", "audio_encoder.adapting.");
        }
        if (name.startsWith("audio_encoder.adapting.")) {
            name = "audio.multi_modal_projector." + name.substring("audio_encoder.adapting.".length());
            name = name.replace(".layer_norm.", ".ln_pre.").replace(".0.", ".linear_1.")
                    .replace(".2.", ".linear_2.");
        }
        return base.apply(name);
    };

    /**
     * {@code GlmASRWhisperEncoderModel.set_gguf_parameters} (ultravox.py:39-48): the base audio keys through the
     * {@code d_model} aliases, the {@code glma} projector, the mel-bin count, the tower epsilon and the merge factor.
     */
    public static void glmaMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.audioMeta(ctx, kv, HfMmprojHooks.audioInt(cfg, "hidden_size", "d_model"),
                HfMmprojHooks.audioInt(cfg, "intermediate_size", "encoder_ffn_dim"),
                HfMmprojHooks.audioInt(cfg, "num_attention_heads", "encoder_attention_heads"),
                HfMmprojHooks.audioBlocks(cfg));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, GLMA);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, HfMmprojHooks.audioInt(cfg, "num_mel_bins"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, audioEps(cfg, DEFAULT_EPS));
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_STACK_FACTOR, requiredRootInt(cfg, "merge_factor"));
    }

    /** The conv-stem weights keep F16, the patch kernels the base rule (ultravox.py:50-54). */
    public static final HfForceType GLMA_FORCE = convF16();

    /**
     * {@code GlmASRWhisperEncoderModel.modify_tensors} (ultravox.py:56-85): the two audio boundary rows become the
     * vision BOS/EOS rows, the adapter's projection layer drops, the conv-stem biases widen to a one-wide row, and
     * everything else takes its mapped name.
     */
    public static void glmaTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("audio_encoder.audio_bos_eos_token.")) {
            long[] shape = src.shape();
            if (shape.length < 1 || shape[0] < 2) {
                throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has the"
                        + " shape " + java.util.Arrays.toString(shape) + "; the boundary table holds two rows");
            }
            long[] rowShape = java.util.Arrays.copyOfRange(shape, 1, shape.length);
            out.add(ctx.names().ggufName("V_TOK_BOI", -1, ""),
                    ElementTransforms.reshape(RowTransforms.sliceRows(src, 0, 1), rowShape));
            out.add(ctx.names().ggufName("V_TOK_EOI", -1, ""),
                    ElementTransforms.reshape(RowTransforms.sliceRows(src, 1, 1), rowShape));
            return;
        }
        if (hfName.startsWith("audio_encoder.adapting.") && hfName.contains(".proj.")) {
            out.drop(hfName, "the adapter projection is recomputed from the audio tower output");
            return;
        }
        if (convBias(hfName)) {
            src = ElementTransforms.reshape(src, src.shape()[0], 1);
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    // ------------------------------------------------------------------ Voxtral realtime

    /**
     * {@code VoxtralRealtimeEncoderModel.filter_tensors} (ultravox.py:228-237): the projector names carry the
     * {@code audio.} prefix of {@link HfWhisperAudioHooks#WHISPER_FILTER}.
     */
    public static final HfNameFilter VOXTRAL_RT_FILTER = HfWhisperAudioHooks.WHISPER_FILTER;

    /**
     * {@code VoxtralRealtimeEncoderModel.set_gguf_parameters} (ultravox.py:221-227): the base audio keys, the
     * {@code voxtral_rt} projector, the mel bins, the head width, the epsilon, the sliding window and the downsample
     * factor.
     */
    public static void voxtralRtMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.audioMeta(ctx, kv);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, VOXTRAL_RT);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, HfMmprojHooks.audioInt(cfg, "num_mel_bins"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_HEAD_DIM, HfMmprojHooks.audioInt(cfg, "head_dim"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, rmsEps(cfg, DEFAULT_EPS));
        kv.set(GgufKeys.CLIP_AUDIO_WINDOW_SIZE, HfMmprojHooks.audioInt(cfg, "sliding_window"));
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_STACK_FACTOR, requiredRootInt(cfg, "downsample_factor"));
    }

    /** The {@code rms_norm_eps} of the audio configuration, or {@code dflt} (ultravox.py:225). */
    private static double rmsEps(HfConfig cfg, double dflt) {
        Object v = HfMmprojHooks.audioFind(cfg, "rms_norm_eps").orElse(null);
        return v instanceof Number ? ((Number) v).doubleValue() : dflt;
    }

    /** The conv-stem weights keep F16, the patch kernels the base rule (ultravox.py:229-232). */
    public static final HfForceType VOXTRAL_RT_FORCE = convF16();

    /**
     * {@code VoxtralRealtimeEncoderModel.modify_tensors} (ultravox.py:234-252): the conv-stem biases widen, the Q/K
     * rows are un-permuted from the rotate-half layout, everything else takes its mapped name.
     */
    public static void voxtralRtTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (convBias(hfName)) {
            src = ElementTransforms.reshape(src, src.shape()[0], 1);
        }
        if (hfName.contains(".self_attn.q_proj.") || hfName.contains(".self_attn.k_proj.")) {
            int heads = HfMmprojHooks.audioInt(ctx.config(), "num_attention_heads");
            src = RowTransforms.permuteQk(src, heads, heads);
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    // ------------------------------------------------------------------ Granite Speech

    /**
     * {@code GraniteSpeechMmprojModel.filter_tensors} (granite.py:622-629): the attention-distance buffers and the
     * batch-norm counters drop, the rest go through the inherited projector filter.
     */
    public static final HfNameFilter GRANITE_SPEECH_FILTER = (name, base) -> {
        if (name.contains("attention_dists") || name.contains("num_batches_tracked")) {
            return null;
        }
        return base.apply(name);
    };

    /**
     * {@code GraniteSpeechMmprojModel.set_gguf_parameters} (granite.py:590-613): the base audio keys over the
     * {@code encoder_config} aliases, the {@code granite_speech} projector, the mel bins, the fixed epsilon, the
     * convolution geometry and the projector window, downsample rate and head count.
     */
    public static void graniteSpeechMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfConfig cfg = ctx.config();
        JsonView audio = graniteSpeechConfig(cfg);
        int hidden = intOf(cfg, audio, "hidden_dim");
        int heads = intOf(cfg, audio, "num_heads");
        int blocks = intOf(cfg, audio, "num_layers");
        HfMmprojHooks.audioMeta(ctx, kv, hidden, Math.multiplyExact(hidden, intOf(cfg, audio, "feedforward_mult")),
                heads, blocks);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, GRANITE_SPEECH);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, intOf(cfg, audio, "input_dim"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, DEFAULT_EPS);
        kv.set(GgufKeys.CLIP_AUDIO_CHUNK_SIZE, intOf(cfg, audio, "context_size"));
        kv.set(GgufKeys.CLIP_AUDIO_CONV_KERNEL_SIZE, intOf(cfg, audio, "conv_kernel_size"));
        kv.set(GgufKeys.CLIP_AUDIO_MAX_POS_EMB, intOf(cfg, audio, "max_pos_emb"));
        JsonView root = cfg.raw();
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_WINDOW_SIZE, intOf(cfg, root, "window_size"));
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_DOWNSAMPLE_RATE, intOf(cfg, root, "downsample_rate"));
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_HEAD_COUNT,
                intOf(cfg, object(cfg, root.raw("projector_config"), "projector_config"), "num_attention_heads"));
        graniteFeatureLayers(ctx, kv, audio, hidden);
    }

    /**
     * {@code GraniteSpeechPlusMmprojModel.set_gguf_parameters} (granite.py:663-690): the concatenated feature layers
     * and the projector width they require.
     */
    private static void graniteFeatureLayers(HfLoadContext ctx, HfMetadataBuilder kv, JsonView audio, int hidden) {
        HfConfig cfg = ctx.config();
        Object raw = audio.raw("cat_hidden_layers");
        if (raw == null) {
            return;
        }
        if (!(raw instanceof List) || ((List<?>) raw).isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": 'encoder_config.cat_hidden_layers' must be a non-empty"
                    + " array");
        }
        List<Integer> layers = new ArrayList<>(((List<?>) raw).size());
        for (Object e : (List<?>) raw) {
            if (!(e instanceof Number) || ((Number) e).doubleValue() != Math.rint(((Number) e).doubleValue())) {
                throw new GgufFormatException(cfg.source() + ": 'encoder_config.cat_hidden_layers' holds "
                        + e + ", not an integer layer index");
            }
            layers.add(((Number) e).intValue());
        }
        int expected = Math.multiplyExact(hidden, layers.size() + 1);
        JsonView projector = object(cfg, cfg.raw().raw("projector_config"), "projector_config");
        int width = intOf(cfg, projector, "encoder_hidden_size");
        if (width != expected) {
            throw new GgufFormatException(cfg.source() + ": the projector encoder_hidden_size " + width
                    + " does not match the concatenated dimension " + expected + " (hidden_dim " + hidden + " * ("
                    + layers.size() + " + 1)) (granite.py:683-690)");
        }
        kv.set(GgufKeys.CLIP_AUDIO_FEATURE_LAYER, layers);
    }

    /** The Granite Speech block count the tensor plan uses: {@code encoder_config.num_layers}. */
    public static int graniteSpeechBlocks(HfConfig cfg) {
        return HfMmprojHooks.blocksOf(cfg, graniteSpeechConfig(cfg), "encoder_config");
    }

    /** The convolution kernels keep F32 (granite.py:615-620). */
    public static final HfForceType GRANITE_SPEECH_FORCE = (ggufName, nDims, source, outType) -> {
        if ((ggufName.contains("encoder") || ggufName.contains("projector")) && ggufName.contains(".conv")
                && ggufName.endsWith(".weight")) {
            return GGMLType.F32;
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };

    /**
     * {@code GraniteSpeechMmprojModel.modify_tensors} (granite.py:631-661): the convolution batch-norm statistics fold
     * into the kernel, the fused {@code to_kv} splits into K and V, the up/down convolutions lose their one-wide
     * kernel axis and the depthwise convolutions their one-wide channel axis.
     */
    public static void graniteSpeechTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (graniteBatchNorm(ctx, hfName, src, out)) {
            return;
        }
        if (hfName.contains(".attn.to_kv.weight")) {
            RowSource[] kv = RowTransforms.splitRows(src, src.shape()[0] / 2, src.shape()[0] - src.shape()[0] / 2);
            out.add(ctx.names().mapOrThrow(hfName.replace("to_kv", "to_k")), kv[0]);
            out.add(ctx.names().mapOrThrow(hfName.replace("to_kv", "to_v")), kv[1]);
            return;
        }
        if ((hfName.contains("up_conv") || hfName.contains("down_conv")) && hfName.endsWith(".weight")) {
            long[] shape = src.shape();
            if (shape.length == 3 && shape[2] == 1) {
                src = ElementTransforms.reshape(src, shape[0], shape[1]);
            }
        }
        if (hfName.contains("depth_conv") && hfName.endsWith(".weight")) {
            long[] shape = src.shape();
            if (shape.length == 3 && shape[1] == 1) {
                src = ElementTransforms.reshape(src, shape[0], shape[2]);
            }
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /**
     * Folds the four {@code encoder.layers.<bid>.conv.batch_norm} statistics into the kernel once the layer's values
     * are all buffered (granite.py:633-651); false when the name is not one of them.
     */
    private static boolean graniteBatchNorm(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (!hfName.contains("batch_norm") || !hfName.contains("encoder.layers.")) {
            return false;
        }
        String marker = "encoder.layers.";
        int at = hfName.indexOf(marker);
        String rest = hfName.substring(at + marker.length());
        int dot = rest.indexOf('.');
        int bid = parseIndex(rest.substring(0, dot), hfName, ctx.config().source());
        String prefix = "encoder.layers." + bid + ".conv.batch_norm";
        BatchNorm bn = new BatchNorm(4);
        if (!bn.add(ctx, "granite.speech.bn." + bid, prefix, hfName, src, out)) {
            return true;
        }
        foldBatchNorm(ctx, bn, prefix + ".weight", prefix + ".bias", out);
        return true;
    }

    // ------------------------------------------------------------------ LFM2 audio

    /**
     * {@code LFM2AudioModel.filter_tensors} (lfm2.py:485-500): the text model, the training loss and the audio
     * output heads drop; the rest go through the inherited projector filter.
     */
    public static final HfNameFilter LFM2A_FILTER = (name, base) -> {
        if (name.startsWith("lfm.") || name.contains("audio_loss_weight")
                || name.contains("codebook_offsets") || name.contains("depth_embeddings")
                || name.contains("depth_linear") || name.contains("depthformer")) {
            return null;
        }
        return base.apply(name);
    };

    /**
     * {@code LFM2AudioModel.set_gguf_parameters} (lfm2.py:470-482): the base audio keys over the {@code encoder}
     * aliases with a feed-forward width of {@code d_model}, the {@code lfm2a} projector, the mel bins and the fixed
     * epsilon.
     */
    public static void lfm2aMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView audio = lfm2Encoder(cfg);
        int hidden = intOf(cfg, audio, "d_model");
        HfMmprojHooks.audioMeta(ctx, kv, hidden, hidden, intOf(cfg, audio, "n_heads"),
                HfMmprojHooks.blocksOf(cfg, audio, "encoder"));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, LFM2A);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, intOf(cfg, audio, "feat_in"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, DEFAULT_EPS);
    }

    /** The LFM2 audio block count the tensor plan uses: the layers of the root {@code encoder}. */
    public static int lfm2aBlocks(HfConfig cfg) {
        return HfMmprojHooks.blocksOf(cfg, lfm2Encoder(cfg), "encoder");
    }

    /** The audio tensors: the conformer folds and reshapes, then the mapped name (lfm2.py inherits them). */
    public static void lfm2aTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        conformerTensors(ctx, hfName, src, out);
    }

    /**
     * The conformer conv kernels keep F32 (gemma.py:315-318): the converter tests the checkpoint name, whose conformer
     * convolutions all land under the {@code a.} prefix of the projector file.
     */
    public static final HfForceType CONFORMER_FORCE = (ggufName, nDims, source, outType) ->
            ggufName.startsWith("a.") && ggufName.contains("conv")
                    ? GGMLType.F32 : HfMmprojHooks.patchKernelForce(ggufName, outType);

    // ------------------------------------------------------------------ d1-omni audio

    /**
     * {@code D1OmniMmprojModel.filter_tensors} (lfm2.py:440-458): the text model and heads drop, the vision and audio
     * prefixes take the names the shared name map knows, and the audio adapter's residual block continues the
     * projector's module numbering.
     */
    public static final HfNameFilter D1OMNI_FILTER = (name, base) -> {
        if (name.startsWith("encoder.") || name.startsWith("head.")) {
            return null;
        }
        name = name.replace("vision.tower.", "vision_tower.").replace("vision.projector.",
                "multi_modal_projector.");
        name = name.replace("audio.encoder.", "conformer.");
        String[][] adapter = {{"adapter.norm", "0"}, {"adapter.linear_1", "1"}, {"adapter.linear_2", "3"},
                {"residual.ln", "4"}, {"residual.down", "5"}, {"residual.up", "6"}};
        for (String[] m : adapter) {
            name = name.replace("audio." + m[0] + ".", "audio_adapter.model." + m[1] + ".");
        }
        return base.apply(name);
    };

    /**
     * The d1-omni audio keys ({@code D1OmniMmprojModel.set_gguf_parameters}, lfm2.py:433-438): the base audio keys
     * over the audio config with the expansion-factor feed-forward width, the {@code d1omni_a} projector, the mel bins
     * and the fixed epsilon.
     */
    public static void d1OmniAudioMeta(HfLoadContext ctx, HfMetadataBuilder kv) {
        HfConfig cfg = ctx.config();
        JsonView audio = audioConfig(cfg);
        int hidden = intOf(cfg, audio, "d_model");
        HfMmprojHooks.audioMeta(ctx, kv, hidden,
                Math.multiplyExact(hidden, intOf(cfg, audio, "ff_expansion_factor")),
                intOf(cfg, audio, "n_heads"), HfMmprojHooks.blocksOf(cfg, audio, "audio_config"));
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, D1OMNI_A);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, intOf(cfg, audio, "feat_in"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, DEFAULT_EPS);
    }

    // ------------------------------------------------------------------ conformer tensors

    /**
     * {@code ConformerAudioModel.modify_tensors} (gemma.py:318-357): the convolution batch-norm statistics fold into
     * the kernel, the pre-encode conv bias widens to {@code [c, 1, 1]}, the depthwise and pointwise kernels lose their
     * unit axes, and every other name takes its mapped form.
     */
    public static void conformerTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        conformerTensors(ctx, hfName, src, out, ".weight", ".bias");
    }

    /**
     * {@link #conformerTensors(HfLoadContext, String, RowSource, TensorSink)} with the converter's extra mapped
     * suffixes (Gemma's scaled-dot-product bounds, {@code .input_max} and its siblings).
     */
    public static void conformerTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out,
                                        String... suffixes) throws IOException {
        if (conformerBatchNorm(ctx, hfName, src, out)) {
            return;
        }
        if (hfName.startsWith("conformer.pre_encode.conv.") && hfName.endsWith(".bias")) {
            src = ElementTransforms.reshape(src, src.shape()[0], 1, 1);
        }
        if (hfName.contains("conv.depthwise_conv") && hfName.endsWith(".weight")) {
            long[] shape = src.shape();
            if (shape.length != 3 || shape[1] != 1) {
                throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has the"
                        + " shape " + java.util.Arrays.toString(shape) + "; a depthwise convolution is [c, 1, k]");
            }
            src = ElementTransforms.reshape(src, shape[0], shape[2]);
        }
        if (hfName.contains("conv.pointwise_conv") && hfName.endsWith(".weight")) {
            long[] shape = src.shape();
            if (shape.length != 3 || shape[2] != 1) {
                throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has the"
                        + " shape " + java.util.Arrays.toString(shape) + "; a pointwise convolution is [c, k, 1]");
            }
            src = ElementTransforms.reshape(src, shape[0], shape[1]);
        }
        addMapped(ctx, hfName, src, out, suffixes);
    }

    /** Adds {@code hfName} under its mapped GGUF name, trying {@code suffixes} (base.py map_tensor_name). */
    static void addMapped(HfLoadContext ctx, String hfName, RowSource src, TensorSink out, String... suffixes) {
        out.add(mappedName(ctx, hfName, suffixes), src);
    }

    /** The mapped GGUF name of {@code hfName}, trying {@code suffixes} (base.py map_tensor_name). */
    static String mappedName(HfLoadContext ctx, String hfName, String... suffixes) {
        java.util.Optional<com.eonmux.neuralfree.gguf.hf.names.HfTensorNameMap.Mapped> mapped =
                ctx.names().map(hfName, suffixes);
        if (!mapped.isPresent()) {
            throw new GgufFormatException("Can not map tensor '" + hfName + "'");
        }
        return mapped.get().ggufName();
    }

    /**
     * The linear patch projection {@code [n_embd, patch * patch * 3]} repacked as the Conv2d kernel
     * {@code [n_embd, 3, patch, patch]} (gemma.py:1047-1053, lfm2.py:404-405).
     */
    static RowSource patchProjection(HfLoadContext ctx, String hfName, RowSource src) {
        long[] shape = src.shape();
        if (shape.length != 2 || shape[1] % 3 != 0) {
            throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has the shape "
                    + java.util.Arrays.toString(shape) + "; the patch projection is [n_embd, patch * patch * 3]");
        }
        int perChannel = (int) (shape[1] / 3);
        int patch = (int) Math.sqrt(perChannel);
        if ((long) patch * patch != perChannel) {
            throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has " + perChannel
                    + " values per channel, not a square patch");
        }
        int[] index = new int[(int) shape[1]];
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < patch; h++) {
                for (int w = 0; w < patch; w++) {
                    index[c * patch * patch + h * patch + w] = (h * patch + w) * 3 + c;
                }
            }
        }
        return ElementTransforms.reshape(ElementTransforms.gatherInner(src, index), shape[0], 3, patch, patch);
    }

    /**
     * Buffers the five {@code conformer.layers.<bid>.conv.batch_norm} statistics and folds them once the layer's set
     * is complete (gemma.py:320-344); false when the name is not one of them.
     */
    private static boolean conformerBatchNorm(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (!hfName.contains("batch_norm")) {
            return false;
        }
        int at = hfName.indexOf("conformer.layers.");
        if (at < 0) {
            return false;
        }
        String rest = hfName.substring(at + "conformer.layers.".length());
        int dot = rest.indexOf('.');
        if (dot <= 0) {
            return false;
        }
        int bid = parseIndex(rest.substring(0, dot), hfName, ctx.config().source());
        String prefix = "conformer.layers." + bid + ".conv.batch_norm";
        BatchNorm bn = new BatchNorm(5);
        if (!bn.add(ctx, "conformer.bn." + bid, prefix, hfName, src, out)) {
            return true;
        }
        foldBatchNorm(ctx, bn, prefix + ".weight", prefix + ".bias", out);
        return true;
    }

    // ------------------------------------------------------------------ shared helpers

    /** A per-channel convolution batch-norm's statistics, buffered until the layer's set is complete. */
    private static final class BatchNorm {
        private final Map<String, RowSource> sources = new LinkedHashMap<>();
        private final int expected;

        BatchNorm(int expected) {
            this.expected = expected;
        }

        /** Buffers the part of {@code prefix} that {@code hfName} names; true once all parts are present. */
        boolean add(HfLoadContext ctx, String stateKey, String prefix, String hfName, RowSource src, TensorSink out) {
            Object state = ctx.state(stateKey, Map.class, LinkedHashMap::new);
            @SuppressWarnings("unchecked")
            Map<String, RowSource> held = (Map<String, RowSource>) state;
            if (held.put(hfName, src) != null) {
                throw new GgufFormatException("the batch-norm tensor '" + hfName + "' appears twice");
            }
            sources.putAll(held);
            return sources.size() >= expected;
        }
    }

    /**
     * {@code weight / sqrt(running_var + eps)} and {@code bias - running_mean * a} (gemma.py:333-338): the folded
     * kernel and bias under the batch-norm's own names.
     */
    private static void foldBatchNorm(HfLoadContext ctx, BatchNorm bn, String weightName, String biasName,
                                      TensorSink out) throws IOException {
        RowSource weight = bn.sources.get(weightName);
        RowSource bias = bn.sources.get(biasName);
        RowSource mean = bn.sources.get(weightName.replace(".weight", ".running_mean"));
        RowSource var = bn.sources.get(weightName.replace(".weight", ".running_var"));
        if (weight == null || bias == null || mean == null || var == null) {
            throw new GgufFormatException("the batch-norm of '" + weightName + "' is incomplete: weight, bias,"
                    + " running_mean and running_var are required");
        }
        float[] w = HfGenHooks.floats(weight, weightName);
        float[] b = HfGenHooks.floats(bias, biasName);
        float[] m = HfGenHooks.floats(mean, mean.describe());
        float[] v = HfGenHooks.floats(var, var.describe());
        if (w.length != b.length || w.length != m.length || w.length != v.length) {
            throw new GgufFormatException("the batch-norm of '" + weightName + "' mixes " + w.length + ", "
                    + b.length + ", " + m.length + " and " + v.length + " values");
        }
        float[] a = new float[w.length];
        float[] c = new float[w.length];
        for (int i = 0; i < w.length; i++) {
            a[i] = w[i] / (float) Math.sqrt(v[i] + BATCH_NORM_EPS);
            c[i] = b[i] - m[i] * a[i];
        }
        out.add(ctx.names().mapOrThrow(weightName), RopeTensors.fromFloats(weight.shape(), a));
        out.add(ctx.names().mapOrThrow(biasName), RopeTensors.fromFloats(bias.shape(), c));
    }

    /** True for the Whisper conv-stem biases, which widen to a one-wide row. */
    static boolean convBias(String hfName) {
        return hfName.contains("conv1.bias") || hfName.contains("conv2.bias");
    }

    /** The conv-stem weights keep F16 (ultravox.py:50-54, :122-125). */
    static HfForceType convF16() {
        return (ggufName, nDims, source, outType) -> ggufName.contains(".conv") && ggufName.endsWith(".weight")
                ? GGMLType.F16 : HfMmprojHooks.patchKernelForce(ggufName, outType);
    }

    /** The audio config of a {@code whisper_config} / {@code audio_config} checkpoint. */
    static JsonView audioConfig(HfConfig cfg) {
        return HfMmprojHooks.audioConfig(cfg).orElseThrow(() -> missingConfig(cfg, "audio_config"));
    }

    /** The {@code encoder} object of the LFM2 audio checkpoint. */
    static JsonView lfm2Encoder(HfConfig cfg) {
        JsonView root = cfg.raw();
        if (!root.has("encoder")) {
            throw missingConfig(cfg, "encoder");
        }
        return object(cfg, root.raw("encoder"), "encoder");
    }

    /** The {@code encoder_config} object of a Granite Speech checkpoint. */
    static JsonView graniteSpeechConfig(HfConfig cfg) {
        JsonView root = cfg.raw();
        if (!root.has("encoder_config")) {
            throw missingConfig(cfg, "encoder_config");
        }
        return object(cfg, root.raw("encoder_config"), "encoder_config");
    }

    /** The layer-norm epsilon of the audio config, or {@code dflt}. */
    static double audioEps(HfConfig cfg, double dflt) {
        Object v = HfMmprojHooks.audioFind(cfg, "layer_norm_eps").orElse(null);
        return v instanceof Number ? ((Number) v).doubleValue() : dflt;
    }

    /** The integer member {@code key} of {@code view}, required. */
    static int intOf(HfConfig cfg, JsonView view, String key) {
        Object v = view.raw(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(view.where(key) + ": the checkpoint requires this integer");
        }
        double d = ((Number) v).doubleValue();
        if (d != Math.rint(d) || d < Integer.MIN_VALUE || d > Integer.MAX_VALUE) {
            throw new GgufFormatException(view.where(key) + ": expected an integer, got " + v);
        }
        return (int) d;
    }

    /** The integer {@code key} of the root configuration, required. */
    static int requiredRootInt(HfConfig cfg, String key) {
        JsonView root = cfg.raw();
        if (!root.has(key)) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint requires '" + key + "'");
        }
        return intOf(cfg, root, key);
    }

    /** {@code view} as an object, refused by name otherwise. */
    static JsonView object(HfConfig cfg, Object value, String what) {
        if (!(value instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": '" + what + "' must be an object, got "
                    + (value == null ? "nothing" : value.getClass().getSimpleName()));
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> map = (Map<String, Object>) value;
        return JsonView.of(map, what);
    }

    static GgufFormatException missingConfig(HfConfig cfg, String key) {
        return new GgufFormatException(cfg.source() + ": the checkpoint has no '" + key + "'; this projector"
                + " converts the audio tower");
    }

    private static int parseIndex(String digits, String where, String source) {
        if (digits.isEmpty() || digits.length() > 9) {
            throw new GgufFormatException("tensor '" + where + "' of " + source + " has no usable block index '"
                    + digits + "'");
        }
        for (int i = 0; i < digits.length(); i++) {
            if (digits.charAt(i) < '0' || digits.charAt(i) > '9') {
                throw new GgufFormatException("tensor '" + where + "' of " + source + " has no usable block index '"
                        + digits + "'");
            }
        }
        return Integer.parseInt(digits);
    }
}
