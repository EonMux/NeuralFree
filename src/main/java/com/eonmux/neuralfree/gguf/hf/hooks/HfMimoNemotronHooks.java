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
import com.eonmux.neuralfree.gguf.hf.st.StRowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The MiMo-V2 vision-and-audio projector file and the Nemotron Omni projector file, ports of llama.cpp
 * {@code MiMoV2VisionAudioModel} (conversion/mimo.py:345-476) and {@code NemotronOmniVLModel}
 * (conversion/nemotron.py:259-340): the MiMo window-attention vision tower with its RVQ and local-attention audio
 * tokenizer, and the RADIO vision tower. The MiMo class declares both encoders; the Nemotron class writes the vision
 * keys only and still converts the Parakeet weights under the audio prefix, as the converter does. Every method is a
 * public static hook reference.
 */
public final class HfMimoNemotronHooks {

    /** The MiMo vision projector ({@code VisionProjectorType.MIMOVL}). */
    public static final String MIMOVL = "mimovl";
    /** The MiMo audio projector ({@code VisionProjectorType.MIMO_AUDIO}). */
    public static final String MIMO_AUDIO = "mimo_audio";
    /** The Nemotron Omni vision projector ({@code VisionProjectorType.NEMOTRON_OMNI}). */
    public static final String NEMOTRON_OMNI = "nemotron_omni";

    /** The RVQ codebook pages of the MiMo audio tokenizer (mimo.py:441). */
    private static final Pattern RVQ_CODEBOOK =
            Pattern.compile("^encoder\\.quantizer\\.vq\\.layers\\.(\\d+)\\._codebook\\.embed$");
    /** The speech-embedding pages of the MiMo checkpoint (mimo.py:404). */
    private static final Pattern SPEECH_EMBEDDING = Pattern.compile("^speech_embeddings\\.(\\d+)\\.weight$");

    private HfMimoNemotronHooks() {
    }

    // ------------------------------------------------------------------ MiMo-V2

    /** {@code MiMoV2VisionAudioModel.filter_tensors} (mimo.py:437-442). */
    public static final HfNameFilter MIMO_FILTER = (name, base) -> {
        if (name.startsWith("visual.") || name.startsWith("speech_embeddings.")
                || name.startsWith("audio_encoder.")) {
            return base.apply(name);
        }
        return null;
    };

    /** {@code MiMoV2VisionAudioModel.set_gguf_parameters} (mimo.py:379-433). */
    public static void mimoMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView vision = cfg.vision().orElseThrow(() -> HfAudioTowerHooks.missingConfig(cfg, "vision_config"));
        int imageSize = HfMmprojHooks.visionIntOpt(cfg, "image_size").orElse(560);
        int heads = firstInt(cfg, vision, "num_heads", "num_attention_heads");
        int blocks = firstInt(cfg, vision, "depth", "num_hidden_layers");
        int hidden = HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS);
        int ffn = HfMmprojHooks.visionInt(cfg, HfMmprojHooks.FFN_KEYS);
        HfMmprojHooks.visionMeta(ctx, kv, imageSize, HfMmprojHooks.visionInt(cfg, "patch_size"), hidden, ffn, heads,
                blocks);
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_TYPE, MIMOVL);
        kv.set(GgufKeys.CLIP_USE_SILU, true);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_HEAD_COUNT_KV,
                optInt(cfg, vision, "num_key_value_heads").orElse(8));
        kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, intOf(cfg, vision, "spatial_merge_size"));
        kv.set("clip.vision.window_size", optInt(cfg, vision, "visual_token_window_size").orElse(-1));
        kv.set(GgufKeys.CLIP_VISION_WA_PATTERN_MODE, intList(cfg, vision, "vit_window_attn_types"));
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                optF64(cfg, vision, "rms_norm_eps").orElse(1e-6));
        JsonView pre = HfMmprojHooks.preprocessor(cfg);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, requiredLong(cfg, pre, "min_pixels"));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, requiredLong(cfg, pre, "max_pixels"));

        JsonView audio = mimoAudioConfig(cfg);
        HfMmprojHooks.audioMetaConfig(cfg, kv, intOf(cfg, audio, "d_model"),
                intOf(cfg, audio, "encoder_ffn_dim"), intOf(cfg, audio, "encoder_attention_heads"),
                intOf(cfg, audio, "encoder_layers"), "audio_tokenizer/config.json");
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, MIMO_AUDIO);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, intOf(cfg, audio, "n_mels"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, optF64(cfg, audio, "layer_norm_eps").orElse(1e-5));
        int[] sizes = rvqCodebookSizes(ctx);
        kv.set(GgufKeys.CLIP_AUDIO_RVQ_NUM_QUANTIZERS, sizes.length);
        kv.set(GgufKeys.CLIP_AUDIO_RVQ_CODEBOOK_SIZE, sizes);
        int layers = intOf(cfg, audio, "encoder_layers");
        int swa = optInt(cfg, audio, "swa_per_block").orElse(1);
        List<Integer> pattern = new ArrayList<>(layers);
        boolean hybrid = Boolean.TRUE.equals(audio.raw("hybrid_attention"));
        for (int i = 0; i < layers; i++) {
            pattern.add(hybrid && swa > 1 && i % swa < swa - 1 ? 0 : -1);
        }
        kv.set(GgufKeys.CLIP_AUDIO_WA_PATTERN_MODE, pattern);
        kv.set(GgufKeys.CLIP_AUDIO_WINDOW_SIZE, firstListInt(cfg, audio, "encoder_attn_window_size"));
        JsonView audioRoot = HfAudioTowerHooks.object(cfg, cfg.raw().raw("audio_config"), "audio_config");
        kv.set(GgufKeys.CLIP_AUDIO_LOCAL_BLOCK_COUNT, intOf(cfg, audioRoot, "input_local_layers"));
        kv.set(GgufKeys.CLIP_AUDIO_LOCAL_GROUP_SIZE, intOf(cfg, audioRoot, "group_size"));
    }

    /** {@code MiMoV2VisionAudioModel.modify_tensors} (mimo.py:444-476). */
    public static void mimoTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.equals("visual.patch_embed.proj.weight")) {
            HfQwenVlVisionHooks.splitPatchFrames(ctx, hfName, src, out);
            return;
        }
        Matcher m = SPEECH_EMBEDDING.matcher(hfName);
        if (m.matches()) {
            speechEmbedding(ctx, Integer.parseInt(m.group(1)), src, out);
            return;
        }
        if (HfAudioTowerHooks.convBias(hfName)) {
            src = ElementTransforms.reshape(src, src.shape()[0], 1);
        }
        if (hfName.equals("audio_encoder.projection.mlp.0.weight")) {
            out.add(ctx.names().ggufName("A_MMPROJ", 1, ".weight"), src);
            return;
        }
        if (hfName.equals("audio_encoder.projection.mlp.2.weight")) {
            out.add(ctx.names().ggufName("A_MMPROJ", 2, ".weight"), src);
            return;
        }
        out.add(HfAudioTowerHooks.mappedName(ctx, hfName, ".weight", ".bias"), src);
    }

    /** Buffers the per-channel speech embeddings and stacks them once every channel is present (mimo.py:404-413). */
    private static void speechEmbedding(HfLoadContext ctx, int channel, RowSource src, TensorSink out) {
        Map<Integer, RowSource> pages = ctx.state("mimo.speech.embeddings", Map.class, LinkedHashMap::new);
        @SuppressWarnings("unchecked")
        Map<Integer, RowSource> held = (Map<Integer, RowSource>) pages;
        if (held.put(channel, src) != null) {
            throw new GgufFormatException("the speech embedding channel " + channel + " appears twice");
        }
        int channels = intOf(ctx.config(),
                HfAudioTowerHooks.object(ctx.config(), ctx.config().raw().raw("audio_config"), "audio_config"),
                "audio_channels");
        if (held.size() < channels) {
            return;
        }
        List<RowSource> ordered = new ArrayList<>(channels);
        for (int i = 0; i < channels; i++) {
            RowSource page = held.get(i);
            if (page == null) {
                throw new GgufFormatException("the speech embeddings are missing channel " + i);
            }
            ordered.add(page);
        }
        long[] shape = ordered.get(0).shape();
        if (shape.length != 2) {
            throw new GgufFormatException("a speech embedding page must be 2-D");
        }
        out.add(ctx.names().ggufName("A_MM_CODE_EMBD", -1, ".weight"),
                ElementTransforms.reshape(RowTransforms.concatRows(ordered), channels, shape[0], shape[1]));
        held.clear();
    }

    /** {@code MiMoV2VisionAudioModel.generate_extra_tensors} (mimo.py:459-476). */
    public static void mimoExtra(HfLoadContext ctx, TensorSink out) throws IOException {
        SafetensorsFile file = ctx.checkpoint().openSideFile("audio_tokenizer/model.safetensors");
        Map<Integer, RowSource> codebooks = new LinkedHashMap<>();
        for (StTensor t : file.tensors()) {
            String name = t.name();
            if (name.startsWith("decoder.")) {
                continue;
            }
            if (name.endsWith("_codebook.cluster_size") || name.endsWith("_codebook.embed_avg")
                    || name.endsWith("_codebook.inited")) {
                continue;
            }
            Matcher m = RVQ_CODEBOOK.matcher(name);
            if (m.matches()) {
                codebooks.put(Integer.parseInt(m.group(1)), StRowSource.of(file, t));
                continue;
            }
            mimoTensors(ctx, name, StRowSource.of(file, t), out);
        }
        if (codebooks.isEmpty()) {
            throw new GgufFormatException(ctx.config().source() + ": audio_tokenizer/model.safetensors has no RVQ"
                    + " codebooks");
        }
        int n = codebooks.size();
        List<RowSource> ordered = new ArrayList<>(n);
        long maxBins = 0;
        for (int i = 0; i < n; i++) {
            RowSource cb = codebooks.get(i);
            if (cb == null) {
                throw new GgufFormatException("the RVQ codebooks are not numbered 0.." + (n - 1));
            }
            ordered.add(cb);
            maxBins = Math.max(maxBins, cb.shape()[0]);
        }
        long dim = ordered.get(0).shape()[1];
        long total = Math.multiplyExact(Math.multiplyExact((long) n, maxBins), dim);
        if (total > Integer.MAX_VALUE) {
            throw new GgufFormatException("the merged RVQ codebooks hold " + total + " values");
        }
        float[] values = new float[(int) total];
        for (int i = 0; i < n; i++) {
            RowSource cb = ordered.get(i);
            if (cb.shape()[1] != dim) {
                throw new GgufFormatException("the RVQ codebooks differ in width");
            }
            float[] page = HfGenHooks.floats(cb, cb.describe());
            long bins = cb.shape()[0];
            System.arraycopy(page, 0, values, (int) (i * maxBins * dim), (int) (bins * dim));
        }
        out.add(ctx.names().ggufName("A_ENC_RVQ_CODEBOOK", -1, ""),
                RopeTensors.fromFloats(new long[]{n, maxBins, dim}, values));
    }

    /** {@code MiMoV2VisionAudioModel.tensor_force_quant} (mimo.py:425-435). */
    public static final HfForceType MIMO_FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.equals("a.rvq.codebook") || ggufName.equals("mm.a.code_embd.weight")) {
            return GGMLType.F32;
        }
        if ((ggufName.contains("encoder.conv") || ggufName.contains("encoder.down_sample_layer"))
                && ggufName.endsWith(".weight")) {
            return GGMLType.F32;
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };

    // ------------------------------------------------------------------ Nemotron Omni

    /** The fixed warmup image size of the Omni tower (nemotron.py:271). */
    private static final int OMNI_IMAGE_SIZE = 512;

    /**
     * {@code NemotronOmniVLModel.filter_tensors} (nemotron.py:288-310) around {@code NemotronNanoV2VLModel}'s
     * (nemotron.py:199-230): the Omni-local prefixes pass through untouched, the video embedder and the final layer
     * norm take the inherited projector filter, and the RADIO names pass the Nano prefix rule.
     */
    public static final HfNameFilter NEMOTRON_FILTER = (name, base) -> {
        if (name.equals("vision_model.summary_idxs")) {
            return null;
        }
        if (name.startsWith("vision_model.embeddings.") || name.startsWith("vision_model.encoder.layer.")
                || name.startsWith("vision_projector.mlp1.")) {
            return name;
        }
        if (name.contains("radio_model.model.patch_generator.video_embedder")
                || name.startsWith("vision_projector.vision_final_layernorm.")) {
            return base.apply(name);
        }
        if (name.contains("input_conditioner") || name.contains("radio_model.model.patch_generator.video_embedder")) {
            return null;
        }
        if (!name.startsWith("vision_model.radio_model.model.") && !name.startsWith("mlp1.")
                && !name.startsWith("sound_encoder.") && !name.startsWith("sound_projection.")) {
            return null;
        }
        String filtered = name.contains("patch_generator.pos_embed") && !name.endsWith(".weight")
                ? name + ".weight" : name;
        if (filtered.contains("conv.norm") && filtered.contains("num_batches")) {
            return null;
        }
        return base.apply(filtered);
    };

    /**
     * {@code NemotronOmniVLModel.set_gguf_parameters} (nemotron.py:270-286): the RADIO tower's geometry and colors
     * from the vision configuration, the {@code nemotron_omni} projector type, GELU, the downsample ratio and the
     * patch-area pixel bounds. The class leaves the audio tower out of the metadata, as the converter does; its
     * Parakeet weights still convert under the {@code a.} prefix.
     */
    public static void nemotronMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        if (!cfg.vision().isPresent()) {
            throw HfAudioTowerHooks.missingConfig(cfg, "vision_config");
        }
        int patchSize = HfMmprojHooks.visionInt(cfg, "patch_size");
        int hidden = HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS);
        int ffn = Math.multiplyExact(hidden, HfMmprojHooks.visionInt(cfg, "mlp_ratio"));
        int blocks = HfMmprojHooks.visionBlocks(cfg);
        int heads = HfMmprojHooks.visionInt(cfg, "num_attention_heads", "num_heads", "heads",
                "vt_num_attention_heads");
        JsonView pre = HfMmprojHooks.preprocessor(cfg);
        HfMmprojHooks.visionMeta(ctx, kv, OMNI_IMAGE_SIZE, patchSize, hidden, ffn, heads, blocks,
                requiredColor(cfg, pre, "norm_mean"), requiredColor(cfg, pre, "norm_std"));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, NEMOTRON_OMNI);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 1e-6);
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        double downsample = cfg.raw().has("downsample_ratio") ? cfg.raw().f64("downsample_ratio") : 0.5;
        if (!(downsample > 0.0)) {
            throw new GgufFormatException(cfg.source() + ": 'downsample_ratio' is " + downsample);
        }
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, (int) (1.0 / downsample));
        long patchArea = (long) patchSize * patchSize;
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS,
                Math.multiplyExact(requiredLong(cfg, pre, "min_num_patches"), patchArea));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS,
                Math.multiplyExact(requiredLong(cfg, pre, "max_num_patches"), patchArea));
    }

    /** {@code NemotronOmniVLModel.modify_tensors} (nemotron.py:312-340). */
    public static void nemotronTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("vision_model.embeddings.")) {
            if (hfName.equals("vision_model.embeddings.video_patch_projection.weight")) {
                src = ElementTransforms.reshape(src, src.shape()[0], 6, patchSize(ctx), patchSize(ctx));
                out.add(ctx.names().ggufName("V_ENC_EMBD_PATCH", -1, ".weight.1"), src);
                return;
            }
            if (hfName.equals("vision_model.embeddings.patch_projection.weight")) {
                src = ElementTransforms.reshape(src, src.shape()[0], 3, patchSize(ctx), patchSize(ctx));
            } else if (hfName.equals("vision_model.embeddings.position_embedding")) {
                src = ElementTransforms.squeezeConv(src);
                hfName += ".weight";
            } else if (hfName.contains(".layer_scale") && hfName.endsWith(".lambda1")) {
                hfName = hfName.substring(0, hfName.length() - "lambda1".length()) + "weight";
            }
            out.add(HfAudioTowerHooks.mappedName(ctx, hfName, ".weight", ".bias"), src);
            return;
        }
        if (hfName.startsWith("vision_projector.mlp1.")) {
            out.add(HfAudioTowerHooks.mappedName(ctx, projectorName(ctx, hfName), ".weight", ".bias"), src);
            return;
        }
        if (hfName.startsWith("vision_projector.vision_final_layernorm.")) {
            out.add(HfAudioTowerHooks.mappedName(ctx, hfName, ".weight", ".bias"), src);
            return;
        }
        if (hfName.contains("patch_generator.pos_embed")) {
            out.add(HfAudioTowerHooks.mappedName(ctx, hfName, ".weight"), src);
            return;
        }
        if (hfName.contains("patch_generator.video_embedder")) {
            int temporal = intOfRoot(ctx.config(), "video_temporal_patch_size", -1);
            if (temporal != 2) {
                throw new GgufFormatException(ctx.config().source() + ": unsupported video_temporal_patch_size "
                        + temporal + "; the converter reshapes the video embedder for two temporal slices"
                        + " (nemotron.py:329-335)");
            }
            out.add(ctx.names().ggufName("V_ENC_EMBD_PATCH", -1, ".weight.1"),
                    ElementTransforms.reshape(src, src.shape()[0], temporal * 3, patchSize(ctx),
                            patchSize(ctx)));
            return;
        }
        if (hfName.contains("patch_generator.embedder")) {
            out.add(HfAudioTowerHooks.mappedName(ctx, hfName, ".weight"),
                    ElementTransforms.reshape(src, src.shape()[0], 3, patchSize(ctx), patchSize(ctx)));
            return;
        }
        if (hfName.startsWith("sound_encoder.") || hfName.startsWith("sound_projection.")) {
            src = soundShape(ctx, hfName, src);
            out.add(HfAudioTowerHooks.mappedName(ctx, hfName, ".weight", ".bias"), src);
            return;
        }
        out.drop(hfName, "the Nemotron Omni projector keeps the RADIO vision tower, its projector and the"
                + " Parakeet speech tower");
    }

    /** The Parakeet convolution reshapes of {@code NemotronNanoV2VLModel.modify_tensors} (nemotron.py:246-256). */
    private static RowSource soundShape(HfLoadContext ctx, String hfName, RowSource src) {
        long[] shape = src.shape();
        if (hfName.contains("depthwise_conv") && hfName.endsWith(".weight")) {
            if (shape.length != 3) {
                throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has the"
                        + " shape " + Arrays.toString(shape) + "; a depthwise convolution is [c, 1, k]");
            }
            return ElementTransforms.reshape(src, 1, 1, shape[0], shape[2]);
        }
        if (hfName.contains("pointwise_conv") && hfName.endsWith(".weight")) {
            if (shape.length == 3 && shape[2] == 1) {
                return ElementTransforms.reshape(src, shape[0], shape[1]);
            }
            return src;
        }
        if ((hfName.contains("subsampling.layers") || hfName.contains("pointwise_conv"))
                && hfName.endsWith(".bias") && shape.length == 1) {
            return ElementTransforms.reshape(src, 1, shape[0], 1, 1);
        }
        return src;
    }

    /** The nested projector names mapped onto the {@code mlp1} names the name table knows (nemotron.py:323-329). */
    private static String projectorName(HfLoadContext ctx, String name) {
        switch (name) {
            case "vision_projector.mlp1.norm.weight":
                return "mlp1.0.weight";
            case "vision_projector.mlp1.linear1.weight":
                return "mlp1.1.weight";
            case "vision_projector.mlp1.linear2.weight":
                return "mlp1.3.weight";
            default:
                return name;
        }
    }

    /** {@code NemotronOmniVLModel.tensor_force_quant} (nemotron.py:292-296) and the parent's sound rule. */
    public static final HfForceType NEMOTRON_FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.equals("v.position_embd.weight") || ggufName.equals("v.class_embd")) {
            return GGMLType.F32;
        }
        if (ggufName.startsWith("a.") || ggufName.startsWith("mm.a.")) {
            if (ggufName.contains("bias") || ggufName.contains("norm")
                    || (ggufName.contains("conv") && ggufName.endsWith(".weight"))) {
                return GGMLType.F32;
            }
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };

    // ------------------------------------------------------------------ helpers

    private static JsonView mimoAudioConfig(HfConfig cfg) throws IOException {
        JsonView root = HfAudioTowerHooks.object(cfg, cfg.raw().raw("audio_config"), "audio_config");
        JsonView file = HfBertVariantHooks.readJsonView(cfg.sideFile("audio_tokenizer/config.json"));
        Map<String, Object> merged = new LinkedHashMap<>();
        for (String key : file.keys()) {
            merged.put(key, file.raw(key));
        }
        merged.put("hidden_size", required(cfg, file, "d_model"));
        merged.put("intermediate_size", required(cfg, file, "encoder_ffn_dim"));
        merged.put("num_attention_heads", required(cfg, file, "encoder_attention_heads"));
        merged.put("n_layers", required(cfg, file, "encoder_layers"));
        return JsonView.of(merged, "audio_tokenizer/config.json");
    }

    /** The RVQ codebook sizes of the side tokenizer (mimo.py:459-476). */
    private static int[] rvqCodebookSizes(HfLoadContext ctx) throws IOException {
        SafetensorsFile file = ctx.checkpoint().openSideFile("audio_tokenizer/model.safetensors");
        Map<Integer, Integer> sizes = new LinkedHashMap<>();
        for (StTensor t : file.tensors()) {
            Matcher m = RVQ_CODEBOOK.matcher(t.name());
            if (m.matches()) {
                sizes.put(Integer.parseInt(m.group(1)), (int) t.shape()[0]);
            }
        }
        if (sizes.isEmpty()) {
            throw new GgufFormatException(ctx.config().source() + ": audio_tokenizer/model.safetensors has no RVQ"
                    + " codebooks");
        }
        int[] out = new int[sizes.size()];
        for (int i = 0; i < out.length; i++) {
            Integer size = sizes.get(i);
            if (size == null) {
                throw new GgufFormatException("the RVQ codebooks are not numbered 0.." + (out.length - 1));
            }
            out[i] = size;
        }
        return out;
    }

    private static int patchSize(HfLoadContext ctx) {
        int patch = HfMmprojHooks.visionInt(ctx.config(), "patch_size");
        if (patch <= 0) {
            throw new GgufFormatException(ctx.config().source() + ": the vision configuration requires a positive"
                    + " 'patch_size'");
        }
        return patch;
    }

    /** The normalization color {@code key} of the preprocessor configuration, refused by name when absent. */
    private static float[] requiredColor(HfConfig cfg, JsonView pre, String key) {
        if (!(pre.raw(key) instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": the preprocessor configuration has no '" + key
                    + "'; the Nemotron Omni tower writes it as clip.vision."
                    + ("norm_mean".equals(key) ? "image_mean" : "image_std"));
        }
        return colorOrDefault(pre, key, null);
    }

    private static float[] colorOrDefault(JsonView pre, String key, float[] dflt) {
        Object v = pre.raw(key);
        if (!(v instanceof List)) {
            return dflt.clone();
        }
        List<?> list = (List<?>) v;
        float[] out = new float[list.size()];
        for (int i = 0; i < out.length; i++) {
            if (!(list.get(i) instanceof Number)) {
                throw new GgufFormatException(pre.where(key) + ": entry " + i + " is not a number");
            }
            out[i] = ((Number) list.get(i)).floatValue();
        }
        return out;
    }

    private static int intOfRoot(HfConfig cfg, String key, int dflt) {
        Object v = cfg.flat().get(key);
        return v instanceof Number ? ((Number) v).intValue() : dflt;
    }

    private static int firstInt(HfConfig cfg, JsonView view, String... keys) {
        for (String key : keys) {
            if (view.has(key)) {
                return intOf(cfg, view, key);
            }
        }
        throw new GgufFormatException(view.where(keys[0]) + ": the checkpoint requires one of "
                + Arrays.toString(keys));
    }

    private static java.util.OptionalInt optInt(HfConfig cfg, JsonView view, String key) {
        return view.has(key) ? java.util.OptionalInt.of(intOf(cfg, view, key)) : java.util.OptionalInt.empty();
    }

    private static java.util.OptionalDouble optF64(HfConfig cfg, JsonView view, String key) {
        Object v = view.raw(key);
        return v instanceof Number ? java.util.OptionalDouble.of(((Number) v).doubleValue())
                : java.util.OptionalDouble.empty();
    }

    private static int intOf(HfConfig cfg, JsonView view, String key) {
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

    private static long requiredLong(HfConfig cfg, JsonView view, String key) {
        Object v = view.raw(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(view.where(key) + ": the checkpoint requires this integer");
        }
        return ((Number) v).longValue();
    }

    private static List<Integer> intList(HfConfig cfg, JsonView view, String key) {
        Object v = view.raw(key);
        if (!(v instanceof List)) {
            throw new GgufFormatException(view.where(key) + ": the checkpoint requires an array of integers");
        }
        List<Integer> out = new ArrayList<>(((List<?>) v).size());
        for (Object e : (List<?>) v) {
            if (!(e instanceof Number)) {
                throw new GgufFormatException(view.where(key) + ": holds " + e + ", not an integer");
            }
            out.add(((Number) e).intValue());
        }
        return out;
    }

    private static int firstListInt(HfConfig cfg, JsonView view, String key) {
        List<Integer> list = intList(cfg, view, key);
        if (list.isEmpty()) {
            throw new GgufFormatException(view.where(key) + ": the array is empty");
        }
        return list.get(0);
    }

    private static Object required(HfConfig cfg, JsonView view, String key) {
        if (!view.has(key)) {
            throw new GgufFormatException(view.where(key) + ": the checkpoint requires this key");
        }
        return view.raw(key);
    }
}
