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
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The d1-omni and LFM3-Omni projector files and the LFM3 audio tower through the Hugging Face loader, ports of
 * llama.cpp {@code D1OmniMmprojModel} (conversion/lfm2.py:413-462), {@code LFM3AudioMmprojModel} (conversion/lfm3.py:
 * 309-573) and {@code LFM3OmniMmprojModel} (lfm3.py:575-609): both towers of d1, the LFM3-VL vision mixin combined
 * with the LFM3 audio encoder, the feature-extractor checks, the depthformer and liquid-codec generator halves, and
 * the per-family name renames. Every method is a public static hook reference.
 */
public final class HfLfmOmniHooks {

    /** {@code VisionProjectorType.D1OMNI_V}. */
    public static final String D1OMNI_V = "d1omni_v";
    /** {@code VisionProjectorType.LFM3A}. */
    public static final String LFM3A = "lfm3a";
    /** {@code VisionProjectorType.LFM3A_GEN}. */
    public static final String LFM3A_GEN = "lfm3a_gen";

    /** The dynamic vision image size of d1-omni (lfm2.py:421). */
    private static final int D1_IMAGE_SIZE = 256;
    /** The required LFM3 audio feature extractor (lfm3.py:369-374). */
    private static final Map<String, Object> FEATURE_EXTRACTOR = featureExtractor();
    /** The required LFM3 liquid-codec configuration (lfm3.py:355-360). */
    private static final Map<String, Object> CODEC = codec();
    /** The required LFM3 generation shape (lfm3.py:365). */
    private static final Map<String, Object> GENERATION = generation();
    /** The d1-omni default downsample factor (lfm2.py:432). */
    private static final int D1_DOWNSAMPLE = 2;

    /** The audio defaults of the LFM3-Audio model card (lfm3.py:363). */
    private static final Map<String, Double> AUDIO_SAMPLING = sampling(1.0, 128.0, 1.1, 4096.0);
    /** The audio defaults of LFM3-Omni (lfm3.py:579). */
    private static final Map<String, Double> OMNI_AUDIO_SAMPLING = sampling(0.4, null, 1.2, 64.0);

    private static final Map<String, String> DEPTH_LAYERS = depthLayers();
    private static final Map<String, String> CODEC_LAYERS = codecLayers();
    private static final Map<String, Integer> DOWNSAMPLE = downsampleLayers();

    private HfLfmOmniHooks() {
    }

    // ------------------------------------------------------------------ d1-omni

    /** The dynamic image size and fixed colors d1-omni writes (lfm2.py:421-423). */
    public static void d1OmniMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv, D1_IMAGE_SIZE, HfMmprojHooks.visionInt(cfg, "patch_size"),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.FFN_KEYS),
                HfMmprojHooks.visionInt(cfg, "num_attention_heads"), HfMmprojHooks.visionBlocks(cfg),
                new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f});
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_TYPE, D1OMNI_V);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64(cfg, "layer_norm_eps"));
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, cfg.raw().has("downsample_factor")
                ? cfg.raw().i32("downsample_factor") : D1_DOWNSAMPLE);
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        HfAudioTowerHooks.d1OmniAudioMeta(ctx, kv);
    }

    /** {@code D1OmniMmprojModel.modify_tensors} and the shared conformer transforms (lfm2.py:456-462). */
    public static void d1OmniTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("conformer.") || hfName.startsWith("audio_adapter.")) {
            HfAudioTowerHooks.conformerTensors(ctx, hfName, src, out, ".weight", ".bias", ".input_max", ".input_min",
                    ".output_max", ".output_min");
            return;
        }
        if (hfName.endsWith("patch_embedding.weight")) {
            src = HfAudioTowerHooks.patchProjection(ctx, hfName, src);
        }
        out.add(HfAudioTowerHooks.mappedName(ctx, hfName, ".weight", ".bias"), src);
    }

    // ------------------------------------------------------------------ LFM3 audio

    /** The LFM3 audio-only view of a {@code Lfm3AudioForConditionalGeneration} checkpoint (lfm3.py:311-318). */
    public static final HfNameFilter LFM3A_FILTER = (name, base) -> {
        if (name.startsWith("encoder.") || name.startsWith("mlp.") || name.startsWith("depthformer.")
                || name.startsWith("depth_") || name.startsWith("codebook_embedding.")
                || name.startsWith("mask_embedding") || name.startsWith("a.gen.wav.")) {
            return base.apply(name);
        }
        return null;
    };

    /**
     * The LFM3 audio tower's metadata (lfm3.py:376-433): the feature-extractor checks, the base audio keys with the
     * LFM2 feed-forward width, the {@code lfm3a} projector, the KV head count, epsilon, mel bins and sliding window,
     * and, when the checkpoint ships the codec, the depthformer and sampling keys of the generator half.
     */
    public static void lfm3AudioMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        lfm3AudioMeta(ctx, kv, false);
    }

    /** {@link #lfm3AudioMeta(HfLoadContext, HfMetadataBuilder, HfPlanView)} for the Omni checkpoint. */
    public static void lfm3OmniAudioMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        lfm3AudioMeta(ctx, kv, true);
    }

    /**
     * {@code LFM3OmniMmprojModel.set_gguf_parameters} (lfm3.py:617-620): the audio tower of
     * {@link #lfm3OmniAudioMeta} followed by the {@code LFM3VLVisionMixin} vision keys, the projector type under
     * {@code clip.vision.projector_type}.
     */
    public static void lfm3OmniMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        lfm3OmniAudioMeta(ctx, kv, plan);
        // LFM3OmniMmprojModel._video_max_frames is 384, half the vision-only class's 768 (lfm3.py:581)
        HfLfm3VisionHooks.meta(ctx, kv, plan, GgufKeys.CLIP_VISION_PROJECTOR_TYPE, LFM3_OMNI_VIDEO_MAX_FRAMES);
    }

    /** The video frame bound of the Omni vision half (lfm3.py:581). */
    private static final int LFM3_OMNI_VIDEO_MAX_FRAMES = 384;

    /** The LFM3 audio block count the tensor plan uses: the layers of the merged audio configuration. */
    public static int lfm3AudioBlocks(HfConfig cfg) {
        return HfMmprojHooks.blocksOf(cfg, lfm3AudioConfig(cfg, false), "audio");
    }

    private static void lfm3AudioMeta(HfLoadContext ctx, HfMetadataBuilder kv, boolean omni) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView audio = lfm3AudioConfig(cfg, omni);
        checkFeatureExtractor(cfg);
        double theta = ropeTheta(cfg, audio);
        if (theta != 1e6) {
            throw new GgufFormatException(cfg.source() + ": unsupported audio encoder rope_theta " + theta
                    + "; the clip LFM3 audio encoder uses 1e6 (lfm3.py:386)");
        }
        HfMmprojHooks.audioMetaConfig(cfg, kv, intOf(cfg, audio, "hidden_size"),
                lfm2FeedForwardLength(cfg, audio), intOf(cfg, audio, "num_attention_heads"),
                HfMmprojHooks.blocksOf(cfg, audio, "audio"), "audio");
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, LFM3A);
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_HEAD_COUNT_KV, intOf(cfg, audio, "num_key_value_heads"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, number(cfg, audio, "norm_eps"));
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, intOf(cfg, audio, "num_mel_bins"));
        kv.set(GgufKeys.CLIP_AUDIO_WINDOW_SIZE, intOf(cfg, audio, "sliding_window"));
        if (!audioOutput(cfg)) {
            return;
        }
        Generation gen = generationConfig(cfg, omni);
        for (Map.Entry<String, Object> e : GENERATION.entrySet()) {
            if (!valuesEqual(e.getValue(), gen.value(e.getKey()))) {
                throw new GgufFormatException(cfg.source() + ": unsupported " + e.getKey() + "="
                        + gen.value(e.getKey()) + ", expected " + e.getValue() + " (lfm3.py:398-400)");
            }
        }
        JsonView depth = depthConfig(cfg, omni);
        checkDepth(cfg, depth);
        int audioVocab = gen.audioVocab();
        int codebooks = gen.codebooks();
        kv.set(GgufKeys.CLIP_HAS_GEN_AUDIO_ENCODER, true);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, LFM3A_GEN);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_PROJECTION_DIM, HfMmprojHooks.textWidth(cfg));
        kv.set(GgufKeys.CLIP_GEN_AUDIO_EMBEDDING_LENGTH, intOf(cfg, depth, "hidden_size"));
        kv.set(GgufKeys.CLIP_GEN_AUDIO_FEED_FORWARD_LENGTH, intOf(cfg, depth, "intermediate_size"));
        kv.set(GgufKeys.CLIP_GEN_AUDIO_BLOCK_COUNT, intOf(cfg, depth, "num_hidden_layers"));
        kv.set(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT, intOf(cfg, depth, "num_attention_heads"));
        kv.set(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT_KV, intOf(cfg, depth, "num_key_value_heads"));
        kv.set(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_LAYER_NORM_EPSILON, number(cfg, depth, "norm_eps"));
        writeSampling(cfg, kv, omni ? OMNI_AUDIO_SAMPLING : AUDIO_SAMPLING);
        if (codebooks <= 0 || audioVocab <= 0) {
            throw new GgufFormatException(cfg.source() + ": the generator needs a positive codebook count and"
                    + " vocabulary");
        }
    }

    /** {@code LFM3AudioMmprojModel.modify_tensors} (lfm3.py:520-556). */
    public static void lfm3AudioTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        lfm3AudioTensors(ctx, hfName, src, out, false);
    }

    /** {@code LFM3OmniMmprojModel.modify_tensors} (lfm3.py:601-609). */
    public static void lfm3OmniTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("model.vision_tower.") || hfName.startsWith("model.multi_modal_projector.")) {
            HfLfm3VisionHooks.tensors(ctx, hfName, src, out);
            return;
        }
        String name = omniAudioRename(hfName);
        if (name == null) {
            out.drop(hfName, "the LFM3-Omni projector keeps the vision tower and the audio encoder and generator");
            return;
        }
        lfm3AudioTensors(ctx, name, src, out, true);
    }

    /** The HF-native Omni audio names mapped onto the LFM3-Audio checkpoint layout (lfm3.py:583-591). */
    private static String omniAudioRename(String name) {
        String[][] renames = {{"model.audio_encoder.lfm.", "encoder.lfm.lfm."},
                {"model.audio_encoder.", "encoder."},
                {"model.audio_projector.layer_norm.", "mlp.model.0."},
                {"model.audio_projector.linear_1.", "mlp.model.1."},
                {"model.audio_projector.linear_2.", "mlp.model.3."},
                {"model.depth_decoder.model.", "depthformer.model."},
                {"model.depth_decoder.", ""},
                {"model.codebook_embedding.", "codebook_embedding."}};
        for (String[] r : renames) {
            if (name.startsWith(r[0])) {
                return r[1] + name.substring(r[0].length());
            }
        }
        return name.startsWith("encoder.") || name.startsWith("mlp.") || name.startsWith("depthformer.")
                || name.startsWith("depth_") || name.startsWith("codebook_embedding.")
                || name.startsWith("mask_embedding") ? name : null;
    }

    private static void lfm3AudioTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out,
                                         boolean omni) throws IOException {
        if (hfName.startsWith("a.gen.wav.")) {
            out.add(hfName, src);
            return;
        }
        if (hfName.startsWith("depthformer.") || hfName.startsWith("depth_")
                || hfName.startsWith("codebook_embedding.") || hfName.startsWith("mask_embedding")) {
            if (audioOutput(ctx.config())) {
                depthTensor(ctx, hfName, src, out, omni);
            } else {
                out.drop(hfName, "the checkpoint has no audio_tokenizer; the LFM3 generator half is not exported");
            }
            return;
        }
        if (!hfName.startsWith("encoder.") && !hfName.startsWith("mlp.")) {
            out.drop(hfName, "the LFM3 audio projector keeps the encoder, its projector and the generator");
            return;
        }
        if (hfName.startsWith("encoder.downsample.")) {
            downsample(ctx, hfName, src, out);
            return;
        }
        if (hfName.endsWith("conv.conv.weight")) {
            // the depthwise conv's channel axis is a singleton, as in the reference's squeeze(1)
            src = ElementTransforms.squeezeDim(src, 1);
        }
        out.add(HfAudioTowerHooks.mappedName(ctx, hfName, ".weight", ".bias"), src);
    }

    /** {@code encoder.downsample.<layer>} rewritten to the pre-encoder conv names (lfm3.py:532-545). */
    private static void downsample(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        String rest = hfName.substring("encoder.downsample.".length());
        int dot = rest.indexOf('.');
        if (dot <= 0 || rest.indexOf('.', dot + 1) >= 0) {
            throw new GgufFormatException("tensor '" + hfName + "' does not name one downsample layer and suffix");
        }
        String layer = rest.substring(0, dot);
        String suffix = "." + rest.substring(dot + 1);
        if ("lin_out".equals(layer)) {
            if (suffix.equals(".weight")) {
                // a kernel-1 conv's trailing singleton axis, as in the reference's squeeze(-1)
                src = ElementTransforms.squeezeDim(src, -1);
            }
            out.add(ctx.names().ggufName("A_ENC_OUT", -1, suffix), src);
            return;
        }
        Integer idx = DOWNSAMPLE.get(layer);
        if (idx == null) {
            throw new GgufFormatException("unsupported LFM3 audio downsample layer '" + layer + "' of tensor '"
                    + hfName + "'");
        }
        if (suffix.equals(".bias")) {
            src = ElementTransforms.reshape(src, src.shape()[0], 1, 1);
        }
        out.add(ctx.names().ggufName("A_ENC_CONV1D", idx, suffix), src);
    }

    /** {@code LFM3AudioMmprojModel._depthformer_tensor} (lfm3.py:558-573). */
    private static void depthTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out, boolean omni)
            throws IOException {
        if (hfName.startsWith("depthformer.model.layers.")) {
            String rest = hfName.substring("depthformer.model.layers.".length());
            int dot = rest.indexOf('.');
            if (dot <= 0) {
                throw new GgufFormatException("tensor '" + hfName + "' has no depthformer layer index");
            }
            int bid = parseIndex(rest.substring(0, dot), hfName);
            String key = stripWeight(rest.substring(dot + 1), hfName);
            String tensor = DEPTH_LAYERS.get(key);
            if (tensor == null) {
                throw new GgufFormatException("unexpected depthformer tensor '" + hfName + "' (lfm3.py:562)");
            }
            out.add(ctx.names().ggufName(tensor, bid, ".weight"), src);
            return;
        }
        if (hfName.equals("depthformer.model.embedding_norm.weight")) {
            out.add(ctx.names().ggufName("A_GEN_CODE_OUTPUT_NORM", -1, ".weight"), src);
            return;
        }
        if (hfName.startsWith("depth_linear.")) {
            out.add(ctx.names().ggufName("A_GEN_CODE_PROJ_IN", -1, suffixOf(hfName)), src);
            return;
        }
        if (hfName.equals("mask_embedding")) {
            out.add(ctx.names().ggufName("A_GEN_CODE_MASK_EMBD", -1, ""), src);
            return;
        }
        if (hfName.equals("codebook_embedding.weight")) {
            Generation gen = generationConfig(ctx.config(), omni);
            long[] shape = src.shape();
            long expected = (long) gen.codebooks() * gen.audioVocab();
            if (shape.length != 2 || shape[0] != expected) {
                throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has the"
                        + " shape " + Arrays.toString(shape) + "; expected [" + expected + ", ...]");
            }
            out.add(ctx.names().ggufName("A_GEN_CODE_OUT_EMBD", -1, ".weight"), src);
            return;
        }
        if (hfName.startsWith("depth_embeddings.")) {
            depthEmbedding(ctx, hfName, src, out, omni);
            return;
        }
        throw new GgufFormatException("unexpected LFM3 depthformer tensor '" + hfName + "'");
    }

    /** The tied per-codebook embedding/head pages, stacked once every codebook is buffered. */
    private static void depthEmbedding(HfLoadContext ctx, String hfName, RowSource src, TensorSink out, boolean omni)
            throws IOException {
        String rest = hfName.substring("depth_embeddings.".length());
        int dot = rest.indexOf('.');
        if (dot <= 0) {
            throw new GgufFormatException("tensor '" + hfName + "' has no codebook index");
        }
        int idx = parseIndex(rest.substring(0, dot), hfName);
        String key = rest.substring(dot + 1);
        Generation gen = generationConfig(ctx.config(), omni);
        int codebooks = gen.codebooks();
        if (idx < 0 || idx >= codebooks) {
            throw new GgufFormatException("tensor '" + hfName + "' names codebook " + idx + " outside the "
                    + codebooks + " codebooks");
        }
        DepthPages state = ctx.state("lfm3.depth.pages", DepthPages.class, DepthPages::new);
        if (key.equals("embedding.weight")) {
            state.embed.put(idx, src);
            // the checkpoint may list the tied head before its embedding page; the check waits for both
            checkTied(ctx, state, idx, hfName);
            // the stack is emitted once the last embedding page arrives; the pages stay buffered so a tied
            // to_logits tensor that arrives later can still be compared against them
            if (state.embed.size() == codebooks) {
                if (!state.logits.isEmpty()) {
                    throw new GgufFormatException("the tied LM head of codebook "
                            + state.logits.keySet().iterator().next() + " has no embedding page");
                }
                out.add(ctx.names().ggufName("A_GEN_CODE_EMBD", -1, ".weight"),
                        stack(ctx, state.embed, codebooks, "depth embedding"));
            }
        } else if (key.equals("embedding_norm.weight")) {
            state.norm.put(idx, src);
            if (state.norm.size() == codebooks) {
                out.add(ctx.names().ggufName("A_GEN_CODE_HEAD_NORM", -1, ".weight"),
                        stack(ctx, state.norm, codebooks, "depth head norm"));
            }
        } else if (key.equals("to_logits.weight")) {
            state.logits.put(idx, src);
            checkTied(ctx, state, idx, hfName);
            if (state.embed.size() == codebooks && !state.embed.containsKey(idx)) {
                throw new GgufFormatException("the tied LM head of codebook " + idx + " has no embedding page");
            }
        } else {
            throw new GgufFormatException("unexpected tensor '" + hfName + "'");
        }
    }

    /** Checks a tied LM head against its embedding page once both have arrived, then forgets the head page. */
    private static void checkTied(HfLoadContext ctx, DepthPages state, int idx, String hfName) throws IOException {
        RowSource embed = state.embed.get(idx);
        RowSource head = state.logits.get(idx);
        if (embed == null || head == null) {
            return;
        }
        if (!Arrays.equals(embed.shape(), head.shape())) {
            throw new GgufFormatException("the tied LM head of codebook " + idx + " ('" + hfName + "') has shape "
                    + Arrays.toString(head.shape()) + ", its embedding page has " + Arrays.toString(embed.shape()));
        }
        if (!Arrays.equals(HfHookSupport.floats(embed, "depth_embeddings." + idx + ".embedding.weight"),
                HfHookSupport.floats(head, hfName))) {
            throw new GgufFormatException("depth_embeddings." + idx + " is not tied: '" + hfName
                    + "' differs from its embedding page");
        }
        state.logits.remove(idx);
    }

    /**
     * {@code torch.stack(pages, dim=0)} over the codebook index, preserving each page's rank: the embedding pages
     * are {@code [n_vocab, n_embd]} and the head norm pages one vector per codebook, so the norm table is
     * {@code [n_embd, n_codebooks]} (or {@code [n_embd, 1, n_codebooks]} when the checkpoint stores
     * {@code [1, n_embd]} pages), as the reference writes {@code stack(depth_norm)} (lfm3.py:571-573).
     */
    private static RowSource stack(HfLoadContext ctx, Map<Integer, RowSource> pages, int codebooks, String what) {
        List<RowSource> ordered = new ArrayList<>(codebooks);
        for (int i = 0; i < codebooks; i++) {
            RowSource page = pages.get(i);
            if (page == null) {
                throw new GgufFormatException("the " + what + " pages are incomplete: codebook " + i + " is missing");
            }
            ordered.add(page);
        }
        long[] shape = ordered.get(0).shape();
        for (RowSource page : ordered) {
            if (!Arrays.equals(shape, page.shape())) {
                throw new GgufFormatException("the " + what + " pages differ in shape");
            }
        }
        long[] outShape = new long[shape.length + 1];
        outShape[0] = codebooks;
        System.arraycopy(shape, 0, outShape, 1, shape.length);
        return ElementTransforms.reshape(RowTransforms.concatRows(ordered), outShape);
    }

    /** {@code LFM3AudioMmprojModel.tensor_force_quant} (lfm3.py:490-494). */
    public static final HfForceType LFM3A_FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.startsWith("a.conv1d.") || ggufName.contains("conv_dw")
                || ggufName.startsWith("a.gen.wav.quant.")) {
            return GGMLType.F32;
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };

    /**
     * The Omni class's {@code tensor_force_quant}: its MRO runs {@code LFM3VLVisionMixin} first, so the patch
     * kernels and the depthwise spatial conv take the vision rule and every other tensor the audio one
     * (lfm3.py:216-224, :490-494).
     */
    public static final HfForceType LFM3_OMNI_FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.contains("patch_embd")) {
            return GGMLType.F16;
        }
        if (ggufName.contains("conv2d_spatial")) {
            return GGMLType.F32;
        }
        return LFM3A_FORCE.force(ggufName, nDims, source, outType);
    };

    // ------------------------------------------------------------------ LFM3 generator extras

    /** {@code LFM3AudioMmprojModel._codec_tensors} (lfm3.py:498-518): the liquid-codec generator tensors. */
    public static void lfm3Codec(HfLoadContext ctx, TensorSink out) throws IOException {
        HfConfig cfg = ctx.config();
        if (!audioOutput(cfg)) {
            return;
        }
        SafetensorsFile file = ctx.checkpoint().openSideFile("audio_tokenizer/model.safetensors");
        JsonView codec = sideJson(cfg, "audio_tokenizer/config.json");
        for (Map.Entry<String, Object> e : CODEC.entrySet()) {
            if (!valuesEqual(e.getValue(), codec.raw(e.getKey()))) {
                throw new GgufFormatException(cfg.source() + ": unsupported audio_tokenizer " + e.getKey() + "="
                        + codec.raw(e.getKey()) + ", expected " + e.getValue() + " (lfm3.py:503-506)");
            }
        }
        long fsq = 1;
        Object levelsRaw = codec.raw("fsq_levels");
        List<?> levels = levelsRaw instanceof List ? (List<?>) levelsRaw : Collections.emptyList();
        for (Object level : levels) {
            fsq *= ((Number) level).longValue();
        }
        Generation gen = generationConfig(cfg, omniConfig(cfg));
        if (fsq != gen.audioVocab() - 1) {
            throw new GgufFormatException(cfg.source() + ": audio_vocab_size " + gen.audioVocab()
                    + " does not match the codec, expected codebook size + 1 (" + (fsq + 1) + ") (lfm3.py:508)");
        }
        out.add(ctx.names().ggufName("A_GEN_WAV_QUANT_FIRST_CB", -1, ".weight"), fsqTable(ctx, levels));
        String quantizer = file.tensor("quantizer.fsq.project_out.weight") != null ? "quantizer.fsq." : "quantizer.";
        for (String suffix : new String[]{".weight", ".bias"}) {
            out.add(ctx.names().ggufName("A_GEN_WAV_QUANT_FIRST_OUT", -1, suffix),
                    source(file, quantizer + "project_out." + suffix.substring(1)));
            out.add(ctx.names().ggufName("A_GEN_WAV_TFM_IN_PROJ", -1, suffix),
                    source(file, "from_quantizer." + suffix.substring(1)));
            out.add(ctx.names().ggufName("A_GEN_WAV_TFM_OUT_PROJ", -1, suffix),
                    source(file, "decoder.lin." + suffix.substring(1)));
        }
        out.add(ctx.names().ggufName("A_GEN_WAV_TFM_OUTPUT_NORM", -1, ".weight"),
                source(file, "decoder.lfm.embedding_norm.weight"));
        for (StTensor t : file.tensors()) {
            String name = t.name();
            if (!name.startsWith("decoder.lfm.layers.")) {
                continue;
            }
            String rest = name.substring("decoder.lfm.layers.".length());
            int dot = rest.indexOf('.');
            if (dot <= 0 || rest.indexOf('.', dot + 1) < 0) {
                throw new GgufFormatException("codec tensor '" + name + "' has no block index and parameter name");
            }
            int bid = parseIndex(rest.substring(0, dot), name);
            String key = stripWeight(rest.substring(dot + 1), name);
            String tensor = CODEC_LAYERS.get(key);
            if (tensor == null) {
                throw new GgufFormatException("unexpected LFM3 codec tensor '" + name + "'");
            }
            RowSource src = StRowSource.of(file, t);
            if ("conv.conv".equals(key)) {
                src = ElementTransforms.squeezeDim(src, 1);
            }
            out.add(ctx.names().ggufName(tensor, bid, ".weight"), src);
        }
    }

    /** The FSQ code table: index to the quantized codes of every level (lfm3.py:509-517). */
    private static RowSource fsqTable(HfLoadContext ctx, List<?> levels) {
        int dim = levels.size();
        long size = 1;
        long[] ls = new long[dim];
        for (int i = 0; i < dim; i++) {
            ls[i] = ((Number) levels.get(i)).longValue();
            size *= ls[i];
        }
        if (size > Integer.MAX_VALUE) {
            throw new GgufFormatException(ctx.config().source() + ": the FSQ table has " + size + " entries");
        }
        float[] codes = new float[(int) size * dim];
        for (int idx = 0; idx < size; idx++) {
            long basis = 1;
            for (int d = 0; d < dim; d++) {
                long level = ls[d];
                long half = level / 2;
                long digit = (idx / basis) % level;
                codes[idx * dim + d] = (float) ((double) (digit - half) / (double) half);
                basis *= level;
            }
        }
        return RopeTensors.fromFloats(new long[]{size, dim}, codes);
    }

    // ------------------------------------------------------------------ configuration helpers

    /** The audio configuration object of an LFM3 checkpoint (lfm3.py:378-383, :594-601). */
    private static JsonView lfm3AudioConfig(HfConfig cfg, boolean omni) {
        JsonView root = cfg.raw();
        if (omni) {
            JsonView enc = HfAudioTowerHooks.object(cfg, root.raw("audio_encoder_config"), "audio_encoder_config");
            JsonView text = HfAudioTowerHooks.object(cfg, enc.raw("text_config"), "audio_encoder_config.text_config");
            Map<String, Object> merged = new LinkedHashMap<>();
            for (String key : text.keys()) {
                merged.put(key, text.raw(key));
            }
            merged.put("num_mel_bins", required(cfg, enc, "num_mel_bins"));
            merged.put("sliding_window", required(cfg, enc, "sliding_window"));
            return JsonView.of(merged, "audio_encoder_config");
        }
        JsonView enc = HfAudioTowerHooks.object(cfg, root.raw("encoder"), "encoder");
        JsonView lfm = HfAudioTowerHooks.object(cfg, enc.raw("lfm"), "encoder.lfm");
        Map<String, Object> merged = new LinkedHashMap<>();
        for (String key : lfm.keys()) {
            merged.put(key, lfm.raw(key));
        }
        merged.put("num_mel_bins", required(cfg, enc, "dim_in"));
        merged.put("sliding_window", required(cfg, enc, "sliding_window"));
        return JsonView.of(merged, "encoder");
    }

    /** The required LFM3 audio feature extractor values (lfm3.py:376-384). */
    private static void checkFeatureExtractor(HfConfig cfg) throws IOException {
        JsonView pre = HfMmprojHooks.preprocessor(cfg);
        JsonView fe = HfAudioTowerHooks.object(cfg, pre.raw("feature_extractor"), "preprocessor feature_extractor");
        Object type = fe.raw("feature_extractor_type");
        if (!"Lfm3AudioFeatureExtractor".equals(type) && !"Lfm3OmniFeatureExtractor".equals(type)) {
            throw new GgufFormatException(cfg.source() + ": unsupported feature_extractor_type " + type
                    + " (lfm3.py:378)");
        }
        for (Map.Entry<String, Object> e : FEATURE_EXTRACTOR.entrySet()) {
            Object actual = fe.raw(e.getKey());
            if (!valuesEqual(e.getValue(), actual)) {
                throw new GgufFormatException(cfg.source() + ": unsupported feature_extractor." + e.getKey() + "="
                        + actual + ", expected " + e.getValue() + " (lfm3.py:379-382)");
            }
        }
    }

    /** True when the checkpoint ships the audio codec directory (lfm3.py:388). */
    private static boolean audioOutput(HfConfig cfg) {
        return Files.isDirectory(cfg.dir().resolve("audio_tokenizer"));
    }

    /** True when the checkpoint layout is the LFM3-Omni one. */
    private static boolean omniConfig(HfConfig cfg) {
        return cfg.raw().has("audio_encoder_config");
    }

    /** The depthformer configuration of the generator half (lfm3.py:592-601). */
    private static JsonView depthConfig(HfConfig cfg, boolean omni) {
        JsonView root = cfg.raw();
        String key = omni ? "depth_config" : "depthformer";
        return HfAudioTowerHooks.object(cfg, root.raw(key), key);
    }

    /** The depthformer validation of the converter (lfm3.py:401-407). */
    private static void checkDepth(HfConfig cfg, JsonView depth) {
        Object causal = depth.raw("is_causal");
        Object sandwich = required(cfg, depth, "use_sandwich_norm");
        Object types = required(cfg, depth, "layer_types");
        if (Boolean.TRUE.equals(causal) || Boolean.TRUE.equals(sandwich)) {
            throw new GgufFormatException(cfg.source() + ": unsupported depthformer, expected a bidirectional"
                    + " all-attention stack (lfm3.py:402)");
        }
        if (!(types instanceof List) || ((List<?>) types).isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": unsupported depthformer, expected a bidirectional"
                    + " all-attention stack (lfm3.py:402)");
        }
        for (Object t : (List<?>) types) {
            if (!"full_attention".equals(t)) {
                throw new GgufFormatException(cfg.source() + ": unsupported depthformer, expected a bidirectional"
                        + " all-attention stack (lfm3.py:402)");
            }
        }
        JsonView rope = HfAudioTowerHooks.object(cfg, depth.raw("rope_parameters"),
                "depthformer.rope_parameters");
        JsonView full = HfAudioTowerHooks.object(cfg, rope.raw("full_attention"),
                "depthformer.rope_parameters.full_attention");
        if (number(cfg, full, "rope_theta") != 1e6) {
            throw new GgufFormatException(cfg.source() + ": unsupported depthformer rope_theta (lfm3.py:406)");
        }
    }

    /** The generation shape of the generator half (lfm3.py:389-396, :596-601). */
    private static Generation generationConfig(HfConfig cfg, boolean omni) {
        JsonView root = cfg.raw();
        if (omni) {
            return Generation.of(root.has("num_codebooks") ? intOf(cfg, root, "num_codebooks") : -1,
                    root.has("codebook_size") ? intOf(cfg, root, "codebook_size") : -1, null, root);
        }
        return Generation.of(intOf(cfg, root, "codebooks"), intOf(cfg, root, "audio_vocab_size"),
                intOf(cfg, root, "maskgit_steps"), root);
    }

    /** The audio defaults: {@code generation_config.json} over the model card's values. */
    private static void writeSampling(HfConfig cfg, HfMetadataBuilder kv, Map<String, Double> defaults) {
        JsonView gen = cfg.generation().orElse(null);
        Double temperature = samplingValue(gen, "audio_temperature", defaults.get("audio_temperature"));
        Double topK = samplingValue(gen, "audio_top_k", defaults.get("audio_top_k"));
        Double penalty = samplingValue(gen, "audio_repetition_penalty",
                defaults.get("audio_repetition_penalty"));
        Double window = samplingValue(gen, "audio_repetition_window", defaults.get("audio_repetition_window"));
        if (temperature != null) {
            kv.set(GgufKeys.CLIP_GEN_AUDIO_SAMPLING_TEMP, temperature.floatValue());
        }
        if (topK != null) {
            kv.set(GgufKeys.CLIP_GEN_AUDIO_SAMPLING_TOP_K, topK.intValue());
        }
        if (penalty != null) {
            kv.set(GgufKeys.CLIP_GEN_AUDIO_SAMPLING_PENALTY_REPEAT, penalty.floatValue());
        }
        if (window != null) {
            kv.set(GgufKeys.CLIP_GEN_AUDIO_SAMPLING_PENALTY_LAST_N, window.intValue());
        }
    }

    /**
     * The sampling value of {@code key}: the {@code generation_config.json} value when the key is present (null when
     * it is not a number, so a present null suppresses the default as the reference's {@code dict.get} does), else
     * {@code dflt}.
     */
    private static Double samplingValue(JsonView gen, String key, Double dflt) {
        if (gen != null && gen.has(key)) {
            Object v = gen.raw(key);
            return v instanceof Number ? ((Number) v).doubleValue() : null;
        }
        return dflt;
    }

    /** The LFM2 feed-forward width of an audio configuration (lfm2.py:17-32). */
    private static int lfm2FeedForwardLength(HfConfig cfg, JsonView view) {
        Object ff = view.has("block_ff_dim") ? view.raw("block_ff_dim") : view.raw("intermediate_size");
        if (!(ff instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": the audio config has no block_ff_dim or"
                    + " intermediate_size");
        }
        long dim = ((Number) ff).longValue();
        Object auto = required(cfg, view, "block_auto_adjust_ff_dim");
        Object multiplier = required(cfg, view, "block_ffn_dim_multiplier");
        Object multiple = required(cfg, view, "block_multiple_of");
        if (Boolean.TRUE.equals(auto)) {
            dim = (long) ((double) 2.0 * dim / 3.0);
            if (multiplier instanceof Number) {
                dim = (long) (((Number) multiplier).doubleValue() * dim);
            }
            if (!(multiple instanceof Number)) {
                throw new GgufFormatException(cfg.source() + ": the audio config has no block_multiple_of");
            }
            long m = ((Number) multiple).longValue();
            dim = m * ((dim + m - 1) / m);
        }
        if (dim <= 0 || dim > Integer.MAX_VALUE) {
            throw new GgufFormatException(cfg.source() + ": the audio feed-forward width " + dim + " is invalid");
        }
        return (int) dim;
    }

    private static double ropeTheta(HfConfig cfg, JsonView audio) {
        JsonView rope = HfAudioTowerHooks.object(cfg, audio.raw("rope_parameters"), "audio rope_parameters");
        return number(cfg, rope, "rope_theta");
    }

    private static RowSource source(SafetensorsFile file, String name) {
        StTensor t = file.tensor(name);
        if (t == null) {
            throw new GgufFormatException("'" + file.path() + "' has no tensor '" + name + "'");
        }
        return StRowSource.of(file, t);
    }

    private static JsonView sideJson(HfConfig cfg, String relPath) throws IOException {
        java.nio.file.Path file = cfg.sideFile(relPath);
        if (!Files.isRegularFile(file)) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint has no '" + relPath + "'");
        }
        return HfBertVariantHooks.readJsonView(file);
    }

    private static Object required(HfConfig cfg, JsonView view, String key) {
        if (!view.has(key)) {
            throw new GgufFormatException(view.where(key) + ": the checkpoint requires this key");
        }
        return view.raw(key);
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

    private static double number(HfConfig cfg, JsonView view, String key) {
        Object v = view.raw(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(view.where(key) + ": the checkpoint requires this number");
        }
        return ((Number) v).doubleValue();
    }

    private static int parseIndex(String digits, String where) {
        if (digits.isEmpty() || digits.length() > 9 || !digits.chars().allMatch(Character::isDigit)) {
            throw new GgufFormatException("tensor '" + where + "' has no usable block index '" + digits + "'");
        }
        return Integer.parseInt(digits);
    }

    private static String stripWeight(String rest, String where) {
        if (!rest.endsWith(".weight")) {
            throw new GgufFormatException("tensor '" + where + "' is not a weight");
        }
        return rest.substring(0, rest.length() - ".weight".length());
    }

    private static String suffixOf(String name) {
        return "." + name.substring(name.lastIndexOf('.') + 1);
    }

    /**
     * Equality as the reference's Python {@code ==} over JSON values: numbers compare by value (the parser stores
     * integers as {@link Long} where the expected tables hold {@code Integer}), lists element by element, the rest
     * with {@link Object#equals}. The converter's expected tables include the integer list {@code fsq_levels}.
     */
    private static boolean valuesEqual(Object expected, Object actual) {
        if (expected instanceof Number && actual instanceof Number) {
            return ((Number) expected).doubleValue() == ((Number) actual).doubleValue();
        }
        if (expected instanceof List && actual instanceof List) {
            List<?> e = (List<?>) expected;
            List<?> a = (List<?>) actual;
            if (e.size() != a.size()) {
                return false;
            }
            for (int i = 0; i < e.size(); i++) {
                if (!valuesEqual(e.get(i), a.get(i))) {
                    return false;
                }
            }
            return true;
        }
        return expected == null ? actual == null : expected.equals(actual);
    }

    // ------------------------------------------------------------------ constant tables

    private static Map<String, Object> featureExtractor() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("sampling_rate", 16000);
        m.put("feature_size", 128);
        m.put("n_fft", 400);
        m.put("win_length", 400);
        m.put("hop_length", 160);
        m.put("log_floor", -12.0);
        m.put("normalization_offset", 4.0);
        m.put("normalization_divisor", 4.0);
        m.put("encoder_stride", 8);
        return Collections.unmodifiableMap(m);
    }

    private static Map<String, Object> codec() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("model_type", "liquid_codec");
        m.put("fsq_levels", Arrays.asList(8, 7, 6, 6));
        m.put("n_q", 12);
        m.put("quantizer_dim", 256);
        m.put("encoder_dim", 512);
        m.put("stft_n_fft", 1280);
        m.put("stft_hop_length", 320);
        m.put("stft_win_length", 1280);
        m.put("stft_frame_stack", 6);
        m.put("decoder_window", 30);
        m.put("sampling_rate", 24000);
        return Collections.unmodifiableMap(m);
    }

    private static Map<String, Object> generation() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("codebooks", 12);
        m.put("maskgit_schedule", Arrays.asList(4, 4, 4));
        m.put("interleaved_n_text", 6);
        m.put("interleaved_n_audio", 12);
        return Collections.unmodifiableMap(m);
    }

    private static Map<String, Double> sampling(double temperature, Double topK, double penalty, double window) {
        Map<String, Double> m = new LinkedHashMap<>();
        m.put("audio_temperature", temperature);
        m.put("audio_top_k", topK);
        m.put("audio_repetition_penalty", penalty);
        m.put("audio_repetition_window", window);
        return Collections.unmodifiableMap(m);
    }

    private static Map<String, String> depthLayers() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("operator_norm", "A_GEN_CODE_ATTN_NORM");
        m.put("self_attn.q_proj", "A_GEN_CODE_ATTN_Q");
        m.put("self_attn.q_norm", "A_GEN_CODE_ATTN_Q_NORM");
        m.put("self_attn.k_proj", "A_GEN_CODE_ATTN_K");
        m.put("self_attn.k_norm", "A_GEN_CODE_ATTN_K_NORM");
        m.put("self_attn.v_proj", "A_GEN_CODE_ATTN_V");
        m.put("self_attn.o_proj", "A_GEN_CODE_ATTN_OUT");
        m.put("self_attn.gate_proj", "A_GEN_CODE_ATTN_GATE");
        m.put("ffn_norm", "A_GEN_CODE_FFN_NORM");
        m.put("feed_forward.w1", "A_GEN_CODE_FFN_GATE");
        m.put("feed_forward.w3", "A_GEN_CODE_FFN_UP");
        m.put("feed_forward.w2", "A_GEN_CODE_FFN_DOWN");
        return Collections.unmodifiableMap(m);
    }

    private static Map<String, String> codecLayers() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("operator_norm", "A_GEN_WAV_TFM_ATTN_NORM");
        m.put("post_operator_norm", "A_GEN_WAV_TFM_ATTN_POST_NORM");
        m.put("self_attn.q_proj", "A_GEN_WAV_TFM_ATTN_Q");
        m.put("self_attn.q_norm", "A_GEN_WAV_TFM_ATTN_Q_NORM");
        m.put("self_attn.k_proj", "A_GEN_WAV_TFM_ATTN_K");
        m.put("self_attn.k_norm", "A_GEN_WAV_TFM_ATTN_K_NORM");
        m.put("self_attn.v_proj", "A_GEN_WAV_TFM_ATTN_V");
        m.put("self_attn.o_proj", "A_GEN_WAV_TFM_ATTN_OUT");
        m.put("self_attn.gate_proj", "A_GEN_WAV_TFM_ATTN_GATE");
        m.put("conv.in_proj", "A_GEN_WAV_TFM_CONV_PW1");
        m.put("conv.conv", "A_GEN_WAV_TFM_CONV_DW");
        m.put("conv.out_proj", "A_GEN_WAV_TFM_CONV_PW2");
        m.put("ffn_norm", "A_GEN_WAV_TFM_FFN_NORM");
        m.put("post_ffn_norm", "A_GEN_WAV_TFM_FFN_POST_NORM");
        m.put("feed_forward.w1", "A_GEN_WAV_TFM_FFN_GATE");
        m.put("feed_forward.w3", "A_GEN_WAV_TFM_FFN_UP");
        m.put("feed_forward.w2", "A_GEN_WAV_TFM_FFN_DOWN");
        return Collections.unmodifiableMap(m);
    }

    private static Map<String, Integer> downsampleLayers() {
        Map<String, Integer> m = new LinkedHashMap<>();
        m.put("conv_in", 0);
        m.put("conv_mid", 2);
        m.put("lin_mid", 3);
        m.put("conv_end", 5);
        m.put("lin_end", 6);
        return Collections.unmodifiableMap(m);
    }

    /** The buffered depthformer pages of one load. */
    static final class DepthPages {
        final Map<Integer, RowSource> embed = new LinkedHashMap<>();
        final Map<Integer, RowSource> norm = new LinkedHashMap<>();
        /** The tied head pages whose embedding page had not arrived yet, checked once it does. */
        final Map<Integer, RowSource> logits = new LinkedHashMap<>();
    }

    /** The derived generation shape: codebook count, audio vocabulary and maskgit schedule. */
    private static final class Generation {
        private final int codebooks;
        private final int audioVocab;
        private final Integer milegitSteps;
        private final JsonView root;

        private Generation(int codebooks, int audioVocab, Integer steps, JsonView root) {
            this.codebooks = codebooks;
            this.audioVocab = audioVocab;
            this.milegitSteps = steps;
            this.root = root;
        }

        static Generation of(int codebooks, int audioVocab, Integer steps, JsonView root) {
            return new Generation(codebooks, audioVocab, steps, root);
        }

        int codebooks() {
            return codebooks;
        }

        int audioVocab() {
            return audioVocab;
        }

        Object value(String key) {
            // the layouts name the derived counts differently: codebooks/audio_vocab_size in config.json of the
            // LFM3-Audio checkpoint, num_codebooks/codebook_size in the Omni one
            if ("codebooks".equals(key)) {
                return codebooks;
            }
            if ("audio_vocab_size".equals(key)) {
                return audioVocab;
            }
            Object v = root.raw(key);
            if ("maskgit_schedule".equals(key) && v == null) {
                int steps = milegitSteps == null ? 0 : milegitSteps;
                if (steps <= 0 || codebooks % steps != 0) {
                    throw new GgufFormatException("the maskgit step count " + steps + " does not divide the "
                            + codebooks + " codebooks");
                }
                List<Integer> schedule = new ArrayList<>(steps);
                for (int i = 0; i < steps; i++) {
                    schedule.add(codebooks / steps);
                }
                return schedule;
            }
            return v;
        }
    }
}
