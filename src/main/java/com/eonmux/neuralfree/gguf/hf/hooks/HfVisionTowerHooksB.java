package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The MoonViT (Kimi-K2.5, GLM-5.2-Vision), MiniMax-M3, Muse-Glimmer / Muse-Spark, Phi-4, Granite Vision 4.1 and
 * Nemotron Nano V2 VL / RADIO projector classes, ports of the llama.cpp converter classes {@code KimiK25Model}
 * (kimivl.py:57), {@code Glm5vModel} (kimivl.py:161), {@code MiniMaxM3VisionModel} (minimax.py:210), {@code
 * MuseGlimmerVisionModel} (muse_glimmer.py:83), {@code MuseSparkVisionModel} (muse_spark.py:146), {@code
 * Phi4VisionMmprojModel} (phi.py:216), {@code Granite4VisionMmprojModel} (granite.py:692) and {@code
 * NemotronNanoV2VLModel} (nemotron.py:124). The classes share the layout rewrites of their towers: the MoonViT Q/K
 * permutation and patch-merger names, the MiniMax temporal patch split and 3-D rope permutation, the Muse patch
 * slab sum and inverse rope permutation, the Phi-4 convolution kernel and truncated tower, the Granite stacked
 * projectors and the RADIO position resampling.
 */
public final class HfVisionTowerHooksB {

    /** {@code VisionProjectorType.KIMIK25}. */
    public static final String KIMIK25 = "kimik25";
    /** {@code VisionProjectorType.MINIMAXM3}. */
    public static final String MINIMAX_M3 = "minimax_m3";
    /** {@code VisionProjectorType.MUSE_GLIMMER}. */
    public static final String MUSE_GLIMMER = "muse-glimmer";
    /** {@code VisionProjectorType.MUSE_SPARK_AUDIO}. */
    public static final String MUSE_SPARK_AUDIO = "muse-spark-audio";
    /** {@code VisionProjectorType.PHI4}. */
    public static final String PHI4 = "phi4";
    /** {@code VisionProjectorType.GRANITE4_VISION}. */
    public static final String GRANITE4_VISION = "granite4_vision";
    /** {@code VisionProjectorType.NEMOTRON_V2_VL}. */
    public static final String NEMOTRON_V2_VL = "nemotron_v2_vl";
    /** {@code VisionProjectorType.PARAKEET}. */
    public static final String PARAKEET = "parakeet";

    /** The fixed ViT-Huge geometry of the RADIO backbone (nemotron.py:126-130). */
    private static final int NEMOTRON_HIDDEN = 1280;
    private static final int NEMOTRON_FFN = 5120;
    private static final int NEMOTRON_BLOCKS = 32;
    private static final int NEMOTRON_HEADS = 16;
    private static final int NEMOTRON_IMAGE_SIZE = 512;
    /** The dataset colors of the Nemotron and SenseNova towers. */
    private static final List<Float> IMAGENET_MEAN = floats(0.485f, 0.456f, 0.406f);
    private static final List<Float> IMAGENET_STD = floats(0.229f, 0.224f, 0.225f);
    /** The Kimi-K2.5 position table defaults (kimivl.py:84-86). */
    private static final int KIMI_POS_HEIGHT = 64;
    private static final int KIMI_POS_WIDTH = 64;
    private static final int KIMI_POS_TIME = 4;
    /** The Granite Vision image size written when the vision config has none (granite.py:748). */
    private static final int GRANITE_IMAGE_SIZE = 384;
    /** The id runs the Granite projector names carry (granite.py:806). */
    private static final Pattern ID = Pattern.compile("\\.([0-9]+)\\.");

    private HfVisionTowerHooksB() {
    }

    private static List<Float> floats(float... values) {
        List<Float> out = new ArrayList<>(values.length);
        for (float v : values) {
            out.add(v);
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * The tensor-map block count of Muse-Spark: the vision tower's layers, or the summed audio stage depths when the
     * speech encoder is deeper (both encoders share one name map).
     */
    public static int museSparkBlocks(HfConfig cfg) {
        int vision = HfMmprojHooks.visionBlocks(cfg);
        JsonView audio = HfMmprojHooks.object(cfg, cfg.raw(), "audio_config");
        if (!audio.has("depths")) {
            return vision;
        }
        int audioBlocks = 0;
        for (int depth : HfMmprojHooks.intArray(cfg, audio, "depths")) {
            audioBlocks += depth;
        }
        return Math.max(vision, audioBlocks);
    }

    private static JsonView vision(HfConfig cfg) {
        return cfg.vision().orElseThrow(() -> new GgufFormatException(cfg.source()
                + ": the checkpoint has no 'vision_config'"));
    }

    // ------------------------------------------------------------------ Kimi-K2.5 / GLM-5.2-Vision (MoonViT)

    /** {@code KimiK25Model.filter_tensors} (kimivl.py:104-115): the vision tower and the projector. */
    public static final HfNameFilter KIMI_FILTER = (name, base) ->
            name.contains("vision_tower") || name.contains("mm_projector") ? base.apply(name) : null;

    /**
     * {@code KimiK25Model.set_gguf_parameters} (kimivl.py:73-101): the base vision keys of the MoonViT tower,
     * {@code kimik25}, the position-table geometry, GELU, the projector epsilon, the merge factor and the pixel
     * bounds derived from the patch size and the processor's patch limit.
     *
     * @throws GgufFormatException naming the missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void kimiMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView vision = vision(cfg);
        int patch = HfMmprojHooks.intOr(cfg, vision, 14, "patch_size");
        int posHeight = HfMmprojHooks.intOr(cfg, vision, KIMI_POS_HEIGHT, "init_pos_emb_height");
        HfMmprojHooks.visionMeta(ctx, kv, posHeight * patch);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, KIMIK25);
        kv.setTyped("vision.pos_emb_height", com.eonmux.neuralfree.gguf.GGUFValueType.UINT32, posHeight);
        kv.setTyped("vision.pos_emb_width", com.eonmux.neuralfree.gguf.GGUFValueType.UINT32,
                HfMmprojHooks.intOr(cfg, vision, KIMI_POS_WIDTH, "init_pos_emb_width"));
        kv.setTyped("vision.pos_emb_time", com.eonmux.neuralfree.gguf.GGUFValueType.UINT32,
                HfMmprojHooks.intOr(cfg, vision, KIMI_POS_TIME, "init_pos_emb_time"));
        Object act = HfMmprojHooks.visionFind(cfg, "projector_hidden_act").orElse("gelu");
        kv.set(GgufKeys.CLIP_USE_GELU, "gelu".equals(String.valueOf(act)));
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "projector_ln_eps").orElse(1e-5));
        int merge = mergeKernel(cfg)[0];
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, merge);
        int inPatchLimit = HfMmprojHooks.preIntOr(cfg, "in_patch_limit", 16_384);
        long pixelsPerPatch = (long) patch * patch;
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, 8L * pixelsPerPatch);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, inPatchLimit * pixelsPerPatch);
    }

    /**
     * {@code KimiK25Model.modify_tensors} (kimivl.py:118-157): the fused {@code wqkv} is split, its Q and K halves
     * permuted from the interleaved to the split rope layout and fused again; the time table loses its middle axis;
     * the projector's numbered layers take their {@code linear_1} / {@code linear_2} names.
     *
     * @throws GgufFormatException when the fused projection does not split into three equal heads of four groups
     */
    public static void kimiTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        String name = "Glm5vForConditionalGeneration".equals(ctx.config().architecture())
                && hfName.startsWith("mm_projector.linear_")
                ? "mm_projector.proj.linear_" + hfName.substring("mm_projector.linear_".length()) : hfName;
        if (name.contains("wqkv")) {
            out.add(ctx.names().mapOrThrow(name), permuteQkv(ctx, name, src));
            return;
        }
        if (name.contains("pos_emb.time_weight")) {
            long[] shape = src.shape();
            if (shape.length != 3 || shape[1] != 1) {
                throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has the shape "
                        + Arrays.toString(shape) + "; the MoonViT time table is [T, 1, C] (kimivl.py:150-152)");
            }
            out.add(ctx.names().mapOrThrow(name), ElementTransforms.reshape(src, shape[0], shape[2]));
            return;
        }
        if (name.contains("mm_projector.proj.0.")) {
            name = name.replace(".proj.0.", ".proj.linear_1.");
        } else if (name.contains("mm_projector.proj.2.")) {
            name = name.replace(".proj.2.", ".proj.linear_2.");
        }
        out.add(ctx.names().mapOrThrow(name), src);
    }

    /**
     * {@code KimiK25Model.permute} over the three fused projections (kimivl.py:130-146): the weight's rows and the
     * bias's entries move per head as {@code reshape(n, hd/4, 2, 2, ...).permute(0, 2, 1, 3, ...)}.
     */
    private static RowSource permuteQkv(HfLoadContext ctx, String name, RowSource src) {
        int heads = HfMmprojHooks.visionIntOpt(ctx.config(), "num_attention_heads", "num_heads", "heads")
                .orElse(16);
        long[] shape = src.shape();
        long outDim = shape[0];
        if (outDim % 3 != 0) {
            throw new GgufFormatException("MoonViT fused projection '" + name + "' has " + outDim
                    + " outputs, not three equal Q/K/V parts");
        }
        long headDim = outDim / 3 / heads;
        if (heads <= 0 || headDim <= 0 || headDim % 4 != 0) {
            throw new GgufFormatException("MoonViT fused projection '" + name + "' has " + heads + " heads of "
                    + headDim + " rows; the rope permutation needs a positive head width divisible by 4");
        }
        int part = Math.toIntExact(outDim / 3);
        int[] index = new int[part];
        int quarter = (int) (headDim / 4);
        for (int h = 0; h < heads; h++) {
            int headStart = h * (int) headDim;
            for (int i = 0; i < headDim; i++) {
                int b = i / (quarter * 2);
                int rest = i % (quarter * 2);
                int a = rest / 2;
                int c = rest % 2;
                index[headStart + i] = headStart + a * 4 + b * 2 + c;
            }
        }
        RowSource q = RowTransforms.gatherRows(RowTransforms.sliceRows(src, 0, part), 1, index);
        RowSource k = RowTransforms.gatherRows(RowTransforms.sliceRows(src, part, part), 1, index);
        RowSource v = RowTransforms.sliceRows(src, 2L * part, part);
        return RowTransforms.concatRows(Arrays.asList(q, k, v));
    }

    /** The merge kernel {@code [h, w]}, the default 2x2 (kimivl.py:66). */
    private static int[] mergeKernel(HfConfig cfg) {
        List<Integer> list = HfMmprojHooks.intList(cfg, vision(cfg), "merge_kernel_size");
        if (list.isEmpty()) {
            return new int[]{2, 2};
        }
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }

    // ------------------------------------------------------------------ MiniMax-M3

    /** {@code MiniMaxM3VisionModel.filter_tensors} (minimax.py:216-221). */
    public static final HfNameFilter MINIMAX_FILTER = (name, base) ->
            name.startsWith("vision_tower.") || name.startsWith("multi_modal_projector.")
                    || name.startsWith("patch_merge_mlp.") ? base.apply(name) : null;

    /**
     * {@code MiniMaxM3VisionModel.set_gguf_parameters} (minimax.py:223-235): the base vision keys, {@code minimax_m3},
     * GELU, the tower's own LayerNorm epsilon and the image-token merge size.
     *
     * @throws GgufFormatException naming the missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void minimaxMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, MINIMAX_M3);
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(1e-5));
        Object comp = HfMmprojHooks.visionFind(cfg, "img_token_compression_config").orElse(null);
        int merge = 2;
        if (comp instanceof Map) {
            merge = HfMmprojHooks.intOr(cfg, JsonView.of(mapOf(comp), "img_token_compression_config"), 2,
                    "spatial_merge_size");
        }
        kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, merge);
    }

    /**
     * {@code MiniMaxM3VisionModel.modify_tensors} (minimax.py:237-282): the Conv3d patch embedding splits into one
     * Conv2d kernel per temporal slice and the ViT Q/K rows are permuted from the rope layout.
     *
     * @throws GgufFormatException when the patch embedding is not a Conv3d weight or Q/K's head width is not the
     *                             fixed 80
     */
    public static void minimaxTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if ("vision_tower.vision_model.embeddings.patch_embedding.weight".equals(hfName)) {
            long[] shape = src.shape();
            if (shape.length != 5) {
                throw new GgufFormatException("unexpected patch_embedding rank " + shape.length + " for " + hfName
                        + " (minimax.py:239)");
            }
            String base = ctx.names().ggufName("V_ENC_EMBD_PATCH", -1);
            for (int t = 0; t < shape[2]; t++) {
                String suffix = t == 0 ? ".weight" : ".weight." + t;
                out.add(base + suffix, ElementTransforms.squeezeConv(RowTransforms.sliceDim(src, 2, t, 1)));
            }
            return;
        }
        String mapped = ctx.names().mapOrThrow(hfName);
        if (mapped.contains(".attn_q.") || mapped.contains(".attn_k.")) {
            out.add(mapped, permuteVitQk(ctx, hfName, mapped, src));
            return;
        }
        out.add(mapped, src);
    }

    /**
     * {@code MiniMaxM3VisionModel._permute_vit_qk} (minimax.py:262-282): the 3-D rope groups reorder inside each
     * head, {@code [Ta Ha Wa | Tb Hb Wb | pad]} to {@code [Ta Tb | Ha Hb | Wa Wb | pad]}.
     */
    private static RowSource permuteVitQk(HfLoadContext ctx, String hfName, String mapped, RowSource src) {
        int heads = HfMmprojHooks.visionInt(ctx.config(), "num_attention_heads", "num_heads", "heads");
        long[] shape = src.shape();
        long outDim = shape[0];
        if (heads <= 0 || outDim % heads != 0) {
            throw new GgufFormatException("MiniMax-M3 tensor '" + hfName + "' has " + outDim + " rows for " + heads
                    + " heads");
        }
        int headDim = Math.toIntExact(outDim / heads);
        int axisDim = 2 * ((2 * (headDim / 2) / 3) / 2);
        int ah = axisDim / 2;
        int half = 3 * ah;
        if (headDim != 80 || axisDim % 2 != 0 || 3 * axisDim > headDim) {
            throw new GgufFormatException("MiniMax-M3 tensor '" + hfName + "' has a head width of " + headDim
                    + "; the 3-D rope permutation expects 80");
        }
        int[] within = new int[headDim];
        int at = 0;
        at = append(within, at, 0, ah);
        at = append(within, at, half, half + ah);
        at = append(within, at, ah, 2 * ah);
        at = append(within, at, half + ah, half + 2 * ah);
        at = append(within, at, 2 * ah, 3 * ah);
        at = append(within, at, half + 2 * ah, half + 3 * ah);
        append(within, at, 2 * half, headDim);
        int[] index = new int[Math.toIntExact(outDim)];
        for (int h = 0; h < heads; h++) {
            for (int i = 0; i < headDim; i++) {
                index[h * headDim + i] = h * headDim + within[i];
            }
        }
        return RowTransforms.gatherRows(src, 1, index);
    }

    private static int append(int[] target, int at, int from, int to) {
        for (int i = from; i < to; i++) {
            target[at++] = i;
        }
        return at;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> mapOf(Object value) {
        return (Map<String, Object>) value;
    }

    // ------------------------------------------------------------------ Muse-Glimmer / Muse-Spark

    /** {@code MuseGlimmerVisionModel.filter_tensors} (muse_glimmer.py:104-110). */
    public static final HfNameFilter MUSE_GLIMMER_FILTER = (name, base) ->
            name.startsWith("model.vision_tower.") || name.startsWith("model.vision_adapter.")
                    || name.startsWith("model.vision_projection.") ? base.apply(name) : null;

    /** {@code MuseSparkVisionModel.filter_tensors} (muse_spark.py:180-186): its audio tower takes the base filter. */
    public static final HfNameFilter MUSE_SPARK_FILTER = (name, base) -> {
        if (name.startsWith("model.audio_tower.")) {
            return base.apply(name);
        }
        return MUSE_GLIMMER_FILTER.filter(name, base);
    };

    /**
     * {@code MuseGlimmerVisionModel.set_gguf_parameters} (muse_glimmer.py:91-99): the base vision keys at the
     * nominal grid {@code pos_emb_height * patch_size * merge_size}, {@code muse-glimmer}, the epsilon and the merge.
     *
     * @throws GgufFormatException naming the missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void museGlimmerMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView vision = vision(cfg);
        int patch = HfMmprojHooks.visionInt(cfg, "patch_size");
        int merge = HfMmprojHooks.visionInt(cfg, "merge_size");
        int imageSize = HfMmprojHooks.visionInt(cfg, "pos_emb_height") * patch * merge;
        HfMmprojHooks.visionMeta(ctx, kv, imageSize);
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_TYPE, MUSE_GLIMMER);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, HfMmprojHooks.visionF64(cfg, "layer_norm_eps"));
        kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, merge);
    }

    /**
     * {@code MuseSparkVisionModel.set_gguf_parameters} (muse_spark.py:160-178): the Muse-Glimmer vision keys and the
     * audio stages' type, mel count, epsilon, convolution kernel, widths, patch sizes and depths.
     *
     * @throws GgufFormatException naming the missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void museSparkMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        museGlimmerMeta(ctx, kv, plan);
        JsonView audio = audio(cfg);
        int hidden = lastInt(cfg, audio, "hidden_sizes");
        double ratio = HfMmprojHooks.f64Or(cfg, audio, 1.0, "mlp_ratio");
        int blocks = sumInt(cfg, audio, "depths");
        HfMmprojHooks.audioMeta(ctx, kv, hidden, (int) (hidden * ratio), 1, blocks);
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, MUSE_SPARK_AUDIO);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, 1);
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, HfMmprojHooks.f64Or(cfg, audio, 1e-5,
                "rms_norm_eps"));
        kv.set(GgufKeys.CLIP_AUDIO_CONV_KERNEL_SIZE, HfMmprojHooks.intOf(cfg, audio, "audio_config",
                "conv_kernel_size"));
        kv.set(GgufKeys.CLIP_AUDIO_HIDDEN_SIZES, HfMmprojHooks.intList(cfg, audio, "hidden_sizes"));
        kv.set(GgufKeys.CLIP_AUDIO_PATCH_SIZES, HfMmprojHooks.intList(cfg, audio, "patch_sizes"));
        kv.set(GgufKeys.CLIP_AUDIO_DEPTHS, HfMmprojHooks.intList(cfg, audio, "depths"));
    }

    /** The Muse projector layers the vision encoder maps through the clip table (muse_glimmer.py:112-117, :121-134). */
    public static void museTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        int heads = HfMmprojHooks.visionInt(ctx.config(), "num_attention_heads", "num_heads", "heads");
        RowSource data = src;
        if (hfName.contains(".attn.q_proj.") || hfName.contains(".attn.k_proj.")) {
            data = RowTransforms.rotateHalfUnpermute(src, heads);
        }
        if (hfName.endsWith("patch_embedder.patch_embedding.weight")) {
            JsonView vision = vision(ctx.config());
            int temporal = HfMmprojHooks.intOf(ctx.config(), vision, "vision_config", "patch_temporal");
            int patch = HfMmprojHooks.intOf(ctx.config(), vision, "vision_config", "patch_size");
            long[] shape = data.shape();
            if (shape.length != 2 || shape[0] <= 0
                    || shape[1] != 3L * temporal * patch * patch) {
                throw new GgufFormatException("Muse patch embedding has shape " + Arrays.toString(shape)
                        + "; expected [n_embd, " + (3L * temporal * patch * patch) + "] (muse_glimmer.py:118-121)");
            }
            RowSource viewed = ElementTransforms.reshape(data, shape[0], temporal, 3, patch, patch);
            data = ElementTransforms.sumAxis(viewed, 1);
        }
        String suffix = hfName.endsWith(".bias") ? ".bias" : ".weight";
        String stem = hfName.endsWith(".bias") || hfName.endsWith(".weight")
                ? hfName.substring(0, hfName.lastIndexOf('.')) : hfName;
        int index;
        if ("model.vision_adapter.fc1".equals(stem)) {
            index = 0;
        } else if ("model.vision_adapter.fc2".equals(stem)) {
            index = 1;
        } else if ("model.vision_projection".equals(stem)) {
            index = 2;
        } else {
            out.add(ctx.names().mapOrThrow(hfName), data);
            return;
        }
        out.add(ctx.names().ggufName("V_MMPROJ", index, suffix), data);
    }

    /**
     * {@code MuseSparkVisionModel.modify_tensors} (muse_spark.py:188-222): the audio head tensors, the per-stage
     * patch embedder and the flat convolution blocks, plus the Muse-Glimmer vision path.
     *
     * @throws GgufFormatException when an audio tensors' stage or block name is malformed
     */
    public static void museSparkTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("model.audio_tower.")) {
            out.add(audioName(ctx, hfName), src);
            return;
        }
        museTensors(ctx, hfName, src, out);
    }

    /** The GGUF name of an audio tensor of Muse-Spark (muse_spark.py:190-220). */
    private static String audioName(HfLoadContext ctx, String hfName) {
        String suffix = hfName.substring(hfName.lastIndexOf('.'));
        String stem = hfName.substring(0, hfName.length() - suffix.length());
        String tensor;
        if ("model.audio_tower.norm".equals(stem)) {
            tensor = "A_PRE_NORM";
        } else if ("model.audio_tower.proj".equals(stem)) {
            tensor = "A_MMPROJ_FC";
        } else if ("model.audio_tower.post_proj_norm".equals(stem)) {
            tensor = "A_POST_NORM";
        } else {
            String[] parts = stem.split("\\.");
            if (parts.length < 6) {
                throw new GgufFormatException("Unexpected Muse-Spark audio tensor '" + hfName + "'");
            }
            int stage = parseIndex(parts[3], hfName);
            if ("patch_embedder".equals(parts[4])) {
                tensor = "proj".equals(parts[5]) ? "A_ENC_PATCH_PROJ" : "A_ENC_PATCH_NORM";
                return ctx.names().ggufName(tensor, stage, suffix);
            }
            int[] depths = HfMmprojHooks.intArray(ctx.config(), audio(ctx.config()), "depths");
            int index = parseIndex(parts[5], hfName);
            int bid = 0;
            for (int i = 0; i < stage && i < depths.length; i++) {
                bid += depths[i];
            }
            bid += index;
            String tail = join(parts, 6);
            switch (tail) {
                case "conv": tensor = "A_ENC_CONV1D"; break;
                case "norm": tensor = "A_ENC_CONV1D_NORM"; break;
                case "mlp.fc1": tensor = "A_ENC_FFN_UP"; break;
                case "mlp.fc2": tensor = "A_ENC_FFN_DOWN"; break;
                default:
                    throw new GgufFormatException("Unexpected Muse-Spark audio tensor '" + hfName + "'");
            }
            return ctx.names().ggufName(tensor, bid, suffix);
        }
        return ctx.names().ggufName(tensor, -1, suffix);
    }

    private static int parseIndex(String s, String hfName) {
        try {
            return Integer.parseInt(s);
        } catch (NumberFormatException e) {
            throw new GgufFormatException("Unexpected Muse-Spark audio tensor '" + hfName + "'", e);
        }
    }

    private static String join(String[] parts, int from) {
        StringBuilder b = new StringBuilder();
        for (int i = from; i < parts.length; i++) {
            if (i > from) {
                b.append('.');
            }
            b.append(parts[i]);
        }
        return b.toString();
    }

    /** The last entry of an integer list of the audio config (muse_spark.py:153-155). */
    private static int lastInt(HfConfig cfg, JsonView view, String key) {
        List<Integer> list = HfMmprojHooks.intList(cfg, view, key);
        if (list.isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": the audio config has no '" + key + "'");
        }
        return list.get(list.size() - 1);
    }

    /** The sum of an integer list of the audio config (muse_spark.py:157). */
    private static int sumInt(HfConfig cfg, JsonView view, String key) {
        int sum = 0;
        for (int v : HfMmprojHooks.intList(cfg, view, key)) {
            sum += v;
        }
        return sum;
    }

    private static JsonView audio(HfConfig cfg) {
        return HfMmprojHooks.audioConfig(cfg).orElseThrow(() -> new GgufFormatException(cfg.source()
                + ": the checkpoint has no audio configuration"));
    }

    // ------------------------------------------------------------------ Phi-4

    /** {@code Phi4VisionMmprojModel.filter_tensors} (phi.py:305-322). */
    public static final HfNameFilter PHI4_FILTER = (name, base) -> {
        String renamed = name.replace("model.vision_tower.vision_tower.", "vision_tower.");
        if (!renamed.startsWith("vision_tower.") && !renamed.startsWith("model.mm_projector.")
                && !renamed.startsWith("mm_projector.")) {
            return null;
        }
        if (renamed.contains(".vision_model.head.") || renamed.contains(".vision_model.post_layernorm.")) {
            return null;
        }
        return base.apply(renamed);
    };

    /** The exported block count: every vision layer but the last (phi.py:230-234). */
    public static int phi4Blocks(HfConfig cfg) {
        JsonView vision = vision(cfg);
        int total = HfMmprojHooks.blocksOf(cfg, vision, "vision_config");
        if (total < 2) {
            throw new GgufFormatException(cfg.source() + ": Phi-4 vision mmproj conversion requires at least 2"
                    + " vision layers, got " + total + " (phi.py:226-229)");
        }
        return total - 1;
    }

    /**
     * {@code Phi4VisionMmprojModel.set_gguf_parameters} (phi.py:284-292): the base keys of the truncated tower, whose
     * patch size comes from the processor and whose image size is the position table's square grid, {@code phi4},
     * the pixel bounds and GELU.
     *
     * @throws GgufFormatException when the position table is not a square patch grid
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void phi4Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        int patch = HfMmprojHooks.preInt(cfg, "patch_size");
        String posName = "v.position_embd.weight";
        if (!plan.has(posName)) {
            throw new GgufFormatException(cfg.source() + ": Phi-4 vision mmproj conversion could not find"
                    + " position_embedding.weight (phi.py:250-253)");
        }
        long[] ne = plan.ne(posName);
        long tokens = ne[ne.length - 1];
        int side = (int) Math.sqrt(tokens);
        if ((long) side * side != tokens) {
            throw new GgufFormatException("Unexpected Phi-4 position embedding shape: " + Arrays.toString(ne)
                    + " (phi.py:256-258)");
        }
        int minPatches = intOrGlobal(cfg, "min_num_patches");
        int maxPatches = intOrGlobal(cfg, "max_num_patches");
        long pixelsPerPatch = (long) patch * patch;
        HfMmprojHooks.visionMeta(ctx, kv, side * patch, patch,
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.FFN_KEYS),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.HEAD_KEYS), phi4Blocks(cfg));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, PHI4);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, minPatches * pixelsPerPatch);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, maxPatches * pixelsPerPatch);
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(1e-6));
    }

    /**
     * {@code Phi4VisionMmprojModel.modify_tensors} (phi.py:324-360): the last vision block is dropped, the flat patch
     * embedding becomes a convolution kernel, and the projector's layers {@code 0} and {@code 2} take {@code mm.0}
     * and {@code mm.2}.
     *
     * @throws GgufFormatException when the patch embedding does not split into channels and patches
     */
    public static void phi4Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("vision_tower.")) {
            int bid = HfLoadContext.bidOf(hfName);
            if (bid >= 0 && bid == phi4Blocks(ctx.config())) {
                return;
            }
            RowSource data = src;
            if (hfName.endsWith("vision_model.embeddings.patch_embedding.weight")) {
                int patch = prePatch(ctx);
                long[] shape = src.shape();
                if (shape.length != 2) {
                    throw new GgufFormatException("Unexpected Phi-4 patch embedding shape: " + Arrays.toString(shape)
                            + " (phi.py:330-333)");
                }
                long patchArea = (long) patch * patch;
                if (shape[1] % patchArea != 0) {
                    throw new GgufFormatException("Phi-4 patch embedding input dim " + shape[1]
                            + " is not divisible by patch area " + patchArea + " (phi.py:335-340)");
                }
                int channels = Math.toIntExact(shape[1] / patchArea);
                RowSource viewed = ElementTransforms.reshape(src, shape[0], patch, patch, channels);
                data = ElementTransforms.permuteDims(viewed, 0, 3, 1, 2);
            }
            out.add(ctx.names().mapOrThrow(hfName), data);
            return;
        }
        String local = hfName.replace("model.mm_projector.", "").replace("mm_projector.", "");
        if (!local.startsWith("0.") && !local.startsWith("2.")) {
            return;
        }
        String suffix = local.endsWith(".bias") ? ".bias" : ".weight";
        int index = Integer.parseInt(local.substring(0, 1));
        out.add(ctx.names().ggufName("V_MMPROJ", index, suffix), src);
    }

    private static int prePatch(HfLoadContext ctx) {
        try {
            return HfMmprojHooks.preInt(ctx.config(), "patch_size");
        } catch (IOException e) {
            throw new GgufFormatException(ctx.config().source() + ": the preprocessor config has no patch_size", e);
        }
    }

    /** {@code preprocessor_config[key]} else {@code global_config[key]}, required (phi.py:262-269). */
    private static int intOrGlobal(HfConfig cfg, String key) {
        Object raw;
        try {
            raw = HfMmprojHooks.preprocessor(cfg).raw(key);
        } catch (IOException e) {
            throw new GgufFormatException(cfg.source() + ": cannot read the preprocessor configuration", e);
        }
        if (raw == null) {
            raw = cfg.raw().raw(key);
        }
        if (!(raw instanceof Number) || raw instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": Phi-4 vision mmproj conversion requires " + key
                    + " (phi.py:262-269)");
        }
        return ((Number) raw).intValue();
    }

    // ------------------------------------------------------------------ Granite Vision 4.1

    /** {@code Granite4VisionMmprojModel.filter_tensors} (granite.py:773-778). */
    public static final HfNameFilter GRANITE_FILTER = (name, base) ->
            name.contains("vision_model.head") || name.startsWith("lm_head") ? null : base.apply(name);

    /**
     * {@code Granite4VisionMmprojModel.set_gguf_parameters} (granite.py:738-771): the base keys, the
     * {@code granite4_vision} projector, GELU, the preprocessor image size, the QFormer sides, the tapped vision
     * layers, the per-projector spatial offsets and the flattened grid pinpoints.
     *
     * @throws GgufFormatException naming a malformed downsample rate or missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void graniteMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView vision = vision(cfg);
        int imageSize = HfMmprojHooks.visionIntOpt(cfg, "image_size").orElse(GRANITE_IMAGE_SIZE);
        HfMmprojHooks.visionMeta(ctx, kv, imageSize);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, GRANITE4_VISION);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(1e-6));
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        kv.set(GgufKeys.CLIP_VISION_PREPROC_IMAGE_SIZE, imageSize);
        Object rate = cfg.raw().raw("downsample_rate");
        if (!(rate instanceof String)) {
            throw new GgufFormatException(cfg.source() + ": the configuration has no 'downsample_rate'");
        }
        String[] parts = ((String) rate).split("/");
        if (parts.length != 2) {
            throw new GgufFormatException(cfg.source() + ": Invalid 'downsample_rate' value: " + rate
                    + " (granite.py:751)");
        }
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_QUERY_SIDE, parsePositive(parts[0], cfg, "downsample_rate"));
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_WINDOW_SIDE, parsePositive(parts[1], cfg, "downsample_rate"));
        List<int[]> projectorMap = projectorMap(cfg);
        List<Integer> featureLayers = new ArrayList<>(projectorMap.size());
        List<Integer> offsets = new ArrayList<>(projectorMap.size());
        for (int[] entry : projectorMap) {
            featureLayers.add(entry[0]);
            offsets.add(entry[2] == 1 ? entry[3] : -1);
        }
        kv.set(GgufKeys.CLIP_VISION_FEATURE_LAYER, featureLayers);
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SPATIAL_OFFSETS, offsets);
        Object pinpoints = cfg.raw().raw("image_grid_pinpoints");
        if (pinpoints instanceof List && !((List<?>) pinpoints).isEmpty()) {
            List<Integer> flat = new ArrayList<>();
            for (Object pair : (List<?>) pinpoints) {
                if (!(pair instanceof List) || ((List<?>) pair).size() != 2) {
                    throw new GgufFormatException(cfg.source() + ": 'image_grid_pinpoints' holds "
                            + (pair == null ? "null" : "a non-pair") + " (granite.py:765-770)");
                }
                Object h = ((List<?>) pair).get(0);
                Object w = ((List<?>) pair).get(1);
                flat.add(integral(cfg, w, "image_grid_pinpoints"));
                flat.add(integral(cfg, h, "image_grid_pinpoints"));
            }
            kv.set(GgufKeys.CLIP_VISION_IMAGE_GRID_PINPOINTS, flat);
        }
    }

    /**
     * {@code Granite4VisionMmprojModel.modify_tensors} (granite.py:780-819): a projector tensor's layer ids are
     * replaced by its index among the stacked projectors, then mapped.
     *
     * @throws GgufFormatException when a projector tensor carries an unexpected number of ids
     */
    public static void graniteTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        List<int[]> projectorMap = projectorMap(ctx.config());
        String name = hfName;
        for (int projIdx = 0; projIdx < projectorMap.size(); projIdx++) {
            int[] entry = projectorMap.get(projIdx);
            String prefix = "model." + (entry[2] == 1 ? "spatial" : "layerwise") + "_projectors." + entry[3];
            if (!hfName.startsWith(prefix)) {
                continue;
            }
            Matcher m = ID.matcher(name);
            List<int[]> spans = new ArrayList<>();
            while (m.find()) {
                spans.add(new int[]{m.start(), m.start(1), m.end(1)});
            }
            if (spans.isEmpty() || spans.size() > 2) {
                throw new GgufFormatException("Unexpected Granite Vision projector tensor '" + hfName + "'");
            }
            if (spans.size() == 1) {
                int[] s = spans.get(0);
                name = name.substring(0, s[1]) + projIdx + name.substring(s[2]);
            } else {
                int[] first = spans.get(0);
                int[] second = spans.get(1);
                name = name.substring(0, first[0]) + name.substring(first[2], second[1]) + projIdx
                        + name.substring(second[2]);
            }
            break;
        }
        out.add(ctx.names().mapOrThrow(name), src);
    }

    /** {@code get_normalized_projector_map} (granite.py:699-735): vision layer, text layer, type, type index. */
    private static List<int[]> projectorMap(HfConfig cfg) {
        JsonView vision = vision(cfg);
        int nVision = HfMmprojHooks.blocksOf(cfg, vision, "vision_config");
        int nText = HfMmprojHooks.intOf(cfg, cfg.text(), "text_config", "num_hidden_layers");
        List<int[]> out = new ArrayList<>();
        Object deepstack = cfg.raw().raw("deepstack_layer_map");
        List<int[]> deepstackEntries = new ArrayList<>();
        if (deepstack instanceof List) {
            for (Object entry : (List<?>) deepstack) {
                if (!(entry instanceof List) || ((List<?>) entry).size() != 2) {
                    throw new GgufFormatException(cfg.source() + ": 'deepstack_layer_map' holds a non-pair");
                }
                deepstackEntries.add(new int[]{integral(cfg, ((List<?>) entry).get(0), "deepstack_layer_map"),
                        integral(cfg, ((List<?>) entry).get(1), "deepstack_layer_map")});
            }
        }
        Collections.sort(deepstackEntries, (a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0])
                : Integer.compare(a[1], b[1]));
        int deepstackIdx = 0;
        for (int[] entry : deepstackEntries) {
            int vl = entry[0] < 0 ? nVision + entry[0] : entry[0];
            int ll = entry[1] < 0 ? nText + entry[1] : entry[1];
            out.add(new int[]{vl, ll, 0, deepstackIdx++});
        }
        Object spatial = cfg.raw().raw("spatial_target_layers");
        if (spatial instanceof List && !((List<?>) spatial).isEmpty()) {
            Object rawLayer = cfg.raw().raw("spatial_vision_layer");
            int svl = rawLayer instanceof Number && !(rawLayer instanceof Boolean)
                    ? ((Number) rawLayer).intValue() : -1;
            if (svl < 0) {
                svl = nVision + svl;
            }
            int spatialIdx = 0;
            for (Object layer : (List<?>) spatial) {
                out.add(new int[]{svl, integral(cfg, layer, "spatial_target_layers"), 1, spatialIdx++});
            }
        }
        Collections.sort(out, (a, b) -> Integer.compare(a[1], b[1]));
        return out;
    }

    private static int parsePositive(String value, HfConfig cfg, String key) {
        try {
            int n = Integer.parseInt(value);
            if (n > 0) {
                return n;
            }
        } catch (NumberFormatException e) {
            // fall through to the refusal below
        }
        throw new GgufFormatException(cfg.source() + ": Invalid '" + key + "' value: " + value);
    }

    private static int integral(HfConfig cfg, Object value, String key) {
        if (!(value instanceof Number) || value instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' holds a non-integer");
        }
        return ((Number) value).intValue();
    }

    // ------------------------------------------------------------------ Nemotron Nano V2 VL / RADIO

    /** {@code NemotronNanoV2VLModel.filter_tensors} (nemotron.py:199-227). */
    public static final HfNameFilter NEMOTRON_FILTER = (name, base) -> {
        String kept = base.apply(name);
        if (kept == null) {
            return null;
        }
        if (kept.contains("input_conditioner")) {
            return null;
        }
        if (kept.contains("radio_model.model.patch_generator.video_embedder")) {
            return null;
        }
        if (!kept.startsWith("vision_model.radio_model.model.") && !kept.startsWith("mlp1.")
                && !kept.startsWith("sound_encoder.") && !kept.startsWith("sound_projection.")) {
            return null;
        }
        if (kept.contains("patch_generator.pos_embed") && !kept.endsWith(".weight")) {
            kept += ".weight";
        }
        if (kept.contains("conv.norm") && kept.contains("num_batches")) {
            return null;
        }
        return kept;
    };

    /** The RADIO backbone's fixed block count (nemotron.py:126-130, :3108). */
    public static int nemotronBlocks(HfConfig cfg) {
        return NEMOTRON_BLOCKS;
    }

    /**
     * {@code NemotronNanoV2VLModel.set_gguf_parameters} (nemotron.py:137-178): the RADIO tower at its fixed ViT-Huge
     * geometry, the audio stages when the checkpoint has a {@code sound_config}, {@code nemotron_v2_vl}, GELU, the
     * epsilon and the downsample scale factor.
     *
     * @throws GgufFormatException naming the missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void nemotronMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        int imageSize = rootIntOr(cfg, NEMOTRON_IMAGE_SIZE, "force_image_size");
        Object sound = cfg.raw().raw("sound_config");
        boolean audio = sound instanceof Map;
        kv.set(GgufKeys.CLIP_HAS_VISION_ENCODER, true);
        kv.set(GgufKeys.CLIP_VISION_PROJECTION_DIM, HfMmprojHooks.textWidth(cfg));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_SIZE, imageSize);
        kv.set(GgufKeys.CLIP_VISION_PATCH_SIZE, HfMmprojHooks.visionInt(cfg, "patch_size"));
        kv.set(GgufKeys.CLIP_VISION_EMBEDDING_LENGTH, NEMOTRON_HIDDEN);
        kv.set(GgufKeys.CLIP_VISION_FEED_FORWARD_LENGTH, NEMOTRON_FFN);
        kv.set(GgufKeys.CLIP_VISION_BLOCK_COUNT, NEMOTRON_BLOCKS);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_HEAD_COUNT, NEMOTRON_HEADS);
        JsonView pre = HfMmprojHooks.preprocessor(cfg);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MEAN,
                pre.has("image_mean") ? pre.list("image_mean") : IMAGENET_MEAN);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_STD,
                pre.has("image_std") ? pre.list("image_std") : IMAGENET_STD);
        if (audio) {
            JsonView soundView = JsonView.of(mapOf(sound), "sound_config");
            kv.set(GgufKeys.CLIP_HAS_AUDIO_ENCODER, true);
            kv.set(GgufKeys.CLIP_AUDIO_PROJECTION_DIM, HfMmprojHooks.textWidth(cfg));
            kv.set(GgufKeys.CLIP_AUDIO_EMBEDDING_LENGTH, HfMmprojHooks.intOf(cfg, soundView, "sound_config",
                    "hidden_size"));
            kv.set(GgufKeys.CLIP_AUDIO_FEED_FORWARD_LENGTH, HfMmprojHooks.intOf(cfg, soundView, "sound_config",
                    "intermediate_size"));
            kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_HEAD_COUNT, HfMmprojHooks.intOf(cfg, soundView, "sound_config",
                    "num_attention_heads"));
            kv.set(GgufKeys.CLIP_AUDIO_BLOCK_COUNT, HfMmprojHooks.blocksOf(cfg, soundView, "sound_config"));
            kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, HfMmprojHooks.intOf(cfg, soundView, "sound_config",
                    "num_mel_bins"));
            kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, 1e-5);
            kv.set(GgufKeys.CLIP_AUDIO_SUBSAMPLING_FACTOR, HfMmprojHooks.intOf(cfg, soundView, "sound_config",
                    "subsampling_factor"));
            kv.set(GgufKeys.CLIP_AUDIO_CONV_KERNEL_SIZE, HfMmprojHooks.intOf(cfg, soundView, "sound_config",
                    "conv_kernel_size"));
            kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, PARAKEET);
            kv.set(GgufKeys.CLIP_VISION_PROJECTOR_TYPE, NEMOTRON_V2_VL);
        } else {
            kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, NEMOTRON_V2_VL);
        }
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 1e-6);
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        double ratio = rootF64Or(cfg, 0.5, "downsample_ratio");
        if (!(ratio > 0.0)) {
            throw new GgufFormatException(cfg.source() + ": 'downsample_ratio' is " + ratio);
        }
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, (int) (1.0 / ratio));
    }

    /**
     * {@code NemotronNanoV2VLModel.modify_tensors} (nemotron.py:229-257): the RADIO position table is resampled to
     * the fixed patch grid, the patch embedder becomes a convolution kernel, the depthwise/pointwise convolutions
     * take their ggml layouts and the sound stage's projector rows land under {@code mm.a}.
     *
     * @throws GgufFormatException when a resampled tensor's shape does not fit the configuration
     * @throws IOException         when the tensor data cannot be read
     */
    public static void nemotronTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfConfig cfg = ctx.config();
        RowSource data = src;
        if (hfName.contains("patch_generator.pos_embed")) {
            data = resamplePosition(cfg, src);
        }
        if (hfName.contains("patch_generator.embedder")) {
            int patch = HfMmprojHooks.visionIntOpt(cfg, "patch_size").orElse(14);
            long[] shape = data.shape();
            if (shape.length != 2 || shape[1] != 3L * patch * patch) {
                throw new GgufFormatException("RADIO patch embedder has shape " + Arrays.toString(shape)
                        + "; expected [n_embd, " + (3L * patch * patch) + "] (nemotron.py:199-204)");
            }
            data = ElementTransforms.reshape(data, shape[0], 3, patch, patch);
        }
        if (hfName.contains("depthwise_conv.weight")) {
            long[] shape = data.shape();
            if (shape.length != 3) {
                throw new GgufFormatException("RADIO depthwise convolution has shape " + Arrays.toString(shape)
                        + "; expected [channels, 1, kernel]");
            }
            data = ElementTransforms.permuteDims(ElementTransforms.reshape(data, shape[0], shape[1], shape[2], 1),
                    3, 1, 0, 2);
        }
        if (hfName.contains("pointwise_conv") && hfName.endsWith(".weight")) {
            long[] shape = data.shape();
            if (shape.length == 3 && shape[2] == 1) {
                data = ElementTransforms.reshape(data, shape[0], shape[1]);
            }
        }
        if ((hfName.contains("subsampling.layers") || hfName.contains("pointwise_conv"))
                && hfName.endsWith(".bias")) {
            long[] shape = data.shape();
            if (shape.length == 1) {
                data = ElementTransforms.reshape(data, 1, shape[0], 1, 1);
            }
        }
        String mapped = ctx.names().mapOrThrow(hfName);
        if (hfName.startsWith("sound_projection.") && mapped.startsWith("mm.model.mlp.")) {
            mapped = mapped.replace("mm.model.mlp.", "mm.a.mlp.");
        }
        out.add(mapped, data);
    }

    /** The RADIO position table resized from its stored grid to the fixed image's patch grid (nemotron.py:230-245). */
    private static RowSource resamplePosition(HfConfig cfg, RowSource src) throws IOException {
        long[] shape = src.shape();
        if (shape.length != 3 || shape[0] != 1 || shape[2] <= 0) {
            throw new GgufFormatException("RADIO position table has shape " + Arrays.toString(shape)
                    + "; expected [1, tokens, n_embd]");
        }
        int patch = HfMmprojHooks.visionIntOpt(cfg, "patch_size").orElse(14);
        int imageSize = rootIntOr(cfg, NEMOTRON_IMAGE_SIZE, "force_image_size");
        int target = imageSize / patch;
        int stored = (int) Math.sqrt(shape[1]);
        if ((long) stored * stored != shape[1]) {
            throw new GgufFormatException("Unexpected RADIO position table shape " + Arrays.toString(shape));
        }
        if (stored == target) {
            return src;
        }
        int channels = Math.toIntExact(shape[2]);
        RowSource grid = ElementTransforms.reshape(src, 1, stored, stored, channels);
        RowSource resized = ElementTransforms.resizeBilinear(grid, target, target);
        return ElementTransforms.reshape(resized, 1, (long) target * target, channels);
    }

    /** The audio and convolution tensors keep F32 where their layout rewrite needs it (nemotron.py:180-190). */
    public static final HfForceType NEMOTRON_FORCE = (ggufName, nDims, source, outType) -> {
        boolean audio = ggufName.startsWith("a.") || ggufName.startsWith("mm.a.");
        if (audio && (ggufName.contains("bias") || ggufName.contains("norm")
                || (ggufName.contains("conv") && ggufName.endsWith(".weight")))) {
            return GGMLType.F32;
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };

    private static int rootIntOr(HfConfig cfg, int dflt, String key) {
        Object raw = cfg.raw().raw(key);
        return raw instanceof Number && !(raw instanceof Boolean) ? ((Number) raw).intValue() : dflt;
    }

    private static double rootF64Or(HfConfig cfg, double dflt, String key) {
        Object raw = cfg.raw().raw(key);
        return raw instanceof Number && !(raw instanceof Boolean) ? ((Number) raw).doubleValue() : dflt;
    }

    /** The Muse-Spark audio projector keeps the base storage rules; the vision side is the Muse-Glimmer path. */
    public static final HfForceType PATCH_KERNELS = HfMmprojHooks.PATCH_KERNELS;

}
