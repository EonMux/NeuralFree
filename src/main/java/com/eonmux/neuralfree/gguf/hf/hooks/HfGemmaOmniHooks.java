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
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;

/**
 * The Gemma 3n and Gemma 4 vision-and-audio projector files through the Hugging Face loader, ports of llama.cpp
 * {@code Gemma3nVisionAudioModel} (conversion/gemma.py:368-475), {@code Gemma4VisionAudioModel} (gemma.py:978-1057)
 * and its {@code EmbeddingGemma2VisionAudioModel} / {@code Gemma4UnifiedVisionAudioModel} subclasses
 * (gemma.py:1059-1090): the two tower geometries, the MobileNetV5 block mapping and embedding projection of
 * Gemma 3n, the conformer audio towers, the Gemma 4 patch-embedding repack and the unified tower overrides. Every
 * method is a public static hook reference.
 */
public final class HfGemmaOmniHooks {

    /** The Gemma 3n vision projector ({@code VisionProjectorType.GEMMA3NV}). */
    public static final String GEMMA3NV = "gemma3nv";
    /** The Gemma 3n audio projector ({@code VisionProjectorType.GEMMA3NA}). */
    public static final String GEMMA3NA = "gemma3na";
    /** The Gemma 4 vision projector ({@code VisionProjectorType.GEMMA4V}). */
    public static final String GEMMA4V = "gemma4v";
    /** The Gemma 4 audio projector ({@code VisionProjectorType.GEMMA4A}). */
    public static final String GEMMA4A = "gemma4a";
    /** The Gemma 4 unified vision projector ({@code VisionProjectorType.GEMMA4UV}). */
    public static final String GEMMA4UV = "gemma4uv";
    /** The Gemma 4 unified audio projector ({@code VisionProjectorType.GEMMA4UA}). */
    public static final String GEMMA4UA = "gemma4ua";

    /** The forced image size of the Gemma 4 tower (gemma.py:989). */
    private static final int GEMMA4_IMAGE_SIZE = 224;
    /** The Gemma 3n default tower feed-forward width before the 4x expansion (gemma.py:401). */
    private static final int GEMMA3N_FFN = 2048;
    /** The Gemma 3n default head count (gemma.py:402). */
    private static final int GEMMA3N_HEADS = 8;
    /** The Gemma 3n default audio feed-forward width (gemma.py:416). */
    private static final int GEMMA3N_AUDIO_FFN = 6144;
    /** The vision sequence length of a Gemma 3n image (gemma.py:405). */
    private static final int GEMMA3N_IMAGE_SEQ = 256;
    /** The faked Gemma 3n vision block count the converter writes (gemma.py:399). */
    private static final int GEMMA3N_BLOCKS = 128;

    /** The suffixes Gemma's mapper retries (gemma.py's {@code map_tensor_name} tuple). */
    private static final String[] GEMMA_SUFFIXES = {".weight", ".bias", ".input_max", ".input_min", ".output_max",
            ".output_min"};

    /** The MobileNetV5 block parts the converter maps outside the name tables (gemma.py:375-396). */
    private static final Set<String> MOBILENET_PARTS = mobilenetParts();

    private HfGemmaOmniHooks() {
    }

    private static Set<String> mobilenetParts() {
        Set<String> s = new LinkedHashSet<>(Arrays.asList(
                "conv_exp.weight", "bn1.weight", "conv_pwl.weight", "bn2.weight",
                "dw_start.conv.weight", "dw_start.bn.weight", "dw_mid.conv.weight", "dw_mid.bn.weight",
                "pw_exp.conv.weight", "pw_exp.bn.weight", "pw_proj.conv.weight", "pw_proj.bn.weight",
                "layer_scale.gamma", "attn.query.proj.weight", "attn.key.proj.weight",
                "attn.value.proj.weight", "attn.output.proj.weight", "attn.key.down_conv.weight",
                "attn.key.norm.weight", "attn.value.down_conv.weight", "attn.value.norm.weight", "norm.weight"));
        return Collections.unmodifiableSet(s);
    }

    // ------------------------------------------------------------------ Gemma 3n

    /**
     * {@code Gemma3nVisionAudioModel.filter_tensors}: the converter defines none, so only the inherited projector
     * filter applies; the tensor hook drops everything the file does not carry.
     */
    public static final HfNameFilter GEMMA3N_FILTER = (name, base) -> base.apply(name);

    /** {@code Gemma3nVisionAudioModel.set_gguf_parameters} (gemma.py:418-432). */
    public static void gemma3nMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView pre = HfMmprojHooks.preprocessor(cfg);
        int imageSize = preprocessorImageSize(cfg, pre);
        int blocks = GEMMA3N_BLOCKS;
        int hidden = HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS);
        int ffn = Math.multiplyExact(HfMmprojHooks.visionIntOpt(cfg, "intermediate_size").orElse(GEMMA3N_FFN), 4);
        int heads = HfMmprojHooks.visionIntOpt(cfg, "num_attention_heads").orElse(GEMMA3N_HEADS);
        Object seq = pre.raw("image_seq_length");
        int imageSeq = seq instanceof Number ? ((Number) seq).intValue() : GEMMA3N_IMAGE_SEQ;
        if (imageSeq <= 0) {
            throw new GgufFormatException(cfg.source() + ": 'image_seq_length' is " + imageSeq);
        }
        HfMmprojHooks.visionMeta(ctx, kv, imageSize, Math.max(1, imageSize / imageSeq), hidden, ffn, heads,
                blocks, new float[]{0f, 0f, 0f}, new float[]{1f, 1f, 1f});
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_TYPE, GEMMA3NV);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(1e-6));
        gemma3nAudioMeta(cfg, kv);
    }

    /** The Gemma 3n audio keys (gemma.py:427-432) over the conformer configuration. */
    private static void gemma3nAudioMeta(HfConfig cfg, HfMetadataBuilder kv) {
        JsonView audio = HfAudioTowerHooks.audioConfig(cfg);
        HfMmprojHooks.audioMetaConfig(cfg, kv, HfAudioTowerHooks.intOf(cfg, audio, "hidden_size"),
                HfAudioTowerHooks.intOf(cfg, audio, "intermediate_size") > 0
                        ? HfAudioTowerHooks.intOf(cfg, audio, "intermediate_size") : GEMMA3N_AUDIO_FFN,
                firstInt(cfg, audio, "num_attention_heads", "conf_num_attention_heads"),
                firstInt(cfg, audio, "n_layers", "conf_num_hidden_layers"), "audio_config");
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, GEMMA3NA);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, firstInt(cfg, audio, "input_feat_size", "feat_in"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, 1e-5);
    }

    /** {@code Gemma3nVisionAudioModel.modify_tensors} (gemma.py:434-475). */
    public static void gemma3nTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (isAudioTensor(hfName)) {
            String name = hfName.replace("model.audio_tower.conformer.", "conformer.layers.");
            HfAudioTowerHooks.conformerTensors(ctx, name, src, out, GEMMA_SUFFIXES);
            return;
        }
        if (!hfName.startsWith("model.embed_vision.") && !hfName.startsWith("model.vision_tower.")) {
            out.drop(hfName, "the Gemma 3n vision projector keeps the vision tower and its embedding projection");
            return;
        }
        String mapped;
        if (hfName.startsWith("model.vision_tower.timm_model.blocks.")) {
            mapped = mobilenetBlock(ctx, hfName);
            out.add(mapped, src);
            return;
        }
        mapped = mappedName(ctx, hfName);
        if (mapped.endsWith("conv_stem.conv.bias") || mapped.endsWith("layer_scale.gamma")) {
            src = ElementTransforms.reshape(src, 1, src.shape()[0], 1, 1);
        }
        out.add(mapped, src);
    }

    /** The converter's {@code custom_map} (gemma.py:447-459): a double-indexed MobileNetV5 block tensor. */
    private static String mobilenetBlock(HfLoadContext ctx, String name) {
        String[] parts = name.split("\\.");
        if (parts.length < 7 || !"blocks".equals(parts[3])) {
            throw new GgufFormatException("unknown Gemma 3n MobileNet tensor '" + name + "'");
        }
        String bid = parts[4];
        String sid = parts[5];
        String part = String.join(".", Arrays.copyOfRange(parts, 6, parts.length));
        if (!MOBILENET_PARTS.contains(part)) {
            throw new GgufFormatException("unknown Gemma 3n MobileNet tensor '" + name + "'");
        }
        return "v.blk." + bid + "." + sid + "." + part;
    }

    /** {@code Gemma3nVisionAudioModel.tensor_force_quant} (gemma.py:434-441) over the conformer's audio rule. */
    public static final HfForceType GEMMA3N_FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.contains("input_projection") || ggufName.contains("input_proj")) {
            return GGMLType.F16;
        }
        if (ggufName.contains(".embeddings.") || ggufName.contains("stem")) {
            return GGMLType.F32;
        }
        return HfAudioTowerHooks.CONFORMER_FORCE.force(ggufName, nDims, source, outType);
    };

    // ------------------------------------------------------------------ Gemma 4

    /**
     * {@code Gemma4VisionAudioModel.filter_tensors}: the converter defines none, so only the inherited projector
     * filter applies; the tensor hook drops everything the file does not carry.
     */
    public static final HfNameFilter GEMMA4_FILTER = (name, base) -> base.apply(name);

    /** {@code EmbeddingGemma2VisionAudioModel}: the same towers without the {@code model.} prefix. */
    public static final HfNameFilter EMBEDDING_GEMMA2_FILTER = (name, base) ->
            base.apply(name.startsWith("model.") ? name : "model." + name);

    /** {@code Gemma4VisionAudioModel.set_gguf_parameters} (gemma.py:997-1025). */
    public static void gemma4Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        gemma4VisionMeta(ctx, kv, GEMMA4V);
        if (ctx.config().audio().isPresent()) {
            gemma4AudioMeta(ctx, kv, GEMMA4A);
        }
    }

    /** {@code Gemma4UnifiedVisionAudioModel.set_gguf_parameters} (gemma.py:1067-1090). */
    public static void gemma4UnifiedMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfConfig cfg = ctx.config();
        if (!cfg.vision().isPresent()) {
            throw HfAudioTowerHooks.missingConfig(cfg, "vision_config");
        }
        HfMmprojHooks.visionMetaConfig(ctx, kv, GEMMA4_IMAGE_SIZE,
                HfMmprojHooks.visionInt(cfg, "patch_size"), HfMmprojHooks.visionInt(cfg, "mm_embed_dim"),
                0, 0, 0, "vision_config", null, null);
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_TYPE, GEMMA4UV);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(1e-6));
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        JsonView audio = HfAudioTowerHooks.audioConfig(cfg);
        HfMmprojHooks.audioMetaConfig(cfg, kv, HfAudioTowerHooks.intOf(cfg, audio, "audio_embed_dim"),
                0, 0, 0, "audio_config");
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, GEMMA4UA);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, audioMelBins(cfg, audio));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON,
                layerNormEps(audio, 1e-6));
    }

    /** The Gemma 4 vision keys (gemma.py:1004-1010). */
    private static void gemma4VisionMeta(HfLoadContext ctx, HfMetadataBuilder kv, String projector) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv, GEMMA4_IMAGE_SIZE, HfMmprojHooks.visionInt(cfg, "patch_size"),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.FFN_KEYS),
                HfMmprojHooks.visionInt(cfg, "num_attention_heads"), HfMmprojHooks.visionBlocks(cfg));
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_TYPE, projector);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(1e-6));
        Object act = HfMmprojHooks.visionFind(cfg, "hidden_activation").orElse("gelu_pytorch_tanh");
        if (!"gelu_pytorch_tanh".equals(act)) {
            throw new GgufFormatException(cfg.source() + ": hidden_activation '" + act + "' is unsupported: the"
                    + " Gemma 4 vision tower runs gelu_pytorch_tanh (gemma.py:1009)");
        }
        kv.set(GgufKeys.CLIP_USE_GELU, true);
    }

    /** The Gemma 4 audio keys (gemma.py:1012-1020). */
    private static void gemma4AudioMeta(HfLoadContext ctx, HfMetadataBuilder kv, String projector) {
        HfConfig cfg = ctx.config();
        JsonView audio = HfAudioTowerHooks.audioConfig(cfg);
        int hidden = HfAudioTowerHooks.intOf(cfg, audio, "hidden_size");
        HfMmprojHooks.audioMetaConfig(cfg, kv, hidden, Math.multiplyExact(hidden, 4),
                HfAudioTowerHooks.intOf(cfg, audio, "num_attention_heads"),
                HfMmprojHooks.blocksOf(cfg, audio, "audio_config"), "audio_config");
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, projector);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, audioMelBins(cfg, audio));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, layerNormEps(audio, 1e-6));
    }

    /** {@code Gemma4VisionAudioModel.modify_tensors} (gemma.py:1027-1057). */
    public static void gemma4Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (src.shape().length == 0) {
            src = ElementTransforms.reshape(src, 1);
        }
        if (isAudioTensor(hfName)) {
            String name = hfName.replace("model.audio_tower.", "conformer.").replace(".linear.", ".");
            if (name.endsWith("per_dim_key_scale") || name.endsWith("per_dim_scale")) {
                src = softplus(src);
                name += ".weight";
            }
            if (name.contains("lconv1d.depthwise_conv1d") && name.endsWith(".weight")) {
                long[] shape = src.shape();
                if (shape.length != 3 || shape[1] != 1) {
                    throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source()
                            + " has the shape " + Arrays.toString(shape) + "; a depthwise convolution is [c, 1, k]");
                }
                src = ElementTransforms.reshape(src, shape[0], shape[2]);
            }
            HfAudioTowerHooks.conformerTensors(ctx, name, src, out, GEMMA_SUFFIXES);
            return;
        }
        if (!hfName.startsWith("model.vision_tower.") && !hfName.startsWith("model.vision_embedder.")) {
            out.drop(hfName, "the Gemma 4 projector keeps the vision and audio towers");
            return;
        }
        String name = hfName.replace("model.vision_tower.encoder.", "vision_model.model.")
                .replace(".linear.weight", ".weight");
        if (name.endsWith("layer_scalar") || name.endsWith("position_embedding_table")) {
            name += ".weight";
        }
        if (name.endsWith("patch_embedder.input_proj.weight")) {
            src = patchEmbedding(ctx, hfName, src);
        }
        out.add(HfAudioTowerHooks.mappedName(ctx, name, GEMMA_SUFFIXES), src);
    }

    /** {@code EmbeddingGemma2VisionAudioModel.modify_tensors} (gemma.py:1059-1064); the filter added the prefix. */
    public static void embeddingGemma2Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        gemma4Tensors(ctx, hfName, src, out);
    }

    /**
     * {@code Gemma4UnifiedVisionAudioModel.modify_tensors} (gemma.py:1091-1136): the transformer-less tower's
     * position table is permuted to one leading row, its {@code pos_norm} renamed to {@code patch_ln3}, and the
     * patch projection and its first layer norm reordered from the im2col CHW layout to the weight's HWC columns
     * before the Gemma 4 mapping.
     */
    public static void gemma4UnifiedTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.endsWith("pos_embedding")) {
            long[] shape = src.shape();
            if (shape.length != 3 || shape[0] != 1) {
                throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has the"
                        + " shape " + Arrays.toString(shape) + "; the unified position table is [1, patches,"
                        + " n_embd]");
            }
            src = ElementTransforms.reshape(src, shape[1], 1, shape[2]);
            hfName += ".weight";
        } else if (hfName.contains(".pos_norm.")) {
            hfName = hfName.replace(".pos_norm.", ".patch_ln3.");
        } else if (hfName.endsWith("patch_dense.weight") || hfName.endsWith("patch_ln1.weight")
                || hfName.endsWith("patch_ln1.bias")) {
            src = ElementTransforms.gatherInner(src, channelPermutation(ctx, hfName));
        }
        gemma4Tensors(ctx, hfName, src, out);
    }

    /**
     * The HWC column of every CHW position of a {@code p x p} patch, where {@code p} is
     * {@code model_patch_size} or {@code patch_size * pooling_kernel_size} (gemma.py:1105-1136).
     */
    private static int[] channelPermutation(HfLoadContext ctx, String hfName) {
        OptionalInt model = HfMmprojHooks.visionIntOpt(ctx.config(), "model_patch_size");
        int p = model.isPresent() ? model.getAsInt()
                : Math.multiplyExact(HfMmprojHooks.visionInt(ctx.config(), "patch_size"),
                        HfMmprojHooks.visionInt(ctx.config(), "pooling_kernel_size"));
        int[] perm = new int[Math.multiplyExact(Math.multiplyExact(p, p), 3)];
        for (int i = 0; i < perm.length; i++) {
            int ch = i / (p * p);
            int row = (i % (p * p)) / p;
            int col = i % p;
            perm[i] = row * p * 3 + col * 3 + ch;
        }
        return perm;
    }

    /** {@code Gemma4VisionAudioModel.tensor_force_quant} (gemma.py:1021-1025). */
    public static final HfForceType GEMMA4_FORCE = (ggufName, nDims, source, outType) -> {
        if ((ggufName.startsWith("a.") || ggufName.startsWith("mm.a.")) && ggufName.contains("conv")) {
            return GGMLType.F32;
        }
        if (ggufName.contains("position_embedding_table")) {
            return GGMLType.F32;
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };

    // ------------------------------------------------------------------ shared helpers

    /** {@code ConformerAudioModel.is_audio_tensor} (gemma.py:312-314). */
    private static boolean isAudioTensor(String name) {
        return name.contains("audio") || name.contains("codebook") || name.contains("conformer")
                || name.contains("depth_embedding") || name.contains("depthformer")
                || name.contains("depth_linear");
    }

    /** The mapped GGUF name of {@code hfName}, or a refusal naming it. */
    private static String mappedName(HfLoadContext ctx, String hfName) {
        return HfAudioTowerHooks.mappedName(ctx, hfName, ".weight", ".bias");
    }

    /** The Gemma 4 patch projection reshaped from a linear map to a Conv2d kernel (gemma.py:1047-1053). */
    private static RowSource patchEmbedding(HfLoadContext ctx, String hfName, RowSource src) {
        return HfAudioTowerHooks.patchProjection(ctx, hfName, src);
    }

    /** {@code torch.nn.functional.softplus} (gemma.py:1036). */
    private static RowSource softplus(RowSource src) throws IOException {
        float[] values = HfGenHooks.floats(src, src.describe());
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) Math.log1p(Math.exp(-Math.abs(values[i])))
                    + Math.max(values[i], 0f);
        }
        return RopeTensors.fromFloats(src.shape(), values);
    }

    /** The preprocessor's {@code size.height}, 768 without one (gemma.py:403-407). */
    private static int preprocessorImageSize(HfConfig cfg, JsonView pre) {
        Object size = pre.raw("size");
        if (size instanceof Map) {
            Object height = ((Map<?, ?>) size).get("height");
            if (height instanceof Number) {
                return ((Number) height).intValue();
            }
        }
        return 768;
    }

    /** The first integral member of {@code view} among {@code aliases}. */
    private static int firstInt(HfConfig cfg, JsonView view, String... aliases) {
        for (String alias : aliases) {
            if (view.has(alias)) {
                return HfAudioTowerHooks.intOf(cfg, view, alias);
            }
        }
        throw new GgufFormatException(view.where(aliases[0]) + ": the checkpoint requires one of "
                + Arrays.toString(aliases));
    }

    /** The mel-bin count: {@code input_feat_size}, then {@code feat_in}, then {@code n_mels}. */
    private static int audioMelBins(HfConfig cfg, JsonView audio) {
        return firstInt(cfg, audio, "input_feat_size", "feat_in", "n_mels");
    }

    /** The audio config's {@code layer_norm_eps}, or {@code dflt}. */
    private static double layerNormEps(JsonView audio, double dflt) {
        Object v = audio.raw("layer_norm_eps");
        return v instanceof Number ? ((Number) v).doubleValue() : dflt;
    }
}
