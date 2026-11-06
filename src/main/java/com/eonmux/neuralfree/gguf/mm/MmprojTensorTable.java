package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Tensor requirement table of one projector family: the tensors of an mmproj file the family's forward graph
 * reads, as name-level requirements checked before any weight is loaded, so a file that cannot run fails naming
 * the first missing tensor (the counterpart of the per-graph {@code create_tensor} lists of llama.cpp
 * tools/mtmd/clip-model.h). Shape checks belong to {@link ClipVisionModel} and {@link ClipAudioModel}.
 *
 * <p>Every name may hold one {@code %d}, expanded over the encoder blocks. A {@linkplain Requirement requirement}
 * is a group of alternatives; the file must hold every name of at least one alternative per group (a fused
 * {@code v.blk.%d.attn_qkv.weight} or all of the separate q/k/v weights). Immutable; build with {@link #builder}
 * or a {@link #forFamily} factory.
 */
public final class MmprojTensorTable {

    /** The granite-speech QFormer projector has two blocks (clip.cpp {@code load_tensors}). */
    static final int GRANITE_QFORMER_BLOCKS = 2;

    /**
     * One requirement: alternatives of name sets, of which one must exist whole. A per-layer requirement lets
     * every block pick its own alternative (a dense or a MoE FFN per block); otherwise one alternative must hold
     * for every block.
     */
    public static final class Requirement {
        private final List<List<String>> alternatives;
        private final boolean perLayer;

        private Requirement(List<List<String>> alternatives, boolean perLayer) {
            this.alternatives = alternatives;
            this.perLayer = perLayer;
        }

        /** The alternative name sets, each of which must exist whole (unmodifiable). */
        public List<List<String>> alternatives() {
            return alternatives;
        }

        /** True when each block may satisfy a different alternative. */
        public boolean perLayer() {
            return perLayer;
        }
    }

    private final String encoderGraph;
    private final String projectorGraph;
    private final List<Requirement> required;
    private final List<String> optional;

    private MmprojTensorTable(String encoderGraph, String projectorGraph, List<Requirement> required,
                              List<String> optional) {
        this.encoderGraph = encoderGraph;
        this.projectorGraph = projectorGraph;
        this.required = required;
        this.optional = optional;
    }

    /**
     * Requirement groups of an encoder/projector family, in check order.
     *
     * @throws IllegalArgumentException for an unknown graph name, or a projector graph of another modality than
     *                                  the encoder graph
     */
    public static MmprojTensorTable forFamily(String encoderGraph, String projectorGraph) {
        Objects.requireNonNull(encoderGraph, "encoderGraph");
        Objects.requireNonNull(projectorGraph, "projectorGraph");
        if (ProjectorEntry.ENCODER_QWEN3TTS_GEN.equals(encoderGraph)) {
            if (!ProjectorEntry.PROJECTOR_QWEN3TTS_GEN.equals(projectorGraph)) {
                throw new IllegalArgumentException("the qwen3tts generator runs the "
                        + ProjectorEntry.PROJECTOR_QWEN3TTS_GEN + " graph, not " + projectorGraph);
            }
            return qwen3TtsGen(encoderGraph, projectorGraph);
        }
        if (ProjectorEntry.ENCODER_POCKETTTS_GEN.equals(encoderGraph)) {
            if (!ProjectorEntry.PROJECTOR_POCKETTTS_GEN.equals(projectorGraph)) {
                throw new IllegalArgumentException("the pockettts generator runs the "
                        + ProjectorEntry.PROJECTOR_POCKETTTS_GEN + " graph, not " + projectorGraph);
            }
            return pocketTtsGen(encoderGraph, projectorGraph);
        }
        if (ProjectorEntry.ENCODER_LFM3A_GEN.equals(encoderGraph)) {
            if (!ProjectorEntry.PROJECTOR_LFM3A_GEN.equals(projectorGraph)) {
                throw new IllegalArgumentException("the lfm3a generator runs the "
                        + ProjectorEntry.PROJECTOR_LFM3A_GEN + " graph, not " + projectorGraph);
            }
            return lfm3aGen(encoderGraph, projectorGraph);
        }
        ProjectorModality modality = ProjectorEntry.modalityOf(encoderGraph);
        if (projectorModality(projectorGraph) != modality) {
            throw new IllegalArgumentException("projector graph " + projectorGraph + " does not run on the "
                    + modality + " encoder graph " + encoderGraph);
        }
        Builder b = new Builder(encoderGraph, projectorGraph);
        encoder(b, encoderGraph);
        if (ProjectorEntry.ENCODER_WHISPER.equals(encoderGraph)) {
            // the whisper graph reads separate q/k/v with q and v biases and biased layer norms
            b.requireLayers("a.blk.%d.ln1.weight", "a.blk.%d.ln1.bias", "a.blk.%d.ln2.weight",
                            "a.blk.%d.ln2.bias", "a.blk.%d.attn_q.weight", "a.blk.%d.attn_q.bias",
                            "a.blk.%d.attn_k.weight", "a.blk.%d.attn_v.weight", "a.blk.%d.attn_v.bias",
                            "a.blk.%d.attn_out.weight", "a.blk.%d.ffn_up.weight", "a.blk.%d.ffn_down.weight")
                    .optional("a.blk.%d.attn_out.bias", "a.blk.%d.ffn_up.bias", "a.blk.%d.ffn_down.bias",
                            "a.blk.%d.ffn_gate.weight", "a.blk.%d.ffn_gate.bias");
        } else if (ownsBlockLayout(encoderGraph)) {
            audioBlocks(b, encoderGraph);
        } else if (ProjectorEntry.ENCODER_LFM3VL_VIT.equals(encoderGraph)) {
            lfm3vlBlocks(b);
        } else if (!ProjectorEntry.ENCODER_YASA2_CONVNEXT.equals(encoderGraph)
                && !ProjectorEntry.ENCODER_GEMMA4UV.equals(encoderGraph)
                && !ProjectorEntry.ENCODER_MOBILENETV5.equals(encoderGraph)) {
            // the yasa2 ConvNeXt and the MobileNetV5 conv stack have no transformer blocks; their graphs check
            // their stage tensors, and the gemma4uv embedder has none either
            vitLayerRequirements(b, modality.tensorPrefix(),
                    ProjectorEntry.ENCODER_DOTS_VIT.equals(encoderGraph));
        }
        if (readsVitExtras(encoderGraph)) {
            b.optional("v.blk.%d.ls1.weight", "v.blk.%d.ls2.weight", "v.blk.%d.attn_q_norm.weight",
                    "v.blk.%d.attn_k_norm.weight", "v.blk.%d.attn_post_norm.weight",
                    "v.blk.%d.ffn_post_norm.weight", "v.blk.%d.out_scale.weight");
        } else if (ProjectorEntry.ENCODER_MINICPMV46_VIT.equals(encoderGraph)) {
            b.optional("v.blk.%d.ls1.weight", "v.blk.%d.ls2.weight");
        }
        if (!visionProjector(b, projectorGraph)) {
            projector(b, projectorGraph);
        }
        return b.build();
    }

    /**
     * True for the encoders built on clip.cpp {@code build_vit}, which apply the layer scales, Q/K norms, post
     * norms and block output scale when the file carries them.
     */
    static boolean readsVitExtras(String encoderGraph) {
        switch (encoderGraph) {
            case ProjectorEntry.ENCODER_SIGLIP_VIT:
            case ProjectorEntry.ENCODER_SIGLIP_RESIZED_VIT:
            case ProjectorEntry.ENCODER_MINICPMV_VIT:
            case ProjectorEntry.ENCODER_INTERNVL_VIT:
            case ProjectorEntry.ENCODER_PIXTRAL_VIT:
            case ProjectorEntry.ENCODER_LLAMA4_VIT:
            case ProjectorEntry.ENCODER_PADDLEOCR_VIT:
            case ProjectorEntry.ENCODER_KIMIVL_VIT:
            case ProjectorEntry.ENCODER_KIMIK25_VIT:
            case ProjectorEntry.ENCODER_DOTS_VIT:
            case ProjectorEntry.ENCODER_HUNYUANVL_VIT:
            case ProjectorEntry.ENCODER_GLM4V_VIT:
            case ProjectorEntry.ENCODER_NEMOTRON_VIT:
            case ProjectorEntry.ENCODER_MINIMAX_M3_VIT:
            case ProjectorEntry.ENCODER_MUSE_GLIMMER_VIT:
            case ProjectorEntry.ENCODER_STEP3VL_VIT:
            case ProjectorEntry.ENCODER_DEEPSEEK4V_VIT:
            case ProjectorEntry.ENCODER_DEEPSEEKOCR_SAM_CLIP:
            case ProjectorEntry.ENCODER_DEEPSEEKOCR2_SAM_QWEN2:
            case ProjectorEntry.ENCODER_GEMMA4V_VIT:
                return true;
            default:
                return false;
        }
    }

    /**
     * The encoder-level tensors of the DeepSeek-OCR SAM backbone; its per-block tensors
     * ({@code v.sam.blk.N.*}, counted by {@code clip.vision.sam.block_count}) are checked with the projector.
     */
    private static void samBackbone(Builder b) {
        b.require("v.sam.patch_embd.weight", "v.sam.patch_embd.bias", "v.sam.pos_embd.weight",
                "v.sam.neck.0.weight", "v.sam.neck.1.weight", "v.sam.neck.1.bias", "v.sam.neck.2.weight",
                "v.sam.neck.3.weight", "v.sam.neck.3.bias", "v.sam.net_2.weight", "v.sam.net_3.weight");
    }

    private static void encoder(Builder b, String encoderGraph) {
        switch (encoderGraph) {
            case ProjectorEntry.ENCODER_CLIP_VIT:
                // the llava converters drop the post-norm of the vision tower; the graph applies it when present
                b.optional("v.class_embd", "v.pre_ln.weight", "v.pre_ln.bias", "v.patch_embd.bias",
                                "v.post_ln.weight", "v.post_ln.bias")
                        .require("v.patch_embd.weight", "v.position_embd.weight");
                break;
            case ProjectorEntry.ENCODER_SIGLIP_VIT:
                b.optional("v.class_embd", "v.pre_ln.weight", "v.pre_ln.bias", "v.patch_embd.bias")
                        .require("v.patch_embd.weight", "v.position_embd.weight",
                                "v.post_ln.weight", "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_QWEN2VL_VIT:
                // the qwen encoders carry no learned position rows; their attention rotates with 2D m-rope,
                // and 2.5 normalizes with RMS, so the post-norm bias is optional too
                b.require("v.patch_embd.weight", "v.post_ln.weight")
                        .optional("v.patch_embd.weight.1", "v.position_embd.weight", "v.post_ln.bias",
                                "v.patch_embd.bias");
                break;
            case ProjectorEntry.ENCODER_QWEN3VL_VIT:
                // the learned position rows are required; qwen3vl.cpp also asserts the patch bias, which
                // ling3vl.cpp adds only when present (the projector tables decide)
                b.require("v.patch_embd.weight", "v.position_embd.weight")
                        .optional("v.patch_embd.bias", "v.patch_embd.weight.1", "v.pre_ln.weight", "v.pre_ln.bias",
                                "v.post_ln.weight", "v.post_ln.bias", "v.deepstack.%d.norm.weight", "v.deepstack.%d.norm.bias",
                                "v.deepstack.%d.fc1.weight", "v.deepstack.%d.fc1.bias",
                                "v.deepstack.%d.fc2.weight", "v.deepstack.%d.fc2.bias");
                break;
            case ProjectorEntry.ENCODER_PADDLEOCR_VIT:
            case ProjectorEntry.ENCODER_KIMIVL_VIT:
            case ProjectorEntry.ENCODER_KIMIK25_VIT:
            case ProjectorEntry.ENCODER_HUNYUANVL_VIT:
                b.require("v.patch_embd.weight", "v.position_embd.weight")
                        .optional("v.patch_embd.bias", "v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight",
                                "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_DOTS_VIT:
                // the trunk norm after the blocks is mm.post_norm (clip.cpp load_tensors), RMS like every norm
                b.require("v.patch_embd.weight", "mm.post_norm.weight")
                        .optional("v.patch_embd.bias", "v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.bias",
                                "v.blk.%d.exp_probs_b.weight");
                break;
            case ProjectorEntry.ENCODER_SIGLIP_RESIZED_VIT:
            case ProjectorEntry.ENCODER_MINICPMV_VIT:
            case ProjectorEntry.ENCODER_MINICPMV46_VIT:
                b.require("v.patch_embd.weight", "v.position_embd.weight")
                        .optional("v.patch_embd.bias", "v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight",
                                "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_INTERNVL_VIT:
                b.require("v.patch_embd.weight", "v.class_embd", "v.position_embd.weight")
                        .optional("v.patch_embd.bias", "v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight",
                                "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_PIXTRAL_VIT:
                // RMS-normalized: the pre-norm has no bias; positions rotate, no learned rows
                b.require("v.patch_embd.weight")
                        .optional("v.patch_embd.bias", "v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight",
                                "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_LLAMA4_VIT:
                b.require("v.patch_embd.weight", "v.class_embd", "v.position_embd.weight")
                        .optional("v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight", "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_COGVLM_VIT:
                b.require("v.patch_embd.weight", "v.class_embd", "v.position_embd.weight",
                                "v.blk.%d.attn_qkv.weight", "v.blk.%d.attn_qkv.bias")
                        .optional("v.patch_embd.bias");
                break;
            case ProjectorEntry.ENCODER_GLM4V_VIT:
                // GLM-OCR carries no learned position rows, GLM-5.3-Flash no embedding norm
                b.require("v.patch_embd.weight", "v.patch_embd.weight.1", "v.patch_embd.bias")
                        .optional("v.norm_embd.weight", "v.norm_embd.bias", "v.position_embd.weight",
                                "v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight", "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_YOUTUVL_VIT:
                b.require("v.patch_embd.weight", "v.blk.%d.attn_q.weight", "v.blk.%d.attn_q.bias",
                                "v.blk.%d.attn_k.weight", "v.blk.%d.attn_k.bias", "v.blk.%d.attn_v.weight",
                                "v.blk.%d.attn_v.bias")
                        .optional("v.patch_embd.bias", "v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight",
                                "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_GRANITE4_VIT:
                b.require("v.patch_embd.weight", "v.position_embd.weight", "v.blk.%d.attn_q.weight",
                                "v.blk.%d.attn_k.weight", "v.blk.%d.attn_v.weight")
                        .optional("v.patch_embd.bias");
                break;
            case ProjectorEntry.ENCODER_YASA2_CONVNEXT:
                // the stages are discovered by name and shape-checked by Yasa2Graph; v.backbone_ln is unread
                b.require("v.patch_embd.weight", "v.stage.0.blk.0.dw.weight")
                        .optional("v.patch_embd.bias", "v.patch_ln.weight", "v.patch_ln.bias",
                                "v.backbone_ln.weight", "v.backbone_ln.bias", "v.vision_pos_embed");
                break;
            case ProjectorEntry.ENCODER_NEMOTRON_VIT:
                // the register rows are v.class_embd; Omni's six-channel video kernel is never read for images
                b.require("v.patch_embd.weight", "v.class_embd", "v.position_embd.weight")
                        .optional("v.patch_embd.bias", "v.patch_embd.weight.1", "v.pre_ln.weight", "v.pre_ln.bias",
                                "v.post_ln.weight", "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_MINIMAX_M3_VIT:
                // minimax-m3.cpp asserts both patch kernels and no patch bias; positions rotate
                b.require("v.patch_embd.weight", "v.patch_embd.weight.1")
                        .optional("v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight", "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_MUSE_GLIMMER_VIT:
            case ProjectorEntry.ENCODER_STEP3VL_VIT:
                // learned position rows; step3vl.cpp also asserts the absence of a class token
                b.require("v.patch_embd.weight", "v.position_embd.weight")
                        .optional("v.patch_embd.bias", "v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight",
                                "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_LFM3VL_VIT:
                // the second kernel is the temporal slice of the patchifier; a still image runs both
                b.require("v.patch_embd.weight", "v.embedding_norm.weight").optional("v.patch_embd.weight.1");
                break;
            case ProjectorEntry.ENCODER_DEEPSEEK4V_VIT:
                // RMS-normalized, positions rotate: no learned rows
                b.require("v.patch_embd.weight")
                        .optional("v.patch_embd.bias", "v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight",
                                "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_SENSENOVA_PATCH:
                // sensenova-u1.cpp asserts the patch bias; the embedder has no blocks
                b.require("v.patch_embd.weight", "v.patch_embd.bias");
                break;
            case ProjectorEntry.ENCODER_DEEPSEEKOCR_SAM_CLIP:
                // the CLIP tower reads the SAM rows, not v.patch_embd; its class token and positions are required
                samBackbone(b);
                b.require("v.class_embd", "v.position_embd.weight")
                        .optional("v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight", "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_DEEPSEEKOCR2_SAM_QWEN2:
                // the Qwen2 encoder reads the SAM rows and the learned queries, no patch or position rows
                samBackbone(b);
                b.optional("v.pre_ln.weight", "v.post_ln.weight");
                break;
            case ProjectorEntry.ENCODER_GEMMA4V_VIT:
                // gemma4v.cpp adds no patch bias; the column and row tables share v.position_embd
                b.require("v.patch_embd.weight", "v.position_embd.weight")
                        .optional("v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight", "v.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_MIMOVL_VIT:
                // mimovl.cpp: both bias-free patch kernels, the fused Q/K/V and its bias in every block, the
                // sinks of the windowed blocks (checked against clip.vision.wa_pattern_mode at load)
                b.require("v.patch_embd.weight", "v.patch_embd.weight.1")
                        .requireLayers("v.blk.%d.attn_qkv.weight", "v.blk.%d.attn_qkv.bias")
                        .optional("v.post_ln.weight", "v.post_ln.bias", "v.blk.%d.attn_sinks");
                break;
            case ProjectorEntry.ENCODER_MOBILENETV5:
                // the blocks v.blk.<stage>.<index>.* are discovered stage by stage (MobileNetV5Layout)
                b.require(MobileNetV5Layout.STEM_CONV)
                        .optional(MobileNetV5Layout.STEM_BIAS, MobileNetV5Layout.STEM_NORM,
                                MobileNetV5Layout.MSFA_EXPAND, MobileNetV5Layout.MSFA_EXPAND_NORM,
                                MobileNetV5Layout.MSFA_PROJECT, MobileNetV5Layout.MSFA_PROJECT_NORM,
                                MobileNetV5Layout.MSFA_NORM);
                break;
            case ProjectorEntry.ENCODER_GEMMA4UV:
                b.require("v.patch_embd.weight", "v.patch_embd.bias", "v.position_embd.weight",
                        "v.patch_norm.1.weight", "v.patch_norm.1.bias", "v.patch_norm.2.weight",
                        "v.patch_norm.2.bias", "v.patch_norm.3.weight", "v.patch_norm.3.bias");
                break;
            case ProjectorEntry.ENCODER_WHISPER:
                b.require("a.conv1d.1.weight", "a.conv1d.1.bias", "a.conv1d.2.weight", "a.conv1d.2.bias",
                                "a.position_embd.weight")
                        .optional("a.pre_ln.weight", "a.pre_ln.bias", "a.post_ln.weight", "a.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_QWEN3A:
                b.require("a.conv2d.1.weight", "a.conv2d.1.bias", "a.conv2d.2.weight", "a.conv2d.2.bias",
                                "a.conv2d.3.weight", "a.conv2d.3.bias", "a.conv_out.weight",
                                "a.position_embd.weight")
                        .optional("a.conv_out.bias", "a.pre_ln.weight", "a.pre_ln.bias", "a.post_ln.weight",
                                "a.post_ln.bias");
                break;
            case ProjectorEntry.ENCODER_CONFORMER:
            case ProjectorEntry.ENCODER_LFM3A:
            case ProjectorEntry.ENCODER_GEMMA4A:
            case ProjectorEntry.ENCODER_GEMMA4UA:
            case ProjectorEntry.ENCODER_PARAKEET:
            case ProjectorEntry.ENCODER_GRANITE_SPEECH:
            case ProjectorEntry.ENCODER_VOXTRAL_RT:
            case ProjectorEntry.ENCODER_MIMO_AUDIO:
            case ProjectorEntry.ENCODER_MUSE_SPARK_AUDIO:
            case ProjectorEntry.ENCODER_DOTS3NOTE_A:
            case ProjectorEntry.ENCODER_GEMMA3N_USM:
                // graph-level and block tensors together in audioBlocks
                break;
            default:
                throw new IllegalArgumentException("unknown encoder graph " + encoderGraph);
        }
    }

    /** True for the audio towers whose tensors {@link #audioBlocks} lists (all but whisper and qwen3a). */
    private static boolean ownsBlockLayout(String encoderGraph) {
        return ProjectorEntry.modalityOf(encoderGraph) == ProjectorModality.AUDIO
                && !ProjectorEntry.ENCODER_WHISPER.equals(encoderGraph)
                && !ProjectorEntry.ENCODER_QWEN3A.equals(encoderGraph);
    }

    /**
     * The qwen3tts code predictor and code2wav tensors: what {@code Qwen3TtsClip} reads before it resolves the
     * graphs. The code2wav layer counts are discovered from the tensor table (the reference hardcodes them), so
     * only the first layer and the top-level names are checked here; the graphs validate every shape.
     */
    private static MmprojTensorTable qwen3TtsGen(String encoderGraph, String projectorGraph) {
        Builder b = new Builder(encoderGraph, projectorGraph);
        b.require("a.gen.code.embd.weight", "a.gen.code.head.weight", "a.gen.code.out_embd.weight",
                        "a.gen.code.output_norm.weight")
                .requireLayers("a.gen.code.blk.%d.ln1.weight", "a.gen.code.blk.%d.attn_q.weight",
                        "a.gen.code.blk.%d.attn_q_norm.weight", "a.gen.code.blk.%d.attn_k.weight",
                        "a.gen.code.blk.%d.attn_k_norm.weight", "a.gen.code.blk.%d.attn_v.weight",
                        "a.gen.code.blk.%d.attn_out.weight", "a.gen.code.blk.%d.ln2.weight",
                        "a.gen.code.blk.%d.ffn_gate.weight", "a.gen.code.blk.%d.ffn_up.weight",
                        "a.gen.code.blk.%d.ffn_down.weight")
                .require("a.gen.wav.quant.first.codebook.weight", "a.gen.wav.quant.first.out_proj.weight",
                        "a.gen.wav.quant.rest.codebook.weight", "a.gen.wav.quant.rest.out_proj.weight",
                        "a.gen.wav.pre_conv.weight", "a.gen.wav.pre_conv.bias",
                        "a.gen.wav.tfm.in_proj.weight", "a.gen.wav.tfm.in_proj.bias",
                        "a.gen.wav.tfm.out_proj.weight", "a.gen.wav.tfm.out_proj.bias",
                        "a.gen.wav.tfm.output_norm.weight", "a.gen.wav.tfm.blk.0.ln1.weight",
                        "a.gen.wav.tfm.blk.0.attn_q.weight", "a.gen.wav.tfm.blk.0.attn_k.weight",
                        "a.gen.wav.tfm.blk.0.attn_v.weight", "a.gen.wav.tfm.blk.0.attn_out.weight",
                        "a.gen.wav.tfm.blk.0.ls1.weight", "a.gen.wav.tfm.blk.0.ln2.weight",
                        "a.gen.wav.tfm.blk.0.ffn_gate.weight", "a.gen.wav.tfm.blk.0.ffn_up.weight",
                        "a.gen.wav.tfm.blk.0.ffn_down.weight", "a.gen.wav.tfm.blk.0.ls2.weight",
                        "a.gen.wav.up.blk.0.conv.weight", "a.gen.wav.up.blk.0.conv.bias",
                        "a.gen.wav.up.blk.0.dwconv.weight", "a.gen.wav.up.blk.0.dwconv.bias",
                        "a.gen.wav.up.blk.0.norm.weight", "a.gen.wav.up.blk.0.norm.bias",
                        "a.gen.wav.up.blk.0.pw1.weight", "a.gen.wav.up.blk.0.pw1.bias",
                        "a.gen.wav.up.blk.0.pw2.weight", "a.gen.wav.up.blk.0.pw2.bias",
                        "a.gen.wav.up.blk.0.gamma",
                        "a.gen.wav.dac.entry.weight", "a.gen.wav.dac.entry.bias",
                        "a.gen.wav.dac.blk.0.snake.alpha", "a.gen.wav.dac.blk.0.snake.beta",
                        "a.gen.wav.dac.blk.0.conv.weight", "a.gen.wav.dac.blk.0.conv.bias",
                        "a.gen.wav.dac.blk.0.res.0.act1.alpha", "a.gen.wav.dac.blk.0.res.0.act1.beta",
                        "a.gen.wav.dac.blk.0.res.0.conv1.weight", "a.gen.wav.dac.blk.0.res.0.conv1.bias",
                        "a.gen.wav.dac.blk.0.res.0.act2.alpha", "a.gen.wav.dac.blk.0.res.0.act2.beta",
                        "a.gen.wav.dac.blk.0.res.0.conv2.weight", "a.gen.wav.dac.blk.0.res.0.conv2.bias",
                        "a.gen.wav.dac.post_snake.alpha", "a.gen.wav.dac.post_snake.beta",
                        "a.gen.wav.dac.post_conv.weight", "a.gen.wav.dac.post_conv.bias")
                .optional("a.gen.code.proj_in.weight", "a.gen.code.proj_in.bias");
        return b.build();
    }

    /**
     * The pockettts flow network and mimi decoder tensors: what {@code PocketTtsClip} reads before it resolves the
     * graphs. The mimi transformer layers expand over the file's block count and the three SEANet stages are named
     * explicitly (the reference hardcodes them); the graphs validate every shape.
     */
    private static MmprojTensorTable pocketTtsGen(String encoderGraph, String projectorGraph) {
        Builder b = new Builder(encoderGraph, projectorGraph);
        b.require("a.gen.flow.input_proj.weight", "a.gen.flow.input_proj.bias",
                        "a.gen.flow.cond_embd.weight", "a.gen.flow.cond_embd.bias",
                        "a.gen.flow.final.ada.weight", "a.gen.flow.final.ada.bias",
                        "a.gen.flow.final.proj.weight", "a.gen.flow.final.proj.bias",
                        "a.gen.out_eos.weight", "a.gen.out_eos.bias", "a.gen.input_linear.weight",
                        "a.gen.emb_mean", "a.gen.emb_std", "a.gen.wav.quant_out.weight",
                        "a.gen.wav.upsample.weight", "a.gen.wav.seanet.conv_in.weight",
                        "a.gen.wav.seanet.conv_in.bias", "a.gen.wav.seanet.conv_out.weight",
                        "a.gen.wav.seanet.conv_out.bias",
                        "a.gen.flow.blk.0.norm.weight", "a.gen.flow.blk.0.norm.bias",
                        "a.gen.flow.blk.0.up.weight", "a.gen.flow.blk.0.up.bias",
                        "a.gen.flow.blk.0.down.weight", "a.gen.flow.blk.0.down.bias",
                        "a.gen.flow.blk.0.ada.weight", "a.gen.flow.blk.0.ada.bias")
                .requireLayers("a.gen.wav.tfm.blk.%d.ln1.weight", "a.gen.wav.tfm.blk.%d.ln1.bias",
                        "a.gen.wav.tfm.blk.%d.ln2.weight", "a.gen.wav.tfm.blk.%d.ln2.bias",
                        "a.gen.wav.tfm.blk.%d.attn_q.weight", "a.gen.wav.tfm.blk.%d.attn_k.weight",
                        "a.gen.wav.tfm.blk.%d.attn_v.weight", "a.gen.wav.tfm.blk.%d.attn_out.weight",
                        "a.gen.wav.tfm.blk.%d.ffn_up.weight", "a.gen.wav.tfm.blk.%d.ffn_down.weight",
                        "a.gen.wav.tfm.blk.%d.ls1.weight", "a.gen.wav.tfm.blk.%d.ls2.weight");
        for (int time = 0; time < 2; time++) {
            String t = "a.gen.flow.time." + time + ".";
            b.require(t + "freqs", t + "up.weight", t + "up.bias", t + "down.weight", t + "down.bias",
                    t + "norm");
        }
        for (int stage = 0; stage < 3; stage++) {
            String s = "a.gen.wav.seanet.blk." + stage + ".";
            b.require(s + "scale_conv.weight", s + "scale_conv.bias", s + "res_conv1.weight",
                    s + "res_conv1.bias", s + "res_conv2.weight", s + "res_conv2.bias");
        }
        return b.build();
    }

    /**
     * The lfm3a MaskGIT depthformer and liquid-codec decoder tensors: what {@code Lfm3aGenClip} reads before it
     * resolves the graphs. The codec decoder's layer count and mix of conv and attention blocks are discovered
     * from the tensor table (the reference hardcodes the liquid-codec shape, which the converter checks), so every
     * layer is checked for one of the two block kinds; the graphs validate every shape. A depthformer block runs
     * attention only; a codec block carries the FFN and both sandwich norms, and its attention blocks the per-head
     * sigmoid gate.
     */
    private static MmprojTensorTable lfm3aGen(String encoderGraph, String projectorGraph) {
        Builder b = new Builder(encoderGraph, projectorGraph);
        b.require("a.gen.code.embd.weight", "a.gen.code.out_embd.weight", "a.gen.code.proj_in.weight",
                        "a.gen.code.proj_in.bias", "a.gen.code.mask_embd",
                        "a.gen.code.output_norm.weight", "a.gen.code.head_norm.weight")
                .requireLayers("a.gen.code.blk.%d.ln1.weight", "a.gen.code.blk.%d.attn_q.weight",
                        "a.gen.code.blk.%d.attn_q_norm.weight", "a.gen.code.blk.%d.attn_k.weight",
                        "a.gen.code.blk.%d.attn_k_norm.weight", "a.gen.code.blk.%d.attn_v.weight",
                        "a.gen.code.blk.%d.attn_out.weight", "a.gen.code.blk.%d.attn_gate.weight",
                        "a.gen.code.blk.%d.ln2.weight", "a.gen.code.blk.%d.ffn_gate.weight",
                        "a.gen.code.blk.%d.ffn_up.weight", "a.gen.code.blk.%d.ffn_down.weight")
                .optional("a.gen.code.blk.%d.attn_post_norm.weight", "a.gen.code.blk.%d.ffn_post_norm.weight")
                .require("a.gen.wav.quant.first.codebook.weight", "a.gen.wav.quant.first.out_proj.weight",
                        "a.gen.wav.quant.first.out_proj.bias", "a.gen.wav.tfm.in_proj.weight",
                        "a.gen.wav.tfm.in_proj.bias", "a.gen.wav.tfm.out_proj.weight", "a.gen.wav.tfm.out_proj.bias",
                        "a.gen.wav.tfm.output_norm.weight")
                .requireLayers("a.gen.wav.tfm.blk.%d.ln1.weight", "a.gen.wav.tfm.blk.%d.ln2.weight",
                        "a.gen.wav.tfm.blk.%d.attn_post_norm.weight", "a.gen.wav.tfm.blk.%d.ffn_post_norm.weight",
                        "a.gen.wav.tfm.blk.%d.ffn_gate.weight", "a.gen.wav.tfm.blk.%d.ffn_up.weight",
                        "a.gen.wav.tfm.blk.%d.ffn_down.weight")
                .requireEachLayerEither(
                        new String[]{"a.gen.wav.tfm.blk.%d.conv_pw1.weight", "a.gen.wav.tfm.blk.%d.conv_dw.weight",
                                "a.gen.wav.tfm.blk.%d.conv_pw2.weight"},
                        new String[]{"a.gen.wav.tfm.blk.%d.attn_q.weight", "a.gen.wav.tfm.blk.%d.attn_q_norm.weight",
                                "a.gen.wav.tfm.blk.%d.attn_k.weight", "a.gen.wav.tfm.blk.%d.attn_k_norm.weight",
                                "a.gen.wav.tfm.blk.%d.attn_v.weight", "a.gen.wav.tfm.blk.%d.attn_out.weight",
                                "a.gen.wav.tfm.blk.%d.attn_gate.weight"});
        return b.build();
    }

    /**
     * The encoder tensors of the audio towers with their own layout: the {@code get_tensor} lists of clip.cpp
     * {@code load_tensors} for the projector types, the blocks of its generic layer loop (an output projection
     * and an FFN per block, lfm3a's conv blocks excepted), and the tensors each graph reads unconditionally.
     */
    private static void audioBlocks(Builder b, String encoderGraph) {
        String p = "a.blk.%d.";
        switch (encoderGraph) {
            case ProjectorEntry.ENCODER_CONFORMER:
                // conformer.cpp adds the q/k/v/out, conv and second-FFN biases unconditionally
                conformerFront(b);
                conformerBlock(b);
                relPosBlock(b);
                b.requireLayers(p + "attn_q.bias", p + "attn_k.bias", p + "attn_v.bias", p + "attn_out.bias",
                                p + "conv_dw.bias", p + "conv_pw1.bias", p + "conv_pw2.bias", p + "ffn_up_1.bias",
                                p + "ffn_down_1.bias")
                        .optional("a.position_embd.weight", p + "ln1.bias", p + "ln2.bias", p + "ffn_up.bias",
                                p + "ffn_down.bias");
                break;
            case ProjectorEntry.ENCODER_LFM3A:
                conformerFront(b);
                b.require("a.post_ln.weight")
                        .requireLayers(p + "ffn_up.weight", p + "ffn_down.weight")
                        .optional(p + "ln1.weight", p + "ln2.weight", p + "ffn_gate.weight")
                        .requireEachLayerEither(
                                new String[]{p + "conv_pw1.weight", p + "conv_dw.weight", p + "conv_pw2.weight"},
                                new String[]{p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight",
                                        p + "attn_out.weight", p + "attn_q_norm.weight", p + "attn_k_norm.weight"});
                break;
            case ProjectorEntry.ENCODER_GEMMA4A:
                // gemma4a.cpp runs each optional operator (attention, conv module, norms) when present
                b.require("a.conv1d.0.weight", "a.conv1d.1.weight", "a.input_projection.weight")
                        .requireLayers(p + "attn_out.weight", p + "ffn_norm.weight", p + "ffn_up.weight",
                                p + "ffn_down.weight", p + "conv_pw1.weight", p + "conv_dw.weight",
                                p + "conv_pw2.weight", p + "ffn_norm_1.weight", p + "ffn_up_1.weight",
                                p + "ffn_down_1.weight")
                        .optional("a.conv1d.0.bias", "a.conv1d.1.bias", "a.conv1d.0.norm.weight",
                                "a.conv1d.1.norm.weight", "a.input_projection.bias", "a.pre_encode.out.weight",
                                "a.pre_encode.out.bias", p + "attn_q.weight", p + "attn_k.weight",
                                p + "attn_v.weight", p + "attn_out.bias", p + "attn_pre_norm.weight", p + "ln1.weight",
                                p + "per_dim_scale.weight", p + "per_dim_k_scale.weight", p + "attn_k_rel.weight",
                                p + "attn_post_norm.weight", p + "ffn_post_norm.weight", p + "conv_norm.weight",
                                p + "conv_dw.bias", p + "norm_conv.weight", p + "ffn_post_norm_1.weight",
                                p + "ln2.weight");
                break;
            case ProjectorEntry.ENCODER_GEMMA4UA:
                break;
            case ProjectorEntry.ENCODER_PARAKEET:
                // parakeet.cpp reads no projection bias; its block norms add their biases unconditionally
                b.require("a.mel_filters", "a.window");
                conformerFront(b);
                conformerBlock(b);
                relPosBlock(b);
                b.requireLayers(p + "ln1.bias", p + "ln2.bias", p + "conv_norm_mean", p + "conv_norm_var");
                break;
            case ProjectorEntry.ENCODER_GRANITE_SPEECH:
                b.require("a.input_projection.weight", "a.input_projection.bias", "a.enc_ctc_out.weight",
                                "a.enc_ctc_out.bias", "a.enc_ctc_out_mid.weight", "a.enc_ctc_out_mid.bias");
                conformerBlock(b);
                // granite-speech.cpp projects Q, K and V without biases; the block norms take optional biases
                b.requireLayers(p + "attn_rel_pos_emb", p + "attn_out.bias", p + "conv_pw1.bias",
                                p + "conv_pw2.bias", p + "ffn_up.bias", p + "ffn_down.bias", p + "ffn_up_1.bias",
                                p + "ffn_down_1.bias")
                        .optional(p + "ln1.bias", p + "ln2.bias");
                break;
            case ProjectorEntry.ENCODER_VOXTRAL_RT:
                // voxtral-rt.cpp adds the Q and V biases unconditionally and has no key bias
                b.require("a.conv1d.1.weight", "a.conv1d.1.bias", "a.conv1d.2.weight", "a.conv1d.2.bias",
                                "a.post_ln.weight")
                        .requireLayers(p + "ln1.weight", p + "ln2.weight", p + "attn_q.weight", p + "attn_q.bias",
                                p + "attn_k.weight", p + "attn_v.weight", p + "attn_v.bias", p + "attn_out.weight",
                                p + "ffn_up.weight", p + "ffn_down.weight")
                        .optional(p + "attn_out.bias", p + "ffn_gate.weight", p + "ffn_down.bias");
                break;
            case ProjectorEntry.ENCODER_MIMO_AUDIO:
                b.require("a.conv1d.1.weight", "a.conv1d.1.bias", "a.conv1d.2.weight", "a.conv1d.2.bias",
                        "a.downsample.conv.weight", "a.downsample.norm.weight", "a.downsample.norm.bias",
                        "a.rvq.codebook.weight");
                vitLayerRequirements(b, "a", false);
                break;
            case ProjectorEntry.ENCODER_MUSE_SPARK_AUDIO:
                // the stage tensors a.patch.N are indexed by stage, not block; stage 0 always exists
                b.require("a.patch.0.weight", "a.patch.0.bias", "a.patch.0.norm.weight", "a.pre_ln.weight",
                                "a.post_ln.weight")
                        .requireLayers("a.conv1d.%d.weight", "a.conv1d.%d.bias", "a.conv1d.%d.norm.weight",
                                p + "ffn_up.weight", p + "ffn_up.bias", p + "ffn_down.weight", p + "ffn_down.bias");
                break;
            case ProjectorEntry.ENCODER_DOTS3NOTE_A:
                b.require("a.conv2d.1.weight", "a.conv2d.1.bias", "a.conv2d.2.weight", "a.conv2d.2.bias",
                        "a.conv2d.3.weight", "a.conv2d.3.bias", "a.conv_out.weight");
                vitLayerRequirements(b, "a", false);
                break;
            case ProjectorEntry.ENCODER_GEMMA3N_USM:
                gemma3nBlocks(b);
                break;
            default:
                throw new IllegalArgumentException("unknown audio encoder graph " + encoderGraph);
        }
    }

    /** The NeMo subsampling stem: conv1d 0, 2, 3, 5, 6 and the output projection. */
    private static void conformerFront(Builder b) {
        b.require("a.conv1d.0.weight", "a.conv1d.0.bias", "a.conv1d.2.weight", "a.conv1d.2.bias",
                "a.conv1d.3.weight", "a.conv1d.3.bias", "a.conv1d.5.weight", "a.conv1d.5.bias",
                "a.conv1d.6.weight", "a.conv1d.6.bias", "a.pre_encode.out.weight", "a.pre_encode.out.bias");
    }

    /**
     * The conformer block tensors the lfm2a, parakeet and granite-speech loaders share: the two half-step FFNs and
     * their norms, the attention, the conv module and the block norm weights.
     */
    private static void conformerBlock(Builder b) {
        String p = "a.blk.%d.";
        b.requireLayers(p + "ln1.weight", p + "ln2.weight", p + "attn_q.weight", p + "attn_k.weight",
                p + "attn_v.weight", p + "attn_out.weight", p + "ffn_norm.weight", p + "ffn_norm.bias", p + "ffn_up.weight", p + "ffn_down.weight",
                p + "ffn_norm_1.weight", p + "ffn_norm_1.bias", p + "ffn_up_1.weight", p + "ffn_down_1.weight",
                p + "norm_conv.weight", p + "norm_conv.bias", p + "conv_norm.weight", p + "conv_norm.bias",
                p + "conv_pw1.weight", p + "conv_dw.weight", p + "conv_pw2.weight");
    }

    /** The relative-position attention inputs of the lfm2a and parakeet blocks. */
    private static void relPosBlock(Builder b) {
        String p = "a.blk.%d.";
        b.requireLayers(p + "pos_bias_u", p + "pos_bias_v", p + "linear_pos.weight");
    }

    /**
     * The LFM3-VL blocks (models/lfm3vl.cpp): RMS norms and the FFN in every block, and per block either the conv
     * mixer (in/out projections, the optional depthwise spatial conv) or the attention (separate q/k/v, the output
     * projection, the optional Q/K norms), never both (clip.cpp {@code load_tensors}).
     */
    private static void lfm3vlBlocks(Builder b) {
        String p = "v.blk.%d.";
        b.requireLayers(p + "ln1.weight", p + "ln2.weight", p + "ffn_up.weight", p + "ffn_down.weight")
                .optional(p + "ln1.bias", p + "ln2.bias", p + "ffn_up.bias", p + "ffn_down.bias",
                        p + "ffn_gate.weight", p + "ffn_gate.bias", p + "conv2d_spatial.weight", p + "attn_out.bias",
                        p + "attn_q_norm.weight", p + "attn_k_norm.weight")
                .requireEachLayerEither(new String[]{p + "conv2d_in_proj.weight", p + "conv2d_out_proj.weight"},
                        new String[]{p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight",
                                p + "attn_out.weight"});
    }

    /**
     * The block tensors of the shared pre-norm ViT tower ({@code build_vit}) under a tensor prefix; with
     * {@code moe} each block holds either the dense FFN or the routed experts.
     */
    private static void vitLayerRequirements(Builder b, String prefix, boolean moe) {
        String p = prefix + ".blk.%d.";
        if (moe) {
            b.requireLayers(p + "ln1.weight", p + "ln2.weight", p + "attn_out.weight")
                    .requireEachLayerEither(new String[]{p + "ffn_up.weight", p + "ffn_down.weight"},
                            new String[]{p + "ffn_gate_inp.weight", p + "ffn_gate_exps.weight",
                                    p + "ffn_up_exps.weight", p + "ffn_down_exps.weight"});
        } else {
            b.requireLayers(p + "ln1.weight", p + "ln2.weight", p + "attn_out.weight", p + "ffn_up.weight",
                    p + "ffn_down.weight");
        }
        b.optional(p + "ln1.bias", p + "ln2.bias", p + "attn_out.bias", p + "ffn_up.bias",
                        p + "ffn_down.bias", p + "ffn_gate.weight", p + "ffn_gate.bias", p + "attn_qkv.bias",
                        p + "attn_q.bias", p + "attn_k.bias", p + "attn_v.bias")
                .requireEitherLayer(new String[]{p + "attn_qkv.weight"},
                        new String[]{p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"});
    }

    /** The sub-sample stem and the conformer block tensors of the Gemma 3n audio encoder (Gemma3nAudioModel). */
    private static void gemma3nBlocks(Builder b) {
        String p = "a.blk.%d.";
        b.require("a.conv1d.0.weight", "a.conv1d.0.norm.weight", "a.conv1d.1.weight", "a.conv1d.1.norm.weight",
                "a.pre_encode.out.weight");
        b.requireLayers(p + "ffn_norm.weight", p + "ffn_up.weight", p + "ffn_down.weight",
                p + "ffn_post_norm.weight", p + "ln1.weight", p + "attn_q.weight", p + "attn_k.weight",
                p + "attn_v.weight", p + "per_dim_scale", p + "linear_pos.weight", p + "attn_out.weight",
                p + "ln2.weight", p + "conv_norm.weight", p + "conv_pw1.weight", p + "conv_dw.weight",
                p + "norm_conv.weight", p + "conv_pw2.weight", p + "ffn_norm_1.weight", p + "ffn_up_1.weight",
                p + "ffn_down_1.weight", p + "ffn_post_norm_1.weight", p + "layer_pre_norm.weight");
    }

    /** The modality a projector graph's input comes from. */
    private static ProjectorModality projectorModality(String projectorGraph) {
        switch (projectorGraph) {
            case ProjectorEntry.PROJECTOR_QWEN3TTS_GEN:
            case ProjectorEntry.PROJECTOR_POCKETTTS_GEN:
            case ProjectorEntry.PROJECTOR_LFM3A_GEN:
                return ProjectorModality.GEN_AUDIO;
            case ProjectorEntry.PROJECTOR_LLAVA_MLP:
            case ProjectorEntry.PROJECTOR_LLAVA_MLP_NORM:
            case ProjectorEntry.PROJECTOR_QWEN2VL_MERGER:
            case ProjectorEntry.PROJECTOR_GEMMA3:
            case ProjectorEntry.PROJECTOR_MINICPMV_RESAMPLER:
            case ProjectorEntry.PROJECTOR_INTERNVL:
            case ProjectorEntry.PROJECTOR_PIXTRAL:
            case ProjectorEntry.PROJECTOR_IDEFICS3:
            case ProjectorEntry.PROJECTOR_LLAMA4:
            case ProjectorEntry.PROJECTOR_PHI4:
            case ProjectorEntry.PROJECTOR_QWEN3VL_MERGER:
            case ProjectorEntry.PROJECTOR_PADDLEOCR:
            case ProjectorEntry.PROJECTOR_KIMIVL:
            case ProjectorEntry.PROJECTOR_KIMIK25:
            case ProjectorEntry.PROJECTOR_DOTS:
            case ProjectorEntry.PROJECTOR_HUNYUANVL:
            case ProjectorEntry.PROJECTOR_JANUS:
            case ProjectorEntry.PROJECTOR_LFM2:
            case ProjectorEntry.PROJECTOR_COHERE2V:
            case ProjectorEntry.PROJECTOR_LIGHTONOCR:
            case ProjectorEntry.PROJECTOR_LDP:
            case ProjectorEntry.PROJECTOR_LDPV2:
            case ProjectorEntry.PROJECTOR_GLM_EDGE:
            case ProjectorEntry.PROJECTOR_COGVLM:
            case ProjectorEntry.PROJECTOR_GLM4V:
            case ProjectorEntry.PROJECTOR_YOUTUVL:
            case ProjectorEntry.PROJECTOR_GRANITE4:
            case ProjectorEntry.PROJECTOR_YASA2:
            case ProjectorEntry.PROJECTOR_NEMOTRON:
            case ProjectorEntry.PROJECTOR_LING3VL:
            case ProjectorEntry.PROJECTOR_MINIMAX_M3:
            case ProjectorEntry.PROJECTOR_MUSE_GLIMMER:
            case ProjectorEntry.PROJECTOR_LFM3VL:
            case ProjectorEntry.PROJECTOR_SENSENOVA:
            case ProjectorEntry.PROJECTOR_STEP3VL:
            case ProjectorEntry.PROJECTOR_DEEPSEEK4V:
            case ProjectorEntry.PROJECTOR_DEEPSEEK41V:
            case ProjectorEntry.PROJECTOR_DEEPSEEKOCR:
            case ProjectorEntry.PROJECTOR_DEEPSEEKOCR2:
            case ProjectorEntry.PROJECTOR_GEMMA4V:
            case ProjectorEntry.PROJECTOR_GEMMA4UV:
            case ProjectorEntry.PROJECTOR_MINICPMV46_MERGER:
            case ProjectorEntry.PROJECTOR_MIMOVL_MERGER:
            case ProjectorEntry.PROJECTOR_GEMMA3NV:
                return ProjectorModality.VISION;
            case ProjectorEntry.PROJECTOR_ULTRAVOX:
            case ProjectorEntry.PROJECTOR_QWEN2A:
            case ProjectorEntry.PROJECTOR_VOXTRAL:
            case ProjectorEntry.PROJECTOR_MUSIC_FLAMINGO:
            case ProjectorEntry.PROJECTOR_MERALION:
            case ProjectorEntry.PROJECTOR_GLMA:
            case ProjectorEntry.PROJECTOR_QWEN3A:
            case ProjectorEntry.PROJECTOR_CONFORMER:
            case ProjectorEntry.PROJECTOR_LFM3A:
            case ProjectorEntry.PROJECTOR_GEMMA4A:
            case ProjectorEntry.PROJECTOR_GEMMA4UA:
            case ProjectorEntry.PROJECTOR_PARAKEET:
            case ProjectorEntry.PROJECTOR_GRANITE_SPEECH:
            case ProjectorEntry.PROJECTOR_VOXTRAL_RT:
            case ProjectorEntry.PROJECTOR_MIMO_AUDIO:
            case ProjectorEntry.PROJECTOR_MUSE_SPARK_AUDIO:
            case ProjectorEntry.PROJECTOR_DOTS3NOTE_A:
            case ProjectorEntry.PROJECTOR_GEMMA3NA:
                return ProjectorModality.AUDIO;
            default:
                throw new IllegalArgumentException("unknown projector graph " + projectorGraph);
        }
    }

    private static void projector(Builder b, String projectorGraph) {
        switch (projectorGraph) {
            case ProjectorEntry.PROJECTOR_LLAVA_MLP:
                b.require("mm.0.weight", "mm.0.bias").optional("mm.2.weight", "mm.2.bias");
                break;
            case ProjectorEntry.PROJECTOR_LLAVA_MLP_NORM:
                b.require("mm.0.weight", "mm.0.bias", "mm.1.weight", "mm.1.bias",
                        "mm.3.weight", "mm.3.bias", "mm.4.weight", "mm.4.bias");
                break;
            case ProjectorEntry.PROJECTOR_QWEN2VL_MERGER:
                // merger.mlp.0 and merger.mlp.2 of the checkpoint (clip.cpp load_tensors)
                b.require("mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias");
                break;
            case ProjectorEntry.PROJECTOR_QWEN3VL_MERGER:
                // qwen3vl.cpp asserts the patch bias
                b.require("v.patch_embd.bias", "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias");
                break;
            case ProjectorEntry.PROJECTOR_PADDLEOCR:
            case ProjectorEntry.PROJECTOR_KIMIVL:
            case ProjectorEntry.PROJECTOR_KIMIK25:
                b.require("mm.input_norm.weight", "mm.input_norm.bias", "mm.1.weight", "mm.1.bias", "mm.2.weight",
                        "mm.2.bias");
                break;
            case ProjectorEntry.PROJECTOR_DOTS:
                b.require("mm.input_norm.weight", "mm.input_norm.bias", "mm.0.weight", "mm.0.bias", "mm.2.weight",
                        "mm.2.bias");
                break;
            case ProjectorEntry.PROJECTOR_HUNYUANVL:
                b.require("mm.pre_norm.weight", "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias",
                        "v.image_newline", "mm.model.fc.weight", "mm.model.fc.bias", "mm.image_begin",
                        "mm.image_end", "mm.post_norm.weight")
                        .optional("v.view_seperator");
                break;
            case ProjectorEntry.PROJECTOR_JANUS:
                b.require("mm.0.weight", "mm.0.bias", "mm.1.weight", "mm.1.bias");
                break;
            case ProjectorEntry.PROJECTOR_LFM2:
                b.require("mm.1.weight", "mm.1.bias", "mm.2.weight", "mm.2.bias")
                        .optional("mm.input_norm.weight", "mm.input_norm.bias");
                break;
            case ProjectorEntry.PROJECTOR_COHERE2V:
                b.require("mm.1.weight", "mm.1.bias", "mm.2.weight", "mm.2.bias");
                break;
            case ProjectorEntry.PROJECTOR_LIGHTONOCR:
                b.require("mm.1.weight", "mm.2.weight")
                        .optional("mm.1.bias", "mm.2.bias", "mm.input_norm.weight", "mm.patch_merger.weight");
                break;
            case ProjectorEntry.PROJECTOR_GEMMA3:
                b.require("mm.input_projection.weight", "mm.soft_emb_norm.weight");
                break;
            case ProjectorEntry.PROJECTOR_MINICPMV_RESAMPLER:
                // pos_embed_k is required by the reference loader although its graph computes the key
                // positions from sin-cos tables
                b.require("resampler.pos_embed_k", "resampler.query", "resampler.proj.weight",
                        "resampler.kv.weight",
                        "resampler.attn.q.weight", "resampler.attn.q.bias",
                        "resampler.attn.k.weight", "resampler.attn.k.bias",
                        "resampler.attn.v.weight", "resampler.attn.v.bias",
                        "resampler.attn.out.weight", "resampler.attn.out.bias",
                        "resampler.ln_q.weight", "resampler.ln_q.bias",
                        "resampler.ln_kv.weight", "resampler.ln_kv.bias",
                        "resampler.ln_post.weight", "resampler.ln_post.bias");
                break;
            case ProjectorEntry.PROJECTOR_INTERNVL:
                b.require("mm.model.mlp.0.weight", "mm.model.mlp.0.bias", "mm.model.mlp.1.weight",
                        "mm.model.mlp.1.bias", "mm.model.mlp.3.weight", "mm.model.mlp.3.bias");
                break;
            case ProjectorEntry.PROJECTOR_PIXTRAL:
                b.require("mm.1.weight", "mm.2.weight", "v.token_embd.img_break")
                        .optional("mm.1.bias", "mm.2.bias", "mm.input_norm.weight", "mm.patch_merger.weight");
                break;
            case ProjectorEntry.PROJECTOR_IDEFICS3:
                b.require("mm.model.fc.weight");
                break;
            case ProjectorEntry.PROJECTOR_LLAMA4:
                b.require("mm.model.fc.weight", "mm.model.mlp.1.weight", "mm.model.mlp.2.weight");
                break;
            case ProjectorEntry.PROJECTOR_PHI4:
                b.require("mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias");
                break;
            case ProjectorEntry.PROJECTOR_YASA2:
                b.require("mm.0.weight", "mm.2.weight").optional("mm.0.bias", "mm.2.bias");
                break;
            case ProjectorEntry.PROJECTOR_ULTRAVOX:
                b.require("mm.a.mlp.1.weight", "mm.a.mlp.2.weight", "mm.a.norm_pre.weight", "mm.a.norm_mid.weight");
                break;
            case ProjectorEntry.PROJECTOR_QWEN2A:
                b.require("mm.a.fc.weight", "mm.a.fc.bias");
                break;
            case ProjectorEntry.PROJECTOR_VOXTRAL:
                b.require("mm.a.mlp.1.weight", "mm.a.mlp.2.weight");
                break;
            case ProjectorEntry.PROJECTOR_MUSIC_FLAMINGO:
            case ProjectorEntry.PROJECTOR_QWEN3A:
                b.require("mm.a.mlp.1.weight", "mm.a.mlp.1.bias", "mm.a.mlp.2.weight", "mm.a.mlp.2.bias");
                break;
            case ProjectorEntry.PROJECTOR_MERALION:
                b.require("mm.a.norm_pre.weight", "mm.a.norm_pre.bias", "mm.a.mlp.0.weight", "mm.a.mlp.0.bias",
                        "mm.a.mlp.1.weight", "mm.a.mlp.1.bias", "mm.a.mlp.2.weight", "mm.a.mlp.2.bias",
                        "mm.a.mlp.3.weight", "mm.a.mlp.3.bias");
                break;
            case ProjectorEntry.PROJECTOR_GEMMA3NA:
                b.require("mm.a.embedding.weight", "mm.a.hard_emb_norm.weight", "mm.a.soft_emb_norm.weight",
                        "mm.a.input_projection.weight");
                break;
            case ProjectorEntry.PROJECTOR_GLMA:
                b.require("mm.a.norm_pre.weight", "mm.a.norm_pre.bias", "mm.a.mlp.1.weight", "mm.a.mlp.1.bias",
                        "mm.a.mlp.2.weight", "mm.a.mlp.2.bias", "v.boi", "v.eoi");
                break;
            case ProjectorEntry.PROJECTOR_CONFORMER:
                // d1omni_a's residual adapter mlp.4..6 is checked by ConformerTower for that projector type
                b.require("mm.a.mlp.0.weight", "mm.a.mlp.0.bias", "mm.a.mlp.1.weight", "mm.a.mlp.1.bias",
                                "mm.a.mlp.3.weight", "mm.a.mlp.3.bias")
                        .optional("mm.a.mlp.4.weight", "mm.a.mlp.4.bias", "mm.a.mlp.5.weight", "mm.a.mlp.5.bias",
                                "mm.a.mlp.6.weight", "mm.a.mlp.6.bias");
                break;
            case ProjectorEntry.PROJECTOR_LFM3A:
                b.require("mm.a.mlp.0.weight", "mm.a.mlp.0.bias", "mm.a.mlp.1.weight", "mm.a.mlp.1.bias",
                        "mm.a.mlp.3.weight", "mm.a.mlp.3.bias");
                break;
            case ProjectorEntry.PROJECTOR_GEMMA4A:
                // clip.cpp reads the output width from mm.a.input_projection
                b.require("mm.a.input_projection.weight").optional("mm.a.soft_emb_norm.weight");
                break;
            case ProjectorEntry.PROJECTOR_GEMMA4UA:
                b.require("mm.a.input_projection.weight");
                break;
            case ProjectorEntry.PROJECTOR_PARAKEET:
                // parakeet.cpp: RMS norm, then the bias-free ReLU-squared MLP
                b.require("mm.a.norm_pre.weight", "mm.a.mlp.1.weight", "mm.a.mlp.2.weight");
                break;
            case ProjectorEntry.PROJECTOR_GRANITE_SPEECH:
                b.require("a.proj_query", "a.proj_norm.weight", "a.proj_norm.bias", "a.proj_linear.weight",
                        "a.proj_linear.bias");
                for (int il = 0; il < GRANITE_QFORMER_BLOCKS; il++) {
                    for (String t : new String[]{"self_attn_q", "self_attn_k", "self_attn_v", "self_attn_out",
                            "self_attn_norm", "cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_out",
                            "cross_attn_norm", "ffn_up", "ffn_down", "ffn_norm"}) {
                        b.require("a.proj_blk." + il + "." + t + ".weight", "a.proj_blk." + il + "." + t + ".bias");
                    }
                }
                break;
            case ProjectorEntry.PROJECTOR_VOXTRAL_RT:
                b.require("mm.a.mlp.1.weight", "mm.a.mlp.2.weight");
                break;
            case ProjectorEntry.PROJECTOR_MIMO_AUDIO:
                // the local blocks mm.a.local_blk.N run over clip.audio.local_block_count, checked by MimoAudioTower
                b.require("mm.a.code_embd.weight", "mm.a.local_norm.weight", "mm.a.mlp.1.weight",
                        "mm.a.mlp.2.weight");
                break;
            case ProjectorEntry.PROJECTOR_MUSE_SPARK_AUDIO:
                b.require("mm.a.fc.weight", "mm.a.fc.bias");
                break;
            case ProjectorEntry.PROJECTOR_DOTS3NOTE_A:
                b.require("mm.a.norm_pre.weight", "mm.a.norm_pre.bias", "mm.a.mlp.1.weight", "mm.a.mlp.1.bias",
                        "mm.a.mlp.3.weight", "mm.a.mlp.3.bias");
                break;
            default:
                throw new IllegalArgumentException("unknown projector graph " + projectorGraph);
        }
    }

    /**
     * The projectors of the MobileVLM, GLM-Edge, CogVLM, GLM-4V, Youtu-VL, Granite Vision 4.1, Nemotron, Ling3-VL,
     * MiniMax-M3, Muse Glimmer, LFM3-VL, SenseNova-U1, Step3-VL, DeepSeek-V4 / V4.1, DeepSeek-OCR / OCR 2 and
     * Gemma 4 families.
     */
    private static boolean visionProjector(Builder b, String projectorGraph) {
        switch (projectorGraph) {
            case ProjectorEntry.PROJECTOR_NEMOTRON:
                b.require("mm.model.mlp.0.weight", "mm.model.mlp.1.weight", "mm.model.mlp.3.weight");
                return true;
            case ProjectorEntry.PROJECTOR_LING3VL:
                // ling3vl.cpp runs both patch kernels over a still image (build_inp_temporal_merge)
                b.require("v.patch_embd.weight.1", "mm.input_norm.weight", "mm.input_norm.bias", "mm.0.weight",
                        "mm.0.bias", "mm.2.weight", "mm.2.bias");
                return true;
            case ProjectorEntry.PROJECTOR_MINIMAX_M3:
                b.require("mm.1.weight", "mm.1.bias", "mm.2.weight", "mm.2.bias", "mm.merger.fc1.weight",
                        "mm.merger.fc1.bias", "mm.merger.fc2.weight", "mm.merger.fc2.bias");
                return true;
            case ProjectorEntry.PROJECTOR_MUSE_GLIMMER:
                b.require("mm.0.weight", "mm.1.weight", "mm.2.weight");
                return true;
            case ProjectorEntry.PROJECTOR_LFM3VL:
                b.require("mm.input_norm.weight", "mm.input_norm.bias", "mm.0.weight", "mm.0.bias", "mm.2.weight",
                        "mm.2.bias");
                return true;
            case ProjectorEntry.PROJECTOR_LDP:
                b.require("mm.model.mlp.1.weight", "mm.model.mlp.1.bias", "mm.model.mlp.3.weight",
                        "mm.model.mlp.3.bias");
                for (int blk = 1; blk <= 2; blk++) {
                    String p = "mm.model.mb_block." + blk + ".block.";
                    b.require(p + "0.0.weight", p + "0.1.weight", p + "0.1.bias", p + "1.fc1.weight",
                            p + "1.fc1.bias", p + "1.fc2.weight", p + "1.fc2.bias", p + "2.0.weight",
                            p + "2.1.weight", p + "2.1.bias");
                }
                return true;
            case ProjectorEntry.PROJECTOR_LDPV2:
                b.require("mm.model.mlp.0.weight", "mm.model.mlp.0.bias", "mm.model.mlp.2.weight",
                        "mm.model.mlp.2.bias", "mm.model.peg.0.weight", "mm.model.peg.0.bias");
                return true;
            case ProjectorEntry.PROJECTOR_GLM_EDGE:
                b.require("adapter.conv.weight", "adapter.conv.bias", "adapter.linear.linear.weight",
                        "adapter.linear.norm1.weight", "adapter.linear.norm1.bias",
                        "adapter.linear.dense_h_to_4h.weight", "adapter.linear.gate.weight",
                        "adapter.linear.dense_4h_to_h.weight", "adapter.boi", "adapter.eoi");
                return true;
            case ProjectorEntry.PROJECTOR_COGVLM:
                b.require("mm.model.fc.weight", "mm.post_fc_norm.weight", "mm.post_fc_norm.bias", "mm.up.weight",
                        "mm.gate.weight", "mm.down.weight", "v.boi", "v.eoi");
                return true;
            case ProjectorEntry.PROJECTOR_GLM4V:
                b.require("mm.patch_merger.weight", "mm.patch_merger.bias", "mm.model.fc.weight",
                                "mm.post_norm.weight", "mm.up.weight", "mm.gate.weight", "mm.down.weight")
                        .optional("mm.post_norm.bias", "mm.up.bias", "mm.gate.bias", "mm.down.bias");
                return true;
            case ProjectorEntry.PROJECTOR_YOUTUVL:
                b.require("mm.input_norm.weight", "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias");
                return true;
            case ProjectorEntry.PROJECTOR_GRANITE4:
                // every file has at least projector 0; the others are resolved against the feature layer count
                b.require("v.image_newline", "v.proj_blk.0.img_pos", "v.proj_blk.0.query");
                for (String t : Granite4VisionGraph.BLOCK_TENSORS) {
                    b.require("v.proj_blk.0." + t + ".weight", "v.proj_blk.0." + t + ".bias");
                }
                return true;
            case ProjectorEntry.PROJECTOR_SENSENOVA:
                b.require("mm.0.weight", "mm.0.bias");
                return true;
            case ProjectorEntry.PROJECTOR_STEP3VL:
                b.require("mm.0.weight", "mm.1.weight", "mm.model.fc.weight").optional("mm.0.bias", "mm.1.bias");
                return true;
            case ProjectorEntry.PROJECTOR_DEEPSEEK4V:
                b.require("mm.1.weight", "mm.1.bias", "mm.2.weight", "mm.2.bias", "v.image_newline",
                        "v.token_embd.img_start", "v.token_embd.img_end", "v.token_embd.img_pad");
                return true;
            case ProjectorEntry.PROJECTOR_DEEPSEEK41V:
                b.require("mm.1.weight", "mm.1.bias", "mm.2.weight", "mm.2.bias", "v.image_newline",
                        "v.token_embd.img_start", "v.token_embd.img_end");
                return true;
            case ProjectorEntry.PROJECTOR_DEEPSEEKOCR:
                b.require("mm.model.fc.weight", "mm.model.fc.bias", "v.image_newline", "v.view_seperator");
                return true;
            case ProjectorEntry.PROJECTOR_DEEPSEEKOCR2:
                // a file carries the queries of the views it encodes: 1024 for the global view, 768 for tiles
                b.require("mm.model.fc.weight", "mm.model.fc.bias", "v.view_seperator")
                        .optional("v.resample_query_768.weight", "v.resample_query_1024.weight");
                return true;
            case ProjectorEntry.PROJECTOR_GEMMA4V:
                b.require("mm.input_projection.weight")
                        .optional("v.std_bias", "v.std_scale", "mm.input_projection.input_min",
                                "mm.input_projection.input_max", "mm.input_projection.output_min",
                                "mm.input_projection.output_max");
                return true;
            case ProjectorEntry.PROJECTOR_GEMMA4UV:
                b.require("mm.input_projection.weight");
                return true;
            case ProjectorEntry.PROJECTOR_MINICPMV46_MERGER:
                // the ViT merger is required for a 4x merge only and the separator rows for minicpmv4_7 only;
                // the projector tensors check both against the hparams
                b.require("mm.input_norm.weight", "mm.up.weight", "mm.down.weight")
                        .optional("mm.input_norm.bias", "mm.up.bias", "mm.down.bias", "v.tok_embd_sep");
                for (String t : MiniCpmv46Graph.MERGER_TENSORS) {
                    b.optional(t);
                }
                return true;
            case ProjectorEntry.PROJECTOR_MIMOVL_MERGER:
                b.require("mm.0.weight", "mm.2.weight");
                return true;
            case ProjectorEntry.PROJECTOR_GEMMA3NV:
                b.require(MobileNetV5Layout.SOFT_EMB_NORM, MobileNetV5Layout.INPUT_PROJECTION);
                return true;
            default:
                return false;
        }
    }

    /** The encoder graph the table belongs to (a {@code ProjectorEntry.ENCODER_*} name). */
    public String encoderGraph() {
        return encoderGraph;
    }

    /** The projector graph the table belongs to (a {@code ProjectorEntry.PROJECTOR_*} name). */
    public String projectorGraph() {
        return projectorGraph;
    }

    /** Requirement groups in check order (unmodifiable). */
    public List<Requirement> required() {
        return required;
    }

    /** Names the graph reads when present (unmodifiable). */
    public List<String> optional() {
        return optional;
    }

    /**
     * Checks the table against {@code tc}: every requirement group must resolve for every layer
     * ({@code %d} over {@code [0, nLayer)}).
     *
     * @throws MissingTensorException naming the architecture, layer and first absent alternative of a group
     */
    public void validate(TensorCatalog tc, int nLayer) {
        Objects.requireNonNull(tc, "tensors");
        if (nLayer < 0) {
            throw new IllegalArgumentException("block count " + nLayer + " is negative");
        }
        for (Requirement r : required) {
            checkGroup(tc, nLayer, r);
        }
    }

    private static void checkGroup(TensorCatalog tc, int nLayer, Requirement group) {
        if (group.perLayer()) {
            for (int il = 0; il < nLayer; il++) {
                checkLayer(tc, il, group);
            }
            return;
        }
        for (List<String> alternative : group.alternatives()) {
            if (firstAbsent(tc, nLayer, alternative) == null) {
                return;
            }
        }
        Absent absent = firstAbsent(tc, nLayer, group.alternatives().get(0));
        throw new MissingTensorException(tc.archName(), absent.layer,
                group.alternatives().size() == 1 ? absent.name : absent.name + " (or one of the alternative sets"
                        + " of this requirement)");
    }

    /**
     * One block of a per-layer requirement: some alternative must exist whole for block {@code il}; otherwise the
     * refusal names the first absent tensor of the alternative the block carries most of.
     */
    private static void checkLayer(TensorCatalog tc, int il, Requirement group) {
        List<String> closest = null;
        int closestPresent = -1;
        for (List<String> alternative : group.alternatives()) {
            if (firstAbsent(tc, il, il + 1, alternative) == null) {
                return;
            }
            int present = 0;
            for (String name : alternative) {
                if (tc.has(String.format(Locale.ROOT, name, il))) {
                    present++;
                }
            }
            if (present > closestPresent) {
                closest = alternative;
                closestPresent = present;
            }
        }
        Absent absent = firstAbsent(tc, il, il + 1, closest);
        throw new MissingTensorException(tc.archName(), absent.layer, absent.name
                + " (or one of the alternative sets of this requirement)");
    }

    /** A tensor of an alternative the file lacks: its expanded name and layer (-1 for a model-level name). */
    private static final class Absent {
        final String name;
        final int layer;

        Absent(String name, int layer) {
            this.name = name;
            this.layer = layer;
        }
    }

    /** The first name of {@code names} the file lacks, in table then layer order; null when all exist. */
    private static Absent firstAbsent(TensorCatalog tc, int nLayer, List<String> names) {
        return firstAbsent(tc, 0, nLayer, names);
    }

    /** {@link #firstAbsent(TensorCatalog, int, List)} over the blocks {@code [from, to)}. */
    private static Absent firstAbsent(TensorCatalog tc, int from, int to, List<String> names) {
        for (String name : names) {
            if (name.contains("%d")) {
                for (int il = from; il < to; il++) {
                    String expanded = String.format(Locale.ROOT, name, il);
                    if (!tc.has(expanded)) {
                        return new Absent(expanded, il);
                    }
                }
            } else if (!tc.has(name)) {
                return new Absent(name, -1);
            }
        }
        return null;
    }

    /** Incremental builder of a family table. */
    public static final class Builder {
        private final String encoderGraph;
        private final String projectorGraph;
        private final List<Requirement> required = new ArrayList<>();
        private final List<String> optional = new ArrayList<>();

        private Builder(String encoderGraph, String projectorGraph) {
            this.encoderGraph = encoderGraph;
            this.projectorGraph = projectorGraph;
        }

        /** Requires every name to exist ({@code %d} expands over the layers). */
        public Builder require(String... names) {
            return addAlternatives(new String[][]{names});
        }

        /** Alias of {@link #require(String...)} for the per-layer names. */
        public Builder requireLayers(String... names) {
            return require(names);
        }

        /** Requires that, per layer, every name of one alternative exists. */
        public Builder requireEitherLayer(String[]... alternatives) {
            return addAlternatives(alternatives);
        }

        /** Requires that every block holds every name of one alternative, each block choosing its own. */
        public Builder requireEachLayerEither(String[]... alternatives) {
            return addAlternatives(true, alternatives);
        }

        private Builder addAlternatives(String[]... alternatives) {
            return addAlternatives(false, alternatives);
        }

        private Builder addAlternatives(boolean perLayer, String[]... alternatives) {
            List<List<String>> sets = new ArrayList<>();
            for (String[] a : alternatives) {
                if (a.length == 0) {
                    throw new IllegalArgumentException("an alternative needs at least one tensor name");
                }
                sets.add(Collections.unmodifiableList(new ArrayList<>(Arrays.asList(a))));
            }
            required.add(new Requirement(Collections.unmodifiableList(sets), perLayer));
            return this;
        }

        /** Records names the graph reads when present, without requiring them. */
        public Builder optional(String... names) {
            optional.addAll(Arrays.asList(names));
            return this;
        }

        public MmprojTensorTable build() {
            return new MmprojTensorTable(encoderGraph, projectorGraph,
                    Collections.unmodifiableList(new ArrayList<>(required)),
                    Collections.unmodifiableList(new ArrayList<>(optional)));
        }
    }
}
