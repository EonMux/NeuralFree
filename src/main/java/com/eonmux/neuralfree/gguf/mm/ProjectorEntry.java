package com.eonmux.neuralfree.gguf.mm;

import java.util.Objects;

/**
 * One row of {@link ProjectorRegistry}: the projector family a {@code clip.projector_type} value runs as.
 * A runnable row names the encoder graph that reads the image or audio tensor and the projector graph that maps
 * the encoder output to text-model embedding rows, and carries the family's tensor requirement table. A rejected
 * row records why a family known to the reference is not runnable (no graphs are registered for it); {@code get}
 * refuses it with that reason. A runnable vision row also carries the {@link VisionPreprocessing} its images are
 * prepared and counted with: its type's own, or the one a plugin row declares for its key.
 *
 * <p>Plugin jars contribute rows of this type through {@code ProjectorProvider}. Immutable.
 */
public final class ProjectorEntry {

    /**
     * The llava ViT (models/llava.cpp): class token appended last, learned position rows, optional pre/post layer
     * norm, the blocks up to the deepest feature layer (the second to last block by default) and the feature
     * layers stacked along the channels.
     */
    public static final String ENCODER_CLIP_VIT = "clip_vit";
    /** SigLIP-style ViT: no class embedding, no pre-norm, learned position rows. */
    public static final String ENCODER_SIGLIP_VIT = "siglip_vit";
    /** Qwen2-VL / Qwen2.5-VL ViT: 2D m-rope attention, no class embedding, RMS or layer norm. */
    public static final String ENCODER_QWEN2VL_VIT = "qwen2vl_vit";
    /**
     * SigLIP ViT over any patch grid: the learned position rows are resized to the grid with an antialiased
     * bilinear filter (clip.cpp {@code resize_position_embeddings}).
     */
    public static final String ENCODER_SIGLIP_RESIZED_VIT = "siglip_resized_vit";
    /** MiniCPM-V ViT: no class embedding, position rows gathered from a 70x70 bucket table, any patch grid. */
    public static final String ENCODER_MINICPMV_VIT = "minicpmv_vit";
    /** InternViT: class token first, learned position rows, layer scale, RMS norm for the 6B encoder. */
    public static final String ENCODER_INTERNVL_VIT = "internvl_vit";
    /** Pixtral ViT: RMS norm, 2D RoPE with interleaved row/column frequencies, any patch grid. */
    public static final String ENCODER_PIXTRAL_VIT = "pixtral_vit";
    /** Llama 4 ViT: unfold-linear patch embedding, class token last, learned position rows plus 2D RoPE. */
    public static final String ENCODER_LLAMA4_VIT = "llama4_vit";
    /**
     * Qwen3-VL ViT: two-frame patch convolution, learned positions resized to the grid, 2D m-rope, merge-group
     * token order, and the deepstack taps of the blocks that carry them.
     */
    public static final String ENCODER_QWEN3VL_VIT = "qwen3vl_vit";
    /** PaddleOCR-VL ViT: SigLIP-style layer-norm ViT with resized positions and the qwen 2D m-rope, raster order. */
    public static final String ENCODER_PADDLEOCR_VIT = "paddleocr_vit";
    /** Kimi-VL MoonViT: layer-norm ViT with resized positions and a 2D RoPE (columns first, rows second). */
    public static final String ENCODER_KIMIVL_VIT = "kimivl_vit";
    /**
     * Kimi K2.5 MoonViT: the Kimi-VL tower whose learned positions are a {@code [n_embd, W, H]} grid resized to the
     * patch grid with the bicubic filter.
     */
    public static final String ENCODER_KIMIK25_VIT = "kimik25_vit";
    /**
     * dots.ocr / dots3note ViT: RMS-norm ViT with the qwen 2D m-rope in raster order, optional sigmoid-routed MoE
     * FFN blocks, and the {@code mm.post_norm} trunk norm.
     */
    public static final String ENCODER_DOTS_VIT = "dots_vit";
    /** HunyuanVL ViT: layer-norm ViT whose learned position grid is resampled to the image on the host. */
    public static final String ENCODER_HUNYUANVL_VIT = "hunyuanvl_vit";
    /** CogVLM ViT: class token last, learned position rows, post-norm blocks (norm after attention and FFN). */
    public static final String ENCODER_COGVLM_VIT = "cogvlm_vit";
    /**
     * GLM-4V ViT: two patch convolutions summed in merge-group order, optional embedding norm, bicubic-resized
     * position rows, RMS norm and 2D m-rope.
     */
    public static final String ENCODER_GLM4V_VIT = "glm4v_vit";
    /**
     * Youtu-VL ViT: linear patch embedding in merge-group order, layer norm, 2D m-rope, window attention outside
     * the listed full-attention layers.
     */
    public static final String ENCODER_YOUTUVL_VIT = "youtuvl_vit";
    /** Granite Vision 4.1 SigLIP tower whose block outputs feed the WindowQFormer projectors. */
    public static final String ENCODER_GRANITE4_VIT = "granite4_vit";
    /**
     * Yasa2 (Reka Edge) ConvNeXt-V2 backbone: patch convolution, downsampling stages of depthwise-conv / GRN
     * blocks, the learned position rows and the 8x8 average pool.
     */
    public static final String ENCODER_YASA2_CONVNEXT = "yasa2_convnext";
    /**
     * Nemotron RADIO ViT: patch conv, learned position rows (the fixed grid for v2-VL, bilinear-resized for Omni),
     * register tokens prepended, the layer-norm ViT, registers dropped.
     */
    public static final String ENCODER_NEMOTRON_VIT = "nemotron_vit";
    /** MiniMax-M3 ViT: two patch convolutions in merge-group order, layer norm, NEOX rope on the h/w dim thirds. */
    public static final String ENCODER_MINIMAX_M3_VIT = "minimax_m3_vit";
    /**
     * Muse Glimmer ViT: patch conv, bilinear-resized position rows, window-grouped rows, 2D rope, block-diagonal
     * window attention outside every fourth and the last block.
     */
    public static final String ENCODER_MUSE_GLIMMER_VIT = "muse_glimmer_vit";
    /**
     * LFM3-VL tower: patch conv(s) in raster order, RMS embedding norm, then per block either a gated depthwise
     * conv mixer or GQA attention with Q/K norms and 2D m-rope, each followed by the FFN.
     */
    public static final String ENCODER_LFM3VL_VIT = "lfm3vl_vit";
    /** SenseNova-U1 patch embedder: the biased patch conv, GELU and the one-head 2D RoPE; no encoder blocks. */
    public static final String ENCODER_SENSENOVA_PATCH = "sensenova_patch";
    /**
     * Step3-VL ViT: the Kimi-VL tower (learned position grid resized antialiased, layer norm, 2D RoPE of
     * {@code build_rope_2d}) without a class token.
     */
    public static final String ENCODER_STEP3VL_VIT = "step3vl_vit";
    /**
     * DeepSeek-V4 ViT (deepseek4v, deepseek41v): RMS norm, no learned positions, the qwen vision m-rope over the
     * patch row and column.
     */
    public static final String ENCODER_DEEPSEEK4V_VIT = "deepseek4v_vit";
    /**
     * DeepSeek-OCR: the SAM ViT-B backbone (windowed attention with decomposed relative positions, the conv neck
     * and two stride-2 convs), then the CLIP-L tower over the SAM rows with a class token and learned positions.
     */
    public static final String ENCODER_DEEPSEEKOCR_SAM_CLIP = "deepseekocr_sam_clip";
    /**
     * DeepSeek-OCR 2: the SAM ViT-B backbone, then the Qwen2 encoder (RMS norm, GQA, NEOX rope) over the SAM rows
     * and the learned queries, the queries attending causally.
     */
    public static final String ENCODER_DEEPSEEKOCR2_SAM_QWEN2 = "deepseekocr2_sam_qwen2";
    /**
     * Gemma 4 ViT: inputs scaled to {@code 2x - 1}, a bias-free patch convolution, learned column and row
     * position tables, RMS-norm blocks with post norms, an output scale and clippable linears, the NEOX 2D RoPE,
     * unit attention scale and RMS-normalized values.
     */
    public static final String ENCODER_GEMMA4V_VIT = "gemma4v_vit";
    /**
     * Gemma 4 unified embedder: no blocks; layer-normed patch vectors of {@code patch_size * n_merge} pixels, a
     * linear projection, layer norm, the column and row position tables and a closing layer norm.
     */
    public static final String ENCODER_GEMMA4UV = "gemma4uv_embed";
    /**
     * MiniCPM-V 4.6 ViT: position rows gathered from a 70x70 bucket table, a layer-norm tower with layer scales,
     * and, for a 4x merge, the window-attention ViT merger after block {@code insert_layer_id} that halves each
     * side of the patch grid.
     */
    public static final String ENCODER_MINICPMV46_VIT = "minicpmv46_vit";
    /**
     * MiMo-VL ViT: two bias-free patch convolutions summed in 2x2 merge-group order, RMS-norm blocks with fused
     * grouped-query Q/K/V heads as wide as the fused projection says, the qwen 2D m-rope and a SwiGLU FFN; per
     * block ({@code clip.vision.wa_pattern_mode}) full attention, or a banded window over the row-major or
     * column-major merge-unit order with per-head attention sinks; then the layer post-norm.
     */
    public static final String ENCODER_MIMOVL_VIT = "mimovl_vit";
    /**
     * Gemma 3n MobileNetV5: a convolutional encoder, no ViT blocks; the stem conv, edge residual, universal inverted
     * residual and multi-query attention blocks in four downsampling stages, and the multi-scale fusion adapter.
     */
    public static final String ENCODER_MOBILENETV5 = "mobilenetv5";

    /**
     * qwen3tts generator: no media encoder runs; the file holds the talker's code predictor and code2wav decoder
     * (tensor prefix {@code a.gen.}), which the {@code qwen3tts_gen} projector row opens.
     */
    public static final String ENCODER_QWEN3TTS_GEN = "qwen3tts_gen";

    /**
     * pockettts generator: no media encoder runs; the file holds the flow-matching decoder and the streaming mimi
     * decoder (tensor prefix {@code a.gen.}), which the {@code pockettts_gen} projector row opens.
     */
    public static final String ENCODER_POCKETTTS_GEN = "pockettts_gen";

    /**
     * lfm3a generator: no media encoder runs; the file holds the MaskGIT depthformer and the liquid-codec decoder
     * (tensor prefix {@code a.gen.}), which the {@code lfm3a_gen} projector row opens.
     */
    public static final String ENCODER_LFM3A_GEN = "lfm3a_gen";

    /** qwen3tts generator projector: the code predictor and code2wav graphs of {@link #ENCODER_QWEN3TTS_GEN}. */
    public static final String PROJECTOR_QWEN3TTS_GEN = "qwen3tts_gen_projector";
    /** pockettts generator projector: the flow network and mimi decoder graphs of {@link #ENCODER_POCKETTTS_GEN}. */
    public static final String PROJECTOR_POCKETTTS_GEN = "pockettts_gen_projector";

    /** lfm3a generator projector: the depthformer and codec graphs of {@link #ENCODER_LFM3A_GEN}. */
    public static final String PROJECTOR_LFM3A_GEN = "lfm3a_gen_projector";
    /** LLaVA MLP: {@code mm.0} + GELU + optional {@code mm.2}. */
    public static final String PROJECTOR_LLAVA_MLP = "llava_mlp";
    /** LLaVA MLP with norms: {@code mm.0}, norm, GELU, {@code mm.3}, norm. */
    public static final String PROJECTOR_LLAVA_MLP_NORM = "llava_mlp_norm";
    /** Qwen2-VL merger: 2x2 pixel shuffle, then {@code mm.0} + GELU + {@code mm.2}. */
    public static final String PROJECTOR_QWEN2VL_MERGER = "qwen2vl_merger";
    /** Gemma3: average pool over the patch grid, RMS norm, {@code mm.input_projection}. */
    public static final String PROJECTOR_GEMMA3 = "gemma3";
    /**
     * MiniCPM-V resampler: one cross-attention of the learned queries over the encoder rows (2D sin-cos keys),
     * post norm and projection.
     */
    public static final String PROJECTOR_MINICPMV_RESAMPLER = "minicpmv_resampler";
    /** InternVL: pixel shuffle, layer norm, {@code mm.model.mlp.1} + GELU + {@code mm.model.mlp.3}. */
    public static final String PROJECTOR_INTERNVL = "internvl_mlp";
    /** Pixtral: optional patch merger, {@code mm.1} + GELU + {@code mm.2}, an {@code [IMG_BREAK]} row per line. */
    public static final String PROJECTOR_PIXTRAL = "pixtral_mlp";
    /** Idefics3: pixel shuffle, then {@code mm.model.fc}. */
    public static final String PROJECTOR_IDEFICS3 = "idefics3_shuffle";
    /** Llama 4: pixel shuffle, two GELU MLP layers, then {@code mm.model.fc}. */
    public static final String PROJECTOR_LLAMA4 = "llama4_shuffle_mlp";
    /** Phi-4 multimodal: {@code mm.0} + GELU + {@code mm.2}. */
    public static final String PROJECTOR_PHI4 = "phi4_mlp";
    /** Qwen3-VL: {@code mm.0} + GELU + {@code mm.2} over merged rows, the deepstack features concatenated after. */
    public static final String PROJECTOR_QWEN3VL_MERGER = "qwen3vl_merger";
    /** PaddleOCR-VL: layer norm, pixel shuffle, {@code mm.1} + the encoder activation + {@code mm.2}. */
    public static final String PROJECTOR_PADDLEOCR = "paddleocr_mlp";
    /** Kimi-VL: pixel shuffle, a layer norm per merged patch, {@code mm.1} + GELU + {@code mm.2}. */
    public static final String PROJECTOR_KIMIVL = "kimivl_mlp";
    /** Kimi K2.5: the Kimi-VL projector with its layer norm at the encoder epsilon. */
    public static final String PROJECTOR_KIMIK25 = "kimik25_mlp";
    /** dots.ocr: layer norm, pixel shuffle, {@code mm.0} + GELU-erf + {@code mm.2}. */
    public static final String PROJECTOR_DOTS = "dots_mlp";
    /**
     * HunyuanVL perceiver: RMS norm, strided conv + GELU + pointwise conv, a newline column, {@code mm.model.fc},
     * the begin/end rows and the closing RMS norm.
     */
    public static final String PROJECTOR_HUNYUANVL = "hunyuanvl_perceiver";
    /** Janus-Pro: {@code mm.0} + the encoder activation + {@code mm.1}. */
    public static final String PROJECTOR_JANUS = "janus_mlp";
    /** LFM2-VL / d1-omni: pixel shuffle, optional layer norm, {@code mm.1} + GELU (erf for d1-omni) + {@code mm.2}. */
    public static final String PROJECTOR_LFM2 = "lfm2_mlp";
    /** Cohere2 vision: pixel shuffle, {@code mm.1}, swapped SwiGLU, {@code mm.2}. */
    public static final String PROJECTOR_COHERE2V = "cohere2v_swiglu";
    /** LightOnOCR: the Pixtral patch merger and MLP without {@code [IMG_BREAK]} rows. */
    public static final String PROJECTOR_LIGHTONOCR = "lightonocr_mlp";
    /** MobileVLM LDP: GELU MLP, then two depthwise-conv / squeeze-excite blocks, the second at stride 2. */
    public static final String PROJECTOR_LDP = "ldp";
    /** MobileVLM V2 LDPv2: GELU MLP, 2x2 average pool, depthwise-conv position encoding with residual. */
    public static final String PROJECTOR_LDPV2 = "ldpv2";
    /** GLM-Edge adapter: 2x2 stride-2 conv, linear, layer norm, GELU, SwiGLU, then the BOI/EOI rows. */
    public static final String PROJECTOR_GLM_EDGE = "glm_edge_adapter";
    /** CogVLM: {@code mm.model.fc}, layer norm, GELU, SwiGLU, then the BOI/EOI rows. */
    public static final String PROJECTOR_COGVLM = "cogvlm_glu";
    /** GLM-4V: conv patch merger, {@code mm.model.fc}, layer norm, GELU-erf, then the gated FFN. */
    public static final String PROJECTOR_GLM4V = "glm4v_merger";
    /** Youtu-VL: RMS norm, 2x2 merge, {@code mm.0} + GELU + {@code mm.2}. */
    public static final String PROJECTOR_YOUTUVL = "youtuvl_merger";
    /** Granite Vision 4.1: one WindowQFormer per feature layer, streams stacked along the channels. */
    public static final String PROJECTOR_GRANITE4 = "granite4_qformer";
    /** Yasa2: {@code mm.0} + GELU-erf + {@code mm.2} over the pooled tokens. */
    public static final String PROJECTOR_YASA2 = "yasa2_mlp";
    /** Nemotron: pixel shuffle, RMS norm, {@code mm.model.mlp.1} + ReLU^2 + {@code mm.model.mlp.3}. */
    public static final String PROJECTOR_NEMOTRON = "nemotron_mlp";
    /** Ling3-VL: per-patch layer norm, 2x2 merge, {@code mm.0} + GELU + {@code mm.2}. */
    public static final String PROJECTOR_LING3VL = "ling3vl_merger";
    /** MiniMax-M3: per-patch {@code mm.1} + GELU-erf + {@code mm.2}, merge, then the fc1/fc2 GELU-erf merger. */
    public static final String PROJECTOR_MINIMAX_M3 = "minimax_m3_merger";
    /** Muse Glimmer: channel-outer pixel shuffle, three bias-free linears with GELU-erf between. */
    public static final String PROJECTOR_MUSE_GLIMMER = "muse_glimmer_mlp";
    /** LFM3-VL: layer norm, 2x2 merge, {@code mm.0} + GELU-erf + {@code mm.2}. */
    public static final String PROJECTOR_LFM3VL = "lfm3vl_mlp";
    /** SenseNova-U1: the biased 2x2 stride-2 conv {@code mm.0} over the rotated patch grid. */
    public static final String PROJECTOR_SENSENOVA = "sensenova_conv";
    /** Step3-VL: two 3x3 stride-2 convs {@code mm.0} / {@code mm.1} with optional biases, then {@code mm.model.fc}. */
    public static final String PROJECTOR_STEP3VL = "step3vl_conv";
    /**
     * DeepSeek-V4: the padded {@code n_merge} unfold, the GELU-erf MLP {@code mm.1} / {@code mm.2}, then the token
     * block of lead pads, start, row-pair interleaved rows with newlines, pads and end.
     */
    public static final String PROJECTOR_DEEPSEEK4V = "deepseek4v_aligner";
    /** DeepSeek-V4.1: the DeepSeek-V4 aligner, then the reading-order block of start, rows with newlines and end. */
    public static final String PROJECTOR_DEEPSEEK41V = "deepseek41v_aligner";
    /**
     * DeepSeek-OCR: {@code mm.model.fc} over the CLIP rows concatenated with the SAM rows, then a newline per row
     * (and the view separator after the global view).
     */
    public static final String PROJECTOR_DEEPSEEKOCR = "deepseekocr_fc";
    /** DeepSeek-OCR 2: {@code mm.model.fc} over the query rows, then the view separator after the global view. */
    public static final String PROJECTOR_DEEPSEEKOCR2 = "deepseekocr2_fc";
    /**
     * No projector: a tower of a CLIP two-tower file ({@link ClipVisionModel#twoTower}), whose rows are the output
     * and whose heads and projections the dual encoder applies.
     */
    public static final String PROJECTOR_NONE = "none";
    /**
     * Gemma 4: {@code n_merge x n_merge} average pool, {@code sqrt(n_embd)} scale, the optional standardization,
     * a weightless RMS norm and the clippable {@code mm.input_projection}.
     */
    public static final String PROJECTOR_GEMMA4V = "gemma4v_pool";
    /** Gemma 4 unified: a weightless RMS norm and {@code mm.input_projection}. */
    public static final String PROJECTOR_GEMMA4UV = "gemma4uv_proj";
    /**
     * MiniCPM-V 4.6 merger: 2x2 neighbor concat, layer norm, {@code mm.up} + GELU + {@code mm.down}, then the
     * separator rows of the image's suffix.
     */
    public static final String PROJECTOR_MINICPMV46_MERGER = "minicpmv46_merger";
    /** MiMo-VL merger: every 2x2 group as one row, the bias-free {@code mm.0} + GELU + {@code mm.2}. */
    public static final String PROJECTOR_MIMOVL_MERGER = "mimovl_merger";
    /**
     * Gemma 3n embedder: the {@code sqrt(n_embd)} scale, RMS norm times {@code mm.soft_emb_norm},
     * {@code mm.input_projection} and a weightless RMS norm.
     */
    public static final String PROJECTOR_GEMMA3NV = "gemma3nv_embed";

    /** Whisper audio encoder: two conv1d blocks, learned position rows, the pre-norm ViT tower. */
    public static final String ENCODER_WHISPER = "whisper_enc";
    /** Qwen3-ASR audio encoder: three stride-2 conv2d blocks per 100-frame chunk, then the ViT tower. */
    public static final String ENCODER_QWEN3A = "qwen3a_enc";
    /** NeMo conformer encoder (lfm2a, d1omni_a): conv subsampling, relative-position conformer blocks. */
    public static final String ENCODER_CONFORMER = "conformer_enc";
    /** LFM3-Audio encoder: causal conv downsampler and the LFM2 stack of short-conv and windowed attention blocks. */
    public static final String ENCODER_LFM3A = "lfm3a_enc";
    /** Gemma 4 audio conformer: conv stem with per-channel norms, chunked local attention with softcap and RPE. */
    public static final String ENCODER_GEMMA4A = "gemma4a_enc";
    /** Gemma 4 unified audio embedder: raw 640-sample waveform frames through RMS norm and one projection. */
    public static final String ENCODER_GEMMA4UA = "gemma4ua_enc";
    /** Parakeet CTC encoder: conv subsampling, relative-position conformer blocks with folded batch norms. */
    public static final String ENCODER_PARAKEET = "parakeet_enc";
    /** Granite speech conformer: chunked attention with Shaw RPE, a CTC branch and the QFormer projector input. */
    public static final String ENCODER_GRANITE_SPEECH = "granite_speech_enc";
    /** Voxtral realtime encoder: causal conv stem and a sliding-window RoPE transformer over the mel frames. */
    public static final String ENCODER_VOXTRAL_RT = "voxtral_rt_enc";
    /** Mimo audio encoder: whisper-style conv stem, the ViT tower with RoPE, RVQ quantization and the local transformer. */
    public static final String ENCODER_MIMO_AUDIO = "mimo_audio_enc";
    /** Muse Spark encoder: a causal conv pyramid over the raw waveform, no attention and no spectrogram. */
    public static final String ENCODER_MUSE_SPARK_AUDIO = "muse_spark_enc";
    /** dots3note audio encoder: conv stem, the ViT tower with partial RoPE and the GELU-erf adapter. */
    public static final String ENCODER_DOTS3NOTE_A = "dots3note_a_enc";
    /** Gemma 3n audio encoder: two sub-sample conv blocks, then the USM conformer blocks. */
    public static final String ENCODER_GEMMA3N_USM = "gemma3n_usm";

    /** Ultravox: stacked frames, RMS pre-norm, swapped SwiGLU {@code mm.a.mlp.1}, RMS mid-norm, {@code mm.a.mlp.2}. */
    public static final String PROJECTOR_ULTRAVOX = "ultravox";
    /** Qwen2-Audio: average-pooled frames through {@code mm.a.fc}. */
    public static final String PROJECTOR_QWEN2A = "qwen2a_fc";
    /** Voxtral: pooled and stacked frames through the bias-free GELU-erf MLP {@code mm.a.mlp.1/2}. */
    public static final String PROJECTOR_VOXTRAL = "voxtral_mlp";
    /** MusicFlamingo: pooled frames through the biased GELU-erf MLP {@code mm.a.mlp.1/2}. */
    public static final String PROJECTOR_MUSIC_FLAMINGO = "musicflamingo_mlp";
    /** MERaLiON: stacked frames, layer norm, SiLU {@code mm.a.mlp.0}, gate/pool GLU, {@code mm.a.mlp.3}. */
    public static final String PROJECTOR_MERALION = "meralion_glu";
    /** GLM-Audio: layer norm, stacked frames, GELU-erf MLP, then the BOI/EOI rows. */
    public static final String PROJECTOR_GLMA = "glma_mlp";
    /** Qwen3-ASR: the GELU-erf MLP {@code mm.a.mlp.1/2}. */
    public static final String PROJECTOR_QWEN3A = "qwen3a_mlp";
    /** LFM2-Audio / d1-omni audio: layer norm, the GELU-erf MLP, the d1-omni residual block. */
    public static final String PROJECTOR_CONFORMER = "conformer_mlp";
    /** LFM3-Audio: layer norm, the GELU-erf MLP. */
    public static final String PROJECTOR_LFM3A = "lfm3a_mlp";
    /** Gemma 4 audio: RMS norm, the GELU-erf MLP, then the optional out projection and multimodal embedder. */
    public static final String PROJECTOR_GEMMA4A = "gemma4a_mlp";
    /** Gemma 4 unified audio: RMS norm and the input projection over the raw waveform frames. */
    public static final String PROJECTOR_GEMMA4UA = "gemma4ua_proj";
    /** Parakeet: RMS norm and the ReLU-squared MLP. */
    public static final String PROJECTOR_PARAKEET = "parakeet_mlp";
    /** Granite speech: the QFormer projector, a learned query attending over windowed encoder frames. */
    public static final String PROJECTOR_GRANITE_SPEECH = "granite_qformer";
    /** Voxtral realtime: the frame stack and the GELU-erf MLP. */
    public static final String PROJECTOR_VOXTRAL_RT = "voxtral_rt_mlp";
    /** Mimo audio: the code embeddings summed over the RVQ quantizers and the local transformer's GELU MLP. */
    public static final String PROJECTOR_MIMO_AUDIO = "mimo_audio_proj";
    /** Muse Spark: the RMS trunk norms and the fc into the text hidden size. */
    public static final String PROJECTOR_MUSE_SPARK_AUDIO = "muse_spark_fc";
    /** dots3note audio: layer norm and the GELU-erf MLP. */
    public static final String PROJECTOR_DOTS3NOTE_A = "dots3note_a_mlp";
    /** Gemma 3n: every fourth row through the soft-token embedder, padded with the last hard token's row. */
    public static final String PROJECTOR_GEMMA3NA = "gemma3na_embedder";

    private final ProjectorType type;
    private final String key;
    private final ProjectorModality modality;
    private final String encoderGraph;
    private final String projectorGraph;
    private final MmprojTensorTable tensors;
    private final VisionPreprocessing preprocessing;
    private final String rejection;

    private ProjectorEntry(String key, ProjectorType type, ProjectorModality modality, String encoderGraph,
                           String projectorGraph, MmprojTensorTable tensors, VisionPreprocessing preprocessing,
                           String rejection) {
        this.key = key;
        this.type = type;
        this.modality = modality;
        this.encoderGraph = encoderGraph;
        this.projectorGraph = projectorGraph;
        this.tensors = tensors;
        this.preprocessing = preprocessing;
        this.rejection = rejection;
    }

    /**
     * A runnable row keyed by the type's own name.
     *
     * @throws IllegalArgumentException for an unknown graph name or the UNKNOWN type
     */
    public static ProjectorEntry of(ProjectorType type, ProjectorModality modality, String encoderGraph,
                                    String projectorGraph) {
        return keyed(typeName(type), type, modality, encoderGraph, projectorGraph);
    }

    /**
     * A runnable row under an explicit registry key, which a plugin may choose freely (a new family name, the
     * {@code architectures[0]} of a Hugging Face projector checkpoint); {@code type} may be
     * {@link ProjectorType#UNKNOWN} for such rows. A vision row prepares its images with the type's
     * {@link ProjectorType#visionPreprocessing()} (none for {@link ProjectorType#UNKNOWN}).
     *
     * @throws IllegalArgumentException for a blank key, an unknown graph name or a keyed UNKNOWN row without
     *                                  modality
     */
    public static ProjectorEntry keyed(String key, ProjectorType type, ProjectorModality modality,
                                       String encoderGraph, String projectorGraph) {
        Objects.requireNonNull(type, "type");
        return build(key, type, modality, encoderGraph, projectorGraph,
                modality == ProjectorModality.VISION ? type.visionPreprocessing() : null);
    }

    /**
     * A runnable vision row under an explicit registry key that declares its own image preprocessing: the
     * preprocessing family its images are prepared with, the formula its image token count follows and its patch
     * merge rule. This is how a plugin family the built-in types do not cover declares its token count.
     *
     * @throws IllegalArgumentException for a blank key, an unknown graph name, or a modality other than
     *                                  {@link ProjectorModality#VISION}
     */
    public static ProjectorEntry keyed(String key, ProjectorType type, ProjectorModality modality,
                                       String encoderGraph, String projectorGraph,
                                       VisionPreprocessing preprocessing) {
        Objects.requireNonNull(preprocessing, "preprocessing");
        Objects.requireNonNull(modality, "modality");
        if (modality != ProjectorModality.VISION) {
            throw new IllegalArgumentException("projector key '" + key + "' declares image preprocessing for a "
                    + modality + " row; only vision rows prepare images");
        }
        return build(key, type, modality, encoderGraph, projectorGraph, preprocessing);
    }

    private static ProjectorEntry build(String key, ProjectorType type, ProjectorModality modality,
                                        String encoderGraph, String projectorGraph,
                                        VisionPreprocessing preprocessing) {
        Objects.requireNonNull(key, "key");
        if (key.isBlank()) {
            throw new IllegalArgumentException("the registry key is blank");
        }
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(modality, "modality");
        MmprojTensorTable tensors = MmprojTensorTable.forFamily(encoderGraph, projectorGraph);
        if (modalityOf(encoderGraph) != modality) {
            throw new IllegalArgumentException("encoder graph " + encoderGraph + " reads "
                    + modalityOf(encoderGraph) + " input, the row declares " + modality);
        }
        return new ProjectorEntry(key, type, modality, encoderGraph, projectorGraph, tensors, preprocessing, null);
    }

    /**
     * The modality an encoder graph reads.
     *
     * @throws IllegalArgumentException for an unknown graph name
     */
    public static ProjectorModality modalityOf(String encoderGraph) {
        Objects.requireNonNull(encoderGraph, "encoderGraph");
        switch (encoderGraph) {
            case ENCODER_CLIP_VIT:
            case ENCODER_SIGLIP_VIT:
            case ENCODER_QWEN2VL_VIT:
            case ENCODER_SIGLIP_RESIZED_VIT:
            case ENCODER_MINICPMV_VIT:
            case ENCODER_INTERNVL_VIT:
            case ENCODER_PIXTRAL_VIT:
            case ENCODER_LLAMA4_VIT:
            case ENCODER_QWEN3VL_VIT:
            case ENCODER_PADDLEOCR_VIT:
            case ENCODER_KIMIVL_VIT:
            case ENCODER_KIMIK25_VIT:
            case ENCODER_DOTS_VIT:
            case ENCODER_HUNYUANVL_VIT:
            case ENCODER_COGVLM_VIT:
            case ENCODER_GLM4V_VIT:
            case ENCODER_YOUTUVL_VIT:
            case ENCODER_GRANITE4_VIT:
            case ENCODER_YASA2_CONVNEXT:
            case ENCODER_NEMOTRON_VIT:
            case ENCODER_MINIMAX_M3_VIT:
            case ENCODER_MUSE_GLIMMER_VIT:
            case ENCODER_LFM3VL_VIT:
            case ENCODER_SENSENOVA_PATCH:
            case ENCODER_STEP3VL_VIT:
            case ENCODER_DEEPSEEK4V_VIT:
            case ENCODER_DEEPSEEKOCR_SAM_CLIP:
            case ENCODER_DEEPSEEKOCR2_SAM_QWEN2:
            case ENCODER_GEMMA4V_VIT:
            case ENCODER_GEMMA4UV:
            case ENCODER_MINICPMV46_VIT:
            case ENCODER_MIMOVL_VIT:
            case ENCODER_MOBILENETV5:
                return ProjectorModality.VISION;
            case ENCODER_QWEN3TTS_GEN:
            case ENCODER_POCKETTTS_GEN:
            case ENCODER_LFM3A_GEN:
                return ProjectorModality.GEN_AUDIO;
            case ENCODER_WHISPER:
            case ENCODER_QWEN3A:
            case ENCODER_CONFORMER:
            case ENCODER_LFM3A:
            case ENCODER_GEMMA4A:
            case ENCODER_GEMMA4UA:
            case ENCODER_PARAKEET:
            case ENCODER_GRANITE_SPEECH:
            case ENCODER_VOXTRAL_RT:
            case ENCODER_MIMO_AUDIO:
            case ENCODER_MUSE_SPARK_AUDIO:
            case ENCODER_DOTS3NOTE_A:
            case ENCODER_GEMMA3N_USM:
                return ProjectorModality.AUDIO;
            default:
                throw new IllegalArgumentException("unknown encoder graph " + encoderGraph);
        }
    }

    /**
     * A rejected row: the family is known to the reference but has no forward graphs in NeuralFree.
     *
     * @throws IllegalArgumentException for an empty reason or the UNKNOWN type
     */
    public static ProjectorEntry rejected(ProjectorType type, String reason) {
        return keyedRejected(typeName(type), type, reason);
    }

    /** A rejected row under an explicit registry key. */
    public static ProjectorEntry keyedRejected(String key, ProjectorType type, String reason) {
        Objects.requireNonNull(key, "key");
        if (key.isBlank()) {
            throw new IllegalArgumentException("the registry key is blank");
        }
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(reason, "reason");
        if (reason.isBlank()) {
            throw new IllegalArgumentException("the rejection reason is empty");
        }
        if (type == ProjectorType.UNKNOWN) {
            throw new IllegalArgumentException("the UNKNOWN projector type has no row");
        }
        return new ProjectorEntry(key, type, null, null, null, null, null, reason);
    }

    private static String typeName(ProjectorType type) {
        if (type == ProjectorType.UNKNOWN) {
            throw new IllegalArgumentException("the UNKNOWN projector type has no row");
        }
        return type.typeName();
    }

    /** The registry key the row is filed under ({@code clip.projector_type} values, or a plugin's own key). */
    public String key() {
        return key;
    }

    /** The projector type this row describes. */
    public ProjectorType type() {
        return type;
    }

    /** The modality, or null for a rejected row. */
    public ProjectorModality modality() {
        return modality;
    }

    /** The encoder graph, or null for a rejected row. */
    public String encoderGraph() {
        return encoderGraph;
    }

    /** The projector graph, or null for a rejected row. */
    public String projectorGraph() {
        return projectorGraph;
    }

    /** The tensor requirement table, or null for a rejected row. */
    public MmprojTensorTable tensors() {
        return tensors;
    }

    /**
     * The image preprocessing of a runnable vision row, or null for an audio row, a rejected row, or a vision
     * row whose type NeuralFree does not preprocess.
     */
    public VisionPreprocessing visionPreprocessing() {
        return preprocessing;
    }

    /** The refusal reason, or null when the row is runnable. */
    public String rejection() {
        return rejection;
    }

    /** True when the row names forward graphs that run the family. */
    public boolean runnable() {
        return rejection == null;
    }

    @Override
    public String toString() {
        return "ProjectorEntry{" + type + (runnable()
                ? ", " + modality.name().toLowerCase(java.util.Locale.ROOT) + ", encoder=" + encoderGraph
                        + ", projector=" + projectorGraph
                : ", rejected: " + rejection) + "}";
    }
}
