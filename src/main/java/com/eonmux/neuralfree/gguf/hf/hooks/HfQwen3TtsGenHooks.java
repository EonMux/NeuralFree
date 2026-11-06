package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.names.HfNameTables;
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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The qwen3tts generator projector, a port of llama.cpp's {@code Qwen3TTSSpeakerEncoderModel}
 * (conversion/qwen3tts.py:190-479): the code predictor (the talker's 16-code frame predictor), the code2wav DAC
 * decoder and, when the checkpoint carries one, the ECAPA-TDNN speaker encoder of voice cloning.
 *
 * <p>The checkpoint's main safetensors file carries the code predictor as 15 separate codebook pages and 15 separate
 * LM-head pages, which are stacked into the 3-D tensors the clip file reads
 * ({@code a.gen.code.embd.weight}, {@code a.gen.code.head.weight}). The code2wav graphs live in the separate
 * {@code speech_tokenizer} model; its tensors are generated from that file: the RVQ codebooks fold
 * {@code embedding_sum / clamp(cluster_usage, 1e-5)}, the SnakeBeta parameters fold {@code exp(alpha)} and
 * {@code 1 / (exp(beta) + 1e-9)}, and the decoder's convolutions keep their stored layouts. A convolution stored as
 * PyTorch weight-norm parameters ({@code weight_g} / {@code weight_v}) is folded into the direction-scaled kernel
 * before it is planned.
 *
 * <p>The speaker encoder's {@code speaker_encoder.*} tensors become the {@code a.enc} blocks and the final
 * {@code mm.a.fc} projection the audio encoder half of the same file reads, and the file declares
 * {@code clip.audio.projector_type = qwen3tts_spkenc}. The CustomVoice and VoiceDesign checkpoints carry none of
 * them and export the generator alone; a checkpoint with only part of the encoder has those tensors dropped, since
 * the file would otherwise declare an encoder it does not hold. Every method is a public static hook reference.
 */
public final class HfQwen3TtsGenHooks {

    /** The {@code clip.gen.audio.projector_type} of the generator this hook converts. */
    public static final String PROJECTOR_TYPE = "qwen3tts_gen";

    /** The acoustic codebook pages of the code predictor (qwen3tts.py:211). */
    private static final int CODEBOOKS = 15;

    /** The ECAPA-TDNN stages, which the speaker encoder's block numbers must fit (qwen3tts.py:235-240). */
    private static final int SPEAKER_ENCODER_STAGES = 4;

    /** The checkpoint tensors of the generator's code predictor. */
    private static final String SPEAKER_ENCODER_PREFIX = "speaker_encoder.";
    /** The tensors every voice-clone checkpoint carries: the stem, the aggregation, both pooling ends and the fc. */
    private static final String[] SPEAKER_ENCODER_ANCHORS = {"speaker_encoder.blocks.0.conv.weight",
            "speaker_encoder.mfa.conv.weight", "speaker_encoder.asp.tdnn.conv.weight",
            "speaker_encoder.asp.conv.weight", "speaker_encoder.fc.weight"};
    private static final String CODEC_EMBEDDING = "talker.model.codec_embedding.weight";
    private static final String CODE_PREFIX = "talker.code_predictor.";
    private static final String CODE_NORM = "talker.code_predictor.model.norm.weight";
    private static final String CODE_PROJ = "talker.code_predictor.small_to_mtp_projection.";
    private static final String CODE_EMBD = "talker.code_predictor.model.codec_embedding.";
    private static final String CODE_HEAD = "talker.code_predictor.lm_head.";
    private static final String CODE_LAYERS = "talker.code_predictor.model.layers.";

    /** The code2wav decoder's side files (qwen3tts.py:261-268, 360-366). */
    private static final String WAV_CONFIG = "speech_tokenizer/config.json";
    private static final String WAV_WEIGHTS = "speech_tokenizer/model.safetensors";
    private static final String WAV_CONFIG_KEY = "decoder_config";

    /** Per-load state keys of the buffered codebook pages. */
    private static final String EMBEDDING_STATE = "qwen3tts.gen.code.embd";
    private static final String HEAD_STATE = "qwen3tts.gen.code.head";

    /** The code predictor's layer parts (qwen3tts.py:196-208). */
    private static final Map<String, String> LAYER_TENSORS = layerTensors();

    /** The projection's bias suffix. */
    private static final String PROJ_BIAS = ".bias";

    private HfQwen3TtsGenHooks() {
    }

    private static Map<String, String> layerTensors() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("input_layernorm", "A_GEN_CODE_ATTN_NORM");
        m.put("self_attn.q_proj", "A_GEN_CODE_ATTN_Q");
        m.put("self_attn.q_norm", "A_GEN_CODE_ATTN_Q_NORM");
        m.put("self_attn.k_proj", "A_GEN_CODE_ATTN_K");
        m.put("self_attn.k_norm", "A_GEN_CODE_ATTN_K_NORM");
        m.put("self_attn.v_proj", "A_GEN_CODE_ATTN_V");
        m.put("self_attn.o_proj", "A_GEN_CODE_ATTN_OUT");
        m.put("post_attention_layernorm", "A_GEN_CODE_FFN_NORM");
        m.put("mlp.gate_proj", "A_GEN_CODE_FFN_GATE");
        m.put("mlp.up_proj", "A_GEN_CODE_FFN_UP");
        m.put("mlp.down_proj", "A_GEN_CODE_FFN_DOWN");
        return Collections.unmodifiableMap(m);
    }

    // ------------------------------------------------------------------ configuration

    /**
     * {@code Qwen3TTSSpeakerEncoderModel.__init__} (qwen3tts.py:216-225): the projector's text width is the talker's
     * hidden size, which the converter exposes as the flattened {@code hidden_size}.
     *
     * @throws GgufFormatException when the checkpoint has no {@code talker_config.hidden_size}
     */
    public static HfConfig fixup(HfConfig cfg) {
        JsonView talker = cfg.raw().optObj("talker_config");
        if (talker.isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": a Qwen3 TTS generator requires a 'talker_config'"
                    + " object");
        }
        int hidden = requiredInt(talker, "hidden_size", "the talker hidden size");
        Map<String, Object> overrides = new LinkedHashMap<>();
        overrides.put("hidden_size", hidden);
        return cfg.withFlat(overrides);
    }

    /**
     * {@code Qwen3TTSSpeakerEncoderModel.set_gguf_parameters} (qwen3tts.py:231-259): the ECAPA speaker encoder
     * half (fixed geometry, mel front end of 128 bins) and the generator half's code predictor keys. The code2wav
     * hyperparameters are hardcoded on the loader side (qwen3tts.py:259).
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView predictor = predictorConfig(cfg);
        kv.set(GgufKeys.CLIP_HAS_GEN_AUDIO_ENCODER, true);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, PROJECTOR_TYPE);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_PROJECTION_DIM, HfMmprojHooks.textWidth(cfg));
        kv.set(GgufKeys.CLIP_GEN_AUDIO_EMBEDDING_LENGTH, requiredInt(predictor, "hidden_size",
                "the code predictor hidden size"));
        kv.set(GgufKeys.CLIP_GEN_AUDIO_FEED_FORWARD_LENGTH, requiredInt(predictor, "intermediate_size",
                "the code predictor feed-forward width"));
        int blocks = requiredInt(predictor, "num_hidden_layers", "the code predictor block count");
        kv.set(GgufKeys.CLIP_GEN_AUDIO_BLOCK_COUNT, blocks);
        int heads = requiredInt(predictor, "num_attention_heads", "the code predictor head count");
        kv.set(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT, heads);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT_KV,
                optInt(predictor, "num_key_value_heads", heads, "the code predictor key/value head count"));
        Object eps = predictor.raw("rms_norm_eps");
        if (!(eps instanceof Number) || eps instanceof Boolean) {
            throw missing(predictor, "rms_norm_eps");
        }
        kv.set(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_LAYER_NORM_EPSILON, ((Number) eps).floatValue());
        speakerMeta(ctx, cfg, kv);
    }

    /**
     * The qwen3tts speaker encoder's keys (qwen3tts.py:231-247), written when the checkpoint carries the encoder.
     * They describe the ECAPA-TDNN backbone the clip file's audio encoder half reads: the Slaney mel front-end
     * (24 kHz, 1024-point FFT, 128 bins) and the three SE-Res2Net stages.
     */
    private static void speakerMeta(HfLoadContext ctx, HfConfig cfg, HfMetadataBuilder kv) {
        if (!hasSpeakerEncoder(ctx)) {
            return;
        }
        kv.set(GgufKeys.CLIP_HAS_AUDIO_ENCODER, true);
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, "qwen3tts_spkenc");
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTION_DIM, HfMmprojHooks.textWidth(cfg));
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, 128);
        kv.set(GgufKeys.CLIP_AUDIO_BLOCK_COUNT, 3);
        // ECAPA-TDNN has no attention or feed-forward stack; these are placeholders the loader checks only for
        // presence (qwen3tts.py:240-247)
        kv.set(GgufKeys.CLIP_AUDIO_EMBEDDING_LENGTH, 1536);
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_HEAD_COUNT, 1);
        kv.set(GgufKeys.CLIP_AUDIO_FEED_FORWARD_LENGTH, 1536);
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, 1e-5f);
    }

    /**
     * Whether the checkpoint carries the voice-clone encoder: its projection, the ECAPA-TDNN stem, the aggregation
     * convolution and both attentive-pooling convolutions are all needed, and a checkpoint that has only some of
     * them is not a complete encoder (the CustomVoice and VoiceDesign variants ship none at all).
     */
    private static boolean hasSpeakerEncoder(HfLoadContext ctx) {
        for (String name : SPEAKER_ENCODER_ANCHORS) {
            if (!ctx.index().has(name)) {
                return false;
            }
        }
        return true;
    }

    /**
     * The block count of the projector view: the larger of the code predictor's layer count and the four ECAPA-TDNN
     * stages, so that the tensor name map covers both towers' numbered tensors.
     */
    public static int blockCount(HfConfig cfg) {
        return Math.max(SPEAKER_ENCODER_STAGES,
                requiredInt(predictorConfig(cfg), "num_hidden_layers", "the code predictor block count"));
    }

    // ------------------------------------------------------------------ name filter

    /**
     * {@code Qwen3TTSSpeakerEncoderModel.filter_tensors} (qwen3tts.py:287-298): the code predictor, the codebook-0
     * embedding it feeds back to the talker and the speaker encoder survive; everything else is left out.
     */
    public static HfNameFilter filter(HfLoadContext ctx) {
        return (name, base) -> {
            if (name.equals(CODEC_EMBEDDING) || name.startsWith(CODE_PREFIX)
                    || name.startsWith(SPEAKER_ENCODER_PREFIX)) {
                return base.apply(name);
            }
            return null;
        };
    }

    // ------------------------------------------------------------------ the code predictor

    /**
     * {@code Qwen3TTSSpeakerEncoderModel.modify_tensors} (qwen3tts.py:300-355) for the generator tensors: the codec-0
     * embedding, the output norm, the talker-to-predictor projection, the stacked codebook and LM-head pages, and the
     * predictor layers.
     *
     * @throws GgufFormatException for a kept name the generator has no tensor for
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.equals(CODEC_EMBEDDING)) {
            out.add(ctx.names().ggufName("A_GEN_CODE_OUT_EMBD", -1, ".weight"), src);
            return;
        }
        if (hfName.equals(CODE_NORM)) {
            out.add(ctx.names().ggufName("A_GEN_CODE_OUTPUT_NORM", -1, ".weight"), src);
            return;
        }
        if (hfName.startsWith(CODE_PROJ)) {
            String suffix = hfName.substring(CODE_PROJ.length() - 1);
            if (!suffix.equals(".weight") && !suffix.equals(PROJ_BIAS)) {
                throw new GgufFormatException("qwen3tts generator: unsupported projection tensor '" + hfName + "'");
            }
            out.add(ctx.names().ggufName("A_GEN_CODE_PROJ_IN", -1, suffix), src);
            return;
        }
        if (hfName.startsWith(CODE_EMBD)) {
            page(ctx, EMBEDDING_STATE, "A_GEN_CODE_EMBD", "codebook embedding", hfName, CODE_EMBD, src, out);
            return;
        }
        if (hfName.startsWith(CODE_HEAD)) {
            page(ctx, HEAD_STATE, "A_GEN_CODE_HEAD", "LM head", hfName, CODE_HEAD, src, out);
            return;
        }
        if (hfName.startsWith(CODE_LAYERS)) {
            layer(ctx, hfName, src, out);
            return;
        }
        if (hfName.startsWith(SPEAKER_ENCODER_PREFIX)) {
            if (hasSpeakerEncoder(ctx)) {
                speaker(ctx, hfName, src, out);
            } else {
                out.drop(hfName, "the checkpoint carries no complete speaker encoder");
            }
            return;
        }
        throw new GgufFormatException("qwen3tts generator: the checkpoint tensor '" + hfName
                + "' has no generator tensor");
    }

    /** One buffered codebook page: its source, and whether the stacked tensor has been planned. */
    private static final class Pages {
        final Map<Integer, RowSource> pages = new LinkedHashMap<>();
        boolean emitted;
    }

    /** Buffers one page and plans the stacked 3-D tensor once all {@value #CODEBOOKS} pages are present. */
    private static void page(HfLoadContext ctx, String stateKey, String tensor, String what, String hfName,
                             String prefix, RowSource src, TensorSink out) {
        String rest = hfName.substring(prefix.length());
        int dot = rest.indexOf('.');
        if (dot <= 0 || !rest.substring(dot).equals(".weight")) {
            throw new GgufFormatException("qwen3tts generator: the " + what + " page '" + hfName
                    + "' is not named '<index>.weight'");
        }
        int index = parseIndex(rest.substring(0, dot), hfName);
        if (index >= CODEBOOKS) {
            throw new GgufFormatException("qwen3tts generator: the " + what + " page index " + index
                    + " of '" + hfName + "' is outside the " + CODEBOOKS + " pages of the reference");
        }
        Pages state = ctx.state(stateKey, Pages.class, Pages::new);
        if (state.pages.put(index, src) != null) {
            throw new GgufFormatException("qwen3tts generator: the " + what + " page " + index + " appears twice"
                    + " ('" + hfName + "')");
        }
        if (state.pages.size() < CODEBOOKS) {
            return;
        }
        List<RowSource> ordered = new ArrayList<>(CODEBOOKS);
        for (int i = 0; i < CODEBOOKS; i++) {
            ordered.add(state.pages.get(i));
        }
        out.add(ctx.names().ggufName(tensor, -1, ".weight"), stack(ordered, what));
        state.pages.clear();
        state.emitted = true;
    }

    /** {@code torch.stack(pages, dim=0)}: the source shapes are PyTorch order, so the GGUF shape is reversed. */
    private static RowSource stack(List<RowSource> pages, String what) {
        long[] shape = pages.get(0).shape();
        if (shape.length != 2) {
            throw new GgufFormatException("qwen3tts generator: each " + what + " page must be 2-D, got "
                    + java.util.Arrays.toString(shape));
        }
        for (int i = 1; i < pages.size(); i++) {
            if (!java.util.Arrays.equals(shape, pages.get(i).shape())) {
                throw new GgufFormatException("qwen3tts generator: the " + what + " page " + i + " has shape "
                        + java.util.Arrays.toString(pages.get(i).shape()) + ", page 0 has "
                        + java.util.Arrays.toString(shape));
            }
        }
        return ElementTransforms.reshape(RowTransforms.concatRows(pages), pages.size(), shape[0], shape[1]);
    }

    /** One {@code talker.code_predictor.model.layers.<bid>.<key>.weight} tensor (qwen3tts.py:338-345). */
    private static void layer(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        String rest = hfName.substring(CODE_LAYERS.length());
        int dot = rest.indexOf('.');
        if (dot <= 0) {
            throw new GgufFormatException("qwen3tts generator: the predictor layer tensor '" + hfName
                    + "' has no layer number");
        }
        int bid = parseIndex(rest.substring(0, dot), hfName);
        String keyAndSuffix = rest.substring(dot + 1);
        if (!keyAndSuffix.endsWith(".weight")) {
            throw new GgufFormatException("qwen3tts generator: the predictor layer tensor '" + hfName
                    + "' is not a weight");
        }
        String key = keyAndSuffix.substring(0, keyAndSuffix.length() - ".weight".length());
        String tensor = LAYER_TENSORS.get(key);
        if (tensor == null) {
            throw new GgufFormatException("qwen3tts generator: the predictor layer tensor '" + hfName
                    + "' has no generator tensor for its part '" + key + "'");
        }
        out.add(ctx.names().ggufName(tensor, bid, ".weight"), src);
    }

    // ------------------------------------------------------------------ the speaker encoder

    /**
     * The ECAPA-TDNN speaker encoder tensors (tensor_mapping.py:2520-2900): the stem {@code blocks.0.conv}, each
     * SE-Res2Net block's TDNNs, squeeze-and-excitation pair and seven Res2Net convolutions, the multi-layer feature
     * aggregation, the attentive-statistics-pooling pair and the final projection.
     */
    private static void speaker(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        String suffix = suffixOf(hfName);
        if (!suffix.equals(".weight") && !suffix.equals(".bias")) {
            throw new GgufFormatException("qwen3tts speaker encoder: the tensor '" + hfName
                    + "' is neither a weight nor a bias");
        }
        if (hfName.equals("speaker_encoder.fc" + suffix)) {
            add(ctx, out, "A_MMPROJ_FC", -1, suffix, src);
            return;
        }
        if (hfName.equals("speaker_encoder.mfa.conv" + suffix)) {
            add(ctx, out, "A_ENC_CONV_OUT", -1, suffix, src);
            return;
        }
        if (hfName.equals("speaker_encoder.asp.conv" + suffix)) {
            add(ctx, out, "A_ENC_ASP_ATTN", -1, suffix, src);
            return;
        }
        if (hfName.equals("speaker_encoder.asp.tdnn.conv" + suffix)) {
            add(ctx, out, "A_ENC_ASP_TDNN", -1, suffix, src);
            return;
        }
        String blocks = "speaker_encoder.blocks.";
        if (!hfName.startsWith(blocks)) {
            throw new GgufFormatException("qwen3tts speaker encoder: the checkpoint tensor '" + hfName
                    + "' has no speaker-encoder tensor");
        }
        String rest = hfName.substring(blocks.length());
        int dot = rest.indexOf('.');
        if (dot <= 0) {
            throw new GgufFormatException("qwen3tts speaker encoder: the tensor '" + hfName + "' has no block index");
        }
        int bid = parseIndex(rest.substring(0, dot), hfName);
        String key = rest.substring(dot + 1);
        if (key.equals("conv" + suffix)) {
            add(ctx, out, "A_ENC_CONV1D", bid, suffix, src);
            return;
        }
        if (key.equals("tdnn1.conv" + suffix)) {
            add(ctx, out, "A_ENC_CONV_PW1", bid, suffix, src);
            return;
        }
        if (key.equals("tdnn2.conv" + suffix)) {
            add(ctx, out, "A_ENC_CONV_PW2", bid, suffix, src);
            return;
        }
        if (key.equals("se_block.conv1" + suffix)) {
            add(ctx, out, "A_ENC_SE_CONV1", bid, suffix, src);
            return;
        }
        if (key.equals("se_block.conv2" + suffix)) {
            add(ctx, out, "A_ENC_SE_CONV2", bid, suffix, src);
            return;
        }
        String res2 = "res2net_block.blocks.";
        if (key.startsWith(res2)) {
            String inner = key.substring(res2.length());
            int innerDot = inner.indexOf('.');
            if (innerDot <= 0) {
                throw new GgufFormatException("qwen3tts speaker encoder: the tensor '" + hfName
                        + "' has no Res2Net block index");
            }
            int xid = parseIndex(inner.substring(0, innerDot), hfName);
            String template = HfNameTables.tensorNames().get("A_ENC_CONV_RES2");
            String name = template.replace("{bid}", Integer.toString(bid))
                    .replace("{xid}", Integer.toString(xid)) + suffix;
            out.add(name, src);
            return;
        }
        throw new GgufFormatException("qwen3tts speaker encoder: the checkpoint tensor '" + hfName
                + "' has no speaker-encoder tensor");
    }

    /** The {@code .weight} or {@code .bias} suffix of a checkpoint tensor name. */
    private static String suffixOf(String name) {
        int dot = name.lastIndexOf('.');
        return dot < 0 ? "" : name.substring(dot);
    }

    private static void add(HfLoadContext ctx, TensorSink out, String tensor, int bid, String suffix, RowSource src) {
        out.add(ctx.names().ggufName(tensor, bid, suffix), src);
    }

    // ------------------------------------------------------------------ the code2wav decoder

    /**
     * {@code Qwen3TTSSpeakerEncoderModel._generate_code2wav_tensors} (qwen3tts.py:357-479): the RVQ codebooks, the
     * pre-convolution, the sliding-window pre-transformer, the upsample stages and the DAC decoder, read from
     * {@code speech_tokenizer/model.safetensors}.
     */
    public static void extra(HfLoadContext ctx, TensorSink out) throws IOException {
        requireEmitted(ctx, EMBEDDING_STATE, "codebook embedding");
        requireEmitted(ctx, HEAD_STATE, "LM head");
        JsonView decoder = decoderConfig(ctx.config());
        SafetensorsFile file = ctx.checkpoint().openSideFile(WAV_WEIGHTS);
        int tfmLayers = requiredInt(decoder, "num_hidden_layers", "the code2wav transformer block count");
        int upsample = listSize(decoder, "upsampling_ratios", "the upsample stage count");
        int dacBlocks = listSize(decoder, "upsample_rates", "the DAC block count");
        quantizer(ctx, out, file);
        preConv(ctx, out, file);
        transformer(ctx, out, file, tfmLayers);
        upsample(ctx, out, file, upsample);
        dac(ctx, out, file, dacBlocks);
    }

    private static void requireEmitted(HfLoadContext ctx, String stateKey, String what) {
        Pages state = ctx.state(stateKey, Pages.class, Pages::new);
        if (!state.emitted) {
            throw new GgufFormatException("qwen3tts generator: the code predictor's " + what + " pages are"
                    + " incomplete: the checkpoint holds " + state.pages.size() + " of " + CODEBOOKS + " pages ("
                    + state.pages.keySet() + ")");
        }
    }

    /** The RVQ codebooks and the quantizer's 1x1 convolutions (qwen3tts.py:385-391). */
    private static void quantizer(HfLoadContext ctx, TensorSink out, SafetensorsFile file) throws IOException {
        String first = "decoder.quantizer.rvq_first";
        String rest = "decoder.quantizer.rvq_rest";
        out.add(weightName(ctx, "A_GEN_WAV_QUANT_FIRST_IN"),
                ElementTransforms.squeezeConv(source(file, first + ".input_proj.weight")));
        out.add(weightName(ctx, "A_GEN_WAV_QUANT_FIRST_OUT"),
                ElementTransforms.squeezeConv(source(file, first + ".output_proj.weight")));
        out.add(weightName(ctx, "A_GEN_WAV_QUANT_FIRST_CB"), codebook(file, first, 1));
        out.add(weightName(ctx, "A_GEN_WAV_QUANT_REST_IN"),
                ElementTransforms.squeezeConv(source(file, rest + ".input_proj.weight")));
        out.add(weightName(ctx, "A_GEN_WAV_QUANT_REST_OUT"),
                ElementTransforms.squeezeConv(source(file, rest + ".output_proj.weight")));
        out.add(weightName(ctx, "A_GEN_WAV_QUANT_REST_CB"), codebook(file, rest, CODEBOOKS));
    }

    /**
     * {@code rvq_codebook}: {@code embedding_sum / clamp(cluster_usage, 1e-5)}, stacked when the quantizer has several
     * layers (qwen3tts.py:374-381).
     */
    private static RowSource codebook(SafetensorsFile file, String prefix, int layers) throws IOException {
        List<RowSource> books = new ArrayList<>(layers);
        for (int i = 0; i < layers; i++) {
            String at = prefix + ".vq.layers." + i + "._codebook.";
            RowSource sum = source(file, at + "embedding_sum");
            RowSource usage = source(file, at + "cluster_usage");
            float[] values = HfGenHooks.floats(sum, at + "embedding_sum");
            float[] counts = HfGenHooks.floats(usage, at + "cluster_usage");
            long[] shape = sum.shape();
            long rows = shape.length < 2 ? 1 : shape[0];
            long inner = values.length / rows;
            if (inner * rows != values.length || counts.length != rows) {
                throw new GgufFormatException("qwen3tts generator: the codebook '" + at + "' holds "
                        + values.length + " values and " + counts.length + " cluster counts for shape "
                        + java.util.Arrays.toString(shape));
            }
            for (int r = 0; r < rows; r++) {
                float divisor = Math.max(counts[r], 1e-5f);
                for (long d = 0; d < inner; d++) {
                    int at2 = (int) (r * inner + d);
                    values[at2] = values[at2] / divisor;
                }
            }
            books.add(RopeTensors.fromFloats(shape, values));
        }
        return layers == 1 ? books.get(0) : stack(books, "codebook");
    }

    /** The causal pre-convolution (qwen3tts.py:393-395). */
    private static void preConv(HfLoadContext ctx, TensorSink out, SafetensorsFile file) throws IOException {
        weightBias(ctx, out, file, "A_GEN_WAV_PRE_CONV", "decoder.pre_conv.conv");
    }

    /** The sliding-window pre-transformer (qwen3tts.py:397-420). */
    private static void transformer(HfLoadContext ctx, TensorSink out, SafetensorsFile file, int layers)
            throws IOException {
        weightBias(ctx, out, file, "A_GEN_WAV_TFM_IN_PROJ", "decoder.pre_transformer.input_proj");
        weightBias(ctx, out, file, "A_GEN_WAV_TFM_OUT_PROJ", "decoder.pre_transformer.output_proj");
        out.add(weightName(ctx, "A_GEN_WAV_TFM_OUTPUT_NORM"), weight(file, "decoder.pre_transformer.norm"));
        Map<String, String> parts = new LinkedHashMap<>();
        parts.put("input_layernorm.weight", "A_GEN_WAV_TFM_ATTN_NORM");
        parts.put("self_attn.q_proj.weight", "A_GEN_WAV_TFM_ATTN_Q");
        parts.put("self_attn.k_proj.weight", "A_GEN_WAV_TFM_ATTN_K");
        parts.put("self_attn.v_proj.weight", "A_GEN_WAV_TFM_ATTN_V");
        parts.put("self_attn.o_proj.weight", "A_GEN_WAV_TFM_ATTN_OUT");
        parts.put("self_attn_layer_scale.scale", "A_GEN_WAV_TFM_ATTN_SCALE");
        parts.put("post_attention_layernorm.weight", "A_GEN_WAV_TFM_FFN_NORM");
        parts.put("mlp.gate_proj.weight", "A_GEN_WAV_TFM_FFN_GATE");
        parts.put("mlp.up_proj.weight", "A_GEN_WAV_TFM_FFN_UP");
        parts.put("mlp.down_proj.weight", "A_GEN_WAV_TFM_FFN_DOWN");
        parts.put("mlp_layer_scale.scale", "A_GEN_WAV_TFM_FFN_SCALE");
        for (int bid = 0; bid < layers; bid++) {
            for (Map.Entry<String, String> part : parts.entrySet()) {
                out.add(ctx.names().ggufName(part.getValue(), bid, ".weight"),
                        source(file, "decoder.pre_transformer.layers." + bid + "." + part.getKey()));
            }
        }
    }

    /** The two causal ConvTranspose1d + ConvNeXt stages (qwen3tts.py:422-438). */
    private static void upsample(HfLoadContext ctx, TensorSink out, SafetensorsFile file, int stages)
            throws IOException {
        for (int bid = 0; bid < stages; bid++) {
            String at = "decoder.upsample." + bid + ".";
            out.add(ctx.names().ggufName("A_GEN_WAV_UP_CONV", bid, ".weight"), weight(file, at + "0.conv"));
            out.add(ctx.names().ggufName("A_GEN_WAV_UP_CONV", bid, ".bias"), source(file, at + "0.conv.bias"));
            out.add(ctx.names().ggufName("A_GEN_WAV_UP_DWCONV", bid, ".weight"),
                    weight(file, at + "1.dwconv.conv"));
            out.add(ctx.names().ggufName("A_GEN_WAV_UP_DWCONV", bid, ".bias"),
                    source(file, at + "1.dwconv.conv.bias"));
            out.add(ctx.names().ggufName("A_GEN_WAV_UP_NORM", bid, ".weight"), source(file, at + "1.norm.weight"));
            out.add(ctx.names().ggufName("A_GEN_WAV_UP_NORM", bid, ".bias"), source(file, at + "1.norm.bias"));
            out.add(ctx.names().ggufName("A_GEN_WAV_UP_PW1", bid, ".weight"), source(file, at + "1.pwconv1.weight"));
            out.add(ctx.names().ggufName("A_GEN_WAV_UP_PW1", bid, ".bias"), source(file, at + "1.pwconv1.bias"));
            out.add(ctx.names().ggufName("A_GEN_WAV_UP_PW2", bid, ".weight"), source(file, at + "1.pwconv2.weight"));
            out.add(ctx.names().ggufName("A_GEN_WAV_UP_PW2", bid, ".bias"), source(file, at + "1.pwconv2.bias"));
            out.add(ctx.names().ggufName("A_GEN_WAV_UP_GAMMA", bid), source(file, at + "1.gamma"));
        }
    }

    /** The DAC decoder: entry convolution, blocks and the final SnakeBeta and convolution (qwen3tts.py:441-479). */
    private static void dac(HfLoadContext ctx, TensorSink out, SafetensorsFile file, int blocks) throws IOException {
        weightBias(ctx, out, file, "A_GEN_WAV_DAC_ENTRY", "decoder.decoder.0.conv");
        for (int bid = 0; bid < blocks; bid++) {
            int py = bid + 1;
            String block = "decoder.decoder." + py + ".";
            out.add(ctx.names().ggufName("A_GEN_WAV_DAC_UP_SNAKE", bid, ".alpha"),
                    snake(file, block + "block.0.alpha", true));
            out.add(ctx.names().ggufName("A_GEN_WAV_DAC_UP_SNAKE", bid, ".beta"),
                    snake(file, block + "block.0.beta", false));
            weightBias(ctx, out, file, "A_GEN_WAV_DAC_UP_CONV", block + "block.1.conv", bid);
            for (int xid = 0; xid < 3; xid++) {
                residual(ctx, out, file, bid, block + "block." + (xid + 2) + ".", xid);
            }
        }
        out.add(baseName(ctx, "A_GEN_WAV_DAC_POST_SNAKE") + ".alpha",
                snake(file, "decoder.decoder.5.alpha", true));
        out.add(baseName(ctx, "A_GEN_WAV_DAC_POST_SNAKE") + ".beta",
                snake(file, "decoder.decoder.5.beta", false));
        weightBias(ctx, out, file, "A_GEN_WAV_DAC_POST_CONV", "decoder.decoder.6.conv");
    }

    /** One DAC residual unit: SnakeBeta, dilated convolution, SnakeBeta, pointwise convolution. */
    private static void residual(HfLoadContext ctx, TensorSink out, SafetensorsFile file, int bid, String at,
                                 int xid) throws IOException {
        out.add(residualName("A_GEN_WAV_DAC_RES_ACT1", bid, xid, ".alpha"), snake(file, at + "act1.alpha", true));
        out.add(residualName("A_GEN_WAV_DAC_RES_ACT1", bid, xid, ".beta"), snake(file, at + "act1.beta", false));
        out.add(residualName("A_GEN_WAV_DAC_RES_CONV1", bid, xid, ".weight"), weight(file, at + "conv1.conv"));
        out.add(residualName("A_GEN_WAV_DAC_RES_CONV1", bid, xid, ".bias"), source(file, at + "conv1.conv.bias"));
        out.add(residualName("A_GEN_WAV_DAC_RES_ACT2", bid, xid, ".alpha"), snake(file, at + "act2.alpha", true));
        out.add(residualName("A_GEN_WAV_DAC_RES_ACT2", bid, xid, ".beta"), snake(file, at + "act2.beta", false));
        out.add(residualName("A_GEN_WAV_DAC_RES_CONV2", bid, xid, ".weight"), weight(file, at + "conv2.conv"));
        out.add(residualName("A_GEN_WAV_DAC_RES_CONV2", bid, xid, ".bias"), source(file, at + "conv2.conv.bias"));
    }

    // ------------------------------------------------------------------ side tensor helpers

    private static JsonView decoderConfig(HfConfig cfg) throws IOException {
        Path file = cfg.sideFile(WAV_CONFIG);
        if (!java.nio.file.Files.isRegularFile(file)) {
            throw new GgufFormatException("qwen3tts generator: the checkpoint has no '" + WAV_CONFIG
                    + "'; the code2wav decoder's geometry comes from it");
        }
        JsonView decoder = HfBertVariantHooks.readJsonView(file).optObj(WAV_CONFIG_KEY);
        if (decoder.isEmpty()) {
            throw new GgufFormatException(file + ": '" + WAV_CONFIG_KEY + "' is missing; the code2wav decoder's"
                    + " geometry comes from it");
        }
        return decoder;
    }

    /** The required {@code talker_config.code_predictor_config} object. */
    private static JsonView predictorConfig(HfConfig cfg) {
        JsonView predictor = cfg.raw().optObj("talker_config").optObj("code_predictor_config");
        if (predictor.isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": a Qwen3 TTS generator requires"
                    + " 'talker_config.code_predictor_config'");
        }
        return predictor;
    }

    private static RowSource source(SafetensorsFile file, String name) {
        StTensor t = file.tensor(name);
        if (t == null) {
            throw new GgufFormatException("qwen3tts generator: '" + file.path() + "' has no tensor '" + name + "'");
        }
        return StRowSource.of(file, t);
    }

    /**
     * The kernel of a convolution stored under {@code base}: {@code base.weight}, or the PyTorch weight-norm pair
     * {@code base.weight_g} / {@code base.weight_v} (also {@code parametrizations.weight.original0/original1}) folded
     * into {@code g * v / ||v||} over every dimension but the first.
     */
    private static RowSource weight(SafetensorsFile file, String base) throws IOException {
        StTensor direct = file.tensor(base + ".weight");
        if (direct != null) {
            return StRowSource.of(file, direct);
        }
        StTensor g = file.tensor(base + ".weight_g");
        StTensor v = file.tensor(base + ".weight_v");
        if (g == null || v == null) {
            g = file.tensor(base + ".parametrizations.weight.original0");
            v = file.tensor(base + ".parametrizations.weight.original1");
        }
        if (g == null || v == null) {
            throw new GgufFormatException("qwen3tts generator: '" + file.path() + "' has no tensor '" + base
                    + ".weight' (nor its weight-norm parameters)");
        }
        return foldWeightNorm(file, g, v);
    }

    /** {@code weight_norm(weight_v, weight_g)}: each outer row scaled by {@code g / ||v||}. */
    private static RowSource foldWeightNorm(SafetensorsFile file, StTensor g, StTensor v) throws IOException {
        RowSource vs = StRowSource.of(file, v);
        float[] values = HfGenHooks.floats(vs, v.name());
        float[] gains = HfGenHooks.floats(StRowSource.of(file, g), g.name());
        long[] shape = vs.shape();
        long rows = shape.length == 0 ? 1 : shape[0];
        long inner = values.length / rows;
        if (inner * rows != values.length || (gains.length != rows && gains.length != 1)) {
            throw new GgufFormatException("qwen3tts generator: the weight-norm pair '" + v.name() + "' / '"
                    + g.name() + "' does not match shape " + java.util.Arrays.toString(shape));
        }
        for (int r = 0; r < rows; r++) {
            double sum = 0.0;
            for (long i = 0; i < inner; i++) {
                double x = values[(int) (r * inner + i)];
                sum += x * x;
            }
            float norm = (float) Math.sqrt(sum);
            float gain = gains.length == 1 ? gains[0] : gains[r];
            float factor = norm == 0f ? 0f : gain / norm;
            for (long i = 0; i < inner; i++) {
                int at = (int) (r * inner + i);
                values[at] = values[at] * factor;
            }
        }
        return RopeTensors.fromFloats(shape, values);
    }

    /**
     * A SnakeBeta parameter folded the way the graph wants it (qwen3tts.py:370-372): {@code exp(alpha)}, or
     * {@code 1 / (exp(beta) + 1e-9)}.
     */
    private static RowSource snake(SafetensorsFile file, String name, boolean alpha) throws IOException {
        RowSource src = source(file, name);
        float[] values = HfGenHooks.floats(src, name);
        for (int i = 0; i < values.length; i++) {
            double e = Math.exp(values[i]);
            values[i] = alpha ? (float) e : (float) (1.0 / (e + 1e-9));
        }
        return RopeTensors.fromFloats(src.shape(), values);
    }

    private static void weightBias(HfLoadContext ctx, TensorSink out, SafetensorsFile file, String tensor,
                                   String base) throws IOException {
        weightBias(ctx, out, file, tensor, base, -1);
    }

    private static void weightBias(HfLoadContext ctx, TensorSink out, SafetensorsFile file, String tensor,
                                   String base, int bid) throws IOException {
        out.add(ctx.names().ggufName(tensor, bid, ".weight"), weight(file, base));
        out.add(ctx.names().ggufName(tensor, bid, ".bias"), source(file, base + ".bias"));
    }

    private static String weightName(HfLoadContext ctx, String tensor) {
        return ctx.names().ggufName(tensor, -1, ".weight");
    }

    private static String baseName(HfLoadContext ctx, String tensor) {
        return ctx.names().ggufName(tensor, -1);
    }

    /** A DAC residual tensor: the template carries both the block and the residual index. */
    private static String residualName(String tensor, int bid, int xid, String suffix) {
        String template = HfNameTables.tensorNames().get(tensor);
        if (template == null) {
            throw new IllegalArgumentException("unknown converter tensor '" + tensor + "'");
        }
        return template.replace("{bid}", Integer.toString(bid)).replace("{xid}", Integer.toString(xid)) + suffix;
    }

    // ------------------------------------------------------------------ validation

    private static int requiredInt(JsonView view, String key, String what) {
        Object v = view.raw(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw missing(view, key);
        }
        double d = ((Number) v).doubleValue();
        if (d != Math.rint(d) || d < Integer.MIN_VALUE || d > Integer.MAX_VALUE) {
            throw new GgufFormatException(view.where(key) + ": " + what + " must be an integer, got " + v);
        }
        return (int) d;
    }

    private static int optInt(JsonView view, String key, int dflt, String what) {
        return view.has(key) ? requiredInt(view, key, what) : dflt;
    }

    private static int listSize(JsonView view, String key, String what) {
        Object v = view.raw(key);
        if (!(v instanceof List) || ((List<?>) v).isEmpty()) {
            throw new GgufFormatException(view.where(key) + ": " + what + " needs a non-empty array, got "
                    + (v == null ? "nothing" : v));
        }
        return ((List<?>) v).size();
    }

    private static GgufFormatException missing(JsonView view, String key) {
        return new GgufFormatException(view.where(key) + ": the qwen3tts generator requires this key");
    }

    private static int parseIndex(String digits, String where) {
        if (digits.isEmpty() || digits.length() > 9) {
            throw new GgufFormatException("qwen3tts generator: the tensor '" + where + "' has no usable index '"
                    + digits + "'");
        }
        for (int i = 0; i < digits.length(); i++) {
            char c = digits.charAt(i);
            if (c < '0' || c > '9') {
                throw new GgufFormatException("qwen3tts generator: the tensor '" + where + "' has no usable index '"
                        + digits + "'");
            }
        }
        return Integer.parseInt(digits);
    }

    // ------------------------------------------------------------------ forced storage types

    /**
     * {@code Qwen3TTSSpeakerEncoderModel.tensor_force_quant} (qwen3tts.py:270-285): the convolution kernels
     * {@code ggml_conv_1d} runs keep F16, the ConvTranspose1d kernels F32, the code predictor's {@code ffn_down}
     * F32 (its intermediate peaks above the F16 range), and every other tensor the base mmproj rule.
     */
    public static final HfForceType FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.endsWith(".weight") && (ggufName.equals("a.gen.wav.pre_conv.weight")
                || ggufName.equals("a.gen.wav.dac.entry.weight")
                || ggufName.equals("a.gen.wav.dac.post_conv.weight")
                || (ggufName.contains(".up.blk.") && ggufName.endsWith(".dwconv.weight"))
                || (ggufName.contains(".dac.blk.") && (ggufName.endsWith(".conv1.weight")
                        || ggufName.endsWith(".conv2.weight"))))) {
            return GGMLType.F16;
        }
        if (ggufName.endsWith(".conv.weight")
                && (ggufName.contains(".up.blk.") || ggufName.contains(".dac.blk."))) {
            return GGMLType.F32;
        }
        if (ggufName.startsWith("a.gen.code.blk.") && ggufName.endsWith(".ffn_down.weight")) {
            return GGMLType.F32;
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };
}
