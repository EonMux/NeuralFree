package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;

/**
 * The Whisper-family audio towers and their projectors through the Hugging Face loader, ports of llama.cpp {@code
 * WhisperEncoderModel} and its subclasses (LC/conversion/ultravox.py:89-237): the audio tower with its
 * {@code d_model} / {@code encoder_ffn_dim} / {@code encoder_attention_heads} aliases, the conv-stem biases widened
 * to a row, the {@code audio.} projector prefix, the frame-stack factor of the ultravox, voxtral and MERaLiON
 * projectors, and the MERaLiON adapter names. Every method is a public static hook reference.
 */
public final class HfWhisperAudioHooks {

    /** The projector of Qwen2-Audio ({@code VisionProjectorType.QWEN2A}). */
    public static final String QWEN2A = "qwen2a";
    /** The projector of Ultravox ({@code VisionProjectorType.ULTRAVOX}). */
    public static final String ULTRAVOX = "ultravox";
    /** The projector of Voxtral ({@code VisionProjectorType.VOXTRAL}). */
    public static final String VOXTRAL = "voxtral";
    /** The projector of Audio-Flamingo-3 ({@code VisionProjectorType.MUSIC_FLAMINGO}). */
    public static final String MUSIC_FLAMINGO = "musicflamingo";
    /** The projector of MERaLiON-2 ({@code VisionProjectorType.MERALION}). */
    public static final String MERALION = "meralion";
    /** The default layer-norm epsilon of the Whisper tower (ultravox.py:113). */
    private static final double DEFAULT_EPS = 1e-5;

    private HfWhisperAudioHooks() {
    }

    // ------------------------------------------------------------------ filters

    /** The projector carries the {@code audio.} prefix, so its names cannot clash with a vision tower (ultravox.py:115-120). */
    public static final HfNameFilter WHISPER_FILTER = (name, base) ->
            name.startsWith("multi_modal_projector") ? base.apply("audio." + name) : base.apply(name);

    /** {@code text_decoder.} drops, {@code speech_encoder.} renamed to {@code audio_tower.}; the projector names
     * carry the {@code audio.} prefix of the inherited Whisper filter (ultravox.py:163-186). */
    public static final HfNameFilter MERALION_FILTER = (name, base) -> {
        if (name.startsWith("text_decoder.")) {
            return null;
        }
        if (name.startsWith("speech_encoder.")) {
            name = "audio_tower." + name.substring("speech_encoder.".length());
        }
        if (name.startsWith("multi_modal_projector")) {
            name = "audio." + name;
        }
        return base.apply(name);
    };

    // ------------------------------------------------------------------ forced storage types

    /** The conv-stem weights keep F16 (ultravox.py:122-125), the patch kernels the base rule. */
    public static final HfForceType CONV_F16 = (ggufName, nDims, source, outType) ->
            ggufName.contains(".conv") && ggufName.endsWith(".weight") ? GGMLType.F16
                    : HfMmprojHooks.patchKernelForce(ggufName, outType);

    /** Audio-Flamingo-3 is trained in BF16, so its conv-stem weights keep F32 (ultravox.py:219-225). */
    public static final HfForceType CONV_F32 = (ggufName, nDims, source, outType) ->
            ggufName.contains(".conv") && ggufName.endsWith(".weight") ? GGMLType.F32
                    : HfMmprojHooks.patchKernelForce(ggufName, outType);

    // ------------------------------------------------------------------ metadata

    /**
     * The base audio keys with the Whisper config aliases (ultravox.py:97-102), the {@code qwen2a} projector, the
     * mel-bin count and the tower epsilon (ultravox.py:104-118).
     */
    public static void qwen2aMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        whisperMeta(ctx, kv, QWEN2A, null);
    }

    /** {@link #qwen2aMeta} with the {@code ultravox} projector and the root {@code stack_factor} (ultravox.py:136-143). */
    public static void ultravoxMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        whisperMeta(ctx, kv, ULTRAVOX, null);
        if (!cfg.raw().has("stack_factor")) {
            throw new GgufFormatException(cfg.source() + ": 'stack_factor' of the checkpoint is missing;"
                    + " clip.audio.projector.stack_factor follows it (ultravox.py:142)");
        }
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_STACK_FACTOR, cfg.raw().i32("stack_factor"));
    }

    /** {@link #qwen2aMeta} with the {@code voxtral} projector and the stack factor 4 (ultravox.py:223-229). */
    public static void voxtralMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        whisperMeta(ctx, kv, VOXTRAL, 4);
    }

    /** {@link #qwen2aMeta} with the {@code musicflamingo} projector (ultravox.py:232-236). */
    public static void musicFlamingoMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        whisperMeta(ctx, kv, MUSIC_FLAMINGO, null);
    }

    private static void whisperMeta(HfLoadContext ctx, HfMetadataBuilder kv, String projector, Integer stackFactor)
            throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.audioMeta(ctx, kv, HfMmprojHooks.audioInt(cfg, "hidden_size", "d_model"),
                HfMmprojHooks.audioInt(cfg, "intermediate_size", "encoder_ffn_dim"),
                HfMmprojHooks.audioInt(cfg, "num_attention_heads", "encoder_attention_heads"),
                HfMmprojHooks.audioBlocks(cfg));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, projector);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, HfMmprojHooks.audioInt(cfg, "num_mel_bins"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, audioEps(cfg));
        if (stackFactor != null) {
            kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_STACK_FACTOR, stackFactor);
        }
    }

    /** The tower epsilon: {@code layer_norm_eps}, default 1e-5. */
    private static double audioEps(HfConfig cfg) {
        Object v = HfMmprojHooks.audioFind(cfg, "layer_norm_eps").orElse(null);
        return v instanceof Number ? ((Number) v).doubleValue() : DEFAULT_EPS;
    }

    /**
     * The MERaLiON keys (ultravox.py:146-176): the base audio keys over {@code speech_config}, the {@code meralion}
     * projector, the mel bins, the epsilon and the stack factor of {@code speech_mlp_scale_factor}, default 15.
     */
    public static void meralionMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView speech = speechConfig(cfg);
        HfMmprojHooks.audioMeta(ctx, kv, HfMmprojHooks.intOf(cfg, speech, "speech_config", "hidden_size"),
                HfMmprojHooks.intOf(cfg, speech, "speech_config", "intermediate_size"),
                HfMmprojHooks.intOf(cfg, speech, "speech_config", "num_attention_heads"),
                HfMmprojHooks.blocksOf(cfg, speech, "speech_config"));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, MERALION);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, speech.i32("num_mel_bins"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, speech.optF64("layer_norm_eps", DEFAULT_EPS));
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_STACK_FACTOR, cfg.raw().has("speech_mlp_scale_factor")
                ? cfg.raw().i32("speech_mlp_scale_factor") : 15);
    }

    /** The {@code speech_config} of MERaLiON ({@code get_audio_config}, ultravox.py:150-152). */
    private static JsonView speechConfig(HfConfig cfg) {
        JsonView root = cfg.raw();
        if (root.has("speech_config")) {
            return root.obj("speech_config");
        }
        throw new GgufFormatException(cfg.source() + ": the checkpoint has no 'speech_config'; the MERaLiON"
                + " projector converts its speech tower");
    }

    /** The MERaLiON speech tower's block count ({@code find_aparam(n_block_keys)} over {@code speech_config}). */
    public static int meralionBlocks(HfConfig cfg) {
        return HfMmprojHooks.blocksOf(cfg, speechConfig(cfg), "speech_config");
    }

    // ------------------------------------------------------------------ tensors

    /**
     * The Whisper tower (ultravox.py:128-132): the conv-stem biases widened from {@code [n]} to a one-wide row,
     * everything else under its mapped name.
     */
    public static void qwen2aTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (convBias(hfName)) {
            src = ElementTransforms.reshape(src, src.shape()[0], 1);
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** True for the two conv-stem biases of the Whisper front end. */
    private static boolean convBias(String hfName) {
        return hfName.contains("conv1.bias") || hfName.contains("conv2.bias");
    }

    /**
     * The MERaLiON tower and adapter (ultravox.py:190-212): {@code ln_speech} as {@code mm.a.norm_pre} and the four
     * adapter weights as {@code mm.a.mlp.0} &hellip; {@code mm.a.mlp.3}, the rest as {@link #qwen2aTensors}.
     */
    public static void meralionTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        String suffix = "." + hfName.substring(hfName.lastIndexOf('.') + 1);
        if (hfName.startsWith("ln_speech.")) {
            out.add(ctx.names().ggufName("A_MM_NORM_PRE", -1, suffix), src);
            return;
        }
        if (hfName.startsWith("speech_audio_adapter.")) {
            int bid;
            if (hfName.contains(".mlp_adapter.0.")) {
                bid = 0;
            } else if (hfName.contains(".gate_proj.")) {
                bid = 1;
            } else if (hfName.contains(".pool_proj.")) {
                bid = 2;
            } else if (hfName.contains(".out_proj.")) {
                bid = 3;
            } else {
                return;
            }
            out.add(ctx.names().ggufName("A_MMPROJ", bid, suffix), src);
            return;
        }
        qwen2aTensors(ctx, hfName, src, out);
    }
}
