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

import java.io.IOException;

/**
 * The Qwen audio towers and the Qwen Omni projector files, ports of llama.cpp {@code Qwen25AudioModel}
 * (conversion/qwenvl.py:145-165), {@code Qwen25OmniModel} (qwenvl.py:167-208), {@code Qwen3OmniMmprojModel}
 * (conversion/qwen3vl.py:159-231) and {@code Qwen3ASRMmprojModel} (qwen3vl.py:233-237): the audio converter aliases,
 * the mel front end, the generated sinusoidal position table, the two-convolution bias widening of the older tower,
 * the three-convolution bias widening and Q/K/V split of the newer one, and the {@code qwen2.5o} mixed-modality
 * projector that resolves to a vision and an audio type. Every method is a public static hook reference.
 */
public final class HfQwenOmniHooks {

    /** The Qwen3 audio projector ({@code VisionProjectorType.QWEN3A}). */
    public static final String QWEN3A = "qwen3a";
    /** The legacy mixed-modality projector of Qwen2.5-Omni ({@code VisionProjectorType.QWEN25O}). */
    public static final String QWEN25O = "qwen2.5o";

    /** The length of the generated sinusoidal position table (qwenvl.py:156). */
    private static final int POSITION_TABLE_LENGTH = 1500;
    /** The largest sinusoid timescale (qwenvl.py:155). */
    private static final double MAX_TIMESCALE = 10000.0;
    /** The default layer-norm epsilon of the audio tower. */
    private static final double DEFAULT_EPS = 1e-5;

    private HfQwenOmniHooks() {
    }

    // ------------------------------------------------------------------ audio metadata

    /**
     * {@code Qwen25AudioModel.set_gguf_parameters} (qwenvl.py:152-161): the base audio keys through the {@code
     * d_model} / {@code encoder_ffn_dim} / {@code encoder_attention_heads} aliases, the mel-bin count and the tower
     * epsilon.
     */
    static void audioMeta(HfLoadContext ctx, HfMetadataBuilder kv) {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.audioMeta(ctx, kv, HfMmprojHooks.audioInt(cfg, "hidden_size", "d_model"),
                HfMmprojHooks.audioInt(cfg, "intermediate_size", "encoder_ffn_dim"),
                HfMmprojHooks.audioInt(cfg, "num_attention_heads", "encoder_attention_heads"),
                HfMmprojHooks.audioBlocks(cfg));
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, HfMmprojHooks.audioInt(cfg, "num_mel_bins"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, HfAudioTowerHooks.audioEps(cfg, DEFAULT_EPS));
    }

    /** The Qwen2.5-Omni vision and audio file (qwenvl.py:167-208). */
    public static void qwen25OmniMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenVlVisionHooks.qwen25OmniVisionMeta(ctx, kv, plan);
        audioMeta(ctx, kv);
    }

    /**
     * The Qwen3 Omni vision and audio file (qwen3vl.py:159-231): the Qwen3-VL vision keys under
     * {@code clip.vision.projector_type}, the audio keys, and the {@code qwen3a} audio projector.
     */
    public static void qwen3OmniMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        if (cfg.vision().isPresent()) {
            HfQwenVlVisionHooks.qwen3vlMeta(ctx, kv, plan, GgufKeys.CLIP_VISION_PROJECTOR_TYPE);
        }
        if (cfg.audio().isPresent()) {
            audioMeta(ctx, kv);
            kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, QWEN3A);
        }
    }

    /**
     * {@code Qwen3ASRMmprojModel} (qwen3vl.py:233-237): the Qwen3-Omni audio half alone, with the common
     * {@code clip.projector_type}.
     */
    public static void qwen3AsrMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        if (!cfg.audio().isPresent()) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint has no 'audio_config'; Qwen3-ASR"
                    + " converts an audio tower");
        }
        audioMeta(ctx, kv);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, QWEN3A);
    }

    // ------------------------------------------------------------------ filters

    /** {@code Qwen25OmniModel.filter_tensors} (qwenvl.py:189-205). */
    public static final HfNameFilter QWEN25_OMNI_FILTER = (name, base) -> {
        if (name.startsWith("thinker.")) {
            name = name.substring("thinker.".length());
        }
        if (!name.startsWith("visual.") && !name.startsWith("audio_tower.")) {
            return null;
        }
        if (name.contains("audio_bos_eos_token")) {
            return null;
        }
        return base.apply(name);
    };

    /** {@code Qwen3OmniMmprojModel.filter_tensors} (qwen3vl.py:196-221). */
    public static final HfNameFilter QWEN3_OMNI_FILTER = (name, base) -> {
        if (name.startsWith("lm_head.") || name.startsWith("mtp.")) {
            return null;
        }
        if (name.startsWith("model.visual.")) {
            name = name.substring("model.".length());
        }
        if (name.startsWith("thinker.audio_tower.")) {
            name = name.substring("thinker.".length());
        }
        if (!name.contains("visual.") && !name.contains("audio_tower.")) {
            return null;
        }
        return base.apply(name);
    };

    // ------------------------------------------------------------------ tensors

    /**
     * {@code Qwen25OmniModel.modify_tensors} (qwenvl.py:207-216): the vision half runs the Qwen2-VL transforms, the
     * audio half the Whisper conv-stem widening.
     */
    public static void qwen25OmniTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.contains("visual.")) {
            HfQwenVlVisionHooks.qwen2vlTensors(ctx, hfName, src, out);
            return;
        }
        if (hfName.contains("audio_tower.")) {
            audioTensors(ctx, hfName, src, out);
        }
    }

    /**
     * {@code Qwen3OmniMmprojModel.modify_tensors} (qwen3vl.py:223-231): the vision names take the Qwen3-VL merger
     * layout, the audio names the Qwen3 conv geometry.
     */
    public static void qwen3OmniTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("thinker.visual.")) {
            hfName = hfName.substring("thinker.".length());
        } else if (hfName.startsWith("model.visual.")) {
            hfName = hfName.substring("model.".length());
        }
        if (hfName.startsWith("thinker.audio_tower.")) {
            hfName = hfName.substring("thinker.".length());
        }
        if (hfName.startsWith("visual.")) {
            hfName = renameMerger(hfName);
            HfQwenVlVisionHooks.qwen3vlTensors(ctx, hfName, src, out);
            return;
        }
        if (hfName.contains("audio_tower.")) {
            audioTensors(ctx, hfName, src, out);
        }
    }

    /** The Qwen3-Omni merger names mapped onto the Qwen3-VL layout (qwen3vl.py:229-237). */
    private static String renameMerger(String name) {
        if (name.startsWith("visual.merger_list.")) {
            name = "visual.deepstack_merger_list." + name.substring("visual.merger_list.".length());
        }
        return name.replace(".ln_q", ".norm").replace(".mlp.0", ".linear_fc1").replace(".mlp.2", ".linear_fc2");
    }

    /**
     * The Qwen audio tower (qwenvl.py:163-165, qwen3vl.py:227-231): the conv-stem biases widen to a row, the Qwen3
     * conv2d bias to {@code [c, 1, 1]}, everything else takes its mapped name.
     */
    private static void audioTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.contains("conv2d") && hfName.endsWith(".bias")) {
            src = ElementTransforms.reshape(src, src.shape()[0], 1, 1);
        } else if (HfAudioTowerHooks.convBias(hfName)) {
            src = ElementTransforms.reshape(src, src.shape()[0], 1);
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** The conv kernels keep F16 (qwenvl.py:163-165). */
    public static final HfForceType FORCE = HfAudioTowerHooks.convF16();

    // ------------------------------------------------------------------ extra tensors

    /**
     * {@code Qwen25AudioModel.generate_extra_tensors} (qwenvl.py:150-161): the sinusoidal position table of the audio
     * tower, {@code [1500, channels]}, written as {@code a.position_embd.weight}.
     */
    public static void positionTable(HfLoadContext ctx, TensorSink out) {
        int channels = HfMmprojHooks.audioInt(ctx.config(), "hidden_size", "d_model");
        if (channels < 4 || channels % 2 != 0) {
            throw new GgufFormatException(ctx.config().source() + ": the audio tower width " + channels
                    + " cannot hold a sinusoid table; it must be even and at least 4");
        }
        int half = channels / 2;
        double increment = Math.log(MAX_TIMESCALE) / (half - 1);
        float[] values = new float[POSITION_TABLE_LENGTH * channels];
        for (int i = 0; i < half; i++) {
            double inv = Math.exp(-increment * i);
            for (int t = 0; t < POSITION_TABLE_LENGTH; t++) {
                double scaled = t * inv;
                int at = t * channels + i;
                values[at] = (float) Math.sin(scaled);
                values[at + half] = (float) Math.cos(scaled);
            }
        }
        out.add(ctx.names().ggufName("A_ENC_EMBD_POS", -1, ".weight"),
                RopeTensors.fromFloats(new long[]{POSITION_TABLE_LENGTH, channels}, values));
    }

    /** The generated position table's GGUF name, for tests. */
    public static String positionTableName(HfLoadContext ctx) {
        return ctx.names().ggufName("A_ENC_EMBD_POS", -1, ".weight");
    }
}
