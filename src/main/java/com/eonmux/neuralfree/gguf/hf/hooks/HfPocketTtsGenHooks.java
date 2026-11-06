package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.io.IOException;
import java.util.Arrays;

/**
 * The {@code PocketTTSMmprojModel} of llama.cpp's converter (conversion/pockettts.py:177-381): the flow-matching
 * decoder, the mimi decoder and the mimi speaker encoder of a pocket-tts checkpoint, written as a clip generator
 * file whose {@code clip.gen.audio.projector_type} is {@code pockettts_gen} and, when the checkpoint carries the
 * voice-prompt tensors, whose {@code clip.audio.projector_type} is {@code pockettts_spkenc}.
 *
 * <p>The checkpoint is one safetensors part holding the text backbone ({@code flow_lm.transformer}) and, beside it,
 * the flow network ({@code flow_lm.flow_net}), the latent projection and end-of-speech head, the mimi quantizer and
 * upsampler, the decoder transformer and the decoder SEANet stack. The text backbone converts through the
 * {@code POCKETTTS} arch; this view keeps the generator tensors and the speaker encoder (the mimi encoder, the
 * encoder transformer and {@code speaker_proj}) and drops the learned input vectors, which the backbone conversion
 * consumes. The decoder geometry is read from the decoder transformer's own tensors, not the encoder's config, so a
 * checkpoint whose two transformers differ still declares the layers the file carries.
 */
public final class HfPocketTtsGenHooks {

    /** The Hugging Face architecture of the checkpoint (shared with the text entry). */
    public static final String ARCH = "PocketTTSModel";

    /** SEANet decoder module indices of the three stages (pockettts.py:31-34, {@code _DEC_*_IDX}). */
    private static final int DECODER_CONV_IN = 0;
    private static final int DECODER_CONV_OUT = 3 * 3 + 2;
    /** The SEANet encoder's first and last module indices (pockettts.py:28-34). */
    private static final int ENCODER_CONV_IN = 0;
    private static final int ENCODER_CONV_OUT = 3 * 3 + 2;

    private HfPocketTtsGenHooks() {
    }

    /**
     * {@code PocketTTSMmprojModel.set_gguf_parameters} (pockettts.py:187-225): the generation keys (the
     * flow-matching decoder and the mimi decoder) and, for a checkpoint carrying the voice-prompt tensors, the
     * speaker encoder keys (the mimi encoder and its projection).
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfTensorIndex idx = ctx.index();
        int layers = decoderBlocks(ctx);
        int width = intOf(idx, "mimi.decoder_transformer.transformer.layers.0.norm1.weight", -1,
                "the decoder transformer width");
        long[] linear = shapeOf(idx, "mimi.decoder_transformer.transformer.layers.0.linear1.weight");
        if (linear.length != 2 || linear[1] != width) {
            throw new GgufFormatException(cfg.source() + ": the decoder transformer's linear1.weight has shape "
                    + Arrays.toString(linear) + ", expected [feed_forward, " + width + "]");
        }
        if (width % 64 != 0) {
            throw new GgufFormatException(cfg.source() + ": the decoder transformer is " + width + " wide; its"
                    + " attention uses head_dim 64, so the width must be a multiple of 64");
        }
        kv.set(GgufKeys.CLIP_HAS_GEN_AUDIO_ENCODER, true);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, "pockettts_gen");
        kv.set(GgufKeys.CLIP_GEN_AUDIO_PROJECTION_DIM, HfMmprojHooks.textWidth(cfg));
        kv.set(GgufKeys.CLIP_GEN_AUDIO_EMBEDDING_LENGTH, width);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_FEED_FORWARD_LENGTH, (int) linear[0]);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_BLOCK_COUNT, layers);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT, width / 64);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_LAYER_NORM_EPSILON, 1e-5f);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_MODEL_VARIANT, cfg.dir().getFileName().toString());
        if (hasSpeakerEncoder(ctx)) {
            speakerMeta(ctx, cfg, kv);
        }
    }

    /**
     * The voice-prompt encoder's keys (pockettts.py:191-200): the mimi encoder's transformer geometry and the raw
     * waveform "mel" of one row.
     */
    private static void speakerMeta(HfLoadContext ctx, HfConfig cfg, HfMetadataBuilder kv) {
        HfTensorIndex idx = ctx.index();
        String first = "mimi.encoder_transformer.transformer.layers.0.";
        int width = intOf(idx, first + "norm1.weight", -1, "the speaker transformer width");
        long[] linear = shapeOf(idx, first + "linear1.weight");
        if (linear.length != 2 || linear[1] != width) {
            throw new GgufFormatException(cfg.source() + ": the speaker transformer's linear1.weight has shape "
                    + Arrays.toString(linear) + ", expected [feed_forward, " + width + "]");
        }
        if (width % 64 != 0) {
            throw new GgufFormatException(cfg.source() + ": the speaker transformer is " + width + " wide; its"
                    + " attention uses head_dim 64, so the width must be a multiple of 64");
        }
        kv.set(GgufKeys.CLIP_HAS_AUDIO_ENCODER, true);
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, "pockettts_spkenc");
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTION_DIM, HfMmprojHooks.textWidth(cfg));
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, 1);
        kv.set(GgufKeys.CLIP_AUDIO_BLOCK_COUNT, speakerBlocks(ctx));
        kv.set(GgufKeys.CLIP_AUDIO_EMBEDDING_LENGTH, width);
        kv.set(GgufKeys.CLIP_AUDIO_FEED_FORWARD_LENGTH, (int) linear[0]);
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_HEAD_COUNT, width / 64);
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, 1e-5f);
    }

    /**
     * Whether the checkpoint carries the voice-prompt encoder: its projection, the mimi encoder's SEANet stack and
     * its transformer are all needed, and a checkpoint that has only one of them is not a complete encoder.
     */
    private static boolean hasSpeakerEncoder(HfLoadContext ctx) {
        HfTensorIndex idx = ctx.index();
        return idx.has("flow_lm.speaker_proj_weight")
                && idx.has("mimi.downsample.conv.conv.weight")
                && idx.has("mimi.encoder.model.0.conv.weight")
                && idx.has("mimi.encoder_transformer.transformer.layers.0.norm1.weight");
    }

    private static int speakerBlocks(HfLoadContext ctx) {
        int count = 0;
        for (String name : ctx.index().names()) {
            if (name.startsWith("mimi.encoder_transformer.transformer.layers.")
                    && name.endsWith(".norm1.weight")) {
                count++;
            }
        }
        if (count == 0) {
            throw new GgufFormatException(ctx.config().source() + ": the checkpoint has no"
                    + " mimi.encoder_transformer.transformer.layers.*.norm1.weight for the speaker encoder");
        }
        return count;
    }

    /**
     * The block count the tensor planner uses: the decoder transformer's layer count, which the file's
     * {@code clip.gen.audio.block_count} must match.
     */
    public static int decoderBlocks(HfLoadContext ctx) {
        int count = 0;
        for (String name : ctx.index().names()) {
            if (name.startsWith("mimi.decoder_transformer.transformer.layers.")
                    && name.endsWith(".norm1.weight")) {
                count++;
            }
        }
        if (count == 0) {
            throw new GgufFormatException(ctx.config().source() + ": the checkpoint has no"
                    + " mimi.decoder_transformer.transformer.layers.*.norm1.weight that the pockettts mimi decoder"
                    + " needs");
        }
        return count;
    }

    /**
     * {@code PocketTTSMmprojModel.modify_tensors} (pockettts.py:238-286): the generator tensors, with the fused
     * mimi attention projection split into Q/K/V, the quantizer's output projection squeezed and the SEANet and
     * upsample kernels forced to F16 by {@link #SEANET_F16}. Tensors of the backbone, and the speaker-encoder
     * tensors of a checkpoint without a complete encoder, are dropped with their owner named.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.equals("flow_lm.bos_emb") || hfName.equals("flow_lm.bos_before_voice")
                || hfName.equals("flow_lm.conditioner.embed.weight")) {
            out.drop(hfName, "the learned input rows are folded into the POCKETTTS backbone embedding table");
            return;
        }
        if (hfName.startsWith("flow_lm.transformer.") || hfName.startsWith("flow_lm.out_norm.")) {
            out.drop(hfName, "flow_lm.transformer is the POCKETTTS backbone");
            return;
        }
        if (hfName.equals("flow_lm.speaker_proj_weight")) {
            if (hasSpeakerEncoder(ctx)) {
                add(ctx, out, "A_ENC_SPEAKER_PROJ", -1, ".weight", src);
            } else {
                out.drop(hfName, "the checkpoint carries no complete speaker encoder");
            }
            return;
        }
        if (hfName.equals("flow_lm.input_linear.weight")) {
            add(ctx, out, "A_GEN_INPUT_LINEAR", -1, ".weight", src);
            return;
        }
        if (hfName.equals("flow_lm.emb_mean") || hfName.equals("flow_lm.emb_std")) {
            add(ctx, out, hfName.equals("flow_lm.emb_mean") ? "A_GEN_EMB_MEAN" : "A_GEN_EMB_STD", -1, "", src);
            return;
        }
        if (hfName.startsWith("flow_lm.out_eos.")) {
            add(ctx, out, "A_GEN_OUT_EOS", -1, suffix(hfName), src);
            return;
        }
        if (hfName.startsWith("flow_lm.flow_net.")) {
            flowNet(ctx, hfName, src, out);
            return;
        }
        if (hfName.equals("mimi.upsample.convtr.convtr.weight")) {
            add(ctx, out, "A_GEN_WAV_UPSAMPLE", -1, ".weight", src);
            return;
        }
        if (hfName.equals("mimi.quantizer.output_proj.weight")) {
            add(ctx, out, "A_GEN_WAV_QUANT_OUT", -1, ".weight", ElementTransforms.squeezeConv(src));
            return;
        }
        if (hfName.equals("mimi.downsample.conv.conv.weight")) {
            add(ctx, out, "A_ENC_DOWNSAMPLE_CONV", -1, ".weight", src);
            return;
        }
        if (hfName.startsWith("mimi.downsample.")) {
            out.drop(hfName, "the speaker encoder's downsample convolution carries only a kernel");
            return;
        }
        if (hfName.contains("_transformer.transformer.layers.")) {
            if (hfName.startsWith("mimi.decoder_transformer.")) {
                mimiTransformer(ctx, hfName, src, out, true);
                return;
            }
            if (hasSpeakerEncoder(ctx)) {
                mimiTransformer(ctx, hfName, src, out, false);
                return;
            }
            out.drop(hfName, "the checkpoint carries no complete speaker encoder");
            return;
        }
        if (hfName.startsWith("mimi.encoder.model.")) {
            if (hasSpeakerEncoder(ctx)) {
                seanet(ctx, hfName, src, out, false);
                return;
            }
            out.drop(hfName, "the checkpoint carries no complete speaker encoder");
            return;
        }
        if (hfName.startsWith("mimi.decoder.model.")) {
            seanet(ctx, hfName, src, out, true);
            return;
        }
        out.drop(hfName, "the pockettts projector has no tensor for this name");
    }

    /** {@code _flow_net_tensor} (pockettts.py:288-333). */
    private static void flowNet(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        String key = hfName.substring("flow_lm.flow_net.".length());
        String suffix = suffix(key);
        String base = key.lastIndexOf('.') < 0 ? key : key.substring(0, key.lastIndexOf('.'));
        String simple = simpleFlowTensor(base);
        if (simple != null) {
            add(ctx, out, simple, -1, suffix, src);
            return;
        }
        if (key.startsWith("time_embed.")) {
            int bid = ctx.bid(hfName);
            String rest = key.substring("time_embed.".length());
            rest = rest.substring(rest.indexOf('.') + 1);
            String tensor = timeTensor(rest);
            if (tensor != null) {
                add(ctx, out, tensor, bid, timeSuffix(rest), src);
                return;
            }
        } else if (key.startsWith("res_blocks.")) {
            int bid = ctx.bid(hfName);
            String rest = key.substring("res_blocks.".length());
            rest = rest.substring(rest.indexOf('.') + 1);
            String inner = rest.lastIndexOf('.') < 0 ? rest : rest.substring(0, rest.lastIndexOf('.'));
            String tensor = blockTensor(inner);
            if (tensor != null) {
                add(ctx, out, tensor, bid, suffix(rest), src);
                return;
            }
        }
        out.drop(hfName, "the pockettts flow network has no tensor for this name");
    }

    private static String simpleFlowTensor(String base) {
        switch (base) {
            case "input_proj": return "A_GEN_FLOW_INPUT_PROJ";
            case "cond_embed": return "A_GEN_FLOW_COND_EMBD";
            case "final_layer.linear": return "A_GEN_FLOW_FINAL_PROJ";
            case "final_layer.adaLN_modulation.1": return "A_GEN_FLOW_FINAL_ADA";
            default: return null;
        }
    }

    private static String timeTensor(String rest) {
        switch (rest) {
            case "freqs": return "A_GEN_FLOW_TIME_FREQS";
            case "mlp.0": case "mlp.0.weight": case "mlp.0.bias": return "A_GEN_FLOW_TIME_UP";
            case "mlp.2": case "mlp.2.weight": case "mlp.2.bias": return "A_GEN_FLOW_TIME_DOWN";
            case "mlp.3.alpha": return "A_GEN_FLOW_TIME_NORM";
            default: return null;
        }
    }

    private static String timeSuffix(String rest) {
        if (rest.equals("freqs") || rest.equals("mlp.3.alpha")) {
            return "";
        }
        return suffix(rest);
    }

    private static String blockTensor(String inner) {
        switch (inner) {
            case "in_ln": return "A_GEN_FLOW_BLK_NORM";
            case "mlp.0": return "A_GEN_FLOW_BLK_UP";
            case "mlp.2": return "A_GEN_FLOW_BLK_DOWN";
            case "adaLN_modulation.1": return "A_GEN_FLOW_BLK_ADA";
            default: return null;
        }
    }

    /** {@code _mimi_tfm_tensor} (pockettts.py:335-346) for the decoder or the speaker-encoder transformer. */
    private static void mimiTransformer(HfLoadContext ctx, String hfName, RowSource src, TensorSink out,
                                        boolean decoder) throws IOException {
        int bid = ctx.bid(hfName);
        String marker = (decoder ? "mimi.decoder_transformer.transformer.layers."
                : "mimi.encoder_transformer.transformer.layers.") + bid + ".";
        int at = hfName.indexOf(marker);
        if (at < 0) {
            throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " names no"
                    + " mimi transformer layer");
        }
        String keyWithSuffix = hfName.substring(at + marker.length());
        if (keyWithSuffix.equals("self_attn.in_proj.weight")) {
            long[] shape = src.shape();
            long third = shape[0] / 3;
            if (shape.length != 2 || shape[0] <= 0 || shape[0] % 3 != 0) {
                throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has shape "
                        + Arrays.toString(shape) + "; the fused Q/K/V projection needs three equal row groups");
            }
            RowSource[] qkv = RowTransforms.splitRows(src, third, third, third);
            add(ctx, out, decoder ? "A_GEN_WAV_TFM_ATTN_Q" : "A_ENC_ATTN_Q", bid, ".weight", qkv[0]);
            add(ctx, out, decoder ? "A_GEN_WAV_TFM_ATTN_K" : "A_ENC_ATTN_K", bid, ".weight", qkv[1]);
            add(ctx, out, decoder ? "A_GEN_WAV_TFM_ATTN_V" : "A_ENC_ATTN_V", bid, ".weight", qkv[2]);
            return;
        }
        String base = keyWithSuffix.lastIndexOf('.') < 0 ? keyWithSuffix
                : keyWithSuffix.substring(0, keyWithSuffix.lastIndexOf('.'));
        String tensor = transformerTensor(base, decoder);
        if (tensor == null) {
            tensor = transformerTensor(keyWithSuffix, decoder);
        }
        if (tensor == null) {
            out.drop(hfName, "the pockettts mimi transformer has no tensor for this name");
            return;
        }
        String suffix = keyWithSuffix.endsWith(".scale") ? ".weight" : suffix(keyWithSuffix);
        add(ctx, out, tensor, bid, suffix, src);
    }

    private static String transformerTensor(String key, boolean decoder) {
        switch (key) {
            case "norm1": return decoder ? "A_GEN_WAV_TFM_ATTN_NORM" : "A_ENC_INPUT_NORM";
            case "norm2": return decoder ? "A_GEN_WAV_TFM_FFN_NORM" : "A_ENC_OUTPUT_NORM";
            case "self_attn.out_proj": return decoder ? "A_GEN_WAV_TFM_ATTN_OUT" : "A_ENC_OUTPUT";
            case "linear1": return decoder ? "A_GEN_WAV_TFM_FFN_UP" : "A_ENC_FFN_UP";
            case "linear2": return decoder ? "A_GEN_WAV_TFM_FFN_DOWN" : "A_ENC_FFN_DOWN";
            case "layer_scale_1.scale": return decoder ? "A_GEN_WAV_TFM_ATTN_SCALE" : "A_ENC_ATTN_SCALE";
            case "layer_scale_2.scale": return decoder ? "A_GEN_WAV_TFM_FFN_SCALE" : "A_ENC_FFN_SCALE_LS";
            default: return null;
        }
    }

    /** {@code _seanet_tensor} for the decoder or the speaker-encoder SEANet stack (pockettts.py:348-381). */
    private static void seanet(HfLoadContext ctx, String hfName, RowSource src, TensorSink out, boolean decoder)
            throws IOException {
        int idx = ctx.bid(hfName);
        String suffix = suffix(hfName);
        if (idx == (decoder ? DECODER_CONV_IN : ENCODER_CONV_IN)) {
            add(ctx, out, decoder ? "A_GEN_WAV_SEANET_CONV_IN" : "A_ENC_SEANET_CONV_IN", -1, suffix, src);
            return;
        }
        if (idx == (decoder ? DECODER_CONV_OUT : ENCODER_CONV_OUT)) {
            add(ctx, out, decoder ? "A_GEN_WAV_SEANET_CONV_OUT" : "A_ENC_SEANET_CONV_OUT", -1, suffix, src);
            return;
        }
        int scaleBase = decoder ? 2 : 3;
        int resBase = decoder ? 3 : 1;
        for (int stage = 0; stage < 3; stage++) {
            if (idx == scaleBase + 3 * stage) {
                add(ctx, out, decoder ? "A_GEN_WAV_SEANET_SCALE_CONV" : "A_ENC_SEANET_SCALE_CONV", stage, suffix, src);
                return;
            }
            if (idx == resBase + 3 * stage) {
                int inner = blockIndex(ctx, hfName);
                String tensor = inner == 1 ? (decoder ? "A_GEN_WAV_SEANET_RES_CONV1" : "A_ENC_SEANET_RES_CONV1")
                        : inner == 3 ? (decoder ? "A_GEN_WAV_SEANET_RES_CONV2" : "A_ENC_SEANET_RES_CONV2") : null;
                if (tensor == null) {
                    out.drop(hfName, "the pockettts SEANet residual unit has no convolution at this block index");
                    return;
                }
                add(ctx, out, tensor, stage, suffix, src);
                return;
            }
        }
        out.drop(hfName, "the pockettts mimi decoder has no SEANet module at index " + idx);
    }

    private static int blockIndex(HfLoadContext ctx, String hfName) {
        int at = hfName.indexOf(".block.");
        if (at < 0) {
            return -1;
        }
        String rest = hfName.substring(at + ".block.".length());
        int dot = rest.indexOf('.');
        String part = dot < 0 ? rest : rest.substring(0, dot);
        try {
            return Integer.parseInt(part);
        } catch (NumberFormatException e) {
            throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has the block"
                    + " index '" + part + "'");
        }
    }

    /** The SEANet, downsample and upsample kernels keep F16, like {@code tensor_force_quant} (pockettts.py:227-233). */
    public static final HfForceType SEANET_F16 = (ggufName, nDims, source, outType) ->
            ggufName.contains(".seanet.") || ggufName.equals("a.downsample.conv.weight")
                    || ggufName.equals("a.gen.wav.upsample.weight") ? GGMLType.F16 : null;

    // ------------------------------------------------------------------ helpers

    private static void add(HfLoadContext ctx, TensorSink out, String tensor, int bid, String suffix,
                            RowSource src) {
        out.add(ctx.names().ggufName(tensor, bid, suffix), src);
    }

    private static String suffix(String name) {
        return "." + name.substring(name.lastIndexOf('.') + 1);
    }

    private static long[] shapeOf(HfTensorIndex idx, String name) {
        if (!idx.has(name)) {
            throw new GgufFormatException("the checkpoint has no tensor '" + name + "'");
        }
        return idx.get(name).shape();
    }

    private static int intOf(HfTensorIndex idx, String name, int dim, String what) {
        long[] shape = shapeOf(idx, name);
        int at = dim < 0 ? shape.length - 1 : dim;
        if (at < 0 || at >= shape.length) {
            throw new GgufFormatException("the checkpoint tensor '" + name + "' has shape "
                    + Arrays.toString(shape) + "; " + what + " is not dimension " + at);
        }
        long v = shape[at];
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new GgufFormatException("the checkpoint tensor '" + name + "' has " + v + " values in dimension "
                    + at + "; " + what + " must be a positive int");
        }
        return (int) v;
    }
}
