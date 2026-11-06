package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The Soprano TTS projector through the Hugging Face loader, a port of llama.cpp {@code SopranoModel}
 * (conversion/soprano.py:51-114): the Vocos decoder read from {@code decoder.pth}, whose geometry comes from the
 * tensor shapes, written as a clip generator file whose {@code clip.gen.audio.projector_type} is {@code soprano}.
 * The ISTFT window is checked against the periodic Hann window the runtime builds itself. Every method is a public
 * static hook reference.
 */
public final class HfSopranoHooks {

    /** The Hugging Face architecture of the projector view (soprano.py:36-45). */
    public static final String ARCH = "SopranoModel";

    /** The decoder weights (soprano.py:55, :91). */
    private static final String DECODER = "decoder.pth";

    /** The decoder block keys and their GGUF tensors (soprano.py:100-108). */
    private static final Map<String, String> CONVNEXT = convnext();

    private HfSopranoHooks() {
    }

    private static Map<String, String> convnext() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("dwconv", "A_GEN_WAV_UP_DWCONV");
        m.put("norm", "A_GEN_WAV_UP_NORM");
        m.put("pwconv1", "A_GEN_WAV_UP_PW1");
        m.put("pwconv2", "A_GEN_WAV_UP_PW2");
        m.put("gamma", "A_GEN_WAV_UP_GAMMA");
        return java.util.Collections.unmodifiableMap(m);
    }

    /**
     * Every main-checkpoint tensor drops: {@code SopranoModel.get_tensors} (soprano.py:85-86) replaces the checkpoint
     * index with the decoder read from {@code decoder.pth} ({@link #tensors(HfLoadContext, TensorSink)}).
     */
    public static final HfNameFilter FILTER = (name, base) -> null;

    /** The per-name counterpart of {@link #FILTER}: the main checkpoint contributes no tensor to the file. */
    public static void drop(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        out.drop(hfName, "the Soprano projector writes the decoder.pth weights; the main checkpoint is the text model");
    }

    /** {@code SopranoModel.set_gguf_parameters} (soprano.py:76-89) over the decoder's own shape. */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        Geometry g = geometry(ctx.config());
        kv.set(GgufKeys.CLIP_HAS_GEN_AUDIO_ENCODER, true);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, "soprano");
        kv.set(GgufKeys.CLIP_GEN_AUDIO_PROJECTION_DIM, HfMmprojHooks.textWidth(ctx.config()));
        kv.set(GgufKeys.CLIP_GEN_AUDIO_EMBEDDING_LENGTH, g.width);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_FEED_FORWARD_LENGTH, g.feedForward);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_BLOCK_COUNT, g.blocks);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT, 1);
        kv.set(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_LAYER_NORM_EPSILON, 1e-6);
    }

    /** The block count the tensor plan uses: the decoder's ConvNeXt block count ({@code get_audio_config}). */
    public static int decoderBlocks(HfLoadContext ctx) {
        try {
            return geometry(ctx.config()).blocks;
        } catch (IOException e) {
            throw new GgufFormatException(ctx.config().source() + ": the Soprano decoder cannot be read: "
                    + e.getMessage(), e);
        }
    }

    /**
     * {@code SopranoModel.get_tensors} and {@code modify_tensors} (soprano.py:91-114): the decoder weights, with
     * the input projection squeezed and the ConvNeXt blocks under the upsample names; the ISTFT window is checked
     * and dropped.
     */
    public static void tensors(HfLoadContext ctx, TensorSink out) throws IOException {
        HfConfig cfg = ctx.config();
        Path path = cfg.dir().resolve(DECODER);
        if (!Files.isRegularFile(path)) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint has no '" + DECODER + "'; the Soprano"
                    + " decoder weights live there");
        }
        try (CheckpointContainer decoder = CheckpointContainers.open(path, cfg.dir(), false)) {
            for (String name : decoder.tensors().stream().map(t -> t.name())
                    .collect(java.util.stream.Collectors.toList())) {
                RowSource src = decoder.source(name);
                if (name.equals("head.istft.window")) {
                    requireHannWindow(ctx, name, src);
                    continue;
                }
                if (name.equals("decoder.embed.weight") && src.shape().length == 3
                        && src.shape()[src.shape().length - 1] == 1) {
                    src = ElementTransforms.reshape(src, src.shape()[0], src.shape()[1]);
                }
                if (name.startsWith("decoder.convnext.")) {
                    convnext(ctx, name, src, out);
                    continue;
                }
                int dot = name.lastIndexOf('.');
                if (dot <= 0) {
                    throw new GgufFormatException("unexpected Soprano tensor '" + name + "'");
                }
                String key = name.substring(0, dot);
                String tensor = simpleTensor(key);
                if (tensor == null) {
                    throw new GgufFormatException("unexpected Soprano tensor '" + name + "'");
                }
                out.add(ctx.names().ggufName(tensor, -1, "." + name.substring(dot + 1)), src);
            }
        }
    }

    /** One {@code decoder.convnext.<block>.<key>...} tensor (soprano.py:115-124). */
    private static void convnext(HfLoadContext ctx, String name, RowSource src, TensorSink out) {
        String[] parts = name.split("\\.");
        if (parts.length < 5) {
            throw new GgufFormatException("unexpected Soprano tensor '" + name + "'");
        }
        int block;
        try {
            block = Integer.parseInt(parts[2]);
        } catch (NumberFormatException e) {
            throw new GgufFormatException("Soprano tensor '" + name + "' has no block index");
        }
        String tensor = CONVNEXT.get(parts[3]);
        if (tensor == null) {
            throw new GgufFormatException("unexpected Soprano tensor '" + name + "'");
        }
        StringBuilder suffix = new StringBuilder();
        for (int i = 4; i < parts.length; i++) {
            suffix.append('.').append(parts[i]);
        }
        out.add(ctx.names().ggufName(tensor, block, suffix.toString()), src);
    }

    private static String simpleTensor(String key) {
        switch (key) {
            case "decoder.embed": return "A_GEN_WAV_INPUT";
            case "decoder.norm": return "A_GEN_WAV_NORM";
            case "decoder.final_layer_norm": return "A_GEN_WAV_OUTPUT_NORM";
            case "head.out": return "A_GEN_WAV_OUTPUT";
            default: return null;
        }
    }

    /**
     * True when {@code src} is the periodic Hann window the ISTFT builds, refused otherwise (soprano.py:94-99).
     */
    private static void requireHannWindow(HfLoadContext ctx, String name, RowSource src) throws IOException {
        float[] values = HfGenHooks.floats(src, name);
        int n = values.length;
        for (int i = 0; i < n; i++) {
            float want = (float) (0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / n)));
            if (Math.abs(values[i] - want) > 1e-6f) {
                throw new GgufFormatException(ctx.config().source() + ": '" + name + "' is not the periodic Hann"
                        + " window the mtmd ISTFT builds (entry " + i + " is " + values[i] + ", expected " + want
                        + ")");
            }
        }
    }

    /** {@code SopranoModel.tensor_force_quant} (soprano.py:94-97). */
    public static final HfForceType FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.contains("dwconv.weight") || ggufName.equals("a.gen.wav.input.weight")) {
            return GGMLType.F16;
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };

    /** The decoder geometry read from the weights (soprano.py:59-68). */
    private static Geometry geometry(HfConfig cfg) throws IOException {
        Path path = cfg.dir().resolve(DECODER);
        if (!Files.isRegularFile(path)) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint has no '" + DECODER + "'");
        }
        try (CheckpointContainer decoder = CheckpointContainers.open(path, cfg.dir(), false)) {
            long[] embed = shape(decoder, "decoder.embed.weight");
            if (embed.length != 3 && embed.length != 2) {
                throw new GgufFormatException(cfg.source() + ": decoder.embed.weight has the shape "
                        + Arrays.toString(embed));
            }
            int width = (int) embed[0];
            int input = (int) embed[1];
            int textWidth = HfMmprojHooks.textWidth(cfg);
            if (input != textWidth) {
                throw new GgufFormatException(cfg.source() + ": the Soprano decoder input dimension " + input
                        + " does not match the backbone dimension " + textWidth + " (soprano.py:65)");
            }
            long[] pw = shape(decoder, "decoder.convnext.0.pwconv1.weight");
            if (pw.length < 1) {
                throw new GgufFormatException(cfg.source() + ": decoder.convnext.0.pwconv1.weight has no shape");
            }
            int blocks = 0;
            for (com.eonmux.neuralfree.io.safetensors.StTensor t : decoder.tensors()) {
                if (t.name().endsWith(".pwconv1.weight")) {
                    blocks++;
                }
            }
            if (blocks == 0) {
                throw new GgufFormatException(cfg.source() + ": the Soprano decoder has no ConvNeXt blocks");
            }
            return new Geometry(width, (int) pw[0], blocks);
        }
    }

    private static long[] shape(CheckpointContainer decoder, String name) {
        com.eonmux.neuralfree.io.safetensors.StTensor t = decoder.tensor(name);
        if (t == null) {
            throw new GgufFormatException("decoder.pth has no tensor '" + name + "'");
        }
        return t.shape();
    }

    /** The decoder geometry: width, ConvNeXt feed-forward width and block count. */
    private static final class Geometry {
        final int width;
        final int feedForward;
        final int blocks;

        Geometry(int width, int feedForward, int blocks) {
            this.width = width;
            this.feedForward = feedForward;
            this.blocks = blocks;
        }
    }
}
