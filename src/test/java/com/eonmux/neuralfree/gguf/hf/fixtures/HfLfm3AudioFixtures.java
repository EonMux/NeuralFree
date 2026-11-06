package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Synthetic Hugging Face checkpoints for the LFM3 audio projector export: a four-block LFM3 audio encoder (two
 * short-conv blocks, a sliding-window attention block and a trailing conv block), the LayerNorm-GELU-linear
 * projector, and, when the checkpoint ships the liquid codec, the MaskGIT depthformer with twelve tied per-codebook
 * embedding pages and the codec decoder side file. The geometry and the tensor values are given in ggml order (the
 * source's {@code ne} dims), so a test can hand in the same tensors the runtime fixtures use and compare the
 * converted encoder against the reference graph; the writer reverses the dims to the PyTorch layout the checkpoint
 * stores, exactly the transform the converter undoes.
 */
public final class HfLfm3AudioFixtures {

    /** Mel bins of the audio encoder and of the runtime's mel frontend. */
    public static final int N_MEL = 16;
    /** Audio encoder width and head geometry. */
    public static final int HIDDEN = 8;
    public static final int HEADS = 2;
    public static final int KV_HEADS = 1;
    public static final int FF = 12;
    /** Short-conv window and kernel taps. */
    public static final int WINDOW = 4;
    public static final int TAPS = 3;
    /** Encoder blocks, with the kind of each (a short conv or an attention block). */
    public static final int LAYERS = 4;
    public static final boolean[] CONV = {true, false, true, true};
    /** The text width the projector writes and its adapter width. */
    public static final int TEXT = 6;
    public static final int ADAPTER = 10;
    /** The audio norm epsilon. */
    public static final float EPS = 1e-5f;
    /** The depthformer geometry: width, FFN, heads, blocks. */
    public static final int DEPTH_HIDDEN = 4;
    public static final int DEPTH_FF = 8;
    public static final int DEPTH_HEADS = 2;
    public static final int DEPTH_KV = 1;
    public static final int DEPTH_LAYERS = 2;
    /** The codebook count and the FSQ level shape the converter requires. */
    public static final int CODEBOOKS = 12;
    public static final int[] FSQ_LEVELS = {8, 7, 6, 6};
    public static final int AUDIO_VOCAB = fsqCodebookSize() + 1;
    /** The liquid-codec decoder width, FFN and head geometry. */
    public static final int CODEC_HIDDEN = 8;
    public static final int CODEC_FF = 8;
    public static final int CODEC_HEADS = 2;
    public static final int CODEC_KV = 1;
    public static final int CODEC_TAPS = 3;

    private static final int[] DOWNSAMPLE_CHANNELS = {4, 4, 3};

    private HfLfm3AudioFixtures() {
    }

    /** The ggml-order tensors of one audio checkpoint, keyed by their GGUF name. */
    public interface Tower {

        /** The {@code ne} dims of {@code ggufName}, or null when the tower does not carry it. */
        long[] ne(String ggufName);

        /** The values of {@code ggufName} in ggml order, or null when the tower does not carry it. */
        float[] values(String ggufName);
    }

    /** Deterministic values for every tensor of the encoder, projector, depthformer and codec extras. */
    public static Tower defaultTower() {
        Map<String, long[]> shapes = new LinkedHashMap<>();
        encoderShapes(shapes);
        generatorShapes(shapes);
        Map<String, float[]> values = new LinkedHashMap<>();
        int seed = 1;
        for (Map.Entry<String, long[]> e : shapes.entrySet()) {
            values.put(e.getKey(), SyntheticHf.values(seed++, Math.toIntExact(count(e.getValue()))));
        }
        return tower(shapes, values);
    }

    /** A {@link Tower} over explicit ggml-order shapes and values. */
    public static Tower tower(Map<String, long[]> shapes, Map<String, float[]> values) {
        return new Tower() {
            @Override
            public long[] ne(String ggufName) {
                long[] s = shapes.get(ggufName);
                return s == null ? null : s.clone();
            }

            @Override
            public float[] values(String ggufName) {
                float[] v = values.get(ggufName);
                return v == null ? null : v.clone();
            }
        };
    }

    /**
     * The {@code Lfm3AudioForConditionalGeneration} checkpoint of {@code tower}, with the audio codec side files when
     * {@code withCodec} is set and every tensor stored as {@code dtype}.
     */
    public static SyntheticHf.Builder audioCheckpoint(Tower tower, boolean withCodec, StDType dtype) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(audioConfigJson()));
        b.file("preprocessor_config.json", SyntheticHf.json(preprocessor()));
        audio(b, tower, names("encoder."), dtype);
        if (withCodec) {
            generator(b, tower, "depthformer.", "depth_embeddings.", "codebook_embedding.", "mask_embedding",
                    "depth_linear.", dtype);
        }
        return b;
    }

    /**
     * The {@code Lfm3OmniForConditionalGeneration} checkpoint with the audio encoder under its HF-native
     * {@code model.audio_encoder}/{@code model.audio_projector} names and the generator under
     * {@code model.depth_decoder}/{@code model.codebook_embedding}.
     */
    public static SyntheticHf.Builder omniCheckpoint(Tower tower, boolean withCodec, StDType dtype) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(omniConfigJson()));
        b.file("preprocessor_config.json", SyntheticHf.json(omniPreprocessor()));
        audio(b, tower, omniNames(), dtype);
        if (withCodec) {
            generator(b, tower, "model.depth_decoder.", "model.depth_decoder.depth_embeddings.",
                    "model.codebook_embedding.", "model.depth_decoder.mask_embedding",
                    "model.depth_decoder.depth_linear.", dtype);
        }
        return b;
    }

    /** Writes {@code audio_tokenizer/config.json} and its decoder weights under the model directory. */
    public static void writeCodec(Path dir) throws IOException {
        Path codec = dir.resolve("audio_tokenizer");
        Files.createDirectories(codec);
        Files.write(codec.resolve("config.json"), codecConfig().getBytes(StandardCharsets.UTF_8));
        SafetensorsWriter.Builder w = SafetensorsWriter.builder();
        int levels = FSQ_LEVELS.length;
        String p = "decoder.lfm.layers.0.";
        int dh = CODEC_HIDDEN / CODEC_HEADS;
        tensor(w, 30, "quantizer.fsq.project_out.weight", new long[]{CODEC_HIDDEN, (long) levels * CODEBOOKS});
        tensor(w, 31, "quantizer.fsq.project_out.bias", new long[]{CODEC_HIDDEN});
        tensor(w, 32, "from_quantizer.weight", new long[]{CODEC_HIDDEN, CODEC_HIDDEN});
        tensor(w, 33, "from_quantizer.bias", new long[]{CODEC_HIDDEN});
        tensor(w, 34, "decoder.lin.weight", new long[]{CODEC_HIDDEN, CODEC_HIDDEN});
        tensor(w, 35, "decoder.lin.bias", new long[]{CODEC_HIDDEN});
        tensor(w, 36, "decoder.lfm.embedding_norm.weight", new long[]{CODEC_HIDDEN});
        tensor(w, 40, p + "operator_norm.weight", new long[]{CODEC_HIDDEN});
        tensor(w, 41, p + "post_operator_norm.weight", new long[]{CODEC_HIDDEN});
        tensor(w, 42, p + "self_attn.q_proj.weight", new long[]{CODEC_HIDDEN, CODEC_HIDDEN});
        tensor(w, 43, p + "self_attn.q_norm.weight", new long[]{dh});
        tensor(w, 44, p + "self_attn.k_proj.weight", new long[]{CODEC_HIDDEN, (long) CODEC_KV * dh});
        tensor(w, 45, p + "self_attn.k_norm.weight", new long[]{dh});
        tensor(w, 46, p + "self_attn.v_proj.weight", new long[]{CODEC_HIDDEN, (long) CODEC_KV * dh});
        tensor(w, 47, p + "self_attn.o_proj.weight", new long[]{CODEC_HIDDEN, CODEC_HIDDEN});
        tensor(w, 48, p + "self_attn.gate_proj.weight", new long[]{CODEC_HIDDEN, CODEC_HEADS});
        tensor(w, 49, p + "conv.in_proj.weight", new long[]{3L * CODEC_HIDDEN, CODEC_HIDDEN});
        tensor(w, 50, p + "conv.conv.weight", new long[]{CODEC_HIDDEN, 1, CODEC_TAPS});
        tensor(w, 51, p + "conv.out_proj.weight", new long[]{CODEC_HIDDEN, CODEC_HIDDEN});
        tensor(w, 52, p + "ffn_norm.weight", new long[]{CODEC_HIDDEN});
        tensor(w, 53, p + "post_ffn_norm.weight", new long[]{CODEC_HIDDEN});
        tensor(w, 54, p + "feed_forward.w1.weight", new long[]{CODEC_FF, CODEC_HIDDEN});
        tensor(w, 55, p + "feed_forward.w3.weight", new long[]{CODEC_FF, CODEC_HIDDEN});
        tensor(w, 56, p + "feed_forward.w2.weight", new long[]{CODEC_HIDDEN, CODEC_FF});
        w.metadata("format", "pt");
        w.build().write(codec.resolve("model.safetensors"));
    }

    /** The depthformer embedding page the checkpoint stores: {@code [vocab, width]}. */
    public static long[] depthEmbeddingShape() {
        return new long[]{AUDIO_VOCAB, DEPTH_HIDDEN};
    }

    /** The per-codebook norm page the checkpoint stores: {@code [1, width]}. */
    public static long[] depthNormShape() {
        return new long[]{1, DEPTH_HIDDEN};
    }

    // ------------------------------------------------------------------ tower shapes

    /** The GGUF names and ggml dims of the encoder and its projector. */
    private static void encoderShapes(Map<String, long[]> shapes) {
        shapes.put("a.conv1d.0.weight", new long[]{3, 3, 1, DOWNSAMPLE_CHANNELS[0]});
        shapes.put("a.conv1d.0.bias", new long[]{1, 1, DOWNSAMPLE_CHANNELS[0]});
        shapes.put("a.conv1d.2.weight", new long[]{3, 3, 1, DOWNSAMPLE_CHANNELS[0]});
        shapes.put("a.conv1d.2.bias", new long[]{1, 1, DOWNSAMPLE_CHANNELS[0]});
        shapes.put("a.conv1d.3.weight", new long[]{1, 1, DOWNSAMPLE_CHANNELS[0], DOWNSAMPLE_CHANNELS[1]});
        shapes.put("a.conv1d.3.bias", new long[]{1, 1, DOWNSAMPLE_CHANNELS[1]});
        shapes.put("a.conv1d.5.weight", new long[]{3, 3, 1, DOWNSAMPLE_CHANNELS[1]});
        shapes.put("a.conv1d.5.bias", new long[]{1, 1, DOWNSAMPLE_CHANNELS[1]});
        shapes.put("a.conv1d.6.weight", new long[]{1, 1, DOWNSAMPLE_CHANNELS[1], DOWNSAMPLE_CHANNELS[2]});
        shapes.put("a.conv1d.6.bias", new long[]{1, 1, DOWNSAMPLE_CHANNELS[2]});
        shapes.put("a.pre_encode.out.weight", new long[]{2L * DOWNSAMPLE_CHANNELS[2], HIDDEN});
        shapes.put("a.pre_encode.out.bias", new long[]{HIDDEN});
        int dh = HIDDEN / HEADS;
        for (int il = 0; il < LAYERS; il++) {
            String p = "a.blk." + il + ".";
            shapes.put(p + "ln1.weight", new long[]{HIDDEN});
            shapes.put(p + "ln2.weight", new long[]{HIDDEN});
            shapes.put(p + "ffn_up.weight", new long[]{HIDDEN, FF});
            if (il < 2) {
                shapes.put(p + "ffn_gate.weight", new long[]{HIDDEN, FF});
            }
            shapes.put(p + "ffn_down.weight", new long[]{FF, HIDDEN});
            if (CONV[il]) {
                shapes.put(p + "conv_pw1.weight", new long[]{HIDDEN, 3L * HIDDEN});
                shapes.put(p + "conv_dw.weight", new long[]{TAPS, HIDDEN});
                shapes.put(p + "conv_pw2.weight", new long[]{HIDDEN, HIDDEN});
            } else {
                shapes.put(p + "attn_q.weight", new long[]{HIDDEN, HIDDEN});
                shapes.put(p + "attn_k.weight", new long[]{HIDDEN, (long) KV_HEADS * dh});
                shapes.put(p + "attn_v.weight", new long[]{HIDDEN, (long) KV_HEADS * dh});
                shapes.put(p + "attn_out.weight", new long[]{HIDDEN, HIDDEN});
                shapes.put(p + "attn_q_norm.weight", new long[]{dh});
                shapes.put(p + "attn_k_norm.weight", new long[]{dh});
            }
        }
        shapes.put("a.post_ln.weight", new long[]{HIDDEN});
        shapes.put("mm.a.mlp.0.weight", new long[]{HIDDEN});
        shapes.put("mm.a.mlp.0.bias", new long[]{HIDDEN});
        shapes.put("mm.a.mlp.1.weight", new long[]{HIDDEN, ADAPTER});
        shapes.put("mm.a.mlp.1.bias", new long[]{ADAPTER});
        shapes.put("mm.a.mlp.3.weight", new long[]{ADAPTER, TEXT});
        shapes.put("mm.a.mlp.3.bias", new long[]{TEXT});
    }

    /** The GGUF names and ggml dims of the depthformer and its page tables. */
    private static void generatorShapes(Map<String, long[]> shapes) {
        int dh = DEPTH_HIDDEN / DEPTH_HEADS;
        for (int il = 0; il < DEPTH_LAYERS; il++) {
            String p = "a.gen.code.blk." + il + ".";
            shapes.put(p + "ln1.weight", new long[]{DEPTH_HIDDEN});
            shapes.put(p + "attn_q.weight", new long[]{DEPTH_HIDDEN, DEPTH_HIDDEN});
            shapes.put(p + "attn_q_norm.weight", new long[]{dh});
            shapes.put(p + "attn_k.weight", new long[]{DEPTH_HIDDEN, (long) DEPTH_KV * dh});
            shapes.put(p + "attn_k_norm.weight", new long[]{dh});
            shapes.put(p + "attn_v.weight", new long[]{DEPTH_HIDDEN, (long) DEPTH_KV * dh});
            shapes.put(p + "attn_out.weight", new long[]{DEPTH_HIDDEN, DEPTH_HIDDEN});
            shapes.put(p + "attn_gate.weight", new long[]{DEPTH_HIDDEN, DEPTH_HEADS});
            shapes.put(p + "ln2.weight", new long[]{DEPTH_HIDDEN});
            shapes.put(p + "ffn_gate.weight", new long[]{DEPTH_HIDDEN, DEPTH_FF});
            shapes.put(p + "ffn_up.weight", new long[]{DEPTH_HIDDEN, DEPTH_FF});
            shapes.put(p + "ffn_down.weight", new long[]{DEPTH_FF, DEPTH_HIDDEN});
        }
        shapes.put("a.gen.code.output_norm.weight", new long[]{DEPTH_HIDDEN});
        shapes.put("depth_linear.weight", new long[]{TEXT, DEPTH_HIDDEN});
        shapes.put("depth_linear.bias", new long[]{DEPTH_HIDDEN});
        shapes.put("mask_embedding", new long[]{DEPTH_HIDDEN});
        shapes.put("codebook_embedding.weight", new long[]{TEXT, (long) CODEBOOKS * AUDIO_VOCAB});
        shapes.put("a.gen.code.embd.weight", new long[]{DEPTH_HIDDEN, AUDIO_VOCAB, CODEBOOKS});
        shapes.put("a.gen.code.head_norm.weight", new long[]{DEPTH_HIDDEN, 1, CODEBOOKS});
    }

    // ------------------------------------------------------------------ HF side

    /** The HF names of one checkpoint layout. */
    private static final class Names {
        final String downsample;
        final String layer;
        final String postNorm;
        final String projector;

        Names(String downsample, String layer, String postNorm, String projector) {
            this.downsample = downsample;
            this.layer = layer;
            this.postNorm = postNorm;
            this.projector = projector;
        }
    }

    private static Names names(String audioPrefix) {
        return new Names(audioPrefix + "downsample.", audioPrefix + "lfm.lfm.layers.",
                audioPrefix + "lfm.lfm.embedding_norm.weight", "mlp.model.");
    }

    private static Names omniNames() {
        return new Names("model.audio_encoder.downsample.", "model.audio_encoder.lfm.layers.",
                "model.audio_encoder.lfm.embedding_norm.weight", "model.audio_projector.");
    }

    /** The audio encoder and projector tensors under the layout's names. */
    private static void audio(SyntheticHf.Builder b, Tower t, Names n, StDType dtype) {
        putWeight(b, t, n.downsample + "conv_in.weight", "a.conv1d.0.weight", dtype);
        putBias(b, t, n.downsample + "conv_in.bias", "a.conv1d.0.bias", dtype);
        putWeight(b, t, n.downsample + "conv_mid.weight", "a.conv1d.2.weight", dtype);
        putBias(b, t, n.downsample + "conv_mid.bias", "a.conv1d.2.bias", dtype);
        putWeight(b, t, n.downsample + "lin_mid.weight", "a.conv1d.3.weight", dtype);
        putBias(b, t, n.downsample + "lin_mid.bias", "a.conv1d.3.bias", dtype);
        putWeight(b, t, n.downsample + "conv_end.weight", "a.conv1d.5.weight", dtype);
        putBias(b, t, n.downsample + "conv_end.bias", "a.conv1d.5.bias", dtype);
        putWeight(b, t, n.downsample + "lin_end.weight", "a.conv1d.6.weight", dtype);
        putBias(b, t, n.downsample + "lin_end.bias", "a.conv1d.6.bias", dtype);
        long[] out = t.ne("a.pre_encode.out.weight");
        write(b, n.downsample + "lin_out.weight", new long[]{out[1], out[0], 1},
                t.values("a.pre_encode.out.weight"), dtype);
        putBias(b, t, n.downsample + "lin_out.bias", "a.pre_encode.out.bias", dtype);
        for (int il = 0; il < LAYERS; il++) {
            String p = "a.blk." + il + ".";
            String h = n.layer + il + ".";
            putWeight(b, t, h + "operator_norm.weight", p + "ln1.weight", dtype);
            putWeight(b, t, h + "ffn_norm.weight", p + "ln2.weight", dtype);
            putWeight(b, t, h + "feed_forward.w3.weight", p + "ffn_up.weight", dtype);
            if (t.ne(p + "ffn_gate.weight") != null) {
                putWeight(b, t, h + "feed_forward.w1.weight", p + "ffn_gate.weight", dtype);
            }
            putWeight(b, t, h + "feed_forward.w2.weight", p + "ffn_down.weight", dtype);
            if (CONV[il]) {
                putWeight(b, t, h + "conv.in_proj.weight", p + "conv_pw1.weight", dtype);
                long[] dw = t.ne(p + "conv_dw.weight");
                write(b, h + "conv.conv.weight", new long[]{dw[1], 1, dw[0]}, t.values(p + "conv_dw.weight"),
                        dtype);
                putWeight(b, t, h + "conv.out_proj.weight", p + "conv_pw2.weight", dtype);
            } else {
                putWeight(b, t, h + "self_attn.q_proj.weight", p + "attn_q.weight", dtype);
                putWeight(b, t, h + "self_attn.q_layernorm.weight", p + "attn_q_norm.weight", dtype);
                putWeight(b, t, h + "self_attn.k_proj.weight", p + "attn_k.weight", dtype);
                putWeight(b, t, h + "self_attn.k_layernorm.weight", p + "attn_k_norm.weight", dtype);
                putWeight(b, t, h + "self_attn.v_proj.weight", p + "attn_v.weight", dtype);
                putWeight(b, t, h + "self_attn.out_proj.weight", p + "attn_out.weight", dtype);
            }
        }
        putWeight(b, t, n.postNorm, "a.post_ln.weight", dtype);
        putWeight(b, t, projectorName(n, 0, "weight"), "mm.a.mlp.0.weight", dtype);
        putBias(b, t, projectorName(n, 0, "bias"), "mm.a.mlp.0.bias", dtype);
        putWeight(b, t, projectorName(n, 1, "weight"), "mm.a.mlp.1.weight", dtype);
        putBias(b, t, projectorName(n, 1, "bias"), "mm.a.mlp.1.bias", dtype);
        putWeight(b, t, projectorName(n, 3, "weight"), "mm.a.mlp.3.weight", dtype);
        putBias(b, t, projectorName(n, 3, "bias"), "mm.a.mlp.3.bias", dtype);
    }

    /** The audio projector module name: {@code mlp.model.i} or the Omni {@code layer_norm/linear_1/linear_2}. */
    private static String projectorName(Names n, int index, String suffix) {
        if ("mlp.model.".equals(n.projector)) {
            return n.projector + index + "." + suffix;
        }
        String module = index == 0 ? "layer_norm" : index == 1 ? "linear_1" : "linear_2";
        return n.projector + module + "." + suffix;
    }

    /** The depthformer and its tied per-codebook pages under the layout's names. */
    private static void generator(SyntheticHf.Builder b, Tower t, String depthformer, String embeddings,
                                  String codebook, String mask, String linear, StDType dtype) {
        for (int il = 0; il < DEPTH_LAYERS; il++) {
            String p = "a.gen.code.blk." + il + ".";
            String h = depthformer + "model.layers." + il + ".";
            putWeight(b, t, h + "operator_norm.weight", p + "ln1.weight", dtype);
            putWeight(b, t, h + "self_attn.q_proj.weight", p + "attn_q.weight", dtype);
            putWeight(b, t, h + "self_attn.q_norm.weight", p + "attn_q_norm.weight", dtype);
            putWeight(b, t, h + "self_attn.k_proj.weight", p + "attn_k.weight", dtype);
            putWeight(b, t, h + "self_attn.k_norm.weight", p + "attn_k_norm.weight", dtype);
            putWeight(b, t, h + "self_attn.v_proj.weight", p + "attn_v.weight", dtype);
            putWeight(b, t, h + "self_attn.o_proj.weight", p + "attn_out.weight", dtype);
            putWeight(b, t, h + "self_attn.gate_proj.weight", p + "attn_gate.weight", dtype);
            putWeight(b, t, h + "ffn_norm.weight", p + "ln2.weight", dtype);
            putWeight(b, t, h + "feed_forward.w1.weight", p + "ffn_gate.weight", dtype);
            putWeight(b, t, h + "feed_forward.w3.weight", p + "ffn_up.weight", dtype);
            putWeight(b, t, h + "feed_forward.w2.weight", p + "ffn_down.weight", dtype);
        }
        putWeight(b, t, depthformer + "model.embedding_norm.weight", "a.gen.code.output_norm.weight", dtype);
        putWeight(b, t, linear + "weight", "depth_linear.weight", dtype);
        putBias(b, t, linear + "bias", "depth_linear.bias", dtype);
        write(b, mask, t.ne("mask_embedding"), t.values("mask_embedding"), dtype);
        putWeight(b, t, codebook + "weight", "codebook_embedding.weight", dtype);
        long[] page = depthEmbeddingShape();
        long pageSize = count(page);
        float[] table = t.values("a.gen.code.embd.weight");
        float[] norms = t.values("a.gen.code.head_norm.weight");
        for (int i = 0; i < CODEBOOKS; i++) {
            float[] embed = Arrays.copyOfRange(table, Math.toIntExact(i * pageSize),
                    Math.toIntExact((i + 1) * pageSize));
            write(b, embeddings + i + ".embedding.weight", page, embed, dtype);
            write(b, embeddings + i + ".embedding_norm.weight", depthNormShape(),
                    Arrays.copyOfRange(norms, i * DEPTH_HIDDEN, (i + 1) * DEPTH_HIDDEN), dtype);
            write(b, embeddings + i + ".to_logits.weight", page, embed, dtype);
        }
    }

    // ------------------------------------------------------------------ tensor helpers

    /** Writes {@code ggufName}'s values under {@code hfName} as the reversal of the ggml dims. */
    private static void putWeight(SyntheticHf.Builder b, Tower t, String hfName, String ggufName, StDType dtype) {
        long[] ne = t.ne(ggufName);
        if (ne == null) {
            throw new IllegalArgumentException("the tower has no tensor '" + ggufName + "'");
        }
        long[] shape = new long[ne.length];
        for (int i = 0; i < ne.length; i++) {
            shape[i] = ne[ne.length - 1 - i];
        }
        write(b, hfName, shape, t.values(ggufName), dtype);
    }

    /** Writes a bias: the checkpoint stores it flat, the converter reshapes it back to the ggml dims. */
    private static void putBias(SyntheticHf.Builder b, Tower t, String hfName, String ggufName, StDType dtype) {
        long[] ne = t.ne(ggufName);
        if (ne == null) {
            throw new IllegalArgumentException("the tower has no tensor '" + ggufName + "'");
        }
        write(b, hfName, new long[]{count(ne)}, t.values(ggufName), dtype);
    }

    private static void write(SyntheticHf.Builder b, String name, long[] shape, float[] values, StDType dtype) {
        if (shape == null) {
            throw new IllegalArgumentException("no shape for '" + name + "'");
        }
        if (values.length != count(shape)) {
            throw new IllegalArgumentException(name + " holds " + values.length + " values, its shape is "
                    + Arrays.toString(shape));
        }
        switch (dtype) {
            case BF16:
                b.tensor(name, dtype, shape, SyntheticHf.bf16(values));
                break;
            case F16:
                b.tensor(name, dtype, shape, SyntheticHf.f16(values));
                break;
            case F32:
                b.tensor(name, dtype, shape, SyntheticHf.f32(values));
                break;
            default:
                throw new IllegalArgumentException("not a float dtype: " + dtype);
        }
    }

    private static void tensor(SafetensorsWriter.Builder w, int seed, String name, long[] shape) {
        w.add(name, StDType.F32, shape, SyntheticHf.f32(SyntheticHf.values(seed, Math.toIntExact(count(shape)))));
    }

    private static long count(long[] shape) {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("shape " + Arrays.toString(shape) + " holds too many values");
        }
        return n;
    }

    private static int fsqCodebookSize() {
        int n = 1;
        for (int level : FSQ_LEVELS) {
            n *= level;
        }
        return n;
    }

    // ------------------------------------------------------------------ configurations

    /** The {@code config.json} text of the audio-only checkpoint. */
    public static String audioConfigJson() {
        return "{'architectures':['Lfm3AudioForConditionalGeneration'],'model_type':'lfm3_audio',"
                + "'hidden_size':" + TEXT + ",'encoder':{'dim_in':" + N_MEL + ",'sliding_window':" + WINDOW + ","
                + "'lfm':{" + audioLfm() + "}},"
                + "'codebooks':" + CODEBOOKS + ",'audio_vocab_size':" + AUDIO_VOCAB + ",'maskgit_steps':3,"
                + "'interleaved_n_text':6,'interleaved_n_audio':12,"
                + "'depthformer':{" + depthformer() + "}}";
    }

    /** The {@code config.json} text of the Omni checkpoint. */
    public static String omniConfigJson() {
        return "{'architectures':['Lfm3OmniForConditionalGeneration'],'model_type':'lfm3_omni',"
                + "'hidden_size':" + TEXT + ",'projector_hidden_act':'gelu','projector_bias':true,"
                + "'projector_layer_norm_eps':1e-06,'downsample_factor':2,"
                + "'num_codebooks':" + CODEBOOKS + ",'codebook_size':" + AUDIO_VOCAB + ","
                + "'maskgit_schedule':[4,4,4],'interleaved_n_text':6,'interleaved_n_audio':12,"
                + "'audio_encoder_config':{'num_mel_bins':" + N_MEL + ",'sliding_window':" + WINDOW + ","
                + "'text_config':{" + audioLfm() + "}},"
                + "'depth_config':{" + depthformer() + "},"
                + "'vision_config':{'hidden_size':8,'intermediate_size':16,'num_attention_heads':2,"
                + "'num_hidden_layers':2,'num_key_value_heads':1,'patch_size':2,'temporal_patch_size':2,"
                + "'in_channels':3,'conv_kernel_size':3,'norm_eps':1e-06,"
                + "'rope_parameters':{'rope_theta':1000000.0}}}";
    }

    private static String audioLfm() {
        return "'hidden_size':" + HIDDEN + ",'num_hidden_layers':" + LAYERS + ",'num_attention_heads':" + HEADS
                + ",'num_key_value_heads':" + KV_HEADS + ",'norm_eps':1e-05,'block_ff_dim':" + FF + ","
                + "'block_auto_adjust_ff_dim':false,'block_ffn_dim_multiplier':null,'block_multiple_of':1,"
                + "'rope_parameters':{'rope_theta':1000000.0}";
    }

    private static String depthformer() {
        return "'hidden_size':" + DEPTH_HIDDEN + ",'intermediate_size':" + DEPTH_FF + ","
                + "'num_hidden_layers':" + DEPTH_LAYERS + ",'num_attention_heads':" + DEPTH_HEADS + ","
                + "'num_key_value_heads':" + DEPTH_KV + ",'norm_eps':1e-05,'is_causal':false,"
                + "'use_sandwich_norm':false,'layer_types':['full_attention','full_attention'],"
                + "'rope_parameters':{'full_attention':{'rope_theta':1000000.0}}";
    }

    private static String featureExtractor() {
        return "'feature_extractor_type':'Lfm3AudioFeatureExtractor','sampling_rate':16000,'feature_size':128,"
                + "'n_fft':400,'win_length':400,'hop_length':160,'log_floor':-12.0,'normalization_offset':4.0,"
                + "'normalization_divisor':4.0,'encoder_stride':8";
    }

    private static String preprocessor() {
        return "{'feature_extractor':{" + featureExtractor() + "}}";
    }

    private static String omniPreprocessor() {
        return "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5],"
                + "'size':{'shortest_edge':4096,'longest_edge':25165824},'resample':2,"
                + "'feature_extractor':{" + featureExtractor() + "}}";
    }

    private static String codecConfig() {
        return SyntheticHf.json("{'model_type':'liquid_codec','fsq_levels':[8,7,6,6],'n_q':12,"
                + "'quantizer_dim':256,'encoder_dim':512,'stft_n_fft':1280,'stft_hop_length':320,"
                + "'stft_win_length':1280,'stft_frame_stack':6,'decoder_window':30,'sampling_rate':24000}");
    }
}
