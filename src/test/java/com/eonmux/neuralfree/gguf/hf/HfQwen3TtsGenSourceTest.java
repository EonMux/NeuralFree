package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.PlannedTensor;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.speech.Qwen3TtsClip;
import com.eonmux.neuralfree.gguf.speech.Qwen3TtsSpeakerEncoder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The qwen3tts generator projector view ({@code convert_hf_to_gguf --mmproj}): the code predictor's stacked
 * codebook pages and the code2wav tensors generated from {@code speech_tokenizer}, their values against the
 * converter's math (qwen3tts.py:190-479), and the projector type and hyperparameters of the file.
 */
public class HfQwen3TtsGenSourceTest {

    private static final int TALKER = 6;
    private static final int PRED = 8;
    private static final int PRED_HEADS = 2;
    private static final int PRED_KV = 1;
    private static final int PRED_DHEAD = 4;
    private static final int PRED_FF = 12;
    private static final int PAGES = 15;
    private static final int VOCAB = 4;

    private static final int FIRST_DIM = 3;
    private static final int FIRST_VOCAB = 4;
    private static final int REST_DIM = 3;
    private static final int REST_VOCAB = 4;
    private static final int DECODER = 32;
    private static final int TFM_FF = 8;
    private static final int UP_FF = 8;
    private static final int DAC_CH = 3;
    private static final int TFM_LAYERS = 1;
    private static final int UP_STAGES = 1;
    private static final int DAC_BLOCKS = 1;

    /** Speaker-encoder geometry: eight channels split into eight Res2Net chunks, a two-value squeeze bottleneck. */
    private static final int SPK_CHANNELS = 8;
    private static final int SPK_CHUNK = 1;
    private static final int SPK_SE = 2;
    private static final int SPK_ASP = 4;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private ModelSourceOptions options() throws IOException {
        return ModelSourceOptions.builder().cacheDir(tmp.newFolder().toPath()).build();
    }

    // ------------------------------------------------------------------ checkpoint fixture

    private static int count(int... shape) {
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        return n;
    }

    /** {@link SyntheticHf#values} of {@code seed} over {@code shape}, the expected values of a planned tensor. */
    private float[] expected(int seed, int... shape) {
        return SyntheticHf.values(seed, count(shape));
    }

    /** Adds an F32 tensor of the main checkpoint file, remembering its seed. */
    private void f32(SyntheticHf.Builder b, int seed, String name, int... shape) {
        long[] dims = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            dims[i] = shape[i];
        }
        b.tensor(name, StDType.F32, dims, SyntheticHf.f32(expected(seed, shape)));
    }

    private static String config() {
        return SyntheticHf.json("{'architectures':['Qwen3TTSForConditionalGeneration'],'model_type':'qwen3_tts',"
                + "'talker_config':{'hidden_size':" + TALKER + ",'text_vocab_size':8,'vocab_size':16,"
                + "'codec_eos_token_id':5,'code_predictor_config':{'hidden_size':" + PRED
                + ",'intermediate_size':" + PRED_FF + ",'num_hidden_layers':1,'num_attention_heads':" + PRED_HEADS
                + ",'num_key_value_heads':" + PRED_KV + ",'rms_norm_eps':1e-5}}}");
    }

    private static String decoderConfig() {
        return SyntheticHf.json("{'decoder_config':{'num_hidden_layers':" + TFM_LAYERS + ","
                + "'upsampling_ratios':[2],'upsample_rates':[2]}}");
    }

    /** Which speaker encoder, if any, the checkpoint carries. */
    private enum Speaker {
        /** The CustomVoice / VoiceDesign checkpoint: no encoder at all. */
        NONE,
        /** A broken checkpoint holding one encoder tensor: not a complete encoder, so all of them are dropped. */
        PARTIAL,
        /** A voice-clone checkpoint: the complete ECAPA-TDNN. */
        COMPLETE
    }

    /** The main checkpoint file and the speech_tokenizer side files of a tiny qwen3tts generator. */
    private Path checkpoint() throws IOException {
        return checkpoint(Speaker.NONE);
    }

    private Path checkpoint(Speaker speaker) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config());
        f32(b, 10, "talker.model.codec_embedding.weight", VOCAB, TALKER);
        f32(b, 11, "talker.code_predictor.model.norm.weight", PRED);
        f32(b, 12, "talker.code_predictor.small_to_mtp_projection.weight", PRED, TALKER);
        f32(b, 13, "talker.code_predictor.small_to_mtp_projection.bias", PRED);
        int seed = 20;
        for (int page = 0; page < PAGES; page++) {
            f32(b, seed++, "talker.code_predictor.model.codec_embedding." + page + ".weight", VOCAB, TALKER);
        }
        for (int page = 0; page < PAGES; page++) {
            f32(b, seed++, "talker.code_predictor.lm_head." + page + ".weight", VOCAB, PRED);
        }
        String p = "talker.code_predictor.model.layers.0.";
        f32(b, 100, p + "input_layernorm.weight", PRED);
        f32(b, 101, p + "self_attn.q_proj.weight", PRED_HEADS * PRED_DHEAD, PRED);
        f32(b, 102, p + "self_attn.q_norm.weight", PRED_DHEAD);
        f32(b, 103, p + "self_attn.k_proj.weight", PRED_KV * PRED_DHEAD, PRED);
        f32(b, 104, p + "self_attn.k_norm.weight", PRED_DHEAD);
        f32(b, 105, p + "self_attn.v_proj.weight", PRED_KV * PRED_DHEAD, PRED);
        f32(b, 106, p + "self_attn.o_proj.weight", PRED, PRED_HEADS * PRED_DHEAD);
        f32(b, 107, p + "post_attention_layernorm.weight", PRED);
        f32(b, 108, p + "mlp.gate_proj.weight", PRED_FF, PRED);
        f32(b, 109, p + "mlp.up_proj.weight", PRED_FF, PRED);
        f32(b, 110, p + "mlp.down_proj.weight", PRED, PRED_FF);
        // the voice-clone encoder, or a lone piece of one in the partial variant
        if (speaker == Speaker.COMPLETE) {
            speakerEncoder(b);
        } else if (speaker == Speaker.PARTIAL) {
            f32(b, 120, "speaker_encoder.fc.weight", 16, 16);
        }
        Path dir = b.write(tmp.newFolder("qwen3tts").toPath());
        writeSpeechTokenizer(dir);
        return dir;
    }

    /**
     * The ECAPA-TDNN speaker encoder of a voice-clone checkpoint, with the reference geometry cut down to eight
     * channels: a five-tap stem over the 128 mel bins, three SE-Res2Net blocks of one-channel Res2Net chunks, the
     * 1536-value aggregation convolution, the attentive-pooling pair and the talker-width projection.
     */
    private void speakerEncoder(SyntheticHf.Builder b) {
        int channels = SPK_CHANNELS;
        int chunk = SPK_CHUNK;
        f32(b, 500, "speaker_encoder.blocks.0.conv.weight", channels, 128, 5);
        f32(b, 501, "speaker_encoder.blocks.0.conv.bias", channels);
        int seed = 510;
        for (int bid = 1; bid <= 3; bid++) {
            String p = "speaker_encoder.blocks." + bid + ".";
            f32(b, seed++, p + "tdnn1.conv.weight", channels, channels, 1);
            f32(b, seed++, p + "tdnn1.conv.bias", channels);
            f32(b, seed++, p + "tdnn2.conv.weight", channels, channels, 1);
            f32(b, seed++, p + "tdnn2.conv.bias", channels);
            f32(b, seed++, p + "se_block.conv1.weight", SPK_SE, channels, 1);
            f32(b, seed++, p + "se_block.conv1.bias", SPK_SE);
            f32(b, seed++, p + "se_block.conv2.weight", channels, SPK_SE, 1);
            f32(b, seed++, p + "se_block.conv2.bias", channels);
            for (int xid = 0; xid < 7; xid++) {
                String r = p + "res2net_block.blocks." + xid + ".";
                f32(b, seed++, r + "conv.weight", chunk, chunk, 3);
                f32(b, seed++, r + "conv.bias", chunk);
            }
        }
        f32(b, 600, "speaker_encoder.mfa.conv.weight", channels, 3 * channels, 1);
        f32(b, 601, "speaker_encoder.mfa.conv.bias", channels);
        f32(b, 610, "speaker_encoder.asp.tdnn.conv.weight", SPK_ASP, 3 * channels, 1);
        f32(b, 611, "speaker_encoder.asp.tdnn.conv.bias", SPK_ASP);
        f32(b, 620, "speaker_encoder.asp.conv.weight", channels, SPK_ASP, 1);
        f32(b, 621, "speaker_encoder.asp.conv.bias", channels);
        f32(b, 630, "speaker_encoder.fc.weight", TALKER, 2 * channels, 1);
        f32(b, 631, "speaker_encoder.fc.bias", TALKER);
    }

    /** The code2wav tensors of {@code speech_tokenizer/model.safetensors}. */
    private void writeSpeechTokenizer(Path dir) throws IOException {
        Path at = dir.resolve("speech_tokenizer");
        Files.createDirectories(at);
        Files.write(at.resolve("config.json"), decoderConfig().getBytes(java.nio.charset.StandardCharsets.UTF_8));
        SafetensorsWriter.Builder w = SafetensorsWriter.builder();
        side(w, 200, "decoder.quantizer.rvq_first.input_proj.weight", FIRST_VOCAB, FIRST_DIM, 1);
        side(w, 201, "decoder.quantizer.rvq_first.output_proj.weight", DECODER, FIRST_DIM, 1);
        side(w, 202, "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum", FIRST_VOCAB, FIRST_DIM);
        side(w, 203, "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage", FIRST_VOCAB);
        side(w, 204, "decoder.quantizer.rvq_rest.input_proj.weight", REST_VOCAB, REST_DIM, 1);
        side(w, 205, "decoder.quantizer.rvq_rest.output_proj.weight", DECODER, REST_DIM, 1);
        int seed = 210;
        for (int layer = 0; layer < PAGES; layer++) {
            side(w, seed++, "decoder.quantizer.rvq_rest.vq.layers." + layer + "._codebook.embedding_sum",
                    REST_VOCAB, REST_DIM);
            side(w, seed++, "decoder.quantizer.rvq_rest.vq.layers." + layer + "._codebook.cluster_usage",
                    REST_VOCAB);
        }
        side(w, 300, "decoder.pre_conv.conv.weight", DECODER, DECODER, 2);
        side(w, 301, "decoder.pre_conv.conv.bias", DECODER);
        side(w, 310, "decoder.pre_transformer.input_proj.weight", DECODER, DECODER);
        side(w, 311, "decoder.pre_transformer.input_proj.bias", DECODER);
        side(w, 312, "decoder.pre_transformer.output_proj.weight", DECODER, DECODER);
        side(w, 313, "decoder.pre_transformer.output_proj.bias", DECODER);
        side(w, 314, "decoder.pre_transformer.norm.weight", DECODER);
        String p = "decoder.pre_transformer.layers.0.";
        side(w, 320, p + "input_layernorm.weight", DECODER);
        side(w, 321, p + "self_attn.q_proj.weight", DECODER, DECODER);
        side(w, 322, p + "self_attn.k_proj.weight", DECODER, DECODER);
        side(w, 323, p + "self_attn.v_proj.weight", DECODER, DECODER);
        side(w, 324, p + "self_attn.o_proj.weight", DECODER, DECODER);
        side(w, 325, p + "self_attn_layer_scale.scale", DECODER);
        side(w, 326, p + "post_attention_layernorm.weight", DECODER);
        side(w, 327, p + "mlp.gate_proj.weight", TFM_FF, DECODER);
        side(w, 328, p + "mlp.up_proj.weight", TFM_FF, DECODER);
        side(w, 329, p + "mlp.down_proj.weight", DECODER, TFM_FF);
        side(w, 330, p + "mlp_layer_scale.scale", DECODER);
        side(w, 340, "decoder.upsample.0.0.conv.weight", DECODER, DECODER, 2);
        side(w, 341, "decoder.upsample.0.0.conv.bias", DECODER);
        side(w, 342, "decoder.upsample.0.1.dwconv.conv.weight", DECODER, 1, 2);
        side(w, 343, "decoder.upsample.0.1.dwconv.conv.bias", DECODER);
        side(w, 344, "decoder.upsample.0.1.norm.weight", DECODER);
        side(w, 345, "decoder.upsample.0.1.norm.bias", DECODER);
        side(w, 346, "decoder.upsample.0.1.pwconv1.weight", UP_FF, DECODER);
        side(w, 347, "decoder.upsample.0.1.pwconv1.bias", UP_FF);
        side(w, 348, "decoder.upsample.0.1.pwconv2.weight", DECODER, UP_FF);
        side(w, 349, "decoder.upsample.0.1.pwconv2.bias", DECODER);
        side(w, 350, "decoder.upsample.0.1.gamma", DECODER);
        // the entry convolution is stored as a weight-norm pair
        side(w, 360, "decoder.decoder.0.conv.weight_g", DECODER, 1, 1);
        side(w, 361, "decoder.decoder.0.conv.weight_v", DECODER, DECODER, 2);
        side(w, 362, "decoder.decoder.0.conv.bias", DECODER);
        side(w, 370, "decoder.decoder.1.block.0.alpha", DECODER);
        side(w, 371, "decoder.decoder.1.block.0.beta", DECODER);
        side(w, 372, "decoder.decoder.1.block.1.conv.weight", DECODER, DAC_CH, 4);
        side(w, 373, "decoder.decoder.1.block.1.conv.bias", DAC_CH);
        for (int xid = 0; xid < 3; xid++) {
            String r = "decoder.decoder.1.block." + (xid + 2) + ".";
            side(w, 380 + xid * 10, r + "act1.alpha", DAC_CH);
            side(w, 381 + xid * 10, r + "act1.beta", DAC_CH);
            side(w, 382 + xid * 10, r + "conv1.conv.weight", DAC_CH, DAC_CH, 3);
            side(w, 383 + xid * 10, r + "conv1.conv.bias", DAC_CH);
            side(w, 384 + xid * 10, r + "act2.alpha", DAC_CH);
            side(w, 385 + xid * 10, r + "act2.beta", DAC_CH);
            side(w, 386 + xid * 10, r + "conv2.conv.weight", DAC_CH, DAC_CH, 1);
            side(w, 387 + xid * 10, r + "conv2.conv.bias", DAC_CH);
        }
        side(w, 420, "decoder.decoder.5.alpha", DAC_CH);
        side(w, 421, "decoder.decoder.5.beta", DAC_CH);
        side(w, 422, "decoder.decoder.6.conv.weight", 1, DAC_CH, 2);
        side(w, 423, "decoder.decoder.6.conv.bias", 1);
        w.metadata("format", "pt");
        w.build().write(at.resolve("model.safetensors"));
    }

    private void side(SafetensorsWriter.Builder w, int seed, String name, int... shape) {
        long[] dims = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            dims[i] = shape[i];
        }
        w.add(name, StDType.F32, dims, SyntheticHf.f32(SyntheticHf.values(seed, count(shape))));
    }

    // ------------------------------------------------------------------ tests

    private static float[] floats(HfModelSource s, String name) throws IOException {
        return s.plan().readF32(name, HfPlanView.MAX_READ_ELEMENTS);
    }

    @Test
    public void theProjectorCarriesTheGeneratorMetadataAndTensors() throws IOException {
        try (HfModelSource s = HfModelSource.openProjector(checkpoint(), options())) {
            GgufKv kv = GgufKv.of(s);
            assertEquals("clip", kv.archName());
            assertTrue(kv.bool(kv.key(GgufKeys.CLIP_HAS_GEN_AUDIO_ENCODER), false));
            assertEquals("qwen3tts_gen", kv.str(kv.key(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE), null));
            assertEquals(TALKER, kv.u32(kv.key(GgufKeys.CLIP_GEN_AUDIO_PROJECTION_DIM), 0));
            assertEquals(PRED, kv.u32(kv.key(GgufKeys.CLIP_GEN_AUDIO_EMBEDDING_LENGTH), 0));
            assertEquals(PRED_FF, kv.u32(kv.key(GgufKeys.CLIP_GEN_AUDIO_FEED_FORWARD_LENGTH), 0));
            assertEquals(1, kv.u32(kv.key(GgufKeys.CLIP_GEN_AUDIO_BLOCK_COUNT), 0));
            assertEquals(PRED_HEADS, kv.u32(kv.key(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT), 0));
            assertEquals(PRED_KV, kv.u32(kv.key(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT_KV), 0));
            assertEquals(1e-5f, kv.f32(kv.key(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_LAYER_NORM_EPSILON), 0f), 0f);
            assertEquals("clip", s.getMetadata().get(GgufKeys.GENERAL_ARCHITECTURE));
            for (String name : new String[]{"a.gen.code.out_embd.weight", "a.gen.code.output_norm.weight",
                    "a.gen.code.proj_in.weight", "a.gen.code.proj_in.bias", "a.gen.code.embd.weight",
                    "a.gen.code.head.weight", "a.gen.code.blk.0.attn_q_norm.weight",
                    "a.gen.code.blk.0.ffn_down.weight",
                    "a.gen.wav.quant.first.in_proj.weight", "a.gen.wav.quant.first.out_proj.weight",
                    "a.gen.wav.quant.rest.in_proj.weight", "a.gen.wav.quant.rest.out_proj.weight",
                    "a.gen.wav.quant.first.codebook.weight",
                    "a.gen.wav.quant.rest.codebook.weight", "a.gen.wav.pre_conv.weight",
                    "a.gen.wav.tfm.in_proj.weight", "a.gen.wav.tfm.out_proj.weight",
                    "a.gen.wav.tfm.output_norm.weight", "a.gen.wav.tfm.blk.0.ln1.weight",
                    "a.gen.wav.tfm.blk.0.ls1.weight", "a.gen.wav.tfm.blk.0.attn_q.weight",
                    "a.gen.wav.tfm.blk.0.ffn_down.weight", "a.gen.wav.tfm.blk.0.ls2.weight",
                    "a.gen.wav.up.blk.0.conv.weight", "a.gen.wav.up.blk.0.dwconv.weight",
                    "a.gen.wav.up.blk.0.gamma", "a.gen.wav.dac.blk.0.res.0.conv1.weight",
                    "a.gen.wav.dac.entry.weight", "a.gen.wav.dac.post_snake.alpha",
                    "a.gen.wav.dac.post_conv.weight"}) {
                assertTrue(name + " is planned", s.plan().has(name));
            }
            for (PlannedTensor t : s.plan().tensors()) {
                assertFalse("a checkpoint without a complete speaker encoder plans no encoder tensor: "
                        + t.ggufName(), t.ggufName().startsWith("a.conv1d.") || t.ggufName().startsWith("a.asp_")
                        || t.ggufName().startsWith("mm.a.fc"));
            }
            assertFalse("the generator alone declares no audio encoder",
                    kv.bool(kv.key(GgufKeys.CLIP_HAS_AUDIO_ENCODER), false));
            assertEquals(null, kv.str(kv.key(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE), null));
        }
    }

    @Test
    public void anIncompleteSpeakerEncoderIsDropped() throws IOException {
        try (HfModelSource s = HfModelSource.openProjector(checkpoint(Speaker.PARTIAL), options())) {
            GgufKv kv = GgufKv.of(s);
            assertFalse("a lone projection does not make an encoder",
                    kv.bool(kv.key(GgufKeys.CLIP_HAS_AUDIO_ENCODER), false));
            assertFalse("the lone projection is dropped, not planned", s.plan().has("mm.a.fc.weight"));
        }
    }

    @Test
    public void aVoiceCloneCheckpointCarriesTheSpeakerEncoder() throws IOException {
        Path dir = checkpoint(Speaker.COMPLETE);
        ModelSourceOptions options = options();
        Path out = tmp.getRoot().toPath().resolve("qwen3tts-voice-clone.gguf");
        try (HfModelSource s = HfModelSource.openProjector(dir, options)) {
            GgufKv kv = GgufKv.of(s);
            assertTrue(kv.bool(kv.key(GgufKeys.CLIP_HAS_AUDIO_ENCODER), false));
            assertEquals(Qwen3TtsSpeakerEncoder.PROJECTOR_TYPE,
                    kv.str(kv.key(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE), null));
            assertEquals(TALKER, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_PROJECTION_DIM), 0));
            assertEquals(128, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS), 0));
            assertEquals(3, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_BLOCK_COUNT), 0));
            assertEquals(1536, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_EMBEDDING_LENGTH), 0));
            assertEquals(1, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_ATTENTION_HEAD_COUNT), 0));
            assertEquals(1536, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_FEED_FORWARD_LENGTH), 0));
            assertEquals(1e-5f, kv.f32(kv.key(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON), 0f), 0f);
            // the checkpoint's [out, in, kernel] convolutions become the GGUF's [kernel, in, out] layouts
            assertArrayEquals(new long[]{5, 128, SPK_CHANNELS}, s.plan().ne("a.conv1d.0.weight"));
            assertArrayEquals(new long[]{1, SPK_CHANNELS, SPK_CHANNELS}, s.plan().ne("a.blk.1.conv_pw1.weight"));
            assertArrayEquals(new long[]{1, SPK_CHANNELS, SPK_SE}, s.plan().ne("a.blk.1.se_conv1.weight"));
            assertArrayEquals(new long[]{3, SPK_CHUNK, SPK_CHUNK}, s.plan().ne("a.blk.1.res2.0.weight"));
            assertArrayEquals(new long[]{1, 3 * SPK_CHANNELS, SPK_CHANNELS}, s.plan().ne("a.conv_out.weight"));
            assertArrayEquals(new long[]{1, SPK_ASP, SPK_CHANNELS}, s.plan().ne("a.asp_attn.weight"));
            assertArrayEquals(new long[]{1, 2 * SPK_CHANNELS, TALKER}, s.plan().ne("mm.a.fc.weight"));
            assertArrayEquals(values(501, SPK_CHANNELS), floats(s, "a.conv1d.0.bias"), 0f);
            assertArrayEquals(values(610, SPK_ASP, 3 * SPK_CHANNELS), floats(s, "a.asp_tdnn.weight"), 0f);
            GgufWriter.write(s, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (Qwen3TtsClip clip = Qwen3TtsClip.open(out.toString(), options)) {
            assertNotNull(clip.speakerEncoder());
            assertEquals(TALKER, clip.speakerEncoder().width());
            float[] samples = new float[24000];
            for (int i = 0; i < samples.length; i++) {
                samples[i] = (float) (0.2 * Math.sin(2.0 * Math.PI * 220.0 * i / 24000.0));
            }
            float[][] rows = clip.speakerEncoder().encode(new AudioClip(samples, 24000));
            assertEquals(1, rows.length);
            assertEquals(TALKER, rows[0].length);
            for (float value : rows[0]) {
                assertTrue("the embedding is finite", Float.isFinite(value));
            }
        }
    }

    @Test
    public void theCodebookPagesAreStackedInPageOrder() throws IOException {
        try (HfModelSource s = HfModelSource.openProjector(checkpoint(), options())) {
            assertArrayEquals(new long[]{TALKER, VOCAB, PAGES}, s.plan().ne("a.gen.code.embd.weight"));
            assertArrayEquals(new long[]{PRED, VOCAB, PAGES}, s.plan().ne("a.gen.code.head.weight"));
            float[] embd = floats(s, "a.gen.code.embd.weight");
            float[] head = floats(s, "a.gen.code.head.weight");
            assertEquals(PAGES * VOCAB * TALKER, embd.length);
            assertEquals(PAGES * VOCAB * PRED, head.length);
            for (int page = 0; page < PAGES; page++) {
                float[] e = expected(20 + page, VOCAB, TALKER);
                float[] h = expected(20 + PAGES + page, VOCAB, PRED);
                for (int i = 0; i < e.length; i++) {
                    assertEquals("codebook page " + page + " value " + i, e[i], embd[page * e.length + i], 0f);
                }
                for (int i = 0; i < h.length; i++) {
                    assertEquals("head page " + page + " value " + i, h[i], head[page * h.length + i], 0f);
                }
            }
        }
    }

    @Test
    public void theCodebooksFoldTheEmaStatistics() throws IOException {
        try (HfModelSource s = HfModelSource.openProjector(checkpoint(), options())) {
            assertArrayEquals(new long[]{FIRST_DIM, FIRST_VOCAB},
                    s.plan().ne("a.gen.wav.quant.first.codebook.weight"));
            assertArrayEquals(new long[]{REST_DIM, REST_VOCAB, PAGES},
                    s.plan().ne("a.gen.wav.quant.rest.codebook.weight"));
            assertCodebook(s, "a.gen.wav.quant.first.codebook.weight", 202, 203, FIRST_VOCAB, FIRST_DIM, 0);
            float[] rest = floats(s, "a.gen.wav.quant.rest.codebook.weight");
            for (int layer = 0; layer < PAGES; layer++) {
                int sum = 210 + layer * 2;
                float[] expected = folded(values(sum, REST_VOCAB, REST_DIM), values(sum + 1, REST_VOCAB),
                        REST_VOCAB, REST_DIM);
                for (int i = 0; i < expected.length; i++) {
                    assertEquals("rest codebook " + layer + " value " + i, expected[i],
                            rest[layer * expected.length + i], 0f);
                }
            }
        }
    }

    private void assertCodebook(HfModelSource s, String tensor, int sumSeed, int usageSeed, int vocab, int dim,
                                int layer) throws IOException {
        float[] got = floats(s, tensor);
        float[] expected = folded(values(sumSeed, vocab, dim), values(usageSeed, vocab), vocab, dim);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(tensor + " value " + i, expected[i], got[layer * expected.length + i], 0f);
        }
    }

    private static float[] values(int seed, int... shape) {
        return SyntheticHf.values(seed, count(shape));
    }

    private static float[] folded(float[] sum, float[] usage, int vocab, int dim) {
        float[] out = sum.clone();
        for (int r = 0; r < vocab; r++) {
            float divisor = Math.max(usage[r], 1e-5f);
            for (int d = 0; d < dim; d++) {
                int at = r * dim + d;
                out[at] = out[at] / divisor;
            }
        }
        return out;
    }

    @Test
    public void theSnakeParametersFoldTheirExponentials() throws IOException {
        try (HfModelSource s = HfModelSource.openProjector(checkpoint(), options())) {
            float[] alpha = floats(s, "a.gen.wav.dac.blk.0.snake.alpha");
            float[] beta = floats(s, "a.gen.wav.dac.blk.0.snake.beta");
            float[] wantA = exp(values(370, DECODER), true);
            float[] wantB = exp(values(371, DECODER), false);
            assertArrayEquals(wantA, alpha, 0f);
            assertArrayEquals(wantB, beta, 0f);
            float[] postBeta = floats(s, "a.gen.wav.dac.post_snake.beta");
            assertArrayEquals(exp(values(421, DAC_CH), false), postBeta, 0f);
        }
    }

    private static float[] exp(float[] v, boolean alpha) {
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            double e = Math.exp(v[i]);
            out[i] = alpha ? (float) e : (float) (1.0 / (e + 1e-9));
        }
        return out;
    }

    @Test
    public void theCode2WavTensorsKeepTheConvertersValues() throws IOException {
        try (HfModelSource s = HfModelSource.openProjector(checkpoint(), options())) {
            // the quantizer's 1x1 kernels drop their trailing unit dimension, values in place
            assertArrayEquals(new long[]{FIRST_DIM, FIRST_VOCAB},
                    s.plan().ne("a.gen.wav.quant.first.in_proj.weight"));
            assertArrayEquals(values(200, FIRST_VOCAB, FIRST_DIM),
                    floats(s, "a.gen.wav.quant.first.in_proj.weight"), 1e-3f);
            // convolutions keep the checkpoint's [in, out, kernel] layout, ne reversed
            assertArrayEquals(new long[]{2, DECODER, DECODER}, s.plan().ne("a.gen.wav.pre_conv.weight"));
            assertArrayEquals(values(300, DECODER, DECODER, 2),
                    floats(s, "a.gen.wav.pre_conv.weight"), 1e-3f);
            assertArrayEquals(values(314, DECODER), floats(s, "a.gen.wav.tfm.output_norm.weight"), 1e-3f);
            assertArrayEquals(values(350, DECODER), floats(s, "a.gen.wav.up.blk.0.gamma"), 1e-3f);
            assertArrayEquals(values(382, DAC_CH, DAC_CH, 3),
                    floats(s, "a.gen.wav.dac.blk.0.res.0.conv1.weight"), 1e-3f);
        }
    }

    @Test
    public void aWeightNormConvolutionFoldsIntoTheKernel() throws IOException {
        try (HfModelSource s = HfModelSource.openProjector(checkpoint(), options())) {
            float[] gain = values(360, DECODER);
            float[] direction = values(361, DECODER, DECODER, 2);
            float[] want = new float[direction.length];
            int per = direction.length / DECODER;
            for (int r = 0; r < DECODER; r++) {
                double sum = 0.0;
                for (int i = 0; i < per; i++) {
                    double x = direction[r * per + i];
                    sum += x * x;
                }
                float factor = (float) (gain[r] / Math.sqrt(sum));
                for (int i = 0; i < per; i++) {
                    want[r * per + i] = (float) (direction[r * per + i] * factor);
                }
            }
            // the entry kernel is stored F16, so the planned tensor carries its rounded values
            assertArrayEquals(want, floats(s, "a.gen.wav.dac.entry.weight"), 1e-4f);
        }
    }

    @Test
    public void theConvolutionKernelsKeepTheirStorageTypes() throws IOException {
        try (HfModelSource s = HfModelSource.openProjector(checkpoint(), options())) {
            assertEquals(GGMLType.F16, s.plan().type("a.gen.wav.pre_conv.weight"));
            assertEquals(GGMLType.F16, s.plan().type("a.gen.wav.dac.entry.weight"));
            assertEquals(GGMLType.F16, s.plan().type("a.gen.wav.dac.post_conv.weight"));
            assertEquals(GGMLType.F16, s.plan().type("a.gen.wav.up.blk.0.dwconv.weight"));
            assertEquals(GGMLType.F16, s.plan().type("a.gen.wav.dac.blk.0.res.0.conv1.weight"));
            assertEquals(GGMLType.F16, s.plan().type("a.gen.wav.dac.blk.0.res.0.conv2.weight"));
            assertEquals(GGMLType.F32, s.plan().type("a.gen.wav.up.blk.0.conv.weight"));
            assertEquals(GGMLType.F32, s.plan().type("a.gen.wav.dac.blk.0.conv.weight"));
            assertEquals(GGMLType.F32, s.plan().type("a.gen.code.blk.0.ffn_down.weight"));
        }
    }

    @Test
    public void aMissingCodebookPageRefusesTheConversion() throws IOException {
        Path dir = checkpoint();
        try (HfModelSource s = HfModelSource.openProjector(dir, options())) {
            assertTrue(s.plan().has("a.gen.code.embd.weight"));
        }
        // drop one page from the main checkpoint and reload
        SyntheticHf.Builder b = SyntheticHf.builder().config(config());
        f32(b, 10, "talker.model.codec_embedding.weight", VOCAB, TALKER);
        f32(b, 11, "talker.code_predictor.model.norm.weight", PRED);
        for (int page = 0; page < PAGES - 1; page++) {
            f32(b, 20 + page, "talker.code_predictor.model.codec_embedding." + page + ".weight", VOCAB, TALKER);
        }
        for (int page = 0; page < PAGES; page++) {
            f32(b, 20 + PAGES + page, "talker.code_predictor.lm_head." + page + ".weight", VOCAB, PRED);
        }
        Path incomplete = b.write(tmp.newFolder("incomplete").toPath());
        writeSpeechTokenizer(incomplete);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfModelSource.openProjector(incomplete, options()));
        assertTrue(e.getMessage(), e.getMessage().contains("incomplete"));
        assertTrue(e.getMessage(), e.getMessage().contains("14"));
    }

    @Test
    public void aCheckpointWithoutItsSpeechTokenizerRefuses() throws IOException {
        Path dir = checkpoint();
        Files.delete(dir.resolve("speech_tokenizer").resolve("config.json"));
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfModelSource.openProjector(dir, options()));
        assertTrue(e.getMessage(), e.getMessage().contains("speech_tokenizer/config.json"));
    }

    @Test
    public void theExportedFileOpensAsAQwen3TtsGenerator() throws IOException {
        Path dir = checkpoint();
        ModelSourceOptions options = options();
        Path out = tmp.getRoot().toPath().resolve("qwen3tts-gen.gguf");
        // the seam nftool export --mmproj resolves a checkpoint through
        try (ModelSource s = ModelSources.openProjector(dir, options)) {
            GgufWriter.write(s, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (Qwen3TtsClip clip = Qwen3TtsClip.open(out.toString(), options)) {
            assertEquals(PRED, clip.predictor().nEmbd());
            assertEquals(TALKER, clip.predictor().talkerWidth());
            assertEquals(PAGES, clip.predictor().nCodebooks());
            assertEquals(PAGES + 1, clip.predictor().codesPerFrame());
            assertEquals(PAGES, clip.code2Wav().codebooks());
            assertEquals(4, clip.code2Wav().hop());
            assertEquals(TFM_LAYERS, clip.code2Wav().nLayer());
        }
    }
}
