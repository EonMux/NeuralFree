package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.mm.MmprojModel;
import com.eonmux.neuralfree.gguf.speech.PocketTtsClip;
import com.eonmux.neuralfree.gguf.speech.PocketTtsFlow;
import com.eonmux.neuralfree.gguf.speech.PocketTtsSpeakerEncoder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The pockettts generator exporter ({@code PocketTTSMmprojModel}, conversion/pockettts.py): a synthetic Hugging
 * Face checkpoint converts to an mmproj GGUF whose metadata and tensor names the runtime graphs of
 * {@link PocketTtsClip} read, the file validates as a GEN_AUDIO mmproj, and a selected weight reaches the flow
 * network unchanged.
 */
public class HfPocketTtsGenRoundTripTest {

    private static final int EMB = 8;
    private static final int LATENT = 3;
    private static final int FLOW = 4;
    private static final int FREQ = 2;
    private static final int WIDTH = 64;
    private static final int FF = 64;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private ModelSourceOptions options() throws IOException {
        Path cache = tmp.getRoot().toPath().resolve("cache");
        java.nio.file.Files.createDirectories(cache);
        return ModelSourceOptions.builder().cacheDir(cache).build();
    }

    private static void flowTensor(SyntheticHf.Builder b, String name, long... shape) {
        b.f32(name, shape);
    }

    private Path checkpoint() throws IOException {
        return checkpoint(false);
    }

    /** @param speakerEncoder whether the checkpoint also carries the voice-prompt encoder tensors */
    private Path checkpoint(boolean speakerEncoder) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder()
                .config(SyntheticHf.json("{'architectures':['PocketTTSModel'],'model_type':'pockettts',"
                        + "'hidden_size':" + EMB + "}"));
        // the text backbone tensors the checkpoint detector reads; the generator view drops them
        b.f32("flow_lm.bos_emb", EMB);
        b.f32("flow_lm.conditioner.embed.weight", 16, EMB);
        b.f32("flow_lm.transformer.layers.0.norm1.weight", EMB);
        b.f32("flow_lm.transformer.layers.0.linear1.weight", 16, EMB);
        if (speakerEncoder) {
            speakerEncoder(b);
        } else {
            // the speaker-encoder transformer, present in every checkpoint and dropped by the generator view
            b.f32("mimi.encoder_transformer.transformer.layers.0.norm1.weight", WIDTH);
            b.f32("mimi.encoder_transformer.transformer.layers.0.linear1.weight", FF, WIDTH);
        }

        // the latent projection, with a known column so the round trip can read it back
        float[] inputLinear = new float[LATENT * EMB];
        for (int o = 0; o < EMB; o++) {
            for (int i = 0; i < LATENT; i++) {
                inputLinear[o * LATENT + i] = o * EMB + i;
            }
        }
        b.tensor("flow_lm.input_linear.weight", StDType.F32, new long[]{EMB, LATENT},
                SyntheticHf.f32(inputLinear));
        b.f32("flow_lm.emb_mean", LATENT);
        b.f32("flow_lm.emb_std", LATENT);
        b.f32("flow_lm.out_eos.weight", 1, EMB);
        b.f32("flow_lm.out_eos.bias", 1);
        b.f32("flow_lm.flow_net.input_proj.weight", FLOW, LATENT);
        b.f32("flow_lm.flow_net.input_proj.bias", FLOW);
        b.f32("flow_lm.flow_net.cond_embed.weight", FLOW, EMB);
        b.f32("flow_lm.flow_net.cond_embed.bias", FLOW);
        for (int t = 0; t < 2; t++) {
            String p = "flow_lm.flow_net.time_embed." + t + ".";
            b.f32(p + "freqs", FREQ);
            b.f32(p + "mlp.0.weight", FLOW, 2 * FREQ);
            b.f32(p + "mlp.0.bias", FLOW);
            b.f32(p + "mlp.2.weight", FLOW, FLOW);
            b.f32(p + "mlp.2.bias", FLOW);
            b.f32(p + "mlp.3.alpha", FLOW);
        }
        for (int r = 0; r < 2; r++) {
            String p = "flow_lm.flow_net.res_blocks." + r + ".";
            b.f32(p + "in_ln.weight", FLOW);
            b.f32(p + "in_ln.bias", FLOW);
            b.f32(p + "mlp.0.weight", FLOW, FLOW);
            b.f32(p + "mlp.0.bias", FLOW);
            b.f32(p + "mlp.2.weight", FLOW, FLOW);
            b.f32(p + "mlp.2.bias", FLOW);
            b.f32(p + "adaLN_modulation.1.weight", 3 * FLOW, FLOW);
            b.f32(p + "adaLN_modulation.1.bias", 3 * FLOW);
        }
        b.f32("flow_lm.flow_net.final_layer.adaLN_modulation.1.weight", 2 * FLOW, FLOW);
        b.f32("flow_lm.flow_net.final_layer.adaLN_modulation.1.bias", 2 * FLOW);
        b.f32("flow_lm.flow_net.final_layer.linear.weight", LATENT, FLOW);
        b.f32("flow_lm.flow_net.final_layer.linear.bias", LATENT);

        // the mimi quantizer, upsampler and SEANet decoder
        b.f32("mimi.quantizer.output_proj.weight", WIDTH, LATENT, 1);
        b.f32("mimi.upsample.convtr.convtr.weight", WIDTH, 1, 32);
        b.f32("mimi.decoder.model.0.conv.weight", WIDTH, WIDTH, 3);
        b.f32("mimi.decoder.model.0.conv.bias", WIDTH);
        for (int s = 0; s < 3; s++) {
            int[] strides = {6, 5, 4};
            String scale = "mimi.decoder.model." + (2 + 3 * s) + ".";
            b.f32(scale + "conv.weight", WIDTH, 1, 2 * strides[s]);
            b.f32(scale + "conv.bias", WIDTH);
            String res = "mimi.decoder.model." + (3 + 3 * s) + ".";
            b.f32(res + "block.1.conv.weight", WIDTH, WIDTH, 3);
            b.f32(res + "block.1.conv.bias", WIDTH);
            b.f32(res + "block.3.conv.weight", WIDTH, WIDTH, 1);
            b.f32(res + "block.3.conv.bias", WIDTH);
        }
        b.f32("mimi.decoder.model.11.conv.weight", 1, WIDTH, 3);
        b.f32("mimi.decoder.model.11.conv.bias", 1);

        // the decoder transformer
        String t = "mimi.decoder_transformer.transformer.layers.0.";
        b.f32(t + "norm1.weight", WIDTH);
        b.f32(t + "norm1.bias", WIDTH);
        b.f32(t + "norm2.weight", WIDTH);
        b.f32(t + "norm2.bias", WIDTH);
        b.f32(t + "self_attn.in_proj.weight", 3 * WIDTH, WIDTH);
        b.f32(t + "self_attn.out_proj.weight", WIDTH, WIDTH);
        b.f32(t + "linear1.weight", FF, WIDTH);
        b.f32(t + "linear2.weight", WIDTH, FF);
        b.f32(t + "layer_scale_1.scale", WIDTH);
        b.f32(t + "layer_scale_2.scale", WIDTH);
        return b.write(tmp.newFolder("pockettts-checkpoint").toPath());
    }

    /**
     * The voice-prompt encoder of a pocket-tts checkpoint (pockettts.py:201-225): the mimi encoder's SEANet stack,
     * its transformer, the strided downsample and the projection to the backbone width.
     */
    private void speakerEncoder(SyntheticHf.Builder b) {
        int[] strides = {4, 5, 6};
        b.f32("mimi.encoder.model.0.conv.weight", WIDTH, 1, 3);
        b.f32("mimi.encoder.model.0.conv.bias", WIDTH);
        for (int s = 0; s < 3; s++) {
            int res = 1 + 3 * s;
            int scale = 3 + 3 * s;
            b.f32("mimi.encoder.model." + scale + ".conv.weight", WIDTH, WIDTH, 2 * strides[s]);
            b.f32("mimi.encoder.model." + scale + ".conv.bias", WIDTH);
            b.f32("mimi.encoder.model." + res + ".block.1.conv.weight", WIDTH, WIDTH, 3);
            b.f32("mimi.encoder.model." + res + ".block.1.conv.bias", WIDTH);
            b.f32("mimi.encoder.model." + res + ".block.3.conv.weight", WIDTH, WIDTH, 1);
            b.f32("mimi.encoder.model." + res + ".block.3.conv.bias", WIDTH);
        }
        b.f32("mimi.encoder.model.11.conv.weight", WIDTH, WIDTH, 3);
        b.f32("mimi.encoder.model.11.conv.bias", WIDTH);
        String t = "mimi.encoder_transformer.transformer.layers.0.";
        b.f32(t + "norm1.weight", WIDTH);
        b.f32(t + "norm1.bias", WIDTH);
        b.f32(t + "norm2.weight", WIDTH);
        b.f32(t + "norm2.bias", WIDTH);
        b.f32(t + "self_attn.in_proj.weight", 3 * WIDTH, WIDTH);
        b.f32(t + "self_attn.out_proj.weight", WIDTH, WIDTH);
        b.f32(t + "linear1.weight", FF, WIDTH);
        b.f32(t + "linear2.weight", WIDTH, FF);
        b.f32(t + "layer_scale_1.scale", WIDTH);
        b.f32(t + "layer_scale_2.scale", WIDTH);
        b.f32("mimi.downsample.conv.conv.weight", LATENT, WIDTH, 2 * PocketTtsSpeakerEncoder.DOWNSAMPLE);
        b.f32("flow_lm.speaker_proj_weight", EMB, LATENT);
    }

    private Path export(Path dir) throws IOException {
        Path out = tmp.getRoot().toPath().resolve("pockettts-mmproj.gguf");
        ModelSourceOptions options = options();
        try (HfModelSource s = HfModelSource.openProjector(dir, options)) {
            GgufWriter.write(s, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        return out;
    }

    @Test
    public void convertsTheGeneratorMetadataAndTensorNames() throws IOException {
        Path dir = checkpoint();
        Path out = export(dir);
        ModelSourceOptions options = options();
        try (ModelSource source = ModelSources.open(out.toString(), options)) {
            GgufKv kv = GgufKv.of(source);
            assertEquals("clip", kv.archName());
            assertTrue(kv.bool(GgufKeys.CLIP_HAS_GEN_AUDIO_ENCODER, false));
            assertEquals(PocketTtsFlow.PROJECTOR_TYPE, kv.str(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, null));
            assertEquals(WIDTH, kv.u32(GgufKeys.CLIP_GEN_AUDIO_EMBEDDING_LENGTH, -1));
            assertEquals(1, kv.u32(GgufKeys.CLIP_GEN_AUDIO_BLOCK_COUNT, -1));
            assertEquals(1, kv.u32(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT, -1));
            assertEquals(FF, kv.u32(GgufKeys.CLIP_GEN_AUDIO_FEED_FORWARD_LENGTH, -1));
            assertEquals(EMB, kv.u32(GgufKeys.CLIP_GEN_AUDIO_PROJECTION_DIM, -1));
            assertEquals(1e-5f, kv.f32Required(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_LAYER_NORM_EPSILON), 0f);
            assertEquals(dir.getFileName().toString(),
                    kv.str(GgufKeys.CLIP_GEN_AUDIO_MODEL_VARIANT, null));
            assertNotNull(source.getTensors().stream()
                    .filter(t -> t.name().equals("a.gen.input_linear.weight"))
                    .findFirst().orElse(null));
            assertNotNull(source.getTensors().stream()
                    .filter(t -> t.name().equals("a.gen.wav.seanet.blk.1.scale_conv.weight"))
                    .findFirst().orElse(null));
            assertNotNull(source.getTensors().stream()
                    .filter(t -> t.name().equals("a.gen.wav.tfm.blk.0.attn_q.weight"))
                    .findFirst().orElse(null));
            // conv kernels keep F16: ggml_conv_1d has no BF16 path
            assertEquals(com.eonmux.neuralfree.gguf.GGMLType.F16, source.getTensors().stream()
                    .filter(t -> t.name().equals("a.gen.wav.seanet.blk.1.scale_conv.weight"))
                    .findFirst().orElseThrow().type());
            assertEquals(com.eonmux.neuralfree.gguf.GGMLType.F16, source.getTensors().stream()
                    .filter(t -> t.name().equals("a.gen.wav.upsample.weight"))
                    .findFirst().orElseThrow().type());
        }
    }

    @Test
    public void convertsACheckpointWithoutAConfigJson() throws IOException {
        Path dir = checkpoint();
        // the gated checkpoint ships no config.json; the detector derives the geometry from the tensor shapes
        java.nio.file.Files.delete(dir.resolve("config.json"));
        Path out = export(dir);
        ModelSourceOptions options = options();
        try (PocketTtsClip clip = PocketTtsClip.open(out.toString(), options)) {
            assertEquals(EMB, clip.flow().condWidth());
            assertEquals(LATENT, clip.flow().latent());
            assertEquals(WIDTH, clip.decoder().width());
            assertEquals(1, clip.decoder().nLayer());
            assertEquals(LATENT, clip.decoder().latent());
        }
    }

    @Test
    public void aVoiceCloneCheckpointCarriesTheSpeakerEncoder() throws IOException {
        Path dir = checkpoint(true);
        Path out = tmp.getRoot().toPath().resolve("pockettts-voice-clone.gguf");
        try (HfModelSource s = HfModelSource.openProjector(dir, options())) {
            GgufKv kv = GgufKv.of(s);
            assertTrue(kv.bool(kv.key(GgufKeys.CLIP_HAS_AUDIO_ENCODER), false));
            assertEquals(PocketTtsSpeakerEncoder.PROJECTOR_TYPE,
                    kv.str(kv.key(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE), null));
            assertEquals(EMB, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_PROJECTION_DIM), -1));
            assertEquals(1, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS), -1));
            assertEquals(1, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_BLOCK_COUNT), -1));
            assertEquals(WIDTH, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_EMBEDDING_LENGTH), -1));
            assertEquals(FF, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_FEED_FORWARD_LENGTH), -1));
            assertEquals(1, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_ATTENTION_HEAD_COUNT), -1));
            for (String name : new String[]{"a.seanet.conv_in.weight", "a.seanet.conv_out.weight",
                    "a.seanet.blk.0.scale_conv.weight", "a.seanet.blk.2.res_conv1.weight", "a.downsample.conv.weight",
                    "a.speaker_proj.weight", "a.blk.0.ln1.weight", "a.blk.0.attn_q.weight", "a.blk.0.attn_k.weight",
                    "a.blk.0.attn_v.weight", "a.blk.0.attn_out.weight", "a.blk.0.ffn_up.weight",
                    "a.blk.0.ffn_down.weight", "a.blk.0.ls1.weight", "a.blk.0.ls2.weight"}) {
                assertTrue(name + " is planned", s.plan().has(name));
            }
            // the checkpoint's [out, in, kernel] kernels become the GGUF's [kernel, in, out] layouts
            assertArrayEquals(new long[]{3, 1, WIDTH}, s.plan().ne("a.seanet.conv_in.weight"));
            assertArrayEquals(new long[]{2 * 4, WIDTH, WIDTH}, s.plan().ne("a.seanet.blk.0.scale_conv.weight"));
            assertArrayEquals(new long[]{2 * PocketTtsSpeakerEncoder.DOWNSAMPLE, WIDTH, LATENT},
                    s.plan().ne("a.downsample.conv.weight"));
            assertArrayEquals(new long[]{LATENT, EMB}, s.plan().ne("a.speaker_proj.weight"));
            GgufWriter.write(s, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (PocketTtsClip clip = PocketTtsClip.open(out.toString(), options())) {
            assertNotNull(clip.speakerEncoder());
            assertEquals(EMB, clip.speakerEncoder().width());
            float[] samples = new float[24000];
            for (int i = 0; i < samples.length; i++) {
                samples[i] = (float) (0.2 * Math.sin(2.0 * Math.PI * 220.0 * i / 24000.0));
            }
            float[][] rows = clip.speakerEncoder().encode(new AudioClip(samples, 24000));
            assertEquals(13, rows.length);
            assertEquals(EMB, rows[0].length);
            for (float value : rows[0]) {
                assertTrue("the embedding is finite", Float.isFinite(value));
            }
        }
    }

    @Test
    public void theExportedFileOpensAsAPocketTtsGeneratorAndMmproj() throws IOException {
        Path out = export(checkpoint());
        ModelSourceOptions options = options();
        try (PocketTtsClip clip = PocketTtsClip.open(out.toString(), options)) {
            assertEquals(EMB, clip.flow().condWidth());
            assertEquals(LATENT, clip.flow().latent());
            assertEquals(FLOW, clip.flow().hidden());
            assertEquals(2, clip.flow().blocks());
            assertEquals(WIDTH, clip.decoder().width());
            assertEquals(LATENT, clip.decoder().latent());
            assertEquals(1920, clip.decoder().hop());
            // the selected input_linear column survives the conversion: row o starts with o * EMB
            float[] projected = clip.flow().project(new float[]{1f, 0f, 0f});
            assertEquals(0f, projected[0], 1e-6f);
            assertEquals(EMB, projected[1], 1e-6f);
            assertEquals(2f * EMB, projected[2], 1e-6f);
        }
        try (MmprojModel mmproj = MmprojModel.open(out.toString(), options)) {
            assertTrue(mmproj.supportsGenAudio());
        }
    }
}
