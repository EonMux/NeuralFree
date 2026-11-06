package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.mm.MmprojModel;
import com.eonmux.neuralfree.gguf.mm.ProjectorModality;
import com.eonmux.neuralfree.gguf.speech.Lfm3aGenClip;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The LFM3 audio projector conversions (lfm3.py {@code LFM3AudioMmprojModel} and {@code LFM3OmniMmprojModel}): a
 * synthetic Hugging Face checkpoint with the audio encoder, its projector, the MaskGIT depthformer and a liquid-codec
 * side file exports to an mmproj GGUF that both serving sides open — {@link MmprojModel} for the audio encoder and
 * {@link Lfm3aGenClip} for the speech generator, whose loader validates every generator tensor shape. The Omni
 * checkpoint carries its vision tower beside the same audio side under the HF-native names; an encoder-only
 * checkpoint omits the generator half, and a codec or feature extractor the clip graphs do not support is refused by
 * name.
 */
public class HfLfm3AudioSourceTest {

    /** The text width the projector projects into. */
    private static final int TEXT = 8;
    /** The audio encoder width, FFN, heads and layers (four cover the projector's {@code mlp.model.0/1/3} indices). */
    private static final int AUDIO = 8;
    private static final int AUDIO_FF = 16;
    private static final int AUDIO_HEADS = 2;
    private static final int AUDIO_KV_HEADS = 1;
    private static final int AUDIO_LAYERS = 4;
    /** The MaskGIT depthformer width, FFN and heads. */
    private static final int DEPTH = 8;
    private static final int DEPTH_FF = 16;
    private static final int DEPTH_HEADS = 2;
    private static final int DEPTH_KV_HEADS = 1;
    private static final int CODEBOOKS = 12;
    /** The FSQ levels' product (8 * 7 * 6 * 6) plus the end code. */
    private static final int VOCAB = 2017;
    /** The liquid-codec decoder width, FFN and the quantizer width. */
    private static final int DEC = 8;
    private static final int DEC_FF = 16;
    private static final int QUANT = 8;
    /** The polar output width of the codec: an even 2 * (1280 / 2 + 1). */
    private static final int POLAR = 2 * (1280 / 2 + 1);

    private static final String FEATURE_EXTRACTOR = "{'feature_extractor_type':'Lfm3AudioFeatureExtractor',"
            + "'sampling_rate':16000,'feature_size':128,'n_fft':400,'win_length':400,'hop_length':160,"
            + "'log_floor':-12.0,'normalization_offset':4.0,'normalization_divisor':4.0,'encoder_stride':8}";
    private static final String PREPROCESSOR = "{'feature_extractor':" + FEATURE_EXTRACTOR + ","
            + "'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5],"
            + "'size':{'shortest_edge':4096,'longest_edge':25165824},'resample':2}";

    /** The audio-only checkpoint's member names. */
    private static final Layout AUDIO_ONLY = new Layout("encoder.", "encoder.lfm.lfm.", "mlp.model.",
            new String[]{"0", "1", "3"}, "", "depthformer.model.", "codebook_embedding.");
    /** The Omni checkpoint's HF-native audio member names (lfm3.py:583-591). */
    private static final Layout OMNI = new Layout("model.audio_encoder.", "model.audio_encoder.lfm.",
            "model.audio_projector.", new String[]{"layer_norm", "linear_1", "linear_2"},
            "", "model.depth_decoder.model.", "model.codebook_embedding.");

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path cache;

    private ModelSourceOptions options() throws IOException {
        if (cache == null) {
            cache = tmp.newFolder("cache").toPath();
        }
        return ModelSourceOptions.builder().cacheDir(cache).build();
    }

    // ------------------------------------------------------------------ checkpoint

    /**
     * A synthetic LFM3-Audio checkpoint: the audio encoder and projector, the generation shape, the depthformer and,
     * when {@code codec}, the liquid-codec side files.
     */
    private Path audioCheckpoint(String featureExtractor, boolean codec, int audioVocab) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Lfm3AudioForConditionalGeneration'],'model_type':'lfm3_audio',"
                + "'lfm':{'hidden_size':" + TEXT + "},"
                + "'encoder':{'dim_in':4,'sliding_window':4,'lfm':" + audioConfig() + "},"
                + "'codebooks':" + CODEBOOKS + ",'audio_vocab_size':" + audioVocab + ",'maskgit_steps':3,"
                + "'interleaved_n_text':6,'interleaved_n_audio':12,"
                + "'depthformer':" + depthConfig() + "}"));
        b.file("preprocessor_config.json", SyntheticHf.json("{'feature_extractor':" + featureExtractor + "}"));
        audioEncoder(b, AUDIO_ONLY);
        depthformer(b, audioVocab, AUDIO_ONLY);
        Path dir = b.write(tmp.newFolder("lfm3a").toPath());
        if (codec) {
            codec(dir);
        }
        return dir;
    }

    /** A synthetic LFM3-Omni checkpoint: the vision tower beside the audio encoder, generator and codec. */
    private Path omniCheckpoint() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Lfm3OmniForConditionalGeneration'],'model_type':'lfm3_omni',"
                + "'text_config':{'hidden_size':" + TEXT + "},"
                + "'projector_hidden_act':'gelu','projector_bias':true,'projector_layer_norm_eps':1e-06,"
                + "'downsample_factor':2,"
                + "'vision_config':{'hidden_size':8,'intermediate_size':16,'num_attention_heads':2,"
                + "'num_hidden_layers':2,'num_key_value_heads':1,'patch_size':2,'in_channels':3,"
                + "'conv_kernel_size':3,'temporal_patch_size':2,'norm_eps':1e-06,"
                + "'rope_parameters':{'rope_theta':1000000.0}},"
                + "'audio_encoder_config':{'num_mel_bins':4,'sliding_window':4,'text_config':"
                + audioConfig() + "},"
                + "'num_codebooks':" + CODEBOOKS + ",'codebook_size':" + VOCAB + ",'maskgit_schedule':[4,4,4],"
                + "'interleaved_n_text':6,'interleaved_n_audio':12,"
                + "'depth_config':" + depthConfig() + "}"));
        b.file("preprocessor_config.json", SyntheticHf.json(PREPROCESSOR));
        vision(b);
        audioEncoder(b, OMNI);
        depthformer(b, VOCAB, OMNI);
        Path dir = b.write(tmp.newFolder("lfm3-omni").toPath());
        codec(dir);
        return dir;
    }

    /** The audio encoder's LFM2 stack (lfm3.py:378-383). */
    private static String audioConfig() {
        return "{'hidden_size':" + AUDIO + ",'intermediate_size':" + AUDIO_FF + ",'num_attention_heads':"
                + AUDIO_HEADS + ",'num_key_value_heads':" + AUDIO_KV_HEADS + ",'num_hidden_layers':"
                + AUDIO_LAYERS + ",'layer_types':['conv','full_attention','conv','full_attention'],"
                + "'norm_eps':1e-05,'block_ff_dim':" + AUDIO_FF + ",'block_auto_adjust_ff_dim':false,"
                + "'block_multiple_of':8,'block_ffn_dim_multiplier':null,"
                + "'rope_parameters':{'rope_theta':1000000.0}}";
    }

    /** The MaskGIT depthformer configuration (lfm3.py:401-407). */
    private static String depthConfig() {
        return "{'hidden_size':" + DEPTH + ",'intermediate_size':" + DEPTH_FF + ",'num_hidden_layers':1,"
                + "'num_attention_heads':" + DEPTH_HEADS + ",'num_key_value_heads':" + DEPTH_KV_HEADS + ","
                + "'norm_eps':1e-05,'is_causal':false,'use_sandwich_norm':false,"
                + "'layer_types':['full_attention'],"
                + "'rope_parameters':{'full_attention':{'rope_theta':1000000.0}}}";
    }

    /** The LFM3-VL vision tower and projector (lfm3.py:225-266). */
    private static void vision(SyntheticHf.Builder b) {
        b.f32("model.vision_tower.patchifier.weight", 8, 3, 2, 2, 2);
        b.f32("model.vision_tower.embedding_norm.weight", 8);
        for (int il = 0; il < 2; il++) {
            String layer = "model.vision_tower.layers." + il + ".";
            b.f32(layer + "operator_norm.weight", 8);
            b.f32(layer + "ffn_norm.weight", 8);
            b.f32(layer + "feed_forward.w1.weight", 16, 8);
            b.f32(layer + "feed_forward.w3.weight", 16, 8);
            b.f32(layer + "feed_forward.w2.weight", 8, 16);
            if (il == 0) {
                b.f32(layer + "conv.in_proj.weight", 8, 8);
                b.f32(layer + "conv.spatial_conv.weight", 8, 1, 3, 3);
                b.f32(layer + "conv.out_proj.weight", 8, 8);
            } else {
                b.f32(layer + "self_attn.q_proj.weight", 8, 8);
                b.f32(layer + "self_attn.k_proj.weight", 4, 8);
                b.f32(layer + "self_attn.v_proj.weight", 4, 8);
                b.f32(layer + "self_attn.out_proj.weight", 8, 8);
            }
        }
        b.f32("model.multi_modal_projector.layer_norm.weight", 8);
        b.f32("model.multi_modal_projector.layer_norm.bias", 8);
        b.f32("model.multi_modal_projector.linear_1.weight", 8, 8);
        b.f32("model.multi_modal_projector.linear_1.bias", 8);
        b.f32("model.multi_modal_projector.linear_2.weight", 8, 8);
        b.f32("model.multi_modal_projector.linear_2.bias", 8);
    }

    /** The audio encoder's downsample stack, conformer-style blocks and projector (lfm3.py:532-545). */
    private static void audioEncoder(SyntheticHf.Builder b, Layout layout) {
        String down = layout.audio + "downsample.";
        b.f32(down + "conv_in.weight", AUDIO, 4, 3);
        b.f32(down + "conv_in.bias", AUDIO);
        b.f32(down + "conv_mid.weight", AUDIO, AUDIO, 3);
        b.f32(down + "conv_mid.bias", AUDIO);
        b.f32(down + "lin_mid.weight", AUDIO, AUDIO);
        b.f32(down + "lin_mid.bias", AUDIO);
        b.f32(down + "conv_end.weight", AUDIO, AUDIO, 3);
        b.f32(down + "conv_end.bias", AUDIO);
        b.f32(down + "lin_end.weight", AUDIO, AUDIO);
        b.f32(down + "lin_end.bias", AUDIO);
        b.f32(down + "lin_out.weight", AUDIO, AUDIO, 1);
        b.f32(down + "lin_out.bias", AUDIO);
        b.f32(layout.lfm + "embedding_norm.weight", AUDIO);
        int headDim = AUDIO / AUDIO_HEADS;
        for (int il = 0; il < AUDIO_LAYERS; il++) {
            String layer = layout.lfm + "layers." + il + ".";
            b.f32(layer + "operator_norm.weight", AUDIO);
            if (il % 2 == 0) {
                b.f32(layer + "conv.in_proj.weight", 3 * AUDIO, AUDIO);
                b.f32(layer + "conv.conv.weight", AUDIO, 1, 4);
                b.f32(layer + "conv.out_proj.weight", AUDIO, AUDIO);
            } else {
                b.f32(layer + "self_attn.q_proj.weight", AUDIO, AUDIO);
                b.f32(layer + "self_attn.q_layernorm.weight", headDim);
                b.f32(layer + "self_attn.k_proj.weight", AUDIO_KV_HEADS * headDim, AUDIO);
                b.f32(layer + "self_attn.k_layernorm.weight", headDim);
                b.f32(layer + "self_attn.v_proj.weight", AUDIO_KV_HEADS * headDim, AUDIO);
                b.f32(layer + "self_attn.out_proj.weight", AUDIO, AUDIO);
            }
            b.f32(layer + "ffn_norm.weight", AUDIO);
            b.f32(layer + "feed_forward.w1.weight", AUDIO_FF, AUDIO);
            b.f32(layer + "feed_forward.w3.weight", AUDIO_FF, AUDIO);
            b.f32(layer + "feed_forward.w2.weight", AUDIO, AUDIO_FF);
        }
        b.f32(layout.mlp + layout.mlpNames[0] + ".weight", AUDIO);
        b.f32(layout.mlp + layout.mlpNames[0] + ".bias", AUDIO);
        b.f32(layout.mlp + layout.mlpNames[1] + ".weight", AUDIO, AUDIO);
        b.f32(layout.mlp + layout.mlpNames[1] + ".bias", AUDIO);
        b.f32(layout.mlp + layout.mlpNames[2] + ".weight", AUDIO, AUDIO);
        b.f32(layout.mlp + layout.mlpNames[2] + ".bias", AUDIO);
    }

    /** The MaskGIT depthformer and its tied per-codebook heads (lfm3.py:558-573). */
    private static void depthformer(SyntheticHf.Builder b, int vocab, Layout layout) {
        int headDim = DEPTH / DEPTH_HEADS;
        String layer = layout.depthModel + "layers.0.";
        b.f32(layer + "operator_norm.weight", DEPTH);
        b.f32(layer + "self_attn.q_proj.weight", DEPTH, DEPTH);
        b.f32(layer + "self_attn.q_norm.weight", headDim);
        b.f32(layer + "self_attn.k_proj.weight", DEPTH_KV_HEADS * headDim, DEPTH);
        b.f32(layer + "self_attn.k_norm.weight", headDim);
        b.f32(layer + "self_attn.v_proj.weight", DEPTH_KV_HEADS * headDim, DEPTH);
        b.f32(layer + "self_attn.o_proj.weight", DEPTH, DEPTH);
        b.f32(layer + "self_attn.gate_proj.weight", DEPTH_HEADS, DEPTH);
        b.f32(layer + "ffn_norm.weight", DEPTH);
        b.f32(layer + "feed_forward.w1.weight", DEPTH_FF, DEPTH);
        b.f32(layer + "feed_forward.w3.weight", DEPTH_FF, DEPTH);
        b.f32(layer + "feed_forward.w2.weight", DEPTH, DEPTH_FF);
        b.f32(layout.depthModel + "embedding_norm.weight", DEPTH);
        b.f32(layout.depth + "depth_linear.weight", DEPTH * CODEBOOKS, TEXT);
        b.f32(layout.depth + "depth_linear.bias", DEPTH * CODEBOOKS);
        b.f32(layout.depth + "mask_embedding", DEPTH);
        b.f32(layout.codebook + "weight", vocab * CODEBOOKS, TEXT);
        for (int i = 0; i < CODEBOOKS; i++) {
            String page = layout.depth + "depth_embeddings." + i + ".";
            b.f32(page + "embedding.weight", vocab, DEPTH);
            b.f32(page + "embedding_norm.weight", DEPTH);
            b.f32(page + "to_logits.weight", vocab, DEPTH);
        }
    }

    /** The liquid-codec side files: {@code config.json} and the decoder's {@code model.safetensors}. */
    private static void codec(Path dir) throws IOException {
        Path codecDir = dir.resolve("audio_tokenizer");
        Files.createDirectories(codecDir);
        Files.write(codecDir.resolve("config.json"), SyntheticHf.json("{'model_type':'liquid_codec',"
                + "'fsq_levels':[8,7,6,6],'n_q':" + CODEBOOKS + ",'quantizer_dim':256,'encoder_dim':512,"
                + "'stft_n_fft':1280,'stft_hop_length':320,'stft_win_length':1280,'stft_frame_stack':6,"
                + "'decoder_window':30,'sampling_rate':24000}").getBytes(StandardCharsets.UTF_8));
        SafetensorsWriter.Builder w = SafetensorsWriter.builder();
        tensor(w, "quantizer.fsq.project_out.weight", QUANT, 4 * CODEBOOKS);
        tensor(w, "quantizer.fsq.project_out.bias", QUANT);
        tensor(w, "from_quantizer.weight", DEC, QUANT);
        tensor(w, "from_quantizer.bias", DEC);
        tensor(w, "decoder.lin.weight", POLAR, DEC);
        tensor(w, "decoder.lin.bias", POLAR);
        tensor(w, "decoder.lfm.embedding_norm.weight", DEC);
        String layer = "decoder.lfm.layers.0.";
        tensor(w, layer + "operator_norm.weight", DEC);
        tensor(w, layer + "post_operator_norm.weight", DEC);
        tensor(w, layer + "conv.in_proj.weight", 3 * DEC, DEC);
        tensor(w, layer + "conv.conv.weight", DEC, 1, 3);
        tensor(w, layer + "conv.out_proj.weight", DEC, DEC);
        tensor(w, layer + "ffn_norm.weight", DEC);
        tensor(w, layer + "post_ffn_norm.weight", DEC);
        tensor(w, layer + "feed_forward.w1.weight", DEC_FF, DEC);
        tensor(w, layer + "feed_forward.w3.weight", DEC_FF, DEC);
        tensor(w, layer + "feed_forward.w2.weight", DEC, DEC_FF);
        w.metadata("format", "pt");
        w.build().write(codecDir.resolve("model.safetensors"));
    }

    private static void tensor(SafetensorsWriter.Builder w, String name, long... shape) {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        w.add(name, StDType.F32, shape, SyntheticHf.f32(SyntheticHf.values(name.hashCode(), (int) n)));
    }

    /** Exports the projector view of {@code dir} and returns the mmproj GGUF's path. */
    private Path export(Path dir, String name) throws IOException {
        Path out = tmp.getRoot().toPath().resolve(name + "-mmproj.gguf");
        ModelSourceOptions options = options();
        try (HfModelSource s = HfModelSource.openProjector(dir, options)) {
            GgufWriter.write(s, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        return out;
    }

    // ------------------------------------------------------------------ tests

    @Test
    public void theAudioExportOpensAsAnAudioEncoderAndSpeechGenerator() throws IOException {
        Path out = export(audioCheckpoint(FEATURE_EXTRACTOR, true, VOCAB), "lfm3a");
        ModelSourceOptions options = options();
        try (MmprojModel mm = MmprojModel.open(out.toString(), options)) {
            assertFalse(mm.caps().inpVision());
            assertTrue(mm.caps().inpAudio());
            assertTrue(mm.caps().genAudio());
            assertTrue(mm.supportsAudio());
            assertTrue(String.valueOf(mm.unrunnableReason(ProjectorModality.GEN_AUDIO)), mm.supportsGenAudio());
            var hp = mm.audioHparams();
            assertEquals("lfm3a", hp.projectorType().typeName());
            assertEquals(TEXT, hp.projectionDim());
            assertEquals(AUDIO, hp.nEmbd());
            assertEquals(AUDIO_LAYERS, hp.nLayer());
            assertEquals(AUDIO_HEADS, hp.nHead());
            assertEquals(AUDIO_KV_HEADS, hp.nHeadKv());
            assertEquals(4, hp.audio().nMelBins());
            assertTrue(mm.tensorNames().contains("a.conv1d.0.weight"));
            assertTrue(mm.tensorNames().contains("a.pre_encode.out.weight"));
            assertTrue(mm.tensorNames().contains("a.post_ln.weight"));
            assertTrue(mm.tensorNames().contains("mm.a.mlp.1.weight"));
            assertTrue(mm.tensorNames().contains("a.blk.0.conv_pw1.weight"));
            assertTrue(mm.tensorNames().contains("a.blk.1.attn_q.weight"));
            assertTrue(mm.tensorNames().contains("a.gen.code.proj_in.weight"));
            assertTrue(mm.tensorNames().contains("a.gen.code.mask_embd"));
            assertTrue(mm.tensorNames().contains("a.gen.code.out_embd.weight"));
            assertTrue(mm.tensorNames().contains("a.gen.code.head_norm.weight"));
            assertTrue(mm.tensorNames().contains("a.gen.wav.quant.first.codebook.weight"));
            assertTrue(mm.tensorNames().contains("a.gen.wav.tfm.blk.0.conv_pw1.weight"));
        }
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(out.toString(), options)) {
            assertEquals(VOCAB, clip.vocab());
            assertEquals(DEPTH, clip.depthformer().nEmbd());
            assertEquals(TEXT, clip.depthformer().talkerWidth());
            assertEquals(1f, clip.temp(), 1e-6f);
            assertEquals(128, clip.topK());
            assertEquals(1.1f, clip.penaltyRepeat(), 1e-6f);
            assertEquals(4096, clip.penaltyLastN());
            assertEquals(VOCAB - 1, clip.codec().vocab());
            assertEquals(CODEBOOKS, clip.codec().nCodebooks());
        }
    }

    @Test
    public void theOmniExportCarriesBothTowersAndTheGenerator() throws IOException {
        Path out = export(omniCheckpoint(), "lfm3-omni");
        try (MmprojModel mm = MmprojModel.open(out.toString(), options())) {
            assertTrue(mm.caps().inpVision());
            assertTrue(mm.caps().inpAudio());
            assertTrue(mm.caps().genAudio());
            assertTrue(mm.supportsVision());
            assertTrue(mm.supportsAudio());
            assertTrue(String.valueOf(mm.unrunnableReason(ProjectorModality.GEN_AUDIO)), mm.supportsGenAudio());
            assertEquals("lfm3vl", mm.visionHparams().projectorType().typeName());
            assertEquals("lfm3a", mm.audioHparams().projectorType().typeName());
            assertTrue(mm.tensorNames().contains("v.patch_embd.weight.1"));
            assertTrue(mm.tensorNames().contains("v.blk.0.conv2d_spatial.weight"));
            assertTrue(mm.tensorNames().contains("v.blk.1.attn_q.weight"));
            assertTrue(mm.tensorNames().contains("mm.0.weight"));
            assertTrue(mm.tensorNames().contains("a.post_ln.weight"));
            assertTrue(mm.tensorNames().contains("mm.a.mlp.3.weight"));
            assertTrue(mm.tensorNames().contains("a.gen.code.out_embd.weight"));
        }
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(out.toString(), options())) {
            assertEquals(VOCAB, clip.vocab());
            assertEquals(0.4f, clip.temp(), 1e-6f);
            assertEquals(1.2f, clip.penaltyRepeat(), 1e-6f);
            assertEquals(64, clip.penaltyLastN());
        }
    }

    @Test
    public void anEncoderOnlyCheckpointOmitsTheGeneratorHalf() throws IOException {
        Path out = export(audioCheckpoint(FEATURE_EXTRACTOR, false, VOCAB), "lfm3a-encoder");
        ModelSourceOptions options = options();
        try (MmprojModel mm = MmprojModel.open(out.toString(), options)) {
            assertTrue(mm.caps().inpAudio());
            assertFalse(mm.caps().genAudio());
            assertFalse(mm.tensorNames().contains("a.gen.code.proj_in.weight"));
        }
        try {
            Lfm3aGenClip.open(out.toString(), options).close();
            fail("an encoder-only file carries no generator");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("lfm3a"));
        }
    }

    @Test
    public void aMismatchedCodecIsRefusedByTheAudioVocabulary() throws IOException {
        try {
            export(audioCheckpoint(FEATURE_EXTRACTOR, true, VOCAB + 1), "lfm3a-bad-codec");
            fail("the FSQ table does not match the audio vocabulary");
        } catch (GgufFormatException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("audio_vocab_size"));
        }
    }

    @Test
    public void anUnsupportedFeatureExtractorIsRefusedByName() throws IOException {
        String extractor = FEATURE_EXTRACTOR.replace("'hop_length':160", "'hop_length':320");
        try {
            export(audioCheckpoint(extractor, true, VOCAB), "lfm3a-bad-front-end");
            fail("the clip front end hardcodes the hop length");
        } catch (GgufFormatException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("feature_extractor.hop_length"));
        }
    }

    /** The member names of one LFM3 audio checkpoint layout. */
    private static final class Layout {
        final String audio;
        final String lfm;
        final String mlp;
        final String[] mlpNames;
        final String depth;
        final String depthModel;
        final String codebook;

        Layout(String audio, String lfm, String mlp, String[] mlpNames, String depth, String depthModel,
               String codebook) {
            this.audio = audio;
            this.lfm = lfm;
            this.mlp = mlp;
            this.mlpNames = Arrays.copyOf(mlpNames, mlpNames.length);
            this.depth = depth;
            this.depthModel = depthModel;
            this.codebook = codebook;
        }
    }
}
