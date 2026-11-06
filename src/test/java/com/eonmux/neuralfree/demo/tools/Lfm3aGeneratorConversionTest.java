package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.speech.Lfm3aClipFixture;
import com.eonmux.neuralfree.gguf.speech.Lfm3aCodecSession;
import com.eonmux.neuralfree.gguf.speech.Lfm3aDepthformer;
import com.eonmux.neuralfree.gguf.speech.Lfm3aGenClip;
import com.eonmux.neuralfree.gguf.speech.Lfm3aSpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.SpeechOptions;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The LFM3 audio generator the converter writes, opened by the runtime. A synthetic Hugging Face checkpoint holding
 * the tensors of {@link Lfm3aClipFixture#deterministicEos()} is exported with {@code nftool export --mmproj} (the
 * full-size codec geometry is shrunk through {@code --metadata} so the fixture's constants apply), and the written
 * clip file runs through {@link Lfm3aGenClip}, {@link Lfm3aDepthformer}, {@link Lfm3aCodecSession} and
 * {@link Lfm3aSpeechGenerator}. The generator reproduces the fixture's constant path: one frame of code 0 followed by
 * the end code, and the decoded waveform equals the fixture's own codec decode.
 */
public class Lfm3aGeneratorConversionTest {

    /** The liquid-codec geometry the converter's config requires; the tensors keep it and the runtime overrides it. */
    private static final int CODEC_LEVELS = 4;
    private static final int[] FSQ_LEVELS = {8, 7, 6, 6};
    /** The codec geometry of {@link Lfm3aClipFixture}, injected through {@code --metadata}. */
    private static final int WINDOW = 6;
    private static final int FRAME_STACK = 2;
    private static final int N_FFT = 16;
    private static final int HOP = 4;
    /** The audio vocabulary the converter derives from the FSQ levels. */
    private static final int AUDIO_VOCAB = fsqSize() + 1;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theConvertedAudioGeneratorReproducesTheFixturePipeline() throws Exception {
        run(false);
    }

    @Test
    public void theConvertedOmniAudioGeneratorReproducesTheFixturePipeline() throws Exception {
        run(true);
    }

    private void run(boolean omni) throws Exception {
        GgufTestWriter fixture = Lfm3aClipFixture.deterministicEos();
        Path dir = checkpoint(fixture, omni).write(tmp.newFolder(omni ? "omni" : "audio").toPath());
        writeCodec(dir, fixture);
        Path out = tmp.newFolder("out").toPath().resolve("mmproj-lfm3a-gen.gguf");
        export(dir, out);

        // the fixture's own pipeline: the expected codes are the constant path's first frame of zeros and the
        // waveform is the fixture codec's decode of that frame plus the flushed tail
        int codebooks = (int) fixture.tensorShape("a.gen.code.embd.weight")[2];
        AudioClip expected;
        try (Lfm3aGenClip original = Lfm3aGenClip.open(
                Lfm3aClipFixture.deterministicEos().writeTempFile().toString());
             Lfm3aCodecSession session = original.codec().newSession()) {
            expected = concat(session.decodeClip(new int[codebooks]), session.flushClip());
        }

        try (Lfm3aGenClip clip = Lfm3aGenClip.open(out.toString())) {
            Lfm3aDepthformer depthformer = clip.depthformer();
            assertEquals(codebooks, depthformer.nCodebooks());
            assertEquals(AUDIO_VOCAB, depthformer.vocab());
            // a masked pass runs the converted depthformer graph over the converted tensor names
            int[] masked = new int[codebooks];
            Arrays.fill(masked, -1);
            // the fixture backbone's hidden state: the second axis carries the value the constant path reads
            float[] hidden = new float[depthformer.talkerWidth()];
            hidden[1] = 5f;
            Lfm3aDepthformer.Frame frame = depthformer.pass(hidden, masked);
            assertNotNull(frame.logits());
            assertEquals((long) codebooks * AUDIO_VOCAB, frame.logits().length);

            AudioClip decoded;
            try (Lfm3aCodecSession session = clip.codec().newSession()) {
                decoded = concat(session.decodeClip(new int[codebooks]), session.flushClip());
            }
            assertAudioEquals(expected, decoded);

            try (GGUFModel backbone = new GGUFModel(Lfm3aBackboneFixturePath.write().toString(), config());
                 Lfm3aSpeechGenerator generator = Lfm3aSpeechGenerator.open(backbone, clip)) {
                assertEquals("lfm3a", generator.name());
                AudioClip audio = generator.synthesize("Hello world", SpeechOptions.defaults());
                assertEquals(24000, audio.sampleRate());
                assertAudioEquals(expected, audio);
            }
        }
    }

    // ------------------------------------------------------------------ HF checkpoint

    private static SyntheticHf.Builder checkpoint(GgufTestWriter w, boolean omni) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(omni ? omniConfig() : audioConfig()));
        b.file("preprocessor_config.json", SyntheticHf.json(omni ? omniPreprocessor() : audioPreprocessor()));
        b.file("generation_config.json", SyntheticHf.json("{'audio_temperature':0.0,"
                + "'audio_repetition_penalty':1.0}"));
        writeDepthformer(b, w, omni);
        return b;
    }

    private static void writeDepthformer(SyntheticHf.Builder b, GgufTestWriter w, boolean omni) {
        String depthformer = omni ? "model.depth_decoder." : "depthformer.";
        String root = omni ? "model.depth_decoder." : "";
        String layers = depthformer + "model.layers.";
        long[] embdNe = w.tensorShape("a.gen.code.embd.weight");
        int width = (int) embdNe[0];
        int fixtureVocab = (int) embdNe[1];
        int codebooks = (int) embdNe[2];
        float[] embd = floats(w, "a.gen.code.embd.weight");
        long[] outNe = w.tensorShape("a.gen.code.out_embd.weight");
        int talker = (int) outNe[0];
        float[] out = floats(w, "a.gen.code.out_embd.weight");

        for (String name : w.tensorNames()) {
            if (!name.startsWith("a.gen.code.") || name.equals("a.gen.code.embd.weight")
                    || name.equals("a.gen.code.head_norm.weight") || name.equals("a.gen.code.out_embd.weight")) {
                continue;
            }
            long[] ne = w.tensorShape(name);
            float[] values = floats(w, name);
            if (name.equals("a.gen.code.proj_in.weight")) {
                write(b, root + "depth_linear.weight", reverse(ne), values);
            } else if (name.equals("a.gen.code.proj_in.bias")) {
                write(b, root + "depth_linear.bias", flat(ne), values);
            } else if (name.equals("a.gen.code.mask_embd")) {
                write(b, root + "mask_embedding", ne, values);
            } else if (name.equals("a.gen.code.output_norm.weight")) {
                write(b, depthformer + "model.embedding_norm.weight", ne, values);
            } else if (name.startsWith("a.gen.code.blk.")) {
                writeLayer(b, name, ne, values, layers);
            } else {
                throw new IllegalArgumentException("unexpected depthformer tensor '" + name + "'");
            }
        }

        // the tied tables: the per-codebook pages of the fixture padded to the converter's FSQ vocabulary
        float[] norms = floats(w, "a.gen.code.head_norm.weight");
        for (int p = 0; p < codebooks; p++) {
            float[] page = new float[AUDIO_VOCAB * width];
            for (int v = 0; v < AUDIO_VOCAB; v++) {
                int src = v < fixtureVocab - 1 ? v : v == AUDIO_VOCAB - 1 ? fixtureVocab - 1 : -1;
                if (src >= 0) {
                    System.arraycopy(embd, src * width + p * width * fixtureVocab, page, v * width, width);
                }
            }
            String base = root + "depth_embeddings." + p + ".";
            write(b, base + "embedding.weight", new long[]{AUDIO_VOCAB, width}, page);
            write(b, base + "embedding_norm.weight", new long[]{1, width},
                    Arrays.copyOfRange(norms, p * width, (p + 1) * width));
            write(b, base + "to_logits.weight", new long[]{AUDIO_VOCAB, width}, page);
        }

        // the tied talker table: codebook p's fixture page lands at rows p * AUDIO_VOCAB, the padded vocabulary
        float[] codebook = new float[codebooks * AUDIO_VOCAB * talker];
        for (int p = 0; p < codebooks; p++) {
            for (int r = 0; r < fixtureVocab; r++) {
                System.arraycopy(out, (p * fixtureVocab + r) * talker, codebook,
                        (p * AUDIO_VOCAB + r) * talker, talker);
            }
        }
        String codebookName = omni ? "model.codebook_embedding.weight" : "codebook_embedding.weight";
        write(b, codebookName, new long[]{(long) codebooks * AUDIO_VOCAB, talker}, codebook);
    }

    private static void writeLayer(SyntheticHf.Builder b, String name, long[] ne, float[] values,
                                   String layers) {
        String rest = name.substring("a.gen.code.blk.".length());
        int dot = rest.indexOf('.');
        String h = layers + rest.substring(0, dot) + ".";
        String part = rest.substring(dot + 1);
        String module;
        if (part.startsWith("ln1.")) {
            module = "operator_norm.weight";
        } else if (part.startsWith("ln2.")) {
            module = "ffn_norm.weight";
        } else if (part.startsWith("attn_q_norm.")) {
            module = "self_attn.q_norm.weight";
        } else if (part.startsWith("attn_k_norm.")) {
            module = "self_attn.k_norm.weight";
        } else if (part.startsWith("attn_q.")) {
            module = "self_attn.q_proj.weight";
        } else if (part.startsWith("attn_k.")) {
            module = "self_attn.k_proj.weight";
        } else if (part.startsWith("attn_v.")) {
            module = "self_attn.v_proj.weight";
        } else if (part.startsWith("attn_out.")) {
            module = "self_attn.o_proj.weight";
        } else if (part.startsWith("attn_gate.")) {
            module = "self_attn.gate_proj.weight";
        } else if (part.startsWith("ffn_gate.")) {
            module = "feed_forward.w1.weight";
        } else if (part.startsWith("ffn_up.")) {
            module = "feed_forward.w3.weight";
        } else if (part.startsWith("ffn_down.")) {
            module = "feed_forward.w2.weight";
        } else {
            throw new IllegalArgumentException("unexpected depthformer layer tensor '" + name + "'");
        }
        write(b, h + module, ne.length == 1 ? ne : reverse(ne), values);
    }

    // ------------------------------------------------------------------ codec side file

    /** Writes the liquid-codec decoder of the fixture under the converter's side-file names. */
    private static void writeCodec(Path dir, GgufTestWriter w) throws IOException {
        Path codec = dir.resolve("audio_tokenizer");
        Files.createDirectories(codec);
        Files.write(codec.resolve("config.json"), codecConfig().getBytes(StandardCharsets.UTF_8));
        SafetensorsWriter.Builder s = SafetensorsWriter.builder();
        int fixtureLevels = (int) w.tensorShape("a.gen.wav.quant.first.codebook.weight")[0];
        int codebooks = (int) w.tensorShape("a.gen.code.embd.weight")[2];
        long[] outNe = w.tensorShape("a.gen.wav.quant.first.out_proj.weight");
        int dec = (int) outNe[1];
        float[] outWeight = floats(w, "a.gen.wav.quant.first.out_proj.weight");
        // the converter's FSQ row of code 0 is all -1 per level; scaling the fixture's two columns by half and
        // leaving the extra levels at zero keeps the first projection equal to the fixture's
        float[] projected = new float[dec * CODEC_LEVELS * codebooks];
        for (int d = 0; d < dec; d++) {
            for (int cb = 0; cb < codebooks; cb++) {
                for (int l = 0; l < CODEC_LEVELS; l++) {
                    if (l < fixtureLevels) {
                        projected[d * CODEC_LEVELS * codebooks + cb * CODEC_LEVELS + l] =
                                0.5f * outWeight[(cb * fixtureLevels + l) + d * fixtureLevels * codebooks];
                    }
                }
            }
        }
        add(s, "quantizer.fsq.project_out.weight", new long[]{dec, (long) CODEC_LEVELS * codebooks},
                projected);
        add(s, "quantizer.fsq.project_out.bias", new long[]{dec}, floats(w, "a.gen.wav.quant.first.out_proj.bias"));
        copy(s, w, "a.gen.wav.tfm.in_proj.weight", "from_quantizer.weight");
        copy(s, w, "a.gen.wav.tfm.in_proj.bias", "from_quantizer.bias");
        copy(s, w, "a.gen.wav.tfm.out_proj.weight", "decoder.lin.weight");
        copy(s, w, "a.gen.wav.tfm.out_proj.bias", "decoder.lin.bias");
        copy(s, w, "a.gen.wav.tfm.output_norm.weight", "decoder.lfm.embedding_norm.weight");
        for (String name : w.tensorNames()) {
            if (!name.startsWith("a.gen.wav.tfm.blk.")) {
                continue;
            }
            String rest = name.substring("a.gen.wav.tfm.blk.".length());
            int dot = rest.indexOf('.');
            String p = "decoder.lfm.layers." + rest.substring(0, dot) + ".";
            String part = rest.substring(dot + 1);
            String hf;
            if (part.equals("ln1.weight")) {
                hf = p + "operator_norm.weight";
            } else if (part.equals("ln2.weight")) {
                hf = p + "ffn_norm.weight";
            } else if (part.equals("attn_post_norm.weight")) {
                hf = p + "post_operator_norm.weight";
            } else if (part.equals("ffn_post_norm.weight")) {
                hf = p + "post_ffn_norm.weight";
            } else if (part.equals("conv_pw1.weight")) {
                hf = p + "conv.in_proj.weight";
            } else if (part.equals("conv_dw.weight")) {
                long[] ne = w.tensorShape(name);
                add(s, p + "conv.conv.weight", new long[]{ne[1], 1, ne[0]}, floats(w, name));
                continue;
            } else if (part.equals("conv_pw2.weight")) {
                hf = p + "conv.out_proj.weight";
            } else if (part.equals("attn_q.weight")) {
                hf = p + "self_attn.q_proj.weight";
            } else if (part.equals("attn_q_norm.weight")) {
                hf = p + "self_attn.q_norm.weight";
            } else if (part.equals("attn_k.weight")) {
                hf = p + "self_attn.k_proj.weight";
            } else if (part.equals("attn_k_norm.weight")) {
                hf = p + "self_attn.k_norm.weight";
            } else if (part.equals("attn_v.weight")) {
                hf = p + "self_attn.v_proj.weight";
            } else if (part.equals("attn_out.weight")) {
                hf = p + "self_attn.o_proj.weight";
            } else if (part.equals("attn_gate.weight")) {
                hf = p + "self_attn.gate_proj.weight";
            } else if (part.equals("ffn_gate.weight")) {
                hf = p + "feed_forward.w1.weight";
            } else if (part.equals("ffn_up.weight")) {
                hf = p + "feed_forward.w3.weight";
            } else if (part.equals("ffn_down.weight")) {
                hf = p + "feed_forward.w2.weight";
            } else {
                throw new IllegalArgumentException("unexpected codec tensor '" + name + "'");
            }
            long[] ne = w.tensorShape(name);
            add(s, hf, reverse(ne), floats(w, name));
        }
        s.metadata("format", "pt");
        s.build().write(codec.resolve("model.safetensors"));
    }

    private static void copy(SafetensorsWriter.Builder s, GgufTestWriter w, String gguf, String hf) {
        long[] ne = w.tensorShape(gguf);
        add(s, hf, ne.length == 1 ? ne : reverse(ne), floats(w, gguf));
    }

    private static void add(SafetensorsWriter.Builder s, String name, long[] shape, float[] values) {
        assertEquals(name + " holds " + values.length + " values", count(shape), values.length);
        s.add(name, StDType.F32, shape, SyntheticHf.f32(values));
    }

    // ------------------------------------------------------------------ export

    private static void export(Path dir, Path out) throws Exception {
        Path metadata = dir.resolveSibling(dir.getFileName() + "-meta.json");
        Files.write(metadata, SyntheticHf.json("{'clip.gen.wav.window':" + WINDOW + ","
                + "'clip.gen.wav.frame_stack':" + FRAME_STACK + ",'clip.gen.wav.n_fft':" + N_FFT + ","
                + "'clip.gen.wav.hop_length':" + HOP + "}").getBytes(StandardCharsets.UTF_8));
        List<Path> files = ExportCommand.export(new String[]{dir.toString(), out.toString(), "--mmproj",
                "--outtype", "f32", "--metadata", metadata.toString()});
        assertTrue(files.toString(), files.contains(out));
    }

    // ------------------------------------------------------------------ helpers

    private static void write(SyntheticHf.Builder b, String name, long[] shape, float[] values) {
        assertEquals(name + " holds " + values.length + " values", count(shape), values.length);
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(values));
    }

    private static long[] reverse(long[] ne) {
        long[] shape = new long[ne.length];
        for (int i = 0; i < ne.length; i++) {
            shape[i] = ne[ne.length - 1 - i];
        }
        return shape;
    }

    private static long[] flat(long[] ne) {
        return new long[]{count(ne)};
    }

    private static long count(long[] shape) {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        return n;
    }

    private static float[] floats(GgufTestWriter w, String name) {
        byte[] raw = w.tensorData(name);
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[raw.length / Float.BYTES];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getFloat(i * Float.BYTES);
        }
        return out;
    }

    private static int fsqSize() {
        int n = 1;
        for (int level : FSQ_LEVELS) {
            n *= level;
        }
        return n;
    }

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(256).ubatch(256)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    private static AudioClip concat(AudioClip a, AudioClip b) {
        float[] samples = new float[a.length() + b.length()];
        System.arraycopy(a.samples(), 0, samples, 0, a.length());
        System.arraycopy(b.samples(), 0, samples, a.length(), b.length());
        return new AudioClip(samples, a.sampleRate());
    }

    private static void assertAudioEquals(AudioClip expected, AudioClip actual) {
        assertEquals(expected.sampleRate(), actual.sampleRate());
        assertEquals(expected.length(), actual.length());
        float[] want = expected.samples();
        float[] got = actual.samples();
        for (int i = 0; i < want.length; i++) {
            if (Math.abs(want[i] - got[i]) > 2e-3f) {
                assertArrayEquals("sample " + i + " differs", want, got, 2e-3f);
            }
        }
        assertArrayEquals(want, got, 2e-3f);
    }

    /** The backbone fixture path, named so the test reads clearly. */
    private static final class Lfm3aBackboneFixturePath {
        static Path write() throws IOException {
            return com.eonmux.neuralfree.gguf.speech.Lfm3aBackboneFixture.ttsBackbone();
        }
    }

    // ------------------------------------------------------------------ configuration

    private static String audioConfig() {
        return "{'architectures':['Lfm3AudioForConditionalGeneration'],'model_type':'lfm3_audio',"
                + "'hidden_size':4,'encoder':{'dim_in':16,'sliding_window':4,'lfm':{" + audioLfm() + "}},"
                + "'codebooks':12,'audio_vocab_size':" + AUDIO_VOCAB + ",'maskgit_steps':3,"
                + "'interleaved_n_text':6,'interleaved_n_audio':12,'depthformer':{" + depthformer() + "}}";
    }

    private static String omniConfig() {
        return "{'architectures':['Lfm3OmniForConditionalGeneration'],'model_type':'lfm3_omni',"
                + "'hidden_size':4,'projector_hidden_act':'gelu','projector_bias':true,"
                + "'projector_layer_norm_eps':1e-06,'downsample_factor':2,"
                + "'num_codebooks':12,'codebook_size':" + AUDIO_VOCAB + ",'maskgit_schedule':[4,4,4],"
                + "'interleaved_n_text':6,'interleaved_n_audio':12,"
                + "'audio_encoder_config':{'num_mel_bins':16,'sliding_window':4,'text_config':{" + audioLfm() + "}},"
                + "'depth_config':{" + depthformer() + "},"
                + "'vision_config':{'hidden_size':8,'intermediate_size':16,'num_attention_heads':2,"
                + "'num_hidden_layers':2,'num_key_value_heads':1,'patch_size':2,'temporal_patch_size':2,"
                + "'in_channels':3,'conv_kernel_size':3,'norm_eps':1e-06,"
                + "'rope_parameters':{'rope_theta':1000000.0}}}";
    }

    private static String audioLfm() {
        return "'hidden_size':8,'num_hidden_layers':4,'num_attention_heads':2,'num_key_value_heads':1,"
                + "'norm_eps':1e-05,'block_ff_dim':12,'block_auto_adjust_ff_dim':false,"
                + "'block_ffn_dim_multiplier':null,'block_multiple_of':1,"
                + "'rope_parameters':{'rope_theta':1000000.0}";
    }

    private static String depthformer() {
        return "'hidden_size':4,'intermediate_size':8,'num_hidden_layers':2,'num_attention_heads':2,"
                + "'num_key_value_heads':1,'norm_eps':1e-05,'is_causal':false,'use_sandwich_norm':false,"
                + "'layer_types':['full_attention','full_attention'],"
                + "'rope_parameters':{'full_attention':{'rope_theta':1000000.0}}";
    }

    private static String featureExtractor() {
        return "'feature_extractor_type':'Lfm3AudioFeatureExtractor','sampling_rate':16000,'feature_size':128,"
                + "'n_fft':400,'win_length':400,'hop_length':160,'log_floor':-12.0,'normalization_offset':4.0,"
                + "'normalization_divisor':4.0,'encoder_stride':8";
    }

    private static String audioPreprocessor() {
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
