package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The LFM3 audio projector the converter writes, opened by the runtime: a synthetic Hugging Face checkpoint built
 * from {@link Lfm3aTowerTest#fixture()} is exported through the projector view and the written clip file is loaded
 * through {@link MmprojModel#openAudioEncoder()}. The encoded rows must equal the fixture's double-precision
 * reference graph, so the converter's downsampler, short-conv, attention and projector names, shapes and values are
 * proven end to end. The Omni checkpoint carries the same encoder under its HF-native {@code model.audio_encoder}
 * names beside the vision tower.
 */
public class Lfm3aMmprojRuntimeTest {

    /** The text width {@link Lfm3aTowerTest#fixture()} projects into. */
    private static final int TEXT = Lfm3aTowerTest.TEXT;
    /** The audio tower geometry of the fixture. */
    private static final int HIDDEN = Lfm3aTowerTest.HIDDEN;
    private static final int FF = Lfm3aTowerTest.FF;
    private static final int HEADS = Lfm3aTowerTest.HEADS;
    private static final int KV_HEADS = Lfm3aTowerTest.KV_HEADS;
    private static final int LAYERS = Lfm3aTowerTest.CONV.length + 1;
    private static final int N_MEL = Lfm3aTowerTest.N_MEL;
    private static final int WINDOW = Lfm3aTowerTest.WINDOW;

    private static final String FEATURE_EXTRACTOR = "'feature_extractor_type':'Lfm3AudioFeatureExtractor',"
            + "'sampling_rate':16000,'feature_size':128,'n_fft':400,'win_length':400,'hop_length':160,"
            + "'log_floor':-12.0,'normalization_offset':4.0,'normalization_divisor':4.0,'encoder_stride':8";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theAudioCheckpointRunsTheConvertedTower() throws Exception {
        MmprojFixture fixture = Lfm3aTowerTest.fixture();
        Path dir = checkpoint(fixture, false).write(tmp.newFolder("lfm3a").toPath());
        Path file = export(dir, tmp.newFolder("lfm3a-out").toPath().resolve("mmproj-audio.gguf"));

        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 77, 5);
        try (MmprojModel model = MmprojModel.open(file.toString());
             CpuAudioEncoder encoder = model.openAudioEncoder()) {
            assertTrue(model.caps().inpAudio());
            float[][] out = encoder.encode(chunk);
            AudioTowerSupport.assertRows(Lfm3aTowerTest.reference(fixture, chunk), out, 2e-4);
            assertEquals(10, out.length);
            assertEquals(TEXT, out[0].length);
        }
    }

    @Test
    public void theOmniAudioHalfRunsTheConvertedTower() throws Exception {
        MmprojFixture fixture = Lfm3aTowerTest.fixture();
        Path dir = checkpoint(fixture, true).write(tmp.newFolder("lfm3omni").toPath());
        Path file = export(dir, tmp.newFolder("lfm3omni-out").toPath().resolve("mmproj-omni.gguf"));

        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 65, 9);
        try (MmprojModel model = MmprojModel.open(file.toString());
             CpuAudioEncoder encoder = model.openAudioEncoder()) {
            assertTrue(model.caps().inpAudio());
            assertTrue(model.caps().inpVision());
            float[][] out = encoder.encode(chunk);
            AudioTowerSupport.assertRows(Lfm3aTowerTest.reference(fixture, chunk), out, 2e-4);
            assertEquals(TEXT, out[0].length);
        }
    }

    // ------------------------------------------------------------------ the synthetic checkpoint

    /** The checkpoint of {@code fixture}: the audio-only layout, or the Omni one beside the vision tower. */
    private static SyntheticHf.Builder checkpoint(MmprojFixture fixture, boolean omni) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(omni ? omniConfig() : audioConfig()));
        b.file("preprocessor_config.json", SyntheticHf.json(omni ? omniPreprocessor() : audioPreprocessor()));
        for (Map.Entry<String, long[]> e : fixture.shapes.entrySet()) {
            String gguf = e.getKey();
            long[] ne = e.getValue();
            b.tensor(hfName(gguf, omni), StDType.F32, hfShape(gguf, ne),
                    SyntheticHf.f32(fixture.get(gguf)));
        }
        identityBlock(b, omni);
        return b;
    }

    /**
     * The fourth encoder block: its weights are zero and its norms one, so it adds nothing and the three-block
     * reference graph matches the runtime's four-block one. The block exists because the checkpoint's projector is
     * the block-3 {@code mlp.model.3} linear, which the converter's name map only reaches when the audio
     * configuration declares more than three blocks.
     */
    private static void identityBlock(SyntheticHf.Builder b, boolean omni) {
        String h = (omni ? "model.audio_encoder.lfm.layers." : "encoder.lfm.lfm.layers.")
                + Lfm3aTowerTest.CONV.length + ".";
        float[] ones = new float[HIDDEN];
        java.util.Arrays.fill(ones, 1f);
        b.tensor(h + "operator_norm.weight", StDType.F32, new long[]{HIDDEN}, SyntheticHf.f32(ones));
        b.tensor(h + "ffn_norm.weight", StDType.F32, new long[]{HIDDEN}, SyntheticHf.f32(ones));
        b.tensor(h + "feed_forward.w3.weight", StDType.F32, new long[]{FF, HIDDEN},
                SyntheticHf.f32(new float[HIDDEN * FF]));
        b.tensor(h + "feed_forward.w2.weight", StDType.F32, new long[]{HIDDEN, FF},
                SyntheticHf.f32(new float[HIDDEN * FF]));
        b.tensor(h + "conv.in_proj.weight", StDType.F32, new long[]{3 * HIDDEN, HIDDEN},
                SyntheticHf.f32(new float[3 * HIDDEN * HIDDEN]));
        b.tensor(h + "conv.conv.weight", StDType.F32, new long[]{HIDDEN, 1, Lfm3aTowerTest.TAPS},
                SyntheticHf.f32(new float[HIDDEN * Lfm3aTowerTest.TAPS]));
        b.tensor(h + "conv.out_proj.weight", StDType.F32, new long[]{HIDDEN, HIDDEN},
                SyntheticHf.f32(new float[HIDDEN * HIDDEN]));
    }

    /** The HF name of a GGUF audio tensor under the checkpoint layout. */
    private static String hfName(String gguf, boolean omni) {
        String encoder = omni ? "model.audio_encoder." : "encoder.";
        String layers = omni ? "model.audio_encoder.lfm.layers." : "encoder.lfm.lfm.layers.";
        if (gguf.equals("a.pre_encode.out.weight") || gguf.equals("a.pre_encode.out.bias")) {
            return encoder + "downsample.lin_out." + suffix(gguf);
        }
        for (int i = 0; i < DOWNSAMPLE.length; i++) {
            String prefix = "a.conv1d." + DOWNSAMPLE_INDEX[i] + ".";
            if (gguf.startsWith(prefix)) {
                return encoder + "downsample." + DOWNSAMPLE[i] + "." + gguf.substring(prefix.length());
            }
        }
        if (gguf.equals("a.post_ln.weight")) {
            return omni ? "model.audio_encoder.lfm.embedding_norm.weight"
                    : "encoder.lfm.lfm.embedding_norm.weight";
        }
        if (gguf.startsWith("mm.a.mlp.")) {
            String rest = gguf.substring("mm.a.mlp.".length());
            int dot = rest.indexOf('.');
            int bid = Integer.parseInt(rest.substring(0, dot));
            String tail = rest.substring(dot + 1);
            if (omni) {
                String module = bid == 0 ? "layer_norm" : bid == 1 ? "linear_1" : "linear_2";
                return "model.audio_projector." + module + "." + tail;
            }
            return "mlp.model." + bid + "." + tail;
        }
        if (gguf.startsWith("a.blk.")) {
            String rest = gguf.substring("a.blk.".length());
            int dot = rest.indexOf('.');
            int bid = Integer.parseInt(rest.substring(0, dot));
            String tail = rest.substring(dot + 1);
            String h = layers + bid + ".";
            if (tail.startsWith("ln1.")) {
                return h + "operator_norm.weight";
            }
            if (tail.startsWith("ln2.")) {
                return h + "ffn_norm.weight";
            }
            if (tail.startsWith("ffn_up.")) {
                return h + "feed_forward.w3.weight";
            }
            if (tail.startsWith("ffn_gate.")) {
                return h + "feed_forward.w1.weight";
            }
            if (tail.startsWith("ffn_down.")) {
                return h + "feed_forward.w2.weight";
            }
            if (tail.startsWith("conv_pw1.")) {
                return h + "conv.in_proj.weight";
            }
            if (tail.startsWith("conv_dw.")) {
                return h + "conv.conv.weight";
            }
            if (tail.startsWith("conv_pw2.")) {
                return h + "conv.out_proj.weight";
            }
            if (tail.startsWith("attn_q_norm.")) {
                return h + "self_attn.q_layernorm.weight";
            }
            if (tail.startsWith("attn_k_norm.")) {
                return h + "self_attn.k_layernorm.weight";
            }
            if (tail.startsWith("attn_q.")) {
                return h + "self_attn.q_proj.weight";
            }
            if (tail.startsWith("attn_k.")) {
                return h + "self_attn.k_proj.weight";
            }
            if (tail.startsWith("attn_v.")) {
                return h + "self_attn.v_proj.weight";
            }
            if (tail.startsWith("attn_out.")) {
                return h + "self_attn.out_proj.weight";
            }
        }
        throw new IllegalArgumentException("no HF name for the fixture tensor '" + gguf + "'");
    }

    /** The checkpoint shape of a GGUF tensor: the reverse of the ggml dims, with the squeezes undone. */
    private static long[] hfShape(String gguf, long[] ne) {
        if (gguf.endsWith(".bias")) {
            return new long[]{count(ne)};
        }
        if (gguf.equals("a.pre_encode.out.weight")) {
            // encoder.downsample.lin_out is a kernel-1 Conv1d: [out, in, 1]
            return new long[]{ne[1], ne[0], 1};
        }
        if (gguf.endsWith("conv_dw.weight")) {
            // conv.conv is a depthwise Conv1d: [channels, 1, taps]
            return new long[]{ne[1], 1, ne[0]};
        }
        long[] shape = new long[ne.length];
        for (int i = 0; i < ne.length; i++) {
            shape[i] = ne[ne.length - 1 - i];
        }
        return shape;
    }

    private static String suffix(String name) {
        return name.substring(name.lastIndexOf('.') + 1);
    }

    /** Exports the projector view; F32 keeps the fixture's stored values exact. */
    private static Path export(Path dir, Path out) throws IOException {
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(dir.resolveSibling(dir.getFileName() + "-cache"))
                .outType(ModelSourceOptions.OutType.F32).build();
        try (ModelSource src = ModelSources.openProjector(dir, options)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        return out;
    }

    private static long count(long[] shape) {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        return n;
    }

    // ------------------------------------------------------------------ configuration

    private static final int[] DOWNSAMPLE_INDEX = {0, 2, 3, 5, 6};
    private static final String[] DOWNSAMPLE = {"conv_in", "conv_mid", "lin_mid", "conv_end", "lin_end"};

    private static String audioConfig() {
        return "{'architectures':['Lfm3AudioForConditionalGeneration'],'model_type':'lfm3_audio',"
                + "'hidden_size':" + TEXT + ",'encoder':{'dim_in':" + N_MEL + ",'sliding_window':" + WINDOW + ","
                + "'lfm':{" + audioLfm() + "}}}";
    }

    private static String omniConfig() {
        return "{'architectures':['Lfm3OmniForConditionalGeneration'],'model_type':'lfm3_omni',"
                + "'hidden_size':" + TEXT + ",'projector_hidden_act':'gelu','projector_bias':true,"
                + "'projector_layer_norm_eps':1e-06,'downsample_factor':2,"
                + "'audio_encoder_config':{'num_mel_bins':" + N_MEL + ",'sliding_window':" + WINDOW + ","
                + "'text_config':{" + audioLfm() + "}},"
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

    private static String audioPreprocessor() {
        return "{'feature_extractor':{" + FEATURE_EXTRACTOR + "}}";
    }

    /** The Omni preprocessor with the 384-frame video bound of {@code LFM3OmniMmprojModel} (lfm3.py:581). */
    private static String omniPreprocessor() {
        return "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5],"
                + "'size':{'shortest_edge':4096,'longest_edge':25165824},'resample':2,"
                + "'video_processor':{'fps':2.0,'min_frames':4,'temporal_patch_size':2,"
                + "'size':{'shortest_edge':4096,'longest_edge':25165824},'max_frames':384},"
                + "'feature_extractor':{" + FEATURE_EXTRACTOR + "}}";
    }
}
