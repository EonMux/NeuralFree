package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfLfm3AudioFixtures;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The LFM3 audio projector export ({@code LFM3AudioMmprojModel}, lfm3.py:309-573): the feature-extractor and
 * generation checks, the {@code clip.audio} keys, the downsampler, short-conv and attention tensors, the
 * LayerNorm-GELU-linear projector, the depthformer with its tied per-codebook pages, the FSQ table of the liquid
 * codec and the forced F32 storage. The Omni checkpoint ({@code LFM3OmniMmprojModel}, lfm3.py:575-609) maps its
 * HF-native audio names onto the same layout beside the LFM3-VL vision tower.
 */
public class HfLfm3AudioSourcesTest {

    private static final int CODEBOOKS = HfLfm3AudioFixtures.CODEBOOKS;
    private static final int AUDIO_VOCAB = HfLfm3AudioFixtures.AUDIO_VOCAB;
    private static final int[] FSQ_LEVELS = HfLfm3AudioFixtures.FSQ_LEVELS;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private ModelSourceOptions options() throws IOException {
        return ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();
    }

    @Test
    public void theAudioCheckpointConvertsItsEncoderAndGenerator() throws IOException {
        HfLfm3AudioFixtures.Tower tower = HfLfm3AudioFixtures.defaultTower();
        SyntheticHf.Builder b = HfLfm3AudioFixtures.audioCheckpoint(tower, true, StDType.F16);
        Path dir = b.write(tmp.newFolder("lfm3a").toPath());
        HfLfm3AudioFixtures.writeCodec(dir);

        try (ModelSource src = ModelSources.openProjector(dir, options())) {
            assertEquals("clip", src.getString("general.architecture"));
            assertTrue(bool(src, "clip.has_audio_encoder"));
            assertEquals("lfm3a", src.getString("clip.audio.projector_type"));
            assertEquals(HfLfm3AudioFixtures.LAYERS, number(src, "clip.audio.block_count"));
            assertEquals(HfLfm3AudioFixtures.HIDDEN, number(src, "clip.audio.embedding_length"));
            assertEquals(HfLfm3AudioFixtures.FF, number(src, "clip.audio.feed_forward_length"));
            assertEquals(HfLfm3AudioFixtures.HEADS, number(src, "clip.audio.attention.head_count"));
            assertEquals(HfLfm3AudioFixtures.KV_HEADS, number(src, "clip.audio.attention.head_count_kv"));
            assertEquals(HfLfm3AudioFixtures.N_MEL, number(src, "clip.audio.num_mel_bins"));
            assertEquals(HfLfm3AudioFixtures.WINDOW, number(src, "clip.audio.window_size"));
            assertEquals(HfLfm3AudioFixtures.TEXT, number(src, "clip.audio.projection_dim"));
            assertEquals(HfLfm3AudioFixtures.EPS, decimal(src, "clip.audio.attention.layer_norm_epsilon"), 1e-9);

            assertTrue(bool(src, "clip.has_gen_audio_encoder"));
            assertEquals("lfm3a_gen", src.getString("clip.gen.audio.projector_type"));
            assertEquals(HfLfm3AudioFixtures.DEPTH_LAYERS, number(src, "clip.gen.audio.block_count"));
            assertEquals(HfLfm3AudioFixtures.DEPTH_HIDDEN, number(src, "clip.gen.audio.embedding_length"));
            assertEquals(HfLfm3AudioFixtures.DEPTH_FF, number(src, "clip.gen.audio.feed_forward_length"));
            assertEquals(HfLfm3AudioFixtures.DEPTH_HEADS, number(src, "clip.gen.audio.attention.head_count"));
            assertEquals(HfLfm3AudioFixtures.DEPTH_KV,
                    number(src, "clip.gen.audio.attention.head_count_kv"));
            assertEquals(HfLfm3AudioFixtures.TEXT, number(src, "clip.gen.audio.projection_dim"));
            assertEquals(1.0f, decimal(src, "clip.gen.audio.sampling.temp"), 1e-6);
            assertEquals(128, number(src, "clip.gen.audio.sampling.top_k"));
            assertEquals(1.1f, decimal(src, "clip.gen.audio.sampling.penalty_repeat"), 1e-6);
            assertEquals(4096, number(src, "clip.gen.audio.sampling.penalty_last_n"));

            Set<String> names = names(src);
            for (String name : new String[]{
                    "a.conv1d.0.weight", "a.conv1d.0.bias", "a.conv1d.2.weight", "a.conv1d.2.bias",
                    "a.conv1d.3.weight", "a.conv1d.5.weight", "a.conv1d.6.weight",
                    "a.pre_encode.out.weight", "a.pre_encode.out.bias",
                    "a.blk.0.conv_pw1.weight", "a.blk.0.conv_dw.weight", "a.blk.0.conv_pw2.weight",
                    "a.blk.1.attn_q.weight", "a.blk.1.attn_q_norm.weight", "a.blk.1.attn_k.weight",
                    "a.blk.1.attn_k_norm.weight", "a.blk.1.attn_v.weight", "a.blk.1.attn_out.weight",
                    "a.blk.3.ffn_up.weight", "a.blk.3.ffn_down.weight", "a.post_ln.weight",
                    "mm.a.mlp.0.weight", "mm.a.mlp.0.bias", "mm.a.mlp.1.weight", "mm.a.mlp.1.bias",
                    "mm.a.mlp.3.weight", "mm.a.mlp.3.bias",
                    "a.gen.code.embd.weight", "a.gen.code.head_norm.weight", "a.gen.code.out_embd.weight",
                    "a.gen.code.proj_in.weight", "a.gen.code.proj_in.bias", "a.gen.code.mask_embd",
                    "a.gen.code.output_norm.weight", "a.gen.code.blk.0.attn_gate.weight",
                    "a.gen.code.blk.1.ffn_down.weight",
                    "a.gen.wav.quant.first.codebook.weight", "a.gen.wav.quant.first.out_proj.weight",
                    "a.gen.wav.quant.first.out_proj.bias",
                    "a.gen.wav.tfm.in_proj.weight", "a.gen.wav.tfm.in_proj.bias",
                    "a.gen.wav.tfm.out_proj.weight", "a.gen.wav.tfm.out_proj.bias",
                    "a.gen.wav.tfm.output_norm.weight", "a.gen.wav.tfm.blk.0.conv_pw1.weight",
                    "a.gen.wav.tfm.blk.0.conv_dw.weight", "a.gen.wav.tfm.blk.0.conv_pw2.weight",
                    "a.gen.wav.tfm.blk.0.attn_post_norm.weight", "a.gen.wav.tfm.blk.0.ffn_post_norm.weight"}) {
                assertTrue(name + " is missing from " + names, names.contains(name));
            }
            assertFalse("a text tensor leaked", names.contains("model.embed_tokens.weight"));

            assertArrayEquals(new long[]{4, (long) fsqCodebookSize()}, dims(src,
                    "a.gen.wav.quant.first.codebook.weight"));
            assertArrayEquals(new long[]{HfLfm3AudioFixtures.DEPTH_HIDDEN, 1, CODEBOOKS},
                    dims(src, "a.gen.code.head_norm.weight"));
            assertArrayEquals(new long[]{HfLfm3AudioFixtures.DEPTH_HIDDEN, AUDIO_VOCAB, CODEBOOKS},
                    dims(src, "a.gen.code.embd.weight"));
            assertArrayEquals(new long[]{HfLfm3AudioFixtures.HIDDEN, HfLfm3AudioFixtures.ADAPTER},
                    dims(src, "mm.a.mlp.1.weight"));

            assertEquals(GGMLType.F32, ((HfModelSource) src).plan().type("a.conv1d.0.weight"));
            assertEquals(GGMLType.F32, ((HfModelSource) src).plan().type("a.blk.0.conv_dw.weight"));
            assertEquals(GGMLType.F32,
                    ((HfModelSource) src).plan().type("a.gen.wav.quant.first.codebook.weight"));
            assertEquals("an unforced tensor keeps the source type", GGMLType.F16,
                    ((HfModelSource) src).plan().type("a.blk.0.conv_pw1.weight"));

            try (CpuWeights w = CpuWeights.open(src)) {
                float[] table = w.floats("a.gen.wav.quant.first.codebook.weight");
                assertArrayEquals(new float[]{-1f, -1f, -1f, -1f}, Arrays.copyOf(table, 4), 1e-6f);
                assertArrayEquals(new float[]{0.75f, 1f, 2f / 3f, 2f / 3f},
                        Arrays.copyOfRange(table, table.length - 4, table.length), 1e-6f);
                assertArrayEquals(tower.values("a.blk.0.ffn_up.weight"), w.floats("a.blk.0.ffn_up.weight"),
                        1e-6f);
                assertArrayEquals(tower.values("a.gen.code.embd.weight"), w.floats("a.gen.code.embd.weight"),
                        1e-6f);
            }
        }
    }

    @Test
    public void aCheckpointWithoutTheCodecExportsNoGenerator() throws IOException {
        Path dir = HfLfm3AudioFixtures.audioCheckpoint(HfLfm3AudioFixtures.defaultTower(), false, StDType.F16)
                .write(tmp.newFolder("lfm3a-nocodec").toPath());
        try (ModelSource src = ModelSources.openProjector(dir, options())) {
            assertTrue(bool(src, "clip.has_audio_encoder"));
            assertFalse(src.getMetadata().containsKey("clip.has_gen_audio_encoder"));
            assertFalse(src.getMetadata().containsKey("clip.gen.audio.projector_type"));
            Set<String> names = names(src);
            for (String name : names) {
                assertFalse("the generator leaked: " + name, name.startsWith("a.gen."));
            }
            assertTrue(names.contains("mm.a.mlp.3.weight"));
        }
    }

    @Test
    public void aDepthheadThatDiffersFromItsEmbeddingIsRefused() throws IOException {
        HfLfm3AudioFixtures.Tower tower = HfLfm3AudioFixtures.defaultTower();
        SyntheticHf.Builder b = HfLfm3AudioFixtures.audioCheckpoint(tower, true, StDType.F16);
        long[] page = HfLfm3AudioFixtures.depthEmbeddingShape();
        float[] ones = new float[Math.toIntExact(page[0] * page[1])];
        Arrays.fill(ones, 1f);
        b.tensor("depth_embeddings.2.to_logits.weight", StDType.F16, page, SyntheticHf.f16(ones));
        Path dir = b.write(tmp.newFolder("lfm3a-untied").toPath());
        HfLfm3AudioFixtures.writeCodec(dir);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.openProjector(dir, options()));
        assertTrue(e.getMessage(), e.getMessage().contains("not tied"));
    }

    @Test
    public void aDifferentFeatureExtractorIsRefusedByName() throws IOException {
        SyntheticHf.Builder b = HfLfm3AudioFixtures.audioCheckpoint(HfLfm3AudioFixtures.defaultTower(), false,
                StDType.F16);
        b.file("preprocessor_config.json", SyntheticHf.json("{'feature_extractor':{"
                + "'feature_extractor_type':'Lfm3AudioFeatureExtractor','sampling_rate':16000,'feature_size':128,"
                + "'n_fft':400,'win_length':400,'hop_length':320,'log_floor':-12.0,'normalization_offset':4.0,"
                + "'normalization_divisor':4.0,'encoder_stride':8}}"));
        Path dir = b.write(tmp.newFolder("lfm3a-fe").toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.openProjector(dir, options()));
        assertTrue(e.getMessage(), e.getMessage().contains("feature_extractor.hop_length"));
    }

    @Test
    public void aCodecThatDiffersFromTheGeneratorIsRefusedByName() throws IOException {
        SyntheticHf.Builder b = HfLfm3AudioFixtures.audioCheckpoint(HfLfm3AudioFixtures.defaultTower(), true,
                StDType.F16);
        Path dir = b.write(tmp.newFolder("lfm3a-codec").toPath());
        HfLfm3AudioFixtures.writeCodec(dir);
        Files.write(dir.resolve("audio_tokenizer").resolve("config.json"),
                SyntheticHf.json("{'model_type':'liquid_codec','fsq_levels':[8,7,6,6],'n_q':12,"
                        + "'quantizer_dim':256,'encoder_dim':512,'stft_n_fft':1280,'stft_hop_length':320,"
                        + "'stft_win_length':1280,'stft_frame_stack':6,'decoder_window':30,"
                        + "'sampling_rate':16000}").getBytes(java.nio.charset.StandardCharsets.UTF_8));
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.openProjector(dir, options()));
        assertTrue(e.getMessage(), e.getMessage().contains("sampling_rate"));
    }

    @Test
    public void aCausalDepthformerIsRefused() throws IOException {
        SyntheticHf.Builder b = HfLfm3AudioFixtures.audioCheckpoint(HfLfm3AudioFixtures.defaultTower(), true,
                StDType.F16);
        b.config(SyntheticHf.json(HfLfm3AudioFixtures.audioConfigJson()
                .replace("'is_causal':false", "'is_causal':true")));
        Path dir = b.write(tmp.newFolder("lfm3a-causal").toPath());
        HfLfm3AudioFixtures.writeCodec(dir);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.openProjector(dir, options()));
        assertTrue(e.getMessage(), e.getMessage().contains("bidirectional"));
    }

    @Test
    public void aMissingFeedForwardKeyIsRefusedByName() throws IOException {
        SyntheticHf.Builder b = HfLfm3AudioFixtures.audioCheckpoint(HfLfm3AudioFixtures.defaultTower(), false,
                StDType.F16);
        // llama.cpp reads the three block keys straight out of the hparams (lfm2.py:17-32)
        b.config(SyntheticHf.json(HfLfm3AudioFixtures.audioConfigJson()
                .replace("'block_multiple_of':1,", "")));
        Path dir = b.write(tmp.newFolder("lfm3a-ff").toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.openProjector(dir, options()));
        assertTrue(e.getMessage(), e.getMessage().contains("block_multiple_of"));
    }

    @Test
    public void theOmniCheckpointMapsItsAudioNamesBesideTheVisionTower() throws IOException {
        Path dir = HfLfm3AudioFixtures.omniCheckpoint(HfLfm3AudioFixtures.defaultTower(), true, StDType.F16)
                .write(tmp.newFolder("lfm3omni").toPath());
        HfLfm3AudioFixtures.writeCodec(dir);
        try (ModelSource src = ModelSources.openProjector(dir, options())) {
            assertEquals("lfm3vl", src.getString("clip.vision.projector_type"));
            assertEquals("lfm3a", src.getString("clip.audio.projector_type"));
            assertEquals("lfm3a_gen", src.getString("clip.gen.audio.projector_type"));
            assertTrue(bool(src, "clip.has_audio_encoder"));
            assertTrue(bool(src, "clip.has_vision_encoder"));
            assertEquals(HfLfm3AudioFixtures.N_MEL, number(src, "clip.audio.num_mel_bins"));
            assertEquals(HfLfm3AudioFixtures.WINDOW, number(src, "clip.audio.window_size"));
            assertEquals(HfLfm3AudioFixtures.HIDDEN, number(src, "clip.audio.embedding_length"));
            assertEquals(8, number(src, "clip.vision.embedding_length"));
            assertEquals(HfLfm3AudioFixtures.TEXT, number(src, "clip.vision.projection_dim"));
            assertEquals(0.4f, decimal(src, "clip.gen.audio.sampling.temp"), 1e-6);
            assertEquals(1.2f, decimal(src, "clip.gen.audio.sampling.penalty_repeat"), 1e-6);
            assertEquals(64, number(src, "clip.gen.audio.sampling.penalty_last_n"));
            assertFalse("the Omni model card sets no top-k",
                    src.getMetadata().containsKey("clip.gen.audio.sampling.top_k"));

            Set<String> names = names(src);
            for (String name : names) {
                assertFalse("an HF-native name leaked: " + name, name.startsWith("model."));
            }
            for (String name : new String[]{"a.conv1d.0.weight", "a.blk.0.conv_pw1.weight",
                    "a.blk.1.attn_q.weight", "a.post_ln.weight", "mm.a.mlp.0.weight", "mm.a.mlp.1.weight",
                    "mm.a.mlp.3.weight", "a.gen.code.embd.weight", "a.gen.code.head_norm.weight",
                    "a.gen.code.out_embd.weight", "a.gen.code.mask_embd",
                    "a.gen.wav.quant.first.codebook.weight"}) {
                assertTrue(name + " is missing from " + names, names.contains(name));
            }
        }
    }

    @Test
    public void bothArchitecturesAreRegisteredAsProjectors() throws IOException {
        for (String name : new String[]{"Lfm3AudioForConditionalGeneration",
                "Lfm3OmniForConditionalGeneration"}) {
            assertTrue(name, HfArchRegistry.has(name, HfArchEntry.Kind.MMPROJ));
            String config = name.startsWith("Lfm3Audio") ? HfLfm3AudioFixtures.audioConfigJson()
                    : HfLfm3AudioFixtures.omniConfigJson();
            Path dir = SyntheticHf.builder().config(SyntheticHf.json(config))
                    .f32("model.embed_tokens.weight", 4, 4)
                    .write(tmp.newFolder(name).toPath());
            HfConfig cfg = HfConfigLoader.load(dir, Collections.emptyMap());
            HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.MMPROJ);
            assertEquals(HfArchEntry.Kind.MMPROJ, entry.kind());
            assertTrue(entry.hfNames().contains(name));
        }
    }

    private static int fsqCodebookSize() {
        int n = 1;
        for (int level : FSQ_LEVELS) {
            n *= level;
        }
        return n;
    }

    private static Set<String> names(ModelSource src) {
        Set<String> out = new HashSet<>();
        for (GGUFTensorInfo t : src.getTensors()) {
            out.add(t.name());
        }
        return out;
    }

    private static long[] dims(ModelSource src, String name) {
        for (GGUFTensorInfo t : src.getTensors()) {
            if (t.name().equals(name)) {
                return t.dims();
            }
        }
        throw new AssertionError(name + " is missing");
    }

    private static boolean bool(ModelSource src, String key) {
        Object v = src.getMetadata().get(key);
        assertTrue(key + " is missing", v instanceof Boolean);
        return (Boolean) v;
    }

    private static long number(ModelSource src, String key) {
        Object v = src.getMetadata().get(key);
        assertTrue(key + " is missing", v instanceof Number);
        return ((Number) v).longValue();
    }

    private static double decimal(ModelSource src, String key) {
        Object v = src.getMetadata().get(key);
        assertTrue(key + " is missing", v instanceof Number);
        return ((Number) v).doubleValue();
    }
}
