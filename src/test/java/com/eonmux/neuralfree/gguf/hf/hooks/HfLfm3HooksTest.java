package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfCheckpointDetectors;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Checkpoint;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Sink;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.config;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.entry;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.plan;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.seq;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfLfm3Hooks} on synthetic LFM3 checkpoints: the per-layer key/value vector, the sliding-window pattern and
 * short-convolution cache, the partial-rotary {@code rope_freqs} factors, the convolution squeeze, the media tensor
 * filter, the audio/omni templates, the d1 decision metadata, and the d1 checkpoint detector.
 */
public class HfLfm3HooksTest {

    private static final String BASE = "'hidden_size': 8, 'num_hidden_layers': 3, 'vocab_size': 16,"
            + " 'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 4,"
            + " 'max_position_embeddings': 16, 'layer_types': ['full_attention', 'conv', 'sliding_attention'],"
            + " 'rope_parameters': {'full_attention': {'rope_theta': 10000.0, 'partial_rotary_factor': 0.5},"
            + " 'sliding_attention': {'rope_theta': 1000.0}},"
            + " 'sliding_window': 8, 'conv_L_cache': 4, 'norm_eps': 1e-05";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Checkpoint ckpt;

    @After
    public void close() throws IOException {
        if (ckpt != null) {
            ckpt.close();
        }
    }

    private static HfConfig lfm3(String extra) {
        return config("{'architectures': ['Lfm3ForCausalLM'], " + BASE + extra + "}");
    }

    @Test
    public void theFixupZeroesTheConvolutionLayersKeyValueHeads() {
        HfConfig fixed = HfLfm3Hooks.fixup(lfm3(""));
        assertEquals(Arrays.asList(1, 0, 1), fixed.findIntList("num_key_value_heads").orElse(null));
    }

    @Test
    public void aNonScalarKeyValueHeadCountIsRefused() {
        HfConfig cfg = config("{'architectures': ['Lfm3ForCausalLM'], " + BASE
                + ", 'num_key_value_heads': [1, 1, 1]}");
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfLfm3Hooks.fixup(cfg));
        assertTrue(e.getMessage(), e.getMessage().contains("num_key_value_heads"));
    }

    @Test
    public void theMetadataWritesTheHeadWidthPatternWindowAndShortConvCache() throws IOException {
        HfMetadataBuilder b = new HfMetadataBuilder("lfm3");
        HfLoadContext ctx = context(lfm3(""), entry("Lfm3ForCausalLM"), HfTensorIndex.of(Map.of()));
        entry("Lfm3ForCausalLM").meta().apply(ctx, b, null);
        HfMetadata m = b.build();
        assertEquals(3, m.get("lfm3.block_count"));
        assertEquals(Arrays.asList(1, 0, 1), m.get("lfm3.attention.head_count_kv"));
        assertEquals(4, m.get("lfm3.rope.dimension_count"));
        assertEquals(Arrays.asList(false, false, true), m.get("lfm3.attention.sliding_window_pattern"));
        assertEquals(8, m.get("lfm3.attention.sliding_window"));
        assertEquals(4, m.get("lfm3.shortconv.l_cache"));
        assertEquals(1e-05f, m.get("lfm3.attention.layer_norm_rms_epsilon"));
        assertEquals(10000.0f, m.get("lfm3.rope.freq_base"));
        assertEquals(1000.0f, m.get("lfm3.rope.freq_base_swa"));
    }

    @Test
    public void theRopeFreqsMaskTheUnrotatedPairs() throws IOException {
        HfLoadContext ctx = context(lfm3(""), entry("Lfm3ForCausalLM"), HfTensorIndex.of(Map.of()));
        Sink out = new Sink();
        HfLfm3Hooks.extra(ctx, out);
        assertArrayEquals(new float[]{1f, 1e30f}, floats(out.get("rope_freqs.weight")), 0f);
    }

    @Test
    public void theConvolutionWeightsSqueezeAndTheOthersMapPlainly() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("model.layers.1.conv.conv.weight", new long[]{8, 1, 4}, seq(32))
                .f32("model.layers.0.self_attn.q_proj.weight", new long[]{8, 8}, seq(64))
                .index();
        HfLoadContext ctx = context(lfm3(""), entry("Lfm3ForCausalLM"), index);
        Sink out = plan(ctx);
        assertArrayEquals(new long[]{8, 4}, out.get("blk.1.shortconv.conv.weight").shape());
        assertArrayEquals(new long[]{8, 8}, out.get("blk.0.attn_q.weight").shape());
    }

    @Test
    public void theFilterDropsTheMediaTensorsAndStripsTheLfmPrefix() {
        assertEquals("model.layers.0.self_attn.q_proj.weight",
                HfLfm3Hooks.FILTER.filter("lfm.model.layers.0.self_attn.q_proj.weight", n -> n));
        assertNull(HfLfm3Hooks.FILTER.filter("encoder.layers.0.linear.weight", n -> n));
        assertNull(HfLfm3Hooks.FILTER.filter("model.codebook_embedding.weight", n -> n));
        assertNull(HfLfm3Hooks.FILTER.filter("depthformer.model.layers.0.norm.weight", n -> n));
    }

    @Test
    public void theAudioCheckpointGetsTheAsrAndTtsTemplates() throws IOException {
        Map<String, Object> md = vocab("lfm3_audio");
        String asr = String.valueOf(md.get(String.format(GgufKeys.TOKENIZER_CHAT_TEMPLATE_N, "asr")));
        String tts = String.valueOf(md.get(String.format(GgufKeys.TOKENIZER_CHAT_TEMPLATE_N, "tts")));
        assertTrue(asr, asr.contains("Perform live ASR."));
        assertTrue(tts, tts.contains("Perform TTS in "));
        assertFalse(tts, tts.contains("<think></think>"));
    }

    @Test
    public void theOmniCheckpointGetsTheTtsTemplateWithTheThinkBlock() throws IOException {
        Map<String, Object> md = vocab("lfm3_omni");
        String tts = String.valueOf(md.get(String.format(GgufKeys.TOKENIZER_CHAT_TEMPLATE_N, "tts")));
        assertFalse(md.containsKey(String.format(GgufKeys.TOKENIZER_CHAT_TEMPLATE_N, "asr")));
        assertTrue(tts, tts.contains("Perform TTS in "));
        assertTrue(tts, tts.contains("<think></think>"));
    }

    @Test
    public void aTextCheckpointGetsNoSpeechTemplate() throws IOException {
        Map<String, Object> md = vocab("lfm3");
        assertFalse(md.containsKey(String.format(GgufKeys.TOKENIZER_CHAT_TEMPLATE_N, "tts")));
        assertFalse(md.containsKey(String.format(GgufKeys.TOKENIZER_CHAT_TEMPLATE_N, "asr")));
    }

    @Test
    public void theD1MetadataRecordsTheBidirectionalAttentionAndTemperatures() throws IOException {
        tmp.newFile("calibration.json");
        Files.writeString(tmp.getRoot().toPath().resolve("calibration.json"), "{\"default\": 0.7}");
        String json = "{'architectures': ['Lfm3D1Model'], 'model_type': 'lfm3_vl', 'd1_family': 'lfm3_vl',"
                + " 'd1_codes': 2, 'd1_attention': 'set', " + BASE + "}";
        HfMetadataBuilder b = new HfMetadataBuilder("lfm3");
        HfLoadContext ctx = context(configIn(tmp.getRoot().toPath(), json), entry("Lfm3D1Model"),
                HfTensorIndex.of(Map.of()));
        entry("Lfm3D1Model").meta().apply(ctx, b, null);
        HfMetadata m = b.build();
        assertEquals(false, m.get("lfm3.attention.causal"));
        assertEquals("lfm3-d1", m.get("lfm3.decision.type"));
        assertEquals(0.7f, m.get("lfm3.decision.temperature.noul"));
        assertEquals(0.7f, m.get("lfm3.decision.temperature.choice"));
        assertEquals(0.7f, m.get("lfm3.decision.temperature.score"));
    }

    @Test
    public void theOmniD1MetadataUsesItsOwnDecisionType() throws IOException {
        tmp.newFile("calibration.json");
        Files.writeString(tmp.getRoot().toPath().resolve("calibration.json"), "{\"default\": 0.5}");
        String json = "{'architectures': ['Lfm3D1Model'], 'model_type': 'lfm3_omni', 'd1_family': 'lfm3_omni',"
                + " 'd1_codes': 2, 'd1_attention': 'set', " + BASE + "}";
        HfMetadataBuilder b = new HfMetadataBuilder("lfm3");
        HfLoadContext ctx = context(configIn(tmp.getRoot().toPath(), json), entry("Lfm3D1Model"),
                HfTensorIndex.of(Map.of()));
        entry("Lfm3D1Model").meta().apply(ctx, b, null);
        assertEquals("lfm3-d1-omni", b.build().get("lfm3.decision.type"));
    }

    @Test
    public void anUnsupportedD1LayoutIsRefused() throws IOException {
        tmp.newFile("calibration.json");
        Files.writeString(tmp.getRoot().toPath().resolve("calibration.json"), "{\"default\": 0.7}");
        String json = "{'architectures': ['Lfm3D1Model'], 'model_type': 'lfm3_vl', 'd1_family': 'lfm3_vl',"
                + " 'd1_codes': 3, 'd1_attention': 'set', " + BASE + "}";
        HfMetadataBuilder b = new HfMetadataBuilder("lfm3");
        HfLoadContext ctx = context(config(json), entry("Lfm3D1Model"),
                HfTensorIndex.of(Map.of()));
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> entry("Lfm3D1Model").meta().apply(ctx, b, null));
        assertTrue(e.getMessage(), e.getMessage().contains("d1_codes"));
    }

    @Test
    public void theD1VocabularyAddsTheSystemoneTemplateWithAnEmptyThinkBlock() throws IOException {
        HfConfig cfg = config("{'architectures': ['Lfm3D1Model'], 'model_type': 'lfm3_vl', 'vocab_size': 16,"
                + " 'num_hidden_layers': 1}");
        HfArchEntry e = entry("Lfm3D1Model");
        HfLoadContext ctx = HfLoadContext.builder(cfg, e, Arch.LFM3)
                .tokenizerFiles(tokenizer()).build();
        Map<String, Object> md = e.vocab().build(ctx, new com.eonmux.neuralfree.gguf.tokenizer.hf
                .HfPreTokenizerResolver());
        String template = String.valueOf(md.get(String.format(GgufKeys.TOKENIZER_CHAT_TEMPLATE_N, "systemone")));
        assertTrue(template, template.contains("<|startoftext|>"));
        assertTrue(template, template.endsWith("<think></think>"));
    }

    @Test
    public void theDetectorRecognizesAnLfm3D1Checkpoint() throws IOException {
        Path dir = tmp.newFolder("d1").toPath();
        Files.writeString(dir.resolve("config.json"), "{\"architectures\": [\"D1Model\"],"
                + " \"model_type\": \"lfm3_vl\", \"text_config\": {\"architectures\": [\"Lfm3VlForConditionalGeneration\"],"
                + " \"num_hidden_layers\": 3, \"hidden_size\": 8}}");
        assertEquals(java.util.Optional.of("Lfm3D1"), HfCheckpointDetectors.match(dir));
        HfConfig cfg = HfConfigLoader.load(dir, ModelSourceOptions.builder().build());
        assertEquals("Lfm3D1Model", cfg.architecture());
        assertEquals("lfm3_vl", cfg.findStringOr(null, "d1_family"));
    }

    @Test
    public void anLfm2D1CheckpointIsNotTakenAsLfm3() throws IOException {
        Path dir = tmp.newFolder("d1-lfm2").toPath();
        Files.writeString(dir.resolve("config.json"), "{\"architectures\": [\"Lfm2ForCausalLM\"],"
                + " \"model_type\": \"lfm2\", \"auto_map\": {\"AutoModel\": \"x.D1Model\"},"
                + " \"text_config\": {\"architectures\": [\"Lfm2ForCausalLM\"], \"num_hidden_layers\": 2}}");
        assertFalse(HfCheckpointDetectors.match(dir).orElse("").equals("Lfm3D1"));
    }

    /** A GPT-2 tokenizer in a GGUF file beside an empty directory. */
    private HfTokenizerFiles tokenizer() throws IOException {
        Path dir = tmp.newFolder("tokenizer").toPath();
        Path gguf = tmp.getRoot().toPath().resolve("tokenizer.gguf");
        new GgufTestWriter().architecture("gpt2").str(GgufKeys.TOKENIZER_MODEL, "gpt2")
                .strArray(GgufKeys.TOKENIZER_LIST, "a", "b", "<eos>").writeTo(gguf);
        return HfTokenizerFiles.locate(dir).withTokenizerDir(gguf);
    }

    private static HfConfig configIn(Path dir, String singleQuotedJson) {
        return HfConfigLoader.fromJson(dir, singleQuotedJson.replace('\'', '"'));
    }

    private Map<String, Object> vocab(String modelType) throws IOException {
        HfConfig cfg = config("{'architectures': ['Lfm3ForCausalLM'], 'model_type': '" + modelType + "',"
                + " 'vocab_size': 16, 'hidden_size': 8, 'num_hidden_layers': 3}");
        HfArchEntry e = entry("Lfm3ForCausalLM");
        HfLoadContext ctx = HfLoadContext.builder(cfg, e, Arch.LFM3)
                .tokenizerFiles(tokenizer()).build();
        return e.vocab().build(ctx, new com.eonmux.neuralfree.gguf.tokenizer.hf.HfPreTokenizerResolver());
    }
}
