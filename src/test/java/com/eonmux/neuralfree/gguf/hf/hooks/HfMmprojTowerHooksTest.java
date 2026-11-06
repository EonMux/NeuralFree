package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
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
import java.util.function.UnaryOperator;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The projector hooks of the audio towers and the multi-encoder families against llama.cpp's converter classes: the
 * {@code clip.*} keys each {@code set_gguf_parameters} writes and the GGUF names its {@code modify_tensors} plans,
 * through the registered {@link HfArchEntry} so the block count and tensor name map are the delivered ones.
 */
public class HfMmprojTowerHooksTest {

    private static final int TEXT = 8;
    private static final String COLORS = "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5]}";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ------------------------------------------------------------------ helpers

    private HfLoadContext ctx(String architecture, String body, Map<String, RowSource> index) throws IOException {
        return ctx(tmp.newFolder().toPath(), architecture, body, index);
    }

    private HfLoadContext ctx(Path dir, String architecture, String body, Map<String, RowSource> index)
            throws IOException {
        String json = ("{'architectures':['" + architecture + "'], " + body + "}").replace('\'', '"');
        // a body may close the configuration root itself
        if (closes(json) > opens(json)) {
            json = json.substring(0, json.lastIndexOf('}'));
        }
        HfConfig cfg = HfConfigLoader.fromJson(dir, json);
        HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.MMPROJ);
        return HfLoadContext.builder(cfg, entry, Arch.CLIP)
                .index(HfTensorIndex.of(index))
                .tokenizerFiles(HfTokenizerFiles.locate(dir))
                .build();
    }

    private static int opens(String s) {
        return (int) s.chars().filter(c -> c == '{').count();
    }

    private static int closes(String s) {
        return (int) s.chars().filter(c -> c == '}').count();
    }

    private static HfMetadata meta(HfLoadContext ctx) throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.CLIP.ggufName());
        ctx.entry().meta().apply(ctx, kv, null);
        return kv.build();
    }

    private static HfQwenHooksTest.RecordingSink tensors(HfLoadContext ctx, String hfName, long... shape)
            throws IOException {
        HfQwenHooksTest.RecordingSink sink = new HfQwenHooksTest.RecordingSink();
        ctx.entry().tensors().apply(ctx, hfName, seq(shape), sink);
        return sink;
    }

    private static void file(Path dir, String name, String json) throws IOException {
        Path p = dir.resolve(name);
        Files.createDirectories(p.getParent());
        Files.write(p, json.replace('\'', '"').getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    private static int u32(HfMetadata m, String key) {
        return ((Number) m.get(key)).intValue();
    }

    private static Object get(HfMetadata m, String key) {
        return m.get(key);
    }

    // ------------------------------------------------------------------ GLM-ASR

    @Test
    public void glmAsrNamesItsBoundaryRowsAndAdapter() throws IOException {
        HfLoadContext ctx = ctx("GlmasrModel", "'hidden_size':" + TEXT + ",'merge_factor':2,"
                + "'audio_config':{'hidden_size':8,'d_model':8,'intermediate_size':16,'encoder_ffn_dim':16,"
                + "'num_attention_heads':2,'encoder_attention_heads':2,'num_mel_bins':4,'num_hidden_layers':2,"
                + "'layer_norm_eps':1e-05}", Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("glma", get(m, "clip.projector_type"));
        assertEquals(4, u32(m, "clip.audio.num_mel_bins"));
        assertEquals(2, u32(m, "clip.audio.block_count"));
        assertEquals(8, u32(m, "clip.audio.projection_dim"));
        assertEquals(2, u32(m, "clip.audio.projector.stack_factor"));
        assertEquals(1e-5, ((Number) get(m, "clip.audio.attention.layer_norm_epsilon")).doubleValue(), 1e-9);

        assertEquals("audio.multi_modal_projector.ln_pre.weight",
                HfAudioTowerHooks.GLMA_FILTER.filter("audio_encoder.layer_norm.weight", UnaryOperator.identity()));
        HfQwenHooksTest.RecordingSink sink = tensors(ctx, "audio_tower.conv1.bias", 8);
        assertEquals("a conv-stem bias becomes a one-wide row (ultravox.py:83-85)",
                Arrays.asList(8L, 1L), boxed(sink.tensors.get("a.conv1d.1.bias").shape()));
        HfQwenHooksTest.RecordingSink boundary = tensors(ctx, "audio_encoder.audio_bos_eos_token.weight", 2, 8);
        assertEquals(Arrays.asList("v.boi", "v.eoi"), new java.util.ArrayList<>(boundary.tensors.keySet()));
        assertEquals(Arrays.asList(8L), boxed(boundary.tensors.get("v.boi").shape()));
        HfQwenHooksTest.RecordingSink dropped = tensors(ctx, "audio_encoder.adapting.proj.weight", 8, 8);
        assertTrue(dropped.tensors.isEmpty());
        assertEquals(1, dropped.dropped.size());
    }

    // ------------------------------------------------------------------ LFM2 audio

    @Test
    public void lfm2AudioReadsItsEncoderObjectAndFoldsItsConformer() throws IOException {
        HfLoadContext ctx = ctx("Lfm2AudioForConditionalGeneration", "'hidden_size':" + TEXT + ",'encoder':"
                + "{'d_model':8,'n_heads':2,'feat_in':4,'num_hidden_layers':2}", Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("lfm2a", get(m, "clip.projector_type"));
        assertEquals(4, u32(m, "clip.audio.num_mel_bins"));
        assertEquals(8, u32(m, "clip.audio.feed_forward_length"));
        assertEquals(2, u32(m, "clip.audio.block_count"));
        assertEquals(1e-5, ((Number) get(m, "clip.audio.attention.layer_norm_epsilon")).doubleValue(), 1e-9);
        assertEquals(2, ctx.blockCount());
    }

    // ------------------------------------------------------------------ Granite Speech

    @Test
    public void graniteSpeechNamesItsConformerAndQFormerKeys() throws IOException {
        HfLoadContext ctx = ctx("GraniteSpeechForConditionalGeneration", "'hidden_size':" + TEXT + ","
                + "'encoder_config':{'hidden_dim':8,'feedforward_mult':2,'num_heads':2,'num_layers':2,"
                + "'input_dim':4,'context_size':4,'conv_kernel_size':3,'max_pos_emb':8},"
                + "'window_size':4,'downsample_rate':2,"
                + "'projector_config':{'num_attention_heads':2,'encoder_hidden_size':8}}", Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("granite_speech", get(m, "clip.projector_type"));
        assertEquals(4, u32(m, "clip.audio.num_mel_bins"));
        assertEquals(16, u32(m, "clip.audio.feed_forward_length"));
        assertEquals(2, u32(m, "clip.audio.block_count"));
        assertEquals(4, u32(m, "clip.audio.chunk_size"));
        assertEquals(3, u32(m, "clip.audio.conv_kernel_size"));
        assertEquals(8, u32(m, "clip.audio.max_pos_emb"));
        assertEquals(4, u32(m, "clip.audio.projector.window_size"));
        assertEquals(2, u32(m, "clip.audio.projector.downsample_rate"));
        assertEquals(2, u32(m, "clip.audio.projector.head_count"));
    }

    @Test
    public void graniteSpeechPlusWritesItsFeatureLayers() throws IOException {
        HfLoadContext ctx = ctx("GraniteSpeechPlusForConditionalGeneration", "'hidden_size':" + TEXT + ","
                + "'encoder_config':{'hidden_dim':8,'feedforward_mult':2,'num_heads':2,'num_layers':2,"
                + "'input_dim':4,'context_size':4,'conv_kernel_size':3,'max_pos_emb':8,"
                + "'cat_hidden_layers':[0,2]},'window_size':4,'downsample_rate':2,"
                + "'projector_config':{'num_attention_heads':2,'encoder_hidden_size':24}}", Collections.emptyMap());
        assertEquals(Arrays.asList(0, 2), get(meta(ctx), "clip.audio.feature_layer"));
    }

    @Test
    public void graniteSpeechPlusRefusesAMismatchedProjectorWidth() throws IOException {
        HfLoadContext ctx = ctx("GraniteSpeechPlusForConditionalGeneration", "'hidden_size':" + TEXT + ","
                + "'encoder_config':{'hidden_dim':8,'feedforward_mult':2,'num_heads':2,'num_layers':2,"
                + "'input_dim':4,'context_size':4,'conv_kernel_size':3,'max_pos_emb':8,"
                + "'cat_hidden_layers':[0,2]},'window_size':4,'downsample_rate':2,"
                + "'projector_config':{'num_attention_heads':2,'encoder_hidden_size':8}}", Collections.emptyMap());
        com.eonmux.neuralfree.gguf.hparams.GgufFormatException e = assertThrows(
                com.eonmux.neuralfree.gguf.hparams.GgufFormatException.class, () -> meta(ctx));
        assertTrue(e.getMessage(), e.getMessage().contains("encoder_hidden_size"));
    }

    // ------------------------------------------------------------------ Voxtral realtime

    @Test
    public void voxtralRealtimeNamesItsWindowAndHeadWidth() throws IOException {
        HfLoadContext ctx = ctx("VoxtralRealtimeForConditionalGeneration", "'hidden_size':" + TEXT + ","
                + "'downsample_factor':4,'audio_config':{'hidden_size':8,'intermediate_size':16,"
                + "'num_attention_heads':2,'num_hidden_layers':2,'num_mel_bins':4,'head_dim':4,"
                + "'rms_norm_eps':1e-06,'sliding_window':8}}", Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("voxtral_rt", get(m, "clip.projector_type"));
        assertEquals(4, u32(m, "clip.audio.attention.head_dim"));
        assertEquals(8, u32(m, "clip.audio.window_size"));
        assertEquals(4, u32(m, "clip.audio.projector.stack_factor"));
        assertEquals(1e-6, ((Number) get(m, "clip.audio.attention.layer_norm_epsilon")).doubleValue(), 1e-9);

        HfQwenHooksTest.RecordingSink sink = tensors(ctx, "audio_tower.layers.0.self_attn.q_proj.weight", 8, 8);
        assertNotNull(sink.tensors.get("a.blk.0.attn_q.weight"));
        assertEquals("rotate-half rows are un-permuted (ultravox.py:245-249)",
                Arrays.asList(8L, 8L), boxed(sink.tensors.get("a.blk.0.attn_q.weight").shape()));
    }

    // ------------------------------------------------------------------ Qwen3-ASR

    @Test
    public void qwen3AsrCarriesTheAudioHalfAndItsPositionTable() throws IOException {
        HfLoadContext ctx = ctx("Qwen3ASRForConditionalGeneration", "'hidden_size':" + TEXT + ","
                + "'audio_config':{'hidden_size':8,'d_model':8,'intermediate_size':16,'encoder_ffn_dim':16,"
                + "'num_attention_heads':2,'encoder_attention_heads':2,'num_mel_bins':4,"
                + "'num_hidden_layers':2,'layer_norm_eps':1e-05}", Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("qwen3a", get(m, "clip.projector_type"));
        assertTrue((Boolean) get(m, "clip.has_audio_encoder"));
        assertFalse(m.has("clip.has_vision_encoder"));
        assertEquals(2, u32(m, "clip.audio.block_count"));
        assertEquals(2, ctx.blockCount());

        HfQwenHooksTest.RecordingSink sink = new HfQwenHooksTest.RecordingSink();
        ctx.entry().extra().generate(ctx, sink);
        assertNotNull(sink.tensors.get("a.position_embd.weight"));
        assertEquals(Arrays.asList(1500L, 8L), boxed(sink.tensors.get("a.position_embd.weight").shape()));
    }

    // ------------------------------------------------------------------ Qwen2.5-Omni / Qwen3-Omni

    @Test
    public void qwen25OmniWritesTheLegacyMixedModalityProjector() throws IOException {
        Path dir = tmp.newFolder().toPath();
        file(dir, "preprocessor_config.json", COLORS);
        HfLoadContext ctx = ctx(dir, "Qwen2_5OmniModel", "'hidden_size':" + TEXT + ","
                + "'thinker_config':{'text_config':{'hidden_size':8},'vision_config':{'depth':2,'num_heads':2,'hidden_size':16,"
                + "'intermediate_size':32,'patch_size':2,'spatial_merge_size':2,'fullatt_block_indexes':[0]},"
                + "'audio_config':{'hidden_size':8,'d_model':8,'intermediate_size':16,"
                + "'encoder_ffn_dim':16,'num_attention_heads':2,'encoder_attention_heads':2,"
                + "'num_mel_bins':4,'encoder_layers':2,'layer_norm_eps':1e-05}}}", Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("qwen2.5o", get(m, "clip.projector_type"));
        assertFalse(m.has("clip.audio.projector_type"));
        assertTrue((Boolean) get(m, "clip.has_vision_encoder"));
        assertTrue((Boolean) get(m, "clip.has_audio_encoder"));
        assertEquals("the multi-encoder block count", 128, ctx.blockCount());
        assertEquals(2, u32(m, "clip.audio.block_count"));
        assertEquals(4, u32(m, "clip.audio.num_mel_bins"));
    }

    @Test
    public void qwen3OmniNamesEachTowerWithItsOwnProjector() throws IOException {
        Path dir = tmp.newFolder().toPath();
        file(dir, "preprocessor_config.json", COLORS);
        HfLoadContext ctx = ctx(dir, "Qwen3OmniMoeForConditionalGeneration", "'hidden_size':" + TEXT + ","
                + "'thinker_config':{'text_config':{'hidden_size':8},'vision_config':{'depth':2,'num_heads':2,'hidden_size':16,"
                + "'intermediate_size':32,'patch_size':16,'spatial_merge_size':2,'deepstack_visual_indexes':[1]},"
                + "'audio_config':{'hidden_size':8,'d_model':8,'intermediate_size':16,"
                + "'encoder_ffn_dim':16,'num_attention_heads':2,'encoder_attention_heads':2,"
                + "'num_mel_bins':4,'encoder_layers':2,'layer_norm_eps':1e-05}}}", Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("qwen3vl_merger", get(m, "clip.vision.projector_type"));
        assertEquals("qwen3a", get(m, "clip.audio.projector_type"));
        assertFalse(m.has("clip.projector_type"));
        assertEquals(128, ctx.blockCount());

        assertEquals("audio_tower.conv1.bias",
                HfQwenOmniHooks.QWEN3_OMNI_FILTER.filter("thinker.audio_tower.conv1.bias",
                        UnaryOperator.identity()));
        assertEquals("visual.merger.ln_q.weight",
                HfQwenOmniHooks.QWEN3_OMNI_FILTER.filter("model.visual.merger.ln_q.weight",
                        UnaryOperator.identity()));
    }

    // ------------------------------------------------------------------ Gemma 3n / Gemma 4

    @Test
    public void gemma3nWritesBothTowersUnderTheMultiTowerMap() throws IOException {
        Path dir = tmp.newFolder().toPath();
        file(dir, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5],"
                + "'size':{'height':64,'width':64},'image_seq_length':16}");
        HfLoadContext ctx = ctx(dir, "Gemma3nForConditionalGeneration", "'hidden_size':" + TEXT + ","
                + "'vision_config':{'hidden_size':16,'intermediate_size':32,'num_attention_heads':2,"
                + "'layer_norm_eps':1e-06},"
                + "'audio_config':{'hidden_size':8,'conf_num_attention_heads':2,"
                + "'conf_num_hidden_layers':2,'input_feat_size':4,'intermediate_size':16}}",
                Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("gemma3nv", get(m, "clip.vision.projector_type"));
        assertEquals("gemma3na", get(m, "clip.audio.projector_type"));
        assertEquals(64, u32(m, "clip.vision.image_size"));
        assertEquals(4, u32(m, "clip.vision.patch_size"));
        assertEquals(4, u32(m, "clip.audio.num_mel_bins"));
        assertEquals(128, ctx.blockCount());
    }

    @Test
    public void gemma4WritesTheVisionAndAudioTowers() throws IOException {
        Path dir = tmp.newFolder().toPath();
        file(dir, "preprocessor_config.json", COLORS);
        HfLoadContext ctx = ctx(dir, "Gemma4ForConditionalGeneration", "'hidden_size':" + TEXT + ","
                + "'vision_config':{'hidden_size':16,'intermediate_size':32,'num_attention_heads':2,"
                + "'num_hidden_layers':2,'patch_size':16,'hidden_activation':'gelu_pytorch_tanh'},"
                + "'audio_config':{'hidden_size':8,'num_attention_heads':2,'num_hidden_layers':2,"
                + "'input_feat_size':4,'layer_norm_eps':1e-06}}", Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("gemma4v", get(m, "clip.vision.projector_type"));
        assertEquals("gemma4a", get(m, "clip.audio.projector_type"));
        assertEquals(224, u32(m, "clip.vision.image_size"));
        assertEquals(4, u32(m, "clip.audio.num_mel_bins"));

        HfQwenHooksTest.RecordingSink sink = tensors(ctx, "model.audio_tower.layers.0.self_attn.per_dim_scale", 1);
        assertNotNull("softplus of the per-dim scale (gemma.py:1030-1034)", sink.tensors.get("a.blk.0.per_dim_scale.weight"));
    }

    @Test
    public void gemma4UnifiedReordersThePatchColumns() throws IOException {
        Path dir = tmp.newFolder().toPath();
        file(dir, "preprocessor_config.json", COLORS);
        HfLoadContext ctx = ctx(dir, "Gemma4UnifiedForConditionalGeneration", "'hidden_size':" + TEXT + ","
                + "'vision_config':{'hidden_size':8,'mm_embed_dim':8,'patch_size':2,'pooling_kernel_size':2,"
                + "'num_hidden_layers':0,'num_attention_heads':0,'layer_norm_eps':1e-06},"
                + "'audio_config':{'audio_embed_dim':8,'hidden_size':8,'input_feat_size':4,"
                + "'num_hidden_layers':0,'num_attention_heads':0,'layer_norm_eps':1e-06}}",
                Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("gemma4uv", get(m, "clip.vision.projector_type"));
        assertEquals("gemma4ua", get(m, "clip.audio.projector_type"));
        assertEquals(8, u32(m, "clip.vision.embedding_length"));
        assertNotNull(tensors(ctx, "model.vision_embedder.pos_norm.weight", 8).tensors.get("v.patch_norm.3.weight"));
    }

    // ------------------------------------------------------------------ MiMo / D1-omni / LFM3-omni

    @Test
    public void mimoWritesItsRadiovisionAndDotsAudioKeys() throws IOException {
        Path dir = tmp.newFolder().toPath();
        file(dir, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5],"
                + "'min_pixels':256,'max_pixels':4096}");
        HfLoadContext ctx = ctx(dir, "MiMoV2ForCausalLM", "'hidden_size':" + TEXT + ","
                + "'vision_config':{'hidden_size':16,'intermediate_size':32,'num_heads':2,'depth':2,"
                + "'num_key_value_heads':1,'patch_size':2,'spatial_merge_size':2,'rms_norm_eps':1e-06,'visual_token_window_size':8,'vit_window_attn_types':[-1,-1]},"
                + "'audio_config':{'audio_channels':4},"
                + "'audio_tokenizer':{'d_model':8,'encoder_ffn_dim':16,'encoder_attention_heads':2,"
                + "'encoder_layers':2,'n_mels':4,'layer_norm_eps':1e-05,'hybrid_attention':false,"
                + "'swa_per_block':1,'encoder_attn_window_size':[8]}}", Collections.emptyMap());
        // the side tokenizer configuration is what the audio geometry comes from
        file(dir, "audio_tokenizer/config.json", "{'d_model':8,'encoder_ffn_dim':16,'encoder_attention_heads':2,"
                + "'encoder_layers':2,'n_mels':4,'layer_norm_eps':1e-05,'hybrid_attention':false,"
                + "'swa_per_block':1,'encoder_attn_window_size':[8]}");
        assertEquals(128, ctx.blockCount());
        assertEquals("visual.patch_embed.proj.weight",
                HfMimoNemotronHooks.MIMO_FILTER.filter("visual.patch_embed.proj.weight",
                        UnaryOperator.identity()));
        assertEquals(null, HfMimoNemotronHooks.MIMO_FILTER.filter("model.layers.0.mlp.up_proj.weight",
                UnaryOperator.identity()));

        assertNotNull(tensors(ctx, "audio_encoder.audio_adapter.proj.1.weight", 8, 8)
                .tensors.get("mm.a.mlp.1.weight"));
        assertNotNull(tensors(ctx, "audio_encoder.input_local_transformer.layers.0.self_attn.q_proj.weight", 8, 8)
                .tensors.get("mm.a.local_blk.0.attn_q.weight"));
        assertNotNull(tensors(ctx, "audio_encoder.projection.mlp.0.weight", 16, 8)
                .tensors.get("mm.a.mlp.1.weight"));
        assertNotNull(tensors(ctx, "audio_encoder.projection.mlp.2.weight", 8, 16)
                .tensors.get("mm.a.mlp.2.weight"));
    }

    @Test
    public void d1OmniWritesBothConformerAndVisionKeys() throws IOException {
        Path dir = tmp.newFolder().toPath();
        file(dir, "preprocessor_config.json", COLORS);
        HfLoadContext ctx = ctx(dir, "D1OmniModel", "'hidden_size':" + TEXT + ","
                + "'vision_config':{'hidden_size':16,'intermediate_size':32,'num_attention_heads':2,"
                + "'num_hidden_layers':2,'patch_size':2,'layer_norm_eps':1e-06},"
                + "'audio_config':{'d_model':8,'ff_expansion_factor':2,'n_heads':2,'num_hidden_layers':2,"
                + "'feat_in':4}}", Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("d1omni_v", get(m, "clip.vision.projector_type"));
        assertEquals("d1omni_a", get(m, "clip.audio.projector_type"));
        assertEquals(128, ctx.blockCount());

        assertEquals("conformer.layers.0.conv.batch_norm.weight",
                HfAudioTowerHooks.D1OMNI_FILTER.filter("audio.encoder.layers.0.conv.batch_norm.weight",
                        UnaryOperator.identity()));
        assertEquals("audio_adapter.model.5.weight",
                HfAudioTowerHooks.D1OMNI_FILTER.filter("audio.residual.down.weight", UnaryOperator.identity()));
    }

    @Test
    public void lfm3OmniWritesTheVisionAndAudioKeys() throws IOException {
        Path dir = tmp.newFolder().toPath();
        file(dir, "preprocessor_config.json", lfm3OmniPreprocessor(null));
        HfLoadContext ctx = ctx(dir, "Lfm3OmniForConditionalGeneration", lfm3OmniBody(), Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("lfm3vl", get(m, "clip.vision.projector_type"));
        assertEquals("lfm3a", get(m, "clip.audio.projector_type"));
        assertEquals(4, u32(m, "clip.audio.num_mel_bins"));
        assertEquals(8, u32(m, "clip.audio.window_size"));
        assertEquals(128, ctx.blockCount());
    }

    @Test
    public void lfm3OmniAcceptsItsOwnVideoFrameBound() throws IOException {
        Path dir = tmp.newFolder().toPath();
        file(dir, "preprocessor_config.json", lfm3OmniPreprocessor(LFM3_OMNI_VIDEO));
        HfMetadata m = meta(ctx(dir, "Lfm3OmniForConditionalGeneration", lfm3OmniBody(),
                Collections.emptyMap()));
        assertEquals("lfm3vl", get(m, "clip.vision.projector_type"));
    }

    @Test
    public void lfm3OmniRefusesTheVisionOnlyVideoFrameBound() throws IOException {
        Path dir = tmp.newFolder().toPath();
        file(dir, "preprocessor_config.json", lfm3OmniPreprocessor(LFM3_VL_VIDEO));
        com.eonmux.neuralfree.gguf.hparams.GgufFormatException e = assertThrows(
                com.eonmux.neuralfree.gguf.hparams.GgufFormatException.class,
                () -> meta(ctx(dir, "Lfm3OmniForConditionalGeneration", lfm3OmniBody(),
                        Collections.emptyMap())));
        assertTrue(e.getMessage(), e.getMessage().contains("max_frames"));
    }

    /** The video processor the Omni class checks against its own 384-frame bound (lfm3.py:581). */
    private static final String LFM3_OMNI_VIDEO = "{'fps':2.0,'min_frames':4,'temporal_patch_size':2,"
            + "'size':{'shortest_edge':4096,'longest_edge':25165824},'max_frames':384}";
    /** The video processor of the vision-only class, which the Omni checkpoint must refuse (lfm3.py:209). */
    private static final String LFM3_VL_VIDEO = "{'fps':2.0,'min_frames':4,'temporal_patch_size':2,"
            + "'size':{'shortest_edge':4096,'longest_edge':25165824},'max_frames':768}";

    private static String lfm3OmniPreprocessor(String video) {
        return "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5],"
                + "'size':{'shortest_edge':4096,'longest_edge':25165824},'resample':2,"
                + (video == null ? "" : "'video_processor':" + video + ",")
                + "'feature_extractor':{'feature_extractor_type':'Lfm3OmniFeatureExtractor','sampling_rate':16000,"
                + "'feature_size':128,'n_fft':400,'win_length':400,'hop_length':160,'log_floor':-12.0,"
                + "'normalization_offset':4.0,'normalization_divisor':4.0,'encoder_stride':8}}";
    }

    private static String lfm3OmniBody() {
        return "'hidden_size':" + TEXT + ","
                + "'projector_hidden_act':'gelu','projector_bias':true,'projector_layer_norm_eps':1e-06,"
                + "'downsample_factor':2,"
                + "'vision_config':{'hidden_size':16,'intermediate_size':32,'num_attention_heads':2,"
                + "'num_hidden_layers':2,'num_key_value_heads':1,'patch_size':2,'in_channels':3,"
                + "'conv_kernel_size':3,'norm_eps':1e-06,"
                + "'rope_parameters':{'rope_theta':1e6}},"
                + "'audio_encoder_config':{'num_mel_bins':4,'sliding_window':8,"
                + "'text_config':{'hidden_size':8,'intermediate_size':16,'num_attention_heads':2,"
                + "'num_key_value_heads':1,'num_hidden_layers':2,'norm_eps':1e-06,"
                + "'block_auto_adjust_ff_dim':false,'block_ffn_dim_multiplier':null,'block_multiple_of':1,"
                + "'rope_parameters':{'rope_theta':1e6}}}}";
    }

    // ------------------------------------------------------------------ Nemotron Omni

    @Test
    public void nemotronOmniReadsTheRadiovisionConfigAndLeavesAudioUnadvertised() throws IOException {
        Path dir = tmp.newFolder().toPath();
        file(dir, "preprocessor_config.json", "{'norm_mean':[0.485,0.456,0.406],"
                + "'norm_std':[0.229,0.224,0.225],'min_num_patches':16,'max_num_patches':256}");
        HfLoadContext ctx = ctx(dir, "NemotronH_Omni_Reasoning_V3", "'hidden_size':" + TEXT + ","
                + "'downsample_ratio':0.5,"
                + "'vision_config':{'hidden_size':1280,'mlp_ratio':4,'num_hidden_layers':2,"
                + "'num_attention_heads':16,'patch_size':16},"
                + "'sound_config':{'num_mel_bins':4,'subsampling_factor':8,'conv_kernel_size':3}}",
                Collections.emptyMap());
        HfMetadata m = meta(ctx);
        assertEquals("nemotron_omni", get(m, "clip.projector_type"));
        assertEquals(5120, u32(m, "clip.vision.feed_forward_length"));
        assertEquals(2, u32(m, "clip.vision.block_count"));
        assertEquals(2, u32(m, "clip.vision.projector.scale_factor"));
        assertFalse("the converter's Omni class advertises the vision encoder only",
                m.has("clip.has_audio_encoder"));

        assertEquals("vision_model.embeddings.patch_projection.weight",
                HfMimoNemotronHooks.NEMOTRON_FILTER.filter("vision_model.embeddings.patch_projection.weight",
                        UnaryOperator.identity()));
        assertEquals("vision_model.radio_model.model.patch_generator.pos_embed.weight",
                HfMimoNemotronHooks.NEMOTRON_FILTER.filter("vision_model.radio_model.model.patch_generator.pos_embed",
                        UnaryOperator.identity()));

        HfQwenHooksTest.RecordingSink sink = tensors(ctx,
                "vision_model.embeddings.patch_projection.weight", 1280, 16 * 16 * 3);
        assertEquals(Arrays.asList(1280L, 3L, 16L, 16L), boxed(sink.tensors.get("v.patch_embd.weight").shape()));

        HfQwenHooksTest.RecordingSink sound = tensors(ctx,
                "sound_encoder.encoder.layers.0.conv.depthwise_conv.weight", 8, 1, 3);
        assertEquals(Arrays.asList(1L, 1L, 8L, 3L), boxed(sound.tensors.get("a.blk.0.conv_dw.weight").shape()));
    }

    // ------------------------------------------------------------------ Pixtral params.json

    @Test
    public void mistralNativeParamsJsonWritesThePixtralProjector() throws IOException {
        Path dir = tmp.newFolder().toPath();
        file(dir, "params.json", "{'dim':8,'norm_eps':1e-05,'vision_encoder':{'hidden_size':16,"
                + "'intermediate_size':32,'num_attention_heads':2,'n_layers':2,'patch_size':2,'image_size':64,"
                + "'rope_theta':10000,'mm_projector_id':'patch_merge','spatial_merge_size':2}}");
        file(dir, "tekken.json", "{'special_tokens':[{'token_str':'[IMG_BREAK]','rank':7}]}");
        HfConfig cfg = HfConfigLoader.load(dir, Collections.emptyMap());
        HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.MMPROJ);
        HfLoadContext ctx = HfLoadContext.builder(cfg, entry, Arch.CLIP)
                .tokenizerFiles(HfTokenizerFiles.locate(dir)).build();
        HfMetadata m = meta(ctx);
        assertEquals("pixtral", get(m, "clip.projector_type"));
        assertTrue((Boolean) get(m, "clip.use_silu"));
        assertEquals(2, u32(m, "clip.vision.spatial_merge_size"));

        HfQwenHooksTest.RecordingSink sink = tensors(ctx, "vision_language_adapter.w_in.weight", 16, 16);
        assertNotNull(sink.tensors.get("mm.1.weight"));
        HfQwenHooksTest.RecordingSink out = tensors(ctx, "vision_language_adapter.w_out.bias", 8);
        assertNotNull(out.tensors.get("mm.2.bias"));
        assertNotNull(tensors(ctx, "vision_encoder.transformer.layers.0.attention.wq.weight", 16, 16).tensors.get("v.blk.0.attn_q.weight"));
    }

    // ------------------------------------------------------------------ Soprano

    @Test
    public void sopranoRefusesACheckpointWithoutItsDecoder() throws IOException {
        HfLoadContext ctx = ctx("SopranoModel", "'hidden_size':" + TEXT, Collections.emptyMap());
        com.eonmux.neuralfree.gguf.hparams.GgufFormatException e = assertThrows(
                com.eonmux.neuralfree.gguf.hparams.GgufFormatException.class, () -> meta(ctx));
        assertTrue(e.getMessage(), e.getMessage().contains("decoder.pth"));
        assertEquals("the main checkpoint contributes no tensor",
                null, HfSopranoHooks.FILTER.filter("model.layers.0.self_attn.q_proj.weight",
                        UnaryOperator.identity()));
    }


    // ------------------------------------------------------------------ export round trip

    @Test
    public void anExportedQwen3AsrFileOpensAsAQwen3aProjector() throws Exception {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen3ASRForConditionalGeneration'],'model_type':'qwen3_asr','hidden_size':8,"
                + "'audio_config':{'hidden_size':8,'d_model':8,'intermediate_size':16,'encoder_ffn_dim':16,"
                + "'num_attention_heads':2,'encoder_attention_heads':2,'num_mel_bins':4,"
                + "'num_hidden_layers':4,'layer_norm_eps':1e-05}}"));
        b.f32("audio_tower.conv2d1.weight", 8, 4, 3, 3);
        b.f32("audio_tower.conv2d1.bias", 8);
        b.f32("audio_tower.conv2d2.weight", 8, 8, 3, 3);
        b.f32("audio_tower.conv2d2.bias", 8);
        b.f32("audio_tower.conv2d3.weight", 8, 8, 3, 3);
        b.f32("audio_tower.conv2d3.bias", 8);
        b.f32("audio_tower.conv_out.weight", 8, 8);
        Path dir = b.write(tmp.newFolder("qwen3asr").toPath());
        Path file = tmp.newFolder("qwen3asr-out").toPath().resolve("mmproj-qwen3asr.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("qwen3asr-cache").toPath()).build();
        try (HfModelSource s = HfModelSource.openProjector(dir, options)) {
            GgufWriter.write(s, file, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        com.eonmux.neuralfree.gguf.ModelSource written =
                com.eonmux.neuralfree.gguf.ModelSources.parseGguf(file.toString());
        try (com.eonmux.neuralfree.gguf.ModelSource source = written) {
            com.eonmux.neuralfree.gguf.hparams.GgufKv kv =
                    com.eonmux.neuralfree.gguf.hparams.GgufKv.of(source);
            assertEquals("clip", kv.archName());
            assertTrue(kv.bool("clip.has_audio_encoder", false));
            assertFalse(kv.bool("clip.has_vision_encoder", false));
            assertEquals("qwen3a", kv.str("clip.projector_type", null));
            assertEquals(4, kv.u32("clip.audio.block_count", 0));
        }
    }

    /** Boxes a shape for {@code assertEquals}. */
    private static List<Long> boxed(long[] shape) {
        Long[] v = new Long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            v[i] = shape[i];
        }
        return Arrays.asList(v);
    }
}
