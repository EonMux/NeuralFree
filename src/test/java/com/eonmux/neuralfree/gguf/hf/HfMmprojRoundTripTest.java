package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;

import com.eonmux.neuralfree.gguf.mm.MmprojModel;
import com.eonmux.neuralfree.gguf.mm.ProjectorModality;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The {@code --mmproj} export round trip: a Hugging Face projector view written through {@link GgufWriter} is an
 * mmproj GGUF the serving side opens ({@code MmprojModel.open}), with the {@code clip.*} hyperparameters the
 * converters write, the encoder and projector tensors the family tables require, and the weights the source
 * planned; a vision encoder binds over them. This is the end of the path {@code nftool export --mmproj} runs.
 */
public class HfMmprojRoundTripTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private ModelSourceOptions options() throws IOException {
        return ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();
    }

    /** Exports the projector view of {@code dir} and opens it through the serving side. */
    private MmprojModel export(Path dir, String name) throws IOException {
        Path out = tmp.getRoot().toPath().resolve(name + "-mmproj.gguf");
        ModelSourceOptions options = options();
        try (HfModelSource s = HfModelSource.openProjector(dir, options)) {
            GgufWriter.write(s, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        return MmprojModel.open(out.toString(), options);
    }

    private static void file(SyntheticHf.Builder b, String name, String json) {
        b.file(name, SyntheticHf.json(json));
    }

    private static final String COLORS = "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5]}";

    // ------------------------------------------------------------------ fixtures

    private Path qwen2vl() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen2VLForConditionalGeneration'],'model_type':'qwen2_vl','hidden_size':8,"
                + "'rms_norm_eps':1e-06,'vision_config':{'depth':3,'num_heads':2,'embed_dim':8,'hidden_size':16,"
                + "'patch_size':2,'spatial_merge_size':2}}"));
        b.f32("visual.patch_embed.proj.weight", 8, 3, 2, 2, 2);
        b.f32("visual.patch_embed.proj.bias", 8);
        for (int n = 0; n < 3; n++) {
            b.f32("visual.blocks." + n + ".norm1.weight", 8);
            b.f32("visual.blocks." + n + ".norm2.weight", 8);
            b.f32("visual.blocks." + n + ".attn.qkv.weight", 24, 8);
            b.f32("visual.blocks." + n + ".attn.qkv.bias", 24);
            b.f32("visual.blocks." + n + ".attn.proj.weight", 8, 8);
            b.f32("visual.blocks." + n + ".attn.proj.bias", 8);
            b.f32("visual.blocks." + n + ".mlp.fc1.weight", 16, 8);
            b.f32("visual.blocks." + n + ".mlp.fc1.bias", 16);
            b.f32("visual.blocks." + n + ".mlp.fc2.weight", 8, 16);
            b.f32("visual.blocks." + n + ".mlp.fc2.bias", 8);
        }
        b.f32("visual.merger.ln_q.weight", 8);
        b.f32("visual.merger.ln_q.bias", 8);
        b.f32("visual.merger.mlp.0.weight", 16, 32);
        b.f32("visual.merger.mlp.0.bias", 16);
        b.f32("visual.merger.mlp.2.weight", 8, 16);
        b.f32("visual.merger.mlp.2.bias", 8);
        file(b, "preprocessor_config.json", COLORS);
        return b.write(tmp.newFolder("rt-qwen2vl").toPath());
    }

    private Path gemma3() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Gemma3ForConditionalGeneration'],'model_type':'gemma3','hidden_size':8,'vision_config':"
                + "{'model_type':'siglip_vision_model','hidden_size':8,'intermediate_size':16,"
                + "'num_attention_heads':2,'image_size':64,'patch_size':4,'layer_norm_eps':1e-06,"
                + "'num_hidden_layers':2,'hidden_act':'gelu_pytorch_tanh'}}"));
        b.f32("vision_tower.vision_model.embeddings.patch_embedding.weight", 8, 3, 4, 4);
        b.f32("vision_tower.vision_model.embeddings.position_embedding.weight", 256, 8);
        b.f32("vision_tower.vision_model.post_layernorm.weight", 8);
        b.f32("vision_tower.vision_model.post_layernorm.bias", 8);
        for (int n = 0; n < 2; n++) {
            b.f32("vision_tower.vision_model.encoder.layers." + n + ".layer_norm1.weight", 8);
            b.f32("vision_tower.vision_model.encoder.layers." + n + ".layer_norm1.bias", 8);
            b.f32("vision_tower.vision_model.encoder.layers." + n + ".layer_norm2.weight", 8);
            b.f32("vision_tower.vision_model.encoder.layers." + n + ".layer_norm2.bias", 8);
            b.f32("vision_tower.vision_model.encoder.layers." + n + ".self_attn.q_proj.weight", 8, 8);
            b.f32("vision_tower.vision_model.encoder.layers." + n + ".self_attn.k_proj.weight", 8, 8);
            b.f32("vision_tower.vision_model.encoder.layers." + n + ".self_attn.v_proj.weight", 8, 8);
            b.f32("vision_tower.vision_model.encoder.layers." + n + ".self_attn.out_proj.weight", 8, 8);
            b.f32("vision_tower.vision_model.encoder.layers." + n + ".mlp.fc1.weight", 16, 8);
            b.f32("vision_tower.vision_model.encoder.layers." + n + ".mlp.fc1.bias", 16);
            b.f32("vision_tower.vision_model.encoder.layers." + n + ".mlp.fc2.weight", 8, 16);
            b.f32("vision_tower.vision_model.encoder.layers." + n + ".mlp.fc2.bias", 8);
        }
        b.f32("multi_modal_projector.mm_soft_emb_norm.weight", 8);
        b.f32("multi_modal_projector.mm_input_projection.weight", 8, 8);
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5],"
                + "'image_seq_length':64}");
        return b.write(tmp.newFolder("rt-gemma3").toPath());
    }

    private Path pixtral() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['LlavaForConditionalGeneration'],'model_type':'llava','hidden_size':8,'spatial_merge_size':2,"
                + "'vision_config':{'model_type':'pixtral','hidden_size':8,'intermediate_size':16,"
                + "'num_attention_heads':2,'image_size':64,'patch_size':4,'hidden_act':'gelu',"
                + "'num_hidden_layers':4}}"));
        b.f32("vision_tower.patch_conv.weight", 8, 3, 4, 4);
        for (int n = 0; n < 4; n++) {
            b.f32("vision_tower.transformer.layers." + n + ".attention.q_proj.weight", 8, 8);
            b.f32("vision_tower.transformer.layers." + n + ".attention.k_proj.weight", 8, 8);
            b.f32("vision_tower.transformer.layers." + n + ".attention.v_proj.weight", 8, 8);
            b.f32("vision_tower.transformer.layers." + n + ".attention.o_proj.weight", 8, 8);
            b.f32("vision_tower.transformer.layers." + n + ".attention_norm.weight", 8);
            b.f32("vision_tower.transformer.layers." + n + ".ffn_norm.weight", 8);
            b.f32("vision_tower.transformer.layers." + n + ".feed_forward.up_proj.weight", 16, 8);
            b.f32("vision_tower.transformer.layers." + n + ".feed_forward.down_proj.weight", 8, 16);
        }
        b.f32("multi_modal_projector.linear_1.weight", 8, 8);
        b.f32("multi_modal_projector.linear_2.weight", 8, 8);
        b.f32("multi_modal_projector.patch_merger.merging_layer.weight", 8, 32);
        b.f32("pre_mm_projector_norm.weight", 8);
        b.f32("model.embed_tokens.weight", 16, 8);
        file(b, "tokenizer_config.json", "{'added_tokens_decoder':{'7':{'content':'[IMG_BREAK]',"
                + "'special':true}}}");
        file(b, "preprocessor_config.json", COLORS);
        return b.write(tmp.newFolder("rt-pixtral").toPath());
    }

    private Path qwen3vl() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen3VLForConditionalGeneration'],'model_type':'qwen3_vl','text_config':{'hidden_size':8,"
                + "'rms_norm_eps':1e-06},'vision_config':{'depth':2,'num_heads':2,'hidden_size':8,"
                + "'intermediate_size':16,'patch_size':2,'spatial_merge_size':2,'num_position_embeddings':16,"
                + "'deepstack_visual_indexes':[1]}}"));
        b.f32("model.visual.patch_embed.proj.weight", 8, 3, 2, 2, 2);
        b.f32("model.visual.patch_embed.proj.bias", 8);
        b.f32("visual.pos_embed.weight", 16, 8);
        for (int n = 0; n < 2; n++) {
            b.f32("visual.blocks." + n + ".norm1.weight", 8);
            b.f32("visual.blocks." + n + ".norm2.weight", 8);
            b.f32("visual.blocks." + n + ".attn.qkv.weight", 24, 8);
            b.f32("visual.blocks." + n + ".attn.qkv.bias", 24);
            b.f32("visual.blocks." + n + ".attn.proj.weight", 8, 8);
            b.f32("visual.blocks." + n + ".mlp.fc1.weight", 16, 8);
            b.f32("visual.blocks." + n + ".mlp.fc1.bias", 16);
            b.f32("visual.blocks." + n + ".mlp.fc2.weight", 8, 16);
            b.f32("visual.blocks." + n + ".mlp.fc2.bias", 8);
        }
        b.f32("visual.merger.norm.weight", 8);
        b.f32("visual.merger.linear_fc1.weight", 16, 32);
        b.f32("visual.merger.linear_fc1.bias", 16);
        b.f32("visual.merger.linear_fc2.weight", 8, 16);
        b.f32("visual.merger.linear_fc2.bias", 8);
        b.f32("model.visual.deepstack_merger_list.0.norm.weight", 8);
        b.f32("model.visual.deepstack_merger_list.0.linear_fc1.weight", 16, 8);
        b.f32("model.visual.deepstack_merger_list.0.linear_fc2.weight", 8, 16);
        file(b, "preprocessor_config.json", COLORS);
        return b.write(tmp.newFolder("rt-qwen3vl").toPath());
    }

    private Path qwen2audio() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen2AudioForConditionalGeneration'],'model_type':'qwen2_audio','hidden_size':8,"
                + "'audio_config':{'hidden_size':8,'d_model':8,'encoder_ffn_dim':16,'encoder_attention_heads':2,"
                + "'num_mel_bins':4,'num_hidden_layers':3,'layer_norm_eps':1e-05}}"));
        b.f32("audio_tower.conv1.weight", 8, 4, 3);
        b.f32("audio_tower.conv1.bias", 8);
        b.f32("audio_tower.conv2.weight", 8, 8, 3);
        b.f32("audio_tower.conv2.bias", 8);
        b.f32("audio_tower.embed_positions.weight", 300, 8);
        for (int n = 0; n < 3; n++) {
            b.f32("audio_tower.layers." + n + ".self_attn.q_proj.weight", 8, 8);
            b.f32("audio_tower.layers." + n + ".self_attn.q_proj.bias", 8);
            b.f32("audio_tower.layers." + n + ".self_attn.k_proj.weight", 8, 8);
            b.f32("audio_tower.layers." + n + ".self_attn.v_proj.weight", 8, 8);
            b.f32("audio_tower.layers." + n + ".self_attn.v_proj.bias", 8);
            b.f32("audio_tower.layers." + n + ".self_attn.out_proj.weight", 8, 8);
            b.f32("audio_tower.layers." + n + ".self_attn_layer_norm.weight", 8);
            b.f32("audio_tower.layers." + n + ".self_attn_layer_norm.bias", 8);
            b.f32("audio_tower.layers." + n + ".final_layer_norm.weight", 8);
            b.f32("audio_tower.layers." + n + ".final_layer_norm.bias", 8);
            b.f32("audio_tower.layers." + n + ".fc1.weight", 16, 8);
            b.f32("audio_tower.layers." + n + ".fc1.bias", 16);
            b.f32("audio_tower.layers." + n + ".fc2.weight", 8, 16);
            b.f32("audio_tower.layers." + n + ".fc2.bias", 8);
        }
        b.f32("audio_tower.layer_norm.weight", 8);
        b.f32("multi_modal_projector.linear.weight", 8, 8);
        b.f32("multi_modal_projector.linear.bias", 8);
        return b.write(tmp.newFolder("rt-qwen2audio").toPath());
    }

    // ------------------------------------------------------------------ round trips

    @Test
    public void theQwen2vlExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(qwen2vl(), "qwen2vl")) {
            assertTrue(mm.caps().toString(), mm.caps().inpVision());
            assertFalse(mm.caps().inpAudio());
            assertTrue(mm.supportsVision());
            var hp = mm.visionHparams();
            assertEquals("qwen2vl_merger", hp.projectorType().typeName());
            assertEquals(8, hp.projectionDim());
            assertEquals(8, hp.nEmbd());
            assertEquals(16, hp.nFf());
            assertEquals(3, hp.nLayer());
            assertEquals(2, hp.nHead());
            assertEquals(2, hp.nMerge());
            assertEquals(1e-6f, hp.eps(), 1e-9f);
            assertTrue("the mm.0 / mm.2 merger loads", mm.tensorNames().contains("mm.0.weight"));
            assertTrue(mm.tensorNames().contains("v.patch_embd.weight.1"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theGemma3ExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(gemma3(), "gemma3")) {
            var hp = mm.visionHparams();
            assertEquals("gemma3", hp.projectorType().typeName());
            assertEquals(64, hp.imageSize());
            assertEquals(4, hp.patchSize());
            assertEquals(2, hp.nMerge());
            assertArrayEquals(new float[]{0.5f, 0.5f, 0.5f}, hp.imageMean(), 1e-6f);
            assertArrayEquals(new float[]{0.5f, 0.5f, 0.5f}, hp.imageStd(), 1e-6f);
            assertTrue(mm.tensorNames().contains("mm.input_projection.weight"));
            assertTrue(mm.tensorNames().contains("mm.soft_emb_norm.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void thePixtralExportCarriesItsImageBreakRow() throws IOException {
        try (MmprojModel mm = export(pixtral(), "pixtral")) {
            assertEquals("pixtral", mm.visionHparams().projectorType().typeName());
            assertTrue(mm.tensorNames().contains("v.token_embd.img_break"));
            assertTrue(mm.tensorNames().contains("mm.1.weight"));
            assertTrue(mm.tensorNames().contains("mm.2.weight"));
            assertTrue(mm.tensorNames().contains("mm.patch_merger.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theQwen3vlExportKeepsItsDeepstackTaps() throws IOException {
        try (MmprojModel mm = export(qwen3vl(), "qwen3vl")) {
            var hp = mm.visionHparams();
            assertEquals("qwen3vl_merger", hp.projectorType().typeName());
            assertEquals("the image size computed from the position grid (qwen3vl.py:55)", 8, hp.imageSize());
            assertTrue(mm.tensorNames().contains("v.deepstack.1.norm.weight"));
            assertTrue(mm.tensorNames().contains("mm.0.weight"));
        }
    }

    @Test
    public void theQwen2AudioExportOpensItsAudioHparams() throws IOException {
        try (MmprojModel mm = export(qwen2audio(), "qwen2audio")) {
            assertFalse(mm.caps().inpVision());
            assertTrue(mm.caps().inpAudio());
            assertTrue(mm.supportsAudio());
            var hp = mm.audioHparams();
            assertEquals("qwen2a", hp.projectorType().typeName());
            assertEquals(8, hp.projectionDim());
            assertEquals(8, hp.nEmbd());
            assertEquals(3, hp.nLayer());
            assertEquals(4, hp.audio().nMelBins());
            assertTrue(mm.tensorNames().contains("mm.a.fc.weight"));
            assertTrue(mm.tensorNames().contains("a.conv1d.1.bias"));
        }
    }
}
