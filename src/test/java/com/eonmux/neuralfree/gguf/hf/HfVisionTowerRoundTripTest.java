package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.mm.MmprojModel;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The {@code --mmproj} export round trip of the vision-tower classes ported from llama.cpp's remaining
 * {@code MmprojModel} subclasses: a synthetic Hugging Face checkpoint is exported through {@link GgufWriter} and the
 * serving side ({@link MmprojModel#open}) must declare the family runnable, parse its hparams and bind every tensor
 * the family's graphs read. Each fixture carries the complete tower and projector the {@code MmprojTensorTable}
 * requires, so a missing conversion rewrite fails the open.
 */
public class HfVisionTowerRoundTripTest {

    private static final int DIM = 8;
    private static final int FF = 16;
    private static final int HEADS = 2;
    private static final int BLOCKS = 4;

    private static final String COLORS = "'image_mean':[0.5,0.5,0.5],'image_std':[0.25,0.5,0.75]";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private ModelSourceOptions options() throws IOException {
        return ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();
    }

    private MmprojModel export(Path dir, String name) throws IOException {
        Path out = tmp.getRoot().toPath().resolve(name + "-mmproj.gguf");
        ModelSourceOptions options = options();
        try (HfModelSource s = HfModelSource.openProjector(dir, options)) {
            GgufWriter.write(s, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        return MmprojModel.open(out.toString(), options);
    }

    private static void f32(SyntheticHf.Builder b, int seed, String name, int... shape) {
        long[] dims = new long[shape.length];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = shape[i];
        }
        b.tensor(name, StDType.F32, dims, SyntheticHf.f32(SyntheticHf.values(seed, count(shape))));
    }

    private static int count(int... shape) {
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        return n;
    }

    private static void file(SyntheticHf.Builder b, String name, String json) {
        b.file(name, SyntheticHf.json(json));
    }

    /** The Q/K/V attention tensors of a block; {@code heads} is the per-head width. */
    private static void attention(SyntheticHf.Builder b, int seed, String prefix, boolean fused, int out, int in) {
        if (fused) {
            f32(b, seed, prefix + "qkv.weight", out, in);
            return;
        }
        f32(b, seed, prefix + "q_proj.weight", out, in);
        f32(b, seed + 1, prefix + "k_proj.weight", out, in);
        f32(b, seed + 2, prefix + "v_proj.weight", out, in);
    }

    /** The SigLIP block at {@code prefix} (norm1, norm2, qkv, out, fc1, fc2). */
    private static void siglipBlock(SyntheticHf.Builder b, int seed, String prefix, boolean fused) {
        f32(b, seed, prefix + "layer_norm1.weight", DIM);
        f32(b, seed + 1, prefix + "layer_norm1.bias", DIM);
        f32(b, seed + 2, prefix + "layer_norm2.weight", DIM);
        f32(b, seed + 3, prefix + "layer_norm2.bias", DIM);
        attention(b, seed + 4, prefix + "self_attn.", fused, DIM, DIM);
        f32(b, seed + 8, prefix + "self_attn.out_proj.weight", DIM, DIM);
        f32(b, seed + 9, prefix + "self_attn.out_proj.bias", DIM);
        f32(b, seed + 10, prefix + "mlp.fc1.weight", FF, DIM);
        f32(b, seed + 11, prefix + "mlp.fc1.bias", FF);
        f32(b, seed + 12, prefix + "mlp.fc2.weight", DIM, FF);
        f32(b, seed + 13, prefix + "mlp.fc2.bias", DIM);
    }

    /** The SigLIP tower shared by Cohere2, Janus-Pro and LFM2. */
    private static void siglipTower(SyntheticHf.Builder b, String root, boolean fused) {
        f32(b, 1, root + "embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 2, root + "embeddings.position_embedding.weight", 16, DIM);
        f32(b, 3, root + "post_layernorm.weight", DIM);
        f32(b, 4, root + "post_layernorm.bias", DIM);
        for (int n = 0; n < BLOCKS; n++) {
            siglipBlock(b, 100 + n * 20, root + "encoder.layers." + n + ".", fused);
        }
    }

    // ------------------------------------------------------------------ fixtures

    private Path cohere2() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Cohere2VisionForConditionalGeneration'],'hidden_size':" + DIM + ",'downsample_factor':2,"
                + "'vision_config':{'hidden_size':" + DIM + ",'layer_norm_eps':1e-05,'intermediate_size':" + FF
                + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':" + BLOCKS + ",'patch_size':2,"
                + "'image_size':8}}"));
        siglipTower(b, "model.vision_tower.", false);
        f32(b, 301, "model.multi_modal_projector.linear_1.weight", DIM, 4 * DIM);
        f32(b, 311, "model.multi_modal_projector.linear_1.bias", DIM);
        f32(b, 302, "model.multi_modal_projector.linear_2.weight", DIM, 4);
        f32(b, 312, "model.multi_modal_projector.linear_2.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'max_patches':12}");
        return b.write(tmp.newFolder("rt-cohere2").toPath());
    }

    private Path janus() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['JanusForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':"
                + DIM + ",'mlp_ratio':2.0,'num_attention_heads':" + HEADS + ",'num_hidden_layers':" + BLOCKS
                + ",'patch_size':2,'image_size':8,'layer_norm_eps':1e-06,'hidden_act':'gelu'}}"));
        siglipTower(b, "model.vision_model.", false);
        f32(b, 400, "model.aligner.fc1.weight", DIM, DIM);
        f32(b, 401, "model.aligner.fc1.bias", DIM);
        f32(b, 402, "model.aligner.hidden_layers.0.weight", DIM, DIM);
        f32(b, 403, "model.aligner.hidden_layers.0.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-janus").toPath());
    }

    private Path lfm2() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Lfm2VlForConditionalGeneration'],'hidden_size':" + DIM + ",'downsample_factor':2,"
                + "'vision_feature_layer':-1,'vision_config':{'image_size':256,'patch_size':16,'hidden_size':"
                + DIM + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'layer_norm_eps':1e-06}}"));
        f32(b, 1, "model.vision_tower.vision_model.embeddings.patch_embedding.weight", DIM, 3 * 16 * 16);
        f32(b, 2, "model.vision_tower.vision_model.embeddings.position_embedding.weight", 256, DIM);
        f32(b, 3, "model.vision_tower.vision_model.post_layernorm.weight", DIM);
        f32(b, 4, "model.vision_tower.vision_model.post_layernorm.bias", DIM);
        for (int n = 0; n < BLOCKS; n++) {
            siglipBlock(b, 100 + n * 20, "model.vision_tower.vision_model.encoder.layers." + n + ".", false);
        }
        f32(b, 501, "model.multi_modal_projector.linear_1.weight", DIM, 4 * DIM);
        f32(b, 511, "model.multi_modal_projector.linear_1.bias", DIM);
        f32(b, 502, "model.multi_modal_projector.linear_2.weight", DIM, DIM);
        f32(b, 512, "model.multi_modal_projector.linear_2.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'resample':2,'do_image_splitting':false}");
        return b.write(tmp.newFolder("rt-lfm2").toPath());
    }

    private Path phi4() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Phi4ForCausalLMV'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':" + DIM
                + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':5,"
                + "'patch_size':4}}"));
        b.f32("model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight", DIM, 3 * 4 * 4);
        b.f32("model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight", 256, DIM);
        for (int n = 0; n < 4; n++) {
            siglipBlock(b, 100 + n * 20, "model.vision_tower.vision_tower.vision_model.encoder.layers." + n + ".",
                    false);
        }
        f32(b, 600, "model.mm_projector.0.weight", DIM, DIM);
        f32(b, 601, "model.mm_projector.0.bias", DIM);
        f32(b, 602, "model.mm_projector.2.weight", DIM, DIM);
        f32(b, 603, "model.mm_projector.2.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'patch_size':4,'min_num_patches':16,"
                + "'max_num_patches':64}");
        return b.write(tmp.newFolder("rt-phi4").toPath());
    }

    /** The Qwen3-VL tower of the pass-through classes (Clef, H2O, Qwen4Exp, PPLX, vLLM-D3). */
    private static void qwen3vlTower(SyntheticHf.Builder b) {
        f32(b, 1, "visual.patch_embed.proj.weight", DIM, 3, 2, 2, 2);
        f32(b, 2, "visual.patch_embed.proj.bias", DIM);
        f32(b, 3, "visual.pos_embed.weight", 16, DIM);
        for (int n = 0; n < BLOCKS; n++) {
            String p = "visual.blocks." + n + ".";
            f32(b, 100 + n * 20, p + "norm1.weight", DIM);
            f32(b, 101 + n * 20, p + "norm2.weight", DIM);
            f32(b, 102 + n * 20, p + "attn.qkv.weight", DIM + 2 * DIM, DIM);
            f32(b, 103 + n * 20, p + "attn.qkv.bias", DIM + 2 * DIM);
            f32(b, 104 + n * 20, p + "attn.proj.weight", DIM, DIM);
            f32(b, 105 + n * 20, p + "attn.proj.bias", DIM);
            f32(b, 106 + n * 20, p + "mlp.fc1.weight", FF, DIM);
            f32(b, 107 + n * 20, p + "mlp.fc1.bias", FF);
            f32(b, 108 + n * 20, p + "mlp.fc2.weight", DIM, FF);
            f32(b, 109 + n * 20, p + "mlp.fc2.bias", DIM);
        }
        f32(b, 300, "visual.merger.norm.weight", DIM);
        f32(b, 301, "visual.merger.linear_fc1.weight", FF, 4 * DIM);
        f32(b, 302, "visual.merger.linear_fc1.bias", FF);
        f32(b, 303, "visual.merger.linear_fc2.weight", DIM, FF);
        f32(b, 304, "visual.merger.linear_fc2.bias", DIM);
    }

    private static String qwen3vlConfig(String architecture) {
        return "{'architectures':['" + architecture + "'],'model_type':'qwen3_vl','text_config':{'hidden_size':"
                + DIM + ",'rms_norm_eps':1e-06},'vision_config':{'depth':" + BLOCKS + ",'num_heads':" + HEADS
                + ",'hidden_size':" + DIM + ",'intermediate_size':" + FF + ",'patch_size':2,"
                + "'spatial_merge_size':2,'num_position_embeddings':16,'deepstack_visual_indexes':[]}}";
    }

    private Path clef() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(qwen3vlConfig("ClefModel")));
        qwen3vlTower(b);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-clef").toPath());
    }

    private Path exaone() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Exaone4_5_ForConditionalGeneration'],'hidden_size':" + DIM + ",'rms_norm_eps':1e-06,"
                + "'vision_config':{'embed_dim':" + DIM + ",'hidden_size':" + FF + ",'num_attention_heads':"
                + HEADS + ",'num_key_value_heads':1,'num_hidden_layers':" + BLOCKS + ",'patch_size':2,"
                + "'image_size':8,'fullatt_block_indexes':[1]}}"));
        f32(b, 1, "model.visual.patch_embed.proj.weight", DIM, 3, 2, 2, 2);
        for (int n = 0; n < BLOCKS; n++) {
            String p = "model.visual.blocks." + n + ".";
            f32(b, 100 + n * 20, p + "norm1.weight", DIM);
            f32(b, 101 + n * 20, p + "norm2.weight", DIM);
            f32(b, 102 + n * 20, p + "attn.qkv.weight", DIM + DIM, DIM);
            f32(b, 103 + n * 20, p + "attn.qkv.bias", DIM + DIM);
            f32(b, 104 + n * 20, p + "attn.proj.weight", DIM, DIM);
            f32(b, 105 + n * 20, p + "attn.proj.bias", DIM);
            f32(b, 106 + n * 20, p + "mlp.fc1.weight", FF, DIM);
            f32(b, 107 + n * 20, p + "mlp.fc1.bias", FF);
            f32(b, 108 + n * 20, p + "mlp.fc2.weight", DIM, FF);
            f32(b, 109 + n * 20, p + "mlp.fc2.bias", DIM);
        }
        f32(b, 299, "model.visual.merger.ln_q.weight", DIM);
        f32(b, 300, "model.visual.merger.mlp.0.weight", FF, 4 * DIM);
        f32(b, 301, "model.visual.merger.mlp.0.bias", FF);
        f32(b, 302, "model.visual.merger.mlp.2.weight", DIM, FF);
        f32(b, 303, "model.visual.merger.mlp.2.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'min_pixels':64,'max_pixels':1024}");
        return b.write(tmp.newFolder("rt-exaone").toPath());
    }

    // ------------------------------------------------------------------ round trips

    @Test
    public void theCohere2ExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(cohere2(), "cohere2")) {
            assertTrue(mm.supportsVision());
            assertEquals("cohere2v", mm.visionHparams().projectorType().typeName());
            assertEquals(8, mm.visionHparams().imageSize());
            assertTrue(mm.tensorNames().contains("mm.1.weight"));
            assertTrue(mm.tensorNames().contains("mm.2.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theJanusProExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(janus(), "janus")) {
            assertTrue(mm.supportsVision());
            assertEquals("janus_pro", mm.visionHparams().projectorType().typeName());
            assertTrue(mm.tensorNames().contains("mm.0.weight"));
            assertTrue(mm.tensorNames().contains("mm.1.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theLfm2ExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(lfm2(), "lfm2")) {
            assertTrue(mm.supportsVision());
            assertEquals("lfm2", mm.visionHparams().projectorType().typeName());
            assertEquals(256, mm.visionHparams().imageSize());
            assertTrue(mm.tensorNames().contains("mm.1.weight"));
            assertTrue(mm.tensorNames().contains("mm.2.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void thePhi4ExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(phi4(), "phi4")) {
            assertTrue(mm.supportsVision());
            assertEquals("phi4", mm.visionHparams().projectorType().typeName());
            assertEquals(4, mm.visionHparams().nLayer());
            assertTrue(mm.tensorNames().contains("mm.0.weight"));
            assertTrue(mm.tensorNames().contains("mm.2.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theQwen3vlDerivedExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(clef(), "clef")) {
            assertTrue(mm.supportsVision());
            assertEquals("qwen3vl_merger", mm.visionHparams().projectorType().typeName());
            assertEquals(8, mm.visionHparams().imageSize());
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theExaone45ExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(exaone(), "exaone")) {
            assertTrue(mm.supportsVision());
            assertEquals("exaone4_5", mm.visionHparams().projectorType().typeName());
            assertEquals(1, mm.visionHparams().nHeadKv());
            assertEquals(2, mm.visionHparams().nWaPattern());
            mm.openCpuVisionEncoder().close();
        }
    }

    // ------------------------------------------------------------------ batch two fixtures

    private Path paddle() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['PaddleOCRVisionModel'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':" + DIM
                + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'rms_norm_eps':1e-05}}"));
        f32(b, 1, "model.vision_model.embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 2, "model.vision_model.embeddings.position_embedding.weight", 16, DIM);
        for (int n = 0; n < BLOCKS; n++) {
            siglipBlock(b, 100 + n * 20, "model.vision_model.encoder.layers." + n + ".", false);
        }
        f32(b, 300, "mlp_AR.pre_norm.weight", DIM);
        f32(b, 301, "mlp_AR.pre_norm.bias", DIM);
        f32(b, 302, "mlp_AR.linear_1.weight", DIM, 4 * DIM);
        f32(b, 303, "mlp_AR.linear_1.bias", DIM);
        f32(b, 304, "mlp_AR.linear_2.weight", DIM, DIM);
        f32(b, 305, "mlp_AR.linear_2.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'min_pixels':64,'max_pixels':1024}");
        return b.write(tmp.newFolder("rt-paddle").toPath());
    }

    private Path youtu() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['YoutuVLForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':"
                + DIM + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'image_size':8,'spatial_merge_size':2,'window_size':4,"
                + "'fullatt_block_indexes':[1],'hidden_act':'gelu'}}"));
        f32(b, 1, "siglip2.vision_model.embeddings.patch_embedding.weight", DIM, 3 * 2 * 2);
        f32(b, 2, "model.vision_tower.embeddings.position_embedding.weight", 16, DIM);
        for (int n = 0; n < BLOCKS; n++) {
            String p = "siglip2.vision_model.encoder.layers." + n + ".self_attn.";
            siglipBlock(b, 100 + n * 20, "siglip2.vision_model.encoder.layers." + n + ".", false);
            f32(b, 140 + n * 20, p + "q_proj.bias", DIM);
            f32(b, 141 + n * 20, p + "k_proj.bias", DIM);
            f32(b, 142 + n * 20, p + "v_proj.bias", DIM);
        }
        f32(b, 300, "multi_modal_projector.norm.weight", DIM);
        f32(b, 301, "multi_modal_projector.linear_0.weight", DIM, 4 * DIM);
        f32(b, 302, "multi_modal_projector.linear_0.bias", DIM);
        f32(b, 303, "multi_modal_projector.linear_2.weight", DIM, DIM);
        f32(b, 304, "multi_modal_projector.linear_2.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-youtu").toPath());
    }

    private Path hunyuan() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['HunYuanVLForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':{'image_size':8,"
                + "'patch_size':2,'hidden_size':" + DIM + ",'intermediate_size':" + FF
                + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':" + BLOCKS
                + ",'rms_norm_eps':1e-05,'spatial_merge_size':2}}"));
        f32(b, 1, "vit.embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 2, "vit.embeddings.position_embedding.weight", 17, DIM);
        for (int n = 0; n < BLOCKS; n++) {
            String p = "vit.layers." + n + ".";
            f32(b, 100 + n * 20, p + "input_layernorm.weight", DIM);
            f32(b, 101 + n * 20, p + "post_attention_layernorm.weight", DIM);
            f32(b, 102 + n * 20, p + "self_attn.q_proj.weight", DIM, DIM);
            f32(b, 103 + n * 20, p + "self_attn.k_proj.weight", DIM, DIM);
            f32(b, 104 + n * 20, p + "self_attn.v_proj.weight", DIM, DIM);
            f32(b, 105 + n * 20, p + "self_attn.o_proj.weight", DIM, DIM);
            f32(b, 106 + n * 20, p + "mlp.dense_h_to_4h.weight", FF, DIM);
            f32(b, 107 + n * 20, p + "mlp.dense_4h_to_h.weight", DIM, FF);
        }
        f32(b, 300, "vit.perceive.before_rms.weight", DIM);
        f32(b, 301, "vit.perceive.proj.0.weight", DIM, DIM, 2, 2);
        f32(b, 302, "vit.perceive.proj.0.bias", DIM);
        f32(b, 303, "vit.perceive.proj.2.weight", DIM, DIM, 1, 1);
        f32(b, 304, "vit.perceive.proj.2.bias", DIM);
        f32(b, 305, "vit.perceive.image_newline", DIM);
        f32(b, 306, "vit.perceive.mlp.weight", DIM, DIM);
        f32(b, 307, "vit.perceive.mlp.bias", DIM);
        f32(b, 308, "vit.perceive.image_begin", DIM);
        f32(b, 309, "vit.perceive.image_end", DIM);
        f32(b, 310, "vit.perceive.after_rms.weight", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'min_pixels':64,'max_pixels':1024}");
        return b.write(tmp.newFolder("rt-hunyuan").toPath());
    }

    private Path senseNova() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['NEOVisionModel'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':" + DIM
                + ",'intermediate_size':" + FF + ",'num_attention_heads':1,'num_hidden_layers':0,'patch_size':2,"
                + "'image_size':8,'downsample_ratio':0.5,'layer_norm_eps':1e-06}}"));
        f32(b, 1, "vision_model.embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 2, "vision_model.embeddings.patch_embedding.bias", DIM);
        f32(b, 3, "vision_model.embeddings.dense_embedding.weight", DIM, DIM, 2, 2);
        f32(b, 4, "vision_model.embeddings.dense_embedding.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-sensenova").toPath());
    }

    private Path dotsOcr() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['DotsOCRForCausalLM'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':" + DIM
                + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'spatial_merge_size':2,'rms_norm_eps':1e-05}}"));
        f32(b, 1, "vision_tower.patch_embed.patchifier.proj.weight", DIM, 3, 2, 2);
        for (int n = 0; n < BLOCKS; n++) {
            String p = "vision_tower.blocks." + n + ".";
            f32(b, 100 + n * 20, p + "norm1.weight", DIM);
            f32(b, 101 + n * 20, p + "norm2.weight", DIM);
            f32(b, 102 + n * 20, p + "attn.qkv.weight", 3 * DIM, DIM);
            f32(b, 103 + n * 20, p + "attn.proj.weight", DIM, DIM);
            f32(b, 104 + n * 20, p + "mlp.fc1.weight", FF, DIM);
            f32(b, 105 + n * 20, p + "mlp.fc2.weight", DIM, FF);
            f32(b, 106 + n * 20, p + "mlp.fc3.weight", FF, DIM);
        }
        f32(b, 300, "vision_tower.post_trunk_norm.weight", DIM);
        f32(b, 301, "vision_tower.merger.ln_q.weight", DIM);
        f32(b, 302, "vision_tower.merger.ln_q.bias", DIM);
        f32(b, 303, "vision_tower.merger.mlp.0.weight", DIM, 4 * DIM);
        f32(b, 304, "vision_tower.merger.mlp.0.bias", DIM);
        f32(b, 305, "vision_tower.merger.mlp.2.weight", DIM, DIM);
        f32(b, 306, "vision_tower.merger.mlp.2.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'min_pixels':64,'max_pixels':1024}");
        return b.write(tmp.newFolder("rt-dotsocr").toPath());
    }

    private Path minicpm() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['MiniCPMV4_6ForConditionalGeneration'],'hidden_size':" + DIM + ",'insert_layer_id':1,"
                + "'vision_config':{'image_size':8,"
                + "'patch_size':2,'hidden_size':" + DIM + ",'intermediate_size':" + FF
                + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':" + BLOCKS
                + ",'layer_norm_eps':1e-06}}"));
        f32(b, 1, "model.vision_tower.embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 2, "model.vision_tower.embeddings.patch_embedding.bias", DIM);
        f32(b, 3, "model.vision_tower.embeddings.position_embedding.weight", 16, DIM);
        for (int n = 0; n < BLOCKS; n++) {
            siglipBlock(b, 100 + n * 20, "model.vision_tower.encoder.layers." + n + ".", false);
        }
        f32(b, 300, "model.merger.mlp.0.pre_norm.weight", 4 * DIM);
        f32(b, 301, "model.merger.mlp.0.linear_1.weight", DIM, 4 * DIM);
        f32(b, 302, "model.merger.mlp.0.linear_2.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'downsample_mode':'4x'}");
        return b.write(tmp.newFolder("rt-minicpm").toPath());
    }

    private Path cogVlm() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['CogVLMForCausalLM'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':" + DIM
                + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'image_size':8,'layer_norm_eps':1e-05}}"));
        f32(b, 1, "model.vision.patch_embedding.proj.weight", DIM, 3, 2, 2);
        f32(b, 2, "model.vision.patch_embedding.cls_embedding", DIM);
        f32(b, 3, "model.vision.patch_embedding.position_embedding.weight", 17, DIM);
        for (int n = 0; n < BLOCKS; n++) {
            String p = "model.vision.transformer.layers." + n + ".";
            f32(b, 100 + n * 20, p + "input_layernorm.weight", DIM);
            f32(b, 101 + n * 20, p + "input_layernorm.bias", DIM);
            f32(b, 102 + n * 20, p + "post_attention_layernorm.weight", DIM);
            f32(b, 103 + n * 20, p + "post_attention_layernorm.bias", DIM);
            f32(b, 104 + n * 20, p + "attention.query_key_value.weight", 3 * DIM, DIM);
            f32(b, 105 + n * 20, p + "attention.query_key_value.bias", 3 * DIM);
            f32(b, 106 + n * 20, p + "attention.dense.weight", DIM, DIM);
            f32(b, 107 + n * 20, p + "attention.dense.bias", DIM);
            f32(b, 108 + n * 20, p + "mlp.fc1.weight", FF, DIM);
            f32(b, 109 + n * 20, p + "mlp.fc1.bias", FF);
            f32(b, 110 + n * 20, p + "mlp.fc2.weight", DIM, FF);
            f32(b, 111 + n * 20, p + "mlp.fc2.bias", DIM);
        }
        f32(b, 300, "model.vision.linear_proj.linear_proj.weight", DIM, DIM);
        f32(b, 301, "model.vision.linear_proj.norm1.weight", DIM);
        f32(b, 302, "model.vision.linear_proj.norm1.bias", DIM);
        f32(b, 303, "model.vision.linear_proj.dense_h_to_4h.weight", DIM, DIM);
        f32(b, 304, "model.vision.linear_proj.dense_4h_to_h.weight", DIM, DIM);
        f32(b, 305, "model.vision.linear_proj.gate_proj.weight", DIM, DIM);
        f32(b, 306, "model.vision.boi", DIM);
        f32(b, 307, "model.vision.eoi", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-cogvlm").toPath());
    }

    @Test
    public void thePaddleOcrExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(paddle(), "paddle")) {
            assertTrue(mm.supportsVision());
            assertEquals("paddleocr", mm.visionHparams().projectorType().typeName());
            assertEquals(32, mm.visionHparams().imageSize());
            assertTrue(mm.tensorNames().contains("mm.input_norm.weight"));
            assertTrue(mm.tensorNames().contains("mm.2.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theYoutuVlExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(youtu(), "youtu")) {
            assertTrue(mm.supportsVision());
            assertEquals("youtuvl", mm.visionHparams().projectorType().typeName());
            assertEquals(4, mm.visionHparams().attnWindowSize());
            assertTrue(mm.tensorNames().contains("mm.input_norm.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theHunyuanVlExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(hunyuan(), "hunyuan")) {
            assertTrue(mm.supportsVision());
            assertEquals("hunyuanvl", mm.visionHparams().projectorType().typeName());
            assertTrue(mm.tensorNames().contains("mm.pre_norm.weight"));
            assertTrue(mm.tensorNames().contains("mm.model.fc.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theSenseNovaExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(senseNova(), "sensenova")) {
            assertTrue(mm.supportsVision());
            assertEquals("sensenova_u1", mm.visionHparams().projectorType().typeName());
            assertEquals(0, mm.visionHparams().nLayer());
            assertTrue(mm.tensorNames().contains("mm.0.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theDotsOcrExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(dotsOcr(), "dotsocr")) {
            assertTrue(mm.supportsVision());
            assertEquals("dots_ocr", mm.visionHparams().projectorType().typeName());
            assertEquals(0, mm.visionHparams().imageSize());
            assertTrue(mm.tensorNames().contains("mm.input_norm.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theMiniCpmExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(minicpm(), "minicpm")) {
            assertTrue(mm.supportsVision());
            assertEquals("minicpmv4_6", mm.visionHparams().projectorType().typeName());
            assertEquals(2, mm.visionHparams().nMerge());
            assertTrue(mm.tensorNames().contains("mm.up.weight"));
            assertTrue(mm.tensorNames().contains("mm.down.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theCogVlmExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(cogVlm(), "cogvlm")) {
            assertTrue(mm.supportsVision());
            assertEquals("cogvlm", mm.visionHparams().projectorType().typeName());
            assertTrue(mm.tensorNames().contains("mm.post_fc_norm.weight"));
            assertTrue(mm.tensorNames().contains("v.boi"));
            mm.openCpuVisionEncoder().close();
        }
    }

    // ------------------------------------------------------------------ batch three fixtures

    private Path kimi() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['KimiK25ForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':"
                + DIM + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'init_pos_emb_height':2,'init_pos_emb_width':2,"
                + "'init_pos_emb_time':4,'merge_kernel_size':[2,2],'projector_ln_eps':1e-05}}"));
        f32(b, 1, "vision_tower.patch_embed.proj.weight", DIM, 3, 2, 2);
        f32(b, 2, "vision_tower.patch_embed.pos_emb.weight", 16, DIM);
        for (int n = 0; n < BLOCKS; n++) {
            String p = "vision_tower.encoder.blocks." + n + ".";
            f32(b, 100 + n * 20, p + "norm0.weight", DIM);
            f32(b, 101 + n * 20, p + "norm1.weight", DIM);
            f32(b, 102 + n * 20, p + "wqkv.weight", 3 * DIM, DIM);
            f32(b, 103 + n * 20, p + "wo.weight", DIM, DIM);
            f32(b, 104 + n * 20, p + "mlp.fc0.weight", FF, DIM);
            f32(b, 105 + n * 20, p + "mlp.fc1.weight", DIM, FF);
        }
        f32(b, 300, "mm_projector.pre_norm.weight", DIM);
        f32(b, 301, "mm_projector.pre_norm.bias", DIM);
        f32(b, 302, "mm_projector.proj.0.weight", DIM, 4 * DIM);
        f32(b, 303, "mm_projector.proj.0.bias", DIM);
        f32(b, 304, "mm_projector.proj.2.weight", DIM, DIM);
        f32(b, 305, "mm_projector.proj.2.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-kimi").toPath());
    }

    private Path minimax() throws IOException {
        int e = 80;
        int f = 160;
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['MiniMaxM3SparseForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':"
                + "{'hidden_size':" + e + ",'intermediate_size':" + f + ",'num_attention_heads':1"
                + ",'num_hidden_layers':" + BLOCKS + ",'patch_size':2,'image_size':8,'layer_norm_eps':1e-05,"
                + "'img_token_compression_config':{'spatial_merge_size':2}}}"));
        f32(b, 1, "vision_tower.vision_model.embeddings.patch_embedding.weight", e, 3, 2, 2, 2);
        for (int n = 0; n < BLOCKS; n++) {
            String p = "vision_tower.vision_model.encoder.layers." + n + ".";
            f32(b, 100 + n * 20, p + "layer_norm1.weight", e);
            f32(b, 101 + n * 20, p + "layer_norm2.weight", e);
            f32(b, 102 + n * 20, p + "self_attn.q_proj.weight", e, e);
            f32(b, 103 + n * 20, p + "self_attn.k_proj.weight", e, e);
            f32(b, 104 + n * 20, p + "self_attn.v_proj.weight", e, e);
            f32(b, 105 + n * 20, p + "self_attn.out_proj.weight", e, e);
            f32(b, 106 + n * 20, p + "mlp.fc1.weight", f, e);
            f32(b, 107 + n * 20, p + "mlp.fc2.weight", e, f);
        }
        f32(b, 300, "multi_modal_projector.linear_1.weight", DIM, e);
        f32(b, 301, "multi_modal_projector.linear_1.bias", DIM);
        f32(b, 302, "multi_modal_projector.linear_2.weight", DIM, DIM);
        f32(b, 303, "multi_modal_projector.linear_2.bias", DIM);
        f32(b, 304, "patch_merge_mlp.linear_1.weight", DIM, 4 * DIM);
        f32(b, 305, "patch_merge_mlp.linear_1.bias", DIM);
        f32(b, 306, "patch_merge_mlp.linear_2.weight", DIM, DIM);
        f32(b, 307, "patch_merge_mlp.linear_2.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-minimax").toPath());
    }

    private Path muse() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['MuseGlimmerForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':"
                + DIM + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'merge_size':2,'pos_emb_height':2,'patch_temporal':1,"
                + "'layer_norm_eps':1e-05}}"));
        f32(b, 1, "model.vision_tower.patch_embedder.patch_embedding.weight", DIM, 3 * 2 * 2);
        f32(b, 2, "model.vision_tower.embeddings.position_embedding.weight", 16, DIM);
        for (int n = 0; n < BLOCKS; n++) {
            String p = "model.vision_tower.layers." + n + ".";
            f32(b, 100 + n * 20, p + "norm1.weight", DIM);
            f32(b, 101 + n * 20, p + "norm2.weight", DIM);
            f32(b, 102 + n * 20, p + "attn.q_proj.weight", HEADS * 4, DIM);
            f32(b, 103 + n * 20, p + "attn.k_proj.weight", HEADS * 4, DIM);
            f32(b, 104 + n * 20, p + "attn.v_proj.weight", HEADS * 4, DIM);
            f32(b, 105 + n * 20, p + "attn.proj.weight", DIM, HEADS * 4);
            f32(b, 106 + n * 20, p + "mlp.fc1.weight", FF, DIM);
            f32(b, 107 + n * 20, p + "mlp.fc2.weight", DIM, FF);
        }
        f32(b, 300, "model.vision_adapter.fc1.weight", DIM, 4 * DIM);
        f32(b, 301, "model.vision_adapter.fc2.weight", DIM, DIM);
        f32(b, 302, "model.vision_projection.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-muse").toPath());
    }

    private Path step3() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['StepVLForConditionalGeneration'],'hidden_size':" + DIM + ",'understand_projector_stride':2,"
                + "'vision_config':{'hidden_size':1536,'intermediate_size':8960,'num_attention_heads':12,"
                + "'num_hidden_layers':47,'patch_size':14,'image_size':728}}"));
        f32(b, 1, "vision_model.conv1.weight", DIM, 3, 14, 14);
        f32(b, 2, "vision_model.positional_embedding", 2704, DIM);
        for (int n = 0; n < 47; n++) {
            String p = "vision_model.transformer.resblocks." + n + ".";
            f32(b, 100 + n * 20, p + "ln_1.weight", DIM);
            f32(b, 101 + n * 20, p + "ln_2.weight", DIM);
            f32(b, 102 + n * 20, p + "attn.in_proj.weight", 3 * DIM, DIM);
            f32(b, 103 + n * 20, p + "attn.out_proj.weight", DIM, DIM);
            f32(b, 104 + n * 20, p + "mlp.c_fc.weight", FF, DIM);
            f32(b, 105 + n * 20, p + "mlp.c_proj.weight", DIM, FF);
        }
        f32(b, 300, "vision_model.vit_downsampler1.weight", DIM, DIM);
        f32(b, 301, "vision_model.vit_downsampler2.weight", DIM, DIM);
        f32(b, 302, "vit_large_projector.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-step3").toPath());
    }

    private Path nemotron() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['NemotronH_Nano_VL_V2'],'hidden_size':" + DIM + ",'force_image_size':64,"
                + "'downsample_ratio':0.5,'vision_config':{'patch_size':8}}"));
        f32(b, 1, "vision_model.radio_model.model.patch_generator.pos_embed.weight", 1, 16, DIM);
        f32(b, 2, "vision_model.radio_model.model.patch_generator.embedder.weight", DIM, 3 * 8 * 8);
        f32(b, 3, "vision_model.radio_model.model.patch_generator.cls_token.token", DIM);
        for (int n = 0; n < 32; n++) {
            String p = "vision_model.radio_model.model.blocks." + n + ".";
            f32(b, 100 + n * 20, p + "norm1.weight", DIM);
            f32(b, 101 + n * 20, p + "norm2.weight", DIM);
            f32(b, 102 + n * 20, p + "attn.qkv.weight", 3 * DIM, DIM);
            f32(b, 103 + n * 20, p + "attn.proj.weight", DIM, DIM);
            f32(b, 104 + n * 20, p + "mlp.fc1.weight", FF, DIM);
            f32(b, 105 + n * 20, p + "mlp.fc2.weight", DIM, FF);
        }
        f32(b, 900, "mlp1.0.weight", DIM, DIM);
        f32(b, 901, "mlp1.1.weight", DIM, DIM);
        f32(b, 902, "mlp1.3.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-nemotron").toPath());
    }

    @Test
    public void theKimiExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(kimi(), "kimi")) {
            assertTrue(mm.supportsVision());
            assertEquals("kimik25", mm.visionHparams().projectorType().typeName());
            assertTrue(mm.tensorNames().contains("mm.input_norm.weight"));
            assertTrue(mm.tensorNames().contains("mm.1.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theMiniMaxExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(minimax(), "minimax")) {
            assertTrue(mm.supportsVision());
            assertEquals("minimax_m3", mm.visionHparams().projectorType().typeName());
            assertTrue(mm.tensorNames().contains("v.patch_embd.weight.1"));
            assertTrue(mm.tensorNames().contains("mm.merger.fc1.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theMuseGlimmerExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(muse(), "muse")) {
            assertTrue(mm.supportsVision());
            assertEquals("muse-glimmer", mm.visionHparams().projectorType().typeName());
            assertTrue(mm.tensorNames().contains("mm.0.weight"));
            assertTrue(mm.tensorNames().contains("mm.2.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theStep3VlExportCarriesItsProjector() throws IOException {
        try (MmprojModel mm = export(step3(), "step3")) {
            assertTrue(mm.supportsVision());
            assertEquals("step3vl", mm.visionHparams().projectorType().typeName());
            assertEquals(728, mm.visionHparams().imageSize());
            assertTrue(mm.tensorNames().contains("mm.0.weight"));
            assertTrue(mm.tensorNames().contains("mm.1.weight"));
            assertTrue(mm.tensorNames().contains("mm.model.fc.weight"));
        }
    }

    @Test
    public void theNemotronExportCarriesItsProjector() throws IOException {
        try (MmprojModel mm = export(nemotron(), "nemotron")) {
            assertTrue(mm.supportsVision());
            assertEquals("nemotron_v2_vl", mm.visionHparams().projectorType().typeName());
            assertEquals(1280, mm.visionHparams().nEmbd());
            assertEquals(32, mm.visionHparams().nLayer());
            assertTrue(mm.tensorNames().contains("mm.model.mlp.0.weight"));
            assertTrue(mm.tensorNames().contains("mm.model.mlp.1.weight"));
        }
    }

    // ------------------------------------------------------------------ batch four fixtures

    private static final int G_DIM = 64;
    private static final int G_FF = 128;

    private Path granite() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Granite4VisionForConditionalGeneration'],'hidden_size':" + DIM + ",'downsample_rate':'2/2',"
                + "'deepstack_layer_map':[[-1,-1]],'spatial_target_layers':[],"
                + "'image_grid_pinpoints':[[8,8]],'text_config':{'num_hidden_layers':4},"
                + "'vision_config':{'hidden_size':" + G_DIM + ",'intermediate_size':" + G_FF
                + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':" + BLOCKS + ",'patch_size':2,"
                + "'image_size':8,'layer_norm_eps':1e-06}}"));
        f32(b, 1, "model.vision_model.embeddings.patch_embedding.weight", G_DIM, 3, 2, 2);
        f32(b, 2, "model.vision_tower.vision_model.embeddings.position_embedding.weight", 16, G_DIM);
        for (int n = 0; n < BLOCKS; n++) {
            String p = "model.vision_tower.vision_model.encoder.layers." + n + ".";
            f32(b, 100 + n * 20, p + "layer_norm1.weight", G_DIM);
            f32(b, 101 + n * 20, p + "layer_norm1.bias", G_DIM);
            f32(b, 102 + n * 20, p + "layer_norm2.weight", G_DIM);
            f32(b, 103 + n * 20, p + "layer_norm2.bias", G_DIM);
            f32(b, 104 + n * 20, p + "self_attn.q_proj.weight", G_DIM, G_DIM);
            f32(b, 105 + n * 20, p + "self_attn.k_proj.weight", G_DIM, G_DIM);
            f32(b, 106 + n * 20, p + "self_attn.v_proj.weight", G_DIM, G_DIM);
            f32(b, 107 + n * 20, p + "self_attn.out_proj.weight", G_DIM, G_DIM);
            f32(b, 108 + n * 20, p + "mlp.fc1.weight", G_FF, G_DIM);
            f32(b, 109 + n * 20, p + "mlp.fc2.weight", G_DIM, G_FF);
        }
        String qf = "model.layerwise_projectors.qformer.encoder.layer.0.";
        f32(b, 300, "model.layerwise_projectors.0.query", 4, G_DIM);
        f32(b, 301, "model.layerwise_projectors.0.image_positions", 4, G_DIM);
        f32(b, 302, "model.image_newline", DIM);
        f32(b, 303, "model.layerwise_projectors.0.norm.weight", G_DIM);
        f32(b, 304, "model.layerwise_projectors.0.norm.bias", G_DIM);
        f32(b, 305, "model.layerwise_projectors.0.qformer.layernorm.weight", G_DIM);
        f32(b, 306, "model.layerwise_projectors.0.qformer.layernorm.bias", G_DIM);
        f32(b, 307, "model.layerwise_projectors.0.out_linear.weight", DIM, G_DIM);
        f32(b, 308, "model.layerwise_projectors.0.out_linear.bias", DIM);
        for (String attn : new String[]{"attention", "crossattention"}) {
            String a = qf + attn + ".";
            f32(b, 320, a + "attention.query.weight", G_DIM, G_DIM);
            f32(b, 321, a + "attention.query.bias", G_DIM);
            f32(b, 322, a + "attention.key.weight", G_DIM, G_DIM);
            f32(b, 323, a + "attention.key.bias", G_DIM);
            f32(b, 324, a + "attention.value.weight", G_DIM, G_DIM);
            f32(b, 325, a + "attention.value.bias", G_DIM);
            f32(b, 326, a + "output.dense.weight", G_DIM, G_DIM);
            f32(b, 327, a + "output.dense.bias", G_DIM);
            f32(b, 328, a + "output.LayerNorm.weight", G_DIM);
            f32(b, 329, a + "output.LayerNorm.bias", G_DIM);
        }
        f32(b, 340, qf + "intermediate_query.dense.weight", G_FF, G_DIM);
        f32(b, 341, qf + "intermediate_query.dense.bias", G_FF);
        f32(b, 342, qf + "output_query.dense.weight", G_DIM, G_FF);
        f32(b, 343, qf + "output_query.dense.bias", G_DIM);
        f32(b, 344, qf + "output_query.LayerNorm.weight", G_DIM);
        f32(b, 345, qf + "output_query.LayerNorm.bias", G_DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-granite").toPath());
    }

    private Path dots3() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Dots3NoteForCausalLM'],'hidden_size':" + DIM + ",'vision_config':{'adapter_type':'patch_merger',"
                + "'pre_pixel_shuffle':true,'router_scoring_func':'sigmoid','router_scale':1.0,"
                + "'temporal_patch_size':1,'use_bias':false,'use_qk_norm':true,'rms_norm_eps':1e-06,"
                + "'capacity_factor':1.0,'pyramid_num_routed':[0,0,0,0],'embed_dim':" + DIM + ",'patch_size':2,"
                + "'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'spatial_merge_size':2},'audio_config':{'use_conv2d_stem':true,'use_rope':true,"
                + "'use_rms_norm':true,'use_causal':false,'merge_factor':1,'chunk_seconds':60,"
                + "'rope_parameters':{'partial_rotary_factor':0.5,'rope_theta':10000.0},"
                + "'whisper_config':{'activation_function':'swiglu','d_model':" + DIM + ",'encoder_ffn_dim':" + FF
                + ",'encoder_attention_heads':" + HEADS + ",'encoder_layers':" + BLOCKS
                + ",'num_mel_bins':4}}}"));
        f32(b, 1, "vision_encoder.patch_embed.proj.weight", DIM, 3, 2, 2);
        for (int n = 0; n < BLOCKS; n++) {
            String p = "vision_encoder.blocks." + n + ".";
            f32(b, 100 + n * 20, p + "norm_1.weight", DIM);
            f32(b, 101 + n * 20, p + "norm_2.weight", DIM);
            f32(b, 102 + n * 20, p + "attn.qkv.weight", 3 * DIM, DIM);
            f32(b, 103 + n * 20, p + "attn.proj.weight", DIM, DIM);
            f32(b, 104 + n * 20, p + "mlp.fc1.weight", FF, DIM);
            f32(b, 105 + n * 20, p + "mlp.fc2.weight", DIM, FF);
            f32(b, 106 + n * 20, p + "mlp.fc3.weight", FF, DIM);
        }
        f32(b, 300, "vision_encoder.post_trunk_norm.weight", DIM);
        f32(b, 301, "vision_encoder.adapter.ln_q.weight", DIM);
        f32(b, 302, "vision_encoder.adapter.ln_q.bias", DIM);
        f32(b, 303, "vision_encoder.adapter.mlp.0.weight", DIM, 4 * DIM);
        f32(b, 304, "vision_encoder.adapter.mlp.0.bias", DIM);
        f32(b, 305, "vision_encoder.adapter.mlp.2.weight", DIM, DIM);
        f32(b, 306, "vision_encoder.adapter.mlp.2.bias", DIM);
        String sp = "audio_encoder.dots_encoder.speech_encoder.";
        for (int n = 1; n <= 3; n++) {
            f32(b, 320 + n, sp + "conv2d" + n + ".weight", DIM, DIM, 3, 3);
            f32(b, 330 + n, sp + "conv2d" + n + ".bias", DIM);
        }
        f32(b, 340, sp + "conv_out.weight", DIM, DIM, 1, 1);
        f32(b, 341, sp + "layer_norm.weight", DIM);
        f32(b, 342, sp + "layer_norm.bias", DIM);
        for (int n = 0; n < BLOCKS; n++) {
            String p = sp + "layers." + n + ".";
            f32(b, 400 + n * 20, p + "self_attn_layer_norm.weight", DIM);
            f32(b, 401 + n * 20, p + "self_attn_layer_norm.bias", DIM);
            f32(b, 402 + n * 20, p + "final_layer_norm.weight", DIM);
            f32(b, 403 + n * 20, p + "final_layer_norm.bias", DIM);
            f32(b, 404 + n * 20, p + "self_attn.q_proj.weight", DIM, DIM);
            f32(b, 405 + n * 20, p + "self_attn.k_proj.weight", DIM, DIM);
            f32(b, 406 + n * 20, p + "self_attn.v_proj.weight", DIM, DIM);
            f32(b, 407 + n * 20, p + "self_attn.out_proj.weight", DIM, DIM);
            f32(b, 408 + n * 20, p + "fc1.weight", 2 * FF, DIM);
            f32(b, 409 + n * 20, p + "fc2.weight", DIM, FF);
        }
        for (int n : new int[]{0, 1, 3}) {
            f32(b, 600 + n, "audio_encoder.audio_adapter.proj." + n + ".weight", DIM, DIM);
            f32(b, 610 + n, "audio_encoder.audio_adapter.proj." + n + ".bias", DIM);
        }
        file(b, "preprocessor_config.json", "{" + COLORS + ",'min_pixels':64,'max_pixels':1024}");
        return b.write(tmp.newFolder("rt-dots3").toPath());
    }

    @Test
    public void theGraniteVisionExportOpensAsARunnableProjector() throws IOException {
        try (MmprojModel mm = export(granite(), "granite")) {
            assertTrue(mm.supportsVision());
            assertEquals("granite4_vision", mm.visionHparams().projectorType().typeName());
            assertEquals(2, mm.visionHparams().visionFamily().querySide());
            assertEquals(2, mm.visionHparams().visionFamily().windowSide());
            assertTrue(mm.tensorNames().contains("v.proj_blk.0.query"));
            assertTrue(mm.tensorNames().contains("v.proj_blk.0.img_pos"));
            assertTrue(mm.tensorNames().contains("v.proj_blk.0.self_attn_q.weight"));
            mm.openCpuVisionEncoder().close();
        }
    }

    @Test
    public void theDots3NoteExportOpensBothEncoders() throws IOException {
        try (MmprojModel mm = export(dots3(), "dots3")) {
            assertTrue(mm.supportsVision());
            assertTrue(mm.supportsAudio());
            assertEquals("dots3note_v", mm.visionHparams().projectorType().typeName());
            assertEquals("dots3note_a", mm.audioHparams().projectorType().typeName());
            assertTrue(mm.tensorNames().contains("mm.input_norm.weight"));
            assertTrue(mm.tensorNames().contains("v.blk.0.ffn_gate.weight"));
            assertTrue(mm.tensorNames().contains("a.conv2d.1.weight"));
            assertTrue(mm.tensorNames().contains("a.blk.0.ffn_up.weight"));
        }
    }

    // ------------------------------------------------------------------ batch five fixtures

    private Path museSpark() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['MuseSparkForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':"
                + DIM + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'merge_size':2,'pos_emb_height':2,'patch_temporal':1,"
                + "'layer_norm_eps':1e-05},'audio_config':{'hidden_sizes':[8,16],'mlp_ratio':2.0,"
                + "'depths':[1,1],'rms_norm_eps':1e-05,'conv_kernel_size':3,'patch_sizes':[2,2]}}"));
        f32(b, 1, "model.vision_tower.patch_embedder.patch_embedding.weight", DIM, 3 * 2 * 2);
        f32(b, 2, "model.vision_tower.embeddings.position_embedding.weight", 16, DIM);
        for (int n = 0; n < BLOCKS; n++) {
            String p = "model.vision_tower.layers." + n + ".";
            f32(b, 100 + n * 20, p + "norm1.weight", DIM);
            f32(b, 101 + n * 20, p + "norm2.weight", DIM);
            f32(b, 102 + n * 20, p + "attn.q_proj.weight", HEADS * 4, DIM);
            f32(b, 103 + n * 20, p + "attn.k_proj.weight", HEADS * 4, DIM);
            f32(b, 104 + n * 20, p + "attn.v_proj.weight", HEADS * 4, DIM);
            f32(b, 105 + n * 20, p + "attn.proj.weight", DIM, HEADS * 4);
            f32(b, 106 + n * 20, p + "mlp.fc1.weight", FF, DIM);
            f32(b, 107 + n * 20, p + "mlp.fc2.weight", DIM, FF);
        }
        f32(b, 300, "model.vision_adapter.fc1.weight", DIM, 4 * DIM);
        f32(b, 301, "model.vision_adapter.fc2.weight", DIM, DIM);
        f32(b, 302, "model.vision_projection.weight", DIM, DIM);
        f32(b, 400, "model.audio_tower.norm.weight", 16);
        // the adapter's first block input is the last stage width
        f32(b, 401, "model.audio_tower.proj.weight", DIM, 16);
        f32(b, 403, "model.audio_tower.proj.bias", DIM);
        f32(b, 402, "model.audio_tower.post_proj_norm.weight", DIM);
        f32(b, 410, "model.audio_tower.stages.0.patch_embedder.proj.weight", 16, 2);
        f32(b, 411, "model.audio_tower.stages.0.patch_embedder.proj.bias", 16);
        f32(b, 412, "model.audio_tower.stages.0.patch_embedder.norm.weight", 16);
        for (int block = 0; block < 2; block++) {
            String p = "model.audio_tower.stages.0.blocks." + block + ".";
            f32(b, 420 + block * 10, p + "conv.weight", 16, 1, 3);
            f32(b, 421 + block * 10, p + "conv.bias", 16);
            f32(b, 422 + block * 10, p + "norm.weight", 16);
            f32(b, 423 + block * 10, p + "mlp.fc1.weight", 32, 16);
            f32(b, 424 + block * 10, p + "mlp.fc1.bias", 32);
            f32(b, 425 + block * 10, p + "mlp.fc2.weight", 16, 32);
            f32(b, 426 + block * 10, p + "mlp.fc2.bias", 16);
        }
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-muse-spark").toPath());
    }

    @Test
    public void theMuseSparkExportOpensBothEncoders() throws IOException {
        try (MmprojModel mm = export(museSpark(), "muse-spark")) {
            assertTrue(mm.supportsVision());
            assertTrue(String.valueOf(mm.unrunnableReason(com.eonmux.neuralfree.gguf.mm.ProjectorModality.AUDIO)),
                    mm.supportsAudio());
            assertEquals("muse-glimmer", mm.visionHparams().projectorType().typeName());
            assertEquals("muse-spark-audio", mm.audioHparams().projectorType().typeName());
            assertTrue(mm.tensorNames().contains("a.pre_ln.weight"));
            assertTrue(mm.tensorNames().contains("a.patch.0.weight"));
            assertTrue(mm.tensorNames().contains("a.conv1d.0.weight"));
            assertTrue(mm.tensorNames().contains("a.blk.0.ffn_up.weight"));
            assertTrue(mm.tensorNames().contains("mm.a.fc.weight"));
        }
    }

    @Test
    public void theNemotronExportOpensBothEncoders() throws IOException {
        try (MmprojModel mm = export(nemotronWithAudio(), "nemotron-audio")) {
            assertTrue(mm.supportsVision());
            assertTrue(String.valueOf(mm.unrunnableReason(
                    com.eonmux.neuralfree.gguf.mm.ProjectorModality.AUDIO)), mm.supportsAudio());
            assertEquals("parakeet", mm.audioHparams().projectorType().typeName());
            assertEquals(4, mm.audioHparams().audio().nMelBins());
            assertTrue(mm.tensorNames().contains("a.mel_filters"));
            assertTrue(mm.tensorNames().contains("a.pre_encode.out.weight"));
            assertTrue(mm.tensorNames().contains("a.blk.0.linear_pos.weight"));
            assertTrue(mm.tensorNames().contains("mm.a.norm_pre.weight"));
            assertTrue(mm.tensorNames().contains("mm.a.mlp.1.weight"));
            assertTrue(mm.tensorNames().contains("mm.a.mlp.2.weight"));
        }
    }

    private Path nemotronWithAudio() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['NemotronH_Nano_VL_V2'],'hidden_size':" + DIM + ",'force_image_size':64,"
                + "'downsample_ratio':0.5,'vision_config':{'patch_size':8},'sound_config':{'hidden_size':8,"
                + "'intermediate_size':16,'num_attention_heads':2,'num_hidden_layers':2,'num_mel_bins':4,"
                + "'subsampling_factor':8,'conv_kernel_size':3}}"));
        f32(b, 1, "vision_model.radio_model.model.patch_generator.pos_embed.weight", 1, 16, DIM);
        f32(b, 2, "vision_model.radio_model.model.patch_generator.embedder.weight", DIM, 3 * 8 * 8);
        f32(b, 3, "vision_model.radio_model.model.patch_generator.cls_token.token", DIM);
        for (int n = 0; n < 32; n++) {
            String p = "vision_model.radio_model.model.blocks." + n + ".";
            f32(b, 100 + n * 20, p + "norm1.weight", DIM);
            f32(b, 101 + n * 20, p + "norm2.weight", DIM);
            f32(b, 102 + n * 20, p + "attn.qkv.weight", 3 * DIM, DIM);
            f32(b, 103 + n * 20, p + "attn.proj.weight", DIM, DIM);
            f32(b, 104 + n * 20, p + "mlp.fc1.weight", FF, DIM);
            f32(b, 105 + n * 20, p + "mlp.fc2.weight", DIM, FF);
        }
        f32(b, 900, "mlp1.0.weight", DIM, DIM);
        f32(b, 901, "mlp1.1.weight", DIM, DIM);
        f32(b, 902, "mlp1.3.weight", DIM, DIM);
        String enc = "sound_encoder.encoder.";
        f32(b, 1000, enc + "feature_extractor.featurizer.fb", 8, 4);
        f32(b, 1001, enc + "feature_extractor.featurizer.window", 8);
        for (int n : new int[]{0, 2, 3, 5, 6}) {
            f32(b, 1010 + n, enc + "subsampling.layers." + n + ".weight", DIM, DIM, 3, 1);
            f32(b, 1020 + n, enc + "subsampling.layers." + n + ".bias", DIM);
        }
        f32(b, 1030, enc + "subsampling.linear.weight", DIM, DIM);
        f32(b, 1031, enc + "subsampling.linear.bias", DIM);
        for (int n = 0; n < 2; n++) {
            String p = enc + "layers." + n + ".";
            f32(b, 1100 + n * 40, p + "norm_self_att.weight", DIM);
            f32(b, 1101 + n * 40, p + "norm_self_att.bias", DIM);
            f32(b, 1102 + n * 40, p + "norm_out.weight", DIM);
            f32(b, 1103 + n * 40, p + "norm_out.bias", DIM);
            f32(b, 1104 + n * 40, p + "self_attn.q_proj.weight", DIM, DIM);
            f32(b, 1105 + n * 40, p + "self_attn.k_proj.weight", DIM, DIM);
            f32(b, 1106 + n * 40, p + "self_attn.v_proj.weight", DIM, DIM);
            f32(b, 1107 + n * 40, p + "self_attn.o_proj.weight", DIM, DIM);
            f32(b, 1108 + n * 40, p + "self_attn.bias_u", DIM);
            f32(b, 1109 + n * 40, p + "self_attn.bias_v", DIM);
            f32(b, 1110 + n * 40, p + "self_attn.relative_k_proj.weight", DIM, DIM);
            f32(b, 1111 + n * 40, p + "norm_feed_forward1.weight", DIM);
            f32(b, 1112 + n * 40, p + "norm_feed_forward1.bias", DIM);
            f32(b, 1113 + n * 40, p + "feed_forward1.linear1.weight", FF, DIM);
            f32(b, 1114 + n * 40, p + "feed_forward1.linear2.weight", DIM, FF);
            f32(b, 1115 + n * 40, p + "norm_feed_forward2.weight", DIM);
            f32(b, 1116 + n * 40, p + "norm_feed_forward2.bias", DIM);
            f32(b, 1117 + n * 40, p + "feed_forward2.linear1.weight", FF, DIM);
            f32(b, 1118 + n * 40, p + "feed_forward2.linear2.weight", DIM, FF);
            f32(b, 1119 + n * 40, p + "norm_conv.weight", DIM);
            f32(b, 1120 + n * 40, p + "norm_conv.bias", DIM);
            f32(b, 1121 + n * 40, p + "conv.norm.weight", DIM);
            f32(b, 1122 + n * 40, p + "conv.norm.bias", DIM);
            f32(b, 1123 + n * 40, p + "conv.norm.running_mean", DIM);
            f32(b, 1124 + n * 40, p + "conv.norm.running_var", DIM);
            f32(b, 1125 + n * 40, p + "conv.pointwise_conv1.weight", 2 * DIM, DIM, 1);
            f32(b, 1126 + n * 40, p + "conv.depthwise_conv.weight", DIM, 1, 3);
            f32(b, 1127 + n * 40, p + "conv.pointwise_conv2.weight", DIM, DIM, 1);
        }
        f32(b, 1200, "sound_projection.norm.weight", DIM);
        f32(b, 1201, "sound_projection.norm.bias", DIM);
        f32(b, 1202, "sound_projection.linear1.weight", DIM, DIM);
        f32(b, 1203, "sound_projection.linear1.bias", DIM);
        f32(b, 1204, "sound_projection.linear2.weight", DIM, DIM);
        f32(b, 1205, "sound_projection.linear2.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b.write(tmp.newFolder("rt-nemotron-audio").toPath());
    }
}
