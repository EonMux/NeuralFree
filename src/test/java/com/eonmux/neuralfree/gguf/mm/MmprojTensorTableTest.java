package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The tensor requirement tables: alternatives over fused or separate attention weights and layer expansion. */
public class MmprojTensorTableTest {

    private static TensorCatalog catalog(String... names) {
        List<GGUFTensorInfo> infos = new ArrayList<>();
        for (String name : names) {
            infos.add(new GGUFTensorInfo(name, new long[]{1}, GGMLType.F32, 0L));
        }
        return TensorCatalog.of("clip", infos);
    }

    private static List<String> requiredNames(MmprojTensorTable t) {
        List<String> names = new ArrayList<>();
        for (MmprojTensorTable.Requirement r : t.required()) {
            for (List<String> alternative : r.alternatives()) {
                names.addAll(alternative);
            }
        }
        return names;
    }

    private static MmprojTensorTable table(String encoder, String projector) {
        return ProjectorEntry.of(ProjectorType.fromName("mlp"), ProjectorModality.VISION, encoder, projector)
                .tensors();
    }

    @Test
    public void fusedOrSeparateAttentionWeightsBothSatisfyTheRequirement() {
        MmprojTensorTable t = table(ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LLAVA_MLP);
        String[][] variants = {
                {"v.blk.%d.attn_qkv.weight"},
                {"v.blk.%d.attn_q.weight", "v.blk.%d.attn_k.weight", "v.blk.%d.attn_v.weight"},
        };
        for (String[] variant : variants) {
            List<String> complete = new ArrayList<>(List.of("v.patch_embd.weight", "v.position_embd.weight",
                    "v.post_ln.weight", "v.post_ln.bias", "mm.0.weight", "mm.0.bias"));
            for (int il = 0; il < 2; il++) {
                for (String n : variant) {
                    complete.add(String.format(java.util.Locale.ROOT, n, il));
                }
                complete.add("v.blk." + il + ".ln1.weight");
                complete.add("v.blk." + il + ".ln2.weight");
                complete.add("v.blk." + il + ".attn_out.weight");
                complete.add("v.blk." + il + ".ffn_up.weight");
                complete.add("v.blk." + il + ".ffn_down.weight");
            }
            t.validate(catalog(complete.toArray(new String[0])), 2);
        }
        // one separate weight alone does not satisfy the alternatives
        List<String> missingOne = new ArrayList<>(List.of("v.patch_embd.weight", "v.position_embd.weight",
                "v.post_ln.weight", "v.post_ln.bias", "mm.0.weight", "mm.0.bias",
                "v.blk.0.ln1.weight", "v.blk.0.ln2.weight", "v.blk.0.attn_out.weight",
                "v.blk.0.ffn_up.weight", "v.blk.0.ffn_down.weight",
                "v.blk.0.attn_q.weight", "v.blk.0.attn_k.weight"));
        MissingTensorException e = assertThrows(MissingTensorException.class,
                () -> t.validate(catalog(missingOne.toArray(new String[0])), 1));
        assertTrue(e.getMessage().contains("attn_qkv.weight"));
    }

    @Test
    public void layerNamesExpandOverEveryBlock() {
        MmprojTensorTable t = table(ProjectorEntry.ENCODER_QWEN2VL_VIT, ProjectorEntry.PROJECTOR_QWEN2VL_MERGER);
        List<String> names = new ArrayList<>(List.of("v.patch_embd.weight", "v.post_ln.weight",
                "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias"));
        for (int il = 0; il < 3; il++) {
            names.add("v.blk." + il + ".ln1.weight");
            names.add("v.blk." + il + ".ln2.weight");
            names.add("v.blk." + il + ".attn_out.weight");
            names.add("v.blk." + il + ".ffn_up.weight");
            names.add("v.blk." + il + ".ffn_down.weight");
            names.add("v.blk." + il + ".attn_qkv.weight");
        }
        t.validate(catalog(names.toArray(new String[0])), 3);
        names.remove("v.blk.2.ln1.weight");
        MissingTensorException e = assertThrows(MissingTensorException.class,
                () -> t.validate(catalog(names.toArray(new String[0])), 3));
        assertTrue(e.getMessage(), e.getMessage().contains("'v.blk.2.ln1.weight' (layer 2)"));
    }

    @Test
    public void aGroupNamesItsFirstAbsentTensorNotItsFirstName() {
        MmprojTensorTable t = table(ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LLAVA_MLP);
        List<String> names = new ArrayList<>(List.of("v.patch_embd.weight", "v.position_embd.weight",
                "v.post_ln.weight", "v.post_ln.bias", "mm.0.weight", "v.blk.0.ln1.weight", "v.blk.0.ln2.weight",
                "v.blk.0.attn_out.weight",
                "v.blk.0.ffn_up.weight", "v.blk.0.ffn_down.weight", "v.blk.0.attn_qkv.weight"));
        MissingTensorException e = assertThrows(MissingTensorException.class,
                () -> t.validate(catalog(names.toArray(new String[0])), 1));
        assertTrue(e.getMessage(), e.getMessage().contains("'mm.0.bias'"));
    }

    @Test
    public void familiesDifferInTheirModelLevelRequirements() {
        // clip and siglip need the learned position rows, qwen needs the post-norm but not the position rows
        assertTrue(table(ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LLAVA_MLP).optional()
                .contains("v.class_embd"));
        assertThrows(MissingTensorException.class,
                () -> table(ProjectorEntry.ENCODER_SIGLIP_VIT, ProjectorEntry.PROJECTOR_GEMMA3)
                        .validate(catalog("v.patch_embd.weight", "v.post_ln.weight", "v.post_ln.bias"), 1));
        assertEquals(4, table(ProjectorEntry.ENCODER_QWEN2VL_VIT, ProjectorEntry.PROJECTOR_GEMMA3).required()
                .size());
    }

    @Test
    public void whisperTablesUseTheAudioPrefixAndTheReferenceBiases() {
        MmprojTensorTable t = MmprojTensorTable.forFamily(ProjectorEntry.ENCODER_WHISPER,
                ProjectorEntry.PROJECTOR_QWEN2A);
        List<String> names = new ArrayList<>(List.of("a.conv1d.1.weight", "a.conv1d.1.bias",
                "a.conv1d.2.weight", "a.conv1d.2.bias", "a.position_embd.weight", "mm.a.fc.weight",
                "mm.a.fc.bias"));
        for (String n : new String[]{"ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias", "attn_q.weight",
                "attn_q.bias", "attn_k.weight", "attn_v.weight", "attn_v.bias", "attn_out.weight",
                "ffn_up.weight", "ffn_down.weight"}) {
            names.add("a.blk.0." + n);
        }
        t.validate(catalog(names.toArray(new String[0])), 1);
        assertTrue(t.optional().contains("a.post_ln.weight"));
        // whisper-enc.cpp reads the query and value biases unconditionally
        names.remove("a.blk.0.attn_v.bias");
        MissingTensorException e = assertThrows(MissingTensorException.class,
                () -> t.validate(catalog(names.toArray(new String[0])), 1));
        assertTrue(e.getMessage(), e.getMessage().contains("attn_v.bias"));
    }

    @Test
    public void qwen3aTableNeedsTheConvStackAndTheProjectorMlp() {
        MmprojTensorTable t = MmprojTensorTable.forFamily(ProjectorEntry.ENCODER_QWEN3A,
                ProjectorEntry.PROJECTOR_QWEN3A);
        for (String n : new String[]{"a.conv2d.1.weight", "a.conv2d.3.bias", "a.conv_out.weight",
                "a.position_embd.weight", "mm.a.mlp.1.weight", "mm.a.mlp.2.bias"}) {
            assertTrue(n, requiredNames(t).contains(n));
        }
        assertTrue(t.optional().contains("a.conv_out.bias"));
    }

    @Test
    public void gemma3nTableNeedsTheSubsampleNormsAndEveryConformerTensor() {
        MmprojTensorTable t = MmprojTensorTable.forFamily(ProjectorEntry.ENCODER_GEMMA3N_USM,
                ProjectorEntry.PROJECTOR_GEMMA3NA);
        List<String> required = requiredNames(t);
        for (String n : new String[]{"a.conv1d.0.norm.weight", "a.conv1d.1.norm.weight", "a.pre_encode.out.weight",
                "a.blk.%d.per_dim_scale", "a.blk.%d.linear_pos.weight", "a.blk.%d.conv_dw.weight",
                "a.blk.%d.ffn_up_1.weight", "a.blk.%d.layer_pre_norm.weight", "mm.a.embedding.weight",
                "mm.a.hard_emb_norm.weight", "mm.a.input_projection.weight"}) {
            assertTrue(n, required.contains(n));
        }
        assertFalse(required.contains("a.position_embd.weight"));
        // the fixture holds exactly the table: every name is read
        t.validate(Gemma3nAudioFixture.tensors(1).catalog(), Gemma3nAudioFixture.LAYERS);
    }

    @Test
    public void encodersAndProjectorsOfDifferentModalitiesAreRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MmprojTensorTable.forFamily(ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_QWEN2A));
        assertTrue(e.getMessage(), e.getMessage().contains(ProjectorEntry.PROJECTOR_QWEN2A));
        assertThrows(IllegalArgumentException.class,
                () -> MmprojTensorTable.forFamily(ProjectorEntry.ENCODER_WHISPER,
                        ProjectorEntry.PROJECTOR_LLAVA_MLP));
    }

    @Test
    public void everyVisionGraphRefusesTheAudioGraphs() {
        String[] encoders = {ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.ENCODER_SIGLIP_VIT,
                ProjectorEntry.ENCODER_QWEN2VL_VIT, ProjectorEntry.ENCODER_SIGLIP_RESIZED_VIT,
                ProjectorEntry.ENCODER_MINICPMV_VIT, ProjectorEntry.ENCODER_INTERNVL_VIT,
                ProjectorEntry.ENCODER_PIXTRAL_VIT, ProjectorEntry.ENCODER_LLAMA4_VIT};
        for (String enc : encoders) {
            assertEquals(enc, ProjectorModality.VISION, ProjectorEntry.modalityOf(enc));
            assertThrows(enc, IllegalArgumentException.class,
                    () -> MmprojTensorTable.forFamily(enc, ProjectorEntry.PROJECTOR_ULTRAVOX));
        }
        String[] projectors = {ProjectorEntry.PROJECTOR_MINICPMV_RESAMPLER, ProjectorEntry.PROJECTOR_INTERNVL,
                ProjectorEntry.PROJECTOR_PIXTRAL, ProjectorEntry.PROJECTOR_IDEFICS3, ProjectorEntry.PROJECTOR_LLAMA4,
                ProjectorEntry.PROJECTOR_PHI4};
        for (String proj : projectors) {
            assertThrows(proj, IllegalArgumentException.class,
                    () -> MmprojTensorTable.forFamily(ProjectorEntry.ENCODER_WHISPER, proj));
            assertThrows(proj, IllegalArgumentException.class,
                    () -> MmprojTensorTable.forFamily(ProjectorEntry.ENCODER_QWEN3A, proj));
        }
    }

    @Test
    public void layerScaleAndQkNormsAreReadByTheVitExtraEncodersOnly() {
        assertTrue(MmprojTensorTable.forFamily(ProjectorEntry.ENCODER_PIXTRAL_VIT, ProjectorEntry.PROJECTOR_PIXTRAL)
                .optional().contains("v.blk.%d.attn_q_norm.weight"));
        for (String n : MmprojTensorTable.forFamily(ProjectorEntry.ENCODER_QWEN3A, ProjectorEntry.PROJECTOR_QWEN3A)
                .optional()) {
            assertFalse(n, n.contains("ls1") || n.contains("q_norm") || n.startsWith("v.blk."));
        }
    }

    @Test
    public void unknownGraphNamesAreRefused() {
        assertThrows(IllegalArgumentException.class,
                () -> MmprojTensorTable.forFamily("mobilenet_v9", ProjectorEntry.PROJECTOR_GEMMA3));
        assertThrows(IllegalArgumentException.class,
                () -> MmprojTensorTable.forFamily(ProjectorEntry.ENCODER_CLIP_VIT, "resampler"));
    }
}
