package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Files holding only the MTP blocks of their model ({@code nextn_flags}, llama-model.cpp:3901-3920): the trunk tensors
 * are not required, the loader checks pass, and the file is then refused as a model of its own. An MTP drafter of
 * any draft file requires no trunk tensor and reads only the tensors outside the trunk blocks.
 */
public class MtpOnlyFileTest {

    private static final int N_EMBD = 64;
    private static final String ARCH = "glm4moe";

    private static ModelHparams glm4Moe() {
        return new ModelHparamsBuilder(ArchRegistry.traits(Arch.GLM4_MOE)).layerCounts(2, 1).nEmbd(N_EMBD).build();
    }

    private static List<TensorRequirement> tensors() {
        return ArchRegistry.get(Arch.GLM4_MOE).tensors();
    }

    private static GGUFTensorInfo t(String name, long... ne) {
        return new GGUFTensorInfo(name, ne, GGMLType.F32, 0L);
    }

    /** token_embd, output_norm and the NextN block {@code blk.1} with the trunk block body. */
    private static List<GGUFTensorInfo> mtpOnly() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("token_embd.weight", N_EMBD, 32));
        ts.add(t("output_norm.weight", N_EMBD));
        String p = "blk.1.";
        ts.add(t(p + "attn_norm.weight", N_EMBD));
        ts.add(t(p + "attn_qkv.weight", N_EMBD, 3L * N_EMBD));
        ts.add(t(p + "attn_output.weight", N_EMBD, N_EMBD));
        ts.add(t(p + "post_attention_norm.weight", N_EMBD));
        ts.add(t(p + "exp_probs_b.bias", 4));
        ts.add(t(p + "ffn_gate_inp.weight", N_EMBD, 4));
        ts.add(t(p + "ffn_gate_exps.weight", N_EMBD, 32, 4));
        ts.add(t(p + "ffn_down_exps.weight", 32, N_EMBD, 4));
        ts.add(t(p + "ffn_up_exps.weight", N_EMBD, 32, 4));
        ts.add(t(p + "nextn.eh_proj.weight", 2L * N_EMBD, N_EMBD));
        ts.add(t(p + "nextn.enorm.weight", N_EMBD));
        ts.add(t(p + "nextn.hnorm.weight", N_EMBD));
        ts.add(t(p + "nextn.shared_head_norm.weight", N_EMBD));
        return ts;
    }

    @Test
    public void mtpOnlyFilePassesTheLoaderChecks() {
        ModelHparams hp = glm4Moe();
        TensorCatalog tc = TensorCatalog.of(ARCH, mtpOnly());
        assertTrue(ModelSpecValidator.isMtpOnly(hp, tc));
        ModelSpecValidator.requireTensors(hp, tc, tensors());
        assertEquals(Collections.emptyList(), TensorClaims.unclaimed(hp, tc, tensors()));
    }

    @Test
    public void mtpOnlyFileIsRefusedAsAModel() {
        ModelHparams hp = glm4Moe();
        TensorCatalog tc = TensorCatalog.of(ARCH, mtpOnly());
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> ModelSpecBuilder.refuseMtpOnly(hp, tc));
        assertTrue(e.getMessage(), e.getMessage().contains("MTP block"));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.0.attn_norm.weight"));
    }

    @Test
    public void aFileWithTheTrunkProbeStillRequiresEveryTrunkTensor() {
        ModelHparams hp = glm4Moe();
        List<GGUFTensorInfo> ts = mtpOnly();
        ts.add(t("blk.0.attn_norm.weight", N_EMBD));
        TensorCatalog tc = TensorCatalog.of(ARCH, ts);
        assertFalse(ModelSpecValidator.isMtpOnly(hp, tc));
        MissingTensorException e = assertThrows(MissingTensorException.class,
                () -> ModelSpecValidator.requireTensors(hp, tc, tensors()));
        assertTrue(e.missing().toString(), e.missing().contains("blk.0.attn_output.weight"));
    }

    @Test
    public void onlyNextnFlagsLoadersRecogniseMtpOnlyFiles() {
        ModelHparams llama = new ModelHparamsBuilder(ArchRegistry.traits(Arch.LLAMA)).layerCounts(2, 1)
                .nEmbd(N_EMBD).build();
        assertFalse(ModelSpecValidator.isMtpOnly(llama, TensorCatalog.of("llama", mtpOnly())));
        ModelHparams trunkOnly = new ModelHparamsBuilder(ArchRegistry.traits(Arch.GLM4_MOE)).layerCounts(2, 0)
                .nEmbd(N_EMBD).build();
        assertFalse(ModelSpecValidator.isMtpOnly(trunkOnly, TensorCatalog.of(ARCH, mtpOnly())));
        assertEquals("blk.0.hc_attn_norm.weight", ModelSpecValidator.trunkProbe(Arch.QWEN4EXP));
    }

    @Test
    public void anMtpDrafterRequiresNoTrunkTensorEvenWhenTheFileHoldsSome() {
        ModelHparams hp = glm4Moe();
        List<GGUFTensorInfo> ts = mtpOnly();
        ts.add(t("blk.0.attn_norm.weight", N_EMBD));
        TensorCatalog tc = TensorCatalog.of(ARCH, ts);
        ModelSpecValidator.requireTensors(hp, tc, tensors(), true);
        assertThrows(MissingTensorException.class, () -> ModelSpecValidator.requireTensors(hp, tc, tensors(), false));
    }

    @Test
    public void trunkTensorsAreTheBlocksBeforeTheNextnBlocks() {
        assertTrue(ModelSpecValidator.isTrunkTensor("blk.0.attn_norm.weight", 1));
        assertTrue(ModelSpecValidator.isTrunkTensor("model.layers.0.self_attn.q_proj.weight", 1));
        assertFalse(ModelSpecValidator.isTrunkTensor("blk.1.nextn.eh_proj.weight", 1));
        assertFalse(ModelSpecValidator.isTrunkTensor("blk.1.attn_norm.weight", 1));
        assertFalse(ModelSpecValidator.isTrunkTensor("token_embd.weight", 1));
        assertFalse(ModelSpecValidator.isTrunkTensor("blk.x.attn_norm.weight", 1));
        assertFalse(ModelSpecValidator.isTrunkTensor("blk.0", 1));
        assertFalse(ModelSpecValidator.isTrunkTensor("enc.blk.0.attn_q.weight", 1));
    }
}
