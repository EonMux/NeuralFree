package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.RopeScalingType;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.UnusedTensorException;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * llama.cpp's {@code done_getting_tensors} check (src/llama-model-loader.cpp:1400-1410): every tensor of the file is
 * claimed by the architecture's tensor list, the generic per-tensor scale pass or a NextN block, and only the tensors
 * the loader creates for the file's layout are claimed.
 */
public class TensorClaimsTest {

    private static final int N_EMBD = 64;

    private static ModelHparams hparams(Arch arch, int blocks, int nextn) {
        return new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(blocks, nextn).nEmbd(N_EMBD).build();
    }

    private static List<TensorRequirement> tensors(Arch arch) {
        return ArchRegistry.get(arch).tensors();
    }

    private static GGUFTensorInfo t(String name, long... ne) {
        return new GGUFTensorInfo(name, ne, GGMLType.F32, 0L);
    }

    private static List<GGUFTensorInfo> llamaLayer(int il) {
        String p = "blk." + il + ".";
        return Arrays.asList(t(p + "attn_norm.weight", N_EMBD), t(p + "attn_q.weight", N_EMBD, N_EMBD),
                t(p + "attn_k.weight", N_EMBD, N_EMBD), t(p + "attn_v.weight", N_EMBD, N_EMBD),
                t(p + "attn_output.weight", N_EMBD, N_EMBD), t(p + "ffn_norm.weight", N_EMBD),
                t(p + "ffn_gate.weight", N_EMBD, 128), t(p + "ffn_up.weight", N_EMBD, 128),
                t(p + "ffn_down.weight", 128, N_EMBD));
    }

    private static List<GGUFTensorInfo> llamaFile(int layers) {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("token_embd.weight", N_EMBD, 32));
        ts.add(t("output_norm.weight", N_EMBD));
        for (int il = 0; il < layers; il++) {
            ts.addAll(llamaLayer(il));
        }
        return ts;
    }

    @Test
    public void everyListedTensorIsClaimed() {
        TensorCatalog tc = TensorCatalog.of("llama", llamaFile(2));
        assertEquals(Collections.emptyList(),
                TensorClaims.unclaimed(hparams(Arch.LLAMA, 2, 0), tc, tensors(Arch.LLAMA)));
    }

    @Test
    public void unlistedTensorsAreRefusedByName() {
        List<GGUFTensorInfo> ts = llamaFile(2);
        ts.add(t("blk.1.ffn_norm_ada_down.weight", N_EMBD));
        ts.add(t("v.blk.0.attn_q.weight", N_EMBD, N_EMBD));
        TensorCatalog tc = TensorCatalog.of("llama", ts);
        ModelHparams hp = hparams(Arch.LLAMA, 2, 0);
        assertEquals(Arrays.asList("blk.1.ffn_norm_ada_down.weight", "v.blk.0.attn_q.weight"),
                TensorClaims.unclaimed(hp, tc, tensors(Arch.LLAMA)));
        UnusedTensorException e = assertThrows(UnusedTensorException.class,
                () -> TensorClaims.requireAllClaimed(hp, tc, tensors(Arch.LLAMA)));
        assertEquals(2, e.unused().size());
        assertEquals(tc.size(), e.fileTensors());
        assertTrue(e.getMessage(), e.getMessage().contains("blk.1.ffn_norm_ada_down.weight"));
    }

    @Test
    public void blocksPastTheBlockCountAreRefused() {
        List<GGUFTensorInfo> ts = llamaFile(2);
        ts.addAll(llamaLayer(2));
        TensorCatalog tc = TensorCatalog.of("llama", ts);
        List<String> unused = TensorClaims.unclaimed(hparams(Arch.LLAMA, 2, 0), tc, tensors(Arch.LLAMA));
        assertEquals(llamaLayer(2).size(), unused.size());
        assertTrue(unused.contains("blk.2.attn_norm.weight"));
    }

    @Test
    public void scalesOfLoadedProjectionsAreClaimed() {
        List<GGUFTensorInfo> ts = llamaFile(1);
        ts.add(t("blk.0.attn_q.scale", 1));
        ts.add(t("blk.0.ffn_down.input_scale", 1));
        ts.add(t("blk.0.ffn_down_exps.scale", 1));
        TensorCatalog tc = TensorCatalog.of("llama", ts);
        assertEquals(Collections.singletonList("blk.0.ffn_down_exps.scale"),
                TensorClaims.unclaimed(hparams(Arch.LLAMA, 1, 0), tc, tensors(Arch.LLAMA)));
    }

    @Test
    public void nextnBlocksClaimTheirDecoderLayerOnlyForMtpArchitectures() {
        String p = "blk.2.";
        List<GGUFTensorInfo> ts = Arrays.asList(t(p + "attn_norm.weight", N_EMBD),
                t(p + "attn_qkv.weight", N_EMBD, 3L * N_EMBD), t(p + "attn_output.weight", N_EMBD, N_EMBD),
                t(p + "post_attention_norm.weight", N_EMBD), t(p + "ffn_norm.weight", N_EMBD),
                t(p + "ffn_up.weight", N_EMBD, 256), t(p + "ffn_down.weight", 128, N_EMBD),
                t(p + "post_ffw_norm.weight", N_EMBD), t(p + "nextn.eh_proj.weight", 2L * N_EMBD, N_EMBD),
                t(p + "nextn.enorm.weight", N_EMBD), t(p + "nextn.hnorm.weight", N_EMBD));
        TensorCatalog glm = TensorCatalog.of("glm4", ts);
        assertEquals(Collections.emptyList(),
                TensorClaims.unclaimed(hparams(Arch.GLM4, 3, 1), glm, tensors(Arch.GLM4)));

        assertFalse(TensorClaims.nextnBlocksRepeatTheTrunk(hparams(Arch.GLM4, 3, 1), tensors(Arch.GLM4)));

        TensorCatalog llama = TensorCatalog.of("llama", ts);
        assertEquals(ts.size(), TensorClaims.unclaimed(hparams(Arch.LLAMA, 3, 1), llama, tensors(Arch.LLAMA)).size());
    }

    private static List<GGUFTensorInfo> nextnHead(int il) {
        String p = "blk." + il + ".";
        return Arrays.asList(t(p + "nextn.eh_proj.weight", 2L * N_EMBD, N_EMBD), t(p + "nextn.enorm.weight", N_EMBD),
                t(p + "nextn.hnorm.weight", N_EMBD));
    }

    private static List<GGUFTensorInfo> experts(int il) {
        String p = "blk." + il + ".";
        return Arrays.asList(t(p + "ffn_gate_inp.weight", N_EMBD, 4), t(p + "ffn_gate_exps.weight", N_EMBD, 32, 4),
                t(p + "ffn_down_exps.weight", 32, N_EMBD, 4), t(p + "ffn_up_exps.weight", N_EMBD, 32, 4));
    }

    @Test
    public void nextnBlocksOfTrunkBodiedLoadersClaimTheTrunkTensors() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        for (int il = 0; il < 2; il++) {
            String p = "blk." + il + ".";
            ts.add(t(p + "attn_norm.weight", N_EMBD));
            ts.add(t(p + "attn_qkv.weight", N_EMBD, 3L * N_EMBD));
            ts.add(t(p + "attn_output.weight", N_EMBD, N_EMBD));
            ts.add(t(p + "post_attention_norm.weight", N_EMBD));
            ts.add(t(p + "exp_probs_b.bias", 4));
            ts.addAll(experts(il));
        }
        ts.addAll(nextnHead(1));
        TensorCatalog tc = TensorCatalog.of("glm4moe", ts);
        ModelHparams hp = hparams(Arch.GLM4_MOE, 2, 1);
        assertTrue(TensorClaims.nextnBlocksRepeatTheTrunk(hp, tensors(Arch.GLM4_MOE)));
        assertEquals(Collections.emptyList(), TensorClaims.unclaimed(hp, tc, tensors(Arch.GLM4_MOE)));

        List<GGUFTensorInfo> dense = new ArrayList<>(ts);
        dense.add(t("blk.1.ffn_gate.weight", N_EMBD, 128));
        assertEquals(Collections.singletonList("blk.1.ffn_gate.weight"), TensorClaims.unclaimed(hp,
                TensorCatalog.of("glm4moe", dense), tensors(Arch.GLM4_MOE)));
    }

    @Test
    public void modelsWithoutATrunkHaveNoTrunkBodyToRepeat() {
        ModelHparams hp = hparams(Arch.GEMMA4_ASSISTANT, 2, 2);
        assertFalse(TensorClaims.nextnBlocksRepeatTheTrunk(hp, tensors(Arch.GEMMA4_ASSISTANT)));
    }

    @Test
    public void exaoneMoeNextnBlocksCarryADenseFfn() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        String p = "blk.1.";
        ts.add(t(p + "attn_norm.weight", N_EMBD));
        ts.add(t(p + "attn_q_norm.weight", 16));
        ts.add(t(p + "attn_k_norm.weight", 16));
        ts.add(t(p + "attn_qkv.weight", N_EMBD, 3L * N_EMBD));
        ts.add(t(p + "attn_output.weight", N_EMBD, N_EMBD));
        ts.add(t(p + "ffn_norm.weight", N_EMBD));
        ts.add(t(p + "ffn_gate.weight", N_EMBD, 128));
        ts.add(t(p + "ffn_down.weight", 128, N_EMBD));
        ts.add(t(p + "ffn_up.weight", N_EMBD, 128));
        ts.addAll(nextnHead(1));
        ModelHparams hp = hparams(Arch.EXAONE_MOE, 2, 1);
        assertFalse(TensorClaims.nextnBlocksRepeatTheTrunk(hp, tensors(Arch.EXAONE_MOE)));
        assertEquals(Collections.emptyList(),
                TensorClaims.unclaimed(hp, TensorCatalog.of("exaone-moe", ts), tensors(Arch.EXAONE_MOE)));

        ts.addAll(experts(1));
        assertEquals(experts(1).size(),
                TensorClaims.unclaimed(hp, TensorCatalog.of("exaone-moe", ts), tensors(Arch.EXAONE_MOE)).size());
    }

    private static TensorCatalog catalog(String arch, List<GGUFTensorInfo> base, GGUFTensorInfo... extra) {
        List<GGUFTensorInfo> ts = new ArrayList<>(base);
        ts.addAll(Arrays.asList(extra));
        return TensorCatalog.of(arch, ts);
    }

    /** A one-block llama file whose attention input is the fused {@code attn_qkv}. */
    private static List<GGUFTensorInfo> fusedLlamaFile() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("token_embd.weight", N_EMBD, 32));
        ts.add(t("output_norm.weight", N_EMBD));
        for (GGUFTensorInfo ti : llamaLayer(0)) {
            if (!ti.name().matches("blk\\.0\\.attn_[qkv]\\.weight")) {
                ts.add(ti);
            }
        }
        ts.add(t("blk.0.attn_qkv.weight", N_EMBD, 3L * N_EMBD));
        return ts;
    }

    @Test
    public void separateProjectionsNeverClaimTheFusedBias() {
        TensorCatalog tc = catalog("llama", llamaFile(1), t("blk.0.attn_qkv.bias", 3L * N_EMBD),
                t("blk.0.attn_q.bias", N_EMBD));
        assertEquals(Collections.singletonList("blk.0.attn_qkv.bias"),
                TensorClaims.unclaimed(hparams(Arch.LLAMA, 1, 0), tc, tensors(Arch.LLAMA)));
    }

    @Test
    public void fusedProjectionClaimsSeparateBiasesOnlyWithoutTheFusedBias() {
        ModelHparams hp = hparams(Arch.LLAMA, 1, 0);
        GGUFTensorInfo qb = t("blk.0.attn_q.bias", N_EMBD);
        GGUFTensorInfo kb = t("blk.0.attn_k.bias", N_EMBD);
        GGUFTensorInfo vb = t("blk.0.attn_v.bias", N_EMBD);
        assertEquals(Collections.emptyList(), TensorClaims.unclaimed(hp,
                catalog("llama", fusedLlamaFile(), qb, kb, vb), tensors(Arch.LLAMA)));
        assertEquals(Arrays.asList("blk.0.attn_q.bias", "blk.0.attn_k.bias", "blk.0.attn_v.bias"),
                TensorClaims.unclaimed(hp, catalog("llama", fusedLlamaFile(), t("blk.0.attn_qkv.bias", 3L * N_EMBD),
                        qb, kb, vb), tensors(Arch.LLAMA)));
    }

    @Test
    public void fusedProjectionLeavesSeparateWeightsUnclaimed() {
        TensorCatalog tc = catalog("llama", fusedLlamaFile(), t("blk.0.attn_q.weight", N_EMBD, N_EMBD));
        assertEquals(Collections.singletonList("blk.0.attn_q.weight"),
                TensorClaims.unclaimed(hparams(Arch.LLAMA, 1, 0), tc, tensors(Arch.LLAMA)));
    }

    @Test
    public void fusedGateUpExpertsLeaveTheSeparatePairUnclaimedExceptInSkippedNextnBlocks() {
        GGUFTensorInfo[] both = {t("blk.0.ffn_gate_up_exps.weight", N_EMBD, 64, 4),
                t("blk.0.ffn_gate_exps.weight", N_EMBD, 32, 4), t("blk.0.ffn_up_exps.weight", N_EMBD, 32, 4)};
        assertEquals(Arrays.asList("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"),
                TensorClaims.unclaimed(hparams(Arch.DEEPSEEK2, 1, 0),
                        catalog("deepseek2", Collections.<GGUFTensorInfo>emptyList(), both), tensors(Arch.DEEPSEEK2)));

        List<GGUFTensorInfo> nextn = new ArrayList<>(nextnHead(1));
        nextn.add(t("blk.1.ffn_gate_up_exps.weight", N_EMBD, 64, 4));
        nextn.add(t("blk.1.ffn_gate_exps.weight", N_EMBD, 32, 4));
        nextn.add(t("blk.1.ffn_up_exps.weight", N_EMBD, 32, 4));
        ModelHparams hp = hparams(Arch.DEEPSEEK2, 2, 1);
        assertTrue(TensorClaims.nextnBlocksRepeatTheTrunk(hp, tensors(Arch.DEEPSEEK2)));
        assertEquals(Collections.emptyList(),
                TensorClaims.unclaimed(hp, TensorCatalog.of("deepseek2", nextn), tensors(Arch.DEEPSEEK2)));
    }

    private static ModelHparams llamaRope(RopeScalingType type) {
        return new ModelHparamsBuilder(ArchRegistry.traits(Arch.LLAMA)).layerCounts(1, 0).nEmbd(N_EMBD)
                .ropeScalingType(type).build();
    }

    private static TensorCatalog ropeFactors(String arch) {
        return TensorCatalog.of(arch, Arrays.asList(t("rope_freqs.weight", 8), t("rope_factors_long.weight", 8),
                t("rope_factors_short.weight", 8)));
    }

    @Test
    public void ropeFactorsFollowTheTrainingScalingType() {
        assertEquals(Arrays.asList("rope_factors_long.weight", "rope_factors_short.weight"),
                TensorClaims.unclaimed(llamaRope(RopeScalingType.YARN), ropeFactors("llama"), tensors(Arch.LLAMA)));
        assertEquals(Collections.singletonList("rope_freqs.weight"),
                TensorClaims.unclaimed(llamaRope(RopeScalingType.LONGROPE), ropeFactors("llama"),
                        tensors(Arch.LLAMA)));
        assertEquals(Collections.singletonList("rope_freqs.weight"),
                TensorClaims.unclaimed(hparamsWithRope(Arch.DECI, RopeScalingType.LONGROPE), ropeFactors("deci"),
                        tensors(Arch.DECI)));
    }

    @Test
    public void ropeFreqsOnlyLoadersNeverClaimLongRopeFactors() {
        for (Arch arch : Arrays.asList(Arch.ARCEE, Arch.EXAONE, Arch.COGVLM)) {
            assertEquals(arch.ggufName(), Arrays.asList("rope_factors_long.weight", "rope_factors_short.weight"),
                    TensorClaims.unclaimed(hparamsWithRope(arch, RopeScalingType.LONGROPE),
                            ropeFactors(arch.ggufName()), tensors(arch)));
        }
    }

    private static ModelHparams hparamsWithRope(Arch arch, RopeScalingType type) {
        return new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(1, 0).nEmbd(N_EMBD)
                .ropeScalingType(type).build();
    }

    @Test
    public void dots3noteNextnBlocksCarryMlaAndADenseFfn() {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.DOTS3NOTE)).layerCounts(3, 1)
                .nEmbd(N_EMBD).nLayerDenseLead(1).build();
        String p = "blk.2.";
        List<GGUFTensorInfo> ts = new ArrayList<>(nextnHead(2));
        for (String name : new String[] {"attn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
                "attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_k_b.weight", "attn_v_b.weight",
                "attn_output.weight", "ffn_norm.weight", "attn_k_norm.weight", "attn_gate.weight",
                "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight", "nextn.embed_tokens.weight",
                "nextn.shared_head_norm.weight"}) {
            ts.add(t(p + name, N_EMBD));
        }
        assertFalse(TensorClaims.nextnBlocksRepeatTheTrunk(hp, tensors(Arch.DOTS3NOTE)));
        assertEquals(Collections.emptyList(),
                TensorClaims.unclaimed(hp, TensorCatalog.of("dots3note", ts), tensors(Arch.DOTS3NOTE)));

        ts.add(t(p + "indexer.proj.weight", N_EMBD));
        ts.add(t(p + "ffn_gate_inp.weight", N_EMBD));
        assertEquals(Arrays.asList(p + "indexer.proj.weight", p + "ffn_gate_inp.weight"),
                TensorClaims.unclaimed(hp, TensorCatalog.of("dots3note", ts), tensors(Arch.DOTS3NOTE)));
    }

    @Test
    public void modernBertWithoutDecisionLayersLeavesTheScorerBiasesUnclaimed() {
        ModelHparams hp = hparams(Arch.MODERN_BERT, 1, 0);
        TensorCatalog tc = TensorCatalog.of("modern-bert", Arrays.asList(t("cls.weight", N_EMBD, N_EMBD),
                t("cls.bias", N_EMBD), t("cls.norm.weight", N_EMBD), t("cls.norm.bias", N_EMBD),
                t("token_types.weight", N_EMBD, 4)));
        assertEquals(Arrays.asList("cls.bias", "cls.norm.bias", "token_types.weight"),
                TensorClaims.unclaimed(hp, tc, tensors(Arch.MODERN_BERT)));
    }
}
