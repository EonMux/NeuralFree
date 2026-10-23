package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Routed-expert resolution: gpt-oss SwiGLU-OAI experts (fused gate_up scaled by {@code ffn_up_exps.scale}, never
 * clamped) and the k2-horizon dense lead checked against {@code ffn_gate_inp} presence.
 */
public class MoeResolverTest {

    private static final int N_EMBD = 64;
    private static final int N_FF = 128;
    private static final int N_FF_EXP = 32;
    private static final int N_EXPERT = 4;
    private static final int N_USED = 2;

    private static GGUFTensorInfo t(String name, long... ne) {
        return new GGUFTensorInfo(name, ne, GGMLType.F32, 0L);
    }

    private static Map<String, Object> common(String arch, int blocks) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", arch);
        m.put(arch + ".context_length", 4096);
        m.put(arch + ".embedding_length", N_EMBD);
        m.put(arch + ".block_count", blocks);
        m.put(arch + ".feed_forward_length", N_FF);
        m.put(arch + ".attention.head_count", 4);
        m.put(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        m.put(arch + ".expert_count", N_EXPERT);
        m.put(arch + ".expert_used_count", N_USED);
        m.put(arch + ".expert_feed_forward_length", N_FF_EXP);
        return m;
    }

    private static ModelHparams gptOss() {
        Map<String, Object> m = common("gpt-oss", 1);
        m.put("gpt-oss.attention.sliding_window", 128);
        return HparamsLoader.load(GgufKv.ofMap(m), Arch.OPENAI_MOE);
    }

    private static ModelHparams k2Horizon() {
        Map<String, Object> m = common("k2-horizon", 2);
        m.put("k2-horizon.leading_dense_block_count", 1);
        return HparamsLoader.load(GgufKv.ofMap(m), Arch.K2_HORIZON);
    }

    private static void router(List<GGUFTensorInfo> ts, int il) {
        ts.add(t("blk." + il + ".ffn_gate_inp.weight", N_EMBD, N_EXPERT));
        ts.add(t("blk." + il + ".ffn_down_exps.weight", N_FF_EXP, N_EMBD, N_EXPERT));
    }

    @Test
    public void gptOssFusedGateUpUsesUpScaleAndNoClamp() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        router(ts, 0);
        ts.add(t("blk.0.ffn_gate_up_exps.weight", N_EMBD, 2L * N_FF_EXP, N_EXPERT));
        ts.add(t("blk.0.ffn_gate_up_exps.bias", 2L * N_FF_EXP, N_EXPERT));
        ts.add(t("blk.0.ffn_up_exps.scale", N_EXPERT));
        ModelHparams hp = gptOss();
        FfnResolver.Result r = new FfnResolver(hp, TensorCatalog.of("gpt-oss", ts)).resolve(0);
        assertEquals(FfnKind.MOE, r.kind());
        assertNull(r.ffn());
        MoeSpec moe = r.moe();
        assertEquals(N_EXPERT, moe.nExpert());
        assertEquals(N_USED, moe.nUsed());
        assertEquals(N_FF_EXP, moe.nFfExp());
        assertEquals(GatingFunc.SOFTMAX_WEIGHT, moe.gating());
        assertEquals(MoeSpec.SelectionMode.PROBS, moe.selectionMode());
        assertEquals(GluOp.SWIGLU_OAI, moe.glu());
        assertTrue(moe.fusedGateUp());
        assertTrue(moe.expertScales());
        assertTrue(moe.expertBiases());
        assertEquals(FfnResolver.OAI_ALPHA, moe.params().alpha(), 0f);
        assertEquals(FfnResolver.OAI_LIMIT, moe.params().limit(), 0f);
        assertFalse(moe.normWeights());
        assertNull(moe.sharedExpert());
    }

    @Test
    public void gptOssSeparateGateAndUpAreNotFused() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        router(ts, 0);
        ts.add(t("blk.0.ffn_gate_exps.weight", N_EMBD, N_FF_EXP, N_EXPERT));
        ts.add(t("blk.0.ffn_up_exps.weight", N_EMBD, N_FF_EXP, N_EXPERT));
        MoeSpec moe = new MoeResolver(gptOss(), TensorCatalog.of("gpt-oss", ts)).resolve(0);
        assertFalse(moe.fusedGateUp());
        assertFalse(moe.expertScales());
        assertEquals(GluOp.SWIGLU_OAI, moe.glu());
    }

    @Test
    public void expertScaleMustHoldOneValuePerExpert() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        router(ts, 0);
        ts.add(t("blk.0.ffn_gate_up_exps.weight", N_EMBD, 2L * N_FF_EXP, N_EXPERT));
        ts.add(t("blk.0.ffn_up_exps.scale", 1));
        MoeResolver resolver = new MoeResolver(gptOss(), TensorCatalog.of("gpt-oss", ts));
        assertThrows(IllegalStateException.class, () -> resolver.resolve(0));
    }

    @Test
    public void k2HorizonDenseLeadThenExperts() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.ffn_gate.weight", N_EMBD, N_FF));
        ts.add(t("blk.0.ffn_up.weight", N_EMBD, N_FF));
        ts.add(t("blk.0.ffn_down.weight", N_FF, N_EMBD));
        router(ts, 1);
        ts.add(t("blk.1.ffn_gate_exps.weight", N_EMBD, N_FF_EXP, N_EXPERT));
        ts.add(t("blk.1.ffn_up_exps.weight", N_EMBD, N_FF_EXP, N_EXPERT));
        ModelHparams hp = k2Horizon();
        FfnResolver ffn = new FfnResolver(hp, TensorCatalog.of("k2-horizon", ts));
        FfnResolver.Result lead = ffn.resolve(0);
        assertEquals(FfnKind.DENSE, lead.kind());
        assertEquals(GluOp.SWIGLU, lead.ffn().glu());
        FfnResolver.Result moe = ffn.resolve(1);
        assertEquals(FfnKind.MOE, moe.kind());
        assertEquals(GatingFunc.SIGMOID, moe.moe().gating());
        assertEquals(GluOp.SWIGLU, moe.moe().glu());
    }

    @Test
    public void routerInsideTheDenseLeadIsRejected() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        router(ts, 0);
        ts.add(t("blk.0.ffn_gate_exps.weight", N_EMBD, N_FF_EXP, N_EXPERT));
        ts.add(t("blk.0.ffn_up_exps.weight", N_EMBD, N_FF_EXP, N_EXPERT));
        MoeResolver resolver = new MoeResolver(k2Horizon(), TensorCatalog.of("k2-horizon", ts));
        assertThrows(IllegalStateException.class, () -> resolver.isMoeLayer(0));
        assertFalse(resolver.isMoeLayer(1));
    }

    @Test
    public void mistral3RenormalizesSoftmaxRoutingAndIgnoresSharedExpertTensors() {
        Map<String, Object> m = common("mistral3", 1);
        ModelHparams hp = HparamsLoader.load(GgufKv.ofMap(m), Arch.MISTRAL3);
        List<GGUFTensorInfo> ts = new ArrayList<>();
        router(ts, 0);
        ts.add(t("blk.0.ffn_gate_exps.weight", N_EMBD, N_FF_EXP, N_EXPERT));
        ts.add(t("blk.0.ffn_up_exps.weight", N_EMBD, N_FF_EXP, N_EXPERT));
        ts.add(t("blk.0.ffn_gate_shexp.weight", N_EMBD, N_FF));
        ts.add(t("blk.0.ffn_up_shexp.weight", N_EMBD, N_FF));
        ts.add(t("blk.0.ffn_down_shexp.weight", N_FF, N_EMBD));
        MoeSpec moe = new MoeResolver(hp, TensorCatalog.of("mistral3", ts)).resolve(0);
        assertEquals(GatingFunc.SOFTMAX, moe.gating());
        assertTrue(moe.normWeights());
        assertNull(moe.sharedExpert());
        assertEquals(1.0f, moe.sharedCombineScale(), 0f);
    }

    @Test
    public void sharedCombineScaleNeedsASharedExpert() {
        assertThrows(IllegalArgumentException.class, () -> MoeSpec.builder()
                .experts(N_EXPERT, N_USED, N_FF_EXP)
                .sharedCombineScale(MoeResolver.COHERE2MOE_SHARED_COMBINE_SCALE)
                .build());
    }
}
