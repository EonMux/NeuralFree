package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.arch.KdaGateMode;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** Recurrent mixers: qwen3next gated DeltaNet with GROUPED heads and kimi-linear KDA with a low-rank gate. */
public class RecurrentResolverTest {

    private static final int N_EMBD = 64;

    private static GGUFTensorInfo t(String name, long... ne) {
        return new GGUFTensorInfo(name, ne, GGMLType.F32, 0L);
    }

    private static Map<String, Object> common(String arch, int blocks) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", arch);
        m.put(arch + ".context_length", 4096);
        m.put(arch + ".embedding_length", N_EMBD);
        m.put(arch + ".block_count", blocks);
        m.put(arch + ".feed_forward_length", 128);
        m.put(arch + ".attention.head_count", 4);
        m.put(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        m.put(arch + ".expert_count", 4);
        m.put(arch + ".expert_used_count", 2);
        m.put(arch + ".expert_feed_forward_length", 32);
        return m;
    }

    /** Four blocks, full attention every fourth: layers 0-2 are GDN with 2 K heads, 4 V heads, 16-dim heads. */
    private static ModelHparams qwen3next() {
        Map<String, Object> m = common("qwen3next", 4);
        m.put("qwen3next.ssm.conv_kernel", 4);
        m.put("qwen3next.ssm.inner_size", 64);
        m.put("qwen3next.ssm.state_size", 16);
        m.put("qwen3next.ssm.time_step_rank", 4);
        m.put("qwen3next.ssm.group_count", 2);
        return HparamsLoader.load(GgufKv.ofMap(m), Arch.QWEN3NEXT);
    }

    /** Layer 0 is KDA (no KV heads), layer 1 MLA attention. */
    private static ModelHparams kimiLinear() {
        Map<String, Object> m = common("kimi-linear", 2);
        m.put("kimi-linear.attention.head_count_kv", new int[] {0, 4});
        m.put("kimi-linear.attention.key_length_mla", 24);
        m.put("kimi-linear.attention.value_length_mla", 16);
        m.put("kimi-linear.attention.kv_lora_rank", 32);
        m.put("kimi-linear.ssm.conv_kernel", 4);
        m.put("kimi-linear.kda.head_dim", 16);
        m.put("kimi-linear.expert_shared_count", 1);
        m.put("kimi-linear.expert_gating_func", 2);
        return HparamsLoader.load(GgufKv.ofMap(m), Arch.KIMI_LINEAR);
    }

    @Test
    public void qwen3nextGatedDeltaNetUsesGroupedHeads() {
        ModelHparams hp = qwen3next();
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.attn_qkv.weight", N_EMBD, 128));
        ts.add(t("blk.0.ssm_ba.weight", N_EMBD, 8));
        RecurrentResolver resolver = new RecurrentResolver(hp, TensorCatalog.of("qwen3next", ts));
        assertTrue(resolver.isRecurrentLayer(0));
        assertFalse(resolver.isRecurrentLayer(3));
        RecurrentSpec spec = resolver.resolve(0);
        assertEquals(MixerKind.GDN, spec.kind());
        assertEquals(GdnHeadMap.GROUPED, spec.headMap());
        assertEquals(2, spec.nKHeads());
        assertEquals(4, spec.nVHeads());
        assertEquals(16, spec.headKDim());
        assertEquals(16, spec.headVDim());
        assertTrue(spec.fusedBetaAlpha());
        assertFalse(spec.legacySsmIn());
        assertEquals(hp.nEmbdR(), spec.convStateFloats());
        assertEquals(hp.nEmbdS(), spec.stateFloats());
    }

    @Test
    public void qwen3nextWithoutFusedQkvUsesLegacySsmIn() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.1.ssm_in.weight", N_EMBD, 256));
        RecurrentSpec spec = new RecurrentResolver(qwen3next(), TensorCatalog.of("qwen3next", ts)).resolve(1);
        assertTrue(spec.legacySsmIn());
        assertFalse(spec.fusedBetaAlpha());
    }

    @Test
    public void attentionLayerIsNotRecurrent() {
        RecurrentResolver resolver = new RecurrentResolver(qwen3next(), TensorCatalog.of("qwen3next", new ArrayList<>()));
        assertThrows(IllegalArgumentException.class, () -> resolver.resolve(3));
    }

    @Test
    public void kimiLinearKdaUsesLowRankGateAndSoftplusDecay() {
        ModelHparams hp = kimiLinear();
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.ssm_g_a.weight", N_EMBD, 16));
        ts.add(t("blk.0.ssm_g_b.weight", 16, 64));
        RecurrentResolver resolver = new RecurrentResolver(hp, TensorCatalog.of("kimi-linear", ts));
        assertFalse(resolver.isRecurrentLayer(1));
        RecurrentSpec spec = resolver.resolve(0);
        assertEquals(MixerKind.KDA, spec.kind());
        assertEquals(KdaGateMode.LOWRANK, spec.kdaGateMode());
        assertEquals(RecurrentSpec.DecayGate.SOFTPLUS, spec.decayGate());
        assertEquals(4, spec.nHead());
        assertEquals(16, spec.headDim());
        assertEquals(hp.nEmbdR(), spec.convStateFloats());
        assertEquals(hp.nEmbdS(), spec.stateFloats());
    }

    @Test
    public void kimiLinearLowRankGateNeedsBothProjections() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.ssm_g_a.weight", N_EMBD, 16));
        RecurrentResolver resolver = new RecurrentResolver(kimiLinear(), TensorCatalog.of("kimi-linear", ts));
        assertThrows(MissingTensorException.class, () -> resolver.resolve(0));
    }
}
