package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
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
 * Attention resolution (heads, SWA, softcap, sinks, KQ scale, gates, deci linear replacement) and the residual,
 * head and embedding traces of gemma2, deci and minimax-01.
 */
public class AttentionResolverTest {

    private static final int N_EMBD = 64;
    private static final int N_HEAD = 4;
    private static final int HEAD_DIM = N_EMBD / N_HEAD;
    private static final int N_FF = 128;
    private static final int N_VOCAB = 100;
    private static final int N_CTX = 4096;

    private static Map<String, Object> base(String arch, int blocks) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", arch);
        m.put(arch + ".context_length", N_CTX);
        m.put(arch + ".embedding_length", N_EMBD);
        m.put(arch + ".block_count", blocks);
        m.put(arch + ".feed_forward_length", N_FF);
        m.put(arch + ".attention.head_count", N_HEAD);
        m.put(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        return m;
    }

    private static ModelHparams load(Map<String, Object> m, Arch arch) {
        return HparamsLoader.load(GgufKv.ofMap(m), arch);
    }

    private static GGUFTensorInfo t(String name, long... ne) {
        return new GGUFTensorInfo(name, ne, GGMLType.F32, 0L);
    }

    private static void attention(List<GGUFTensorInfo> ts, int il, int qRows, int kvRows) {
        ts.add(t("blk." + il + ".attn_norm.weight", N_EMBD));
        ts.add(t("blk." + il + ".attn_q.weight", N_EMBD, qRows));
        ts.add(t("blk." + il + ".attn_k.weight", N_EMBD, kvRows));
        ts.add(t("blk." + il + ".attn_v.weight", N_EMBD, kvRows));
        ts.add(t("blk." + il + ".attn_output.weight", qRows, N_EMBD));
    }

    private static AttentionResolver resolver(ModelHparams hp, TensorCatalog tc) {
        return new AttentionResolver(hp, tc, new RopeResolver(hp, tc, N_CTX), new NormResolver(hp, tc));
    }

    private static TensorCatalog catalog(Arch arch, List<GGUFTensorInfo> ts) {
        return TensorCatalog.of(arch.ggufName(), ts);
    }

    @Test
    public void llamaGroupedQueryAttention() {
        Map<String, Object> m = base("llama", 1);
        m.put("llama.attention.head_count_kv", 2);
        List<GGUFTensorInfo> ts = new ArrayList<>();
        attention(ts, 0, N_EMBD, 2 * HEAD_DIM);
        ModelHparams hp = load(m, Arch.LLAMA);
        AttentionResolver r = resolver(hp, catalog(Arch.LLAMA, ts));
        assertEquals(MixerKind.ATTENTION, r.attentionKind(0));
        AttentionSpec a = r.resolve(0);
        assertEquals(N_HEAD, a.nHead());
        assertEquals(2, a.nHeadKv());
        assertEquals(HEAD_DIM, a.headDimK());
        assertEquals(HEAD_DIM, a.nRot());
        assertFalse(a.noRope());
        assertTrue(a.causal());
        assertFalse(a.isSwaLayer());
        assertEquals(AttnGateKind.NONE, a.gate());
        assertEquals(0f, a.softcap(), 0f);
        assertFalse(a.sinks());
        assertTrue(a.usesMemory());
        assertEquals(0, a.kvOwnerLayer());
        assertEquals(1.0f / (float) Math.sqrt(HEAD_DIM), a.kqScale(), 0f);
    }

    @Test
    public void gemma2SandwichSoftcapsAndSlidingWindow() {
        Map<String, Object> m = base("gemma2", 2);
        m.put("gemma2.attention.sliding_window", 256);
        m.put("gemma2.attn_logit_softcapping", 50.0f);
        m.put("gemma2.final_logit_softcapping", 30.0f);
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("token_embd.weight", N_EMBD, N_VOCAB));
        ts.add(t("output_norm.weight", N_EMBD));
        for (int il = 0; il < 2; il++) {
            attention(ts, il, N_EMBD, N_EMBD);
            ts.add(t("blk." + il + ".post_attention_norm.weight", N_EMBD));
            ts.add(t("blk." + il + ".ffn_norm.weight", N_EMBD));
            ts.add(t("blk." + il + ".post_ffw_norm.weight", N_EMBD));
            ts.add(t("blk." + il + ".ffn_up.weight", N_EMBD, N_FF));
            ts.add(t("blk." + il + ".ffn_down.weight", N_FF, N_EMBD));
        }
        ModelHparams hp = load(m, Arch.GEMMA2);
        TensorCatalog tc = catalog(Arch.GEMMA2, ts);
        AttentionResolver r = resolver(hp, tc);

        AttentionSpec swa = r.resolve(0);
        assertTrue(swa.isSwaLayer());
        assertEquals(SwaType.STANDARD, swa.swaType());
        assertEquals(256, swa.nSwa());
        assertEquals(50.0f, swa.softcap(), 0f);
        assertEquals(1.0f / (float) Math.sqrt(HEAD_DIM), swa.kqScale(), 1e-7f);
        AttentionSpec full = r.resolve(1);
        assertFalse(full.isSwaLayer());
        assertEquals(50.0f, full.softcap(), 0f);

        HeadSpec head = new HeadResolver(hp, tc).resolve();
        assertEquals(30.0f, head.finalSoftcap(), 0f);
        assertTrue(head.tied());
        assertEquals("output_norm.weight", head.outputNorm().name());
        EmbeddingSpec embd = new EmbeddingResolver(hp, tc).resolve();
        assertEquals((float) Math.sqrt(N_EMBD), embd.embdScale(), 1e-6f);
    }

    @Test
    public void gptOssSinksAndRotaryKqScale() {
        int headK = 32;
        int nRot = 16;
        Map<String, Object> m = base("gpt-oss", 2);
        m.put("gpt-oss.attention.head_count_kv", 2);
        m.put("gpt-oss.attention.key_length", headK);
        m.put("gpt-oss.attention.value_length", headK);
        m.put("gpt-oss.rope.dimension_count", nRot);
        m.put("gpt-oss.expert_count", 4);
        m.put("gpt-oss.expert_used_count", 2);
        m.put("gpt-oss.expert_feed_forward_length", 32);
        m.put("gpt-oss.attention.sliding_window", 128);
        List<GGUFTensorInfo> ts = new ArrayList<>();
        for (int il = 0; il < 2; il++) {
            attention(ts, il, N_HEAD * headK, 2 * headK);
            ts.add(t("blk." + il + ".attn_sinks.weight", N_HEAD));
            ts.add(t("blk." + il + ".post_attention_norm.weight", N_EMBD));
        }
        AttentionResolver r = resolver(load(m, Arch.OPENAI_MOE), catalog(Arch.OPENAI_MOE, ts));
        AttentionSpec a0 = r.resolve(0);
        assertTrue(a0.sinks());
        assertTrue(a0.isSwaLayer());
        assertEquals(nRot, a0.nRot());
        assertEquals(1.0f / (float) Math.sqrt(nRot), a0.kqScale(), 0f);
        assertFalse(r.resolve(1).isSwaLayer());
    }

    @Test
    public void deciLayerVariants() {
        Map<String, Object> m = base("deci", 3);
        m.put("deci.attention.head_count", new int[]{N_HEAD, 0, N_HEAD});
        m.put("deci.attention.head_count_kv", new int[]{2, 0, 0});
        m.put("deci.feed_forward_length", new int[]{N_FF, N_FF, 0});
        List<GGUFTensorInfo> ts = new ArrayList<>();
        attention(ts, 0, N_EMBD, 2 * HEAD_DIM);
        ts.add(t("blk.2.attn_norm.weight", N_EMBD));
        ts.add(t("blk.2.attn_output.weight", N_EMBD, N_EMBD));
        ModelHparams hp = load(m, Arch.DECI);
        TensorCatalog tc = catalog(Arch.DECI, ts);
        AttentionResolver r = resolver(hp, tc);

        assertEquals(MixerKind.ATTENTION, r.attentionKind(0));
        assertEquals(MixerKind.NONE, r.attentionKind(1));
        assertNull(r.resolve(1));
        assertEquals(MixerKind.ATTENTION_LINEAR_REPLACE, r.attentionKind(2));
        AttentionSpec lin = r.resolve(2);
        assertTrue(lin.linearReplace());
        assertTrue(lin.noRope());
        assertEquals(0, lin.nHeadKv());

        ResidualResolver res = new ResidualResolver(hp, tc);
        assertFalse(res.resolve(0).discardLayerOutput());
        assertTrue(res.resolve(0).mixerResidual());
        assertFalse(res.resolve(1).mixerResidual());
        assertTrue(res.resolve(2).discardLayerOutput());
        assertTrue(res.resolve(2).mixerResidual());
    }

    @Test
    public void minimax01ResidualScalesTheNormedInput() {
        float scale = 1.5f;
        Map<String, Object> m = base("minimax-01", 2);
        m.put("minimax-01.residual_scale", scale);
        m.put("minimax-01.full_attention_interval", 2);
        ModelHparams hp = load(m, Arch.MINIMAX_01);
        ResidualResolver res = new ResidualResolver(hp, catalog(Arch.MINIMAX_01, new ArrayList<GGUFTensorInfo>()));
        for (int il = 0; il < 2; il++) {
            ResidualSpec s = res.resolve(il);
            assertEquals(ResidualKind.NORMED_INPUT_SCALED, s.kind());
            assertEquals(scale, s.residualScale(), 0f);
        }
    }

    @Test
    public void gateKindFollowsTheTensorShape() {
        ModelHparams hp = load(base("llama", 3), Arch.LLAMA);
        List<GGUFTensorInfo> ts = new ArrayList<>();
        attention(ts, 0, N_EMBD, N_EMBD);
        ts.add(t("blk.0.attn_gate.weight", N_EMBD, N_HEAD));
        attention(ts, 1, N_EMBD, N_EMBD);
        ts.add(t("blk.1.attn_gate.weight", N_EMBD, N_EMBD));
        attention(ts, 2, 2 * N_EMBD, N_EMBD);
        AttentionResolver r = resolver(hp, catalog(Arch.LLAMA, ts));
        assertEquals(AttnGateKind.PER_HEAD_SIGMOID, r.gate(0, N_HEAD, HEAD_DIM));
        assertEquals(AttnGateKind.ELEMENTWISE_SIGMOID, r.gate(1, N_HEAD, HEAD_DIM));
        assertEquals(AttnGateKind.Q_INTERLEAVED, r.gate(2, N_HEAD, HEAD_DIM));
    }

    @Test
    public void outputBiasIsAddedOnlyWhereTheGraphUsesIt() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        attention(ts, 0, N_EMBD, N_EMBD);
        ts.add(t("blk.0.attn_output.bias", N_EMBD));
        AttentionSpec llama = resolver(load(base("llama", 1), Arch.LLAMA), catalog(Arch.LLAMA, ts)).resolve(0);
        assertTrue(llama.outBias());
        AttentionSpec refact = resolver(load(base("refact", 1), Arch.REFACT), catalog(Arch.REFACT, ts)).resolve(0);
        assertFalse(refact.outBias());
        assertFalse(refact.qkNormAfterRope());
    }

    @Test
    public void k2HorizonGateIsAPerElementBase2Softplus() {
        ModelHparams hp = load(base("k2-horizon", 1), Arch.K2_HORIZON);
        List<GGUFTensorInfo> ts = new ArrayList<>();
        attention(ts, 0, N_EMBD, N_EMBD);
        ts.add(t("blk.0.attn_gate.weight", N_EMBD, N_EMBD));
        assertEquals(AttnGateKind.SOFTPLUS_BASE2,
                resolver(hp, catalog(Arch.K2_HORIZON, ts)).gate(0, N_HEAD, HEAD_DIM));

        List<GGUFTensorInfo> perHead = new ArrayList<>();
        attention(perHead, 0, N_EMBD, N_EMBD);
        perHead.add(t("blk.0.attn_gate.weight", N_EMBD, N_HEAD));
        AttentionResolver r = resolver(hp, catalog(Arch.K2_HORIZON, perHead));
        assertThrows(IllegalStateException.class, () -> r.gate(0, N_HEAD, HEAD_DIM));
    }
}
