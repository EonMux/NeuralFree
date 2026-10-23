package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.PostNormRole;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
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
 * Norm placement from tensor presence: POST_ONLY (olmo2), PRE_FFN (gpt-oss), SANDWICH (gemma2), plain pre-norm,
 * the shared attention norm, the Q/K norm kinds and their order against RoPE, k2-horizon's grouped norms and the
 * mistral3 adaptive FFN norm.
 */
public class NormResolverTest {

    private static final int N_EMBD = 64;
    private static final int N_HEAD = 4;
    private static final int HEAD_DIM = N_EMBD / N_HEAD;
    private static final int N_FF = 128;
    private static final float RMS_EPS = 1e-6f;

    private static Map<String, Object> base(String arch, int blocks) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", arch);
        m.put(arch + ".context_length", 4096);
        m.put(arch + ".embedding_length", N_EMBD);
        m.put(arch + ".block_count", blocks);
        m.put(arch + ".feed_forward_length", N_FF);
        m.put(arch + ".attention.head_count", N_HEAD);
        m.put(arch + ".attention.layer_norm_rms_epsilon", RMS_EPS);
        return m;
    }

    private static ModelHparams load(Map<String, Object> m, Arch arch) {
        return HparamsLoader.load(GgufKv.ofMap(m), arch);
    }

    private static GGUFTensorInfo t(String name, long... ne) {
        return new GGUFTensorInfo(name, ne, GGMLType.F32, 0L);
    }

    private static void denseFfn(List<GGUFTensorInfo> ts, int il) {
        ts.add(t("blk." + il + ".ffn_up.weight", N_EMBD, N_FF));
        ts.add(t("blk." + il + ".ffn_down.weight", N_FF, N_EMBD));
    }

    private static NormResolver resolver(ModelHparams hp, Arch arch, List<GGUFTensorInfo> ts) {
        return new NormResolver(hp, TensorCatalog.of(arch.ggufName(), ts));
    }

    @Test
    public void llamaPreNormHasNoPostRole() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.attn_norm.weight", N_EMBD));
        ts.add(t("blk.0.ffn_norm.weight", N_EMBD));
        denseFfn(ts, 0);
        NormResolver r = resolver(load(base("llama", 1), Arch.LLAMA), Arch.LLAMA, ts);
        NormSpec n = r.resolve(0);
        assertEquals(NormKind.RMS, n.kind());
        assertEquals(RMS_EPS, n.eps(), 0f);
        assertTrue(n.attnNorm());
        assertTrue(n.ffnNorm());
        assertEquals(PostNormRole.NONE, n.postRole());
        assertFalse(n.sharedAttnNormForFfn());
        assertEquals(QkNormKind.NONE, n.qkNorm());
        assertEquals("blk.0.ffn_norm.weight", r.ffnNormTensor(0).name());
        assertNull(r.postFfnNormTensor(0));
    }

    @Test
    public void olmo2IsPostOnlyWithFullProjectionQkNorm() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.post_attention_norm.weight", N_EMBD));
        ts.add(t("blk.0.post_ffw_norm.weight", N_EMBD));
        ts.add(t("blk.0.attn_q_norm.weight", N_EMBD));
        ts.add(t("blk.0.attn_k_norm.weight", N_EMBD));
        denseFfn(ts, 0);
        NormResolver r = resolver(load(base("olmo2", 1), Arch.OLMO2), Arch.OLMO2, ts);
        NormSpec n = r.resolve(0);
        assertEquals(PostNormRole.POST_ONLY, n.postRole());
        assertFalse(n.attnNorm());
        assertFalse(n.ffnNorm());
        assertTrue(n.postAttnNorm());
        assertTrue(n.postFfnNorm());
        assertEquals(QkNormKind.RMS_FULL, n.qkNorm());
        assertNull(r.ffnNormTensor(0));
        assertEquals("blk.0.post_ffw_norm.weight", r.postFfnNormTensor(0).name());
    }

    @Test
    public void gptOssPostAttentionNormIsTheFfnInputNorm() {
        Map<String, Object> m = base("gpt-oss", 2);
        m.put("gpt-oss.expert_count", 4);
        m.put("gpt-oss.expert_used_count", 2);
        m.put("gpt-oss.expert_feed_forward_length", 32);
        m.put("gpt-oss.attention.sliding_window", 128);
        List<GGUFTensorInfo> ts = new ArrayList<>();
        for (int il = 0; il < 2; il++) {
            ts.add(t("blk." + il + ".attn_norm.weight", N_EMBD));
            ts.add(t("blk." + il + ".post_attention_norm.weight", N_EMBD));
            ts.add(t("blk." + il + ".ffn_gate_inp.weight", N_EMBD, 4));
            ts.add(t("blk." + il + ".ffn_down_exps.weight", 32, N_EMBD, 4));
        }
        NormResolver r = resolver(load(m, Arch.OPENAI_MOE), Arch.OPENAI_MOE, ts);
        for (int il = 0; il < 2; il++) {
            NormSpec n = r.resolve(il);
            assertEquals(PostNormRole.PRE_FFN, n.postRole());
            assertTrue(n.attnNorm());
            assertFalse(n.ffnNorm());
            assertTrue(n.ffnInputIsPostAttnNorm());
            assertFalse(n.sharedAttnNormForFfn());
            assertEquals("blk." + il + ".post_attention_norm.weight", r.ffnNormTensor(il).name());
        }
    }

    @Test
    public void gemma2IsSandwichWithPerHeadQkNormAbsent() {
        Map<String, Object> m = base("gemma2", 2);
        m.put("gemma2.attention.sliding_window", 256);
        List<GGUFTensorInfo> ts = new ArrayList<>();
        for (int il = 0; il < 2; il++) {
            ts.add(t("blk." + il + ".attn_norm.weight", N_EMBD));
            ts.add(t("blk." + il + ".post_attention_norm.weight", N_EMBD));
            ts.add(t("blk." + il + ".ffn_norm.weight", N_EMBD));
            ts.add(t("blk." + il + ".post_ffw_norm.weight", N_EMBD));
            denseFfn(ts, il);
        }
        NormResolver r = resolver(load(m, Arch.GEMMA2), Arch.GEMMA2, ts);
        NormSpec n = r.resolve(1);
        assertEquals(PostNormRole.SANDWICH, n.postRole());
        assertTrue(n.attnNorm());
        assertTrue(n.ffnNorm());
        assertTrue(n.postAttnNorm());
        assertTrue(n.postFfnNorm());
        assertEquals(QkNormKind.NONE, n.qkNorm());
        assertEquals("blk.1.ffn_norm.weight", r.ffnNormTensor(1).name());
        assertEquals("blk.1.post_ffw_norm.weight", r.postFfnNormTensor(1).name());
    }

    @Test
    public void missingFfnNormSharesTheAttentionNorm() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.attn_norm.weight", N_EMBD));
        denseFfn(ts, 0);
        NormResolver r = resolver(load(base("llama", 1), Arch.LLAMA), Arch.LLAMA, ts);
        NormSpec n = r.resolve(0);
        assertTrue(n.sharedAttnNormForFfn());
        assertFalse(n.ffnNorm());
        assertNull(r.ffnNormTensor(0));
    }

    @Test
    public void perHeadQkNormWeightIsRmsHead() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.attn_norm.weight", N_EMBD));
        ts.add(t("blk.0.ffn_norm.weight", N_EMBD));
        ts.add(t("blk.0.attn_q_norm.weight", HEAD_DIM));
        ts.add(t("blk.0.attn_k_norm.weight", HEAD_DIM));
        denseFfn(ts, 0);
        NormResolver r = resolver(load(base("llama", 1), Arch.LLAMA), Arch.LLAMA, ts);
        assertEquals(QkNormKind.RMS_HEAD, r.qkNorm(0));
        assertEquals(QkNormKind.RMS_HEAD, r.resolve(0).qkNorm());
    }

    @Test
    public void llamaQkNormRunsBeforeRopeWithItsBias() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.attn_norm.weight", N_EMBD));
        ts.add(t("blk.0.ffn_norm.weight", N_EMBD));
        ts.add(t("blk.0.attn_q_norm.weight", HEAD_DIM));
        ts.add(t("blk.0.attn_k_norm.weight", HEAD_DIM));
        denseFfn(ts, 0);
        NormResolver r = resolver(load(base("llama", 1), Arch.LLAMA), Arch.LLAMA, ts);
        assertFalse(r.qkNormAfterRope(0));
        assertTrue(r.qkNormBias(0));
        assertEquals(1, r.resolve(0).groups());
        assertFalse(r.resolve(0).ffnAda());
    }

    @Test
    public void mptFullWidthQkLayerNormIsLnFull() {
        Map<String, Object> m = base("mpt", 1);
        m.put("mpt.attention.layer_norm_epsilon", 1e-5f);
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.attn_norm.weight", N_EMBD));
        ts.add(t("blk.0.ffn_norm.weight", N_EMBD));
        ts.add(t("blk.0.attn_q_norm.weight", N_EMBD));
        ts.add(t("blk.0.attn_k_norm.weight", N_EMBD));
        denseFfn(ts, 0);
        NormResolver r = resolver(load(m, Arch.MPT), Arch.MPT, ts);
        assertEquals(QkNormKind.LN_FULL, r.qkNorm(0));
        assertFalse(r.qkNormAfterRope(0));
    }

    @Test
    public void k2HorizonGroupsItsBlockNormsAndNormalizesQkPerHead() {
        Map<String, Object> m = base("k2-horizon", 1);
        m.put("k2-horizon.attention.group_norm_groups", 4);
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("blk.0.attn_norm.weight", N_EMBD));
        ts.add(t("blk.0.ffn_norm.weight", N_EMBD));
        ts.add(t("blk.0.attn_q_norm.weight", HEAD_DIM));
        ts.add(t("blk.0.attn_k_norm.weight", HEAD_DIM));
        denseFfn(ts, 0);
        ModelHparams hp = load(m, Arch.K2_HORIZON);
        NormResolver r = resolver(hp, Arch.K2_HORIZON, ts);
        assertEquals(4, NormResolver.normGroups(hp));
        assertEquals(4, r.resolve(0).groups());
        assertEquals(QkNormKind.RMS_HEAD, r.qkNorm(0));
    }

    @Test
    public void ungroupedArchitecturesKeepOneNormGroup() {
        Map<String, Object> m = base("llama", 1);
        m.put("llama.attention.group_norm_groups", 4);
        assertEquals(1, NormResolver.normGroups(load(m, Arch.LLAMA)));
    }

    private static void mistral3Layer(List<GGUFTensorInfo> ts, int il, boolean adaUp) {
        ts.add(t("blk." + il + ".attn_norm.weight", N_EMBD));
        ts.add(t("blk." + il + ".ffn_norm.weight", N_EMBD));
        denseFfn(ts, il);
        ts.add(t("blk." + il + ".ffn_norm_ada_down.weight", N_EMBD, 8));
        if (adaUp) {
            ts.add(t("blk." + il + ".ffn_norm_ada_up.weight", 8, N_EMBD));
        }
    }

    @Test
    public void mistral3AdaTensorsConditionTheFfnNorm() {
        Map<String, Object> m = base("mistral3", 2);
        m.put("mistral3.realtime_delay_tokens", 3);
        List<GGUFTensorInfo> ts = new ArrayList<>();
        mistral3Layer(ts, 0, true);
        mistral3Layer(ts, 1, true);
        NormResolver r = resolver(load(m, Arch.MISTRAL3), Arch.MISTRAL3, ts);
        NormSpec n = r.resolve(1);
        assertTrue(n.ffnAda());
        assertEquals("blk.1.ffn_norm_ada_down.weight", n.ffnAdaDown().name());
        assertEquals("blk.1.ffn_norm_ada_up.weight", n.ffnAdaUp().name());
        assertEquals(3, n.ffnAdaDelay());
    }

    @Test
    public void mistral3AdaNormNeedsBothProjections() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        mistral3Layer(ts, 0, true);
        mistral3Layer(ts, 1, false);
        NormResolver r = resolver(load(base("mistral3", 2), Arch.MISTRAL3), Arch.MISTRAL3, ts);
        assertTrue(r.resolve(0).ffnAda());
        assertThrows(MissingTensorException.class, () -> r.resolve(1));
    }

    @Test
    public void mistral3RoutedLayersSkipTheAdaNorm() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        mistral3Layer(ts, 0, true);
        ts.add(t("blk.1.attn_norm.weight", N_EMBD));
        ts.add(t("blk.1.ffn_norm.weight", N_EMBD));
        ts.add(t("blk.1.ffn_gate_inp.weight", N_EMBD, 4));
        NormResolver r = resolver(load(base("mistral3", 2), Arch.MISTRAL3), Arch.MISTRAL3, ts);
        assertFalse(r.resolve(1).ffnAda());
    }
}
