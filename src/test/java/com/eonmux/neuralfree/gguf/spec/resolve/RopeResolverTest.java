package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;

/**
 * Context and per-layer RoPE parameters: plain NORM, YaRN, long/short frequency factors, per-layer SWA bases
 * (gemma2), olmo2's unscaled SWA layers and qwen3vl IMROPE sections.
 */
public class RopeResolverTest {

    private static final int N_EMBD = 64;
    private static final int N_HEAD = 4;
    private static final int HEAD_DIM = N_EMBD / N_HEAD;
    private static final int N_CTX = 4096;

    private static Map<String, Object> base(String arch, int blocks) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", arch);
        m.put(arch + ".context_length", N_CTX);
        m.put(arch + ".embedding_length", N_EMBD);
        m.put(arch + ".block_count", blocks);
        m.put(arch + ".feed_forward_length", 128);
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

    private static RopeResolver rope(ModelHparams hp, List<GGUFTensorInfo> ts, int nCtxSeq) {
        return new RopeResolver(hp, TensorCatalog.of(hp.arch().ggufName(), ts), nCtxSeq);
    }

    @Test
    public void llamaNormRopeWithoutScaling() {
        ModelHparams hp = load(base("llama", 1), Arch.LLAMA);
        RopeSpec r = rope(hp, new ArrayList<GGUFTensorInfo>(), N_CTX).forLayer(0);
        assertEquals(RopeType.NORM, r.type());
        assertEquals(HEAD_DIM, r.nDims());
        assertEquals(0, r.nOffs());
        assertEquals(10000.0f, r.freqBase(), 0f);
        assertEquals(1.0f, r.freqScale(), 0f);
        assertEquals(RopeMath.thetaScale(10000.0f, HEAD_DIM), r.thetaScale(), 0f);
        assertEquals(0.0f, r.extFactor(), 0f);
        assertEquals(1.0f, r.attnFactor(), 0f);
        assertEquals(N_CTX, r.nCtxOrig());
        assertNull(r.freqFactors());
    }

    @Test
    public void yarnScalingEnablesExtrapolation() {
        Map<String, Object> m = base("llama", 1);
        m.put("llama.rope.scaling.type", "yarn");
        m.put("llama.rope.scaling.factor", 4.0f);
        m.put("llama.rope.scaling.original_context_length", 1024);
        ModelHparams hp = load(m, Arch.LLAMA);
        RopeResolver rr = rope(hp, new ArrayList<GGUFTensorInfo>(), N_CTX);
        assertEquals(0.25f, rr.freqScale(), 0f);
        assertEquals(1.0f, rr.extFactor(), 0f);
        assertEquals(RopeMath.yarnAttnFactor(1.0f, 1.0f, 0.25f, 0f, false, 1.0f), rr.attnFactor(), 0f);
        RopeSpec r = rr.forLayer(0);
        assertEquals(1024, r.nCtxOrig());
        float[] corr = RopeMath.corrDims(HEAD_DIM, 1024, 10000.0f, 32.0f, 1.0f);
        assertEquals(corr[0], r.corrLo(), 0f);
        assertEquals(corr[1], r.corrHi(), 0f);
    }

    @Test
    public void frequencyFactorsFollowTheContextLength() {
        Map<String, Object> m = base("llama", 1);
        m.put("llama.rope.scaling.original_context_length", 2048);
        ModelHparams hp = load(m, Arch.LLAMA);
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("rope_factors_long.weight", HEAD_DIM / 2));
        ts.add(t("rope_factors_short.weight", HEAD_DIM / 2));
        assertEquals("rope_factors_long.weight", rope(hp, ts, 4096).forLayer(0).freqFactors().name());
        assertEquals("rope_factors_short.weight", rope(hp, ts, 2048).forLayer(0).freqFactors().name());
        ts.add(t("rope_freqs.weight", HEAD_DIM / 2));
        assertEquals("rope_freqs.weight", rope(hp, ts, 4096).forLayer(0).freqFactors().name());
    }

    @Test
    public void gemma2SlidingLayersUseTheSwaBase() {
        Map<String, Object> m = base("gemma2", 2);
        m.put("gemma2.attention.sliding_window", 256);
        m.put("gemma2.rope.freq_base", 1000000.0f);
        m.put("gemma2.rope.freq_base_swa", 10000.0f);
        ModelHparams hp = load(m, Arch.GEMMA2);
        RopeResolver rr = rope(hp, new ArrayList<GGUFTensorInfo>(), N_CTX);
        RopeSpec swa = rr.forLayer(0);
        assertEquals(RopeType.NEOX, swa.type());
        assertEquals(10000.0f, swa.freqBase(), 0f);
        assertEquals(RopeMath.thetaScale(10000.0f, HEAD_DIM), swa.thetaScale(), 0f);
        assertEquals(1000000.0f, rr.forLayer(1).freqBase(), 0f);
    }

    @Test
    public void olmo2SlidingLayersAreUnscaled() {
        Map<String, Object> m = base("olmo2", 4);
        m.put("olmo2.attention.sliding_window", 128);
        m.put("olmo2.rope.scaling.factor", 2.0f);
        m.put("olmo2.rope.freq_base", 500000.0f);
        ModelHparams hp = load(m, Arch.OLMO2);
        RopeResolver rr = rope(hp, new ArrayList<GGUFTensorInfo>(), N_CTX);
        RopeSpec swa = rr.forLayer(0);
        assertEquals(1.0f, swa.freqScale(), 0f);
        assertEquals(500000.0f, swa.freqBase(), 0f);
        RopeSpec full = rr.forLayer(3);
        assertEquals(0.5f, full.freqScale(), 0f);
        assertEquals(500000.0f, full.freqBase(), 0f);
    }

    @Test
    public void qwen3vlImropeCarriesSections() {
        Map<String, Object> m = base("qwen3vl", 1);
        m.put("qwen3vl.rope.dimension_sections", new int[]{4, 2, 2, 0});
        ModelHparams hp = load(m, Arch.QWEN3VL);
        RopeSpec r = rope(hp, new ArrayList<GGUFTensorInfo>(), N_CTX).forLayer(0);
        assertEquals(RopeType.IMROPE, r.type());
        assertArrayEquals(new int[]{4, 2, 2, 0}, r.sections());
        assertEquals(hp.mropeTextPositions(), r.textPositions());
        assertEquals(4, r.positionsPerToken());
    }

    @Test
    public void starcoderLearnedPositionsApplyNoRope() {
        Map<String, Object> m = base("starcoder", 1);
        m.put("starcoder.attention.layer_norm_epsilon", 1e-5f);
        RopeResolver rr = rope(load(m, Arch.STARCODER), new ArrayList<GGUFTensorInfo>(), N_CTX);
        assertFalse(rr.appliesRope(0));
        assertEquals(0, rr.nRot(0));
        assertNull(rr.forLayer(0));
    }

    @Test
    public void ropelessArchitectureRejectsExplicitRope() {
        Map<String, Object> m = base("gpt2", 1);
        m.put("gpt2.attention.layer_norm_epsilon", 1e-5f);
        RopeResolver rr = rope(load(m, Arch.GPT2), new ArrayList<GGUFTensorInfo>(), N_CTX);
        assertFalse(rr.appliesRope(0));
        assertThrows(IllegalStateException.class, () -> rr.forLayer(0, HEAD_DIM, 0));
    }

    @Test
    public void contextLengthMustBePositive() {
        ModelHparams hp = load(base("llama", 1), Arch.LLAMA);
        TensorCatalog tc = TensorCatalog.of("llama", new ArrayList<GGUFTensorInfo>());
        assertThrows(IllegalArgumentException.class, () -> new RopeResolver(hp, tc, 0));
    }
}
