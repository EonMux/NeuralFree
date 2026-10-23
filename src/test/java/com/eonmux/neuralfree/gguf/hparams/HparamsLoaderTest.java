package com.eonmux.neuralfree.gguf.hparams;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchEntry;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.RopeScalingType;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.arch.UnsupportedArchException;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HparamsLoader} against map-backed metadata. Registry entries are built in the test so the loader
 * steps are exercised independently of the architecture tables.
 */
public class HparamsLoaderTest {

    private static ArchEntry entry(Arch arch, ArchHparamsHook hook) {
        return entry(arch, SupportClass.GEN, RopeType.NORM, false, hook);
    }

    private static ArchEntry entry(Arch arch, SupportClass support, RopeType rope, boolean mropeWhenSections,
                                   ArchHparamsHook hook) {
        ArchTraits t = ArchTraits.builder(arch).ropeType(rope).memoryKind(MemoryKind.KV).support(support)
                .mropeWhenSections(mropeWhenSections).build();
        return ArchEntry.of(t, hook);
    }

    private static Map<String, Object> base(String arch, int blocks) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", arch);
        m.put(arch + ".context_length", 4096);
        m.put(arch + ".embedding_length", 512);
        m.put(arch + ".block_count", blocks);
        m.put(arch + ".feed_forward_length", 1024);
        m.put(arch + ".attention.head_count", 8);
        m.put(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        return m;
    }

    private static ModelHparams load(Map<String, Object> m, ArchEntry e) {
        return HparamsLoader.load(GgufKv.ofMap(m), e);
    }

    @Test
    public void arrayValuedHeadCountKvGivesPerLayerValues() {
        Map<String, Object> m = base("llama", 4);
        m.put("llama.attention.head_count_kv", new int[] {8, 4, 2, 1});
        ModelHparams hp = load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE));
        assertArrayEquals(new int[] {8, 4, 2, 1}, hp.nHeadKvArray());
        assertEquals(4, hp.nHeadKv(1));
        assertEquals(8, hp.nGqa(3));
        assertEquals(8, hp.nHead(2));
        assertEquals(64, hp.headK(0));
        assertEquals(64 * 2, hp.nEmbdKGqa(2));
        assertEquals(64 * 8, hp.nEmbdKGqaMax());
        assertTrue(hp.isNEmbdKGqaVariable());
    }

    @Test
    public void headCountKvDefaultsToHeadCount() {
        ModelHparams hp = load(base("llama", 2), entry(Arch.LLAMA, ArchHparamsHook.NONE));
        assertArrayEquals(new int[] {8, 8}, hp.nHeadKvArray());
    }

    @Test
    public void wrongLengthHeadCountKvThrows() {
        Map<String, Object> m = base("llama", 4);
        m.put("llama.attention.head_count_kv", new int[] {8, 4, 2});
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE)));
        assertTrue(e.getMessage().contains("llama.attention.head_count_kv"));
    }

    @Test
    public void nextnBlocksAreExcludedFromTrunk() {
        Map<String, Object> m = base("glm4moe", 48);
        m.put("glm4moe.nextn_predict_layers", 1);
        ModelHparams hp = load(m, entry(Arch.GLM4_MOE, ArchHparamsHook.NONE));
        assertEquals(48, hp.nLayerAll());
        assertEquals(1, hp.nextnLayers());
        assertEquals(47, hp.nLayer());
        assertEquals(48, hp.nHeadArray().length);
    }

    @Test
    public void nextnLargerThanBlockCountThrows() {
        Map<String, Object> m = base("llama", 2);
        m.put("llama.nextn_predict_layers", 3);
        assertThrows(GgufFormatException.class, () -> load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE)));
    }

    @Test
    public void contextLengthIsRequiredForGenerativeModels() {
        Map<String, Object> m = base("llama", 2);
        m.remove("llama.context_length");
        assertThrows(GgufFormatException.class, () -> load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE)));
    }

    @Test
    public void encoderWithoutContextLengthFallsBack() {
        Map<String, Object> m = base("bert", 2);
        m.remove("bert.context_length");
        ModelHparams hp = load(m, entry(Arch.BERT, SupportClass.EMBED, RopeType.NONE, false, ArchHparamsHook.NONE));
        assertEquals(HparamsLoader.ENCODER_FALLBACK_CTX, hp.nCtxTrain());
    }

    @Test
    public void ropeScalingAndHeadDims() {
        Map<String, Object> m = base("llama", 2);
        m.put("llama.rope.scaling.type", "yarn");
        m.put("llama.rope.scaling.factor", 4.0f);
        m.put("llama.rope.scaling.original_context_length", 8192);
        m.put("llama.rope.freq_base", 500000.0f);
        ModelHparams hp = load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE));
        assertEquals(RopeScalingType.YARN, hp.ropeScalingType());
        assertEquals(0.25f, hp.ropeFreqScale(), 0f);
        assertEquals(8192, hp.nCtxOrigYarn());
        assertEquals(500000.0f, hp.ropeFreqBase(0), 0f);
        assertEquals(64, hp.nRot(0));
        assertEquals(RopeType.NORM, hp.ropeType());
    }

    @Test
    public void unknownRopeScalingTypeThrows() {
        Map<String, Object> m = base("llama", 2);
        m.put("llama.rope.scaling.type", "quadratic");
        assertThrows(GgufFormatException.class, () -> load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE)));
    }

    @Test
    public void llamaRotaryDimensionMustMatchHeadSize() {
        Map<String, Object> m = base("llama", 2);
        m.put("llama.rope.dimension_count", 32);
        assertThrows(GgufFormatException.class, () -> load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE)));
    }

    @Test
    public void hookRunsAfterGenericKeys() {
        Map<String, Object> m = base("gemma3", 6);
        m.put("gemma3.attention.sliding_window", 1024);
        ArchHparamsHook hook = (b, kv) -> {
            b.normRmsEps(kv.f32(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, 0f));
            if (HparamsSupport.readSlidingWindow(b, kv, false) && b.nSwa() > 0) {
                b.swaType(SwaType.STANDARD);
                HparamsSupport.setSwaPattern(b, 3, false);
            }
        };
        ModelHparams hp = load(m, entry(Arch.GEMMA3, hook));
        assertEquals(1e-6f, hp.normRmsEps(), 0f);
        assertEquals(1024, hp.nSwa());
        assertArrayEquals(new boolean[] {true, true, false, true, true, false}, hp.isSwaArray());
        assertTrue(hp.isSwaAny());
    }

    @Test
    public void swaTypeWithoutWindowFailsValidation() {
        ArchHparamsHook hook = (b, kv) -> b.swaType(SwaType.STANDARD);
        assertThrows(GgufFormatException.class, () -> load(base("gemma3", 2), entry(Arch.GEMMA3, hook)));
    }

    @Test
    public void expertCountsAreValidated() {
        Map<String, Object> m = base("qwen3moe", 2);
        m.put("qwen3moe.expert_count", 8);
        m.put("qwen3moe.expert_used_count", 10);
        assertThrows(GgufFormatException.class, () -> load(m, entry(Arch.QWEN3MOE, ArchHparamsHook.NONE)));
    }

    @Test
    public void moeLayersFollowDenseLead() {
        Map<String, Object> m = base("deepseek2", 3);
        m.put("deepseek2.expert_count", 4);
        m.put("deepseek2.expert_used_count", 2);
        m.put("deepseek2.leading_dense_block_count", 1);
        m.put("deepseek2.expert_gating_func", 2);
        ArchHparamsHook denseLead = (b, kv) -> b
                .nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, 0))
                .gating(HparamsSupport.readGating(kv, b.traits().defaultGating()));
        ModelHparams hp = load(m, entry(Arch.DEEPSEEK2, denseLead));
        assertArrayEquals(new boolean[] {false, true, true}, hp.moeArray());
        assertEquals(GatingFunc.SIGMOID, hp.gating());
        assertEquals(2, hp.nExpertUsedMax());
    }

    @Test
    public void graniteGraphExpertsRenormalizeTheSelectedWeights() {
        Map<String, Object> mini = base("minicpm", 2);
        mini.put("minicpm.expert_count", 8);
        mini.put("minicpm.expert_used_count", 2);
        ModelHparams minicpm = HparamsLoader.load(GgufKv.ofMap(mini), ArchRegistry.get(Arch.MINICPM));
        assertEquals(GatingFunc.SOFTMAX, minicpm.gating());
        assertTrue(minicpm.expertWeightsNorm());

        Map<String, Object> gr = base("granite", 2);
        gr.put("granite.expert_count", 8);
        gr.put("granite.expert_used_count", 2);
        gr.put("granite.logit_scale", 8.0f);
        ModelHparams granite = HparamsLoader.load(GgufKv.ofMap(gr), ArchRegistry.get(Arch.GRANITE));
        assertEquals(GatingFunc.SOFTMAX, granite.gating());
        assertTrue(granite.expertWeightsNorm());
    }

    @Test
    public void moeEveryNRuleAppliesToBertGraphOnly() {
        ArchHparamsHook everyTwo = (b, kv) -> b.moeEveryN(2);

        Map<String, Object> k2 = base("k2-horizon", 4);
        k2.put("k2-horizon.expert_count", 4);
        k2.put("k2-horizon.expert_used_count", 2);
        ModelHparams horizon = load(k2, entry(Arch.K2_HORIZON, everyTwo));
        assertArrayEquals(new boolean[] {true, true, true, true}, horizon.moeArray());

        Map<String, Object> nb = base("nomic-bert-moe", 4);
        nb.put("nomic-bert-moe.expert_count", 4);
        nb.put("nomic-bert-moe.expert_used_count", 2);
        ModelHparams nomic = load(nb, entry(Arch.NOMIC_BERT_MOE, SupportClass.EMBED, RopeType.NEOX, false, everyTwo));
        assertArrayEquals(new boolean[] {false, true, false, true}, nomic.moeArray());
    }

    @Test
    public void swaPatternArrayIsLoadedPerLayer() {
        Map<String, Object> m = base("gemma4", 4);
        m.put("gemma4.attention.sliding_window_pattern", new boolean[] {true, false, true, true, false, true});
        ArchHparamsHook hook = (b, kv) -> HparamsSupport.loadSwaPattern(b, kv, 1, false);
        ModelHparams hp = load(m, entry(Arch.GEMMA4, hook));
        assertArrayEquals(new boolean[] {true, false, true, true}, hp.isSwaArray());

        Map<String, Object> scalar = base("gemma4", 4);
        scalar.put("gemma4.attention.sliding_window_pattern", 2);
        assertArrayEquals(new boolean[] {true, false, true, false}, load(scalar, entry(Arch.GEMMA4, hook)).isSwaArray());
    }

    @Test
    public void kvOwnersFollowSharedKvRule() {
        ArchHparamsHook hook = (b, kv) -> {
            b.swaType(SwaType.STANDARD).nSwa(512);
            HparamsSupport.setSwaPattern(b, 2, false);
            b.nLayerKvFromStart(4);
        };
        ModelHparams hp = load(base("gemma3n", 6), entry(Arch.GEMMA3N, hook));
        assertArrayEquals(new int[] {0, 1, 2, 3, 2, 3}, hp.kvOwnerLayerArray());
        assertTrue(hp.hasKv(3));
        assertFalse(hp.hasKv(4));
    }

    @Test
    public void conditionalMropeAndDflashRopeTypes() {
        Map<String, Object> m = base("glm4", 2);
        m.put("glm4.rope.dimension_sections", new int[] {8, 12, 12, 0});
        ArchHparamsHook sections = (b, kv) -> HparamsSupport.readRopeSections(b, kv, false);
        ModelHparams glm = load(m, entry(Arch.GLM4, SupportClass.GEN, RopeType.NORM, true, sections));
        assertEquals(RopeType.MROPE, glm.ropeType());
        assertEquals(MropeTextPositions.SECTIONS, glm.nPosPerEmbd());

        ModelHparams plain = load(base("glm4", 2), entry(Arch.GLM4, SupportClass.GEN, RopeType.NORM, true, sections));
        assertEquals(RopeType.NORM, plain.ropeType());

        ModelHparams dflash = load(base("dflash", 2),
                entry(Arch.DFLASH, SupportClass.DRAFT, RopeType.NEOX, false, ArchHparamsHook.NONE));
        assertEquals(RopeType.NEOX, dflash.ropeType());
    }

    @Test
    public void classifierLabelsAndPooling() {
        Map<String, Object> m = base("bert", 2);
        m.put("bert.classifier.output_labels", new String[] {"neg", "neu", "pos"});
        m.put("bert.pooling_type", 2);
        m.put("bert.classifier.activation", "gelu");
        ModelHparams hp = load(m, entry(Arch.BERT, SupportClass.CLS, RopeType.NONE, false, ArchHparamsHook.NONE));
        assertEquals(3, hp.nClsOut());
        assertArrayEquals(new String[] {"neg", "neu", "pos"}, hp.classifierLabels());
        assertEquals(PoolingKind.CLS, hp.pooling());
        assertNull(hp.poolingCls());
    }

    @Test
    public void unsupportedArchitecturesAreRejected() {
        Map<String, Object> clip = base("clip", 2);
        UnsupportedArchException e = assertThrows(UnsupportedArchException.class,
                () -> HparamsLoader.load(GgufKv.ofMap(clip), ArchRegistry.get(Arch.CLIP)));
        assertTrue(e.getMessage().contains("clip"));

        Map<String, Object> unknown = base("not-an-arch", 2);
        UnsupportedArchException u = assertThrows(UnsupportedArchException.class,
                () -> HparamsLoader.load(GgufKv.ofMap(unknown)));
        assertTrue(u.getMessage().contains("not-an-arch"));
    }

    @Test
    public void architectureMismatchThrows() {
        assertThrows(GgufFormatException.class, () -> load(base("qwen2", 2), entry(Arch.LLAMA, ArchHparamsHook.NONE)));
    }

    @Test
    public void mropeTextPositionsLayout() {
        int[] dst = new int[8];
        MropeTextPositions.PPP0.fill(5, 2, dst);
        assertArrayEquals(new int[] {5, 6, 5, 6, 5, 6, 0, 0}, dst);
        MropeTextPositions.P000.fill(5, 2, dst);
        assertArrayEquals(new int[] {5, 6, 0, 0, 0, 0, 0, 0}, dst);
    }
}
