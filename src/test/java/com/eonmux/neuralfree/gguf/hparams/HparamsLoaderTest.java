package com.eonmux.neuralfree.gguf.hparams;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchEntry;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.PoolerAct;
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
        return ArchEntry.of(traits(arch, support, rope, mropeWhenSections).build(), hook);
    }

    private static ArchTraits.Builder traits(Arch arch, SupportClass support, RopeType rope,
                                             boolean mropeWhenSections) {
        return ArchTraits.builder(arch).ropeType(rope).memoryKind(MemoryKind.KV).support(support)
                .mropeWhenSections(mropeWhenSections);
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
        // the registry's traits carry the bert graph family the rule keys on
        ModelHparams nomic = load(nb, ArchEntry.of(ArchRegistry.get(Arch.NOMIC_BERT_MOE).traits(), everyTwo));
        assertArrayEquals(new boolean[] {false, true, false, true}, nomic.moeArray());

        // the same arch outside the bert graph ignores the rule, as k2-horizon does
        ModelHparams ownGraph = load(nb, entry(Arch.NOMIC_BERT_MOE, SupportClass.EMBED, RopeType.NEOX, false, everyTwo));
        assertArrayEquals(new boolean[] {true, true, true, true}, ownGraph.moeArray());
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
    public void thePoolerAndSparseClassifierKeysParse() {
        Map<String, Object> m = base("bert", 2);
        m.put("bert.pooling_type", 0);
        m.put("bert.pooling.normalize", true);
        m.put("bert.classifier.scope", "token");
        m.put("bert.classifier.pooler_activation", "sigmoid");
        m.put("bert.classifier.token_direct", true);
        m.put("bert.classifier.sparse", "log1p_relu_sum");
        m.put("bert.classifier.sparse_cls_token", 2);
        m.put("bert.classifier.sparse_sep_token", 3);
        ModelHparams hp = load(m, entry(Arch.BERT, SupportClass.EMBED, RopeType.NONE, false, ArchHparamsHook.NONE));
        assertEquals(PoolerAct.SIGMOID, hp.poolerActivation());
        assertTrue(hp.classifierTokenDirect());
        assertEquals("log1p_relu_sum", hp.classifierSparse());
        assertEquals(2, hp.sparseClsToken());
        assertEquals(3, hp.sparseSepToken());
        assertTrue(hp.poolingNormalize());
    }

    @Test
    public void theTanhPoolerActivationParses() {
        Map<String, Object> m = base("bert", 2);
        m.put("bert.classifier.pooler_activation", "tanh");
        ModelHparams hp = load(m, entry(Arch.BERT, SupportClass.EMBED, RopeType.NONE, false, ArchHparamsHook.NONE));
        assertEquals(PoolerAct.TANH, hp.poolerActivation());
    }

    @Test
    public void thePoolerAndSparseClassifierKeysDefaultToTheirNeutralValues() {
        ModelHparams hp = load(base("bert", 2),
                entry(Arch.BERT, SupportClass.EMBED, RopeType.NONE, false, ArchHparamsHook.NONE));
        assertNull(hp.poolerActivation());
        assertFalse(hp.classifierTokenDirect());
        assertNull(hp.classifierSparse());
        assertEquals(101, hp.sparseClsToken());
        assertEquals(102, hp.sparseSepToken());
        assertFalse(hp.poolingNormalize());
    }

    @Test
    public void anUnknownPoolerActivationOrSparseKindRefusesByName() {
        Map<String, Object> act = base("bert", 2);
        act.put("bert.classifier.pooler_activation", "relu");
        GgufFormatException a = assertThrows(GgufFormatException.class, () -> load(act,
                entry(Arch.BERT, SupportClass.EMBED, RopeType.NONE, false, ArchHparamsHook.NONE)));
        assertTrue(a.getMessage(), a.getMessage().contains("relu"));

        Map<String, Object> sparse = base("bert", 2);
        sparse.put("bert.classifier.sparse", "log1p");
        GgufFormatException s = assertThrows(GgufFormatException.class, () -> load(sparse,
                entry(Arch.BERT, SupportClass.EMBED, RopeType.NONE, false, ArchHparamsHook.NONE)));
        assertTrue(s.getMessage(), s.getMessage().contains("log1p"));
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

    // ------------------------------------------------------------------ NTK and FoPE keys

    @Test
    public void ntkScalingReadsTheFactorAndMixedB() {
        Map<String, Object> m = base("llama", 1);
        m.put("llama.rope.scaling.type", "ntk");
        m.put("llama.rope.scaling.factor", 4.0f);
        m.put("llama.rope.scaling.mixed_b", 0.25f);
        ModelHparams hp = load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE));
        assertEquals(RopeScalingType.NTK, hp.ropeScalingType());
        assertEquals(4.0f, hp.ropeScalingFactor(), 0f);
        assertEquals(0.25f, hp.ropeScalingMixedB(), 0f);
        // the divisors carry the scale: freq_scale stays 1
        assertEquals(1.0f, hp.ropeFreqScale(), 0f);
    }

    @Test
    public void ntkScalingWithoutFactorIsRefused() {
        Map<String, Object> m = base("llama", 1);
        m.put("llama.rope.scaling.type", "ntk");
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE)));
        assertTrue(e.getMessage(), e.getMessage().contains("llama.rope.scaling.factor"));
    }

    @Test
    public void fopeKeysLoad() {
        Map<String, Object> m = base("llama", 1);
        m.put("llama.rope.fope.use", true);
        m.put("llama.rope.fope.sep_head", true);
        m.put("llama.rope.fope.num_inv_freq", 16);
        m.put("llama.rope.fope.init_factor", 0.1f);
        ModelHparams hp = load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE));
        assertTrue(hp.ropeFope());
        assertTrue(hp.ropeFopeSepHead());
        assertEquals(16, hp.ropeFopeNumInvFreq());
        assertEquals(0.1f, hp.ropeFopeInitFactor(), 0f);
        // the tuning keys without the use flag are a malformed file
        Map<String, Object> bad = base("llama", 1);
        bad.put("llama.rope.fope.sep_head", true);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> load(bad, entry(Arch.LLAMA, ArchHparamsHook.NONE)));
        assertTrue(e.getMessage(), e.getMessage().contains("llama.rope.fope.use"));
    }

    @Test
    public void perGroupTopKFlagLoads() {
        Map<String, Object> m = grouped("llama");
        m.put("llama.expert_group_used_count", 2);
        m.put("llama.expert_per_group_top_k", true);
        ModelHparams hp = load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE));
        assertTrue(hp.moePerGroupTopK());
        assertEquals(2, hp.nGroupUsed());
    }

    /** Every group contributes, so a file without a kept-group count keeps all of them. */
    @Test
    public void perGroupTopKDefaultsTheKeptGroupsToAllOfThem() {
        Map<String, Object> m = grouped("llama");
        m.put("llama.expert_per_group_top_k", true);
        ModelHparams hp = load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE));
        assertEquals(2, hp.nExpertGroups());
        assertEquals(2, hp.nGroupUsed());
    }

    @Test
    public void perGroupTopKWithoutGroupsIsRefused() {
        Map<String, Object> m = base("llama", 1);
        m.put("llama.expert_count", 4);
        m.put("llama.expert_used_count", 2);
        m.put("llama.expert_per_group_top_k", true);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> load(m, entry(Arch.LLAMA, ArchHparamsHook.NONE)));
        assertTrue(e.getMessage(), e.getMessage().contains("llama.expert_per_group_top_k"));
    }

    /** The expert keys of a two-group router over four experts. */
    private static Map<String, Object> grouped(String arch) {
        Map<String, Object> m = base(arch, 1);
        m.put(arch + ".expert_count", 4);
        m.put(arch + ".expert_used_count", 2);
        m.put(arch + ".expert_group_count", 2);
        return m;
    }

    @Test
    public void mropeTextPositionsLayout() {
        int[] dst = new int[8];
        MropeTextPositions.PPP0.fill(5, 2, dst);
        assertArrayEquals(new int[] {5, 6, 5, 6, 5, 6, 0, 0}, dst);
        MropeTextPositions.P000.fill(5, 2, dst);
        assertArrayEquals(new int[] {5, 6, 0, 0, 0, 0, 0, 0}, dst);
    }

    @Test
    public void decisionTypeTemperaturesAndScoreBiasParse() {
        Map<String, Object> m = base("vllm-decision2", 2);
        m.put("vllm-decision2.decision.type", "vllm-decision2");
        m.put("vllm-decision2.decision.temperature.choice", 0.6f);
        m.put("vllm-decision2.decision.temperature.choice.3_5", 0.7f);
        m.put("vllm-decision2.decision.temperature.score", 0.5f);
        m.put("vllm-decision2.decision.score_bias.4.0", -1.0f);
        m.put("vllm-decision2.decision.score_bias.4.3", 1.0f);
        ModelHparams hp = load(m, entry(Arch.VLLM_DECISION2, SupportClass.CLS, RopeType.NEOX, false,
                ArchHparamsHook.NONE));
        assertEquals("vllm-decision2", hp.decisionType());
        assertEquals(3, hp.decisionTemperatures().size());
        assertEquals(0.7f, hp.decisionTemperatures().get("choice.3_5"), 1e-6f);
        assertEquals(1, hp.decisionScoreBias().size());
        assertArrayEquals(new float[] {-1.0f, 0.0f, 0.0f, 1.0f}, hp.decisionScoreBias().get(4), 1e-6f);
    }

    @Test
    public void decisionHeadMetadataDefaultsToEmptyAndNull() {
        ModelHparams hp = load(base("bert", 2),
                entry(Arch.BERT, SupportClass.EMBED, RopeType.NONE, false, ArchHparamsHook.NONE));
        assertNull(hp.decisionType());
        assertTrue(hp.decisionTemperatures().isEmpty());
        assertTrue(hp.decisionScoreBias().isEmpty());
    }

    @Test
    public void decisionHeadMetadataIsImmutable() {
        Map<String, Object> m = base("vllm-decision2", 2);
        m.put("vllm-decision2.decision.temperature.choice", 0.6f);
        m.put("vllm-decision2.decision.score_bias.3.0", 0.25f);
        ModelHparams hp = load(m, entry(Arch.VLLM_DECISION2, SupportClass.CLS, RopeType.NEOX, false,
                ArchHparamsHook.NONE));
        assertThrows(UnsupportedOperationException.class, () -> hp.decisionTemperatures().put("noul", 1.0f));
        assertThrows(UnsupportedOperationException.class, () -> hp.decisionScoreBias().put(3, new float[3]));
    }

    @Test
    public void anInvalidDecisionTemperatureRefusesNamingTheKey() {
        Map<String, Object> m = base("vllm-decision2", 2);
        m.put("vllm-decision2.decision.temperature.choice", 0.0f);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> load(m, entry(Arch.VLLM_DECISION2, SupportClass.CLS, RopeType.NEOX, false,
                        ArchHparamsHook.NONE)));
        assertTrue(e.getMessage(), e.getMessage().contains("invalid decision temperature"));
        assertTrue(e.getMessage(), e.getMessage().contains("vllm-decision2.decision.temperature.choice"));
    }

    @Test
    public void invalidDecisionScoreBiasKeysRefuseNamingTheKey() {
        for (String bad : new String[] {"decision.score_bias.4.4", "decision.score_bias.1.0",
                "decision.score_bias.256.0", "decision.score_bias.x.0", "decision.score_bias.4"}) {
            Map<String, Object> m = base("vllm-decision2", 2);
            m.put("vllm-decision2." + bad, 1.0f);
            GgufFormatException e = assertThrows(GgufFormatException.class,
                    () -> load(m, entry(Arch.VLLM_DECISION2, SupportClass.CLS, RopeType.NEOX, false,
                            ArchHparamsHook.NONE)));
            assertTrue(bad, e.getMessage().contains("invalid decision score bias: vllm-decision2." + bad));
        }
    }
}
