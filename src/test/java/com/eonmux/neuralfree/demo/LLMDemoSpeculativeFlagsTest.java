package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The speculative decoding flags of {@link LLMDemo} ({@link SpeculativeFlags}) in llama.cpp's and vLLM's
 * spellings reach the engine configuration: the type list, each n-gram type's sizes, the lookup cache files, the
 * suffix limits, the draft context's KV types and ubatch and the verification mode; invalid values are usage
 * errors naming their flag.
 */
public class LLMDemoSpeculativeFlagsTest {

    private static final String[] SPECULATIVE_PROPS = {
            EngineConfig.PROP_DRAFT, EngineConfig.PROP_DRAFT_TYPE, EngineConfig.PROP_MTP,
            SpeculativeOptions.PROP_VERIFY, SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.n",
            SpeculativeOptions.PROP_DFLASH_DEVICE
    };

    private final Map<String, String> saved = new HashMap<>();

    @Before
    public void clearSpeculativeProperties() {
        for (String key : SPECULATIVE_PROPS) {
            saved.put(key, System.getProperty(key));
            System.clearProperty(key);
        }
    }

    @After
    public void restoreSpeculativeProperties() {
        for (Map.Entry<String, String> e : saved.entrySet()) {
            if (e.getValue() == null) {
                System.clearProperty(e.getKey());
            } else {
                System.setProperty(e.getKey(), e.getValue());
            }
        }
    }

    private static EngineConfig engine(String... args) throws UsageException {
        return LLMDemoOptions.parse(args).engine;
    }

    private static SpeculativeOptions options(String... args) throws UsageException {
        return engine(args).speculativeOptions();
    }

    private static void assertRefused(String flag, String... args) {
        UsageException e = assertThrows(UsageException.class, () -> engine(args));
        assertTrue(e.getMessage(), e.getMessage().startsWith(flag));
    }

    @Test
    public void draftBackendSamplingChoosesWhereADflashDraftDrafts() throws UsageException {
        assertTrue(options().dflashOnDevice());
        assertFalse(options("--no-spec-draft-backend-sampling").dflashOnDevice());
        assertTrue(options("--no-spec-draft-backend-sampling", "--spec-draft-backend-sampling").dflashOnDevice());
    }

    /** The draft-tree flags combine with the drafter-side draft-head and W12 flags in one command line. */
    @Test
    public void draftTreeFlagsCombineWithTheDrafterPlacementFlags() throws UsageException {
        SpeculativeOptions o = options("--spec-draft-tree", "--spec-draft-tree-nodes", "8",
                "--spec-tree-temp", "0.5", "--w12", "--draft-head-vocab", "1024");
        assertTrue(o.treeVerify());
        assertEquals(8, o.treeNodes());
        assertEquals(0.5, o.treeTemp(), 1e-12);
        assertTrue(o.w12());
        assertEquals(1024, o.draftHead().vocabSize());
        assertRefused("--spec-draft-tree-nodes", "--spec-draft-tree-nodes", "64");
        assertRefused("--spec-draft-tree-temp", "--spec-draft-tree-temp", "0");
    }

    @Test
    public void heterogeneousVocabDraftingIsASwitch() throws UsageException {
        assertFalse(options("--draft", "draft.gguf").heterogeneousVocab());
        assertTrue(options("--draft", "draft.gguf", "--spec-draft-heterogeneous-vocab").heterogeneousVocab());
        assertFalse(options("--draft", "draft.gguf", "--spec-draft-heterogeneous-vocab",
                "--no-spec-draft-heterogeneous-vocab").heterogeneousVocab());
        assertRefused("neuralfree.draft.heterogeneous.vocab",
                "--spec-draft-heterogeneous-vocab");
    }

    @Test
    public void backendSamplingTurnsEverySelectionOnAndItsAbsenceOff() throws UsageException {
        System.setProperty(GenerationLoop.PROP_SAMPLING_DEVICE, "false");
        EngineConfig on = engine("--backend-sampling");
        assertTrue(on.speculativeOptions().dflashOnDevice());
        assertTrue(on.speculativeOptions().verifyOnDevice());
        assertEquals("false", System.getProperty(GenerationLoop.PROP_SAMPLING_DEVICE));
        EngineConfig off = engine("--no-backend-sampling");
        assertFalse(off.speculativeOptions().dflashOnDevice());
        assertFalse(off.speculativeOptions().verifyOnDevice());
        assertEquals("false", System.getProperty(GenerationLoop.PROP_SAMPLING_DEVICE));
    }

    @Test
    public void aGranularFlagOverridesBackendSampling() throws UsageException {
        assertFalse(options("--backend-sampling", "--no-spec-draft-backend-sampling").dflashOnDevice());
        assertTrue(options("--no-backend-sampling", "--spec-draft-backend-sampling").dflashOnDevice());
    }

    @Test
    public void specTypeTakesAListAndRepeatedFlagsAppendInLlamaCppOrder() throws UsageException {
        EngineConfig c = engine("--spec-type", "suffix,ngram-simple", "--spec-type", "ngram-mod");
        assertEquals(List.of(DraftType.NGRAM, DraftType.NGRAM_MOD, DraftType.SUFFIX), c.draftTypes());
        assertTrue(c.speculative());
        EngineConfig withFile = engine("--draft", "draft.gguf", "--spec-type", "draft-simple,ngram-map-k");
        assertEquals(List.of(DraftType.NGRAM_MAP_K, DraftType.MODEL), withFile.draftTypes());
        assertEquals(DraftType.MODEL, withFile.draftType());
    }

    @Test
    public void aFlagListReplacesThePropertyList() throws UsageException {
        System.setProperty(EngineConfig.PROP_DRAFT_TYPE, "suffix");
        assertEquals(List.of(DraftType.NGRAM), engine("--spec-type", "ngram-simple").draftTypes());
        assertEquals(List.of(DraftType.SUFFIX), engine().draftTypes());
    }

    @Test
    public void theDraftFileNeedsAFileTypeInTheList() {
        UsageException e = assertThrows(UsageException.class,
                () -> engine("--draft", "draft.gguf", "--spec-type", "suffix"));
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_DRAFT));
        assertRefused("--spec-type", "--spec-type", "draft-simple,draft-mtp");
    }

    @Test
    public void eachNgramTypeTakesItsSizes() throws UsageException {
        SpeculativeOptions o = options("--spec-type", "ngram-simple,ngram-map-k4v,ngram-mod",
                "--spec-ngram-simple-size-n", "4", "--spec-ngram-simple-size-m", "12",
                "--spec-ngram-map-k4v-size-n", "2", "--spec-ngram-map-k4v-size-m", "6",
                "--spec-ngram-map-k4v-min-hits", "3",
                "--spec-ngram-mod-n-match", "5", "--spec-ngram-mod-n-max", "20", "--spec-ngram-mod-n-min", "2");
        NgramDrafter.Options simple = o.ngramOptions(NgramDrafter.Kind.SIMPLE);
        assertEquals(4, simple.sizeN());
        assertEquals(12, simple.sizeM());
        NgramDrafter.Options k4v = o.ngramOptions(NgramDrafter.Kind.MAP_K4V);
        assertEquals(2, k4v.sizeN());
        assertEquals(6, k4v.sizeM());
        assertEquals(3, k4v.minHits());
        NgramDrafter.Options mod = o.ngramOptions(NgramDrafter.Kind.MOD);
        assertEquals(5, mod.sizeN());
        assertEquals(20, mod.sizeM());
        assertEquals(2, mod.draftMin());
        assertEquals("untouched kinds keep the defaults", NgramDrafter.DEFAULT_SIZE_N,
                o.ngramOptions(NgramDrafter.Kind.MAP_K).sizeN());
    }

    @Test
    public void theFlagsOverrideTheSizeProperties() throws UsageException {
        System.setProperty(SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.n", "5");
        assertEquals(5, options().ngramOptions(NgramDrafter.Kind.SIMPLE).sizeN());
        assertEquals(6, options("--spec-ngram-simple-size-n", "6").ngramOptions(NgramDrafter.Kind.SIMPLE).sizeN());
    }

    @Test
    public void invalidSizesAreRefusedNamingTheFlag() throws UsageException {
        assertRefused("--spec-ngram-simple-min-hits", "--spec-ngram-simple-min-hits", "2");
        assertRefused("--spec-ngram-simple-size-m", "--spec-ngram-simple-size-m", "500");
        assertRefused("--spec-ngram-map-k-size-n", "--spec-ngram-map-k-size-n", "0");
        assertRefused("--spec-ngram-mod-n-min", "--spec-ngram-mod-n-min", "x");
        // ngram-simple's lookup pattern needs two tokens; the range is checked when the sizes combine
        assertRefused("--spec-ngram-simple-*", "--spec-ngram-simple-size-n", "1");
        // a floor above the cap is llama.cpp's: a chain of the full n-max is kept
        assertEquals(6, options("--spec-ngram-mod-n-max", "4", "--spec-ngram-mod-n-min", "6")
                .ngramOptions(NgramDrafter.Kind.MOD).draftMin());
        assertRefused("--spec-ngram-simple-*", "--spec-ngram-simple-size-n", "3", "--spec-ngram-simple-size-n-min",
                "4");
    }

    @Test
    public void thePromptLookupRangeHasLlamaCppAndVllmSpellings() throws UsageException {
        NgramDrafter.Options range = options("--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "5",
                "--spec-ngram-simple-size-n-min", "2").ngramOptions(NgramDrafter.Kind.SIMPLE);
        assertEquals(2, range.sizeNMin());
        assertEquals(5, range.sizeN());
        NgramDrafter.Options v0 = options("--ngram-prompt-lookup-max", "6", "--ngram-prompt-lookup-min", "1")
                .ngramOptions(NgramDrafter.Kind.SIMPLE);
        assertEquals(1, v0.sizeNMin());
        assertEquals(6, v0.sizeN());
        assertEquals("a single size follows -size-n", 7, options("--spec-ngram-simple-size-n", "7")
                .ngramOptions(NgramDrafter.Kind.SIMPLE).sizeNMin());
        assertRefused("--spec-ngram-simple-size-n-min", "--spec-ngram-simple-size-n-min", "0");
    }

    @Test
    public void speculativeConfigJsonSetsWhatItsFlagsSet() throws UsageException {
        EngineConfig ngram = engine("--speculative-config", "{\"method\": \"ngram\", \"num_speculative_tokens\": 5,"
                + " \"prompt_lookup_max\": 4, \"prompt_lookup_min\": 2}");
        assertEquals(List.of(DraftType.NGRAM), ngram.draftTypes());
        assertEquals(5, ngram.draftMax());
        NgramDrafter.Options simple = ngram.speculativeOptions().ngramOptions(NgramDrafter.Kind.SIMPLE);
        assertEquals(List.of(2, 4), List.of(simple.sizeNMin(), simple.sizeN()));

        EngineConfig suffix = engine("--speculative-config", "{\"method\": \"suffix\","
                + " \"suffix_decoding_max_tree_depth\": 24, \"suffix_decoding_max_cached_requests\": 7,"
                + " \"suffix_decoding_max_spec_factor\": 2, \"suffix_decoding_min_token_prob\": 0.2}");
        assertEquals(List.of(DraftType.SUFFIX), suffix.draftTypes());
        assertEquals("suffix decoding defaults the draft max to the tree depth, as vLLM", 24, suffix.draftMax());
        assertEquals(24, suffix.speculativeOptions().suffixOptions().maxTreeDepth());
        assertEquals(7, suffix.speculativeOptions().suffixOptions().maxCachedRequests());
        assertEquals(2.0, suffix.speculativeOptions().suffixOptions().maxSpecFactor(), 0.0);
        assertEquals(0.2, suffix.speculativeOptions().suffixOptions().minTokenProb(), 0.0);

        EngineConfig file = engine("--speculative-config", "{\"method\": \"eagle3\", \"model\": \"eagle.gguf\","
                + " \"num_speculative_tokens\": 3, \"rejection_sample_method\": \"standard\"}");
        assertEquals("eagle.gguf", file.draftModelPath());
        assertEquals(DraftType.AUTO, file.draftType());
        assertEquals(VerifyMode.REJECTION, file.speculativeOptions().verifyMode());
        assertEquals(DraftType.MTP, engine("--speculative-config", "{\"method\": \"deepseek_mtp\"}").draftType());
        assertEquals(DraftType.MODEL, engine("--speculative-config",
                "{\"method\": \"draft_model\", \"model\": \"small.gguf\"}").draftType());
        assertEquals("a later flag overrides a key", 9, engine("--speculative-config",
                "{\"method\": \"ngram\", \"num_speculative_tokens\": 5}", "--draft-max", "9").draftMax());
    }

    @Test
    public void speculativeConfigRefusalsNameTheKey() {
        assertRefused("--speculative-config key 'draft_tensor_parallel_size'", "--speculative-config",
                "{\"method\": \"ngram\", \"draft_tensor_parallel_size\": 1}");
        assertRefused("--speculative-config key 'enforce_eager'", "--speculative-config",
                "{\"method\": \"ngram\", \"enforce_eager\": true}");
        assertRefused("--speculative-config num_speculative_tokens=0: --spec-draft-n-max", "--speculative-config",
                "{\"method\": \"ngram\", \"num_speculative_tokens\": 0}");
        assertRefused("--speculative-config prompt_lookup_min=", "--speculative-config",
                "{\"method\": \"ngram\", \"prompt_lookup_max\": 2, \"prompt_lookup_min\": 3}");
        assertRefused("--speculative-config method ngram_gpu", "--speculative-config", "{\"method\": \"ngram_gpu\"}");
        assertRefused("--speculative-config is not valid JSON", "--speculative-config", "{method: ngram}");
    }

    @Test
    public void theSuffixRequestCacheTakesItsFlag() throws UsageException {
        assertEquals(3, options("--spec-type", "suffix", "--spec-suffix-max-cached-requests", "3")
                .suffixOptions().maxCachedRequests());
        assertEquals(0, options("--spec-suffix-max-cached-requests", "0").suffixOptions().maxCachedRequests());
        assertRefused("--spec-suffix-max-cached-requests", "--spec-suffix-max-cached-requests", "-1");
    }

    @Test
    public void theLookupCacheFilesNeedTheNgramCacheType() throws UsageException {
        SpeculativeOptions o = options("--spec-type", "ngram-cache", "-lcs", "static.bin",
                "--lookup-cache-dynamic", "dynamic.bin");
        assertEquals("static.bin", o.lookupCacheStatic());
        assertEquals("dynamic.bin", o.lookupCacheDynamic());
        assertEquals("dynamic.bin", options("--spec-type", "ngram-cache", "-lcd", "dynamic.bin")
                .lookupCacheDynamic());
        UsageException e = assertThrows(UsageException.class,
                () -> engine("--spec-type", "ngram-simple", "--lookup-cache-static", "static.bin"));
        assertTrue(e.getMessage(), e.getMessage().contains("ngram-cache"));
    }

    @Test
    public void theSuffixLimitsReachTheSuffixDrafter() throws UsageException {
        SpeculativeOptions o = options("--spec-type", "suffix", "--spec-suffix-max-tree-depth", "32",
                "--spec-suffix-min-token-prob", "0.25");
        assertEquals(32, o.suffixOptions().maxTreeDepth());
        assertEquals(0.25, o.suffixOptions().minTokenProb(), 0.0);
        assertRefused("--spec-suffix-max-tree-depth", "--spec-suffix-max-tree-depth", "0");
    }

    @Test
    public void theDraftContextTakesItsKvTypesAndUbatch() throws UsageException {
        EngineConfig c = engine("-ctkd", "q8_0", "--spec-draft-type-v", "f32", "-ubd", "16");
        assertEquals(KvDType.Q8_0, c.memory().draftTypeK());
        assertEquals(KvDType.F32, c.memory().draftTypeV());
        assertEquals(16, c.memory().draftUbatch());
        assertRefused("-ctkd", "-ctkd", "q9");
    }

    @Test
    public void theVerificationModeHasLlamaCppAndVllmSpellings() throws UsageException {
        assertEquals(VerifyMode.MATCH, options().verifyMode());
        assertEquals(VerifyMode.REJECTION, options("--spec-verify", "rejection").verifyMode());
        assertEquals(VerifyMode.BLOCK, options("--rejection-sample-method", "block").verifyMode());
        assertEquals(VerifyMode.REJECTION, options("--spec-draft-sampling", "probabilistic").verifyMode());
        assertEquals(VerifyMode.MATCH, options("--spec-draft-sampling", "greedy").verifyMode());
        SpeculativeOptions fly = options("--spec-verify", "fly", "--spec-fly-window", "4",
                "--spec-fly-entropy-threshold", "0.5", "--spec-fly-entropy-top-k", "8");
        assertEquals(VerifyMode.FLY, fly.verifyMode());
        assertEquals(4, fly.flyWindow());
        assertEquals(0.5, fly.flyEntropyThreshold(), 0.0);
        assertEquals(8, fly.flyEntropyTopK());
        assertRefused("--spec-verify", "--spec-verify", "lossy");
        assertRefused("--spec-draft-sampling", "--spec-draft-sampling", "beam");
    }

    @Test
    public void theDraftSplitProbabilityReachesTheDrafter() throws UsageException {
        assertEquals(SpeculativeOptions.DEFAULT_P_SPLIT, options().pSplit(), 0.0);
        assertEquals(0.25, options("--spec-draft-p-split", "0.25").pSplit(), 0.0);
        assertEquals(0.4, options("--draft-p-split", "0.4").pSplit(), 0.0);
        assertRefused("--spec-draft-p-split", "--spec-draft-p-split", "1.5");
        assertRefused("--spec-draft-p-split", "--spec-draft-p-split", "high");
    }

    @Test
    public void syntheticAcceptanceSelectsTheSyntheticMode() throws UsageException {
        SpeculativeOptions len = options("--spec-synth-len", "2.5");
        assertEquals(VerifyMode.SYNTHETIC, len.verifyMode());
        assertEquals(2.5, len.synthLength(), 0.0);
        SpeculativeOptions rates = options("--spec-synth-rates", "0.9,0.5");
        assertArrayEquals(new double[]{0.9, 0.5}, rates.synthRates(), 0.0);
        UsageException mixed = assertThrows(UsageException.class,
                () -> engine("--spec-verify", "rejection", "--spec-synth-len", "2"));
        assertTrue(mixed.getMessage(), mixed.getMessage().contains(SpeculativeOptions.PROP_VERIFY));
        assertRefused("--spec-synth-rates", "--spec-synth-rates", "0.5,x");
    }
}
