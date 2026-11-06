package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import com.eonmux.neuralfree.gguf.engine.draft.DrafterChain;
import com.eonmux.neuralfree.gguf.engine.draft.NgramCache;
import com.eonmux.neuralfree.gguf.engine.draft.NgramCacheFile;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.SuffixDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.SyntheticVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The lookup drafters reach a real target through the configuration alone: {@link LLMModelFactory#speculate} binds
 * the drafters of {@code neuralfree.draft.type} without a draft file, sized by {@link SpeculativeOptions}, chains
 * several in llama.cpp's ask order, reads and writes the {@code ngram-cache} files as llama.cpp does, and verifies
 * with the configured mode. Greedy speculative output equals the plain greedy output of the same model (the
 * verification is lossless), over the synthetic llama on the CPU engine.
 */
public class LookupSpeculationTest {

    private static final int NEW_TOKENS = 24;

    private static Path modelFile;
    private static EngineModel model;
    private static int[] plain;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @BeforeClass
    public static void openModel() throws IOException {
        modelFile = SyntheticModels.llama().writeTempFile();
        model = LLMModelFactory.open(modelFile.toString(), base().build());
        plain = model.generate(prompt(), request(), null).tokens();
        model.resetKVCache();
    }

    @AfterClass
    public static void closeModel() throws IOException {
        if (model != null) {
            model.close();
        }
        Files.deleteIfExists(modelFile);
    }

    /** The CPU engine with exact F32 caches, wide enough for the lookup prompt, and {@link #shortSizes()}. */
    private static EngineConfig.Builder base() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(512).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory).draftMax(6)
                .speculativeOptions(shortSizes().build());
    }

    /**
     * N-gram sizes a {@value #NEW_TOKENS}-token generation can match in: llama.cpp's defaults (12 / 48, ngram-mod
     * 24 / 64 / 48) need longer repeats than the run the synthetic model produces.
     */
    private static SpeculativeOptions.Builder shortSizes() {
        SpeculativeOptions.Builder b = SpeculativeOptions.builder();
        for (NgramDrafter.Kind kind : new NgramDrafter.Kind[]{NgramDrafter.Kind.SIMPLE, NgramDrafter.Kind.MAP_K,
                NgramDrafter.Kind.MAP_K4V, NgramDrafter.Kind.MOD}) {
            b.ngram(NgramDrafter.Options.builder(kind).sizeN(3).sizeM(8).draftMin(0).build());
        }
        return b;
    }

    /**
     * {@code (4, 5, v)} triples then {@code (4, 5)}: the synthetic model continues with a run of one token, which
     * the history then repeats, so every lookup type finds proposals.
     */
    private static int[] prompt() {
        int[] p = new int[3 * 100 + 2];
        int w = 0;
        for (int v = 6; v <= 105; v++) {
            p[w++] = 4;
            p[w++] = 5;
            p[w++] = v;
        }
        p[w++] = 4;
        p[w] = 5;
        return p;
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(NEW_TOKENS).sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static EngineConfig lookup(String types) {
        return base().draftTypes(DraftTypes.parse(types)).build();
    }

    /** Generates with a session over {@code cfg}, checks the output against plain greedy and returns the stats. */
    private static SpeculativeDecoder.Stats generate(SpeculativeSession s) {
        int[] tokens = s.generate(prompt(), request(), null).tokens();
        assertArrayEquals(s.drafterKind() + " keeps the plain greedy output", plain, tokens);
        return s.lastStats();
    }

    @Test
    public void ngramSimpleAttachesFromTheConfigurationWithItsSizes() throws IOException {
        EngineConfig cfg = base().draftType(DraftType.NGRAM).speculativeOptions(shortSizes()
                .ngram(NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE).sizeN(2).sizeM(5).build())
                .build()).build();
        try (SpeculativeSession s = LLMModelFactory.speculate(model, cfg)) {
            assertEquals("ngram-simple", s.drafterKind());
            NgramDrafter drafter = (NgramDrafter) s.drafter();
            assertEquals(NgramDrafter.Kind.SIMPLE, drafter.options().kind());
            assertEquals(2, drafter.options().sizeN());
            assertEquals(5, drafter.options().sizeM());
            SpeculativeDecoder.Stats stats = generate(s);
            assertTrue("the lookup drafted: " + stats, stats.drafted() > 0);
            assertTrue("drafts were accepted: " + stats, stats.accepted() > 0);
        }
    }

    @Test
    public void everyLookupTypeDraftsWithoutADraftFile() throws IOException {
        for (String type : new String[]{"ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache",
                "suffix"}) {
            try (SpeculativeSession s = LLMModelFactory.speculate(model, lookup(type))) {
                assertEquals(type, s.drafterKind());
                SpeculativeDecoder.Stats stats = generate(s);
                assertTrue(type + " drafted: " + stats, stats.drafted() > 0);
            }
        }
    }

    @Test
    public void severalTypesChainInLlamaCppOrder() throws IOException {
        try (SpeculativeSession s = LLMModelFactory.speculate(model, lookup("suffix,ngram-mod,ngram-simple"))) {
            assertEquals("ngram-simple+ngram-mod+suffix", s.drafterKind());
            List<DraftModel> members = ((DrafterChain) s.drafter()).drafters();
            assertEquals(3, members.size());
            assertEquals(NgramDrafter.Kind.SIMPLE, ((NgramDrafter) members.get(0)).options().kind());
            assertEquals(NgramDrafter.Kind.MOD, ((NgramDrafter) members.get(1)).options().kind());
            assertTrue(members.get(2) instanceof SuffixDrafter);
            assertTrue(generate(s).drafted() > 0);
        }
    }

    /**
     * A greedy draft model under rejection verification proposes one-hot rows, which the verifier accepts with
     * probability {@code p(d)}: the output stays the plain greedy one and the drafts keep being accepted.
     */
    @Test
    public void aGreedyDraftModelUnderRejectionKeepsThePlainOutput() throws IOException {
        EngineConfig cfg = base().draftModelPath(modelFile.toString())
                .speculativeOptions(SpeculativeOptions.builder().verifyMode(VerifyMode.REJECTION).build()).build();
        try (SpeculativeSession s = LLMModelFactory.speculate(model, cfg)) {
            assertTrue("the greedy draft proposed", generate(s).drafted() > 0);
        } finally {
            model.resetKVCache();
        }
    }

    @Test
    public void aWatermarkedDraftModelSpeculationIsDetectable() throws IOException {
        com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig watermark =
                com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig.builder(123L)
                        .algorithm(com.eonmux.neuralfree.gguf.engine.watermark.WatermarkAlgorithm.DUAL_KEY_GUMBEL)
                        .contextWidth(4).build();
        EngineConfig cfg = base().draftModelPath(modelFile.toString())
                .speculativeOptions(SpeculativeOptions.builder().verifyMode(VerifyMode.REJECTION).build()).build();
        GenerationRequest sampling = GenerationRequest.builder().maxNewTokens(64)
                .sampler(SamplerConfig.builder().temperature(0.8f).topK(0).topP(1f).minP(0f).seed(5)
                        .watermark(watermark).build()).build();
        GenerationResult generated;
        try (SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            generated = session.generate(prompt(), sampling, null);
            assertTrue("the draft model proposed", session.lastStats().drafted() > 0);
        } finally {
            model.resetKVCache();
        }
        assertTrue("the watermarked speculative output is detected",
                com.eonmux.neuralfree.gguf.engine.watermark.WatermarkDetector.create(watermark)
                        .detect(generated.tokens()).isWatermarked());
    }

    @Test
    public void aSingleKeyWatermarkedRequestIsRefusedByNameOnTheSpeculativePath() throws IOException {
        EngineConfig cfg = base().draftType(DraftType.NGRAM).build();
        GenerationRequest watermarked = GenerationRequest.builder().maxNewTokens(NEW_TOKENS)
                .sampler(SamplerConfig.greedyDecoding().toBuilder()
                        .watermark(com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig.builder(9L)
                                .build())
                        .build())
                .build();
        try (SpeculativeSession s = LLMModelFactory.speculate(model, cfg)) {
            IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                    () -> s.generate(prompt(), watermarked, null));
            assertTrue(refused.getMessage(), refused.getMessage().contains("allow_target_only_watermarking"));
        }
        model.resetKVCache();
    }

    @Test
    public void probabilisticDraftingIsRefusedOnTheSingleSequencePathByName() {
        EngineConfig cfg = base().draftType(DraftType.NGRAM).speculativeOptions(SpeculativeOptions.builder()
                .probabilisticDraft(true).build()).build();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> LLMModelFactory.speculate(model, cfg));
        assertTrue(e.getMessage(), e.getMessage().contains(SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD));
    }

    @Test
    public void theDynamicLookupCacheIsWrittenOnCloseAndGrowsAcrossRuns() throws IOException {
        Path dynamic = tmp.getRoot().toPath().resolve("dynamic.bin");
        EngineConfig cfg = base().draftType(DraftType.NGRAM_CACHE).speculativeOptions(shortSizes()
                .lookupCacheDynamic(dynamic.toString()).build()).build();
        NgramCache.Key key = NgramCache.Key.of(new int[]{4, 5}, 0, 2);

        try (SpeculativeSession s = LLMModelFactory.speculate(model, cfg)) {
            generate(s);
            assertFalse("the dynamic cache is written when the session closes", Files.exists(dynamic));
        }
        NgramCache first = NgramCacheFile.load(dynamic);
        assertEquals("the run's (4, 5) -> 6 n-gram", 1, first.count(key, 6));

        try (SpeculativeSession s = LLMModelFactory.speculate(model, cfg)) {
            generate(s);
        }
        NgramCache second = NgramCacheFile.load(dynamic);
        assertEquals("the loaded counts plus the second run's", 2, second.count(key, 6));
        assertTrue(second.ngrams() >= first.ngrams());
    }

    @Test
    public void theStaticLookupCacheIsReadAndAMissingOneRefusedByName() throws IOException {
        NgramCache cache = new NgramCache();
        cache.addSequence(1, NgramCache.MAX_KEY, prompt());
        Path stat = tmp.newFile("static.bin").toPath();
        NgramCacheFile.save(cache, stat);
        EngineConfig cfg = base().draftType(DraftType.NGRAM_CACHE).speculativeOptions(shortSizes()
                .lookupCacheStatic(stat.toString()).build()).build();
        try (SpeculativeSession s = LLMModelFactory.speculate(model, cfg)) {
            assertTrue(generate(s).drafted() > 0);
        }

        String missing = tmp.getRoot().toPath().resolve("missing.bin").toString();
        EngineConfig refused = base().draftType(DraftType.NGRAM_CACHE).speculativeOptions(shortSizes()
                .lookupCacheStatic(missing).build()).build();
        FileNotFoundException e = assertThrows(FileNotFoundException.class,
                () -> LLMModelFactory.speculate(model, refused));
        assertTrue(e.getMessage(), e.getMessage().contains(SpeculativeOptions.PROP_LOOKUP_CACHE_STATIC));
        // the refusal released the target's drafter slot
        try (SpeculativeSession s = LLMModelFactory.speculate(model, lookup("ngram-simple"))) {
            assertEquals("ngram-simple", s.drafterKind());
        }
    }

    @Test
    public void theVerificationModeReachesTheDecoder() throws IOException {
        for (VerifyMode mode : Arrays.asList(VerifyMode.REJECTION, VerifyMode.BLOCK)) {
            EngineConfig cfg = base().draftType(DraftType.NGRAM).speculativeOptions(shortSizes()
                    .verifyMode(mode).build()).build();
            try (SpeculativeSession s = LLMModelFactory.speculate(model, cfg)) {
                assertEquals(mode, s.verifyMode());
                // greedy target chains reduce the lossless modes to the match
                assertTrue(generate(s).drafted() > 0);
            }
        }
    }

    @Test
    public void syntheticVerificationWithoutTheOptInRefusesAndReleasesTheDrafters() throws IOException {
        assertNull(System.getProperty(SyntheticVerifier.PROP_ENABLED));
        EngineConfig cfg = base().draftType(DraftType.NGRAM).speculativeOptions(shortSizes()
                .synthLength(2.0).build()).build();
        assertEquals(VerifyMode.SYNTHETIC, cfg.speculativeOptions().verifyMode());
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> LLMModelFactory.speculate(model, cfg));
        assertTrue(e.getMessage(), e.getMessage().contains(SyntheticVerifier.PROP_ENABLED));
        try (SpeculativeSession s = LLMModelFactory.speculate(model, lookup("ngram-simple"))) {
            assertEquals(VerifyMode.MATCH, s.verifyMode());
        }
    }

    @Test
    public void theLookupTypesChainWithTheModelsOwnMtpHead() throws IOException {
        Path mtpFile = SyntheticModels.glm4moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(2).build(), 1)
                .writeTempFile();
        try (EngineModel mtpModel = LLMModelFactory.open(mtpFile.toString(), base().mtp(true).build())) {
            GenerationRequest req = request();
            int[] expected = mtpModel.generate(prompt(), req, null).tokens();
            mtpModel.resetKVCache();
            EngineConfig cfg = base().draftTypes(DraftTypes.parse("ngram-simple,draft-mtp")).build();
            assertTrue(cfg.mtp());
            try (SpeculativeSession s = LLMModelFactory.speculate(mtpModel, cfg)) {
                assertEquals("ngram-simple+mtp", s.drafterKind());
                List<DraftModel> members = ((DrafterChain) s.drafter()).drafters();
                assertTrue(members.get(0) instanceof NgramDrafter);
                assertArrayEquals(expected, s.generate(prompt(), req, null).tokens());
                assertTrue(s.lastStats().drafted() > 0);
            }
        } finally {
            Files.deleteIfExists(mtpFile);
        }
    }
}
