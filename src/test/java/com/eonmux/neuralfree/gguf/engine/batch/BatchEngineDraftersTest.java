package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.DraftTypes;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.draft.NgramCache;
import com.eonmux.neuralfree.gguf.engine.draft.NgramCacheFile;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
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

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.TreeMap;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Every drafter of the batch engine with per-sequence state (vLLM serves every proposer in its batched scheduler):
 * the lookup types, their chains and a separate draft model draft for several running sequences at once, and under
 * greedy decoding every sequence finishes with the tokens of the same prompt run alone through
 * {@link GenerationLoop}; the {@code ngram-cache} dynamic file collects every sequence's n-grams; rejection sampling
 * in the batch verify step keeps the target's sampling distribution with a probabilistic draft model; the
 * configurations the batch engine does not run refuse by name.
 */
public class BatchEngineDraftersTest {

    private static final int N_CTX = 768;
    private static final int NEW_TOKENS = 16;

    private static Path model;
    /** A draft model of the target's vocabulary with other weights. */
    private static Path otherDraft;
    private static GenerationResult refA;
    private static GenerationResult refB;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @BeforeClass
    public static void writeModels() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        otherDraft = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().seed(7).build())
                .writeTempFile();
        refA = reference(promptA(), greedy());
        refB = reference(promptB(), greedy());
    }

    @AfterClass
    public static void deleteModels() throws IOException {
        Files.deleteIfExists(model);
        Files.deleteIfExists(otherDraft);
    }

    // ------------------------------------------------------------------ fixtures

    private static EngineConfig.Builder config(int nCtx, int parallel) {
        MemoryConfig mc = MemoryConfig.builder().nCtx(nCtx).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .parallel(parallel).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(mc).draftMax(4);
    }

    /**
     * {@code (a, b, v)} triples over v = {@code from}..{@code to}, ending on the pair: the synthetic model's greedy
     * continuation repeats tokens its history holds, so every lookup type finds proposals.
     */
    private static int[] triples(int a, int b, int from, int to) {
        int[] p = new int[3 * (to - from + 1) + 2];
        int w = 0;
        for (int v = from; v <= to; v++) {
            p[w++] = a;
            p[w++] = b;
            p[w++] = v;
        }
        p[w++] = a;
        p[w] = b;
        return p;
    }

    /** {@code ngram-simple} over the 3-gram the triples repeat; llama.cpp's default 12-gram finds no match. */
    private static NgramDrafter.Options simple3() {
        return NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE).sizeN(3).sizeM(8).build();
    }

    /**
     * Every n-gram kind keyed by the triples' 3-gram with 8-token proposals: llama.cpp's default keys (12 for
     * the maps, 24 for {@code ngram-mod}) find no match in them.
     */
    private static SpeculativeOptions tripleLookups() {
        SpeculativeOptions.Builder b = SpeculativeOptions.builder().ngram(simple3());
        for (NgramDrafter.Kind kind : new NgramDrafter.Kind[]{NgramDrafter.Kind.MAP_K, NgramDrafter.Kind.MAP_K4V}) {
            b.ngram(NgramDrafter.Options.builder(kind).sizeN(3).sizeM(8).build());
        }
        return b.ngram(NgramDrafter.Options.builder(NgramDrafter.Kind.MOD).sizeN(3).sizeM(8).draftMin(0).build())
                .build();
    }

    private static int[] promptA() {
        return triples(4, 5, 6, 105);
    }

    private static int[] promptB() {
        return triples(7, 8, 20, 99);
    }

    private static GenerationRequest greedy() {
        return GenerationRequest.builder().maxNewTokens(NEW_TOKENS).sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** The same prompt and request through the single-sequence GenerationLoop. */
    private static GenerationResult reference(int[] prompt, GenerationRequest request) throws IOException {
        EngineConfig cfg = config(N_CTX, 1).build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request, (TokenCallback) null);
        }
    }

    private static void assertDone(BatchSequence sequence) {
        assertTrue("expected DONE but was " + sequence.state()
                        + (sequence.error() == null ? "" : ": " + sequence.error()),
                sequence.state() == BatchSequence.State.DONE);
    }

    private static void assertSame(String what, GenerationResult reference, BatchSequence batched) {
        assertDone(batched);
        assertEquals(what + " stop reason", reference.stopReason(), batched.result().stopReason());
        assertArrayEquals(what + " token ids", reference.tokens(), batched.result().tokens());
        assertEquals(what + " text", reference.text(), batched.result().text());
    }

    /** Runs prompts A and B side by side on an engine over {@code cfg}; checks both against the references. */
    private static BatchStats runBoth(String what, EngineConfig cfg) throws IOException {
        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            BatchSequence a = engine.submit(promptA(), greedy(), (TokenCallback) null);
            BatchSequence b = engine.submit(promptB(), greedy(), (TokenCallback) null);
            engine.drain();
            assertSame(what + " A", refA, a);
            assertSame(what + " B", refB, b);
            return engine.stats();
        }
    }

    // ------------------------------------------------------------------ greedy equality

    @Test
    public void everyLookupTypeDraftsPerSequenceAndKeepsTheSingleSequenceOutputs() throws IOException {
        for (String types : new String[]{"ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod",
                "ngram-cache", "suffix", "ngram-map-k,suffix", "suffix,ngram-mod,ngram-simple"}) {
            BatchStats stats = runBoth(types, config(N_CTX, 2).draftTypes(DraftTypes.parse(types))
                    .speculativeOptions(tripleLookups()).build());
            assertTrue(types + " drafted: " + stats, stats.specDrafted() > 0);
            assertTrue(stats.specAccepted() <= stats.specDrafted());
        }
    }

    @Test
    public void vllmsPromptLookupRangeDraftsPerSequenceWhereTheSingleSizeCannot() throws IOException {
        NgramDrafter.Options single = NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE).build();
        BatchStats none = runBoth("size 12", config(N_CTX, 2).draftType(EngineConfig.DraftType.NGRAM)
                .speculativeOptions(SpeculativeOptions.builder().ngram(single).build()).build());
        assertEquals("llama.cpp's single 12-gram finds no match: " + none, 0, none.specDrafted());

        NgramDrafter.Options range = NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE).sizeN(12).sizeNMin(2)
                .sizeM(8).build();
        BatchStats stats = runBoth("range 2..12", config(N_CTX, 2).draftType(EngineConfig.DraftType.NGRAM)
                .speculativeOptions(SpeculativeOptions.builder().ngram(range).build()).build());
        assertTrue("the 2..12 range matches the shorter repeats: " + stats, stats.specDrafted() > 0);
        assertTrue(stats.specAccepted() <= stats.specDrafted());
    }

    @Test
    public void aDraftModelDraftsEverySequenceAndKeepsTheSingleSequenceOutputs() throws IOException {
        EngineConfig self = config(N_CTX, 2).draftType(EngineConfig.DraftType.MODEL)
                .draftModelPath(model.toString()).build();
        BatchStats stats = runBoth("the target as its own draft", self);
        assertTrue("the draft model drafted: " + stats, stats.specDrafted() > 0);
        // the target drafting for itself proposes its own greedy picks: only a round's stop position is lost
        assertTrue("accepted " + stats.specAccepted() + " of " + stats.specDrafted() + " in "
                + stats.specRounds() + " rounds", stats.specAccepted() >= stats.specDrafted() - stats.specRounds());

        EngineConfig other = config(N_CTX, 2).draftModelPath(otherDraft.toString()).build();
        assertTrue(runBoth("another draft model", other).specDrafted() > 0);

        EngineConfig chain = config(N_CTX, 2).draftTypes(DraftTypes.parse("ngram-simple,draft-simple"))
                .draftModelPath(otherDraft.toString())
                .speculativeOptions(SpeculativeOptions.builder().ngram(simple3()).build()).build();
        assertTrue(runBoth("ngram-simple then the draft model", chain).specDrafted() > 0);
    }

    /** A drafter recording the widest {@link #draftBatch} call it served. */
    private static final class Recording implements BatchDrafter {
        private final BatchDrafter inner;
        int widest;

        Recording(BatchDrafter inner) {
            this.inner = inner;
        }

        @Override public int[] draft(int seq, int lastToken, int n) {
            return inner.draft(seq, lastToken, n);
        }

        @Override public int[][] draftBatch(List<Request> requests) {
            widest = Math.max(widest, requests.size());
            return inner.draftBatch(requests);
        }

        @Override public float[] draftProbs(int seq, int position) {
            return inner.draftProbs(seq, position);
        }

        @Override public void accept(int seq, int nAccepted) {
            inner.accept(seq, nAccepted);
        }

        @Override public void reset(int seq) {
            inner.reset(seq);
        }

        @Override public void close() {
            inner.close();
        }
    }

    /**
     * Two short prompts prefill within the first steps, so their decode rounds overlap: each step's proposals for
     * both sequences come from one draft call, and both outputs stay the single-sequence ones.
     */
    @Test
    public void aStepDraftsEveryDecodingSequenceInOneCall() throws IOException {
        int[] first = {1, 10, 20, 30, 40, 50};
        int[] second = {1, 60, 70, 80, 90, 100};
        GenerationResult refFirst = reference(first, greedy());
        GenerationResult refSecond = reference(second, greedy());
        EngineConfig cfg = config(N_CTX, 2).draftModelPath(otherDraft.toString()).build();
        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             Recording drafter = new Recording(new BatchModelDrafter(LLMModelFactory.openBatchDraftModel(two, cfg),
                     two, 0f, false));
             BatchEngine engine = BatchEngine.forModel(two, cfg, drafter,
                     BatchSpec.of(4, 0, cfg.speculativeOptions()))) {
            BatchSequence a = engine.submit(first, greedy(), (TokenCallback) null);
            BatchSequence b = engine.submit(second, greedy(), (TokenCallback) null);
            engine.drain();
            assertSame("first", refFirst, a);
            assertSame("second", refSecond, b);
            assertEquals("both running sequences drafted in one batched call", 2, drafter.widest);
        }
    }

    @Test
    public void theOtherVerifyModesKeepGreedyOutputsBesideASamplingSequence() throws IOException {
        for (VerifyMode mode : new VerifyMode[]{VerifyMode.REJECTION, VerifyMode.BLOCK}) {
            EngineConfig cfg = config(N_CTX, 3).draftTypes(DraftTypes.parse("ngram-simple,draft-simple"))
                    .draftModelPath(otherDraft.toString())
                    .speculativeOptions(SpeculativeOptions.builder().verifyMode(mode).probabilisticDraft(true)
                            .ngram(simple3()).build())
                    .build();
            GenerationRequest sampling = GenerationRequest.builder().maxNewTokens(NEW_TOKENS)
                    .sampler(SamplerConfig.builder().temperature(1f).topK(8).seed(11).build()).build();
            try (GGUFModel two = new GGUFModel(model.toString(), cfg);
                 BatchEngine engine = BatchEngine.forModel(two, cfg)) {
                BatchSequence a = engine.submit(promptA(), greedy(), (TokenCallback) null);
                BatchSequence s = engine.submit(promptB(), sampling, (TokenCallback) null);
                BatchSequence b = engine.submit(promptB(), greedy(), (TokenCallback) null);
                engine.drain();
                assertSame(mode + " A", refA, a);
                assertSame(mode + " B", refB, b);
                assertDone(s);
                assertEquals(NEW_TOKENS, s.result().tokenCount());
                assertTrue(mode + " verified rounds", engine.stats().specRounds() > 0);
            }
        }
    }

    // ------------------------------------------------------------------ the dynamic lookup cache

    @Test
    public void theDynamicLookupCacheCollectsEverySequenceWhenTheEngineCloses() throws IOException {
        Path dynamic = tmp.getRoot().toPath().resolve("dynamic.bin");
        EngineConfig cfg = config(N_CTX, 2).draftType(EngineConfig.DraftType.NGRAM_CACHE)
                .speculativeOptions(SpeculativeOptions.builder().lookupCacheDynamic(dynamic.toString()).build())
                .build();
        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            BatchSequence a = engine.submit(promptA(), greedy(), (TokenCallback) null);
            BatchSequence b = engine.submit(promptB(), greedy(), (TokenCallback) null);
            // a third sequence reuses a finished one's slot: the n-grams of the slot's earlier sequence stay
            BatchSequence again = engine.submit(promptA(), greedy(), (TokenCallback) null);
            engine.drain();
            assertSame("A", refA, a);
            assertSame("B", refB, b);
            assertSame("A again", refA, again);
            assertFalse("the dynamic cache is written when the engine closes", Files.exists(dynamic));
        }
        NgramCache saved = NgramCacheFile.load(dynamic);
        assertEquals("both A sequences' (4, 5) -> 6", 2, saved.count(NgramCache.Key.of(new int[]{4, 5}, 0, 2), 6));
        assertEquals("B's (7, 8) -> 20", 1, saved.count(NgramCache.Key.of(new int[]{7, 8}, 0, 2), 20));
    }

    // ------------------------------------------------------------------ draft probabilities

    @Test
    public void aProbabilisticDraftReportsTheDistributionItsTokensWereDrawnFrom() throws IOException {
        EngineConfig cfg = config(256, 2).draftModelPath(otherDraft.toString()).build();
        try (GGUFModel target = new GGUFModel(model.toString(), cfg);
             BatchModelDrafter drafter = new BatchModelDrafter(LLMModelFactory.openBatchDraftModel(target, cfg),
                     target, 0f, true)) {
            List<BatchDrafter.Request> requests = new ArrayList<>();
            requests.add(new BatchDrafter.Request(0, 10, 3, 1f, new SplittableRandom(3)));
            requests.add(new BatchDrafter.Request(1, 20, 3, 0f, new SplittableRandom(3)));
            int[][] drafts = drafter.draftBatch(requests);
            assertEquals(3, drafts[0].length);
            for (int i = 0; i < drafts[0].length; i++) {
                float[] q = drafter.draftProbs(0, i);
                double sum = 0;
                for (float v : q) {
                    sum += v;
                }
                assertEquals("the proposal row is a distribution", 1.0, sum, 1e-4);
                assertTrue("the drawn token has mass", q[drafts[0][i]] > 0f);
            }
            assertEquals(3, drafts[1].length);
            assertNull("a greedy request's picks are one-hot", drafter.draftProbs(1, 0));
            assertArrayEquals("the greedy draft is deterministic after the draft memory re-syncs", drafts[1],
                    drafter.draft(1, 20, 3));
            assertThrows(IllegalStateException.class, () -> drafter.draftProbs(1, 3));
        }
    }

    // ------------------------------------------------------------------ rejection sampling distribution

    private static final int SAMPLES = 1200;
    private static final int SAMPLED = 3;
    private static final float TEMPERATURE = 0.5f;
    private static final int[] SHORT_PROMPT = {1, 10, 20, 30, 40, 50, 60, 70};

    /** A top-4 sampling request with its own seed. */
    private static GenerationRequest sampling(long seed) {
        return GenerationRequest.builder().maxNewTokens(SAMPLED)
                .sampler(SamplerConfig.builder().temperature(TEMPERATURE).topK(4).topP(1f).minP(0f).seed(seed)
                        .build())
                .build();
    }

    /** The sampled token triples of {@link #SAMPLES} requests, keyed as one string per request. */
    private static List<String> sample(BatchEngine engine, long firstSeed) {
        List<BatchSequence> runs = new ArrayList<>(SAMPLES);
        for (int i = 0; i < SAMPLES; i++) {
            runs.add(engine.submit(SHORT_PROMPT, sampling(firstSeed + i), (TokenCallback) null));
        }
        engine.drain();
        List<String> out = new ArrayList<>(SAMPLES);
        for (BatchSequence s : runs) {
            assertDone(s);
            int[] t = s.result().tokens();
            assertEquals(SAMPLED, t.length);
            out.add(t[0] + "," + t[1] + "," + t[2]);
        }
        return out;
    }

    /** Rejection verification with a probabilistic draft model of {@code draft}. */
    private static EngineConfig rejection(Path draft) {
        return config(512, 8).draftModelPath(draft.toString()).draftMax(3)
                .speculativeOptions(SpeculativeOptions.builder().verifyMode(VerifyMode.REJECTION)
                        .probabilisticDraft(true).build())
                .build();
    }

    /** Samples through rejection verification with a probabilistic draft model of {@code draft}. */
    private static List<String> speculativeSample(Path draft, long firstSeed) throws IOException {
        EngineConfig cfg = rejection(draft);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            List<String> out = sample(engine, firstSeed);
            BatchStats stats = engine.stats();
            assertTrue("the draft model drafted: " + stats, stats.specDrafted() > 0);
            assertTrue("some proposals were accepted: " + stats, stats.specAccepted() > 0);
            assertTrue("some proposals were rejected: " + stats, stats.specAccepted() < stats.specDrafted());
            return out;
        }
    }

    /**
     * Sampled outputs of 1200 top-4 requests through eight concurrent sequences: rejection verification with a
     * probabilistic draft model (the target itself, and a model of other weights) gives the plain engine's joint law
     * of the three sampled tokens. The proposal-row plumbing under a misreported distribution is pinned by
     * {@code VerifyRoundDistributionTest}, where the bias is large enough to detect.
     */
    @Test
    public void rejectionSamplingInTheBatchKeepsTheTargetsSamplingDistribution() throws IOException {
        List<String> plain;
        EngineConfig plainCfg = config(512, 8).build();
        try (GGUFModel m = new GGUFModel(model.toString(), plainCfg);
             BatchEngine engine = BatchEngine.forModel(m, plainCfg)) {
            plain = sample(engine, 1);
        }
        ChiSquare self = ChiSquare.twoSample(plain, speculativeSample(model, 1 + SAMPLES));
        ChiSquare other = ChiSquare.twoSample(plain, speculativeSample(otherDraft, 1 + 2 * SAMPLES));
        assertTrue("rejection sampling with the target as its own draft matches plain sampling: " + self,
                self.statistic < self.critical);
        assertTrue("rejection sampling with another draft model matches plain sampling: " + other,
                other.statistic < other.critical);
    }

    /** Pearson's two-sample chi-square over equal-size samples, cells under 10 joint counts pooled. */
    private static final class ChiSquare {
        final double statistic;
        final int dof;
        final double critical;

        private ChiSquare(double statistic, int dof) {
            this.statistic = statistic;
            this.dof = dof;
            // Wilson-Hilferty upper 0.0005 quantile (z = 3.29)
            double h = 2.0 / (9.0 * dof);
            this.critical = dof * Math.pow(1 - h + 3.29 * Math.sqrt(h), 3);
        }

        static ChiSquare twoSample(List<String> a, List<String> b) {
            Map<String, int[]> cells = new TreeMap<>();
            for (String s : a) {
                cells.computeIfAbsent(s, k -> new int[2])[0]++;
            }
            for (String s : b) {
                cells.computeIfAbsent(s, k -> new int[2])[1]++;
            }
            double x2 = 0;
            int used = 0;
            int pooledA = 0;
            int pooledB = 0;
            for (int[] c : cells.values()) {
                if (c[0] + c[1] < 10) {
                    pooledA += c[0];
                    pooledB += c[1];
                    continue;
                }
                x2 += (double) (c[0] - c[1]) * (c[0] - c[1]) / (c[0] + c[1]);
                used++;
            }
            if (pooledA + pooledB > 0) {
                x2 += (double) (pooledA - pooledB) * (pooledA - pooledB) / (pooledA + pooledB);
                used++;
            }
            return new ChiSquare(x2, Math.max(1, used - 1));
        }

        @Override
        public String toString() {
            return "chi2=" + statistic + " dof=" + dof + " critical=" + critical;
        }
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void whatTheBatchEngineDoesNotRunRefusesByName() throws IOException {
        EngineConfig base = config(256, 2).build();
        try (GGUFModel two = new GGUFModel(model.toString(), base)) {
            UnsupportedOperationException mtp = assertThrows(UnsupportedOperationException.class,
                    () -> BatchEngine.forModel(two, config(256, 2).mtp(true).build()));
            assertTrue(mtp.getMessage(), mtp.getMessage().contains(EngineConfig.PROP_MTP));

            IllegalArgumentException probabilistic = assertThrows(IllegalArgumentException.class,
                    () -> BatchEngine.forModel(two, config(256, 2).draftType(EngineConfig.DraftType.NGRAM)
                            .speculativeOptions(SpeculativeOptions.builder().probabilisticDraft(true).build())
                            .build()));
            assertTrue(probabilistic.getMessage(),
                    probabilistic.getMessage().contains(SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD));

            String missing = tmp.getRoot().toPath().resolve("absent.gguf").toString();
            UncheckedIOException absent = assertThrows(UncheckedIOException.class,
                    () -> BatchEngine.forModel(two, config(256, 2).draftModelPath(missing).build()));
            assertTrue(absent.getMessage(), absent.getMessage().contains("absent.gguf"));

            if (!Boolean.getBoolean(SyntheticVerifier.PROP_ENABLED)) {
                IllegalStateException synthetic = assertThrows(IllegalStateException.class,
                        () -> BatchEngine.forModel(two, config(256, 2).draftType(EngineConfig.DraftType.NGRAM)
                                .speculativeOptions(SpeculativeOptions.builder().verifyMode(VerifyMode.SYNTHETIC)
                                        .synthLength(2.0).build())
                                .build()));
                assertTrue(synthetic.getMessage(), synthetic.getMessage().contains(SyntheticVerifier.PROP_ENABLED));
            }
        }
    }

    // ------------------------------------------------------------------ speculative watermarking

    /** A dual-key watermark over the first {@code width} tokens, deduplication off. */
    private static com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig watermark(long key, int width) {
        return com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig.builder(key)
                .algorithm(com.eonmux.neuralfree.gguf.engine.watermark.WatermarkAlgorithm.DUAL_KEY_GUMBEL)
                .contextWidth(width)
                .deduplicateContexts(
                        com.eonmux.neuralfree.gguf.engine.watermark.WatermarkContextScope.NONE)
                .build();
    }

    @Test
    public void aWatermarkedDraftDrawsUnderTheDraftKeyAndReportsItsProposal() throws IOException {
        com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig wm = watermark(31L, 4);
        EngineConfig cfg = config(256, 2).draftModelPath(otherDraft.toString()).build();
        try (GGUFModel target = new GGUFModel(model.toString(), cfg);
             BatchModelDrafter drafter = new BatchModelDrafter(LLMModelFactory.openBatchDraftModel(target, cfg),
                     target, 0f, false)) {
            assertTrue(drafter.supportsWatermarking());
            List<BatchDrafter.Request> plain = new ArrayList<>();
            plain.add(new BatchDrafter.Request(0, 10, 3, 1f, new SplittableRandom(3)));
            int[][] ordinary = drafter.draftBatch(plain);
            assertNull("a greedy draft reports one-hot proposals", drafter.draftProbs(0, 0));
            assertArrayEquals("the draft memory re-syncs, so the plain draft repeats", ordinary[0],
                    drafter.draft(0, 10, 3));

            List<BatchDrafter.Request> marked = new ArrayList<>();
            marked.add(new BatchDrafter.Request(0, 10, 3, 1f, new SplittableRandom(3), wm, 0));
            int[][] watermarked = drafter.draftBatch(marked);
            assertTrue("the keyed draw differs from the greedy picks",
                    !java.util.Arrays.equals(ordinary[0], watermarked[0]));
            for (int i = 0; i < watermarked[0].length; i++) {
                float[] q = drafter.draftProbs(0, i);
                assertTrue("a watermarked draft reports its proposal", q != null);
                double sum = 0.0;
                for (float v : q) {
                    sum += v;
                }
                assertEquals("proposal " + i + " is a distribution", 1.0, sum, 1e-4);
                assertTrue("the drawn token has proposal mass", q[watermarked[0][i]] > 0f);
            }
        }
    }

    @Test
    public void aWatermarkedSpeculativeRequestIsDetectable() throws IOException {
        com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig wm = watermark(123L, 4);
        EngineConfig cfg = config(N_CTX, 2).draftModelPath(otherDraft.toString())
                .speculativeOptions(SpeculativeOptions.builder().verifyMode(VerifyMode.REJECTION).build()).build();
        GenerationRequest.Builder request = GenerationRequest.builder().maxNewTokens(64)
                .sampler(SamplerConfig.builder().temperature(0.8f).topK(0).topP(1f).minP(0f).seed(5)
                        .watermark(wm).build());
        GenerationRequest marked = request.build();
        GenerationRequest plain = request.build().toBuilder()
                .sampler(marked.sampler().toBuilder().watermark(null).build()).build();
        int[] markedTokens;
        int[] plainTokens;
        try (GGUFModel target = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(target, cfg)) {
            BatchSequence a = engine.submit(promptA(), marked, (TokenCallback) null);
            BatchSequence b = engine.submit(promptA(), plain, (TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertDone(b);
            markedTokens = a.result().tokens();
            plainTokens = b.result().tokens();
            assertTrue("the watermarked round drafted", engine.stats().specDrafted() > 0);
        }
        assertTrue("the watermarked output is detected",
                com.eonmux.neuralfree.gguf.engine.watermark.WatermarkDetector.create(wm)
                        .detect(markedTokens).isWatermarked());
        assertFalse("the plain output is not detected",
                com.eonmux.neuralfree.gguf.engine.watermark.WatermarkDetector.create(wm)
                        .detect(plainTokens).isWatermarked());
    }

    @Test
    public void anAdaptivePChainIsRefusedAtSubmitBesideAVerifyModeOtherThanMatchAndMirostatRuns() throws IOException {
        EngineConfig rejection = config(N_CTX, 2).draftType(EngineConfig.DraftType.NGRAM)
                .speculativeOptions(SpeculativeOptions.builder().verifyMode(VerifyMode.REJECTION).build()).build();
        GenerationRequest adaptiveP = GenerationRequest.builder().maxNewTokens(4)
                .sampler(SamplerConfig.builder().temperature(1f).samplers("temperature;adaptive_p").seed(1).build())
                .build();
        try (GGUFModel two = new GGUFModel(model.toString(), rejection);
             BatchEngine engine = BatchEngine.forModel(two, rejection)) {
            IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                    () -> engine.submit(promptA(), adaptiveP, (TokenCallback) null));
            assertTrue(refused.getMessage(), refused.getMessage().contains("adaptive_p"));
            assertTrue(engine.idle());

            // mirostat adapts on the emitted token's surprise, so it runs under every verify mode
            GenerationRequest mirostat = GenerationRequest.builder().maxNewTokens(4)
                    .sampler(SamplerConfig.builder().mirostat(2).seed(1).build()).build();
            BatchSequence s = engine.submit(promptA(), mirostat, (TokenCallback) null);
            engine.drain();
            assertDone(s);
        }
    }
}
