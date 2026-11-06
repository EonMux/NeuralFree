package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Lookahead decoding over the synthetic llama on the CPU engine: the deterministic path keeps the plain greedy
 * output token for token (the verification is exact), its observed n-grams are accepted on a repeating prompt, and
 * the memory and request shapes it cannot serve are refused by name.
 */
public class LookaheadDecoderTest {

    private static final int WINDOW = 4;
    private static final int NGRAM = 3;
    private static final int MAX_NGRAMS = 2;
    private static final int SEQUENCES = WINDOW + MAX_NGRAMS + 1;
    private static final int N_CTX = 1024;
    private static final int N_NEW = 16;

    private static Path modelFile;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @BeforeClass
    public static void writeModel() throws IOException {
        modelFile = SyntheticModels.llama().writeTempFile();
    }

    @AfterClass
    public static void deleteModel() throws IOException {
        if (modelFile != null) {
            Files.deleteIfExists(modelFile);
        }
    }

    private static LookaheadOptions options() {
        return LookaheadOptions.builder().enabled(true).window(WINDOW).ngramSize(NGRAM)
                .maxNgrams(MAX_NGRAMS).build();
    }

    /** Paged memory with the lookahead sequences, unified KV and exact caches. */
    private static EngineConfig config() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32)
                .parallel(SEQUENCES).kvUnified(true).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory)
                .lookaheadOptions(options()).build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** The same prompt and request through the plain single-sequence loop. */
    private static GenerationResult plain(int[] prompt, GenerationRequest request) throws IOException {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory).build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg)) {
            return GenerationLoop.forModel(model, cfg).generate(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    /** A prompt whose token n-grams repeat, so the history offers verification candidates. */
    private static int[] repeatingPrompt() {
        int[] prompt = new int[3 * 20 + 2];
        int w = 0;
        for (int v = 6; v <= 25; v++) {
            prompt[w++] = 4;
            prompt[w++] = 5;
            prompt[w++] = v;
        }
        prompt[w++] = 4;
        prompt[w] = 5;
        return prompt;
    }

    @Test
    public void greedyOutputMatchesThePlainLoop() throws IOException {
        int[] prompt = new int[] {4, 5, 6, 4, 5, 7, 4, 5, 8, 4, 5};
        GenerationRequest request = request();
        GenerationResult expected = plain(prompt, request);
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), config())) {
            LookaheadDecoder decoder = LookaheadDecoder.forModel(model, config());
            GenerationResult actual = decoder.generate(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals("stop reason", expected.stopReason(), actual.stopReason());
            assertArrayEquals("tokens", expected.tokens(), actual.tokens());
            assertEquals("text", expected.text(), actual.text());
        }
    }

    @Test
    public void repeatingPromptAcceptsObservedNgrams() throws IOException {
        int[] prompt = repeatingPrompt();
        GenerationRequest request = request();
        GenerationResult expected = plain(prompt, request);
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), config())) {
            LookaheadOptions options = options();
            java.util.concurrent.atomic.AtomicReference<LookaheadDecoder.Stats> seen =
                    new java.util.concurrent.atomic.AtomicReference<>();
            LookaheadDecoder decoder = LookaheadDecoder.builder(model.forwardBackend(), model.tokenizer(), options)
                    .samplingDefaults(SamplingDefaults.forModel(model, config()))
                    .statsSink(seen::set)
                    .build();
            GenerationResult actual = decoder.generate(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertArrayEquals("tokens", expected.tokens(), actual.tokens());
            assertEquals("stop reason", expected.stopReason(), actual.stopReason());
            LookaheadDecoder.Stats stats = seen.get();
            assertTrue("the repeating prompt offers n-grams the target confirms", stats.accepted() > 0);
            assertTrue("an accepted n-gram saves a step", stats.emitted() > stats.rounds());
        }
    }

    @Test
    public void longGenerationKeepsThePlainGreedyOutput() throws IOException {
        int[] prompt = repeatingPrompt();
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(48)
                .sampler(SamplerConfig.greedyDecoding()).build();
        GenerationResult expected = plain(prompt, request);
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), config())) {
            LookaheadDecoder decoder = LookaheadDecoder.forModel(model, config());
            GenerationResult actual = decoder.generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals("stop reason", expected.stopReason(), actual.stopReason());
            assertArrayEquals("tokens", expected.tokens(), actual.tokens());
        }
    }

    @Test
    public void eachGenerationRestartsFromThePrompt() throws IOException {
        int[] prompt = repeatingPrompt();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), config())) {
            LookaheadDecoder decoder = LookaheadDecoder.forModel(model, config());
            int[] first = decoder.generate(prompt, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null)
                    .tokens();
            int[] second = decoder.generate(prompt, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null)
                    .tokens();
            assertArrayEquals("a second generation resets the memory", first, second);
        }
    }

    @Test
    public void promptLogprobsRefusedByName() throws IOException {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(4)
                .sampler(SamplerConfig.greedyDecoding()).promptLogprobs(1).build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), config())) {
            LookaheadDecoder decoder = LookaheadDecoder.forModel(model, config());
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> decoder.generate(repeatingPrompt(), request,
                            (com.eonmux.neuralfree.gguf.TokenCallback) null));
            assertTrue(e.getMessage(), e.getMessage().contains("prompt_logprobs"));
        }
    }

    @Test
    public void tokenLogprobsCoverEveryEmittedToken() throws IOException {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(8)
                .sampler(SamplerConfig.greedyDecoding()).logprobs(2).build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), config())) {
            LookaheadDecoder decoder = LookaheadDecoder.forModel(model, config());
            GenerationResult result = decoder.generate(repeatingPrompt(), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals("one capture per emitted token", result.tokens().length, result.logprobs().size());
        }
    }

    @Test
    public void seededSamplingRunsToItsBudgetAndRepeats() throws IOException {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(12)
                .sampler(SamplerConfig.builder().temperature(0.8f).seed(7).build()).build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), config())) {
            LookaheadDecoder decoder = LookaheadDecoder.forModel(model, config());
            GenerationResult first = decoder.generate(repeatingPrompt(), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals(StopReason.MAX_TOKENS, first.stopReason());
            assertEquals(12, first.tokens().length);
            GenerationResult second = decoder.generate(repeatingPrompt(), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertArrayEquals("the seeded sampler restarts with the generation", first.tokens(), second.tokens());
        }
    }

    @Test
    public void statsReportTheStepsAndTheirAcceptedTokens() throws IOException {
        int[] prompt = repeatingPrompt();
        GenerationRequest request = request();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), config())) {
            java.util.concurrent.atomic.AtomicReference<LookaheadDecoder.Stats> seen =
                    new java.util.concurrent.atomic.AtomicReference<>();
            LookaheadDecoder decoder = LookaheadDecoder.builder(model.forwardBackend(), model.tokenizer(), options())
                    .samplingDefaults(SamplingDefaults.forModel(model, config()))
                    .statsSink(seen::set)
                    .build();
            GenerationResult result = decoder.generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            LookaheadDecoder.Stats stats = seen.get();
            assertTrue("the decoder reported its stats", stats != null);
            assertTrue("at least one lookahead step", stats.rounds() > 0);
            assertEquals("emitted tokens", result.tokens().length, stats.emitted());
            assertTrue("accepted cannot exceed emitted", stats.accepted() <= stats.emitted());
            assertTrue("accepted tokens per step", stats.acceptanceRate() >= 0.0);
        }
    }

    @Test
    public void cancellationBeforeTheFirstTokenEmitsNothing() throws IOException {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(8)
                .sampler(SamplerConfig.greedyDecoding()).cancellation(() -> true).build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), config())) {
            LookaheadDecoder decoder = LookaheadDecoder.forModel(model, config());
            GenerationResult result = decoder.generate(repeatingPrompt(), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals(StopReason.CANCELLED, result.stopReason());
            assertEquals(0, result.tokens().length);
        }
    }

    @Test
    public void maxTokensStopKeepsTheBudget() throws IOException {
        int[] prompt = new int[] {4, 5, 6, 4, 5, 7, 4, 5};
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(5)
                .sampler(SamplerConfig.greedyDecoding()).build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), config())) {
            LookaheadDecoder decoder = LookaheadDecoder.forModel(model, config());
            GenerationResult result = decoder.generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals(StopReason.MAX_TOKENS, result.stopReason());
            assertEquals(5, result.tokens().length);
        }
    }

    @Test
    public void tooFewSequencesRefusedByName() throws IOException {
        MemoryConfig small = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(SEQUENCES - 1).kvUnified(true).build();
        EngineConfig cfg = EngineConfig.builder().memory(small).build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg)) {
            LookaheadDecoder decoder = LookaheadDecoder.builder(model.forwardBackend(), model.tokenizer(), options())
                    .build();
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> decoder.generate(new int[] {4, 5}, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null));
            assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.parallel"));
            assertTrue(e.getMessage(), e.getMessage().contains(String.valueOf(SEQUENCES)));
        }
    }

    @Test
    public void contiguousMemoryRefusedByTheConfiguration() {
        MemoryConfig contiguous = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.builder().memory(contiguous).lookaheadOptions(options()).build());
        assertTrue(e.getMessage(), e.getMessage().contains(LookaheadOptions.PROP_ENABLED));
        assertTrue(e.getMessage(), e.getMessage().contains("paged"));
    }

    @Test
    public void speculativeDecodingExcluded() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(SEQUENCES).kvUnified(true).build();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.builder().memory(memory).lookaheadOptions(options())
                        .draftTypes(DraftTypes.parse("ngram-simple")).build());
        assertTrue(e.getMessage(), e.getMessage().contains(LookaheadOptions.PROP_ENABLED));
        assertTrue(e.getMessage(), e.getMessage().contains("speculative"));
    }

}
