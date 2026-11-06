package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.DraftTypes;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Speculative decoding through the serving stack: an engine opened with {@code neuralfree.draft.type}
 * {@code ngram-simple}, or with a lookup chain and a separate draft model under rejection verification, drafts and
 * verifies inside the batch engine the HTTP endpoints submit into, so every greedy choice returns the same tokens as
 * the plain engine while the spec counters move. An engine configured by vLLM's {@code --speculative-config} launch
 * JSON searches its prompt-lookup size range the same way. A grammar-constrained
 * request speculates too: its proposals are cut by the grammar before they are verified, and its greedy completion is
 * the plain engine's.
 */
class EngineSpeculativeTest {

    private static EngineService plain;
    private static EngineService speculative;
    private static EngineService range;
    /** A lookup chain and a separate draft model, verified by rejection sampling. */
    private static EngineService drafted;
    /** The bound meters; micrometer reads them through a weak reference, so the test keeps them reachable. */
    private static volatile EngineMetrics bound;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        plain = new EngineService(model.toString(), paged().build());
        // the 3-gram the (4, 5, v) prompt repeats; llama.cpp's default 12-gram finds no match in it
        speculative = new EngineService(model.toString(), paged().draftType(EngineConfig.DraftType.NGRAM)
                .speculativeOptions(SpeculativeOptions.builder().ngram(NgramDrafter.Options
                        .builder(NgramDrafter.Kind.SIMPLE).sizeN(3).sizeM(8).build()).build()).build());
        Properties launch = new Properties();
        launch.putAll(ServerLaunchArgs.parse("--speculative-config", "{\"method\": \"ngram\","
                + " \"num_speculative_tokens\": 3, \"prompt_lookup_max\": 12, \"prompt_lookup_min\": 2}")
                .properties());
        range = new EngineService(model.toString(), EngineConfig.fromProperties(launch).toBuilder()
                .memory(paged().build().memory()).build());
        drafted = new EngineService(model.toString(), paged()
                .draftTypes(DraftTypes.parse("ngram-map-k,suffix,draft-simple")).draftModelPath(model.toString())
                .speculativeOptions(SpeculativeOptions.builder().verifyMode(VerifyMode.REJECTION)
                        .probabilisticDraft(true).build())
                .build());
    }

    @AfterAll
    static void close() {
        plain.close();
        speculative.close();
        range.close();
        drafted.close();
        bound = null;
    }

    /** A paged serving config wide enough for the lookup prompt (per-slot context nCtx / parallel). */
    private static EngineConfig.Builder paged() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(768).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build();
        return EngineConfig.builder().memory(mc);
    }

    /**
     * The prompt whose stored {@code (4, 5, v)} triples make the prompt-lookup drafter propose from the first
     * decode step on: the fixture's greedy continuation is a run of token 86, which the prompt holds after the
     * (4, 5) pair (pinned once against the synthetic model's fixed weights).
     */
    private static int[] lookupPrompt() {
        int[] prompt = new int[3 * 100 + 2];
        int w = 0;
        for (int v = 6; v <= 105; v++) {
            prompt[w++] = 4;
            prompt[w++] = 5;
            prompt[w++] = v;
        }
        prompt[w++] = 4;
        prompt[w++] = 5;
        return prompt;
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder()
                .maxNewTokens(8)
                .sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    /** Submits to {@code service} and waits out the terminal state. */
    private static com.eonmux.neuralfree.gguf.engine.batch.BatchSequence run(EngineService service, int[] prompt)
            throws InterruptedException {
        EngineService.Submission submission = service.submit(prompt, request(),
                (com.eonmux.neuralfree.gguf.TokenCallback) (token, text) -> { });
        assertTrue(submission.await(30, TimeUnit.SECONDS), "the generation outlived its wait");
        return submission.sequence();
    }

    @Test
    void theSpeculativeEngineServesTheSameTokensAsThePlainOneAndCountsRounds() throws InterruptedException {
        int[] prompt = lookupPrompt();
        com.eonmux.neuralfree.gguf.engine.batch.BatchSequence plainRun = run(plain, prompt);
        com.eonmux.neuralfree.gguf.engine.batch.BatchSequence specRun = run(speculative, prompt);

        assertEquals(com.eonmux.neuralfree.gguf.engine.batch.BatchSequence.State.DONE, plainRun.state());
        assertEquals(com.eonmux.neuralfree.gguf.engine.batch.BatchSequence.State.DONE, specRun.state());
        assertArrayEquals(plainRun.result().tokens(), specRun.result().tokens(),
                "speculation is lossless over the serving path");
        assertEquals(plainRun.result().text(), specRun.result().text());
        assertTrue(speculative.stats().specDrafted() > 0);
        assertTrue(speculative.stats().specRounds() > 0);
        assertTrue(speculative.stats().specAccepted() <= speculative.stats().specDrafted());
    }

    /** A greedy {@code /completion} body over the lookup prompt, with {@code grammar} (GBNF, JSON-escaped) or none. */
    private static String completionBody(String grammar) {
        return "{\"prompt\": " + Arrays.toString(lookupPrompt()) + ", \"n_predict\": 8, \"temperature\": 0"
                + (grammar == null ? "" : ", \"grammar\": \"" + grammar + "\"") + "}";
    }

    /**
     * The synthetic llama's ids 4..105 are its byte tokens {@code 0x01..0x66}: the lookup prompt's first proposal
     * {@code 4 5 87} spells {@code \x01 \x02 T}, and the greedy continuation is a run of {@code S} (id 86). The
     * grammar {@code [\x01\x02S]+} admits that run and cuts the proposal before {@code T}, so the constrained
     * request verifies fewer drafted rows than the unconstrained one on the same prompt, while its completion is
     * the plain engine's.
     */
    @Test
    void aGrammarConstrainedGreedyCompletionIsTheSameWithAndWithoutTheDrafter() {
        String constrained = completionBody("root ::= [\\\\x01\\\\x02S]+");
        LlamaCompletionResource plainApi = new LlamaCompletionResource(plain);
        LlamaCompletionResource specApi = new LlamaCompletionResource(speculative);

        long before = speculative.stats().specDrafted();
        specApi.completion(completionBody(null));
        long unconstrainedDrafted = speculative.stats().specDrafted() - before;
        com.eonmux.neuralfree.utils.json.JsonView specBody = view(specApi.completion(constrained).getEntity());
        long constrainedDrafted = speculative.stats().specDrafted() - before - unconstrainedDrafted;
        com.eonmux.neuralfree.utils.json.JsonView plainBody = view(plainApi.completion(constrained).getEntity());

        String content = plainBody.str("content");
        assertTrue(content.matches("S+"), "constrained content was '" + content + "'");
        assertEquals(content, specBody.str("content"));
        assertEquals(plainBody.raw("tokens"), specBody.raw("tokens"), "speculation is lossless under the grammar");
        assertEquals(plainBody.i64("tokens_predicted"), specBody.i64("tokens_predicted"));
        assertTrue(constrainedDrafted > 0, "the admitted prefix of the proposal was verified");
        assertTrue(constrainedDrafted < unconstrainedDrafted, "the grammar cut the proposal: "
                + constrainedDrafted + " drafted rows against " + unconstrainedDrafted + " unconstrained");
    }

    @Test
    void vllmsPromptLookupRangeDraftsWhereTheSingleLlamaCppSizeCannot() throws InterruptedException {
        int[] prompt = lookupPrompt();
        com.eonmux.neuralfree.gguf.engine.batch.BatchSequence plainRun = run(plain, prompt);
        com.eonmux.neuralfree.gguf.engine.batch.BatchSequence rangeRun = run(range, prompt);
        assertEquals(com.eonmux.neuralfree.gguf.engine.batch.BatchSequence.State.DONE, rangeRun.state());
        assertArrayEquals(plainRun.result().tokens(), rangeRun.result().tokens());
        assertTrue(range.stats().specDrafted() > 0, "the 2..12 range matches the prompt's shorter repeats");
        assertTrue(range.stats().specAccepted() <= range.stats().specDrafted());
    }

    @Test
    void theMetricsExposeTheSpecCounters() throws InterruptedException {
        EngineLifecycle lifecycle = new EngineLifecycle() {
            @Override
            EngineService current() {
                return speculative;
            }
        };
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        bound = new EngineMetrics(lifecycle);
        bound.bindTo(registry);
        run(speculative, lookupPrompt());
        assertTrue(registry.get("neuralfree.spec.rounds").functionCounter().count() > 0);
        assertTrue(registry.get("neuralfree.spec.drafted.tokens").functionCounter().count() > 0);
        assertTrue(registry.get("neuralfree.spec.accepted.tokens").functionCounter().count()
                <= registry.get("neuralfree.spec.drafted.tokens").functionCounter().count());
    }

    @Test
    void theLookupChainAndADraftModelServeTheSameTokensUnderRejectionVerification() throws InterruptedException {
        int[] prompt = lookupPrompt();
        com.eonmux.neuralfree.gguf.engine.batch.BatchSequence plainRun = run(plain, prompt);
        com.eonmux.neuralfree.gguf.engine.batch.BatchSequence draftedRun = run(drafted, prompt);
        assertEquals(com.eonmux.neuralfree.gguf.engine.batch.BatchSequence.State.DONE, draftedRun.state(),
                String.valueOf(draftedRun.error()));
        assertArrayEquals(plainRun.result().tokens(), draftedRun.result().tokens(),
                "greedy requests keep their tokens under rejection verification");
        assertTrue(drafted.stats().specDrafted() > 0);
        assertTrue(drafted.stats().specAccepted() > 0);
    }
}
