package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The load-balance methods of the data-parallel pool ({@code neuralfree.server.load.balance}): round robin rotates
 * whatever the load or the prefix affinity, minimum tokens routes by the prompt tokens in flight, and the in-flight
 * count returns to zero once the requests finish. Each route is read off the replica's in-flight prompt tokens,
 * which count from the moment of submission.
 */
class EngineServiceDpBalanceTest {

    private static final TokenCallback IGNORED = (token, text) -> { };

    private static Path model;

    private EngineService pool;
    private final List<EngineService.Submission> submitted = new ArrayList<>();

    @BeforeAll
    static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    @AfterEach
    void close() throws InterruptedException {
        if (pool != null) {
            pool.abortAll();
            for (EngineService.Submission submission : submitted) {
                submission.await(30, TimeUnit.SECONDS);
            }
            pool.close();
        }
    }

    private void open(LoadBalance method) throws IOException {
        pool = new EngineService(model.toString(), ServerTestSupport.pagedConfig(512, 2),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null)
                        .withDataParallel(new DpLayout(2, 0, method)));
    }

    /** A greedy request that keeps its replica busy until aborted. */
    private static GenerationRequest longRequest() {
        return GenerationRequest.builder().maxNewTokens(400).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static GenerationRequest shortRequest() {
        return GenerationRequest.builder().maxNewTokens(2).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private EngineService.Submission submit(int[] prompt, GenerationRequest request) throws InterruptedException {
        EngineService.Submission submission = pool.submit(prompt, request, IGNORED);
        submitted.add(submission);
        return submission;
    }

    private long[] inFlight() {
        return new long[]{pool.replicaPromptTokensInFlight(0), pool.replicaPromptTokensInFlight(1)};
    }

    private long served(int replica) {
        return pool.replicaStats(replica).promptTokens() + pool.replicaStats(replica).promptTokensCached();
    }

    /** Waits until both replicas published idle snapshots. */
    private void awaitIdle() throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (pool.stats().running() + pool.stats().waiting() > 0) {
            assertTrue(System.nanoTime() < deadline, "the pool never went idle");
            Thread.sleep(5);
        }
    }

    @Test
    @Timeout(120)
    void roundRobinRotatesWhateverTheLoad() throws InterruptedException, IOException {
        open(LoadBalance.ROUND_ROBIN);
        int[] prompt = pool.encode("the rotating prompt", true, false);
        submit(prompt, longRequest());
        submit(prompt, longRequest());
        submit(prompt, longRequest());
        assertEquals(2L * prompt.length, inFlight()[0], "the first and third requests went to replica 0");
        assertEquals(prompt.length, inFlight()[1], "the second went to replica 1, though replica 0 was busier");
        assertEquals(3, pool.abortAll());
        for (EngineService.Submission submission : submitted) {
            assertTrue(submission.await(30, TimeUnit.SECONDS));
        }
        assertEquals(0L, inFlight()[0] + inFlight()[1], "finished requests leave the in-flight count");
    }

    @Test
    @Timeout(120)
    void roundRobinIgnoresThePrefixAffinityOfAnIdlePool() throws InterruptedException, IOException {
        open(LoadBalance.ROUND_ROBIN);
        int[] prompt = pool.encode("the repeated prompt", true, false);
        for (int i = 0; i < 4; i++) {
            assertTrue(submit(prompt, shortRequest()).await(30, TimeUnit.SECONDS));
            awaitIdle();
        }
        assertEquals(2L * prompt.length, served(0), "replica 0 served the first and third repeat");
        assertEquals(2L * prompt.length, served(1),
                "replica 1 served the second and fourth, where shortest_queue keeps the replica with the prefix");
    }

    @Test
    @Timeout(120)
    void minimumTokensRoutesByThePromptTokensInFlight() throws InterruptedException, IOException {
        open(LoadBalance.MINIMUM_TOKENS);
        int[] longPrompt = pool.encode("a long prompt that keeps going with many words so that it holds far more"
                + " tokens than the short one does, and then some more words for good measure", true, false);
        int[] shortPrompt = pool.encode("hi", true, false);
        assertTrue(longPrompt.length > 3 * shortPrompt.length, "the long prompt outweighs two short ones");

        submit(longPrompt, longRequest());
        assertEquals(longPrompt.length, inFlight()[0], "an idle tie goes to the first replica in turn");
        submit(shortPrompt, longRequest());
        assertEquals(shortPrompt.length, inFlight()[1]);
        submit(shortPrompt, longRequest());
        submit(shortPrompt, longRequest());
        assertEquals(longPrompt.length, inFlight()[0], "replica 0 holds more tokens, so it took nothing more");
        assertEquals(3L * shortPrompt.length, inFlight()[1],
                "replica 1 took both, though it held more requests (shortest_queue would split them)");
    }

    @Test
    @Timeout(120)
    void minimumTokensBreaksATokenTieByTheRequestCount() throws InterruptedException, IOException {
        open(LoadBalance.MINIMUM_TOKENS);
        int[] prompt = pool.encode("the same prompt", true, false);
        submit(prompt, longRequest());
        submit(prompt, longRequest());
        assertEquals(prompt.length, inFlight()[0]);
        assertEquals(prompt.length, inFlight()[1], "the second went to the replica with no tokens in flight");
        submit(prompt, longRequest());
        submit(prompt, longRequest());
        assertEquals(2L * prompt.length, inFlight()[0]);
        assertEquals(2L * prompt.length, inFlight()[1], "ties spread over the replicas");
    }
}
