package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.backend;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A multi-input pooling call over a memoryless encoder runs its one-shot forwards in as few pump handoffs as the pump
 * allows, and keeps the per-request contract of one handoff per input: another caller's request waiting for the pump
 * runs between the batch's inputs, the request budget bounds each input's wait rather than the whole batch, and a
 * batch whose caller gave up stops before its next input, so the pump is free again within about one input.
 */
class PoolingOneShotHandoffTest {

    /** The encoder fixture's longest input (its training context is 128). */
    private static final int TOKENS = 120;

    private static Path encoder;

    @BeforeAll
    static void open() throws IOException {
        encoder = SyntheticModels.bert().writeTempFile();
    }

    private static EngineService service(Duration budget) throws IOException {
        EngineConfig config = EngineConfig.builder().backend(backend()).build();
        return new EngineService(encoder.toString(), config, new ServerOptions(null, budget));
    }

    private static int[] input(int seed) {
        int[] tokens = new int[TOKENS];
        for (int t = 0; t < tokens.length; t++) {
            tokens[t] = 4 + (seed * 7 + t * 13) % 300;
        }
        return tokens;
    }

    private static int[][] batch(int n) {
        int[][] out = new int[n][];
        for (int i = 0; i < n; i++) {
            out[i] = input(i);
        }
        return out;
    }

    /** The median nanoseconds of one input's forward on {@code service}, after a warm-up. */
    private static long oneInputNanos(EngineService service) throws IOException {
        long[] times = new long[9];
        service.pooling().embed(null, input(0));
        for (int i = 0; i < times.length; i++) {
            long start = System.nanoTime();
            service.pooling().embed(null, input(i));
            times[i] = System.nanoTime() - start;
        }
        Arrays.sort(times);
        return Math.max(times[times.length / 2], 1);
    }

    private static CompletableFuture<float[][]> embedAllAsync(EngineService service, int[][] inputs) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return service.pooling().embedAll(null, inputs);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    @Test
    void anotherCallersRequestRunsBetweenTheBatchesInputs() throws Exception {
        try (EngineService service = service(ServerOptions.DEFAULT_TIMEOUT)) {
            long perInput = oneInputNanos(service);
            int n = (int) Math.min(20_000, Math.max(200, TimeUnit.MILLISECONDS.toNanos(2_000) / perInput));
            int[][] inputs = batch(n);
            CompletableFuture<float[][]> long_ = embedAllAsync(service, inputs);
            Thread.sleep(Math.max(50, TimeUnit.NANOSECONDS.toMillis(perInput * n / 10)));
            assertFalse(long_.isDone(), "the batch of " + n + " inputs is still running");
            float[] single = service.pooling().embed(null, input(n + 1));
            assertFalse(long_.isDone(), "the single request answered while the batch was still running");
            float[][] answers = long_.get(120, TimeUnit.SECONDS);
            assertEquals(n, answers.length);
            assertArrayEquals(service.pooling().embed(null, inputs[n - 1]), answers[n - 1], 1e-6f);
            assertEquals(answers[0].length, single.length);
        }
    }

    @Test
    void theBudgetBoundsEachInputsWaitNotTheWholeBatch() throws Exception {
        long perInput;
        try (EngineService probe = service(ServerOptions.DEFAULT_TIMEOUT)) {
            perInput = oneInputNanos(probe);
        }
        Duration budget = Duration.ofNanos(Math.max(TimeUnit.MILLISECONDS.toNanos(250), 20 * perInput));
        int n = (int) Math.min(20_000, 4 * budget.toNanos() / perInput);
        try (EngineService service = service(budget)) {
            float[][] answers = service.pooling().embedAll(null, batch(n));
            assertEquals(n, answers.length, "a batch running four budgets answers every input");
        }
    }

    @Test
    void aBatchWhoseCallerGaveUpFreesThePumpWithinAboutOneInput() throws Exception {
        long perInput;
        try (EngineService probe = service(ServerOptions.DEFAULT_TIMEOUT)) {
            perInput = oneInputNanos(probe);
        }
        int n = (int) Math.min(20_000, Math.max(200, TimeUnit.MILLISECONDS.toNanos(3_000) / perInput));
        // a budget a quarter of one input: the caller sees no input finish within it and gives up
        try (EngineService service = service(Duration.ofNanos(Math.max(perInput / 4, 1)))) {
            IOException refused = assertThrows(IOException.class, () -> service.pooling().embedAll(null, batch(n)));
            assertTrue(refused.getMessage().contains("budget"), refused.getMessage());
            CountDownLatch probe = new CountDownLatch(1);
            long start = System.nanoTime();
            service.replicas().get(0).work().put(probe::countDown);
            assertTrue(probe.await(60, TimeUnit.SECONDS), "the pump took the next task");
            long waited = System.nanoTime() - start;
            assertTrue(waited < perInput * n / 4, "the pump was free after " + TimeUnit.NANOSECONDS.toMillis(waited)
                    + " ms, a batch of " + n + " inputs runs " + TimeUnit.NANOSECONDS.toMillis(perInput * n) + " ms");
        }
    }
}
