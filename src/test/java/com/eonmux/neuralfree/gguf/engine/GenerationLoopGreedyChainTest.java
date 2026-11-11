package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The generation loop's device-greedy chain ({@code ForwardBackend.chainsGreedyOnDevice}): a backend that chains
 * gets each greedy step's successor enqueued before the host offers the step's token, and the run must still be the
 * serial loop's run. Over the CPU forward path of a synthetic model, a chaining backend and a serial one generate
 * the same tokens, text and stop reason, and leave the same stored sequence, for a budget stop at every length, a
 * stop string at every generated token, a cancellation at every step and a full context; a chained step the run did
 * not use is dropped with one rollback, and the loop asks the backend before every chained step whether it still
 * chains. A step that fails on the device reaches the caller as its own exception, also when the step chained on it
 * fails with the same one, and the memory starts the next run clean.
 */
public class GenerationLoopGreedyChainTest {

    private static final int[] PROMPT = {4, 17, 30, 43, 56, 69};
    private static final int TOKENS = 8;

    private static Path model;

    @BeforeClass
    public static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    @AfterClass
    public static void deleteModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    private static EngineConfig config(int nCtx, boolean shift) {
        MemoryConfig mc = MemoryConfig.builder().nCtx(nCtx).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .contextShift(shift).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(mc).build();
    }

    private static GenerationRequest.Builder greedy(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.builder().greedy(true).penaltyLastN(0).build());
    }

    /** One generation's outcome and the sequence the memory stored after it. */
    private static final class Run {
        final GenerationResult result;
        final int[] stored;
        final int rollbacks;
        final int chained;

        Run(GenerationResult result, int[] stored, int rollbacks, int chained) {
            this.result = result;
            this.stored = stored;
            this.rollbacks = rollbacks;
            this.chained = chained;
        }
    }

    /** Generates {@code request} on a fresh model, through a backend that chains greedy steps or not. */
    private static Run run(EngineConfig config, GenerationRequest request, boolean chains) throws IOException {
        return run(config, request, chains ? Integer.MAX_VALUE : 0);
    }

    /**
     * Generates {@code request} on a fresh model, through a backend that answers that it chains greedy steps to the
     * first {@code chainingAnswers} questions and that it does not afterwards.
     */
    private static Run run(EngineConfig config, GenerationRequest request, int chainingAnswers) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), config)) {
            ChainCountingBackend backend = new ChainCountingBackend(m.forwardBackend(), chainingAnswers);
            GenerationResult result = GenerationLoop.builder(backend, m.tokenizer())
                    .contextShift(config.memory().contextShift())
                    .build()
                    .generate(PROMPT, request, null);
            return new Run(result, m.forwardBackend().memory().sequence().tokens(), backend.rollbacks,
                    backend.chained);
        }
    }

    private static void assertSameRun(String what, Run serial, Run chained) {
        assertArrayEquals(what + ": tokens", serial.result.tokens(), chained.result.tokens());
        assertEquals(what + ": text", serial.result.text(), chained.result.text());
        assertEquals(what + ": stop reason", serial.result.stopReason(), chained.result.stopReason());
        assertArrayEquals(what + ": stored sequence", serial.stored, chained.stored);
        assertEquals(what + ": the serial loop never rolls back", 0, serial.rollbacks);
        assertEquals(what + ": the serial loop never chains", 0, serial.chained);
    }

    @Test
    public void aBudgetStopAtEveryLengthMatchesTheSerialLoop() throws IOException {
        EngineConfig config = config(64, false);
        for (int n = 1; n <= TOKENS; n++) {
            Run serial = run(config, greedy(n).build(), false);
            Run chained = run(config, greedy(n).build(), true);
            assertSameRun("budget " + n, serial, chained);
            assertEquals("budget " + n + " generates its tokens", n, chained.result.tokens().length);
            if (n > 1) {
                assertTrue("budget " + n + " chains its later steps: " + chained.chained, chained.chained > 0);
                assertEquals("budget " + n + " drops the step chained after its last token", 1,
                        chained.rollbacks);
            }
        }
    }

    @Test
    public void aStopStringAtEveryTokenMatchesTheSerialLoop() throws IOException {
        EngineConfig config = config(64, false);
        Run reference = run(config, greedy(TOKENS).build(), false);
        String text = reference.result.text();
        int stops = 0;
        for (int i = 1; i < text.length(); i++) {
            String stop = text.substring(Math.max(0, i - 2), i + 1);
            if (text.indexOf(stop) != Math.max(0, i - 2) || stop.isBlank()) {
                continue;
            }
            Run serial = run(config, greedy(TOKENS).stopString(stop).build(), false);
            Run chained = run(config, greedy(TOKENS).stopString(stop).build(), true);
            assertSameRun("stop string '" + stop + "'", serial, chained);
            assertEquals("stop string '" + stop + "' stops the run", StopReason.STOP_STRING,
                    chained.result.stopReason());
            stops++;
        }
        assertTrue("the generated text offers stop strings: '" + text + "'", stops > 0);
    }

    @Test
    public void aCancellationAtEveryStepMatchesTheSerialLoop() throws IOException {
        EngineConfig config = config(64, false);
        for (int at = 1; at <= TOKENS; at++) {
            Run serial = run(config, greedy(TOKENS).cancellation(cancelAt(at)).build(), false);
            Run chained = run(config, greedy(TOKENS).cancellation(cancelAt(at)).build(), true);
            assertSameRun("cancel at check " + at, serial, chained);
        }
    }

    @Test
    public void aFullContextMatchesTheSerialLoop() throws IOException {
        for (int nCtx = PROMPT.length + 1; nCtx <= PROMPT.length + TOKENS; nCtx++) {
            EngineConfig config = config(nCtx, false);
            Run serial = run(config, greedy(TOKENS + 4).build(), false);
            Run chained = run(config, greedy(TOKENS + 4).build(), true);
            assertSameRun("context " + nCtx, serial, chained);
            assertEquals("context " + nCtx + " ends full", StopReason.CONTEXT_FULL, chained.result.stopReason());
        }
    }

    @Test
    public void aContextShiftMatchesTheSerialLoop() throws IOException {
        EngineConfig config = config(PROMPT.length + 4, true);
        Run serial = run(config, greedy(TOKENS + 6).build(), false);
        Run chained = run(config, greedy(TOKENS + 6).build(), true);
        assertSameRun("context shift", serial, chained);
        assertEquals("the run shifts its context instead of ending", TOKENS + 6, chained.result.tokens().length);
    }

    @Test
    public void aBackendThatStopsChainingIsAskedBeforeEveryChainedStep() throws IOException {
        EngineConfig config = config(64, false);
        Run serial = run(config, greedy(TOKENS).build(), false);
        for (int answers = 1; answers <= 4; answers++) {
            Run partly = run(config, greedy(TOKENS).build(), answers);
            assertSameRun("chaining for " + answers + " answers", serial, partly);
            assertEquals("the loop chains only while the backend says it can", answers - 1, partly.chained);
        }
    }

    @Test
    public void aFailedStepReachesTheCallerAsItselfAndTheNextRunStartsClean() throws IOException {
        EngineConfig config = config(64, false);
        Run reference = run(config, greedy(TOKENS).build(), false);
        for (int failAt = 1; failAt <= 4; failAt++) {
            try (GGUFModel m = new GGUFModel(model.toString(), config)) {
                FailingChainBackend backend = new FailingChainBackend(m.forwardBackend(), failAt);
                GenerationLoop loop = GenerationLoop.builder(backend, m.tokenizer()).build();
                RuntimeException thrown = assertThrows(RuntimeException.class,
                        () -> loop.generate(PROMPT, greedy(TOKENS).build(), null));
                assertSame("fail at chained step " + failAt + ": the device's own exception", backend.failure,
                        thrown);
                assertEquals("fail at chained step " + failAt + ": the step chained on the failed one adds no copy",
                        0, thrown.getSuppressed().length);
                assertEquals("fail at chained step " + failAt + ": the failed run leaves no sequence", 0,
                        m.forwardBackend().memory().sequence().nPast());
                GenerationResult again = GenerationLoop.builder(m.forwardBackend(), m.tokenizer()).build()
                        .generate(PROMPT, greedy(TOKENS).build(), null);
                assertArrayEquals("fail at chained step " + failAt + ": the next run is the serial run",
                        reference.result.tokens(), again.tokens());
            }
        }
    }

    /** A cancellation hook that reports a cancel from its {@code at}-th poll on. */
    private static java.util.function.BooleanSupplier cancelAt(int at) {
        AtomicInteger polls = new AtomicInteger();
        return () -> polls.incrementAndGet() >= at;
    }

    /**
     * The CPU forward path, chaining greedy steps when asked: every call runs the real forward path, and the backend
     * counts the steps chained and the rollbacks of the loop.
     */
    private static class ChainCountingBackend implements ForwardBackend {

        private final ForwardBackend delegate;
        private int chainingAnswers;
        int chained;
        int rollbacks;

        ChainCountingBackend(ForwardBackend delegate, int chainingAnswers) {
            this.delegate = delegate;
            this.chainingAnswers = chainingAnswers;
        }

        @Override
        public boolean chainsGreedyOnDevice() {
            if (chainingAnswers <= 0) {
                return false;
            }
            chainingAnswers--;
            return true;
        }

        @Override
        public GreedyStep chainGreedy(GreedyStep previous, int pos) {
            chained++;
            return ForwardBackend.super.chainGreedy(previous, pos);
        }

        @Override
        public float[] prefill(int[] tokens, int pos0) {
            return delegate.prefill(tokens, pos0);
        }

        @Override
        public float[] decode(int token, int pos) {
            return delegate.decode(token, pos);
        }

        @Override
        public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
            return delegate.forwardRows(tokens, pos0, outputRows);
        }

        @Override
        public float[][] forwardAll(int[] tokens, int[] outputRows) {
            return delegate.forwardAll(tokens, outputRows);
        }

        @Override
        public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
            return delegate.hiddenRows(tokens, tokenTypes);
        }

        @Override
        public void reset() {
            delegate.reset();
        }

        @Override
        public boolean canRollback(int nTokens) {
            return delegate.canRollback(nTokens);
        }

        @Override
        public void rollback(int nTokens) {
            rollbacks++;
            delegate.rollback(nTokens);
        }

        @Override
        public void shiftContext(int nKeep, int nDiscard) {
            delegate.shiftContext(nKeep, nDiscard);
        }

        @Override
        public ModelMemory memory() {
            return delegate.memory();
        }

        @Override
        public void setHiddenTap(HiddenTap tap) {
            delegate.setHiddenTap(tap);
        }

        @Override
        public void close() {
            delegate.close();
        }
    }

    /**
     * A chaining backend whose {@code failAt}-th chained step stores its token and then fails on the device: its
     * id, and the id of the step chained on it (which stores a placeholder entry, as a device-chained step does
     * before its input id is known), throw the same exception, as a GPU chain reports a step whose input step
     * failed.
     */
    private static final class FailingChainBackend extends ChainCountingBackend {

        final RuntimeException failure = new IllegalStateException("device lost");
        private final int failAt;
        private GreedyStep failed;

        FailingChainBackend(ForwardBackend delegate, int failAt) {
            super(delegate, Integer.MAX_VALUE);
            this.failAt = failAt;
        }

        @Override
        public GreedyStep chainGreedy(GreedyStep previous, int pos) {
            chained++;
            if (previous == failed) {
                decode(0, pos);
                return () -> {
                    throw failure;
                };
            }
            GreedyStep step = startGreedy(previous.id(), pos);
            if (chained == failAt) {
                failed = () -> {
                    throw failure;
                };
                return failed;
            }
            return step;
        }
    }
}
