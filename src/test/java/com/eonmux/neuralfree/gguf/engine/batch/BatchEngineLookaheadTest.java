package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LookaheadOptions;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Lookahead (Jacobi) decoding inside the batch engine: one running request owns a group of {@code W + G + 1}
 * sequences and each step runs its own Jacobi and n-gram batch, so a served request computes the tokens the plain
 * {@link GenerationLoop} computes for it, several lookahead requests run side by side, the engine reports the step
 * and acceptance counters, and the requests the scheme cannot serve refuse by name.
 */
public class BatchEngineLookaheadTest {

    private static final int WINDOW = 4;
    private static final int NGRAM = 3;
    private static final int MAX_NGRAMS = 4;
    private static final int SEQUENCES = WINDOW + MAX_NGRAMS + 1;
    private static final int N_CTX = 2048;
    private static final int N_NEW = 16;

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

    private static LookaheadOptions lookahead() {
        return LookaheadOptions.builder().enabled(true).window(WINDOW).ngramSize(NGRAM)
                .maxNgrams(MAX_NGRAMS).build();
    }

    private static EngineConfig config(int requests) {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .parallel(SEQUENCES * requests).kvUnified(true).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory)
                .lookaheadOptions(lookahead()).build();
    }

    private static int[] prompt(int first) {
        int[] p = new int[12];
        for (int i = 0; i < p.length; i++) {
            p[i] = 4 + (i % 3) + first;
        }
        return p;
    }

    private static GenerationRequest greedy() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static GenerationResult plain(int[] prompt) throws IOException {
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, greedy(), (TokenCallback) null);
        }
    }

    private static void assertDone(BatchSequence sequence) {
        assertTrue("expected DONE but was " + sequence.state()
                        + (sequence.error() == null ? "" : ": " + sequence.error()),
                sequence.state() == BatchSequence.State.DONE);
    }

    @Test
    public void servedLookaheadMatchesThePlainGreedyLoop() throws IOException {
        int[] prompt = prompt(0);
        GenerationResult expected = plain(prompt);
        EngineConfig cfg = config(1);
        try (GGUFModel one = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(one, cfg)) {
            BatchSequence sequence = engine.submit(prompt, greedy(), (TokenCallback) null);
            engine.drain();
            assertDone(sequence);
            assertArrayEquals("lookahead serving keeps the plain greedy tokens", expected.tokens(),
                    sequence.result().tokens());
            assertEquals("stop reason", expected.stopReason(), sequence.result().stopReason());
            assertEquals("text", expected.text(), sequence.result().text());
            BatchStats stats = engine.stats();
            assertTrue("lookahead steps ran: " + stats, stats.lookaheadRounds() > 0);
            assertTrue("tokens were emitted: " + stats, stats.lookaheadEmitted() > 0);
            assertTrue("accepted tokens did not outrun the steps: " + stats,
                    stats.lookaheadAccepted() <= stats.lookaheadEmitted());
        }
    }

    @Test
    public void twoLookaheadRequestsRunSideBySide() throws IOException {
        int[] a = prompt(0);
        int[] b = prompt(1);
        GenerationResult expectedA = plain(a);
        GenerationResult expectedB = plain(b);
        EngineConfig cfg = config(2);
        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            BatchSequence first = engine.submit(a, greedy(), (TokenCallback) null);
            BatchSequence second = engine.submit(b, greedy(), (TokenCallback) null);
            engine.drain();
            assertDone(first);
            assertDone(second);
            assertArrayEquals("request A", expectedA.tokens(), first.result().tokens());
            assertArrayEquals("request B", expectedB.tokens(), second.result().tokens());
            assertTrue("both requests ran lookahead steps: " + engine.stats(),
                    engine.stats().lookaheadRounds() >= 2);
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
    public void repeatingPromptAcceptsObservedNgrams() throws IOException {
        int[] prompt = repeatingPrompt();
        GenerationResult expected = plain(prompt);
        EngineConfig cfg = config(1);
        try (GGUFModel one = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(one, cfg)) {
            BatchSequence sequence = engine.submit(prompt, greedy(), (TokenCallback) null);
            engine.drain();
            assertDone(sequence);
            assertArrayEquals("lookahead serving keeps the plain greedy tokens", expected.tokens(),
                    sequence.result().tokens());
            BatchStats stats = engine.stats();
            assertTrue("the repeating prompt offers observed n-grams: " + stats, stats.lookaheadAccepted() > 0);
            assertTrue("an accepted n-gram saves a step: " + stats,
                    stats.lookaheadEmitted() > stats.lookaheadRounds());
            assertTrue("acceptance does not outrun the emission: " + stats,
                    stats.lookaheadAccepted() < stats.lookaheadEmitted());
        }
    }

    @Test
    public void mediaAndShiftedRequestsRefuseByName() throws IOException {
        EngineConfig cfg = config(1);
        try (GGUFModel one = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(one, cfg)) {
            GenerationRequest shifted = GenerationRequest.builder().maxNewTokens(4)
                    .sampler(SamplerConfig.greedyDecoding()).nKeep(2).build();
            UnsupportedOperationException nKeep = assertThrows(UnsupportedOperationException.class,
                    () -> engine.submit(prompt(0), shifted, (TokenCallback) null));
            assertTrue(nKeep.getMessage(), nKeep.getMessage().contains("n_keep"));

            GenerationRequest grouped = GenerationRequest.builder().maxNewTokens(4)
                    .sampler(SamplerConfig.greedyDecoding()).grpAttnN(2).grpAttnW(4).build();
            UnsupportedOperationException extend = assertThrows(UnsupportedOperationException.class,
                    () -> engine.submit(prompt(0), grouped, (TokenCallback) null));
            assertTrue(extend.getMessage(), extend.getMessage().contains("grp_attn_n"));
        }
    }
}
