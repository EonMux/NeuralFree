package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine samples a step's host rows in parallel ({@code OutputRouter}): sequences whose chains hold only
 * their own state draw on worker threads, the rest (a grammar, a logprobs capture) on the engine's thread, and every
 * sequence still carries the tokens, text and stop of the same prompt and request run alone through
 * {@link GenerationLoop} — for seeded whole-row chains (top-k off, top-p, min-p, typical, penalties, DRY, mirostat)
 * beside one another, beside prompts completing in the same steps, and beside the serial chains. A failure or a
 * cancellation inside a parallel step ends each request once, the way the serial route ends it: a piece that fails
 * retires itself and every later piece of the step with its error, an earlier piece keeps its token, and a cancelled
 * request finishes when the route reaches it.
 */
public class BatchParallelSamplingTest {

    private static final int PARALLEL = 7;
    private static final int N_NEW = 10;

    private static EngineConfig config(boolean paged) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            mc = mc.parallel(PARALLEL);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    private static GenerationRequest request(SamplerConfig sampler) {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(sampler).build();
    }

    /** Seeded chains that sample whole rows on the host, each with its own state. */
    private static List<GenerationRequest> independentRequests() {
        List<GenerationRequest> out = new ArrayList<>();
        out.add(request(SamplerConfig.builder().topK(0).topP(0.9f).minP(0f).temperature(0.8f).seed(11L).build()));
        out.add(request(SamplerConfig.builder().topK(0).topP(1f).minP(0.05f).typicalP(0.9f).temperature(1.1f)
                .seed(12L).build()));
        out.add(request(SamplerConfig.builder().topK(40).topP(0.95f).temperature(0.7f).penaltyLastN(16)
                .repeatPenalty(1.3f).frequencyPenalty(0.2f).seed(13L).build()));
        out.add(request(SamplerConfig.builder().topK(0).topP(0.8f).temperature(0.9f).dryMultiplier(0.8f)
                .dryBase(1.75f).dryAllowedLength(2).dryPenaltyLastN(32).seed(14L).build()));
        out.add(request(SamplerConfig.builder().mirostat(2).mirostatTau(4f).mirostatEta(0.2f).temperature(0.9f)
                .seed(17L).build()));
        return out;
    }

    private static GenerationResult reference(Path model, int[] prompt, GenerationRequest request)
            throws IOException {
        EngineConfig cfg = config(false);
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    private static void assertMatches(String what, GenerationResult reference, BatchSequence sequence) {
        assertTrue(what + " expected DONE but was " + sequence.state()
                        + (sequence.error() == null ? "" : ": " + sequence.error()),
                sequence.state() == BatchSequence.State.DONE);
        GenerationResult batched = sequence.result();
        assertEquals(what + " stop reason", reference.stopReason(), batched.stopReason());
        assertArrayEquals(what + " token ids", reference.tokens(), batched.tokens());
        assertEquals(what + " text", reference.text(), batched.text());
    }

    @Test
    public void parallelHostSamplingMatchesEachSequenceRunAlone() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        List<GenerationRequest> requests = new ArrayList<>(independentRequests());
        requests.add(GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.builder().topK(0).topP(0.9f).temperature(0.8f).seed(15L).build())
                .constraint(ConstraintSpec.grammar("root ::= [a-z ]+"))
                .build());
        requests.add(GenerationRequest.builder().maxNewTokens(N_NEW).logprobs(3)
                .sampler(SamplerConfig.builder().topK(0).topP(0.9f).temperature(0.8f).seed(16L).build())
                .build());
        List<int[]> prompts = new ArrayList<>();
        List<GenerationResult> refs = new ArrayList<>();
        for (int i = 0; i < requests.size(); i++) {
            prompts.add(tokens(i + 1, 5 + 3 * i));
            refs.add(reference(model, prompts.get(i), requests.get(i)));
        }
        EngineConfig cfg = config(true);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            List<BatchSequence> running = new ArrayList<>();
            for (int i = 0; i < requests.size(); i++) {
                running.add(engine.submit(prompts.get(i), requests.get(i),
                        (com.eonmux.neuralfree.gguf.TokenCallback) null));
            }
            engine.drain();
            for (int i = 0; i < running.size(); i++) {
                assertMatches("sequence " + i, refs.get(i), running.get(i));
            }
            assertTrue("the independent chains sampled on worker threads",
                    engine.stats().parallelSampledRows() > 0);
            assertTrue("only the independent chains sampled in parallel",
                    engine.stats().parallelSampledRows() <= (long) independentRequests().size() * N_NEW);
        }
    }

    private static SamplerConfig seeded(long seed) {
        return SamplerConfig.builder().topK(0).topP(0.9f).temperature(0.8f).seed(seed).build();
    }

    /**
     * A logprobs request (serial) fails in its callback on its second token while the later request B, sampled in
     * parallel, was cancelled for the same step: the serial route retires A, B and C with A's error, so B finishes
     * once, as FAILED, and nothing counts it cancelled.
     */
    @Test
    public void aCancelledParallelPieceAfterAFailingPieceRetiresOnce() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(true);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            AtomicInteger aTokens = new AtomicInteger();
            AtomicBoolean cancelB = new AtomicBoolean();
            GenerationRequest a = GenerationRequest.builder().maxNewTokens(N_NEW).logprobs(2).sampler(seeded(1L))
                    .build();
            GenerationRequest b = GenerationRequest.builder().maxNewTokens(N_NEW).cancellation(cancelB::get)
                    .sampler(seeded(2L)).build();
            GenerationRequest c = GenerationRequest.builder().maxNewTokens(N_NEW).sampler(seeded(3L)).build();
            BatchSequence sa = engine.submit(tokens(1, 5), a, (TokenCallback) (id, t) -> {
                if (aTokens.incrementAndGet() == 2) {
                    throw new IllegalStateException("callback A fails on its second token");
                }
            });
            BatchSequence sb = engine.submit(tokens(2, 5), b, (TokenCallback) null);
            BatchSequence sc = engine.submit(tokens(3, 5), c, (TokenCallback) null);
            AtomicInteger aFinish = new AtomicInteger();
            AtomicInteger bFinish = new AtomicInteger();
            AtomicInteger cFinish = new AtomicInteger();
            sa.onFinish(aFinish::incrementAndGet);
            sb.onFinish(bFinish::incrementAndGet);
            sc.onFinish(cFinish::incrementAndGet);
            engine.pump();
            assertEquals("the prompts completed with one token each", 1, aTokens.get());
            long parallelBefore = engine.stats().parallelSampledRows();
            assertTrue("B and C sampled their first tokens in parallel", parallelBefore >= 2);
            cancelB.set(true);
            engine.pump();
            for (BatchSequence s : List.of(sa, sb, sc)) {
                assertEquals(BatchSequence.State.FAILED, s.state());
                assertEquals("callback A fails on its second token", s.error().getMessage());
            }
            assertEquals("A finishes once", 1, aFinish.get());
            assertEquals("B finishes once", 1, bFinish.get());
            assertEquals("C finishes once", 1, cFinish.get());
            assertEquals(3L, engine.stats().failed());
            assertEquals(0L, engine.stats().finished(StopReason.CANCELLED));
            assertEquals("no token of the failed step was offered", parallelBefore,
                    engine.stats().parallelSampledRows());
        }
    }

    /**
     * Inside a parallel step, B's callback fails on its second token: A, earlier in the batch, keeps sampling and
     * matches its run alone, while B and the later C retire with B's error, once each. A cancelled D, sampled in
     * parallel beside them on an earlier step, finishes cancelled once.
     */
    @Test
    public void aFailureInsideAParallelStepRetiresFromThatPieceOn() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        GenerationRequest a = request(seeded(21L));
        GenerationResult refA = reference(model, tokens(1, 5), a);
        EngineConfig cfg = config(true);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            AtomicBoolean cancelD = new AtomicBoolean();
            AtomicInteger bTokens = new AtomicInteger();
            BatchSequence sd = engine.submit(tokens(4, 5), GenerationRequest.builder().maxNewTokens(N_NEW)
                    .cancellation(cancelD::get).sampler(seeded(24L)).build(), (TokenCallback) null);
            BatchSequence sa = engine.submit(tokens(1, 5), a, (TokenCallback) null);
            BatchSequence sb = engine.submit(tokens(2, 5), request(seeded(22L)), (TokenCallback) (id, t) -> {
                if (bTokens.incrementAndGet() == 2) {
                    throw new IllegalStateException("callback B fails on its second token");
                }
            });
            BatchSequence sc = engine.submit(tokens(3, 5), request(seeded(23L)), (TokenCallback) null);
            AtomicInteger[] finishes = new AtomicInteger[4];
            BatchSequence[] all = {sa, sb, sc, sd};
            for (int i = 0; i < all.length; i++) {
                AtomicInteger n = new AtomicInteger();
                finishes[i] = n;
                all[i].onFinish(n::incrementAndGet);
            }
            engine.pump();
            cancelD.set(true);
            engine.drain();
            assertMatches("A", refA, sa);
            for (BatchSequence s : List.of(sb, sc)) {
                assertEquals(BatchSequence.State.FAILED, s.state());
                assertEquals("callback B fails on its second token", s.error().getMessage());
            }
            assertEquals(BatchSequence.State.CANCELLED, sd.state());
            for (int i = 0; i < all.length; i++) {
                assertEquals("request " + i + " finishes once", 1, finishes[i].get());
            }
            assertEquals(2L, engine.stats().failed());
            assertEquals(1L, engine.stats().finished(StopReason.CANCELLED));
            assertTrue(engine.stats().parallelSampledRows() > 0);
        }
    }

    /**
     * A serial decode A fails in its callback in the step where B and C complete prompts that reuse a cached prefix:
     * the serial route retires B and C before their prefill is recorded, and the parallel route leaves the counters
     * the same, so the step counts none of their reused tokens as cached prompt work. Without the failure the same
     * step samples B and C in parallel and counts their reuse.
     */
    @Test
    public void aFailingEarlierPieceLeavesTheParallelPromptsUncounted() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        long[] parallelFail = cachedStep(model, true, true);
        long[] serialFail = cachedStep(model, false, true);
        long[] parallelPass = cachedStep(model, true, false);
        assertEquals("B and C sample in parallel beside the passing A", 2L, parallelPass[3]);
        assertTrue("B and C reuse the cached prefix", parallelPass[4] > 0);
        assertEquals("the passing step counts B's and C's reuse", parallelPass[4], parallelPass[1]);
        assertEquals("the failing step retires A, B and C", 3L, serialFail[0]);
        assertEquals("the serial route counts no reuse of the retired prompts", 0L, serialFail[1]);
        assertArrayEquals("the parallel route leaves the counters as the serial route", serialFail, parallelFail);
    }

    /**
     * One engine: a cached prompt, then a decoding A (logprobs, so it samples serially; its callback throws on the
     * second token when {@code fail}) beside B and C, whose prompts extend the cached one and complete in A's second
     * step; B and C sample in parallel unless {@code parallel} is false (logprobs keep them serial).
     *
     * @return over A's second step: {failed, promptTokensCached delta, generatedTokens delta, parallelSampledRows
     *         delta, B's plus C's reused prefix}
     */
    private static long[] cachedStep(Path model, boolean parallel, boolean fail) throws IOException {
        MemoryConfig mc = MemoryConfig.builder().nCtx(256).ubatch(16).parallel(3).typeK(KvDType.F32)
                .typeV(KvDType.F32).prefixReuse(true).kvPaged(true).build();
        EngineConfig cfg = EngineConfig.builder().memory(mc).build();
        int[] cached = tokens(9, 12);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            engine.submit(cached, request(seeded(30L)), (TokenCallback) null);
            engine.drain();
            AtomicInteger aTokens = new AtomicInteger();
            GenerationRequest a = GenerationRequest.builder().maxNewTokens(N_NEW).logprobs(2).sampler(seeded(31L))
                    .build();
            engine.submit(tokens(1, 5), a, (TokenCallback) (id, t) -> {
                if (aTokens.incrementAndGet() == 2 && fail) {
                    throw new IllegalStateException("callback A fails on its second token");
                }
            });
            engine.pump();
            assertEquals("A prefilled alone and sampled one token", 1, aTokens.get());
            List<BatchSequence> later = new ArrayList<>();
            for (int i = 0; i < 2; i++) {
                int[] prompt = java.util.Arrays.copyOf(cached, cached.length + 1);
                prompt[cached.length] = 4 + i;
                GenerationRequest.Builder r = GenerationRequest.builder().maxNewTokens(N_NEW)
                        .sampler(seeded(32L + i));
                later.add(engine.submit(prompt, (parallel ? r : r.logprobs(2)).build(), (TokenCallback) null));
            }
            BatchStats before = engine.stats();
            engine.pump();
            BatchStats after = engine.stats();
            assertEquals("A samples its second token in the step", 2, aTokens.get());
            for (BatchSequence s : later) {
                assertEquals(fail ? BatchSequence.State.FAILED : BatchSequence.State.RUNNING, s.state());
            }
            return new long[]{after.failed() - before.failed(),
                    after.promptTokensCached() - before.promptTokensCached(),
                    after.generatedTokens() - before.generatedTokens(),
                    after.parallelSampledRows() - before.parallelSampledRows(),
                    later.get(0).reusedPrefix() + later.get(1).reusedPrefix()};
        }
    }

    @Test
    public void onlyChainsHoldingTheirOwnStateSampleOffTheEngineThread() {
        for (GenerationRequest r : independentRequests()) {
            // the vocabulary-free chain: DRY without restart sequences needs no vocabulary to build
            SamplerConfig config = r.sampler().dryMultiplier() == 0f ? r.sampler()
                    : r.sampler().toBuilder().drySequenceBreakers(List.of()).build();
            assertTrue(config.toString(), Sampler.create(config).samplesIndependently());
        }
        assertTrue("greedy", Sampler.create(SamplerConfig.greedyDecoding()).samplesIndependently());
        assertTrue("a logit bias is fixed", Sampler.create(SamplerConfig.greedyDecoding(), new int[]{3},
                java.util.Map.of(5, 1.5f)).samplesIndependently());
        Sampler masked = Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                List.of(new com.eonmux.neuralfree.gguf.engine.sampling.AllowedTokensStage(new int[]{4, 5, 6})));
        assertFalse("a mask stage", masked.samplesIndependently());
        Sampler custom = Sampler.create(SamplerConfig.greedyDecoding(), new int[0], List.of(new BuiltInLookalike()));
        assertFalse("a custom stage reporting a built-in kind", custom.samplesIndependently());
    }

    /**
     * A chain holding a stage instance its caller handed in samples on the engine's thread even when the instance is
     * one of the built-in final classes: a logits processor factory or an extra-stage list may return one instance
     * to several requests, whose chains would then share its scratch on worker threads.
     */
    @Test
    public void aCallerSuppliedBuiltInStageKeepsItsChainOnTheEngineThread() {
        SamplerConfig config = SamplerConfig.builder().topK(0).topP(1f).temperature(0.8f).seed(5L).build();
        List<com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage> none = List.of();
        assertTrue("no caller stages", Sampler.create(config, new int[0], none, none).samplesIndependently());
        com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage shared =
                new com.eonmux.neuralfree.gguf.engine.sampling.TopPStage(0.9f, 1);
        assertFalse("a built-in stage as a logits processor",
                Sampler.create(config, new int[0], none, List.of(shared)).samplesIndependently());
        assertFalse("a built-in stage as an extra stage",
                Sampler.create(config, new int[0], List.of(shared)).samplesIndependently());
    }

    /** A Java-API stage that reports a built-in kind while it may hold state shared across requests. */
    private static final class BuiltInLookalike extends com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage {
        @Override
        public com.eonmux.neuralfree.gguf.engine.sampling.StageKind kind() {
            return com.eonmux.neuralfree.gguf.engine.sampling.StageKind.TOP_P;
        }

        @Override
        public void apply(Sampler.Candidates candidates) {
        }
    }
}
