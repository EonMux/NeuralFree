package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** The batch engine's counters and gauges ({@link BatchStats}) over finished, queued and cancelled requests. */
public class BatchStatsTest {

    private static final int N_TOKENS = 12;

    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    @Test
    public void theCountersFollowEveryPrefillDecodeAndFinish() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(6).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            assertEquals(0L, engine.stats().promptTokens());
            BatchSequence a = engine.submit(tokens(1), request, (TokenCallback) null);
            BatchSequence b = engine.submit(tokens(2), request, (TokenCallback) null);
            engine.submit(tokens(3), request.toBuilder().cancellation(() -> true).build(), (TokenCallback) null);
            BatchStats queued = engine.stats();
            assertEquals(3, queued.waiting());
            assertEquals(0, queued.running());
            engine.drain();
            BatchStats s = engine.stats();
            assertEquals("every prompt token prefilled or reused", 2L * N_TOKENS,
                    s.promptTokens() + s.promptTokensCached());
            assertEquals(a.result().tokens().length + b.result().tokens().length, s.generatedTokens());
            assertEquals(2L, s.finished(StopReason.MAX_TOKENS));
            assertEquals(1L, s.finished(StopReason.CANCELLED));
            assertEquals(0L, s.failed());
            assertEquals("a host backend reads every sampled row back", 0L, s.deviceSampledRows());
            assertTrue(s.decodeSteps() > 0);
            assertTrue(s.rowsPerDecode() >= 1.0 && s.rowsPerDecode() <= 2.0);
            assertTrue(s.promptSeconds() > 0 && s.decodeSeconds() > 0);
            assertEquals(N_TOKENS + 6, s.maxSequenceLength());
            assertEquals(0, s.running());
            assertEquals(0, s.waiting());
            assertTrue("the cached prefixes still hold rows", s.kvUsage() > 0 && s.kvUsage() <= 1);
            assertEquals(0L, s.preemptions());
        }
    }

    @Test
    public void plusCombinesTwoSnapshots() {
        BatchStats.Builder a = new BatchStats.Builder();
        a.promptTokens = 5;
        a.promptNanos = 1_000;
        a.generatedTokens = 3;
        a.deviceSampledRows = 5;
        a.decodeSteps = 2;
        a.decodeRows = 3;
        a.finished[StopReason.MAX_TOKENS.ordinal()] = 1;
        a.failed = 1;
        a.specRounds = 4;
        a.specDrafted = 9;
        a.specAccepted = 6;
        a.maxSequenceLength = 40;
        a.running = 1;
        a.queued = 2;
        a.usedCells = 10;
        a.totalCells = 100;
        a.encoderCacheRows = 16;
        a.encoderCacheHits = 3;
        a.encoderCacheEvictions = 1;
        BatchStats.Builder b = new BatchStats.Builder();
        b.promptTokens = 7;
        b.promptNanos = 500;
        b.generatedTokens = 4;
        b.deviceSampledRows = 2;
        b.decodeSteps = 2;
        b.decodeRows = 2;
        b.finished[StopReason.MAX_TOKENS.ordinal()] = 2;
        b.specRounds = 2;
        b.specDrafted = 5;
        b.specAccepted = 3;
        b.maxSequenceLength = 60;
        b.usedCells = 30;
        b.totalCells = 100;
        b.encoderCacheRows = 8;
        b.encoderCacheHits = 2;
        b.encoderCacheEvictions = 4;
        BatchStats sum = a.build().plus(b.build());
        assertEquals(12L, sum.promptTokens());
        assertEquals(7L, sum.generatedTokens());
        assertEquals(7L, sum.deviceSampledRows());
        assertEquals(4L, sum.decodeSteps());
        assertEquals(1.25, sum.rowsPerDecode(), 1e-9);
        assertEquals(1500.0 / 1e9, sum.promptSeconds(), 1e-12);
        assertEquals(3L, sum.finished(StopReason.MAX_TOKENS));
        assertEquals(1L, sum.failed());
        assertEquals(6L, sum.specRounds());
        assertEquals(14L, sum.specDrafted());
        assertEquals(9L, sum.specAccepted());
        assertEquals(60, sum.maxSequenceLength());
        assertEquals(1, sum.running());
        assertEquals(2, sum.waiting());
        assertEquals(0.2, sum.kvUsage(), 1e-9);
        assertEquals("the replicas' encoder caches add", 24, sum.encoderCacheRows());
        assertEquals(5L, sum.encoderCacheHits());
        assertEquals(5L, sum.encoderCacheEvictions());
        assertEquals(0L, BatchStats.EMPTY.plus(BatchStats.EMPTY).promptTokens());
        assertEquals(0, BatchStats.EMPTY.plus(BatchStats.EMPTY).waiting());
    }
}
