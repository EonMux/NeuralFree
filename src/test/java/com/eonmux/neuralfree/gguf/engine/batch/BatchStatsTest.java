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
}
