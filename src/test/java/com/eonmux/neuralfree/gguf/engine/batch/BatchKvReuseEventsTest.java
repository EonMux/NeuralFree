package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvCacheEvent;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventBatch;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublishers;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessorArgs;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The prefix cache's reuse reports (vLLM's {@code emit_cached_block_events}): a request whose {@code vllm_xargs}
 * set {@code kv_cache_report_mode} to {@code full} publishes a {@code BlockStored} for every whole cached block it
 * reuses, carrying the hash, tokens, extra keys and spec their store carried; the default {@code incremental} mode
 * reports only the blocks an engine step stores, so a repeat of a cached prompt reports nothing.
 */
public class BatchKvReuseEventsTest {

    private static final int PAGE = 4;

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).kvPaged(true)
                .kvPageSize(PAGE).parallel(1).typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(true).build())
                .build();
    }

    private static GenerationRequest request(int n) {
        return GenerationRequest.builder().maxNewTokens(n).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** A request that asks for {@code reportMode} through the {@code vllm_xargs} object. */
    private static GenerationRequest request(int n, Object reportMode) {
        Map<String, Object> args = Collections.singletonMap(Submission.REPORT_MODE, reportMode);
        return GenerationRequest.builder().maxNewTokens(n).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding())
                .logitsProcessorArgs(LogitsProcessorArgs.of(args)).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 11 + i * 7) % 250;
        }
        return t;
    }

    private static void run(BatchEngine engine, int[] prompt, GenerationRequest request) {
        BatchSequence seq = engine.submit(prompt, request, (TokenCallback) null);
        for (int step = 0; step < 2_000 && !seq.finished(); step++) {
            engine.pump();
        }
        assertEquals("the request completed: " + seq.error(), BatchSequence.State.DONE, seq.state());
    }

    /** The {@code BlockStored} events of every batch so far, in order. */
    private static List<KvCacheEvent.BlockStored> stores(List<KvEventBatch> batches) {
        List<KvCacheEvent.BlockStored> out = new ArrayList<>();
        for (KvEventBatch batch : batches) {
            for (KvCacheEvent e : batch.events()) {
                if (e instanceof KvCacheEvent.BlockStored) {
                    out.add((KvCacheEvent.BlockStored) e);
                }
            }
        }
        return out;
    }

    /** The chain hashes the store events so far carry, in order. */
    private static List<Long> chain(List<KvEventBatch> batches) {
        List<Long> out = new ArrayList<>();
        for (KvCacheEvent.BlockStored store : stores(batches)) {
            for (long hash : store.blockHashInts()) {
                out.add(hash);
            }
        }
        return out;
    }

    @Test
    public void aFullModeRequestReportsEveryCachedBlockItReuses() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        List<KvEventBatch> batches = new ArrayList<>();
        int[] x = tokens(21, 9);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg, KvEventPublishers.listener(batches::add))) {
            run(engine, x, request(3));
            List<Long> stored = chain(batches);
            assertTrue("the first run stores the prompt's blocks", stored.size() >= 2);
            batches.clear();

            run(engine, x, request(3, Submission.REPORT_MODE_FULL));
            List<KvCacheEvent.BlockStored> reused = stores(batches);
            assertEquals("the repeat stores nothing new and reports one run of reused blocks: " + batches, 1,
                    reused.size());
            KvCacheEvent.BlockStored report = reused.get(0);
            assertNull("a cache hit starts at its chain's first block", report.parentBlockHashInt());
            int whole = (x.length - 1) / PAGE * PAGE;
            assertEquals("the report covers the whole blocks the request reuses", whole, report.tokenIds().length);
            assertEquals(whole / PAGE, report.blockHashInts().length);
            assertEquals(PAGE, report.blockSize());
            assertEquals("the blocks have the medium the cached pool names",
                    stores(batches).get(0).medium(), report.medium());
            for (int k = 0; k < report.blockHashInts().length; k++) {
                assertEquals("block " + k + " carries its stored hash", (long) stored.get(k),
                        report.blockHashInts()[k]);
            }
            for (int i = 0; i < report.tokenIds().length; i++) {
                assertEquals("the report carries the prompt's token " + i, x[i], report.tokenIds()[i]);
            }
        }
    }

    /**
     * A prompt shorter than one block reuses only rows of the cached partial block, not a whole block: as in vLLM's
     * {@code emit_cached_block_events}, the report covers whole blocks only, so nothing is published.
     */
    @Test
    public void aFullModeRequestWithNoWholeCachedBlockReportsNothing() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        List<KvEventBatch> batches = new ArrayList<>();
        int[] x = tokens(23, PAGE - 1);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg, KvEventPublishers.listener(batches::add))) {
            run(engine, x, request(3));
            assertFalse("the first run stores the prompt's blocks", stores(batches).isEmpty());
            batches.clear();

            run(engine, x, request(3, Submission.REPORT_MODE_FULL));
            assertTrue("a hit inside one block reports no whole cached block: " + batches, stores(batches).isEmpty());
        }
    }

    @Test
    public void theDefaultReportModeReportsOnlyStores() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        List<KvEventBatch> batches = new ArrayList<>();
        int[] x = tokens(22, 9);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg, KvEventPublishers.listener(batches::add))) {
            run(engine, x, request(3));

            batches.clear();
            run(engine, x, request(3));
            assertTrue("a repeat of a cached prompt stores and reports nothing", stores(batches).isEmpty());

            batches.clear();
            run(engine, x, request(3, "incremental"));
            assertTrue("the incremental mode reports no reused block", stores(batches).isEmpty());

            batches.clear();
            run(engine, x, request(3, Boolean.TRUE));
            assertTrue("only the string full asks for reuse reports, as vLLM compares it", stores(batches).isEmpty());

            batches.clear();
            run(engine, x, request(3, Submission.REPORT_MODE_FULL));
            assertFalse("the full mode still reports the reused blocks", stores(batches).isEmpty());
        }
    }
}
