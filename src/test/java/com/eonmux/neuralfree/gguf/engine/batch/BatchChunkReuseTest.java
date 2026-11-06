package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine's KV chunk reuse ({@link ChunkReuse}, llama.cpp server's {@code n_cache_reuse}): past the
 * common prefix, the cached chain's runs of at least {@code n_cache_reuse} rows re-append at the prompt's own
 * rows instead of prefilling, a shifted chunk's keys re-rotating by the shift between the cached rows and the
 * prompt ones. The re-appended rows carry their donor prefix in their deeper-layer state (the accuracy trade
 * llama.cpp's opt-in flag accepts), so the tests assert the mechanics: which rows the chunks skip, the counts
 * the engine reports, and that a run without a qualifying chunk keeps the plain prefix cache — whose reuse is
 * exact, so those generations still match the full prefill's.
 */
public class BatchChunkReuseTest {

    private static final int PAGE = 8;
    private static final int N_CTX = 64;
    private static final int BUDGET = 6;

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(PAGE).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).kvPageSize(PAGE).prefixReuse(true).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static EngineConfig unpagedConfig() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(PAGE).parallel(1).kvPaged(false)
                .typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(true).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static EngineConfig uncacheableConfig() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(PAGE).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).kvPageSize(PAGE).prefixReuse(false).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static GenerationRequest request(int reuse) {
        return GenerationRequest.builder()
                .maxNewTokens(BUDGET)
                .sampler(SamplerConfig.greedyDecoding())
                .nCacheReuse(reuse)
                .build();
    }

    /** Caches {@code prompt} whole: one decode, so its rows commit and store as the block chain. */
    private static void cache(BatchEngine engine, int[] prompt) {
        engine.submit(prompt, GenerationRequest.builder()
                        .maxNewTokens(1).sampler(SamplerConfig.greedyDecoding()).build(),
                (com.eonmux.neuralfree.gguf.TokenCallback) null);
        engine.drain();
    }

    /** Eight deterministic tokens named by one letter, blocks distinct from one another. */
    private static int[] block(char name) {
        int[] t = new int[PAGE];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (name * 31 + i * 13) % 250;
        }
        return t;
    }

    private static int[] join(int[]... parts) {
        int n = 0;
        for (int[] part : parts) {
            n += part.length;
        }
        int[] out = new int[n];
        int at = 0;
        for (int[] part : parts) {
            System.arraycopy(part, 0, out, at, part.length);
            at += part.length;
        }
        return out;
    }

    private static int[] first(int[] tokens, int n) {
        int[] out = new int[n];
        System.arraycopy(tokens, 0, out, 0, n);
        return out;
    }

    /**
     * Runs {@code prompt} on a fresh engine with the reuse off: the reference generation the plain prefix
     * reuse must match, the same rows a full prefill computes.
     */
    private static GenerationResult fullPrefill(Path model, int[] prompt) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            BatchSequence seq = engine.submit(prompt, request(0), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue(String.valueOf(seq.error()), seq.state() == BatchSequence.State.DONE);
            return seq.result();
        }
    }

    /**
     * An aligned divergence: the prompt repeats the cached p and c blocks and then diverges, so the whole match
     * adopts as the prefix and the cached chain past the boundary (the d block) never matches the prompt — no
     * chunk forms and the plain prefix reuse stands, exact against the full prefill.
     */
    @Test
    public void anAlignedMatchAdoptsThePrefixAndNeverChunks() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] cached = join(block('p'), block('c'), block('d'));
        int[] prompt = join(block('p'), block('c'), block('e'));
        int[] reference = fullPrefill(model, prompt).tokens();
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            cache(engine, cached);
            BatchSequence reused = engine.submit(prompt, request(4),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue(String.valueOf(reused.error()), reused.state() == BatchSequence.State.DONE);
            assertEquals("the p and c blocks adopt whole as the prefix", 16, reused.reusedPrefix());
            assertArrayEquals("the plain prefix reuse decodes like the full prefill",
                    reference, reused.result().tokens());
            assertEquals("no chunk forms past the aligned boundary", 0, engine.stats().chunkRowsReused());
        }
    }

    /**
     * A shifted chunk: the prompt repeats the cached p block and then takes the cached d block, which the chain
     * holds one block deeper — the chunk's eight rows move from the cached rows 16-23 to the prompt rows 8-15,
     * a shift of -8 whose keys re-rotate, and the prompt prefills only its tail. The chunk skips its rows and
     * the generation completes greedily.
     */
    @Test
    public void aShiftedChunkSkipsItsRowsAndDecodes() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] cached = join(block('p'), block('c'), block('d'));
        int[] prompt = join(block('p'), block('d'), block('e'));
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            cache(engine, cached);
            BatchSequence reused = engine.submit(prompt, request(8),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue(String.valueOf(reused.error()), reused.state() == BatchSequence.State.DONE);
            assertEquals("the prefix and the shifted chunk skip the prompt's first 16 rows",
                    16, reused.reusedPrefix());
            assertEquals("the chunk's rows re-appended from the cached pages", 8,
                    engine.stats().chunkRowsReused());
            assertEquals("the chunk leaves the tail to decode", BUDGET, reused.result().tokens().length);
        }
    }

    /**
     * A chunk that starts inside the boundary block: the prompt repeats the cached p block and the first five c
     * rows, then takes the cached d block's first seven rows — the scan walks the boundary block's diverged rows
     * alone (the cache cursor advances without the prompt one) and the d block's first six rows re-append at the
     * prompt rows 13-18, a shift of -3, never covering the prompt's last row. Only the rows past the chunk prefill.
     */
    @Test
    public void aChunkAfterTheBoundaryBlockWalksItsRowsAlone() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] c = block('c');
        int[] cached = join(block('p'), c, block('d'));
        int[] prompt = join(block('p'), first(c, 5), first(block('d'), 7));
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            cache(engine, cached);
            BatchSequence reused = engine.submit(prompt, request(4),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue(String.valueOf(reused.error()), reused.state() == BatchSequence.State.DONE);
            assertEquals("the prefix, the boundary's five rows and the chunk skip 19 prompt rows",
                    19, reused.reusedPrefix());
            assertEquals("the d block's rows before the prompt's last row re-append", 6,
                    engine.stats().chunkRowsReused());
            assertEquals("the chunk leaves the last row to decode", BUDGET, reused.result().tokens().length);
        }
    }

    /** A run shorter than {@code n_cache_reuse} never re-appends: only the common prefix is reused. */
    @Test
    public void aRunShorterThanTheReuseLengthPrefills() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] cached = join(block('p'), block('c'), block('d'));
        int[] prompt = join(block('p'), block('d'), block('e'));
        int[] reference = fullPrefill(model, prompt).tokens();
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            cache(engine, cached);
            BatchSequence shortOfReuse = engine.submit(prompt, request(9),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue(String.valueOf(shortOfReuse.error()), shortOfReuse.state() == BatchSequence.State.DONE);
            assertEquals("the eight-row run falls short of the reuse length", 8, shortOfReuse.reusedPrefix());
            assertArrayEquals("the plain prefix reuse decodes like the full prefill",
                    reference, shortOfReuse.result().tokens());
            assertEquals(0, engine.stats().chunkRowsReused());
        }
    }

    /** The reuse off ({@code n_cache_reuse 0}) leaves the plain prefix cache behavior standing. */
    @Test
    public void zeroReuseKeepsThePlainPrefixCache() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] cached = join(block('p'), block('c'), block('d'));
        int[] prompt = join(block('p'), block('c'), block('e'));
        int[] reference = fullPrefill(model, prompt).tokens();
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            cache(engine, cached);
            BatchSequence plain = engine.submit(prompt, request(0),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue(String.valueOf(plain.error()), plain.state() == BatchSequence.State.DONE);
            assertEquals("the p and c blocks adopt whole", 16, plain.reusedPrefix());
            assertArrayEquals(reference, plain.result().tokens());
            assertEquals(0, engine.stats().chunkRowsReused());
        }
    }

    /** An engine without the prefix cache refuses a reuse request by name. */
    @Test
    public void anEngineWithoutThePrefixCacheRefusesTheReuseByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), uncacheableConfig());
             BatchEngine engine = BatchEngine.forModel(m, uncacheableConfig())) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> engine.submit(join(block('p'), block('c')), request(4),
                            (com.eonmux.neuralfree.gguf.TokenCallback) null));
            assertTrue(e.getMessage(), e.getMessage().contains("n_cache_reuse"));
        }
    }

    /** An engine without paged memory cannot batch at all, so a reuse request cannot even queue. */
    @Test
    public void anEngineWithoutPagedSequencesBuildsNowhere() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        GGUFModel m = new GGUFModel(model.toString(), unpagedConfig());
        try {
            assertThrows(UnsupportedOperationException.class,
                    () -> BatchEngine.forModel(m, unpagedConfig()));
        } finally {
            m.close();
        }
    }

    /** A negative {@code n_cache_reuse} never builds (the request field rejects it). */
    @Test
    public void aNegativeReuseCountRefuses() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> GenerationRequest.builder().maxNewTokens(1)
                        .sampler(SamplerConfig.greedyDecoding()).nCacheReuse(-1));
        assertTrue(e.getMessage(), e.getMessage().contains("n_cache_reuse"));
    }
}
