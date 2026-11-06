package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EmbedsPrompt;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.gguf.testutil.SyntheticRoutedModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.function.UnaryOperator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Prefix reuse over memories whose cached KV blocks alone cannot restore a prefix — recurrent state (mamba2),
 * hybrid recurrent and attention layers (qwen3next), sliding-window groups (gemma2), and both (lfm3) — through
 * context checkpoints (llama.cpp's {@code --ctx-checkpoints}, vLLM's Mamba "align" cache): a conversation's next
 * turn resumes at the retirement checkpoint, a repeated prompt at the checkpoint before its last page, a prompt
 * that diverges midway at the last spaced checkpoint before the divergence, and a preempted sequence recomputes
 * from its own. Every result equals the run of the same prompt on a fresh engine (tokens and their log
 * probabilities), and a freed slot's recurrent state never leaks into the next sequence it serves.
 */
public class BatchContextCheckpointTest {

    private static final int PAGE = 4;
    private static final int N_NEW = 6;
    private static final float LOGPROB_TOLERANCE = 1e-3f;

    private static Path mamba2() throws IOException {
        return named("mamba2", SyntheticModels.mamba2());
    }

    private static Path qwen3next() throws IOException {
        return named("qwen3next", SyntheticModels.qwen3next());
    }

    /** lfm2 (short convolutions beside full attention). */
    private static Path lfm2() throws IOException {
        return named("lfm2", SyntheticModels.lfm2());
    }

    /** gemma2 with an 8-row window, so the prompts outgrow it and its group trims. */
    private static Path gemma2() throws IOException {
        return named("gemma2", SyntheticModels.gemma2(
                SyntheticModels.Config.gemma2().toBuilder().slidingWindow(8).build()));
    }

    /** lfm3 (sliding-window attention beside short convolutions) with an 8-row window. */
    private static Path lfm3() throws IOException {
        return named("lfm3", SyntheticModels.lfm3(
                SyntheticModels.Config.lfm3().toBuilder().slidingWindow(8).build(), true));
    }

    /** Writes the fixture to a temporary file whose name starts with {@code arch}, so failures name the model. */
    private static Path named(String arch, GgufTestWriter w) throws IOException {
        Path path = Files.createTempFile(arch + "-", ".gguf");
        path.toFile().deleteOnExit();
        w.writeTo(path);
        return path;
    }

    private static Path[] models() throws IOException {
        return new Path[] {mamba2(), qwen3next(), gemma2(), lfm3()};
    }

    private static MemoryConfig.Builder memory(boolean reuse) {
        return MemoryConfig.builder().nCtx(128).ubatch(8).kvPaged(true).kvPageSize(PAGE).parallel(1)
                .typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(reuse);
    }

    private static EngineConfig config(MemoryConfig.Builder memory) {
        return EngineConfig.builder().memory(memory.build()).build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).ignoreEos(true).logprobs(1)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** A deterministic prompt of {@code n} tokens; {@code seed} varies every token. */
    private static int[] prompt(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 37 + i * 11 + (i * i) % 13) % 300;
        }
        return t;
    }

    /** {@code head} followed by {@code tail}. */
    private static int[] concat(int[] head, int[] tail) {
        int[] out = Arrays.copyOf(head, head.length + tail.length);
        System.arraycopy(tail, 0, out, head.length, tail.length);
        return out;
    }

    private static BatchSequence submit(BatchEngine engine, int[] prompt) {
        BatchSequence seq = engine.submit(prompt, request(), (TokenCallback) null);
        engine.drain();
        assertEquals("the sequence finished: " + seq.error(), BatchSequence.State.DONE, seq.state());
        return seq;
    }

    /** The result of {@code prompt} alone on a fresh engine without prefix reuse. */
    private static GenerationResult reference(Path model, int[] prompt) throws IOException {
        EngineConfig cfg = config(memory(false));
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            return submit(engine, prompt).result();
        }
    }

    private static void assertSameResult(String what, GenerationResult expected, GenerationResult actual) {
        assertArrayEquals(what + ": tokens", expected.tokens(), actual.tokens());
        for (int i = 0; i < expected.logprobs().size(); i++) {
            TokenScore e = expected.logprobs().get(i)[0];
            TokenScore a = actual.logprobs().get(i)[0];
            assertEquals(what + ": token " + i, e.tokenId(), a.tokenId());
            assertEquals(what + ": logprob of token " + i, e.logprob(), a.logprob(), LOGPROB_TOLERANCE);
        }
    }

    /** Runs {@code body} on an engine over {@code model} with {@code memory}. */
    private static void withEngine(Path model, MemoryConfig.Builder memory, EngineBody body) throws IOException {
        EngineConfig cfg = config(memory);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            body.run(engine);
        }
    }

    private interface EngineBody {
        void run(BatchEngine engine) throws IOException;
    }

    /**
     * The paged reference the other tests compare with is itself the contiguous loop's result on the windowed
     * models: a prefill chunk whose rows read windows below its last row keeps them through the trunk (the trim
     * after a step follows the next query), so a prompt longer than the window computes the same tokens.
     */
    @Test
    public void aWindowedPrefillPastTheWindowMatchesTheContiguousLoop() throws IOException {
        for (Path model : new Path[] {gemma2(), lfm3()}) {
            int[] p = prompt(18, 30);
            EngineConfig contiguous = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(128).ubatch(8)
                    .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
            GenerationResult loop;
            try (GGUFModel m = new GGUFModel(model.toString(), contiguous)) {
                loop = GenerationLoop.forModel(m, contiguous).generate(p, request(), (TokenCallback) null);
            }
            assertSameResult(model.getFileName() + " paged against contiguous", loop, reference(model, p));
        }
    }

    @Test
    public void aFreedSlotStartsTheNextSequenceFromAFreshRecurrentState() throws IOException {
        for (Path model : new Path[] {mamba2(), qwen3next(), lfm2(), lfm3()}) {
            int[] second = prompt(2, 11);
            GenerationResult ref = reference(model, second);
            withEngine(model, memory(false), engine -> {
                submit(engine, prompt(1, 10));
                BatchSequence b = submit(engine, second);
                assertEquals("no prefix reuse", 0, b.reusedPrefix());
                assertSameResult(model.getFileName() + " after a slot reuse", ref, b.result());
            });
        }
    }

    @Test
    public void aNextTurnResumesAtTheRetirementCheckpoint() throws IOException {
        for (Path model : models()) {
            int[] first = prompt(3, 13);
            withEngine(model, memory(true), engine -> {
                BatchSequence a = submit(engine, first);
                int[] turn = concat(a.result().withPrompt(first), prompt(4, 5));
                GenerationResult ref = reference(model, turn);
                BatchSequence b = submit(engine, turn);
                assertEquals(model.getFileName() + ": the next turn resumes at the retired sequence's committed"
                        + " rows", first.length + N_NEW - 1, b.reusedPrefix());
                assertSameResult(model.getFileName() + " next turn", ref, b.result());
            });
        }
    }

    @Test
    public void aRepeatedPromptResumesAtTheCheckpointBeforeItsLastPage() throws IOException {
        for (Path model : models()) {
            int[] p = prompt(5, 18);
            GenerationResult ref = reference(model, p);
            withEngine(model, memory(true), engine -> {
                BatchSequence a = submit(engine, p);
                assertSameResult(model.getFileName() + " first run", ref, a.result());
                BatchSequence b = submit(engine, p);
                assertEquals(model.getFileName() + ": the repeat resumes at the last page boundary before its last"
                        + " token", (p.length - 1) / PAGE * PAGE, b.reusedPrefix());
                assertSameResult(model.getFileName() + " repeat", ref, b.result());
            });
        }
    }

    @Test
    public void aPromptThatDivergesMidwayResumesAtTheLastSpacedCheckpointBeforeIt() throws IOException {
        for (Path model : models()) {
            int[] shared = prompt(6, 26);
            int[] first = concat(shared, prompt(7, 14));
            int[] second = concat(shared, prompt(8, 9));
            GenerationResult ref = reference(model, second);
            // four slots size the sliding-window pool with room for the checkpoints' windows beside one sequence
            withEngine(model, memory(true).parallel(4).checkpointMinStep(8), engine -> {
                submit(engine, first);
                BatchSequence b = submit(engine, second);
                assertEquals(model.getFileName() + ": the spaced checkpoint at 24 rows serves the shared 26", 24,
                        b.reusedPrefix());
                assertSameResult(model.getFileName() + " diverging prompt", ref, b.result());
            });
        }
    }

    /**
     * A one-slot sliding-window pool holds the running sequence's window and little more: the checkpoints the
     * prefill takes on its way pin pages the sequence needs next, so they yield to it before each step instead of
     * starving the prefill, and every result stays the reference's.
     */
    @Test
    public void aSlidingWindowPoolWithoutRoomYieldsItsCheckpointsToTheRunningSequence() throws IOException {
        for (Path model : new Path[] {gemma2(), lfm3()}) {
            int[] shared = prompt(15, 26);
            int[] first = concat(shared, prompt(16, 14));
            int[] second = concat(shared, prompt(17, 9));
            GenerationResult refFirst = reference(model, first);
            GenerationResult refSecond = reference(model, second);
            withEngine(model, memory(true).checkpointMinStep(0), engine -> {
                assertSameResult(model.getFileName() + " first", refFirst, submit(engine, first).result());
                assertSameResult(model.getFileName() + " second", refSecond, submit(engine, second).result());
                assertSameResult(model.getFileName() + " repeat", refSecond, submit(engine, second).result());
            });
        }
    }

    /**
     * A reserve of sliding-window pool room ({@code neuralfree.swa.checkpoint.reserve}) keeps the checkpoints a
     * one-slot prefill takes on its way: the spaced checkpoint before a diverging prompt survives its own request
     * and serves the divergence (without the reserve the same checkpoint yields to the running sequence, as the
     * test above pins).
     */
    @Test
    public void anSwaPoolReserveKeepsTheMidPrefillCheckpointsAlive() throws IOException {
        for (Path model : new Path[] {gemma2(), lfm3()}) {
            int[] shared = prompt(6, 26);
            int[] first = concat(shared, prompt(7, 14));
            int[] second = concat(shared, prompt(8, 9));
            GenerationResult ref = reference(model, second);
            withEngine(model, memory(true).checkpointMinStep(8).swaCheckpointReserve(12), engine -> {
                submit(engine, first);
                BatchSequence b = submit(engine, second);
                assertEquals(model.getFileName() + ": the spaced checkpoint at 24 rows survives on the reserved"
                        + " pool", 24, b.reusedPrefix());
                assertSameResult(model.getFileName() + " diverging prompt", ref, b.result());
            });
        }
    }

    @Test
    public void theCheckpointLimitDropsTheLeastRecentlyUsedCheckpoint() throws IOException {
        Path model = qwen3next();
        int[] shared = prompt(9, 26);
        int[] first = concat(shared, prompt(10, 14));
        int[] second = concat(shared, prompt(11, 9));
        GenerationResult ref = reference(model, second);
        withEngine(model, memory(true).checkpointMinStep(8).ctxCheckpoints(1), engine -> {
            submit(engine, first);
            BatchSequence b = submit(engine, second);
            assertEquals("one checkpoint per slot keeps only the retirement checkpoint, past the divergence", 0,
                    b.reusedPrefix());
            assertSameResult("limited checkpoints", ref, b.result());
        });
    }

    @Test
    public void zeroCheckpointsTurnThePrefixCacheOffForAMemoryThatNeedsThem() throws IOException {
        for (Path model : new Path[] {mamba2(), gemma2()}) {
            int[] p = prompt(12, 18);
            GenerationResult ref = reference(model, p);
            withEngine(model, memory(true).ctxCheckpoints(0), engine -> {
                submit(engine, p);
                BatchSequence b = submit(engine, p);
                assertEquals(model.getFileName() + ": nothing reused", 0, b.reusedPrefix());
                assertSameResult(model.getFileName() + " without checkpoints", ref, b.result());
            });
        }
    }

    @Test
    public void aPreemptedSequenceRecomputesToItsReferenceResult() throws IOException {
        UnaryOperator<MemoryConfig.Builder> small = b -> b.nCtx(64).parallel(2).maxBatchedTokens(7);
        for (Path model : new Path[] {mamba2(), qwen3next(), lfm2(), gemma2(), lfm3()}) {
            int[] pa = prompt(13, 30);
            int[] pb = prompt(14, 30);
            GenerationResult refA = reference(model, pa);
            GenerationResult refB = reference(model, pb);
            withEngine(model, small.apply(memory(true)), engine -> {
                BatchSequence a = engine.submit(pa, request(), (TokenCallback) null);
                BatchSequence b = engine.submit(pb, request(), (TokenCallback) null);
                engine.drain();
                assertEquals(BatchSequence.State.DONE, a.state());
                assertEquals(BatchSequence.State.DONE, b.state());
                assertTrue(model.getFileName() + ": the shared pool preempted a sequence",
                        engine.stats().preemptions() >= 1);
                assertSameResult(model.getFileName() + " sequence a", refA, a.result());
                assertSameResult(model.getFileName() + " sequence b", refB, b.result());
            });
        }
    }

    /**
     * A retired conversation's free blocks are older than the preempted sequence's, so the eviction that follows
     * the preemption takes them first: the preempted sequence's checkpoint at its committed rows survives, and its
     * re-admission restores it whole and decodes on with nothing recomputed.
     */
    @Test
    public void aPreemptedSequenceWhoseCheckpointSurvivesResumesWithoutRecompute() throws IOException {
        UnaryOperator<MemoryConfig.Builder> small = b -> b.nCtx(64).parallel(2).maxBatchedTokens(7);
        for (Path model : new Path[] {mamba2(), qwen3next()}) {
            int[] pa = prompt(19, 30);
            int[] pb = prompt(20, 30);
            GenerationResult refA = reference(model, pa);
            GenerationResult refB = reference(model, pb);
            withEngine(model, small.apply(memory(true)), engine -> {
                submit(engine, prompt(21, 20));
                BatchSequence a = engine.submit(pa, request(), (TokenCallback) null);
                BatchSequence b = engine.submit(pb, request(), (TokenCallback) null);
                engine.drain();
                BatchStats stats = engine.stats();
                assertTrue(model.getFileName() + ": a sequence was preempted, " + stats, stats.preemptions() >= 1);
                assertEquals(model.getFileName() + ": the re-admission restored the checkpoint whole, " + stats, 0,
                        stats.recomputedTokens());
                assertSameResult(model.getFileName() + " sequence a", refA, a.result());
                assertSameResult(model.getFileName() + " sequence b", refB, b.result());
            });
        }
    }

    /**
     * A step over a recurrent memory starts only when the pages of all its ubatches fit (a recurrent state cannot
     * roll back a half-run step); a chunk wider than the ubatch over sliding-window and convolution layers still
     * fits the one-slot window pool, which its ubatches trim as they go, and computes the reference tokens.
     */
    @Test
    public void aWindowedRecurrentChunkWiderThanTheUbatchFitsItsWindowPool() throws IOException {
        Path model = lfm3();
        int[] p = prompt(22, 40);
        GenerationResult ref = reference(model, p);
        withEngine(model, memory(true).maxBatchedTokens(32), engine -> {
            assertSameResult("wide chunks", ref, submit(engine, p).result());
            assertSameResult("wide chunks, repeated", ref, submit(engine, p).result());
        });
    }

    @Test
    public void aSpeculativeDrafterOverRecurrentStateRefusesByName() throws IOException {
        Path model = mamba2();
        EngineConfig cfg = EngineConfig.builder().memory(memory(true).build())
                .draftType(EngineConfig.DraftType.NGRAM).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            try {
                BatchEngine.forModel(m, cfg).close();
                fail("a drafter over recurrent state must refuse");
            } catch (UnsupportedOperationException e) {
                assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_DRAFT_TYPE));
                assertTrue(e.getMessage(), e.getMessage().contains("recurrent state"));
            }
        }
    }
    /** The embedding table rows of {@code tokens}: prompt-embedding rows the model computes like the tokens. */
    private static float[][] tableRows(Path model, int[] tokens) throws IOException {
        try (GGUFFileParser parser = new GGUFFileParser(model.toString())) {
            GGUFTensorInfo table = parser.getTensor("token_embd.weight");
            float[] flat = parser.readTensorDataAsFloat(table);
            int width = (int) table.dims()[0];
            float[][] rows = new float[tokens.length][];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = Arrays.copyOfRange(flat, tokens[i] * width, (tokens[i] + 1) * width);
            }
            return rows;
        }
    }

    private static GenerationRequest salted(String salt) {
        return GenerationRequest.builder().maxNewTokens(N_NEW).ignoreEos(true).logprobs(1)
                .sampler(SamplerConfig.greedyDecoding()).cacheSalt(salt).build();
    }

    private static BatchSequence submit(BatchEngine engine, EmbedsPrompt prompt, GenerationRequest request) {
        BatchSequence seq = engine.submit(prompt, request, (TokenCallback) null);
        engine.drain();
        assertEquals("the sequence finished: " + seq.error(), BatchSequence.State.DONE, seq.state());
        return seq;
    }

    /** Runs a {@link MediaPrompt} on the engine and asserts it finished. */
    private static BatchSequence submit(BatchEngine engine, MediaPrompt prompt, GenerationRequest request) {
        BatchSequence seq = engine.submit(prompt, request, (TokenCallback) null);
        engine.drain();
        assertEquals("the sequence finished: " + seq.error(), BatchSequence.State.DONE, seq.state());
        return seq;
    }

    /** The result of the embeddings {@code prompt} alone on a fresh engine without prefix reuse. */
    private static GenerationResult reference(Path model, EmbedsPrompt prompt) throws IOException {
        EngineConfig cfg = config(memory(false));
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            return submit(engine, prompt, request()).result();
        }
    }

    /**
     * A checkpoint hangs on a cached block, so it restores only under that block's identity: a next turn under
     * another cache salt restores nothing, while the same turn under the same salt resumes at the retirement
     * checkpoint.
     */
    @Test
    public void aCheckpointRestoresOnlyUnderTheSameCacheSalt() throws IOException {
        for (Path model : models()) {
            int[] first = prompt(29, 13);
            withEngine(model, memory(true).parallel(4), engine -> {
                BatchSequence a = submit(engine, first);
                int[] turn = concat(a.result().withPrompt(first), prompt(30, 5));
                GenerationResult ref = reference(model, turn);
                BatchSequence b = engine.submit(turn, salted("tenant-b"), (TokenCallback) null);
                engine.drain();
                assertEquals("the sequence finished: " + b.error(), BatchSequence.State.DONE, b.state());
                assertEquals(model.getFileName() + ": another salt restores no checkpoint", 0, b.reusedPrefix());
                assertSameResult(model.getFileName() + " salted turn", ref, b.result());
                BatchSequence c = submit(engine, turn);
                assertEquals(model.getFileName() + ": the same salt resumes at the retirement checkpoint",
                        first.length + N_NEW - 1, c.reusedPrefix());
                assertSameResult(model.getFileName() + " next turn", ref, c.result());
            });
        }
    }

    /**
     * A checkpoint over media rows restores only for the same rows at the same places (their digests key the
     * cached blocks): a next turn over other prompt-embedding rows restores nothing, the same rows resume at the
     * retirement checkpoint. Over the rotary recurrent, hybrid and sliding-window models.
     */
    @Test
    public void aCheckpointRestoresOnlyForTheSameMediaRows() throws IOException {
        for (Path model : new Path[] {qwen3next(), gemma2(), lfm3()}) {
            int[] head = prompt(23, 3);
            int[] tail = prompt(24, 4);
            float[][] media = tableRows(model, prompt(25, 6));
            float[][] other = tableRows(model, prompt(26, 6));
            // four slots size the sliding-window pool with room for the checkpoints' windows beside one sequence
            withEngine(model, memory(true).parallel(4), engine -> {
                EmbedsPrompt first = EmbedsPrompt.of(new int[][] {head, tail}, new float[][][] {media});
                BatchSequence a = submit(engine, first, request());
                int[] turnTail = concat(a.result().withPrompt(tail), prompt(27, 3));
                EmbedsPrompt turn = EmbedsPrompt.of(new int[][] {head, turnTail}, new float[][][] {media});
                EmbedsPrompt otherMedia = EmbedsPrompt.of(new int[][] {head, turnTail}, new float[][][] {other});

                BatchSequence b = submit(engine, otherMedia, request());
                assertEquals(model.getFileName() + ": other media rows restore no checkpoint", 0, b.reusedPrefix());
                assertSameResult(model.getFileName() + " other media", reference(model, otherMedia), b.result());

                BatchSequence c = submit(engine, turn, request());
                assertEquals(model.getFileName() + ": the same media rows resume at the retirement checkpoint",
                        head.length + media.length + tail.length + N_NEW - 1, c.reusedPrefix());
                assertSameResult(model.getFileName() + " next turn", reference(model, turn), c.result());
            });
        }
    }

    /**
     * The message boundary a prompt names (llama.cpp's message spans): the prefill takes a checkpoint at the row
     * even when it sits mid-page (the checkpoint hangs on a stored partial block), and a prompt that branches the
     * conversation after it restores there, re-running only the rows after the boundary.
     */
    @Test
    public void aPromptBoundaryCheckpointServesAPromptThatBranchesPastIt() throws IOException {
        for (Path model : models()) {
            int[] head = prompt(31, 16);
            int[] first = concat(head, prompt(32, 6));
            int[] branch = concat(head, prompt(33, 4));
            GenerationResult ref = reference(model, branch);
            MediaPrompt named = MediaPrompt.builder().addText(first).checkpointRow(14).build();
            // the reserve keeps a one-slot sliding-window pool's checkpoints alive beside the running sequence
            withEngine(model, memory(true).swaCheckpointReserve(12), engine -> {
                submit(engine, named, request());
                BatchSequence b = submit(engine, branch);
                assertEquals(model.getFileName() + ": the boundary checkpoint at 14 rows serves the branch", 14,
                        b.reusedPrefix());
                assertSameResult(model.getFileName() + " branch", ref, b.result());
            });
        }
    }

    /**
     * A media prompt takes the checkpoints of its prefill's way too (llama.cpp skips checkpoints behind media
     * chunks; here the prompt-embedding chunk pieces end on the checkpoint rows the text chunks' prefill ends
     * on): a prompt that branches after an early checkpoint restores there and recomputes only its own tail.
     */
    @Test
    public void aMediaPromptTakesItsMidPrefillCheckpoints() throws IOException {
        for (Path model : new Path[] {qwen3next(), gemma2()}) {
            int[] open = prompt(34, 2);
            float[][] embeds = tableRows(model, prompt(35, 10));
            int[] mid = prompt(36, 2);
            int[] head = concat(open, prompt(37, 16));
            EmbedsPrompt first = EmbedsPrompt.of(new int[][] {open, mid, head}, new float[][][] {embeds, embeds});
            // the branch shares the first 26 rows (through the second text run) and diverges inside the head run
            EmbedsPrompt branchPrompt = EmbedsPrompt.of(new int[][] {open, mid, prompt(38, 4)},
                    new float[][][] {embeds, embeds});
            GenerationResult ref = reference(model, branchPrompt);
            // four slots size the sliding-window pool with room for the checkpoints' windows beside one sequence
            withEngine(model, memory(true).checkpointMinStep(8).parallel(4), engine -> {
                submit(engine, first, request());
                BatchSequence b = submit(engine, branchPrompt, request());
                assertEquals(model.getFileName() + ": the spaced checkpoint at 24 rows serves the branch", 24,
                        b.reusedPrefix());
                assertSameResult(model.getFileName() + " branch", ref, b.result());
            });
        }
    }

    /**
     * A DSV4 memory restores its cached prefixes at context checkpoints that carry the compressor streams and
     * rollback planes beside the sliding-window group's pages (llama.cpp's compressor state is per sequence and
     * accumulates, so only a checkpoint makes a prefix restorable): a repeated prompt resumes at the checkpoint
     * before its last page, a prompt that diverges midway at the last spaced checkpoint, both with the
     * reference's result.
     */
    @Test
    public void aDsv4MemoryRestoresItsPrefixesAtCheckpoints() throws IOException {
        Path model = named("deepseek4", SyntheticRoutedModels.deepseek4());
        int[] p = prompt(28, 18);
        GenerationResult ref = reference(model, p);
        withEngine(model, memory(true), engine -> {
            BatchSequence a = submit(engine, p);
            assertSameResult("first run", ref, a.result());
            BatchSequence again = submit(engine, p);
            assertEquals("the repeat resumes at the checkpoint before its last page",
                    (p.length - 1) / PAGE * PAGE, again.reusedPrefix());
            assertSameResult("repeat", ref, again.result());
        });
        int[] shared = prompt(39, 16);
        int[] first = concat(shared, prompt(40, 8));
        int[] diverging = concat(shared, prompt(41, 4));
        GenerationResult divergingRef = reference(model, diverging);
        withEngine(model, memory(true).checkpointMinStep(8), engine -> {
            submit(engine, first);
            BatchSequence b = submit(engine, diverging);
            assertEquals("the spaced checkpoint at 16 rows serves the shared 16 rows", 16,
                    b.reusedPrefix());
            assertSameResult("diverging", divergingRef, b.result());
        });
    }
}

