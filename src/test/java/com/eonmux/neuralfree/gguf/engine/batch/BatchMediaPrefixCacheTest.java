package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EmbedsPrompt;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.BlockExtraKey;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvCacheEvent;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventBatch;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublishers;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.media.MediaLimits;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The prefix cache over media and prompt-embedding prompts on the synthetic qwen2vl (M-RoPE), pages of four rows:
 * a block's media rows are keyed by their item's identity (vLLM's {@code mm_hash}) and offset, so a repeat of the
 * same image at the same place reuses every row but the last, a different image reuses only the text before it, a
 * text-only prompt still shares that text, a reuse never ends inside an image, prompt-embedding rows share row by
 * row, media sharing the encoder cache's rows under a step encoder budget keep their reuse, a preempted media
 * sequence recomputes over its cached blocks, and the stored blocks publish their media keys: a client's uuid is
 * the published mm_hash, and concurrent identical media prompts fork off the running donor. Every cached run
 * decodes the tokens of an engine without a prefix cache.
 */
public class BatchMediaPrefixCacheTest {

    private static final int PAGE = 4;
    private static final int N_EMBD = 64;
    private static final int HEAD = 6;
    /** A 4 x 4 grid: sixteen rows over four positions. */
    private static final int GRID = 4;

    private static EngineConfig config(int nCtx, boolean reuse) {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(nCtx).ubatch(8).kvPaged(true)
                .kvPageSize(PAGE).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(reuse)
                .build()).build();
    }

    private static GenerationRequest request(int n) {
        return GenerationRequest.builder().maxNewTokens(n).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 11 + i * 7) % 250;
        }
        return t;
    }

    private static float[][] image(float v) {
        float[][] r = new float[GRID * GRID][N_EMBD];
        for (int i = 0; i < r.length; i++) {
            for (int d = 0; d < N_EMBD; d++) {
                r[i][d] = v * (1 + 0.03f * ((i * 3 + d) % 7)) - 0.1f * ((d % 3) - 1);
            }
        }
        return r;
    }

    private static float[][] embedRows(int seed, int n) {
        float[][] r = new float[n][N_EMBD];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < N_EMBD; d++) {
                r[i][d] = 0.05f * (((seed * 5 + i * 3 + d) % 11) - 5);
            }
        }
        return r;
    }

    /** {@code head}, the image grid at the cursor, then {@code tail} (when non-null). */
    private static MediaPrompt withImage(int[] head, float[][] image, int[] tail) {
        MediaPrompt.Builder b = MediaPrompt.builder().addText(head)
                .addMedia(MediaRopePositions.image2d(head.length, GRID, GRID), image, true);
        if (tail != null) {
            b.addText(tail);
        }
        return b.build();
    }

    /** {@code head}, the image named {@code key} at the cursor, then {@code tail} (when non-null). */
    private static MediaPrompt withNamedImage(int[] head, MediaKey key, float[][] image, int[] tail) {
        MediaPrompt.Builder b = MediaPrompt.builder().addText(head)
                .addMedia(key, 0, MediaRopePositions.image2d(head.length, GRID, GRID), image, true);
        if (tail != null) {
            b.addText(tail);
        }
        return b.build();
    }

    private static BatchSequence run(BatchEngine engine, MediaPrompt prompt, GenerationRequest request) {
        BatchSequence seq = engine.submit(prompt, request, (TokenCallback) null);
        for (int step = 0; step < 2_000 && !seq.finished(); step++) {
            engine.pump();
        }
        assertEquals("the request completed: " + seq.error(), BatchSequence.State.DONE, seq.state());
        return seq;
    }

    /** The tokens each prompt decodes on an engine without a prefix cache. */
    private static int[][] uncached(Path model, MediaPrompt[] prompts, int n) throws IOException {
        EngineConfig cfg = config(128, false);
        int[][] out = new int[prompts.length][];
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            for (int i = 0; i < prompts.length; i++) {
                BatchSequence seq = run(engine, prompts[i], request(n));
                assertEquals(0, seq.reusedPrefix());
                out[i] = seq.result().tokens();
            }
        }
        return out;
    }

    /** Runs the prompts one after the other on a cached engine; returns each one's reused rows. */
    private static int[] cachedReuse(Path model, MediaPrompt[] prompts, int[][] expected) throws IOException {
        EngineConfig cfg = config(128, true);
        int[] reused = new int[prompts.length];
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            for (int i = 0; i < prompts.length; i++) {
                BatchSequence seq = run(engine, prompts[i], request(expected[i].length));
                reused[i] = seq.reusedPrefix();
                assertArrayEquals("prompt " + i + " decodes as uncached", expected[i], seq.result().tokens());
            }
        }
        return reused;
    }

    @Test
    public void theSameImageSharesItsBlocksAndADifferentOneOnlyTheTextBeforeIt() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        int[] head = tokens(1, HEAD);
        int[] tail = tokens(2, 5);
        MediaPrompt first = withImage(head, image(0.3f), tail);
        MediaPrompt same = withImage(head.clone(), image(0.3f), tail.clone());
        MediaPrompt other = withImage(head, image(0.7f), tail);
        MediaPrompt textOnly = MediaPrompt.ofText(concat(head, tokens(3, 5)));
        MediaPrompt elsewhere = withImage(tokens(4, HEAD), image(0.3f), tail);
        MediaPrompt[] prompts = {first, same, other, textOnly, elsewhere};
        int[][] expected = uncached(model, prompts, 4);

        int[] reused = cachedReuse(model, prompts, expected);
        int rows = first.nTokens();
        assertEquals(0, reused[0]);
        assertEquals("the same image at the same place reuses every row but the last", rows - 1, reused[1]);
        assertEquals("a different image shares only the text before it", HEAD, reused[2]);
        assertEquals("a text-only prompt shares the text prefix before the media", HEAD, reused[3]);
        assertEquals("the same image after other text shares nothing", 0, reused[4]);
    }

    @Test
    public void aReuseStopsShortOfAnImageItWouldEndInside() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        int[] head = tokens(5, HEAD);
        MediaPrompt endsInImage = withImage(head, image(0.4f), null);
        MediaPrompt[] prompts = {endsInImage, withImage(head, image(0.4f), null)};
        int[][] expected = uncached(model, prompts, 4);

        int[] reused = cachedReuse(model, prompts, expected);
        assertEquals("the last row always prefills, and the image it ends prefills whole", HEAD, reused[1]);
    }

    @Test
    public void promptEmbeddingRowsShareRowByRow() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        float[][] rows = embedRows(1, 10);
        float[][] diverging = Arrays.copyOf(rows, 10);
        assertNotEquals(Arrays.deepToString(Arrays.copyOfRange(rows, 6, 10)), Arrays.deepToString(embedRows(2, 4)));
        System.arraycopy(embedRows(2, 4), 0, diverging, 6, 4);
        MediaPrompt[] prompts = {
                EmbedsPrompt.ofRows(rows).toMediaPrompt(),
                EmbedsPrompt.ofRows(rows).toMediaPrompt(),
                EmbedsPrompt.ofRows(diverging).toMediaPrompt(),
                EmbedsPrompt.of(new int[][] {tokens(6, 3), tokens(7, 2)}, new float[][][] {rows}).toMediaPrompt(),
                EmbedsPrompt.of(new int[][] {tokens(6, 3), tokens(8, 4)},
                        new float[][][] {Arrays.copyOf(rows, 7)}).toMediaPrompt()};
        int[][] expected = uncached(model, prompts, 4);

        int[] reused = cachedReuse(model, prompts, expected);
        assertEquals(0, reused[0]);
        assertEquals("the same rows reuse all but the last", rows.length - 1, reused[1]);
        assertEquals("rows share up to the first differing row", 6, reused[2]);
        assertEquals(0, reused[3]);
        assertEquals("tokens, then the rows they share", 3 + 7, reused[4]);
    }

    /**
     * {@code head}, image A, {@code middle}, image B, then five tail tokens; with {@code identified} the images
     * carry encoder-cache identities, on fresh row arrays each call.
     */
    private static MediaPrompt twoImages(int[] head, int[] middle, boolean identified) {
        MediaPrompt.Builder b = MediaPrompt.builder().addText(head);
        addImage(b, identified ? "image:aaa#0" : null, image(0.3f));
        b.addText(middle);
        addImage(b, identified ? "image:bbb#0" : null, image(0.5f));
        return b.addText(tokens(22, 5)).build();
    }

    private static void addImage(MediaPrompt.Builder b, String id, float[][] rows) {
        MediaRopePositions at = MediaRopePositions.image2d(b.position(), GRID, GRID);
        if (id == null) {
            b.addMedia(at, rows, true);
        } else {
            b.addMedia(id, at, rows, true);
        }
    }

    @Test
    public void sharedEncoderOutputAndATightEncoderBudgetKeepTheMediaPrefixReuse() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        int[] head = tokens(20, HEAD);
        int[] middle = tokens(21, 3);
        int[] otherMiddle = tokens(23, 3);
        assertNotEquals(middle[0], otherMiddle[0]);
        int[][] expected = uncached(model, new MediaPrompt[] {twoImages(head, middle, false),
                twoImages(head, middle, false), twoImages(head, otherMiddle, false)}, 4);

        EngineConfig cfg = EngineConfig.builder().memory(config(128, true).memory().toBuilder()
                .maxEncoderTokens(GRID * GRID).encoderCacheRows(4 * GRID * GRID).build()).build();
        MediaPrompt[] prompts = {twoImages(head, middle, true), twoImages(head, middle, true),
                twoImages(head, otherMiddle, true)};
        int[] reused = new int[prompts.length];
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            for (int i = 0; i < prompts.length; i++) {
                BatchSequence seq = run(engine, prompts[i], request(4));
                reused[i] = seq.reusedPrefix();
                assertArrayEquals("prompt " + i + " decodes as uncached", expected[i], seq.result().tokens());
            }
            assertEquals("each image's rows store once", 2 * GRID * GRID, engine.stats().encoderCacheRows());
            assertEquals("the repeats find both images' encoder output", 4, engine.stats().encoderCacheHits());
        }
        assertEquals(0, reused[0]);
        assertEquals("the same images at the same places reuse every row but the last",
                prompts[1].nTokens() - 1, reused[1]);
        assertEquals("the reuse ends at the first image's last row, where the text diverges",
                HEAD + GRID * GRID, reused[2]);
    }

    @Test
    public void aPreemptedMediaSequenceRecomputesOverItsCachedBlocks() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        MediaPrompt a = withImage(tokens(11, HEAD), image(0.2f), tokens(12, 5));
        MediaPrompt b = withImage(tokens(13, HEAD), image(0.6f), tokens(14, 5));
        long[] recomputed = new long[2];
        int[][][] results = new int[2][][];
        for (int pass = 0; pass < 2; pass++) {
            EngineConfig cfg = config(64, pass == 1);
            try (GGUFModel m = new GGUFModel(model.toString(), cfg);
                 BatchEngine engine = BatchEngine.forModel(m, cfg)) {
                BatchSequence first = engine.submit(a, request(30), (TokenCallback) null);
                BatchSequence second = engine.submit(b, request(30), (TokenCallback) null);
                for (int step = 0; step < 4_000 && !(first.finished() && second.finished()); step++) {
                    engine.pump();
                }
                assertEquals(String.valueOf(first.error()), BatchSequence.State.DONE, first.state());
                assertEquals(String.valueOf(second.error()), BatchSequence.State.DONE, second.state());
                assertTrue("the two outgrow the 64 shared cells", engine.stats().preemptions() > 0);
                recomputed[pass] = engine.stats().recomputedTokens();
                results[pass] = new int[][] {first.result().tokens(), second.result().tokens()};
            }
        }
        assertTrue("the cached blocks spare recompute rows: " + recomputed[1] + " of " + recomputed[0],
                recomputed[1] < recomputed[0]);
        assertArrayEquals(results[0][0], results[1][0]);
        assertArrayEquals(results[0][1], results[1][1]);
    }

    @Test
    public void aStoredMediaBlockPublishesItsItemHashAndOffset() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        EngineConfig cfg = config(128, true);
        List<KvEventBatch> batches = new ArrayList<>();
        MediaPrompt prompt = withImage(tokens(15, HEAD), image(0.5f), tokens(16, 5));
        String hash = prompt.chunks().get(1).item().id();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg, KvEventPublishers.listener(batches::add))) {
            run(engine, prompt, request(2));
        }
        List<List<BlockExtraKey>> keys = new ArrayList<>();
        List<Integer> tokenIds = new ArrayList<>();
        for (KvEventBatch batch : batches) {
            for (KvCacheEvent e : batch.events()) {
                KvCacheEvent.BlockStored s = (KvCacheEvent.BlockStored) e;
                keys.addAll(s.extraKeys());
                for (int t : s.tokenIds()) {
                    tokenIds.add(t);
                }
            }
        }
        assertNull("a text block mixes in no extra key", keys.get(0));
        assertEquals("the image starts two rows into block 1", Collections.singletonList(
                BlockExtraKey.media(hash, HEAD - PAGE)), keys.get(1));
        assertEquals("block 2 continues the image begun two rows before it", Collections.singletonList(
                BlockExtraKey.media(hash, HEAD - 2 * PAGE)), keys.get(2));
        for (int r = HEAD; r < HEAD + GRID * GRID; r++) {
            assertEquals("media rows publish no token", CacheRows.NO_TOKEN, (int) tokenIds.get(r));
        }
        assertEquals(prompt.chunks().get(0).tokens()[0], (int) tokenIds.get(0));
    }

    @Test
    public void theItemKeyNamesTheRowsAndTheBlockKeyAddsTheLayoutAndMaskRule() {
        float[][] rows = image(0.3f);
        MediaKey key = MediaChunk.media(MediaRopePositions.image2d(HEAD, GRID, GRID), rows, true).item();
        assertEquals(MediaKey.Source.ROWS, key.source());
        assertEquals(64, key.id().length());
        assertEquals("the same rows are one item wherever they sit", key,
                MediaChunk.media(MediaRopePositions.image2d(HEAD + 1, GRID, GRID), image(0.3f), false).item());
        assertNotEquals("other pixels", key,
                MediaChunk.media(MediaRopePositions.image2d(HEAD, GRID, GRID), image(0.31f), true).item());

        int[] head = tokens(1, HEAD);
        CacheRows base = CacheRows.of(withImage(head, rows, null));
        CacheRows same = CacheRows.of(withImage(head.clone(), image(0.3f), null));
        CacheRows causal = CacheRows.of(MediaPrompt.builder().addText(head)
                .addMedia(MediaRopePositions.image2d(HEAD, GRID, GRID), rows, false).build());
        CacheRows regrid = CacheRows.of(MediaPrompt.builder().addText(head)
                .addMedia(MediaRopePositions.image2d(HEAD, 2 * GRID, GRID / 2), rows, true).build());
        int all = HEAD + GRID * GRID;
        assertEquals(all, CacheRows.commonLength(base, 0, same, 0, all));
        assertEquals("another mask rule shares no media row", HEAD, CacheRows.commonLength(base, 0, causal, 0, all));
        assertEquals("another grid shares no media row", HEAD, CacheRows.commonLength(base, 0, regrid, 0, all));
        CacheRows asEmbeds = CacheRows.of(MediaPrompt.builder().addText(head).addEmbeds(HEAD, rows).build());
        CacheRows asItem = CacheRows.of(MediaPrompt.builder().addText(head)
                .addMedia(MediaRopePositions.normal(HEAD, rows.length), rows, false).build());
        assertEquals("prompt embeddings are not a media item", HEAD,
                CacheRows.commonLength(asEmbeds, 0, asItem, 0, all));
    }

    @Test
    public void aUuidNamesTheItemItsBlocksPublishAsTheMmHash() throws IOException {
        // vLLM's mm_hash from the uuid: the client's uuid keys the item's blocks and is the published hash
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        int[] head = tokens(21, HEAD);
        int[] tail = tokens(22, 5);
        MediaKey cat = MediaKey.ofUuid(MediaLimits.Modality.IMAGE, "cat", null);
        MediaKey dog = MediaKey.ofUuid(MediaLimits.Modality.IMAGE, "dog", null);
        MediaPrompt[] prompts = {withNamedImage(head, cat, image(0.3f), tail),
                withNamedImage(head, dog, image(0.3f), tail)};
        int[][] expected = uncached(model, prompts, 4);

        EngineConfig cfg = config(128, true);
        List<KvEventBatch> batches = new ArrayList<>();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg, KvEventPublishers.listener(batches::add))) {
            BatchSequence first = run(engine, prompts[0], request(4));
            assertArrayEquals(expected[0], first.result().tokens());
            BatchSequence again = run(engine, withNamedImage(head, cat, image(0.3f), tail.clone()), request(4));
            assertEquals("the same uuid reuses every row but the last", prompts[0].nTokens() - 1,
                    again.reusedPrefix());
            assertArrayEquals(expected[0], again.result().tokens());
            BatchSequence other = run(engine, prompts[1], request(4));
            assertEquals("another uuid is another item: only the text before it shares", HEAD, other.reusedPrefix());
            assertArrayEquals(expected[1], other.result().tokens());
        }
        List<List<BlockExtraKey>> keys = new ArrayList<>();
        for (KvEventBatch batch : batches) {
            for (KvCacheEvent e : batch.events()) {
                if (e instanceof KvCacheEvent.BlockStored) {
                    keys.addAll(((KvCacheEvent.BlockStored) e).extraKeys());
                }
            }
        }
        assertEquals("the published mm_hash of the image block is the uuid", Collections.singletonList(
                BlockExtraKey.media("cat", HEAD - PAGE)), keys.get(1));
    }

    @Test
    public void concurrentIdenticalMediaPromptsForkOffTheRunningDonor() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        int[] head = tokens(27, HEAD);
        int[] tail = tokens(28, 5);
        MediaPrompt prompt = withImage(head, image(0.4f), tail);
        int rows = prompt.nTokens();
        int[][] expected = uncached(model, new MediaPrompt[] {prompt}, 4);
        EngineConfig cfg = config(128, true);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence first = engine.submit(prompt, request(4), (TokenCallback) null);
            BatchSequence second = engine.submit(withImage(head.clone(), image(0.4f), tail.clone()), request(4),
                    (TokenCallback) null);
            for (int step = 0; step < 2_000 && !(first.finished() && second.finished()); step++) {
                engine.pump();
            }
            assertEquals(String.valueOf(second.error()), BatchSequence.State.DONE, second.state());
            assertEquals(0, first.reusedPrefix());
            assertEquals("the twin admitted beside it forks off its committed rows", rows - 1, second.reusedPrefix());
            assertArrayEquals(expected[0], first.result().tokens());
            assertArrayEquals(expected[0], second.result().tokens());
            assertTrue(engine.idle());
        }
    }

    private static int[] concat(int[] a, int[] b) {
        int[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
