package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.image.ImageTokenBudget;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.media.MediaIoOptions;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.media.MediaLimits.Modality;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The media cache and the projector release of {@link MediaContext}: vLLM's media {@code uuid} keys an item's
 * encoding so a repeat skips reading, decoding and encoding it, an item whose encoding was dropped encodes again from
 * its processed tiles or samples without reading its data, a uuid neither layer holds without data is refused, the
 * chunks carry the item's key (the uuid, else the digest of the bytes, with the loader options as hash factors), the
 * byte budget evicts the least recently used encoding and 0 caches nothing, and a released projector refuses to
 * encode until it reopens with the same rows.
 */
public class MediaContextCacheTest {

    private static final int WIDTH = 16;

    private static MediaContext open(Path file, long cacheBytes) throws IOException {
        return MediaContext.open(file.toString(), WIDTH, false, ImageTokenBudget.none(), t -> false,
                MmprojOptions.cpu(), cacheBytes);
    }

    private static MediaContext llava(long cacheBytes) throws IOException {
        return open(MmprojTestFiles.llava(WIDTH), cacheBytes);
    }

    /** The chunks of one image appended alone. */
    private static List<MediaChunk> image(MediaContext ctx, String uuid, Supplier<byte[]> data) {
        MediaPrompt.Builder b = MediaPrompt.builder();
        ctx.appendImage(b, uuid, data, 0, MediaContextCacheTest::chars);
        return b.build().chunks();
    }

    private static int[] chars(String text) {
        int[] out = new int[text.length()];
        for (int i = 0; i < out.length; i++) {
            out[i] = text.charAt(i);
        }
        return out;
    }

    /** Data that fails the test when read. */
    private static Supplier<byte[]> unread() {
        return () -> {
            throw new AssertionError("the cached item's data must not be read");
        };
    }

    @Test
    public void aRepeatedUuidServesTheCachedEncodingWithoutReadingItsData() throws IOException {
        byte[] png = MediaContextTest.png(12, 8);
        AtomicInteger reads = new AtomicInteger();
        try (MediaContext ctx = llava(MediaContext.DEFAULT_CACHE_BYTES)) {
            assertFalse(ctx.hasCached(Modality.IMAGE, "cat-1"));
            List<MediaChunk> first = image(ctx, "cat-1", () -> {
                reads.incrementAndGet();
                return png;
            });
            assertEquals(1, reads.get());
            assertTrue(ctx.hasCached(Modality.IMAGE, "cat-1"));
            assertFalse("uuids are keyed per modality", ctx.hasCached(Modality.AUDIO, "cat-1"));

            long hits = ctx.cacheHits();
            assertEquals(first, image(ctx, "cat-1", unread()));
            assertEquals(first, image(ctx, "cat-1", null));
            assertEquals(hits + 2, ctx.cacheHits());
            assertEquals(1, ctx.cachedItems());
        }
    }

    @Test
    public void theUuidKeysTheItemRatherThanItsBytes() throws IOException {
        // vLLM uses the client's uuid as the item's hash: the client vouches that one uuid names one item
        try (MediaContext ctx = llava(MediaContext.DEFAULT_CACHE_BYTES)) {
            byte[] a = MediaContextTest.png(12, 8);
            byte[] b = MediaContextTest.png(8, 12);
            List<MediaChunk> first = image(ctx, "same", () -> a);
            assertEquals(first, image(ctx, "same", () -> b));
            assertNotEquals("without the uuid the other image encodes", first, image(ctx, null, () -> b));
            assertEquals("an item without a uuid is cached by its digest", 2, ctx.cachedItems());
        }
    }

    @Test
    public void aUuidItemsChunksCarryTheItemsKeyAndPieceAsTheirIdentity() throws IOException {
        // one identity for both caches: the engine's encoder cache shares rows by the item key and piece
        byte[] png = MediaContextTest.png(12, 8);
        byte[] other = MediaContextTest.png(8, 12);
        try (MediaContext ctx = llava(MediaContext.DEFAULT_CACHE_BYTES)) {
            String named = mediaId(image(ctx, "cat-2", () -> png));
            assertEquals("the uuid and the piece are the identity",
                    ctx.keyOf(Modality.IMAGE, "cat-2", null) + "#0", named);
            assertEquals("a uuid-only repeat keeps the identity", named, mediaId(image(ctx, "cat-2", null)));
            assertNotEquals("other bytes under another uuid name other rows", named,
                    mediaId(image(ctx, "dog", () -> other)));
            assertEquals("the same bytes without a uuid name the digest",
                    ctx.keyOf(Modality.IMAGE, null, png) + "#0", mediaId(image(ctx, null, () -> png)));
        }
    }

    /** The identity of the only media chunk among {@code chunks}. */
    private static String mediaId(List<MediaChunk> chunks) {
        String id = null;
        for (MediaChunk c : chunks) {
            if (!c.isText()) {
                assertEquals("one media chunk", null, id);
                id = c.id();
            }
        }
        assertTrue("a media chunk carries an identity", id != null);
        return id;
    }

    @Test
    public void aUuidTheCacheDoesNotHoldWithoutDataIsRefusedNamingIt() throws IOException {
        try (MediaContext ctx = llava(MediaContext.DEFAULT_CACHE_BYTES)) {
            MediaInputException e = assertThrows(MediaInputException.class, () -> image(ctx, "never-sent", null));
            assertTrue(e.getMessage(), e.getMessage().contains("uuid 'never-sent'"));
            assertTrue(e.getMessage(), e.getMessage().contains("not in the media cache"));
        }
    }

    @Test
    public void aZeroBudgetCachesNothing() throws IOException {
        byte[] png = MediaContextTest.png(8, 8);
        try (MediaContext ctx = llava(0)) {
            List<MediaChunk> first = image(ctx, "k", () -> png);
            assertEquals(0, ctx.cachedItems());
            List<MediaChunk> byDigest = image(ctx, null, () -> png);
            assertNotEquals("a uuid and a digest name two items", first, byDigest);
            assertTrue(Arrays.deepEquals(mediaChunk(first).embeddings(), mediaChunk(byDigest).embeddings()));
            assertEquals(0, ctx.cachedItems());
            assertEquals(0, ctx.processedItems());
            assertThrows(MediaInputException.class, () -> image(ctx, "k", null));
        }
    }

    @Test
    public void theBudgetEvictsTheLeastRecentlyUsedEncoding() throws IOException {
        // the uuids key the items, so one image's bytes serve every uuid at the same size
        byte[] png = MediaContextTest.png(8, 8);
        long one;
        try (MediaContext probe = llava(MediaContext.DEFAULT_CACHE_BYTES)) {
            image(probe, "x", () -> png);
            one = probe.cachedBytes();
        }
        assertTrue(one > 0);
        assertEquals("an encoding's size is its rows of floats", 0, one % ((long) WIDTH * Float.BYTES));
        try (MediaContext ctx = llava(2 * one)) {
            assertEquals(2 * one, ctx.cacheBudget());
            image(ctx, "a", () -> png);
            image(ctx, "b", () -> png);
            image(ctx, "a", unread());
            image(ctx, "c", () -> png);
            assertEquals(2, ctx.cachedItems());
            assertEquals(2 * one, ctx.cachedBytes());
            long hits = ctx.cacheHits();
            image(ctx, "c", unread());
            image(ctx, "a", unread());
            assertEquals("a and c kept their encodings", hits + 2, ctx.cacheHits());
            image(ctx, "b", () -> png);
            assertEquals("b's encoding was evicted", hits + 2, ctx.cacheHits());

            ctx.clearCache();
            assertEquals(0, ctx.cachedItems());
            assertEquals(0, ctx.cachedBytes());
            assertFalse(ctx.hasCached(Modality.IMAGE, "a"));
        }
    }

    @Test
    public void anEncodedHitNeverReachesTheProcessedLayer() throws IOException {
        byte[] png = MediaContextTest.png(12, 8);
        try (MediaContext ctx = llava(MediaContext.DEFAULT_CACHE_BYTES)) {
            List<MediaChunk> first = image(ctx, null, () -> png);
            assertEquals(first, image(ctx, null, () -> png.clone()));
            assertEquals(2, ctx.cacheQueries());
            assertEquals(1, ctx.cacheHits());
            assertEquals("only the first append looked up the processed layer", 1, ctx.processedQueries());
            assertEquals(0, ctx.processedHits());
            assertEquals(1, ctx.processedItems());
            assertTrue("the prepared tiles' RGB bytes", ctx.processedBytes() > 0 && ctx.processedBytes() % 3 == 0);
        }
    }

    @Test
    public void anImageWhoseEncodingWasDroppedEncodesFromItsProcessedTilesWithoutItsData() throws IOException {
        byte[] png = MediaContextTest.png(12, 8);
        try (MediaContext ctx = llava(MediaContext.DEFAULT_CACHE_BYTES)) {
            List<MediaChunk> first = image(ctx, "cat-3", () -> png);
            ctx.clearEncodings();
            assertEquals(0, ctx.cachedItems());
            assertTrue("the processed layer still holds the uuid", ctx.hasCached(Modality.IMAGE, "cat-3"));
            assertEquals("re-encoding the prepared tiles gives the same rows and identity", first,
                    image(ctx, "cat-3", unread()));
            assertEquals(1, ctx.processedHits());
            assertEquals("the new encoding is cached again", 1, ctx.cachedItems());

            ctx.clearEncodings();
            assertEquals("a uuid sent alone is served from the processed layer", first, image(ctx, "cat-3", null));

            ctx.clearCache();
            assertEquals(0, ctx.processedItems());
            assertEquals(0, ctx.processedBytes());
            assertFalse(ctx.hasCached(Modality.IMAGE, "cat-3"));
            MediaInputException e = assertThrows(MediaInputException.class, () -> image(ctx, "cat-3", null));
            assertTrue(e.getMessage(), e.getMessage().contains("uuid 'cat-3'"));
        }
    }

    @Test
    public void aDigestKeyedImageSkipsItsDecodeAfterItsEncodingWasDropped() throws IOException {
        byte[] png = MediaContextTest.png(12, 8);
        try (MediaContext ctx = llava(MediaContext.DEFAULT_CACHE_BYTES)) {
            List<MediaChunk> first = image(ctx, null, () -> png);
            ctx.clearEncodings();
            assertEquals(first, image(ctx, null, () -> png));
            assertEquals("the digest keys both layers", 1, ctx.processedHits());
        }
    }

    @Test
    public void anAudioClipsProcessedSamplesServeItsUuidAfterItsEncodingWasDropped() throws IOException {
        byte[] wav = MediaContextTest.wav(1);
        try (MediaContext ctx = open(MmprojTestFiles.qwen2aAudio(WIDTH), MediaContext.DEFAULT_CACHE_BYTES)) {
            MediaPrompt.Builder first = MediaPrompt.builder();
            ctx.appendAudio(first, "clip-7", () -> wav, "clip.wav", MediaContextCacheTest::chars);
            assertEquals("16 kHz mono for one second, as floats", 16000L * Float.BYTES, ctx.processedBytes());
            ctx.clearEncodings();
            MediaPrompt.Builder again = MediaPrompt.builder();
            ctx.appendAudio(again, "clip-7", null, "clip.wav", MediaContextCacheTest::chars);
            assertEquals(first.build().chunks(), again.build().chunks());
            assertEquals(1, ctx.processedHits());
        }
    }

    @Test
    public void aReleasedProjectorRefusesToEncodeUntilItReopensWithTheSameRows() throws IOException {
        byte[] cat = MediaContextTest.png(12, 8);
        byte[] dog = MediaContextTest.png(8, 12);
        try (MediaContext ctx = llava(MediaContext.DEFAULT_CACHE_BYTES)) {
            List<MediaChunk> before = image(ctx, null, () -> dog);
            List<MediaChunk> cached = image(ctx, "cat", () -> cat);
            ctx.release();
            ctx.release();
            assertTrue(ctx.released());
            assertTrue("the capabilities stay while released", ctx.supportsVision());
            assertTrue(ctx.toString(), ctx.toString().contains("released"));
            assertThrows(MediaContext.ReleasedException.class, () -> image(ctx, "other", () -> dog));
            assertEquals("a cached encoding needs no projector", cached, image(ctx, "cat", null));

            ctx.reopen();
            ctx.reopen();
            assertFalse(ctx.released());
            ctx.clearCache();
            assertRowsEqual(before, image(ctx, null, () -> dog));
        }
    }

    /** The media chunks of {@code a} and {@code b} hold equal rows. */
    private static void assertRowsEqual(List<MediaChunk> a, List<MediaChunk> b) {
        assertEquals(a, b);
        for (int i = 0; i < a.size(); i++) {
            if (!a.get(i).isText()) {
                assertTrue("chunk " + i, Arrays.deepEquals(a.get(i).embeddings(), b.get(i).embeddings()));
            }
        }
    }

    /** The single media chunk of a one-image prompt. */
    private static MediaChunk mediaChunk(List<MediaChunk> chunks) {
        MediaChunk out = null;
        for (MediaChunk c : chunks) {
            if (!c.isText()) {
                assertNull("one media chunk", out);
                out = c;
            }
        }
        return out;
    }

    @Test
    public void theChunksCarryTheItemKeyTheCacheStoresItUnder() throws Exception {
        byte[] png = MediaContextTest.png(12, 8);
        try (MediaContext ctx = llava(MediaContext.DEFAULT_CACHE_BYTES)) {
            MediaKey named = mediaChunk(image(ctx, "cat-1", () -> png)).item();
            assertEquals("the uuid is the item's mm_hash", "cat-1", named.id());
            assertEquals(ctx.keyOf(Modality.IMAGE, "cat-1", null), named);
            MediaKey digest = mediaChunk(image(ctx, null, () -> png)).item();
            assertEquals(MediaKey.hex(MessageDigest.getInstance("SHA-256").digest(png)), digest.id());
            assertEquals(MediaKey.Source.CONTENT, digest.source());
        }
    }

    @Test
    public void loaderOptionsKeyTheItemsAndShapeTheirDecode() throws IOException {
        byte[] png = MediaContextTest.png(12, 8);
        MediaIoOptions gray = MediaIoOptions.parse("{\"image\": {\"image_mode\": \"L\"}}");
        try (MediaContext rgb = llava(MediaContext.DEFAULT_CACHE_BYTES);
             MediaContext l = MediaContext.open(MmprojTestFiles.llava(WIDTH).toString(), WIDTH, false,
                     ImageTokenBudget.none(), t -> false, MmprojOptions.cpu(), MediaContext.DEFAULT_CACHE_BYTES,
                     gray)) {
            assertEquals(gray, l.ioOptions());
            MediaKey key = mediaChunk(image(l, "cat", () -> png)).item();
            assertNotEquals("the options are hash factors of a uuid, as in vLLM", "cat", key.id());
            assertEquals(l.keyOf(Modality.IMAGE, "cat", null), key);
            assertNotEquals(rgb.keyOf(Modality.IMAGE, "cat", null), key);
            assertEquals("the audio keys keep their uuid", "cat", l.keyOf(Modality.AUDIO, "cat", null).id());
            float[][] grayRows = mediaChunk(image(l, null, () -> png)).embeddings();
            float[][] colourRows = mediaChunk(image(rgb, null, () -> png)).embeddings();
            assertFalse("a grayscale load encodes other rows", Arrays.deepEquals(grayRows, colourRows));
        }
    }

    @Test
    public void aClosedContextNeitherReleasesNorReopens() throws IOException {
        MediaContext ctx = llava(MediaContext.DEFAULT_CACHE_BYTES);
        ctx.release();
        ctx.close();
        ctx.release();
        assertThrows(IllegalStateException.class, ctx::reopen);
        byte[] png = MediaContextTest.png(8, 8);
        assertThrows(IllegalStateException.class, () -> image(ctx, null, () -> png));
    }

    @Test
    public void audioTakesItsUuidToo() throws IOException {
        byte[] wav = MediaContextTest.wav(1);
        try (MediaContext ctx = open(MmprojTestFiles.qwen2aAudio(WIDTH), MediaContext.DEFAULT_CACHE_BYTES)) {
            MediaPrompt.Builder first = MediaPrompt.builder();
            ctx.appendAudio(first, "clip-1", () -> wav, "clip.wav", MediaContextCacheTest::chars);
            assertTrue(ctx.hasCached(Modality.AUDIO, "clip-1"));
            assertFalse(ctx.hasCached(Modality.IMAGE, "clip-1"));
            MediaPrompt.Builder second = MediaPrompt.builder();
            ctx.appendAudio(second, "clip-1", null, "clip.wav", MediaContextCacheTest::chars);
            assertEquals(first.build().chunks(), second.build().chunks());
        }
    }
}
