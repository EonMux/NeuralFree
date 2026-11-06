package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.image.ImageTokenBudget;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.media.MediaIoOptions;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.media.MediaLimits.Modality;
import com.eonmux.neuralfree.gguf.video.VideoOptions;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The shared encoder cache of {@link MediaContext} (vLLM's encoder cache transfer): a producer saves the encoder
 * output of every item it encodes, a consumer loads it without reading the item's data or running its encoder, the
 * consumer counts the item as cached before it loads it, and an entry a producer of another projector width wrote is
 * refused by name.
 */
public class MediaContextSharedCacheTest {

    private static final int WIDTH = 16;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static MediaContext open(Path file, EncoderCacheConnector connector) throws IOException {
        return MediaContext.open(file.toString(), WIDTH, false, ImageTokenBudget.none(), t -> false,
                MmprojOptions.cpu(), MediaContext.DEFAULT_CACHE_BYTES, MediaIoOptions.defaults(),
                VideoOptions.defaults(), connector, Collections.emptyList());
    }

    private static List<MediaChunk> image(MediaContext ctx, String uuid, Supplier<byte[]> data) {
        MediaPrompt.Builder b = MediaPrompt.builder();
        ctx.appendImage(b, uuid, data, 0, MediaContextSharedCacheTest::chars);
        return b.build().chunks();
    }

    private static int[] chars(String text) {
        int[] out = new int[text.length()];
        for (int i = 0; i < out.length; i++) {
            out[i] = text.charAt(i);
        }
        return out;
    }

    private static Supplier<byte[]> unread() {
        return () -> {
            throw new AssertionError("a shared encoder cache hit must not read the item's data");
        };
    }

    @Test
    public void aConsumerLoadsWhatAProducerSavedWithoutItsDataOrEncoder() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        Path dir = tmp.newFolder("shared").toPath();
        byte[] png = MediaContextTest.png(12, 8);
        DirectoryEncoderCacheConnector producer =
                new DirectoryEncoderCacheConnector(dir, EncoderCacheConnector.Role.PRODUCER);
        List<MediaChunk> produced;
        try (MediaContext ctx = open(file, producer)) {
            produced = image(ctx, "cat-1", () -> png);
            assertEquals(1, ctx.sharedSaves());
        }
        MediaKey key = MediaKey.ofUuid(Modality.IMAGE, "cat-1", null);
        assertTrue("the producer saved the item", producer.has(key));

        DirectoryEncoderCacheConnector consumer =
                new DirectoryEncoderCacheConnector(dir, EncoderCacheConnector.Role.CONSUMER);
        try (MediaContext ctx = open(file, consumer)) {
            assertFalse("nothing is cached locally yet", ctx.cachedItems() > 0);
            assertTrue("a uuid only the shared cache holds is cached", ctx.hasCached(Modality.IMAGE, "cat-1"));
            List<MediaChunk> loaded = image(ctx, "cat-1", unread());
            assertEquals(produced, loaded);
            assertEquals(1, ctx.cachedItems());
            assertEquals(1, ctx.sharedLoads());
            assertEquals(0, ctx.sharedSaves());
            assertEquals("the load fills the encoded layer", produced, image(ctx, "cat-1", null));
        }
    }

    @Test
    public void aConsumerAlsoSavesAndReloadsAfterItsCacheIsDropped() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        Path dir = tmp.newFolder("shared").toPath();
        byte[] png = MediaContextTest.png(10, 10);
        DirectoryEncoderCacheConnector both =
                new DirectoryEncoderCacheConnector(dir, EncoderCacheConnector.Role.BOTH);
        try (MediaContext ctx = open(file, both)) {
            List<MediaChunk> first = image(ctx, "cat-2", () -> png);
            assertEquals(1, ctx.sharedSaves());
            ctx.clearEncodings();
            assertEquals("the second append comes from the shared cache", first, image(ctx, "cat-2", unread()));
            assertEquals(1, ctx.sharedLoads());
            assertEquals(1, ctx.sharedSaves());
        }
        assertTrue(both.has(MediaKey.ofUuid(Modality.IMAGE, "cat-2", null)));
    }

    @Test
    public void anEntryOfAnotherProjectorWidthIsRefusedNamingIt() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        Path dir = tmp.newFolder("shared").toPath();
        DirectoryEncoderCacheConnector both =
                new DirectoryEncoderCacheConnector(dir, EncoderCacheConnector.Role.BOTH);
        both.put(MediaKey.ofUuid(Modality.IMAGE, "wide", null),
                EncoderCacheEntry.of(Collections.singletonList(
                        EncoderCacheEntry.Chunk.rows(new float[][]{new float[WIDTH * 2]}))));
        try (MediaContext ctx = open(file, both)) {
            MediaInputException e = assertThrows(MediaInputException.class, () -> image(ctx, "wide", null));
            assertTrue(e.getMessage(), e.getMessage().contains(String.valueOf(WIDTH * 2)));
            assertTrue(e.getMessage(), e.getMessage().contains("another projector"));
        }
    }

    @Test
    public void withoutAConnectorARepeatedUuidStillNeedsItsData() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        try (MediaContext ctx = open(file, null)) {
            MediaInputException e = assertThrows(MediaInputException.class, () -> image(ctx, "lonely", null));
            assertTrue(e.getMessage(), e.getMessage().contains("not in the media cache"));
        }
    }
}
