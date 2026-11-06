package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link MediaChunk} and {@link MediaPrompt}: a chunk counts one KV row per token or embedding and the positions
 * its base advances by (a media chunk can take fewer positions than rows), a prompt sums its chunks, and every
 * invalid shape refuses by name.
 */
public class MediaPromptTest {

    private static float[][] rows(int n, int width, float v) {
        float[][] r = new float[n][width];
        for (float[] row : r) {
            java.util.Arrays.fill(row, v);
        }
        return r;
    }

    @Test
    public void aTextChunkCountsOneRowAndOnePositionPerToken() {
        MediaChunk c = MediaChunk.text(new int[]{5, 6, 7});
        assertTrue(c.isText());
        assertEquals(3, c.nTokens());
        assertEquals(3, c.nPositions());
        assertFalse(c.nonCausal());
        assertArrayEquals(new int[]{5, 6, 7}, c.tokens());
    }

    @Test
    public void aMediaChunkCountsRowsAgainstItsPositions() {
        MediaRopePositions p = MediaRopePositions.image2d(0, 4, 2);
        MediaChunk c = MediaChunk.media(p, rows(8, 6, 0.5f), true);
        assertFalse(c.isText());
        assertEquals(8, c.nTokens());
        assertEquals(4, p.nPositions());
        assertEquals(4, c.nPositions());
        assertTrue(c.nonCausal());
        assertEquals(6, c.embeddings()[0].length);
    }

    @Test
    public void aPromptSumsChunksRowsAndPositions() {
        MediaPrompt p = MediaPrompt.builder()
                .addText(new int[]{1, 2, 3})
                .addMedia(MediaRopePositions.image2d(3, 4, 2), rows(8, 6, 1), true)
                .addText(new int[]{4, 5})
                .build();
        assertEquals(3, p.chunks().size());
        assertEquals(1, p.nMediaChunks());
        assertFalse(p.isTextOnly());
        assertEquals(6, p.mediaWidth());
        assertEquals(3 + 8 + 2, p.nTokens());
        assertEquals(3 + 4 + 2, p.nPositions());
    }

    @Test
    public void samplerTokensSkipMediaRows() {
        MediaPrompt p = MediaPrompt.builder()
                .addText(new int[]{1, 2})
                .addMedia(MediaRopePositions.image2d(2, 2, 2), rows(4, 4, 1), false)
                .addText(new int[]{3})
                .build();
        assertArrayEquals(new int[]{1, 2, 3}, p.samplerTokens());
    }

    @Test
    public void aTextOnlyPromptIsThePlainTokenRun() {
        MediaPrompt p = MediaPrompt.ofText(new int[]{9, 8, 7});
        assertTrue(p.isTextOnly());
        assertEquals(3, p.nTokens());
        assertEquals(3, p.nPositions());
        assertArrayEquals(new int[]{9, 8, 7}, p.samplerTokens());
    }

    @Test
    public void validationRefusesBadShapesByName() {
        assertThrows(IllegalArgumentException.class, () -> MediaChunk.text(new int[0]));
        assertThrows(IllegalArgumentException.class, () -> MediaChunk.text(new int[]{-1}));
        assertThrows(NullPointerException.class, () -> MediaChunk.media(null, rows(1, 2, 0), false));
        assertThrows(IllegalArgumentException.class,
                () -> MediaChunk.media(MediaRopePositions.image2d(0, 2, 2), rows(3, 2, 0), false));
        assertThrows(IllegalArgumentException.class,
                () -> MediaChunk.media(MediaRopePositions.normal(0, 2), new float[][]{{1}, {2, 3}}, false));
        assertThrows(IllegalStateException.class, () -> MediaPrompt.builder().build());
        // one media width per prompt
        MediaPrompt.Builder b = MediaPrompt.builder()
                .addMedia(MediaRopePositions.image2d(0, 2, 2), rows(4, 4, 0), false);
        assertThrows(IllegalArgumentException.class,
                () -> b.addMedia(MediaRopePositions.image2d(4, 2, 2), rows(4, 8, 0), false));
    }

    @Test
    public void appendTextExtendsTheTrailingTextChunkLikeMtmdsAddText() {
        MediaPrompt.Builder b = MediaPrompt.builder();
        assertTrue(b.isEmpty());
        b.appendText(new int[0]).appendText(null);
        assertTrue(b.isEmpty());
        b.appendText(new int[]{1, 2}).appendText(new int[]{3});
        assertEquals(3, b.position());
        b.addMedia(MediaRopePositions.image2d(3, 4, 2), rows(8, 4, 0), false);
        assertEquals(7, b.position());
        b.appendText(new int[]{4}).appendText(new int[]{5});
        assertFalse(b.isEmpty());
        MediaPrompt p = b.build();
        assertEquals(3, p.chunks().size());
        assertArrayEquals(new int[]{1, 2, 3}, p.chunks().get(0).tokens());
        assertArrayEquals(new int[]{4, 5}, p.chunks().get(2).tokens());
        assertThrows(IllegalArgumentException.class, () -> MediaPrompt.builder().appendText(new int[]{-2}));
    }

    @Test
    public void chunksWithEqualContentsAreEqual() {
        assertEquals(MediaChunk.text(new int[]{1, 2}), MediaChunk.text(new int[]{1, 2}));
        assertNotEquals(MediaChunk.text(new int[]{1, 2}),
                MediaChunk.media(MediaRopePositions.normal(0, 2), rows(2, 2, 0), false));
        assertEquals(MediaChunk.media(MediaRopePositions.normal(0, 2), rows(2, 2, 0), true),
                MediaChunk.media(MediaRopePositions.normal(0, 2), rows(2, 2, 0), true));
    }

    @Test
    public void anIdentifiedChunkHoldsItsRowsAsGivenAndNamesItsId() {
        float[][] held = rows(4, 4, 1);
        MediaChunk c = MediaChunk.media("img:abc", MediaRopePositions.image2d(0, 4, 1), held, false);
        assertFalse(c.isText());
        assertEquals("img:abc", c.id());
        assertEquals(4, c.nTokens());
        assertSame("the rows are held, not copied", held, c.sharedRows());
        assertArrayEquals(held[1], c.embeddings()[1], 0f);
        assertEquals(MediaChunk.media("img:abc", MediaRopePositions.image2d(0, 4, 1), held, false), c);
        assertThrows(IllegalArgumentException.class,
                () -> MediaChunk.media(" ", MediaRopePositions.normal(0, 1), rows(1, 2, 0), false));
        assertThrows(IllegalArgumentException.class,
                () -> MediaChunk.media(null, MediaRopePositions.normal(0, 1), rows(1, 2, 0), false));
        assertNull("an unidentified chunk names no id", MediaChunk.text(new int[]{1}).id());
    }

    @Test
    public void anIdentifiedPromptAdoptsTheSharedRowsItsCacheReturns() {
        float[][] own = rows(4, 4, 1);
        float[][] canonical = rows(4, 4, 2);
        MediaPrompt p = MediaPrompt.builder()
                .addText(new int[]{1, 2})
                .addMedia("img:abc", MediaRopePositions.image2d(2, 4, 1), own, false)
                .addMedia(MediaRopePositions.normal(6, 2), rows(2, 4, 3), false)
                .addText(new int[]{3})
                .build();

        MediaPrompt shared = p.withSharedMedia((id, rows) -> id.equals("img:abc") ? canonical : rows);
        assertSame("the identified chunk holds the canonical rows", canonical,
                shared.chunks().get(1).sharedRows());
        assertSame("the prompt object changes only where a chunk changed", p,
                p.withSharedMedia((id, rows) -> rows));
        assertSame("the caller's prompt keeps its own rows", own, p.chunks().get(1).sharedRows());
        assertEquals("shape and order survive the swap", p.nTokens(), shared.nTokens());
        assertEquals(p.nPositions(), shared.nPositions());
        assertSame("unidentified chunks pass through", shared.chunks().get(2), p.chunks().get(2));
    }

    @Test
    public void theMediaItemRuleSetsMediaApartFromTextAndPromptEmbeddings() {
        assertFalse(MediaChunk.text(new int[]{1}).isMediaItem());
        assertFalse(MediaChunk.embeds(0, rows(2, 4, 0)).isMediaItem());
        assertTrue(MediaChunk.media(MediaRopePositions.normal(0, 2), rows(2, 4, 0), false).isMediaItem());
        assertTrue(MediaChunk.media("img:a#0", MediaRopePositions.normal(0, 2), rows(2, 4, 0), false).isMediaItem());
    }

    @Test
    public void anIdentityNamesTheEncoderOutputWithoutChangingTheItemKey() {
        MediaRopePositions at = MediaRopePositions.image2d(2, 2, 2);
        MediaChunk plain = MediaChunk.media(at, rows(4, 4, 1), true);
        MediaChunk named = MediaChunk.media("image:abc#0", at, rows(4, 4, 1), true);
        assertEquals("the prefix cache keys the rows, not the encoder cache's identity", plain.item(),
                named.item());
        assertNotEquals("identities still tell chunks apart", plain, named);

        MediaPrompt p = MediaPrompt.builder()
                .addText(new int[]{1, 2})
                .addMedia("image:abc#0", at, rows(4, 4, 1), true)
                .addEmbeds(4, rows(2, 4, 5))
                .build();
        MediaPrompt shared = p.withSharedMedia((id, r) -> rows(4, 4, 1));
        assertEquals(p.chunks().get(1).item(), shared.chunks().get(1).item());
        assertTrue("a prompt-embedding run passes the swap unchanged", shared.chunks().get(2).isPromptEmbeds());
        assertSame(p.chunks().get(2), shared.chunks().get(2));
    }
}
