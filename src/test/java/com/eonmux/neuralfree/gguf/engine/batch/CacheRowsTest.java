package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.kvevents.BlockExtraKey;
import com.eonmux.neuralfree.gguf.engine.lora.LoraPlan;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The prefix cache's keyed rows ({@link CacheRows}): a token row keys by its token, a media row by its item's
 * key and index, a prompt-embedding row by its own digest; rows store the positions the memory holds (an
 * image grid's rows share its base); a reuse never ends inside a media item; fed decodes continue past the
 * prompt's positions; and a block's extra keys follow vLLM's order and offsets.
 */
public class CacheRowsTest {

    private static final int WIDTH = 4;
    private static final int[] HEAD = {5, 6, 7};
    private static final int[] TAIL = {8, 9};

    private static float[][] rows(int n, float v) {
        float[][] r = new float[n][WIDTH];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < WIDTH; d++) {
                r[i][d] = v + 0.25f * i - 0.5f * d;
            }
        }
        return r;
    }

    /** Text, a 3 x 2 image grid (six rows over three positions), text. */
    private static MediaPrompt imagePrompt(float v) {
        return MediaPrompt.builder().addText(HEAD)
                .addMedia(MediaRopePositions.image2d(HEAD.length, 3, 2), rows(6, v), true)
                .addText(TAIL).build();
    }

    @Test
    public void aTextPromptKeysItsTokensAtTheirOwnPositions() {
        CacheRows text = CacheRows.of(MediaPrompt.ofText(HEAD));
        assertTrue(text.tokensOnly());
        assertArrayEquals(HEAD, text.tokens(0, HEAD.length));
        assertNull("rows store their own index", text.positions(0, HEAD.length));
        assertEquals(2, text.cut(2));
    }

    @Test
    public void aMediaPromptKeysItsImageRowsAtTheGridsStoredPositions() {
        CacheRows keyed = CacheRows.of(imagePrompt(1f));
        assertFalse(keyed.tokensOnly());
        assertEquals(11, keyed.length());
        int[] tokens = keyed.tokens(0, keyed.length());
        for (int r = 3; r < 9; r++) {
            assertEquals(CacheRows.NO_TOKEN, tokens[r]);
        }
        assertArrayEquals("the grid's rows share base position 3, the tail continues at 3 + max(3, 2)",
                new int[] {0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7}, keyed.positions(0, 11));
        assertArrayEquals(new int[] {8, 9, 10}, keyed.withFed(new int[] {40, 41, 42}).positions(11, 14));
        assertArrayEquals(new int[] {40, 41}, keyed.withFed(new int[] {40, 41}).tokens(11, 13));
    }

    @Test
    public void rowsAgreeOnlyForTheSameItemAtTheSameIndex() {
        CacheRows a = CacheRows.of(imagePrompt(1f));
        assertEquals(11, CacheRows.commonLength(a, 0, CacheRows.of(imagePrompt(1f)), 0, 11));
        assertEquals("another image differs from its first row", 3,
                CacheRows.commonLength(a, 0, CacheRows.of(imagePrompt(2f)), 0, 11));
        assertEquals("a text-only prompt shares the text before the image", 3,
                CacheRows.commonLength(a, 0, CacheRows.ofTokens(new int[] {5, 6, 7, 8}), 0, 4));
        assertEquals("rows of one item at other indexes differ", 0, CacheRows.commonLength(a, 3, a, 4, 4));
        assertEquals("rows of one item at their own indexes agree", 1, CacheRows.commonLength(a, 3, a, 3, 1));
    }

    @Test
    public void aReuseNeverEndsInsideAMediaItem() {
        CacheRows keyed = CacheRows.of(imagePrompt(1f));
        assertEquals(3, keyed.cut(3));
        for (int r = 4; r < 9; r++) {
            assertEquals("a reuse ending at row " + r + " backs up to the image's first row", 3, keyed.cut(r));
        }
        assertEquals("the row after the image ends the item", 9, keyed.cut(9));
        assertEquals(10, keyed.cut(10));
    }

    @Test
    public void promptEmbeddingRowsKeyRowByRowAndSplitAnywhere() throws Exception {
        float[][] r = rows(5, 0.5f);
        CacheRows a = CacheRows.of(MediaPrompt.builder().addText(HEAD).addEmbeds(HEAD.length, r).build());
        float[][] changed = Arrays.copyOf(r, 5);
        changed[3] = rows(1, 9f)[0];
        CacheRows b = CacheRows.of(MediaPrompt.builder().addText(HEAD).addEmbeds(HEAD.length, changed).build());
        assertEquals("rows share up to the first differing row", HEAD.length + 3,
                CacheRows.commonLength(a, 0, b, 0, 8));
        for (int k = 0; k <= 8; k++) {
            assertEquals("a prompt-embedding run splits anywhere", k, a.cut(k));
        }
        assertArrayEquals("one position per row", new int[] {0, 1, 2, 3, 4, 5, 6, 7}, a.positions(0, 8));

        MessageDigest expected = MessageDigest.getInstance("SHA-256");
        // block rows [2, 6): the last text token, then embedding rows 0..2
        ByteBuffer bytes = ByteBuffer.allocate(3 * WIDTH * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 3; i++) {
            for (float v : r[i]) {
                bytes.putFloat(v);
            }
        }
        expected.update(bytes.array());
        assertEquals("the block's key digests its embedding rows' little-endian float32 bytes",
                Arrays.asList(BlockExtraKey.salt("s"), BlockExtraKey.promptEmbeds(MediaKey.hex(expected.digest()))),
                a.extraKeys(2, 6, PrefixKey.of("s", LoraPlan.DEFAULT)));
    }

    @Test
    public void aBlocksExtraKeysListItsItemsSaltAndEmbeddings() {
        MediaChunk image = MediaChunk.media(MediaRopePositions.image2d(HEAD.length, 3, 2), rows(6, 1f), true);
        CacheRows keyed = CacheRows.of(imagePrompt(1f));
        String hash = image.item().id();
        assertEquals(Collections.emptyList(), keyed.extraKeys(0, 3, null));
        assertEquals(Arrays.asList(BlockExtraKey.media(hash, 3), BlockExtraKey.salt("tenant")),
                keyed.extraKeys(0, 4, PrefixKey.of("tenant", LoraPlan.DEFAULT)));
        assertEquals("a block inside the image names its first row before the block",
                Collections.singletonList(BlockExtraKey.media(hash, -1)), keyed.extraKeys(4, 8, null));

        MediaPrompt twice = MediaPrompt.builder().addMedia(MediaRopePositions.normal(0, 2), rows(2, 1f), false)
                .addMedia(MediaRopePositions.normal(2, 2), rows(2, 1f), false).build();
        CacheRows both = CacheRows.of(twice);
        String first = twice.chunks().get(0).item().id();
        String second = twice.chunks().get(1).item().id();
        assertEquals("two items in one block, each at its offset",
                Arrays.asList(BlockExtraKey.media(first, -1), BlockExtraKey.media(second, 1)),
                both.extraKeys(1, 4, null));
    }

    @Test
    public void aSliceKeepsTheKeysOfItsRows() {
        CacheRows keyed = CacheRows.of(imagePrompt(1f));
        CacheRows slice = keyed.slice(2, 6);
        assertEquals(4, slice.length());
        assertEquals(4, CacheRows.commonLength(slice, 0, keyed, 2, 4));
        assertEquals(0, CacheRows.commonLength(slice, 0, keyed, 3, 4));
        assertEquals("the slice's rows are the source's rows", 4, CacheRows.commonLength(keyed, 2, slice, 0, 4));
    }
}
