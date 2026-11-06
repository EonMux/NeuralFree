package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EmbedsPrompt}: the run shapes it accepts (an empty first or last token run, adjacent embedding runs),
 * the counts it reports, the media prompt it runs as on the batch engine, and every invalid input refused by name.
 * Constructors copy, so later caller edits leave the prompt untouched.
 */
public class EmbedsPromptTest {

    private static final float[][] ROWS = {{1, 2, 3}, {4, 5, 6}};

    @Test
    public void aRowsOnlyPromptHoldsTwoEmptyTokenRunsAndTheRows() {
        EmbedsPrompt prompt = EmbedsPrompt.ofRows(ROWS);
        assertEquals(2, prompt.segments());
        assertEquals(0, prompt.tokens(0).length);
        assertEquals(0, prompt.tokens(1).length);
        assertArrayEquals(ROWS, prompt.embeds(0));
        assertEquals(2, prompt.rowCount());
        assertEquals(3, prompt.rowWidth());
        assertTrue(prompt.rowsOnly());
        assertEquals(0, prompt.allTokens().length);
    }

    @Test
    public void aSplicedPromptCarriesTheRunsInOrder() {
        EmbedsPrompt prompt = EmbedsPrompt.of(new int[][] {{7, 8}, {}, {9}},
                new float[][][] {ROWS, new float[][] {{1, 1, 1}}});
        assertEquals(3, prompt.segments());
        assertArrayEquals(new int[] {7, 8}, prompt.tokens(0));
        assertArrayEquals(new int[] {}, prompt.tokens(1));
        assertArrayEquals(new int[] {9}, prompt.tokens(2));
        assertArrayEquals(ROWS, prompt.embeds(0));
        assertArrayEquals(new float[][] {{1, 1, 1}}, prompt.embeds(1));
        assertEquals(6, prompt.rowCount());
        assertFalse(prompt.rowsOnly());
        assertArrayEquals(new int[] {7, 8, 9}, prompt.allTokens());
    }

    @Test
    public void accessorsHandOutCopies() {
        int[] tokens = {5};
        float[][] rows = {{1, 2}};
        EmbedsPrompt prompt = EmbedsPrompt.of(new int[][] {tokens, new int[0]}, new float[][][] {rows});
        tokens[0] = 9;
        rows[0][0] = 9;
        assertEquals(5, prompt.tokens(0)[0]);
        assertEquals(1, prompt.embeds(0)[0][0], 0);
        int[] read = prompt.tokens(0);
        read[0] = 7;
        assertEquals(5, prompt.tokens(0)[0]);
    }

    @Test
    public void brokenShapesRefuseByName() {
        assertThrows(IllegalArgumentException.class,
                () -> EmbedsPrompt.of(new int[0][], new float[0][][]));
        assertThrows(IllegalArgumentException.class, () -> EmbedsPrompt.of(new int[2][], new float[2][][]));
        assertThrows(IllegalArgumentException.class, () -> EmbedsPrompt.ofRows(new float[0][]));
        assertThrows(IllegalArgumentException.class, () -> EmbedsPrompt.ofRows(new float[][] {{1}, {2, 3}}));
        assertThrows(IllegalArgumentException.class, () -> EmbedsPrompt.ofRows(new float[][] {{1, Float.NaN}}));
        assertThrows(NullPointerException.class, () -> EmbedsPrompt.ofRows(new float[][] {null}));
        assertThrows(IllegalArgumentException.class, () -> EmbedsPrompt.of(new int[][] {{-1}}, new float[][][] {ROWS}));
        assertThrows(NullPointerException.class, () -> EmbedsPrompt.of(null, new float[][][] {ROWS}));
    }

    @Test
    public void theLastEmbeddingRunIndexIsRefused() {
        EmbedsPrompt prompt = EmbedsPrompt.ofRows(ROWS);
        assertThrows(IndexOutOfBoundsException.class, () -> prompt.embeds(1));
        assertThrows(IndexOutOfBoundsException.class, () -> prompt.embeds(-1));
    }

    @Test
    public void theMediaPromptHoldsTheRunsAsTextAndCausalNormalChunks() {
        EmbedsPrompt prompt = EmbedsPrompt.of(new int[][] {{7, 8}, {}, {9}},
                new float[][][] {ROWS, new float[][] {{1, 1, 1}}});
        MediaPrompt media = prompt.toMediaPrompt();
        assertEquals("empty token runs leave no chunk; adjacent row runs stay two chunks", 4, media.chunks().size());
        assertEquals(prompt.rowCount(), media.nTokens());
        assertEquals("one position per row", prompt.rowCount(), media.nPositions());
        assertArrayEquals(prompt.allTokens(), media.samplerTokens());
        assertArrayEquals(new int[] {7, 8}, media.chunks().get(0).tokens());
        MediaChunk first = media.chunks().get(1);
        assertFalse(first.isText());
        assertFalse("prompt rows are causal", first.nonCausal());
        assertTrue("the rows are a prompt-embeddings run, not a media item", first.isPromptEmbeds());
        assertEquals(MediaRopePositions.Kind.NORMAL, first.positions().kind());
        assertEquals("the rows start where the tokens before them end", 2, first.positions().pos0());
        assertArrayEquals(ROWS, first.embeddings());
        assertEquals(4, media.chunks().get(2).positions().pos0());
        assertArrayEquals(new int[] {9}, media.chunks().get(3).tokens());
        assertEquals(3, media.mediaWidth());
    }

    @Test
    public void aRowsOnlyPromptIsOneMediaChunkAtPositionZero() {
        MediaPrompt media = EmbedsPrompt.ofRows(ROWS).toMediaPrompt();
        assertEquals(1, media.chunks().size());
        assertEquals(0, media.chunks().get(0).positions().pos0());
        assertEquals(0, media.samplerTokens().length);
        assertFalse(media.isTextOnly());
    }
}
