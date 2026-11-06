package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The prefill cursor of the batch engine ({@link PrefillChunks}): a text prompt slices its pending tail after the
 * reused prefix, a recompute slices the replayed history, a media prompt schedules its remaining whole chunks at
 * their prompt rows and positions (an image grid takes fewer positions than rows), a media recompute replays its
 * chunks then the fed decode tokens past the prompt's positions, a reused prefix resumes inside a text or
 * prompt-embedding chunk but never inside a media item (exactly where the prefix cache may cut it), media items
 * schedule within the step's encoder budget while text and prompt-embedding rows take none, and a media chunk's
 * forward batch carries the chunk's positions, mask rule and output rows.
 */
public class PrefillChunksTest {

    private static final int[] HEAD = {5, 6, 7, 8};
    private static final int[] TAIL = {9, 10, 11};
    /** A 3 x 2 image grid: six rows over max(3, 2) = 3 positions. */
    private static final int GRID_X = 3;
    private static final int GRID_Y = 2;
    private static final int WIDTH = 8;

    private static float[][] rows(int n) {
        float[][] r = new float[n][WIDTH];
        for (int i = 0; i < n; i++) {
            Arrays.fill(r[i], 0.1f * (i + 1));
        }
        return r;
    }

    /** Text, a non-causal image grid, text: 13 rows over 10 positions. */
    private static MediaPrompt mediaPrompt() {
        return MediaPrompt.builder()
                .addText(HEAD)
                .addMedia(MediaRopePositions.image2d(HEAD.length, GRID_X, GRID_Y), rows(GRID_X * GRID_Y), true)
                .addText(TAIL)
                .build();
    }

    private static Slot slot(MediaPrompt prompt, GenerationRequest request) {
        Submission sub = new Submission(new BatchSequence(0), prompt, request, (channel, id, text) -> { },
                null, null, null, java.util.List.of());
        return new Slot(0, sub, null, 64);
    }

    private static Slot slot(MediaPrompt prompt) {
        return slot(prompt, GenerationRequest.builder().build());
    }

    /**
     * A slot whose claim reused {@code reused} prompt rows before the slot opened, as {@code PrefixClaims.claim}
     * settles the reuse before {@code SlotOpener.open}: the position cursor starts at the reused prefix's end.
     */
    private static Slot slot(MediaPrompt prompt, int reused) {
        Submission sub = new Submission(new BatchSequence(0), prompt, GenerationRequest.builder().build(),
                (channel, id, text) -> { }, null, null, null, java.util.List.of());
        sub.reused = reused;
        return new Slot(0, sub, null, 64);
    }

    private static final int BUDGET = 64; // a step's encoder budget no test chunk exceeds unless it says so

    @Test
    public void aFreshMediaPromptSchedulesEveryChunkAtItsPromptRowAndPosition() {
        MediaPrompt prompt = mediaPrompt();
        assertTrue("the grid takes fewer positions than rows", prompt.nPositions() < prompt.nTokens());
        List<Piece> pieces = new ArrayList<>();
        PrefillChunks.chunkPieces(slot(prompt), pieces, BUDGET, BUDGET);

        assertEquals(3, pieces.size());
        assertChunk(pieces.get(0), HEAD.length, 0, 0, false);
        assertChunk(pieces.get(1), GRID_X * GRID_Y, HEAD.length, HEAD.length, false);
        Piece last = pieces.get(2);
        assertChunk(last, TAIL.length, HEAD.length + GRID_X * GRID_Y, HEAD.length + Math.max(GRID_X, GRID_Y), true);
        assertTrue("the completing chunk of a fresh prompt asks logits", last.output);
        assertFalse(pieces.get(0).output || pieces.get(1).output);
    }

    @Test
    public void aMediaPrefillResumesAtTheChunkItsCursorReached() {
        Slot slot = slot(mediaPrompt());
        slot.prefillRows = HEAD.length;
        List<Piece> pieces = new ArrayList<>();
        PrefillChunks.chunkPieces(slot, pieces, BUDGET, BUDGET);

        assertEquals(2, pieces.size());
        assertTrue(pieces.get(0).chunk.nonCausal());
        assertChunk(pieces.get(0), GRID_X * GRID_Y, HEAD.length, HEAD.length, false);
        assertChunk(pieces.get(1), TAIL.length, HEAD.length + GRID_X * GRID_Y, 7, true);
    }

    @Test
    public void aReusedPrefixResumesInsideATextOrEmbeddingChunkButNeverInsideAMediaItem() {
        Slot slot = slot(mediaPrompt());
        slot.submission.reused = HEAD.length + GRID_X * GRID_Y + 1;
        List<Piece> pieces = new ArrayList<>();
        PrefillChunks.chunkPieces(slot, pieces, BUDGET, BUDGET);
        assertEquals(1, pieces.size());
        Piece tail = pieces.get(0);
        assertArrayEquals("the tail chunk runs from its first unreused token", Arrays.copyOfRange(TAIL, 1, 3),
                tail.chunk.tokens());
        assertEquals(HEAD.length + GRID_X * GRID_Y + 1, tail.rowBase);
        assertEquals(tail.rowBase, tail.promptRowBase);
        assertEquals("the tail's second token sits at 4 + 3 + 1", 8, tail.posBase);
        assertTrue(tail.finalChunk);
        assertTrue(tail.output);

        MediaPrompt embeds = MediaPrompt.builder().addText(HEAD).addEmbeds(HEAD.length, rows(5)).build();
        Slot rows = slot(embeds);
        rows.submission.reused = HEAD.length + 2;
        List<Piece> rest = new ArrayList<>();
        PrefillChunks.chunkPieces(rows, rest, BUDGET, BUDGET);
        assertEquals(1, rest.size());
        assertTrue(rest.get(0).chunk.isPromptEmbeds());
        assertEquals(3, rest.get(0).rows);
        assertEquals(HEAD.length + 2, rest.get(0).posBase);
        assertArrayEquals(Arrays.copyOfRange(rows(5), 2, 5), rest.get(0).chunk.embeddings());
        assertEquals(HEAD.length + 2, PrefillChunks.chunkBatch(rest.get(0)).positions()[0]);

        Slot inside = slot(mediaPrompt());
        inside.submission.reused = HEAD.length + 2;
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> PrefillChunks.chunkPieces(inside, new ArrayList<>(), BUDGET, BUDGET));
        assertTrue(e.getMessage(), e.getMessage().contains("media item"));
    }

    @Test
    public void aCheckpointStopEndsTheWalkCuttingTextAndRunningWholeItemsPastIt() {
        // a stop inside the prompt-embedding run cuts the run at it and ends the walk there
        MediaPrompt embeds = MediaPrompt.builder().addText(HEAD).addEmbeds(HEAD.length, rows(5)).addText(TAIL)
                .build();
        Slot embedsSlot = slot(embeds);
        List<Piece> pieces = new ArrayList<>();
        PrefillChunks.chunkPieces(embedsSlot, pieces, BUDGET, BUDGET, 6);
        assertEquals(2, pieces.size());
        assertEquals(4, pieces.get(0).rows);
        assertEquals(2, pieces.get(1).rows);
        assertTrue(pieces.get(1).chunk.isPromptEmbeds());
        assertEquals(6, embedsSlot.pendingCheckpointRow);

        // a stop inside a media item runs the item whole and lands the checkpoint at its end
        Slot item = slot(mediaPrompt());
        List<Piece> whole = new ArrayList<>();
        PrefillChunks.chunkPieces(item, whole, BUDGET, BUDGET, 7);
        assertEquals(2, whole.size());
        assertEquals(GRID_X * GRID_Y, whole.get(1).rows);
        assertTrue(whole.get(1).chunk.isMediaItem());
        assertEquals(HEAD.length + GRID_X * GRID_Y, item.pendingCheckpointRow);
    }

    @Test
    public void aMediaRecomputeReplaysTheChunksThenTheFedTokensPastThePromptPositions() {
        MediaPrompt prompt = mediaPrompt();
        Slot slot = slot(prompt);
        slot.recompute = true;
        slot.fed(40);
        slot.fed(41);
        slot.nextRow = prompt.nTokens() + 2;
        List<Piece> pieces = new ArrayList<>();
        PrefillChunks.chunkPieces(slot, pieces, BUDGET, BUDGET);

        assertEquals(4, pieces.size());
        for (int i = 0; i < 3; i++) {
            assertFalse("a recompute's chunks sample nothing", pieces.get(i).output);
            assertFalse("the replay continues past the prompt", pieces.get(i).finalChunk);
        }
        Piece tail = pieces.get(3);
        assertFalse(tail.isMediaChunk());
        assertFalse(tail.output);
        assertTrue(tail.finalChunk);
        assertEquals(prompt.nPositions(), tail.from);
        assertArrayEquals(new int[] {40, 41}, tail.tokens);
        assertEquals(-1, tail.promptRowBase);

        slot.prefillRows = prompt.nTokens() + 1;
        List<Piece> rest = new ArrayList<>();
        PrefillChunks.chunkPieces(slot, rest, BUDGET, BUDGET);
        assertEquals(1, rest.size());
        assertEquals(prompt.nPositions() + 1, rest.get(0).from);
        assertArrayEquals(new int[] {41}, rest.get(0).tokens);
    }

    @Test
    public void aTextChunkSlicesThePromptTailAfterTheReusedPrefix() {
        int[] tokens = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
        Slot slot = slot(MediaPrompt.ofText(tokens), 3);
        PrefillChunks.chunkWritten(slot, PrefillChunks.chunkPiece(slot, 2)); // the step wrote a 2-row chunk

        Piece middle = PrefillChunks.chunkPiece(slot, 4);
        assertEquals(5, middle.from);
        assertEquals(5, middle.promptRowBase);
        assertArrayEquals(Arrays.copyOfRange(tokens, 5, 9), middle.tokens);
        assertFalse(middle.finalChunk);
        assertFalse(middle.output);
        assertFalse(middle.isMediaChunk());

        Piece last = PrefillChunks.chunkPiece(slot, 5);
        assertArrayEquals(Arrays.copyOfRange(tokens, 5, 10), last.tokens);
        assertTrue(last.finalChunk);
        assertTrue(last.output);
    }

    @Test
    public void aTextRecomputeSlicesThePromptAndTheFedTokens() {
        int[] tokens = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
        Slot slot = slot(MediaPrompt.ofText(tokens));
        slot.recompute = true;
        slot.fed(30);
        slot.fed(31);
        slot.nextRow = tokens.length + 2;
        PrefillChunks.chunkWritten(slot, PrefillChunks.chunkPiece(slot, 4)); // the replay wrote two chunks
        PrefillChunks.chunkWritten(slot, PrefillChunks.chunkPiece(slot, 4));

        Piece piece = PrefillChunks.chunkPiece(slot, 4);
        assertEquals(8, piece.from);
        assertArrayEquals(new int[] {12, 13, 30, 31}, piece.tokens);
        assertTrue(piece.finalChunk);
        assertFalse("a recompute keeps its armed token: no logits", piece.output);
    }

    @Test
    public void aWrittenChunkAdvancesTheCursorAndTheFinalOneEndsTheRecompute() {
        int[] tokens = {4, 5, 6, 7, 8, 9};
        Slot slot = slot(MediaPrompt.ofText(tokens));
        slot.recompute = true;
        slot.nextRow = tokens.length;

        PrefillChunks.chunkWritten(slot, PrefillChunks.chunkPiece(slot, 4));
        assertEquals(4, slot.prefillRows);
        assertTrue(slot.needsPrefill);
        assertTrue(slot.recompute);

        PrefillChunks.chunkWritten(slot, PrefillChunks.chunkPiece(slot, 2));
        assertEquals(6, slot.prefillRows);
        assertFalse(slot.needsPrefill);
        assertFalse(slot.recompute);
    }

    @Test
    public void aMediaChunkBatchCarriesItsPositionsMaskAndOutputRows() {
        MediaPrompt prompt = mediaPrompt();
        List<Piece> pieces = new ArrayList<>();
        PrefillChunks.chunkPieces(slot(prompt), pieces, BUDGET, BUDGET);
        Piece image = pieces.get(1);
        MediaRopePositions positions = image.chunk.positions();

        ForwardBatch batch = PrefillChunks.chunkBatch(image);
        assertEquals(GRID_X * GRID_Y, batch.nTokens());
        assertTrue(batch.nonCausal());
        assertNotNull("an image grid carries its rope sections", batch.ropePositions());
        assertNotNull("an image chunk runs embedding rows", batch.embeddings());
        for (int i = 0; i < batch.nTokens(); i++) {
            assertEquals(positions.position(0, i), batch.positions()[i]);
            assertFalse("a non-completing chunk asks no logits", batch.output(i));
        }

        ForwardBatch text = PrefillChunks.chunkBatch(pieces.get(2));
        assertFalse(text.nonCausal());
        assertArrayEquals(TAIL, text.tokens());
        for (int i = 0; i < TAIL.length; i++) {
            assertEquals(7 + i, text.positions()[i]);
            assertEquals("only the completing row asks logits", i == TAIL.length - 1, text.output(i));
        }
    }

    @Test
    public void aPromptLogprobsChunkAsksEveryRowsLogits() {
        GenerationRequest request = GenerationRequest.builder().promptLogprobs(2).build();
        List<Piece> pieces = new ArrayList<>();
        PrefillChunks.chunkPieces(slot(mediaPrompt(), request), pieces, BUDGET, BUDGET);
        for (Piece p : pieces) {
            assertTrue(p.readRows);
            ForwardBatch batch = PrefillChunks.chunkBatch(p);
            for (int i = 0; i < batch.nTokens(); i++) {
                assertTrue(batch.output(i));
            }
        }
    }

    @Test
    public void theEncoderBudgetSchedulesWholeChunksUntilItIsSpent() {
        // image, image, text: the two grids of six rows exceed a budget of 8 only together
        MediaPrompt prompt = MediaPrompt.builder()
                .addText(HEAD)
                .addMedia(MediaRopePositions.image2d(HEAD.length, GRID_X, GRID_Y), rows(GRID_X * GRID_Y), true)
                .addMedia(MediaRopePositions.image2d(HEAD.length + 3, GRID_X, GRID_Y), rows(GRID_X * GRID_Y), true)
                .addText(TAIL)
                .build();
        Slot slot = slot(prompt);

        List<Piece> first = new ArrayList<>();
        int spent = PrefillChunks.chunkPieces(slot, first, 8, 8);
        assertEquals("the head ran free and the first grid spent the budget", GRID_X * GRID_Y, spent);
        assertEquals(2, first.size());
        assertTrue(first.get(1).chunk.nonCausal());

        slot.prefillRows = HEAD.length + GRID_X * GRID_Y; // the step wrote the head and the first grid
        List<Piece> second = new ArrayList<>();
        spent = PrefillChunks.chunkPieces(slot, second, 8, 8);
        assertEquals("the second grid fits the fresh budget", GRID_X * GRID_Y, spent);
        assertEquals(2, second.size());
        assertChunk(second.get(1), TAIL.length, HEAD.length + 2 * GRID_X * GRID_Y, HEAD.length + 2 * 3, true);

        slot.prefillRows = prompt.nTokens();
        List<Piece> last = new ArrayList<>();
        assertEquals(0, PrefillChunks.chunkPieces(slot, last, 8, 8));
        assertTrue(last.isEmpty());
    }

    /** A head, a 16-row grid larger than an 8-row budget, and {@code tail}. */
    private static MediaPrompt oversizedPrompt(int[] tail) {
        return MediaPrompt.builder()
                .addText(HEAD)
                .addMedia(MediaRopePositions.image2d(HEAD.length, 4, 4), rows(16), true)
                .addText(tail)
                .build();
    }

    @Test
    public void aChunkLargerThanTheWholeBudgetSchedulesOnAnUntouchedBudget() {
        int[] tail = {21, 22};
        List<Piece> pieces = new ArrayList<>();
        int spent = PrefillChunks.chunkPieces(slot(oversizedPrompt(tail)), pieces, 8, 8);
        assertEquals("the oversized grid takes the whole step's budget", 16, spent);
        assertEquals("the head, the grid and the free trailing text", 3, pieces.size());
        assertFalse(pieces.get(1).chunk.isText());
        assertArrayEquals(tail, pieces.get(2).chunk.tokens());
        assertTrue(pieces.get(2).finalChunk);
    }

    @Test
    public void anOversizedChunkWaitsOnceTheStepSpentAnyMediaRows() {
        int[] tail = {21, 22};
        List<Piece> partly = new ArrayList<>();
        assertEquals("another prompt's media spent part of the budget: the grid waits", 0,
                PrefillChunks.chunkPieces(slot(oversizedPrompt(tail)), partly, 3, 8));
        assertEquals("only the head schedules", 1, partly.size());

        // another prompt's oversized chunk left the step's budget negative (8 - 16)
        List<Piece> spentOut = new ArrayList<>();
        assertEquals(0, PrefillChunks.chunkPieces(slot(oversizedPrompt(tail)), spentOut, -8, 8));
        assertEquals(1, spentOut.size());
        assertTrue(spentOut.get(0).chunk.isText());
    }

    @Test
    public void anUnschedulableChunkStopsTheWalkUntilTheBudgetRefills() {
        int[] tail = {31, 32, 33};
        MediaPrompt prompt = MediaPrompt.builder()
                .addText(HEAD)
                .addMedia(MediaRopePositions.image2d(HEAD.length, 3, 4), rows(12), true)
                .addText(tail)
                .build();
        Slot slot = slot(prompt);

        List<Piece> first = new ArrayList<>();
        int spent = PrefillChunks.chunkPieces(slot, first, 6, 12);
        assertEquals("the grid waits: nothing past the head schedules this step", 0, spent);
        assertEquals(1, first.size());

        slot.prefillRows = HEAD.length; // the first step wrote the head
        List<Piece> second = new ArrayList<>();
        spent = PrefillChunks.chunkPieces(slot, second, 12, 12);
        assertEquals(12, spent);
        assertEquals(2, second.size());
        assertChunk(second.get(1), tail.length, HEAD.length + 12, HEAD.length + 4, true);

        slot.prefillRows = HEAD.length + 12;
        List<Piece> after = new ArrayList<>();
        assertEquals("the budget only bounds media rows: the trailing text schedules", 0,
                PrefillChunks.chunkPieces(slot, after, 4, 12));
        assertEquals(1, after.size());
        assertTrue(after.get(0).chunk.isText());
        assertArrayEquals(tail, after.get(0).chunk.tokens());
    }

    /** Text, a five-row prompt-embedding run, a non-causal image grid, text. */
    private static MediaPrompt mixedPrompt() {
        int embedsAt = HEAD.length;
        int gridAt = embedsAt + 5;
        return MediaPrompt.builder()
                .addText(HEAD)
                .addEmbeds(embedsAt, rows(5))
                .addMedia(MediaRopePositions.image2d(gridAt, GRID_X, GRID_Y), rows(GRID_X * GRID_Y), true)
                .addText(TAIL)
                .build();
    }

    @Test
    public void thePrefillSplitsAPromptExactlyWhereThePrefixCacheMayCutIt() {
        MediaPrompt prompt = mixedPrompt();
        CacheRows keyed = CacheRows.of(prompt);
        for (int reuse = 0; reuse < prompt.nTokens(); reuse++) {
            Slot slot = slot(prompt);
            slot.submission.reused = reuse;
            boolean cacheMayEndHere = keyed.cut(reuse) == reuse;
            boolean prefillResumesHere;
            try {
                PrefillChunks.chunkPieces(slot, new ArrayList<>(), BUDGET, BUDGET);
                prefillResumesHere = true;
            } catch (IllegalStateException e) {
                prefillResumesHere = false;
            }
            assertEquals("row " + reuse, cacheMayEndHere, prefillResumesHere);
        }
    }

    @Test
    public void promptEmbeddingRowsTakeNoEncoderBudget() {
        MediaPrompt prompt = mixedPrompt();
        List<Piece> pieces = new ArrayList<>();
        int spent = PrefillChunks.chunkPieces(slot(prompt), pieces, 2, GRID_X * GRID_Y);
        assertEquals("the embedding run schedules free; the grid waits for a fresh budget", 0, spent);
        assertEquals("the head and the embedding run", 2, pieces.size());
        assertTrue(pieces.get(1).chunk.isPromptEmbeds());
        assertEquals(5, pieces.get(1).rows);
    }

    @Test
    public void aReusedPrefixAndTheEncoderBudgetMeetAtTheItemBoundary() {
        MediaPrompt prompt = mixedPrompt();
        int gridRow = HEAD.length + 5;
        Slot slot = slot(prompt);
        slot.submission.reused = gridRow; // the cache served the head and the embedding run

        List<Piece> spentOut = new ArrayList<>();
        assertEquals(0, PrefillChunks.chunkPieces(slot, spentOut, 0, GRID_X * GRID_Y));
        assertTrue("the grid is the first pending row and the budget is spent: nothing schedules",
                spentOut.isEmpty());

        List<Piece> fresh = new ArrayList<>();
        assertEquals(GRID_X * GRID_Y, PrefillChunks.chunkPieces(slot, fresh, GRID_X * GRID_Y, GRID_X * GRID_Y));
        assertEquals(2, fresh.size());
        assertTrue(fresh.get(0).chunk.isMediaItem());
        assertEquals("the grid runs at its prompt row", gridRow, fresh.get(0).rowBase);
        assertChunk(fresh.get(1), TAIL.length, gridRow + GRID_X * GRID_Y, gridRow + GRID_X, true);
    }

    private static void assertChunk(Piece p, int rows, int rowBase, int posBase, boolean finalChunk) {
        assertTrue(p.isMediaChunk());
        assertEquals("rows", rows, p.rows);
        assertEquals("row base", rowBase, p.rowBase);
        assertEquals("prompt row base", rowBase, p.promptRowBase);
        assertEquals("position base", posBase, p.posBase);
        assertEquals("first position", posBase, p.from);
        assertEquals("final chunk", finalChunk, p.finalChunk);
    }
}
