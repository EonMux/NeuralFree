package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The prompt-row capture of the batch engine's prefill (vLLM {@code prompt_logprobs},
 * {@code prompt_logprob_start}, {@code prompt_logprob_token_ids}): the first prompt token has no entry (no row
 * predicts it), every other entry leads with the prompt token at its exact vocabulary rank and carries the
 * log-softmax of the raw logits — the prompt token merged with the top of the requested count, or every token
 * under {@link GenerationRequest#PROMPT_LOGPROBS_ALL}.
 */
public class PromptLogprobsTest {

    /** A four-token vocabulary row: log-softmax is ln(4/7), ln(2/7), ln(1/7), -inf. */
    private static float[] row() {
        return new float[]{(float) Math.log(4), (float) Math.log(2), 0f, Float.NEGATIVE_INFINITY};
    }

    @Test
    public void theFirstPromptTokenHasNoEntryAndLaterEntriesLeadWithIt() {
        float[][] rows = {row(), row(), row()};
        int[] prompt = {10, 1, 2};
        List<TokenScore[]> captures = PromptLogprobs.capture(rows, prompt, 1);
        assertEquals(3, captures.size());
        assertNull("no row predicts the first prompt token", captures.get(0));
        assertEquals("row 0 predicts prompt[1]", 1, captures.get(1)[0].tokenId());
        assertEquals("row 1 predicts prompt[2]", 2, captures.get(2)[0].tokenId());
        assertEquals("count 1 means the prompt token alone", 1, captures.get(1).length);
    }

    @Test
    public void entriesCarryExactLogSoftmaxValuesRanksAndAlternatives() {
        float[] r = row();
        // prompt_logprobs 2 (count 3) over a prompt token ranked 3rd: the token plus the two best others
        List<TokenScore[]> captures = PromptLogprobs.capture(new float[][]{r, r}, new int[]{10, 2}, 3);
        assertEquals(2, captures.size());
        TokenScore[] entry = captures.get(1);
        assertEquals(3, entry.length);
        assertEquals("the prompt token leads", 2, entry[0].tokenId());
        assertEquals((float) Math.log(1.0 / 7), entry[0].logprob(), 1e-6);
        assertEquals("its exact rank among the vocabulary", 3, entry[0].rank());
        assertEquals("the best alternative first", 0, entry[1].tokenId());
        assertEquals((float) Math.log(4.0 / 7), entry[1].logprob(), 1e-6);
        assertEquals(1, entry[1].rank());
        assertEquals("the second best alternative next", 1, entry[2].tokenId());
        assertEquals((float) Math.log(2.0 / 7), entry[2].logprob(), 1e-6);
        assertEquals(2, entry[2].rank());
    }

    /**
     * vLLM's dict merges the prompt token into the top-n it gathers: a prompt token inside the top n adds no
     * extra alternative, so {@code prompt_logprobs: 2} over a 2nd-ranked token answers two entries.
     */
    @Test
    public void aPromptTokenInsideTheTopMergesWithIt() {
        float[] r = row();
        TokenScore[] entry = PromptLogprobs.capture(new float[][]{r, r}, new int[]{10, 1}, 3).get(1);
        assertEquals(2, entry.length);
        assertEquals(1, entry[0].tokenId());
        assertEquals(2, entry[0].rank());
        assertEquals(0, entry[1].tokenId());
    }

    /**
     * {@code prompt_logprob_start}: rows before it are not scored, so every position up to and including it is
     * null; {@code prompt_logprob_token_ids} score their ids per scored row (padding at -inf); a logits mode reports
     * the raw logits.
     */
    @Test
    public void theStartRowAndTokenIdRowsScoreOnlyTheRowsFromTheStart() {
        float[] r = row();
        int[] prompt = {10, 0, 1, 2};
        int[][] ids = {{2, 0}, {1, GenerationRequest.PROMPT_TOKEN_ID_PAD}};
        PromptLogprobs.Collector c = new PromptLogprobs.Collector(prompt, 1, 1, ids, false);
        for (int row = 0; row < prompt.length; row++) {
            c.captureRow(row, r);
        }
        assertTrue(c.complete());
        assertNull(c.result().get(0));
        assertNull("position 1 is predicted by row 0, before the start", c.result().get(1));
        assertEquals(1, c.result().get(2)[0].tokenId());
        assertEquals(2, c.result().get(3)[0].tokenId());
        float[][] scores = c.tokenIdScores();
        assertEquals(2, scores.length);
        assertEquals((float) Math.log(1.0 / 7), scores[0][0], 1e-6);
        assertEquals((float) Math.log(4.0 / 7), scores[0][1], 1e-6);
        assertEquals((float) Math.log(2.0 / 7), scores[1][0], 1e-6);
        assertEquals("padding scores -inf", Float.NEGATIVE_INFINITY, scores[1][1], 0f);

        PromptLogprobs.Collector logits = new PromptLogprobs.Collector(prompt, 0, 2, new int[][]{{0}}, true);
        assertFalse("no scored row yet", logits.complete());
        logits.captureRow(2, r);
        assertTrue(logits.complete());
        assertFalse("token-id rows alone ask for no prompt_logprobs entries", logits.hasEntries());
        assertEquals("a logits mode reports the raw logit", r[0], logits.tokenIdScores()[0][0], 0f);
    }

    @Test
    public void allTokensCaptureOrdersTheWholeVocabularyBestFirst() {
        List<TokenScore[]> captures = PromptLogprobs.capture(new float[][]{row(), row()}, new int[]{10, 2},
                GenerationRequest.PROMPT_LOGPROBS_ALL);
        TokenScore[] entry = captures.get(1);
        assertEquals(4, entry.length);
        assertEquals("the prompt token leads", 2, entry[0].tokenId());
        assertEquals(3, entry[0].rank());
        int[] rest = {0, 1, 3};
        int[] ranks = {1, 2, 4};
        for (int i = 1; i < entry.length; i++) {
            assertEquals("the rest best first", rest[i - 1], entry[i].tokenId());
            assertEquals(ranks[i - 1], entry[i].rank());
        }
    }
}
