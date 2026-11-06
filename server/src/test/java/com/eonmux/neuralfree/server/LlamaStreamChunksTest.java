package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.ReasoningFilter.Channel;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/** The native stream's chunks: one per generated token, its pieces merged in order. */
class LlamaStreamChunksTest {

    private static GenerationBatch.Text piece(int choice, Channel channel, String text, int token, int predicted,
                                              TokenScore[] logprobs) {
        return new GenerationBatch.Text(choice, channel, text, logprobs, token, predicted);
    }

    @Test
    void piecesOfOneTokenMergeAndTheNextTokenReleasesTheChunk() {
        LlamaStreamChunks chunks = new LlamaStreamChunks(2);
        TokenScore[] scores = {new TokenScore(7, -0.5f)};
        assertNull(chunks.offer(piece(0, Channel.REASONING, "think", 7, 1, scores)));
        assertNull(chunks.offer(piece(0, Channel.CONTENT, "ing", 7, 1, null)));
        assertNull(chunks.offer(piece(1, Channel.CONTENT, "other", 9, 1, null)), "choices do not mix");
        GenerationBatch.Text first = chunks.offer(piece(0, Channel.CONTENT, "x", 8, 2, null));
        assertEquals("thinking", first.text());
        assertEquals(7, first.tokenId());
        assertEquals(1, first.predicted());
        assertArrayEquals(scores, first.logprobs());
    }

    @Test
    void textReleasedAtTheEndJoinsTheLastTokenAndFlushEmptiesTheChoice() {
        LlamaStreamChunks chunks = new LlamaStreamChunks(1);
        chunks.offer(piece(0, Channel.CONTENT, "a", 5, 3, null));
        assertNull(chunks.offer(piece(0, Channel.CONTENT, "b", -1, 3, null)));
        GenerationBatch.Text last = chunks.flush(0);
        assertEquals("ab", last.text());
        assertEquals(5, last.tokenId());
        assertNull(chunks.flush(0));
    }

    @Test
    void theEndOfGenerationCaptureDoesNotTakeOverTheLastTokensChunk() {
        LlamaStreamChunks chunks = new LlamaStreamChunks(1);
        TokenScore[] hi = {new TokenScore(5, -0.1f)};
        TokenScore[] eos = {new TokenScore(2, -0.2f)};
        assertNull(chunks.offer(piece(0, Channel.CONTENT, "Hi", 5, 1, hi)));
        GenerationBatch.Text last = chunks.offer(piece(0, Channel.CONTENT, "", 2, 1, eos));
        assertEquals(5, last.tokenId(), "the real token keeps its chunk");
        assertArrayEquals(hi, last.logprobs());
        assertEquals(2, chunks.flush(0).tokenId(), "the end-of-generation capture is its own chunk");
    }

    @Test
    void aTailReleasedAtTheEndJoinsTheEndOfGenerationChunk() {
        LlamaStreamChunks chunks = new LlamaStreamChunks(1);
        TokenScore[] eos = {new TokenScore(2, -0.2f)};
        chunks.offer(piece(0, Channel.CONTENT, "foo", 5, 1, null));
        assertEquals("foo", chunks.offer(piece(0, Channel.CONTENT, "", 2, 1, eos)).text());
        // the stop filter's held tail comes out at the end, attributed to the last generated token
        assertNull(chunks.offer(piece(0, Channel.CONTENT, "<", 5, 1, null)), "no chunk after the end");
        GenerationBatch.Text last = chunks.flush(0);
        assertEquals("<", last.text());
        assertArrayEquals(eos, last.logprobs());
    }
}
