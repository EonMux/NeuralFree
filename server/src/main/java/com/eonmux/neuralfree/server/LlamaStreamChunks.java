package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;

/**
 * The per-token chunks of llama.cpp's native stream: the pieces one generated token releases (a reasoning tag
 * splits its text in two, and text held back for a stop string comes out with a later token) merge into one chunk
 * carrying that token, its text in order and the first logprobs capture among them. A piece of another token, the
 * next one or the end-of-generation token's bare capture, makes the held chunk ready, as does the end of its
 * choice. Not thread-safe: one stream, one thread.
 */
final class LlamaStreamChunks {

    private final GenerationBatch.Text[] pending;

    LlamaStreamChunks(int choices) {
        this.pending = new GenerationBatch.Text[choices];
    }

    /** Adds {@code piece}; returns the chunk of the token before it when {@code piece} starts a new one, else null. */
    GenerationBatch.Text offer(GenerationBatch.Text piece) {
        int i = piece.choice();
        GenerationBatch.Text held = pending[i];
        if (held == null) {
            pending[i] = piece;
            return null;
        }
        // text the end of generation released (no capture of its own) joins the end-of-generation token's chunk
        boolean released = held.text().isEmpty() && piece.logprobs() == null;
        boolean sameToken = piece.tokenId() < 0 || piece.tokenId() == held.tokenId() || released;
        if (held.predicted() != piece.predicted() || !sameToken) {
            pending[i] = piece;
            return held;
        }
        pending[i] = new GenerationBatch.Text(i, ReasoningFilter.Channel.CONTENT, held.text() + piece.text(),
                held.logprobs() != null ? held.logprobs() : piece.logprobs(), held.tokenId(), held.predicted());
        return null;
    }

    /** The last chunk of {@code choice}, or null when none is held. */
    GenerationBatch.Text flush(int choice) {
        GenerationBatch.Text held = pending[choice];
        pending[choice] = null;
        return held;
    }
}
