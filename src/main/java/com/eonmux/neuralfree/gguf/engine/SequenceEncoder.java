package com.eonmux.neuralfree.gguf.engine;

/**
 * The per-sequence encoder of an encoder-decoder {@link ForwardBackend} whose {@link ForwardBackend#forwardBatch}
 * runs several decoder sequences at once ({@link ForwardBackend#sequenceEncoder()}): each sequence attends to the
 * encoding of its own source, as llama.cpp tags every encoder row with its sequence ({@code cross.seq_ids_enc},
 * llama-context.cpp:1776-1785) and masks each decoder row to its sequence's encoder rows
 * (llama-graph.cpp:1093-1105). A caller encodes a sequence's source with {@link #encodeSequence}, then runs the
 * sequence's decoder rows from the {@link #decoderStartToken()} at position 0 through batches.
 */
public interface SequenceEncoder {

    /** The token every decoder sequence starts from at position 0 (llama.cpp completion.cpp:535-538). */
    int decoderStartToken();

    /** Longest source {@link #encodeSequence} accepts: the encoder runs a source in one ubatch. */
    int maxSourceTokens();

    /**
     * Encodes {@code source} into sequence {@code seq}'s cross-attention caches, replacing the sequence's previous
     * encoding; the decoder memory and the other sequences' encodings stay. The encoding lasts until the next
     * encoding of the sequence or a reset of the backend.
     *
     * @throws ContextFullException      when the source is longer than {@link #maxSourceTokens()}; nothing changes
     * @throws IllegalArgumentException  for an empty source or a token outside the vocabulary
     * @throws IndexOutOfBoundsException when {@code seq} is outside the backend's sequences
     */
    void encodeSequence(int seq, int[] source);
}
