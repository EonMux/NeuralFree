package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.LookaheadRun;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;

/**
 * One scheduled piece of a {@code BatchEngine} step: the prefill piece of a waiting sequence (a slice of a text
 * prompt's rows, a whole chunk of a media prompt, or the fed-token tail a recompute replays), or the decode round
 * of a running one (its armed token plus drafts, one row plain), with its rows' tokens, first position and output
 * role (see {@code StepScheduler#step}). A media chunk piece carries its chunk rows and runs its own forward batch
 * ({@code PrefillChunks.chunkBatch}); the other pieces share the step's token batch.
 */
final class Piece {
    final Slot slot;
    final boolean decode;
    /** Rows the piece writes: the chunk's, the replayed tail's, or a decode round's ({@code 1 + drafted}). */
    final int rows;
    /** The first position the rows write. */
    final int from;
    final int[] tokens;
    /** Whether the piece's completing prefill chunk produces logits (a decode round's every row does). */
    final boolean output;
    /** Whether the piece completes the pending prefill or recompute. */
    final boolean finalChunk;
    /**
     * The decode round's tree proposal (the verify budget's cut, its row 0 the armed token), or null for a chain
     * round or a plain decode: its rows run under the per-row ancestor mask at their depth rope positions, and the
     * router walks the tree over them ({@code DecodeRounds#verifyTree}).
     */
    final TreeProposal tree;
    /** The media chunk the piece runs; null for a text or decode piece (they share the token batch). */
    final MediaChunk chunk;
    /**
     * The lookahead (Jacobi) step the piece runs as its own forward batch, or null for every other piece: a
     * lookahead step writes the rows of its whole sequence group and emits several tokens through its own run.
     */
    final LookaheadRun.Step lookahead;
    /**
     * The piece's first row in the sequence's KV-row space: the prompt row of a chunk or text slice (a media
     * chunk's tokens come from the chunk), the replayed row of a recompute tail, or the decode row cursor of a
     * round. The per-row adapter slots compare against it.
     */
    final int rowBase;
    /** The media chunk's first position in the prompt, its rows' write position. */
    final int posBase;
    /** The piece's index in the step's output order; -1 when it produces no logits. Set by {@code StepExecutor}. */
    int outIdx = -1;
    /**
     * The prompt-row index of the piece's first row when its rows are prompt rows (a fresh prompt's text slice, a
     * media chunk at its prompt place, a recompute's replayed prompt rows), or -1 (a decode round, the fed-token
     * tail a recompute replays). Set where the piece is scheduled.
     */
    int promptRowBase = -1;
    /**
     * Whether every row's logits read back for the submission's prompt logprobs ({@code prompt_logprobs}).
     * Set where the piece is scheduled.
     */
    boolean readRows;
    /**
     * Whether every row's final hidden states read back for a pooling request's pooling run (the rows land in the
     * run instead of the sampler). Set where the piece is scheduled; a pooling prompt's rows never sample.
     */
    boolean wantsHidden;

    Piece(Slot slot, boolean decode, int rows, int from, int firstRow, int[] tokens, boolean output,
            boolean finalChunk) {
        this(slot, decode, rows, from, tokens, output, finalChunk, null, firstRow, from, null);
    }

    /** A tree verify round's piece: {@code tree} carries the proposal its walk follows. */
    Piece(Slot slot, boolean decode, int rows, int from, int firstRow, int[] tokens, boolean output,
          boolean finalChunk, TreeProposal tree) {
        this(slot, decode, rows, from, tokens, output, finalChunk, null, firstRow, from, tree);
    }

    Piece(Slot slot, boolean decode, int rows, int from, int[] tokens, boolean output, boolean finalChunk,
          MediaChunk chunk, int rowBase, int posBase) {
        this(slot, decode, rows, from, tokens, output, finalChunk, chunk, rowBase, posBase, null);
    }

    private Piece(Slot slot, boolean decode, int rows, int from, int[] tokens, boolean output, boolean finalChunk,
                  MediaChunk chunk, int rowBase, int posBase, TreeProposal tree) {
        this.slot = slot;
        this.decode = decode;
        this.rows = rows;
        this.from = from;
        this.tokens = tokens;
        this.output = output;
        this.finalChunk = finalChunk;
        this.chunk = chunk;
        this.tree = tree;
        this.lookahead = null;
        this.rowBase = rowBase;
        this.posBase = posBase;
    }

    /**
     * A chunk piece of a media prompt at its prompt place: a whole chunk, or the rest of a text or prompt-embedding
     * chunk past a reused prefix.
     */
    static Piece mediaChunk(Slot slot, MediaChunk chunk, int rowBase, int posBase, boolean output,
                            boolean finalChunk) {
        return new Piece(slot, false, chunk.nTokens(), posBase, null, output, finalChunk, chunk, rowBase,
                posBase);
    }

    /**
     * The lookahead step of {@code slot}: its own forward batch over the slot's sequence group; the executor walks
     * its rows through the slot's run, which emits the step's tokens.
     */
    static Piece lookahead(Slot slot, LookaheadRun.Step step) {
        return new Piece(slot, step);
    }

    private Piece(Slot slot, LookaheadRun.Step step) {
        this.slot = slot;
        this.decode = true;
        this.rows = step.rows();
        this.from = slot.nextPos;
        this.tokens = null;
        this.output = false;
        this.finalChunk = false;
        this.chunk = null;
        this.tree = null;
        this.lookahead = step;
        this.rowBase = slot.nextRow;
        this.posBase = slot.nextPos;
    }

    /** True when the piece runs a media chunk (its own forward batch). */
    boolean isMediaChunk() {
        return chunk != null;
    }

    /** True when the piece runs a lookahead step (its own forward batch). */
    boolean isLookahead() {
        return lookahead != null;
    }
}
