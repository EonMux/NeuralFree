package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.engine.ContextFullException;

/**
 * Thrown by {@code PagedSequences.prepare} when a step cannot get the pages it needs: the page pools hold no free
 * page for the requested tokens, or the requested range pushes a sequence past its context cap
 * ({@code neuralfree.kv.unified=false}). May only escape {@code prepare} before any state changed (all-or-nothing
 * reservation); a caller may shrink the step and {@code prepare} again.
 */
public class MemoryFullException extends ContextFullException {

    private static final long serialVersionUID = 1L;

    private final int nTokens;
    private final int freeCells;

    /**
     * @param requiredPositions positions the step needs (its token count, measured from the first position)
     * @param contextLength     positions the memory holds for the sequence
     * @param nTokens           tokens of the failed step
     * @param freeCells         empty rows the memory held before the step
     */
    public MemoryFullException(int requiredPositions, int contextLength, int nTokens, int freeCells) {
        super(requiredPositions, contextLength);
        this.nTokens = nTokens;
        this.freeCells = freeCells;
    }

    @Override
    public String getMessage() {
        return "memory full: a step of " + nTokens + " tokens needs " + requiredPositions()
                + " positions but only " + freeCells + " of " + contextLength() + " rows are free; lower "
                + "the step size, free other sequences (SequenceMemory.keep) or raise -Dneuralfree.kv.page.size";
    }

    /** Tokens the failed step carried. */
    public int nTokens() {
        return nTokens;
    }

    /** Empty rows the memory held before the step. */
    public int freeCells() {
        return freeCells;
    }
}
