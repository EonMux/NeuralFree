package com.eonmux.neuralfree.gguf.engine;

/**
 * Thrown when a forward pass or a prompt needs more positions than the model memory holds. A generation loop that
 * runs out of context mid-generation reports {@link StopReason#CONTEXT_FULL} instead, unless the prompt alone does
 * not fit.
 */
public class ContextFullException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    private final int requiredPositions;
    private final int contextLength;

    /**
     * @param requiredPositions positions the operation needs (for a prompt: its length)
     * @param contextLength     positions the memory holds
     */
    public ContextFullException(int requiredPositions, int contextLength) {
        super("context full: " + requiredPositions + " positions are needed but the context holds " + contextLength
                + "; raise -Dneuralfree.ctx, shorten the input, or enable -Dneuralfree.context.shift=true"
                + " for models whose memory supports shifting");
        this.requiredPositions = requiredPositions;
        this.contextLength = contextLength;
    }

    /** Positions the failed operation needed. */
    public int requiredPositions() {
        return requiredPositions;
    }

    /** Positions the memory holds. */
    public int contextLength() {
        return contextLength;
    }
}
