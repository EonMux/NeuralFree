package com.eonmux.neuralfree.gguf.engine.batch;

/**
 * The answer of a slot save or restore ({@link BatchEngine#saveSlot}, {@link BatchEngine#restoreSlot}): the tokens
 * the slot file carries (llama.cpp's {@code n_saved} / {@code n_restored}) and its size in bytes ({@code n_written}
 * / {@code n_read}). Immutable.
 */
public final class SlotResult {

    private final int tokens;
    private final long bytes;

    /** @throws IllegalArgumentException for a negative count */
    public SlotResult(int tokens, long bytes) {
        if (tokens < 0 || bytes < 0) {
            throw new IllegalArgumentException("a slot result of " + tokens + " tokens and " + bytes + " bytes");
        }
        this.tokens = tokens;
        this.bytes = bytes;
    }

    /** Tokens the slot file carries. */
    public int tokens() {
        return tokens;
    }

    /** Bytes of the slot file. */
    public long bytes() {
        return bytes;
    }

    @Override
    public String toString() {
        return "SlotResult{tokens=" + tokens + ", bytes=" + bytes + '}';
    }
}
