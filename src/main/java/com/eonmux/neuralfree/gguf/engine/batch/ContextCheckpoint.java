package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.memory.CompressedStateCopy;
import com.eonmux.neuralfree.gguf.memory.RecurrentStateCopy;
import com.eonmux.neuralfree.gguf.memory.paged.PageWindow;

/**
 * One context checkpoint of the prefix cache (llama.cpp's {@code common_prompt_checkpoint}, vLLM's Mamba "align"
 * cached state): the part of a sequence's memory that cannot be rebuilt from cached KV blocks, taken when the
 * sequence had committed exactly {@link #rows} rows — the copy of its recurrent state and the pages of every
 * sliding-window group's window. It hangs on the cached block holding row {@code rows - 1}, so a later prompt whose
 * prefix walks that block chain through {@code rows} rows restores it and continues from there. Owned by the
 * engine thread; the cache releases its pages when it drops the checkpoint.
 */
final class ContextCheckpoint {

    /** Rows of the sequence the checkpoint restores. */
    final int rows;
    /** The recurrent state at {@link #rows}; {@link RecurrentStateCopy#EMPTY} without recurrent state. */
    final RecurrentStateCopy state;
    /** The copied DSV4 compressed streams at {@link #rows}, or null without compressed caches. */
    final CompressedStateCopy compressed;
    /** The window pages per KV group at {@link #rows}; null entries for full-attention groups. */
    private final PageWindow[] windows;
    /** The cached block the checkpoint hangs on. */
    PrefixCache.Block block;
    long lastUse;

    ContextCheckpoint(int rows, RecurrentStateCopy state, CompressedStateCopy compressed, PageWindow[] windows) {
        this.rows = rows;
        this.state = state;
        this.compressed = compressed;
        this.windows = windows.clone();
    }

    /** The copied DSV4 compressed streams at {@link #rows}, or null without compressed caches. */
    CompressedStateCopy compressed() {
        return compressed;
    }

    /**
     * What captured the checkpoint (the engine's {@code Submission}), or null for one no engine run took: the
     * eviction spares the checkpoints of the capture it makes room for, as llama.cpp's spares the current task's.
     */
    Object owner;

    /** The window pages per KV group, null for a full-attention group (a copy of the array). */
    PageWindow[] windows() {
        return windows.clone();
    }

    @Override
    public String toString() {
        return "ContextCheckpoint[" + rows + " rows, state " + state.bytes() + " bytes]";
    }
}
