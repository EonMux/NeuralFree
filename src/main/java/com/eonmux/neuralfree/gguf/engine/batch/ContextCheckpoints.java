package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.memory.CompressedStateCopy;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.RecurrentStateCopy;
import com.eonmux.neuralfree.gguf.memory.paged.PageWindow;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Context checkpoints of a {@link BatchEngine} whose memory cannot reuse a prompt prefix from cached KV blocks
 * alone (llama.cpp's server checkpoints, {@code --ctx-checkpoints}; vLLM's {@code mamba_cache_mode="align"} and
 * {@code SlidingWindowManager}): a recurrent state cannot be rewound to an earlier position, a sliding-window
 * group trims the rows that fall below its window, and a DSV4 compressor state accumulates across positions, so
 * a cached prefix of {@code n} rows is only restorable where a {@link ContextCheckpoint} at exactly {@code n}
 * rows holds the recurrent state, the DSV4 compressed streams and the window pages. This class captures, restores
 * and releases them over the memory, and decides where they are taken:
 * <ul>
 *   <li>when a sequence leaves (retirement or preemption), at its committed rows — the point a conversation's
 *       next turn, or the preempted sequence's own recompute, continues from (llama.cpp keeps a slot's live
 *       state);</li>
 *   <li>on a prefill's way, at the last page boundary before the prompt's final row (llama.cpp takes its
 *       checkpoint a few tokens before the end, so a repeated prompt re-runs only its tail), every
 *       {@code neuralfree.checkpoint.min.step} rows before it (llama.cpp's {@code --checkpoint-min-step}; 0 sets
 *       no minimum, a checkpoint at every chunk's end), and at the message boundaries the prompt names
 *       (llama.cpp's message spans). The prefill's chunks and chunk pieces end there, and the checkpoint hangs on
 *       the cached block that ends there — the chain's tail at a page boundary, a freshly stored partial block at
 *       a boundary row mid-page.</li>
 * </ul>
 * The prefix cache keeps at most {@code neuralfree.ctx.checkpoints} per sequence slot; beyond the limit, the
 * checkpoints within {@code neuralfree.checkpoint.min.step} of the kept checkpoint before them leave first and
 * the oldest created one last (llama.cpp's {@code create_checkpoint}). Engine thread only.
 */
final class ContextCheckpoints {

    private final PagedSequences paged;
    private final ModelMemory memory;
    private final int limit;
    private final int minStep;
    private final int pageSize;
    private final boolean recurrent;
    private final boolean windowed;
    private final boolean compressed;

    /**
     * Checkpoints over {@code memory} and its paged sequences: {@code perSlot} kept per sequence slot,
     * {@code minStep} the least spacing on a prefill's way (0 for none: every chunk's end).
     *
     * @throws IllegalArgumentException      for a non-positive {@code perSlot} or a negative {@code minStep}
     * @throws UnsupportedOperationException when the memory cannot copy a sequence's recurrent state or its DSV4
     *                                       compressed streams
     */
    ContextCheckpoints(PagedSequences paged, ModelMemory memory, int perSlot, int minStep) {
        this.paged = Objects.requireNonNull(paged, "paged");
        this.memory = Objects.requireNonNull(memory, "memory");
        if (perSlot <= 0 || minStep < 0) {
            throw new IllegalArgumentException("context checkpoints need a positive count per slot and a"
                    + " non-negative spacing, got " + perSlot + " and " + minStep);
        }
        String refusal = memory.sequenceStateRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        String compressedRefusal = memory.compressedStateRefusal();
        if (compressedRefusal != null) {
            throw new UnsupportedOperationException(compressedRefusal);
        }
        this.limit = (int) Math.min(Integer.MAX_VALUE, (long) perSlot * paged.maxSequences());
        this.minStep = minStep;
        this.pageSize = paged.pageSize();
        this.recurrent = memory.hasRecurrentState();
        this.compressed = memory.plan().hasCompressed();
        boolean trims = false;
        for (int g = 0; g < paged.groups(); g++) {
            trims |= paged.trims(g);
        }
        this.windowed = trims;
    }

    /**
     * What makes a prefix of {@code memory} restorable only at a checkpoint: "recurrent state", "sliding-window
     * groups", "DSV4 compressed caches", any "and" join of them, or null when cached KV blocks alone restore it.
     */
    static String needs(ModelMemory memory, PagedSequences paged) {
        List<String> why = new ArrayList<>(3);
        if (memory.hasRecurrentState()) {
            why.add("recurrent state");
        }
        for (int g = 0; g < paged.groups(); g++) {
            if (paged.trims(g)) {
                why.add("sliding-window groups");
                break;
            }
        }
        if (memory.plan().hasCompressed()) {
            why.add("DSV4 compressed caches");
        }
        if (why.isEmpty()) {
            return null;
        }
        return String.join(" and ", why);
    }

    /** Whether the checkpoints hold sliding-window pages (some KV group trims). */
    boolean windowed() {
        return windowed;
    }

    /** Checkpoints the prefix cache keeps at most. */
    int limit() {
        return limit;
    }

    /** The least spacing two kept checkpoints keep on a prefill's way (0 for none). */
    int minStep() {
        return minStep;
    }

    /**
     * The checkpoint of sequence {@code seq} at its {@code rows} committed rows: a copy of its recurrent state,
     * a copy of its DSV4 compressed streams, and a hold on every sliding-window group's window pages.
     *
     * @throws IllegalArgumentException when the sequence's pages do not cover exactly {@code rows} rows
     */
    ContextCheckpoint capture(int seq, int rows) {
        RecurrentStateCopy state = recurrent ? memory.saveSequenceState(seq) : RecurrentStateCopy.EMPTY;
        CompressedStateCopy compressedState = compressed ? memory.saveCompressedState(seq) : null;
        PageWindow[] windows = new PageWindow[paged.groups()];
        try {
            for (int g = 0; g < windows.length; g++) {
                if (paged.trims(g)) {
                    windows[g] = paged.holdWindow(seq, g, rows);
                }
            }
        } catch (RuntimeException e) {
            releaseWindows(windows);
            throw e;
        }
        return new ContextCheckpoint(rows, state, compressedState, windows);
    }

    /** Writes the checkpoint's recurrent state and DSV4 streams into sequence {@code seq} (its windows join at
     * adoption). */
    void restore(int seq, ContextCheckpoint checkpoint) {
        if (recurrent) {
            memory.restoreSequenceState(seq, checkpoint.state);
        }
        if (compressed) {
            memory.restoreCompressedState(seq, checkpoint.compressed);
        }
    }

    /** Releases the checkpoint's window pages. */
    void release(ContextCheckpoint checkpoint) {
        releaseWindows(checkpoint.windows());
    }

    private void releaseWindows(PageWindow[] windows) {
        for (PageWindow w : windows) {
            if (w != null) {
                paged.releaseWindow(w);
            }
        }
    }

    /**
     * The row a prefill that has committed {@code committed} of its {@code end} rows stops at next for a
     * checkpoint, or -1: the nearest checkpoint row past {@code committed} — the next spaced row ({@code minStep}
     * past the {@code last} checkpoint, rounded up to a page), the tail row (the last page boundary before the
     * final row), or the prompt's next {@code boundary} row, whichever comes first. Without a minimum spacing
     * every chunk ends on a checkpoint row of its own, so only the tail bounds a chunk; a boundary row below
     * {@code end} lands wherever the message starts, page boundary or not.
     */
    int nextTarget(int committed, int end, int last) {
        return nextTarget(committed, end, last, -1);
    }

    /** {@link #nextTarget(int, int, int)} with the prompt's next message-boundary row ({@code -1} for none). */
    int nextTarget(int committed, int end, int last, int boundary) {
        int tail = tail(end);
        int best = -1;
        if (minStep > 0) {
            int spaced = roundUp((long) last + minStep);
            if (spaced > committed && spaced < tail) {
                best = spaced;
            }
        }
        if (tail > committed && tail > last && (best < 0 || tail < best)) {
            best = tail;
        }
        if (boundary > committed && boundary > last && boundary < end && (best < 0 || boundary < best)) {
            best = boundary;
        }
        return best;
    }

    /**
     * Whether a prefill that has committed {@code committed} of its {@code end} rows stands on a checkpoint row: a
     * page boundary past the {@code last} checkpoint and before the final row that is the tail row or at least
     * {@code minStep} past {@code last}.
     */
    boolean due(int committed, int end, int last) {
        if (committed <= last || committed % pageSize != 0 || committed >= end) {
            return false;
        }
        return committed == tail(end) || committed >= roundUp((long) last + minStep);
    }

    /** The last page boundary before the final of {@code end} rows (0 for a prompt within its first page). */
    private int tail(int end) {
        return (end - 1) / pageSize * pageSize;
    }

    private int roundUp(long rows) {
        long r = (rows + pageSize - 1) / pageSize * pageSize;
        return (int) Math.min(Integer.MAX_VALUE - pageSize, r);
    }
}
