package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.engine.BatchSegments;

/**
 * Per-step flags the forward backend sets before it runs the slots and clears after, read by the mixers of the
 * step (the paged step they read off the memory is the sibling mechanism). Created with the context and shared by
 * every context derived from it, so a backend holds one state over its whole life. Not thread-safe.
 */
final class CpuForwardStepState {

    private boolean nonCausal;
    private long[] treeMask;        // per-ubatch-row ancestor-or-self bitmask; null = no tree mask
    private int treeFresh = -1;     // the mask's fresh rows (the tree rows of the step); -1 = none
    private int[] storePositions;   // per-ubatch-row KV store positions; null = the POS slot stores too
    private CpuRecurrentTreeRound treeRound; // the tree step's recurrent round; null = no recurrent tree step
    private BatchSegments segments; // the sequence runs of a packed memoryless step; null = one sequence

    /** True when the running step attends without the causal mask ({@code llama_set_causal_attn(false)}). */
    boolean nonCausal() {
        return nonCausal;
    }

    /** Sets the flag for the running step. */
    void nonCausal(boolean v) {
        this.nonCausal = v;
    }

    /** The running step's tree mask and its fresh-row count, or null when the step carries none. */
    long[] treeMask() {
        return treeMask;
    }

    /** The running step's fresh tree rows, or -1 when the step carries no tree mask. */
    int treeFresh() {
        return treeFresh;
    }

    /** Sets the running step's tree mask ({@code fresh} fresh rows). */
    void treeMask(long[] mask, int fresh) {
        this.treeMask = mask;
        this.treeFresh = fresh;
    }

    /** The running step's recurrent tree round, or null when the step runs no recurrent tree step. */
    CpuRecurrentTreeRound treeRound() {
        return treeRound;
    }

    /** Sets the running step's recurrent tree round (a tree verify's masked step over a recurrent target). */
    void treeRound(CpuRecurrentTreeRound round) {
        this.treeRound = round;
    }

    /** The running step's KV store positions, or null when the rows store at their POS positions. */
    int[] storePositions() {
        return storePositions;
    }

    /** Sets the running step's KV store positions (a contiguous tree verify's row slots). */
    void storePositions(int[] positions) {
        this.storePositions = positions;
    }

    /** The running memoryless step's sequence runs, or null when its rows form one sequence. */
    BatchSegments segments() {
        return segments;
    }

    /** Sets the running memoryless step's sequence runs (a packed encoder batch). */
    void segments(BatchSegments v) {
        this.segments = v;
    }
}
