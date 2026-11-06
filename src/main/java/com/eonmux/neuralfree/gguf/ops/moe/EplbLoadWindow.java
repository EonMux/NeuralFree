package com.eonmux.neuralfree.gguf.ops.moe;

import java.util.List;

/**
 * The live routed-row window of the runtime expert rearrangement (vLLM's {@code expert_load_window}): the routed
 * rows of the last {@link #windowSize} steps before a rearrangement boundary, counted per (weight layer, logical
 * expert) like a load profile. The rearrangement scheduler opens the recording for the steps the next rearrangement
 * reads ({@link #record}, vLLM's {@code _should_record_current_step}) and drains the counts at the boundary
 * (vLLM's window sum), so the rearrangement balances from the steps it asked for and the recorder costs nothing
 * while it is closed.
 *
 * <p>Not thread-safe: the router read-back of the forward thread notes into it, the step scheduler drives it.</p>
 */
public final class EplbLoadWindow {

    private final MoeExpertHeat counts = new MoeExpertHeat();
    private final int windowSize;
    private boolean recording;

    /**
     * A window over {@code windowSize} steps.
     *
     * @throws IllegalArgumentException when the size is below one
     */
    public EplbLoadWindow(int windowSize) {
        if (windowSize < 1) {
            throw new IllegalArgumentException("the load window must hold at least one step, got " + windowSize);
        }
        this.windowSize = windowSize;
    }

    /** The steps of live load the window keeps. */
    public int windowSize() {
        return windowSize;
    }

    /**
     * Opens or closes the recording of the following steps: open when the rearrangement is within
     * {@link #windowSize} steps.
     */
    public void record(boolean recording) {
        this.recording = recording;
    }

    /** True when the window counts the current step's routed rows. */
    public boolean recording() {
        return recording;
    }

    /**
     * Counts one step's routed ids of {@code layer} when the window records.
     *
     * @throws IllegalArgumentException when {@code ids} does not hold {@code pairs} entries
     */
    public void note(int layer, int[] ids, int pairs) {
        if (recording) {
            counts.note(layer, ids, pairs);
        }
    }

    /** True when no routed rows are counted. */
    public boolean isEmpty() {
        return counts.isEmpty();
    }

    /** The counted routed rows, hottest first, and clears the window. */
    public List<RankedExpert> drain() {
        List<RankedExpert> out = counts.snapshot();
        counts.clear();
        return out;
    }

    @Override
    public String toString() {
        return "EplbLoadWindow{" + windowSize + " step(s), " + (recording ? "recording" : "closed") + "}";
    }
}
