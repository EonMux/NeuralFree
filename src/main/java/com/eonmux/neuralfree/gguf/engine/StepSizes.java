package com.eonmux.neuralfree.gguf.engine;

/**
 * How a run of tokens divides into forward steps: as few steps as the per-step cap allows, of equal size up to one
 * token, so that no short step trails the run.
 *
 * <p>A step pays a fixed cost whatever its rows (uploading or prefetching the experts a streaming model reads,
 * one submission of the queue), and its cost grows with its rows only up to that fixed cost, so the cost is convex
 * in the step's size: {@code k} steps of equal size cost no more in total than {@code k} uneven steps holding the
 * same rows, and a short trailing step is nearly as expensive as a full one. Filling greedily leaves one: 19014
 * tokens at a cap of 8192 take steps of 8192, 8192 and 2630, where the same 3 steps of equal size take 6338 each.
 *
 * <p>{@link #next} answers the size of the next step of a run, so a caller that loops over the rows it still has to
 * store needs no state beyond them. The answers of one run differ by at most one token and it takes as few steps as
 * the cap allows, the first of them {@code ceil(rows / ceil(rows / maxStep))} tokens.
 */
public final class StepSizes {

    private StepSizes() {
    }

    /**
     * The tokens of the next step of a run of {@code remaining} tokens whose steps hold at most {@code maxStep}.
     *
     * @throws IllegalArgumentException for {@code remaining < 1} or {@code maxStep < 1}
     */
    public static int next(int remaining, int maxStep) {
        return next(remaining, maxStep, 0);
    }

    /**
     * {@link #next(int, int)} for a run whose last step must hold at least {@code keepTail} tokens: the trailing
     * tokens a rollback may drop stay in one step (llama.cpp's {@code n_keep_tail}, src/llama-batch.cpp:714-765).
     * The steps stay equal up to one token where the tail allows it; where it does not, the steps before it give up
     * the tokens it needs.
     *
     * <p>A step of at most {@code keepTail} tokens cannot keep the tail apart and splits plainly (llama.cpp asserts
     * {@code n_ubatch > n_keep_tail}), as does a run no longer than one step, whose only step is the last one.
     *
     * @throws IllegalArgumentException for {@code remaining < 1}, {@code maxStep < 1} or a negative {@code keepTail}
     */
    public static int next(int remaining, int maxStep, int keepTail) {
        if (remaining < 1 || maxStep < 1) {
            throw new IllegalArgumentException("a step needs remaining >= 1 and maxStep >= 1, got " + remaining
                    + "/" + maxStep);
        }
        if (keepTail < 0) {
            throw new IllegalArgumentException("keepTail must not be negative, got " + keepTail);
        }
        if (remaining <= maxStep) {
            return remaining;
        }
        if (keepTail >= maxStep) {
            return maxStep;
        }
        int steps = ceilDiv(remaining, maxStep);
        int size = ceilDiv(remaining, steps);
        // an equal split leaves the run's smallest step last; a trailing tail it does not hold takes tokens from
        // the steps before it
        if ((long) remaining - (long) (steps - 1) * size < keepTail) {
            size = ceilDiv(remaining - keepTail, steps - 1);
        }
        return size;
    }

    private static int ceilDiv(int a, int b) {
        return (int) (((long) a + b - 1) / b);
    }
}
