package com.eonmux.neuralfree.acceleration.backend;

/**
 * The per-step derivations of one stage's captured dispatches: a mixer that re-sets step-varying
 * arguments on its kernels before every dispatch (a query position, a key count, a split count and
 * the work size they imply) exposes a patcher that computes the same values from the step position
 * alone. A {@link GpuGraphPatchPlan} built over a capture lets its patchers claim the captured
 * dispatches they recognize ({@link #accepts(GpuGraphDispatch)}), validate their predictions against
 * the values the capture holds ({@link #patch(int, int, GpuGraphPatchSink)} with a recording sink)
 * and, once trusted, feed {@link GpuGraph#replayPatched(GpuGraphPatchPlan, int)}. Not thread-safe.
 */
public interface GpuStepPatcher {

    /**
     * True when this patcher derives the step values of the captured dispatch {@code d}: the kernel
     * is one of this stage's and the dispatch's static argument values (its bound buffers and row
     * offsets) are this stage's. A dispatch several stages bind (a shared cache two layers read)
     * matches the first patcher that accepts it; their step values must then be identical.
     */
    boolean accepts(GpuGraphDispatch d);

    /**
     * Forgets the dispatches this patcher claimed so far. A plan builder calls it before the first
     * {@link #accepts} of a fresh capture, so the claims of an earlier, invalidated capture never
     * survive into the new plan.
     */
    default void resetClaims() {
    }

    /**
     * Hands the step values of the captured dispatch at {@code recordIndex} to {@code sink}: the
     * arguments this stage re-sets per step, with the grid when the step changes it. When the step's
     * command pattern leaves the capture this patcher's stage would enqueue or skip a dispatch
     * against the capture's pattern (a split-K merge that appears only past a key-count threshold),
     * it returns false and patches nothing. A decline must happen before the first sink call, so a
     * refused step leaves no patch applied: the plan aborts at the first false and the caller may
     * patch another position against the same capture.
     *
     * @return true when the patch is the capture's pattern for {@code pos}, false when the pattern
     *         diverges and the caller must run the step body instead
     */
    boolean patch(int pos, int recordIndex, GpuGraphPatchSink sink);

    /**
     * True when this stage's kernels currently hold exactly the step values this patcher derives for
     * {@code pos}: the check a graph replay shadow-runs against the body's dispatches before it
     * trusts the patcher without a body. A stage whose kernels hold other values than the prediction
     * (a derivation the patcher does not model) returns false.
     */
    boolean verify(int pos);
}
