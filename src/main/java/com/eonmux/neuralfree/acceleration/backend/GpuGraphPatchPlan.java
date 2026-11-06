package com.eonmux.neuralfree.acceleration.backend;

/**
 * The per-step patches of a captured graph, built once from the capture: for the step at position
 * {@code pos} it derives every captured dispatch's step-varying arguments and work size and applies
 * them through a {@link GpuGraphPatchSink}, so {@link GpuGraph#replayPatched(GpuGraphPatchPlan, int)}
 * submits the capture without running the step's dispatch body. The plan is exact: it predicts the
 * same values the body would set, because it is built from the same per-step derivations the mixers
 * run at dispatch time. A step whose command pattern leaves the capture (the body would enqueue a
 * dispatch the capture does not hold, or skip one it does) returns {@code false} from
 * {@link #patchStep(int, GpuGraphPatchSink)} and the caller runs the body instead. Not thread-safe.
 */
public interface GpuGraphPatchPlan {

    /**
     * Applies the patches of the step at position {@code pos} to {@code sink}.
     *
     * @return true when every captured dispatch this plan covers holds its step values, false when the
     *         step's command pattern leaves the capture; a caller that receives false must not submit
     *         the graph and runs the step body instead
     */
    boolean patchStep(int pos, GpuGraphPatchSink sink);
}
