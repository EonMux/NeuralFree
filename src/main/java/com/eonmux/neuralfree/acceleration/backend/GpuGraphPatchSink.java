package com.eonmux.neuralfree.acceleration.backend;

/**
 * The receiver of one step's captured-graph patches: a {@link GpuGraphPatchPlan} derives the per-step
 * argument values of the captured dispatches and hands each changed dispatch here; the graph applies
 * the patch to its captured node before submitting the capture as one device submission. Values follow
 * the kernel-argument rules of {@link KernelArgs}: buffers by identity, scalars boxed, arrays cloned.
 * Not thread-safe; a graph calls it from the thread that replays the graph.
 */
public interface GpuGraphPatchSink {

    /**
     * Patches the captured kernel dispatch at position {@code recordIndex} (a
     * {@link GpuGraphDispatch#index()}) with this step's values. Only the listed argument indices are
     * patched; {@code grid}, when not {@code null}, replaces the dispatch's global work size. The
     * graph applies a patch only when it differs from what the node currently holds.
     *
     * @throws IllegalArgumentException when the graph holds no kernel dispatch at {@code recordIndex}
     */
    void patchKernel(int recordIndex, long[] grid, int[] argIndices, Object[] values);
}
