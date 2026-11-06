package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchPlan;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchSink;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The patch plan of one captured decode trunk: the step patchers of the trunk's mixers claim the
 * captured kernel dispatches they recognize, and the plan is trusted only when their predictions
 * reproduce the capture exactly. Building it runs every claimed patch against the capture position
 * through a recording sink and compares each predicted argument value and grid with the captured
 * node's ({@link KernelArgs#sameValue}); any divergence — a derivation a patcher does not model, a
 * foreign kernel inside a group, a pattern the patcher refuses — rejects the whole plan, so the
 * caller keeps re-running the dispatch body on the replay.
 *
 * <p>On a trusted plan, {@link #patchStep(int, GpuGraphPatchSink)} patches the nodes the mixers
 * claimed for the step position and reports a pattern divergence as {@code false}; a false return
 * aborts the replay before the launch, and the caller runs the body. Not thread-safe; one plan per
 * capture, built by the trunk that owns the patchers.
 */
public final class DecodeReplayPlan implements GpuGraphPatchPlan {

    private static final Logger LOGGER = LoggingUtil.getLogger(DecodeReplayPlan.class);

    private final GpuStepPatcher[] owners;
    private final int capturePos;

    private DecodeReplayPlan(GpuStepPatcher[] owners, int capturePos) {
        this.owners = owners;
        this.capturePos = capturePos;
    }

    /**
     * Builds the plan of {@code graph} over {@code patchers} for the step the trunk captured it at:
     * each patcher claims the captured dispatches it recognizes ({@link GpuStepPatcher#accepts}, first
     * claim wins), then every claimed dispatch is patched at {@code capturePos} into a recording sink
     * and the prediction is compared with the captured node. Null when the graph exposes no kernel
     * dispatches, no patcher claimed one, a patcher refuses the capture's pattern, or a prediction
     * misses the captured values — the caller then replays by running the body.
     */
    public static DecodeReplayPlan build(GpuGraph graph, List<GpuStepPatcher> patchers, int capturePos) {
        Objects.requireNonNull(graph, "graph");
        if (patchers == null || patchers.isEmpty()) {
            return null;
        }
        List<GpuGraphDispatch> dispatches = graph.kernelDispatches();
        if (dispatches == null || dispatches.isEmpty()) {
            return null;
        }
        for (GpuStepPatcher p : patchers) {
            p.resetClaims();
        }
        GpuStepPatcher[] owners = new GpuStepPatcher[dispatches.size()];
        for (GpuGraphDispatch d : dispatches) {
            for (GpuStepPatcher p : patchers) {
                if (p.accepts(d)) {
                    owners[d.index()] = p;
                    break;
                }
            }
        }
        boolean any = false;
        for (int i = 0; i < owners.length; i++) {
            GpuStepPatcher p = owners[i];
            if (p == null) {
                continue;
            }
            RecordedSink recorded = new RecordedSink();
            if (!p.patch(capturePos, i, recorded)) {
                LOGGER.debug("Decode replay plan rejects dispatch {} ({}): its patcher refuses the capture's"
                        + " pattern", i, graph.kernelDispatch(i).kernelName());
                return null;
            }
            if (!matches(graph.kernelDispatch(i), recorded)) {
                return null;
            }
            any = true;
        }
        if (!any) {
            LOGGER.debug("Decode replay plan rejects the capture: {} patchers claim none of its {} dispatches",
                    patchers.size(), dispatches.size());
        }
        return any ? new DecodeReplayPlan(owners, capturePos) : null;
    }

    /** The position the trunk captured the graph at; the patchers reproduce its values from it. */
    public int capturePos() {
        return capturePos;
    }

    /** True when a patcher claims the captured dispatch at {@code recordIndex}. */
    public boolean claims(int recordIndex) {
        return recordIndex >= 0 && recordIndex < owners.length && owners[recordIndex] != null;
    }

    @Override
    public boolean patchStep(int pos, GpuGraphPatchSink sink) {
        GpuGraphPatchSink applied = LOGGER.isDebugEnabled() ? new LoggingSink(sink, pos) : sink;
        for (int i = 0; i < owners.length; i++) {
            GpuStepPatcher p = owners[i];
            if (p != null && !p.patch(pos, i, applied)) {
                return false;
            }
        }
        return true;
    }

    /** A sink that logs every patch a trusted step applies, for the replay trace. */
    private static final class LoggingSink implements GpuGraphPatchSink {
        private final GpuGraphPatchSink delegate;
        private final int pos;

        LoggingSink(GpuGraphPatchSink delegate, int pos) {
            this.delegate = delegate;
            this.pos = pos;
        }

        @Override
        public void patchKernel(int recordIndex, long[] grid, int[] argIndices, Object[] values) {
            LOGGER.debug("trusted step {} patches dispatch {} grid {} args {}={}", pos, recordIndex,
                    grid == null ? "null" : Arrays.toString(grid), Arrays.toString(argIndices),
                    Arrays.toString(values));
            delegate.patchKernel(recordIndex, grid, argIndices, values);
        }
    }

    /** True when the recorded patch matches the captured dispatch's grid and argument values. */
    private static boolean matches(GpuGraphDispatch d, RecordedSink recorded) {
        if (recorded.grid != null && !Arrays.equals(recorded.grid, d.grid())) {
            LOGGER.debug("Decode replay plan rejects dispatch {} ({}): the patch predicts grid {} but the"
                    + " capture holds {}", d.index(), d.kernelName(), Arrays.toString(recorded.grid),
                    Arrays.toString(d.grid()));
            return false;
        }
        if (recorded.argIndices.length != recorded.values.length) {
            return false;
        }
        for (int a = 0; a < recorded.argIndices.length; a++) {
            if (!KernelArgs.sameValue(recorded.values[a], d.arg(recorded.argIndices[a]))) {
                LOGGER.debug("Decode replay plan rejects dispatch {} ({}): arg {} predicts {} but the capture"
                        + " holds {}", d.index(), d.kernelName(), recorded.argIndices[a],
                        KernelArgs.describe(recorded.values[a]), KernelArgs.describe(d.arg(recorded.argIndices[a])));
                return false;
            }
        }
        return true;
    }

    /** A sink of the build that records the one patch a patcher hands it. */
    private static final class RecordedSink implements GpuGraphPatchSink {
        private long[] grid;
        private int[] argIndices = new int[0];
        private Object[] values = new Object[0];

        @Override
        public void patchKernel(int recordIndex, long[] grid, int[] argIndices, Object[] values) {
            this.grid = grid == null ? null : grid.clone();
            this.argIndices = argIndices.clone();
            this.values = values.clone();
        }
    }
}
