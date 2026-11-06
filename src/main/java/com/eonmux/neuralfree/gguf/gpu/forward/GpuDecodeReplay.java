package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import com.eonmux.neuralfree.acceleration.backend.GpuUncapturableCommandException;
import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Decode-step graph replay over one queue, the device-graph capture of the decode trunk: once the trunk's
 * shape has stayed stable for a few steps, it records the trunk's kernel dispatches
 * ({@code ComputeQueue.beginCapture}) and submits the capture once with
 * {@link GpuGraph#launch(ComputeQueue, GpuEvent...)}.
 *
 * <p>A trusted replay never runs the step body again. The trunk's mixers expose patchers
 * ({@link GpuMixer#decodeReplayPatchers()} through {@code GpuLayerExecutor#decodeReplayPatchers()}), each
 * deriving its kernels' per-step values (positions, key counts, split counts, grids) from the step position
 * alone. The capture carries every kernel dispatch's captured values; {@link DecodeReplayPlan} is built
 * once over them and trusted only when the patchers' predictions reproduce the capture exactly. The first
 * steps after the capture still run the body through {@link GpuGraph#replay(Runnable)} as a check: every
 * dispatch the body changed must be claimed by the plan, and every patcher must find its kernels holding
 * exactly the values it predicted. After two agreeing steps the trunk replays with
 * {@link GpuGraph#replayPatched(GpuGraphPatchPlan, int)} — the plan's patches land in the captured nodes'
 * parameters and one submission reaches the device, without re-running and re-matching the body's
 * dispatches. A backend that holds a second idle instantiation of the capture takes the follow-up step's
 * patches while the current step runs ({@link GpuGraph#preparePatched}), so a trusted step pays no patch
 * time on its critical path. A step whose pattern leaves the capture (a split-K merge that appears past a
 * key-count threshold) is declined and recaptured, never replayed against a stale graph; a step whose
 * position does not follow the previous one is stale by the same rule. Small and MoE models pay per
 * kernel launch at decode, so the collapsed submission is where their decode time goes.
 *
 * <p>Warmup ({@code neuralfree.gpu.graphWarmupSteps}, default {@value #DEFAULT_WARMUP_STEPS}): a capture
 * happens only after that many consecutive eligible decode steps have run with the same trunk shape, so
 * a step that is still settling (lazily growing caches, the first steps after a prefill) runs directly
 * and is not recorded. Every call that changes the trunk's dispatch sequence (a prefill, a rollback, a
 * memory reset, a context shift, a tap change) calls {@link #invalidate()}, which closes the graph and
 * restarts the warmup count. A value of 0 captures the first eligible step.
 *
 * <p>Eligibility ({@code capturable}): the caller decides, per step, that the trunk only enqueues kernel
 * dispatches (no tap reads, no host transfers, one token against memory), which a capture can hold — and
 * that a replay may skip, since a step that would enqueue more must run the body. A replay whose body does
 * not match its capture (the mixers enqueued a different sequence or arguments the plan cannot patch)
 * closes the graph and runs the trunk directly while the shape settles again; after a few failed replays
 * the attempts stop until {@link #invalidate()}. A capture that reaches a command the queue cannot record
 * (a host transfer or read) aborts cleanly, counts toward the aborted-capture limit and runs the step
 * directly, so an uncapturable trunk never fails generation. A capture whose step failed is aborted as
 * well, so no partial device work survives. A trunk without patchers, or a backend that cannot patch
 * captured nodes (a verify-only graph), replays through {@link GpuGraph#replay(Runnable)}, which re-runs
 * the body and matches its dispatches against the capture.
 * {@code neuralfree.gpu.decodeReplay=false} (default {@code true}) disables the replay entirely. Not
 * thread-safe.
 */
final class GpuDecodeReplay implements AutoCloseable {

    static final String PROPERTY = GpuForwardBackend.DECODE_REPLAY_PROPERTY;
    static final String WARMUP_PROPERTY = "neuralfree.gpu.graphWarmupSteps";
    /** Consecutive stable decode steps before the first capture, the default of {@link #WARMUP_PROPERTY}. */
    static final int DEFAULT_WARMUP_STEPS = 2;
    /** Steps the body runs after a capture before the plan is trusted without it. */
    private static final int VERIFY_STEPS = 2;
    /** Failed replays in a row after which the attempts stop until {@link #invalidate()}. */
    private static final int MAX_FAILED_REPLAYS = 2;
    /** Captures that failed after which the attempts stop until {@link #invalidate()}. */
    private static final int MAX_ABORTED_CAPTURES = 3;
    /** Prints every trusted step's patch and prepare host time ({@code neuralfree.probe.replayHostCost}). */
    private static final boolean HOST_COST_PROBE =
            "1".equals(System.getProperty("neuralfree.probe.replayHostCost"));

    private static final Logger LOGGER = LoggingUtil.getLogger(GpuDecodeReplay.class);

    private final ComputeQueue queue;
    private final String queueType;
    private final boolean configured;
    private final int warmupSteps;
    private GpuGraph graph;
    /** The patch plan of {@link #graph}, or null when no patcher claimed the capture or the backend cannot patch. */
    private DecodeReplayPlan plan;
    /** Body steps run over {@link #plan} since the capture; trusted at {@link #VERIFY_STEPS}. */
    private int verifiedSteps;
    /** The position of the last step this replay ran, for the continuity guard of a trusted replay. */
    private int lastPos = -1;
    private int stableSteps;
    private int failedReplays;
    private int abortedCaptures;

    GpuDecodeReplay(ComputeQueue queue) {
        this.queue = queue;
        this.queueType = queue.getQueueType().name().toLowerCase(Locale.ROOT) + " queue";
        this.configured = configured();
        this.warmupSteps = warmupSteps();
    }

    /**
     * {@code neuralfree.gpu.decodeReplay} as a boolean, default {@code true}.
     *
     * @throws IllegalArgumentException when the property is set to something else than a boolean
     */
    static boolean configured() {
        String value = System.getProperty(PROPERTY);
        if (value == null || value.isEmpty()) {
            return true;
        }
        if (value.equalsIgnoreCase("true")) {
            return true;
        }
        if (value.equalsIgnoreCase("false")) {
            return false;
        }
        throw new IllegalArgumentException("-D" + PROPERTY + " must be a boolean, got '" + value + "'");
    }

    /**
     * {@code neuralfree.gpu.graphWarmupSteps} as a non-negative integer, default
     * {@value #DEFAULT_WARMUP_STEPS}.
     *
     * @throws IllegalArgumentException when the property is not a non-negative integer
     */
    static int warmupSteps() {
        String value = System.getProperty(WARMUP_PROPERTY);
        if (value == null || value.isEmpty()) {
            return DEFAULT_WARMUP_STEPS;
        }
        int steps;
        try {
            steps = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("-D" + WARMUP_PROPERTY + " must be a non-negative integer, got '"
                    + value + "'", e);
        }
        if (steps < 0) {
            throw new IllegalArgumentException("-D" + WARMUP_PROPERTY + " must be a non-negative integer, got '"
                    + value + "'");
        }
        return steps;
    }

    /**
     * Runs the trunk of one decode step at {@code pos}: on a trusted plan, patches the captured graph and
     * submits it without the body; before that, runs the body as the capture's check or captures after the
     * warmup; or runs the trunk directly.
     *
     * @param pos        the step position; a trusted replay requires it to follow the previous step's
     * @param patchers   the trunk mixers' step patchers, or null or empty for a body replay
     * @param capturable says the step's trunk only enqueues kernel dispatches (no tap reads, no host
     *                   transfers), which a capture can hold and a replay may skip
     */
    void trunk(int pos, List<GpuStepPatcher> patchers, boolean capturable, Runnable body) {
        try {
            trunkStep(pos, patchers, capturable, body);
        } finally {
            lastPos = pos;
        }
    }

    /**
     * Runs the trunk of one step that has no position to check (a multi-row verify step), by the body
     * replay: the body runs again and its dispatches are matched against the capture.
     */
    void trunk(boolean capturable, Runnable body) {
        trunkStep(-1, null, capturable, body);
    }

    private void trunkStep(int pos, List<GpuStepPatcher> patchers, boolean capturable, Runnable body) {
        if (!configured || !queue.supportsGraphCapture()) {
            body.run();
            return;
        }
        if (LOGGER.isDebugEnabled()) {
            LOGGER.debug("decode step pos={} {}", pos, this);
        }
        if (graph != null && plan != null && verifiedSteps >= VERIFY_STEPS) {
            trusted(pos, patchers, capturable, body);
            return;
        }
        if (graph != null && plan != null) {
            verifyStep(pos, patchers, body);
            return;
        }
        if (graph != null) {
            bodyReplay(body);
            if (graph != null) {
                return;
            }
        }
        capture(pos, patchers, capturable, body);
    }

    /** Replays a trusted plan: patches and submits, never touching the body. */
    private void trusted(int pos, List<GpuStepPatcher> patchers, boolean capturable, Runnable body) {
        if (!capturable || pos != lastPos + 1) {
            /* A step that enqueues more than the capture, or whose position jumped: the graph may be
             * stale, so it is never replayed. */
            LOGGER.debug("Decode replay re-captures: step {} does not follow {} or the trunk is not capturable",
                    pos, lastPos);
            dropGraph();
            body.run();
            return;
        }
        boolean replayed;
        long patchStart = HOST_COST_PROBE ? System.nanoTime() : 0;
        try {
            replayed = graph.replayPatched(plan, pos);
        } catch (GpuGraphMismatchException e) {
            LOGGER.debug("Decode replay re-captures after: {}", e.getMessage());
            replayed = false;
        }
        if (replayed) {
            failedReplays = 0;
            if (HOST_COST_PROBE) {
                long prepared = System.nanoTime();
                prepareFollowup(pos);
                hostCost(pos, (prepared - patchStart) / 1e6, (System.nanoTime() - prepared) / 1e6);
                return;
            }
            prepareFollowup(pos);
            return;
        }
        /* A declined pattern or a backend refusal: the capture no longer fits the step. */
        dropGraph();
        stableSteps = 0;
        if (++failedReplays >= MAX_FAILED_REPLAYS) {
            LOGGER.debug("Decode replay stops trying until the step shape changes: {} failed replays",
                    failedReplays);
        }
        body.run();
    }

    /**
     * Patches the follow-up step's graph while this step runs, for backends that hold a second idle
     * instantiation: the next trusted replay then submits a graph whose parameters already hold the
     * step's values. A failure here cannot touch the step that just launched; the graph is dropped so
     * the next step recaptures.
     */
    private void prepareFollowup(int pos) {
        try {
            graph.preparePatched(plan, pos + 1, pos);
        } catch (RuntimeException e) {
            LOGGER.debug("Decode replay drops its graph after a failed follow-up patch: {}", e.getMessage());
            dropGraph();
        }
    }

    /** Appends one trusted step's host-time split to {@code build/replay-host-cost.txt}. */
    private static void hostCost(int pos, double patchMs, double prepareMs) {
        try (java.io.PrintWriter out = new java.io.PrintWriter(new java.io.FileWriter(
                Path.of("build", "replay-host-cost.txt").toFile(), true))) {
            out.println(String.format(Locale.ROOT,
                    "replay-host-cost pos %d patch+launch %.3f ms prepare %.3f ms", pos, patchMs, prepareMs));
        } catch (Exception e) {
            LOGGER.debug("Decode replay host-cost probe write failed: {}", e.getMessage());
        }
    }

    /** Runs a body step as the capture's check: the plan must cover exactly what the body changed. */
    private void verifyStep(int pos, List<GpuStepPatcher> patchers, Runnable body) {
        List<GpuGraphDispatch> before = LOGGER.isDebugEnabled() ? graph.kernelDispatches() : null;
        try {
            graph.replay(body);
        } catch (GpuGraphMismatchException e) {
            LOGGER.debug("Decode replay re-captures after a verify step: {}", e.getMessage());
            dropGraph();
            if (++failedReplays >= MAX_FAILED_REPLAYS) {
                LOGGER.debug("Decode replay stops trying until the step shape changes: {} failed replays",
                        failedReplays);
            }
            body.run();
            return;
        }
        verifiedSteps++;
        int[] changed = graph.takeReplayChanges();
        if (changed == null || !cover(changed)) {
            LOGGER.debug("Decode replay drops its patch plan: the step changed dispatches the plan does not"
                    + " claim: {}", unclaimed(graph, before, changed));
            dropGraph();
            if (++failedReplays >= MAX_FAILED_REPLAYS) {
                LOGGER.debug("Decode replay stops trying until the step shape changes: {} failed replays",
                        failedReplays);
            }
            return;
        }
        if (!patchersVerify(pos, patchers)) {
            LOGGER.debug("Decode replay drops its patch plan: a patcher's kernels hold values it does not"
                    + " predict for step {}", pos);
            dropGraph();
            if (++failedReplays >= MAX_FAILED_REPLAYS) {
                LOGGER.debug("Decode replay stops trying until the step shape changes: {} failed replays",
                        failedReplays);
            }
        }
    }

    /**
     * The changed dispatches no patcher claims, with the arguments a verify step moved, for the drop
     * message. {@code before} holds the capture's snapshots, null when debug logging is off.
     */
    private String unclaimed(GpuGraph graph, List<GpuGraphDispatch> before, int[] changed) {
        if (changed == null || changed.length == 0) {
            return "none";
        }
        List<String> names = new ArrayList<>();
        for (int idx : changed) {
            if (plan != null && plan.claims(idx)) {
                continue;
            }
            GpuGraphDispatch now = graph.kernelDispatch(idx);
            String diff = "";
            if (before != null) {
                int[] moved = movedArgs(before.get(idx).args(), now.args());
                if (moved.length > 0) {
                    diff = " args" + java.util.Arrays.toString(moved);
                }
            }
            names.add(idx + " " + now.kernelName() + diff);
        }
        return names.isEmpty() ? "none" : String.join(", ", names);
    }

    /** The argument indices whose values differ, for the drop message. */
    private static int[] movedArgs(Object[] before, Object[] now) {
        List<Integer> moved = new ArrayList<>();
        for (int i = 0; i < before.length && i < now.length; i++) {
            if (!KernelArgs.sameValue(before[i], now[i])) {
                moved.add(i);
            }
        }
        int[] out = new int[moved.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = moved.get(i);
        }
        return out;
    }

    /** True when the plan claims every changed dispatch of the body. */
    private boolean cover(int[] changed) {
        for (int idx : changed) {
            if (!plan.claims(idx)) {
                return false;
            }
        }
        return true;
    }

    /** True when every patcher's kernels hold the values it predicts for {@code pos}. */
    private static boolean patchersVerify(int pos, List<GpuStepPatcher> patchers) {
        for (GpuStepPatcher p : patchers) {
            if (!p.verify(pos)) {
                return false;
            }
        }
        return true;
    }

    /** The body replay without a plan: the body runs again and matches the capture. */
    private void bodyReplay(Runnable body) {
        try {
            graph.replay(body);
            failedReplays = 0;
        } catch (GpuGraphMismatchException e) {
            LOGGER.debug("Decode replay re-captures after: {}", e.getMessage());
            closeGraph();
            stableSteps = 0;
            if (++failedReplays >= MAX_FAILED_REPLAYS) {
                LOGGER.debug("Decode replay stops trying until the step shape changes: {} failed replays",
                        failedReplays);
            }
        }
    }

    /** The warmup and the capture: run the body until the shape is stable, then record it. */
    private void capture(int pos, List<GpuStepPatcher> patchers, boolean capturable, Runnable body) {
        if (!capturable) {
            stableSteps = 0;
            body.run();
            return;
        }
        if (stableSteps < warmupSteps) {
            stableSteps++;
            body.run();
            return;
        }
        if (failedReplays >= MAX_FAILED_REPLAYS || abortedCaptures >= MAX_ABORTED_CAPTURES) {
            body.run();
            return;
        }
        queue.beginCapture();
        try {
            body.run();
        } catch (GpuUncapturableCommandException e) {
            // the recorded commands never ran: drop them and run this step directly
            try {
                queue.abortCapture();
            } catch (RuntimeException abortFailure) {
                e.addSuppressed(abortFailure);
            }
            abortedCaptures++;
            stableSteps = 0;
            LOGGER.debug("Decode replay runs the {} directly: {}", queueType, e.getMessage());
            body.run();
            return;
        } catch (RuntimeException e) {
            try {
                queue.abortCapture();
            } catch (RuntimeException abortFailure) {
                e.addSuppressed(abortFailure);
            }
            abortedCaptures++;
            stableSteps = 0;
            throw e;
        }
        GpuGraph captured = queue.endCapture();
        if (captured == null) {
            LOGGER.debug("The {} recorded no dispatch; its step ran nothing", queueType);
            return;
        }
        // the capture executed nothing: submit it once so the step that recorded it reaches the device
        graph = captured;
        plan = null;
        verifiedSteps = 0;
        if (patchers != null && !patchers.isEmpty() && graph.supportsPatchedReplay()) {
            plan = DecodeReplayPlan.build(graph, patchers, pos);
            if (plan == null) {
                LOGGER.debug("Decode replay keeps the {} body replay: no patcher reproduces the capture",
                        queueType);
            }
        }
        try {
            graph.launch(queue);
        } catch (RuntimeException e) {
            closeGraph();
            throw e;
        }
    }

    /**
     * Drops the captured graph and the attempt limits: anything that can change the trunk's dispatch
     * sequence between steps (a non-decode call, a memory rollback, shift or reset) calls this.
     */
    void invalidate() {
        closeGraph();
        lastPos = -1;
        stableSteps = 0;
        failedReplays = 0;
        abortedCaptures = 0;
    }

    /**
     * Re-arms the continuity guard for a rewind that kept the capture (the speculative round's replay,
     * {@code neuralfree.gpu.chainReplay}): a rollback that only rewound the host's sequence bookkeeping
     * sets the position the next step runs at, so the guard accepts it although it does not follow the
     * last step's. The capture, its plan and the verify count stay; nothing else changes.
     */
    void rearm(int nextPos) {
        lastPos = nextPos - 1;
    }

    /** Drops the graph and its plan, and lets the warmup restart. The step's body runs this step. */
    private void dropGraph() {
        closeGraph();
        plan = null;
        verifiedSteps = 0;
        stableSteps = 0;
    }

    private void closeGraph() {
        if (graph == null) {
            return;
        }
        GpuGraph captured = graph;
        graph = null;
        captured.close();
    }

    /** Releases the captured graph. Idempotent. */
    @Override
    public void close() {
        closeGraph();
    }

    @Override
    public String toString() {
        return "GpuDecodeReplay{" + queueType + (graph != null ? ", " + graph.dispatchCount() + " dispatches"
                + (plan != null ? (verifiedSteps >= VERIFY_STEPS ? ", patched replay" : ", verifying")
                : ", body replay")
                : ", " + stableSteps + "/" + warmupSteps + " warmup steps")
                + (configured ? "" : ", disabled") + "}";
    }
}
