package com.eonmux.neuralfree.acceleration.backend;

/**
 * A captured sequence of device commands that one queue re-enqueues as a single submission: the CUDA
 * graph or pre-recorded Vulkan command buffer of the decode step. A graph is captured from a queue that
 * answers {@code true} to {@link ComputeQueue#supportsGraphCapture()}: {@link ComputeQueue#beginCapture()}
 * starts recording, the kernel dispatches (and, where the backend supports them, device copies and fills)
 * enqueued while recording execute nothing and become this graph, and {@link ComputeQueue#endCapture()}
 * completes it.
 *
 * <p>{@link ComputeQueue#endCapture()} leaves the recorded commands unexecuted: the step that captured
 * them calls {@link #launch(ComputeQueue, GpuEvent...)} to submit them once with the arguments of the
 * capture. {@link #replay(Runnable)} runs the caller's step body again: the body enqueues the same command
 * sequence it enqueued at capture time, and the implementations submit the whole sequence as one unit. A
 * CUDA or HIP graph updates every captured node's parameters (the body's current arguments and grid) and
 * issues one {@code cuGraphLaunch}/{@code hipGraphLaunch}; a Vulkan queue or an OpenCL command buffer
 * verifies the body's dispatches against the recorded commands and resubmits them, updating scalar
 * arguments in place where {@code cl_khr_command_buffer_mutable_dispatch} allows it. The device is touched
 * once, after the body, so a replay that cannot run (a kernel, its order, its count or its arguments
 * differing from the capture) throws {@link GpuGraphMismatchException} before any device work and the
 * caller can capture the step again.
 *
 * <p>{@link #replayPatched(GpuGraphPatchPlan, int)} skips the body: a {@link GpuGraphPatchPlan} built
 * once over {@link #kernelDispatches()} derives each captured dispatch's step values and the graph
 * writes them into the changed nodes' parameters before one submission. Backends that cannot patch
 * recorded nodes refuse it, and a plan whose step pattern leaves the capture is declined, so the
 * caller falls back to {@link #replay(Runnable)}.
 *
 * <p>Buffers bound in a replay must stay open until {@link #close()}; a closed bound buffer is refused by
 * name. A graph is immutable in its structure: changed work sizes or kernels need a new capture, while
 * arguments and grid sizes a backend can update are picked up from the replay body. Not thread-safe:
 * launches, replays and close come from one thread; events a launch returns may be awaited from others.
 */
public interface GpuGraph extends AutoCloseable {

    /** Number of kernel dispatches captured. */
    int dispatchCount();

    /**
     * Submits the captured commands once, with the arguments they held at capture time, as one device
     * submission on {@code queue}. CUDA and HIP graphs may run on any open queue of the backend; a Vulkan
     * graph or an OpenCL command buffer runs only on the queue that captured it, because the recorded
     * command buffer belongs to that queue.
     *
     * @param queue an open queue of the backend the graph was captured on
     * @return an event that completes once the launch has finished, or null when the queue allocates none
     * @throws IllegalArgumentException when the queue is null, belongs to another backend, or is a queue
     *                                  this graph cannot run on
     * @throws IllegalStateException    when the queue or this graph is closed, or the queue is recording
     * @throws RuntimeException         the translated backend failure
     */
    GpuEvent launch(ComputeQueue queue, GpuEvent... waitOnEvents);

    /**
     * Runs {@code body} and submits the captured commands as one device submission. The body must enqueue
     * the captured command sequence, in order, with the arguments of this step.
     *
     * @throws GpuGraphMismatchException when the body's command sequence or its arguments differ from the
     *                                   capture, or a bound buffer is closed
     * @throws IllegalStateException     when the graph or its queue is closed, or another capture or
     *                                   replay is recording on the queue
     */
    void replay(Runnable body);

    /**
     * The graph's captured kernel dispatches in capture order, each addressed by its position in the
     * list, or null when the backend cannot report them (a verify-only graph keeps no argument
     * values). The captured argument values are the capture-time snapshot; a backend that updates
     * nodes from a replay body reports the values the nodes last held. The list is a fresh copy;
     * callers may keep it.
     */
    default java.util.List<GpuGraphDispatch> kernelDispatches() {
        return null;
    }

    /**
     * The current state of the captured kernel dispatch at position {@code index} (a
     * {@link GpuGraphDispatch#index()} of {@link #kernelDispatches()}), or null when the backend
     * cannot report it. The state reflects every node update applied since the capture.
     */
    default GpuGraphDispatch kernelDispatch(int index) {
        return null;
    }

    /**
     * The positions of the kernel dispatches whose captured node a body replay updated since the last
     * call, or null when the backend cannot report them. A body replay that changes a dispatch the
     * caller's patch plan does not cover would leave the replayed graph with a stale node, so a plan
     * builder requires every changed dispatch to be covered. Empty when the last body replay changed
     * nothing.
     */
    default int[] takeReplayChanges() {
        return null;
    }

    /**
     * True when this backend can apply a patch plan's values to the captured nodes and submit without
     * a body ({@link #replayPatched}): the capture reports kernel dispatches and the backend writes
     * changed node parameters. False for a verify-only graph, where the caller always replays through
     * {@link #replay(Runnable)}.
     */
    default boolean supportsPatchedReplay() {
        return false;
    }

    /**
     * Applies {@code plan}'s patches for the step at {@code pos} and submits the captured commands as
     * one device submission, without running a step body: the plan hands each captured dispatch's
     * step values to the graph, which writes the changed nodes' parameters before the submission. A
     * backend that cannot patch captured nodes (a verify-only graph whose recorded arguments are
     * fixed) returns false without touching the device.
     *
     * @return true when the graph was patched and submitted, false when the backend cannot patch or
     *         {@code plan} refused the step ({@link GpuGraphPatchPlan#patchStep}); the caller runs the
     *         step body instead
     * @throws IllegalStateException when the graph or its queue is closed, or another capture or
     *                               replay is recording on the queue
     */
    default boolean replayPatched(GpuGraphPatchPlan plan, int pos) {
        return false;
    }

    /**
     * Patches the captured nodes for the follow-up step at {@code nextPos} ahead of its launch, while
     * the last launched step at {@code lastPos} is still running on the device, so the follow-up's
     * patched replay pays no patch time on the critical path. On a backend with one executable graph
     * per capture this would update a running graph, so the default does nothing and the follow-up's
     * {@link #replayPatched} patches as usual. A backend with a second, idle instantiation patches
     * that one and submits it next; a declined follow-up pattern leaves the state the last launch
     * made, and the caller's next replay reconciles whatever moved. The patch plan stays
     * authoritative: a follow-up that runs at another position than the prepared one is simply
     * rewritten.
     *
     * @return true when the follow-up step's graph is prepared and the next {@link #replayPatched}
     *         will submit it, false when the backend does not prepare (the next replayPatched then
     *         patches its own graph)
     */
    default boolean preparePatched(GpuGraphPatchPlan plan, int nextPos, int lastPos) {
        return false;
    }

    /** True once {@link #close()} has run. */
    boolean isClosed();

    /**
     * Waits for the last submitted replay and releases the graph's device resources. Idempotent.
     *
     * @throws RuntimeException the translated backend failure
     */
    @Override
    void close();
}
