package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Pipeline-parallel execution of a sequence of stage modules: the batch is split along dimension 0 into micro-batches
 * that flow through the stages, each stage on its own fixed worker thread, so different stages work on different
 * micro-batches at once.
 *
 * <p><b>Placement.</b> Stage {@code i} runs on {@code backends.get(i)}; a tensor that arrives from another backend is
 * copied there through the host, and a null backend leaves the tensor where it is. Stages that
 * share a backend are serialized by that backend's lock. The last stage's outputs are concatenated
 * along dimension 0 on the caller's thread ({@link TensorOps#cat}).</p>
 *
 * <p><b>Autograd.</b> Graphs and grad mode are per thread ({@link ComputationGraph}), so the pipeline carries
 * gradients between the stage threads itself. {@link #forward} records when the caller's grad mode is on and every
 * stage is in training mode: each stage thread then runs its stage with grad mode on, turns every received
 * floating-point activation into a leaf (on stage 0 only when the input requires grad) and keeps that leaf and the
 * stage's output, the root of the micro-batch's graph on that thread. The result is then a leaf that requires grad
 * (when it is floating point), so the caller's loss records against it. After the caller's backward pass,
 * {@link #backward(Tensor)} with the gradient that reached the result (its {@code grad()}) backpropagates every
 * micro-batch through the stages in reverse order on their threads, hands each stage's leaf gradient to the stage
 * before it, and so accumulates the parameter gradients of every stage; it returns the gradient of the input.
 * Auxiliary loss terms a stage's forward pass leaves pending ({@link Module#takeAuxiliaryLoss()}, such as a mixture of
 * experts' load-balancing term) are taken on the stage's thread, kept per micro-batch and backpropagated by
 * {@link #backward(Tensor)} as terms added to the caller's loss. Otherwise the stages run under
 * {@link ComputationGraph#noGrad()}, pending auxiliary terms are discarded and the result does not require grad.</p>
 *
 * <p><b>Lifetime.</b> On each stage thread a recorded forward pass runs inside a {@link ComputationGraph.StepScope}.
 * The retained roots, leaves and scopes are released and closed per micro-batch on their threads: by
 * {@link #backward(Tensor)}, also when it fails; when a recorded forward pass fails or is aborted; by the next
 * {@link #forward} when {@code backward} was never called; and by {@link #close()}. Every intermediate tensor is
 * closed once the next stage has consumed it. Calls are serialized.</p>
 */
public final class PipelineParallel implements AutoCloseable {
    private final List<Module> stages;
    private final List<ComputeBackend> backends;
    private final int numMicroBatches;
    private final int numStages;
    private final List<ExecutorService> executors;
    private final AtomicBoolean closed;
    /**
     * The retained graphs of the last recorded forward pass that {@link #backward} has not consumed; guarded by this.
     */
    private Recording recording;

    public PipelineParallel(List<Module> stages, List<ComputeBackend> backends, int numMicroBatches) {
        if (stages == null || stages.isEmpty()) {
            throw new IllegalArgumentException("Stages list cannot be null or empty");
        }
        if (backends == null || backends.isEmpty()) {
            throw new IllegalArgumentException("Backends list cannot be null or empty");
        }
        if (stages.size() != backends.size()) {
            throw new IllegalArgumentException("Number of stages must match number of backends");
        }
        if (numMicroBatches < 1) {
            throw new IllegalArgumentException("Number of micro-batches must be at least 1");
        }

        this.stages = new ArrayList<>(stages);
        this.backends = new ArrayList<>(backends);
        this.numMicroBatches = numMicroBatches;
        this.numStages = stages.size();
        List<ExecutorService> perStage = new ArrayList<>(numStages);
        for (int i = 0; i < numStages; i++) {
            final int stageIdx = i;
            perStage.add(Executors.newSingleThreadExecutor(task -> stageThread(task, stageIdx)));
        }
        this.executors = perStage;
        this.closed = new AtomicBoolean(false);
    }

    /**
     * The pipeline's output for {@code input}: the stages applied in order to each micro-batch of
     * {@code ceil(batch / numMicroBatches)} rows (fewer micro-batches when the batch is smaller), concatenated along
     * dimension 0 as a new tensor the caller owns. When grad mode is on and every stage is in training mode the pass
     * is recorded and a floating-point result is a leaf that requires grad (see the class description); otherwise the
     * result does not require grad. The retained graphs of an earlier recorded pass that was never backpropagated are
     * released first. A failure in any stage stops the others and is rethrown, after the graphs this pass retained are
     * released.
     *
     * @throws IllegalArgumentException when {@code input} is null, rank 0 or has an empty batch dimension
     */
    public synchronized Tensor forward(Tensor input) {
        ensureOpen();
        if (input == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        if (input.shape().rank() == 0 || input.shape().dim(0) == 0) {
            throw new IllegalArgumentException("Input needs a non-empty batch dimension, got shape " + input.shape());
        }
        releasePending();

        boolean record = ComputationGraph.isGradEnabled() && allStagesTraining();
        long[] rows = microBatchRows(input.shape().dim(0));
        List<Tensor> microBatches = splitRows(input, rows);
        Recording rec = record ? new Recording(numStages, microBatches, input) : null;
        Tensor[] outputs = new Tensor[rows.length];
        try {
            runChain(forwardOrder(), microBatches, outputs, (stage, mb, received) ->
                forwardStep(stage, mb, received, rec));
            Tensor result = concatenate(outputs, record);
            if (rec != null) {
                rec.outputShape = result.shape();
                for (int mb = 0; mb < outputs.length; mb++) {
                    rec.outputRows[mb] = outputs[mb].shape().rank() == 0 ? 1 : outputs[mb].shape().dim(0);
                }
                recording = rec;
            }
            return result;
        } catch (RuntimeException | Error e) {
            if (rec != null) {
                release(rec, e);
            }
            throw e;
        } finally {
            closeAll(Arrays.asList(outputs));
            closeAll(microBatches);
        }
    }

    /**
     * Backpropagates the last recorded {@link #forward} pass: {@code gradOutput}, the gradient of the loss with respect
     * to the forward result (usually the result's {@code grad()} after the caller's backward pass), is sliced per
     * micro-batch along dimension 0; each stage, last to first, runs on its own thread the backward pass of each
     * micro-batch's auxiliary loss, when it kept one, and of its root with the gradient that arrived for it,
     * accumulating into its parameters' gradients, and hands the gradient of its leaf to the stage before it (copied
     * through the host across backends). The retained graphs are then released, also when a stage
     * fails. {@code gradOutput} stays the caller's.
     *
     * @return the gradient with respect to the forward pass's input, a new tensor on the input's backend the caller
     *         owns (zeros for micro-batches no gradient reached), or null when the input did not require grad or no
     *         gradient reached it
     * @throws IllegalArgumentException when {@code gradOutput} is null or does not have the forward result's shape
     *                                  (the recorded pass is then kept)
     * @throws IllegalStateException    when no recorded forward pass is pending: the last forward pass did not record
     *                                  (grad mode off or a stage in evaluation mode), or was already backpropagated
     */
    public synchronized Tensor backward(Tensor gradOutput) {
        ensureOpen();
        if (gradOutput == null) {
            throw new IllegalArgumentException("Gradient cannot be null");
        }
        Recording rec = recording;
        if (rec == null) {
            throw new IllegalStateException("No recorded forward pass to backpropagate: forward records only with grad "
                + "mode on and every stage in training mode, and each recorded pass is backpropagated once");
        }
        if (!gradOutput.shape().equals(rec.outputShape)) {
            throw new IllegalArgumentException("Gradient shape " + gradOutput.shape() + " does not match the forward "
                + "result's shape " + rec.outputShape);
        }
        recording = null;

        Tensor[] inputGrads = new Tensor[rec.size()];
        List<Tensor> slices = new ArrayList<>();
        Tensor result;
        try {
            slices.addAll(splitRows(gradOutput, rec.outputRows));
            runChain(backwardOrder(), slices, inputGrads, (stage, mb, received) ->
                backwardStep(rec, stage, mb, received));
            result = rec.inputRequiresGrad ? inputGradient(rec, inputGrads) : null;
        } catch (RuntimeException | Error e) {
            closeAll(Arrays.asList(inputGrads));
            closeAll(slices);
            release(rec, e);
            throw e;
        }
        closeAll(Arrays.asList(inputGrads));
        closeAll(slices);
        try {
            release(rec, null);
        } catch (RuntimeException | Error e) {
            if (result != null) {
                result.close();
            }
            throw e;
        }
        return result;
    }

    // ------------------------------------------------------------------ stage steps

    /**
     * Stage {@code stage} applied to micro-batch {@code mb}, {@code received} on the stage's thread: a detached handle
     * of the stage's output, owned by the receiver. When {@code rec} is not null the step records (grad mode on), a
     * floating-point input is made a leaf (stage 0 only when the pipeline input requires grad) and the leaf, the output
     * when it requires grad, and the auxiliary loss the stage's forward pass left pending when it is a recorded term
     * ({@link Module#takeAuxiliaryLoss()}), are kept in the stage's record; otherwise the step runs under
     * {@code noGrad} and the pending auxiliary terms are discarded. The cross-backend copy and an output that is not
     * kept are closed, never a tensor the stage module holds.
     */
    private Tensor forwardStep(int stage, int mb, Tensor received, Recording rec) {
        Module module = stages.get(stage);
        StageRecord sr = rec == null ? null : rec.stages[stage];
        Tensor local = null;
        Tensor leaf = null;
        Tensor output = null;
        try (GradMode mode = GradMode.of(sr != null)) {
            if (sr != null && sr.scope == null) {
                sr.scope = ComputationGraph.current().step();
            }
            local = onBackend(received, backends.get(stage));
            Tensor in = local;
            if (sr != null && local.dtype().isFloatingPoint() && (stage > 0 || rec.inputRequiresGrad)) {
                leaf = asLeaf(local);
                sr.leaves[mb] = leaf;
                in = leaf;
            }
            output = module.forward(in);
            if (output == null) {
                throw new IllegalStateException("Stage " + stage + " returned null for micro-batch " + mb);
            }
            if (sr != null && output.requiresGrad()) {
                sr.roots[mb] = output;
                sr.ownedRoots[mb] = ComputationGraph.current().getNode(output) != null;
            }
            takeAuxiliaryLoss(module, sr, mb);
            return output.detach();
        } finally {
            Tensor root = sr == null ? null : sr.roots[mb];
            if (local != null && local != received && local != leaf && local != root && !holds(module, local)) {
                local.close();
            }
            if (output != null && output != received && output != local && output != leaf && output != root
                    && !holds(module, output)) {
                output.close();
            }
        }
    }

    /**
     * The backward pass of micro-batch {@code mb} through stage {@code stage} on the stage's thread: the backward pass
     * of the micro-batch's auxiliary loss, when the stage kept one (seeded with 1, as a term added to the loss), then
     * the root's backward pass seeded with {@code gradient} (skipped when no gradient arrived, null). Both accumulate
     * into the stage's parameter gradients and the leaf's gradient; the auxiliary pass retains the graph when the
     * root's pass follows, which then releases the micro-batch's graph. Returns the gradient of the stage's leaf,
     * taken out of the leaf's slot and owned by the receiver, or null without a leaf or a gradient.
     */
    private static Tensor backwardStep(Recording rec, int stage, int mb, Tensor gradient) {
        StageRecord sr = rec.stages[stage];
        Tensor root = sr.roots[mb];
        boolean seeded = root != null && gradient != null;
        Tensor auxiliary = sr.auxiliary[mb];
        if (auxiliary != null) {
            sr.auxiliary[mb] = null;
            try {
                ComputationGraph.current().backward(auxiliary, null, seeded);
            } finally {
                auxiliary.close();
            }
        }
        if (seeded) {
            Tensor seed = onBackend(gradient, root.backend());
            try {
                ComputationGraph.current().backward(root, seed);
            } finally {
                if (seed != gradient) {
                    seed.close();
                }
            }
        }
        Tensor leaf = sr.leaves[mb];
        if (leaf == null || leaf.grad() == null) {
            return null;
        }
        Tensor leafGrad = leaf.grad();
        leaf.setGrad(null);
        return leafGrad;
    }

    /**
     * The gradient of the pipeline input (FLOAT32, as every gradient): the stage-0 leaf gradients moved to the input's
     * backend, zeros for micro-batches without one, concatenated along dimension 0 without recording; null when no
     * micro-batch has one.
     */
    private static Tensor inputGradient(Recording rec, Tensor[] inputGrads) {
        boolean any = false;
        for (Tensor g : inputGrads) {
            any |= g != null;
        }
        if (!any) {
            return null;
        }
        List<Tensor> parts = new ArrayList<>(inputGrads.length);
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (int mb = 0; mb < inputGrads.length; mb++) {
                Tensor g = inputGrads[mb];
                parts.add(g == null ? Tensor.zeros(rec.inputBackend, rec.inputShapes[mb], DType.FLOAT32)
                    : onBackendCopy(g, rec.inputBackend));
            }
            return parts.size() == 1 ? parts.get(0).detach() : TensorOps.cat(parts, 0);
        } finally {
            closeAll(parts);
        }
    }

    // ------------------------------------------------------------------ chain execution

    /** One step of a chain position: the tensor it hands on for {@code received}, which it does not close. */
    @FunctionalInterface
    private interface StageWork {
        Tensor apply(int stage, int microBatch, Tensor received);
    }

    private int[] forwardOrder() {
        int[] order = new int[numStages];
        for (int i = 0; i < numStages; i++) {
            order[i] = i;
        }
        return order;
    }

    private int[] backwardOrder() {
        int[] order = new int[numStages];
        for (int i = 0; i < numStages; i++) {
            order[i] = numStages - 1 - i;
        }
        return order;
    }

    /**
     * Runs one task per chain position, position {@code p} on the thread of stage {@code order[p]}, linked by queues,
     * and waits for every task to finish. Position 0 reads {@code sources} (left open), later positions take what the
     * previous position handed on (and close it after their step); the last position's results go to {@code sinks}.
     * A failing position stops the others without interrupts: it sets {@code aborted}, which the other positions check
     * before each micro-batch, and every non-last position ends by sending {@link Link#END} downstream, so no position
     * waits forever. Only after all tasks have finished are the tensors left in the links closed, so none is enqueued
     * after the cleanup. The first failure is rethrown, later ones added as suppressed.
     */
    private void runChain(int[] order, List<Tensor> sources, Tensor[] sinks, StageWork work) {
        List<BlockingQueue<Link>> links = new ArrayList<>();
        for (int i = 1; i < order.length; i++) {
            links.add(new LinkedBlockingQueue<>());
        }
        AtomicBoolean aborted = new AtomicBoolean(false);
        List<Future<?>> tasks = new ArrayList<>();
        Throwable failure = null;
        try {
            for (int p = 0; p < order.length; p++) {
                final int position = p;
                tasks.add(executors.get(order[p]).submit(() -> {
                    runPosition(position, order, sources, links, sinks, aborted, work);
                    return null;
                }));
            }
        } catch (RuntimeException e) {
            aborted.set(true);
            failure = e;
        }
        failure = awaitAll(tasks, aborted, failure);
        for (BlockingQueue<Link> link : links) {
            for (Link pending : link) {
                if (pending != Link.END && pending.tensor != null) {
                    pending.tensor.close();
                }
            }
            link.clear();
        }
        rethrow(failure);
    }

    /**
     * Chain position {@code position}: applies {@code work} for stage {@code order[position]} to every micro-batch in
     * order, stopping early when the chain is aborted or its upstream ended early; marks the chain aborted when it
     * fails, and a non-last position always ends by sending {@link Link#END}.
     */
    private static void runPosition(int position, int[] order, List<Tensor> sources, List<BlockingQueue<Link>> links,
                                    Tensor[] sinks, AtomicBoolean aborted, StageWork work)
            throws InterruptedException {
        int stage = order[position];
        boolean first = position == 0;
        boolean last = position == order.length - 1;
        try {
            for (int mb = 0; mb < sources.size() && !aborted.get(); mb++) {
                Tensor received;
                if (first) {
                    received = sources.get(mb);
                } else {
                    Link link = links.get(position - 1).take();
                    if (link == Link.END) {
                        break;
                    }
                    received = link.tensor;
                }
                Tensor produced = null;
                try {
                    produced = work.apply(stage, mb, received);
                } finally {
                    if (!first && received != null && received != produced) {
                        received.close();
                    }
                }
                if (last) {
                    sinks[mb] = produced;
                } else {
                    links.get(position).add(new Link(produced));
                }
            }
        } catch (RuntimeException | Error | InterruptedException e) {
            aborted.set(true);
            throw e;
        } finally {
            if (!last) {
                links.get(position).add(Link.END);
            }
        }
    }

    /**
     * Waits for every task, setting {@code aborted} when interrupted (the wait continues, so no task outlives the
     * call; the interrupt is restored afterwards). Returns {@code failure} with the task failures collected.
     */
    private static Throwable awaitAll(List<Future<?>> tasks, AtomicBoolean aborted, Throwable failure) {
        boolean interrupted = false;
        for (Future<?> task : tasks) {
            while (true) {
                try {
                    task.get();
                    break;
                } catch (InterruptedException e) {
                    interrupted = true;
                    if (aborted != null) {
                        aborted.set(true);
                    }
                } catch (ExecutionException e) {
                    failure = collect(failure, e.getCause());
                    break;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
            if (failure == null && aborted != null) {
                failure = new IllegalStateException("Pipeline execution interrupted");
            }
        }
        return failure;
    }

    // ------------------------------------------------------------------ retained graphs

    /** Releases the retained graphs of a recorded pass that {@link #backward} never consumed. */
    private void releasePending() {
        Recording rec = recording;
        if (rec != null) {
            recording = null;
            release(rec, null);
        }
    }

    /**
     * Releases every stage's retained graphs on the stage's own thread ({@link #releaseStage}) and waits for all of
     * them. With {@code primary} (a failure being propagated) the release failures are added to it as suppressed;
     * otherwise the first is thrown.
     */
    private void release(Recording rec, Throwable primary) {
        List<Future<?>> tasks = new ArrayList<>(numStages);
        Throwable failure = null;
        for (int s = 0; s < numStages; s++) {
            StageRecord sr = rec.stages[s];
            try {
                tasks.add(executors.get(s).submit(() -> {
                    releaseStage(sr);
                    return null;
                }));
            } catch (RuntimeException e) {
                failure = collect(failure, e);
            }
        }
        failure = awaitAll(tasks, null, failure);
        if (failure == null) {
            return;
        }
        if (primary != null) {
            if (failure != primary) {
                primary.addSuppressed(failure);
            }
            return;
        }
        rethrow(failure);
    }

    /**
     * On the stage's thread: closes the stage's {@link ComputationGraph.StepScope}, which removes the nodes the pass
     * recorded there that no backward pass released and closes their outputs, then closes the leaves with their
     * gradients, the roots that were node outputs and the auxiliary losses no backward pass consumed. Everything is
     * tried; the first failure is thrown with later ones suppressed.
     */
    private static void releaseStage(StageRecord sr) {
        RuntimeException failure = null;
        if (sr.scope != null) {
            try {
                sr.scope.close();
            } catch (RuntimeException e) {
                failure = e;
            }
            sr.scope = null;
        }
        for (int mb = 0; mb < sr.leaves.length; mb++) {
            try {
                Tensor leaf = sr.leaves[mb];
                sr.leaves[mb] = null;
                if (leaf != null) {
                    Tensor grad = leaf.grad();
                    if (grad != null) {
                        leaf.setGrad(null);
                        grad.close();
                    }
                    leaf.close();
                }
                Tensor root = sr.roots[mb];
                sr.roots[mb] = null;
                if (root != null && sr.ownedRoots[mb]) {
                    root.close();
                }
                Tensor auxiliary = sr.auxiliary[mb];
                sr.auxiliary[mb] = null;
                if (auxiliary != null) {
                    auxiliary.close();
                }
            } catch (RuntimeException e) {
                failure = (RuntimeException) collect(failure, e);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    // ------------------------------------------------------------------ helpers

    private boolean allStagesTraining() {
        for (Module stage : stages) {
            if (!stage.isTraining()) {
                return false;
            }
        }
        return true;
    }

    /** The row counts of the micro-batches: {@code ceil(batch / numMicroBatches)} rows each, the last one fewer. */
    private long[] microBatchRows(long batchSize) {
        long microBatchSize = (batchSize + numMicroBatches - 1) / numMicroBatches;
        List<Long> rows = new ArrayList<>();
        for (int i = 0; i < numMicroBatches; i++) {
            long start = (long) i * microBatchSize;
            if (start >= batchSize) {
                break;
            }
            rows.add(Math.min(start + microBatchSize, batchSize) - start);
        }
        long[] result = new long[rows.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = rows.get(i);
        }
        return result;
    }

    /**
     * Consecutive views of {@code t} along dimension 0 with the given row counts, made without recording (a single
     * part is a detached handle of {@code t}, which may be rank 0); the caller closes them.
     */
    private static List<Tensor> splitRows(Tensor t, long[] rows) {
        List<Tensor> parts = new ArrayList<>(rows.length);
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            if (rows.length == 1) {
                parts.add(t.detach());
                return parts;
            }
            long start = 0;
            for (long count : rows) {
                parts.add(t.slice(0, start, start + count));
                start += count;
            }
            return parts;
        } catch (RuntimeException e) {
            closeAll(parts);
            throw e;
        }
    }

    /**
     * The last stage's outputs concatenated along dimension 0 as a new tensor, without recording
     * ({@link TensorOps#cat}, on the device when the train kernels are available; a single output as a new handle of
     * itself); with {@code asLeaf}, a floating-point result is a leaf that requires grad.
     */
    private static Tensor concatenate(Tensor[] outputs, boolean asLeaf) {
        if (outputs.length == 0) {
            throw new IllegalStateException("No outputs produced");
        }
        for (int mb = 0; mb < outputs.length; mb++) {
            if (outputs[mb] == null) {
                throw new IllegalStateException("No output produced for micro-batch " + mb);
            }
        }
        Tensor raw;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            raw = outputs.length == 1 ? outputs[0].detach() : TensorOps.cat(Arrays.asList(outputs), 0);
        }
        if (!asLeaf || !raw.dtype().isFloatingPoint()) {
            return raw;
        }
        Tensor leaf = raw.withRequiresGrad(true);
        if (leaf != raw) {
            raw.close();
        }
        return leaf;
    }

    /** A new leaf over {@code t}'s storage that requires grad ({@code detach().withRequiresGrad(true)}). */
    private static Tensor asLeaf(Tensor t) {
        Tensor detached = t.detach();
        Tensor leaf = detached.withRequiresGrad(true);
        if (leaf != detached) {
            detached.close();
        }
        return leaf;
    }

    /**
     * On the stage's thread, takes the auxiliary loss terms {@code module}'s forward pass left pending
     * ({@link Module#takeAuxiliaryLoss()}): a recorded term is kept in {@code sr} for micro-batch {@code mb}; any other
     * (a constant of a pass that does not record) is released and closed.
     */
    private static void takeAuxiliaryLoss(Module module, StageRecord sr, int mb) {
        Tensor auxiliary = module.takeAuxiliaryLoss();
        if (auxiliary == null) {
            return;
        }
        ComputationGraph graph = ComputationGraph.current();
        if (sr != null && graph.getNode(auxiliary) != null) {
            sr.auxiliary[mb] = auxiliary;
            return;
        }
        try {
            graph.release(auxiliary);
        } finally {
            auxiliary.close();
        }
    }

    /** Whether {@code t} is a parameter tensor or a buffer of {@code module}, which the pipeline never closes. */
    private static boolean holds(Module module, Tensor t) {
        for (Parameter param : module.parameters()) {
            if (param.tensor() == t) {
                return true;
            }
        }
        for (Tensor buffer : module.namedBuffers().values()) {
            if (buffer == t) {
                return true;
            }
        }
        return false;
    }

    /** {@code t} on {@code target}: itself when {@code target} is null or already its backend, else a host copy. */
    private static Tensor onBackend(Tensor t, ComputeBackend target) {
        if (target == null || t.backend() == target) {
            return t;
        }
        TensorStorage storage = TensorStorage.fromHost(target, t.toByteBuffer(), t.dtype());
        return Tensor.fromStorage(target, t.shape(), t.dtype(), storage);
    }

    /** A new handle of {@code t} on {@code target}: a detached handle on the same backend, else a host copy. */
    private static Tensor onBackendCopy(Tensor t, ComputeBackend target) {
        Tensor moved = onBackend(t, target);
        return moved == t ? t.detach() : moved;
    }

    /** {@code first} with {@code next} added as suppressed, or {@code next} when there is no first failure. */
    private static Throwable collect(Throwable first, Throwable next) {
        if (first == null) {
            return next;
        }
        if (next != first) {
            first.addSuppressed(next);
        }
        return first;
    }

    private static void rethrow(Throwable failure) {
        if (failure == null) {
            return;
        }
        if (failure instanceof RuntimeException) {
            throw (RuntimeException) failure;
        }
        if (failure instanceof Error) {
            throw (Error) failure;
        }
        throw new IllegalStateException("Pipeline stage failed", failure);
    }

    /** Closes every non-null tensor of {@code tensors}. */
    private static void closeAll(List<Tensor> tensors) {
        for (Tensor t : tensors) {
            if (t != null) {
                t.close();
            }
        }
    }

    /** A daemon worker thread for stage {@code stageIdx}, so an unclosed pipeline does not keep the JVM alive. */
    private static Thread stageThread(Runnable task, int stageIdx) {
        Thread thread = new Thread(task, "pipeline-stage-" + stageIdx);
        thread.setDaemon(true);
        return thread;
    }

    /**
     * One entry of a link between chain positions: a tensor (null for none), or {@link #END} when the sender stopped.
     */
    private static final class Link {
        static final Link END = new Link(null);

        final Tensor tensor;

        Link(Tensor tensor) {
            this.tensor = tensor;
        }
    }

    /** The grad mode a stage step runs in: on for a recorded pass, off otherwise; restored on close. */
    private static final class GradMode implements AutoCloseable {
        private final ComputationGraph.EnableGradScope on;
        private final ComputationGraph.NoGradScope off;

        private GradMode(ComputationGraph.EnableGradScope on, ComputationGraph.NoGradScope off) {
            this.on = on;
            this.off = off;
        }

        static GradMode of(boolean enabled) {
            return enabled ? new GradMode(ComputationGraph.enableGrad(), null)
                : new GradMode(null, ComputationGraph.noGrad());
        }

        @Override
        public void close() {
            if (on != null) {
                on.close();
            }
            if (off != null) {
                off.close();
            }
        }
    }

    /** What one stage retains of a recorded pass, per micro-batch; used only on the stage's own thread. */
    private static final class StageRecord {
        final Tensor[] leaves;
        final Tensor[] roots;
        final boolean[] ownedRoots;
        final Tensor[] auxiliary;
        ComputationGraph.StepScope scope;

        StageRecord(int microBatches) {
            this.leaves = new Tensor[microBatches];
            this.roots = new Tensor[microBatches];
            this.ownedRoots = new boolean[microBatches];
            this.auxiliary = new Tensor[microBatches];
        }
    }

    /** The retained state of one recorded forward pass. */
    private static final class Recording {
        final StageRecord[] stages;
        final boolean inputRequiresGrad;
        final ComputeBackend inputBackend;
        final Shape[] inputShapes;
        final long[] outputRows;
        Shape outputShape;

        Recording(int numStages, List<Tensor> microBatches, Tensor input) {
            this.stages = new StageRecord[numStages];
            for (int s = 0; s < numStages; s++) {
                stages[s] = new StageRecord(microBatches.size());
            }
            this.inputRequiresGrad = input.requiresGrad();
            this.inputBackend = input.backend();
            this.inputShapes = new Shape[microBatches.size()];
            for (int mb = 0; mb < inputShapes.length; mb++) {
                inputShapes[mb] = microBatches.get(mb).shape();
            }
            this.outputRows = new long[microBatches.size()];
        }

        int size() {
            return outputRows.length;
        }
    }

    public List<Module> stages() {
        ensureOpen();
        return new ArrayList<>(stages);
    }

    public int numStages() {
        ensureOpen();
        return numStages;
    }

    public int numMicroBatches() {
        ensureOpen();
        return numMicroBatches;
    }

    public Module stage(int index) {
        ensureOpen();
        if (index < 0 || index >= numStages) {
            throw new IndexOutOfBoundsException("Stage index out of bounds: " + index);
        }
        return stages.get(index);
    }

    public ComputeBackend backend(int stageIndex) {
        ensureOpen();
        if (stageIndex < 0 || stageIndex >= numStages) {
            throw new IndexOutOfBoundsException("Stage index out of bounds: " + stageIndex);
        }
        return backends.get(stageIndex);
    }

    public long parameterCount() {
        ensureOpen();
        long count = 0;
        for (Module stage : stages) {
            count += stage.parameterCount();
        }
        return count;
    }

    public void train() {
        ensureOpen();
        for (Module stage : stages) {
            stage.train();
        }
    }

    public void eval() {
        ensureOpen();
        for (Module stage : stages) {
            stage.eval();
        }
    }

    public void zeroGrad() {
        ensureOpen();
        for (Module stage : stages) {
            stage.zeroGrad();
        }
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("PipelineParallel has been closed");
        }
    }

    /**
     * Releases the retained graphs of a recorded pass that was never backpropagated, then shuts the stage threads
     * down; idempotent. A release failure is thrown after the threads are shut down.
     */
    @Override
    public synchronized void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        try {
            releasePending();
        } finally {
            for (ExecutorService executor : executors) {
                executor.shutdown();
            }
        }
    }
}
