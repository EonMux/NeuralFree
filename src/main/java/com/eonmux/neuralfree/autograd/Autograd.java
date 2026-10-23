package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Supplier;

/**
 * The recorder every differentiable operation uses, and {@code torch.autograd.grad}.
 *
 * <p><b>Recording rule.</b> An operation records a node exactly when {@link #needsGrad} holds for its tensor inputs:
 * grad mode is on ({@link ComputationGraph#isGradEnabled()}) and some input requires grad. Its result then requires
 * grad and carries the node's gradient function; otherwise the result does not require grad. The pattern of a
 * recording operation is</p>
 * <pre>{@code
 * if (!Autograd.needsGrad(a, b)) {
 *     return opRaw(a, b);
 * }
 * Tensor raw = Autograd.untracked(() -> opRaw(a, b));
 * return Autograd.record(raw, Arrays.asList(a, b), out -> new OpGrad(...));
 * }</pre>
 *
 * <p><b>Create graph.</b> Inside a backward pass with {@code createGraph = true} ({@link #isCreatingGraph()}) gradient
 * functions run with grad mode on, so the operations they call record. {@link #reattach} gives a gradient function's
 * saved handle (a detached view) a recorded path back to the forward tensor it was saved from, so second-order
 * gradients reach the forward graph.</p>
 */
public final class Autograd {

    private Autograd() {
    }

    /** Whether an operation on {@code a} records: grad mode is on and {@code a} (when not null) requires grad. */
    public static boolean needsGrad(Tensor a) {
        return requires(a) && ComputationGraph.isGradEnabled();
    }

    /** Whether an operation on {@code a} and {@code b} records; null inputs are ignored. */
    public static boolean needsGrad(Tensor a, Tensor b) {
        return (requires(a) || requires(b)) && ComputationGraph.isGradEnabled();
    }

    /** Whether an operation on {@code a}, {@code b} and {@code c} records; null inputs are ignored. */
    public static boolean needsGrad(Tensor a, Tensor b, Tensor c) {
        return (requires(a) || requires(b) || requires(c)) && ComputationGraph.isGradEnabled();
    }

    /** Whether an operation on {@code inputs} records; null entries are ignored. */
    public static boolean needsGrad(List<Tensor> inputs) {
        if (inputs == null) {
            return false;
        }
        for (Tensor input : inputs) {
            if (requires(input)) {
                return ComputationGraph.isGradEnabled();
            }
        }
        return false;
    }

    private static boolean requires(Tensor t) {
        return t != null && t.requiresGrad();
    }

    /**
     * Runs {@code body} with grad mode off and returns its result: the raw computation of a recorded operation, whose
     * internal views and helper operations must not record.
     */
    public static <T> T untracked(Supplier<T> body) {
        Objects.requireNonNull(body, "body");
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            return body.get();
        }
    }

    /**
     * Records {@code raw}, the new result of an operation on {@code inputs}, and returns it marked as requiring grad.
     * {@code raw} is handed over: when marking makes a new handle, {@code raw} is closed; a {@code raw} that is one of
     * the inputs (a view operation returning its operand) is left open and the result is a distinct handle. The
     * gradient function is built from the marked result (which it may save), set as the result's gradient function and
     * recorded in this thread's graph. On failure the function's saved tensors are cleared, the result is closed and
     * the exception is rethrown.
     *
     * @throws IllegalStateException    when {@link #needsGrad(List)} does not hold for {@code inputs} ({@code raw} is
     *                                  closed unless it is an input), or {@code gradFn} returns null
     * @throws IllegalArgumentException when {@code raw} already carries a gradient function (it was recorded; compute
     *                                  it inside {@link #untracked})
     */
    public static Tensor record(Tensor raw, List<Tensor> inputs, Function<Tensor, GradientFunction> gradFn) {
        Objects.requireNonNull(raw, "raw");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(gradFn, "gradFn");
        requireRecordable(Collections.singletonList(raw), inputs);
        Tensor out = marked(raw, inputs);
        GradientFunction fn = null;
        try {
            fn = gradFn.apply(out);
            if (fn == null) {
                throw new IllegalStateException("The gradient function factory returned null");
            }
            out.setGradFn(fn);
            ComputationGraph.current().recordNode(out, fn, inputs, false);
            return out;
        } catch (RuntimeException e) {
            if (fn != null) {
                clearQuietly(fn, e);
            }
            closeQuietly(out, e);
            throw e;
        }
    }

    /**
     * The multi-output form of {@link #record} (used by {@code split}): records one node per result, each with the
     * gradient function {@code gradFn} builds for its position, all over {@code inputs}. The results are handed over as
     * in {@link #record}; a result that is an input or repeats an earlier result gets a distinct handle. All or
     * nothing: on failure no node stays recorded, the built functions' saved tensors are cleared, every result is
     * closed and the exception is rethrown.
     *
     * @throws IllegalStateException    when {@link #needsGrad(List)} does not hold for {@code inputs}, or
     *                                  {@code gradFn} returns null
     * @throws IllegalArgumentException when a result already carries a gradient function
     */
    public static List<Tensor> recordEach(List<Tensor> raws, List<Tensor> inputs,
                                          IntFunction<GradientFunction> gradFn) {
        Objects.requireNonNull(raws, "raws");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(gradFn, "gradFn");
        requireRecordable(raws, inputs);
        List<Tensor> outs = markedAll(raws, inputs);
        ComputationGraph graph = ComputationGraph.current();
        List<GradientFunction> built = new ArrayList<>(outs.size());
        List<Tensor> recorded = new ArrayList<>(outs.size());
        try {
            for (int i = 0; i < outs.size(); i++) {
                GradientFunction fn = gradFn.apply(i);
                if (fn == null) {
                    throw new IllegalStateException("The gradient function factory returned null for result " + i);
                }
                built.add(fn);
                Tensor out = outs.get(i);
                out.setGradFn(fn);
                graph.recordNode(out, fn, inputs, false);
                recorded.add(out);
            }
            return outs;
        } catch (RuntimeException e) {
            graph.forget(recorded);
            for (GradientFunction fn : built) {
                clearQuietly(fn, e);
            }
            for (Tensor out : outs) {
                closeQuietly(out, e);
            }
            throw e;
        }
    }

    /**
     * A handle of a tensor saved for backward that keeps the path to {@code origin}, the forward tensor it was saved
     * from. Outside a create-graph pass, while grad mode is off, or when {@code origin} does not require grad, returns
     * {@code handle} itself (borrowed; nothing is recorded). Otherwise returns a new handle sharing {@code handle}'s
     * storage, recorded as an identity node whose input is {@code origin} (by identity; {@code origin} may be closed)
     * and whose output the graph owns and closes when the node leaves the graph; the caller borrows it and never
     * closes it.
     *
     * @throws IllegalArgumentException when the shapes of {@code handle} and {@code origin} differ
     */
    public static Tensor reattach(Tensor handle, Tensor origin) {
        Objects.requireNonNull(handle, "handle");
        Objects.requireNonNull(origin, "origin");
        if (!tracking(origin)) {
            return handle;
        }
        return identity(handle, origin, true);
    }

    /**
     * A new handle of {@code t} owned by the caller: a detached handle, or inside a create-graph pass, when {@code t}
     * requires grad, the output of a recorded identity node whose input is {@code t} (the caller owns it like any
     * other node output it computed).
     */
    static Tensor secondHandle(Tensor t) {
        if (!tracking(t)) {
            return t.detach();
        }
        return identity(t, t, false);
    }

    /** Whether this thread is inside {@code backward(..., createGraph = true)} or {@code grad(..., true)}. */
    public static boolean isCreatingGraph() {
        return ComputationGraph.isCreatingGraph();
    }

    /**
     * {@code torch.autograd.grad}: the gradients of {@code output} with respect to {@code inputs} (leaves or node
     * outputs of this thread's graph), each owned by the caller, null where no gradient arrives. No gradient slot is
     * written. {@code gradOutput} is the seed (null means ones; it stays the caller's). Only the nodes with a path to
     * some input run; the gradient arriving at a node-output input is captured as a new reference while routing
     * continues to inputs further upstream. Without {@code retainGraph} the part of the graph of {@code output} that is
     * not upstream of an input is released afterwards (the nodes the pass ran and the branches that reach no input),
     * like {@code retain_graph=False}; the graph a node-output input was computed from stays, so the input can still be
     * backpropagated ({@code g = grad(loss, [h]); h.backward(g)}) or released. {@code createGraph} implies {@code retainGraph}: the gradients are recorded (also when
     * called inside {@link ComputationGraph#noGrad}), so they can be differentiated again, and the nodes the pass
     * recorded follow the lifetime rule of {@link ComputationGraph}.
     *
     * @throws IllegalArgumentException      when {@code output} is null, {@code inputs} is null or empty, an input is
     *                                       null or does not require grad, or a shape does not match
     * @throws IllegalStateException         when {@code output} does not require grad, or {@code output}, an input or a
     *                                       tensor the pass reaches is a released non-leaf
     * @throws UnsupportedOperationException when the pass would run a checkpointed segment, or with {@code createGraph}
     *                                       a node whose gradient function does not support it (thrown before any
     *                                       gradient is computed)
     */
    public static List<Tensor> grad(Tensor output, List<Tensor> inputs, Tensor gradOutput, boolean retainGraph,
                                    boolean createGraph) {
        return ComputationGraph.current().gradients(output, inputs, gradOutput, retainGraph || createGraph,
            createGraph);
    }

    // ------------------------------------------------------------------ helpers

    private static boolean tracking(Tensor origin) {
        return ComputationGraph.isCreatingGraph() && ComputationGraph.isGradEnabled() && origin.requiresGrad();
    }

    /** A new handle of {@code source} recorded as the output of an identity node over {@code origin}. */
    private static Tensor identity(Tensor source, Tensor origin, boolean graphOwned) {
        if (!source.shape().equals(origin.shape())) {
            throw new IllegalArgumentException("Cannot reattach a handle of shape " + source.shape()
                + " to a tensor of shape " + origin.shape());
        }
        Tensor detached = source.detach();
        Tensor out;
        try {
            out = detached.withRequiresGrad(true);
        } finally {
            detached.close();
        }
        try {
            ReattachGrad fn = new ReattachGrad();
            out.setGradFn(fn);
            ComputationGraph.current().recordNode(out, fn, Collections.singletonList(origin), graphOwned);
            return out;
        } catch (RuntimeException e) {
            closeQuietly(out, e);
            throw e;
        }
    }

    /** Throws when {@code inputs} do not record, first closing every raw result that is not an input. */
    private static void requireRecordable(List<Tensor> raws, List<Tensor> inputs) {
        if (needsGrad(inputs)) {
            return;
        }
        IllegalStateException e = new IllegalStateException("Autograd.record requires grad mode and an input that "
            + "requires grad; check Autograd.needsGrad first");
        for (Tensor raw : raws) {
            if (raw != null && !containsIdentity(inputs, raw)) {
                closeQuietly(raw, e);
            }
        }
        throw e;
    }

    /** {@code raw} marked as requiring grad, as a handle distinct from every input; see {@link #record}. */
    private static Tensor marked(Tensor raw, List<Tensor> inputs) {
        if (containsIdentity(inputs, raw)) {
            return distinctMarked(raw);
        }
        if (raw.gradFn() != null) {
            throw new IllegalArgumentException("The raw result already carries a gradient function; compute it "
                + "inside Autograd.untracked so it is recorded once");
        }
        Tensor out = raw.withRequiresGrad(true);
        if (out != raw) {
            raw.close();
        }
        return out;
    }

    /** A new handle of {@code t} that requires grad; {@code t} stays as it is. */
    private static Tensor distinctMarked(Tensor t) {
        Tensor detached = t.detach();
        try {
            return detached.withRequiresGrad(true);
        } finally {
            detached.close();
        }
    }

    /**
     * Every raw result marked as by {@link #marked}, a repeated result getting a distinct handle of its first marked
     * form. On failure the results marked so far and the raws not yet handled (except inputs) are closed.
     */
    private static List<Tensor> markedAll(List<Tensor> raws, List<Tensor> inputs) {
        List<Tensor> outs = new ArrayList<>(raws.size());
        Map<Tensor, Tensor> firstMarked = new IdentityHashMap<>();
        int handled = 0;
        try {
            for (Tensor raw : raws) {
                Objects.requireNonNull(raw, "raw result " + handled + " must not be null");
                Tensor earlier = firstMarked.get(raw);
                Tensor out = earlier != null ? distinctMarked(earlier) : marked(raw, inputs);
                firstMarked.putIfAbsent(raw, out);
                outs.add(out);
                handled++;
            }
            return outs;
        } catch (RuntimeException e) {
            for (Tensor out : outs) {
                closeQuietly(out, e);
            }
            for (int i = handled; i < raws.size(); i++) {
                Tensor raw = raws.get(i);
                if (raw != null && !containsIdentity(inputs, raw) && !firstMarked.containsKey(raw)) {
                    closeQuietly(raw, e);
                }
            }
            throw e;
        }
    }

    private static boolean containsIdentity(List<Tensor> tensors, Tensor t) {
        for (Tensor candidate : tensors) {
            if (candidate == t) {
                return true;
            }
        }
        return false;
    }

    /** Closes {@code t}, attaching a close failure to {@code primary}, the failure being propagated. */
    private static void closeQuietly(Tensor t, RuntimeException primary) {
        try {
            t.close();
        } catch (RuntimeException e) {
            primary.addSuppressed(e);
        }
    }

    /** Clears {@code fn}'s saved tensors, attaching a failure to {@code primary}, the failure being propagated. */
    private static void clearQuietly(GradientFunction fn, RuntimeException primary) {
        try {
            fn.clearSavedTensors();
        } catch (RuntimeException e) {
            primary.addSuppressed(e);
        }
    }

    /**
     * The identity node of {@link #reattach}: the gradient passes through unchanged (BackwardPass takes its own
     * reference). Supports create-graph.
     */
    static final class ReattachGrad implements GradientFunction {
        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Collections.singletonList(gradOutput);
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }
}
