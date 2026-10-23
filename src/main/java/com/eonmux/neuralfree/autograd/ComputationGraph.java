package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The per-thread record of autograd operations and the reverse-mode pass over it.
 *
 * <p><b>Grad mode is per thread.</b> The graph, the grad-mode flag ({@link #noGrad}, {@link #enableGrad}) and the
 * create-graph flag ({@link Autograd#isCreatingGraph()}) are thread-local. A worker thread does not inherit its
 * caller's mode or graph: code that runs model work on other threads captures the caller's flags when it submits the
 * work and applies them in the worker with a scope.</p>
 *
 * <p><b>Recording.</b> {@link Autograd#record} (and {@link #record}) adds one node per operation that has an input
 * requiring grad while grad mode is on. A node's output requires grad and carries the node's gradient function
 * ({@link Tensor#gradFn()}), which is never cleared. A <em>leaf</em> is a tensor that requires grad and has no
 * gradient function (parameters, and tensors made with {@code withRequiresGrad(true)}). A tensor with a gradient
 * function whose node is no longer in this thread's graph (released, discarded, cleared, or recorded on another
 * thread) is a <em>released non-leaf</em>: no gradient can flow through it.</p>
 *
 * <p><b>Backward.</b> {@link #backward(Tensor, Tensor, boolean, boolean)} propagates from a root through the nodes that
 * produced it, consumers before producers. A node's output receives its gradient only inside the pass: the gradients of
 * node outputs are summed per output, handed to the node's gradient function and closed. Leaves accumulate their
 * gradient in {@link Tensor#grad()}; a backward from a leaf accumulates the seed into its own slot. A first-order pass
 * runs with grad mode off, so gradient functions that call recording operations record nothing. By default the
 * traversed part of the graph is released afterwards; {@code retainGraph = true} keeps it, with its saved tensors,
 * for a further backward pass. Backward through a released non-leaf throws.</p>
 *
 * <p><b>Create graph.</b> With {@code createGraph = true} (which implies {@code retainGraph}) the pass runs with grad
 * mode on and the create-graph flag set, so the gradients it computes are recorded and can be differentiated again.
 * Every node the pass will run must support it ({@link GradientFunction#supportsCreateGraph()}); otherwise the call
 * throws before any gradient is computed. <em>Lifetime rule:</em> afterwards only the nodes recorded by the pass from
 * which a delivered leaf gradient or a captured result ({@link Autograd#grad}) is computed stay in the graph; the rest
 * are discarded. The kept nodes are released by the next non-retained backward that traverses them, by
 * {@link #release(Tensor)} of a result of the pass, or by the enclosing {@link StepScope}.</p>
 *
 * <p><b>Tensor ownership.</b> Device buffers have no GC-based release, so ownership is explicit:</p>
 * <ul>
 *   <li>Every gradient tensor the pass creates or receives from a gradient function belongs to the graph. A leaf's
 *   gradient belongs to that leaf's gradient slot: every slot gets its own tensor object (views of one buffer stay
 *   alive through the storage reference count), and accumulating into a slot closes the gradient it replaces.</li>
 *   <li>A tensor a gradient function returns that is the incoming gradient, an input, the output or a saved tensor of
 *   its node, or an output the graph owns, is never taken over; its slot gets a new reference to it.</li>
 *   <li>Releasing nodes (after a backward pass, or {@link #release(Tensor)} without one) drops their saved tensors
 *   and closes their outputs, except the root and tensors that nodes still in the graph use. Leaves (parameters,
 *   inputs) and the gradients in the slots of leaves this graph gave gradients to are never closed. A caller that
 *   needs an intermediate result after its graph is released takes its own reference with {@link Tensor#detach()}
 *   first.</li>
 *   <li>Outputs of the identity nodes {@link Autograd#reattach} records belong to the graph: every way a node leaves
 *   the graph (release, {@link #discardRecordedSince}, {@link #clear()}, pruning, {@link StepScope}) closes them.</li>
 *   <li>{@link #zeroGrad()} closes and clears the gradients of the leaves this graph gave gradients to and of the
 *   tensors its nodes reference.</li>
 *   <li>{@link #discardRecordedSince(long)} drops the nodes recorded after a {@link #mark()} that are still in the
 *   graph, such as a discarded auxiliary branch no backward pass or release reached: their saved tensors are dropped
 *   and their outputs stay with whoever holds them. {@link StepScope} does the same for one training step and also
 *   closes the outputs.</li>
 * </ul>
 */
public final class ComputationGraph {
    private static final ThreadLocal<ComputationGraph> INSTANCE = ThreadLocal.withInitial(ComputationGraph::new);
    private static final ThreadLocal<Boolean> GRAD_ENABLED = ThreadLocal.withInitial(() -> Boolean.TRUE);
    private static final ThreadLocal<Boolean> CREATING_GRAPH = ThreadLocal.withInitial(() -> Boolean.FALSE);

    static final String RELEASED_GRAPH_MESSAGE = "the graph that produced this tensor was released or recorded on "
        + "another thread; pass retainGraph=true to backpropagate twice, or detach() it";

    private final List<GraphNode> nodes;
    private final Map<Tensor, GraphNode> tensorToNode;
    private final Set<Tensor> gradientLeaves;
    private long nextSequence;

    private ComputationGraph() {
        this.nodes = new ArrayList<>();
        this.tensorToNode = new IdentityHashMap<>();
        this.gradientLeaves = identitySet();
    }

    public static ComputationGraph current() {
        return INSTANCE.get();
    }

    public static boolean isGradEnabled() {
        return GRAD_ENABLED.get();
    }

    /** Turns grad mode off until the returned scope is closed, which restores the previous mode. */
    public static NoGradScope noGrad() {
        return new NoGradScope();
    }

    /** Turns grad mode on until the returned scope is closed, which restores the previous mode. */
    public static EnableGradScope enableGrad() {
        return new EnableGradScope();
    }

    public static void setGradEnabled(boolean enabled) {
        GRAD_ENABLED.set(enabled);
    }

    /** Whether this thread is inside a backward pass with {@code createGraph = true}. */
    static boolean isCreatingGraph() {
        return CREATING_GRAPH.get();
    }

    /**
     * Records {@code output} as produced by {@code gradFn} from {@code inputs}, when grad mode is on, some input
     * requires grad and {@code gradFn} is not null; otherwise does nothing. The caller marks {@code output} as
     * requiring grad and sets its gradient function first; {@link Autograd#record} does all of it.
     *
     * @throws IllegalArgumentException when {@code output} is null
     */
    public void record(Tensor output, GradientFunction gradFn, List<Tensor> inputs) {
        if (!isGradEnabled()) {
            return;
        }
        if (output == null) {
            throw new IllegalArgumentException("Output tensor cannot be null");
        }
        if (gradFn == null || !anyRequiresGrad(inputs)) {
            return;
        }
        addNode(output, gradFn, inputs, false);
    }

    /**
     * Adds a node without the grad-mode guard of {@link #record}: the caller ({@link Autograd}) checked it. With
     * {@code ownsOutput} the graph closes {@code output} when the node leaves the graph.
     */
    void recordNode(Tensor output, GradientFunction gradFn, List<Tensor> inputs, boolean ownsOutput) {
        if (output == null || gradFn == null) {
            throw new IllegalArgumentException("A recorded node needs an output and a gradient function");
        }
        addNode(output, gradFn, inputs, ownsOutput);
    }

    private void addNode(Tensor output, GradientFunction gradFn, List<Tensor> inputs, boolean ownsOutput) {
        GraphNode node = new GraphNode(output, gradFn, inputs != null ? inputs : Collections.emptyList());
        node.sequence = nextSequence++;
        node.ownsOutput = ownsOutput;
        nodes.add(node);
        tensorToNode.put(output, node);
    }

    /** Removes the nodes of {@code outputs} without touching their saved tensors or outputs (a failed recording). */
    void forget(Collection<Tensor> outputs) {
        Set<GraphNode> forgotten = identitySet();
        for (Tensor output : outputs) {
            GraphNode node = tensorToNode.remove(output);
            if (node != null) {
                forgotten.add(node);
            }
        }
        nodes.removeIf(forgotten::contains);
    }

    /** Whether {@code t} is the output of a node of this graph that owns it ({@link Autograd#reattach}). */
    boolean ownsOutput(Tensor t) {
        GraphNode node = tensorToNode.get(t);
        return node != null && node.ownsOutput;
    }

    /** Backward pass from {@code loss} seeded with ones, releasing the traversed graph afterwards. */
    public void backward(Tensor loss) {
        backward(loss, null, false, false);
    }

    /** Backward pass from {@code loss} seeded with ones; {@code retainGraph} keeps the traversed graph. */
    public void backward(Tensor loss, boolean retainGraph) {
        backward(loss, null, retainGraph, false);
    }

    /** Backward pass from {@code loss} seeded with {@code gradOutput}, releasing the traversed graph afterwards. */
    public void backward(Tensor loss, Tensor gradOutput) {
        backward(loss, gradOutput, false, false);
    }

    /** First-order backward pass; see {@link #backward(Tensor, Tensor, boolean, boolean)}. */
    public void backward(Tensor loss, Tensor gradOutput, boolean retainGraph) {
        backward(loss, gradOutput, retainGraph, false);
    }

    /**
     * Propagates gradients from {@code loss} to every leaf that requires grad and that {@code loss} was computed from.
     * The seed is {@code gradOutput} (which stays the caller's: the pass takes its own reference), or ones of the
     * shape of {@code loss} when it is null. When {@code loss} is a leaf the seed is accumulated into its own gradient
     * slot. Without {@code retainGraph} the traversed nodes are released afterwards (see the class description); nodes
     * that other nodes still in the graph depend on are kept. With {@code createGraph} the pass records the gradient
     * computation (grad mode on even inside {@link #noGrad}), the graph is retained, and the nodes the pass recorded
     * follow the lifetime rule of the class description. When a gradient function fails, the gradients of the pass
     * that were not yet handed to leaves are closed and the traversed graph is left as it was.
     *
     * @throws IllegalArgumentException      when {@code loss} is null, or {@code gradOutput} or a computed gradient
     *                                       does not have the shape of its tensor
     * @throws IllegalStateException         when {@code loss} does not require grad, or the pass reaches a released
     *                                       non-leaf ({@code loss} itself included)
     * @throws UnsupportedOperationException with {@code createGraph}, when a node of the pass does not support it
     */
    public void backward(Tensor loss, Tensor gradOutput, boolean retainGraph, boolean createGraph) {
        if (loss == null) {
            throw new IllegalArgumentException("Loss tensor cannot be null");
        }
        if (!loss.requiresGrad()) {
            throw new IllegalStateException("Cannot call backward() on tensor that does not require grad");
        }
        requireGradientShape(gradOutput, loss);
        GraphNode rootNode = tensorToNode.get(loss);
        if (rootNode == null && loss.gradFn() != null) {
            throw new IllegalStateException(RELEASED_GRAPH_MESSAGE);
        }
        pruneGradientLeaves();
        List<GraphNode> producersFirst = rootNode == null ? Collections.emptyList() : producersFirst(rootNode);
        if (!createGraph) {
            try (PassScope scope = new PassScope(false)) {
                BackwardPass.run(this, producersFirst, loss, gradOutput, gradientLeaves, false);
            }
            if (!retainGraph && rootNode != null) {
                release(producersFirst, only(loss));
            }
            return;
        }
        requireCreateGraphSupport(producersFirst, null);
        long passMark = mark();
        RuntimeException failure = null;
        try (PassScope scope = new PassScope(true)) {
            BackwardPass.run(this, producersFirst, loss, gradOutput, gradientLeaves, true);
        } catch (RuntimeException e) {
            failure = e;
            throw e;
        } finally {
            try {
                pruneRecordedSince(passMark, slotGradients());
            } catch (RuntimeException e) {
                if (failure == null) {
                    throw e;
                }
                failure.addSuppressed(e);
            }
        }
    }

    /**
     * {@link Autograd#grad}: the gradients of {@code output} with respect to {@code targets}, without writing any
     * gradient slot. Only the nodes with a path to some target run. Without {@code retainGraph} the nodes downstream of
     * the targets are released afterwards; the graph a node-output target was computed from stays.
     */
    List<Tensor> gradients(Tensor output, List<Tensor> targets, Tensor gradOutput, boolean retainGraph,
                           boolean createGraph) {
        if (output == null) {
            throw new IllegalArgumentException("Output tensor cannot be null");
        }
        if (!output.requiresGrad()) {
            throw new IllegalStateException("Cannot differentiate a tensor that does not require grad");
        }
        requireTargets(targets);
        requireGradientShape(gradOutput, output);
        GraphNode rootNode = tensorToNode.get(output);
        if (rootNode == null && output.gradFn() != null) {
            throw new IllegalStateException(RELEASED_GRAPH_MESSAGE);
        }
        List<GraphNode> producersFirst = rootNode == null ? Collections.emptyList() : producersFirst(rootNode);
        Set<GraphNode> mask = reachingNodes(producersFirst, targets);
        for (GraphNode node : producersFirst) {
            if (mask.contains(node) && node.gradFn instanceof CheckpointFunction.CheckpointGradFunction) {
                throw new UnsupportedOperationException("Autograd.grad cannot run through a checkpointed segment, "
                    + "whose recompute writes leaf gradient slots; use backward() instead");
            }
        }
        if (!createGraph) {
            List<Tensor> result;
            try (PassScope scope = new PassScope(false)) {
                result = BackwardPass.capture(this, producersFirst, mask, output, gradOutput, targets, false);
            }
            if (!retainGraph && rootNode != null) {
                releaseKeepingResults(producersFirst, output, targets, result);
            }
            return result;
        }
        requireCreateGraphSupport(producersFirst, mask);
        long passMark = mark();
        List<Tensor> result = Collections.emptyList();
        RuntimeException failure = null;
        try (PassScope scope = new PassScope(true)) {
            result = BackwardPass.capture(this, producersFirst, mask, output, gradOutput, targets, true);
        } catch (RuntimeException e) {
            failure = e;
            throw e;
        } finally {
            Set<Tensor> keep = slotGradients();
            keep.addAll(nonNull(result));
            try {
                pruneRecordedSince(passMark, keep);
            } catch (RuntimeException e) {
                if (failure == null) {
                    BackwardPass.closeAll(nonNull(result), e);
                    throw e;
                }
                failure.addSuppressed(e);
            }
        }
        return result;
    }

    /**
     * Releases the graph of {@code output} after a non-retained capture, except the nodes a target was computed from:
     * the nodes the pass ran and the side branches that lead to no target leave the graph, while a node-output target
     * stays open with its own graph, so it can still be backpropagated or released by its holder. The captured handles
     * are the caller's.
     */
    private void releaseKeepingResults(List<GraphNode> producersFirst, Tensor output, List<Tensor> targets,
                                       List<Tensor> result) {
        try {
            Set<GraphNode> upstreamOfTargets = producersOf(targets);
            List<GraphNode> downstream = new ArrayList<>();
            for (GraphNode node : producersFirst) {
                if (!upstreamOfTargets.contains(node)) {
                    downstream.add(node);
                }
            }
            Set<Tensor> roots = only(output);
            roots.addAll(targets);
            release(downstream, roots);
        } catch (RuntimeException e) {
            BackwardPass.closeAll(nonNull(result), e);
            throw e;
        }
    }

    private static List<Tensor> nonNull(List<Tensor> tensors) {
        List<Tensor> present = new ArrayList<>();
        for (Tensor t : tensors) {
            if (t != null) {
                present.add(t);
            }
        }
        return present;
    }

    private void requireTargets(List<Tensor> targets) {
        if (targets == null || targets.isEmpty()) {
            throw new IllegalArgumentException("At least one input to differentiate with respect to is required");
        }
        for (int i = 0; i < targets.size(); i++) {
            Tensor t = targets.get(i);
            if (t == null) {
                throw new IllegalArgumentException("Input " + i + " cannot be null");
            }
            if (!t.requiresGrad()) {
                throw new IllegalArgumentException("Input " + i + " does not require grad");
            }
            if (t.gradFn() != null && !tensorToNode.containsKey(t)) {
                throw new IllegalStateException("Input " + i + ": " + RELEASED_GRAPH_MESSAGE);
            }
        }
    }

    private static void requireGradientShape(Tensor gradOutput, Tensor tensor) {
        if (gradOutput != null && !gradOutput.shape().equals(tensor.shape())) {
            throw new IllegalArgumentException("Gradient shape " + gradOutput.shape() + " does not match tensor shape "
                + tensor.shape());
        }
    }

    /**
     * Throws, naming the operation, for the first node the pass will run (those of {@code producersFirst} in
     * {@code mask}, or all of them when it is null) whose gradient function cannot record its backward.
     */
    private static void requireCreateGraphSupport(List<GraphNode> producersFirst, Set<GraphNode> mask) {
        for (GraphNode node : producersFirst) {
            if ((mask == null || mask.contains(node)) && !node.gradFn.supportsCreateGraph()) {
                throw new UnsupportedOperationException("createGraph is not supported by "
                    + node.gradFn.getClass().getName() + " (node with output " + node.output + ")");
            }
        }
    }

    /**
     * The nodes of {@code producersFirst} with a path to some target: a node whose input is a target or the output of
     * such a node. {@code producersFirst} lists producers before consumers, so one sweep suffices.
     */
    private static Set<GraphNode> reachingNodes(List<GraphNode> producersFirst, List<Tensor> targets) {
        Set<Tensor> targetSet = identitySet();
        targetSet.addAll(targets);
        Set<Tensor> reaching = identitySet();
        Set<GraphNode> mask = identitySet();
        for (GraphNode node : producersFirst) {
            for (Tensor input : node.inputs) {
                if (input != null && (targetSet.contains(input) || reaching.contains(input))) {
                    mask.add(node);
                    reaching.add(node.output);
                    break;
                }
            }
        }
        return mask;
    }

    /**
     * Releases the nodes {@code root} was computed from without a backward pass, exactly as a backward pass from
     * {@code root} without {@code retainGraph} releases them afterwards: for a result whose gradient will not be
     * propagated, such as a prediction whose loss was computed without recording. {@code root} itself stays open.
     * Does nothing when no recorded node produced {@code root}.
     *
     * @throws IllegalArgumentException when {@code root} is null
     */
    public void release(Tensor root) {
        if (root == null) {
            throw new IllegalArgumentException("Root tensor cannot be null");
        }
        GraphNode rootNode = tensorToNode.get(root);
        if (rootNode != null) {
            release(producersFirst(rootNode), only(root));
        }
    }

    /** Drops the saved tensors of every node, keeping the graph structure. */
    public void releaseSavedTensors() {
        for (GraphNode node : nodes) {
            node.gradFn.clearSavedTensors();
        }
    }

    // ------------------------------------------------------------------ traversal

    /**
     * The nodes {@code root} was computed from, {@code root} included, each after every node that produced one of its
     * inputs (iterative depth-first post-order, so long chains do not exhaust the stack).
     */
    private List<GraphNode> producersFirst(GraphNode root) {
        List<GraphNode> order = new ArrayList<>();
        Set<GraphNode> visited = identitySet();
        Deque<Visit> stack = new ArrayDeque<>();
        visited.add(root);
        stack.push(new Visit(root));
        while (!stack.isEmpty()) {
            Visit top = stack.peek();
            if (top.nextInput < top.node.inputs.size()) {
                Tensor input = top.node.inputs.get(top.nextInput++);
                GraphNode producer = input == null ? null : tensorToNode.get(input);
                if (producer != null && visited.add(producer)) {
                    stack.push(new Visit(producer));
                }
            } else {
                stack.pop();
                order.add(top.node);
            }
        }
        return order;
    }

    /** The nodes that produced one of {@code tensors}, and every node upstream of them. */
    private Set<GraphNode> producersOf(List<Tensor> tensors) {
        Set<GraphNode> found = identitySet();
        Deque<GraphNode> stack = new ArrayDeque<>();
        for (Tensor t : tensors) {
            GraphNode producer = tensorToNode.get(t);
            if (producer != null && found.add(producer)) {
                stack.push(producer);
            }
        }
        while (!stack.isEmpty()) {
            for (Tensor input : stack.pop().inputs) {
                GraphNode producer = input == null ? null : tensorToNode.get(input);
                if (producer != null && found.add(producer)) {
                    stack.push(producer);
                }
            }
        }
        return found;
    }

    // ------------------------------------------------------------------ release

    /**
     * Releases the nodes of a finished pass from {@code roots}: every traversed node that no node remaining in the
     * graph depends on drops its saved tensors and leaves the graph, and its output is closed unless it is a root the
     * caller holds, its own input, used by a remaining node, or the gradient in a leaf's slot. An output the graph owns
     * is closed even as a root, since nobody else holds it.
     */
    private void release(List<GraphNode> traversed, Set<Tensor> roots) {
        Set<GraphNode> released = identitySet();
        released.addAll(traversed);
        keepDependenciesOfRemainingNodes(released);

        Set<Tensor> stillUsed = slotGradients();
        for (GraphNode node : nodes) {
            if (!released.contains(node)) {
                stillUsed.add(node.output);
                stillUsed.addAll(node.inputs);
                stillUsed.addAll(BackwardPass.savedTensors(node));
            }
        }

        List<Tensor> outputs = new ArrayList<>();
        RuntimeException failure = null;
        for (GraphNode node : traversed) {
            if (!released.contains(node)) {
                continue;
            }
            try {
                node.gradFn.clearSavedTensors();
            } catch (RuntimeException e) {
                failure = BackwardPass.collect(failure, e);
            }
            tensorToNode.remove(node.output, node);
            boolean kept = (roots.contains(node.output) && !node.ownsOutput) || stillUsed.contains(node.output)
                || containsIdentity(node.inputs, node.output);
            if (!kept) {
                outputs.add(node.output);
            }
        }
        nodes.removeIf(released::contains);
        BackwardPass.closeAll(outputs, failure);
        if (failure != null) {
            throw failure;
        }
    }

    /** Removes from {@code released} every node that a node outside it depends on, directly or transitively. */
    private void keepDependenciesOfRemainingNodes(Set<GraphNode> released) {
        Deque<GraphNode> stack = new ArrayDeque<>();
        for (GraphNode node : nodes) {
            if (!released.contains(node)) {
                stack.push(node);
            }
        }
        while (!stack.isEmpty()) {
            GraphNode node = stack.pop();
            for (Tensor input : node.inputs) {
                GraphNode producer = input == null ? null : tensorToNode.get(input);
                if (producer != null && released.remove(producer)) {
                    stack.push(producer);
                }
            }
        }
    }

    /**
     * Removes {@code discarded} from the graph: drops their saved tensors and closes the outputs the graph owns, or
     * every output with {@code closeOutputs}, except the tensors in {@code spared} and an output that is its node's own
     * input. Every node is tried; the first failure is thrown with later ones suppressed.
     */
    private void discard(List<GraphNode> discarded, boolean closeOutputs, Set<Tensor> spared) {
        RuntimeException failure = null;
        Set<GraphNode> removed = identitySet();
        List<Tensor> outputs = new ArrayList<>();
        for (GraphNode node : discarded) {
            try {
                node.gradFn.clearSavedTensors();
            } catch (RuntimeException e) {
                failure = BackwardPass.collect(failure, e);
            }
            tensorToNode.remove(node.output, node);
            removed.add(node);
            if ((closeOutputs || node.ownsOutput) && !spared.contains(node.output)
                && !containsIdentity(node.inputs, node.output)) {
                outputs.add(node.output);
            }
        }
        nodes.removeIf(removed::contains);
        try {
            BackwardPass.closeAll(outputs, null);
        } catch (RuntimeException e) {
            failure = BackwardPass.collect(failure, e);
        }
        if (failure != null) {
            throw failure;
        }
    }

    /**
     * The create-graph lifetime rule: of the nodes recorded since {@code mark}, keeps those from which a tensor of
     * {@code results} is computed and discards the rest.
     */
    private void pruneRecordedSince(long mark, Set<Tensor> results) {
        List<GraphNode> recorded = recordedSince(mark);
        if (recorded.isEmpty()) {
            return;
        }
        Map<Tensor, GraphNode> producers = new IdentityHashMap<>();
        for (GraphNode node : recorded) {
            producers.put(node.output, node);
        }
        Set<GraphNode> kept = identitySet();
        Deque<Tensor> stack = new ArrayDeque<>(results);
        while (!stack.isEmpty()) {
            GraphNode producer = producers.get(stack.pop());
            if (producer != null && kept.add(producer)) {
                for (Tensor input : producer.inputs) {
                    if (input != null) {
                        stack.push(input);
                    }
                }
            }
        }
        List<GraphNode> pruned = new ArrayList<>();
        for (GraphNode node : recorded) {
            if (!kept.contains(node)) {
                pruned.add(node);
            }
        }
        discard(pruned, false, slotGradients());
    }

    private List<GraphNode> recordedSince(long mark) {
        List<GraphNode> recorded = new ArrayList<>();
        for (GraphNode node : nodes) {
            if (node.sequence >= mark) {
                recorded.add(node);
            }
        }
        return recorded;
    }

    // ------------------------------------------------------------------ gradients and graph state

    /**
     * Closes and clears the gradients of the leaves this graph gave gradients to and of every tensor its nodes
     * reference. A gradient tensor is closed once even when several tensors hold it.
     */
    public void zeroGrad() {
        Set<Tensor> holders = identitySet();
        holders.addAll(gradientLeaves);
        for (GraphNode node : nodes) {
            holders.add(node.output);
            holders.addAll(node.inputs);
        }
        Set<Tensor> grads = identitySet();
        for (Tensor tensor : holders) {
            if (tensor != null && tensor.grad() != null) {
                grads.add(tensor.grad());
                tensor.setGrad(null);
            }
        }
        gradientLeaves.clear();
        BackwardPass.closeAll(grads, null);
    }

    /**
     * Discards every node and drops their saved tensors. Outputs stay with whoever holds them, except the outputs the
     * graph owns ({@link Autograd#reattach}), which are closed; leaves keep their gradients.
     */
    public void clear() {
        List<GraphNode> all = new ArrayList<>(nodes);
        Set<Tensor> slots = slotGradients();
        gradientLeaves.clear();
        discard(all, false, slots);
    }

    /** The current position in this graph's recording order, for {@link #discardRecordedSince(long)}. */
    public long mark() {
        return nextSequence;
    }

    /**
     * Discards the nodes recorded since {@code mark} (a value of {@link #mark()}) that are still in the graph and drops
     * their saved tensors. Outputs stay with whoever holds them, except the outputs the graph owns
     * ({@link Autograd#reattach}), which are closed: the scope that took the mark ends, and a node it recorded that no
     * backward pass or {@link #release(Tensor)} reached is left with no use. Nodes recorded before the mark stay; none
     * of them can depend on a later node. Every node is tried; the first failure is thrown with later ones suppressed.
     */
    public void discardRecordedSince(long mark) {
        discard(recordedSince(mark), false, slotGradients());
    }

    /**
     * Opens a training step: a {@link StepScope} that, when closed, removes every node recorded during the step and
     * closes their outputs, except the tensors spared with {@link StepScope#spare}.
     */
    public StepScope step() {
        return new StepScope(this);
    }

    /** Forgets the leaves whose gradient was cleared outside this graph, such as by an optimizer's zeroGrad. */
    private void pruneGradientLeaves() {
        gradientLeaves.removeIf(t -> t.grad() == null);
    }

    /** The gradients now in the slots of the leaves this graph gave gradients to; never closed by the graph. */
    private Set<Tensor> slotGradients() {
        Set<Tensor> slots = identitySet();
        for (Tensor leaf : gradientLeaves) {
            Tensor g = leaf.grad();
            if (g != null) {
                slots.add(g);
            }
        }
        return slots;
    }

    public int size() {
        return nodes.size();
    }

    public boolean isEmpty() {
        return nodes.isEmpty();
    }

    /** The same as {@link #isGradEnabled()}; kept for source compatibility. */
    public boolean isRecording() {
        return isGradEnabled();
    }

    public GraphNode getNode(Tensor tensor) {
        return tensorToNode.get(tensor);
    }

    public List<GraphNode> getNodes() {
        return Collections.unmodifiableList(nodes);
    }

    // ------------------------------------------------------------------ helpers

    private static boolean anyRequiresGrad(List<Tensor> inputs) {
        if (inputs != null) {
            for (Tensor input : inputs) {
                if (input != null && input.requiresGrad()) {
                    return true;
                }
            }
        }
        return false;
    }

    private static <T> Set<T> identitySet() {
        return BackwardPass.identitySet();
    }

    private static Set<Tensor> only(Tensor t) {
        Set<Tensor> set = identitySet();
        set.add(t);
        return set;
    }

    private static boolean containsIdentity(List<Tensor> tensors, Tensor t) {
        for (Tensor candidate : tensors) {
            if (candidate == t) {
                return true;
            }
        }
        return false;
    }

    /** One frame of the iterative depth-first traversal. */
    private static final class Visit {
        private final GraphNode node;
        private int nextInput;

        Visit(GraphNode node) {
            this.node = node;
        }
    }

    /** Grad mode and the create-graph flag of one backward pass on this thread, both restored on close. */
    private static final class PassScope implements AutoCloseable {
        private final boolean previousGrad;
        private final boolean previousCreating;

        PassScope(boolean createGraph) {
            this.previousGrad = GRAD_ENABLED.get();
            this.previousCreating = CREATING_GRAPH.get();
            GRAD_ENABLED.set(createGraph);
            CREATING_GRAPH.set(createGraph);
        }

        @Override
        public void close() {
            GRAD_ENABLED.set(previousGrad);
            CREATING_GRAPH.set(previousCreating);
        }
    }

    public static final class NoGradScope implements AutoCloseable {
        private final boolean previousState;

        private NoGradScope() {
            this.previousState = isGradEnabled();
            setGradEnabled(false);
        }

        @Override
        public void close() {
            setGradEnabled(previousState);
        }
    }

    /** Grad mode on for the scope's lifetime; {@link #close()} restores the previous mode. */
    public static final class EnableGradScope implements AutoCloseable {
        private final boolean previousState;

        private EnableGradScope() {
            this.previousState = isGradEnabled();
            setGradEnabled(true);
        }

        @Override
        public void close() {
            setGradEnabled(previousState);
        }
    }

    /**
     * One training step of the graph that opened it ({@link #step()}), on the thread that opened it. {@link #close()}
     * removes every node recorded since the step opened that is still in the graph, whatever its results were computed
     * from: their saved tensors are dropped and their outputs closed, except spared tensors ({@link #spare}), gradients
     * in leaf slots and outputs a node recorded before the step uses. A spared node output stays open while its node
     * leaves the graph, so it becomes a released non-leaf. Nodes recorded before the step and their outputs are never
     * touched, so afterwards {@link ComputationGraph#size()} is back at its value when the step opened. Failures are
     * collected: the first is thrown with the others suppressed. Closing twice does nothing.
     */
    public static final class StepScope implements AutoCloseable {
        private final ComputationGraph graph;
        private final long mark;
        private final Set<Tensor> spared;
        private boolean closed;

        private StepScope(ComputationGraph graph) {
            this.graph = graph;
            this.mark = graph.mark();
            this.spared = identitySet();
        }

        /** Keeps {@code t} open when the step closes (a result the step returns); returns {@code t}. */
        public Tensor spare(Tensor t) {
            if (t != null) {
                spared.add(t);
            }
            return t;
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            graph.closeRecordedSince(mark, spared);
        }
    }

    /** {@link StepScope#close()}'s discard: the nodes recorded since {@code mark}, outputs closed. */
    private void closeRecordedSince(long mark, Set<Tensor> spared) {
        Set<Tensor> keepOpen = slotGradients();
        keepOpen.addAll(spared);
        for (GraphNode node : nodes) {
            if (node.sequence < mark) {
                keepOpen.add(node.output);
                keepOpen.addAll(node.inputs);
                keepOpen.addAll(BackwardPass.savedTensors(node));
            }
        }
        discard(recordedSince(mark), true, keepOpen);
    }

    public static final class GraphNode {
        private final Tensor output;
        private final GradientFunction gradFn;
        private final List<Tensor> inputs;
        /** Position in the recording order of the graph that recorded this node. */
        private long sequence;
        /** Whether the graph closes {@link #output} when the node leaves it ({@link Autograd#reattach}). */
        private boolean ownsOutput;

        public GraphNode(Tensor output, GradientFunction gradFn, List<Tensor> inputs) {
            this.output = output;
            this.gradFn = gradFn;
            this.inputs = inputs != null ? new ArrayList<>(inputs) : Collections.emptyList();
        }

        public Tensor output() {
            return output;
        }

        public GradientFunction gradFn() {
            return gradFn;
        }

        public List<Tensor> inputs() {
            return Collections.unmodifiableList(inputs);
        }

        /** Whether the graph owns {@link #output()} and closes it when the node leaves the graph. */
        public boolean ownsOutput() {
            return ownsOutput;
        }

        @Override
        public String toString() {
            return "GraphNode{" +
                "output=" + output +
                ", inputs=" + inputs.size() +
                '}';
        }
    }
}
