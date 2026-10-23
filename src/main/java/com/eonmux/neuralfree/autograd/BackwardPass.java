package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.autograd.ComputationGraph.GraphNode;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * One reverse-mode pass over the nodes a root was computed from, with the gradient ownership rules of the
 * {@link ComputationGraph} class description: every gradient the pass holds is closed once it has been consumed or
 * handed to a leaf's gradient slot or a captured result, and on failure every gradient not yet handed over is closed.
 *
 * <p>The pass runs in one of two modes. {@link #run} ({@code backward}) accumulates leaf gradients into their slots.
 * {@link #capture} ({@link Autograd#grad}) writes no slot: it runs only the nodes of a reachability mask, captures the
 * gradient arriving at each target (a leaf's summed gradient, or a new reference to a node output's gradient while
 * routing continues upstream) and closes the gradients of every other leaf.</p>
 *
 * <p>In a create-graph pass the caller has grad mode and the create-graph flag on, so the gradient functions record
 * their computation; every second handle of a gradient the pass makes is then a recorded identity node
 * ({@link Autograd#secondHandle}), so gradients keep their path to the tensors they were computed from.</p>
 */
final class BackwardPass {
    private final ComputationGraph graph;
    private final List<GraphNode> producersFirst;
    private final Map<Tensor, GraphNode> producers;
    private final Set<Tensor> gradientLeaves;
    private final Map<Tensor, Tensor> pending;
    private final boolean creating;
    /** Capture mode: the nodes that run (null in backward mode: all of them). */
    private final Set<GraphNode> mask;
    /** Capture mode: each target's position in the result (its first position when listed twice). */
    private final Map<Tensor, Integer> targetIndex;
    private final Tensor[] captured;

    private BackwardPass(ComputationGraph graph, List<GraphNode> producersFirst, Set<Tensor> gradientLeaves,
                         boolean creating, Set<GraphNode> mask, List<Tensor> targets) {
        this.graph = graph;
        this.producersFirst = producersFirst;
        this.gradientLeaves = gradientLeaves;
        this.creating = creating;
        this.mask = mask;
        this.producers = new IdentityHashMap<>();
        for (GraphNode node : producersFirst) {
            producers.put(node.output(), node);
        }
        this.pending = new IdentityHashMap<>();
        if (targets == null) {
            this.targetIndex = null;
            this.captured = null;
        } else {
            this.targetIndex = new IdentityHashMap<>();
            for (int i = targets.size() - 1; i >= 0; i--) {
                targetIndex.put(targets.get(i), i);
            }
            this.captured = new Tensor[targets.size()];
        }
    }

    /**
     * Propagates from {@code root}, the output of the last node of {@code producersFirst} (every node listed after the
     * nodes that produced its inputs), or a leaf when {@code producersFirst} is empty, seeded with a new reference to
     * {@code gradOutput} or with ones. Every leaf that receives a gradient is added to {@code gradientLeaves}.
     */
    static void run(ComputationGraph graph, List<GraphNode> producersFirst, Tensor root, Tensor gradOutput,
                    Set<Tensor> gradientLeaves, boolean creating) {
        new BackwardPass(graph, producersFirst, gradientLeaves, creating, null, null).propagate(root, gradOutput);
    }

    /**
     * Propagates from {@code root} as {@link #run} does, running only the nodes in {@code mask}, and returns the
     * gradient of each of {@code targets} (null where none arrives), owned by the caller. No gradient slot is written.
     * A target listed several times gets a separate reference at each position.
     */
    static List<Tensor> capture(ComputationGraph graph, List<GraphNode> producersFirst, Set<GraphNode> mask,
                                Tensor root, Tensor gradOutput, List<Tensor> targets, boolean creating) {
        BackwardPass pass = new BackwardPass(graph, producersFirst, null, creating, mask, targets);
        try {
            pass.propagate(root, gradOutput);
            for (int i = 0; i < targets.size(); i++) {
                int first = pass.targetIndex.get(targets.get(i));
                if (first != i && pass.captured[first] != null) {
                    pass.captured[i] = pass.secondHandle(pass.captured[first]);
                }
            }
            return new ArrayList<>(Arrays.asList(pass.captured));
        } catch (RuntimeException e) {
            List<Tensor> held = new ArrayList<>();
            for (Tensor t : pass.captured) {
                if (t != null) {
                    held.add(t);
                }
            }
            closeAll(held, e);
            throw e;
        }
    }

    private void propagate(Tensor root, Tensor gradOutput) {
        RuntimeException failure = null;
        try {
            Tensor seed = gradOutput != null ? secondHandle(gradOutput) : seed(root);
            if (producersFirst.isEmpty()) {
                deliverToLeaf(root, seed);
                return;
            }
            pending.put(root, seed);
            for (int i = producersFirst.size() - 1; i >= 0; i--) {
                GraphNode node = producersFirst.get(i);
                Tensor grad = pending.remove(node.output());
                if (grad != null) {
                    visit(node, grad);
                }
            }
        } catch (RuntimeException e) {
            failure = e;
            throw e;
        } finally {
            closeAll(pending.values(), failure);
            pending.clear();
        }
    }

    private static Tensor seed(Tensor root) {
        return root.shape().rank() == 0
            ? Tensor.scalar(root.backend(), 1.0f)
            : Tensor.ones(root.backend(), root.shape());
    }

    /**
     * A new reference to {@code t} owned by the pass: a detached handle, or in a create-graph pass, when {@code t}
     * requires grad, a recorded identity node's output whose input is {@code t}.
     */
    private Tensor secondHandle(Tensor t) {
        return creating ? Autograd.secondHandle(t) : t.detach();
    }

    /** Capture mode: a new reference to the total gradient of a node-output target; routing continues with it. */
    private void captureNodeOutput(Tensor output, Tensor grad) {
        if (targetIndex == null) {
            return;
        }
        Integer index = targetIndex.get(output);
        if (index != null) {
            Tensor handle = secondHandle(grad);
            Tensor previous = captured[index];
            captured[index] = handle;
            if (previous != null) {
                closeAll(Collections.singletonList(previous), null);
            }
        }
    }

    /**
     * Handles {@code grad}, the total gradient of {@code node}'s output: captures it when the output is a target, runs
     * the gradient function of {@code node} on it and routes the results when the node runs, then closes it.
     */
    private void visit(GraphNode node, Tensor grad) {
        RuntimeException failure = null;
        try {
            captureNodeOutput(node.output(), grad);
            if (mask == null || mask.contains(node)) {
                route(node, grad);
            }
        } catch (RuntimeException e) {
            failure = e;
            throw e;
        } finally {
            closeAll(Collections.singletonList(grad), failure);
        }
    }

    /**
     * Hands each gradient the function of {@code node} returns to the input at the same position: summed into the
     * pending gradient of a node output, delivered to a leaf. Positions whose input is absent or does not require grad
     * are skipped. A returned tensor is taken over unless it is {@code grad}, an input, the output or a saved tensor of
     * {@code node}, an output the graph owns, or was already taken over for an earlier position; each of those uses
     * gets a new reference instead. Taken-over tensors left unused are closed.
     */
    private void route(GraphNode node, Tensor grad) {
        List<Tensor> returned;
        try {
            returned = node.gradFn().backward(grad);
        } catch (RuntimeException e) {
            throw new RuntimeException("Error during backward pass for node with output: " + node.output(), e);
        }
        if (returned == null) {
            return;
        }
        List<Tensor> inputs = node.inputs();

        Set<Tensor> foreign = identitySet();
        foreign.addAll(inputs);
        foreign.add(node.output());
        foreign.addAll(savedTensors(node));
        foreign.add(grad);
        for (Tensor r : returned) {
            if (r != null && graph.ownsOutput(r)) {
                foreign.add(r);
            }
        }

        Set<Tensor> adopted = identitySet();
        List<Tensor> targets = new ArrayList<>();
        List<Tensor> handles = new ArrayList<>();
        RuntimeException failure = null;
        try {
            for (int i = 0; i < returned.size(); i++) {
                Tensor r = returned.get(i);
                Tensor input = i < inputs.size() ? inputs.get(i) : null;
                if (r == null || input == null || !input.requiresGrad()) {
                    continue;
                }
                targets.add(input);
                handles.add(foreign.contains(r) || !adopted.add(r) ? secondHandle(r) : r);
            }
            for (int k = 0; k < handles.size(); k++) {
                deliver(node, targets.get(k), handles.set(k, null));
            }
        } catch (RuntimeException e) {
            failure = e;
            throw e;
        } finally {
            List<Tensor> unused = new ArrayList<>();
            for (Tensor h : handles) {
                if (h != null) {
                    unused.add(h);
                }
            }
            for (Tensor r : returned) {
                if (r != null && !foreign.contains(r) && !adopted.contains(r)) {
                    unused.add(r);
                }
            }
            closeAll(unused, failure);
        }
    }

    /**
     * Takes ownership of {@code grad} and adds it to the gradient of {@code input}, an input of {@code node}: into the
     * pending gradient of a node output of this pass, or into a leaf.
     *
     * @throws IllegalStateException when {@code input} is a released non-leaf (it has a gradient function but no node
     *                               in this pass)
     */
    private void deliver(GraphNode node, Tensor input, Tensor grad) {
        if (!grad.shape().equals(input.shape())) {
            IllegalArgumentException e = new IllegalArgumentException("Gradient shape " + grad.shape()
                + " does not match tensor shape " + input.shape() + " (input of the node with output "
                + node.output() + ")");
            closeAll(Collections.singletonList(grad), e);
            throw e;
        }
        GraphNode producer = producers.get(input);
        if (producer == null) {
            if (input.gradFn() != null) {
                IllegalStateException e = new IllegalStateException("Cannot backpropagate into " + input
                    + " (input of the node with output " + node.output() + "): "
                    + ComputationGraph.RELEASED_GRAPH_MESSAGE);
                closeAll(Collections.singletonList(grad), e);
                throw e;
            }
            deliverToLeaf(input, grad);
            return;
        }
        if (mask != null && !mask.contains(producer) && !targetIndex.containsKey(input)) {
            closeAll(Collections.singletonList(grad), null);
            return;
        }
        Tensor existing = pending.get(input);
        if (existing == null) {
            pending.put(input, grad);
            return;
        }
        pending.put(input, add(existing, grad));
        closeAll(Collections.singletonList(existing), null);
    }

    /**
     * Hands {@code grad} (owned) to {@code leaf}: in backward mode into its gradient slot; in capture mode into its
     * captured result when it is a target, else it is closed.
     */
    private void deliverToLeaf(Tensor leaf, Tensor grad) {
        if (targetIndex == null) {
            accumulateLeaf(leaf, grad);
            return;
        }
        Integer index = targetIndex.get(leaf);
        if (index == null) {
            closeAll(Collections.singletonList(grad), null);
            return;
        }
        Tensor existing = captured[index];
        captured[index] = existing == null ? grad : add(existing, grad);
        if (existing != null) {
            closeAll(Collections.singletonList(existing), null);
        }
    }

    /**
     * Adds {@code grad} (owned) into the gradient slot of {@code leaf}: the slot takes {@code grad} when it is empty,
     * else the sum, and the gradient it replaces is closed.
     */
    private void accumulateLeaf(Tensor leaf, Tensor grad) {
        Tensor existing = leaf.grad();
        Tensor next = existing == null ? grad : add(existing, grad);
        try {
            leaf.setGrad(next);
        } catch (RuntimeException e) {
            closeAll(Collections.singletonList(next), e);
            throw e;
        }
        gradientLeaves.add(leaf);
        if (existing != null) {
            closeAll(Collections.singletonList(existing), null);
        }
    }

    /** {@code existing + grad} as a new tensor; closes {@code grad}, which the caller handed over. */
    private static Tensor add(Tensor existing, Tensor grad) {
        Tensor sum;
        try {
            sum = TensorOps.add(existing, grad);
        } catch (RuntimeException e) {
            closeAll(Collections.singletonList(grad), e);
            throw e;
        }
        closeAll(Collections.singletonList(grad), null);
        return sum;
    }

    // ------------------------------------------------------------------ shared with ComputationGraph

    /** The saved tensors of {@code node}'s gradient function; empty when it reports none. */
    static List<Tensor> savedTensors(GraphNode node) {
        List<Tensor> saved = node.gradFn().savedTensors();
        return saved != null ? saved : Collections.emptyList();
    }

    static <T> Set<T> identitySet() {
        return Collections.newSetFromMap(new IdentityHashMap<>());
    }

    /** {@code first} with {@code next} added as suppressed, or {@code next} when there is no first failure. */
    static RuntimeException collect(RuntimeException first, RuntimeException next) {
        if (first == null) {
            return next;
        }
        first.addSuppressed(next);
        return first;
    }

    /**
     * Closes every tensor (closing is idempotent per tensor object). When {@code primary} is given, close failures are
     * added to it as suppressed exceptions; otherwise the first failure is thrown once every tensor was tried.
     */
    static void closeAll(Collection<Tensor> tensors, RuntimeException primary) {
        RuntimeException failure = null;
        for (Tensor t : tensors) {
            try {
                t.close();
            } catch (RuntimeException e) {
                if (primary != null) {
                    primary.addSuppressed(e);
                } else {
                    failure = collect(failure, e);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }
}
