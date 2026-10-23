package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.tensor.Tensor;

import java.util.List;

/**
 * The backward rule of one recorded operation ({@link Autograd#record}).
 *
 * <p><b>Ownership.</b> {@link #backward} must not close {@code gradOutput}, which belongs to the graph. The tensors it
 * returns are handed to the graph, which closes them once they are consumed or replaced. Returning {@code gradOutput}
 * itself, the same tensor at several positions, or one of the node's inputs, output or saved tensors is allowed: the
 * graph then takes a new reference for each use and leaves the returned object alone.</p>
 *
 * <p><b>Authoring rules.</b></p>
 * <ul>
 *   <li>G1: save tensors through {@code GradFunctions.SavingGradFn.save} (detached handles released by
 *   {@link #clearSavedTensors()}); never keep a module-owned tensor without saving it, and never save, directly or
 *   through an operation's saves, a tensor whose storage is written in place before the backward (module buffers,
 *   caches, recurrent state): save a fresh tensor or a copy.</li>
 *   <li>G2: compute only with the public facades ({@code TensorOps}, {@code ConvOps}, {@code NormOps}) and tensor
 *   views, never the device dispatchers or CPU references directly, so the CPU fallback and create-graph come for
 *   free. A first-order pass runs with grad mode off, so these calls record nothing.</li>
 *   <li>G3: return one entry per input, null for inputs that need no gradient (need flags captured at construction);
 *   never close {@code gradOutput}; returned tensors follow the ownership contract above.</li>
 *   <li>G4: {@link #supportsCreateGraph()} may return true only when the backward is composed of recorded operations
 *   and never calls {@link Tensor#detach()} on a tensor that may require grad: to pass {@code gradOutput} or a saved
 *   tensor through unchanged it returns that tensor itself (the graph then takes the second reference with a recorded
 *   identity node) or uses {@link Autograd#reattach}. A backward computed from a backward primitive that does not
 *   record returns false.</li>
 *   <li>G5: each implementing class documents its formula, its saved tensors and its tolerance class.</li>
 * </ul>
 */
public interface GradientFunction {
    /**
     * The gradients with respect to the recorded inputs, in input order, given the gradient of the output. An entry
     * may be null for an input that needs no gradient; entries for inputs that do not require grad are discarded.
     */
    List<Tensor> backward(Tensor gradOutput);

    /** The tensors saved during the forward pass for {@link #backward}; never null. */
    List<Tensor> savedTensors();

    /**
     * Drops the saved tensors once the node is released. The inputs and outputs of recorded operations belong to the
     * graph or the caller and are never closed here; a function that saved references of its own (such as
     * {@link Tensor#detach()} views) closes those.
     */
    default void clearSavedTensors() {
    }

    /**
     * Whether {@link #backward} records a differentiable graph of itself inside a create-graph pass (rule G4). A
     * backward pass with {@code createGraph = true}, and {@link Autograd#grad} with it, throw before computing anything
     * when a node they would run returns false.
     */
    default boolean supportsCreateGraph() {
        return false;
    }
}
