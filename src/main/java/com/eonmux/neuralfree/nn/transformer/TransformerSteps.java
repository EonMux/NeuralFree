package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.function.UnaryOperator;

/**
 * Chaining of the transformer blocks' forward steps without leaking intermediates, since device buffers have no
 * GC-based release. An intermediate is closed as soon as the next step has consumed it, also when it is the output of
 * a recorded node: the gradient functions hold their own references to what they saved and the graph routes gradients
 * by identity, so an early close never affects the backward pass, and the graph's own close of the output after the
 * backward pass is idempotent. The caller's input is never closed.
 */
final class TransformerSteps {

    private TransformerSteps() {
    }

    /**
     * {@code step} applied to {@code current}, after which {@code current} is closed unless it is {@code callerInput}
     * (null when there is none) or the step returned it; when the step fails, {@code current} is closed the same way
     * before the exception propagates.
     */
    static Tensor then(Tensor current, Tensor callerInput, UnaryOperator<Tensor> step) {
        Tensor next;
        try {
            next = step.apply(current);
        } catch (RuntimeException e) {
            closeUnless(current, callerInput, e);
            throw e;
        }
        if (next != current) {
            closeUnless(current, callerInput, null);
        }
        return next;
    }

    /** The residual sum {@code x + branch} (recorded with {@code AddGrad} in grad mode); {@code branch} is closed. */
    static Tensor residual(Tensor x, Tensor branch) {
        try {
            return TensorOps.add(x, branch);
        } finally {
            branch.close();
        }
    }

    private static void closeUnless(Tensor t, Tensor callerInput, RuntimeException failure) {
        if (t == callerInput) {
            return;
        }
        if (failure == null) {
            t.close();
            return;
        }
        try {
            t.close();
        } catch (RuntimeException e) {
            failure.addSuppressed(e);
        }
    }
}
