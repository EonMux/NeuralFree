package com.eonmux.neuralfree.gguf.engine;

/**
 * A one-token greedy decode a {@link ForwardBackend} has enqueued ({@link ForwardBackend#startGreedy},
 * {@link ForwardBackend#chainGreedy}): the step's token is stored in memory at its position when the call returns,
 * and {@link #id()} delivers the device's greedy selection over the step's logits once the step has run. A backend
 * that chains on the device ({@link ForwardBackend#chainsGreedyOnDevice()}) may still be running the step when the
 * handle is returned. Used by the thread that drives the backend.
 */
public interface GreedyStep {

    /**
     * Waits for the step and returns the id of its highest logit (lowest id on ties). Every call returns the same
     * id; a step that failed on the device, or read its input id from a step that did, throws that failure on every
     * call instead.
     */
    int id();
}
