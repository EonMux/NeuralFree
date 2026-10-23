package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

/**
 * An {@link LLMModel} built on the spec-driven engine: it exposes its resolved {@link ModelSpec} and the
 * {@link ForwardBackend} that runs it, so drafters, the speculative decoder and the heads can drive the forward
 * passes directly.
 */
public interface EngineModel extends LLMModel {

    /** The backend running this model's forward passes. */
    ForwardBackend forwardBackend();

    /** The resolved model specification. */
    @Override
    ModelSpec spec();

    /** The support class of the resolved architecture. */
    @Override
    default SupportClass supportClass() {
        return spec().supportClass();
    }
}
