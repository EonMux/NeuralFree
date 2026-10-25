package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;

import java.io.IOException;

/**
 * A converter class's {@code generate_extra_tensors}: tensors computed from the configuration or other tensors (rope
 * frequency factors, for example). Runs after every checkpoint tensor is planned.
 */
@FunctionalInterface
public interface HfExtraTensors {

    /** No extra tensors. */
    HfExtraTensors NONE = (ctx, out) -> { };

    /**
     * Plans the extra tensors.
     *
     * @throws IOException when tensor data the hook reads cannot be read
     */
    void generate(HfLoadContext ctx, TensorSink out) throws IOException;
}
