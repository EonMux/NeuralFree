package com.eonmux.neuralfree.gguf.gpu.block;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/** Creates the GPU feed-forward blocks of one {@link FfnKind}; registered with {@link GpuBlockRegistry}. */
public interface GpuFfnFactory {

    /** The FFN kind this factory serves. */
    FfnKind kind();

    /**
     * An unbound block for a layer of {@link #kind()}.
     *
     * @throws UnsupportedOperationException when this layer's variant of the kind has no GPU implementation
     */
    GpuFfnBlock create(LayerSpec layer);
}
