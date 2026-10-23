package com.eonmux.neuralfree.gguf.gpu.block;

import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/** Creates the GPU mixers of one {@link MixerKind}; registered with {@link GpuBlockRegistry}. */
public interface GpuMixerFactory {

    /** The mixer kind this factory serves. */
    MixerKind kind();

    /**
     * An unbound mixer for a layer of {@link #kind()}.
     *
     * @throws UnsupportedOperationException when this layer's variant of the kind has no GPU implementation
     */
    GpuMixer create(LayerSpec layer);
}
