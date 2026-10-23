package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/** Creates the CPU mixers of one {@link MixerKind}; registered with {@link CpuBlockRegistry}. */
public interface CpuMixerFactory {

    /** The mixer kind this factory serves. */
    MixerKind kind();

    /**
     * An unbound mixer for a layer of {@link #kind()}.
     *
     * @throws UnsupportedOperationException when this layer's variant of the kind has no CPU implementation
     */
    CpuMixer create(LayerSpec layer);
}
