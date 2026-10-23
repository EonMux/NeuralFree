package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/** Creates the CPU feed-forward blocks of one {@link FfnKind}; registered with {@link CpuBlockRegistry}. */
public interface CpuFfnFactory {

    /** The FFN kind this factory serves. */
    FfnKind kind();

    /**
     * An unbound block for a layer of {@link #kind()}.
     *
     * @throws UnsupportedOperationException when this layer's variant of the kind has no CPU implementation
     */
    CpuFfnBlock create(LayerSpec layer);
}
