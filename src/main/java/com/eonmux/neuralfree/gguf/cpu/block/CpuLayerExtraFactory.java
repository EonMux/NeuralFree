package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/** Creates the CPU layer extras of one feature id ({@code LayerSpec.Extras}); registered with {@link CpuBlockRegistry}. */
public interface CpuLayerExtraFactory {

    /** The {@code LayerSpec.Extras} id this factory serves. */
    String featureId();

    /**
     * An unbound extra for a layer that lists {@link #featureId()}.
     *
     * @throws UnsupportedOperationException when this layer's variant of the feature has no CPU implementation
     */
    CpuLayerExtra create(LayerSpec layer);
}
