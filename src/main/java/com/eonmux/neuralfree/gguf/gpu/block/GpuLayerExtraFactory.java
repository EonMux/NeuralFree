package com.eonmux.neuralfree.gguf.gpu.block;

import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/** Creates the GPU layer extras of one feature id ({@code LayerSpec.Extras}); registered with {@link GpuBlockRegistry}. */
public interface GpuLayerExtraFactory {

    /** The {@code LayerSpec.Extras} id this factory serves. */
    String featureId();

    /**
     * An unbound extra for a layer that lists {@link #featureId()}.
     *
     * @throws UnsupportedOperationException when this layer's variant of the feature has no GPU implementation
     */
    GpuLayerExtra create(LayerSpec layer);
}
