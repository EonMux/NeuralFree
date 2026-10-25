package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;

import java.io.IOException;

/**
 * A converter class's {@code set_gguf_parameters}: the architecture metadata. Runs after tensor planning, so it can
 * read planned tensors through {@code plan}. A hook whose class calls {@code super().set_gguf_parameters()} calls
 * {@link HfGenericMetadata#apply} first.
 */
@FunctionalInterface
public interface HfMetadataHook {

    /** {@code TextModel.set_gguf_parameters} alone. */
    HfMetadataHook GENERIC = (ctx, kv, plan) -> HfGenericMetadata.apply(ctx, kv);

    /**
     * Writes the metadata into {@code kv}.
     *
     * @throws IOException when tensor data the hook reads cannot be read
     */
    void apply(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException;
}
