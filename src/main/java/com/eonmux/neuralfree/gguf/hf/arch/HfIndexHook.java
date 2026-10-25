package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;

import java.io.IOException;

/**
 * A converter class's own {@code index_tensors}: tensors read from side files, renames the quantization rewrite
 * depends on, and index changes that need more than one name at a time. Runs before the quantization rewrite and the
 * generic {@link com.eonmux.neuralfree.gguf.hf.names.HfTensorFilters}. The per-name renames and drops of the class's
 * {@code filter_tensors} belong in its {@link HfNameFilter}, which side-file names go through as well.
 */
@FunctionalInterface
public interface HfIndexHook {

    /** The index unchanged. */
    HfIndexHook NONE = (ctx, index) -> index;

    /**
     * The adjusted index, derived with {@link HfTensorIndex#withReplaced}.
     *
     * @throws IOException when a side file cannot be read
     */
    HfTensorIndex apply(HfLoadContext ctx, HfTensorIndex index) throws IOException;
}
