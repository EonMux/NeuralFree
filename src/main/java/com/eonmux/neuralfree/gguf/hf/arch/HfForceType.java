package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;

/**
 * A converter class's {@code tensor_force_quant}: the storage type of a planned tensor when the class fixes it,
 * regardless of the output type policy.
 */
@FunctionalInterface
public interface HfForceType {

    /** No override. */
    HfForceType NONE = (ggufName, nDims, source, outType) -> null;

    /**
     * The forced type of {@code ggufName}, or null when the class does not force one.
     *
     * @param nDims   the number of dimensions of the planned tensor
     * @param source  the element format of its source
     * @param outType the requested output type
     */
    GGMLType force(String ggufName, int nDims, ElementFormat source, ModelSourceOptions.OutType outType);
}
