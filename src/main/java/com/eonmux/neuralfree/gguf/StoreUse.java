package com.eonmux.neuralfree.gguf;

/**
 * How a {@link WeightStore} from {@link ModelSource#createMappedStore(StoreUse)} will be read. A source whose tensors
 * are all stored in its files serves every use alike; a source that derives tensors from others uses it to decide
 * where the derived bytes are kept.
 */
public enum StoreUse {
    /** One pass over the tensors placed on the device (the resident weight upload). */
    RESIDENT_UPLOAD,
    /** Repeated reads of the same tensors (weights streamed to the device layer by layer for every token). */
    STREAMING
}
