package com.eonmux.neuralfree.gguf;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * A model checkpoint presented in the GGUF contract: GGUF-keyed metadata with GGUF value types, a tensor table with
 * GGUF names, {@code ne} dims, {@link GGMLType}s and virtual offsets, and the tensor bytes behind those offsets. A GGUF
 * file is its own source ({@link GGUFFileParser}); other checkpoint formats present exactly what the GGUF converter
 * would have written, so every consumer after this seam (hparams, tensor specs, weight stores, tokenizers) is
 * format-neutral. Sources are opened through {@link ModelSources}.
 */
public interface ModelSource extends AutoCloseable {

    /** The container format the source reads. */
    ModelFormat format();

    /** The file or directory the source was opened from, for messages. */
    String getFilePath();

    /**
     * Unmodifiable metadata in the GGUF value model: String, Boolean, Byte, Short, Integer, Long, Float, Double and
     * {@code List<Object>} arrays of those.
     */
    Map<String, Object> getMetadata();

    /** GGUF type of a metadata value, or null when the key is absent. */
    GGUFValueType getValueType(String key);

    /** Element type of an array metadata value, or null when the key is absent or not an array. */
    GGUFValueType getArrayElementType(String key);

    /** A string metadata value, or null when the key is absent or not a string. */
    String getString(String key);

    /** Unmodifiable tensor table: GGUF names, {@code ne} dims, storage types and virtual offsets. */
    List<GGUFTensorInfo> getTensors();

    /** Tensor count the source declares (for a split GGUF, the count in the header of the file opened). */
    long getTensorCount();

    /**
     * A new store reading tensor bytes by virtual offset; the caller closes it.
     *
     * @param use how the store will be read
     * @throws IOException when the tensor files cannot be opened
     */
    WeightStore createMappedStore(StoreUse use) throws IOException;

    /** {@link #createMappedStore(StoreUse)} for the resident weight upload. */
    default WeightStore createMappedStore() throws IOException {
        return createMappedStore(StoreUse.RESIDENT_UPLOAD);
    }

    /**
     * New byte regions of the tensors for host-side readers; the caller closes them.
     *
     * @param mode whether regions are memory-mapped or read into direct buffers
     * @throws IOException when the tensor files cannot be opened
     */
    TensorRegions openRegions(TensorReadMode mode) throws IOException;

    @Override
    void close() throws IOException;
}
