package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGUFValueType;

/**
 * Where tensor hooks put their decisions while a Hugging Face checkpoint is planned: GGUF tensors with their sources,
 * metadata derived from tensor values (llama.cpp writes such keys from {@code modify_tensors} and
 * {@code prepare_tensors}, for example Apertus {@code xielu.*}), and the checkpoint tensors deliberately left out.
 * Hooks that combine several checkpoint tensors into one mark the others consumed so they are not visited again.
 */
public interface TensorSink {

    /**
     * Plans the GGUF tensor {@code ggufName} with the bytes of {@code src}.
     *
     * @throws IllegalArgumentException when the name is already planned or the source cannot be stored as a GGUF
     *                                  tensor
     */
    void add(String ggufName, RowSource src);

    /**
     * Records the metadata key {@code key} with an explicit GGUF type.
     *
     * @param type        the value type; {@link GGUFValueType#ARRAY} for an array
     * @param elementType the element type of an array, null otherwise
     * @param value       the value: a String, Boolean or Number for a scalar, a {@code List} or primitive array for
     *                    an array
     * @throws IllegalArgumentException when the value does not fit the type or the key is already recorded
     */
    void addKv(String key, GGUFValueType type, GGUFValueType elementType, Object value);

    /**
     * Records the scalar metadata key {@code key} with an explicit GGUF type; arrays use
     * {@link #addKv(String, GGUFValueType, GGUFValueType, Object)}.
     *
     * @throws IllegalArgumentException when the value does not fit the type or the key is already recorded
     */
    default void addKv(String key, GGUFValueType type, Object value) {
        addKv(key, type, null, value);
    }

    /** Records that the checkpoint tensor {@code hfName} is left out of the view, and why. */
    void drop(String hfName, String reason);

    /** True when a hook has already used {@code hfName} as part of another tensor. */
    boolean isConsumed(String hfName);

    /** Marks {@code hfName} as used, so the planner does not visit it on its own. */
    void consume(String hfName);
}
