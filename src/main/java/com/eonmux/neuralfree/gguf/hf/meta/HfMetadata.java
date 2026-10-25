package com.eonmux.neuralfree.gguf.hf.meta;

import com.eonmux.neuralfree.gguf.GGUFValueType;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * The metadata of a Hugging Face checkpoint's GGUF view: keys in write order, each with its value in the
 * {@link GGUFValueType} Java value model (arrays as unmodifiable {@code List<Object>}), its type and, for arrays, its
 * element type. Built by {@link HfMetadataBuilder#build()}. Immutable.
 */
public final class HfMetadata {

    private final Map<String, Object> values;
    private final Map<String, GGUFValueType> types;
    private final Map<String, GGUFValueType> elementTypes;

    HfMetadata(Map<String, Object> values, Map<String, GGUFValueType> types,
               Map<String, GGUFValueType> elementTypes) {
        this.values = Collections.unmodifiableMap(new LinkedHashMap<>(values));
        this.types = Collections.unmodifiableMap(new LinkedHashMap<>(types));
        this.elementTypes = Collections.unmodifiableMap(new LinkedHashMap<>(elementTypes));
    }

    /** Key to value, in write order (unmodifiable). */
    public Map<String, Object> values() {
        return values;
    }

    /** Key to value type, in write order (unmodifiable). */
    public Map<String, GGUFValueType> types() {
        return types;
    }

    /** Array key to element type (unmodifiable); scalar keys are absent. */
    public Map<String, GGUFValueType> elementTypes() {
        return elementTypes;
    }

    /** The keys in write order (unmodifiable). */
    public Set<String> keys() {
        return values.keySet();
    }

    /** True when {@code key} is present. */
    public boolean has(String key) {
        return values.containsKey(key);
    }

    /** The value of {@code key}, or null when absent. */
    public Object get(String key) {
        return values.get(Objects.requireNonNull(key, "key"));
    }

    /** The value type of {@code key}, or null when absent. */
    public GGUFValueType type(String key) {
        return types.get(Objects.requireNonNull(key, "key"));
    }

    /** The element type of the array key {@code key}, or null when absent or scalar. */
    public GGUFValueType elementType(String key) {
        return elementTypes.get(Objects.requireNonNull(key, "key"));
    }

    /** The number of keys. */
    public int size() {
        return values.size();
    }

    @Override
    public String toString() {
        return "HfMetadata{" + values.size() + " keys}";
    }
}
