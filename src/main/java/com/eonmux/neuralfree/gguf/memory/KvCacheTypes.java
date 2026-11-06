package com.eonmux.neuralfree.gguf.memory;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * The K and V row types of a cache chosen together. Most names are one {@link KvDType} for both sides, but vLLM's
 * TurboQuant presets ({@code turboquant_k8v4}, {@code turboquant_4bit_nc}, {@code turboquant_k3v4_nc},
 * {@code turboquant_3bit_nc}, vllm/model_executor/layers/quantization/turboquant/config.py TQ_PRESETS) name a key
 * stream and a value stream of different geometry, so one name sets both caches. {@code fp8_inc} is Intel Gaudi's
 * name for the E4M3 dtype and is the same cache as {@code fp8_e4m3}; {@code nvfp4} / {@code nvfp4_4over6} set both
 * sides to the NVFP4 layout; {@code fp8_ds_mla} / {@code nvfp4_ds_mla} are the DeepSeek MLA latent rows that fill the
 * one K row both sides read. Immutable.
 */
public final class KvCacheTypes {

    private static final Map<String, KvCacheTypes> PRESETS = buildPresets();

    private final KvDType typeK;
    private final KvDType typeV;

    private KvCacheTypes(KvDType typeK, KvDType typeV) {
        this.typeK = Objects.requireNonNull(typeK, "typeK");
        this.typeV = Objects.requireNonNull(typeV, "typeV");
    }

    private static Map<String, KvCacheTypes> buildPresets() {
        Map<String, KvCacheTypes> m = new LinkedHashMap<>();
        m.put("turboquant_k8v4", new KvCacheTypes(KvDType.TQ_K8_KEYS, KvDType.TQ_4BIT_VALUES));
        m.put("turboquant_4bit_nc", new KvCacheTypes(KvDType.TQ_4BIT_KEYS, KvDType.TQ_4BIT_VALUES));
        m.put("turboquant_k3v4_nc", new KvCacheTypes(KvDType.TQ_3BIT_KEYS, KvDType.TQ_4BIT_VALUES));
        m.put("turboquant_3bit_nc", new KvCacheTypes(KvDType.TQ_3BIT_KEYS, KvDType.TQ_3BIT_VALUES));
        return Collections.unmodifiableMap(m);
    }

    /** The names of the TurboQuant presets, in the order they are listed. */
    public static List<String> presets() {
        return Collections.unmodifiableList(new java.util.ArrayList<>(PRESETS.keySet()));
    }

    /** The K row type. */
    public KvDType typeK() {
        return typeK;
    }

    /** The V row type. */
    public KvDType typeV() {
        return typeV;
    }

    /** A pair of one type on both sides. */
    public static KvCacheTypes of(KvDType type) {
        return new KvCacheTypes(type, type);
    }

    /** A pair of the two types. */
    public static KvCacheTypes of(KvDType typeK, KvDType typeV) {
        return new KvCacheTypes(typeK, typeV);
    }

    /** The accepted names: the preset names {@link KvDType#names()} appends, for error messages. */
    public static String names() {
        return String.join(", ", presets());
    }

    /**
     * Parses a cache type name: a TurboQuant preset (case-insensitive), or any name {@link KvDType#parse} accepts, on
     * both sides. A pair of vendor types that do not belong together is refused by name.
     *
     * @param property name of the property or flag being parsed, for the error message
     * @throws IllegalArgumentException for an unknown name or an inconsistent pair
     */
    public static KvCacheTypes parse(String value, String property) {
        if (value == null) {
            throw new IllegalArgumentException(property + " is not set");
        }
        String v = value.trim().toLowerCase(Locale.ROOT);
        KvCacheTypes preset = PRESETS.get(v);
        if (preset != null) {
            return preset;
        }
        return of(KvDType.parse(value, property));
    }

    /**
     * This pair with both sides replaced by {@code type}; used by the per-side settings.
     *
     * @throws IllegalArgumentException when a per-side setting would split a TurboQuant key and value stream
     */
    public static KvCacheTypes checkPair(KvDType typeK, KvDType typeV, String property) {
        boolean consistent = typeK.nvfp4() == typeV.nvfp4()
                && typeK.latentFormat() == typeV.latentFormat()
                && (typeK.turboQuantKey() == typeV.turboQuantValue())
                && (typeK.turboQuantValue() == false)
                && (typeV.turboQuantKey() == false);
        if (!consistent) {
            throw new IllegalArgumentException(property + ": " + typeK.name().toLowerCase(Locale.ROOT) + " on K and "
                    + typeV.name().toLowerCase(Locale.ROOT) + " on V do not form a cache format; a TurboQuant key"
                    + " stream pairs with a TurboQuant value stream, NVFP4 and the DeepSeek MLA latent rows fill both"
                    + " sides, and every other type fills K and V independently through " + MemoryConfig.PROP_KV_TYPE_K
                    + " / " + MemoryConfig.PROP_KV_TYPE_V + " (or a turboquant_* preset with "
                    + MemoryConfig.PROP_KV_TYPE + ")");
        }
        return of(typeK, typeV);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof KvCacheTypes)) {
            return false;
        }
        KvCacheTypes other = (KvCacheTypes) o;
        return typeK == other.typeK && typeV == other.typeV;
    }

    @Override
    public int hashCode() {
        return 31 * typeK.hashCode() + typeV.hashCode();
    }

    @Override
    public String toString() {
        return typeK == typeV ? typeK.name() : typeK.name() + "/" + typeV.name();
    }
}
