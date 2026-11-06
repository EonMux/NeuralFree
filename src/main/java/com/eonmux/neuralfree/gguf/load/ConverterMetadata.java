package com.eonmux.neuralfree.gguf.load;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * The authorship metadata overrides of llama.cpp's converters: the {@code --metadata} JSON object file
 * (gguf-py {@code Metadata.load} with {@code load_metadata_override}, written by {@code set_gguf_meta_model}) and
 * the {@code --model-name} that wins over it, as {@link MetadataOverrides.Override}s on an exported source's GGUF
 * view ({@link #apply}). Immutable.
 *
 * <p>The keys the converter reads keep the types its writer stores:</p>
 * <ul>
 *   <li>the string keys {@code general.name}, {@code author}, {@code version}, {@code organization},
 *       {@code finetune}, {@code basename}, {@code description}, {@code quantized_by}, {@code size_label},
 *       {@code license.name}, {@code license.link}, {@code url}, {@code doi}, {@code uuid}, {@code repo_url},
 *       {@code source.url}, {@code source.doi}, {@code source.uuid}, {@code source.repo_url} and
 *       {@code sampling.sequence}: a JSON string, STRING;</li>
 *   <li>{@code general.sampling.top_k}, {@code penalty_last_n} and {@code mirostat}: a JSON integer, INT32;</li>
 *   <li>{@code general.sampling.top_p}, {@code min_p}, {@code xtc_probability}, {@code xtc_threshold},
 *       {@code temp}, {@code penalty_repeat}, {@code mirostat_tau} and {@code mirostat_eta}: a JSON number,
 *       FLOAT32;</li>
 *   <li>{@code general.tags} and {@code general.languages}: a non-empty array of strings, ARRAY of STRING;</li>
 *   <li>{@code general.base_models} and {@code general.datasets}: an array of objects with the string fields
 *       {@code name}, {@code author}, {@code version}, {@code organization}, {@code description}, {@code url},
 *       {@code doi}, {@code uuid} and {@code repo_url}, written as {@code general.base_model.count} (UINT32) and
 *       {@code general.base_model.<i>.<field>} (and the {@code general.dataset.*} keys); the list replaces the
 *       source's, so every {@code general.base_model.*} key of the source is removed first.</li>
 * </ul>
 * <p>Any other key, which the converter ignores, is written as the GGUF type of its JSON value, as an
 * {@code --override-kv} entry would: a string STRING, a boolean BOOL, an integer INT32 (INT64 outside 32 bits), a
 * decimal FLOAT32 (FLOAT64 outside the float range), and a non-empty array of one kind an ARRAY of that type
 * (integers widening to INT64 and a list mixing integers and decimals stored as floats). Every refusal names the
 * flag and the key: a null, a nested object, an empty or mixed array, a value of the wrong kind for a known key,
 * an unknown field of a base model or dataset entry.</p>
 */
public final class ConverterMetadata {

    /** The {@code --metadata} list key the converter expands into {@code general.base_model.*}. */
    public static final String BASE_MODELS = "general.base_models";
    /** The {@code --metadata} list key the converter expands into {@code general.dataset.*}. */
    public static final String DATASETS = "general.datasets";

    private static final Set<String> STRING_KEYS = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
            GgufKeys.GENERAL_SAMPLING_SEQUENCE, GgufKeys.GENERAL_NAME, GgufKeys.GENERAL_AUTHOR,
            GgufKeys.GENERAL_VERSION, GgufKeys.GENERAL_ORGANIZATION, GgufKeys.GENERAL_FINETUNE,
            GgufKeys.GENERAL_BASENAME, GgufKeys.GENERAL_DESCRIPTION, GgufKeys.GENERAL_QUANTIZED_BY,
            GgufKeys.GENERAL_SIZE_LABEL, GgufKeys.GENERAL_LICENSE_NAME, GgufKeys.GENERAL_LICENSE_LINK,
            GgufKeys.GENERAL_URL, GgufKeys.GENERAL_DOI, GgufKeys.GENERAL_UUID, GgufKeys.GENERAL_REPO_URL,
            GgufKeys.GENERAL_SOURCE_URL, GgufKeys.GENERAL_SOURCE_DOI, GgufKeys.GENERAL_SOURCE_UUID,
            GgufKeys.GENERAL_SOURCE_REPO_URL)));

    private static final Set<String> INT32_KEYS = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
            GgufKeys.GENERAL_SAMPLING_TOP_K, GgufKeys.GENERAL_SAMPLING_PENALTY_LAST_N,
            GgufKeys.GENERAL_SAMPLING_MIROSTAT)));

    private static final Set<String> FLOAT32_KEYS = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
            GgufKeys.GENERAL_SAMPLING_TOP_P, GgufKeys.GENERAL_SAMPLING_MIN_P,
            GgufKeys.GENERAL_SAMPLING_XTC_PROBABILITY, GgufKeys.GENERAL_SAMPLING_XTC_THRESHOLD,
            GgufKeys.GENERAL_SAMPLING_TEMP, GgufKeys.GENERAL_SAMPLING_PENALTY_REPEAT,
            GgufKeys.GENERAL_SAMPLING_MIROSTAT_TAU, GgufKeys.GENERAL_SAMPLING_MIROSTAT_ETA)));

    private static final Set<String> STRING_ARRAY_KEYS = Collections.unmodifiableSet(new LinkedHashSet<>(
            Arrays.asList("general.tags", "general.languages")));

    /** The fields of one base model or dataset entry, in the converter's write order. */
    private static final List<String> ENTRY_FIELDS = Collections.unmodifiableList(Arrays.asList(
            "name", "author", "version", "organization", "description", "url", "doi", "uuid", "repo_url"));

    private static final ConverterMetadata NONE = new ConverterMetadata(Collections.emptyList(),
            Collections.emptyList());

    private final List<MetadataOverrides.Override> overrides;
    /** The key prefixes of the source the overrides replace as a whole ({@code general.base_model.}). */
    private final List<String> replacedPrefixes;

    private ConverterMetadata(List<MetadataOverrides.Override> overrides, List<String> replacedPrefixes) {
        this.overrides = Collections.unmodifiableList(new ArrayList<>(overrides));
        this.replacedPrefixes = Collections.unmodifiableList(new ArrayList<>(replacedPrefixes));
    }

    /** No overrides: {@link #apply} returns the source itself. */
    public static ConverterMetadata none() {
        return NONE;
    }

    /**
     * The overrides of one {@code --metadata} JSON object, in object order.
     *
     * @param what the flag and file the object came from, named in errors
     * @throws IllegalArgumentException naming {@code what} and the key of a value the converter's file cannot hold
     */
    public static ConverterMetadata fromJson(Map<?, ?> object, String what) {
        Objects.requireNonNull(object, "object");
        Objects.requireNonNull(what, "what");
        List<MetadataOverrides.Override> out = new ArrayList<>();
        List<String> replaced = new ArrayList<>();
        for (Map.Entry<?, ?> e : object.entrySet()) {
            if (!(e.getKey() instanceof String) || ((String) e.getKey()).isEmpty()) {
                throw new IllegalArgumentException(what + ": key " + e.getKey() + " is not a metadata key");
            }
            String key = (String) e.getKey();
            Object v = e.getValue();
            if (key.equals(BASE_MODELS) || key.equals(DATASETS)) {
                String family = key.equals(BASE_MODELS) ? "general.base_model" : "general.dataset";
                out.addAll(entries(what, key, family, v));
                replaced.add(family + ".");
            } else {
                out.add(override(what, key, v));
            }
        }
        return new ConverterMetadata(out, replaced);
    }

    /**
     * This metadata with {@code general.name} set to {@code name} last, so it wins over the file
     * ({@code --model-name}, the converter's direct override after the file).
     *
     * @throws IllegalArgumentException for an empty name
     */
    public ConverterMetadata withModelName(String name) {
        Objects.requireNonNull(name, "name");
        if (name.isEmpty()) {
            throw new IllegalArgumentException("--model-name is empty");
        }
        List<MetadataOverrides.Override> l = new ArrayList<>(overrides);
        l.add(MetadataOverrides.Override.scalar(GgufKeys.GENERAL_NAME, GGUFValueType.STRING, name));
        return new ConverterMetadata(l, replacedPrefixes);
    }

    /** The overrides in order; later entries of a key win. */
    public List<MetadataOverrides.Override> overrides() {
        return overrides;
    }

    /** Whether there is nothing to override. */
    public boolean isEmpty() {
        return overrides.isEmpty();
    }

    /**
     * {@code src} with the overrides applied, the replaced {@code general.base_model.*} / {@code general.dataset.*}
     * keys of {@code src} removed; {@code src} itself when there is nothing to apply. The view closes {@code src}.
     */
    public ModelSource apply(ModelSource src) {
        Objects.requireNonNull(src, "src");
        if (overrides.isEmpty()) {
            return src;
        }
        Set<String> removed = new LinkedHashSet<>();
        for (String key : src.getMetadata().keySet()) {
            for (String prefix : replacedPrefixes) {
                if (key.startsWith(prefix)) {
                    removed.add(key);
                }
            }
        }
        return new OverriddenSource(src, overrides, removed);
    }

    // ------------------------------------------------------------------ keys

    /** The override of one key other than the two entry lists. */
    private static MetadataOverrides.Override override(String what, String key, Object v) {
        if (v == null) {
            throw bad(what, key, "null is not a GGUF metadata value");
        }
        if (v instanceof Map) {
            throw bad(what, key, "a JSON object is not a GGUF metadata value");
        }
        if (STRING_KEYS.contains(key)) {
            return MetadataOverrides.Override.scalar(key, GGUFValueType.STRING, string(what, key, v));
        }
        if (INT32_KEYS.contains(key)) {
            return MetadataOverrides.Override.scalar(key, GGUFValueType.INT32, int32(what, key, v));
        }
        if (FLOAT32_KEYS.contains(key)) {
            return MetadataOverrides.Override.scalar(key, GGUFValueType.FLOAT32, float32(what, key, v));
        }
        if (STRING_ARRAY_KEYS.contains(key)) {
            List<?> list = nonEmptyList(what, key, v);
            List<String> strings = new ArrayList<>(list.size());
            for (int i = 0; i < list.size(); i++) {
                strings.add(string(what, key + "[" + i + "]", list.get(i)));
            }
            return MetadataOverrides.Override.array(key, GGUFValueType.STRING, strings);
        }
        return v instanceof List ? genericArray(what, key, (List<?>) v) : genericScalar(what, key, v);
    }

    /**
     * The {@code general.base_model.*} or {@code general.dataset.*} keys of one entry list: the count, then each
     * entry's fields in the converter's order.
     */
    private static List<MetadataOverrides.Override> entries(String what, String key, String family, Object v) {
        if (!(v instanceof List)) {
            throw bad(what, key, "it is not an array of objects");
        }
        List<?> list = (List<?>) v;
        List<MetadataOverrides.Override> out = new ArrayList<>();
        out.add(MetadataOverrides.Override.scalar(family + ".count", GGUFValueType.UINT32, list.size()));
        for (int i = 0; i < list.size(); i++) {
            String at = key + "[" + i + "]";
            if (!(list.get(i) instanceof Map)) {
                throw bad(what, at, "it is not an object");
            }
            Map<?, ?> entry = (Map<?, ?>) list.get(i);
            for (Object field : entry.keySet()) {
                if (!ENTRY_FIELDS.contains(field)) {
                    throw bad(what, at + "." + field, "the field is not one of " + ENTRY_FIELDS);
                }
            }
            for (String field : ENTRY_FIELDS) {
                if (entry.containsKey(field)) {
                    out.add(MetadataOverrides.Override.scalar(family + "." + i + "." + field, GGUFValueType.STRING,
                            string(what, at + "." + field, entry.get(field))));
                }
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ values

    private static String string(String what, String key, Object v) {
        if (!(v instanceof String)) {
            throw bad(what, key, kind(v) + " is not a string");
        }
        return (String) v;
    }

    private static Integer int32(String what, String key, Object v) {
        if (!(v instanceof Long || v instanceof BigInteger)) {
            throw bad(what, key, kind(v) + " is not an integer");
        }
        try {
            return bigInteger((Number) v).intValueExact();
        } catch (ArithmeticException e) {
            throw bad(what, key, v + " does not fit INT32");
        }
    }

    private static Float float32(String what, String key, Object v) {
        if (!(v instanceof Number)) {
            throw bad(what, key, kind(v) + " is not a number");
        }
        double d = ((Number) v).doubleValue();
        if (!Double.isFinite(d) || Math.abs(d) > Float.MAX_VALUE) {
            throw bad(what, key, v + " is not a finite FLOAT32");
        }
        return (float) d;
    }

    private static List<?> nonEmptyList(String what, String key, Object v) {
        if (!(v instanceof List)) {
            throw bad(what, key, kind(v) + " is not an array");
        }
        List<?> list = (List<?>) v;
        if (list.isEmpty()) {
            throw bad(what, key, "an empty array has no GGUF element type to store it as");
        }
        return list;
    }

    /** A key the converter does not read: the GGUF type of its JSON string, boolean or number. */
    private static MetadataOverrides.Override genericScalar(String what, String key, Object v) {
        if (v instanceof String) {
            return MetadataOverrides.Override.scalar(key, GGUFValueType.STRING, v);
        }
        if (v instanceof Boolean) {
            return MetadataOverrides.Override.scalar(key, GGUFValueType.BOOL, v);
        }
        if (v instanceof Double) {
            double d = (Double) v;
            if (!Double.isFinite(d)) {
                throw bad(what, key, v + " is not finite");
            }
            return Math.abs(d) > Float.MAX_VALUE
                    ? MetadataOverrides.Override.scalar(key, GGUFValueType.FLOAT64, d)
                    : MetadataOverrides.Override.scalar(key, GGUFValueType.FLOAT32, (float) d);
        }
        if (v instanceof Long || v instanceof BigInteger) {
            BigInteger b = bigInteger((Number) v);
            if (b.bitLength() < 32) {
                return MetadataOverrides.Override.scalar(key, GGUFValueType.INT32, b.intValue());
            }
            if (b.bitLength() < 64) {
                return MetadataOverrides.Override.scalar(key, GGUFValueType.INT64, b.longValue());
            }
            throw bad(what, key, b + " does not fit INT32 or INT64");
        }
        throw bad(what, key, kind(v) + " is not a GGUF metadata value");
    }

    /** A key the converter does not read: an ARRAY of the one kind its elements share. */
    private static MetadataOverrides.Override genericArray(String what, String key, List<?> list) {
        nonEmptyList(what, key, list);
        boolean strings = false;
        boolean bools = false;
        boolean decimals = false;
        boolean integers = false;
        for (int i = 0; i < list.size(); i++) {
            Object e = list.get(i);
            if (e instanceof String) {
                strings = true;
            } else if (e instanceof Boolean) {
                bools = true;
            } else if (e instanceof Double) {
                decimals = true;
            } else if (e instanceof Long || e instanceof BigInteger) {
                integers = true;
            } else {
                throw bad(what, key + "[" + i + "]", kind(e) + " is not a GGUF array element");
            }
        }
        int kinds = (strings ? 1 : 0) + (bools ? 1 : 0) + (decimals || integers ? 1 : 0);
        if (kinds > 1) {
            throw bad(what, key, "its elements mix strings, booleans and numbers");
        }
        if (strings) {
            return MetadataOverrides.Override.array(key, GGUFValueType.STRING, list);
        }
        if (bools) {
            return MetadataOverrides.Override.array(key, GGUFValueType.BOOL, list);
        }
        return decimals ? floatArray(what, key, list) : intArray(what, key, list);
    }

    /** An integer array: INT32 when every element fits 32 bits, else INT64. */
    private static MetadataOverrides.Override intArray(String what, String key, List<?> list) {
        int bits = 0;
        for (Object e : list) {
            bits = Math.max(bits, bigInteger((Number) e).bitLength());
        }
        if (bits >= 64) {
            throw bad(what, key, "an element does not fit INT32 or INT64");
        }
        List<Object> out = new ArrayList<>(list.size());
        for (Object e : list) {
            BigInteger b = bigInteger((Number) e);
            out.add(bits < 32 ? (Object) b.intValue() : (Object) b.longValue());
        }
        return MetadataOverrides.Override.array(key, bits < 32 ? GGUFValueType.INT32 : GGUFValueType.INT64, out);
    }

    /** A decimal (or mixed integer and decimal) array: FLOAT32, or FLOAT64 when an element is past its range. */
    private static MetadataOverrides.Override floatArray(String what, String key, List<?> list) {
        boolean wide = false;
        List<Double> doubles = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            double d = ((Number) list.get(i)).doubleValue();
            if (!Double.isFinite(d)) {
                throw bad(what, key + "[" + i + "]", list.get(i) + " is not finite");
            }
            wide |= Math.abs(d) > Float.MAX_VALUE;
            doubles.add(d);
        }
        if (wide) {
            return MetadataOverrides.Override.array(key, GGUFValueType.FLOAT64, doubles);
        }
        List<Float> floats = new ArrayList<>(doubles.size());
        for (double d : doubles) {
            floats.add((float) d);
        }
        return MetadataOverrides.Override.array(key, GGUFValueType.FLOAT32, floats);
    }

    /** A JSON integer of the {@link com.eonmux.neuralfree.utils.json.Json} value model as a BigInteger. */
    private static BigInteger bigInteger(Number n) {
        return n instanceof BigInteger ? (BigInteger) n : BigInteger.valueOf(n.longValue());
    }

    private static String kind(Object v) {
        if (v == null) {
            return "null";
        }
        if (v instanceof List) {
            return "an array";
        }
        if (v instanceof Map) {
            return "an object";
        }
        return v instanceof String ? "the string '" + v + "'" : String.valueOf(v);
    }

    private static IllegalArgumentException bad(String what, String key, String reason) {
        return new IllegalArgumentException(what + ": '" + key + "': " + reason);
    }
}
