package com.eonmux.neuralfree.gguf.hf.meta;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hf.names.HfNameTables;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The value type of every metadata key a Hugging Face checkpoint's GGUF view may carry, by {@link GgufKeys} template:
 * the generated {@code gguf_keys.tsv} (the type each {@code GGUFWriter.add_*} method writes, LC/gguf-py/gguf/
 * gguf_writer.py) plus the keys NeuralFree writes that the converter does not ({@code neuralfree.*}, and the
 * tokenizer and adapter keys llama.cpp reads but the converter's writer never emits).
 *
 * <p>Templates use the {@link GgufKeys} placeholders: {@code %s} for the architecture prefix (and for a template name
 * in {@code tokenizer.chat_template.%s} and {@code decision.temperature.%s}), {@code %d} for indexes. The table's
 * {@code {arch}}, {@code {name}}, {@code {id}}, {@code {n}} and {@code {i}} placeholders are rewritten to these, and
 * {@code {dense}} matches the {@code dense_<n>} module names.</p>
 *
 * <p>{@link KeyType#any()} marks the writer methods that take either an integer or a list
 * ({@code add_head_count(int | Sequence[int])}): an integer is written as UINT32, a list as an array whose element
 * type follows the values.</p>
 */
public final class GgufKeyTable {

    /** The type of one key template. Immutable. */
    public static final class KeyType {
        private final GGUFValueType type;
        private final GGUFValueType elementType;
        private final boolean any;
        private final String source;

        KeyType(GGUFValueType type, GGUFValueType elementType, boolean any, String source) {
            this.type = type;
            this.elementType = elementType;
            this.any = any;
            this.source = source;
        }

        /** The value type; for an {@link #any()} key the scalar type, UINT32. */
        public GGUFValueType type() { return type; }

        /** The element type of an array key, else null. */
        public GGUFValueType elementType() { return elementType; }

        /** True for keys whose writer takes an integer or a list. */
        public boolean any() { return any; }

        /** Where the type comes from: the gguf_writer method, or the NeuralFree producer. */
        public String source() { return source; }

        @Override
        public String toString() {
            return any ? "ANY" : type == GGUFValueType.ARRAY ? "ARRAY:" + elementType : type.name();
        }
    }

    private static final Pattern TSV_PLACEHOLDER = Pattern.compile("\\{([a-z]+)}");
    private static final String ARRAY_PREFIX = "ARRAY:";
    private static final String DENSE = "dense";

    private static final class Entry {
        final String template;
        final KeyType type;
        final Pattern templatePattern;

        Entry(String template, KeyType type, Pattern templatePattern) {
            this.template = template;
            this.type = type;
            this.templatePattern = templatePattern;
        }
    }

    private final Map<String, KeyType> exact;
    private final List<Entry> patterns;
    /** The templates with a placeholder besides a leading {@code %s.}, matched by pattern against concrete keys. */
    private final List<Entry> formatted;

    private GgufKeyTable() {
        Map<String, KeyType> ex = new LinkedHashMap<>();
        List<Entry> pat = new ArrayList<>();
        for (HfNameTables.TsvRow row : HfNameTables.read("gguf_keys.tsv",
                "writer_method", "key_template", "value_type").rows()) {
            KeyType t = parseType(row, row.get(2), row.get(0));
            addTsvRow(row, row.get(1), t, ex, pat);
        }
        for (Map.Entry<String, KeyType> e : supplementary().entrySet()) {
            if (ex.put(e.getKey(), e.getValue()) != null) {
                throw new IllegalStateException("GgufKeyTable: '" + e.getKey()
                        + "' is both in gguf_keys.tsv and in the NeuralFree key types");
            }
        }
        this.exact = Collections.unmodifiableMap(ex);
        this.patterns = Collections.unmodifiableList(pat);
        List<Entry> fmt = new ArrayList<>(pat);
        for (Map.Entry<String, KeyType> e : ex.entrySet()) {
            String body = e.getKey().startsWith("%s.") ? e.getKey().substring(3) : e.getKey();
            if (body.contains("%s") || body.contains("%d")) {
                fmt.add(new Entry(e.getKey(), e.getValue(), null));
            }
        }
        this.formatted = Collections.unmodifiableList(fmt);
    }

    private static final class Holder {
        static final GgufKeyTable TABLE = new GgufKeyTable();
    }

    /**
     * The type of the {@link GgufKeys} template {@code template}, or null when the table has no such key. A
     * {@code {dense}} template ({@code %s.{dense}_feat_in}) and the templates it covers ({@code %s.dense_2_feat_in})
     * share its type.
     */
    public static KeyType type(String template) {
        Objects.requireNonNull(template, "template");
        GgufKeyTable t = Holder.TABLE;
        KeyType k = t.exact.get(template);
        if (k != null) {
            return k;
        }
        for (Entry e : t.patterns) {
            if (e.template.equals(template) || e.templatePattern.matcher(template).matches()) {
                return e.type;
            }
        }
        return null;
    }

    /**
     * The template and type of the concrete key {@code key} of a model of architecture {@code arch}
     * ({@code llama.context_length}, {@code tokenizer.chat_template.rerank}), or null when no template matches. An
     * {@code <arch>.} prefix is read as {@code %s.}.
     */
    public static Map.Entry<String, KeyType> resolve(String key, String arch) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(arch, "arch");
        GgufKeyTable t = Holder.TABLE;
        KeyType k = t.exact.get(key);
        if (k != null) {
            return new AbstractMap.SimpleImmutableEntry<>(key, k);
        }
        String prefix = arch + ".";
        if (key.startsWith(prefix)) {
            String archTemplate = "%s." + key.substring(prefix.length());
            k = t.exact.get(archTemplate);
            if (k != null) {
                return new AbstractMap.SimpleImmutableEntry<>(archTemplate, k);
            }
        }
        for (Entry e : t.formatted) {
            if (concretePattern(e.template, arch).matcher(key).matches()) {
                return new AbstractMap.SimpleImmutableEntry<>(e.template, e.type);
            }
        }
        return null;
    }

    /** Every template the table types, in table order (unmodifiable). */
    public static List<String> templates() {
        GgufKeyTable t = Holder.TABLE;
        List<String> out = new ArrayList<>(t.exact.keySet());
        for (Entry e : t.patterns) {
            out.add(e.template);
        }
        return Collections.unmodifiableList(out);
    }

    // ------------------------------------------------------------------ construction

    private static void addTsvRow(HfNameTables.TsvRow row, String tsvTemplate, KeyType type,
                                  Map<String, KeyType> exact, List<Entry> patterns) {
        StringBuilder template = new StringBuilder();
        StringBuilder templateRegex = new StringBuilder();
        boolean dense = false;
        Matcher m = TSV_PLACEHOLDER.matcher(tsvTemplate);
        int last = 0;
        while (m.find()) {
            String literal = tsvTemplate.substring(last, m.start());
            template.append(literal);
            templateRegex.append(Pattern.quote(literal));
            String placeholder;
            switch (m.group(1)) {
                case "arch":
                    if (m.start() != 0) {
                        throw row.defect("{arch} must lead the key: " + tsvTemplate);
                    }
                    placeholder = "%s";
                    break;
                case "name":
                    placeholder = "%s";
                    break;
                case "id":
                case "n":
                case "i":
                    placeholder = "%d";
                    break;
                case DENSE:
                    dense = true;
                    placeholder = "{" + DENSE + "}";
                    break;
                default:
                    throw row.defect("unknown placeholder {" + m.group(1) + "} in " + tsvTemplate);
            }
            template.append(placeholder);
            templateRegex.append(DENSE.equals(m.group(1)) ? "dense_[0-9]+" : Pattern.quote(placeholder));
            last = m.end();
        }
        template.append(tsvTemplate.substring(last));
        templateRegex.append(Pattern.quote(tsvTemplate.substring(last)));
        if (dense) {
            patterns.add(new Entry(template.toString(), type, Pattern.compile(templateRegex.toString())));
        } else if (exact.put(template.toString(), type) != null) {
            throw row.defect("key " + tsvTemplate + " is listed twice");
        }
    }

    private static KeyType parseType(HfNameTables.TsvRow row, String spec, String writer) {
        if ("ANY".equals(spec)) {
            return new KeyType(GGUFValueType.UINT32, null, true, writer);
        }
        try {
            if (spec.startsWith(ARRAY_PREFIX)) {
                GGUFValueType elem = GGUFValueType.valueOf(spec.substring(ARRAY_PREFIX.length()));
                if (elem == GGUFValueType.ARRAY) {
                    throw row.defect("nested arrays are not supported: " + spec);
                }
                return new KeyType(GGUFValueType.ARRAY, elem, false, writer);
            }
            GGUFValueType t = GGUFValueType.valueOf(spec);
            if (t == GGUFValueType.ARRAY) {
                throw row.defect("an array type needs an element type: " + spec);
            }
            return new KeyType(t, null, false, writer);
        } catch (IllegalArgumentException e) {
            throw row.defect("unknown value type '" + spec + "'");
        }
    }

    /** {@code template} as a regex over concrete keys of {@code arch}. */
    private static Pattern concretePattern(String template, String arch) {
        StringBuilder sb = new StringBuilder();
        int i = 0;
        if (template.startsWith("%s.")) {
            sb.append(Pattern.quote(arch + "."));
            i = 3;
        }
        int from = i;
        while (i < template.length()) {
            if (template.startsWith("%s", i) || template.startsWith("%d", i)
                    || template.startsWith("{" + DENSE + "}", i)) {
                sb.append(Pattern.quote(template.substring(from, i)));
                if (template.startsWith("%s", i)) {
                    sb.append(".+");
                    i += 2;
                } else if (template.startsWith("%d", i)) {
                    sb.append("[0-9]+");
                    i += 2;
                } else {
                    sb.append("dense_[0-9]+");
                    i += DENSE.length() + 2;
                }
                from = i;
            } else {
                i++;
            }
        }
        sb.append(Pattern.quote(template.substring(from)));
        return Pattern.compile(sb.toString());
    }

    /**
     * Types of keys outside {@code gguf_keys.tsv}: the {@code neuralfree.*} keys (their {@link GgufKeys} javadoc names
     * the producer), the adapter keys of LC/convert_lora_to_gguf.py:427-454 and LC/conversion/bert.py:557-559, the
     * tokenizer ids llama.cpp reads as {@code uint32} (src/llama-vocab.cpp) and the remaining string keys.
     */
    private static Map<String, KeyType> supplementary() {
        Map<String, KeyType> m = new LinkedHashMap<>();
        String nf = "NeuralFree";
        put(m, GgufKeys.NF_EXTENSION_TYPES, GGUFValueType.BOOL, nf);
        put(m, GgufKeys.NF_SOURCE_FORMAT, GGUFValueType.STRING, nf);
        put(m, GgufKeys.NF_SOURCE_QUANT_METHOD, GGUFValueType.STRING, nf);
        putArray(m, GgufKeys.NF_HF_DROPPED_PREFIXES, GGUFValueType.STRING, nf);
        put(m, GgufKeys.NF_ACT_QUANT_VERSION, GGUFValueType.UINT32, nf);
        putArray(m, GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES, GGUFValueType.STRING, nf);
        putArray(m, GgufKeys.NF_ACT_QUANT_SCHEMES, GGUFValueType.INT32, nf);
        put(m, GgufKeys.NF_ACT_QUANT_SCALE_UB, GGUFValueType.FLOAT32, nf);
        put(m, GgufKeys.NF_SAMPLING_PRESENCE_PENALTY, GGUFValueType.FLOAT32, nf);
        put(m, GgufKeys.NF_SAMPLING_FREQUENCY_PENALTY, GGUFValueType.FLOAT32, nf);
        put(m, GgufKeys.NF_SAMPLING_MAX_NEW_TOKENS, GGUFValueType.INT32, nf);
        putArray(m, GgufKeys.NF_TOKENIZER_TOKEN_ATTRS, GGUFValueType.INT32, nf);
        put(m, GgufKeys.NF_TOKENIZER_NORMALIZER, GGUFValueType.STRING, nf);
        put(m, GgufKeys.NF_TOKENIZER_PRE_TOKENIZER, GGUFValueType.STRING, nf);
        put(m, GgufKeys.NF_TOKENIZER_DECODER, GGUFValueType.STRING, nf);
        put(m, GgufKeys.NF_TOKENIZER_PREPEND_SCHEME, GGUFValueType.STRING, nf);
        putArray(m, GgufKeys.NF_TOKENIZER_PREFIX_IDS, GGUFValueType.INT32, nf);
        putArray(m, GgufKeys.NF_TOKENIZER_SUFFIX_IDS, GGUFValueType.INT32, nf);
        putArray(m, GgufKeys.NF_TOKENIZER_EXTRA_EOG_IDS, GGUFValueType.INT32, nf);
        put(m, GgufKeys.NF_TOKENIZER_CLEAN_SPACES, GGUFValueType.BOOL, nf);
        put(m, GgufKeys.NF_TOKENIZER_BPE_UNK_FUSE, GGUFValueType.BOOL, nf);
        put(m, GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES, GGUFValueType.BOOL, nf);
        put(m, GgufKeys.NF_TOKENIZER_BPE_ADD_PREFIX_SPACE, GGUFValueType.BOOL, nf);
        put(m, GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK, GGUFValueType.BOOL, nf);
        put(m, GgufKeys.NF_TOKENIZER_BPE_CONTINUING_SUBWORD_PREFIX, GGUFValueType.STRING, nf);
        put(m, GgufKeys.NF_TOKENIZER_BPE_END_OF_WORD_SUFFIX, GGUFValueType.STRING, nf);

        put(m, GgufKeys.ADAPTER_TYPE, GGUFValueType.STRING, "convert_lora_to_gguf.py:427");
        put(m, GgufKeys.ADAPTER_LORA_ALPHA, GGUFValueType.FLOAT32, "convert_lora_to_gguf.py:431");
        putArray(m, GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS, GGUFValueType.UINT32, "convert_lora_to_gguf.py:449-454");
        put(m, GgufKeys.ADAPTER_LORA_TASK_NAME, GGUFValueType.STRING, "conversion/bert.py:557");
        put(m, GgufKeys.ADAPTER_LORA_PROMPT_PREFIX, GGUFValueType.STRING, "conversion/bert.py:559");

        String vocab = "llama-vocab.cpp (uint32_t)";
        for (String k : new String[] {GgufKeys.TOKENIZER_CLS_ID, GgufKeys.TOKENIZER_FIM_PRE_ID,
            GgufKeys.TOKENIZER_FIM_SUF_ID, GgufKeys.TOKENIZER_FIM_MID_ID, GgufKeys.TOKENIZER_FIM_PAD_ID,
            GgufKeys.TOKENIZER_FIM_REP_ID, GgufKeys.TOKENIZER_FIM_SEP_ID, GgufKeys.TOKENIZER_PREFIX_ID,
            GgufKeys.TOKENIZER_SUFFIX_ID, GgufKeys.TOKENIZER_MIDDLE_ID}) {
            put(m, k, GGUFValueType.UINT32, vocab);
        }
        put(m, GgufKeys.TOKENIZER_HF_JSON, GGUFValueType.STRING, "gguf_writer.add_string");
        put(m, GgufKeys.TOKENIZER_RWKV, GGUFValueType.STRING, "gguf_writer.add_string");
        put(m, GgufKeys.GENERAL_SOURCE_HF_REPO, GGUFValueType.STRING, "gguf_writer.add_string");
        put(m, GgufKeys.ROPE_SCALE_LINEAR, GGUFValueType.FLOAT32, "llama-model.cpp (float)");
        return m;
    }

    private static void put(Map<String, KeyType> m, String key, GGUFValueType type, String source) {
        m.put(key, new KeyType(type, null, false, source));
    }

    private static void putArray(Map<String, KeyType> m, String key, GGUFValueType elem, String source) {
        m.put(key, new KeyType(GGUFValueType.ARRAY, elem, false, source));
    }
}
