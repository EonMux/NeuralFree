package com.eonmux.neuralfree.gguf.hub.slice;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The counts a model's {@code config.json} declares that a slice changes: layer counts (regular and MTP), expert
 * counts, vocab sizes and {@code first_k_dense_replace}. {@link #of} walks every nested object the way the reference
 * skill does, and {@link #patch} rewrites the entries whose values identify a kept family, returning a line per
 * change for the run's log. The config map is the mutable one {@link com.eonmux.neuralfree.utils.json.Json} parsed;
 * a model without a config has none of these counts and no patching.
 */
final class ConfigAnalysis {

    private static final Set<String> LAYER_KEYS = keys("num_hidden_layers", "num_layers", "n_layer", "n_layers",
            "num_layer", "num_decoder_layers", "encoder_layers", "decoder_layers", "depth",
            "num_encoder_layers", "num_attention_layers", "layer_count");
    private static final Set<String> VOCAB_KEYS = keys("vocab_size", "padded_vocab_size");
    private static final Set<String> NEXTN_KEYS = keys("num_nextn_predict_layers", "num_mtp_layers");
    private static final String[] TOPK_SUBSTRINGS = {"per_tok", "topk", "top_k", "moe_k"};
    private static final String[] EXPERT_KEY_EXCLUDES =
            {"per_tok", "topk", "top_k", "shared", "group", "intermediate", "size", "dim", "dtype"};
    private static final String FIRST_K_DENSE = "first_k_dense_replace";

    /** One scalar entry of a nested config object, mutable through {@link #set}. */
    private static final class Leaf {

        private final Map<String, Object> container;
        private final String key;
        private final Object value;

        private Leaf(Map<String, Object> container, String key, Object value) {
            this.container = container;
            this.key = key;
            this.value = value;
        }

        private void set(Object replacement) {
            container.put(key, replacement);
        }
    }

    private final Map<String, Object> config;
    private final Set<Long> layerCounts = new HashSet<>();
    private final Set<Long> nextnCounts = new HashSet<>();
    private final Set<Long> expertCounts = new HashSet<>();
    private final Set<Long> vocabSizes = new HashSet<>();
    private final long firstKDense;

    private ConfigAnalysis(Map<String, Object> config, long firstKDense) {
        this.config = config;
        this.firstKDense = firstKDense;
    }

    /**
     * Analyzes the parsed {@code config.json}; {@code null} (no config file) analyzes to no counts.
     *
     * @throws IllegalArgumentException when the document is not a JSON object
     */
    static ConfigAnalysis of(Object parsed) {
        if (parsed == null) {
            return new ConfigAnalysis(null, 0L);
        }
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("config.json is not a JSON object");
        }
        Map<String, Object> root = asStringMap(parsed);
        Set<Long> layerCounts = new HashSet<>();
        Set<Long> nextnCounts = new HashSet<>();
        Set<Long> expertCounts = new HashSet<>();
        Set<Long> vocabSizes = new HashSet<>();
        long firstKDense = 0L;
        for (Leaf leaf : leaves(root)) {
            Long value = integerOf(leaf.value);
            if (value == null || value.longValue() <= 0L) {
                continue;
            }
            long v = value.longValue();
            String key = leaf.key;
            if (LAYER_KEYS.contains(key)) {
                layerCounts.add(v);
            } else if (NEXTN_KEYS.contains(key)) {
                nextnCounts.add(v);
            } else if (VOCAB_KEYS.contains(key)) {
                vocabSizes.add(v);
            } else if (FIRST_K_DENSE.equals(key)) {
                firstKDense = v;
            } else if (key.contains("expert") && v > 1 && !containsAny(key, EXPERT_KEY_EXCLUDES)) {
                expertCounts.add(v);
            }
        }
        ConfigAnalysis analysis = new ConfigAnalysis(root, firstKDense);
        analysis.layerCounts.addAll(layerCounts);
        analysis.nextnCounts.addAll(nextnCounts);
        analysis.expertCounts.addAll(expertCounts);
        analysis.vocabSizes.addAll(vocabSizes);
        return analysis;
    }

    /** The mutable config object, or null when the model has no {@code config.json}. */
    Map<String, Object> config() {
        return config;
    }

    /** The layer counts the config declares. */
    Set<Long> layerCounts() {
        return Collections.unmodifiableSet(layerCounts);
    }

    /** The MTP (nextn) layer counts the config declares. */
    Set<Long> nextnCounts() {
        return Collections.unmodifiableSet(nextnCounts);
    }

    /** The expert counts the config declares. */
    Set<Long> expertCounts() {
        return Collections.unmodifiableSet(expertCounts);
    }

    /** The vocab sizes the config declares. */
    Set<Long> vocabSizes() {
        return Collections.unmodifiableSet(vocabSizes);
    }

    /** The number of leading dense layers of a model whose MoE layers follow them; 0 when it declares none. */
    long firstKDense() {
        return firstKDense;
    }

    /**
     * Rewrites the config entries that name the sliced counts, in place, and returns one line per change, as the
     * reference skill patches them: a layer count becomes the kept regular layers, {@code first_k_dense_replace}
     * counts the kept dense layers, an expert count becomes the kept experts, a top-k above them comes down to them,
     * and a vocab size named by {@code newVocab} shrinks. A per-layer list (such as {@code layer_types}) whose
     * length is a layer count is cut to the kept indices.
     *
     * @param families the planned tensor families
     * @param newVocab the vocab size to cut vocab-sized tensors to, or null to leave vocab entries alone
     */
    List<String> patch(Collection<TensorFamily> families, Integer newVocab) {
        List<String> log = new ArrayList<>();
        if (config == null) {
            return log;
        }
        Map<Long, TensorFamily> byCard = new LinkedHashMap<>();
        Long maxNewExperts = null;
        for (TensorFamily family : families) {
            if (family.kept() == null) {
                continue;
            }
            byCard.putIfAbsent((long) family.card(), family);
            byCard.putIfAbsent(family.regularTotal(), family);
            if (family.kind() == TensorFamily.Kind.EXPERT) {
                long newCount = family.kept().size();
                maxNewExperts = maxNewExperts == null ? newCount : Math.max(maxNewExperts, newCount);
            }
        }
        for (Leaf leaf : leaves(config)) {
            String key = leaf.key;
            Object value = leaf.value;
            Long integer = integerOf(value);
            if (integer != null) {
                long v = integer.longValue();
                TensorFamily family = byCard.get(v);
                Object replacement;
                if (LAYER_KEYS.contains(key) && family != null && family.kind() == TensorFamily.Kind.LAYER) {
                    replacement = family.newRegular();
                } else if (FIRST_K_DENSE.equals(key)) {
                    TensorFamily layers = firstLayer(families);
                    if (layers == null) {
                        continue;
                    }
                    long dense = 0L;
                    for (Long index : layers.kept()) {
                        if (index.longValue() < v) {
                            dense++;
                        }
                    }
                    replacement = dense;
                } else if (key.contains("expert") && family != null && family.kind() == TensorFamily.Kind.EXPERT
                        && !containsAny(key, EXPERT_KEY_EXCLUDES)) {
                    replacement = (long) family.kept().size();
                } else if (key.contains("expert") && containsAny(key, TOPK_SUBSTRINGS) && maxNewExperts != null
                        && v > maxNewExperts.longValue()) {
                    replacement = maxNewExperts;
                } else if (VOCAB_KEYS.contains(key) && newVocab != null && vocabSizes.contains(v)) {
                    replacement = (long) newVocab.intValue();
                } else {
                    continue;
                }
                if (!replacement.equals(value)) {
                    leaf.set(replacement);
                    log.add(key + ": " + v + " -> " + replacement);
                }
            } else if (value instanceof List && !((List<?>) value).isEmpty()
                    && allScalars((List<?>) value)) {
                List<?> list = (List<?>) value;
                TensorFamily family = byCard.get((long) list.size());
                if (family == null || family.kind() != TensorFamily.Kind.LAYER
                        || (list.size() != family.card() && list.size() != family.regularTotal())) {
                    continue;
                }
                List<Object> kept = new ArrayList<>();
                for (Long index : family.kept()) {
                    if (index.longValue() < list.size()) {
                        kept.add(list.get(index.intValue()));
                    }
                }
                leaf.set(kept);
                log.add(key + ": list[" + list.size() + "] -> list[" + kept.size() + "]");
            }
        }
        return log;
    }

    private static TensorFamily firstLayer(Collection<TensorFamily> families) {
        for (TensorFamily family : families) {
            if (family.kind() == TensorFamily.Kind.LAYER) {
                return family;
            }
        }
        return null;
    }

    private static boolean allScalars(List<?> list) {
        for (Object o : list) {
            if (o instanceof Map || o instanceof List) {
                return false;
            }
        }
        return true;
    }

    /** The leaves of a nested object, in document order; a list is a leaf. */
    private static List<Leaf> leaves(Map<String, Object> root) {
        List<Leaf> out = new ArrayList<>();
        walk(root, out);
        return out;
    }

    private static void walk(Map<String, Object> map, List<Leaf> out) {
        for (Map.Entry<String, Object> e : map.entrySet()) {
            if (e.getValue() instanceof Map) {
                walk(asStringMap(e.getValue()), out);
            } else {
                out.add(new Leaf(map, e.getKey(), e.getValue()));
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asStringMap(Object value) {
        return (Map<String, Object>) value;
    }

    /** The long value of a JSON integer, or null for every other value. */
    private static Long integerOf(Object value) {
        return value instanceof Long ? (Long) value : null;
    }

    private static boolean containsAny(String key, String[] substrings) {
        for (String s : substrings) {
            if (key.contains(s)) {
                return true;
            }
        }
        return false;
    }

    private static Set<String> keys(String... values) {
        Set<String> set = new HashSet<>();
        Collections.addAll(set, values);
        return Collections.unmodifiableSet(set);
    }
}
