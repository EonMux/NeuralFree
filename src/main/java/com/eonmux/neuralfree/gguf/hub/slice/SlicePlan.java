package com.eonmux.neuralfree.gguf.hub.slice;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * What a slice keeps: the source shards, the renumbered tensor families, the tensors to fetch and the config
 * analysis their config is patched from. Built by {@link SlicePlanner}; immutable once built, except that
 * {@link #patchConfig} rewrites the mutable config map the analysis holds. The inspect report of {@link #inspect()}
 * is the reference skill's {@code --inspect} output.
 */
final class SlicePlan {

    private final List<String> weightFiles;
    private final Map<String, TensorFamily> families;
    private final List<TensorSlice> tensors;
    private final int totalTensors;
    private final long totalBytes;
    private final ConfigAnalysis config;

    SlicePlan(List<String> weightFiles, Map<String, TensorFamily> families, List<TensorSlice> tensors,
              int totalTensors, ConfigAnalysis config) {
        this.weightFiles = Collections.unmodifiableList(new ArrayList<>(weightFiles));
        this.families = Collections.unmodifiableMap(new LinkedHashMap<>(families));
        this.tensors = Collections.unmodifiableList(new ArrayList<>(tensors));
        this.totalTensors = totalTensors;
        long bytes = 0L;
        for (TensorSlice t : tensors) {
            bytes = Math.addExact(bytes, t.byteLength());
        }
        this.totalBytes = bytes;
        this.config = config;
    }

    /** The source shards, in the order their tensors are written. */
    List<String> weightFiles() {
        return weightFiles;
    }

    /** The tensor families in the order their prefixes first appear. */
    Map<String, TensorFamily> families() {
        return families;
    }

    /** The kept tensors, in output order. */
    List<TensorSlice> tensors() {
        return tensors;
    }

    /** How many tensors the source shards hold. */
    int totalTensors() {
        return totalTensors;
    }

    /** The byte length of the kept tensors. */
    long totalBytes() {
        return totalBytes;
    }

    /** The config analysis of the model. */
    ConfigAnalysis config() {
        return config;
    }

    /** Whether any family was recognized as the model's layers. */
    boolean hasLayerFamily() {
        for (TensorFamily family : families.values()) {
            if (family.kind() == TensorFamily.Kind.LAYER) {
                return true;
            }
        }
        return false;
    }

    /** Patches the config to the slice and returns one line per change. */
    List<String> patchConfig(Integer newVocab) {
        return config.patch(families.values(), newVocab);
    }

    /**
     * The reference skill's {@code --inspect} text: the config counts, the tensor families with the indices they
     * keep, the kept tensors (one per family, or every row-sliced one) and the kept total.
     */
    String inspect() {
        StringBuilder out = new StringBuilder();
        Set<Long> vocab = config.vocabSizes();
        out.append("config analysis: layer counts ").append(sorted(config.layerCounts()))
                .append("  nextn ").append(sorted(config.nextnCounts()))
                .append("  experts ").append(sorted(config.expertCounts()))
                .append("  vocab ").append(sorted(vocab))
                .append("  first_k_dense ").append(config.firstKDense())
                .append('\n');
        out.append('\n').append("numeric families in tensor names:\n");
        List<TensorFamily> sortedFamilies = new ArrayList<>(families.values());
        sortedFamilies.sort((a, b) -> a.prefix().compareTo(b.prefix()));
        for (TensorFamily family : sortedFamilies) {
            String kept = family.kept() == null ? "all" : abbreviate(family.kept());
            out.append(String.format(Locale.ROOT, "  %-60s kind=%-6s count=%4d kept=%s%n",
                    family.prefix(), family.kind().name().toLowerCase(Locale.ROOT), family.card(), kept));
        }
        out.append('\n').append("kept tensors (first index of each family; row-sliced ones always):\n");
        Set<String> seen = new HashSet<>();
        for (TensorSlice t : tensors) {
            String key = t.outputName().replaceAll("\\.\\d+\\.", ".#.");
            if (!seen.add(key) && !t.rowSliced()) {
                continue;
            }
            String mark = t.rowSliced()
                    ? "  (sliced from " + Arrays.toString(t.originalShape()) + ")" : "";
            out.append(String.format(Locale.ROOT, "  %-70s %-5s %s%s%n", t.outputName(),
                    t.dtype().headerName(), Arrays.toString(t.outputShape()), mark));
        }
        int dropped = totalTensors - tensors.size();
        out.append('\n').append(tensors.size()).append('/').append(totalTensors).append(" tensors kept, ")
                .append(dropped).append(" dropped, ").append(human(totalBytes)).append('\n');
        return out.toString();
    }

    private static String abbreviate(List<Long> kept) {
        if (kept.size() <= 16) {
            return kept.toString();
        }
        return kept.subList(0, 16) + "...";
    }

    private static String sorted(Set<Long> values) {
        List<Long> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        return sorted.toString();
    }

    /** A byte count in the reference skill's units: bytes, then one decimal of KB, MB, GB or TB. */
    static String human(long bytes) {
        if (bytes < 1024) {
            return bytes + "B";
        }
        double value = bytes;
        for (String unit : new String[]{"KB", "MB", "GB", "TB"}) {
            value /= 1024.0;
            if (value < 1024.0 || "TB".equals(unit)) {
                return String.format(Locale.ROOT, "%.1f%s", value, unit);
            }
        }
        return bytes + "B";
    }
}
