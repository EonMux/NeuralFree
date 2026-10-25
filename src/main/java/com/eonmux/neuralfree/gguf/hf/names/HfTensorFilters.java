package com.eonmux.neuralfree.gguf.hf.names;

import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.UnaryOperator;

/**
 * The converter's name-level tensor filters, applied to a checkpoint index before any tensor hook sees it. The
 * entry's {@link HfNameFilter} (the class's own {@code filter_tensors}) wraps the inherited filters, which are driven
 * by the {@code filters.tsv} rules of {@link HfNameTables}; for each name, in the converter's order:
 * <ol>
 *   <li>the class filter decides whether the tensor is kept. For {@link HfArchEntry.Kind#TEXT},
 *       {@code TextModel.filter_tensors} (base.py:1556-1573) drops multimodal towers: the name, with a leading
 *       {@code model.} removed for the test only, starting with a {@code drop_prefix} rule or containing a
 *       {@code drop_contains} rule. For {@link HfArchEntry.Kind#MMPROJ}, {@code MmprojModel.filter_tensors}
 *       (base.py:2905-2912) drops the language model: names containing a {@code skip_contains} rule;</li>
 *   <li>{@code ModelBase.filter_tensors} (base.py:585-594) renames a kept tensor: a name ending in a
 *       {@code rename_suffix} rule has every occurrence of it replaced, then every occurrence of a
 *       {@code rename_contains} rule is replaced;</li>
 *   <li>{@code prepare_tensors} never converts a name ending in a {@code skip_suffix} rule (base.py:1222-1224).</li>
 * </ol>
 * The drop tests run on the checkpoint name, before the renames, as in the converter.
 */
public final class HfTensorFilters {

    /** The filtered index and the dropped tower prefixes. Immutable. */
    public static final class Result {
        private final HfTensorIndex index;
        private final List<String> droppedPrefixes;

        Result(HfTensorIndex index, List<String> droppedPrefixes) {
            this.index = index;
            this.droppedPrefixes = droppedPrefixes;
        }

        /** The kept tensors under their filtered names, in checkpoint order. */
        public HfTensorIndex index() {
            return index;
        }

        /**
         * The name prefixes of the dropped tensors (the checkpoint name up to the end of the dotted component that
         * matched a drop rule, such as {@code model.visual} or {@code vision_tower}), distinct, in first-seen order
         * (unmodifiable). Skipped suffixes are not towers and are not listed.
         */
        public List<String> droppedPrefixes() {
            return droppedPrefixes;
        }
    }

    private HfTensorFilters() {
    }

    /**
     * Filters {@code index} for a checkpoint of {@code kind} through the name filter of {@code ctx.entry()} bound to
     * {@code ctx}.
     *
     * @throws GgufFormatException when two kept tensors get the same filtered name
     */
    public static Result apply(HfLoadContext ctx, HfTensorIndex index, HfArchEntry.Kind kind) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(index, "index");
        Objects.requireNonNull(kind, "kind");
        HfNameFilter classFilter = ctx.entry().nameFilter(ctx);
        LinkedHashMap<String, RowSource> kept = new LinkedHashMap<>();
        Map<String, String> origin = new LinkedHashMap<>();
        Set<String> dropped = new LinkedHashSet<>();
        String[] tower = new String[1];
        UnaryOperator<String> inherited = n -> {
            tower[0] = droppedPrefix(n, kind);
            return tower[0] != null ? null : rename(n);
        };
        int skipped = 0;
        int classDropped = 0;
        for (String name : index.names()) {
            tower[0] = null;
            String renamed = classFilter.filter(name, inherited);
            if (renamed == null) {
                if (tower[0] != null) {
                    dropped.add(tower[0]);
                } else {
                    classDropped++;
                }
                continue;
            }
            if (isSkipped(renamed)) {
                skipped++;
                continue;
            }
            String first = origin.put(renamed, name);
            if (first != null) {
                throw new GgufFormatException("checkpoint tensors '" + first + "' and '" + name
                        + "' both become '" + renamed + "' after the converter's renames");
            }
            kept.put(renamed, index.get(name));
        }
        if (!dropped.isEmpty() || skipped > 0 || classDropped > 0) {
            ctx.log("tensor filters: kept " + kept.size() + " of " + index.size() + " tensors"
                    + (dropped.isEmpty() ? "" : ", dropped " + dropped)
                    + (classDropped == 0 ? "" : ", dropped " + classDropped + " by the class filter")
                    + (skipped == 0 ? "" : ", skipped " + skipped + " unused buffers"));
        }
        return new Result(HfTensorIndex.of(kept),
                Collections.unmodifiableList(new ArrayList<>(dropped)));
    }

    /**
     * The converter's {@code filter_tensors} of {@code name} for the load {@code ctx}: the name filter of
     * {@code ctx.entry()} bound to {@code ctx}, around {@link #filterName(String, HfArchEntry.Kind)} for the entry's
     * kind; null when the name is dropped. The skip suffixes are not applied ({@link #isSkipped}). Used to bring names
     * from side files (such as the Hadamard packing manifest) into the filtered namespace.
     */
    public static String filterName(HfLoadContext ctx, String name) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(name, "name");
        HfArchEntry.Kind kind = ctx.entry().kind();
        return ctx.entry().nameFilter(ctx).filter(name, n -> filterName(n, kind));
    }

    /**
     * The inherited {@code filter_tensors} of {@code name} for a checkpoint of {@code kind} (the {@code TextModel} or
     * {@code MmprojModel} filter, then {@code ModelBase}'s renames): null when the name is dropped, else the renamed
     * name. The skip suffixes are not applied ({@link #isSkipped}).
     */
    public static String filterName(String name, HfArchEntry.Kind kind) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(kind, "kind");
        return droppedPrefix(name, kind) != null ? null : rename(name);
    }

    /** The {@code ModelBase.filter_tensors} renames of {@code name} (base.py:585-594). */
    public static String rename(String name) {
        String out = name;
        for (HfNameTables.FilterRule r : HfNameTables.filters(HfNameTables.FilterKind.RENAME_SUFFIX)) {
            if (out.endsWith(r.value())) {
                out = out.replace(r.value(), r.replacement());
            }
        }
        for (HfNameTables.FilterRule r : HfNameTables.filters(HfNameTables.FilterKind.RENAME_CONTAINS)) {
            if (out.contains(r.value())) {
                out = out.replace(r.value(), r.replacement());
            }
        }
        return out;
    }

    /** True when {@code prepare_tensors} never converts {@code name} (base.py:1222-1224). */
    public static boolean isSkipped(String name) {
        for (HfNameTables.FilterRule r : HfNameTables.filters(HfNameTables.FilterKind.SKIP_SUFFIX)) {
            if (name.endsWith(r.value())) {
                return true;
            }
        }
        return false;
    }

    /** The dropped prefix of {@code name} when the class filter of {@code kind} removes it, else null. */
    private static String droppedPrefix(String name, HfArchEntry.Kind kind) {
        if (kind == HfArchEntry.Kind.MMPROJ) {
            for (HfNameTables.FilterRule r : HfNameTables.filters(HfNameTables.FilterKind.SKIP_CONTAINS)) {
                int at = name.indexOf(r.value());
                if (at >= 0) {
                    return prefixThrough(name, at + r.value().length());
                }
            }
            return null;
        }
        int offset = 0;
        for (HfNameTables.FilterRule r : HfNameTables.filters(HfNameTables.FilterKind.STRIP_PREFIX)) {
            if (name.startsWith(r.value(), offset)) {
                offset += r.value().length();
            }
        }
        for (HfNameTables.FilterRule r : HfNameTables.filters(HfNameTables.FilterKind.DROP_PREFIX)) {
            if (name.startsWith(r.value(), offset)) {
                return prefixThrough(name, offset + r.value().length());
            }
        }
        for (HfNameTables.FilterRule r : HfNameTables.filters(HfNameTables.FilterKind.DROP_CONTAINS)) {
            int at = name.indexOf(r.value(), offset);
            if (at >= 0) {
                return prefixThrough(name, at + r.value().length());
            }
        }
        return null;
    }

    /**
     * {@code name} up to the end of the dotted component that ends at or contains position {@code end}: a match
     * ending in {@code .} closes its component, otherwise the component runs to the next {@code .}.
     */
    private static String prefixThrough(String name, int end) {
        if (end > 0 && name.charAt(end - 1) == '.') {
            return name.substring(0, end - 1);
        }
        int dot = name.indexOf('.', end);
        return dot < 0 ? name : name.substring(0, dot);
    }
}
