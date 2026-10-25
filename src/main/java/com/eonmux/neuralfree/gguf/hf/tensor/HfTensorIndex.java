package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.io.safetensors.StTensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * The tensors of a Hugging Face checkpoint by their checkpoint names, in order, each with its {@link RowSource}. The
 * order is the checkpoint's ({@link CheckpointContainer#tensors()}) and is the order in which tensors are planned.
 * Index hooks and the quantization rewriter derive new indexes with {@link #withReplaced}. Immutable.
 */
public final class HfTensorIndex {

    private final List<String> names;
    private final Map<String, RowSource> sources;

    private HfTensorIndex(LinkedHashMap<String, RowSource> ordered) {
        this.names = Collections.unmodifiableList(new ArrayList<>(ordered.keySet()));
        this.sources = new HashMap<>(ordered);
    }

    /** Every tensor of {@code checkpoint}, in its order. */
    public static HfTensorIndex of(CheckpointContainer checkpoint) {
        Objects.requireNonNull(checkpoint, "checkpoint");
        LinkedHashMap<String, RowSource> m = new LinkedHashMap<>();
        for (StTensor t : checkpoint.tensors()) {
            if (m.put(t.name(), checkpoint.source(t.name())) != null) {
                throw new IllegalStateException("tensor '" + t.name() + "' appears twice in "
                        + checkpoint.description());
            }
        }
        return new HfTensorIndex(m);
    }

    /** An index of {@code ordered}'s entries in its iteration order. */
    public static HfTensorIndex of(Map<String, ? extends RowSource> ordered) {
        Objects.requireNonNull(ordered, "ordered");
        LinkedHashMap<String, RowSource> m = new LinkedHashMap<>();
        for (Map.Entry<String, ? extends RowSource> e : ordered.entrySet()) {
            m.put(Objects.requireNonNull(e.getKey(), "name"),
                    Objects.requireNonNull(e.getValue(), "source of " + e.getKey()));
        }
        return new HfTensorIndex(m);
    }

    /** The tensor names in order (unmodifiable). */
    public List<String> names() {
        return names;
    }

    /** True when the index holds {@code name}. */
    public boolean has(String name) {
        return sources.containsKey(name);
    }

    /**
     * The source of {@code name}.
     *
     * @throws IllegalArgumentException when the index does not hold it
     */
    public RowSource get(String name) {
        RowSource s = sources.get(name);
        if (s == null) {
            throw new IllegalArgumentException("the checkpoint has no tensor '" + name + "'");
        }
        return s;
    }

    /** The number of tensors. */
    public int size() {
        return names.size();
    }

    /**
     * A new index with {@code replace} applied and {@code remove} left out. A replaced name keeps its position. The
     * names of {@code replace} that this index does not hold are inserted, in {@code replace}'s iteration order, at
     * the position of the first name (in index order) that is replaced or removed (after it when it is replaced), or
     * appended when there is none.
     * A rewrite that turns one module's tensors into new ones therefore keeps that module where it was.
     *
     * @throws IllegalArgumentException when a name of {@code remove} is not in the index or is also in
     *                                  {@code replace}
     */
    public HfTensorIndex withReplaced(Map<String, ? extends RowSource> replace, Set<String> remove) {
        Objects.requireNonNull(replace, "replace");
        Objects.requireNonNull(remove, "remove");
        for (String r : remove) {
            if (!sources.containsKey(r)) {
                throw new IllegalArgumentException("cannot remove tensor '" + r + "': the index does not hold it");
            }
            if (replace.containsKey(r)) {
                throw new IllegalArgumentException("tensor '" + r + "' is both replaced and removed");
            }
        }
        LinkedHashMap<String, RowSource> added = new LinkedHashMap<>();
        for (Map.Entry<String, ? extends RowSource> e : replace.entrySet()) {
            Objects.requireNonNull(e.getValue(), "source of " + e.getKey());
            if (!sources.containsKey(Objects.requireNonNull(e.getKey(), "name"))) {
                added.put(e.getKey(), e.getValue());
            }
        }
        LinkedHashMap<String, RowSource> out = new LinkedHashMap<>();
        boolean inserted = added.isEmpty();
        for (String name : names) {
            boolean replaced = replace.containsKey(name);
            boolean removed = remove.contains(name);
            if (replaced) {
                out.put(name, replace.get(name));
            } else if (!removed) {
                out.put(name, sources.get(name));
            }
            if (!inserted && (replaced || removed)) {
                out.putAll(added);
                inserted = true;
            }
        }
        if (!inserted) {
            out.putAll(added);
        }
        return new HfTensorIndex(out);
    }

    @Override
    public String toString() {
        return "HfTensorIndex{" + names.size() + " tensors}";
    }
}
