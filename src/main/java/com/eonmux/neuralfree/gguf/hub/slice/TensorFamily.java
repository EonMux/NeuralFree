package com.eonmux.neuralfree.gguf.hub.slice;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableSet;
import java.util.Objects;
import java.util.TreeSet;

/**
 * One numeric slot of the tensor names of a model, the unit the slice renumbers: {@code model.layers} for
 * {@code model.layers.12.attn_q.weight}, {@code model.layers.#.mlp.experts} for
 * {@code model.layers.12.mlp.experts.3.w1.weight}, where {@code #} stands for an earlier numeric slot. A family is
 * a layer family (its indices continue with the config's layer count), an expert family (its container names
 * experts) or other (kept whole). {@link #kept()} is null for an "other" family and the sorted output indices
 * otherwise; {@link #renumber()} maps each kept input index to its output index. Not thread-safe; the planner fills
 * it and nothing changes afterwards.
 */
final class TensorFamily {

    /** What a numeric slot is: the model's layers, experts, or a slot the slice keeps whole. */
    enum Kind {
        LAYER,
        EXPERT,
        OTHER
    }

    /** One numeric path component of a tensor name: its position, the normalized prefix before it and its value. */
    static final class Slot {

        private final int position;
        private final String prefix;
        private final long index;

        private Slot(int position, String prefix, long index) {
            this.position = position;
            this.prefix = prefix;
            this.index = index;
        }

        /** The component's position in the dot-separated tensor name. */
        int position() {
            return position;
        }

        /** The path before the component, each earlier numeric component replaced by {@code #}. */
        String prefix() {
            return prefix;
        }

        /** The component's value. */
        long index() {
            return index;
        }
    }

    private final String prefix;
    private Kind kind;
    private final NavigableSet<Long> indices = new TreeSet<>();
    private List<Long> kept;
    private Map<Long, Long> renumber = Collections.emptyMap();
    private long regularTotal;
    private long newRegular;

    TensorFamily(String prefix, Kind kind) {
        this.prefix = Objects.requireNonNull(prefix, "prefix");
        this.kind = Objects.requireNonNull(kind, "kind");
    }

    /**
     * The numeric components of {@code name}, one per digit-only dot-separated component. A component too large for
     * a 64-bit index is no slot: no layer or expert of a model is numbered there.
     */
    static List<Slot> slots(String name) {
        String[] tokens = name.split("\\.", -1);
        List<Slot> out = new ArrayList<>();
        List<String> normalized = new ArrayList<>(tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            if (isDigits(tokens[i])) {
                Long value = parseIndex(tokens[i]);
                if (value != null) {
                    out.add(new Slot(i, String.join(".", normalized), value.longValue()));
                }
                normalized.add("#");
            } else {
                normalized.add(tokens[i]);
            }
        }
        return out;
    }

    private static boolean isDigits(String token) {
        if (token.isEmpty()) {
            return false;
        }
        for (int i = 0; i < token.length(); i++) {
            char c = token.charAt(i);
            if (c < '0' || c > '9') {
                return false;
            }
        }
        return true;
    }

    private static Long parseIndex(String digits) {
        try {
            return Long.valueOf(new BigInteger(digits).longValueExact());
        } catch (ArithmeticException e) {
            return null;
        }
    }

    /** The container segment of one normalized prefix: the text between the last dots. */
    static String container(String prefix) {
        int dot = prefix.lastIndexOf('.');
        return dot < 0 ? prefix : prefix.substring(dot + 1);
    }

    String prefix() {
        return prefix;
    }

    Kind kind() {
        return kind;
    }

    /** Adds one input index to the family. */
    void addIndex(long index) {
        indices.add(index);
    }

    /** How many distinct input indices the family has. */
    int card() {
        return indices.size();
    }

    /** The input indices, ascending. */
    NavigableSet<Long> indices() {
        return Collections.unmodifiableNavigableSet(indices);
    }

    /** Marks a family whose indices are layer indices, {@code regularTotal} of which are not the MTP tail. */
    void makeLayer(long regularTotal) {
        this.kind = Kind.LAYER;
        this.regularTotal = regularTotal;
    }

    /** How many of the family's indices are the model's regular layers, excluding any MTP tail. */
    long regularTotal() {
        return regularTotal;
    }

    /** How many regular layers the slice keeps. */
    long newRegular() {
        return newRegular;
    }

    void setNewRegular(long value) {
        this.newRegular = value;
    }

    /** The kept input indices, ascending; null keeps every index unchanged. */
    List<Long> kept() {
        return kept;
    }

    /** Keeps exactly {@code kept}, ascending, and builds the renumbering onto {@code 0..n-1}. */
    void keep(List<Long> kept) {
        List<Long> sorted = new ArrayList<>(kept);
        Collections.sort(sorted);
        List<Long> distinct = new ArrayList<>(sorted.size());
        for (Long index : sorted) {
            if (distinct.isEmpty() || !distinct.get(distinct.size() - 1).equals(index)) {
                distinct.add(index);
            }
        }
        this.kept = Collections.unmodifiableList(distinct);
        Map<Long, Long> map = new LinkedHashMap<>(distinct.size() * 2);
        for (int i = 0; i < distinct.size(); i++) {
            map.put(distinct.get(i), (long) i);
        }
        this.renumber = Collections.unmodifiableMap(map);
    }

    /** Keeps every index unchanged. */
    void keepAll() {
        this.kept = null;
        this.renumber = Collections.emptyMap();
    }

    /** The output index of one kept input index, or null when the index is dropped. */
    Long renumber(long index) {
        return renumber.get(index);
    }

    @Override
    public String toString() {
        return prefix + " kind=" + kind + " count=" + card() + " kept=" + (kept == null ? "all" : kept);
    }
}
