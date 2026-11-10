package com.eonmux.neuralfree.gguf.engine.sampling;

import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The masks a grammar stage computed, keyed by the grammar state they were computed in: the set of partial parses,
 * the partial UTF-8 tail and the row width. The mask of a vocabulary-ordered row is a function of that state alone
 * (end-of-generation admission follows from the parses), and inside a string or a repeated rule the state recurs
 * token after token, so a recurring state skips the whole rejection walk. A least-recently-used map whose entry count
 * keeps the memo within {@link #BUDGET_BYTES}; owned by one stage, not thread-safe.
 */
final class GrammarMaskMemo {

    /** The memory the memo's masks may hold. */
    static final int BUDGET_BYTES = 512 * 1024;
    private static final int MIN_ENTRIES = 2;
    private static final int MAX_ENTRIES = 64;

    private static final Comparator<GrammarAutomaton.Stack> ORDER = (a, b) -> {
        int c = Integer.compare(a.hashCode(), b.hashCode());
        return c != 0 ? c : Arrays.compare(a.entries, b.entries);
    };

    /** One grammar state: its parses in a canonical order, the partial tail and the row width. */
    static final class Key {
        private final GrammarAutomaton.Stack[] stacks;
        private final int partialValue;
        private final int partialRemain;
        private final int width;
        private final int hash;

        Key(GrammarAutomaton.Stack[] stacks, int partialValue, int partialRemain, int width) {
            GrammarAutomaton.Stack[] sorted = stacks.clone();
            Arrays.sort(sorted, ORDER);
            this.stacks = sorted;
            this.partialValue = partialValue;
            this.partialRemain = partialRemain;
            this.width = width;
            this.hash = ((Arrays.hashCode(sorted) * 31 + partialValue) * 31 + partialRemain) * 31 + width;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Key)) {
                return false;
            }
            Key k = (Key) o;
            return hash == k.hash && width == k.width && partialValue == k.partialValue
                    && partialRemain == k.partialRemain && Arrays.equals(stacks, k.stacks);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }

    private final Map<Key, long[]> masks;

    /** A memo for rows of {@code width} candidates. */
    GrammarMaskMemo(int width) {
        long perEntry = Math.max(8L, ((long) width + 63) / 64 * 8);
        int capacity = (int) Math.max(MIN_ENTRIES, Math.min(MAX_ENTRIES, BUDGET_BYTES / perEntry));
        this.masks = new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Key, long[]> eldest) {
                return size() > capacity;
            }
        };
    }

    /** The rejected row indices of {@code key} as a bit set, or null when the state is not remembered. */
    long[] get(Key key) {
        return masks.get(key);
    }

    /** Remembers the rejected row indices of {@code key}; the bit set is kept, not copied. */
    void put(Key key, long[] rejected) {
        masks.put(key, rejected);
    }
}
