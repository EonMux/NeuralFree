package com.eonmux.neuralfree.gguf.tokenizer;

import java.util.Arrays;
import java.util.Comparator;

/**
 * llama.cpp's {@code naive_trie} over byte strings, laid out in arrays: every node's children sit in one sorted run
 * of {@link #labels} / {@link #targets}, the root's are indexed directly by byte. A key inserted twice keeps the
 * value of its later insertion, as {@code naive_trie::insert} overwrites it. Walks step one byte at a time
 * ({@link #child}) and stop where the trie has no path, so a matcher reads only the prefixes some key has.
 * Immutable and thread-safe.
 */
final class ByteTrie {

    static final int ROOT = 0;
    /** {@link #child} when the trie has no such path, {@link #value} when no key ends at the node. */
    static final int NONE = -1;

    private final int[] rootChildren = new int[256];
    private final int[] childStart;
    private final int[] childCount;
    private final byte[] labels;
    private final int[] targets;
    private final int[] values;

    /**
     * A trie of {@code keys}, key {@code i} carrying {@code values[i]}; a later equal key replaces an earlier one's
     * value. Values must be non-negative.
     */
    ByteTrie(byte[][] keys, int[] values) {
        if (keys.length != values.length) {
            throw new IllegalArgumentException(keys.length + " keys but " + values.length + " values");
        }
        Integer[] order = new Integer[keys.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Comparator<Integer> byKey = (a, b) -> Arrays.compareUnsigned(keys[a], keys[b]);
        Arrays.sort(order, byKey); // stable: equal keys keep insertion order, so the later one is applied last
        int maxNodes = 1;
        for (byte[] key : keys) {
            maxNodes += key.length;
        }
        int[] parentOf = new int[maxNodes];
        byte[] labelOf = new byte[maxNodes];
        int[] valueOf = new int[maxNodes];
        Arrays.fill(valueOf, NONE);
        int nodes = 1;
        byte[] previous = new byte[0];
        int[] path = new int[1];
        for (int k : order) {
            byte[] key = keys[k];
            if (values[k] < 0) {
                throw new IllegalArgumentException("negative value " + values[k]);
            }
            int shared = 0;
            int limit = Math.min(previous.length, key.length);
            while (shared < limit && previous[shared] == key[shared]) {
                shared++;
            }
            if (path.length < key.length + 1) {
                path = Arrays.copyOf(path, Math.max(key.length + 1, path.length * 2));
            }
            for (int d = shared; d < key.length; d++) {
                int node = nodes++;
                parentOf[node] = path[d];
                labelOf[node] = key[d];
                path[d + 1] = node;
            }
            valueOf[path[key.length]] = values[k];
            previous = key;
        }
        // nodes were numbered in key order, so each node's children are numbered in ascending label order
        this.childStart = new int[nodes];
        this.childCount = new int[nodes];
        for (int n = 1; n < nodes; n++) {
            childCount[parentOf[n]]++;
        }
        int at = 0;
        for (int n = 0; n < nodes; n++) {
            childStart[n] = at;
            at += childCount[n];
        }
        this.labels = new byte[Math.max(0, nodes - 1)];
        this.targets = new int[Math.max(0, nodes - 1)];
        int[] filled = new int[nodes];
        Arrays.fill(rootChildren, NONE);
        for (int n = 1; n < nodes; n++) {
            int p = parentOf[n];
            int slot = childStart[p] + filled[p]++;
            labels[slot] = labelOf[n];
            targets[slot] = n;
            if (p == ROOT) {
                rootChildren[labelOf[n] & 0xFF] = n;
            }
        }
        this.values = Arrays.copyOf(valueOf, nodes);
    }

    /** The node one byte below {@code node}, or {@link #NONE}. */
    int child(int node, byte b) {
        if (node == ROOT) {
            return rootChildren[b & 0xFF];
        }
        int lo = childStart[node];
        int hi = lo + childCount[node] - 1;
        int key = b & 0xFF;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            int label = labels[mid] & 0xFF;
            if (label < key) {
                lo = mid + 1;
            } else if (label > key) {
                hi = mid - 1;
            } else {
                return targets[mid];
            }
        }
        return NONE;
    }

    /** The value of the key ending at {@code node}, or {@link #NONE}. */
    int value(int node) {
        return values[node];
    }

    /** {@code naive_trie::get_longest_prefix}: how far the trie path follows {@code s} from {@code off}. */
    int pathDepth(byte[] s, int off) {
        int node = ROOT;
        int depth = 0;
        while (off + depth < s.length) {
            node = child(node, s[off + depth]);
            if (node == NONE) {
                break;
            }
            depth++;
        }
        return depth;
    }
}
