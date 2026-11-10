package com.eonmux.neuralfree.gguf.tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;

/**
 * An Aho-Corasick automaton over a list of special-token texts that tells, in one pass over a text, which of them
 * occur in it. The partition ({@link SpecialTokens#partition}) runs llama.cpp's longest-first sequential split over
 * the tokens that occur in its input only: every fragment the split produces is a substring of that input, so a text
 * that does not occur there can never match, and skipping it leaves the split unchanged. Immutable and thread-safe.
 */
final class SpecialTextIndex {

    private final int patterns;
    /** Per node: where its children start in {@link #labels} / {@link #targets}, and how many. */
    private final int[] childStart;
    private final int[] childCount;
    private final char[] labels;
    private final int[] targets;
    private final int[] fail;
    /** Nodes in breadth-first order: every node's failure target comes before it. */
    private final int[] order;
    /** Per node: the first pattern ending there, chained through {@link #nextOutput}; -1 for none. */
    private final int[] output;
    private final int[] nextOutput;

    /**
     * An index over {@code texts}; a null or empty entry is no pattern and never reported. Pattern {@code k} is
     * reported at index {@code k} of {@link #occurring}.
     */
    SpecialTextIndex(String[] texts) {
        this.patterns = texts.length;
        List<TreeMap<Character, Integer>> children = new ArrayList<>();
        children.add(new TreeMap<>());
        List<Integer> ends = new ArrayList<>();
        ends.add(-1);
        int[] next = new int[texts.length];
        Arrays.fill(next, -1);
        for (int k = 0; k < texts.length; k++) {
            String text = texts[k];
            if (text == null || text.isEmpty()) {
                continue;
            }
            int node = 0;
            for (int i = 0; i < text.length(); i++) {
                Integer child = children.get(node).get(text.charAt(i));
                if (child == null) {
                    child = children.size();
                    children.get(node).put(text.charAt(i), child);
                    children.add(new TreeMap<>());
                    ends.add(-1);
                }
                node = child;
            }
            next[k] = ends.get(node);
            ends.set(node, k);
        }
        int nodes = children.size();
        this.childStart = new int[nodes];
        this.childCount = new int[nodes];
        this.labels = new char[nodes - 1];
        this.targets = new int[nodes - 1];
        this.output = new int[nodes];
        this.nextOutput = next;
        int at = 0;
        for (int n = 0; n < nodes; n++) {
            childStart[n] = at;
            childCount[n] = children.get(n).size();
            for (var e : children.get(n).entrySet()) {
                labels[at] = e.getKey();
                targets[at] = e.getValue();
                at++;
            }
            output[n] = ends.get(n);
        }
        this.fail = new int[nodes];
        this.order = new int[nodes];
        int head = 0;
        int tail = 0;
        order[tail++] = 0;
        while (head < tail) {
            int n = order[head++];
            for (int c = childStart[n]; c < childStart[n] + childCount[n]; c++) {
                int child = targets[c];
                fail[child] = n == 0 ? 0 : step(fail[n], labels[c]);
                order[tail++] = child;
            }
        }
    }

    /** The goto-with-failure transition from {@code node} on {@code ch}. */
    private int step(int node, char ch) {
        while (true) {
            int child = child(node, ch);
            if (child >= 0) {
                return child;
            }
            if (node == 0) {
                return 0;
            }
            node = fail[node];
        }
    }

    private int child(int node, char ch) {
        int lo = childStart[node];
        int hi = lo + childCount[node] - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            char label = labels[mid];
            if (label < ch) {
                lo = mid + 1;
            } else if (label > ch) {
                hi = mid - 1;
            } else {
                return targets[mid];
            }
        }
        return -1;
    }

    /** Which patterns occur in at least one of the text fragments of {@code fragments}, by pattern index. */
    boolean[] occurring(List<SpecialTokens.Fragment> fragments) {
        boolean[] reached = new boolean[fail.length];
        for (SpecialTokens.Fragment f : fragments) {
            if (f.isToken()) {
                continue;
            }
            String text = f.text();
            int node = 0;
            for (int i = 0; i < text.length(); i++) {
                node = step(node, text.charAt(i));
                reached[node] = true;
            }
        }
        boolean[] found = new boolean[patterns];
        for (int i = order.length - 1; i > 0; i--) {
            int n = order[i];
            if (!reached[n]) {
                continue;
            }
            reached[fail[n]] = true;
            for (int k = output[n]; k >= 0; k = nextOutput[k]) {
                found[k] = true;
            }
        }
        return found;
    }
}
