package com.eonmux.neuralfree.gguf.engine.draft;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A suffix tree of token sequences for suffix decoding (Oliaro et al., arXiv 2411.04975), the storage of SGLang's
 * and vLLM's suffix decoding proposer ({@code suffix_decoding.py}, Arctic Inference's {@code SuffixTree}): a trie
 * whose every path from the root is a substring of a stored sequence, so each node holds one substring and the count
 * of its occurrences, and the children of the node matching the context hold the empirical distribution of the token
 * that follows it.
 *
 * <p>Sequences are added incrementally ({@link #extend}): at each token the suffixes of length 1..{@link #maxDepth()}
 * that end at it are entered (the depth cap keeps each insert at {@code maxDepth} traversals). A sequence id can be
 * extended more than once with its appended tokens, which must arrive in order; {@link #dropSequence} forgets one, and
 * {@link #clear()} frees the whole tree. Sequence ids are caller-chosen ints ({@code >= 0}).
 *
 * <p>{@link #speculate} matches the longest prefix of the given context that the tree contains — the tree's paths are
 * substrings, so any suffix of the context matches where a stored sequence contains it — then drafts the chain of its
 * most probable children (the chain speculation of the references' {@code speculate}, tree speculation off), scoring
 * each token by {@code child.count / parent continuation count}, bounded by the maximum speculation tokens and by
 * {@code maxSpecFactor * matchedLength + maxSpecOffset}.
 */
public final class SuffixTree {

    /** One result: the drafted chain, its per-token probabilities, the sum and the matched context length. */
    public static final class Draft {
        private final int[] tokens;
        private final double[] probabilities;
        private final double score;
        private final int matchLength;

        Draft(int[] tokens, double[] probabilities, double score, int matchLength) {
            this.tokens = tokens;
            this.probabilities = probabilities;
            this.score = score;
            this.matchLength = matchLength;
        }

        /** The proposed token ids, in proposal order (empty when nothing was matched). */
        public int[] tokenIds() { return tokens; }

        /** The estimated probability of each drafted token (parallel to {@link #tokenIds()}). */
        public double[] probabilities() { return probabilities; }

        /** The sum of the drafted token probabilities. */
        public double score() { return score; }

        /** How many context tokens were matched in the tree. */
        public int matchLength() { return matchLength; }

        @Override
        public String toString() {
            return "SuffixTree.Draft{" + tokens.length + " tokens, score=" + score + ", matched=" + matchLength + '}';
        }
    }

    private static final class Node {
        final HashMap<Integer, Node> children = new HashMap<>();
        int count;
    }

    /** The suffix positions of one stored sequence, so its next tokens extend, as llama.cpp's suffix states. */
    private static final class SuffixState {
        Node[] nodes = new Node[0];
        int length;
    }

    private final Node root = new Node();
    private final int maxDepth;
    private final Map<Integer, SuffixState> sequences = new HashMap<>();

    /** A tree entering suffixes up to {@code maxDepth} tokens long.
     *
     * @throws IllegalArgumentException for a depth below 1 */
    public SuffixTree(int maxDepth) {
        if (maxDepth < 1) {
            throw new IllegalArgumentException("the suffix tree depth must be at least 1, got " + maxDepth);
        }
        this.maxDepth = maxDepth;
    }

    /** The entered suffix depth cap. */
    public int maxDepth() {
        return maxDepth;
    }

    /** Number of stored sequences (still tracked: removed states age out only via {@link #clear()}). */
    public int sequenceCount() {
        return sequences.size();
    }

    /**
     * Appends {@code tokens} to sequence {@code sequenceId}, creating it when new. A sequence's tokens must arrive in
     * order; the suffixes that end at each of them extend the tree.
     *
     * @throws IllegalArgumentException for a negative sequence id, null tokens or a negative token id
     */
    public void extend(int sequenceId, int[] tokens) {
        Objects.requireNonNull(tokens, "tokens");
        if (sequenceId < 0) {
            throw new IllegalArgumentException("sequence id must not be negative, got " + sequenceId);
        }
        for (int t : tokens) {
            if (t < 0) {
                throw new IllegalArgumentException("a suffix tree sequence holds negative token id " + t);
            }
        }
        SuffixState state = sequences.computeIfAbsent(sequenceId, id -> new SuffixState());
        for (int t : tokens) {
            Node[] next = new Node[Math.min(state.length + 1, maxDepth)];
            Node first = childOf(root, t);
            next[0] = first;
            int kept = Math.min(state.length, maxDepth - 1);
            for (int d = 1; d <= kept; d++) {
                next[d] = childOf(state.nodes[d - 1], t);
            }
            state.nodes = next;
            state.length++;
        }
    }

    /** The {@code token}-child of {@code node}, creating it and counting it. */
    private static Node childOf(Node node, int token) {
        Node child = node.children.get(token);
        if (child == null) {
            child = new Node();
            node.children.put(token, child);
        }
        child.count++;
        return child;
    }

    /**
     * Drops the whole tree and every sequence state, as a proposer does when its cached request is removed.
     */
    public void clear() {
        root.children.clear();
        root.count = 0;
        sequences.clear();
    }

    /** Removes one sequence's suffix state: extension of an evicted sequence's id starts fresh. */
    public void dropSequence(int sequenceId) {
        sequences.remove(sequenceId);
    }

    /**
     * Drafts after the given context: the longest prefix of {@code pattern} that the tree contains is matched (the
     * pattern is the caller's choice of suffix of the generation; a full pattern is the conventional
     * {@code min(maxDepth, tokens)} tail), then its best child chain is drafted.
     *
     * @param pattern        the context to match (a suffix of the token history)
     * @param maxSpecTokens  draft length cap ({@code max_spec_tokens}; 0 or negative keeps nothing)
     * @param maxSpecFactor  draft length factor against the matched length ({@code max_spec_factor})
     * @param maxSpecOffset  draft length offset added to the factor ({@code max_spec_offset})
     * @param minTokenProb   smallest accepted child probability ({@code min_token_prob})
     */
    public Draft speculate(int[] pattern, int maxSpecTokens, double maxSpecFactor, double maxSpecOffset,
                           double minTokenProb) {
        Objects.requireNonNull(pattern, "pattern");
        Node node = root;
        int matched = 0;
        for (int token : pattern) {
            Node child = node.children.get(token);
            if (child == null) {
                break;
            }
            node = child;
            matched++;
        }
        if (matched == 0 || maxSpecTokens <= 0) {
            return new Draft(new int[0], new double[0], 0.0, matched);
        }
        return speculateFrom(node, matched, maxSpecTokens, maxSpecFactor, maxSpecOffset, minTokenProb);
    }

    private static Draft speculateFrom(Node node, int matched, int maxSpecTokens, double maxSpecFactor,
                                       double maxSpecOffset, double minTokenProb) {
        List<Integer> tokens = new ArrayList<>();
        List<Double> probs = new ArrayList<>();
        double score = 0.0;
        while (!node.children.isEmpty()) {
            if (tokens.size() >= maxSpecTokens
                    || tokens.size() >= maxSpecFactor * matched + maxSpecOffset) {
                break;
            }
            Node best = null;
            for (Node child : node.children.values()) {
                if (best == null || child.count > best.count) {
                    best = child;
                }
            }
            long continuation = 0;
            for (Node child : node.children.values()) {
                continuation += child.count;
            }
            double prob = (double) best.count / continuation;
            if (prob < minTokenProb) {
                break;
            }
            tokens.add(bestChildToken(node, best));
            probs.add(prob);
            score += prob;
            node = best;
            if (tokens.size() >= Integer.MAX_VALUE / 2) {
                break;
            }
        }
        int[] t = new int[tokens.size()];
        double[] p = new double[probs.size()];
        for (int i = 0; i < t.length; i++) {
            t[i] = tokens.get(i);
            p[i] = probs.get(i);
        }
        return new Draft(t, p, score, matched);
    }

    /** The token id whose child node is {@code child} (counts are distinct enough to match it again). */
    private static int bestChildToken(Node node, Node child) {
        for (Map.Entry<Integer, Node> e : node.children.entrySet()) {
            if (e.getValue() == child) {
                return e.getKey();
            }
        }
        throw new IllegalStateException("a child node disappeared from its parent");
    }

    @Override
    public String toString() {
        return "SuffixTree{maxDepth=" + maxDepth + ", sequences=" + sequences.size() + '}';
    }
}
