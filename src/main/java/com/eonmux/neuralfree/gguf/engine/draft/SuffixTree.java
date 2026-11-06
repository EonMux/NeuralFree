package com.eonmux.neuralfree.gguf.engine.draft;

import java.util.ArrayList;
import java.util.Arrays;
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
 * extended more than once with its appended tokens, which must arrive in order. {@link #remove} takes a sequence's
 * substrings back out of the tree (Arctic Inference's {@code SuffixTree::remove}, how its global cache evicts a
 * request), {@link #dropSequence} only forgets its extension state, and {@link #clear()} frees the whole tree.
 * Sequence ids are caller-chosen ints ({@code >= 0}).
 *
 * <p>{@link #speculate} follows the references' {@code speculate} with tree speculation off: for every suffix length
 * of the context, shortest first, the context's suffix of that length is matched from the root (the tree's paths are
 * substrings, so it matches where a stored sequence contains it) until one is not found; each matched suffix drafts the
 * path of its most frequent children, each token carrying the running product of {@code child.count / node.count}
 * and the path stopping once that product falls below {@code minTokenProb}, bounded by the maximum speculation tokens
 * and by {@code maxSpecFactor * matchedLength + maxSpecOffset}. The draft with the highest score (the sum of its token
 * probabilities) wins, a tie going to the longer match.
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

        /**
         * The estimated probability of each drafted token, the running product of the path's child frequencies
         * (parallel to {@link #tokenIds()}).
         */
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

    /**
     * The suffix positions of one stored sequence, so its next tokens extend, as llama.cpp's suffix states, and the
     * sequence's tokens, so {@link #remove} can take its substrings back out.
     */
    private static final class SuffixState {
        Node[] nodes = new Node[0];
        int[] tokens = new int[16];
        int length;

        void append(int token) {
            if (length == tokens.length) {
                tokens = Arrays.copyOf(tokens, tokens.length * 2);
            }
            tokens[length] = token;
        }
    }

    /** Absorbs the rounding of {@code matchedLength * maxSpecFactor + maxSpecOffset} before its floor. */
    private static final double LENGTH_EPSILON = 1e-6;

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

    /** Number of sequences the tree tracks (neither {@linkplain #remove removed} nor {@linkplain #dropSequence dropped}). */
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
            state.append(t);
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

    /**
     * Forgets one sequence's extension state and leaves its substrings in the tree: extension of the id starts a
     * fresh sequence.
     */
    public void dropSequence(int sequenceId) {
        sequences.remove(sequenceId);
    }

    /**
     * Takes sequence {@code sequenceId} out of the tree: every substring it entered loses that occurrence, and a node
     * left without occurrences is pruned with its subtree. Extension of the id afterwards starts a fresh sequence.
     *
     * @return whether the tree tracked the sequence
     */
    public boolean remove(int sequenceId) {
        SuffixState state = sequences.remove(sequenceId);
        if (state == null) {
            return false;
        }
        // extend entered, for each start s, the substrings tokens[s .. s + d) with d up to maxDepth
        for (int s = 0; s < state.length; s++) {
            Node node = root;
            int depth = Math.min(maxDepth, state.length - s);
            for (int d = 0; d < depth; d++) {
                int token = state.tokens[s + d];
                Node child = node.children.get(token);
                if (child == null) {
                    throw new IllegalStateException("suffix tree lost the substring of sequence " + sequenceId
                            + " at " + s + " (+" + d + ")");
                }
                if (--child.count == 0) {
                    // every deeper substring of this path occurs no more often than its prefix: all gone too
                    node.children.remove(token);
                    break;
                }
                node = child;
            }
        }
        return true;
    }

    /**
     * Drafts after the given context: every suffix of {@code pattern} the tree contains is matched, and the best of
     * their drafts is returned (the empty draft, with a match length of 0, when not even the last token is known).
     *
     * @param pattern        the context to match (a suffix of the token history, typically its last
     *                       {@link #maxDepth()} tokens)
     * @param maxSpecTokens  draft length cap ({@code max_spec_tokens}; 0 or negative keeps nothing)
     * @param maxSpecFactor  draft length factor against the matched length ({@code max_spec_factor})
     * @param maxSpecOffset  draft length offset added to the factor ({@code max_spec_offset})
     * @param minTokenProb   smallest accepted cumulative token probability ({@code min_token_prob})
     */
    public Draft speculate(int[] pattern, int maxSpecTokens, double maxSpecFactor, double maxSpecOffset,
                           double minTokenProb) {
        Objects.requireNonNull(pattern, "pattern");
        Draft best = new Draft(new int[0], new double[0], 0.0, 0);
        for (int matchLength = 1; matchLength <= pattern.length; matchLength++) {
            Node node = match(pattern, pattern.length - matchLength);
            if (node == null) {
                break;
            }
            double budget = Math.floor(matchLength * maxSpecFactor + maxSpecOffset + LENGTH_EPSILON);
            int maxTokens = (int) Math.max(0.0, Math.min(maxSpecTokens, budget));
            Draft draft = speculatePath(node, matchLength, maxTokens, minTokenProb);
            if (draft.score >= best.score) {
                best = draft;
            }
        }
        return best;
    }

    /** The node of {@code pattern[from ..]}, or null when the tree does not contain it. */
    private Node match(int[] pattern, int from) {
        Node node = root;
        for (int i = from; i < pattern.length; i++) {
            node = node.children.get(pattern[i]);
            if (node == null) {
                return null;
            }
        }
        return node;
    }

    /** The path of most frequent children below {@code node}, as long as its running probability allows. */
    private static Draft speculatePath(Node node, int matchLength, int maxTokens, double minTokenProb) {
        List<Integer> tokens = new ArrayList<>();
        List<Double> probs = new ArrayList<>();
        double score = 0.0;
        double prob = 1.0;
        while (tokens.size() < maxTokens) {
            int bestToken = -1;
            Node best = null;
            for (Map.Entry<Integer, Node> e : node.children.entrySet()) {
                Node child = e.getValue();
                if (best == null || child.count > best.count
                        || (child.count == best.count && e.getKey() < bestToken)) {
                    best = child;
                    bestToken = e.getKey();
                }
            }
            if (best == null) {
                break;
            }
            prob *= (double) best.count / node.count;
            if (prob < minTokenProb) {
                break;
            }
            tokens.add(bestToken);
            probs.add(prob);
            score += prob;
            node = best;
        }
        int[] t = new int[tokens.size()];
        double[] p = new double[probs.size()];
        for (int i = 0; i < t.length; i++) {
            t[i] = tokens.get(i);
            p[i] = probs.get(i);
        }
        return new Draft(t, p, score, matchLength);
    }

    @Override
    public String toString() {
        return "SuffixTree{maxDepth=" + maxDepth + ", sequences=" + sequences.size() + '}';
    }
}
