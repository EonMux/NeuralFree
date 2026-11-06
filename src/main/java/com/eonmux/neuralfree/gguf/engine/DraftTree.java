package com.eonmux.neuralfree.gguf.engine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The proposal tree of one multi-branch speculative round: a flat list of nodes, each a drafted token with its
 * parent node ({@code -1} for a node whose parent is the round's pending token), its depth (1 for the first drafted
 * position) and the draft branch the draft model evaluated it on. Immutable.
 *
 * <p>Node order is the draft model's evaluation order: depth by depth, and inside a depth the branches a parent
 * split into before the parent's own most likely token. A <b>linear</b> proposal has exactly one node at every
 * depth and one branch; the decoder verifies it as an ordinary linear draft. A branching proposal is verified one
 * target forward per depth, each node on its own page-shared target sequence ({@code SpeculativeDecoder}'s tree
 * round).
 */
public final class DraftTree {

    /** One drafted token. Immutable. */
    public static final class Node {
        private final int token;
        private final int parent;
        private final int depth;
        private final int branch;

        /**
         * @param parent the parent node's index, or -1 for a first-depth node
         */
        public Node(int token, int parent, int depth, int branch) {
            if (token < 0) {
                throw new IllegalArgumentException("draft node token " + token);
            }
            if (parent < -1) {
                throw new IllegalArgumentException("draft node at depth " + depth + " has parent " + parent);
            }
            if (depth < 1 || branch < 0) {
                throw new IllegalArgumentException("draft node depth " + depth + " branch " + branch);
            }
            this.token = token;
            this.parent = parent;
            this.depth = depth;
            this.branch = branch;
        }

        /** The drafted token. */
        public int token() {
            return token;
        }

        /** The parent node's index in {@link DraftTree#nodes()}, or -1 for a first-depth node. */
        public int parent() {
            return parent;
        }

        /** The drafted position, 1-based. */
        public int depth() {
            return depth;
        }

        /** The draft branch the model evaluated this token on. */
        public int branch() {
            return branch;
        }

        @Override
        public String toString() {
            return "Node{" + token + " d" + depth + " p" + parent + " b" + branch + '}';
        }
    }

    private final List<Node> nodes;
    private final int depth;
    private final boolean linear;

    /**
     * @param nodes the proposal nodes in draft evaluation order
     * @throws IllegalArgumentException when the nodes disagree about their depths
     */
    public DraftTree(List<Node> nodes) {
        Objects.requireNonNull(nodes, "nodes");
        this.nodes = Collections.unmodifiableList(new ArrayList<>(nodes));
        int deepest = 0;
        int[] perDepth = new int[nodes.size() + 1];
        for (int i = 0; i < this.nodes.size(); i++) {
            Node n = this.nodes.get(i);
            if (n.parent() >= i) {
                throw new IllegalArgumentException("draft node " + i + " has parent " + n.parent()
                        + " that does not precede it");
            }
            if (n.parent() == -1 ? n.depth() != 1 : this.nodes.get(n.parent()).depth() != n.depth() - 1) {
                throw new IllegalArgumentException("draft node " + i + " at depth " + n.depth()
                        + " has parent " + n.parent());
            }
            deepest = Math.max(deepest, n.depth());
            perDepth[n.depth()]++;
        }
        this.depth = deepest;
        boolean chain = true;
        for (int d = 1; d <= deepest; d++) {
            chain &= perDepth[d] == 1;
        }
        this.linear = chain;
    }

    /** The proposal nodes in draft evaluation order. */
    public List<Node> nodes() {
        return nodes;
    }

    /** The deepest drafted position, 0 for an empty proposal. */
    public int depth() {
        return depth;
    }

    /** Number of drafted tokens. */
    public int size() {
        return nodes.size();
    }

    /** Whether the proposal is one chain: one node at every depth. */
    public boolean linear() {
        return linear && !nodes.isEmpty();
    }

    /**
     * The tokens of the linear proposal, depth by depth; empty for a branching proposal.
     */
    public int[] linearTokens() {
        if (!linear()) {
            return new int[0];
        }
        int[] out = new int[depth];
        for (Node n : nodes) {
            out[n.depth() - 1] = n.token();
        }
        return out;
    }

    @Override
    public String toString() {
        return "DraftTree{nodes=" + nodes.size() + ", depth=" + depth + (linear ? ", linear" : ", branching") + '}';
    }
}
