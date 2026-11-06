package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.DraftTree;

import java.util.Objects;

/**
 * One sequence's tree proposal of a decode round: the draft tree and, per node, the cumulative path log-probability
 * its builder scored it at (node 0, the root, at 0.0). The scores drive the verify budget
 * ({@link TreeBudget}): they descend non-increasingly along the node list — a child scores below its parent by its
 * log-softmax — so the first {@code n} nodes of the proposal are the {@code n}-node proposal (a valid tree, the
 * budget's cut being a node prefix, SpecPilot's per-request budget on the best-first DDTree, arXiv 2611.02501).
 * Immutable.
 */
final class TreeProposal {

    /** The proposal's nodes, parents ahead of children. */
    final DraftTree tree;
    /** {@code tree.size()} path scores, node 0's first. */
    final double[] pathScores;

    /**
     * @throws IllegalArgumentException when the scores do not match the tree's node count, a score is not finite,
     *                                  or a node after the first scores above its predecessor
     */
    TreeProposal(DraftTree tree, double[] pathScores) {
        this.tree = Objects.requireNonNull(tree, "tree");
        Objects.requireNonNull(pathScores, "pathScores");
        if (pathScores.length != tree.size()) {
            throw new IllegalArgumentException("a tree proposal carries one path score per node, got "
                    + pathScores.length + " scores for " + tree.size() + " nodes");
        }
        for (int i = 0; i < pathScores.length; i++) {
            if (!Double.isFinite(pathScores[i])) {
                throw new IllegalArgumentException("tree proposal node " + i + " carries a non-finite path score");
            }
            if (i > 0 && pathScores[i] > pathScores[i - 1]) {
                throw new IllegalArgumentException("tree proposal node " + i + " scores above its predecessor:"
                        + " path scores descend along the node list");
            }
        }
        this.pathScores = pathScores.clone();
    }

    /** The proposal cut to its first {@code n} nodes (a node prefix, itself a valid tree). */
    TreeProposal cut(int n) {
        if (n >= tree.size()) {
            return this;
        }
        if (n <= 0) {
            return new TreeProposal(new DraftTree(new java.util.ArrayList<>()), new double[0]);
        }
        return new TreeProposal(new DraftTree(new java.util.ArrayList<>(tree.nodes().subList(0, n))),
                java.util.Arrays.copyOf(pathScores, n));
    }

    /** The proposal's nodes, parents ahead of children. */
    DraftTree tree() {
        return tree;
    }

    /** {@code tree.size()} path scores, node 0's first. */
    double[] pathScores() {
        return pathScores;
    }
}
