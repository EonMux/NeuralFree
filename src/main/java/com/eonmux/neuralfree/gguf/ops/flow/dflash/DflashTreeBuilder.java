package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.engine.DraftTree;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Best-first draft-tree builder over a dflash drafter's per-slot top candidates: SpecPilot's {@code ddtree} tree
 * mode (sglang {@code srt/speculative/specpilot_tree.py}, {@code _tree_build_kernel}, arXiv 2611.02501).
 *
 * <p>A pool entry pairs a tree node with one candidate of the next block row. The entry's score is the parent
 * node's path score plus the candidate's log-softmax over the row's scores at the builder temperature, so a child
 * never outranks its parent (a log-softmax is negative) and a priority queue pops parents before their children:
 * the tree's node list keeps parents ahead of children, deepest-first expansions of the strongest chains first.
 * The block's top candidate of every row extends the strongest path, so a one-branch proposal is the drafter's
 * ordinary selector walk; further branches pick up the secondary candidates the walk would have dropped, best
 * first. Siblings carry distinct tokens: a row's candidates are distinct top-k ids, and a duplicate target id
 * (a non-injective {@code d2t}) keeps only its highest-ranked occurrence.
 *
 * <p>Two budget bounds keep the proposal verifiable in one target forward: at most {@code maxBranches} leaves
 * (every leaf verifies on its own page-shared target sequence) and at most {@code maxRows} layout rows (a leaf's
 * path repeats its ancestors' rows, so a child of an already-branching node costs its whole ancestor chain).
 * Entries whose cost would exceed a bound are skipped; cheaper ones still make the tree.
 */
final class DflashTreeBuilder {

    /** No predecessor candidate: the row is conditioned on the block's anchor token. */
    static final int ANCHOR_PRED = -1;

    /**
     * The candidates of one block: a row per depth, conditioned on the predecessor candidate a tree node chose.
     * Implementations over the DFlash2 and LiLiCorr selector lattices, plain DFlash top-k and the DSpark Markov
     * head provide it; row materialization is cheap (a lattice slice or one bias and top-k over the row).
     */
    interface Table {

        /** First depth holding candidates: 1, or 0 for an anchor-first DSpark draft. */
        int firstDepth();

        /** Deepest depth holding candidates. */
        int lastDepth();

        /**
         * Candidates of {@code depth} after predecessor candidate {@code predIdx} of depth - 1, whose token is
         * {@code predToken} ({@code predIdx} = {@link #ANCHOR_PRED}, {@code predToken} = the anchor, for
         * {@code depth == firstDepth()}). Ids in rank order, best first; larger score is better.
         */
        Row row(int depth, int predIdx, int predToken);
    }

    /** One conditioned candidate row: ids in rank order with their conditional scores. */
    static final class Row {
        final int[] ids;
        final float[] scores;

        Row(int[] ids, float[] scores) {
            if (ids.length != scores.length) {
                throw new IllegalArgumentException("a candidate row needs one score per id, got " + ids.length
                        + " / " + scores.length);
            }
            this.ids = ids;
            this.scores = scores;
        }

        int size() {
            return ids.length;
        }
    }

    /** One pool entry: candidate {@code candIdx} of {@code depth} under tree node {@code parent}. */
    private static final class Entry implements Comparable<Entry> {
        final double score;
        final long order;
        final int parent;
        final int depth;
        final int candIdx;
        final int token;

        Entry(double score, long order, int parent, int depth, int candIdx, int token) {
            this.score = score;
            this.order = order;
            this.parent = parent;
            this.depth = depth;
            this.candIdx = candIdx;
            this.token = token;
        }

        /** Best score first; the earlier insertion wins a tie, so equal-score chains expand in proposal order. */
        @Override
        public int compareTo(Entry o) {
            if (score != o.score) {
                return score > o.score ? -1 : 1;
            }
            return Long.compare(order, o.order);
        }
    }

    private DflashTreeBuilder() {
    }

    /**
     * Builds the tree of at most {@code maxNodes} draft nodes over {@code table}, whose depth-1 rows are
     * conditioned on {@code anchorToken}, the target's pending token. At most {@code maxBranches} leaves and
     * {@code maxRows} layout rows, at the score temperature {@code temp} (SpecPilot's {@code TREE_TEMP}).
     *
     * @return the proposal, empty when no candidate fits the budgets
     * @throws IllegalArgumentException for a non-positive temperature or budgets below one
     */
    static DraftTree build(Table table, int anchorToken, int maxNodes, int maxBranches, int maxRows, float temp) {
        Objects.requireNonNull(table, "table");
        if (!(temp > 0.0) || !Float.isFinite(temp)) {
            throw new IllegalArgumentException("tree temperature must be finite and positive, got " + temp);
        }
        if (maxNodes < 1 || maxBranches < 1 || maxRows < 1) {
            return new DraftTree(new ArrayList<>());
        }
        final int first = table.firstDepth();
        final int last = table.lastDepth();
        if (anchorToken < 0 || last < first) {
            return new DraftTree(new ArrayList<>());
        }
        int[] tokens = new int[maxNodes];
        int[] parents = new int[maxNodes];
        int[] depths = new int[maxNodes];
        boolean[] leaf = new boolean[maxNodes];
        int n = 0;
        int leaves = 0;
        int rows = 0;
        long order = 0;
        PriorityQueue<Entry> pool = new PriorityQueue<>();
        order += pushChildren(pool, table, ANCHOR_PRED, anchorToken, first, -1, 0.0, order, temp);
        while (n < maxNodes && !pool.isEmpty()) {
            Entry e = pool.poll();
            boolean parentLeaf = e.parent >= 0 && leaf[e.parent];
            int cost;
            if (e.parent < 0 || !parentLeaf) {
                // a fresh layout path: the candidate alone, or the ancestor chain it repeats beside its parent
                cost = e.parent < 0 ? 1 : e.depth + 1;
                if (rows + cost > maxRows || leaves + 1 > maxBranches) {
                    continue;
                }
                leaves++;
            } else {
                cost = 1;
                if (rows + cost > maxRows) {
                    continue;
                }
                leaf[e.parent] = false;
            }
            rows += cost;
            tokens[n] = e.token;
            parents[n] = e.parent;
            depths[n] = e.depth;
            leaf[n] = true;
            n++;
            if (e.depth < last) {
                order += pushChildren(pool, table, e.candIdx, e.token, e.depth + 1, n - 1, e.score, order, temp);
            }
        }
        List<DraftTree.Node> nodes = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            nodes.add(new DraftTree.Node(tokens[i], parents[i], depths[i] - first + 1, 0));
        }
        return new DraftTree(nodes);
    }

    /**
     * Pushes the entries for the children of one expansion: the candidates of {@code depth} conditioned on
     * predecessor candidate {@code predIdx} / token {@code predToken}, each scored at {@code parentScore} plus
     * its log-softmax over the row. The first child of a node keeps the row's own ranking (largest log-softmax,
     * which is the row's best candidate at score 0 relative to the row's best). Returns the entries pushed.
     */
    private static int pushChildren(PriorityQueue<Entry> pool, Table table, int predIdx, int predToken, int depth,
                                    int parent, double parentScore, long order, float temp) {
        Row row = table.row(depth, predIdx, predToken);
        int k = row.size();
        if (k == 0) {
            return 0;
        }
        // log-softmax over the row's scores at the temperature: (s_j - max) / T - log sum_k exp((s_k - max) / T)
        float max = row.scores[0];
        for (int j = 1; j < k; j++) {
            max = Math.max(max, row.scores[j]);
        }
        double sum = 0.0;
        for (int j = 0; j < k; j++) {
            sum += Math.exp((row.scores[j] - max) / temp);
        }
        double logSum = Math.log(sum);
        Set<Integer> seen = new HashSet<>();
        int pushed = 0;
        for (int j = 0; j < k; j++) {
            // siblings carry distinct tokens: a row keeps its best-ranked occurrence of every token
            if (!seen.add(row.ids[j])) {
                continue;
            }
            double score = parentScore + ((row.scores[j] - max) / temp - logSum);
            pool.add(new Entry(score, order + pushed, parent, depth, j, row.ids[j]));
            pushed++;
        }
        return pushed;
    }

    /**
     * Plain DFlash candidates: the fixed top-k of every block row's target logits, scored by the logits
     * themselves (dflash.cpp reads the walk's token per row greedily; the tree scores the same rows
     * condition-free). Immutable.
     */
    static final class TopKTable implements Table {
        private final int firstDepth;
        private final int k;
        private final int[] ids;
        private final float[] scores;

        /**
         * @param ids    {@code [rows][k]} candidate ids of every scored row, rank order
         * @param scores their logits
         */
        TopKTable(int firstDepth, int rows, int k, int[] ids, float[] scores) {
            if (rows < 1 || k < 1 || ids.length < rows * k || scores.length < rows * k) {
                throw new IllegalArgumentException("a top-k table of " + rows + " rows x " + k + " needs "
                        + (rows * k) + " ids and scores, got " + ids.length + " / " + scores.length);
            }
            this.firstDepth = firstDepth;
            this.k = k;
            this.ids = ids.clone();
            this.scores = scores.clone();
        }

        @Override
        public int firstDepth() {
            return firstDepth;
        }

        @Override
        public int lastDepth() {
            return firstDepth + ids.length / k - 1;
        }

        @Override
        public Row row(int depth, int predIdx, int predToken) {
            int at = (depth - firstDepth) * k;
            return new Row(java.util.Arrays.copyOfRange(ids, at, at + k),
                    java.util.Arrays.copyOfRange(scores, at, at + k));
        }
    }

    /**
     * DSpark candidates: the top-k of every block row's logits after the Markov bias of the conditioning token
     * ({@code DflashMarkovHead.addBias}), computed per expansion, so a branch conditions on its own chain.
     */
    static final class DsparkTable implements Table {
        private final DflashMarkovHead markov;
        private final int firstDepth;
        private final int lastDepth;
        private final int nVocab;
        private final int k;
        private final float[] logits;

        DsparkTable(DflashMarkovHead markov, float[] logits, int nVocab, int firstDepth, int lastDepth, int k) {
            this.markov = Objects.requireNonNull(markov, "markov");
            this.logits = Objects.requireNonNull(logits, "logits");
            this.nVocab = nVocab;
            this.firstDepth = firstDepth;
            this.lastDepth = lastDepth;
            this.k = k;
        }

        @Override
        public int firstDepth() {
            return firstDepth;
        }

        @Override
        public int lastDepth() {
            return lastDepth;
        }

        @Override
        public Row row(int depth, int predIdx, int predToken) {
            int width = Math.min(k, nVocab);
            int[] ids = new int[width];
            float[] scores = new float[width];
            float[] biased = new float[nVocab];
            int off = depth * nVocab;
            System.arraycopy(logits, off, biased, 0, nVocab);
            markov.addBias(predToken, biased, 0);
            CpuOps.topK(biased, 0, nVocab, width, ids, 0);
            for (int j = 0; j < width; j++) {
                scores[j] = biased[ids[j]];
            }
            return new Row(ids, scores);
        }
    }

    /**
     * Selector-lattice candidates (DFlash2 and LiLiCorr): the row's fixed top-k ids with their conditional
     * scores {@code score(p, j, predIdx)}; the anchor row of a lattice keeps only predecessor 0.
     */
    static final class LatticeTable implements Table {
        private final DflashSelector.Lattice lattice;
        private final int firstDepth;
        private final int lastDepth;

        LatticeTable(DflashSelector.Lattice lattice, int firstDepth, int lastDepth) {
            this.lattice = Objects.requireNonNull(lattice, "lattice");
            this.firstDepth = firstDepth;
            this.lastDepth = lastDepth;
        }

        @Override
        public int firstDepth() {
            return firstDepth;
        }

        @Override
        public int lastDepth() {
            return lastDepth;
        }

        @Override
        public Row row(int depth, int predIdx, int predToken) {
            int k = lattice.topK();
            int[] ids = new int[k];
            float[] scores = new float[k];
            int at = lattice.scoreRow(depth, Math.max(0, predIdx));
            System.arraycopy(lattice.scores(), at, scores, 0, k);
            for (int j = 0; j < k; j++) {
                ids[j] = lattice.candidate(depth, j);
            }
            return new Row(ids, scores);
        }
    }
}
