package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The batch engine's verify budget over one step's tree proposals (SpecPilot's ragged verify budget and per-request
 * node budget on the best-first DDTree, arXiv 2611.02501, {@code SPECPILOT_DB_RV} / {@code SPECPILOT_DB_RP}).
 *
 * <p><b>The tiered total.</b> {@code neuralfree.draft.tree.rows} bounds the tree rows one verify step lays out over
 * all its sequences: a plain integer is the constant total, a tier table {@code bs:T;...} or {@code bs:xF;...} gives
 * the total for every batch size up to {@code bs} ({@code xF} scales {@code ceil(bs * F)}, the reference's tokens
 * tier), the first tier that covers the batch size winning. The total clamps into {@code [bs, bs * width + pad]} —
 * every sequence verifies at least its pending row, and no tier can ask more rows than the full-width trees plus the
 * reference's {@code PAD_MAX} padding allowance. Unset, the total is the full-width total {@code bs * width}.
 *
 * <p><b>The per-request cut.</b> {@code neuralfree.draft.tree.budget} divides the total over the step's proposals by
 * their path scores, keeping every request at least one node: {@code full} (default) verifies every drafted node —
 * the strongest chains first when the step's row caps bind; {@code topt} keeps the {@code total - bs} strongest
 * nodes over all requests, keyed by the running minimum of the path scores minus a per-index epsilon, which makes
 * every request's kept nodes a prefix of its node list; {@code thresh} keeps a request's nodes while
 * {@code gamma * score >= log lambda} ({@code neuralfree.draft.tree.thresh.gamma / .lambda}, lambda at most 0
 * keeping everything); {@code fixed} gives every request {@code neuralfree.draft.tree.fixed} nodes. {@code topt}'s
 * total without a rows tier is {@code round(neuralfree.draft.tree.mean * bs)}, the reference's mean verified nodes
 * per request; {@code neuralfree.draft.tree.min} floors every request's kept nodes (a floor above what the total
 * fills is a policy-studies setting that breaks the sum, as in the reference).
 *
 * <p>Every cut is a per-request node prefix: the node list keeps parents ahead of children and the scores descend
 * along it, so the first {@code n_i} nodes are the {@code n_i}-node proposal and the target's greedy accept
 * restricted to them is the greedy accept on that proposal — the cut moves no token, it only shortens the round.
 * Immutable.
 */
final class TreeBudget {

    /** Extra rows a total tier may ask above the full-width trees (SpecPilot's {@code PAD_MAX}). */
    static final int PAD_MAX = 64;
    /** Per-index epsilon of the {@code topt} keys (the reference's arbitrary tie-break). */
    private static final double EPS = 1e-4;


    /** One tier of the rows table: covers batch sizes up to {@code bsMax}, total {@code t} or {@code ceil(bs * f)}. */
    private static final class Tier {
        final int bsMax;
        final int t;
        final double f;
        final boolean scaled;

        Tier(int bsMax, int t, double f, boolean scaled) {
            this.bsMax = bsMax;
            this.t = t;
            this.f = f;
            this.scaled = scaled;
        }
    }

    private final TreeBudgetPolicy policy;
    /** The rows tiers ascending in {@code bsMax}; empty for the full-width total. */
    private final List<Tier> tiers;
    /** The {@code topt} mean nodes per request by batch size, ascending cut points. */
    private final double[] meanBs;
    private final double[] meanAt;
    private final double gamma;
    private final double lambda;
    private final int fixed;
    private final int minNodes;

    private TreeBudget(TreeBudgetPolicy policy, List<Tier> tiers, double[] meanBs, double[] meanAt, double gamma,
                       double lambda, int fixed, int minNodes) {
        this.policy = policy;
        this.tiers = tiers;
        this.meanBs = meanBs;
        this.meanAt = meanAt;
        this.gamma = gamma;
        this.lambda = lambda;
        this.fixed = fixed;
        this.minNodes = minNodes;
    }

    /**
     * The budget of {@code options}: the rows tiers of {@code neuralfree.draft.tree.rows} and the policy of
     * {@code neuralfree.draft.tree.budget} with its thresholds.
     *
     * @throws IllegalArgumentException naming the property for a value it cannot parse
     */
    static TreeBudget of(SpeculativeOptions o) {
        String rows = o.treeRows();
        List<Tier> tiers = new ArrayList<>();
        if (rows != null && !rows.trim().isEmpty()) {
            String raw = rows.trim();
            if (raw.indexOf(':') < 0) {
                int t = parseIntTier(SpeculativeOptions.PROP_TREE_ROWS, raw);
                tiers.add(new Tier(Integer.MAX_VALUE, t, t, false));
            } else {
                for (String item : raw.split(";")) {
                    String part = item.trim();
                    if (part.isEmpty()) {
                        throw new IllegalArgumentException(SpeculativeOptions.PROP_TREE_ROWS + "=" + rows
                                + " holds an empty tier");
                    }
                    int colon = part.indexOf(':');
                    String bs = part.substring(0, colon).trim();
                    String v = part.substring(colon + 1).trim();
                    int bsMax;
                    try {
                        bsMax = Integer.parseInt(bs);
                    } catch (NumberFormatException e) {
                        throw new IllegalArgumentException(SpeculativeOptions.PROP_TREE_ROWS + "=" + rows
                                + " tier \"" + part + "\" has no batch size", e);
                    }
                    if (bsMax < 1) {
                        throw new IllegalArgumentException(SpeculativeOptions.PROP_TREE_ROWS + "=" + rows
                                + " tier \"" + part + "\" covers no batch size");
                    }
                    if (v.startsWith("x")) {
                        double f = parseNumber(SpeculativeOptions.PROP_TREE_ROWS, rows, v.substring(1));
                        if (!(f >= 0.0)) {
                            throw new IllegalArgumentException(SpeculativeOptions.PROP_TREE_ROWS + "=" + rows
                                    + " tier \"" + part + "\" scales a negative rows-per-request factor");
                        }
                        tiers.add(new Tier(bsMax, 0, f, true));
                    } else {
                        int t = parseIntTier(SpeculativeOptions.PROP_TREE_ROWS, v);
                        if (t < 1) {
                            throw new IllegalArgumentException(SpeculativeOptions.PROP_TREE_ROWS + "=" + rows
                                    + " tier \"" + part + "\" holds fewer rows than one sequence verifies");
                        }
                        tiers.add(new Tier(bsMax, t, t, false));
                    }
                }
                tiers.sort((a, b) -> Integer.compare(a.bsMax, b.bsMax));
            }
        }
        double[] meanBs = new double[0];
        double[] meanAt = new double[0];
        String mean = o.treeMean();
        if (mean != null && !mean.trim().isEmpty()) {
            List<double[]> pts = parseTable(SpeculativeOptions.PROP_TREE_MEAN, mean);
            meanBs = new double[pts.size()];
            meanAt = new double[pts.size()];
            for (int i = 0; i < pts.size(); i++) {
                meanBs[i] = pts.get(i)[0];
                meanAt[i] = pts.get(i)[1];
            }
        }
        return new TreeBudget(o.treeBudgetPolicy(), tiers, meanBs, meanAt, o.treeThreshGamma(), o.treeThreshLambda(),
                o.treeFixed(), o.treeMinNodes());
    }

    private static int parseIntTier(String key, String v) {
        double d = parseNumber(key, v, v);
        if (d != Math.rint(d) || d < 1) {
            throw new IllegalArgumentException(key + "=" + v + " is not a positive whole row total");
        }
        return (int) d;
    }

    private static double parseNumber(String key, String raw, String v) {
        try {
            return Double.parseDouble(v.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + "=" + raw + " entry \"" + v + "\" is not a number", e);
        }
    }

    /** {@code "bs:v;..."} or a single number, as ascending (bs, v) points. */
    private static List<double[]> parseTable(String key, String raw) {
        List<double[]> pts = new ArrayList<>();
        String trimmed = raw.trim();
        if (trimmed.indexOf(':') < 0) {
            double v = parseNumber(key, trimmed, trimmed);
            pts.add(new double[] {1.0, v});
            return pts;
        }
        for (String item : trimmed.split(";")) {
            String part = item.trim();
            if (part.isEmpty()) {
                throw new IllegalArgumentException(key + "=" + raw + " holds an empty entry");
            }
            int colon = part.indexOf(':');
            pts.add(new double[] {parseNumber(key, trimmed, part.substring(0, colon)),
                    parseNumber(key, trimmed, part.substring(colon + 1))});
        }
        pts.sort((a, b) -> Double.compare(a[0], b[0]));
        return pts;
    }

    /** The configured policy. */
    TreeBudgetPolicy policy() {
        return policy;
    }

    /**
     * The total tree rows the step of {@code bs} proposals of at most {@code width} nodes each lays out
     * (SpecPilot's {@code tokens_for_bs}): the first tier that covers {@code bs}, or the full-width total, clamped
     * into {@code [bs, bs * width + PAD_MAX]}.
     */
    int totalFor(int bs, int width) {
        int t;
        if (tiers.isEmpty()) {
            t = bs * width;
        } else {
            Tier tier = null;
            for (Tier candidate : tiers) {
                if (bs <= candidate.bsMax) {
                    tier = candidate;
                    break;
                }
            }
            if (tier == null) {
                tier = tiers.get(tiers.size() - 1);
            }
            t = tier.scaled ? (int) Math.ceil(bs * tier.f) : tier.t;
        }
        return Math.max(bs, Math.min(bs * width + PAD_MAX, t));
    }

    /** The {@code topt} total without a rows tier: {@code round(mean(bs) * bs)} in {@code [bs, bs * width]}. */
    int toptTotalFor(int bs, int width) {
        double mean = meanAt.length == 0 ? 1.0 : interpolate(meanBs, meanAt, bs);
        long t = Math.round(mean * bs);
        return (int) Math.max(bs, Math.min((long) bs * width, t));
    }

    private static double interpolate(double[] xs, double[] ys, double x) {
        if (xs.length == 1 || x <= xs[0]) {
            return ys[0];
        }
        for (int i = 1; i < xs.length; i++) {
            if (x <= xs[i]) {
                double f = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
                return ys[i - 1] + (ys[i] - ys[i - 1]) * f;
            }
        }
        return ys[ys.length - 1];
    }

    /**
     * Divides {@code total} tree nodes over the proposals by the policy: {@code out[i]} is proposal {@code i}'s kept
     * node count in {@code [min(1, nodes), nodes_i]}, a node prefix of it. {@code total} is the step's node budget
     * across proposals; a budget above the proposals' node count keeps everything.
     *
     * @param scores per proposal, its nodes' path scores (non-increasing, node 0 the root at 0.0)
     */
    int[] cut(double[][] scores, int total) {
        int bs = scores.length;
        int[] base = new int[bs];
        int nodes = 0;
        for (int i = 0; i < bs; i++) {
            base[i] = baseCount(scores[i]);
            nodes += base[i];
        }
        int[] out = capToTotal(scores, base, Math.min(total, nodes));
        if (minNodes > 1) {
            for (int i = 0; i < bs; i++) {
                out[i] = Math.max(out[i], Math.min(minNodes, scores[i].length));
            }
        }
        return out;
    }

    /** The policy's per-request node count before the step total caps it. */
    private int baseCount(double[] score) {
        int n = score.length;
        switch (policy) {
            case FULL:
            case TOPT:
                return n;
            case FIXED:
                return Math.max(Math.min(minNodes, n), Math.min(fixed, n));
            case THRESH:
                double logLambda = lambda <= 0.0 ? Double.NEGATIVE_INFINITY : Math.log(lambda);
                for (int j = 1; j < n; j++) {
                    if (gamma * score[j] < logLambda) {
                        return j;
                    }
                }
                return n;
            default:
                throw new IllegalStateException("unhandled policy " + policy);
        }
    }

    /**
     * Caps the per-request counts so their sum is at most {@code total}, dropping the weakest nodes first: the
     * global top-{@code (total - bs)} over the nodes past each request's root, keyed by the running minimum of the
     * path scores minus the per-index epsilon (the reference's {@code rp_topt} keys), so what stays of every request
     * is a prefix of its node list.
     */
    private static int[] capToTotal(double[][] scores, int[] base, int total) {
        int bs = scores.length;
        int[] out = new int[bs];
        int nodes = 0;
        for (int i = 0; i < bs; i++) {
            out[i] = base[i];
            nodes += base[i];
        }
        int extra = total - bs;
        if (extra <= 0) {
            Arrays.fill(out, 1);
            return out;
        }
        if (extra >= nodes - bs) {
            return out;
        }
        // candidates: (request, node j >= 1) with key = runningMin(score[j]) - (j - 1) * EPS; pick the best `extra`
        int candidates = nodes - bs;
        double[] keys = new double[candidates];
        int[] req = new int[candidates];
        int[] idx = new int[candidates];
        int c = 0;
        for (int i = 0; i < bs; i++) {
            double min = Double.POSITIVE_INFINITY;
            for (int j = 1; j < base[i]; j++) {
                min = Math.min(min, scores[i][j]);
                keys[c] = min - (j - 1) * EPS;
                req[c] = i;
                idx[c] = j;
                c++;
            }
        }
        Integer[] order = new Integer[c];
        for (int i = 0; i < c; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> keys[a] != keys[b]
                ? (keys[a] > keys[b] ? -1 : 1)
                : Integer.compare(req[a], req[b]));
        int[] kept = new int[bs];
        for (int i = 0; i < extra; i++) {
            kept[req[order[i]]] = Math.max(kept[req[order[i]]], idx[order[i]]);
        }
        for (int i = 0; i < bs; i++) {
            out[i] = kept[i] + 1;
        }
        return out;
    }
}
