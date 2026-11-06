package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.engine.StopReason;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;

import static org.junit.Assert.assertTrue;

/**
 * Exact-enumeration harness for the {@link Verifier} implementations on a 4-token vocabulary: verify-row
 * distributions, the reference quantities of rejection sampling ({@code min(1, p/q)}, the residual {@code (p - q)+}
 * and its cdf breakpoints), scripted verify rounds driven by a fixed uniform list, and the walk that drives one round
 * per uniform cell so the collected runs carry the exact measure of every outcome.
 */
final class VerifierEnumeration {

    static final int VOCAB = 4;

    private VerifierEnumeration() {
    }

    /** A verify row distribution with the chain draw on the first argmax. */
    static Verifier.Distribution dist(double... probs) {
        int[] tokens = new int[probs.length];
        float[] p = new float[probs.length];
        float[] l = new float[probs.length];
        int argmax = 0;
        for (int i = 0; i < probs.length; i++) {
            tokens[i] = i;
            p[i] = (float) probs[i];
            l[i] = probs[i] <= 0.0 ? Float.NEGATIVE_INFINITY : (float) Math.log(probs[i]);
            if (probs[i] > probs[argmax]) {
                argmax = i;
            }
        }
        return new Verifier.Distribution(tokens, p, l, argmax);
    }

    /** The same with an explicit chain draw. */
    static Verifier.Distribution distSelected(int selected, double... probs) {
        Verifier.Distribution d = dist(probs);
        int[] tokens = new int[d.size()];
        float[] p = new float[d.size()];
        float[] l = new float[d.size()];
        for (int i = 0; i < d.size(); i++) {
            tokens[i] = d.token(i);
            p[i] = d.prob(i);
            l[i] = d.logit(i);
        }
        return new Verifier.Distribution(tokens, p, l, selected);
    }

    /** A greedy chain's row: one-hot probabilities on the argmax, the logits otherwise intact. */
    static Verifier.Distribution greedyRow(int selected, double... logitsValues) {
        float[] l = new float[logitsValues.length];
        for (int i = 0; i < l.length; i++) {
            l[i] = (float) logitsValues[i];
        }
        float[] p = new float[l.length];
        p[selected] = 1f;
        int[] tokens = new int[l.length];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = i;
        }
        return new Verifier.Distribution(tokens, p, l, selected);
    }

    /** The values as an array. */
    static double[] array(double... values) {
        return values;
    }

    /** The sum of the values. */
    static double sum(double[] v) {
        double total = 0.0;
        for (double x : v) {
            total += x;
        }
        return total;
    }

    /** {@code min(1, p/q)} per token. */
    static double[] minRatios(double[] p, double[] q) {
        double[] a = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            a[i] = q[i] > 0 ? Math.min(1.0, p[i] / q[i]) : 0.0;
        }
        return a;
    }

    /** {@code (p - q)+}. */
    static double[] residualWeights(double[] p, double[] q) {
        double[] w = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            w[i] = Math.max(0.0, p[i] - q[i]);
        }
        return w;
    }

    /** The cdf thresholds strictly inside (0, 1) of a weight vector's runs. */
    static double[] cdfBreakpoints(double[] weights) {
        List<Double> out = new ArrayList<>();
        double total = sum(weights);
        double run = 0.0;
        for (double w : weights) {
            if (w > 0.0) {
                run += w;
                if (run < total) {
                    out.add(run / total);
                }
            }
        }
        double[] a = new double[out.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = out.get(i);
        }
        return a;
    }

    /** Threshold set for a grid: entries strictly inside (0, 1). */
    static double[] thresholds(double... in) {
        List<Double> out = new ArrayList<>();
        for (double t : in) {
            if (t > 0.0 && t < 1.0) {
                out.add(t);
            }
        }
        double[] a = new double[out.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = out.get(i);
        }
        return a;
    }

    /** The union of several threshold sets. */
    static double[] union(double[]... grids) {
        List<Double> out = new ArrayList<>();
        for (double[] g : grids) {
            for (double t : g) {
                out.add(t);
            }
        }
        double[] a = new double[out.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = out.get(i);
        }
        return a;
    }

    // ---------------------------------------------------------------- scripted rounds

    /** A scripted verify round: fixed rows and drafts plus a uniform list; an emission never stops the round. */
    static final class ScriptRound implements Verifier.Round {

        private final Verifier.Distribution[] targets;
        private final Verifier.Distribution[] drafts;
        private final int[] draftTokens;
        private final Deque<Double> uniforms;
        final List<Integer> emitted = new ArrayList<>();
        private int lastRow = -1;
        int confirmed;
        int uniformsUsed;
        boolean overflowed;

        ScriptRound(Verifier.Distribution[] targets, Verifier.Distribution[] drafts, int[] draftTokens,
                    double[] uniformValues) {
            this.targets = targets;
            this.drafts = drafts;
            this.draftTokens = draftTokens;
            this.uniforms = new ArrayDeque<>();
            for (double u : uniformValues) {
                this.uniforms.add(u);
            }
        }

        @Override public int draftCount() {
            return draftTokens.length;
        }

        @Override public int draftToken(int position) {
            return draftTokens[position];
        }

        @Override public Verifier.Distribution target(int row) {
            if (row <= lastRow || row >= targets.length) {
                throw new IllegalStateException("verify row " + row + " outside the ascending-once order (last "
                        + lastRow + ")");
            }
            lastRow = row;
            return targets[row];
        }

        @Override public Verifier.Distribution draft(int position) {
            return drafts[position];
        }

        @Override public double uniform() {
            if (uniforms.isEmpty()) {
                // the branch wants a draw this script does not carry: the run is discarded and the
                // walk subdivides the next depth's grid to enumerate that draw (walk's overflow case)
                overflowed = true;
                return 0.5;
            }
            uniformsUsed++;
            return uniforms.poll();
        }

        @Override public StopReason confirm(int position) {
            if (position != confirmed) {
                throw new IllegalStateException("confirm(" + position + ") after " + confirmed);
            }
            emitted.add(draftTokens[position]);
            confirmed++;
            return null;
        }

        @Override public StopReason emit(int token) {
            emitted.add(token);
            return null;
        }

        @Override public int accepted() {
            return confirmed;
        }

        @Override public boolean stopped() {
            return false;
        }
    }

    /** One enumerated outcome: how many draft tokens were confirmed and what the emissions were. */
    static final class Run {
        double prob;
        final int confirmed;
        final int finalToken;
        final List<Integer> emitted;

        Run(double prob, int confirmed, int finalToken, List<Integer> emitted) {
            this.prob = prob;
            this.confirmed = confirmed;
            this.finalToken = finalToken;
            this.emitted = emitted;
        }
    }

    /** The runs with their measures scaled by {@code weight}. */
    static List<Run> weighted(List<Run> runs, double weight) {
        List<Run> out = new ArrayList<>(runs.size());
        for (Run run : runs) {
            out.add(new Run(run.prob * weight, run.confirmed, run.finalToken, run.emitted));
        }
        return out;
    }

    /**
     * Exact enumeration: the per-depth grid holds the union of thresholds any reachable branch's draw at that
     * depth compares; each cell's midpoint drives one scripted round and collects the cell's whole measure (the
     * outcome is constant on the interior of its region). A branch that finished on fewer draws than the script
     * carries records its run once, with the deeper dimensions integrating out. A branch that wanted a draw
     * beyond the script records nothing and is re-driven at the next depth, where that draw is enumerated; the
     * measure it already spent (this depth's width) carries into the subdivision.
     */
    static List<Run> enumerate(Verifier verifier, RoundMaker maker, double[]... depthGrids) {
        List<Run> runs = new ArrayList<>();
        walk(verifier, maker, depthGrids, 0, new ArrayList<Double>(), 1.0, runs);
        return runs;
    }

    private static void walk(Verifier verifier, RoundMaker maker, double[][] grids, int depth,
                             List<Double> chosen, double prob, List<Run> runs) {
        double[] grid = cellBoundaries(grids[Math.min(depth, grids.length - 1)]);
        for (int i = 0; i + 1 < grid.length; i++) {
            double width = grid[i + 1] - grid[i];
            List<Double> uniform = new ArrayList<>(chosen);
            uniform.add((grid[i] + grid[i + 1]) / 2.0);
            ScriptRound round = maker.round(toArray(uniform));
            verifier.verify(round);
            if (round.overflowed) {
                walk(verifier, maker, grids, depth + 1, uniform, prob * width, runs);
                continue;
            }
            runs.add(new Run(prob * width, round.confirmed, round.emitted.get(round.emitted.size() - 1),
                    round.emitted));
        }
    }

    private static double[] toArray(List<Double> list) {
        double[] a = new double[list.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = list.get(i);
        }
        return a;
    }

    /** {@code 0, sorted cell boundaries, 1} of one draw. */
    private static double[] cellBoundaries(double[] thresholdsIn) {
        double[] all = new double[thresholdsIn.length + 2];
        all[0] = 0.0;
        all[all.length - 1] = 1.0;
        System.arraycopy(thresholdsIn, 0, all, 1, thresholdsIn.length);
        Arrays.sort(all);
        int n = 0;
        for (double v : all) {
            if (n == 0 || v != all[n - 1]) {
                all[n++] = v;
            }
        }
        return Arrays.copyOf(all, n);
    }

    /** The measure of the runs that confirmed {@code confirmed} tokens and emitted {@code finalToken} last. */
    static double measure(List<Run> runs, int confirmed, int finalToken) {
        double total = 0.0;
        for (Run run : runs) {
            if (run.confirmed == confirmed && run.finalToken == finalToken) {
                total += run.prob;
            }
        }
        return total;
    }

    /** The measure of the runs that confirmed {@code confirmed} tokens. */
    static double measureConfirmed(List<Run> runs, int confirmed) {
        double total = 0.0;
        for (Run run : runs) {
            if (run.confirmed == confirmed) {
                total += run.prob;
            }
        }
        return total;
    }

    /** The expected confirmed length over the runs. */
    static double expectedLength(List<Run> runs) {
        double total = 0.0;
        for (Run run : runs) {
            total += run.prob * run.confirmed;
        }
        return total;
    }

    /** The emission law over the runs: P(final == token) per token of the 4-token vocabulary. */
    static double[] emissionLaw(List<Run> runs) {
        double[] law = new double[VOCAB];
        for (Run run : runs) {
            law[run.finalToken] += run.prob;
        }
        return law;
    }

    /** The marginal law of the rounds' first emission: the decision token of the opening position. */
    static double[] firstEmissionLaw(List<Run> runs) {
        double[] law = new double[VOCAB];
        for (Run run : runs) {
            law[run.emitted.get(0)] += run.prob;
        }
        return law;
    }

    /** Builds the scripted round of one uniform list. */
    interface RoundMaker {
        ScriptRound round(double[] uniforms);
    }

    /** Asserts {@code |expected - actual| <= delta}, naming the quantity. */
    static void assertEq(double expected, double actual, double delta, String what) {
        assertTrue(what + ": expected " + expected + " got " + actual, Math.abs(expected - actual) <= delta);
    }

    /** Returns a k = 2 round with per-position target and draft rows for the given proposal pair. */
    static RoundMaker twoTokenRound(final double[] p0, final double[] p1, final double[] q0,
                                    final double[] q1, final int first, final int second) {
        return new RoundMaker() {
            @Override public ScriptRound round(double[] uniforms) {
                return new ScriptRound(
                        new Verifier.Distribution[]{dist(p0), dist(p1), distSelected(3, 1.0, 0, 0, 0)},
                        new Verifier.Distribution[]{dist(q0), dist(q1)}, new int[]{first, second}, uniforms);
            }
        };
    }

    /** Drives one scripted round of {@code maker} through {@code verifier}. */
    static ScriptRound round(Verifier verifier, RoundMaker maker, double[] uniforms) {
        ScriptRound script = maker.round(uniforms);
        verifier.verify(script);
        return script;
    }

    /** Appends the runs of {@code add} to {@code target}. */
    static void plusInto(List<Run> target, List<Run> add) {
        target.addAll(add);
    }

    /** A drafter row with all its mass on {@code token}. */
    static Verifier.Distribution oneHotDrafter(int token) {
        double[] q = new double[VOCAB];
        q[token] = 1.0;
        return dist(q);
    }
}
