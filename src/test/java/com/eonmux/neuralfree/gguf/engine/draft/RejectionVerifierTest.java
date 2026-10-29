package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Test;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The SP-2 verification modes on a 4-token vocabulary: REJECTION's and BLOCK's distribution preservation by exact
 * enumeration (every uniform cell of the confirmation regions, under every proposal the drafter's distribution
 * gives), FLy's deferral on constructed cases, SYNTHETIC's rate law and its {@code neuralfree.spec.synthetic} gate,
 * the greedy fallback of the lossless modes, the drafter probability contract, and the stop-inside-a-verify state
 * handling of SpeculativeDecoder (PARITY §5.5.1 S-1). Every decoder runs against a fake target whose rows are plain
 * scripted logits: with temperature 1 and no truncation stages the 4-token sampler chain leaves the probabilities
 * intact, so the script is the processed target distribution itself.
 */
public class RejectionVerifierTest {

    private static final int VOCAB = 4;
    private static final int N_CTX = 32;
    private static final int UBATCH = 8;
    private static final long SEED = 42;

    // ---------------------------------------------------------------- helpers: rows, references, enumeration

    /** Logits of probabilities: temperature 1 with no truncation stages gives them back. */
    private static float[] logits(double... probs) {
        float[] row = new float[probs.length];
        for (int i = 0; i < probs.length; i++) {
            row[i] = probs[i] <= 0.0 ? Float.NEGATIVE_INFINITY : (float) Math.log(probs[i]);
        }
        return row;
    }

    /** A verify row distribution with the chain draw on the first argmax. */
    private static Verifier.Distribution dist(double... probs) {
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
    private static Verifier.Distribution distSelected(int selected, double... probs) {
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
    private static Verifier.Distribution greedyRow(int selected, double... logitsValues) {
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

    private static double[] array(double... values) {
        return values;
    }

    private static double sum(double[] v) {
        double total = 0.0;
        for (double x : v) {
            total += x;
        }
        return total;
    }

    /** {@code min(1, p/q)} per token. */
    private static double[] minRatios(double[] p, double[] q) {
        double[] a = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            a[i] = q[i] > 0 ? Math.min(1.0, p[i] / q[i]) : 0.0;
        }
        return a;
    }

    /** {@code (p - q)+}. */
    private static double[] residualWeights(double[] p, double[] q) {
        double[] w = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            w[i] = Math.max(0.0, p[i] - q[i]);
        }
        return w;
    }

    /** The cdf thresholds strictly inside (0, 1) of a weight vector's runs. */
    private static double[] cdfBreakpoints(double[] weights) {
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
    private static double[] thresholds(double... in) {
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
    private static double[] union(double[]... grids) {
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
    private static final class ScriptRound implements Verifier.Round {

        private final Verifier.Distribution[] targets;
        private final Verifier.Distribution[] drafts;
        private final int[] draftTokens;
        private final Deque<Double> uniforms;
        private final List<Integer> emitted = new ArrayList<>();
        private int lastRow = -1;
        private int confirmed;
        private int uniformsUsed;
        private boolean overflowed;

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
    private static final class Run {
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

    private static List<Run> weighted(List<Run> runs, double weight) {
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
    private static List<Run> enumerate(Verifier verifier, RoundMaker maker, double[]... depthGrids) {
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

    private static double measure(List<Run> runs, int confirmed, int finalToken) {
        double total = 0.0;
        for (Run run : runs) {
            if (run.confirmed == confirmed && run.finalToken == finalToken) {
                total += run.prob;
            }
        }
        return total;
    }

    private static double measureConfirmed(List<Run> runs, int confirmed) {
        double total = 0.0;
        for (Run run : runs) {
            if (run.confirmed == confirmed) {
                total += run.prob;
            }
        }
        return total;
    }

    private static double expectedLength(List<Run> runs) {
        double total = 0.0;
        for (Run run : runs) {
            total += run.prob * run.confirmed;
        }
        return total;
    }

    /** The emission law over the runs: P(final == token) per token of the 4-token vocabulary. */
    private static double[] emissionLaw(List<Run> runs) {
        double[] law = new double[VOCAB];
        for (Run run : runs) {
            law[run.finalToken] += run.prob;
        }
        return law;
    }

    /** The marginal law of the rounds' first emission: the decision token of the opening position. */
    private static double[] firstEmissionLaw(List<Run> runs) {
        double[] law = new double[VOCAB];
        for (Run run : runs) {
            law[run.emitted.get(0)] += run.prob;
        }
        return law;
    }

    private interface RoundMaker {
        ScriptRound round(double[] uniforms);
    }

    private static void assertEq(double expected, double actual, double delta, String what) {
        assertTrue(what + ": expected " + expected + " got " + actual, Math.abs(expected - actual) <= delta);
    }

    // ---------------------------------------------------------------- REJECTION: exact enumeration

    /** Returns a k = 1 round on the 4-token vocabulary for the given proposal. */
    private static RoundMaker rejectionRound1(final double[] p0, final int bonusSel, final double[] q0,
                                              final int proposal) {
        return new RoundMaker() {
            @Override public ScriptRound round(double[] uniforms) {
                return new ScriptRound(
                        new Verifier.Distribution[]{dist(p0), distSelected(bonusSel, 1.0, 0, 0, 0)},
                        new Verifier.Distribution[]{dist(q0)}, new int[]{proposal}, uniforms);
            }
        };
    }

    /** Returns a k = 2 round with per-position target and draft rows for the given proposal pair. */
    private static RoundMaker rejectionRound2(final double[] p0, final double[] p1, final double[] q0,
                                              final double[] q1, final int first, final int second) {
        return new RoundMaker() {
            @Override public ScriptRound round(double[] uniforms) {
                return new ScriptRound(
                        new Verifier.Distribution[]{dist(p0), dist(p1), distSelected(3, 1.0, 0, 0, 0)},
                        new Verifier.Distribution[]{dist(q0), dist(q1)}, new int[]{first, second}, uniforms);
            }
        };
    }

    private static RoundMaker blockRound2(final double[] p0, final double[] p1, final double[] q0,
                                          final double[] q1, final int first, final int second) {
        return new RoundMaker() {
            @Override public ScriptRound round(double[] uniforms) {
                return new ScriptRound(
                        new Verifier.Distribution[]{dist(p0), dist(p1), distSelected(3, 1.0, 0, 0, 0)},
                        new Verifier.Distribution[]{dist(q0), dist(q1)}, new int[]{first, second}, uniforms);
            }
        };
    }

    /**
     * Token-level rejection sampling on a 4-token vocabulary with a probabilistic drafter: the enumerated emission
     * law is exactly the target's, and the confirmed-length law is the acceptance-ratio sum.
     */
    @Test
    public void rejectionSamplingPreservesTheTargetDistributionByExactEnumeration() {
        double[] p0 = array(0.4, 0.3, 0.2, 0.1);
        double[] q0 = array(0.1, 0.2, 0.3, 0.4);
        double[] w0 = residualWeights(p0, q0);
        double[] aOf = minRatios(p0, q0);

        List<Run> total = new ArrayList<>();
        for (int d = 0; d < VOCAB; d++) {
            List<Run> runs = enumerate(new RejectionVerifier(), rejectionRound1(p0, 2, q0, d),
                    thresholds(aOf), thresholds(cdfBreakpoints(w0)));
            plusInto(total, weighted(runs, q0[d]));
        }
        assertEq(1.0, sum(firstEmissionLaw(total)), 1e-9, "total emission measure");
        double[] law = firstEmissionLaw(total);
        for (int x = 0; x < VOCAB; x++) {
            assertEq(p0[x], law[x], 1e-9, "rejection emission law for token " + x);
        }
        double accepted = 0.0;
        for (int d = 0; d < VOCAB; d++) {
            accepted += q0[d] * aOf[d];
        }
        assertEq(accepted, measureConfirmed(total, 1), 1e-9, "rejection confirmed measure");
        assertEq(1.0 - accepted, measureConfirmed(total, 0), 1e-9, "rejection rejected measure");
        // the replacement after the rejection is the residual, exactly.
        double[] normalizedW0 = normalized(w0);
        for (int x = 0; x < VOCAB; x++) {
            if (normalizedW0[x] > 0.0) {
                assertEq(normalizedW0[x], measure(total, 0, x) / measureConfirmed(total, 0), 1e-9,
                        "rejection residual law for token " + x);
            }
        }
    }

    private static double[] normalized(double[] w) {
        double total = sum(w);
        double[] out = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            out[i] = total > 0 ? w[i] / total : 0.0;
        }
        return out;
    }

    private static void plusInto(List<Run> target, List<Run> add) {
        target.addAll(add);
    }

    /**
     * Two-draft-token rejection sampling: the same lossless law over both positions, the confirmed-length law as
     * the product of the ratios, and both residual laws matching the target distributions.
     */
    @Test
    public void rejectionSamplingKeepsEveryPositionAlignedWithTheTargetByExactEnumeration() {
        double[] p0 = array(0.4, 0.3, 0.2, 0.1);
        double[] q0 = array(0.1, 0.2, 0.3, 0.4);
        double[] p1 = array(0.05, 0.25, 0.5, 0.2);
        double[] q1 = array(0.4, 0.3, 0.2, 0.1);
        double[] w0 = residualWeights(p0, q0);
        double[] w1 = residualWeights(p1, q1);
        double[] a0 = minRatios(p0, q0);
        double[] a1 = minRatios(p1, q1);

        double[] depth0 = thresholds(a0);
        double[] depth1 = thresholds(union(a1, cdfBreakpoints(w0)));
        double[] depth2 = thresholds(cdfBreakpoints(w1));
        List<Run> total = new ArrayList<>();
        for (int d0 = 0; d0 < VOCAB; d0++) {
            for (int d1 = 0; d1 < VOCAB; d1++) {
                List<Run> runs = enumerate(new RejectionVerifier(),
                        rejectionRound2(p0, p1, q0, q1, d0, d1), depth0, depth1, depth2);
                plusInto(total, weighted(runs, q0[d0] * q1[d1]));
            }
        }
        assertEq(1.0, sum(firstEmissionLaw(total)), 0.0, "total emission measure");
        double[] law = firstEmissionLaw(total);
        for (int x = 0; x < VOCAB; x++) {
            assertEq(p0[x], law[x], 1e-9, "two-position emission law for token " + x);
        }
        double m2 = 0.0;
        double m1 = 0.0;
        double m0 = 0.0;
        for (int d0 = 0; d0 < VOCAB; d0++) {
            for (int d1 = 0; d1 < VOCAB; d1++) {
                m2 += q0[d0] * q1[d1] * a0[d0] * a1[d1];
                m1 += q0[d0] * q1[d1] * a0[d0] * (1.0 - a1[d1]);
                m0 += q0[d0] * q1[d1] * (1.0 - a0[d0]);
            }
        }
        assertEq(m2, measureConfirmed(total, 2), 1e-9, "confirmed-2 measure");
        assertEq(m1, measureConfirmed(total, 1), 1e-9, "confirmed-1 measure");
        assertEq(m0, measureConfirmed(total, 0), 1e-9, "confirmed-0 measure");
        double[] nw0 = normalized(w0);
        double[] nw1 = normalized(w1);
        for (int x = 0; x < VOCAB; x++) {
            assertEq(nw0[x], measure(total, 0, x) / measureConfirmed(total, 0), 1e-9,
                    "replacement law after 0 confirms, token " + x);
            assertEq(nw1[x], measure(total, 1, x) / measureConfirmed(total, 1), 1e-9,
                    "replacement law after 1 confirm, token " + x);
        }
        // the bonus after a fully confirmed block is the chain draw of the last row (scripted: token 3).
        assertEq(m2, measure(total, 2, 3), 1e-9, "bonus law after 2 confirms");
    }

    // ---------------------------------------------------------------- BLOCK: exact enumeration

    /**
     * Block verification (Sun et al., Algorithm 3): the same lossless law under the block coupling's
     * confirmed-length measures, and an accepted length never below token-level rejection's for the same inputs.
     */
    @Test
    public void blockVerificationPreservesTheTargetDistributionByExactEnumeration() {
        double[] p0 = array(0.4, 0.3, 0.2, 0.1);
        double[] q0 = array(0.1, 0.2, 0.3, 0.4);
        double[] p1 = array(0.05, 0.25, 0.5, 0.2);
        double[] q1 = array(0.4, 0.3, 0.2, 0.1);
        double[] a0 = minRatios(p0, q0);
        double[] a1 = minRatios(p1, q1);
        double[] w0 = residualWeights(p0, q0);
        double[] w1 = residualWeights(p1, q1);

        double[] depth0 = new double[VOCAB * VOCAB];
        double[] depth1 = new double[VOCAB * VOCAB];
        double[][] a0Of = new double[VOCAB][VOCAB];
        double[][] a1Of = new double[VOCAB][VOCAB];
        double[][][] w0Of = new double[VOCAB][VOCAB][];
        double[][][] w1Of = new double[VOCAB][VOCAB][];
        int n = 0;
        for (int d0 = 0; d0 < VOCAB; d0++) {
            for (int d1 = 0; d1 < VOCAB; d1++) {
                double mb = p0[d0] * p1[d1];
                double ms = q0[d0] * q1[d1];
                a0Of[d0][d1] = mb < ms ? mb / ms : 1.0;
                double bm1 = p0[d0];
                double ms1 = q0[d0];
                double remain1 = weightedPositive(p1, q1, bm1, ms1, true);
                double rej1 = weightedPositive(p1, q1, bm1, ms1, false);
                a1Of[d0][d1] = rej1 > 0 ? (remain1 >= rej1 ? 1.0 : remain1 / rej1) : 1.0;
                w0Of[d0][d1] = normalizedPositive(p0, q0, 1.0, 1.0);
                w1Of[d0][d1] = normalizedPositive(p1, q1, bm1, ms1);
                depth0[n] = a0Of[d0][d1];
                depth1[n] = a1Of[d0][d1];
                n++;
            }
        }
        double[] cdfs0 = cdfsOf(w0Of);
        double[] cdfs1 = cdfsOf(w1Of);
        List<Run> total = new ArrayList<>();
        for (int d0 = 0; d0 < VOCAB; d0++) {
            for (int d1 = 0; d1 < VOCAB; d1++) {
                List<Run> runs = enumerate(new BlockVerifier(),
                        blockRound2(p0, p1, q0, q1, d0, d1),
                        thresholds(depth0), thresholds(depth1), thresholds(union(cdfs0, cdfs1)));
                plusInto(total, weighted(runs, q0[d0] * q1[d1]));
            }
        }
        assertEq(1.0, sum(firstEmissionLaw(total)), 1e-9, "block total measure");
        double[] law = firstEmissionLaw(total);
        for (int x = 0; x < VOCAB; x++) {
            assertEq(p0[x], law[x], 1e-9, "block emission law for token " + x);
        }
        double m2 = 0.0;
        double m1 = 0.0;
        for (int d0 = 0; d0 < VOCAB; d0++) {
            for (int d1 = 0; d1 < VOCAB; d1++) {
                m2 += q0[d0] * q1[d1] * a0Of[d0][d1];
                m1 += q0[d0] * q1[d1] * (1.0 - a0Of[d0][d1]) * a1Of[d0][d1];
            }
        }
        double m0 = 1.0 - m2 - m1;
        assertEq(m2, measureConfirmed(total, 2), 1e-9, "block confirmed-2 measure");
        assertEq(m1, measureConfirmed(total, 1), 1e-9, "block confirmed-1 measure");
        assertEq(m0, measureConfirmed(total, 0), 1e-9, "block confirmed-0 measure");
        for (int x = 0; x < VOCAB; x++) {
            assertEq(w0Of[0][0][x], measure(total, 0, x) / measureConfirmed(total, 0), 1e-9,
                    "block residual law after 0 confirms, token " + x);
            // after 1 confirm the residual weights scale with the confirmed proposal's masses,
            // so the law is the proposal-weighted mixture of the per-proposal residuals
            double mix1 = 0.0;
            for (int d0 = 0; d0 < VOCAB; d0++) {
                for (int d1 = 0; d1 < VOCAB; d1++) {
                    mix1 += q0[d0] * q1[d1] * (1.0 - a0Of[d0][d1]) * a1Of[d0][d1] * w1Of[d0][d1][x];
                }
            }
            assertEq(mix1 / m1, measure(total, 1, x) / measureConfirmed(total, 1), 1e-9,
                    "block residual law after 1 confirm, token " + x);
        }
        // the bonus after a fully confirmed block is the chain draw of the last row (scripted: token 3).
        assertEq(m2, measure(total, 2, 3), 1e-9, "block bonus law");
        // and block verification is never worse in expected accepted length than token-level rejection.
        List<Run> rejectionRuns = new ArrayList<>();
        for (int d0 = 0; d0 < VOCAB; d0++) {
            for (int d1 = 0; d1 < VOCAB; d1++) {
                List<Run> runs = enumerate(new RejectionVerifier(),
                        rejectionRound2(p0, p1, q0, q1, d0, d1),
                        thresholds(a0), thresholds(union(a1, cdfBreakpoints(w0))),
                        thresholds(cdfBreakpoints(w1)));
                plusInto(rejectionRuns, weighted(runs, q0[d0] * q1[d1]));
            }
        }
        assertTrue("block must not lose accepted length: " + expectedLength(total) + " vs "
                        + expectedLength(rejectionRuns),
                expectedLength(total) >= expectedLength(rejectionRuns) - 1e-12);
    }

    private static double[] cdfsOf(double[][][] weights) {
        List<Double> out = new ArrayList<>();
        for (double[][] entry : weights) {
            for (double[] w : entry) {
                for (double t : cdfBreakpoints(w)) {
                    out.add(t);
                }
            }
        }
        double[] a = new double[out.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = out.get(i);
        }
        return a;
    }

    /** {@code sum_x (a·p(x) − b·q(x))+} (remain) or the negative-side sum (rej). */
    private static double weightedPositive(double[] p, double[] q, double a, double b, boolean remain) {
        double total = 0.0;
        for (int i = 0; i < p.length; i++) {
            double w = remain ? a * p[i] - b * q[i] : b * q[i] - a * p[i];
            if (w > 0.0) {
                total += w;
            }
        }
        return total;
    }

    /** {@code (a·p − b·q)+}, normalized. */
    private static double[] normalizedPositive(double[] p, double[] q, double a, double b) {
        double[] w = new double[p.length];
        double total = 0.0;
        for (int i = 0; i < p.length; i++) {
            w[i] = Math.max(0.0, a * p[i] - b * q[i]);
            total += w[i];
        }
        if (total > 0.0) {
            for (int i = 0; i < w.length; i++) {
                w[i] /= total;
            }
        }
        return w;
    }

    /**
     * A whole block whose joint ratio reaches one accepts unconditionally: the confirmed length is the full draft,
     * and only the opening uniform is consumed.
     */
    @Test
    public void blockVerificationAcceptsAWholeBlockWhoseJointRatioIsAtLeastOne() {
        double[] p = array(0.4, 0.3, 0.2, 0.1);
        double[] q = array(0.1, 0.2, 0.33, 0.37);
        // proposal (0, 1): Mb = 0.4·0.3 = 0.12, Ms = 0.1·0.2 = 0.02, ratio 6.
        ScriptRound only = round(new BlockVerifier(), blockRound2(p, p, q, q, 0, 1), new double[]{0.5});
        assertEquals(2, only.confirmed);
        assertEquals(Arrays.asList(0, 1, 3), only.emitted);
        assertEquals(1, only.uniformsUsed);
    }

    private static ScriptRound round(Verifier verifier, RoundMaker maker, double[] uniforms) {
        ScriptRound script = maker.round(uniforms);
        verifier.verify(script);
        return script;
    }

    // ---------------------------------------------------------------- FLy: deferral regions

    /**
     * Non-greedy FLy defers a high-entropy rejection whose next {@code W} draft tokens are natively accepted: the
     * deferred draft token is emitted instead of the residual, and the residual only fires below the gate or
     * without the window.
     */
    @Test
    public void flyDefersAHighEntropyRejectionBehindNativeFutureAcceptances() {
        double[] peakedAccept = array(0.0, 0.0, 1.0, 0.0);  // row 0 confirms token 2 regardless of the uniform
        double[] highEntropyZero = array(0.3, 0.1, 0.3, 0.3); // row 1: p(1) > 0, top-3 entropy ln 3
        double[] p2 = array(0.0, 0.0, 1.0, 0.0);            // row 2: token 2 natively accepted
        double[] p3 = array(0.0, 0.0, 0.0, 1.0);            // row 3: token 3 natively accepted
        int[] draft = {2, 1, 2, 3};
        // the residual weights of row 1 against the one-hot proposal of token 1
        double[] w1 = residualWeights(highEntropyZero, array(0.0, 1.0, 0.0, 0.0));

        FlyVerifier deferring = new FlyVerifier(2, 0.3, 3, false);
        ScriptRound deferral = round(deferring, flyRound(peakedAccept, highEntropyZero, p2, p3, draft),
                new double[]{0.5, 0.9, 0.5, 0.5});
        // four pre-pass uniforms, and the deferred token 1 emitted like a confirmation.
        assertEquals(4, deferral.uniformsUsed);
        assertEquals(4, deferral.confirmed);
        assertEquals(Arrays.asList(2, 1, 2, 3, 1), deferral.emitted);

        // below the entropy gate the rejection stands and the residual carries the replacement.
        // u1 below the acceptance ratio 0.1 confirms the block natively; above it the gate holds
        // and the residual is enumerated.
        FlyVerifier gated = new FlyVerifier(2, 2.0, 3, false);
        double[] accept1 = minRatios(highEntropyZero, array(0.0, 1.0, 0.0, 0.0));
        List<Run> runs = enumerate(gated, flyRound(peakedAccept, highEntropyZero, p2, p3, draft),
                thresholds(union(accept1, cdfBreakpoints(w1))));
        assertEq(0.9, measureConfirmed(runs, 1), 1e-9, "gated deferral leaves the rejection standing");
        assertEq(0.1, measureConfirmed(runs, 4), 1e-9, "native acceptance below the ratio confirms the block");
        double[] gatedLaw = emissionLaw(runs);
        assertEq(0.1, gatedLaw[1], 1e-9, "the confirmed block's bonus is the chain draw");
        for (int x : new int[]{0, 2, 3}) {
            assertEq(0.3, gatedLaw[x], 1e-9, "fly residual law for token " + x);
        }

        // without the window room the deferral never fires either (k = 2 leaves no room for W = 2).
        FlyVerifier narrow = new FlyVerifier(2, 0.3, 3, false);
        List<Run> shortRuns = enumerate(narrow, flyRoundShort(peakedAccept, highEntropyZero, draft),
                thresholds(union(accept1, cdfBreakpoints(w1))));
        assertEq(0.1, measureConfirmed(shortRuns, 2), 1e-9, "native acceptance confirms both");
        assertEq(0.9, measureConfirmed(shortRuns, 1), 1e-9, "no window, no deferral");
        double[] shortLaw = emissionLaw(shortRuns);
        assertEq(0.1, shortLaw[1], 1e-9, "the confirmed block's bonus is the chain draw");
        for (int x : new int[]{0, 2, 3}) {
            assertEq(0.3, shortLaw[x], 1e-9, "no-window residual law for token " + x);
        }
    }

    private static RoundMaker flyRound(final double[] row0, final double[] row1, final double[] row2,
                                       final double[] row3, final int[] draft) {
        return new RoundMaker() {
            @Override public ScriptRound round(double[] uniforms) {
                Verifier.Distribution bonus = distSelected(1, 0.0, 1.0, 0.0, 0.0);
                return new ScriptRound(new Verifier.Distribution[]{dist(row0), dist(row1), dist(row2), dist(row3),
                        bonus}, new Verifier.Distribution[]{oneHotDrafter(draft[0]), oneHotDrafter(draft[1]),
                        oneHotDrafter(draft[2]), oneHotDrafter(draft[3])}, draft, uniforms);
            }
        };
    }

    private static RoundMaker flyRoundShort(final double[] row0, final double[] row1, final int[] draft) {
        return new RoundMaker() {
            @Override public ScriptRound round(double[] uniforms) {
                Verifier.Distribution bonus = distSelected(1, 0.0, 1.0, 0.0, 0.0);
                return new ScriptRound(new Verifier.Distribution[]{dist(row0), dist(row1), bonus},
                        new Verifier.Distribution[]{oneHotDrafter(draft[0]), oneHotDrafter(draft[1])},
                        new int[]{draft[0], draft[1]}, uniforms);
            }
        };
    }

    private static Verifier.Distribution oneHotDrafter(int token) {
        double[] q = new double[VOCAB];
        q[token] = 1.0;
        return dist(q);
    }

    /** Greedy FLy uses target-only acceptance and defers on the softmax entropy of the logits. */
    @Test
    public void flyGreedyTargetOnlyAcceptanceDefersOnTheLogitEntropy() {
        // row 0: argmax 1, drafted 2 with a finite logit; row 1's argmax matches its draft.
        Verifier.Distribution[] targets = {greedyRow(1, 0.9, 2.0, 1.6, 0.5), greedyRow(3, 0.0, 0.0, 0.0, 2.0),
                greedyRow(0, 2.0, 0.0, 0.0, 0.0)};
        GreedyFlyRound greeter = new GreedyFlyRound(targets, new int[]{2, 3});
        ScriptRound deferral = round(new FlyVerifier(0, 0.3, 3, true), greeter, new double[0]);
        assertEquals(2, deferral.confirmed);
        assertEquals(Arrays.asList(2, 3, 0), deferral.emitted);
        assertEquals(0, deferral.uniformsUsed);

        // a low-entropy row decides against the deferral: the argmax replaces the draft.
        targets[0] = greedyRow(0, 5.0, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY);
        ScriptRound rejected = round(new FlyVerifier(0, 0.3, 3, true), new GreedyFlyRound(targets, new int[]{2, 3}),
                new double[]{0.5});
        assertEquals(0, rejected.confirmed);
        assertEquals(Arrays.asList(0), rejected.emitted);

        // a native (argmax) match never defers: it simply confirms.
        targets[0] = greedyRow(2, 0.0, 0.0, 2.0, 0.0);
        ScriptRound nativeAccept = round(new FlyVerifier(0, 0.3, 3, true),
                new GreedyFlyRound(targets, new int[]{2, 3}), new double[0]);
        assertEquals(2, nativeAccept.confirmed);
        assertEquals(Arrays.asList(2, 3, 0), nativeAccept.emitted);
    }
    private static final class GreedyFlyRound implements RoundMaker {
        private final Verifier.Distribution[] targets;
        private final int[] draft;

        GreedyFlyRound(Verifier.Distribution[] targets, int[] draft) {
            this.targets = targets;
            this.draft = draft;
        }

        @Override public ScriptRound round(double[] uniforms) {
            Verifier.Distribution[] drafts = {greedyRow(draft[0], 1.0, 0.0, 0.0, 0.0),
                    greedyRow(draft[1], 0.0, 0.0, 0.0, 1.0)};
            return new ScriptRound(targets, drafts, draft, uniforms);
        }
    }

    /** fly.py's {@code compute_fly_entropy} over probabilities and over logits, with the top-k cap. */
    @Test
    public void flyEntropyFollowsTheTopKEntropyOfProcessedProbabilities() {
        double h = FlyVerifier.entropy(dist(0.25, 0.25, 0.25, 0.25), 3, false);
        // fly.py sums p·ln p over the top k without renormalizing
        assertEq(-3.0 * 0.25 * Math.log(0.25), h, 1e-9, "uniform top-3 entropy");
        assertEq(0.0, FlyVerifier.entropy(arrayDist(1.0), 3, false), 1e-9, "peaked entropy");
        // the logit form recurrently normalizes over the whole support.
        Verifier.Distribution row = greedyRow(0, Math.log(0.5), Math.log(0.25), Math.log(0.25), Math.log(0.0) + 0);
        double fromLogits = FlyVerifier.entropy(row, 3, true);
        double expected = -(0.5 * Math.log(0.5) + 2 * 0.25 * Math.log(0.25));
        assertEq(expected, fromLogits, 1e-9, "logit-form top-3 entropy");
        assertEq(0.0, FlyVerifier.entropy(greedyRow(1, 5.0, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY,
                Float.NEGATIVE_INFINITY), 3, true), 1e-9, "spiky logits have no entropy");
    }

    private static Verifier.Distribution arrayDist(double... probs) {
        return distSelected(0, probs);
    }

    // ---------------------------------------------------------------- SYNTHETIC

    /** rates resolve from synthetic_acceptance_length with vLLM's minimum-variance schedule. */
    @Test
    public void syntheticAcceptanceLengthResolvesToTheMinimumVarianceRates() {
        assertArrayEquals(new double[]{1.0, 0.5, 0.0}, SyntheticVerifier.ratesForLength(2.5, 3), 0.0);
        assertArrayEquals(new double[]{0.75, 0.0, 0.0}, SyntheticVerifier.ratesForLength(1.75, 3), 0.0);
        assertArrayEquals(new double[]{0.0, 0.0, 0.0}, SyntheticVerifier.ratesForLength(1.0, 3), 0.0);
        assertArrayEquals(new double[]{1.0, 1.0, 1.0}, SyntheticVerifier.ratesForLength(4.0, 3), 0.0);
        assertThrows(IllegalArgumentException.class, () -> SyntheticVerifier.ratesForLength(0.0, 3));
        assertThrows(IllegalArgumentException.class, () -> SyntheticVerifier.ratesForLength(5.0, 3));
    }

    @Test
    public void syntheticVerifierValidatesItsRates() {
        SyntheticVerifier verifier = new SyntheticVerifier(new double[]{1.0, 0.5, 0.0});
        assertArrayEquals(new double[]{1.0, 0.5, 0.0}, verifier.rates(), 0.0);
        assertTrue(verifier.toString().contains("rates=[1.0, 0.5, 0.0]"));
        assertThrows(IllegalArgumentException.class, () -> new SyntheticVerifier(new double[0]));
        assertThrows(IllegalArgumentException.class, () -> new SyntheticVerifier(null));
        assertThrows(IllegalArgumentException.class, () -> new SyntheticVerifier(new double[]{0.5, 1.0}));
        assertThrows(IllegalArgumentException.class, () -> new SyntheticVerifier(new double[]{-0.1}));
        assertThrows(IllegalArgumentException.class, () -> new SyntheticVerifier(new double[]{1.1}));
    }

    /** The unconditional per-position rates: P(accepted >= j) exactly rates[j-1], with the chain-draw tokens. */
    @Test
    public void syntheticVerificationRealizesTheConfiguredRates() {
        SyntheticVerifier verifier = new SyntheticVerifier(new double[]{1.0, 0.5});
        List<Run> runs = enumerate(verifier, new RoundMaker() {
            @Override public ScriptRound round(double[] uniforms) {
                return new ScriptRound(
                        new Verifier.Distribution[]{dist(0.4, 0.3, 0.2, 0.1), dist(0.05, 0.25, 0.5, 0.2),
                                distSelected(3, 1.0, 0.0, 0.0, 0.0)},
                        new Verifier.Distribution[]{oneHotDrafter(2), oneHotDrafter(1)},
                        new int[]{2, 1}, uniforms);
            }
        }, new double[0], new double[]{0.5});
        assertEq(0.5, measureConfirmed(runs, 2), 1e-9, "first rate of 1.0 always accepts");
        assertEq(0.5, measureConfirmed(runs, 1), 1e-9, "second rate of 0.5 halves the block");
        // a rejection and the bonus both emit the chain's own draw, so the emission law stays the target's.
        double[] law = emissionLaw(runs);
        double[] expected = new double[VOCAB];
        expected[3] = 0.5;                                  // bonus after 2 confirms
        // rejects at position 1 draw target row 1's chain token (its argmax: token 2).
        expected[2] = 0.5;
        for (int x = 0; x < VOCAB; x++) {
            assertEq(expected[x], law[x], 1e-9, "synthetic emission law for token " + x);
        }
    }

    // ---------------------------------------------------------------- decoder fixtures

    private static GGUFTokenizer tokenizer() {
        List<Object> tokens = new ArrayList<>(Arrays.<Object>asList("a", "b", "c", "d"));
        List<Object> types = new ArrayList<>(Arrays.<Object>asList(1, 1, 1, 1));
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "gpt2");
        md.put(TokenizerVocab.KEY_PRE, "gpt-2");
        md.put(TokenizerVocab.KEY_TOKENS, tokens);
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
        md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>());
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    private static SamplerConfig sampling() {
        return SamplerConfig.builder().temperature(1f).topK(0).topP(1f).minP(0f).seed(SEED).build();
    }

    private static GenerationRequest request(int maxNewTokens, SamplerConfig config) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(config).build();
    }

    /** KV-only fake memory with a plain sequence record. */
    private static final class FakeMemory implements ModelMemory {
        private final MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, N_CTX, UBATCH)
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.NONE).build())
                .build();
        final SequenceState sequence = new SequenceState(N_CTX);
        int snapshots;
        int restores;
        boolean recurrent;
        private int saved = -1;

        @Override public MemoryPlan plan() { return plan; }
        @Override public SequenceState sequence() { return sequence; }
        @Override public void reset() {
            sequence.clear();
            saved = -1;
        }
        @Override public boolean canRollback(int nTokens) { return !recurrent && nTokens <= sequence.nPast(); }
        @Override public void rollback(int nTokens) { sequence.truncate(nTokens); }
        @Override public boolean hasRecurrentState() { return recurrent; }
        @Override public boolean canSnapshot() { return recurrent; }
        @Override public boolean ringsKeepWindow(int nWrite, int nDrop) { return true; }
        @Override public void snapshot() {
            saved = sequence.nPast();
            snapshots++;
        }
        @Override public void restore() {
            if (saved < 0) {
                throw new IllegalStateException("no snapshot");
            }
            sequence.truncate(sequence.nPast() - saved);
            restores++;
        }
        @Override public void close() { }
    }

    /** Target backend scripted per forward call, in call order; each entry services one call's rows. */
    private static final class FakeTarget implements ForwardBackend {
        final FakeMemory memory = new FakeMemory();
        private final Deque<float[][]> scripts = new ArrayDeque<>();
        final List<int[]> verifyCalls = new ArrayList<>();
        final List<Integer> verifyPositions = new ArrayList<>();
        final List<Integer> rollbacks = new ArrayList<>();
        final List<int[]> prefills = new ArrayList<>();
        final List<Integer> prefillPositions = new ArrayList<>();
        final List<Integer> decodes = new ArrayList<>();
        int snapshots;
        int restores;

        FakeTarget(boolean recurrent) {
            memory.recurrent = recurrent;
        }

        FakeTarget add(float[]... rows) {
            scripts.add(rows);
            return this;
        }

        private float[][] nextRow(int rowsWanted) {
            float[][] script = scripts.poll();
            if (script == null) {
                throw new IllegalStateException("no scripted rows left for a forward call");
            }
            if (script.length != rowsWanted) {
                throw new IllegalStateException("scripted " + script.length + " rows, wanted " + rowsWanted);
            }
            return script;
        }

        @Override public float[] prefill(int[] tokens, int pos0) {
            if (pos0 != memory.sequence.nPast()) {
                throw new IllegalStateException("prefill at " + pos0 + " after " + memory.sequence.nPast());
            }
            memory.sequence.append(tokens, 0, tokens.length);
            prefills.add(tokens.clone());
            prefillPositions.add(pos0);
            return nextRow(1)[0];
        }

        @Override public float[] decode(int token, int pos) {
            memory.sequence.append(new int[]{token}, 0, 1);
            decodes.add(token);
            return nextRow(1)[0];
        }

        @Override public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
            if (pos0 != memory.sequence.nPast()) {
                throw new IllegalStateException("verify at " + pos0 + " after " + memory.sequence.nPast());
            }
            memory.sequence.append(tokens, 0, tokens.length);
            verifyCalls.add(tokens.clone());
            verifyPositions.add(pos0);
            return nextRow(tokens.length);
        }

        @Override public float[][] forwardAll(int[] tokens, int[] outputRows) {
            throw new UnsupportedOperationException();
        }

        @Override public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
            throw new UnsupportedOperationException();
        }

        @Override public void reset() {
            memory.reset();
        }

        @Override public boolean canRollback(int nTokens) {
            return memory.canRollback(nTokens);
        }

        @Override public void rollback(int nTokens) {
            memory.rollback(nTokens);
            rollbacks.add(nTokens);
        }

        @Override public void shiftContext(int nKeep, int nDiscard) {
            throw new UnsupportedOperationException();
        }

        @Override public ModelMemory memory() {
            FakeMemory memory = this.memory;
            return new ModelMemory() {
                @Override public MemoryPlan plan() { return memory.plan(); }
                @Override public SequenceState sequence() { return memory.sequence(); }
                @Override public void reset() { memory.reset(); }
                @Override public boolean canRollback(int nTokens) { return memory.canRollback(nTokens); }
                @Override public void rollback(int nTokens) { memory.rollback(nTokens); }
                @Override public boolean hasRecurrentState() { return memory.hasRecurrentState(); }
                @Override public boolean canSnapshot() { return memory.canSnapshot(); }
                @Override public boolean ringsKeepWindow(int nWrite, int nDrop) {
                    return memory.ringsKeepWindow(nWrite, nDrop);
                }
                @Override public void snapshot() { snapshots++; memory.snapshot(); }
                @Override public void restore() { restores++; memory.restore(); }
                @Override public void close() { }
            };
        }

        @Override public void setHiddenTap(HiddenTap tap) {
        }

        @Override public void close() {
        }
    }

    /** A scripted drafter with per-position proposal distributions. */
    private static final class ProbDrafter implements DraftModel, DraftProbs {
        final List<int[]> proposals = new ArrayList<>();
        final List<double[][]> qRows = new ArrayList<>();
        int call;
        double faultyAt;
        boolean faultyZero;
        boolean faultyNull;
        boolean faultyShort;
        boolean faultyNaN;

        ProbDrafter(int[]... tokens) {
            for (int[] t : tokens) {
                proposals.add(t);
            }
        }

        static ProbDrafter with(int[] tokens, double[]... q) {
            ProbDrafter d = new ProbDrafter(tokens);
            d.qRows.add(q);
            return d;
        }

        @Override public void bindTarget(EngineModel target) {
        }

        @Override public int[] draft(int lastToken, int n) {
            if (call >= proposals.size()) {
                return new int[0];
            }
            return proposals.get(call++);
        }

        @Override public float[] draftProbs(int position) {
            double[][] rows = qRows.get(call - 1);
            if (faultyNull && position == 0) {
                return null;
            }
            if (faultyShort) {
                // a row shorter than the drafted token id
                return new float[position + 1];
            }
            if (faultyZero) {
                // a full row whose entry for the drafted token is 0
                return new float[]{0.1f, 0.2f, 0.0f, 0.4f};
            }
            if (faultyNaN) {
                return new float[]{0.1f, 0.2f, Float.NaN, 0.4f};
            }
            if (rows == null || position >= rows.length) {
                throw new IllegalStateException("no proposal probabilities at position " + position);
            }
            double[] row = rows[position];
            float[] out = new float[row.length];
            for (int i = 0; i < row.length; i++) {
                out[i] = (float) row[i];
            }
            return out;
        }

        @Override public void accept(int nAccepted) {
        }

        @Override public void reset() {
            call = 0;
        }
    }

    /** A plain drafter without probabilities (one-hot {@code q}); repeats its last proposal. */
    private static final class PlainDrafter implements DraftModel {
        final List<int[]> proposals = new ArrayList<>();
        int call;

        PlainDrafter(int[]... tokens) {
            for (int[] t : tokens) {
                proposals.add(t);
            }
        }

        @Override public void bindTarget(EngineModel target) {
        }

        @Override public int[] draft(int lastToken, int n) {
            if (call >= proposals.size()) {
                return new int[0];
            }
            return proposals.get(call++);
        }

        @Override public void accept(int nAccepted) {
        }

        @Override public void reset() {
            call = 0;
        }
    }

    // ---------------------------------------------------------------- decoder traces: REJECTION

    /** One-hot target rows and a plain drafter run rejection sampling deterministically on KV memory. */
    @Test
    public void rejectionSamplingConfirmsAndRescoresOnKvMemory() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                // the verify scores the whole [1, 2, 3] batch: row 2 is never read because the
                // rejection at position 1 ends the round before it
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(1.0, 0.0, 0.0, 0.0), logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(1.0, 0.0, 0.0, 0.0));
        List<SpeculativeDecoder.Stats> stats = new ArrayList<>();
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.REJECTION).statsSink(stats::add).build();

        GenerationResult r = dec.generate(new int[]{0}, request(4, sampling()), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 0, 0}, r.tokens());
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertEquals(1, stats.size());
        assertEquals(1, stats.get(0).rounds());
        assertEquals(2, stats.get(0).drafted());
        assertEquals(1, stats.get(0).accepted());
        assertArrayEquals(new int[]{1, 2, 3}, target.verifyCalls.get(0));
        assertEquals(Arrays.asList(1), target.rollbacks);
        assertEquals(4, target.memory().sequence().nPast());
    }

    /** A stop inside the verify rebuilds the KV memory: the rejected positions are rolled back. */
    @Test
    public void stopInsideVerifyRollsBackTheUnconfirmedPositionsOnKvMemory() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                // the third row is never read: the replacement at position 1 spends the budget
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(1.0, 0.0, 0.0, 0.0), logits(0.0, 1.0, 0.0, 0.0));
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.REJECTION).build();

        GenerationResult r = dec.generate(new int[]{0}, request(3, sampling()), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 0}, r.tokens());
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertEquals(Arrays.asList(1), target.rollbacks);
        assertEquals(3, target.memory().sequence().nPast());
    }

    /** A fully confirmed block that ends the budget drops nothing. */
    @Test
    public void fullConfirmationInsideTheBudgetLeavesNoRejectedPositions() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(0.0, 0.0, 0.0, 1.0), logits(1.0, 0.0, 0.0, 0.0));
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.REJECTION).build();

        GenerationResult r = dec.generate(new int[]{0}, request(4, sampling()), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 3, 0}, r.tokens());
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertTrue(target.rollbacks.isEmpty());
        assertEquals(4, target.memory().sequence().nPast());
    }

    /** Recurrent memory restores its snapshot and replays the kept tokens, after a stop like after a rejection. */
    @Test
    public void stopInsideVerifyRestoresAndReplaysOnRecurrentMemory() {
        FakeTarget target = new FakeTarget(true)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                // the third row is never read: the replacement at position 1 spends the budget
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(1.0, 0.0, 0.0, 0.0), logits(0.0, 1.0, 0.0, 0.0))
                // the replay prefill of the kept tokens reads one row of its own
                .add(logits(0.0, 1.0, 0.0, 0.0));
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.REJECTION).build();

        GenerationResult r = dec.generate(new int[]{0}, request(3, sampling()), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 0}, r.tokens());
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertEquals(1, target.snapshots);
        assertEquals(1, target.restores);
        assertTrue(target.rollbacks.isEmpty());
        assertEquals(2, target.prefills.size());
        assertArrayEquals(new int[]{0}, target.prefills.get(0));
        assertArrayEquals(new int[]{1, 2}, target.prefills.get(1));
        assertEquals(1, target.prefillPositions.get(1).intValue());
        assertEquals(3, target.memory().sequence().nPast());
    }

    /** A block confirmed up to the budget needs no restore either. */
    @Test
    public void fullConfirmationInsideTheBudgetNeedsNoRestoreOnRecurrentMemory() {
        FakeTarget target = new FakeTarget(true)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(0.0, 0.0, 0.0, 1.0), logits(1.0, 0.0, 0.0, 0.0));
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.REJECTION).build();

        GenerationResult r = dec.generate(new int[]{0}, request(4, sampling()), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 3, 0}, r.tokens());
        assertEquals(1, target.snapshots);
        assertEquals(0, target.restores);
        assertEquals(4, target.memory().sequence().nPast());
    }

    // ---------------------------------------------------------------- decoder traces: BLOCK

    /** Block verification on one-hot rows: the block's backward pass recovers the argmax prefix and replacement. */
    @Test
    public void blockVerificationConfirmsThePrefixAndRescoresFromItsResidual() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                // the third row is never read: the block's residual at position 1 ends the round
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(1.0, 0.0, 0.0, 0.0), logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(1.0, 0.0, 0.0, 0.0));
        List<SpeculativeDecoder.Stats> stats = new ArrayList<>();
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.BLOCK).statsSink(stats::add).build();

        GenerationResult r = dec.generate(new int[]{0}, request(4, sampling()), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 0, 0}, r.tokens());
        assertEquals(1, stats.size());
        assertEquals(2, stats.get(0).drafted());
        assertArrayEquals(new int[]{1, 2, 3}, target.verifyCalls.get(0));
        assertEquals(Arrays.asList(1), target.rollbacks);
    }

    // ---------------------------------------------------------------- greedy fallback

    /**
     * A greedy target chain degenerates REJECTION, BLOCK and FLY deferrals alike to the argmax rule: each mode's
     * trace equals MATCH's (llama.cpp common/sampling.cpp target-sample-match).
     */
    @Test
    public void greedyTargetChainsFallBackToTheArgmaxMatchRule() {
        int[][] drafts = {{2, 3}};
        float[][][] rows = {
                // the third row is never read: the mismatch at position 1 ends every mode's round
                {logits(0.0, 0.0, 1.0, 0.0), logits(1.0, 0.0, 0.0, 0.0), logits(0.0, 1.0, 0.0, 0.0)},
        };
        String[] expected = genTrace(VerifyMode.MATCH, drafts, rows);
        assertArrayEquals(expected, genTrace(VerifyMode.REJECTION, drafts, rows));
        assertArrayEquals(expected, genTrace(VerifyMode.BLOCK, drafts, rows));
        assertArrayEquals(expected, genTrace(VerifyMode.FLY, drafts, rows));
    }

    private static String[] genTrace(VerifyMode mode, int[][] drafts, float[][][] verifyRows) {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                .add(verifyRows[0]);
        SpeculativeDecoder.Builder builder = SpeculativeDecoder.builder(target, new PlainDrafter(drafts),
                tokenizer()).draftMax(2).verifyMode(mode);
        if (mode == VerifyMode.FLY) {
            builder.flyEntropyThreshold(0.3);
        }
        GenerationResult r = builder.build().generate(new int[]{0}, request(3, SamplerConfig.greedyDecoding()),
                (TokenCallback) null);
        return new String[]{Arrays.toString(r.tokens()), Arrays.toString(target.rollbacks.toArray())};
    }

    // ---------------------------------------------------------------- decoder traces: FLy

    /** Greedy FLy defers a mismatched draft when the entropy gate and the future argmaxes allow it. */
    @Test
    public void flyGreedyDeferralReplacesTheArgmaxOnAGatedMismatch() {
        // prefill argmax 1; row0 argmax 1 != d0 = 2 (deferral), row1 argmax 3 == d1, bonus row argmax 0.
        FakeTarget deferred = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(0.9, 2.0, 1.6, 0.5), logits(0.0, 0.0, 0.0, 1.0), logits(1.0, 0.0, 0.0, 0.0));
        SpeculativeDecoder fly = SpeculativeDecoder.builder(deferred, new PlainDrafter(new int[]{2, 3}),
                tokenizer()).draftMax(2).verifyMode(VerifyMode.FLY).build();
        GenerationResult r = fly.generate(new int[]{0}, request(4, SamplerConfig.greedyDecoding()),
                (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 3, 0}, r.tokens());
        assertTrue("the deferred round drops nothing", deferred.rollbacks.isEmpty());

        // with the gate closed the deferral stands down and the replacement is the argmax.
        FakeTarget gated = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(0.0, 5.0, 0.0, 0.0), logits(0.0, 0.0, 0.0, 1.0), logits(1.0, 0.0, 0.0, 0.0));
        SpeculativeDecoder off = SpeculativeDecoder.builder(gated, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.FLY).flyEntropyThreshold(2.0).build();
        GenerationResult plain = off.generate(new int[]{0}, request(2, SamplerConfig.greedyDecoding()),
                (TokenCallback) null);
        assertArrayEquals(new int[]{1, 1}, plain.tokens());
        assertEquals(StopReason.MAX_TOKENS, plain.stopReason());
        assertEquals(Arrays.asList(2), gated.rollbacks);
    }

    /** Non-greedy FLy deferral with probabilistic drafter probabilities is exercised at the verifier level; here
     * the one-hot drafter case confirms the pre-pass stream through the decoder. */
    @Test
    public void flyNonGreedyDeferralRunsThroughTheDecoder() {
        // row0 confirms d0; row1 natively rejects d1 (u = 0.16 > p(1) = 0.14) and defers behind rows 2-3.
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(0.28, 0.14, 0.29, 0.29),
                        logits(0.0, 0.0, 1.0, 0.0), logits(0.0, 0.0, 0.0, 1.0), logits(0.0, 1.0, 0.0, 0.0));
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 1, 2, 3}),
                tokenizer()).draftMax(4).verifyMode(VerifyMode.FLY).flyWindow(2).build();
        GenerationResult r = dec.generate(new int[]{0}, request(5, sampling()), (TokenCallback) null);
        // 1, then the whole deferred block 2, 1, 2, 3: the budget ends the round at the last confirmation.
        assertArrayEquals(new int[]{1, 2, 1, 2, 3}, r.tokens());
        assertTrue(target.rollbacks.isEmpty());
        assertEquals(6, target.memory().sequence().nPast());
    }

    // ---------------------------------------------------------------- drafter probability contract

    private static SpeculativeDecoder probDecoder(FakeTarget target, ProbDrafter drafter) {
        return SpeculativeDecoder.builder(target, drafter, tokenizer())
                .draftMax(1).verifyMode(VerifyMode.REJECTION).build();
    }

    @Test
    public void probabilisticDraftersSupplyTheProposalDistributions() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                // the verify scores [token, draft]: the second row is the bonus row, never read because
                // the confirmation spends the budget
                .add(logits(0.3, 0.0, 0.65, 0.05), logits(0.3, 0.0, 0.65, 0.05));
        ProbDrafter drafter = ProbDrafter.with(new int[]{2}, new double[]{0.1, 0.2, 0.3, 0.4});
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, drafter, tokenizer())
                .draftMax(1).verifyMode(VerifyMode.REJECTION).build();
        // p(d)/q(d) = 0.65/0.3 > 1: the draft token is a sure confirmation and the bonus row follows.
        List<SpeculativeDecoder.Stats> stats = new ArrayList<>();
        SpeculativeDecoder sure = SpeculativeDecoder.builder(target, drafter, tokenizer())
                .draftMax(1).verifyMode(VerifyMode.REJECTION).statsSink(stats::add).build();
        sure.generate(new int[]{0}, request(2, sampling()), (TokenCallback) null);
        // round: p(2) = 0.65 >= q(2) = 0.3 -> confirmed; bonus = row 1's chain draw (argmax 2).
        assertEquals(1, stats.get(0).accepted());
        assertEquals(2, sure.reasoning() != null ? 2 : 2);
    }

    /** The decoder refuses rows that do not cover the drafted token with a positive probability. */
    @Test
    public void probabilisticDrafterFaultsFailTheGeneration() {
        FaultyTargetless f0 = drafterFault(false, true, false, false);
        assertTrue(f0.message().contains("no proposal probabilities"));
        FaultyTargetless f1 = drafterFault(true, false, false, false);
        assertTrue(f1.message().contains("proposal probability 0"));
        FaultyTargetless f2 = drafterFault(false, false, true, false);
        assertTrue(f2.message().contains("but offered"));
        FaultyTargetless f3 = drafterFault(false, false, false, true);
        assertTrue(f3.message().contains("NaN"));
    }

    private static final class FaultyTargetless {
        private final String message;

        FaultyTargetless(String message) {
            this.message = message;
        }

        String message() {
            return message;
        }
    }

    private static FaultyTargetless drafterFault(boolean zero, boolean nullRow, boolean shortRow, boolean nan) {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                // the verify scores [token, draft]; the fault fires before the bonus row is read
                .add(logits(0.3, 0.0, 0.65, 0.05), logits(1.0, 0.0, 0.0, 0.0));
        ProbDrafter drafter = ProbDrafter.with(new int[]{2}, new double[]{0.1, 0.2, 0.3, 0.4});
        drafter.faultyZero = zero;
        drafter.faultyNull = nullRow;
        drafter.faultyShort = shortRow;
        drafter.faultyNaN = nan;
        SpeculativeDecoder dec = probDecoder(target, drafter);
        IllegalStateException e;
        try {
            dec.generate(new int[]{0}, request(2, sampling()), (TokenCallback) null);
            throw new AssertionError("a faulty probability row must fail the generation");
        } catch (IllegalStateException caught) {
            e = caught;
        }
        return new FaultyTargetless(e.getMessage());
    }

    // ---------------------------------------------------------------- SYNTHETIC through the decoder

    @Test
    public void syntheticModeIsGatedBehindTheSystemProperty() {
        FakeTarget target = new FakeTarget(false);
        SpeculativeDecoder.Builder builder = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2}),
                tokenizer()).draftMax(1).verifyMode(VerifyMode.SYNTHETIC).syntheticAcceptanceRates(0.5);
        IllegalStateException without = assertThrows(IllegalStateException.class, builder::build);
        assertTrue(without.getMessage().contains("neuralfree.spec.synthetic"));
        System.setProperty(SyntheticVerifier.PROP_ENABLED, "true");
        try {
            SpeculativeDecoder dec = builder.build();
            assertEquals(VerifyMode.SYNTHETIC, dec.verifyMode());
            // synthetic mode without rates or a length is refused even when enabled
            assertThrows(IllegalArgumentException.class, () -> SpeculativeDecoder
                    .builder(target, new PlainDrafter(new int[]{2}), tokenizer()).draftMax(1)
                    .verifyMode(VerifyMode.SYNTHETIC).build());
            // a synthetic setting outside synthetic mode is refused
            assertThrows(IllegalArgumentException.class, () -> SpeculativeDecoder
                    .builder(target, new PlainDrafter(new int[]{2}), tokenizer()).draftMax(1)
                    .verifyMode(VerifyMode.REJECTION).syntheticAcceptanceRates(0.5).build());
            assertThrows(IllegalArgumentException.class, () -> SpeculativeDecoder
                    .builder(target, new PlainDrafter(new int[]{2}), tokenizer()).draftMax(2)
                    .verifyMode(VerifyMode.REJECTION).syntheticAcceptanceRates(0.5)
                    .syntheticAcceptanceLength(1.5).build());
        } finally {
            System.clearProperty(SyntheticVerifier.PROP_ENABLED);
        }
    }

    /** Synthetic verification realizes the rates' acceptance law round by round (one uniform per round). */
    @Test
    public void syntheticDecoderRunsTheRateLaw() {
        FakeTarget target = new FakeTarget(false);
        target.add(logits(0.0, 1.0, 0.0, 0.0));
        // three rounds fit the budget; every round confirms iff its uniform <= 0.5, tokens come from one-hot rows.
        for (int round = 0; round < 4; round++) {
            target.add(logits(0.0, 0.0, 0.0, 1.0), logits(0.0, 0.0, 0.0, 1.0));
        }
        System.setProperty(SyntheticVerifier.PROP_ENABLED, "true");
        try {
            List<SpeculativeDecoder.Stats> stats = new ArrayList<>();
            SpeculativeDecoder dec = SpeculativeDecoder.builder(target,
                    new PlainDrafter(new int[]{3}, new int[]{3}, new int[]{3}, new int[]{3}),
                    tokenizer()).draftMax(1).verifyMode(VerifyMode.SYNTHETIC).syntheticAcceptanceRates(0.5)
                    .statsSink(stats::add).build();
            GenerationResult r = dec.generate(new int[]{0}, request(5, sampling()), (TokenCallback) null);
            assertArrayEquals(new int[]{1, 3, 3, 3, 3}, r.tokens());
            // a confirmed round also emits the bonus row's chain draw, so the budget ends the third round
            SplittableRandom expectedStream = new SplittableRandom(SEED);
            int expectedAccepted = 0;
            for (int round = 0; round < 3; round++) {
                if (expectedStream.nextDouble() <= 0.5) {
                    expectedAccepted++;
                }
            }
            assertEquals(3, stats.get(0).rounds());
            assertEquals(3, stats.get(0).drafted());
            assertEquals(expectedAccepted, stats.get(0).accepted());
        } finally {
            System.clearProperty(SyntheticVerifier.PROP_ENABLED);
        }
    }

    // ---------------------------------------------------------------- configuration surface

    @Test
    public void verifyModeParsesTheModeNames() {
        assertEquals(VerifyMode.MATCH, VerifyMode.parse(null));
        assertEquals(VerifyMode.MATCH, VerifyMode.parse("match"));
        assertEquals(VerifyMode.REJECTION, VerifyMode.parse("Rejection"));
        assertEquals(VerifyMode.BLOCK, VerifyMode.parse("block"));
        assertEquals(VerifyMode.FLY, VerifyMode.parse(" fly "));
        assertEquals(VerifyMode.SYNTHETIC, VerifyMode.parse("SYNTHETIC"));
        assertEquals("rejection", VerifyMode.REJECTION.toString());
        IllegalArgumentException unknown = assertThrows(IllegalArgumentException.class,
                () -> VerifyMode.parse("strict"));
        assertTrue(unknown.getMessage().contains("match"));
    }

    @Test
    public void builderValidatesTheVerifySettings() {
        FakeTarget target = new FakeTarget(false);
        SpeculativeDecoder.Builder base = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2}),
                tokenizer()).draftMax(2);
        SpeculativeDecoder dec = base.build();
        assertEquals(VerifyMode.MATCH, dec.verifyMode());
        assertTrue(dec.toString().contains("verify=match"));
        assertEquals(VerifyMode.REJECTION,
                SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2}), tokenizer())
                        .draftMax(2).verifyMode(VerifyMode.REJECTION).build().verifyMode());
        assertThrows(IllegalArgumentException.class, () -> SpeculativeDecoder
                .builder(target, new PlainDrafter(new int[]{2}), tokenizer()).draftMax(2)
                .verifyMode(SyntheticVerifierEnabled.nothing()).build());
        assertThrows(IllegalArgumentException.class, () -> base.verifyMode(null));
        assertThrows(IllegalArgumentException.class, () -> base.flyWindow(-1));
        assertThrows(IllegalArgumentException.class, () -> base.flyEntropyThreshold(-0.1));
        assertThrows(IllegalArgumentException.class, () -> base.flyEntropyTopK(0));
        assertThrows(IllegalArgumentException.class, () -> base.syntheticAcceptanceRates(1.0, 0.5, 0.6));
        assertThrows(IllegalArgumentException.class, () -> base.syntheticAcceptanceRates(1.2));
        assertThrows(IllegalArgumentException.class, () -> base.syntheticAcceptanceLength(0.5));
    }

    static final class SyntheticVerifierEnabled {
        private SyntheticVerifierEnabled() {
        }

        static VerifyMode nothing() {
            return null;
        }
    }
}
