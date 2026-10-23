package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;

import java.util.Arrays;
import java.util.Objects;

/**
 * Candidate lattice of the DFlash2 selector and the LiLiCorr head, and the greedy walk over it.
 *
 * <p>Row {@code p} of a lattice (block position {@code p >= 1}; row 0 is unused) holds {@code topK} candidate
 * tokens and {@code topK * topK} transition scores, {@code score(p, j, i)} for successor candidate {@code j} at
 * position {@code p} after predecessor candidate {@code i} at {@code p - 1}. This is the packed {@code t_h_nextn}
 * row {@code [cand (topK) | scores (topK*topK)]} of dflash.cpp:611-614 and :794-797.
 *
 * <p>DFlash2 scores (dflash.cpp:548-639): with candidates {@code cand(p)} the {@code topK} largest logits of
 * position {@code p}, {@code gate(p) = selector_hidden . t_embd(p)} and predecessor ids {@code pred(p)} (the anchor
 * token at {@code p = 1}, else {@code cand(p - 1)}):
 * <pre>
 *   score(p, j, i) = sum_r next[cand(p)_j][r] * prev[pred(p)_i][r] * gate(p)[r] + logit(p)[cand(p)_j]
 * </pre>
 * The walk (common/speculative.cpp:1291-1320) starts from predecessor 0 and at each row takes the first maximum
 * of {@code score(p, ., predecessor)}; with {@code pMin > 0} it stops when the softmax of that row's scores at the
 * maximum, {@code 1 / sum_k exp(s_k - s_max)}, is below {@code pMin}.
 */
public final class DflashSelector {

    private DflashSelector() {
    }

    /** One block's lattice. Rows are filled by their producer before the walk; row 0 stays unused. */
    public static final class Lattice {
        private final int topK;
        private final int nRows;
        private final int[] cand;
        private final float[] score;

        /**
         * @throws IllegalArgumentException for a non-positive candidate count or a row count below 1
         */
        public Lattice(int topK, int nRows) {
            if (topK <= 0 || nRows < 1) {
                throw new IllegalArgumentException("lattice needs topK > 0 and at least one row, got " + topK + "/"
                        + nRows);
            }
            this.topK = topK;
            this.nRows = nRows;
            this.cand = new int[nRows * topK];
            this.score = new float[nRows * topK * topK];
        }

        public int topK() { return topK; }
        public int nRows() { return nRows; }

        /** Candidate {@code j} of row {@code p}. */
        public int candidate(int p, int j) {
            return cand[p * topK + j];
        }

        /** {@code score(p, j, i)}: successor {@code j} after predecessor {@code i}. */
        public float score(int p, int j, int i) {
            return score[(p * topK + i) * topK + j];
        }

        void setCandidate(int p, int j, int token) {
            cand[p * topK + j] = token;
        }

        void setScore(int p, int j, int i, float v) {
            score[(p * topK + i) * topK + j] = v;
        }

        /** Offset of the {@code topK} successor scores after predecessor {@code i} in row {@code p}. */
        int scoreRow(int p, int i) {
            return (p * topK + i) * topK;
        }

        float[] scores() {
            return score;
        }
    }

    /**
     * Greedy walk over rows {@code 1 .. nSteps} of {@code lattice} (speculative.cpp:1291-1320).
     *
     * @param nSteps rows to walk; at most {@code lattice.nRows() - 1}
     * @param pMin   stop threshold on the step probability; 0 disables it
     * @return the drafted tokens, at most {@code nSteps}
     * @throws IllegalArgumentException when {@code nSteps} exceeds the lattice
     */
    public static int[] walk(Lattice lattice, int nSteps, float pMin) {
        Objects.requireNonNull(lattice, "lattice");
        if (nSteps < 0 || nSteps > lattice.nRows() - 1) {
            throw new IllegalArgumentException("walk of " + nSteps + " steps exceeds the " + (lattice.nRows() - 1)
                    + " scored rows");
        }
        final int k = lattice.topK();
        final float[] s = lattice.scores();
        int[] out = new int[nSteps];
        int n = 0;
        int predecessor = 0;
        for (int p = 1; p <= nSteps; p++) {
            final int row = lattice.scoreRow(p, predecessor);
            predecessor = firstMax(s, row, k);
            if (pMin > 0.0f) {
                float sum = 0.0f;
                for (int j = 0; j < k; j++) {
                    sum += (float) Math.exp(s[row + j] - s[row + predecessor]);
                }
                if (1.0f / sum < pMin) {
                    break;
                }
            }
            out[n++] = lattice.candidate(p, predecessor);
        }
        return n == nSteps ? out : Arrays.copyOf(out, n);
    }

    /**
     * DFlash2 lattice of one block of {@code nTokens} positions (dflash.cpp:567-638): rows
     * {@code 1 .. min(nTokens, graphBlockSize) - 1} are scored.
     *
     * @param logits   target-vocabulary logits {@code [nTokens][nVocab]} after the logit transforms and the
     *                 {@code d2t} scatter
     * @param tEmbd    normalized hidden rows {@code [nTokens][nEmbd]}
     * @param anchor   token of block position 0 (the last sampled target token)
     * @param prev     {@code selector_predecessor} ({@code n_vocab} rows of {@code rank})
     * @param next     {@code selector_successor} ({@code n_vocab} rows of {@code rank})
     * @param hidden   {@code selector_hidden} ({@code rank} rows of {@code nEmbd})
     * @throws IllegalArgumentException when a shape is inconsistent
     */
    static Lattice dflash2(float[] logits, int nVocab, float[] tEmbd, int nEmbd, int nTokens, int anchor,
                           int topK, int graphBlockSize, DflashMatrix prev, DflashMatrix next, DflashMatrix hidden) {
        final int blockLen = Math.min(nTokens, graphBlockSize);
        final int rank = hidden.rows();
        if (prev.cols() != rank || next.cols() != rank || hidden.cols() != nEmbd) {
            throw new IllegalArgumentException("DFlash2 selector tensors disagree on rank " + rank + " / width "
                    + nEmbd + ": " + prev + ", " + next + ", " + hidden);
        }
        Lattice lattice = new Lattice(topK, Math.max(1, blockLen));
        int[] candPrev = new int[topK];
        int[] candCur = new int[topK];
        float[] gate = new float[rank];
        float[] succ = new float[topK * rank];
        float[] cond = new float[topK * rank];
        float[] predRow = new float[rank];
        for (int p = 1; p < blockLen; p++) {
            final int lOff = p * nVocab;
            CpuOps.topK(logits, lOff, nVocab, topK, candCur, 0);
            hidden.matvec(tEmbd, p * nEmbd, gate, 0);
            for (int j = 0; j < topK; j++) {
                lattice.setCandidate(p, j, candCur[j]);
                next.row(candCur[j], succ, j * rank);
            }
            final int nPred = p == 1 ? 1 : topK;
            for (int i = 0; i < nPred; i++) {
                prev.row(p == 1 ? anchor : candPrev[i], predRow, 0);
                for (int r = 0; r < rank; r++) {
                    cond[i * rank + r] = predRow[r] * gate[r];
                }
            }
            for (int i = 0; i < topK; i++) {
                final int ci = (nPred == 1 ? 0 : i) * rank;
                for (int j = 0; j < topK; j++) {
                    float dot = 0.0f;
                    final int sj = j * rank;
                    for (int r = 0; r < rank; r++) {
                        dot += succ[sj + r] * cond[ci + r];
                    }
                    lattice.setScore(p, j, i, dot + logits[lOff + candCur[j]]);
                }
            }
            System.arraycopy(candCur, 0, candPrev, 0, topK);
        }
        return lattice;
    }

    /** Index (from {@code off}) of the first largest of {@code x[off .. off + n)}, as {@code std::max_element}. */
    static int firstMax(float[] x, int off, int n) {
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (x[off + i] > x[off + best]) {
                best = i;
            }
        }
        return best;
    }
}
