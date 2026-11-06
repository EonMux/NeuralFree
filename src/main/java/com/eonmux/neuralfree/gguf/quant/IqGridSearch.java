package com.eonmux.neuralfree.gguf.quant;

import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.quant.QuantizeSearch.nearestInt;

/**
 * The sub-block search the IQ2 and IQ3 grid quantizers of llama.cpp ggml-quants.c share
 * ({@code quantize_row_iq2_xxs_impl} :3310, {@code iq2_xs} :3488, {@code iq3_xxs} :3954, {@code iq3_s}
 * :4185, {@code iq2_s} :5130): the sign fold of each 8-value group onto non-negative magnitudes, the scan
 * of trial scales that rounds every grid group to levels and snaps off-grid vectors to their best
 * neighbour ({@code iq2_find_best_neighbour} :3286, {@code iq3_find_best_neighbour} :3930), the
 * least-squares refit of the winning levels, and the grid indices of the result. The arithmetic follows
 * the reference in float, term for term, so the chosen levels and scales equal the reference's.
 * <p>
 * One instance holds the scratch state of one sub-block size and is reused across the sub-blocks of a
 * row; it is not thread-safe, and each row call makes its own.
 */
final class IqGridSearch {

    /** Which groups the final least-squares pass re-rounds at the winning scale. */
    enum Refit {
        /** Every group, when the scale is positive (IQ2_XXS). */
        ALL,
        /** The groups the winning trial left off the grid, when there are any (IQ2_XS, IQ2_S, IQ3_XXS). */
        OFF_GRID,
        /** Every group, when the winning trial left any off the grid (IQ3_S). */
        ALL_WHEN_OFF_GRID
    }

    private final IqGrid grid;
    private final int n;
    private final int lanes;
    private final int groups;
    private final int kMaxQ;

    /** The sign-folded magnitudes of the sub-block (one value may be negative after the parity fold). */
    final float[] xval;
    /** The per-value importance of the sub-block. */
    final float[] weight;
    /** The square roots of {@link #weight}, the neighbour search's weights. */
    final float[] waux;
    /** The chosen levels. */
    final byte[] levels;
    /** The grid index of each group after {@link #gridIndices}. */
    final int[] index;

    private final byte[] trial;
    private final boolean[] onGrid;
    private final boolean[] onGridTrial;

    /**
     * @param grid  the lattice the groups snap to
     * @param n     values per sub-block
     * @param kMaxQ levels per lane ({@code kMaxQ} of the reference: 3 for IQ2, 8 for IQ3)
     */
    IqGridSearch(IqGrid grid, int n, int kMaxQ) {
        this.grid = grid;
        this.n = n;
        this.lanes = grid.lanes();
        this.groups = n / lanes;
        this.kMaxQ = kMaxQ;
        this.xval = new float[n];
        this.weight = new float[n];
        this.waux = new float[n];
        this.levels = new byte[n];
        this.index = new int[groups];
        this.trial = new byte[n];
        this.onGrid = new boolean[groups];
        this.onGridTrial = new boolean[groups];
    }

    /** The imatrix weights {@code qw * sqrt(sigma2 + x^2)} of the sub-block at {@code x[xOff]}, and their roots. */
    void weighted(float[] x, int xOff, float[] qw, int qwOff, float sigma2) {
        for (int i = 0; i < n; ++i) {
            float v = x[xOff + i];
            weight[i] = qw[qwOff + i] * (float) Math.sqrt(sigma2 + v * v);
        }
        roots();
    }

    /** The unweighted importance {@code bias + x^2} of the sub-block at {@code x[xOff]}, and their roots. */
    void unweighted(float[] x, int xOff, float bias) {
        for (int i = 0; i < n; ++i) {
            float v = x[xOff + i];
            weight[i] = bias + v * v;
        }
        roots();
    }

    private void roots() {
        for (int i = 0; i < n; ++i) {
            waux[i] = (float) Math.sqrt(weight[i]);
        }
    }

    /**
     * Folds the signs of 8-value group {@code k} of the sub-block at {@code x[xOff]} into {@link #xval} and
     * returns its sign byte. With {@code evenParity} the formats store 7 sign bits, so an odd count of
     * negatives flips the sign of the least important value back and the 7-bit pattern is returned.
     */
    int foldSigns(float[] x, int xOff, int k, boolean evenParity) {
        int nflip = 0;
        int s = 0;
        for (int i = 0; i < 8; ++i) {
            float v = x[xOff + 8 * k + i];
            if (v >= 0) {
                xval[8 * k + i] = v;
            } else {
                xval[8 * k + i] = -v;
                ++nflip;
                s |= 1 << i;
            }
        }
        if (!evenParity) {
            return s;
        }
        if (nflip % 2 != 0) {
            int imin = 0;
            float x0 = x[xOff + 8 * k];
            float min = weight[8 * k] * x0 * x0;
            for (int i = 1; i < 8; ++i) {
                float xi = x[xOff + 8 * k + i];
                float ax = weight[8 * k + i] * xi * xi;
                if (ax < min) {
                    min = ax;
                    imin = i;
                }
            }
            xval[8 * k + imin] = -xval[8 * k + imin];
            s ^= 1 << imin;
        }
        return s & 127;
    }

    /** The largest folded magnitude. */
    float max() {
        float max = xval[0];
        for (int i = 1; i < n; ++i) {
            max = Math.max(max, xval[i]);
        }
        return max;
    }

    /** Clears the chosen levels, the start the reference gives the scale scan. */
    void clearLevels() {
        Arrays.fill(levels, (byte) 0);
    }

    /**
     * The scan of trial inverse scales {@code (2*kMaxQ - 1 + is*step) / base} for {@code is} in
     * {@code [-range, range]}: each trial rounds every group to levels, snaps off-grid groups to their best
     * neighbour, and wins when its least-squares fit explains more of the weighted signal. Returns the
     * winning scale, {@code scale} when no trial wins; the winning levels land in {@link #levels}.
     */
    float scan(float base, float scale, int range, float step, boolean initialOnGrid) {
        float best = 0;
        Arrays.fill(onGrid, initialOnGrid);
        for (int is = -range; is <= range; ++is) {
            float id = (2 * kMaxQ - 1 + is * step) / base;
            float thisScale = 1 / id;
            for (int k = 0; k < groups; ++k) {
                onGridTrial[k] = place(k, id, thisScale, trial);
            }
            float sumqx = 0;
            float sumq2 = 0;
            for (int i = 0; i < n; ++i) {
                float w = weight[i];
                float q = 2 * trial[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                System.arraycopy(trial, 0, levels, 0, n);
                System.arraycopy(onGridTrial, 0, onGrid, 0, groups);
            }
        }
        return scale;
    }

    /**
     * The final pass: re-rounds the groups {@code mode} names at the winning {@code scale}, snapping off-grid
     * vectors to their best neighbour, and refits the scale to the levels. Returns the refitted scale, or
     * {@code scale} when the pass does not run.
     */
    float refit(float scale, Refit mode) {
        boolean anyOff = false;
        for (int k = 0; k < groups; ++k) {
            anyOff |= !onGrid[k];
        }
        if (!(scale > 0) || (mode != Refit.ALL && !anyOff)) {
            return scale;
        }
        float id = 1 / scale;
        for (int k = 0; k < groups; ++k) {
            if (mode == Refit.OFF_GRID && onGrid[k]) {
                continue;
            }
            place(k, id, scale, levels);
        }
        float sumqx = 0;
        float sumq2 = 0;
        for (int i = 0; i < n; ++i) {
            float w = weight[i];
            float q = 2 * levels[i] + 1;
            sumqx += w * xval[i] * q;
            sumq2 += w * q * q;
        }
        return sumq2 > 0 ? sumqx / sumq2 : scale;
    }

    /**
     * The grid index of every group of {@link #levels}, into {@link #index}.
     *
     * @throws IllegalStateException when a group's levels are off the grid, which the search never leaves
     */
    void gridIndices() {
        for (int k = 0; k < groups; ++k) {
            int u = pack(levels, k);
            int g = grid.indexOf(u);
            if (g < 0) {
                throw new IllegalStateException("level vector " + u + " of group " + k + " is not on the grid");
            }
            index[k] = g;
        }
    }

    /**
     * Rounds group {@code k} at inverse scale {@code id} to levels in {@code out}; an off-grid vector is
     * replaced by the levels of its best neighbour at {@code scale}. Returns true when the rounded vector
     * was on the grid.
     */
    private boolean place(int k, float id, float scale, byte[] out) {
        for (int i = 0; i < lanes; ++i) {
            int l = nearestInt(0.5f * (id * xval[lanes * k + i] - 1));
            out[lanes * k + i] = (byte) Math.max(0, Math.min(kMaxQ - 1, l));
        }
        int u = pack(out, k);
        if (grid.indexOf(u) >= 0) {
            return true;
        }
        bestNeighbour(u, k, scale, out);
        return false;
    }

    /** The neighbour of {@code u} closest to group {@code k} at {@code scale} in weighted squared error. */
    private void bestNeighbour(int u, int k, float scale, byte[] out) {
        int off = lanes * k;
        float bestD2 = Float.MAX_VALUE;
        int gridIndex = -1;
        for (int g : grid.neighbours(u)) {
            float d2 = 0;
            for (int i = 0; i < lanes; ++i) {
                float q = 2 * grid.level(g, i) + 1;
                float diff = scale * q - xval[off + i];
                d2 += waux[off + i] * diff * diff;
            }
            if (d2 < bestD2) {
                bestD2 = d2;
                gridIndex = g;
            }
        }
        if (gridIndex < 0) {
            throw new IllegalStateException("no grid neighbour fits group " + k
                    + "; the values or weights are not finite");
        }
        for (int i = 0; i < lanes; ++i) {
            out[off + i] = (byte) grid.level(gridIndex, i);
        }
    }

    private int pack(byte[] l, int k) {
        int u = 0;
        for (int i = 0; i < lanes; ++i) {
            u |= l[lanes * k + i] << (grid.bits() * i);
        }
        return u;
    }

    /** {@code sum x^2} over {@code n} values in order, the super-block energy the weights scale with. */
    static float sumSquares(float[] x, int off, int n) {
        float s = 0;
        for (int i = 0; i < n; ++i) {
            s += x[off + i] * x[off + i];
        }
        return s;
    }
}
