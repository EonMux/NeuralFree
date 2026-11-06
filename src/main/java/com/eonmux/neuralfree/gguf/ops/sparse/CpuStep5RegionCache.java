package com.eonmux.neuralfree.gguf.ops.sparse;

import java.util.Arrays;

/**
 * The Step5 region-summary cache of one sequence and one full-attention layer: the {@code [region][proxy]} summaries
 * of closed eight-token regions and the ring of the open region's raw {@code [index_k | index_z]} rows (vLLM
 * {@code common/ops/summary_update.py}, {@code common/sparse_summary_cache.py} and {@code common/csa_tail_ring.py}).
 *
 * <p>A region summary is an online softmax over its {@code regionBlock} indexer rows in position order. The ring holds
 * the open region's rows so the row that closes a region can replay them; it holds
 * {@code max(regionBlock, 8 + rollbackSnapshots)} rows ({@link com.eonmux.neuralfree.gguf.spec.Step5Spec#ringCapacity}),
 * one BF16-rounded {@code [key | z]} pair each. Only closed
 * regions are ever scored, so a summary written by a row that a rejected draft later uncloses is rewritten when the
 * region closes again. This class is the mutable view over two caller-owned F32 arrays; it allocates nothing per
 * token.
 */
public final class CpuStep5RegionCache {

    private final int regionBlock;
    private final int capacity;
    private final int proxy;
    private final float[] summary;
    private final int summaryOff;
    private final float[] ring;
    private final int ringOff;

    /**
     * @param summary   closed-region summaries, {@code [region][proxy]} from {@code summaryOff}, values already
     *                  quantized by a previous update (zeroed before the first)
     * @param ring      open-region rows, {@code [row][key | z]} from {@code ringOff}, {@code 2 * proxy} values each
     * @throws IllegalArgumentException for non-positive geometry or a view outside its arrays
     */
    public CpuStep5RegionCache(int regionBlock, int capacity, int proxy, float[] summary, int summaryOff,
                               float[] ring, int ringOff) {
        if (regionBlock <= 0 || capacity < regionBlock || proxy <= 0) {
            throw new IllegalArgumentException("Step5 region cache needs a positive region and proxy and capacity >= "
                    + regionBlock + ", got " + regionBlock + "/" + proxy + "/" + capacity);
        }
        this.regionBlock = regionBlock;
        this.capacity = capacity;
        this.proxy = proxy;
        this.summary = summary;
        this.summaryOff = summaryOff;
        this.ring = ring;
        this.ringOff = ringOff;
        checkView(summary, summaryOff, proxy, "summary");
        checkView(ring, ringOff, 2 * proxy, "ring");
    }

    public int regionBlock() { return regionBlock; }
    public int capacity() { return capacity; }
    public int proxy() { return proxy; }

    /** Float offset of summary {@code region} in the backing array. */
    public int summaryOffset(int region) {
        return summaryOff + region * proxy;
    }

    /** The backing summaries (a view; only {@link #summaryOffset} regions are written by {@link #update}). */
    public float[] summaryArray() {
        return summary;
    }

    /** The backing ring rows, {@code 2 * proxy} values per row from {@link #ringOffset(int)}. */
    public float[] ringArray() {
        return ring;
    }

    /** Float offset of the {@code [key | z]} row of {@code position}: {@code [key(proxy) | z(proxy)]}. */
    public int ringOffset(int position) {
        return ringOff + (position % capacity) * 2 * proxy;
    }

    /**
     * Stores one open-region row and, when {@code position} closes its region, writes that region's summary.
     *
     * @param position token position, {@code >= 0}
     * @param key      indexer key of the token, {@code proxy} values
     * @param keyOff   first key value
     * @param z        region-summary weights of the token, {@code proxy} values
     * @param zOff     first weight value
     * @throws IllegalArgumentException when a region falls outside the backing summaries
     */
    public void update(int position, float[] key, int keyOff, float[] z, int zOff) {
        if (position < 0) {
            throw new IllegalArgumentException("Step5 summary position " + position + " is negative");
        }
        checkView(key, keyOff, proxy, "indexer key");
        checkView(z, zOff, proxy, "indexer z");
        int row = position % capacity;
        int base = ringOff + row * 2 * proxy;
        for (int d = 0; d < proxy; d++) {
            ring[base + d] = Step5Dtypes.bf16(key[keyOff + d]);
            ring[base + proxy + d] = Step5Dtypes.bf16(z[zOff + d]);
        }
        if ((position + 1) % regionBlock != 0) {
            return;
        }
        int region = position / regionBlock;
        if (summaryOff + (region + 1) * proxy > summary.length) {
            throw new IllegalArgumentException("region " + region + " is outside the " + (summary.length - summaryOff)
                    + " summary values");
        }
        summarize(region, region * regionBlock, position);
    }

    /**
     * Replays the rows of region {@code region} at positions {@code [first, position]} and stores the summary: one
     * online update per row in position order, with a single maximum over every proxy dimension as the vLLM Triton
     * kernel computes it.
     */
    private void summarize(int region, int first, int position) {
        double[] numerator = new double[proxy];
        double[] denominator = new double[proxy];
        float maximum = Float.NEGATIVE_INFINITY;
        int slot = first % capacity;
        for (int p = first; p <= position; p++) {
            slot = p % capacity;
            int base = ringOff + slot * 2 * proxy;
            float updatedMax = maximum;
            for (int d = 0; d < proxy; d++) {
                updatedMax = Math.max(updatedMax, ring[base + proxy + d]);
            }
            for (int d = 0; d < proxy; d++) {
                double scale = denominator[d] > 0.0 ? Math.exp(maximum - updatedMax) : 0.0;
                double weight = Math.exp(ring[base + proxy + d] - updatedMax);
                numerator[d] = numerator[d] * scale
                        + weight * (double) ring[base + d];
                denominator[d] = denominator[d] * scale + weight;
            }
            maximum = updatedMax;
        }
        int out = summaryOff + region * proxy;
        for (int d = 0; d < proxy; d++) {
            float mean = (float) (numerator[d] / Math.max(denominator[d], 1e-20));
            summary[out + d] = Step5Dtypes.e4m3(Step5Dtypes.bf16(mean));
        }
    }

    /**
     * Rebuilds the summaries of closed regions {@code [0, closedRegions)} from the ring rows at their positions. A
     * context shift moves the rows first ({@link #shiftRows}) and re-rotates their keys, then calls this: every
     * summary is the online-softmax mean of the rows the region now holds, as if the region had closed there.
     *
     * @throws IllegalArgumentException when a region falls outside the backing summaries
     */
    public void rebuildSummaries(int closedRegions) {
        if (closedRegions < 0) {
            throw new IllegalArgumentException("closed region count " + closedRegions + " is negative");
        }
        if (summaryOff + (long) closedRegions * proxy > summary.length) {
            throw new IllegalArgumentException("regions [0, " + closedRegions + ") are outside the "
                    + (summary.length - summaryOff) + " summary values");
        }
        for (int r = 0; r < closedRegions; r++) {
            summarize(r, r * regionBlock, (r + 1) * regionBlock - 1);
        }
    }

    /**
     * Zeroes the summaries of regions {@code [from, capacity / regionBlock + 1)}: the regions a shift left without
     * rows. Nothing is written for a {@code from} beyond the backing summaries.
     */
    public void clearSummariesFrom(int from) {
        int first = Math.max(0, from);
        int regionsHeld = (summary.length - summaryOff) / proxy;
        for (int r = first; r < regionsHeld; r++) {
            Arrays.fill(summary, summaryOff + r * proxy, summaryOff + (r + 1) * proxy, 0f);
        }
    }

    /**
     * Moves the ring rows of a context shift: position {@code dst >= nKeep} takes the row of
     * {@code dst + nDiscard}; positions below {@code nKeep} stay. The ring must hold every position the shift keeps
     * (its capacity at least the old position count), as a plan reserving the whole context does.
     *
     * @param nKeep    positions kept before the discarded range
     * @param nDiscard rows discarded
     * @param nNew     positions held after the shift ({@code old positions - nDiscard})
     * @throws IllegalArgumentException when a range is invalid or a row lies outside the ring
     */
    public void shiftRows(int nKeep, int nDiscard, int nNew) {
        if (nKeep < 0 || nDiscard <= 0 || nNew < nKeep) {
            throw new IllegalArgumentException("ring shift of keep=" + nKeep + ", discard=" + nDiscard + ", new="
                    + nNew + " is invalid");
        }
        int old = nNew + nDiscard;
        if (old > capacity) {
            throw new IllegalArgumentException("ring capacity " + capacity + " cannot hold the " + old
                    + " positions the shift moves from");
        }
        for (int dst = nKeep; dst < nNew; dst++) {
            int src = dst + nDiscard;
            if (src >= old) {
                throw new IllegalArgumentException("ring shift reads position " + src + " beyond the " + old
                        + " stored positions");
            }
            // destination rows are below their sources, so every copy reads a row not yet written
            System.arraycopy(ring, ringOff + (src % capacity) * 2 * proxy, ring,
                    ringOff + (dst % capacity) * 2 * proxy, 2 * proxy);
        }
    }

    private static void checkView(float[] a, int off, int n, String what) {
        if (a == null || off < 0 || off + n > a.length) {
            throw new IllegalArgumentException(what + " view [" + off + ", " + (off + n) + ") is outside "
                    + (a == null ? "a null array" : "length " + a.length));
        }
    }
}
