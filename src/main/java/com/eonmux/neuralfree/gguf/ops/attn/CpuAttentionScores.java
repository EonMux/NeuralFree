package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuAttnKernels;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuSimd;

import java.util.Arrays;

/**
 * The key scoring and value accumulation of one {@link CpuAttentionCore} work unit: query rows {@code t < nt} of a
 * block (one row for a single-row call) against query heads {@code [h0, h0 + gc)} of one KV head.
 * <p>
 * {@link #prepare} lays the unit's query heads out as lanes ({@code t * gc + g}, padded to whole vectors), so the
 * dots of a group of keys with every (row, head) pair run as one {@link CpuAttnKernels#dots} call over the
 * {@code nt * gc} lanes that hold heads: each lane is the scalar loop {@code dot += q[d] * k[d]} over ascending
 * {@code d}, so a score is bit-identical to one (row, head) pair's own loop, while the vector kernel runs eight lanes
 * and several keys side by side instead of waiting on one dependent chain of adds. The value accumulation
 * {@code o += p V} runs through {@link CpuAttnKernels#accumulate}, the same per-element arithmetic. The kernels are
 * looked up per unit ({@link CpuSimd#attention()}), so an invalid kernel property is reported by name at every call.
 */
final class CpuAttentionScores {

    /** Keys one {@link CpuAttnKernels#dots} call scores: their K rows stay in the core's L1 cache. */
    static final int KEY_GROUP = 8;

    private CpuAttentionScores() {
    }

    /**
     * Lays out heads {@code [h0, h0 + gc)} of {@code nt} query rows (row {@code t} at {@code q[qOff + t * qStride]},
     * position {@code qPos[qFirst + t]}, mask row {@code qFirst + t}) as the unit's lanes.
     */
    static void prepare(CpuAttentionCore.Scratch s, float[] q, int qOff, int qStride, int nt, int[] qPos, int qFirst,
                        int h0, int gc, int dK) {
        for (int t = 0; t < nt; t++) {
            s.qPos[t] = qPos[qFirst + t];
            s.qRow[t] = qFirst + t;
        }
        lay(s, q, qOff, qStride, nt, h0, gc, dK);
    }

    /** {@link #prepare} of the single query row at {@code q[qOff]}, position {@code qPos}, mask row {@code qRow}. */
    static void prepareRow(CpuAttentionCore.Scratch s, float[] q, int qOff, int qPos, int qRow, int h0, int gc,
                           int dK) {
        s.qPos[0] = qPos;
        s.qRow[0] = qRow;
        lay(s, q, qOff, 0, 1, h0, gc, dK);
    }

    private static void lay(CpuAttentionCore.Scratch s, float[] q, int qOff, int qStride, int nt, int h0, int gc,
                            int dK) {
        final int quantum = CpuAttnKernels.LANE_QUANTUM;
        final int lanes = (nt * gc + quantum - 1) / quantum * quantum;
        s.rowsInUnit = nt;
        s.headsInUnit = gc;
        s.lanes = lanes;
        s.kernels = CpuSimd.attention();
        if (s.qT.length < dK * lanes) {
            s.qT = new float[dK * lanes];
        }
        if (s.keys.length < KEY_GROUP * dK) {
            s.keys = new float[KEY_GROUP * dK];
        }
        if (s.dots.length < KEY_GROUP * lanes) {
            s.dots = new float[KEY_GROUP * lanes];
        }
        final float[] qT = s.qT;
        Arrays.fill(qT, 0, dK * lanes, 0.0f);
        for (int t = 0; t < nt; t++) {
            for (int g = 0; g < gc; g++) {
                final int qb = qOff + t * qStride + (h0 + g) * dK;
                final int lane = t * gc + g;
                for (int d = 0; d < dK; d++) {
                    qT[d * lanes + lane] = q[qb + d];
                }
            }
        }
    }

    /**
     * Scores {@code s.scores[(t * gc + g) * KEY_BLOCK + r]} of the prepared unit against keys {@code [b, b + nb)}:
     * the scaled, soft-capped dot plus the position bias (none for a sink key); a key a row does not see scores
     * -INF for it, and a key no row sees is not read. Under a tree mask ({@code treeBits}) a fresh key row at or
     * after {@code freshStart} scores only for the query rows whose bitmask names it.
     */
    static void score(CpuAttentionCore.Params p, CpuAttentionCore.Rows rows, int b, int nb,
                      CpuAttentionCore.Scratch s, int h0, int kvh, long[] treeBits, int freshStart) {
        final int dK = rows.dK();
        final int nt = s.rowsInUnit;
        final int gc = s.headsInUnit;
        final int lanes = s.lanes;
        final float[] sc = s.scores;
        for (int r0 = 0; r0 < nb; r0 += KEY_GROUP) {
            int m = 0;
            for (int r = r0; r < Math.min(nb, r0 + KEY_GROUP); r++) {
                final int row = b + r;
                final int pk = rows.pos(row);
                final boolean sink = rows.sink(row);
                int vis = 0;
                for (int t = 0; t < nt; t++) {
                    if (visible(p, row, pk, sink, s.qPos[t], s.qRow[t], treeBits, freshStart)) {
                        vis |= 1 << t;
                    } else {
                        for (int g = 0; g < gc; g++) {
                            sc[(t * gc + g) * CpuAttentionCore.KEY_BLOCK + r] = Float.NEGATIVE_INFINITY;
                        }
                    }
                }
                if (vis != 0) {
                    readScaledK(p, rows, row, sink, kvh, s.keys, m * dK, dK);
                    s.keyAt[m] = r;
                    s.keyPos[m] = pk;
                    s.keyVis[m] = vis;
                    m++;
                }
            }
            if (m == 0) {
                continue;
            }
            s.kernels.dots(s.qT, lanes, nt * gc, s.keys, 0, dK, m, dK, s.dots, 0);
            for (int j = 0; j < m; j++) {
                final int r = s.keyAt[j];
                final int pk = s.keyPos[j];
                final boolean sink = rows.sink(b + r);
                for (int t = 0; t < nt; t++) {
                    if ((s.keyVis[j] >>> t & 1) == 0) {
                        continue;
                    }
                    for (int g = 0; g < gc; g++) {
                        final int h = h0 + g;
                        final float dot = s.dots[j * lanes + t * gc + g];
                        sc[(t * gc + g) * CpuAttentionCore.KEY_BLOCK + r] = CpuAttentionCore.score(dot, p.scale,
                                p.softcap) + (sink ? 0.0f : p.bias(h, s.qPos[t], pk, s.qRow[t]));
                    }
                }
            }
        }
    }

    /**
     * Whether the query row at {@code qPos} (mask row {@code qRow}) sees key row {@code row} at position {@code pk}.
     * A sink token precedes every cached key: its key index is the query's absolute position plus the sink count,
     * vLLM's seq_lens + sink_len alignment. Under a tree mask a fresh row's visibility is its mask bit alone:
     * siblings share one depth position, which the position rules would wrongly make mutually visible.
     */
    static boolean visible(CpuAttentionCore.Params p, int row, int pk, boolean sink, int qPos, int qRow,
                           long[] treeBits, int freshStart) {
        if (sink) {
            return !p.window.causal() || p.window.visible(qPos + p.sinkTokens(), pk);
        }
        if (treeBits != null && row >= freshStart) {
            return (treeBits[qRow] >>> (row - freshStart) & 1L) != 0L;
        }
        return p.window.visible(qPos, pk);
    }

    /**
     * Reads K head {@code kvh} of {@code row} into {@code dst[off, off + dK)} with an FP8 cache's dequant scale; the
     * sink keys are parameters, not cache rows, so the scale does not apply to them.
     */
    private static void readScaledK(CpuAttentionCore.Params p, CpuAttentionCore.Rows rows, int row, boolean sink,
                                    int kvh, float[] dst, int off, int dK) {
        rows.readK(row, kvh, dst, off);
        if (p.kScale != null && !sink) {
            final float ks = p.kScale[kvh];
            for (int d = off; d < off + dK; d++) {
                dst[d] *= ks;
            }
        }
    }

    /** {@code o += p V} of {@code heads} (row, head) pairs over keys {@code [b, b + nb)}; a value row no pair weights
     * is not read. */
    static void addValues(CpuAttentionCore.Params p, CpuAttentionCore.Rows rows, int b, int nb,
                          CpuAttentionCore.Scratch s, int heads, int kvh, int dV) {
        final float[] sc = s.scores;
        final float[] vbuf = s.v;
        final float[] acc = s.acc;
        final float[] vScale = p.vScale;
        final CpuAttnKernels kernels = s.kernels;
        for (int r = 0; r < nb; r++) {
            boolean used = false;
            for (int g = 0; g < heads && !used; g++) {
                used = sc[g * CpuAttentionCore.KEY_BLOCK + r] != 0.0f;
            }
            if (!used) {
                continue;
            }
            rows.readV(b + r, kvh, vbuf);
            /* The sink values are parameters, not cache rows: an FP8 cache's dequant scale does not apply. */
            if (vScale != null && !rows.sink(b + r)) {
                final float vs = vScale[kvh];
                for (int d = 0; d < dV; d++) {
                    vbuf[d] *= vs;
                }
            }
            for (int g = 0; g < heads; g++) {
                final float pr = sc[g * CpuAttentionCore.KEY_BLOCK + r];
                if (pr != 0.0f) {
                    kernels.accumulate(vbuf, 0, pr, acc, g * dV, dV);
                }
            }
        }
    }

    /** {@code o[oOff + d] += po[pOff + d] * w}: one split's values weighted into the merged state. */
    static void merge(float[] po, int pOff, float w, float[] o, int oOff, int dV) {
        CpuSimd.attention().accumulate(po, pOff, w, o, oOff, dV);
    }
}
