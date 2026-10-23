package com.eonmux.neuralfree.gguf.ops.rope;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.Objects;

/**
 * RoPE position buffers of text tokens, in the layout {@code LLM_RoPE_Ext} reads: section {@code k} of token
 * {@code t} at {@code pos[t + k * nTokens]} (llama-batch.cpp:146-157). The per-architecture pattern of the
 * sections ({@code [p, p, p, 0]} by default, {@code [p, 0, 0, 0]} where the arch says so) is the
 * {@link MropeTextPositions} that the hparams hooks attach to each {@link RopeSpec}; single-section types read
 * section 0 only, which every pattern sets to the token position. Also defines the per-cell deltas of a context
 * shift (llama.cpp {@code seq_rm} + {@code seq_add}).
 */
public final class RopePositions {

    private RopePositions() {
    }

    /**
     * Positions of {@code nTokens} consecutive text tokens from {@code pos0} for {@code spec}:
     * {@link RopeMath#positionsPerToken} sections of {@code nTokens} values each.
     *
     * @throws IllegalArgumentException for a negative token count or position
     */
    public static int[] text(RopeSpec spec, int pos0, int nTokens) {
        Objects.requireNonNull(spec, "spec");
        checkCounts(pos0, nTokens);
        final int sections = RopeMath.positionsPerToken(spec.type());
        final int[] dst = new int[sections * nTokens];
        fill(spec.type(), spec.textPositions(), pos0, nTokens, dst);
        return dst;
    }

    /**
     * Writes the positions of {@code nTokens} consecutive text tokens from {@code pos0}: four sections per the
     * pattern for multi-section types, {@code dst[t] = pos0 + t} otherwise.
     *
     * @throws IllegalArgumentException when {@code dst} is too short or a count is negative
     */
    public static void fill(RopeType type, MropeTextPositions pattern, int pos0, int nTokens, int[] dst) {
        Objects.requireNonNull(pattern, "pattern");
        Objects.requireNonNull(dst, "dst");
        checkCounts(pos0, nTokens);
        if (RopeMath.multiSection(Objects.requireNonNull(type, "type"))) {
            pattern.fill(pos0, nTokens, dst);
            return;
        }
        requireLength(dst, nTokens);
        for (int t = 0; t < nTokens; t++) {
            dst[t] = pos0 + t;
        }
    }

    /**
     * Writes the shared four-section step buffer ({@code GpuStepInputs.ropePos}, {@code 4 * nTokens} values) for text
     * tokens at {@code pos0 ..}: serves every layer, as single-section layers read section 0 only.
     *
     * @throws IllegalArgumentException when {@code dst} holds fewer than {@code 4 * nTokens} values
     */
    public static void fillStep(MropeTextPositions pattern, int pos0, int nTokens, int[] dst) {
        Objects.requireNonNull(pattern, "pattern");
        Objects.requireNonNull(dst, "dst");
        checkCounts(pos0, nTokens);
        pattern.fill(pos0, nTokens, dst);
    }

    /**
     * Writes the positions of {@code nTokens} text tokens with explicit positions {@code tokenPos[off + t]} (for
     * example rows gathered out of order): four sections per the pattern for multi-section types, the positions
     * themselves otherwise.
     *
     * @throws IllegalArgumentException when an array is too short or a position is negative
     */
    public static void fill(RopeType type, MropeTextPositions pattern, int[] tokenPos, int off, int nTokens,
                            int[] dst) {
        Objects.requireNonNull(pattern, "pattern");
        Objects.requireNonNull(tokenPos, "tokenPos");
        Objects.requireNonNull(dst, "dst");
        if (off < 0 || nTokens < 0 || (long) off + nTokens > tokenPos.length) {
            throw new IllegalArgumentException("token positions [" + off + ", " + ((long) off + nTokens)
                    + ") exceed " + tokenPos.length + " values");
        }
        final int sections = RopeMath.positionsPerToken(Objects.requireNonNull(type, "type"));
        requireLength(dst, sections * nTokens);
        for (int t = 0; t < nTokens; t++) {
            final int p = tokenPos[off + t];
            if (p < 0) {
                throw new IllegalArgumentException("token " + t + " has negative position " + p);
            }
            for (int s = 0; s < sections; s++) {
                dst[s * nTokens + t] = sections == 1 ? p : pattern.position(s, p);
            }
        }
    }

    /**
     * Context shift of one sequence (llama.cpp {@code seq_rm(nKeep, nKeep + nDiscard)} then
     * {@code seq_add(nKeep + nDiscard, inf, -nDiscard)}): for every cache row with position {@code rowPos[r]},
     * {@code newPos[r]} is -1 for an empty or discarded row, the position less {@code nDiscard} for rows at or after
     * {@code nKeep + nDiscard}, and unchanged otherwise; {@code delta[r]} is {@code -nDiscard} for the moved rows and
     * 0 for the rest, the input of {@code LLM_KV_RopeShift_*}.
     *
     * @throws IllegalArgumentException for negative counts or output arrays shorter than {@code rowPos}
     */
    public static void contextShift(int[] rowPos, int nKeep, int nDiscard, int[] newPos, int[] delta) {
        Objects.requireNonNull(rowPos, "rowPos");
        Objects.requireNonNull(newPos, "newPos");
        Objects.requireNonNull(delta, "delta");
        if (nKeep < 0 || nDiscard <= 0) {
            throw new IllegalArgumentException("context shift needs nKeep >= 0 and nDiscard > 0, got " + nKeep + " / "
                    + nDiscard);
        }
        if (newPos.length < rowPos.length || delta.length < rowPos.length) {
            throw new IllegalArgumentException("context shift outputs hold " + newPos.length + " / " + delta.length
                    + " values, need " + rowPos.length);
        }
        final long end = (long) nKeep + nDiscard;
        for (int r = 0; r < rowPos.length; r++) {
            final int p = rowPos[r];
            if (p < 0 || (p >= nKeep && p < end)) {
                newPos[r] = -1;
                delta[r] = 0;
            } else if (p >= end) {
                newPos[r] = p - nDiscard;
                delta[r] = -nDiscard;
            } else {
                newPos[r] = p;
                delta[r] = 0;
            }
        }
    }

    private static void checkCounts(int pos0, int nTokens) {
        if (pos0 < 0 || nTokens < 0) {
            throw new IllegalArgumentException("positions need pos0 >= 0 and nTokens >= 0, got " + pos0 + " / "
                    + nTokens);
        }
    }

    private static void requireLength(int[] dst, long n) {
        if (dst.length < n) {
            throw new IllegalArgumentException("position buffer holds " + dst.length + " values, need " + n);
        }
    }
}
