package com.eonmux.neuralfree.gguf.ops.rope;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.Arrays;
import java.util.Objects;

/**
 * Per-pair frequency table of one RoPE configuration, the CPU counterpart of the angle computation in
 * {@code LLM_RoPE_Ext} and {@code LLM_KV_RopeShift_*}: for pair {@code i} it holds the position section, the factor
 * {@code theta_scale^e(i)} (ggml-cuda/rope.cu: {@code powf(theta_scale, i)}) and the frequency divisor
 * {@code ff[i]}, so the cos/sin row of a token is {@code rope_yarn(p_section * theta_scale^e / ff)} per pair
 * (see {@link RopeMath}). One table serves every head and token of a layer; {@link #cosSin} fills a caller-owned
 * row, so a table is immutable and shareable across threads.
 */
public final class RopeFreqTable {

    private final RopeType type;
    private final int nDims;
    private final int pairs;
    private final boolean multi;
    private final int[] section;
    private final float[] thetaPow;
    private final float[] freqFactor;
    private final float freqScale;
    private final float extFactor;
    private final float attnFactor;
    private final float corrLo;
    private final float corrHi;

    private RopeFreqTable(RopeType type, int nDims, float thetaScale, float freqScale, float extFactor,
                          float attnFactor, float corrLo, float corrHi, int[] sections, float[] freqFactors) {
        this.type = type;
        this.nDims = nDims;
        this.pairs = RopeMath.pairCount(type, nDims);
        this.multi = RopeMath.multiSection(type);
        this.freqScale = freqScale;
        this.extFactor = extFactor;
        this.attnFactor = attnFactor;
        this.corrLo = corrLo;
        this.corrHi = corrHi;
        this.section = new int[pairs];
        this.thetaPow = new float[pairs];
        this.freqFactor = new float[pairs];
        for (int i = 0; i < pairs; i++) {
            final int sec = RopeMath.section(type, i, sections);
            section[i] = sec;
            thetaPow[i] = (float) Math.pow(thetaScale, (float) RopeMath.thetaExponent(type, i, sec, sections));
            freqFactor[i] = freqFactors != null ? freqFactors[i] : 1.0f;
        }
    }

    /**
     * Table of a layer's rotation ({@code LLM_RoPE_Ext} arguments from {@code spec}).
     *
     * @param freqFactors per-pair divisors ({@code rope_freqs} or the long/short factors, at least
     *                    {@link RopeMath#pairCount} values), or null for none
     * @throws IllegalArgumentException when the factors are too short or the sections do not fit the type
     */
    public static RopeFreqTable of(RopeSpec spec, float[] freqFactors) {
        Objects.requireNonNull(spec, "spec");
        return of(spec.type(), spec.nDims(), spec.thetaScale(), spec.freqScale(), spec.extFactor(), spec.attnFactor(),
                spec.corrLo(), spec.corrHi(), spec.sections(), freqFactors);
    }

    /**
     * Table of the K-cache context shift of a layer ({@code LLM_KV_RopeShift_*}): the layer's parameters with the
     * {@link RopeMath#shiftType shift type} (NORM or NEOX, one position, exponent {@code i}).
     *
     * @throws IllegalArgumentException for a type that cannot be shifted or factors that are too short
     */
    public static RopeFreqTable forShift(RopeSpec spec, float[] freqFactors) {
        Objects.requireNonNull(spec, "spec");
        return of(RopeMath.shiftType(spec.type()), spec.nDims(), spec.thetaScale(), spec.freqScale(),
                spec.extFactor(), spec.attnFactor(), spec.corrLo(), spec.corrHi(),
                new int[MropeTextPositions.SECTIONS], freqFactors);
    }

    /**
     * Table from explicit kernel parameters.
     *
     * @param sections four section sizes (only read by multi-section types)
     * @throws IllegalArgumentException for NONE, odd or non-positive {@code nDims}, multi-section sizes that cover
     *                                  no pair, or factors shorter than the pair count
     */
    public static RopeFreqTable of(RopeType type, int nDims, float thetaScale, float freqScale, float extFactor,
                                   float attnFactor, float corrLo, float corrHi, int[] sections, float[] freqFactors) {
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(sections, "sections");
        if (type == RopeType.NONE) {
            throw new IllegalArgumentException("rope type NONE has no frequency table");
        }
        if (nDims <= 0 || nDims % 2 != 0) {
            throw new IllegalArgumentException("rope nDims must be even and positive, got " + nDims);
        }
        if (sections.length != MropeTextPositions.SECTIONS) {
            throw new IllegalArgumentException("rope sections need " + MropeTextPositions.SECTIONS + " values, got "
                    + sections.length);
        }
        if (RopeMath.multiSection(type)) {
            int sum = 0;
            for (int s : sections) {
                if (s < 0) {
                    throw new IllegalArgumentException("rope sections must be >= 0: " + Arrays.toString(sections));
                }
                sum += s;
            }
            if (sum <= 0) {
                throw new IllegalArgumentException(type + " rope needs positive sections, got "
                        + Arrays.toString(sections));
            }
        }
        final int pairs = RopeMath.pairCount(type, nDims);
        if (freqFactors != null && freqFactors.length < pairs) {
            throw new IllegalArgumentException("rope frequency factors hold " + freqFactors.length + " values, "
                    + type + " with nDims " + nDims + " reads " + pairs);
        }
        return new RopeFreqTable(type, nDims, thetaScale, freqScale, extFactor, attnFactor, corrLo, corrHi,
                sections.clone(), freqFactors);
    }

    public RopeType type() {
        return type;
    }

    /** Rotated dims parameter ({@code n_dims}). */
    public int nDims() {
        return nDims;
    }

    /** Rotation pairs per head. */
    public int pairs() {
        return pairs;
    }

    /** Positions per token read by {@link #cosSin(int[], int, int, int, float[])}: 4 for multi-section types, else 1. */
    public int positionsPerToken() {
        return multi ? MropeTextPositions.SECTIONS : 1;
    }

    /** Position section of pair {@code i}. */
    public int section(int i) {
        return section[i];
    }

    /** Offset of the first element of pair {@code i} from the first rotated dim. */
    public int pairFirst(int i) {
        return RopeMath.pairFirst(type, i);
    }

    /** Distance between the two elements of a pair. */
    public int pairSpan() {
        return RopeMath.pairSpan(type, nDims);
    }

    /** Head elements the rotation covers from the first rotated dim. */
    public int rotatedWidth() {
        return RopeMath.rotatedWidth(type, nDims);
    }

    /**
     * Fills {@code cs[2i] = cos * mscale}, {@code cs[2i + 1] = sin * mscale} for every pair of token {@code t}, whose
     * section {@code k} position is {@code pos[posOff + t + k * posStride]} (single-section types read
     * {@code pos[posOff + t]} only).
     *
     * @throws IllegalArgumentException when {@code cs} holds fewer than {@code 2 * pairs()} values
     */
    public void cosSin(int[] pos, int posOff, int t, int posStride, float[] cs) {
        requireRow(cs);
        for (int i = 0; i < pairs; i++) {
            final int p = multi ? pos[posOff + t + section[i] * posStride] : pos[posOff + t];
            pair(p, i, cs);
        }
    }

    /**
     * Fills the cos/sin row for one position used by every section (context shift, text NORM/NEOX).
     *
     * @throws IllegalArgumentException when {@code cs} holds fewer than {@code 2 * pairs()} values
     */
    public void cosSin(int p, float[] cs) {
        requireRow(cs);
        for (int i = 0; i < pairs; i++) {
            pair(p, i, cs);
        }
    }

    private void pair(int p, int i, float[] cs) {
        final float thetaBase = (float) p * thetaPow[i];
        RopeMath.cosSin(thetaBase / freqFactor[i], freqScale, extFactor, attnFactor, corrLo, corrHi, i, cs, 2 * i);
    }

    private void requireRow(float[] cs) {
        if (cs.length < 2 * pairs) {
            throw new IllegalArgumentException("cos/sin row holds " + cs.length + " values, need " + 2 * pairs);
        }
    }

    @Override
    public String toString() {
        return "RopeFreqTable{" + type + ", nDims=" + nDims + ", pairs=" + pairs + ", freqScale=" + freqScale
                + ", ext=" + extFactor + ", attn=" + attnFactor + ", corr=[" + corrLo + ", " + corrHi + "]}";
    }
}
