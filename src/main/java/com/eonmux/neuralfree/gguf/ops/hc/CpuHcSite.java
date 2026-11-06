package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.CoefficientParams;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * One bound hyper-connection collapse site on the CPU, the twin of {@link GpuHcSite} with the same steps and the
 * same temporaries ({@link HcSubBlock}): the norms go through {@link CpuOps} / {@link NormMath}, the projections
 * through {@link CpuMatVec}, the coefficients, collapse and post map through {@link CpuHyperConnections}. Arrays
 * come from {@link CpuScratch} at construction; nothing is allocated per token. Not thread-safe.
 */
public final class CpuHcSite {

    /** Minimum tokens per parallel range of the coefficients and the grouped norm. */
    private static final int TOKEN_GRAIN = 1;

    private final HcSubBlock block;
    private final CpuMatVec matVec;
    private final int ubatch;
    private final int hc;
    private final int nEmbd;
    private final int streamWidth;

    private final CpuTensorRef fn;
    private final float[] scale;
    private final float[] base;
    private final float[] norm;
    private final CpuTensorRef down;
    private final CpuTensorRef up;
    private final CpuTensorRef inject;
    /** Merged {@code [down | inject]} weight; null unless the merged gated form. */
    private final CpuTensorRef downInject;

    private final float[] flat;
    private final float[] mixes;
    private final float[] pre;
    private final float[] post;
    private final float[] comb;
    private final float[] lowRank;
    /** Merged {@code [down | inject]} output rows before the split; null unless the merged gated form. */
    private final float[] merged;
    private final float[] gate;

    /**
     * @throws MissingTensorException   when a tensor of the site is absent
     * @throws IllegalArgumentException when a temporary exceeds a Java array or a decoded vector has the wrong size
     */
    public CpuHcSite(CpuForwardContext ctx, HcSubBlock block) {
        this.block = Objects.requireNonNull(block, "block");
        CpuWeights weights = ctx.weights();
        CpuScratch scratch = ctx.scratch();
        String arch = ctx.spec().arch().ggufName();
        this.matVec = ctx.matVec();
        this.ubatch = scratch.ubatch();
        this.hc = block.hc();
        this.nEmbd = block.nEmbd();
        this.streamWidth = block.streamWidth();
        HcSubBlock.Kind kind = block.kind();

        int il = block.il();
        this.fn = kind.projected() ? tensor(weights, arch, il, block.fn()) : null;
        this.scale = kind.projected() ? vector(weights, arch, il, block.scale(), block.coefficients().scaleCount())
                : null;
        this.base = kind.projected() ? vector(weights, arch, il, block.base(), block.coefficients().baseCount())
                : null;
        this.norm = kind.gated() ? vector(weights, arch, il, block.norm(), hc * nEmbd) : null;
        this.down = block.down() != null ? tensor(weights, arch, il, block.down()) : null;
        this.up = kind.gated() ? tensor(weights, arch, il, block.up()) : null;
        this.inject = block.inject() != null ? tensor(weights, arch, il, block.inject()) : null;
        this.downInject = block.downInject() != null ? tensor(weights, arch, il, block.downInject()) : null;

        boolean hasComb = block.post() != null && block.post().hasComb();
        this.flat = kind.projected() || kind.gated() ? scratch.temp(HcSubBlock.TEMP_FLAT, rows(streamWidth)) : null;
        this.mixes = kind.projected() ? scratch.temp(HcSubBlock.TEMP_MIXES, rows(block.mixDim())) : null;
        this.pre = kind.projected() ? scratch.temp(HcSubBlock.TEMP_PRE, rows(hc)) : null;
        this.post = kind.expands() ? scratch.temp(HcSubBlock.TEMP_POST, rows(hc)) : null;
        this.comb = hasComb ? scratch.temp(HcSubBlock.TEMP_COMB, rows(hc * hc)) : null;
        this.lowRank = kind.gated() ? scratch.temp(HcSubBlock.TEMP_LOW_RANK, rows(block.lowRank())) : null;
        this.merged = downInject != null
                ? scratch.temp(HcSubBlock.TEMP_MERGED_DOWN, rows(block.lowRank() + hc)) : null;
        this.gate = kind.gated() ? scratch.temp(HcSubBlock.TEMP_GATE, rows(streamWidth)) : null;
    }

    public HcSubBlock block() {
        return block;
    }

    /**
     * Computes the pre, post and comb coefficients of the last {@code nTokens} stream rows without collapsing them
     * ({@code LLM_HC_PrePostComb}); {@link #collapseWithPre} then collapses with the pre values.
     *
     * @throws IllegalStateException    when the site is not projected (the gated and mean variants have no mixes)
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]}
     */
    public void computeCoefficients(int nTokens, float[] stream, int streamOff) {
        if (!block.kind().projected()) {
            throw new IllegalStateException("hyper-connection " + block + " projects no mixes");
        }
        checkTokens(nTokens);
        CpuOps.norm(NormKind.RMS, block.normEps(), stream, streamOff, flat, 0, nTokens, streamWidth, null, null);
        matVec.matmul(fn, flat, 0, mixes, 0, nTokens);
        final CoefficientParams coef = block.coefficients();
        CpuParallel.parallelFor(nTokens, TOKEN_GRAIN, (from, to) -> CpuHyperConnections.coefficients(coef, from, to,
                mixes, 0, scale, base, pre, 0, post, 0, comb, 0));
    }

    /**
     * Collapses {@code nTokens} stream rows with pre coefficients {@code preValues[preOff ..]}, as a mix computed by
     * {@link #computeCoefficients} on another sub-block would.
     *
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]}
     */
    public void collapseWithPre(int nTokens, float[] stream, int streamOff, float[] preValues, int preOff,
                                float[] out, int outOff) {
        checkTokens(nTokens);
        CpuHyperConnections.requireLength("pre", preValues, (long) preOff + (long) nTokens * hc);
        CpuHyperConnections.pre(block.pre(), hc, nEmbd, nTokens, stream, streamOff, preValues, preOff, out, outOff);
    }

    /** The pre coefficients of the last collapse or {@link #computeCoefficients}. */
    public float[] preCoefficients() {
        if (pre == null) {
            throw new IllegalStateException("hyper-connection " + block + " has no pre coefficients");
        }
        return pre;
    }

    /**
     * Collapses {@code nTokens} stream rows ({@code stream[streamOff + t*S ...]}) into
     * {@code out[outOff + t*n_embd ...]} and computes the post coefficients {@link #expand} uses. {@code out} must
     * not overlap the stream.
     *
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]}
     */
    public void collapse(int nTokens, float[] stream, int streamOff, float[] out, int outOff) {
        checkTokens(nTokens);
        HcSubBlock.Kind kind = block.kind();
        if (kind.projected()) {
            CpuOps.norm(NormKind.RMS, block.normEps(), stream, streamOff, flat, 0, nTokens, streamWidth, null, null);
            matVec.matmul(fn, flat, 0, mixes, 0, nTokens);
            final CoefficientParams coef = block.coefficients();
            CpuParallel.parallelFor(nTokens, TOKEN_GRAIN, (from, to) -> CpuHyperConnections.coefficients(coef, from,
                    to, mixes, 0, scale, base, pre, 0, post, 0, comb, 0));
            CpuHyperConnections.pre(block.pre(), hc, nEmbd, nTokens, stream, streamOff, pre, 0, out, outOff);
        } else if (kind.gated()) {
            groupedNorm(nTokens, stream, streamOff);
            if (downInject != null) {
                mergedGate(nTokens);
            } else {
                int n = nTokens * block.lowRank();
                matVec.matmul(down, flat, 0, lowRank, 0, nTokens);
                CpuOps.scale(lowRank, 0, n, block.lowRankScale());
                CpuOps.unary(ActOp.SILU, ActParams.NONE, lowRank, 0, lowRank, 0, n);
                matVec.matmul(up, lowRank, 0, gate, 0, nTokens);
            }
            CpuHyperConnections.pre(block.pre(), hc, nEmbd, nTokens, flat, 0, gate, 0, out, outOff);
            if (inject != null) {
                matVec.matmul(inject, flat, 0, post, 0, nTokens);
            }
        } else {
            CpuHyperConnections.pre(block.pre(), hc, nEmbd, nTokens, stream, streamOff, null, 0, out, outOff);
        }
    }

    /**
     * The merged qwen4exp gated mix (qwen4exp.cpp:363-380): one matmul with the {@code [hc_*_down |
     * hc_*_inject]} weight, the {@code 1/hc} scale of the whole output before the split, SiLU on the low-rank
     * rows, then the up projection; the inject rows land in {@code post} pre-scaled, so the post map skips the
     * scale ({@code PostParams#gatedInjectPreScaled}).
     */
    private void mergedGate(int nTokens) {
        int rank = block.lowRank();
        int rowW = rank + hc;
        matVec.matmul(downInject, flat, 0, merged, 0, nTokens);
        CpuOps.scale(merged, 0, nTokens * rowW, block.lowRankScale());
        for (int t = 0; t < nTokens; t++) {
            int s = t * rowW;
            System.arraycopy(merged, s + rank, post, t * hc, hc);
            System.arraycopy(merged, s, lowRank, t * rank, rank);
        }
        CpuOps.unary(ActOp.SILU, ActParams.NONE, lowRank, 0, lowRank, 0, nTokens * rank);
        matVec.matmul(up, lowRank, 0, gate, 0, nTokens);
    }

    /**
     * Grouped RMS norm of the streams into {@code flat}: each {@code n_embd} stream row is normalized and scaled by
     * gamma row {@code h} of {@code [n_embd, hc]} (qwen4exp.cpp:325-328).
     */
    private void groupedNorm(int nTokens, float[] stream, int streamOff) {
        final float eps = block.normEps();
        CpuParallel.parallelFor(nTokens, TOKEN_GRAIN, (from, to) -> NormMath.rowNorm(NormKind.RMS, eps,
                (to - from) * hc, nEmbd, nEmbd, hc, stream, streamOff + from * streamWidth, norm, null, flat,
                from * streamWidth));
    }

    /**
     * {@code xout[xoutOff + (t*hc + dst)*n_embd + i] = y[yOff + t*n_embd + i]*p_dst + R} with the coefficients of
     * the last {@link #collapse}. {@code xout} may be {@code x} at the same offset; otherwise the two must not
     * overlap.
     *
     * @throws IllegalStateException    for the head site, which has no post map
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]}
     */
    void expand(int nTokens, float[] y, int yOff, float[] x, int xOff, float[] xout, int xoutOff) {
        if (block.post() == null) {
            throw new IllegalStateException("hyper-connection " + block + " has no post map");
        }
        checkTokens(nTokens);
        CpuHyperConnections.post(block.post(), hc, nEmbd, nTokens, y, yOff, x, xOff, post, 0, comb, 0, xout, xoutOff);
    }

    private void checkTokens(int nTokens) {
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException("hyper-connection " + block + ": token count " + nTokens
                    + " is outside [1, " + ubatch + "]");
        }
    }

    private int rows(long width) {
        long n = Math.multiplyExact((long) ubatch, Math.max(1L, width));
        if (n > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("hyper-connection " + block + ": a temporary of " + n
                    + " floats exceeds a Java array; lower neuralfree.ubatch");
        }
        return (int) n;
    }

    private static CpuTensorRef tensor(CpuWeights weights, String arch, int il, TensorRef ref) {
        CpuTensorRef t = weights.get(ref);
        if (t == null) {
            throw new MissingTensorException(arch, il, ref.name());
        }
        return t;
    }

    private static float[] vector(CpuWeights weights, String arch, int il, TensorRef ref, int expected) {
        float[] v = weights.floats(ref);
        if (v == null) {
            throw new MissingTensorException(arch, il, ref.name());
        }
        if (v.length != expected) {
            throw new IllegalArgumentException("hyper-connection tensor " + ref.name() + " decodes to " + v.length
                    + " floats, expected " + expected);
        }
        return v;
    }

    @Override
    public String toString() {
        return "CpuHcSite{" + block + '}';
    }
}
