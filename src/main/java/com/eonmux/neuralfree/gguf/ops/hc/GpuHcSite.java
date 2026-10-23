package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.CoefficientParams;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.PostParams;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.PreParams;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.Arrays;
import java.util.Objects;

/**
 * One bound hyper-connection collapse site on the GPU ({@link HcSubBlock}): {@link #collapse} turns {@code hc}
 * streams per token into one {@code n_embd} row and leaves the site's post coefficients in its temporaries;
 * {@link #expand} writes {@code Xout = post(y, X)} with them (replacing the residual add).
 *
 * <p>Dispatches per {@link #collapse} of {@code T} tokens, {@code S = hc*n_embd}:
 * <ul>
 *   <li>projected variants: {@code LLM_RowNorm} (RMS, no weight, T rows of S) into {@code hc.flat}; the projection
 *       of {@code fn} into {@code hc.mixes}; {@code LLM_HC_PrePostComb}; {@code LLM_HC_Pre} (weights) from the
 *       stream;</li>
 *   <li>gated variants: {@code LLM_RowNorm} (RMS, {@code hc} weight rows, T*hc rows of n_embd) into
 *       {@code hc.flat}, launched from {@link NormKernels#rowNormArgs} because the gammas may be stored flattened
 *       ({@link HcSubBlock#norm()}); {@code down} into {@code hc.lowrank}; scale by {@code 1/hc} and SiLU in place; {@code up}
 *       into {@code hc.gate}; {@code LLM_HC_Pre} (gated) from {@code hc.flat}; for a sub-block, {@code inject}
 *       into {@code hc.post};</li>
 *   <li>mean head: {@code LLM_HC_Pre} (mean) from the stream.</li>
 * </ul>
 * {@link #expand} is one {@code LLM_HC_Post}. The HC kernels are {@link PreparedDispatch}es bound at construction;
 * per call only the token count, the global size and the caller's buffers and offsets are re-set. Outputs that
 * {@code LLM_HC_PrePostComb} does not produce for the variant are bound to the writable placeholder, and the comb of
 * a post without mixing to the read-only one. Not thread-safe; one forward pass at a time.</p>
 */
public final class GpuHcSite implements AutoCloseable {

    private final HcSubBlock block;
    private final GpuOps ops;
    private final GpuPlaceholders placeholders;
    private final int ubatch;
    private final int local;
    private final int hc;
    private final int nEmbd;
    private final int streamWidth;

    private final GpuWeight fn;
    private final GpuWeight norm;
    private final GpuWeight down;
    private final GpuWeight up;
    private final GpuWeight inject;

    private final ScratchSlot flat;
    private final ScratchSlot mixes;
    private final ScratchSlot pre;
    private final ScratchSlot post;
    private final ScratchSlot comb;
    private final ScratchSlot lowRank;
    private final ScratchSlot gate;

    private final PreparedDispatch coefficients;
    private final PreparedDispatch collapse;
    private final PreparedDispatch expand;

    /**
     * @param ctx     model context
     * @param weights weights holding the site's tensors
     * @param block   resolved site
     * @throws MissingTensorException   when a tensor of the site is not resident
     * @throws IllegalArgumentException when a resident tensor does not match the site's shape or type rules
     */
    public GpuHcSite(GpuForwardContext ctx, GpuWeights weights, HcSubBlock block) {
        this.block = Objects.requireNonNull(block, "block");
        this.ops = ctx.ops();
        GpuScratch scratch = ctx.scratch();
        GpuPlaceholders ph = ctx.placeholders();
        this.placeholders = ph;
        String arch = ctx.spec().arch().ggufName();
        this.ubatch = scratch.ubatch();
        this.local = ops.local();
        this.hc = block.hc();
        this.nEmbd = block.nEmbd();
        this.streamWidth = block.streamWidth();
        HcSubBlock.Kind kind = block.kind();

        int il = block.il();
        this.fn = kind.projected() ? resident(weights, arch, il, block.fn()) : null;
        GpuWeight scale = kind.projected() ? vector(weights, arch, il, block.scale()) : null;
        GpuWeight base = kind.projected() ? vector(weights, arch, il, block.base()) : null;
        this.norm = kind.gated() ? vector(weights, arch, il, block.norm()) : null;
        this.down = kind.gated() ? resident(weights, arch, il, block.down()) : null;
        this.up = kind.gated() ? resident(weights, arch, il, block.up()) : null;
        this.inject = block.inject() != null ? resident(weights, arch, il, block.inject()) : null;

        boolean needsFlat = kind.projected() || kind.gated();
        this.flat = needsFlat ? scratch.temp(HcSubBlock.TEMP_FLAT, rows(streamWidth)) : null;
        this.mixes = kind.projected() ? scratch.temp(HcSubBlock.TEMP_MIXES, rows(block.mixDim())) : null;
        boolean needsPre = kind.projected();
        this.pre = needsPre ? scratch.temp(HcSubBlock.TEMP_PRE, rows(hc)) : null;
        boolean needsPost = kind.expands();
        this.post = needsPost ? scratch.temp(HcSubBlock.TEMP_POST, rows(hc)) : null;
        boolean hasComb = block.post() != null && block.post().hasComb();
        this.comb = hasComb ? scratch.temp(HcSubBlock.TEMP_COMB, rows(hc * hc)) : null;
        this.lowRank = kind.gated() ? scratch.temp(HcSubBlock.TEMP_LOW_RANK, rows(block.lowRank())) : null;
        this.gate = kind.gated() ? scratch.temp(HcSubBlock.TEMP_GATE, rows(streamWidth)) : null;

        PreparedDispatch c = null;
        PreparedDispatch p = null;
        PreparedDispatch e = null;
        try {
            if (kind.projected()) {
                c = prepareCoefficients(block.coefficients(), scale, base, ph);
            }
            p = prepareCollapse(block.pre(), ph);
            if (kind.expands()) {
                e = prepareExpand(block.post(), ph);
            }
        } catch (RuntimeException ex) {
            Closeables.close(p, Closeables.close(c, ex));
            throw ex;
        }
        this.coefficients = c;
        this.collapse = p;
        this.expand = e;
    }

    private PreparedDispatch prepareCoefficients(CoefficientParams p, GpuWeight scale, GpuWeight base,
                                                 GpuPlaceholders ph) {
        boolean hasPost = p.nOut() >= HcKernels.OUT_PRE_POST;
        boolean hasComb = p.nOut() == HcKernels.OUT_PRE_POST_COMB;
        Object[] a = new Object[HcKernels.COEF_ARG_COUNT];
        a[HcKernels.COEF_ARG_N_TOKENS] = ubatch;
        a[HcKernels.COEF_ARG_HC] = hc;
        a[HcKernels.COEF_ARG_EPS] = p.eps();
        a[HcKernels.COEF_ARG_ITERS] = p.iters();
        a[HcKernels.COEF_ARG_N_OUT] = p.nOut();
        a[HcKernels.COEF_ARG_POST_MAG] = p.postMag();
        a[HcKernels.COEF_ARG_POST_EPS] = p.postEps();
        a[HcKernels.COEF_ARG_MIXES_OFF] = mixes.offset();
        a[HcKernels.COEF_ARG_PRE_OFF] = pre.offset();
        a[HcKernels.COEF_ARG_POST_OFF] = hasPost ? post.offset() : 0;
        a[HcKernels.COEF_ARG_COMB_OFF] = hasComb ? comb.offset() : 0;
        a[HcKernels.COEF_ARG_MIXES] = mixes.buffer();
        a[HcKernels.COEF_ARG_SCALE] = scale.shard(0);
        a[HcKernels.COEF_ARG_BASE] = base.shard(0);
        a[HcKernels.COEF_ARG_PRE] = pre.buffer();
        a[HcKernels.COEF_ARG_POST] = hasPost ? post.buffer() : ph.scratchFloats();
        a[HcKernels.COEF_ARG_COMB] = hasComb ? comb.buffer() : ph.scratchFloats();
        return ops.prepare(HcKernels.HC_PRE_POST_COMB, new long[]{HcKernels.perTokenGlobal(ubatch, local)},
                new long[]{local}, a);
    }

    private PreparedDispatch prepareCollapse(PreParams p, GpuPlaceholders ph) {
        Object[] a = new Object[HcKernels.PRE_ARG_COUNT];
        a[HcKernels.PRE_ARG_N_TOKENS] = ubatch;
        a[HcKernels.PRE_ARG_N_EMBD] = nEmbd;
        a[HcKernels.PRE_ARG_HC] = hc;
        a[HcKernels.PRE_ARG_MODE] = p.mode();
        a[HcKernels.PRE_ARG_GSCALE] = p.gscale();
        GpuBuffer weightsOrGate;
        int wOff;
        if (p.mode() == HcKernels.PRE_MODE_WEIGHTS) {
            weightsOrGate = pre.buffer();
            wOff = pre.offset();
        } else if (p.mode() == HcKernels.PRE_MODE_GATED) {
            weightsOrGate = gate.buffer();
            wOff = gate.offset();
        } else {
            weightsOrGate = ph.readOnlyFloats();
            wOff = 0;
        }
        boolean gated = p.mode() == HcKernels.PRE_MODE_GATED;
        a[HcKernels.PRE_ARG_X_OFF] = gated ? flat.offset() : 0;
        a[HcKernels.PRE_ARG_W_OFF] = wOff;
        a[HcKernels.PRE_ARG_OUT_OFF] = 0;
        a[HcKernels.PRE_ARG_X] = gated ? flat.buffer() : ph.readOnlyFloats();
        a[HcKernels.PRE_ARG_W_OR_GATE] = weightsOrGate;
        a[HcKernels.PRE_ARG_OUT] = ph.scratchFloats();
        return ops.prepare(HcKernels.HC_PRE, new long[]{HcKernels.elementGlobal(ubatch, nEmbd, local)},
                new long[]{local}, a);
    }

    private PreparedDispatch prepareExpand(PostParams p, GpuPlaceholders ph) {
        Object[] a = new Object[HcKernels.POST_ARG_COUNT];
        a[HcKernels.POST_ARG_N_TOKENS] = ubatch;
        a[HcKernels.POST_ARG_N_EMBD] = nEmbd;
        a[HcKernels.POST_ARG_HC] = hc;
        a[HcKernels.POST_ARG_HAS_COMB] = p.hasComb() ? 1 : 0;
        a[HcKernels.POST_ARG_POST_ACT] = p.postAct();
        a[HcKernels.POST_ARG_POST_IN_SCALE] = p.postInScale();
        a[HcKernels.POST_ARG_POST_MAG] = p.postMag();
        a[HcKernels.POST_ARG_Y_OFF] = 0;
        a[HcKernels.POST_ARG_X_OFF] = 0;
        a[HcKernels.POST_ARG_POST_OFF] = post.offset();
        a[HcKernels.POST_ARG_COMB_OFF] = p.hasComb() ? comb.offset() : 0;
        a[HcKernels.POST_ARG_XOUT_OFF] = 0;
        a[HcKernels.POST_ARG_Y] = ph.readOnlyFloats();
        a[HcKernels.POST_ARG_X] = ph.readOnlyFloats();
        a[HcKernels.POST_ARG_POST] = post.buffer();
        a[HcKernels.POST_ARG_COMB] = p.hasComb() ? comb.buffer() : ph.readOnlyFloats();
        a[HcKernels.POST_ARG_XOUT] = ph.scratchFloats();
        return ops.prepare(HcKernels.HC_POST, new long[]{HcKernels.elementGlobal(ubatch, nEmbd, local)},
                new long[]{local}, a);
    }

    public HcSubBlock block() {
        return block;
    }

    /**
     * Collapses {@code nTokens} stream rows ({@code stream[streamOff + t*S ...]}, {@code S = hc*n_embd}) into
     * {@code out[outOff + t*n_embd ...]} and computes the post coefficients {@link #expand} uses. {@code out} must
     * not overlap the stream.
     *
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]}
     */
    public void collapse(int nTokens, GpuBuffer stream, int streamOff, GpuBuffer out, int outOff) {
        checkTokens(nTokens);
        HcSubBlock.Kind kind = block.kind();
        if (kind.projected()) {
            ops.norm(NormKind.RMS, null, null, block.normEps(), stream, streamOff, flat.buffer(), flat.offset(),
                    nTokens, streamWidth);
            ops.matmul(fn, flat.buffer(), flat.offset(), mixes.buffer(), mixes.offset(), nTokens);
            coefficients.setInt(HcKernels.COEF_ARG_N_TOKENS, nTokens);
            coefficients.setGlobal(HcKernels.perTokenGlobal(nTokens, local));
            coefficients.dispatch();
        } else if (kind.gated()) {
            int rank = block.lowRank();
            int nRows = nTokens * hc;
            ops.enqueue(GpuOps.ROW_NORM, new long[]{(long) nRows * local}, new long[]{local},
                    NormKernels.rowNormArgs(NormKind.RMS, block.normEps(), nRows, nEmbd, nEmbd, stream, streamOff,
                            flat.buffer(), flat.offset(), norm.shard(0), null, hc, placeholders.readOnlyFloats()));
            ops.matmul(down, flat.buffer(), flat.offset(), lowRank.buffer(), lowRank.offset(), nTokens);
            ops.scale(lowRank.buffer(), lowRank.offset(), nTokens * rank, block.lowRankScale());
            ops.unary(ActOp.SILU, ActParams.NONE, lowRank.buffer(), lowRank.offset(), lowRank.buffer(),
                    lowRank.offset(), nTokens * rank);
            ops.matmul(up, lowRank.buffer(), lowRank.offset(), gate.buffer(), gate.offset(), nTokens);
        }
        collapse.setInt(HcKernels.PRE_ARG_N_TOKENS, nTokens);
        collapse.setGlobal(HcKernels.elementGlobal(nTokens, nEmbd, local));
        if (!kind.gated()) {
            collapse.setBuffer(HcKernels.PRE_ARG_X, stream);
            collapse.setInt(HcKernels.PRE_ARG_X_OFF, streamOff);
        }
        collapse.setBuffer(HcKernels.PRE_ARG_OUT, out);
        collapse.setInt(HcKernels.PRE_ARG_OUT_OFF, outOff);
        collapse.dispatch();
        if (inject != null) {
            ops.matmul(inject, flat.buffer(), flat.offset(), post.buffer(), post.offset(), nTokens);
        }
    }

    /**
     * {@code xout[xoutOff + (t*hc + dst)*n_embd + i] = y[yOff + t*n_embd + i]*p_dst + R} with the coefficients of
     * the last {@link #collapse} ({@code R} mixes or keeps the streams of {@code x}). {@code xout} may be {@code x}
     * at the same offset; otherwise the two must not overlap.
     *
     * @throws IllegalStateException    for the head site, which has no post map
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]}
     */
    void expand(int nTokens, GpuBuffer y, int yOff, GpuBuffer x, int xOff, GpuBuffer xout, int xoutOff) {
        if (expand == null) {
            throw new IllegalStateException("hyper-connection " + block + " has no post map");
        }
        checkTokens(nTokens);
        expand.setInt(HcKernels.POST_ARG_N_TOKENS, nTokens);
        expand.setGlobal(HcKernels.elementGlobal(nTokens, nEmbd, local));
        expand.setBuffer(HcKernels.POST_ARG_Y, y);
        expand.setInt(HcKernels.POST_ARG_Y_OFF, yOff);
        expand.setBuffer(HcKernels.POST_ARG_X, x);
        expand.setInt(HcKernels.POST_ARG_X_OFF, xOff);
        expand.setBuffer(HcKernels.POST_ARG_XOUT, xout);
        expand.setInt(HcKernels.POST_ARG_XOUT_OFF, xoutOff);
        expand.dispatch();
    }

    private void checkTokens(int nTokens) {
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException("hyper-connection " + block + ": token count " + nTokens
                    + " is outside [1, " + ubatch + "]");
        }
    }

    private long rows(long width) {
        return Math.multiplyExact((long) ubatch, Math.max(1L, width));
    }

    private static GpuWeight resident(GpuWeights weights, String arch, int il, TensorRef ref) {
        GpuWeight w = weights.find(ref.name());
        if (w == null) {
            throw new MissingTensorException(arch, il, ref.name());
        }
        if (w.cols() != ref.cols() || w.rows() != ref.rows()) {
            throw new IllegalArgumentException("hyper-connection tensor " + ref.name() + " is resident as " + w
                    + ", expected " + ref.cols() + " x " + ref.rows());
        }
        return w;
    }

    /** A small coefficient or gamma tensor the kernels read directly by element index: one F32 buffer. */
    private static GpuWeight vector(GpuWeights weights, String arch, int il, TensorRef ref) {
        GpuWeight w = resident(weights, arch, il, ref);
        if (w.type() != GGMLType.F32 || !w.isSingleShard()) {
            throw new IllegalArgumentException("hyper-connection tensor " + ref.name()
                    + " must be one F32 buffer for the HC kernels, got " + w);
        }
        return w;
    }

    /** Releases the prepared dispatches; the scratch temporaries and weights stay with their owners. */
    @Override
    public void close() {
        RuntimeException first = Closeables.closeAll(Arrays.asList(coefficients, collapse, expand), null);
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuHcSite{" + block + '}';
    }
}
