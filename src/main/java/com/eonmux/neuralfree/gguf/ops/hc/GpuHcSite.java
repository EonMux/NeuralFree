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
import com.eonmux.neuralfree.gguf.ops.fused.FusedKernels;
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
    /** Merged {@code [down | inject]} weight; null unless the merged gated form. */
    private final GpuWeight downInject;

    private final ScratchSlot flat;
    private final ScratchSlot mixes;
    private final ScratchSlot pre;
    private final ScratchSlot post;
    private final ScratchSlot comb;
    private final ScratchSlot lowRank;
    /** Merged {@code [down | inject]} output rows before the split; null unless the merged gated form. */
    private final ScratchSlot merged;
    private final ScratchSlot gate;

    private final PreparedDispatch coefficients;
    private final PreparedDispatch collapse;
    private final PreparedDispatch expand;
    /** Fused {@code LLM_HcPostNorm} (expand then the leading norm of the next site); null when unused. */
    private final PreparedDispatch expandNorm;

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
        this.down = block.down() != null ? resident(weights, arch, il, block.down()) : null;
        this.up = kind.gated() ? resident(weights, arch, il, block.up()) : null;
        this.inject = block.inject() != null ? resident(weights, arch, il, block.inject()) : null;
        this.downInject = block.downInject() != null ? resident(weights, arch, il, block.downInject()) : null;

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
        this.merged = downInject != null
                ? scratch.temp(HcSubBlock.TEMP_MERGED_DOWN, rows(block.lowRank() + hc)) : null;
        this.gate = kind.gated() ? scratch.temp(HcSubBlock.TEMP_GATE, rows(streamWidth)) : null;

        PreparedDispatch c = null;
        PreparedDispatch p = null;
        PreparedDispatch e = null;
        PreparedDispatch n = null;
        try {
            if (kind.projected()) {
                c = prepareCoefficients(block.coefficients(), scale, base, ph);
            }
            p = prepareCollapse(block.pre(), ph);
            if (kind.expands()) {
                e = prepareExpand(block.post(), ph);
                n = prepareExpandNorm(block.post(), ph);
            }
        } catch (RuntimeException ex) {
            Closeables.close(n, Closeables.close(e, Closeables.close(p, Closeables.close(c, ex))));
            throw ex;
        }
        this.coefficients = c;
        this.collapse = p;
        this.expand = e;
        this.expandNorm = n;
    }

    /**
     * The fused expansion and norm of the consuming site's leading row norm ({@link FusedKernels#HC_POST_NORM}),
     * or null when the backend lacks the kernel, the stream count is outside its bound, or the site does not expand.
     * The norm weight and its row count are bound by {@link #expandNorm} per call.
     */
    private PreparedDispatch prepareExpandNorm(PostParams p, GpuPlaceholders ph) {
        if (!block.kind().expands() || !FusedKernels.hcPostNormUsable(hc)
                || !ops.fused(FusedKernels.HC_POST_NORM)) {
            return null;
        }
        Object[] a = FusedKernels.hcPostNormArgs(ubatch, nEmbd, hc, p.hasComb(), p.postAct(), p.postInScale(),
                p.postMag(), block.normEps(), 0, normScope(), 0, 0, post.offset(), p.hasComb() ? comb.offset() : 0,
                0, 0,
                ph.readOnlyFloats(), ph.readOnlyFloats(), post.buffer(),
                p.hasComb() ? comb.buffer() : ph.readOnlyFloats(), ph.readOnlyFloats(), ph.readOnlyFloats(),
                ph.readOnlyFloats());
        return ops.prepare(FusedKernels.HC_POST_NORM, new long[]{FusedKernels.hcPostNormGlobal(ubatch, local)},
                new long[]{local}, a);
    }

    /** Whether this site's expansion has a fused expand-and-norm launch. */
    public boolean canExpandNorm() {
        return expandNorm != null;
    }

    /** Whether this site's collapse starts with a row norm over the stream that {@link #collapseNormed} skips. */
    public boolean hasLeadingNorm() {
        return flat != null && (block.kind().projected() || block.kind().gated());
    }

    /** The stream-norm temporaries of this site (the buffer the leading norm writes). */
    public GpuBuffer flatBuffer() {
        return flat == null ? null : flat.buffer();
    }

    /** First element of the leading-norm rows in {@link #flatBuffer()}; 0 when absent. */
    public int flatOffset() {
        return flat == null ? 0 : flat.offset();
    }

    /** The per-head gamma buffer of this site's leading norm, or null for a weightless norm. */
    public GpuBuffer normGamma() {
        return norm == null ? null : norm.shard(0);
    }

    /** Weight rows of {@link #normGamma()} ({@code hc}), or 0 for a weightless norm. */
    public int normGammaRows() {
        return norm == null ? 0 : hc;
    }

    /** Epsilon of this site's leading row norm. */
    public float normEps() {
        return block.normEps();
    }

    /** Norm axis of this site's leading row norm: 0 per stream row (gated), 1 per token (projected). */
    public int normScope() {
        return block.kind().gated() ? 0 : 1;
    }

    /**
     * Runs the fused expand and leading norm: {@code xout} receives the expanded stream and {@code normOut} the
     * row-normed stream rows, both in one launch. The norm weight is {@code gamma} of {@code gammaRows} rows of
     * {@code n_embd} values; {@code gammaRows == 0} runs the weightless norm.
     *
     * @throws IllegalStateException when the site has no fused expansion
     */
    public void expandNorm(int nTokens, GpuBuffer y, int yOff, GpuBuffer x, int xOff, GpuBuffer xout, int xoutOff,
                           GpuBuffer gamma, int gammaRows, int scope, float eps, GpuBuffer normOut, int normOff) {
        if (expandNorm == null) {
            throw new IllegalStateException("hyper-connection " + block + " has no fused expansion");
        }
        checkTokens(nTokens);
        HcKernels.PostParams p = block.post();
        expandNorm.setInt(FusedKernels.HC_ARG_N_TOKENS, nTokens);
        expandNorm.setInt(FusedKernels.HC_ARG_HAS_W, gammaRows > 0 ? 1 : 0);
        expandNorm.setInt(FusedKernels.HC_ARG_W_ROWS, Math.max(1, gammaRows));
        expandNorm.setInt(FusedKernels.HC_ARG_SCOPE, scope);
        expandNorm.setFloat(FusedKernels.HC_ARG_EPS, eps);
        expandNorm.setBuffer(FusedKernels.HC_ARG_Y, y);
        expandNorm.setBuffer(FusedKernels.HC_ARG_X, x);
        expandNorm.setBuffer(FusedKernels.HC_ARG_XOUT, xout);
        expandNorm.setBuffer(FusedKernels.HC_ARG_XN, normOut);
        expandNorm.setBuffer(FusedKernels.HC_ARG_GAMMA,
                gamma != null ? gamma : placeholders.readOnlyFloats());
        expandNorm.setInt(FusedKernels.HC_ARG_Y_OFF, yOff);
        expandNorm.setInt(FusedKernels.HC_ARG_X_OFF, xOff);
        expandNorm.setInt(FusedKernels.HC_ARG_XOUT_OFF, xoutOff);
        expandNorm.setInt(FusedKernels.HC_ARG_XN_OFF, normOff);
        expandNorm.setInt(FusedKernels.HC_ARG_COMB_OFF, p.hasComb() ? comb.offset() : 0);
        expandNorm.setGlobal(FusedKernels.hcPostNormGlobal(nTokens, local));
        expandNorm.dispatch();
    }

    /**
     * The steps of {@link #collapse} after its leading row norm, which the caller produced into {@link #flatBuffer()}
     * through {@link #expandNorm}.
     *
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]}
     * @throws IllegalStateException    when this site's collapse has no leading norm
     */
    public void collapseNormed(int nTokens, GpuBuffer stream, int streamOff, GpuBuffer out, int outOff) {
        if (!hasLeadingNorm()) {
            throw new IllegalStateException("hyper-connection " + block + " has no leading row norm");
        }
        checkTokens(nTokens);
        HcSubBlock.Kind kind = block.kind();
        if (kind.projected()) {
            ops.matmul(fn, flat.buffer(), flat.offset(), mixes.buffer(), mixes.offset(), nTokens);
            coefficients.setInt(HcKernels.COEF_ARG_N_TOKENS, nTokens);
            coefficients.setGlobal(HcKernels.perTokenGlobal(nTokens, local));
            coefficients.dispatch();
        } else if (kind.gated()) {
            int rank = block.lowRank();
            if (downInject != null) {
                mergedGate(nTokens);
            } else {
                ops.matmul(down, flat.buffer(), flat.offset(), lowRank.buffer(), lowRank.offset(), nTokens);
                ops.scale(lowRank.buffer(), lowRank.offset(), nTokens * rank, block.lowRankScale());
                ops.unary(ActOp.SILU, ActParams.NONE, lowRank.buffer(), lowRank.offset(), lowRank.buffer(),
                        lowRank.offset(), nTokens * rank);
                ops.matmul(up, lowRank.buffer(), lowRank.offset(), gate.buffer(), gate.offset(), nTokens);
            }
        }
        dispatchCollapse(nTokens, stream, streamOff,
                kind.projected() ? pre.buffer() : gate.buffer(),
                kind.projected() ? pre.offset() : gate.offset(), out, outOff);
        if (inject != null) {
            ops.matmul(inject, flat.buffer(), flat.offset(), post.buffer(), post.offset(), nTokens);
        }
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
     * Computes the pre, post and comb coefficients of the last {@code nTokens} stream rows without collapsing them
     * ({@code LLM_HC_PrePostComb}); {@link #collapseWithPre} then collapses with the pre values.
     *
     * @throws IllegalStateException    when the site is not projected (the gated and mean variants have no mixes)
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]}
     */
    public void computeCoefficients(int nTokens, GpuBuffer stream, int streamOff) {
        if (!block.kind().projected()) {
            throw new IllegalStateException("hyper-connection " + block + " projects no mixes");
        }
        checkTokens(nTokens);
        ops.norm(NormKind.RMS, null, null, block.normEps(), stream, streamOff, flat.buffer(), flat.offset(), nTokens,
                streamWidth);
        ops.matmul(fn, flat.buffer(), flat.offset(), mixes.buffer(), mixes.offset(), nTokens);
        coefficients.setInt(HcKernels.COEF_ARG_N_TOKENS, nTokens);
        coefficients.setGlobal(HcKernels.perTokenGlobal(nTokens, local));
        coefficients.dispatch();
    }

    /** The pre coefficients of the last collapse or {@link #computeCoefficients}. */
    public GpuBuffer preBuffer() {
        if (pre == null) {
            throw new IllegalStateException("hyper-connection " + block + " has no pre coefficients");
        }
        return pre.buffer();
    }

    /** First element of the pre coefficients in {@link #preBuffer()}. */
    public int preOffset() {
        if (pre == null) {
            throw new IllegalStateException("hyper-connection " + block + " has no pre coefficients");
        }
        return pre.offset();
    }

    /**
     * Collapses {@code nTokens} stream rows with pre coefficients {@code pre[preOff ..]}, as a mix computed by
     * {@link #computeCoefficients} on another sub-block would.
     *
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]}
     */
    public void collapseWithPre(int nTokens, GpuBuffer stream, int streamOff, GpuBuffer pre, int preOff,
                                GpuBuffer out, int outOff) {
        if (!block.kind().projected()) {
            throw new IllegalStateException("hyper-connection " + block + " collapses with its own mixes");
        }
        checkTokens(nTokens);
        Objects.requireNonNull(pre, "pre");
        dispatchCollapse(nTokens, stream, streamOff, pre, preOff, out, outOff);
    }

    private void dispatchCollapse(int nTokens, GpuBuffer x, int xOff, GpuBuffer w, int wOff, GpuBuffer out,
                                  int outOff) {
        HcSubBlock.Kind kind = block.kind();
        collapse.setInt(HcKernels.PRE_ARG_N_TOKENS, nTokens);
        collapse.setGlobal(HcKernels.elementGlobal(nTokens, nEmbd, local));
        collapse.setBuffer(HcKernels.PRE_ARG_X, kind.gated() ? flat.buffer() : x);
        collapse.setInt(HcKernels.PRE_ARG_X_OFF, kind.gated() ? flat.offset() : xOff);
        collapse.setBuffer(HcKernels.PRE_ARG_W_OR_GATE, w);
        collapse.setInt(HcKernels.PRE_ARG_W_OFF, wOff);
        collapse.setBuffer(HcKernels.PRE_ARG_OUT, out);
        collapse.setInt(HcKernels.PRE_ARG_OUT_OFF, outOff);
        collapse.dispatch();
    }

    /**
     * Collapses {@code nTokens} stream rows ({@code stream[streamOff + t*S ...]}) into
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
            if (downInject != null) {
                mergedGate(nTokens);
            } else {
                ops.matmul(down, flat.buffer(), flat.offset(), lowRank.buffer(), lowRank.offset(), nTokens);
                ops.scale(lowRank.buffer(), lowRank.offset(), nTokens * rank, block.lowRankScale());
                ops.unary(ActOp.SILU, ActParams.NONE, lowRank.buffer(), lowRank.offset(), lowRank.buffer(),
                        lowRank.offset(), nTokens * rank);
                ops.matmul(up, lowRank.buffer(), lowRank.offset(), gate.buffer(), gate.offset(), nTokens);
            }
        }
        dispatchCollapse(nTokens, stream, streamOff, kind.projected() ? pre.buffer() : kind.gated() ? gate.buffer()
                        : placeholders.readOnlyFloats(), kind.projected() ? pre.offset() : kind.gated() ? gate.offset()
                        : 0, out, outOff);
        if (inject != null) {
            ops.matmul(inject, flat.buffer(), flat.offset(), post.buffer(), post.offset(), nTokens);
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
        ops.matmul(downInject, flat.buffer(), flat.offset(), merged.buffer(), merged.offset(), nTokens);
        ops.scale(merged.buffer(), merged.offset(), nTokens * rowW, block.lowRankScale());
        ops.copy2d(merged.buffer(), merged.offset() + rank, rowW, post.buffer(), post.offset(), hc, nTokens, hc);
        ops.copy2d(merged.buffer(), merged.offset(), rowW, lowRank.buffer(), lowRank.offset(), rank, nTokens, rank);
        ops.unary(ActOp.SILU, ActParams.NONE, lowRank.buffer(), lowRank.offset(), lowRank.buffer(),
                lowRank.offset(), nTokens * rank);
        ops.matmul(up, lowRank.buffer(), lowRank.offset(), gate.buffer(), gate.offset(), nTokens);
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
        RuntimeException first = Closeables.closeAll(Arrays.asList(coefficients, collapse, expand, expandNorm),
                null);
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuHcSite{" + block + '}';
    }
}
