package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.ops.fused.FusedKernels;
import com.eonmux.neuralfree.gguf.ops.fused.GpuMultiTask;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.DualChunkRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Arrays;
import java.util.Objects;

/**
 * The query-side rotation stages of {@link GpuAttentionMixer}: the Q and K norms of every {@link QkNormKind} and
 * gemma's weightless V norm, the RoPE or FoPE rotation of Q and K (NoPE layers skip it), and their fused forms —
 * {@link FusedKernels#RMS_NORM_ROPE} or the persistent multi-task kernel for the norms with the rotations, under
 * {@code neuralfree.gpu.fused}. Also the layer's RoPE frequency factors, which the fused store and the context
 * shift read. Not thread-safe; driven by the owning mixer's bind.
 */
final class GpuNormRopeStages implements AutoCloseable {

    private final GpuAttentionMixer mixer;

    /** Persistent launch of the Q/K norms with their rotations and the V norm; null when unused. */
    private GpuMultiTask normRopeBatch;
    private GpuBuffer ropeFactorBuf;
    private GpuBuffer computedFactorBuf;
    private float[] computedFactorValues;
    private GpuBuffer fopeCosBuf;
    private GpuBuffer fopeSinBuf;
    private GpuBuffer fopeInvBuf;
    private float[] fopeInvValues;

    GpuNormRopeStages(GpuAttentionMixer mixer) {
        this.mixer = Objects.requireNonNull(mixer, "mixer");
    }

    @Override
    public void close() {
        if (normRopeBatch != null) {
            normRopeBatch.close();
            normRopeBatch = null;
        }
    }

    /** The layer's F32 frequency factors the fused K store and the context shift reuse, or null when there are none. */
    GpuBuffer factors() {
        return ropeFactorBuf;
    }

    /** The Q and K norms the graph applies ({@link QkNormGeometry#applies}); {@code skipK} leaves K to the store. */
    void bindNorms(boolean skipK) {
        QkNormKind kind = mixer.spec().qkNorm();
        if (kind == QkNormKind.NONE) {
            return;
        }
        boolean hasQ = mixer.weights().layerOrNull(mixer.layer(), TensorNames.weight(TensorNames.ATTN_Q_NORM)) != null;
        boolean hasK = mixer.weights().layerOrNull(mixer.layer(), TensorNames.weight(TensorNames.ATTN_K_NORM)) != null;
        if (QkNormGeometry.applies(kind, mixer.ctx().spec().arch(), false, hasQ, hasK)) {
            addQkNorm(geometry(kind, mixer.spec().nHead()), TensorNames.ATTN_Q_NORM, mixer.qBuf(), false);
        }
        if (!skipK && mixer.proj().projectsKv()
                && QkNormGeometry.applies(kind, mixer.ctx().spec().arch(), true, hasQ, hasK)) {
            addQkNorm(geometry(kind, mixer.spec().nHeadKv()), TensorNames.ATTN_K_NORM, mixer.kBuf(), true);
        }
    }

    /** The norm geometry of {@code kind} over {@code nHeads} heads of the layer. */
    QkNormGeometry geometry(QkNormKind kind, int nHeads) {
        return QkNormGeometry.of(kind, mixer.spec().headDimK(), nHeads, mixer.hparams().normEps(), mixer.rmsEps(),
                mixer.layer().il());
    }

    /** Gemma's weightless per-head RMS norm of V. */
    void bindVNorm() {
        if (mixer.spec().vNorm() && mixer.proj().projectsKv()) {
            mixer.input(mixer.normStep(NormKind.RMS, mixer.rmsEps(), mixer.vBuf(), mixer.spec().nHeadKv(),
                    mixer.spec().headDimV(), null, null, 1), true);
        }
    }

    /**
     * Q or K norm over the dense rows of {@code x} in geometry {@code g}; a weight of one scalar per head is expanded
     * to {@code n_heads} rows ({@link #broadcastPerHead}). The norm bias is loaded only when the graph applies it.
     */
    private void addQkNorm(QkNormGeometry g, String base, GpuBuffer x, boolean cacheWrite) {
        QkNormOperand op = operand(g, base);
        mixer.input(mixer.normStep(g.normKind(), g.eps(), x, g.rowsPerToken(), g.rowLen(), op.weight, op.bias,
                op.wRows), cacheWrite);
    }

    /** The normalized weight and bias buffers of a Q/K norm, expanded to per-head rows when needed. */
    QkNormOperand operand(QkNormGeometry g, String base) {
        GpuWeight w = mixer.weights().layerOrNull(mixer.layer(), TensorNames.weight(base));
        GpuWeight b = mixer.spec().qkNormBias() ? mixer.weights().layerOrNull(mixer.layer(), TensorNames.bias(base))
                : null;
        int rowLen = g.rowLen();
        int wRows = 1;
        GpuBuffer wb = null;
        GpuBuffer bb = null;
        boolean perHeadScalar = w != null && g.perHeadScalar(w.cols(), w.rows());
        if (perHeadScalar) {
            wRows = g.nHeads();
            wb = broadcastPerHead(w, g.nHeads(), rowLen);
        } else if (w != null) {
            wRows = g.weightRows(w.name(), w.rows() * w.cols());
            wb = mixer.vector(w, (long) wRows * rowLen);
        }
        if (b != null) {
            bb = perHeadScalar && g.perHeadScalar(b.cols(), b.rows())
                    ? broadcastPerHead(b, g.nHeads(), rowLen) : mixer.vector(b, (long) wRows * rowLen);
        }
        return new QkNormOperand(wb, bb, wRows);
    }

    /**
     * Binds the fused norm and rotation path of a layer whose Q and K norms are per-head RMS norms applied before
     * RoPE. With the multi-task kernel the Q norm+rotation, the K norm+rotation and the V norm run as one launch;
     * without it the norm+rotation of each projection is one {@link FusedKernels#RMS_NORM_ROPE} launch. {@code skipK}
     * leaves the K norm and rotation to the fused KV store; when that leaves the multi-task launch with a single
     * body (Q only, no V norm) the plain {@code RMS_NORM_ROPE} is used instead. Returns false when the layer's shapes
     * or the backend do not admit either form, leaving the unfused pair to the caller.
     */
    boolean bindFusedNormRope(GpuBuffer ropePos, boolean skipK) {
        RopeSpec rope = mixer.layer().rope();
        if (rope == null || mixer.spec().noRope() || rope.type() == RopeType.NONE) {
            return false;
        }
        QkNormKind kind = mixer.spec().qkNorm();
        if (kind == QkNormKind.NONE || rope.fope() != null) {
            /* The fused norm+rope kernels evaluate the analytic rotation; FoPE pairs come from coefficients. */
            return false;
        }
        boolean hasQ = mixer.weights().layerOrNull(mixer.layer(), TensorNames.weight(TensorNames.ATTN_Q_NORM)) != null;
        boolean hasK = mixer.weights().layerOrNull(mixer.layer(), TensorNames.weight(TensorNames.ATTN_K_NORM)) != null;
        boolean kv = mixer.proj().projectsKv();
        if (!QkNormGeometry.applies(kind, mixer.ctx().spec().arch(), false, hasQ, hasK)) {
            return false;
        }
        QkNormGeometry qg = geometry(kind, mixer.spec().nHead());
        QkNormGeometry kg = kv && QkNormGeometry.applies(kind, mixer.ctx().spec().arch(), true, hasQ, hasK)
                ? geometry(kind, mixer.spec().nHeadKv()) : null;
        if (!perHeadRms(qg, mixer.spec().nHead(), mixer.spec().headDimK())
                || (kv && !skipK && kg == null)
                || (kg != null && !perHeadRms(kg, mixer.spec().nHeadKv(), mixer.spec().headDimK()))) {
            return false;
        }
        GpuBuffer ff = ropeFactors();
        ropeFactorBuf = ff;
        boolean hasFf = ff != null;
        GpuBuffer ffArg = hasFf ? ff : mixer.ops().placeholders().readOnlyFloats();
        QkNormOperand qop = operand(qg, TensorNames.ATTN_Q_NORM);
        QkNormOperand kop = kg != null ? operand(kg, TensorNames.ATTN_K_NORM) : null;
        GpuBuffer none = mixer.ops().placeholders().readOnlyFloats();

        boolean packed = (!skipK && kop != null) || mixer.spec().vNorm();
        if (GpuMultiTask.usable(mixer.ops()) && GpuMultiTask.localUsable(mixer.local()) && packed) {
            GpuMultiTask batch = new GpuMultiTask(mixer.ops(), mixer.ubatch(),
                    "attn.normrope." + mixer.layer().il());
            try {
                batch.normRope(rope, qg.eps(), mixer.spec().nHead(), mixer.spec().headDimK(), mixer.spec().qDim(),
                        mixer.spec().headDimK(), rope.nOffs(), 0, 0, qop.flags(), qop.wRows, hasFf, ropePos, ffArg,
                        qop.weight != null ? qop.weight : none, qop.bias != null ? qop.bias : none, mixer.qBuf(),
                        mixer.qBuf());
                if (!skipK && kop != null) {
                    batch.normRope(rope, kg.eps(), mixer.spec().nHeadKv(), mixer.spec().headDimK(),
                            mixer.spec().kDim(), mixer.spec().headDimK(), rope.nOffs(), 0, 0, kop.flags(), kop.wRows,
                            hasFf, ropePos, ffArg, kop.weight != null ? kop.weight : none,
                            kop.bias != null ? kop.bias : none, mixer.kBuf(), mixer.kBuf());
                }
                if (mixer.spec().vNorm()) {
                    batch.rowNorm(mixer.spec().nHeadKv(), mixer.spec().headDimV(), mixer.spec().headDimV(),
                            NormKind.RMS, 0, 1, 0, 0, mixer.rmsEps(), mixer.vBuf(), none, none, mixer.vBuf());
                }
                batch.build();
                normRopeBatch = batch;
                mixer.input(TokenStep.multiTask(batch), true);
                return true;
            } catch (RuntimeException e) {
                batch.close();
                if (mixer.ops().fusedMode().requiresFusion()) {
                    throw e;
                }
            }
        }
        if (!mixer.ops().fused(FusedKernels.RMS_NORM_ROPE)) {
            return false;
        }
        addFusedNormRope(rope, qg, qop, mixer.spec().qDim(), mixer.qBuf(), ropePos, ffArg, hasFf, none);
        if (!skipK && kg != null) {
            addFusedNormRope(rope, kg, kop, mixer.spec().kDim(), mixer.kBuf(), ropePos, ffArg, hasFf, none);
        }
        if (mixer.spec().vNorm()) {
            mixer.input(mixer.normStep(NormKind.RMS, mixer.rmsEps(), mixer.vBuf(), mixer.spec().nHeadKv(),
                    mixer.spec().headDimV(), null, null, 1), true);
        }
        return true;
    }

    private void addFusedNormRope(RopeSpec rope, QkNormGeometry g, QkNormOperand op, int rowStride, GpuBuffer x,
                                  GpuBuffer pos, GpuBuffer ff, boolean hasFf, GpuBuffer none) {
        Object[] args = FusedKernels.rmsNormRopeArgs(rope, mixer.ubatch(), g.nHeads(), mixer.spec().headDimK(),
                rowStride, g.rowLen(), rope.nOffs(), mixer.ubatch(), 0, 0, op.flags(), op.wRows, g.eps(), hasFf, pos,
                ff, op.weight != null ? op.weight : none, op.bias != null ? op.bias : none, x, x);
        mixer.input(mixer.step(FusedKernels.RMS_NORM_ROPE, args, FusedKernels.ROPE_ARG_N_TOKENS, 1,
                (long) g.nHeads() * mixer.local(), 0, FusedKernels.ROPE_ARG_POS_STRIDE), true);
    }

    static boolean perHeadRms(QkNormGeometry g, int nHeads, int headDim) {
        return g.normKind() == NormKind.RMS && g.rowsPerToken() == nHeads && g.rowLen() == headDim;
    }

    /**
     * A {@code [nHeads][rowLen]} F32 buffer, owned by this mixer, whose row {@code h} repeats head {@code h}'s scalar
     * of a {@code {1, nHeads}} tensor: the same product as ggml's broadcast multiply of the normalized
     * {@code [rowLen, nHeads, n_tokens]} rows. Filled once with one {@value GpuOps#COPY2D} per head reading its
     * scalar with source stride 0.
     */
    private GpuBuffer broadcastPerHead(GpuWeight w, int nHeads, int rowLen) {
        GpuBuffer scalars = mixer.vector(w, nHeads);
        GpuBuffer rows = mixer.resources().allocate((long) nHeads * rowLen * Float.BYTES);
        long[] global = {mixer.roundUp(rowLen)};
        long[] localSize = {mixer.local()};
        for (int h = 0; h < nHeads; h++) {
            mixer.ops().enqueue(GpuOps.COPY2D, global, localSize, rowLen, 1, 0, h, 1, h * rowLen, scalars, rows);
        }
        return rows;
    }

    /** The normalized weight and bias views of a Q/K norm, or placeholders when the norm is weightless. */
    static final class QkNormOperand {
        final GpuBuffer weight;
        final GpuBuffer bias;
        final int wRows;

        QkNormOperand(GpuBuffer weight, GpuBuffer bias, int wRows) {
            this.weight = weight;
            this.bias = bias;
            this.wRows = wRows;
        }

        int flags() {
            int f = 0;
            if (weight != null) {
                f |= NormKernels.FLAG_HAS_W;
            }
            if (bias != null) {
                f |= NormKernels.FLAG_HAS_B;
            }
            return f;
        }
    }

    // ------------------------------------------------------------------ RoPE

    void bindRope(GpuBuffer ropePos, boolean skipK) {
        RopeSpec rope = mixer.layer().rope();
        if (rope == null || mixer.spec().noRope() || rope.type() == RopeType.NONE) {
            return;
        }
        if (rope.fope() != null) {
            bindFope(rope, ropePos, skipK);
            return;
        }
        GpuBuffer ff = ropeFactors();
        ropeFactorBuf = ff;
        boolean hasFf = ff != null;
        GpuBuffer ffArg = hasFf ? ff : mixer.ops().placeholders().readOnlyFloats();
        if (mixer.dualChunk() != null) {
            bindDualChunkRope(rope, ffArg, hasFf);
            return;
        }
        mixer.input(ropeStep(rope, mixer.spec().nHead(), mixer.spec().qDim(), ropePos, ffArg, hasFf, mixer.qBuf()),
                false);
        if (!skipK && mixer.proj().projectsKv()) {
            mixer.input(ropeStep(rope, mixer.spec().nHeadKv(), mixer.spec().kDim(), ropePos, ffArg, hasFf,
                    mixer.kBuf()), true);
        }
    }

    /** Binds the FoPE rotation of Q and K ({@link RopeKernels#ROPE_FOPE}) with the model's coefficient tensors. */
    private void bindFope(RopeSpec rope, GpuBuffer ropePos, boolean skipK) {
        TensorRef cosRef = rope.fope().cosCoef();
        TensorRef sinRef = rope.fope().sinCoef();
        GpuWeight cos = mixer.weights().getOrNull(cosRef.name());
        GpuWeight sin = mixer.weights().getOrNull(sinRef.name());
        if (cos == null || sin == null) {
            throw new IllegalStateException("layer " + mixer.layer().il() + ": FoPE coefficients " + cosRef.name()
                    + " / " + sinRef.name() + " are not resident");
        }
        long n = cosRef.numElements();
        fopeCosBuf = mixer.vector(cos, n);
        fopeSinBuf = mixer.vector(sin, n);
        float[] inv = rope.fope().invFreq();
        if (fopeInvBuf == null || !Arrays.equals(fopeInvValues, inv)) {
            fopeInvBuf = mixer.resources().allocate((long) inv.length * Float.BYTES);
            GpuEvent e = mixer.ctx().queue().enqueueWriteBuffer(fopeInvBuf, 0, (long) inv.length * Float.BYTES, inv);
            if (e != null) {
                e.close();
            }
            fopeInvValues = inv;
        }
        mixer.input(fopeStep(rope, mixer.spec().nHead(), mixer.spec().qDim(), ropePos, mixer.qBuf()), false);
        if (!skipK && mixer.proj().projectsKv()) {
            mixer.input(fopeStep(rope, mixer.spec().nHeadKv(), mixer.spec().kDim(), ropePos, mixer.kBuf()), true);
        }
    }

    /** The {@link RopeKernels#ROPE_FOPE} step of {@code nHeads} heads over {@code x} ({@code rowStride} wide). */
    private TokenStep fopeStep(RopeSpec rope, int nHeads, int rowStride, GpuBuffer pos, GpuBuffer x) {
        Object[] args = RopeKernels.fopeArgs(mixer.ubatch(), nHeads, mixer.spec().headDimK(), rowStride,
                rope.nDims(), 0, rope.fope().numInvFreq(), rope.fope().numKeyValueHeads(),
                rope.fope().separateHeads(), pos, fopeCosBuf, fopeSinBuf, fopeInvBuf, x);
        return mixer.step(RopeKernels.ROPE_FOPE, args, RopeKernels.ARG_N_TOKENS, 1, nHeads, 0, -1);
    }

    /**
     * The five query rotations and the key rotation of dual-chunk attention: the projected query is copied into the
     * four spare planes, each plane rotates at its own derived position with the same RoPE, and the keys rotate at
     * {@code position % chunk_length} ({@link DualChunkRope}, vLLM {@code DualChunkRotaryEmbedding.forward}).
     */
    private void bindDualChunkRope(RopeSpec rope, GpuBuffer ff, boolean hasFf) {
        int pitch = mixer.dcPlanes().planePitch();
        for (int plane = 1; plane < DualChunkRope.VARIANTS; plane++) {
            mixer.input(mixer.copyStep(mixer.qBuf(), 0, mixer.spec().qDim(), mixer.qBuf(), plane * pitch,
                    mixer.spec().qDim(), 1, mixer.spec().qDim()), false);
        }
        for (int plane = 0; plane < DualChunkRope.VARIANTS; plane++) {
            mixer.input(ropeStep(rope, mixer.spec().nHead(), mixer.spec().qDim(),
                    mixer.dcPlanes().planePos(plane), ff, hasFf, mixer.qBuf(), plane * pitch), false);
        }
        if (mixer.proj().projectsKv()) {
            mixer.input(ropeStep(rope, mixer.spec().nHeadKv(), mixer.spec().kDim(), mixer.dcPlanes().keyPos(), ff,
                    hasFf, mixer.kBuf(), 0), true);
        }
    }

    /**
     * The layer's F32 frequency factors: the divisors the resolver computed (vLLM's fixed and mixed NTK scaling) or
     * the {@code rope_freqs} / long / short factors, or null when the rotation has none.
     */
    private GpuBuffer ropeFactors() {
        RopeSpec rope = mixer.layer().rope();
        if (rope == null) {
            return null;
        }
        float[] computed = rope.computedFreqFactors();
        if (computed != null) {
            if (computedFactorBuf == null || !Arrays.equals(computedFactorValues, computed)) {
                computedFactorBuf = mixer.resources().allocate((long) computed.length * Float.BYTES);
                GpuEvent e = mixer.ctx().queue().enqueueWriteBuffer(computedFactorBuf, 0,
                        (long) computed.length * Float.BYTES, computed);
                if (e != null) {
                    e.close();
                }
                computedFactorValues = computed;
            }
            return computedFactorBuf;
        }
        if (rope.freqFactors() == null) {
            return null;
        }
        TensorRef ref = rope.freqFactors();
        GpuWeight w = mixer.weights().getOrNull(ref.name());
        if (w == null) {
            throw new IllegalStateException("layer " + mixer.layer().il() + ": RoPE frequency factors " + ref.name()
                    + " are not resident");
        }
        return mixer.vector(w, rope.nDims() / 2);
    }

    private TokenStep ropeStep(RopeSpec rope, int nHeads, int rowStride, GpuBuffer pos, GpuBuffer ff, boolean hasFf,
                               GpuBuffer x) {
        return ropeStep(rope, nHeads, rowStride, pos, ff, hasFf, x, 0);
    }

    private TokenStep ropeStep(RopeSpec rope, int nHeads, int rowStride, GpuBuffer pos, GpuBuffer ff, boolean hasFf,
                               GpuBuffer x, int xOff) {
        Object[] args = RopeKernels.ropeExtArgs(rope, mixer.ubatch(), nHeads, mixer.spec().headDimK(), rowStride,
                rope.nOffs(), mixer.ubatch(), xOff, hasFf, pos, ff, x);
        long perToken = RopeKernels.ropeExtWorkItems(rope.type(), 1, nHeads, rope.nDims());
        return mixer.step(RopeKernels.ROPE_EXT, args, RopeKernels.ARG_N_TOKENS, 1, perToken, 0,
                RopeKernels.ARG_POS_STRIDE);
    }
}
