package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * GPU gated DeltaNet mixer ({@link MixerKind#GDN}: qwen3next, qwen35, qwen35moe, clef), following
 * {@code build_layer_attn_linear} of models/qwen3next.cpp:311-553 and models/qwen35.cpp:346-475 for one sequence. With
 * {@code T} tokens, {@code K = n_k_heads * S_k}, {@code V = n_v_heads * S_v} and {@code C = 2K + V} conv channels:
 *
 * <pre>
 * attn_qkv(x) -> Q[T][C] = [q | k | v]; attn_gate(x) -> V slot z[T][V]             (fused projections)
 *   or ssm_in(x) -> ATTN_OUT[T][n_k][q k v z]; LLM_Copy2D_F32 -> Q[T][C], z[T][V]   (legacy qwen3next GGUF)
 * ssm_ba(x) -> a[T][n_k][b | a]                                                  (qwen3next fused beta/alpha)
 *   or ssm_alpha(x) -> a[T][n_v], ssm_beta(x) -> b[T][n_v]                         (separate)
 * LLM_SSM_Conv(Q, SiLU, conv state) -> K[T][C]
 * LLM_RowNorm L2_GDN over the q and k heads: in place in K for one token; for T &gt; 1 the q|k columns are first
 *   compacted into the temporary gdn.qk[T][2K] (rows of the conv output are C wide)
 * LLM_GDN_GatePrep -> g[T][n_v], beta[T][n_v];  LLM_GDN_Scan(q, k, v = K[.][2K..], scan state) -> ATTN_OUT[T][V]
 * LLM_RowNorm RMS(ssm_norm) per value head; LLM_GLU SWIGLU (silu(z) * o) in place; ssm_out -> out
 * </pre>
 *
 * The conv, gate-prep and scan launches are {@link PreparedDispatch}es bound once (two scans: the one-token view
 * reads q and k from the conv output, the batch view from {@code gdn.qk}); per step only the token count and the
 * gate-prep global size change. State layouts: conv {@code conv[c][j]}, scan
 * {@code s[h*S_v*S_k + col*S_k + i]}, row {@code stateRows[0]}. A model without memory runs on zeroed temporaries
 * (see {@link DeltaNetBlocks.GpuStates}). Not thread-safe.
 */
public final class GpuGdnMixer implements GpuMixer {

    /** Temporary holding the compacted q|k columns of a batch. */
    static final String TEMP_QK = "gdn.qk";
    /** Temporary holding the alpha logits (separate) or the fused {@code ssm_ba} rows. */
    static final String TEMP_A = "gdn.a";
    /** Temporary holding the beta logits of the separate layout. */
    static final String TEMP_B = "gdn.b";

    private final List<GpuBuffer> owned = new ArrayList<>();
    private final List<PreparedDispatch> prepared = new ArrayList<>();

    private GpuOps ops;
    private int elLocal;
    private int nK;
    private int nV;
    private int headK;
    private int headV;
    private int keyDim;
    private int valDim;
    private int convCh;
    private boolean legacy;
    private float l2Eps;
    private float rmsEps;
    private CpuGatedDeltaNet.GateShape gateShape;

    private GpuWeight wQkv;
    private GpuWeight wGate;
    private GpuWeight wIn;
    private GpuWeight wBa;
    private GpuWeight wAlpha;
    private GpuWeight wBeta;
    private GpuWeight wOut;
    private GpuWeight normW;
    private GpuBuffer convW;
    private GpuBuffer dtBias;
    private GpuBuffer ssmA;
    /** LoRA deltas of the projections ({@code build_lora_mm}, models/qwen35.cpp, models/qwen3next.cpp). */
    private GpuProjectionDeltas.Site qkvDeltas;
    private GpuProjectionDeltas.Site gateDeltas;
    private GpuProjectionDeltas.Site inDeltas;
    private GpuProjectionDeltas.Site baDeltas;
    private GpuProjectionDeltas.Site alphaDeltas;
    private GpuProjectionDeltas.Site betaDeltas;
    private GpuProjectionDeltas.Site outDeltas;

    private ScratchSlot x;
    private ScratchSlot conv;
    private ScratchSlot z;
    private ScratchSlot o;
    private ScratchSlot qk;
    private ScratchSlot a;
    private ScratchSlot b;
    private ScratchSlot gSlot;
    private ScratchSlot betaSlot;
    private DeltaNetBlocks.GpuStates states;
    private GpuBuffer[] working;

    private PreparedDispatch convDispatch;
    private PreparedDispatch gateDispatch;
    private PreparedDispatch scanOne;
    private PreparedDispatch scanBatch;

    private boolean bound;
    private boolean closed;

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException when the layer is not a GDN layer
     * @throws IllegalStateException    when a tensor, state or scratch slot has the wrong size, or the mixer is
     *                                  already bound or closed
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException("GDN mixer is already bound or closed");
        }
        DeltaNetBlocks.checkKind(MixerKind.GDN, layer);
        try {
            bindChecked(ctx, layer);
            bound = true;
        } catch (RuntimeException e) {
            try {
                release();
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
    }

    private void bindChecked(GpuForwardContext ctx, LayerSpec layer) {
        RecurrentSpec r = layer.recurrent();
        bindShape(ctx, r);
        bindWeights(ctx, layer, r);
        bindScratch(ctx, layer, r);
        prepareDispatches(ctx, r);
    }

    /** Head geometry, epsilons and the element-wise work-group size of the layer. */
    private void bindShape(GpuForwardContext ctx, RecurrentSpec r) {
        ops = ctx.ops();
        elLocal = ops.local();
        gateShape = CpuGatedDeltaNet.GateShape.of(r);
        nK = r.nKHeads();
        nV = r.nVHeads();
        headK = r.headKDim();
        headV = r.headVDim();
        keyDim = nK * headK;
        valDim = nV * headV;
        convCh = r.convChannels();
        legacy = r.legacySsmIn();
        SsmKernels.checkConvShape(convCh, r.dConv());
        l2Eps = DeltaNetBlocks.l2Eps(ctx.spec());
        rmsEps = DeltaNetBlocks.rmsEps(ctx.spec());
    }

    /** Projection matrices (legacy {@code ssm_in} or {@code attn_qkv} + {@code attn_gate}) and the F32 vectors. */
    private void bindWeights(GpuForwardContext ctx, LayerSpec layer, RecurrentSpec r) {
        int nEmbd = DeltaNetBlocks.mixerWidth(layer);
        GpuWeights ws = ctx.weightSource().weightsFor(layer.weightLayer());
        if (legacy) {
            long perKh = 2L * headK + 2L * (nV / nK) * headV;
            wIn = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_IN, nEmbd, nK * perKh);
        } else {
            wQkv = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_QKV, nEmbd, convCh);
            wGate = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_GATE, nEmbd, valDim);
        }
        if (r.fusedBetaAlpha()) {
            wBa = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_BETA_ALPHA, nEmbd, 2L * nV);
        } else {
            wAlpha = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_ALPHA, nEmbd, nV);
            wBeta = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_BETA, nEmbd, nV);
        }
        wOut = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_OUT, valDim, nEmbd);
        bindDeltas(ctx.projectionDeltas(), layer);
        normW = DeltaNetBlocks.gpuNormWeight(ctx, ws.layer(layer, TensorNames.weight(TensorNames.SSM_NORM)), headV,
                owned);
        convW = DeltaNetBlocks.gpuF32(ctx, ws.layer(layer, TensorNames.weight(TensorNames.SSM_CONV1D)),
                (long) convCh * r.dConv(), owned);
        dtBias = DeltaNetBlocks.gpuF32(ctx, ws.layer(layer, TensorNames.bias(TensorNames.SSM_DT)), nV, owned);
        ssmA = DeltaNetBlocks.gpuF32(ctx, DeltaNetBlocks.gpuEither(ws, layer, TensorNames.SSM_A_NOSCAN), nV, owned);
    }

    /** The LoRA delta sites of the bound projections. */
    private void bindDeltas(GpuProjectionDeltas deltas, LayerSpec layer) {
        if (legacy) {
            inDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_IN), wIn);
        } else {
            qkvDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_QKV), wQkv);
            gateDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_GATE), wGate);
        }
        if (wBa != null) {
            baDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_BETA_ALPHA), wBa);
        } else {
            alphaDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_ALPHA), wAlpha);
            betaDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_BETA), wBeta);
        }
        outDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_OUT), wOut);
    }

    /** Scratch slots and temporaries of one ubatch, and the layer's recurrent states. */
    private void bindScratch(GpuForwardContext ctx, LayerSpec layer, RecurrentSpec r) {
        GpuScratch scratch = ctx.scratch();
        int ubatch = scratch.ubatch();
        x = scratch.slot(ScratchSlots.Q).require((long) ubatch * convCh);
        conv = scratch.slot(ScratchSlots.K).require((long) ubatch * convCh);
        z = scratch.slot(ScratchSlots.V).require((long) ubatch * valDim);
        o = scratch.slot(ScratchSlots.ATTN_OUT).require((long) ubatch * (legacy ? wIn.rows() : valDim));
        qk = scratch.temp(TEMP_QK, (long) ubatch * 2 * keyDim);
        a = scratch.temp(TEMP_A, (long) ubatch * (r.fusedBetaAlpha() ? 2 * nV : nV));
        b = r.fusedBetaAlpha() ? a : scratch.temp(TEMP_B, (long) ubatch * nV);
        gSlot = DeltaNetBlocks.atZero(scratch.temp(DeltaNetBlocks.TEMP_G, (long) ubatch * nV),
                DeltaNetKernels.GDN_GATE_PREP);
        betaSlot = DeltaNetBlocks.atZero(scratch.temp(DeltaNetBlocks.TEMP_BETA, (long) ubatch * nV),
                DeltaNetKernels.GDN_GATE_PREP);
        states = DeltaNetBlocks.GpuStates.bind(ctx, layer, r.convStateFloats(), r.stateFloats());
        working = new GpuBuffer[]{x.buffer(), conv.buffer(), z.buffer(), o.buffer(), qk.buffer(), a.buffer(),
                b.buffer()};
    }

    /** The single-token conv, gate-prep and scan dispatches, and the batched scan over the de-interleaved Q/K. */
    private void prepareDispatches(GpuForwardContext ctx, RecurrentSpec r) {
        CpuGatedDeltaNet.ScanShape scanShape = CpuGatedDeltaNet.ScanShape.of(r);
        int dConv = r.dConv();
        GpuBuffer ph = ctx.placeholders().readOnlyFloats();
        int convStride = DeltaNetBlocks.toInt(r.convStateFloats(), "conv state stride");
        convDispatch = keep(ops.prepare(SsmKernels.SSM_CONV, DeltaNetBlocks.global1d(convCh, elLocal),
                new long[]{elLocal}, DeltaNetBlocks.convArgs(convCh, dConv, 1, SsmKernels.CONV_FLAG_SILU,
                        convCh, x.offset(), convCh, conv.offset(), 0, convStride,
                        x.buffer(), convW, ph, states.conv, states.rows, conv.buffer())));

        int abStride = r.fusedBetaAlpha() ? 2 * nV : nV;
        gateDispatch = keep(ops.prepare(DeltaNetKernels.GDN_GATE_PREP,
                new long[]{DeltaNetKernels.gatePrepGlobal(1, nV, gateShape.kdaMode(), headV, elLocal)},
                new long[]{elLocal}, DeltaNetBlocks.gatePrepArgs(gateShape, 1, abStride, a.offset(), abStride,
                        b.offset(), a.buffer(), b.buffer(), dtBias, ssmA, gSlot.buffer(), betaSlot.buffer())));

        int local = DeltaNetBlocks.recurrenceLocal(ctx);
        long[] scanGlobal = {DeltaNetKernels.recurrenceGlobal(nV, headV, local)};
        long[] scanLocal = {local};
        int c0 = conv.offset();
        scanOne = keep(ops.prepare(DeltaNetKernels.GDN_SCAN, scanGlobal, scanLocal,
                DeltaNetBlocks.scanArgs(scanShape, 1, convCh, c0, convCh, c0 + keyDim, convCh, c0 + 2 * keyDim,
                        valDim, o.offset(), conv.buffer(), conv.buffer(), conv.buffer(), gSlot.buffer(),
                        betaSlot.buffer(), states.scan, states.rows, o.buffer())));
        int q0 = qk.offset();
        scanBatch = keep(ops.prepare(DeltaNetKernels.GDN_SCAN, scanGlobal, scanLocal,
                DeltaNetBlocks.scanArgs(scanShape, 1, 2 * keyDim, q0, 2 * keyDim, q0 + keyDim, convCh,
                        c0 + 2 * keyDim, valDim, o.offset(), qk.buffer(), qk.buffer(), conv.buffer(),
                        gSlot.buffer(), betaSlot.buffer(), states.scan, states.rows, o.buffer())));
    }

    private PreparedDispatch keep(PreparedDispatch d) {
        prepared.add(d);
        return d;
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        run(1, in, out);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        run(nTokens, in, out);
    }

    private void run(int n, ScratchSlot in, ScratchSlot out) {
        if (!bound || closed) {
            throw new IllegalStateException("GDN mixer is not bound or already closed");
        }
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ops.scratch().ubatch()) {
            throw new IllegalArgumentException("GDN step of " + n + " tokens is outside [1, "
                    + ops.scratch().ubatch() + "]");
        }
        DeltaNetBlocks.checkStepViews("GDN mixer", in, out, working);
        states.clearTransient(ops);
        project(n, in);

        convDispatch.setInt(SsmKernels.ARG_CONV_N_TOKENS, n);
        convDispatch.dispatch();

        PreparedDispatch scan;
        if (n == 1) {
            ops.norm(NormKind.L2_GDN, null, null, l2Eps, conv.buffer(), conv.offset(), conv.buffer(), conv.offset(),
                    2 * nK, headK);
            scan = scanOne;
        } else {
            ops.copy2d(conv.buffer(), conv.offset(), convCh, qk.buffer(), qk.offset(), 2 * keyDim, n, 2 * keyDim);
            ops.norm(NormKind.L2_GDN, null, null, l2Eps, qk.buffer(), qk.offset(), qk.buffer(), qk.offset(),
                    n * 2 * nK, headK);
            scan = scanBatch;
        }

        gateDispatch.setInt(DeltaNetKernels.GATE_ARG_N_TOKENS, n);
        gateDispatch.setGlobal(DeltaNetKernels.gatePrepGlobal(n, nV, gateShape.kdaMode(), headV, elLocal));
        gateDispatch.dispatch();

        scan.setInt(DeltaNetKernels.SCAN_ARG_N_TOKENS, n);
        scan.dispatch();

        ops.norm(NormKind.RMS, normW, null, rmsEps, o.buffer(), o.offset(), o.buffer(), o.offset(), n * nV, headV);
        ops.glu(GluOp.SWIGLU, ActParams.NONE, z.buffer(), z.offset(), valDim, o.buffer(), o.offset(), valDim,
                o.buffer(), o.offset(), valDim, n, valDim);
        ops.matmul(wOut, o.buffer(), o.offset(), out.buffer(), out.offset(), n);
        outDeltas.applyRows(n, o.buffer(), o.offset(), out.buffer(), out.offset());
    }

    /** Input projections: q|k|v into the conv input, z, and the beta/alpha logits. */
    private void project(int n, ScratchSlot in) {
        if (legacy) {
            ops.matmul(wIn, in.buffer(), in.offset(), o.buffer(), o.offset(), n);
            inDeltas.applyRows(n, in.buffer(), in.offset(), o.buffer(), o.offset());
            deinterleave(n);
        } else {
            ops.matmul(wQkv, in.buffer(), in.offset(), x.buffer(), x.offset(), n);
            qkvDeltas.applyRows(n, in.buffer(), in.offset(), x.buffer(), x.offset());
            ops.matmul(wGate, in.buffer(), in.offset(), z.buffer(), z.offset(), n);
            gateDeltas.applyRows(n, in.buffer(), in.offset(), z.buffer(), z.offset());
        }
        if (wBa != null) {
            ops.matmul(wBa, in.buffer(), in.offset(), a.buffer(), a.offset(), n);
            baDeltas.applyRows(n, in.buffer(), in.offset(), a.buffer(), a.offset());
        } else {
            ops.matmul(wAlpha, in.buffer(), in.offset(), a.buffer(), a.offset(), n);
            alphaDeltas.applyRows(n, in.buffer(), in.offset(), a.buffer(), a.offset());
            ops.matmul(wBeta, in.buffer(), in.offset(), b.buffer(), b.offset(), n);
            betaDeltas.applyRows(n, in.buffer(), in.offset(), b.buffer(), b.offset());
        }
    }

    /**
     * Splits the legacy {@code ssm_in} rows, per key head {@code [q(S_k) | k(S_k) | v(R*S_v) | z(R*S_v)]}
     * (qwen3next.cpp:337-388), into the conv input {@code [q all | k all | v all]} and {@code z}; value head
     * {@code kh*R + r} keeps its position. One copy per part per token, or per key head when that is fewer.
     */
    private void deinterleave(int n) {
        int vw = (nV / nK) * headV;
        int perKh = 2 * headK + 2 * vw;
        int rowW = nK * perKh;
        GpuBuffer src = o.buffer();
        GpuBuffer xb = x.buffer();
        GpuBuffer zb = z.buffer();
        if (n <= nK) {
            for (int t = 0; t < n; t++) {
                int s = o.offset() + t * rowW;
                int d = x.offset() + t * convCh;
                ops.copy2d(src, s, perKh, xb, d, headK, nK, headK);
                ops.copy2d(src, s + headK, perKh, xb, d + keyDim, headK, nK, headK);
                ops.copy2d(src, s + 2 * headK, perKh, xb, d + 2 * keyDim, vw, nK, vw);
                ops.copy2d(src, s + 2 * headK + vw, perKh, zb, z.offset() + t * valDim, vw, nK, vw);
            }
        } else {
            for (int kh = 0; kh < nK; kh++) {
                int s = o.offset() + kh * perKh;
                ops.copy2d(src, s, rowW, xb, x.offset() + kh * headK, convCh, n, headK);
                ops.copy2d(src, s + headK, rowW, xb, x.offset() + keyDim + kh * headK, convCh, n, headK);
                ops.copy2d(src, s + 2 * headK, rowW, xb, x.offset() + 2 * keyDim + kh * vw, convCh, n, vw);
                ops.copy2d(src, s + 2 * headK + vw, rowW, zb, z.offset() + kh * vw, valDim, n, vw);
            }
        }
    }

    // ================================================================== close

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        release();
    }

    private void release() {
        RuntimeException first = Closeables.closeAll(prepared, null);
        prepared.clear();
        first = Closeables.closeAll(owned, first);
        owned.clear();
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuGdnMixer{" + (bound ? nK + "k/" + nV + "v heads, " + headK + "x" + headV
                + (legacy ? ", ssm_in" : "") + (wBa != null ? ", ssm_ba" : "") : "unbound") + '}';
    }
}
