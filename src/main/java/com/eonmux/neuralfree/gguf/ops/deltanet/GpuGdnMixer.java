package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.engine.RecurrentTreeRound;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.block.GpuRecurrentTreeMixer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuRecurrentTreeRound;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.ops.fused.FusedKernels;
import com.eonmux.neuralfree.gguf.ops.fused.GpuMultiTask;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmKernels;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
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
public final class GpuGdnMixer implements GpuMixer, GpuRecurrentTreeMixer {

    /** Temporary holding the compacted q|k columns of a batch. */
    static final String TEMP_QK = "gdn.qk";
    /** Temporary holding the alpha logits (separate), the fused {@code ssm_ba} or merged {@code ssm_beta_alpha} rows. */
    static final String TEMP_A = "gdn.a";
    /** Temporary holding the beta logits of the separate layout. */
    static final String TEMP_B = "gdn.b";
    /** Temporary holding the merged {@code attn_qkv_gate} rows before the split. */
    static final String TEMP_QKV_GATE = "gdn.qkv_gate";

    private final List<GpuBuffer> owned = new ArrayList<>();
    private final List<PreparedDispatch> prepared = new ArrayList<>();

    private GpuOps ops;
    private int layerIndex;
    private int elLocal;
    private int nK;
    private int nV;
    private int headK;
    private int headV;
    private int keyDim;
    private int valDim;
    private int convCh;
    private boolean legacy;
    private boolean mergedQkvGate;
    private float l2Eps;
    private float rmsEps;
    private CpuGatedDeltaNet.GateShape gateShape;

    private GpuWeight wQkv;
    private GpuWeight wGate;
    /** Merged linear-attention input {@code attn_qkv_gate}; null unless the spec says so. */
    private GpuWeight wQkvGate;
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
    private GpuProjectionDeltas.Site qkvGateDeltas;
    private GpuProjectionDeltas.Site inDeltas;
    private GpuProjectionDeltas.Site baDeltas;
    private GpuProjectionDeltas.Site alphaDeltas;
    private GpuProjectionDeltas.Site betaDeltas;
    private GpuProjectionDeltas.Site outDeltas;

    private ScratchSlot x;
    private ScratchSlot conv;
    private ScratchSlot z;
    private ScratchSlot o;
    /** Merged {@code attn_qkv_gate} rows before the split; null unless merged. */
    private ScratchSlot qkvGate;
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
    /** Fused output norm and gate; null when the backend has no such kernel. */
    private Boolean fusedNormGate;
    /** Launch of the L2 norm and gate preparation as one persistent multi-task dispatch; null when unused. */
    private GpuMultiTask gateTaskDecode;
    private GpuMultiTask gateTaskPrefill;
    /** Launch of the input projections as one persistent multi-task dispatch, recorded on first use. */
    private GpuMultiTask projectTask;
    private GpuBuffer projectIn;
    private int projectInOff;
    /**
     * Whether the projection weight set admits the multi-task path: all F32 single-buffer. Deltas register after the
     * mixer is bound, so their absence is checked at each launch ({@link #projectTask}).
     */
    private boolean projectFusable;

    private boolean bound;
    /** The layer's stream width: the rows the mixer reads and writes. */
    private int rowWidth;
    /** Floats of one conv-state row; the stride the conv kernel indexes a stack slot with. */
    private int convStride;
    /** Floats of one scan-state row; the stride the scan kernel indexes a stack slot with. */
    private int scanStride;
    /** Whether the memory holds per-sequence state rows a batched step routes through. */
    private boolean routeState;
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
        layerIndex = layer.il();
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
        mergedQkvGate = r.mergedQkvGate();
        SsmKernels.checkConvShape(convCh, r.dConv());
        convStride = DeltaNetBlocks.toInt(r.convStateFloats(), "conv state stride");
        scanStride = DeltaNetBlocks.toInt(r.stateFloats(), "scan state stride");
        l2Eps = DeltaNetBlocks.l2Eps(ctx.spec());
        rmsEps = DeltaNetBlocks.rmsEps(ctx.spec());
    }

    /** Projection matrices (legacy {@code ssm_in}, {@code attn_qkv} + {@code attn_gate} or the merged weights) and the F32 vectors. */
    private void bindWeights(GpuForwardContext ctx, LayerSpec layer, RecurrentSpec r) {
        int nEmbd = DeltaNetBlocks.mixerWidth(layer);
        this.rowWidth = nEmbd;
        GpuWeights ws = ctx.weightSource().weightsFor(layer.weightLayer());
        if (legacy) {
            long perKh = 2L * headK + 2L * (nV / nK) * headV;
            wIn = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_IN, nEmbd, nK * perKh);
        } else if (r.mergedQkvGate()) {
            wQkvGate = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_QKV_GATE, nEmbd, convCh + valDim);
        } else {
            wQkv = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_QKV, nEmbd, convCh);
            wGate = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_GATE, nEmbd, valDim);
        }
        if (r.fusedBetaAlpha()) {
            wBa = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_BETA_ALPHA, nEmbd, 2L * nV);
        } else if (r.concatBetaAlpha()) {
            wBa = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_BETA_ALPHA_CAT, nEmbd, 2L * nV);
        } else {
            wAlpha = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_ALPHA, nEmbd, nV);
            wBeta = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_BETA, nEmbd, nV);
        }
        wOut = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_OUT, valDim, nEmbd);
        bindDeltas(ctx.projectionDeltas(), layer, r);
        normW = DeltaNetBlocks.gpuNormWeight(ctx, ws.layer(layer, TensorNames.weight(TensorNames.SSM_NORM)), headV,
                owned);
        convW = DeltaNetBlocks.gpuF32(ctx, ws.layer(layer, TensorNames.weight(TensorNames.SSM_CONV1D)),
                (long) convCh * r.dConv(), owned);
        dtBias = DeltaNetBlocks.gpuF32(ctx, ws.layer(layer, TensorNames.bias(TensorNames.SSM_DT)), nV, owned);
        ssmA = DeltaNetBlocks.gpuF32(ctx, DeltaNetBlocks.gpuEither(ws, layer, TensorNames.SSM_A_NOSCAN), nV, owned);
    }

    /** The LoRA delta sites of the bound projections. */
    private void bindDeltas(GpuProjectionDeltas deltas, LayerSpec layer, RecurrentSpec r) {
        if (legacy) {
            inDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_IN), wIn);
        } else if (wQkvGate != null) {
            qkvGateDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_QKV_GATE), wQkvGate);
        } else {
            qkvDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_QKV), wQkv);
            gateDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_GATE), wGate);
        }
        if (wBa != null) {
            baDeltas = deltas.serve(layer, TensorNames.weight(r.concatBetaAlpha() ? TensorNames.SSM_BETA_ALPHA_CAT
                    : TensorNames.SSM_BETA_ALPHA), wBa);
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
        qkvGate = mergedQkvGate ? scratch.temp(TEMP_QKV_GATE, (long) ubatch * (convCh + valDim)) : null;
        qk = scratch.temp(TEMP_QK, (long) ubatch * 2 * keyDim);
        boolean fused = r.fusedBetaAlpha() || r.concatBetaAlpha();
        a = scratch.temp(TEMP_A, (long) ubatch * (fused ? 2 * nV : nV));
        b = fused ? a : scratch.temp(TEMP_B, (long) ubatch * nV);
        gSlot = DeltaNetBlocks.atZero(scratch.temp(DeltaNetBlocks.TEMP_G, (long) ubatch * nV),
                DeltaNetKernels.GDN_GATE_PREP);
        betaSlot = DeltaNetBlocks.atZero(scratch.temp(DeltaNetBlocks.TEMP_BETA, (long) ubatch * nV),
                DeltaNetKernels.GDN_GATE_PREP);
        states = DeltaNetBlocks.GpuStates.bind(ctx, layer, r.convStateFloats(), r.stateFloats());
        routeState = ctx.hasMemory() && ctx.memory() instanceof com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory
                && ctx.memory().pagedSequences() != null;
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

        boolean fused = r.fusedBetaAlpha() || r.concatBetaAlpha();
        int abStride = fused ? 2 * nV : nV;
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
        prepareFused(r);
    }

    /**
     * Binds the fused decode-path kernels the backend provides: the output norm + gate ({@link FusedKernels#GDN_NORM_GATE})
     * and the persistent launch of the L2 norm with the gate preparation ({@link FusedKernels#MULTI_TASK}).
     */
    private void prepareFused(RecurrentSpec r) {
        fusedNormGate = ops.fused(FusedKernels.GDN_NORM_GATE);
        if (!ops.fused(FusedKernels.MULTI_TASK)) {
            return;
        }
        GpuMultiTask decode = null;
        try {
            decode = buildGateTask(r, conv.buffer(), conv.offset());
            gateTaskPrefill = buildGateTask(r, qk.buffer(), qk.offset());
        } catch (RuntimeException e) {
            throw Closeables.close(decode, e);
        }
        gateTaskDecode = decode;
        projectFusable = projectionsFusable();
    }

    /** The L2 norm of the q|k columns and the gate preparation as one task pair. */
    private GpuMultiTask buildGateTask(RecurrentSpec r, GpuBuffer normSource, int normOff) {
        GpuMultiTask t = new GpuMultiTask(ops, ops.scratch().ubatch(), "gdn.gate." + layerIndex);
        GpuBuffer none = ops.placeholders().readOnlyFloats();
        t.rowNorm(2 * nK, headK, headK, NormKind.L2_GDN, 0, 1, normOff, normOff, l2Eps, normSource, none, none,
                normSource);
        boolean fused = r.fusedBetaAlpha() || r.concatBetaAlpha();
        int abStride = fused ? 2 * nV : nV;
        t.gatePrep(gateShape.nVHeads(), gateShape.nKHeads(), gateShape.layout(), gateShape.kdaMode(),
                gateShape.headDim(), abStride, a.offset(), abStride, b.offset(), gateShape.lb(),
                gateShape.betaScale(), a.buffer(), b.buffer(), dtBias, ssmA, gSlot.buffer(), betaSlot.buffer());
        t.build();
        return t;
    }

    /**
     * Whether every input projection is a plain single-buffer F32 matrix, so the multi-task matvec body computes
     * exactly what {@code ops.matmul} would for a projection without LoRA deltas.
     */
    private boolean projectionsFusable() {
        if (legacy) {
            return matvecFusable(wIn);
        }
        if (mergedQkvGate) {
            return matvecFusable(wQkvGate) && (wBa == null || matvecFusable(wBa));
        }
        return matvecFusable(wQkv) && matvecFusable(wGate)
                && (wBa != null ? matvecFusable(wBa) : matvecFusable(wAlpha) && matvecFusable(wBeta));
    }

    private boolean deltasEmpty() {
        return empty(qkvDeltas) && empty(gateDeltas) && empty(qkvGateDeltas) && empty(inDeltas) && empty(baDeltas)
                && empty(alphaDeltas) && empty(betaDeltas) && empty(outDeltas);
    }

    private static boolean empty(GpuProjectionDeltas.Site site) {
        return site == null || site.isEmpty();
    }

    private static boolean matvecFusable(GpuWeight w) {
        return w != null && w.type() == com.eonmux.neuralfree.gguf.GGMLType.F32 && w.isSingleShard()
                && !w.hasInputTransform() && w.hadamard() == null && !w.hasScale()
                && w.rows() <= Integer.MAX_VALUE && w.cols() <= Integer.MAX_VALUE;
    }

    /**
     * The input projections as one multi-task launch for the first input view seen; null when the view changes
     * (batched segments), the weights are not fusible, or a LoRA delta is registered on an input or output
     * projection now, so the caller runs {@code ops.matmul} and applies the deltas.
     */
    private GpuMultiTask projectTask(GpuBuffer inBuf, int inOff) {
        if (!projectFusable || !deltasEmpty()) {
            return null;
        }
        if (projectTask == null) {
            projectIn = inBuf;
            projectInOff = inOff;
            projectTask = new GpuMultiTask(ops, ops.scratch().ubatch(), "gdn.project." + layerIndex);
            if (legacy) {
                projectTask.matvecF32(wIn, inBuf, inOff, o.buffer(), o.offset());
            } else if (mergedQkvGate) {
                projectTask.matvecF32(wQkvGate, inBuf, inOff, qkvGate.buffer(), qkvGate.offset());
                if (wBa != null) {
                    projectTask.matvecF32(wBa, inBuf, inOff, a.buffer(), a.offset());
                }
            } else {
                projectTask.matvecF32(wQkv, inBuf, inOff, x.buffer(), x.offset());
                projectTask.matvecF32(wGate, inBuf, inOff, z.buffer(), z.offset());
                if (wBa != null) {
                    projectTask.matvecF32(wBa, inBuf, inOff, a.buffer(), a.offset());
                } else {
                    projectTask.matvecF32(wAlpha, inBuf, inOff, a.buffer(), a.offset());
                    projectTask.matvecF32(wBeta, inBuf, inOff, b.buffer(), b.offset());
                }
            }
            projectTask.build();
        }
        return projectTask != null && projectIn == inBuf && projectInOff == inOff ? projectTask : null;
    }

    private PreparedDispatch keep(PreparedDispatch d) {
        prepared.add(d);
        return d;
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        run(1, in.buffer(), in.offset(), out.buffer(), out.offset(), 0);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        run(nTokens, in.buffer(), in.offset(), out.buffer(), out.offset(), 0);
    }

    /** Why this mixer runs a batched step: every sequence's rows dispatch as its own contiguous step. */
    @Override
    public String batchRefusal() {
        return null;
    }

    /**
     * One batched step: every segment of {@code s} dispatches its rows as the contiguous step, reading and
     * writing its sequence's recurrent-state rows (the conv and scan re-point the kernels' state-row routing per
     * segment) and reading its rows at their place in the step's input and output.
     */
    @Override
    public void batch(StepLayout s, ScratchSlot in, ScratchSlot out) {
        if (s.contiguousSingle()) {
            if (s.nTokens() == 1) {
                decode(s.pos0(0), in, out);
            } else {
                prefill(s.pos0(0), s.nTokens(), in, out);
            }
            return;
        }
        long n = (long) s.nTokens() * rowWidth;
        in.require(n);
        out.require(n);
        DeltaNetBlocks.checkStepViews("GDN mixer", in, out, working);
        for (int seg = 0; seg < s.nSeg(); seg++) {
            int start = s.tokStart(seg);
            run(s.len(seg), in.buffer(), in.offset() + start * rowWidth, out.buffer(),
                    out.offset() + start * rowWidth, s.seq(seg));
        }
    }

    private void run(int n, ScratchSlot in, ScratchSlot out) {
        run(n, in.buffer(), in.offset(), out.buffer(), out.offset(), 0);
    }

    private void run(int n, GpuBuffer inBuf, int inOff, GpuBuffer outBuf, int outOff, int stateRow) {
        if (!bound || closed) {
            throw new IllegalStateException("GDN mixer is not bound or already closed");
        }
        if (n < 1 || n > ops.scratch().ubatch()) {
            throw new IllegalArgumentException("GDN step of " + n + " tokens is outside [1, "
                    + ops.scratch().ubatch() + "]");
        }
        if (routeState) {
            GpuMixer.routeStateRow(stateRow, states.rows, ops.queue());
        }
        states.clearTransient(ops);
        project(n, inBuf, inOff);

        convDispatch.setInt(SsmKernels.ARG_CONV_N_TOKENS, n);
        convDispatch.dispatch();

        PreparedDispatch scan;
        GpuMultiTask gateTask = n == 1 ? gateTaskDecode : gateTaskPrefill;
        if (n == 1) {
            if (gateTask != null) {
                gateTask.run(1);
            } else {
                ops.norm(NormKind.L2_GDN, null, null, l2Eps, conv.buffer(), conv.offset(), conv.buffer(),
                        conv.offset(), 2 * nK, headK);
            }
            scan = scanOne;
        } else {
            ops.copy2d(conv.buffer(), conv.offset(), convCh, qk.buffer(), qk.offset(), 2 * keyDim, n, 2 * keyDim);
            if (gateTask != null) {
                gateTask.run(n);
            } else {
                ops.norm(NormKind.L2_GDN, null, null, l2Eps, qk.buffer(), qk.offset(), qk.buffer(), qk.offset(),
                        n * 2 * nK, headK);
            }
            scan = scanBatch;
        }

        if (gateTask == null) {
            gateDispatch.setInt(DeltaNetKernels.GATE_ARG_N_TOKENS, n);
            gateDispatch.setGlobal(DeltaNetKernels.gatePrepGlobal(n, nV, gateShape.kdaMode(), headV, elLocal));
            gateDispatch.dispatch();
        }

        scan.setInt(DeltaNetKernels.SCAN_ARG_N_TOKENS, n);
        scan.dispatch();

        if (Boolean.TRUE.equals(fusedNormGate)) {
            ops.gdnNormGate(n, nV, headV, valDim, normW, rmsEps, o.buffer(), o.offset(), z.buffer(), z.offset(),
                    o.buffer(), o.offset());
        } else {
            ops.norm(NormKind.RMS, normW, null, rmsEps, o.buffer(), o.offset(), o.buffer(), o.offset(), n * nV,
                    headV);
            ops.glu(GluOp.SWIGLU, ActParams.NONE, z.buffer(), z.offset(), valDim, o.buffer(), o.offset(), valDim,
                    o.buffer(), o.offset(), valDim, n, valDim);
        }
        ops.matmul(wOut, o.buffer(), o.offset(), outBuf, outOff, n);
        outDeltas.applyRows(n, o.buffer(), o.offset(), outBuf, outOff);
    }

    /** Input projections: q|k|v into the conv input, z, and the beta/alpha logits. */
    private void project(int n, GpuBuffer inBuf, int inOff) {
        // A single-token step is one row per projection, the shape the multi-task matvec body serves; a prefill
        // multiplies whole tiles and keeps the tiled GEMM.
        GpuMultiTask mt = n == 1 ? projectTask(inBuf, inOff) : null;
        if (mt != null) {
            mt.run(n);
            if (legacy) {
                deinterleave(n);
            } else if (mergedQkvGate) {
                splitQkvGate(n);
            }
            return;
        }
        // Every input projection reads the same single row, so its Q8_1 blocks are quantized once.
        int inWidth = (int) (legacy ? wIn.cols() : mergedQkvGate ? wQkvGate.cols() : wQkv.cols());
        GpuOps.SharedQ8 share = n == 1 && GpuOps.shareableQ8Activations(inOff, inWidth)
                ? ops.shareQ8Activations(inBuf, inOff, inWidth) : null;
        try {
            if (legacy) {
                ops.matmul(wIn, inBuf, inOff, o.buffer(), o.offset(), n);
                inDeltas.applyRows(n, inBuf, inOff, o.buffer(), o.offset());
                deinterleave(n);
            } else if (mergedQkvGate) {
                ops.matmul(wQkvGate, inBuf, inOff, qkvGate.buffer(), qkvGate.offset(), n);
                qkvGateDeltas.applyRows(n, inBuf, inOff, qkvGate.buffer(), qkvGate.offset());
                splitQkvGate(n);
            } else {
                ops.matmul(wQkv, inBuf, inOff, x.buffer(), x.offset(), n);
                qkvDeltas.applyRows(n, inBuf, inOff, x.buffer(), x.offset());
                ops.matmul(wGate, inBuf, inOff, z.buffer(), z.offset(), n);
                gateDeltas.applyRows(n, inBuf, inOff, z.buffer(), z.offset());
            }
            if (wBa != null) {
                ops.matmul(wBa, inBuf, inOff, a.buffer(), a.offset(), n);
                baDeltas.applyRows(n, inBuf, inOff, a.buffer(), a.offset());
            } else {
                ops.matmul(wAlpha, inBuf, inOff, a.buffer(), a.offset(), n);
                alphaDeltas.applyRows(n, inBuf, inOff, a.buffer(), a.offset());
                ops.matmul(wBeta, inBuf, inOff, b.buffer(), b.offset(), n);
                betaDeltas.applyRows(n, inBuf, inOff, b.buffer(), b.offset());
            }
        } finally {
            if (share != null) {
                share.close();
            }
        }
    }

    @Override
    public String treeStepRefusal() {
        return null;
    }

    @Override
    public int treeRowWidth() {
        return rowWidth;
    }

    @Override
    public void treeStep(GpuRecurrentTreeRound round, ScratchSlot in, ScratchSlot out) {
        if (!bound || closed) {
            throw new IllegalStateException("GDN mixer is not bound or already closed");
        }
        long[] mask = round.mask();
        int realRows = round.realRows();
        GpuBuffer convStack = round.convStack();
        GpuBuffer scanStack = round.scanStack();
        // the stateful kernels read the round's stack slots instead of the committed rows for the step; the
        // per-row routing still rides the live routing buffer, so every dispatch addresses its slot
        convDispatch.setBuffer(SsmKernels.CONV_ARG_STATE, convStack);
        scanOne.setBuffer(DeltaNetKernels.SCAN_ARG_STATE, scanStack);
        try {
            for (int row = 0; row < realRows; row++) {
                int depth = RecurrentTreeRound.depth(mask[row]);
                if (depth < 0) {
                    throw new IllegalArgumentException("the tree step's row " + row + " is padding, the mixer runs"
                            + " the computed rows");
                }
                int parent = RecurrentTreeRound.parentRow(mask[row], row);
                seedStack(round.queue(), convStack, row, depth, parent, convStride, states.conv);
                seedStack(round.queue(), scanStack, row, depth, parent, scanStride, states.scan);
                run(1, in.buffer(), in.offset() + row * rowWidth, out.buffer(), out.offset() + row * rowWidth, row);
            }
        } finally {
            convDispatch.setBuffer(SsmKernels.CONV_ARG_STATE, states.conv);
            scanOne.setBuffer(DeltaNetKernels.SCAN_ARG_STATE, states.scan);
        }
    }

    /**
     * Enqueues the copy that fills the row's stack slot: the committed rows at the root, else the parent row's
     * slot.
     */
    private static void seedStack(ComputeQueue queue, GpuBuffer stack, int row, int depth, int parent, int stride,
                                  GpuBuffer committed) {
        if (stride == 0) {
            return;
        }
        long bytes = (long) stride * Float.BYTES;
        if (depth == 0) {
            queue.enqueueCopyBuffer(committed, 0, stack, (long) row * bytes, bytes);
        } else {
            queue.enqueueCopyBuffer(stack, (long) parent * bytes, stack, (long) row * bytes, bytes);
        }
    }

    /** Splits the merged {@code attn_qkv_gate} rows into the conv input {@code x} and the gate {@code z}. */
    private void splitQkvGate(int n) {
        int rowW = convCh + valDim;
        ops.copy2d(qkvGate.buffer(), qkvGate.offset(), rowW, x.buffer(), x.offset(), convCh, n, convCh);
        ops.copy2d(qkvGate.buffer(), qkvGate.offset() + convCh, rowW, z.buffer(), z.offset(), valDim, n, valDim);
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
        first = Closeables.closeAll(java.util.Arrays.asList(gateTaskDecode, gateTaskPrefill, projectTask), first);
        gateTaskDecode = null;
        gateTaskPrefill = null;
        projectTask = null;
        first = Closeables.closeAll(owned, first);
        owned.clear();
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuGdnMixer{" + (bound ? nK + "k/" + nV + "v heads, " + headK + "x" + headV
                + (legacy ? ", ssm_in" : "") + (wQkvGate != null ? ", attn_qkv_gate" : "")
                + (wBa != null ? ", " + (gateShape.layout() == DeltaNetKernels.LAYOUT_CONCAT_BA
                        ? "ssm_beta_alpha" : "ssm_ba") : "") : "unbound") + '}';
    }
}
