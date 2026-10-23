package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.KdaGateMode;
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
 * GPU Kimi delta attention mixer ({@link MixerKind#KDA}: kimi-linear, kimi-k3, glm5-next, bailingmoe3), following
 * models/kimi-linear.cpp:286-384, models/kimi-k3.cpp:398-476, models/glm5-next.cpp:820-886 and
 * models/bailingmoe3.cpp:254-298 for one sequence. With {@code T} tokens, {@code H} heads of dim {@code D} and
 * {@code d_inner = H*D}:
 *
 * <pre>
 * attn_qkv(x) -> ATTN_OUT[T][3 d_inner]                 (fused; kimi-linear views it at 0, d_inner, 2 d_inner)
 *   or attn_q / attn_k / attn_v(x) -> ATTN_OUT[T][d_inner], one at a time before its conv
 * LLM_SSM_Conv(SiLU, no bias) per q / k / v with its own ssm_conv1d_{q,k,v} and conv-state slice
 *   (state_off 0, d_inner, 2 d_inner channels) -> Q, K, V [T][d_inner]
 * decay f: ssm_f_b(ssm_f_a(x)) (low rank) or ssm_f_a(x) (full, bailingmoe3) -> ATTN_OUT[T][d_inner]
 * ssm_beta(x) -> kda.b[T][H]; LLM_GDN_GatePrep(kda_mode SOFTPLUS / LB_NEG / LB_POS) -> g[T][H][D], beta[T][H]
 * LLM_RowNorm L2_GDN on every q and k head; LLM_GDN_Scan(per-key gate, out aliasing V) -> V
 * LLM_RowNorm RMS(ssm_norm) per head; output gate g2 -> ATTN_OUT[T][d_inner]:
 *   LOWRANK ssm_g_b(ssm_g_a(x)) | FULL_SSM_G ssm_g(x) | FULL_SSM_G_A ssm_g_a(x)
 * LLM_GLU SIGMUL (sigmoid(g2) * o) in place in V; attn_output -> out
 * </pre>
 *
 * The three convs, the gate prep and the scan are {@link PreparedDispatch}es bound once; per step only the token
 * count and the gate-prep global size change. The conv state holds three slices {@code [q | k | v]}, each
 * {@code conv[c][j]} (kimi-linear.cpp:175-231); the scan state is {@code s[h*D*D + col*D + i]}.
 * A model without memory runs on zeroed temporaries (see {@link DeltaNetBlocks.GpuStates}). Not thread-safe.
 */
public final class GpuKdaMixer implements GpuMixer {

    /** Temporary holding the low-rank decay projection {@code ssm_f_a(x)}. */
    static final String TEMP_FA = "kda.fa";
    /** Temporary holding the low-rank gate projection {@code ssm_g_a(x)}. */
    static final String TEMP_GA = "kda.ga";
    /** Temporary holding the beta logits. */
    static final String TEMP_B = "kda.b";

    private final List<GpuBuffer> owned = new ArrayList<>();
    private final List<PreparedDispatch> prepared = new ArrayList<>();

    private GpuOps ops;
    private int elLocal;
    private int nHead;
    private int headDim;
    private int dInner;
    private float l2Eps;
    private float rmsEps;
    private KdaGateMode gateMode;
    private CpuGatedDeltaNet.GateShape gateShape;

    private GpuWeight wQkv;
    private GpuWeight[] wSeparate;
    private GpuWeight wFa;
    private GpuWeight wFb;
    private GpuWeight wBeta;
    private GpuWeight wGa;
    private GpuWeight wGb;
    private GpuWeight wG;
    private GpuWeight wOut;
    /**
     * LoRA deltas of every projection, added right after the product as {@code build_lora_mm} does
     * (llama-graph.cpp:1629-1667); llama.cpp's kimi-linear graph multiplies these weights plainly.
     */
    private GpuProjectionDeltas.Site qkvDeltas;
    private GpuProjectionDeltas.Site[] separateDeltas;
    private GpuProjectionDeltas.Site faDeltas;
    private GpuProjectionDeltas.Site fbDeltas;
    private GpuProjectionDeltas.Site betaDeltas;
    private GpuProjectionDeltas.Site gaDeltas;
    private GpuProjectionDeltas.Site gbDeltas;
    private GpuProjectionDeltas.Site gDeltas;
    private GpuProjectionDeltas.Site outDeltas;
    private GpuWeight normW;
    private final GpuBuffer[] convW = new GpuBuffer[3];
    private GpuBuffer dtBias;
    private GpuBuffer ssmA;

    private ScratchSlot q;
    private ScratchSlot k;
    private ScratchSlot v;
    private ScratchSlot work;
    private ScratchSlot fa;
    private ScratchSlot ga;
    private ScratchSlot bl;
    private ScratchSlot gSlot;
    private ScratchSlot betaSlot;
    private DeltaNetBlocks.GpuStates states;
    private GpuBuffer[] working;

    private final PreparedDispatch[] convs = new PreparedDispatch[3];
    private PreparedDispatch gateDispatch;
    private PreparedDispatch scanDispatch;

    private boolean bound;
    private boolean closed;

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException when the layer is not a KDA layer
     * @throws IllegalStateException    when a tensor, state or scratch slot has the wrong size, the layer has no
     *                                  gate mode, or the mixer is already bound or closed
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException("KDA mixer is already bound or closed");
        }
        DeltaNetBlocks.checkKind(MixerKind.KDA, layer);
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
        bindShape(ctx, layer, r);
        bindWeights(ctx, layer, r);
        bindScratch(ctx, layer, r);
        prepareDispatches(ctx, r);
    }

    /** Head geometry, gate mode, conv-state shape, epsilons and the element-wise work-group size. */
    private void bindShape(GpuForwardContext ctx, LayerSpec layer, RecurrentSpec r) {
        ops = ctx.ops();
        elLocal = ops.local();
        gateShape = CpuGatedDeltaNet.GateShape.of(r);
        gateMode = r.kdaGateMode();
        if (gateMode == null) {
            throw new IllegalStateException("layer " + DeltaNetBlocks.describe(layer)
                    + ": KDA needs an output-gate mode");
        }
        nHead = r.nHead();
        headDim = r.headDim();
        dInner = nHead * headDim;
        int dConv = convWidth(r);
        SsmKernels.checkConvShape(dInner, dConv);
        if (r.convStateFloats() != 3L * dInner * (dConv - 1)) {
            throw new IllegalStateException("layer " + DeltaNetBlocks.describe(layer) + ": KDA conv state of "
                    + r.convStateFloats() + " floats is not three slices of " + dInner + " x " + (dConv - 1));
        }
        l2Eps = DeltaNetBlocks.l2Eps(ctx.spec());
        rmsEps = DeltaNetBlocks.rmsEps(ctx.spec());
    }

    /** Projection matrices, their LoRA delta sites, the norm weight and the F32 conv / decay vectors. */
    private void bindWeights(GpuForwardContext ctx, LayerSpec layer, RecurrentSpec r) {
        int nEmbd = DeltaNetBlocks.mixerWidth(layer);
        GpuWeights ws = ctx.weightSource().weightsFor(layer.weightLayer());
        wQkv = DeltaNetBlocks.gpuMatrixOrNull(ws, layer, TensorNames.ATTN_QKV, nEmbd, 3L * dInner);
        if (wQkv == null) {
            wSeparate = new GpuWeight[]{
                    DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_Q, nEmbd, dInner),
                    DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_K, nEmbd, dInner),
                    DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_V, nEmbd, dInner)};
        }
        bindDecay(ws, layer, nEmbd);
        wBeta = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_BETA, nEmbd, nHead);
        bindOutputGate(ws, layer, nEmbd);
        wOut = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_OUT, dInner, nEmbd);
        bindDeltas(ctx.projectionDeltas(), layer);
        normW = DeltaNetBlocks.gpuNormWeight(ctx, ws.layer(layer, TensorNames.weight(TensorNames.SSM_NORM)), headDim,
                owned);
        String[] convNames = {TensorNames.SSM_CONV1D_Q, TensorNames.SSM_CONV1D_K, TensorNames.SSM_CONV1D_V};
        long convFloats = (long) dInner * convWidth(r);
        for (int i = 0; i < 3; i++) {
            convW[i] = DeltaNetBlocks.gpuF32(ctx, ws.layer(layer, TensorNames.weight(convNames[i])), convFloats,
                    owned);
        }
        dtBias = DeltaNetBlocks.gpuF32(ctx, ws.layer(layer, TensorNames.bias(TensorNames.SSM_DT)), dInner, owned);
        ssmA = DeltaNetBlocks.gpuF32(ctx, DeltaNetBlocks.gpuEither(ws, layer, TensorNames.SSM_A_NOSCAN), nHead,
                owned);
    }

    /** Scratch slots and temporaries of one ubatch, and the layer's recurrent states. */
    private void bindScratch(GpuForwardContext ctx, LayerSpec layer, RecurrentSpec r) {
        GpuScratch scratch = ctx.scratch();
        int ubatch = scratch.ubatch();
        long rows = (long) ubatch * dInner;
        q = scratch.slot(ScratchSlots.Q).require(rows);
        k = scratch.slot(ScratchSlots.K).require(rows);
        v = scratch.slot(ScratchSlots.V).require(rows);
        work = scratch.slot(ScratchSlots.ATTN_OUT).require(wQkv != null ? 3 * rows : rows);
        if (wFb != null) {
            fa = scratch.temp(TEMP_FA, (long) ubatch * wFa.rows());
        }
        if (wGa != null && wGb != null) {
            ga = scratch.temp(TEMP_GA, (long) ubatch * wGa.rows());
        }
        bl = scratch.temp(TEMP_B, (long) ubatch * nHead);
        gSlot = DeltaNetBlocks.atZero(scratch.temp(DeltaNetBlocks.TEMP_G, rows), DeltaNetKernels.GDN_GATE_PREP);
        betaSlot = DeltaNetBlocks.atZero(scratch.temp(DeltaNetBlocks.TEMP_BETA, (long) ubatch * nHead),
                DeltaNetKernels.GDN_GATE_PREP);
        states = DeltaNetBlocks.GpuStates.bind(ctx, layer, r.convStateFloats(), r.stateFloats());
        working = new GpuBuffer[]{q.buffer(), k.buffer(), v.buffer(), work.buffer(), bl.buffer(),
                fa != null ? fa.buffer() : work.buffer(), ga != null ? ga.buffer() : work.buffer()};
    }

    /** The three single-token q/k/v conv dispatches, the gate prep and the scan. */
    private void prepareDispatches(GpuForwardContext ctx, RecurrentSpec r) {
        CpuGatedDeltaNet.ScanShape scanShape = CpuGatedDeltaNet.ScanShape.of(r);
        int dConv = convWidth(r);
        GpuBuffer ph = ctx.placeholders().readOnlyFloats();
        int convStride = DeltaNetBlocks.toInt(r.convStateFloats(), "conv state stride");
        int xStride = wQkv != null ? 3 * dInner : dInner;
        ScratchSlot[] outs = {q, k, v};
        for (int i = 0; i < 3; i++) {
            int xOff = work.offset() + (wQkv != null ? i * dInner : 0);
            convs[i] = keep(ops.prepare(SsmKernels.SSM_CONV, DeltaNetBlocks.global1d(dInner, elLocal),
                    new long[]{elLocal}, DeltaNetBlocks.convArgs(dInner, dConv, 1, SsmKernels.CONV_FLAG_SILU,
                            xStride, xOff, dInner, outs[i].offset(), i * dInner, convStride,
                            work.buffer(), convW[i], ph, states.conv, states.rows, outs[i].buffer())));
        }

        gateDispatch = keep(ops.prepare(DeltaNetKernels.GDN_GATE_PREP,
                new long[]{DeltaNetKernels.gatePrepGlobal(1, nHead, gateShape.kdaMode(), headDim, elLocal)},
                new long[]{elLocal}, DeltaNetBlocks.gatePrepArgs(gateShape, 1, dInner, work.offset(), nHead,
                        bl.offset(), work.buffer(), bl.buffer(), dtBias, ssmA, gSlot.buffer(), betaSlot.buffer())));

        int local = DeltaNetBlocks.recurrenceLocal(ctx);
        scanDispatch = keep(ops.prepare(DeltaNetKernels.GDN_SCAN,
                new long[]{DeltaNetKernels.recurrenceGlobal(nHead, headDim, local)}, new long[]{local},
                DeltaNetBlocks.scanArgs(scanShape, 1, dInner, q.offset(), dInner, k.offset(), dInner, v.offset(),
                        dInner, v.offset(), q.buffer(), k.buffer(), v.buffer(), gSlot.buffer(), betaSlot.buffer(),
                        states.scan, states.rows, v.buffer())));
    }

    /** Conv kernel width: the stored steps plus the current token. */
    private static int convWidth(RecurrentSpec r) {
        return r.convSteps() + 1;
    }

    /** The delta sites of the projections the layer runs. */
    private void bindDeltas(GpuProjectionDeltas deltas, LayerSpec layer) {
        if (wQkv != null) {
            qkvDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_QKV), wQkv);
        } else {
            String[] names = {TensorNames.ATTN_Q, TensorNames.ATTN_K, TensorNames.ATTN_V};
            separateDeltas = new GpuProjectionDeltas.Site[names.length];
            for (int i = 0; i < names.length; i++) {
                separateDeltas[i] = deltas.serve(layer, TensorNames.weight(names[i]), wSeparate[i]);
            }
        }
        faDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_F_A), wFa);
        if (wFb != null) {
            fbDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_F_B), wFb);
        }
        betaDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_BETA), wBeta);
        if (wGa != null) {
            gaDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_G_A), wGa);
            gbDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_G_B), wGb);
        } else {
            String g = gateMode == KdaGateMode.FULL_SSM_G_A ? TensorNames.SSM_G_A : TensorNames.SSM_G;
            gDeltas = deltas.serve(layer, TensorNames.weight(g), wG);
        }
        outDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_OUT), wOut);
    }

    /**
     * Decay projection: low rank {@code ssm_f_b(ssm_f_a(x))} when {@code ssm_f_b} exists (kimi-linear.cpp:314-315,
     * kimi-k3.cpp:422-423, glm5-next.cpp:835-836), otherwise the full {@code ssm_f_a(x)} of {@code d_inner} rows
     * (bailingmoe3.cpp:82, 271).
     */
    private void bindDecay(GpuWeights ws, LayerSpec layer, int nEmbd) {
        wFb = ws.layerOrNull(layer, TensorNames.weight(TensorNames.SSM_F_B));
        if (wFb != null) {
            wFa = DeltaNetBlocks.gpuMatrixCols(ws, layer, TensorNames.SSM_F_A, nEmbd);
            wFb = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_F_B, wFa.rows(), dInner);
        } else {
            wFa = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_F_A, nEmbd, dInner);
        }
    }

    /**
     * Output gate weights of the layer's {@link KdaGateMode} (kimi-linear.cpp:367-368, kimi-k3.cpp:463,
     * glm5-next.cpp:874-875, bailingmoe3.cpp:292).
     */
    private void bindOutputGate(GpuWeights ws, LayerSpec layer, int nEmbd) {
        switch (gateMode) {
            case LOWRANK:
                wGa = DeltaNetBlocks.gpuMatrixCols(ws, layer, TensorNames.SSM_G_A, nEmbd);
                wGb = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_G_B, wGa.rows(), dInner);
                break;
            case FULL_SSM_G:
                wG = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_G, nEmbd, dInner);
                break;
            case FULL_SSM_G_A:
                wG = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.SSM_G_A, nEmbd, dInner);
                break;
            default:
                throw new IllegalStateException("unhandled KDA gate mode " + gateMode);
        }
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
            throw new IllegalStateException("KDA mixer is not bound or already closed");
        }
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ops.scratch().ubatch()) {
            throw new IllegalArgumentException("KDA step of " + n + " tokens is outside [1, "
                    + ops.scratch().ubatch() + "]");
        }
        DeltaNetBlocks.checkStepViews("KDA mixer", in, out, working);
        states.clearTransient(ops);
        GpuBuffer x = in.buffer();
        int x0 = in.offset();
        if (wQkv != null) {
            ops.matmul(wQkv, x, x0, work.buffer(), work.offset(), n);
            qkvDeltas.applyRows(n, x, x0, work.buffer(), work.offset());
            for (PreparedDispatch c : convs) {
                conv(c, n);
            }
        } else {
            for (int i = 0; i < 3; i++) {
                ops.matmul(wSeparate[i], x, x0, work.buffer(), work.offset(), n);
                separateDeltas[i].applyRows(n, x, x0, work.buffer(), work.offset());
                conv(convs[i], n);
            }
        }

        if (wFb != null) {
            ops.matmul(wFa, x, x0, fa.buffer(), fa.offset(), n);
            faDeltas.applyRows(n, x, x0, fa.buffer(), fa.offset());
            ops.matmul(wFb, fa.buffer(), fa.offset(), work.buffer(), work.offset(), n);
            fbDeltas.applyRows(n, fa.buffer(), fa.offset(), work.buffer(), work.offset());
        } else {
            ops.matmul(wFa, x, x0, work.buffer(), work.offset(), n);
            faDeltas.applyRows(n, x, x0, work.buffer(), work.offset());
        }
        ops.matmul(wBeta, x, x0, bl.buffer(), bl.offset(), n);
        betaDeltas.applyRows(n, x, x0, bl.buffer(), bl.offset());
        gateDispatch.setInt(DeltaNetKernels.GATE_ARG_N_TOKENS, n);
        gateDispatch.setGlobal(DeltaNetKernels.gatePrepGlobal(n, nHead, gateShape.kdaMode(), headDim, elLocal));
        gateDispatch.dispatch();

        ops.norm(NormKind.L2_GDN, null, null, l2Eps, q.buffer(), q.offset(), q.buffer(), q.offset(), n * nHead,
                headDim);
        ops.norm(NormKind.L2_GDN, null, null, l2Eps, k.buffer(), k.offset(), k.buffer(), k.offset(), n * nHead,
                headDim);
        scanDispatch.setInt(DeltaNetKernels.SCAN_ARG_N_TOKENS, n);
        scanDispatch.dispatch();

        ops.norm(NormKind.RMS, normW, null, rmsEps, v.buffer(), v.offset(), v.buffer(), v.offset(), n * nHead,
                headDim);
        if (wGa != null) {
            ops.matmul(wGa, x, x0, ga.buffer(), ga.offset(), n);
            gaDeltas.applyRows(n, x, x0, ga.buffer(), ga.offset());
            ops.matmul(wGb, ga.buffer(), ga.offset(), work.buffer(), work.offset(), n);
            gbDeltas.applyRows(n, ga.buffer(), ga.offset(), work.buffer(), work.offset());
        } else {
            ops.matmul(wG, x, x0, work.buffer(), work.offset(), n);
            gDeltas.applyRows(n, x, x0, work.buffer(), work.offset());
        }
        ops.glu(GluOp.SIGMUL, ActParams.NONE, work.buffer(), work.offset(), dInner, v.buffer(), v.offset(), dInner,
                v.buffer(), v.offset(), dInner, n, dInner);
        ops.matmul(wOut, v.buffer(), v.offset(), out.buffer(), out.offset(), n);
        outDeltas.applyRows(n, v.buffer(), v.offset(), out.buffer(), out.offset());
    }

    private static void conv(PreparedDispatch c, int n) {
        c.setInt(SsmKernels.ARG_CONV_N_TOKENS, n);
        c.dispatch();
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
        return "GpuKdaMixer{" + (bound ? nHead + " heads of " + headDim + ", " + gateShape.decayGate() + ", "
                + gateMode : "unbound") + '}';
    }
}
