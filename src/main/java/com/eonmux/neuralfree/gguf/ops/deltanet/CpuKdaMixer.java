package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.KdaGateMode;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.ssm.CpuSsmConv;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * CPU Kimi delta attention mixer, the twin of {@link GpuKdaMixer} (same buffers, same order, same math):
 * q/k/v projections, three {@link CpuSsmConv} calls on the {@code [q | k | v]} conv-state slices, the decay and beta
 * projections with {@link CpuGatedDeltaNet#gatePrep}, {@code L2_GDN} on every q and k head, the per-key-gate scan
 * (heads in parallel, output in V), the RMS norm of every head, the output gate of the layer's {@link KdaGateMode}
 * applied as {@code sigmoid(g2) * o} ({@link CpuOps#gluRows} SIGMUL), then {@code attn_output}. Working arrays are
 * the scratch slots Q, K, V (conv outputs) and ATTN_OUT (projection input of the convs, then the decay, then
 * {@code g2}) plus bind-time temporaries. Not thread-safe.
 */
public final class CpuKdaMixer implements CpuMixer {

    private CpuMatVec mv;
    private int ubatch;
    private int nHead;
    private int headDim;
    private int dInner;
    private int dConv;
    private float l2Eps;
    private float rmsEps;
    private KdaGateMode gateMode;
    private CpuGatedDeltaNet.ScanShape scanShape;
    private CpuGatedDeltaNet.GateShape gateShape;

    private CpuTensorRef wQkv;
    private CpuTensorRef[] wSeparate;
    private CpuTensorRef wFa;
    private CpuTensorRef wFb;
    private CpuTensorRef wBeta;
    private CpuTensorRef wGa;
    private CpuTensorRef wGb;
    private CpuTensorRef wG;
    private CpuTensorRef wOut;
    /**
     * LoRA deltas of every projection, added right after the product as {@code build_lora_mm} does
     * (llama-graph.cpp:1629-1667); llama.cpp's kimi-linear graph multiplies these weights plainly.
     */
    private CpuProjectionDeltas.Site qkvDeltas;
    private CpuProjectionDeltas.Site[] separateDeltas;
    private CpuProjectionDeltas.Site faDeltas;
    private CpuProjectionDeltas.Site fbDeltas;
    private CpuProjectionDeltas.Site betaDeltas;
    private CpuProjectionDeltas.Site gaDeltas;
    private CpuProjectionDeltas.Site gbDeltas;
    private CpuProjectionDeltas.Site gDeltas;
    private CpuProjectionDeltas.Site outDeltas;
    private float[] normW;
    private final float[][] convW = new float[3][];
    private float[] dtBias;
    private float[] ssmA;

    private final float[][] qkv = new float[3][];
    private float[] work;
    private float[] fa;
    private float[] ga;
    private float[] bl;
    private float[] g;
    private float[] beta;
    private CpuGatedDeltaNet.Rows qRows;
    private CpuGatedDeltaNet.Rows kRows;
    private CpuGatedDeltaNet.Rows vRows;
    private CpuGatedDeltaNet.Rows fRows;
    private CpuGatedDeltaNet.Rows bRows;
    private DeltaNetBlocks.CpuStates states;
    private float[][] working;

    private boolean bound;

    /**
     * @throws IllegalArgumentException when the layer is not a KDA layer
     * @throws IllegalStateException    when a tensor, state or scratch slot has the wrong size, the layer has no
     *                                  gate mode, or the mixer is already bound
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound) {
            throw new IllegalStateException("KDA mixer is already bound");
        }
        DeltaNetBlocks.checkKind(MixerKind.KDA, layer);
        RecurrentSpec r = layer.recurrent();
        bindShape(ctx, layer, r);
        bindWeights(ctx, layer);
        bindScratch(ctx, layer, r);
        bound = true;
    }

    /** Head geometry, gate mode, conv-state shape and epsilons of the layer. */
    private void bindShape(CpuForwardContext ctx, LayerSpec layer, RecurrentSpec r) {
        mv = ctx.matVec();
        ubatch = ctx.scratch().ubatch();
        scanShape = CpuGatedDeltaNet.ScanShape.of(r);
        gateShape = CpuGatedDeltaNet.GateShape.of(r);
        gateMode = r.kdaGateMode();
        if (gateMode == null) {
            throw new IllegalStateException("layer " + DeltaNetBlocks.describe(layer)
                    + ": KDA needs an output-gate mode");
        }
        nHead = r.nHead();
        headDim = r.headDim();
        dInner = nHead * headDim;
        dConv = r.convSteps() + 1;
        SsmKernels.checkConvShape(dInner, dConv);
        if (r.convStateFloats() != 3L * dInner * (dConv - 1)) {
            throw new IllegalStateException("layer " + DeltaNetBlocks.describe(layer) + ": KDA conv state of "
                    + r.convStateFloats() + " floats is not three slices of " + dInner + " x " + (dConv - 1));
        }
        l2Eps = DeltaNetBlocks.l2Eps(ctx.spec());
        rmsEps = DeltaNetBlocks.rmsEps(ctx.spec());
    }

    /** Projection matrices, their LoRA delta sites, the norm weight and the conv / decay vectors. */
    private void bindWeights(CpuForwardContext ctx, LayerSpec layer) {
        int nEmbd = DeltaNetBlocks.mixerWidth(layer);
        CpuWeights ws = ctx.weights();
        wQkv = DeltaNetBlocks.cpuMatrixOrNull(ws, layer, TensorNames.ATTN_QKV, nEmbd, 3L * dInner);
        if (wQkv == null) {
            wSeparate = new CpuTensorRef[]{
                    DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_Q, nEmbd, dInner),
                    DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_K, nEmbd, dInner),
                    DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_V, nEmbd, dInner)};
        }
        if (ws.layerOrNull(layer, TensorNames.weight(TensorNames.SSM_F_B)) != null) {
            wFa = DeltaNetBlocks.cpuMatrixCols(ws, layer, TensorNames.SSM_F_A, nEmbd);
            wFb = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.SSM_F_B, wFa.rows(), dInner);
        } else {
            wFa = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.SSM_F_A, nEmbd, dInner);
        }
        wBeta = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.SSM_BETA, nEmbd, nHead);
        bindOutputGate(ws, layer, nEmbd);
        wOut = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_OUT, dInner, nEmbd);
        bindDeltas(ctx.projectionDeltas(), layer);
        normW = DeltaNetBlocks.cpuFloats(ws, layer, TensorNames.weight(TensorNames.SSM_NORM), headDim);
        String[] convNames = {TensorNames.SSM_CONV1D_Q, TensorNames.SSM_CONV1D_K, TensorNames.SSM_CONV1D_V};
        for (int i = 0; i < 3; i++) {
            convW[i] = DeltaNetBlocks.cpuFloats(ws, layer, TensorNames.weight(convNames[i]), (long) dInner * dConv);
        }
        dtBias = DeltaNetBlocks.cpuFloats(ws, layer, TensorNames.bias(TensorNames.SSM_DT), dInner);
        ssmA = DeltaNetBlocks.cpuEither(ws, layer, TensorNames.SSM_A_NOSCAN, nHead);
    }

    /**
     * The output-gate projection of the gate mode: low-rank {@code ssm_g_a . ssm_g_b}, full {@code ssm_g} or full
     * {@code ssm_g_a}.
     */
    private void bindOutputGate(CpuWeights ws, LayerSpec layer, int nEmbd) {
        switch (gateMode) {
            case LOWRANK:
                wGa = DeltaNetBlocks.cpuMatrixCols(ws, layer, TensorNames.SSM_G_A, nEmbd);
                wGb = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.SSM_G_B, wGa.rows(), dInner);
                break;
            case FULL_SSM_G:
                wG = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.SSM_G, nEmbd, dInner);
                break;
            case FULL_SSM_G_A:
                wG = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.SSM_G_A, nEmbd, dInner);
                break;
            default:
                throw new IllegalStateException("unhandled KDA gate mode " + gateMode);
        }
    }

    /** Scratch rows and temporaries of one ubatch, the row views of the scan and the layer's recurrent states. */
    private void bindScratch(CpuForwardContext ctx, LayerSpec layer, RecurrentSpec r) {
        CpuScratch scratch = ctx.scratch();
        qkv[0] = slot(scratch, CpuScratch.Q, dInner);
        qkv[1] = slot(scratch, CpuScratch.K, dInner);
        qkv[2] = slot(scratch, CpuScratch.V, dInner);
        work = slot(scratch, CpuScratch.ATTN_OUT, wQkv != null ? 3 * dInner : dInner);
        if (wFb != null) {
            fa = scratch.temp(GpuKdaMixer.TEMP_FA, ubatch * DeltaNetBlocks.toInt(wFa.rows(), "ssm_f_a rows"));
        }
        if (wGa != null) {
            ga = scratch.temp(GpuKdaMixer.TEMP_GA, ubatch * DeltaNetBlocks.toInt(wGa.rows(), "ssm_g_a rows"));
        }
        bl = scratch.temp(GpuKdaMixer.TEMP_B, ubatch * nHead);
        g = scratch.temp(DeltaNetBlocks.TEMP_G, ubatch * dInner);
        beta = scratch.temp(DeltaNetBlocks.TEMP_BETA, ubatch * nHead);
        qRows = new CpuGatedDeltaNet.Rows(qkv[0], 0, dInner);
        kRows = new CpuGatedDeltaNet.Rows(qkv[1], 0, dInner);
        vRows = new CpuGatedDeltaNet.Rows(qkv[2], 0, dInner);
        fRows = new CpuGatedDeltaNet.Rows(work, 0, dInner);
        bRows = new CpuGatedDeltaNet.Rows(bl, 0, nHead);
        states = DeltaNetBlocks.CpuStates.bind(ctx, layer, r.convStateFloats(), r.stateFloats());
        working = new float[][]{qkv[0], qkv[1], qkv[2], work, bl, g, beta, fa != null ? fa : work,
                ga != null ? ga : work};
    }

    /** The delta sites of the projections the layer runs. */
    private void bindDeltas(CpuProjectionDeltas deltas, LayerSpec layer) {
        if (wQkv != null) {
            qkvDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_QKV), wQkv, mv);
        } else {
            String[] names = {TensorNames.ATTN_Q, TensorNames.ATTN_K, TensorNames.ATTN_V};
            separateDeltas = new CpuProjectionDeltas.Site[names.length];
            for (int i = 0; i < names.length; i++) {
                separateDeltas[i] = deltas.serve(layer, TensorNames.weight(names[i]), wSeparate[i], mv);
            }
        }
        faDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_F_A), wFa, mv);
        if (wFb != null) {
            fbDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_F_B), wFb, mv);
        }
        betaDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_BETA), wBeta, mv);
        if (wGa != null) {
            gaDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_G_A), wGa, mv);
            gbDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_G_B), wGb, mv);
        } else {
            String g = gateMode == KdaGateMode.FULL_SSM_G_A ? TensorNames.SSM_G_A : TensorNames.SSM_G;
            gDeltas = deltas.serve(layer, TensorNames.weight(g), wG, mv);
        }
        outDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_OUT), wOut, mv);
    }

    private float[] slot(CpuScratch scratch, String name, int width) {
        float[] s = scratch.slot(name);
        long need = (long) ubatch * width;
        if (s.length < need) {
            throw new IllegalStateException("scratch slot " + name + " holds " + s.length + " floats, the KDA mixer"
                    + " needs " + need);
        }
        return s;
    }

    @Override
    public void decode(int pos, float[] in, float[] out) {
        run(1, in, out);
    }

    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        run(n, in, out);
    }

    private void run(int n, float[] in, float[] out) {
        if (!bound) {
            throw new IllegalStateException("KDA mixer is not bound");
        }
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("KDA step of " + n + " tokens is outside [1, " + ubatch + "]");
        }
        DeltaNetBlocks.checkStepArrays("KDA mixer", in, out, working);
        states.clearTransient();
        if (wQkv != null) {
            mv.matmul(wQkv, in, 0, work, 0, n);
            qkvDeltas.applyRows(n, in, 0, work, 0);
            for (int i = 0; i < 3; i++) {
                conv(i, n, i * dInner, 3 * dInner);
            }
        } else {
            for (int i = 0; i < 3; i++) {
                mv.matmul(wSeparate[i], in, 0, work, 0, n);
                separateDeltas[i].applyRows(n, in, 0, work, 0);
                conv(i, n, 0, dInner);
            }
        }

        if (wFb != null) {
            mv.matmul(wFa, in, 0, fa, 0, n);
            faDeltas.applyRows(n, in, 0, fa, 0);
            mv.matmul(wFb, fa, 0, work, 0, n);
            fbDeltas.applyRows(n, fa, 0, work, 0);
        } else {
            mv.matmul(wFa, in, 0, work, 0, n);
            faDeltas.applyRows(n, in, 0, work, 0);
        }
        mv.matmul(wBeta, in, 0, bl, 0, n);
        betaDeltas.applyRows(n, in, 0, bl, 0);
        CpuGatedDeltaNet.gatePrep(gateShape, n, fRows, bRows, dtBias, ssmA, g, 0, beta, 0);

        CpuOps.norm(NormKind.L2_GDN, l2Eps, qkv[0], 0, qkv[0], 0, n * nHead, headDim, null, null);
        CpuOps.norm(NormKind.L2_GDN, l2Eps, qkv[1], 0, qkv[1], 0, n * nHead, headDim, null, null);
        CpuParallel.parallelFor(nHead, 1, (h0, h1) -> CpuGatedDeltaNet.scanHeads(scanShape, n, qRows, kRows, vRows,
                g, 0, beta, 0, states.scan, 0, vRows, h0, h1));

        float[] o = qkv[2];
        CpuOps.norm(NormKind.RMS, rmsEps, o, 0, o, 0, n * nHead, headDim, normW, null);
        if (wGa != null) {
            mv.matmul(wGa, in, 0, ga, 0, n);
            gaDeltas.applyRows(n, in, 0, ga, 0);
            mv.matmul(wGb, ga, 0, work, 0, n);
            gbDeltas.applyRows(n, ga, 0, work, 0);
        } else {
            mv.matmul(wG, in, 0, work, 0, n);
            gDeltas.applyRows(n, in, 0, work, 0);
        }
        CpuOps.gluRows(GluOp.SIGMUL, ActParams.NONE, n, dInner, work, 0, dInner, o, 0, dInner, o, 0, dInner);
        mv.matmul(wOut, o, 0, out, 0, n);
        outDeltas.applyRows(n, o, 0, out, 0);
    }

    /** Causal conv of slice {@code i} (q, k or v) from {@code work} into its slot, with SiLU and no bias. */
    private void conv(int i, int n, int xOff, int xStride) {
        CpuSsmConv.forward(dInner, dConv, n, SsmKernels.CONV_FLAG_SILU, work, xOff, xStride, convW[i], null,
                states.conv, 0, i * dInner, qkv[i], 0, dInner);
    }

    @Override
    public String toString() {
        return "CpuKdaMixer{" + (bound ? nHead + " heads of " + headDim + ", " + gateShape.decayGate() + ", "
                + gateMode : "unbound") + '}';
    }
}
