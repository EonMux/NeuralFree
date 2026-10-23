package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.gguf.arch.GluOp;
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
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.ssm.CpuSsmConv;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * CPU gated DeltaNet mixer, the twin of {@link GpuGdnMixer} (same buffers, same order, same math): projections with
 * {@link CpuMatVec}, the de-interleave of the legacy {@code ssm_in} rows, {@link CpuSsmConv} with SiLU, the
 * {@code L2_GDN} norm of every q and k head ({@link NormMath}, in place in the conv output), {@link CpuGatedDeltaNet}
 * gate prep and scan (value heads in parallel), the RMS norm of every value head and the {@code silu(z)} gate
 * ({@link CpuOps}), then {@code ssm_out}. Working arrays are the scratch slots Q (conv input), K (conv output),
 * V ({@code z}) and ATTN_OUT (scan output; first the legacy {@code ssm_in} rows) plus bind-time temporaries. Not
 * thread-safe.
 */
public final class CpuGdnMixer implements CpuMixer {

    private CpuMatVec mv;
    private int ubatch;
    private int nK;
    private int nV;
    private int headK;
    private int headV;
    private int keyDim;
    private int valDim;
    private int convCh;
    private int dConv;
    private boolean legacy;
    private float l2Eps;
    private float rmsEps;
    private CpuGatedDeltaNet.ScanShape scanShape;
    private CpuGatedDeltaNet.GateShape gateShape;

    private CpuTensorRef wQkv;
    private CpuTensorRef wGate;
    private CpuTensorRef wIn;
    private CpuTensorRef wBa;
    private CpuTensorRef wAlpha;
    private CpuTensorRef wBeta;
    private CpuTensorRef wOut;
    /** LoRA deltas of the projections ({@code build_lora_mm}, models/qwen35.cpp, models/qwen3next.cpp). */
    private CpuProjectionDeltas.Site qkvDeltas;
    private CpuProjectionDeltas.Site gateDeltas;
    private CpuProjectionDeltas.Site inDeltas;
    private CpuProjectionDeltas.Site baDeltas;
    private CpuProjectionDeltas.Site alphaDeltas;
    private CpuProjectionDeltas.Site betaDeltas;
    private CpuProjectionDeltas.Site outDeltas;
    private float[] normW;
    private float[] convW;
    private float[] dtBias;
    private float[] ssmA;

    private float[] x;
    private float[] conv;
    private float[] z;
    private float[] o;
    private float[] g;
    private float[] beta;
    private CpuGatedDeltaNet.Rows qRows;
    private CpuGatedDeltaNet.Rows kRows;
    private CpuGatedDeltaNet.Rows vRows;
    private CpuGatedDeltaNet.Rows oRows;
    private CpuGatedDeltaNet.Rows aRows;
    private CpuGatedDeltaNet.Rows bRows;
    private DeltaNetBlocks.CpuStates states;
    private float[][] working;

    private boolean bound;

    /**
     * @throws IllegalArgumentException when the layer is not a GDN layer
     * @throws IllegalStateException    when a tensor, state or scratch slot has the wrong size, or the mixer is
     *                                  already bound
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound) {
            throw new IllegalStateException("GDN mixer is already bound");
        }
        DeltaNetBlocks.checkKind(MixerKind.GDN, layer);
        RecurrentSpec r = layer.recurrent();
        bindShape(ctx, r);
        bindWeights(ctx, layer, r);
        bindScratch(ctx, layer, r);
        bound = true;
    }

    /** Head geometry, conv shape and epsilons of the layer. */
    private void bindShape(CpuForwardContext ctx, RecurrentSpec r) {
        mv = ctx.matVec();
        ubatch = ctx.scratch().ubatch();
        scanShape = CpuGatedDeltaNet.ScanShape.of(r);
        gateShape = CpuGatedDeltaNet.GateShape.of(r);
        nK = r.nKHeads();
        nV = r.nVHeads();
        headK = r.headKDim();
        headV = r.headVDim();
        keyDim = nK * headK;
        valDim = nV * headV;
        convCh = r.convChannels();
        legacy = r.legacySsmIn();
        dConv = r.dConv();
        SsmKernels.checkConvShape(convCh, dConv);
        l2Eps = DeltaNetBlocks.l2Eps(ctx.spec());
        rmsEps = DeltaNetBlocks.rmsEps(ctx.spec());
    }

    /** Projection matrices, their LoRA delta sites, the norm weight and the conv / decay vectors. */
    private void bindWeights(CpuForwardContext ctx, LayerSpec layer, RecurrentSpec r) {
        int nEmbd = DeltaNetBlocks.mixerWidth(layer);
        CpuWeights ws = ctx.weights();
        if (legacy) {
            long perKh = 2L * headK + 2L * (nV / nK) * headV;
            wIn = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.SSM_IN, nEmbd, nK * perKh);
        } else {
            wQkv = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_QKV, nEmbd, convCh);
            wGate = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_GATE, nEmbd, valDim);
        }
        if (r.fusedBetaAlpha()) {
            wBa = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.SSM_BETA_ALPHA, nEmbd, 2L * nV);
        } else {
            wAlpha = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.SSM_ALPHA, nEmbd, nV);
            wBeta = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.SSM_BETA, nEmbd, nV);
        }
        wOut = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.SSM_OUT, valDim, nEmbd);
        CpuProjectionDeltas deltas = ctx.projectionDeltas();
        if (legacy) {
            inDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_IN), wIn, mv);
        } else {
            qkvDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_QKV), wQkv, mv);
            gateDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_GATE), wGate, mv);
        }
        if (wBa != null) {
            baDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_BETA_ALPHA), wBa, mv);
        } else {
            alphaDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_ALPHA), wAlpha, mv);
            betaDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_BETA), wBeta, mv);
        }
        outDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_OUT), wOut, mv);
        normW = DeltaNetBlocks.cpuFloats(ws, layer, TensorNames.weight(TensorNames.SSM_NORM), headV);
        convW = DeltaNetBlocks.cpuFloats(ws, layer, TensorNames.weight(TensorNames.SSM_CONV1D), (long) convCh * dConv);
        dtBias = DeltaNetBlocks.cpuFloats(ws, layer, TensorNames.bias(TensorNames.SSM_DT), nV);
        ssmA = DeltaNetBlocks.cpuEither(ws, layer, TensorNames.SSM_A_NOSCAN, nV);
    }

    /** Scratch rows and temporaries of one ubatch, the row views of the scan and the layer's recurrent states. */
    private void bindScratch(CpuForwardContext ctx, LayerSpec layer, RecurrentSpec r) {
        CpuScratch scratch = ctx.scratch();
        x = slot(scratch, CpuScratch.Q, convCh);
        conv = slot(scratch, CpuScratch.K, convCh);
        z = slot(scratch, CpuScratch.V, valDim);
        o = slot(scratch, CpuScratch.ATTN_OUT, legacy ? DeltaNetBlocks.toInt(wIn.rows(), "ssm_in rows") : valDim);
        int abStride = r.fusedBetaAlpha() ? 2 * nV : nV;
        float[] aArr = scratch.temp(GpuGdnMixer.TEMP_A, ubatch * abStride);
        float[] bArr = r.fusedBetaAlpha() ? aArr : scratch.temp(GpuGdnMixer.TEMP_B, ubatch * nV);
        g = scratch.temp(DeltaNetBlocks.TEMP_G, ubatch * nV);
        beta = scratch.temp(DeltaNetBlocks.TEMP_BETA, ubatch * nV);
        qRows = new CpuGatedDeltaNet.Rows(conv, 0, convCh);
        kRows = new CpuGatedDeltaNet.Rows(conv, keyDim, convCh);
        vRows = new CpuGatedDeltaNet.Rows(conv, 2 * keyDim, convCh);
        oRows = new CpuGatedDeltaNet.Rows(o, 0, valDim);
        aRows = new CpuGatedDeltaNet.Rows(aArr, 0, abStride);
        bRows = new CpuGatedDeltaNet.Rows(bArr, 0, abStride);
        states = DeltaNetBlocks.CpuStates.bind(ctx, layer, r.convStateFloats(), r.stateFloats());
        working = new float[][]{x, conv, z, o, aArr, bArr, g, beta};
    }

    private float[] slot(CpuScratch scratch, String name, int width) {
        float[] s = scratch.slot(name);
        long need = (long) ubatch * width;
        if (s.length < need) {
            throw new IllegalStateException("scratch slot " + name + " holds " + s.length + " floats, the GDN mixer"
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
            throw new IllegalStateException("GDN mixer is not bound");
        }
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("GDN step of " + n + " tokens is outside [1, " + ubatch + "]");
        }
        DeltaNetBlocks.checkStepArrays("GDN mixer", in, out, working);
        states.clearTransient();
        project(n, in);
        CpuSsmConv.forward(convCh, dConv, n, SsmKernels.CONV_FLAG_SILU, x, 0, convCh, convW, null,
                states.conv, 0, 0, conv, 0, convCh);
        CpuParallel.parallelFor(n, 1, (t0, t1) -> {
            for (int t = t0; t < t1; t++) {
                int off = t * convCh;
                NormMath.rowNorm(NormKind.L2_GDN, l2Eps, 2 * nK, headK, headK, 1, conv, off, null, null, conv, off);
            }
        });
        CpuGatedDeltaNet.gatePrep(gateShape, n, aRows, bRows, dtBias, ssmA, g, 0, beta, 0);
        CpuParallel.parallelFor(nV, 1, (h0, h1) -> CpuGatedDeltaNet.scanHeads(scanShape, n, qRows, kRows, vRows,
                g, 0, beta, 0, states.scan, 0, oRows, h0, h1));
        CpuOps.norm(NormKind.RMS, rmsEps, o, 0, o, 0, n * nV, headV, normW, null);
        CpuOps.gluRows(GluOp.SWIGLU, ActParams.NONE, n, valDim, z, 0, valDim, o, 0, valDim, o, 0, valDim);
        mv.matmul(wOut, o, 0, out, 0, n);
        outDeltas.applyRows(n, o, 0, out, 0);
    }

    private void project(int n, float[] in) {
        if (legacy) {
            mv.matmul(wIn, in, 0, o, 0, n);
            inDeltas.applyRows(n, in, 0, o, 0);
            deinterleave(n);
        } else {
            mv.matmul(wQkv, in, 0, x, 0, n);
            qkvDeltas.applyRows(n, in, 0, x, 0);
            mv.matmul(wGate, in, 0, z, 0, n);
            gateDeltas.applyRows(n, in, 0, z, 0);
        }
        if (wBa != null) {
            mv.matmul(wBa, in, 0, aRows.data(), 0, n);
            baDeltas.applyRows(n, in, 0, aRows.data(), 0);
        } else {
            mv.matmul(wAlpha, in, 0, aRows.data(), 0, n);
            alphaDeltas.applyRows(n, in, 0, aRows.data(), 0);
            mv.matmul(wBeta, in, 0, bRows.data(), 0, n);
            betaDeltas.applyRows(n, in, 0, bRows.data(), 0);
        }
    }

    /** {@link GpuGdnMixer}'s legacy de-interleave: per key head {@code [q | k | v(R heads) | z(R heads)]}. */
    private void deinterleave(int n) {
        int vw = (nV / nK) * headV;
        int perKh = 2 * headK + 2 * vw;
        int rowW = nK * perKh;
        for (int t = 0; t < n; t++) {
            for (int kh = 0; kh < nK; kh++) {
                int s = t * rowW + kh * perKh;
                int d = t * convCh;
                System.arraycopy(o, s, x, d + kh * headK, headK);
                System.arraycopy(o, s + headK, x, d + keyDim + kh * headK, headK);
                System.arraycopy(o, s + 2 * headK, x, d + 2 * keyDim + kh * vw, vw);
                System.arraycopy(o, s + 2 * headK + vw, z, t * valDim + kh * vw, vw);
            }
        }
    }

    @Override
    public String toString() {
        return "CpuGdnMixer{" + (bound ? nK + "k/" + nV + "v heads, " + headK + "x" + headV
                + (legacy ? ", ssm_in" : "") + (wBa != null ? ", ssm_ba" : "") : "unbound") + '}';
    }
}
