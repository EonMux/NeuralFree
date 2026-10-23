package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.gguf.arch.ActOp;
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
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * CPU lightning attention mixer, the twin of {@link GpuLightningMixer} (same buffers, same order, same math):
 * {@code attn_qkv} with SiLU into Q, {@link CpuLightning#forwardHeads} (heads in parallel, layer-scaled slopes with
 * {@code layer_scale = 1}) into K, the RMS norm of the whole {@code H*D} row with {@code attn_norm_2}, the
 * {@code sigmoid(attn_gate(x))} gate from V, then {@code attn_output}. Not thread-safe.
 */
public final class CpuLightningMixer implements CpuMixer {

    private CpuMatVec mv;
    private int ubatch;
    private int nHead;
    private int headDim;
    private int width;
    private float rmsEps;
    private float[] slopes;

    private CpuTensorRef wQkv;
    private CpuTensorRef wGate;
    private CpuTensorRef wOut;
    /** LoRA deltas of {@code attn_qkv}, {@code attn_gate} and {@code attn_output} ({@code build_lora_mm}). */
    private CpuProjectionDeltas.Site qkvDeltas;
    private CpuProjectionDeltas.Site gateDeltas;
    private CpuProjectionDeltas.Site outDeltas;
    private float[] normW;

    private float[] qkv;
    private float[] attn;
    private float[] gate;
    private DeltaNetBlocks.CpuStates states;
    private float[][] working;

    private boolean bound;

    /**
     * @throws IllegalArgumentException when the layer is not a lightning layer or its head dim exceeds the kernel
     * @throws IllegalStateException    when a tensor, state or scratch slot has the wrong size, the slopes are
     *                                  missing, or the mixer is already bound
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound) {
            throw new IllegalStateException("lightning mixer is already bound");
        }
        DeltaNetBlocks.checkKind(MixerKind.LIGHTNING, layer);
        RecurrentSpec r = layer.recurrent();
        mv = ctx.matVec();
        CpuScratch scratch = ctx.scratch();
        ubatch = scratch.ubatch();
        nHead = r.nHead();
        headDim = r.headDim();
        DeltaNetKernels.validateLightning(nHead, headDim);
        width = nHead * headDim;
        slopes = r.slopes();
        if (slopes == null || slopes.length != nHead) {
            throw new IllegalStateException("layer " + DeltaNetBlocks.describe(layer) + ": lightning attention needs "
                    + nHead + " slopes, the spec has " + (slopes == null ? "none" : slopes.length));
        }
        rmsEps = DeltaNetBlocks.rmsEps(ctx.spec());
        int nEmbd = DeltaNetBlocks.mixerWidth(layer);

        CpuWeights ws = ctx.weights();
        wQkv = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_QKV, nEmbd, 3L * width);
        wGate = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_GATE, nEmbd, width);
        wOut = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_OUT, width, nEmbd);
        qkvDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.ATTN_QKV), wQkv, mv);
        gateDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.ATTN_GATE), wGate, mv);
        outDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.ATTN_OUT), wOut, mv);
        normW = DeltaNetBlocks.cpuFloats(ws, layer, TensorNames.weight(TensorNames.ATTN_NORM_2), width);

        qkv = slot(scratch, CpuScratch.Q, 3 * width);
        attn = slot(scratch, CpuScratch.K, width);
        gate = slot(scratch, CpuScratch.V, width);
        states = DeltaNetBlocks.CpuStates.bind(ctx, layer, 0, r.stateFloats());
        working = new float[][]{qkv, attn, gate};
        bound = true;
    }

    private float[] slot(CpuScratch scratch, String name, int rowWidth) {
        float[] s = scratch.slot(name);
        long need = (long) ubatch * rowWidth;
        if (s.length < need) {
            throw new IllegalStateException("scratch slot " + name + " holds " + s.length + " floats, the lightning"
                    + " mixer needs " + need);
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
            throw new IllegalStateException("lightning mixer is not bound");
        }
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("lightning step of " + n + " tokens is outside [1, " + ubatch + "]");
        }
        DeltaNetBlocks.checkStepArrays("lightning mixer", in, out, working);
        states.clearTransient();
        mv.matmul(wQkv, in, 0, qkv, 0, n);
        qkvDeltas.applyRows(n, in, 0, qkv, 0);
        CpuOps.unary(ActOp.SILU, ActParams.NONE, qkv, 0, qkv, 0, 3 * n * width);
        CpuParallel.parallelFor(nHead, 1, (h0, h1) -> CpuLightning.forwardHeads(n, nHead, headDim, 1.0f, slopes,
                qkv, 0, 3 * width, states.scan, 0, attn, 0, width, h0, h1));
        CpuOps.norm(NormKind.RMS, rmsEps, attn, 0, attn, 0, n, width, normW, null);
        mv.matmul(wGate, in, 0, gate, 0, n);
        gateDeltas.applyRows(n, in, 0, gate, 0);
        CpuOps.gluRows(GluOp.SIGMUL, ActParams.NONE, n, width, gate, 0, width, attn, 0, width, attn, 0, width);
        mv.matmul(wOut, attn, 0, out, 0, n);
        outDeltas.applyRows(n, attn, 0, out, 0);
    }

    @Override
    public String toString() {
        return "CpuLightningMixer{" + (bound ? nHead + " heads of " + headDim : "unbound") + '}';
    }
}
