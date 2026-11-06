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
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * CPU lightning attention mixer, the twin of {@link GpuLightningMixer} (same buffers, same order, same math):
 * {@code attn_qkv} into Q, {@link CpuLightning#forwardHeads} (heads in parallel, layer-scaled slopes with
 * {@code layer_scale = 1}) into K, the output norm with {@code attn_norm_2}, the {@code sigmoid(attn_gate(x))} gate
 * from V, then {@code attn_output}. A Bailing MoE V2.5 layer inserts vLLM's steps between the projection and the
 * recurrence — SiLU on the fused Q/K/V, the per-head Q/K RMS norms, the NEOX rotation at the step's positions and
 * the {@code head_dim^-0.5} Q scale — and normalizes the output over {@code group_norm_size} groups. Not
 * thread-safe.
 */
public final class CpuLightningMixer implements CpuMixer {

    private CpuMatVec mv;
    private CpuScratch scratch;
    private int ubatch;
    private int nHead;
    private int headDim;
    private int width;
    /** Stream width of the layer (the mixer's input and output row width). */
    private int stream;
    private float rmsEps;
    private float qkEps;
    private float[] slopes;
    private int nNormGroups;
    private boolean linearSilu;
    private boolean qkNorm;
    private float qScale;
    private RopeFreqTable ropeTable;

    private CpuTensorRef wQkv;
    private CpuTensorRef wGate;
    private CpuTensorRef wOut;
    /** LoRA deltas of {@code attn_qkv}, {@code attn_gate} and {@code attn_output} ({@code build_lora_mm}). */
    private CpuProjectionDeltas.Site qkvDeltas;
    private CpuProjectionDeltas.Site gateDeltas;
    private CpuProjectionDeltas.Site outDeltas;
    private float[] normW;
    private float[] qNormW;
    private float[] kNormW;

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
        this.scratch = scratch;
        ubatch = scratch.ubatch();
        nHead = r.nHead();
        headDim = r.headDim();
        DeltaNetKernels.validateLightning(nHead, headDim);
        width = nHead * headDim;
        stream = DeltaNetBlocks.mixerWidth(layer);
        slopes = r.slopes();
        if (slopes == null || slopes.length != nHead) {
            throw new IllegalStateException("layer " + DeltaNetBlocks.describe(layer) + ": lightning attention needs "
                    + nHead + " slopes, the spec has " + (slopes == null ? "none" : slopes.length));
        }
        rmsEps = DeltaNetBlocks.rmsEps(ctx.spec());
        qkEps = ctx.spec().hparams().normRmsEps();
        nNormGroups = r.nNormGroups();
        linearSilu = r.linearSilu();
        qkNorm = r.qkNorm();
        qScale = r.qScale();
        RopeSpec rope = r.rope();
        ropeTable = rope != null ? RopeFreqTable.of(rope, null) : null;
        if (ropeTable != null && (long) scratch.ints(CpuScratch.POS).length < 4L * ubatch) {
            throw new IllegalStateException("scratch slot " + CpuScratch.POS + " holds fewer than 4 x " + ubatch
                    + " positions");
        }
        int nEmbd = DeltaNetBlocks.mixerWidth(layer);

        CpuWeights ws = ctx.weights();
        wQkv = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_QKV, nEmbd, 3L * width);
        wGate = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_GATE, nEmbd, width);
        wOut = DeltaNetBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_OUT, width, nEmbd);
        qkvDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.ATTN_QKV), wQkv, mv);
        gateDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.ATTN_GATE), wGate, mv);
        outDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.ATTN_OUT), wOut, mv);
        normW = DeltaNetBlocks.cpuFloats(ws, layer, TensorNames.weight(TensorNames.ATTN_NORM_2), width);
        if (qkNorm) {
            qNormW = DeltaNetBlocks.cpuFloats(ws, layer, TensorNames.weight(TensorNames.ATTN_Q_NORM), headDim);
            kNormW = DeltaNetBlocks.cpuFloats(ws, layer, TensorNames.weight(TensorNames.ATTN_K_NORM), headDim);
        }

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
        run(1, 0, 1, 0, 0, 0, in, out);
    }

    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        run(n, 0, n, 0, 0, 0, in, out);
    }

    /**
     * One batched step over the rows of {@code s}: one segment runs whole ({@link #run} at its sequence's state
     * row), several segments run one at a time through a scratch relay, each with its own sequence's state row and
     * its rows' place in the step's position slot.
     */
    @Override
    public void batch(com.eonmux.neuralfree.gguf.memory.StepLayout s, float[] in, float[] out) {
        if (s.nSeg() == 1) {
            run(s.nTokens(), 0, s.nTokens(), 0, 0, s.seq(0), in, out);
            return;
        }
        if (!bound) {
            throw new IllegalStateException("lightning mixer is not bound");
        }
        float[] buf = scratch.temp(TEMP_SEGMENT, Math.multiplyExact(ubatch, stream));
        float[] bufOut = scratch.temp(TEMP_SEGMENT_OUT, Math.multiplyExact(ubatch, stream));
        for (int seg = 0; seg < s.nSeg(); seg++) {
            int start = s.tokStart(seg);
            int len = s.len(seg);
            System.arraycopy(in, start * stream, buf, 0, len * stream);
            run(len, start, s.nTokens(), 0, 0, s.seq(seg), buf, bufOut);
            System.arraycopy(bufOut, 0, out, start * stream, len * stream);
        }
    }

    private static final String TEMP_SEGMENT = "CpuLightningMixer.segmentIn";
    private static final String TEMP_SEGMENT_OUT = "CpuLightningMixer.segmentOut";

    /**
     * One step of {@code n} tokens: the fused projection (SiLU when the spec asks for it), the optional per-head
     * Q/K RMS norms, the NEOX rotation at the step's positions {@code [posOff, posOff + n)} of the position slot
     * whose section stride is {@code posStride}, the Q scale, the lightning recurrence and the gated output norm.
     */
    private void run(int n, int posOff, int posStride, int inOff, int outOff, int seq, float[] in, float[] out) {
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
        mv.matmul(wQkv, in, inOff, qkv, 0, n);
        qkvDeltas.applyRows(n, in, inOff, qkv, 0);
        if (linearSilu) {
            CpuOps.unary(ActOp.SILU, ActParams.NONE, qkv, 0, qkv, 0, 3 * n * width);
        }
        if (qkNorm) {
            NormMath.rowNorm(NormKind.RMS, qkEps, n * nHead, headDim, 3 * headDim, 1, qkv, 0, qNormW, null,
                    qkv, 0);
            NormMath.rowNorm(NormKind.RMS, qkEps, n * nHead, headDim, 3 * headDim, 1, qkv, headDim, kNormW, null,
                    qkv, headDim);
        }
        if (ropeTable != null) {
            int[] pos = scratch.ints(CpuScratch.POS);
            CpuRope.apply(ropeTable, qkv, 0, n, nHead, 3 * headDim, 3 * width, 0, pos, posOff, posStride);
            CpuRope.apply(ropeTable, qkv, headDim, n, nHead, 3 * headDim, 3 * width, 0, pos, posOff, posStride);
        }
        if (qScale != 0f) {
            CpuLightning.scaleQ(n, nHead, headDim, 3 * width, qScale, qkv, 0);
        }
        CpuParallel.parallelFor(nHead, 1, (h0, h1) -> CpuLightning.forwardHeads(n, nHead, headDim, 1.0f, slopes,
                qkv, 0, 3 * width, states.scan, states.scanBase(seq), attn, 0, width, h0, h1));
        if (nNormGroups > 1) {
            NormMath.groupRms(n, width, nNormGroups, rmsEps, attn, 0, width, normW, attn, 0);
        } else {
            CpuOps.norm(NormKind.RMS, rmsEps, attn, 0, attn, 0, n, width, normW, null);
        }
        mv.matmul(wGate, in, inOff, gate, 0, n);
        gateDeltas.applyRows(n, in, inOff, gate, 0);
        CpuOps.gluRows(GluOp.SIGMUL, ActParams.NONE, n, width, gate, 0, width, attn, 0, width, attn, 0, width);
        mv.matmul(wOut, attn, 0, out, outOff, n);
        outDeltas.applyRows(n, attn, 0, out, outOff);
    }

    @Override
    public String toString() {
        return "CpuLightningMixer{" + (bound ? nHead + " heads of " + headDim : "unbound") + '}';
    }
}
