package com.eonmux.neuralfree.gguf.ops.ssm;

import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * CPU twin of {@link GpuShortConvMixer}: {@code shortconv.in_proj -> Q[T][3E] = [b | c | x]}, then
 * {@link CpuShortConv#gatedCausal} (with conv state) or {@link CpuShortConv#gatedCentered} (stateless) computing
 * {@code c * conv(b * x)} into V with {@code bx} in K, then {@code shortconv.out_proj(V) -> out}. Every working
 * array comes from {@link CpuScratch}; a step allocates nothing. Not thread-safe.
 */
public final class CpuShortConvMixer implements CpuMixer {

    private final MixerKind kind;

    private CpuMatVec mv;
    private int ubatch;
    private int nEmbd;
    private int width;
    private CpuTensorRef inProj;
    private CpuTensorRef outProj;
    /** LoRA deltas of {@code shortconv.in_proj} / {@code out_proj} ({@code build_lora_mm}, models/lfm2.cpp). */
    private CpuProjectionDeltas.Site inDeltas;
    private CpuProjectionDeltas.Site outDeltas;
    private float[] convW;
    private float[] convState;
    private float[] q;
    private float[] k;
    private float[] v;
    private boolean bound;
    private boolean closed;

    /**
     * @param kind SHORTCONV or SHORTCONV_CENTERED
     * @throws IllegalArgumentException for another kind
     */
    CpuShortConvMixer(MixerKind kind) {
        this.kind = Objects.requireNonNull(kind, "kind");
        if (kind != MixerKind.SHORTCONV && kind != MixerKind.SHORTCONV_CENTERED) {
            throw new IllegalArgumentException("CpuShortConvMixer does not serve " + kind);
        }
    }

    /**
     * @throws IllegalArgumentException when the layer is not of this mixer's kind
     * @throws IllegalStateException    when a causal layer has no conv state, a tensor or state has the wrong size,
     *                                  a scratch slot is too small, or the mixer is bound or closed
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException(kind + " mixer is already bound or closed");
        }
        SsmBlocks.checkKind(kind, layer);
        RecurrentSpec r = layer.recurrent();
        mv = ctx.matVec();
        CpuScratch scratch = ctx.scratch();
        ubatch = scratch.ubatch();
        nEmbd = r.nEmbd();
        width = r.lCache();
        if (layer.residual().streamWidth() != nEmbd) {
            throw new IllegalStateException("layer " + SsmBlocks.describe(layer) + ": " + kind + " expects a stream of "
                    + nEmbd + " floats, the layer streams " + layer.residual().streamWidth());
        }
        if (kind == MixerKind.SHORTCONV) {
            SsmKernels.checkConvShape(nEmbd, width);
            if (!ctx.hasMemory()) {
                throw new IllegalStateException("layer " + SsmBlocks.describe(layer) + " needs recurrent memory");
            }
            RecurrentStateAccessor state = ctx.memory().recurrent(layer.il());
            if (state == null) {
                throw new IllegalStateException("layer " + SsmBlocks.describe(layer) + " has no recurrent state");
            }
            SsmBlocks.stateFloats(state.tensors(), LayerMemoryPlan.STATE_CONV, (long) (width - 1) * nEmbd, layer);
            convState = state.tensor(LayerMemoryPlan.STATE_CONV);
        } else {
            SsmKernels.checkCenteredShape(nEmbd, width);
        }
        CpuWeights ws = ctx.weights();
        inProj = SsmBlocks.cpuMatrix(ws, layer, TensorNames.SHORTCONV_INPROJ, nEmbd, 3L * nEmbd);
        outProj = SsmBlocks.cpuMatrix(ws, layer, TensorNames.SHORTCONV_OUTPROJ, nEmbd, nEmbd);
        inDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.SHORTCONV_INPROJ), inProj, mv);
        outDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.SHORTCONV_OUTPROJ), outProj,
                mv);
        convW = SsmBlocks.cpuFloats(ws, layer, TensorNames.weight(TensorNames.SHORTCONV_CONV), (long) nEmbd * width,
                true);
        q = slot(scratch, CpuScratch.Q, 3L * nEmbd);
        k = slot(scratch, CpuScratch.K, nEmbd);
        v = slot(scratch, CpuScratch.V, nEmbd);
        bound = true;
    }

    private float[] slot(CpuScratch scratch, String name, long rowWidth) {
        float[] s = scratch.slot(name);
        long need = ubatch * rowWidth;
        if (s.length < need) {
            throw new IllegalStateException(kind + " mixer needs " + need + " floats in scratch slot " + name
                    + ", it holds " + s.length);
        }
        return s;
    }

    @Override
    public void decode(int pos, float[] in, float[] out) {
        forward(1, in, out);
    }

    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        forward(n, in, out);
    }

    private void forward(int t, float[] in, float[] out) {
        if (!bound || closed) {
            throw new IllegalStateException(kind + " mixer is " + (closed ? "closed" : "not bound"));
        }
        if (t < 1 || t > ubatch) {
            throw new IllegalArgumentException(kind + " step of " + t + " tokens is outside [1, " + ubatch + "]");
        }
        int n = t * nEmbd;
        if (in.length < n || out.length < n) {
            throw new IllegalArgumentException(kind + " step of " + t + " tokens needs " + n
                    + " floats of input and output");
        }
        if (in == out || in == q || in == k || in == v || out == q || out == k || out == v) {
            throw new IllegalArgumentException(kind + " mixer input and output must be distinct caller arrays");
        }
        mv.matmul(inProj, in, 0, q, 0, t);
        inDeltas.applyRows(t, in, 0, q, 0);
        if (kind == MixerKind.SHORTCONV) {
            CpuShortConv.gatedCausal(nEmbd, width, t, q, 0, 3 * nEmbd, convW, convState, 0, k, v, 0, nEmbd);
        } else {
            CpuShortConv.gatedCentered(nEmbd, width, t, q, 0, 3 * nEmbd, convW, k, v, 0, nEmbd);
        }
        mv.matmul(outProj, v, 0, out, 0, t);
        outDeltas.applyRows(t, v, 0, out, 0);
    }

    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "CpuShortConvMixer{" + kind + (bound ? ", n_embd=" + nEmbd : ", unbound") + (closed ? ", closed" : "")
                + '}';
    }
}
