package com.eonmux.neuralfree.gguf.ops.ssm;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
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
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuRecurrentState;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * GPU LFM2 short-convolution mixer: {@link MixerKind#SHORTCONV} (causal, with conv state;
 * llm_build_lfm_base::build_lfm_shortconv, models/lfm-base.cpp:9-85) and {@link MixerKind#SHORTCONV_CENTERED} (the
 * stateless centered window of non-causal LFM2: the decision layers, models/lfm2.cpp:355-381, and the
 * non-causal form of models/lfm-base.cpp:49-56, whose state-fed left pad is always zero). With
 * {@code E = n_embd}, {@code L = n_shortconv_l_cache} and {@code T} tokens:
 *
 * <pre>
 * shortconv.in_proj -> Q[T][3E] = [b | c | x]
 * LLM_GLU MUL(b, x) -> K[T][E]                                      bx = b * x
 * SHORTCONV          LLM_SSM_Conv(K, kernel width L, no bias/SiLU, conv state) -> V[T][E]
 * SHORTCONV_CENTERED LLM_ShortConvCentered(K, L taps) -> V[T][E]
 * LLM_GLU MUL(c, V) -> V                                            y = c * conv
 * shortconv.out_proj(V) -> out
 * </pre>
 *
 * The conv launch is a {@link PreparedDispatch} bound once whose token count is re-set per step. The conv state
 * ({@code conv[c][j]}, {@code L - 1} steps) lives in row {@code stateRows[0]} of the layer's
 * recurrent state. LFM2 media masks do not apply to the text-only steps run here. Not thread-safe.
 */
public final class GpuShortConvMixer implements GpuMixer {

    private static final int EL_LOCAL = SsmKernels.ELEMENTWISE_LOCAL;

    private final MixerKind kind;
    private final List<GpuBuffer> owned = new ArrayList<>();

    private GpuOps ops;
    private int ubatch;
    private int nEmbd;
    private GpuWeight inProj;
    private GpuWeight outProj;
    /** LoRA deltas of {@code shortconv.in_proj} / {@code out_proj} ({@code build_lora_mm}, models/lfm2.cpp). */
    private GpuProjectionDeltas.Site inDeltas;
    private GpuProjectionDeltas.Site outDeltas;
    private ScratchSlot q;
    private ScratchSlot k;
    private ScratchSlot v;
    private PreparedDispatch conv;
    private boolean bound;
    private boolean closed;

    /**
     * @param kind SHORTCONV or SHORTCONV_CENTERED
     * @throws IllegalArgumentException for another kind
     */
    GpuShortConvMixer(MixerKind kind) {
        this.kind = Objects.requireNonNull(kind, "kind");
        if (kind != MixerKind.SHORTCONV && kind != MixerKind.SHORTCONV_CENTERED) {
            throw new IllegalArgumentException("GpuShortConvMixer does not serve " + kind);
        }
    }

    /**
     * @throws IllegalArgumentException when the layer is not of this mixer's kind
     * @throws IllegalStateException    when a causal layer has no conv state, a tensor or state has the wrong size,
     *                                  a scratch slot is too small, or the mixer is bound or closed
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException(kind + " mixer is already bound or closed");
        }
        SsmBlocks.checkKind(kind, layer);
        try {
            bindChecked(ctx, layer);
            bound = true;
        } catch (RuntimeException e) {
            closed = true;
            RuntimeException failure = release();
            if (failure != null) {
                e.addSuppressed(failure);
            }
            throw e;
        }
    }

    private void bindChecked(GpuForwardContext ctx, LayerSpec layer) {
        RecurrentSpec r = layer.recurrent();
        ops = ctx.ops();
        GpuScratch scratch = ctx.scratch();
        ubatch = scratch.ubatch();
        nEmbd = r.nEmbd();
        int width = r.lCache();
        if (layer.residual().streamWidth() != nEmbd) {
            throw new IllegalStateException("layer " + SsmBlocks.describe(layer) + ": " + kind + " expects a stream of "
                    + nEmbd + " floats, the layer streams " + layer.residual().streamWidth());
        }
        GpuWeights ws = ctx.weightSource().weightsFor(layer.weightLayer());
        inProj = SsmBlocks.gpuMatrix(ws, layer, TensorNames.SHORTCONV_INPROJ, nEmbd, 3L * nEmbd);
        outProj = SsmBlocks.gpuMatrix(ws, layer, TensorNames.SHORTCONV_OUTPROJ, nEmbd, nEmbd);
        inDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.SHORTCONV_INPROJ), inProj);
        outDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.SHORTCONV_OUTPROJ), outProj);
        GpuBuffer w = SsmBlocks.gpuF32(ctx, ws.layer(layer, TensorNames.weight(TensorNames.SHORTCONV_CONV)),
                (long) nEmbd * width, owned);
        q = scratch.slot(ScratchSlots.Q).require((long) ubatch * 3 * nEmbd);
        k = scratch.slot(ScratchSlots.K).require((long) ubatch * nEmbd);
        v = scratch.slot(ScratchSlots.V).require((long) ubatch * nEmbd);

        if (kind == MixerKind.SHORTCONV) {
            SsmKernels.checkConvShape(nEmbd, width);
            GpuMemory memory = ctx.memory();
            GpuRecurrentState state = memory.recurrent(layer.il());
            if (state == null) {
                throw new IllegalStateException("layer " + SsmBlocks.describe(layer) + " has no recurrent state");
            }
            int stateStride = (int) SsmBlocks.stateFloats(state.tensors(), LayerMemoryPlan.STATE_CONV,
                    (long) (width - 1) * nEmbd, layer);
            Object[] args = {nEmbd, width, 1, 0, nEmbd, k.offset(), nEmbd, v.offset(), 0, stateStride,
                k.buffer(), w, ctx.placeholders().readOnlyFloats(), state.tensor(LayerMemoryPlan.STATE_CONV),
                memory.stepInputs().stateRows(), v.buffer()};
            checkCount(SsmKernels.SSM_CONV, args, SsmKernels.CONV_ARG_COUNT);
            conv = ops.prepare(SsmKernels.SSM_CONV, new long[]{SsmKernels.elementwiseGlobal(nEmbd, EL_LOCAL)},
                    new long[]{EL_LOCAL}, args);
        } else {
            SsmKernels.checkCenteredShape(nEmbd, width);
            if (v.offset() != 0) {
                throw new IllegalStateException("scratch slot " + v.name() + " starts at element " + v.offset()
                        + "; " + SsmKernels.SHORTCONV_CENTERED + " writes it from element 0");
            }
            Object[] args = {nEmbd, width, 1, nEmbd, k.offset(), nEmbd, k.buffer(), w, v.buffer()};
            checkCount(SsmKernels.SHORTCONV_CENTERED, args, SsmKernels.CENTERED_ARG_COUNT);
            conv = ops.prepare(SsmKernels.SHORTCONV_CENTERED,
                    new long[]{SsmKernels.elementwiseGlobal(nEmbd, EL_LOCAL)}, new long[]{EL_LOCAL}, args);
        }
    }

    private static void checkCount(String kernel, Object[] args, int expected) {
        if (args.length != expected) {
            throw new IllegalStateException(kernel + " takes " + expected + " arguments, " + args.length + " built");
        }
    }

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        forward(1, in, out);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        forward(nTokens, in, out);
    }

    private void forward(int t, ScratchSlot in, ScratchSlot out) {
        if (!bound || closed) {
            throw new IllegalStateException(kind + " mixer is " + (closed ? "closed" : "not bound"));
        }
        if (t < 1 || t > ubatch) {
            throw new IllegalArgumentException(kind + " step of " + t + " tokens is outside [1, " + ubatch + "]");
        }
        long n = (long) t * nEmbd;
        in.require(n);
        out.require(n);
        checkExternal(in);
        checkExternal(out);
        int stride = 3 * nEmbd;
        ops.matmul(inProj, in.buffer(), in.offset(), q.buffer(), q.offset(), t);
        inDeltas.applyRows(t, in.buffer(), in.offset(), q.buffer(), q.offset());
        ops.glu(GluOp.MUL, ActParams.NONE, q.buffer(), q.offset(), stride, q.buffer(), q.offset() + 2 * nEmbd, stride,
                k.buffer(), k.offset(), nEmbd, t, nEmbd);
        conv.setInt(kind == MixerKind.SHORTCONV ? SsmKernels.ARG_CONV_N_TOKENS : SsmKernels.ARG_CENTERED_N_TOKENS, t);
        if (kind == MixerKind.SHORTCONV_CENTERED) {
            conv.setGlobal(SsmKernels.elementwiseGlobal(n, EL_LOCAL));
        }
        conv.dispatch();
        ops.glu(GluOp.MUL, ActParams.NONE, q.buffer(), q.offset() + nEmbd, stride, v.buffer(), v.offset(), nEmbd,
                v.buffer(), v.offset(), nEmbd, t, nEmbd);
        ops.matmul(outProj, v.buffer(), v.offset(), out.buffer(), out.offset(), t);
        outDeltas.applyRows(t, v.buffer(), v.offset(), out.buffer(), out.offset());
    }

    /** Rejects a caller view living in one of this mixer's working buffers. */
    private void checkExternal(ScratchSlot s) {
        GpuBuffer b = s.buffer();
        if (b == q.buffer() || b == k.buffer() || b == v.buffer()) {
            throw new IllegalArgumentException(kind + " mixer input/output " + s
                    + " shares a buffer with its working slots");
        }
    }

    /** Releases the prepared conv launch and the F32 copy of the conv weight made at bind. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = release();
        if (failure != null) {
            throw failure;
        }
    }

    private RuntimeException release() {
        RuntimeException first = null;
        if (conv != null) {
            try {
                conv.close();
            } catch (RuntimeException e) {
                first = e;
            }
            conv = null;
        }
        first = Closeables.closeAll(owned, first);
        owned.clear();
        return first;
    }

    @Override
    public String toString() {
        return "GpuShortConvMixer{" + kind + (bound ? ", n_embd=" + nEmbd : ", unbound") + (closed ? ", closed" : "")
                + '}';
    }
}
