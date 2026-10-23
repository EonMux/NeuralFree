package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.ActOp;
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
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * GPU lightning attention mixer ({@link MixerKind#LIGHTNING}, MiniMax-01), following the recurrent branch of
 * models/minimax-01.cpp:277-420 for one sequence. With {@code T} tokens, {@code H} heads of dim {@code D}:
 *
 * <pre>
 * attn_qkv(x) -> Q[T][H][q | k | v]; LLM_Unary SiLU in place                      (minimax-01.cpp:293-299)
 * LLM_LightningAttn(slopes, layer_scale 1, scan state) -> K[T][H*D]               (minimax-01.cpp:288-405)
 * LLM_RowNorm RMS(attn_norm_2) over the whole H*D row                              (minimax-01.cpp:407-410)
 * attn_gate(x) -> V[T][H*D]; LLM_GLU SIGMUL (sigmoid(g) * qkv_norm) in place in K  (minimax-01.cpp:413-419)
 * attn_output -> out                                                               (minimax-01.cpp:421)
 * </pre>
 *
 * The per-head decay slopes of the layer ({@link RecurrentSpec#slopes()}, already multiplied by the layer scale
 * {@code 1 - il/(n_layer-1) + 1e-5}, minimax-01.cpp:288) are uploaded once at bind into a buffer this mixer owns, so
 * the kernel runs with {@code layer_scale = 1}. The kernel has no offsets, so the Q and K slots must start at element
 * 0. The recurrence is a {@link PreparedDispatch} whose token count is re-set per step. State layout:
 * {@code s[h*D*D + v*D + k]}, row {@code stateRows[0]}; llama.cpp also allocates an unused conv state here, which
 * the memory plan omits. The residual {@code f_residual_scale} is the executor's. Not thread-safe.
 */
public final class GpuLightningMixer implements GpuMixer {

    private final List<GpuBuffer> owned = new ArrayList<>();

    private GpuOps ops;
    private int nHead;
    private int headDim;
    private int width;
    private float rmsEps;
    private float[] slopes;

    private GpuWeight wQkv;
    private GpuWeight wGate;
    private GpuWeight wOut;
    private GpuWeight normW;
    /** LoRA deltas of {@code attn_qkv}, {@code attn_gate} and {@code attn_output} ({@code build_lora_mm}). */
    private GpuProjectionDeltas.Site qkvDeltas;
    private GpuProjectionDeltas.Site gateDeltas;
    private GpuProjectionDeltas.Site outDeltas;

    private ScratchSlot qkv;
    private ScratchSlot attn;
    private ScratchSlot gate;
    private DeltaNetBlocks.GpuStates states;
    private GpuBuffer[] working;
    private PreparedDispatch recurrence;

    private boolean bound;
    private boolean closed;

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException when the layer is not a lightning layer or its head dim exceeds the kernel
     * @throws IllegalStateException    when a tensor, state or scratch slot has the wrong size, the slopes are
     *                                  missing, or the mixer is already bound or closed
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException("lightning mixer is already bound or closed");
        }
        DeltaNetBlocks.checkKind(MixerKind.LIGHTNING, layer);
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
        ops = ctx.ops();
        GpuScratch scratch = ctx.scratch();
        int ubatch = scratch.ubatch();
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

        GpuWeights ws = ctx.weightSource().weightsFor(layer.weightLayer());
        wQkv = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_QKV, nEmbd, 3L * width);
        wGate = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_GATE, nEmbd, width);
        wOut = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_OUT, width, nEmbd);
        qkvDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.ATTN_QKV), wQkv);
        gateDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.ATTN_GATE), wGate);
        outDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.ATTN_OUT), wOut);
        normW = DeltaNetBlocks.gpuNormWeight(ctx, ws.layer(layer, TensorNames.weight(TensorNames.ATTN_NORM_2)), width,
                owned);

        qkv = DeltaNetBlocks.atZero(scratch.slot(ScratchSlots.Q).require(3L * ubatch * width),
                DeltaNetKernels.LIGHTNING_ATTN);
        attn = DeltaNetBlocks.atZero(scratch.slot(ScratchSlots.K).require((long) ubatch * width),
                DeltaNetKernels.LIGHTNING_ATTN);
        gate = scratch.slot(ScratchSlots.V).require((long) ubatch * width);
        states = DeltaNetBlocks.GpuStates.bind(ctx, layer, 0, r.stateFloats());
        working = new GpuBuffer[]{qkv.buffer(), attn.buffer(), gate.buffer()};

        GpuBuffer slopeBuf = ctx.backend().allocateBuffer((long) nHead * Float.BYTES, true);
        owned.add(slopeBuf);
        GpuEvent e = ctx.queue().enqueueWriteBuffer(slopeBuf, 0, (long) nHead * Float.BYTES, slopes);
        if (e != null) {
            e.close();
        }

        int local = DeltaNetBlocks.recurrenceLocal(ctx);
        recurrence = ops.prepare(DeltaNetKernels.LIGHTNING_ATTN,
                new long[]{DeltaNetKernels.recurrenceGlobal(nHead, headDim, local)}, new long[]{local},
                DeltaNetBlocks.lightningArgs(1, nHead, headDim, 1.0f, slopeBuf, qkv.buffer(), states.scan,
                        states.rows, attn.buffer()));
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
            throw new IllegalStateException("lightning mixer is not bound or already closed");
        }
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ops.scratch().ubatch()) {
            throw new IllegalArgumentException("lightning step of " + n + " tokens is outside [1, "
                    + ops.scratch().ubatch() + "]");
        }
        DeltaNetBlocks.checkStepViews("lightning mixer", in, out, working);
        states.clearTransient(ops);
        ops.matmul(wQkv, in.buffer(), in.offset(), qkv.buffer(), 0, n);
        qkvDeltas.applyRows(n, in.buffer(), in.offset(), qkv.buffer(), 0);
        ops.unary(ActOp.SILU, ActParams.NONE, qkv.buffer(), 0, qkv.buffer(), 0, 3 * n * width);
        recurrence.setInt(DeltaNetKernels.LIGHTNING_ARG_N_TOKENS, n);
        recurrence.dispatch();
        ops.norm(NormKind.RMS, normW, null, rmsEps, attn.buffer(), 0, attn.buffer(), 0, n, width);
        ops.matmul(wGate, in.buffer(), in.offset(), gate.buffer(), gate.offset(), n);
        gateDeltas.applyRows(n, in.buffer(), in.offset(), gate.buffer(), gate.offset());
        ops.glu(GluOp.SIGMUL, ActParams.NONE, gate.buffer(), gate.offset(), width, attn.buffer(), 0, width,
                attn.buffer(), 0, width, n, width);
        ops.matmul(wOut, attn.buffer(), 0, out.buffer(), out.offset(), n);
        outDeltas.applyRows(n, attn.buffer(), 0, out.buffer(), out.offset());
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
        RuntimeException first = null;
        if (recurrence != null) {
            try {
                recurrence.close();
            } catch (RuntimeException e) {
                first = e;
            }
            recurrence = null;
        }
        first = Closeables.closeAll(owned, first);
        owned.clear();
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuLightningMixer{" + (bound ? nHead + " heads of " + headDim : "unbound") + '}';
    }
}
