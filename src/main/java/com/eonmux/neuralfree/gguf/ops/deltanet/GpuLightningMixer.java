package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * GPU lightning attention mixer ({@link MixerKind#LIGHTNING}, MiniMax-01 and Bailing MoE V2.5), following the
 * recurrent branch of models/minimax-01.cpp:277-420 for one sequence. With {@code T} tokens, {@code H} heads of dim
 * {@code D}:
 *
 * <pre>
 * attn_qkv(x) -> Q[T][H][q | k | v]                                               (minimax-01.cpp:293-299)
 * [SiLU on Q/K/V, per-head Q/K RMS norms, NEOX rotation at the step's positions, Q * D^-1/2]
 * LLM_LightningAttn(slopes, layer_scale 1, scan state) -> K[T][H*D]               (minimax-01.cpp:288-405)
 * RMS norm of the H*D row (whole row or group_norm_size groups)                     (minimax-01.cpp:407-410)
 * attn_gate(x) -> V[T][H*D]; LLM_GLU SIGMUL (sigmoid(g) * qkv_norm) in place in K  (minimax-01.cpp:413-419)
 * attn_output -> out                                                               (minimax-01.cpp:421)
 * </pre>
 *
 * The bracketed line is vLLM's {@code BailingMoELinearAttention}: SiLU and the whole-row output norm are the
 * MiniMax graph's too, while the Q/K norms, the rotation and the Q scale come from the {@code ssm.*} metadata
 * (absent for MiniMax). The per-head decay slopes of the layer ({@link RecurrentSpec#slopes()}, already multiplied
 * by the layer scale {@code 1 - il/(n_layer-1) + 1e-5}, minimax-01.cpp:288) are uploaded once at bind into a buffer
 * this mixer owns, so the kernel runs with {@code layer_scale = 1}. The kernel has no offsets, so the Q and K slots
 * must start at element 0. The recurrence is a {@link PreparedDispatch} whose token count is re-set per step. State
 * layout: {@code s[h*D*D + v*D + k]}, row {@code stateRows[0]}; llama.cpp also allocates an unused conv state here,
 * which the memory plan omits. The residual {@code f_residual_scale} is the executor's. Not thread-safe.
 */
public final class GpuLightningMixer implements GpuMixer {

    /** Dense {@code [tokens][heads][headDim]} copy of Q while the per-head norm, rotation and scale run. */
    private static final String TEMP_Q = "lightning.q";
    /** Dense copy of K for the same steps. */
    private static final String TEMP_K = "lightning.k";

    private final List<GpuBuffer> owned = new ArrayList<>();
    private final List<PreparedDispatch> prepared = new ArrayList<>();

    private GpuOps ops;
    private int nHead;
    private int headDim;
    private int width;
    private float rmsEps;
    private float qkEps;
    private float[] slopes;
    private int nNormGroups;
    private boolean linearSilu;
    private boolean qkNorm;
    private float qScale;
    private int local;
    private int reduceLocal;

    private GpuWeight wQkv;
    private GpuWeight wGate;
    private GpuWeight wOut;
    private GpuWeight normW;
    private GpuWeight qNormW;
    private GpuWeight kNormW;
    /** LoRA deltas of {@code attn_qkv}, {@code attn_gate} and {@code attn_output} ({@code build_lora_mm}). */
    private GpuProjectionDeltas.Site qkvDeltas;
    private GpuProjectionDeltas.Site gateDeltas;
    private GpuProjectionDeltas.Site outDeltas;

    private ScratchSlot qkv;
    private ScratchSlot attn;
    private ScratchSlot gate;
    private ScratchSlot qTmp;
    private ScratchSlot kTmp;
    private GpuBuffer ropePos;
    private GpuBuffer segPos;
    private GpuStepInputs steps;
    private boolean denseQk;
    private long ropePerToken;
    private PreparedDispatch ropeQ;
    private PreparedDispatch ropeK;
    private PreparedDispatch ropeQSeg;
    private PreparedDispatch ropeKSeg;
    private PreparedDispatch groupNorm;
    private DeltaNetBlocks.GpuStates states;
    private GpuBuffer[] working;
    private PreparedDispatch recurrence;

    private boolean bound;
    /** The layer's stream width: the rows the mixer reads and writes. */
    private int rowWidth;
    /** Whether the memory holds per-sequence state rows a batched step routes through. */
    private boolean routeState;
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
        qkEps = ctx.spec().hparams().normRmsEps();
        nNormGroups = r.nNormGroups();
        linearSilu = r.linearSilu();
        qkNorm = r.qkNorm();
        qScale = r.qScale();
        local = ops.local();
        RopeSpec rope = r.rope();
        int nEmbd = DeltaNetBlocks.mixerWidth(layer);
        this.rowWidth = nEmbd;

        GpuWeights ws = ctx.weightSource().weightsFor(layer.weightLayer());
        wQkv = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_QKV, nEmbd, 3L * width);
        wGate = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_GATE, nEmbd, width);
        wOut = DeltaNetBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_OUT, width, nEmbd);
        qkvDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.ATTN_QKV), wQkv);
        gateDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.ATTN_GATE), wGate);
        outDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.ATTN_OUT), wOut);
        normW = DeltaNetBlocks.gpuNormWeight(ctx, ws.layer(layer, TensorNames.weight(TensorNames.ATTN_NORM_2)), width,
                owned);
        if (qkNorm) {
            qNormW = DeltaNetBlocks.gpuNormWeight(ctx, ws.layer(layer, TensorNames.weight(TensorNames.ATTN_Q_NORM)),
                    headDim, owned);
            kNormW = DeltaNetBlocks.gpuNormWeight(ctx, ws.layer(layer, TensorNames.weight(TensorNames.ATTN_K_NORM)),
                    headDim, owned);
        }

        qkv = DeltaNetBlocks.atZero(scratch.slot(ScratchSlots.Q).require(3L * ubatch * width),
                DeltaNetKernels.LIGHTNING_ATTN);
        attn = DeltaNetBlocks.atZero(scratch.slot(ScratchSlots.K).require((long) ubatch * width),
                DeltaNetKernels.LIGHTNING_ATTN);
        gate = scratch.slot(ScratchSlots.V).require((long) ubatch * width);
        states = DeltaNetBlocks.GpuStates.bind(ctx, layer, 0, r.stateFloats());
        routeState = ctx.hasMemory() && ctx.memory() instanceof com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory
                && ctx.memory().pagedSequences() != null;
        working = new GpuBuffer[]{qkv.buffer(), attn.buffer(), gate.buffer()};

        denseQk = qkNorm || rope != null || qScale != 0f;
        if (denseQk) {
            qTmp = scratch.temp(TEMP_Q, (long) ubatch * width);
            kTmp = scratch.temp(TEMP_K, (long) ubatch * width);
        }
        if (rope != null) {
            if (ctx.hasMemory()) {
                steps = ctx.memory().stepInputs();
                ropePos = steps.ropePos();
                segPos = steps.segmentPositions();
            } else {
                ropePos = DeltaNetBlocks.atZero(scratch.slot(ScratchSlots.POS).require(4L * ubatch),
                        RopeKernels.ROPE_EXT).buffer();
            }
            GpuBuffer ph = ops.placeholders().readOnlyFloats();
            ropePerToken = RopeKernels.ropeExtWorkItems(rope.type(), 1, nHead, rope.nDims());
            ropeQ = prepareRope(rope, qTmp.buffer(), ropePos, ph, ropePerToken);
            ropeK = prepareRope(rope, kTmp.buffer(), ropePos, ph, ropePerToken);
            if (segPos != null) {
                ropeQSeg = prepareRope(rope, qTmp.buffer(), segPos, ph, ropePerToken);
                ropeKSeg = prepareRope(rope, kTmp.buffer(), segPos, ph, ropePerToken);
            }
        }
        if (nNormGroups > 1) {
            reduceLocal = Math.min(local, LlmKernelConstants.LLM_DIMX);
            NormKernels.checkReduceLocal(reduceLocal);
            Object[] args = NormKernels.groupRmsArgs(1, width, nNormGroups, rmsEps, attn.buffer(), 0, width,
                    normW.shard(0), attn.buffer());
            args[NormKernels.ARG_GROUP_RMS_N_TOKENS] = 1;
            groupNorm = prepare(NormKernels.GROUP_RMS,
                    NormKernels.reduceGlobal(NormKernels.groupRmsGroups(1, nNormGroups), reduceLocal),
                    reduceLocal, args);
        }

        GpuBuffer slopeBuf = ctx.backend().allocateBuffer((long) nHead * Float.BYTES, true);
        owned.add(slopeBuf);
        GpuEvent e = ctx.queue().enqueueWriteBuffer(slopeBuf, 0, (long) nHead * Float.BYTES, slopes);
        if (e != null) {
            e.close();
        }

        int recLocal = DeltaNetBlocks.recurrenceLocal(ctx);
        recurrence = ops.prepare(DeltaNetKernels.LIGHTNING_ATTN,
                new long[]{DeltaNetKernels.recurrenceGlobal(nHead, headDim, recLocal)}, new long[]{recLocal},
                DeltaNetBlocks.lightningArgs(1, nHead, headDim, 1.0f, slopeBuf, qkv.buffer(), states.scan,
                        states.rows, attn.buffer()));
    }

    /** A lightning-attention rotary dispatch over one dense {@code [tokens][heads][headDim]} buffer. */
    private PreparedDispatch prepareRope(RopeSpec rope, GpuBuffer x, GpuBuffer pos, GpuBuffer placeholder,
                                         long perToken) {
        Object[] args = RopeKernels.ropeExtArgs(rope, 1, nHead, headDim, width, rope.nOffs(), 1, 0, false, pos,
                placeholder, x);
        return prepare(RopeKernels.ROPE_EXT, new long[]{roundUp(perToken)}, local, args);
    }

    private PreparedDispatch prepare(String kernel, long[] global, int localSize, Object[] args) {
        PreparedDispatch d = ops.prepare(kernel, global, new long[]{localSize}, args);
        prepared.add(d);
        return d;
    }

    private long roundUp(long items) {
        return Math.max(1, (items + local - 1) / local) * local;
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        run(1, in.buffer(), in.offset(), out.buffer(), out.offset(), 0, 1, false);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        run(nTokens, in.buffer(), in.offset(), out.buffer(), out.offset(), 0, nTokens, false);
    }

    /** Why this mixer runs a batched step: every sequence's rows dispatch as its own contiguous step. */
    @Override
    public String batchRefusal() {
        return null;
    }

    /**
     * One batched step: every segment of {@code s} dispatches its rows as the contiguous step, reading and
     * writing its sequence's recurrent state (the attention re-points the kernel's state-row routing per
     * segment) and reading its rows at their place in the step's input and output. A segment's RoPE positions are
     * staged into the step inputs' segment positions, whose stride is the segment length.
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
        DeltaNetBlocks.checkStepViews("lightning mixer", in, out, working);
        for (int seg = 0; seg < s.nSeg(); seg++) {
            int start = s.tokStart(seg);
            int len = s.len(seg);
            if (ropePos != null) {
                if (segPos == null) {
                    throw new UnsupportedOperationException("lightning mixer: a batched step needs the memory's"
                            + " segment positions to rotate its segments");
                }
                steps.uploadSegmentPositions(s.pos0(seg), len);
            }
            run(len, in.buffer(), in.offset() + start * rowWidth, out.buffer(),
                    out.offset() + start * rowWidth, s.seq(seg), len, ropePos != null);
        }
    }

    private void run(int n, GpuBuffer inBuf, int inOff, GpuBuffer outBuf, int outOff, int stateRow, int posStride,
                     boolean segmented) {
        if (!bound || closed) {
            throw new IllegalStateException("lightning mixer is not bound or already closed");
        }
        if (n < 1 || n > ops.scratch().ubatch()) {
            throw new IllegalArgumentException("lightning step of " + n + " tokens is outside [1, "
                    + ops.scratch().ubatch() + "]");
        }
        if (routeState) {
            GpuMixer.routeStateRow(stateRow, states.rows, ops.queue());
        }
        states.clearTransient(ops);
        // The input and gate projections of a single-token step both read this row, so its Q8_1 blocks are
        // quantized once.
        GpuOps.SharedQ8 share = n == 1 && GpuOps.shareableQ8Activations(inOff, rowWidth)
                ? ops.shareQ8Activations(inBuf, inOff, rowWidth) : null;
        try {
            ops.matmul(wQkv, inBuf, inOff, qkv.buffer(), 0, n);
            qkvDeltas.applyRows(n, inBuf, inOff, qkv.buffer(), 0);
            if (linearSilu) {
                ops.unary(ActOp.SILU, ActParams.NONE, qkv.buffer(), 0, qkv.buffer(), 0, 3 * n * width);
            }
            if (denseQk) {
                int rows = n * nHead;
                ops.copy2d(qkv.buffer(), 0, 3 * headDim, qTmp.buffer(), 0, headDim, rows, headDim);
                ops.copy2d(qkv.buffer(), headDim, 3 * headDim, kTmp.buffer(), 0, headDim, rows, headDim);
                if (qkNorm) {
                    ops.norm(NormKind.RMS, qNormW, null, qkEps, qTmp.buffer(), 0, qTmp.buffer(), 0, rows, headDim);
                    ops.norm(NormKind.RMS, kNormW, null, qkEps, kTmp.buffer(), 0, kTmp.buffer(), 0, rows, headDim);
                }
                if (ropePos != null) {
                    PreparedDispatch qRope = segmented ? ropeQSeg : ropeQ;
                    PreparedDispatch kRope = segmented ? ropeKSeg : ropeK;
                    qRope.setInt(RopeKernels.ARG_N_TOKENS, n);
                    qRope.setInt(RopeKernels.ARG_POS_STRIDE, posStride);
                    qRope.setGlobal(roundUp(n * ropePerToken));
                    qRope.dispatch();
                    kRope.setInt(RopeKernels.ARG_N_TOKENS, n);
                    kRope.setInt(RopeKernels.ARG_POS_STRIDE, posStride);
                    kRope.setGlobal(roundUp(n * ropePerToken));
                    kRope.dispatch();
                }
                if (qScale != 0f) {
                    ops.scale(qTmp.buffer(), 0, n * width, qScale);
                }
                ops.copy2d(qTmp.buffer(), 0, headDim, qkv.buffer(), 0, 3 * headDim, rows, headDim);
                ops.copy2d(kTmp.buffer(), 0, headDim, qkv.buffer(), headDim, 3 * headDim, rows, headDim);
            }
            recurrence.setInt(DeltaNetKernels.LIGHTNING_ARG_N_TOKENS, n);
            recurrence.dispatch();
            if (groupNorm != null) {
                groupNorm.setInt(NormKernels.ARG_GROUP_RMS_N_TOKENS, n);
                groupNorm.setGlobal(NormKernels.reduceGlobal(NormKernels.groupRmsGroups(n, nNormGroups), reduceLocal));
                groupNorm.dispatch();
            } else {
                ops.norm(NormKind.RMS, normW, null, rmsEps, attn.buffer(), 0, attn.buffer(), 0, n, width);
            }
            ops.matmul(wGate, inBuf, inOff, gate.buffer(), gate.offset(), n);
            gateDeltas.applyRows(n, inBuf, inOff, gate.buffer(), gate.offset());
            ops.glu(GluOp.SIGMUL, ActParams.NONE, gate.buffer(), gate.offset(), width, attn.buffer(), 0, width,
                    attn.buffer(), 0, width, n, width);
            ops.matmul(wOut, attn.buffer(), 0, outBuf, outOff, n);
            outDeltas.applyRows(n, attn.buffer(), 0, outBuf, outOff);
        } finally {
            if (share != null) {
                share.close();
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
        RuntimeException first = null;
        if (recurrence != null) {
            try {
                recurrence.close();
            } catch (RuntimeException e) {
                first = e;
            }
            recurrence = null;
        }
        for (PreparedDispatch d : prepared) {
            try {
                d.close();
            } catch (RuntimeException e) {
                if (first == null) {
                    first = e;
                }
            }
        }
        prepared.clear();
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
