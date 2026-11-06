package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuRecurrentState;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmKernels;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * Inkling's two residual-stream short convolutions on the GPU (vLLM {@code _sconv_add_norm}): after the mixer and
 * after the FFN the branch passes through its layer's conv state and is added back into the residual stream together
 * with the convolution's built-in residual, {@code x += conv(branch) + branch}.
 *
 * <p>The state row holds Inkling's four streams ({@link RecurrentSpec#inklingStreamOffset(int)}): the K and V
 * streams the attention mixer updates, and the attention-output and MLP-output streams of this extra. A step with
 * several sequence segments runs one conv launch per segment over that sequence's state row
 * ({@code GpuStepInputs.stateRows}).
 */
public final class GpuInklingSconvExtra implements GpuLayerExtra {

    /** Argument index of {@code LLM_SSM_Conv}'s token count, its input buffer and its row offsets. */
    private static final int ARG_N_TOKENS = SsmKernels.ARG_CONV_N_TOKENS;
    private static final int ARG_X = 10;
    private static final int ARG_X_OFF = 5;
    private static final int ARG_OUT_OFF = 7;

    private GpuOps ops;
    private int ubatch;
    private int nEmbd;
    private int width;
    private GpuBuffer attnW;
    private GpuBuffer ffnW;
    private PreparedDispatch convAttn;
    private PreparedDispatch convFfn;
    private ScratchSlot temp;
    /** {@code ffn_global_scale}: the dense MLP's output multiplier, 1 without one. */
    private float ffnScale = 1.0f;
    /** The step's state-row buffer; a device memory over a paged layout routes it per segment. */
    private GpuBuffer stateRows;
    private boolean routeState;
    private boolean bound;

    /** The convolutions are chronological state updates, exact over any segment layout. */
    @Override
    public String batchRefusal() {
        return null;
    }

    /**
     * @throws IllegalStateException    when the layer is not an Inkling layer or has no conv state
     * @throws MissingTensorException   when a convolution kernel is absent
     * @throws IllegalArgumentException when a kernel's values do not fill {@code n_embd x conv_width}
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound) {
            throw new IllegalStateException("Inkling short-convolution extra of layer " + layer.il()
                    + " is already bound");
        }
        RecurrentSpec r = layer.recurrent();
        if (r == null || r.kind() != MixerKind.INKLING) {
            throw new IllegalStateException("layer " + layer.il() + " lists '"
                    + LayerSpec.Extras.INKLING_SCONV + "' but its recurrent mixer is not Inkling");
        }
        this.ops = ctx.ops();
        this.ubatch = ctx.scratch().ubatch();
        this.nEmbd = ctx.spec().hparams().nEmbd();
        this.width = r.dConv();
        if (layer.residual().streamWidth() != nEmbd) {
            throw new IllegalStateException("layer " + layer.il() + ": the short convolutions expect a stream of "
                    + nEmbd + " floats, the layer streams " + layer.residual().streamWidth());
        }
        String arch = ctx.spec().arch().ggufName();
        GpuWeights ws = ctx.weightSource().weightsFor(layer.weightLayer());
        this.attnW = kernel(ctx, ws, layer, arch, TensorNames.ATTN_SCONV);
        this.ffnScale = ffnScale(ctx, ws, layer, arch);
        GpuMemory memory = ctx.memory();
        this.stateRows = memory.stepInputs().stateRows();
        this.routeState = memory instanceof DeviceModelMemory && memory.pagedSequences() != null;
        this.temp = ctx.scratch().temp("inkling.sconv." + layer.stack() + "." + layer.il(), (long) ubatch * nEmbd);
        if (temp.offset() != 0) {
            throw new IllegalStateException("scratch slot " + temp.name() + " starts at element " + temp.offset());
        }
        this.convAttn = conv(ctx, layer, attnW, r.inklingStreamOffset(RecurrentSpec.INKLING_ATTN));
        this.ffnW = kernel(ctx, ws, layer, arch, TensorNames.FFN_SCONV);
        this.convFfn = conv(ctx, layer, ffnW, r.inklingStreamOffset(RecurrentSpec.INKLING_MLP));
        bound = true;
    }

    /** {@code base.weight} as an F32 device buffer of {@code n_embd * conv_width} values, checked against the state. */
    private GpuBuffer kernel(GpuForwardContext ctx, GpuWeights ws, LayerSpec layer, String arch, String base) {
        GpuWeight w = ws.find(TensorNames.weight(base));
        if (w == null) {
            w = ws.getOrNull(TensorNames.weight(base));
        }
        if (w == null) {
            throw new MissingTensorException(arch, layer.il(), TensorNames.weight(base));
        }
        long n = (long) nEmbd * width;
        if (w.rows() * w.cols() != n) {
            throw new IllegalArgumentException(arch + " layer " + layer.il() + ": " + w.name() + " holds "
                    + w.rows() * w.cols() + " values, expected " + nEmbd + " x " + width);
        }
        if (w.type() == GGMLType.F32 && w.isSingleShard() && w.inverseHadamard() == null && !w.hasScale()) {
            return w.shard(0);
        }
        ScratchSlot decoded = ctx.scratch().temp("inkling.sconv.w." + layer.stack() + "." + layer.il() + "."
                + base, n);
        if (decoded.offset() != 0) {
            throw new IllegalStateException("scratch slot " + decoded.name() + " starts at element "
                    + decoded.offset());
        }
        int[] rows = new int[(int) w.rows()];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = i;
        }
        ops.embed(w, rows, decoded.buffer(), 0);
        return decoded.buffer();
    }

    /** The dense MLP's output multiplier ({@code ffn_global_scale}) as one F32 value, or 1 without the tensor. */
    private static float ffnScale(GpuForwardContext ctx, GpuWeights ws, LayerSpec layer, String arch) {
        GpuWeight w = ws.find(TensorNames.weight(TensorNames.FFN_GLOBAL_SCALE));
        if (w == null) {
            return 1.0f;
        }
        if (w.rows() * w.cols() != 1) {
            throw new IllegalArgumentException(arch + " layer " + layer.il() + ": " + w.name() + " holds "
                    + w.rows() * w.cols() + " values, expected 1");
        }
        float[] v = new float[1];
        GpuEvent e = ctx.queue().enqueueReadBuffer(w.shard(0), w.shardRowStart(0) * w.rowBytes(), Float.BYTES, v);
        if (e != null) {
            e.close();
        }
        return v[0];
    }

    /** The convolution of one stream: its weight buffer, its state channels and the layer's conv state. */
    private PreparedDispatch conv(GpuForwardContext ctx, LayerSpec layer, GpuBuffer w, int stateOff) {
        GpuRecurrentState state = ctx.memory().recurrent(layer.il());
        if (state == null) {
            throw new IllegalStateException("layer " + layer.il() + " has no recurrent state for its short"
                    + " convolutions");
        }
        int stride = -1;
        for (LayerMemoryPlan.StateTensor t : state.tensors()) {
            if (t.name().equals(LayerMemoryPlan.STATE_CONV)) {
                stride = (int) t.floats();
            }
        }
        if (stride < 0) {
            throw new IllegalStateException("layer " + layer.il() + " has no conv state");
        }
        if (stride != layer.recurrent().convStateFloats()) {
            throw new IllegalStateException("layer " + layer.il() + ": conv state holds " + stride + " floats, the"
                    + " layer geometry sizes it at " + layer.recurrent().convStateFloats());
        }
        GpuMemory memory = ctx.memory();
        GpuBuffer stateRows = memory.stepInputs().stateRows();
        int local = SsmKernels.ELEMENTWISE_LOCAL;
        /* The input buffer starts as this extra's own temporary; every call re-binds the caller's rows. */
        Object[] args = {nEmbd, width, 1, 0, nEmbd, 0, nEmbd, 0, stateOff, stride, temp.buffer(), w,
            ctx.placeholders().readOnlyFloats(), state.tensor(LayerMemoryPlan.STATE_CONV), stateRows,
            temp.buffer()};
        if (args.length != SsmKernels.CONV_ARG_COUNT) {
            throw new IllegalStateException(SsmKernels.SSM_CONV + " takes " + SsmKernels.CONV_ARG_COUNT
                    + " arguments, " + args.length + " built");
        }
        long items = (long) ubatch * nEmbd;
        return ops.prepare(SsmKernels.SSM_CONV, new long[]{Math.max(1, (items + local - 1) / local) * local},
                new long[]{local}, args);
    }

    @Override
    public boolean afterMixer(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
        run(convAttn, 1.0f, 0, nTokens, 0, x, branch);
        return true;
    }

    @Override
    public boolean afterFfn(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
        run(convFfn, ffnScale, 0, nTokens, 0, x, branch);
        return true;
    }

    @Override
    public boolean afterMixer(StepLayout s, ScratchSlot x, ScratchSlot branch) {
        segments(s, (seq, start, len) -> run(convAttn, 1.0f, seq, len, start, x, branch));
        return true;
    }

    @Override
    public boolean afterFfn(StepLayout s, ScratchSlot x, ScratchSlot branch) {
        segments(s, (seq, start, len) -> run(convFfn, ffnScale, seq, len, start, x, branch));
        return true;
    }

    /**
     * One segment: {@code x += conv(branch) + branch} over its rows, over its sequence's state row. {@code scale}
     * multiplies the branch before the convolution reads it ({@code ffn_global_scale} on the MLP stream).
     */
    private void run(PreparedDispatch conv, float scale, int seq, int nTokens, int start, ScratchSlot x,
                     ScratchSlot branch) {
        ensureBound();
        if (nTokens < 1) {
            return;
        }
        if (routeState) {
            GpuMixer.routeStateRow(seq, stateRows, ops.queue());
        }
        int in = branch.offset() + start * nEmbd;
        if (scale != 1.0f) {
            ops.scale(branch.buffer(), in, nTokens * nEmbd, scale);
        }
        conv.setBuffer(ARG_X, branch.buffer());
        conv.setInt(ARG_N_TOKENS, nTokens);
        conv.setInt(ARG_X_OFF, in);
        conv.setInt(ARG_OUT_OFF, 0);
        conv.dispatch();
        int dst = x.offset() + start * nEmbd;
        int count = nTokens * nEmbd;
        ops.add(x.buffer(), dst, branch.buffer(), in, x.buffer(), dst, count);
        ops.add(x.buffer(), dst, temp.buffer(), 0, x.buffer(), dst, count);
    }

    private void segments(StepLayout s, SegmentHook hook) {
        ensureBound();
        if (s.contiguousSingle()) {
            hook.run(0, 0, s.nTokens());
            return;
        }
        if (s.nSeg() <= 0) {
            throw new IllegalArgumentException("a step without segments has no rows");
        }
        for (int seg = 0; seg < s.nSeg(); seg++) {
            hook.run(s.seq(seg), s.tokStart(seg), s.len(seg));
        }
    }

    /** One segment of a step. */
    @FunctionalInterface
    private interface SegmentHook {
        void run(int seq, int start, int len);
    }

    private void ensureBound() {
        if (!bound) {
            throw new IllegalStateException("Inkling short-convolution extra used before bind()");
        }
    }

    @Override
    public void close() {
        RuntimeException first = null;
        for (PreparedDispatch d : new PreparedDispatch[]{convAttn, convFfn}) {
            if (d != null) {
                try {
                    d.close();
                } catch (RuntimeException e) {
                    first = e;
                }
            }
        }
        convAttn = null;
        convFfn = null;
        bound = false;
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuInklingSconvExtra{" + (bound ? "n_embd=" + nEmbd : "unbound") + '}';
    }
}
