package com.eonmux.neuralfree.gguf.ops.act;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSource;
import com.eonmux.neuralfree.gguf.ops.act.FfnBlocks.DenseLayout;
import com.eonmux.neuralfree.gguf.ops.gemma3n.Gemma3nKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The dense feed-forward block ({@code FfnKind.DENSE}) on the GPU: llama.cpp {@code build_ffn}
 * (src/llama-graph.cpp:1922-2139) in all four forms of {@link DenseLayout.Form}.
 *
 * <p>Per step, over {@code n} token rows:
 * <ol>
 *   <li>Projections. Each projection {@code y = W x} is followed by the {@link GpuProjectionDeltas} registered on
 *       it (LoRA), its bias ({@code <base>.bias}, {@value #BATCH_BIAS_ADD}, unless {@link FfnSpec#biases()} is
 *       false) and then its output scale {@code w_s}
 *       ({@link GpuWeight#scale()}, or the resident {@code <base>.scale} tensor through {@code LLM_MulRowScale}),
 *       which is llama.cpp's order {@code (W x + delta + b) * s} (llama-graph.cpp:1959-1995, 2116-2136). The PRISM
 *       input rotation and row shards are {@link GpuOps#matmul}'s.</li>
 *   <li>Gemma3n activation sparsity: with {@code activationSparsityStdMul > 0} the gate rows pass through
 *       {@code LLM_GaussianTopK} before the GLU (src/models/gemma3n.cpp:212-219).</li>
 *   <li>Activation: {@code LLM_GLU} of gate and up (separate, or the two halves of a fused {@code ffn_up}), or
 *       {@code LLM_Unary} on the gate (sequential) or up (ungated) rows, followed for GELU by the division by
 *       {@code ffn.act.scales} when the layer has them (mpt).</li>
 *   <li>bitnet's {@code ffn_sub_norm} (RMS, {@code LLM_RowNorm}) on the activated rows
 *       (src/models/bitnet.cpp:126-138).</li>
 *   <li>The down projection into {@code out}.</li>
 * </ol>
 * Working rows live at element 0 of the {@code FFN_GATE_UP} slot and the {@link DenseLayout} temporaries, so the
 * offset-free bias and row-scale kernels address them directly; a biased or tensor-scaled down projection bound for
 * an output view at an offset is finished in {@link DenseLayout#TEMP_DOWN} and copied.</p>
 *
 * <p>Every element-wise and row kernel is a {@link PreparedDispatch} built at bind time for {@code ubatch} rows; a
 * step re-sets only its row count and global size. The {@code ffn.act.scales} divisor is read back once at bind time
 * and kept as reciprocals in a buffer this block owns. Not thread-safe.</p>
 */
public final class GpuDenseFfn implements GpuFfnBlock {

    /** {@code data[s*dim + d] += bias[d]} for {@code s < seq_len} ({@code kernels/llm/eltwise.cl}). */
    public static final String BATCH_BIAS_ADD = GpuOps.BATCH_BIAS_ADD;
    public static final int ARG_BIAS_DIM = 0;
    public static final int ARG_BIAS_SEQ_LEN = 1;
    public static final int ARG_BIAS_BIAS = 2;
    public static final int ARG_BIAS_DATA = 3;
    public static final int BIAS_ARG_COUNT = 4;

    /** Kernels this block launches besides the projection families of {@link GpuOps}, for prewarming. */
    public static final List<String> KERNELS = Collections.unmodifiableList(Arrays.asList(
            ActKernels.GLU, ActKernels.UNARY, BATCH_BIAS_ADD, NormKernels.ROW_NORM, NormKernels.MUL_ROW_SCALE,
            Gemma3nKernels.GAUSSIAN_TOPK));

    private final List<PreparedDispatch> prepared = new ArrayList<>();
    private GpuForwardContext ctx;
    private GpuOps ops;
    private DenseLayout layout;
    private String where;
    private int ubatch;
    private int local;
    private boolean biases;
    private ScratchSlot gateUp;
    private ScratchSlot upTemp;
    private ScratchSlot actTemp;
    private ScratchSlot downTemp;
    private Projection gate;
    private Projection up;
    private Projection down;
    private PreparedDispatch topK;
    private int topKLocal;
    private PreparedDispatch glu;
    private PreparedDispatch unary;
    private PreparedDispatch actScale;
    private PreparedDispatch subNorm;
    private GpuBuffer actScaleReciprocals;
    private boolean closed;

    /**
     * Binds the block: resolves the layer's tensors, validates them against its spec, reserves the working buffers
     * and prepares every element-wise dispatch.
     *
     * @throws IllegalStateException when already bound or closed, or the tensors contradict the spec
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext context, LayerSpec layer) {
        if (closed || ctx != null) {
            throw new IllegalStateException("dense FFN block is " + (closed ? "closed" : "already bound"));
        }
        Objects.requireNonNull(context, "context");
        FfnSpec spec = FfnBlocks.denseSpec(layer);
        ctx = context;
        try {
            bindLayer(layer, spec);
        } catch (RuntimeException e) {
            release(e);
            unbind();
            throw e;
        }
    }

    /** Drops every reference a failed bind may have left, so the block reads as unbound. */
    private void unbind() {
        ctx = null;
        ops = null;
        layout = null;
        where = null;
        gateUp = null;
        upTemp = null;
        actTemp = null;
        downTemp = null;
        gate = null;
        up = null;
        down = null;
        topK = null;
        glu = null;
        unary = null;
        actScale = null;
        subNorm = null;
    }

    private void bindLayer(LayerSpec layer, FfnSpec spec) {
        ops = ctx.ops();
        where = ctx.spec().arch().ggufName() + " layer " + layer.stack() + "." + layer.il();
        ubatch = ctx.scratch().ubatch();
        local = ops.local();
        biases = spec.biases();
        GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
        GpuWeight upW = weights.layer(layer, spec.upWeight());
        GpuWeight gateW = weights.layerOrNull(layer, spec.gateWeight());
        GpuWeight downW = weights.layer(layer, spec.downWeight());
        layout = DenseLayout.of(layer, shape(upW), gateW != null ? shape(gateW) : null, shape(downW));

        gateUp = dense(ctx.scratch().slot(ScratchSlots.FFN_GATE_UP), layout.gateUpWidth());
        if (layout.upTempWidth() > 0) {
            upTemp = dense(ctx.scratch().temp(DenseLayout.TEMP_UP, rows(layout.upTempWidth())),
                    layout.upTempWidth());
        }
        if (layout.form() == DenseLayout.Form.FUSED_GLU) {
            actTemp = dense(ctx.scratch().temp(DenseLayout.TEMP_ACT, rows(layout.nFf())), layout.nFf());
        }

        if (layout.hasGate()) {
            gate = projection(weights, layer, gateW, layout.gateBase(), gateUp.buffer());
        }
        up = projection(weights, layer, upW, layout.upBase(),
                upTemp != null ? upTemp.buffer() : gateUp.buffer());
        down = projectionDown(weights, layer, downW);

        if (layout.sparsity() > 0f) {
            topKLocal = Gemma3nKernels.rowLocal(local);
            topK = prepare(Gemma3nKernels.GAUSSIAN_TOPK, Gemma3nKernels.rowGlobal(ubatch, topKLocal), topKLocal,
                    Gemma3nKernels.gaussianTopKArgs(ubatch, layout.nFf(), layout.gateStride(), 0, layout.sparsity(),
                            gateUp.buffer()));
        }
        if (layout.usesGlu()) {
            glu = prepare(ActKernels.GLU, global(ubatch, layout.nFf()), local, gluArgs());
        } else {
            unary = prepare(ActKernels.UNARY, global(ubatch, layout.nFf()), local,
                    ActKernels.unaryArgs(layout.act(), layout.params(), ActKernels.elementCount(ubatch, layout.nFf()),
                            gateUp.buffer(), 0, gateUp.buffer(), 0));
            bindActScales(weights, layer);
        }
        if (layout.subNorm()) {
            bindSubNorm(weights, layer);
        }
    }

    private Object[] gluArgs() {
        int nFf = layout.nFf();
        GpuBuffer g = gateUp.buffer();
        if (layout.form() == DenseLayout.Form.FUSED_GLU) {
            return ActKernels.gluArgs(layout.glu(), layout.params(), ubatch, nFf,
                    g, 0, 2 * nFf, g, nFf, 2 * nFf, actTemp.buffer(), 0, nFf);
        }
        return ActKernels.gluArgs(layout.glu(), layout.params(), ubatch, nFf,
                g, 0, nFf, upTemp.buffer(), 0, nFf, g, 0, nFf);
    }

    /**
     * {@code ffn.act.scales}: llama.cpp divides the GELU output by them per column (llama-graph.cpp:2032-2057). The
     * values are read back once and their reciprocals multiplied in with {@code LLM_MulRowScale}.
     */
    private void bindActScales(GpuWeights weights, LayerSpec layer) {
        if (!layout.actScalesApply()) {
            return;
        }
        GpuWeight s = weights.layerOrNull(layer, DenseLayout.ACT_SCALES);
        if (s == null) {
            return;
        }
        int nFf = layout.nFf();
        checkVector(s, nFf, "activation scales");
        float[] values = readBack(layer, s, nFf);
        float[] reciprocals = new float[nFf];
        for (int i = 0; i < nFf; i++) {
            float v = values[i];
            float r = 1.0f / v;
            if (!Float.isFinite(v) || !Float.isFinite(r)) {
                throw new IllegalStateException(where + ": " + s.name() + "[" + i + "] = " + v
                        + " is not a finite divisor with a finite reciprocal");
            }
            reciprocals[i] = r;
        }
        actScaleReciprocals = ctx.backend().allocateBuffer((long) nFf * Float.BYTES, true);
        GpuEvent e = ctx.queue().enqueueWriteBuffer(actScaleReciprocals, 0, (long) nFf * Float.BYTES, reciprocals);
        closeEvent(e);
        actScale = prepare(NormKernels.MUL_ROW_SCALE, global(ubatch, nFf), local,
                NormKernels.mulRowScaleArgs(ubatch, nFf, actScaleReciprocals, nFf, gateUp.buffer()));
    }

    /** Reads a resident F32 vector to the host once the layer's weights are resident on the compute queue. */
    private float[] readBack(LayerSpec layer, GpuWeight w, int n) {
        LayerWeightSource source = ctx.weightSource();
        source.prefetch(layer.weightLayer());
        source.await(layer.weightLayer());
        float[] host = new float[n];
        ComputeQueue q = ctx.queue();
        closeEvent(q.enqueueReadBuffer(w.shard(0), 0, (long) n * Float.BYTES, host));
        q.finish();
        return host;
    }

    private void bindSubNorm(GpuWeights weights, LayerSpec layer) {
        int nFf = layout.nFf();
        GpuWeight w = weights.layer(layer, DenseLayout.SUB_NORM);
        checkVector(w, nFf, "sub-norm weight");
        NormKernels.checkReduceLocal(local);
        GpuBuffer act = activated();
        subNorm = prepare(NormKernels.ROW_NORM, NormKernels.reduceGlobal(NormKernels.rowNormGroups(ubatch), local),
                local, NormKernels.rowNormArgs(NormKind.RMS, layout.subNormEps(), ubatch, nFf, nFf, act, 0, act, 0,
                        w.shard(0), null, 1, ctx.placeholders().readOnlyFloats()));
    }

    private Projection projection(GpuWeights weights, LayerSpec layer, GpuWeight w, String base, GpuBuffer target) {
        int rows = (int) w.rows();
        PreparedDispatch bias = biasDispatch(weights, layer, base, rows, target);
        PreparedDispatch tensorScale = w.hasScale() ? null : tensorScaleDispatch(weights, layer, base, rows, target);
        String refusal = w.hadamard() != null ? w.name() + " rotates its input (PRISM Hadamard) inside the matmul; a"
                + " LoRA delta would need the rotated rows" : null;
        GpuProjectionDeltas.Site deltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(base), refusal);
        return new Projection(w, rows, bias, tensorScale, deltas);
    }

    /** The down projection; a temporary serves output views at an offset when the bias or row scale needs one. */
    private Projection projectionDown(GpuWeights weights, LayerSpec layer, GpuWeight w) {
        String base = layout.downBase();
        boolean offsetFree = bias(weights, layer, base) != null
                || (!w.hasScale() && weights.layerOrNull(layer, TensorNames.scale(base)) != null);
        if (offsetFree) {
            downTemp = dense(ctx.scratch().temp(DenseLayout.TEMP_DOWN, rows(layout.nOut())), layout.nOut());
        }
        return projection(weights, layer, w, base, offsetFree ? downTemp.buffer() : null);
    }

    /**
     * The projection's {@code <base>.bias}; null when the file has none or the graph ignores it
     * ({@link FfnSpec#biases()} false).
     */
    private GpuWeight bias(GpuWeights weights, LayerSpec layer, String base) {
        return biases ? weights.layerOrNull(layer, TensorNames.bias(base)) : null;
    }

    private PreparedDispatch biasDispatch(GpuWeights weights, LayerSpec layer, String base, int rows,
                                          GpuBuffer target) {
        GpuWeight b = bias(weights, layer, base);
        if (b == null) {
            return null;
        }
        checkVector(b, rows, "bias");
        return prepare(BATCH_BIAS_ADD, global(1, rows), local, biasAddArgs(rows, ubatch, b.shard(0), target));
    }

    private PreparedDispatch tensorScaleDispatch(GpuWeights weights, LayerSpec layer, String base, int rows,
                                                 GpuBuffer target) {
        GpuWeight s = weights.layerOrNull(layer, TensorNames.scale(base));
        if (s == null) {
            return null;
        }
        checkVector(s, 1, "output scale");
        return prepare(NormKernels.MUL_ROW_SCALE, global(ubatch, rows), local,
                NormKernels.mulRowScaleArgs(ubatch, rows, s.shard(0), 1, target));
    }

    /**
     * Arguments of {@value #BATCH_BIAS_ADD} in signature order: {@code data[s*dim + d] += bias[d]} for
     * {@code s < seqLen}, {@code d < dim}; the rows start at element 0 of {@code data}.
     *
     * @throws IllegalArgumentException for a non-positive width, a negative row count or a view outside its buffer
     */
    public static Object[] biasAddArgs(int dim, int seqLen, GpuBuffer bias, GpuBuffer data) {
        if (dim < 1 || seqLen < 0) {
            throw new IllegalArgumentException("bias add needs dim >= 1 and seq_len >= 0, got " + dim + " / " + seqLen);
        }
        ActKernels.elementCount(seqLen, dim);
        ActKernels.checkFloatView("bias", bias, 0, dim, 1, dim);
        ActKernels.checkFloatView("biased rows", data, 0, dim, seqLen, dim);
        return new Object[]{dim, seqLen, bias, data};
    }

    // ------------------------------------------------------------------ step

    /**
     * Runs the FFN over {@code nTokens} rows of {@code in} (the up projection's input width) into {@code out} (the
     * down projection's output width).
     *
     * @throws IllegalStateException    when not bound or closed
     * @throws IllegalArgumentException for a row count outside {@code [1, ubatch]}, a slot too small for the rows,
     *                                  or an output slot that shares a buffer with the FFN's working rows
     */
    @Override
    public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
        if (closed || ctx == null) {
            throw new IllegalStateException("dense FFN block is " + (closed ? "closed" : "not bound"));
        }
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException(where + ": FFN row count " + nTokens + " outside [1, " + ubatch + "]");
        }
        in.require((long) nTokens * layout.nIn());
        out.require((long) nTokens * layout.nOut());
        GpuBuffer act = activated();
        if (out.buffer() == act || out.buffer() == gateUp.buffer()
                || (upTemp != null && out.buffer() == upTemp.buffer())) {
            throw new IllegalArgumentException(where + ": FFN output slot " + out + " shares a buffer with the FFN"
                    + " working rows");
        }
        GpuBuffer x = in.buffer();
        int xOff = in.offset();
        GpuBuffer g = gateUp.buffer();
        switch (layout.form()) {
            case PARALLEL_GLU:
                gate.run(ops, nTokens, x, xOff, g, 0);
                sparsity(nTokens);
                up.run(ops, nTokens, x, xOff, upTemp.buffer(), 0);
                runGlu(nTokens);
                break;
            case FUSED_GLU:
                up.run(ops, nTokens, x, xOff, g, 0);
                sparsity(nTokens);
                runGlu(nTokens);
                break;
            case SEQUENTIAL:
                up.run(ops, nTokens, x, xOff, upTemp.buffer(), 0);
                gate.run(ops, nTokens, upTemp.buffer(), 0, g, 0);
                runUnary(nTokens);
                break;
            default:
                up.run(ops, nTokens, x, xOff, g, 0);
                runUnary(nTokens);
                break;
        }
        if (subNorm != null) {
            subNorm.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, nTokens);
            subNorm.setGlobal(NormKernels.reduceGlobal(NormKernels.rowNormGroups(nTokens), local));
            subNorm.dispatch();
        }
        int nOut = layout.nOut();
        if (downTemp != null && out.offset() != 0) {
            down.run(ops, nTokens, act, 0, downTemp.buffer(), 0);
            ops.copy(downTemp.buffer(), 0, out.buffer(), out.offset(), nTokens * nOut);
        } else {
            down.run(ops, nTokens, act, 0, out.buffer(), out.offset());
        }
    }

    private void sparsity(int nTokens) {
        if (topK != null) {
            topK.setInt(Gemma3nKernels.ARG_N_TOKENS, nTokens);
            topK.setGlobal(Gemma3nKernels.rowGlobal(nTokens, topKLocal));
            topK.dispatch();
        }
    }

    private void runGlu(int nTokens) {
        glu.setInt(ActKernels.ARG_GLU_N_ROWS, nTokens);
        glu.setGlobal(global(nTokens, layout.nFf()));
        glu.dispatch();
    }

    private void runUnary(int nTokens) {
        int n = nTokens * layout.nFf();
        unary.setInt(ActKernels.ARG_UNARY_N, n);
        unary.setGlobal(global(nTokens, layout.nFf()));
        unary.dispatch();
        if (actScale != null) {
            actScale.setInt(NormKernels.ARG_ROW_SCALE_N_ROWS, nTokens);
            actScale.setGlobal(global(nTokens, layout.nFf()));
            actScale.dispatch();
        }
    }

    /** Buffer holding the activated {@code [n][n_ff]} rows from element 0. */
    private GpuBuffer activated() {
        return actTemp != null ? actTemp.buffer() : gateUp.buffer();
    }

    // ------------------------------------------------------------------ helpers

    /**
     * One projection with its deltas, optional bias and output scale, in llama.cpp's order. The weight's own output
     * scale is folded into the matmul unless a bias or a delta must come before it.
     */
    private static final class Projection {
        private final GpuWeight weight;
        private final GpuWeight unscaled;
        private final int rows;
        private final int cols;
        private final PreparedDispatch bias;
        private final PreparedDispatch tensorScale;
        private final GpuProjectionDeltas.Site deltas;

        Projection(GpuWeight w, int rows, PreparedDispatch bias, PreparedDispatch tensorScale,
                   GpuProjectionDeltas.Site deltas) {
            this.weight = w;
            this.unscaled = w.hasScale() ? w.withScale(1.0f) : w;
            this.rows = rows;
            this.cols = (int) w.cols();
            this.bias = bias;
            this.tensorScale = tensorScale;
            this.deltas = Objects.requireNonNull(deltas, "deltas");
        }

        /**
         * {@code target[targetOff + t*rows + r] = (W x_t + delta(x_t) + b)[r] * s}, the input rows dense at
         * {@code inOff}. A projection with a bias or a tensor scale writes at {@code targetOff == 0}.
         */
        void run(GpuOps ops, int n, GpuBuffer in, int inOff, GpuBuffer target, int targetOff) {
            if ((bias != null || tensorScale != null) && targetOff != 0) {
                throw new IllegalStateException(weight.name() + ": biased or row-scaled projection needs its rows at"
                        + " element 0, got offset " + targetOff);
            }
            boolean hasDeltas = !deltas.isEmpty();
            boolean postScale = weight.hasScale() && (bias != null || hasDeltas);
            ops.matmul(postScale ? unscaled : weight, in, inOff, target, targetOff, n);
            if (hasDeltas) {
                deltas.apply(n, in, inOff, cols, target, targetOff, rows);
            }
            if (bias != null) {
                bias.setInt(ARG_BIAS_SEQ_LEN, n);
                bias.setBuffer(ARG_BIAS_DATA, target);
                bias.dispatch();
            }
            if (postScale) {
                ops.scale(target, targetOff, n * rows, weight.scale());
            }
            if (tensorScale != null) {
                tensorScale.setInt(NormKernels.ARG_ROW_SCALE_N_ROWS, n);
                tensorScale.setBuffer(NormKernels.ARG_ROW_SCALE_X, target);
                tensorScale.setGlobal(ActKernels.global1d((long) n * rows, ops.local()));
                tensorScale.dispatch();
            }
        }
    }

    private PreparedDispatch prepare(String kernel, long[] global, int localSize, Object[] args) {
        PreparedDispatch d = ops.prepare(kernel, global, new long[]{localSize}, args);
        prepared.add(d);
        return d;
    }

    private long[] global(int nRows, int width) {
        return ActKernels.global1d((long) nRows * width, local);
    }

    private long rows(int width) {
        return (long) ubatch * width;
    }

    /** Checks that a scratch view starts at element 0 of its buffer and holds {@code ubatch} rows of a width. */
    private ScratchSlot dense(ScratchSlot s, int width) {
        if (s.offset() != 0) {
            throw new IllegalStateException(where + ": FFN working slot " + s + " must start at element 0");
        }
        return s.require(rows(width));
    }

    private void checkVector(GpuWeight v, int n, String what) {
        if (v.type() != GGMLType.F32 || !v.isSingleShard() || v.cols() * v.rows() != n) {
            throw new IllegalStateException(where + ": " + what + " " + v.name() + " must be a single-buffer F32"
                    + " vector of " + n + " values, got " + v);
        }
    }

    private static long[] shape(GpuWeight w) {
        return new long[]{w.cols(), w.rows()};
    }

    private static void closeEvent(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    /**
     * Closes the prepared dispatches and, after the queue drains, the activation-scale buffer; returns the first
     * failure with later ones suppressed, chained onto {@code first} when given.
     */
    private RuntimeException release(RuntimeException first) {
        RuntimeException failure = Closeables.closeAll(prepared, first);
        prepared.clear();
        if (actScaleReciprocals != null) {
            try {
                ctx.queue().finish();
                actScaleReciprocals.close();
            } catch (RuntimeException e) {
                failure = Closeables.chain(failure, e);
            }
            actScaleReciprocals = null;
        }
        return failure;
    }

    /** Closes the prepared dispatches and, after the queue drains, the activation-scale buffer. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = release(null);
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "GpuDenseFfn{" + (layout != null ? where + ", " + layout : "unbound") + (closed ? ", closed" : "") + '}';
    }
}
