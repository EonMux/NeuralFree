package com.eonmux.neuralfree.gguf.gpu.head;

import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.ops.flow.FlowKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.utils.Closeables;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Output stage of the GPU forward path ({@link HeadSpec}): from the final residual stream {@link ScratchSlots#X} to
 * the hidden rows the LM or classifier heads read ({@code t_embd}) and to logits.
 *
 * <ol>
 *   <li>{@link #hidden}: the requested output rows are selected from {@code X} (one copy for a contiguous run,
 *       {@value FlowKernels#GATHER_ROWS} otherwise; llama.cpp {@code inp_out_ids}); every bound
 *       {@link GpuHeadPreprocessor} rewrites them to {@code n_embd} (hyper-connection head, ...); then the output
 *       norm with its bias, of the trunk's norm kind and epsilon, over {@link HeadSpec#outputNormGroups()} slices of
 *       the row; then the {@link HeadSpec#embeddingProjection()} with its LoRA deltas
 *       (models/gemma-embedding2.cpp:206-212), which sets {@link #hiddenWidth()}.</li>
 *   <li>{@link #logits}: {@code output · h} ({@code token_embd} when tied), {@code + scale * B (A h)} of the LoRA
 *       deltas registered on {@code output.weight} ({@code build_lora_mm}, llama-graph.cpp:1629-1667; a tied head
 *       serves the same name, so a {@code token_embd} adapter adapts only the lookup), {@code + output.bias}
 *       ({@value #BIAS_ADD}), the logit scale ({@code / s} or {@code * s} per
 *       {@link HeadSpec#logitScaleDivides()}), then the final softcap {@code cap * tanh(x / cap)}, then
 *       {@code -FLT_MAX} over the masked logit range of every row (models/chameleon.cpp:186-198).</li>
 *   <li>{@link #argmax}: {@value #ARGMAX} per row on the device, so greedy decoding reads back one id.</li>
 * </ol>
 * Graphs whose block norms carry no weight and that have no {@code output_norm} tensor end with a weightless output
 * norm of the block kind (models/olmo.cpp:128-130, models/talkie.cpp:137); the HRM schedule already ends every pass
 * with its weightless norm (models/hrm-text.cpp:158, 203-211) and gets none.
 *
 * <p>The returned slots are views of {@link ScratchSlots#NORMED}, {@link ScratchSlots#X},
 * {@link ScratchSlots#LOGITS} or the head's own temporaries; they stay valid until the next step. No call allocates
 * device memory or waits on the device. Not thread-safe.</p>
 */
public final class GpuLogitsHead implements AutoCloseable {

    /** {@code data[s*dim + d] += bias[d]} ({@code kernels/llm/eltwise.cl}). */
    public static final String BIAS_ADD = GpuOps.BATCH_BIAS_ADD;
    /** Row-wise first maximum ({@code kernels/llm/sampling.cl}). */
    public static final String ARGMAX = "LLM_Argmax";
    /** Kernels this stage launches besides the {@link GpuOps} families, for prewarming. */
    public static final List<String> KERNELS = Collections.unmodifiableList(Arrays.asList(
            BIAS_ADD, ARGMAX, FlowKernels.GATHER_ROWS, NormKernels.ROW_NORM));

    /** Temporary holding the gathered row indices. */
    static final String TEMP_ROWS = "head.rows";
    /** Temporary holding the argmax ids. */
    static final String TEMP_ARGMAX = "head.argmax";
    /** Temporary holding the projected embedding rows. */
    static final String TEMP_EMBD = "head.embd";

    private final GpuForwardContext ctx;
    private final GpuOps ops;
    private final List<GpuHeadPreprocessor> preprocessors;
    private final boolean outputNorm;
    private final GpuWeight normWeight;
    private final GpuWeight normBias;
    private final NormKind normKind;
    private final float normEps;
    private final int normGroups;
    private final GpuWeight projection;
    /** LoRA deltas of the embedding projection, or null without one. */
    private final GpuProjectionDeltas.Site projectionDeltas;
    private final GpuWeight output;
    private final GpuWeight outputBias;
    /** LoRA deltas of the output head, or null without an LM head. */
    private final GpuProjectionDeltas.Site outputDeltas;
    private final int vocab;
    private final float logitMultiplier;
    private final float softcap;
    private final int maskStart;
    private final int maskEnd;
    private final int streamWidth;
    /** Width of the normed rows ({@code n_embd}). */
    private final int normWidth;
    private final int hiddenWidth;
    private final int ubatch;
    private final int reduceLocal;
    private final ScratchSlot x;
    private final ScratchSlot normed;
    private final ScratchSlot logitsSlot;
    private final ScratchSlot rowsSlot;
    private final ScratchSlot argmaxSlot;
    private final ScratchSlot embdSlot;
    private final ByteBuffer rowBytes;
    private boolean closed;

    /**
     * Binds the head to a model context and binds every preprocessor whose {@link GpuHeadPreprocessor#appliesTo}
     * accepts the model, in list order; the head closes them.
     *
     * @throws MissingTensorException   when a head tensor the spec names is not resident
     * @throws IllegalStateException    when the final stream is wider than {@code n_embd} and no preprocessor
     *                                  collapses it
     * @throws IllegalArgumentException when a head tensor does not match {@code n_embd}
     */
    public GpuLogitsHead(GpuForwardContext ctx, List<GpuHeadPreprocessor> preprocessors) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(preprocessors, "preprocessors");
        this.ops = ctx.ops();
        ModelSpec spec = ctx.spec();
        String arch = spec.arch().ggufName();
        HeadSpec head = spec.head();
        GpuWeights weights = ctx.weights();
        List<LayerSpec> layers = spec.layers();
        LayerSpec last = layers.get(layers.size() - 1);
        NormSpec lastNorm = last.norm();
        this.streamWidth = last.residual().streamWidth();
        this.normWidth = spec.embedding().nEmbd();
        this.hiddenWidth = head.hiddenWidth(normWidth);
        this.ubatch = ctx.scratch().ubatch();
        this.normKind = lastNorm.kind();
        this.normEps = lastNorm.eps();
        this.normGroups = head.outputNormGroups();
        this.outputNorm = head.outputNorm() != null
                || (allWeightless(layers) && spec.schedule().kind() != LayerSchedule.Kind.HRM);
        this.normWeight = head.outputNorm() != null ? vector(weights, arch, head.outputNorm()) : null;
        this.normBias = head.outputNormBias() != null ? vector(weights, arch, head.outputNormBias()) : null;
        if (normGroups > 1 && (normWeight == null || normWidth % normGroups != 0)) {
            throw new IllegalArgumentException(arch + ": an output norm over " + normGroups + " groups needs"
                    + " output_norm.weight and an n_embd " + normWidth + " divisible by the groups");
        }
        TensorRef proj = head.embeddingProjection();
        if (proj != null) {
            this.projection = require(weights, arch, proj.name());
            if (projection.cols() != normWidth || projection.rows() != hiddenWidth) {
                throw new IllegalArgumentException(arch + ": embedding projection " + projection.name() + " is "
                        + projection.cols() + " x " + projection.rows() + ", expected " + normWidth + " x "
                        + hiddenWidth);
            }
        } else {
            this.projection = null;
        }
        this.projectionDeltas = projection != null
                ? ctx.projectionDeltas().serveModel(TensorNames.weight(TensorNames.OUTPUT), projection) : null;
        TensorRef out = head.output();
        if (out != null) {
            this.output = require(weights, arch, out.name());
            if (output.cols() != normWidth) {
                throw new IllegalArgumentException(arch + ": output head " + output.name() + " has " + output.cols()
                        + " columns, n_embd is " + normWidth);
            }
            if (output.rows() > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(arch + ": output head " + output.name() + " has " + output.rows()
                        + " rows");
            }
            this.vocab = (int) output.rows();
            this.outputBias = head.outputBias() != null ? vector(weights, arch, head.outputBias()) : null;
            if (outputBias != null && outputBias.cols() != vocab) {
                throw new IllegalArgumentException(arch + ": output bias " + outputBias.name() + " has "
                        + outputBias.cols() + " values, the head has " + vocab + " rows");
            }
        } else {
            this.output = null;
            this.vocab = 0;
            this.outputBias = null;
        }
        this.outputDeltas = output != null
                ? ctx.projectionDeltas().serveModel(TensorNames.weight(TensorNames.OUTPUT), output) : null;
        float s = head.logitScale();
        this.logitMultiplier = head.logitScaleDivides() ? 1.0f / s : s;
        this.softcap = head.finalSoftcap();
        this.maskStart = head.maskedLogitsStart();
        this.maskEnd = head.maskedLogitsEnd();
        if (maskEnd > vocab) {
            throw new IllegalArgumentException(arch + ": masked logit range ends at " + maskEnd + ", the head has "
                    + vocab + " rows");
        }
        this.reduceLocal = Math.min(ops.local(), LlmKernelConstants.LLM_DIMX);
        this.x = ctx.scratch().slot(ScratchSlots.X).require((long) ubatch * streamWidth);
        this.normed = ctx.scratch().slot(ScratchSlots.NORMED).require((long) ubatch * streamWidth);
        this.logitsSlot = vocab > 0 ? ctx.scratch().slot(ScratchSlots.LOGITS).require((long) ubatch * vocab) : null;
        if (logitsSlot != null && logitsSlot.offset() != 0) {
            throw new IllegalStateException("the " + BIAS_ADD + " and " + ARGMAX + " kernels read the logits from"
                    + " element 0, the LOGITS slot starts at " + logitsSlot.offset());
        }
        this.rowsSlot = ctx.scratch().temp(TEMP_ROWS, ubatch);
        this.argmaxSlot = ctx.scratch().temp(TEMP_ARGMAX, ubatch);
        this.embdSlot = projection != null ? ctx.scratch().temp(TEMP_EMBD, (long) ubatch * hiddenWidth) : null;
        if (rowsSlot.offset() != 0 || argmaxSlot.offset() != 0) {
            throw new IllegalStateException("head temporaries must start at element 0");
        }
        this.rowBytes = ByteBuffer.allocate(ubatch * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        prewarmKernels();
        this.preprocessors = bindPreprocessors(ctx, preprocessors);
        if (streamWidth != normWidth && this.preprocessors.isEmpty()) {
            closePreprocessors(this.preprocessors);
            throw new IllegalStateException(arch + ": the final stream holds " + streamWidth + " values per token but"
                    + " no head preprocessor collapses it to n_embd " + normWidth);
        }
    }

    /**
     * Prepares the kernels this head launches at the work-group sizes it launches them with
     * ({@link GpuOps#prewarm}): the row gather, and with an LM head the output bias add and the argmax.
     */
    private void prewarmKernels() {
        ops.prewarm(FlowKernels.GATHER_ROWS, new long[]{FlowKernels.GATHER_LOCAL});
        if (outputBias != null) {
            ops.prewarm(BIAS_ADD, new long[]{ops.local()});
        }
        if (logitsSlot != null) {
            ops.prewarm(ARGMAX, new long[]{reduceLocal});
        }
    }

    private static boolean allWeightless(List<LayerSpec> layers) {
        for (LayerSpec l : layers) {
            if (!l.norm().weightless()) {
                return false;
            }
        }
        return true;
    }

    private static List<GpuHeadPreprocessor> bindPreprocessors(GpuForwardContext ctx, List<GpuHeadPreprocessor> all) {
        List<GpuHeadPreprocessor> bound = new ArrayList<>();
        try {
            for (GpuHeadPreprocessor p : all) {
                Objects.requireNonNull(p, "head preprocessor");
                if (p.appliesTo(ctx.spec())) {
                    p.bind(ctx);
                    bound.add(p);
                }
            }
        } catch (RuntimeException e) {
            closePreprocessors(bound);
            throw e;
        }
        return Collections.unmodifiableList(bound);
    }

    private static void closePreprocessors(List<GpuHeadPreprocessor> list) {
        RuntimeException first = Closeables.closeAll(list, null);
        if (first != null) {
            throw first;
        }
    }

    private GpuWeight vector(GpuWeights weights, String arch, TensorRef ref) {
        GpuWeight w = require(weights, arch, ref.name());
        if (w.type() != GGMLType.F32 || !w.isSingleShard()) {
            throw new IllegalArgumentException(arch + ": head vector " + w.name() + " must be a single-buffer F32"
                    + " tensor, got " + w);
        }
        return w;
    }

    private static GpuWeight require(GpuWeights weights, String arch, String name) {
        GpuWeight w = weights.find(name);
        if (w == null) {
            w = weights.getOrNull(name);
        }
        if (w == null) {
            throw new MissingTensorException(arch, -1, name);
        }
        return w;
    }

    /** True when the model has an LM head. */
    public boolean hasLmHead() {
        return output != null;
    }

    /** Floats per logits row (head rows including vocabulary padding); 0 without an LM head. */
    public int vocab() {
        return vocab;
    }

    /** Floats per hidden row returned by {@link #hidden}: {@code n_embd}, or the embedding projection's width. */
    public int hiddenWidth() {
        return hiddenWidth;
    }

    /** Head preprocessors bound to this model, in application order. */
    public List<GpuHeadPreprocessor> preprocessors() {
        return preprocessors;
    }

    /**
     * Selects {@code count} rows of the step's {@code nTokens} stream rows, in the order of {@code rows}, and
     * enqueues the head preprocessors and the output norm on them.
     *
     * @param rows step-local row indices (repeats allowed); the first {@code count} are used
     * @return {@code count} rows of {@link #hiddenWidth()} floats
     * @throws IllegalArgumentException when {@code count} is outside {@code [1, nTokens]} or a row is outside the
     *                                  step
     */
    public ScratchSlot hidden(int nTokens, int[] rows, int count) {
        ensureOpen();
        Objects.requireNonNull(rows, "rows");
        if (nTokens < 1 || nTokens > ubatch || count < 1 || count > rows.length || count > ubatch) {
            throw new IllegalArgumentException("head selection of " + count + " rows from " + nTokens
                    + " tokens is outside [1, " + ubatch + "]");
        }
        for (int i = 0; i < count; i++) {
            if (rows[i] < 0 || rows[i] >= nTokens) {
                throw new IllegalArgumentException("output row " + rows[i] + " is outside the " + nTokens
                        + " rows of the step");
            }
        }
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStageStart(-1, ExecutionObserver.Stage.OUTPUT_NORM);
        }
        ScratchSlot h = select(rows, count);
        for (GpuHeadPreprocessor p : preprocessors) {
            p.apply(count, h);
        }
        if (outputNorm) {
            outputNorm(h, count);
            h = normed;
        }
        if (projection != null) {
            ops.matmul(projection, h.buffer(), h.offset(), embdSlot.buffer(), embdSlot.offset(), count);
            projectionDeltas.applyRows(count, h.buffer(), h.offset(), embdSlot.buffer(), embdSlot.offset());
            h = embdSlot;
        }
        ScratchSlot result = h.slice(0, (long) count * hiddenWidth);
        if (observed) {
            obs.onStageEnd(-1, ExecutionObserver.Stage.OUTPUT_NORM, result, count, hiddenWidth);
        }
        return result;
    }

    /** Output norm of {@code count} rows of {@code h} into {@link ScratchSlots#NORMED}, grouped when configured. */
    private void outputNorm(ScratchSlot h, int count) {
        if (normGroups == 1) {
            ops.norm(normKind, normWeight, normBias, normEps, h.buffer(), h.offset(), normed.buffer(),
                    normed.offset(), count, normWidth);
            return;
        }
        int rows = count * normGroups;
        int slice = normWidth / normGroups;
        ops.enqueue(NormKernels.ROW_NORM, new long[]{(long) rows * ops.local()}, new long[]{ops.local()},
                NormKernels.rowNormArgs(normKind, normEps, rows, slice, slice, h.buffer(), h.offset(),
                        normed.buffer(), normed.offset(), normWeight.shard(0),
                        normBias != null ? normBias.shard(0) : null, normGroups,
                        ctx.placeholders().readOnlyFloats()));
    }

    /**
     * Rows of {@code X} in request order: X itself for the identity selection when nothing rewrites the rows
     * in place, one copy for a contiguous run, a gather otherwise.
     */
    private ScratchSlot select(int[] rows, int count) {
        int first = rows[0];
        boolean contiguous = true;
        for (int i = 1; i < count && contiguous; i++) {
            contiguous = rows[i] == first + i;
        }
        if (contiguous && first == 0 && preprocessors.isEmpty()) {
            return x;
        }
        if (contiguous) {
            ops.copy(x.buffer(), x.offset() + first * streamWidth, normed.buffer(), normed.offset(),
                    count * streamWidth);
            return normed;
        }
        rowBytes.clear();
        for (int i = 0; i < count; i++) {
            rowBytes.putInt(rows[i]);
        }
        GpuEvent ev = ctx.queue().enqueueWriteBuffer(rowsSlot.buffer(), 0, (long) count * Integer.BYTES,
                rowBytes.array());
        if (ev != null) {
            ev.close();
        }
        ops.enqueue(FlowKernels.GATHER_ROWS, new long[]{FlowKernels.gatherGlobal(count, streamWidth)},
                new long[]{FlowKernels.GATHER_LOCAL}, count, streamWidth, streamWidth, streamWidth, x.offset(),
                normed.offset(), rowsSlot.buffer(), x.buffer(), normed.buffer());
        return normed;
    }

    /**
     * Enqueues the logits of {@code nRows} hidden rows.
     *
     * @param hidden rows returned by {@link #hidden}
     * @return {@code nRows * vocab()} logits from element 0 of {@link ScratchSlots#LOGITS}
     * @throws UnsupportedOperationException when the model has no LM head
     */
    public ScratchSlot logits(int nRows, ScratchSlot hidden) {
        ensureOpen();
        requireLmHead();
        Objects.requireNonNull(hidden, "hidden");
        if (nRows < 1 || nRows > ubatch) {
            throw new IllegalArgumentException("logits of " + nRows + " rows outside [1, " + ubatch + "]");
        }
        hidden.require((long) nRows * hiddenWidth);
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStageStart(-1, ExecutionObserver.Stage.LOGITS);
        }
        int n = Math.multiplyExact(nRows, vocab);
        ops.matmul(output, hidden.buffer(), hidden.offset(), logitsSlot.buffer(), 0, nRows);
        outputDeltas.applyRows(nRows, hidden.buffer(), hidden.offset(), logitsSlot.buffer(), 0);
        if (outputBias != null) {
            ops.enqueue(BIAS_ADD, new long[]{roundUp(vocab, ops.local())}, new long[]{ops.local()}, vocab, nRows,
                    outputBias.shard(0), logitsSlot.buffer());
        }
        if (logitMultiplier != 1.0f) {
            ops.unary(ActOp.SCALE_BIAS, ActParams.builder().scaleBias(logitMultiplier, 0.0f).build(),
                    logitsSlot.buffer(), 0, logitsSlot.buffer(), 0, n);
        }
        if (softcap > 0.0f) {
            ops.unary(ActOp.SOFTCAP, ActParams.builder().softcap(softcap, 1.0f).build(), logitsSlot.buffer(), 0,
                    logitsSlot.buffer(), 0, n);
        }
        if (maskEnd > maskStart) {
            int pattern = Float.floatToRawIntBits(-Float.MAX_VALUE);
            for (int r = 0; r < nRows; r++) {
                ops.fill(logitsSlot.buffer(), ((long) r * vocab + maskStart) * Float.BYTES,
                        (long) (maskEnd - maskStart) * Float.BYTES, pattern);
            }
        }
        ScratchSlot result = logitsSlot.slice(0, n);
        if (observed) {
            obs.onStageEnd(-1, ExecutionObserver.Stage.LOGITS, result, nRows, vocab);
        }
        return result;
    }

    /**
     * Enqueues the greedy choice of each of the first {@code nRows} logits rows: the first maximum, NaN never
     * selected (llama.cpp greedy sampler order).
     *
     * @return {@code nRows} int ids from element 0 of the head's argmax temporary
     */
    public ScratchSlot argmax(int nRows) {
        ensureOpen();
        requireLmHead();
        if (nRows < 1 || nRows > ubatch) {
            throw new IllegalArgumentException("argmax of " + nRows + " rows outside [1, " + ubatch + "]");
        }
        ops.enqueue(ARGMAX, new long[]{(long) nRows * reduceLocal}, new long[]{reduceLocal}, vocab,
                logitsSlot.buffer(), argmaxSlot.buffer(), nRows);
        return argmaxSlot.slice(0, nRows);
    }

    private void requireLmHead() {
        if (output == null) {
            throw new UnsupportedOperationException(ctx.spec().arch().ggufName() + " has no LM head");
        }
    }

    private static long roundUp(long n, int local) {
        return (n + local - 1) / local * local;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("logits head is closed");
        }
    }

    /** Closes the bound preprocessors. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        closePreprocessors(preprocessors);
    }

    @Override
    public String toString() {
        return "GpuLogitsHead{" + (output != null ? output.name() + " x " + vocab : "no LM head")
                + (outputNorm ? ", norm " + normKind : "")
                + (logitMultiplier != 1.0f ? ", scale " + logitMultiplier : "")
                + (softcap > 0f ? ", softcap " + softcap : "")
                + (projection != null ? ", projection " + projection.name() : "")
                + (maskEnd > maskStart ? ", masked [" + maskStart + ", " + maskEnd + ")" : "")
                + ", preprocessors=" + preprocessors.size() + '}';
    }
}
