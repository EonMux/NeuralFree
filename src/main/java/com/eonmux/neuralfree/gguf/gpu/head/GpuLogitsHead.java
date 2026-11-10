package com.eonmux.neuralfree.gguf.gpu.head;

import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.engine.CertifiedArgmaxMode;
import com.eonmux.neuralfree.gguf.engine.CertifiedArgmaxTuning;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
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
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
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
 *   <li>{@link #logits}: {@code h = project_out · h} when {@link HeadSpec#outputProjection()} is present (OPT's
 *       {@code word_embed_proj_dim != hidden_size}); then {@code output · h} ({@code token_embd} when tied),
 *       {@code + scale * B (A h)} of the LoRA
 *       deltas registered on {@code output.weight} ({@code build_lora_mm}, llama-graph.cpp:1629-1667; a tied head
 *       serves the same name, so a {@code token_embd} adapter adapts only the lookup), {@code + output.bias}
 *       ({@value #BIAS_ADD}), the logit scale ({@code / s} or {@code * s} per
 *       {@link HeadSpec#logitScaleDivides()}), then the final softcap {@code cap * tanh(x / cap)}, then
 *       {@code -FLT_MAX} over the masked logit range of every row (models/chameleon.cpp:186-198).</li>
 *   <li>{@link #argmax}: {@value #ARGMAX} per row on the device, so greedy decoding reads back one id. A head that
 *       certifies argmax ({@link #certifiesArgmax}) serves the greedy readout instead through
 *       {@link #argmaxCertified}, which decides the ids on the device from the head's INT8 screen under its error
 *       bounds ({@link GpuCertArgmax}) without materializing the logits rows.</li>
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

    /**
     * A vocabulary-parallel LM head (vLLM's {@code ParallelLMHead}): every rank holds a row slice of the output
     * weight, computes its slice of the logits and the model device gathers the slices into the full logits row. The
     * head runs its own rank-0 slice through {@code output}; the split gathers the remaining ranks.
     */
    public interface VocabSplit {
        /** First output row the model device holds. */
        int headStart();

        /** Output rows the model device holds ({@code output.rows()}). */
        int headCount();

        /** Rows of the whole output head. */
        int headVocab();

        /**
         * The model device's logits-slice buffer, at element 0: rank 0's product is written there (the bias add reads
         * element 0), then {@link #gather} places it in the full logits row.
         */
        ScratchSlot shardLogits();

        /**
         * Enqueues the peer ranks' output products of the first {@code nRows} hidden rows and gathers every rank's
         * slice (rank 0's included, from {@code shardLogits}) into {@code fullLogits}, row {@code r} at columns
         * {@code [rank start, rank end)} of row {@code r}.
         */
        void gather(int nRows, ScratchSlot hidden, ScratchSlot shardLogits, ScratchSlot fullLogits);
    }

    /** {@code data[s*dim + d] += bias[d]} ({@code kernels/llm/eltwise.cl}). */
    public static final String BIAS_ADD = GpuOps.BATCH_BIAS_ADD;
    /** Row-wise first maximum ({@code kernels/llm/sampling.cl}). */
    public static final String ARGMAX = "LLM_Argmax";
    /** The tree verify's device accept walk over one masked batch's greedy ids ({@code kernels/llm/sampling.cl}). */
    public static final String TREE_ACCEPT = "LLM_TreeAccept";
    /** Kernels this stage launches besides the {@link GpuOps} families, for prewarming. */
    public static final List<String> KERNELS = Collections.unmodifiableList(Arrays.asList(
            BIAS_ADD, ARGMAX, TREE_ACCEPT, FlowKernels.GATHER_ROWS, NormKernels.ROW_NORM,
            NormKernels.MUL_ROW_SCALE));

    /** Temporary holding the gathered row indices. */
    static final String TEMP_ROWS = "head.rows";
    /** Temporary holding the argmax ids. */
    static final String TEMP_ARGMAX = "head.argmax";
    /** Temporary holding the projected embedding rows. */
    static final String TEMP_EMBD = "head.embd";
    /** Temporary holding the rows produced by {@code project_out} before the LM head. */
    static final String TEMP_PROJECTED = "head.projectOut";
    /** Temporary holding rank 0's vocabulary slice when the head is vocabulary-parallel. */
    static final String TEMP_SHARD = "head.vocabShard";

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
    /** {@code project_out}, or null without one. */
    private final GpuWeight outputProjection;
    /** LoRA deltas of {@code project_out}, or null without one. */
    private final GpuProjectionDeltas.Site outputProjectionDeltas;
    private final GpuWeight output;
    private final GpuWeight outputBias;
    /** Per-vocabulary logit multiplier applied after the projection; null when absent. */
    private final GpuWeight logitScaleVector;
    /** Vocabulary-parallel head, or null when the model device holds the whole output weight. */
    private final VocabSplit vocabSplit;
    /** Rows of the resident output weight: the whole vocabulary, or rank 0's slice. */
    private final int shardVocab;
    private final ScratchSlot shardSlot;
    /** LoRA deltas of the output head, or null without an LM head. */
    private final GpuProjectionDeltas.Site outputDeltas;
    private final int vocab;
    private final float logitMultiplier;
    private final float softcap;
    private final float sigmoidScale;
    private final float sigmoidShift;
    private final float sigmoidTemperature;
    private final int maskStart;
    private final int maskEnd;
    private final int streamWidth;
    /** Width of the normed rows ({@code n_embd}). */
    private final int normWidth;
    private final int hiddenWidth;
    /** Width of the rows the LM head reads ({@link HeadSpec#headInputWidth(int)}). */
    private final int lmWidth;
    private final int ubatch;
    private final int reduceLocal;
    private final ScratchSlot x;
    private final ScratchSlot normed;
    private final ScratchSlot logitsSlot;
    private final ScratchSlot rowsSlot;
    private final ScratchSlot argmaxSlot;
    private final ScratchSlot embdSlot;
    private final ScratchSlot projSlot;
    private final ByteBuffer rowBytes;
    private final boolean rowTopKKernel;
    private final boolean certifiable;
    private GpuRowTopK topKReadout;
    private GpuLogitPrefilter prefilterReadout;
    private GpuCertArgmax certReadout;
    private boolean prefilterKernel;
    /** Whether the certified kernels run at the matvec work-group size; null until first queried. */
    private Boolean certKernelsReady;
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
        this(ctx, preprocessors, null);
    }

    /**
     * Binds the head with a vocabulary-parallel output weight: the resident {@code output} holds only rank 0's row
     * slice, the head keeps the full vocabulary for masking, argmax and top-k, and {@link VocabSplit#gather} fills
     * the remaining rows.
     *
     * @throws IllegalArgumentException when the resident output weight does not match the split's rank-0 slice
     */
    public GpuLogitsHead(GpuForwardContext ctx, List<GpuHeadPreprocessor> preprocessors, VocabSplit vocabSplit) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(preprocessors, "preprocessors");
        this.vocabSplit = vocabSplit;
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
        this.lmWidth = head.headInputWidth(normWidth);
        TensorRef outProj = head.outputProjection();
        if (outProj != null) {
            this.outputProjection = require(weights, arch, outProj.name());
            if (outputProjection.cols() != normWidth || outputProjection.rows() != lmWidth) {
                throw new IllegalArgumentException(arch + ": project_out " + outputProjection.name() + " is "
                        + outputProjection.cols() + " x " + outputProjection.rows() + ", expected " + normWidth
                        + " x " + lmWidth);
            }
            this.outputProjectionDeltas = ctx.projectionDeltas().serveModel(outProj.name(), outputProjection);
        } else {
            this.outputProjection = null;
            this.outputProjectionDeltas = null;
        }
        TensorRef out = head.output();
        if (out != null) {
            this.output = require(weights, arch, out.name());
            if (output.cols() != lmWidth) {
                throw new IllegalArgumentException(arch + ": output head " + output.name() + " has " + output.cols()
                        + " columns, expected " + (lmWidth == normWidth ? "n_embd " + normWidth
                                : "the head input width " + lmWidth));
            }
            if (output.rows() > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(arch + ": output head " + output.name() + " has " + output.rows()
                        + " rows");
            }
            if (vocabSplit != null) {
                if (vocabSplit.headStart() != 0 || output.rows() != vocabSplit.headCount()
                        || output.rows() > vocabSplit.headVocab()) {
                    throw new IllegalArgumentException(arch + ": output head " + output.name() + " has "
                            + output.rows() + " rows, not the vocabulary split's rank-0 slice ["
                            + vocabSplit.headStart() + ", " + (vocabSplit.headStart() + vocabSplit.headCount())
                            + ") of " + vocabSplit.headVocab());
                }
                this.shardVocab = (int) output.rows();
                this.vocab = vocabSplit.headVocab();
            } else {
                this.shardVocab = (int) output.rows();
                this.vocab = this.shardVocab;
            }
            this.outputBias = head.outputBias() != null ? vector(weights, arch, head.outputBias()) : null;
            this.logitScaleVector = head.logitScaleVector() != null
                    ? vector(weights, arch, head.logitScaleVector()) : null;
            if (logitScaleVector != null && logitScaleVector.cols() != vocab) {
                throw new IllegalArgumentException(arch + ": logit scale vector " + logitScaleVector.name()
                        + " has " + logitScaleVector.cols() + " values, the head has " + vocab + " rows");
            }
            if (outputBias != null) {
                if (vocabSplit == null && outputBias.cols() != vocab) {
                    throw new IllegalArgumentException(arch + ": output bias " + outputBias.name() + " has "
                            + outputBias.cols() + " values, the head has " + vocab + " rows");
                }
                if (vocabSplit != null && outputBias.cols() != shardVocab) {
                    throw new IllegalArgumentException(arch + ": output bias " + outputBias.name() + " has "
                            + outputBias.cols() + " values, rank 0's vocabulary slice holds " + shardVocab);
                }
            }
        } else {
            if (vocabSplit != null) {
                throw new IllegalArgumentException(arch + ": the vocabulary split needs an LM head, the model has"
                        + " none");
            }
            this.output = null;
            this.vocab = 0;
            this.shardVocab = 0;
            this.outputBias = null;
            this.logitScaleVector = null;
        }
        this.outputDeltas = output != null
                ? ctx.projectionDeltas().serveModel(TensorNames.weight(TensorNames.OUTPUT), output) : null;
        float s = head.logitScale();
        this.logitMultiplier = head.logitScaleDivides() ? 1.0f / s : s;
        this.softcap = head.finalSoftcap();
        this.sigmoidScale = head.sigmoidCapped() ? head.sigmoidCapScale() : 0f;
        this.sigmoidShift = head.sigmoidCapShift();
        this.sigmoidTemperature = head.sigmoidCapTemperature();
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
        this.shardSlot = vocabSplit != null ? vocabSplit.shardLogits() : logitsSlot;
        if (vocabSplit != null && (shardSlot == null || shardSlot.offset() != 0
                || shardSlot.floats() < (long) ubatch * shardVocab)) {
            throw new IllegalStateException("the vocabulary split's rank-0 logits buffer holds "
                    + (shardSlot == null ? 0 : shardSlot.floats()) + " floats at offset "
                    + (shardSlot == null ? 0 : shardSlot.offset()) + ", " + ((long) ubatch * shardVocab)
                    + " are needed at element 0");
        }
        this.embdSlot = projection != null ? ctx.scratch().temp(TEMP_EMBD, (long) ubatch * hiddenWidth) : null;
        this.projSlot = outputProjection != null
                ? ctx.scratch().temp(TEMP_PROJECTED, (long) ubatch * lmWidth) : null;
        if (rowsSlot.offset() != 0 || argmaxSlot.offset() != 0) {
            throw new IllegalStateException("head temporaries must start at element 0");
        }
        this.rowBytes = ByteBuffer.allocate(ubatch * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        this.rowTopKKernel = ops.prewarm(GemmKernels.ROW_TOPK, new long[]{reduceLocal});
        this.prefilterKernel = logitsSlot != null
                && ops.prewarm(GemmKernels.LOGITS_PREFILTER, new long[]{reduceLocal});
        this.certifiable = initCertifiable();
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
            p.apply(count, h, rows);
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
        ScratchSlot h = hidden;
        if (outputProjection != null) {
            ops.matmul(outputProjection, hidden.buffer(), hidden.offset(), projSlot.buffer(), projSlot.offset(),
                    nRows);
            outputProjectionDeltas.applyRows(nRows, hidden.buffer(), hidden.offset(), projSlot.buffer(),
                    projSlot.offset());
            h = projSlot;
        }
        ops.matmul(output, h.buffer(), h.offset(), shardSlot.buffer(), shardSlot.offset(), nRows);
        outputDeltas.applyRows(nRows, h.buffer(), h.offset(), shardSlot.buffer(), shardSlot.offset());
        if (outputBias != null) {
            ops.enqueue(BIAS_ADD, new long[]{roundUp(shardVocab, ops.local())}, new long[]{ops.local()}, shardVocab,
                    nRows, outputBias.shard(0), shardSlot.buffer());
        }
        if (vocabSplit != null) {
            vocabSplit.gather(nRows, h, shardSlot, logitsSlot);
        }
        if (logitMultiplier != 1.0f) {
            ops.unary(ActOp.SCALE_BIAS, ActParams.builder().scaleBias(logitMultiplier, 0.0f).build(),
                    logitsSlot.buffer(), 0, logitsSlot.buffer(), 0, n);
        }
        if (logitScaleVector != null) {
            ops.enqueue(NormKernels.MUL_ROW_SCALE, new long[]{roundUp(n, ops.local())}, new long[]{ops.local()},
                    NormKernels.mulRowScaleArgs(nRows, vocab, logitScaleVector.shard(0), vocab, logitsSlot.buffer()));
        }
        if (softcap > 0.0f) {
            ops.unary(ActOp.SOFTCAP, ActParams.builder().softcap(softcap, 1.0f).build(), logitsSlot.buffer(), 0,
                    logitsSlot.buffer(), 0, n);
        }
        if (sigmoidScale != 0f && sigmoidTemperature != 0f) {
            ops.unary(ActOp.SCALE_BIAS, ActParams.builder()
                            .scaleBias(1.0f / sigmoidTemperature, sigmoidShift / sigmoidTemperature).build(),
                    logitsSlot.buffer(), 0, logitsSlot.buffer(), 0, n);
            ops.unary(ActOp.SIGMOID, ActParams.NONE, logitsSlot.buffer(), 0, logitsSlot.buffer(), 0, n);
            ops.unary(ActOp.SCALE_BIAS, ActParams.builder().scaleBias(sigmoidScale, 0.0f).build(),
                    logitsSlot.buffer(), 0, logitsSlot.buffer(), 0, n);
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

    /**
     * Whether this head carries nothing the certified lane cannot certify by name: a mode that serves
     * the stored type on this backend ({@link CertifiedArgmaxMode}, the measured pairs under
     * {@code auto} and every supported type under {@code on}), a plain single-buffer head weight with
     * no input transform, scale, Hadamard rotation or LoRA deltas, and no output bias, logit scale
     * vector, logit multiplier, softcap, sigmoid cap, masked range or vocabulary split. The hidden-side
     * stages (output norm, head preprocessors, {@code project_out} and its deltas) shape the hidden rows
     * both paths read, so they certify implicitly.
     */
    private boolean initCertifiable() {
        if (output == null || vocabSplit != null
                || !CertifiedArgmaxMode.configured().permitsScreen(ctx.backend().backendType(),
                        output.type(), output.rows() * output.cols())) {
            return false;
        }
        if (output.type() == GGMLType.Q8_0 && output.cols() % output.type().getBlockSize() != 0) {
            return false;
        }
        if (output.hasInputTransform() || output.hadamard() != null || output.hasScale()
                || !output.isSingleShard()) {
            return false;
        }
        return (outputDeltas == null || outputDeltas.isEmpty()) && outputBias == null && logitScaleVector == null
                && logitMultiplier == 1.0f && softcap <= 0.0f && sigmoidScale == 0f && maskEnd <= maskStart;
    }

    /** Whether the certified readout is configured for this head ({@link #certifiesRows} gates per call). */
    public boolean certifiesArgmax() {
        ensureOpen();
        return certifiable;
    }

    /**
     * Whether {@link #argmaxCertified} serves a readout of {@code nRows} rows: {@link #certifiesArgmax},
     * the certified kernels at the matvec work-group size, and the float matvec family serving the
     * {@code nRows}-row projection ({@link GpuOps#matvecFamilyServes}) &mdash; the rescore repeats that
     * lane's arithmetic, so a token count a fast, tiled or int8 lane serves keeps the exact readout.
     */
    public boolean certifiesRows(int nRows) {
        ensureOpen();
        if (!certifiable || nRows < 1 || nRows > ubatch || !kernelsReady()) {
            return false;
        }
        return ops.matvecFamilyServes(output, nRows);
    }

    /** Whether the certified kernels run at the matvec work-group size, remembered from the first query. */
    private boolean kernelsReady() {
        if (certKernelsReady == null) {
            certKernelsReady = ops.prewarm(GpuCertArgmax.SCREEN, new long[]{ops.matvecLocal()})
                    && ops.prewarm(GpuCertArgmax.rescoreKernel(output.type()), new long[]{ops.matvecLocal()});
        }
        return certKernelsReady;
    }

    /**
     * Enqueues the certified greedy choice of each of the first {@code nRows} hidden rows: the head's
     * INT8 screen, the error-bound threshold, the exact rescore of the certified candidates and the
     * in-kernel full-scan fallback, all on the device ({@link GpuCertArgmax}). Every committed id equals
     * the argmax of the exact logits rows, first index on ties.
     *
     * @param hidden rows returned by {@link #hidden}
     * @return {@code nRows} int ids from element 0 of the head's argmax temporary
     * @throws UnsupportedOperationException when the head does not certify argmax
     *                                       ({@link #certifiesArgmax}) or the row count runs on a
     *                                       projection lane the rescore does not repeat
     *                                       ({@link #certifiesRows})
     */
    public ScratchSlot argmaxCertified(int nRows, ScratchSlot hidden) {
        ensureOpen();
        requireLmHead();
        Objects.requireNonNull(hidden, "hidden");
        if (nRows < 1 || nRows > ubatch) {
            throw new IllegalArgumentException("certified argmax of " + nRows + " rows outside [1, " + ubatch + "]");
        }
        if (!certifiable || !kernelsReady()) {
            throw new UnsupportedOperationException(ctx.spec().arch().ggufName() + ": this head does not certify"
                    + " argmax; its greedy ids keep the exact readout");
        }
        if (!ops.matvecFamilyServes(output, nRows)) {
            throw new UnsupportedOperationException(ctx.spec().arch().ggufName() + ": the " + nRows
                    + "-row head projection runs on a lane the certified rescore does not repeat; the exact"
                    + " readout serves it");
        }
        hidden.require((long) nRows * hiddenWidth);
        if (certReadout == null) {
            certReadout = GpuCertArgmax.bind(ctx, output, ubatch, CertifiedArgmaxTuning.kmax(),
                    argmaxSlot.buffer());
            if (certReadout == null) {
                certKernelsReady = Boolean.FALSE;
                throw new UnsupportedOperationException(ctx.spec().arch().ggufName() + ": the certified"
                        + " rescore is not available on " + ctx.backend().backendType() + "; the greedy ids"
                        + " keep the exact readout");
            }
        }
        certReadout.run(nRows, hidden);
        return argmaxSlot.slice(0, nRows);
    }

    /** Whether the device row top-k readout runs on this backend ({@code LLM_Row_TopK}). */
    public boolean rowTopKAvailable() {
        ensureOpen();
        return rowTopKKernel;
    }

    /**
     * Whether the pre-top-k logit edits run on this backend ({@value GemmKernels#LOGITS_PREFILTER}): the fixed
     * logit-bias table, the penalties and DRY, applied to the logits rows before {@link #topK}.
     */
    public boolean prefilterAvailable() {
        ensureOpen();
        return prefilterKernel;
    }

    /**
     * The bound prefilter's DRY breaker uploads made and skipped ({@link GpuLogitPrefilter#breakerUploadCounts}),
     * zeros before the first prefiltered call.
     */
    public long[] prefilterBreakerUploads() {
        ensureOpen();
        return prefilterReadout == null ? new long[2] : prefilterReadout.breakerUploadCounts();
    }

    /**
     * Applies one prefilter row per output row of the first {@code prefilter.rows()} logits rows, in place: the
     * fixed logit-bias stage, the penalties and DRY, before the row top-k readout sees the row.
     *
     * @throws UnsupportedOperationException when the backend provides no prefilter kernel
     * @throws IllegalArgumentException      when a row names a token outside the vocabulary
     */
    public void prefilter(LogitPrefilter prefilter) {
        ensureOpen();
        requireLmHead();
        Objects.requireNonNull(prefilter, "prefilter");
        if (!prefilterKernel) {
            throw new UnsupportedOperationException(GemmKernels.LOGITS_PREFILTER + " is not available on "
                    + ctx.backend().backendType() + ", so sampled rows are read back whole");
        }
        if (prefilter.isEmpty()) {
            return;
        }
        if (this.prefilterReadout == null) {
            GpuLogitPrefilter bound = GpuLogitPrefilter.bind(ops, ctx.backend(), ctx.queue(), logitsSlot.buffer(),
                    ubatch, vocab);
            if (bound == null) {
                throw new UnsupportedOperationException(GemmKernels.LOGITS_PREFILTER + " is not available on "
                        + ctx.backend().backendType() + ", so sampled rows are read back whole");
            }
            this.prefilterReadout = bound;
        }
        this.prefilterReadout.run(prefilter);
    }

    /**
     * Enqueues the top-k readout of the first {@code nRows} logits rows ({@code LLM_Row_TopK}): per row the k best
     * candidates in rank order (larger logit first, the lower id on ties) and the row's softmax denominator, so a
     * truncated sampler chain samples from a few read-back values per row (llama.cpp backend sampling {@code -bs}).
     * The readout binds at its first {@code k} and rebinds for a later, larger {@code k}, so a caller may widen the
     * cone between steps; a narrower {@code k} reuses the bound readout and reads its first candidates.
     *
     * @throws UnsupportedOperationException when the backend provides no row top-k kernel
     * @throws IllegalArgumentException      when {@code nRows} is outside {@code [1, ubatch]} or {@code k} is
     *                                       below one
     */
    public GpuRowTopK topK(int nRows, int k) {
        ensureOpen();
        requireLmHead();
        if (nRows < 1 || nRows > ubatch) {
            throw new IllegalArgumentException("top-k of " + nRows + " rows outside [1, " + ubatch + "]");
        }
        if (k < 1) {
            throw new IllegalArgumentException("top-k readback of " + k + " candidates");
        }
        if (topKReadout != null && k > topKReadout.k()) {
            topKReadout.close();
            topKReadout = null;
        }
        if (topKReadout == null) {
            GpuRowTopK bound = GpuRowTopK.bind(ops, ctx.backend(), logitsSlot.buffer(), ubatch, vocab, k);
            if (bound == null) {
                throw new UnsupportedOperationException(GemmKernels.ROW_TOPK + " is not available on "
                        + ctx.backend().backendType() + ", so sampled rows are read back whole");
            }
            topKReadout = bound;
        }
        topKReadout.run(nRows);
        return topKReadout;
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

    /** Closes the bound preprocessors, the top-k readout, the logit prefilter and the certified readout. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (topKReadout != null) {
            topKReadout.close();
            topKReadout = null;
        }
        if (prefilterReadout != null) {
            prefilterReadout.close();
            prefilterReadout = null;
        }
        if (certReadout != null) {
            certReadout.close();
            certReadout = null;
        }
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
