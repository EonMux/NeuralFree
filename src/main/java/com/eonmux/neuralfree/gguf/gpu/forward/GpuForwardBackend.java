package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.DraftTree;
import com.eonmux.neuralfree.gguf.engine.EmbeddingRowSupport;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.engine.OutputKind;
import com.eonmux.neuralfree.gguf.engine.RecurrentTreeRound;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.RopeSectionSupport;
import com.eonmux.neuralfree.gguf.engine.TreeAccept;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuRecurrentTreeMixer;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * The GPU {@link ForwardBackend}: one embedding stage, one {@link GpuLayerExecutor} over the trunk and one
 * {@link GpuLogitsHead}, driven per ubatch by a {@link GpuPrefillDriver}. Every method enqueues its whole call and
 * synchronizes once, when the requested logits, hidden rows or greedy ids are read back.
 *
 * <p>Memory: causal calls ({@link #prefill}, {@link #decode}, {@link #forwardRows}) write the context's
 * {@link GpuMemory} and continue it at its current length; a call starting earlier rolls the tail back first when
 * the memory allows it. Models planned without memory (encoders, diffusion, classifiers) serve {@link #forwardAll}
 * and {@link #hiddenRows} in one step of at most one ubatch. {@link #hiddenRows} on a model with memory runs
 * causally on an empty memory and leaves it empty. On a paged memory ({@code neuralfree.parallel > 1} or
 * {@code neuralfree.kv.paged=true}), {@link #forwardBatch} evaluates a {@link ForwardBatch} over several sequences
 * at once: multi-sequence ubatches through the device page tables and the paged attention kernels, one logits row
 * per requested batch row ({@link #hiddenBatch}: one hidden row; {@link #forwardBatchTopK}: the row's best
 * candidates, {@code LLM_Row_TopK}). A batch's rows are token rows or precomputed
 * embedding rows ({@code prompt_embeds}, media embeddings), mixed in one ubatch as the embedding stage writes each
 * by its kind; a media batch's RoPE sections and mask rule ({@code ForwardBatch.nonCausal}) reach the paged
 * attention, and the hidden tap receives every row at its batch row, as on the CPU backend.
 * {@link #prefillEmbeddings} runs embedding rows on the contiguous memory.</p>
 *
 * <p>The context's queue, scratch, weights, memory and placeholders belong to whoever built the context; this
 * backend closes only the blocks and head preprocessors it bound. Not thread-safe.</p>
 */
public final class GpuForwardBackend implements ForwardBackend {

    /** The system property {@code false} sets keeps decode steps out of a captured graph. */
    public static final String DECODE_REPLAY_PROPERTY = "neuralfree.gpu.decodeReplay";
    /**
     * The system property {@code true} extends the trunk replay into a speculative round's steps: a
     * rollback keeps the drafter's captured decode instead of dropping it, and the memory-backed
     * multi-row verify step captures its own graph. On CUDA the default is {@code true}: the patched
     * replay's per-step patch cost sits outside the launched step (the follow-up graph is prepared
     * while the current one runs), and keeping the capture across rollbacks saves the capture and its
     * two verify steps in every speculative round (measured on an RTX 5080 Laptop with the decode
     * replay on: chain throughput 32.9 against 31.7 tokens/s on a 4B target with a 0.6B drafter, 117.8
     * against 115.6 tokens/s on a 0.6B F16 pair, 657.3 against 641.0 tokens/s on a synthetic pair).
     * Other backends default to {@code false}; a device that cannot capture or whose replay cannot
     * patch recorded arguments gains nothing from a kept capture.
     */
    public static final String CHAIN_REPLAY_PROPERTY = "neuralfree.gpu.chainReplay";

    private final GpuForwardContext ctx;
    private final GpuEmbeddingStage embedding;
    private final GpuLayerExecutor executor;
    private final GpuLogitsHead head;
    private final GpuPrefillDriver driver;
    /** Whether a speculative round's steps keep and capture their trunks ({@code chainReplayConfigured(backend)}). */
    private final boolean chainReplay;
    /** The open recurrent tree round, or null when no tree verify runs over a recurrent target. */
    private GpuRecurrentTreeRound treeRound;
    private HiddenTap tap;
    private boolean closed;

    /**
     * Whether {@code neuralfree.gpu.chainReplay} is explicitly set to {@code true}. The model factory checks it
     * before the GPU load, where the backend is not known yet: an explicit {@code true} asks for a GPU-only
     * behavior, so a refused GPU load is rethrown instead of falling back to the CPU engine, which ignores the
     * setting. The per-backend default ({@code true} on CUDA, off elsewhere) is
     * {@link #chainReplayConfigured(ComputeBackend.BackendType)}.
     *
     * @throws IllegalArgumentException when the property is set to something else than a boolean
     */
    public static boolean chainReplayConfigured() {
        Boolean explicit = chainReplayProperty();
        return explicit != null && explicit;
    }

    /**
     * {@code neuralfree.gpu.chainReplay} as a boolean; the default is {@code true} on CUDA and
     * {@code false} on every other backend. An explicit value overrides the default on every backend.
     *
     * @throws IllegalArgumentException when the property is set to something else than a boolean
     */
    static boolean chainReplayConfigured(ComputeBackend.BackendType backend) {
        Boolean explicit = chainReplayProperty();
        return explicit != null ? explicit : backend == ComputeBackend.BackendType.CUDA;
    }

    /**
     * The parsed {@code neuralfree.gpu.chainReplay} property, or null when it is unset (the caller applies its
     * default).
     *
     * @throws IllegalArgumentException when the property is set to something else than a boolean
     */
    private static Boolean chainReplayProperty() {
        String value = System.getProperty(CHAIN_REPLAY_PROPERTY);
        if (value == null || value.isEmpty()) {
            return null;
        }
        if (value.equalsIgnoreCase("true")) {
            return Boolean.TRUE;
        }
        if (value.equalsIgnoreCase("false")) {
            return Boolean.FALSE;
        }
        throw new IllegalArgumentException("-D" + CHAIN_REPLAY_PROPERTY + " must be a boolean, got '" + value
                + "'");
    }

    /**
     * Binds every block of the model.
     *
     * @param registry      the mixer, FFN and extra factories of every family, including
     *                      {@link BuiltInGpuExtras#factories()}
     * @param preprocessors head preprocessors of the families ({@code HcBlocks.gpuHeadPreprocessors()}, ...)
     * @throws UnsupportedOperationException when the registry lacks a block the spec needs
     * @throws IllegalStateException         when the spec and the registered extras disagree
     */
    public GpuForwardBackend(GpuForwardContext ctx, GpuBlockRegistry registry,
                             List<GpuHeadPreprocessor> preprocessors) {
        this(ctx, registry, preprocessors, null, null);
    }

    /**
     * Binds every block of the model with a vocabulary-parallel embedding and LM head: the resident token table and
     * output weight hold only the model device's row slices, the lookup sums every rank's rows and the head gathers
     * every rank's logits slice.
     */
    public GpuForwardBackend(GpuForwardContext ctx, GpuBlockRegistry registry,
                             List<GpuHeadPreprocessor> preprocessors,
                             GpuEmbeddingStage.VocabParallel embeddingSplit, GpuLogitsHead.VocabSplit headSplit) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(registry, "registry");
        Objects.requireNonNull(preprocessors, "preprocessors");
        ModelSpec spec = ctx.spec();
        registry.validate(spec);
        ctx.kernels().prewarm(kernels(spec));
        this.embedding = new GpuEmbeddingStage(ctx, embeddingSplit);
        GpuLayerExecutor exec = GpuLayerExecutor.forTrunk(ctx, registry);
        GpuLogitsHead h;
        try {
            h = new GpuLogitsHead(ctx, preprocessors, headSplit);
        } catch (RuntimeException e) {
            Closeables.close(exec, e);
            throw e;
        }
        GpuPrefillDriver d;
        try {
            d = new GpuPrefillDriver(ctx, embedding, exec, h);
        } catch (RuntimeException e) {
            Closeables.close(h, Closeables.close(exec, e));
            throw e;
        }
        this.executor = exec;
        this.head = h;
        this.driver = d;
        this.chainReplay = chainReplayConfigured(ctx.backend().backendType());
        this.tap = ctx.hiddenTap();
    }

    /** Kernels the executor, the head and the built-in extras launch, plus attention when a layer attends. */
    public static List<String> kernels(ModelSpec spec) {
        Set<String> names = new LinkedHashSet<>();
        names.addAll(GpuLayerExecutor.KERNELS);
        names.addAll(BuiltInGpuExtras.KERNELS);
        names.addAll(GpuLogitsHead.KERNELS);
        names.add(GpuOps.UNARY);
        for (LayerSpec l : spec.layers()) {
            if (l.attention() != null || l.crossAttention() != null || l.mla() != null) {
                names.addAll(AttnKernels.all());
                break;
            }
        }
        return new ArrayList<>(names);
    }

    // ================================================================== causal calls

    /** Installs the runner of every PREFILL step's trunk (the prefill context-parallel coordinator), or null. */
    public void trunkRunner(GpuPrefillDriver.TrunkRunner runner) {
        driver.trunkRunner(runner);
    }

    /** Installs the runner of every batched PREFILL step's trunk (the coordinator's paged mode), or null. */
    public void batchTrunkRunner(GpuPrefillDriver.BatchTrunkRunner runner) {
        driver.batchTrunkRunner(runner);
    }

    /** The trunk executor this backend drives (the prefill context-parallel coordinator's rank 0 phases). */
    public GpuLayerExecutor layerExecutor() {
        return executor;
    }

    @Override
    public float[] prefill(int[] tokens, int pos0) {
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("prefill without tokens");
        }
        float[][] rows = causal(tokens, pos0, new int[]{tokens.length - 1}, GpuPrefillDriver.Output.LOGITS).rows();
        return rows[0];
    }

    @Override
    public float[] decode(int token, int pos) {
        return causal(new int[]{token}, pos, new int[]{0}, GpuPrefillDriver.Output.LOGITS).rows()[0];
    }

    /** Greedy selection on the device ({@code LLM_Argmax}); only the id is read back. */
    @Override
    public int decodeGreedyOnDevice(int token, int pos) {
        return causal(new int[]{token}, pos, new int[]{0}, GpuPrefillDriver.Output.ARGMAX).ids()[0];
    }

    @Override
    public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
        Objects.requireNonNull(tokens, "tokens");
        return causal(tokens, pos0, outputRows, GpuPrefillDriver.Output.LOGITS).rows();
    }

    /** Greedy selection of every requested row on the device ({@code LLM_Argmax}); only the ids are read back. */
    @Override
    public int[] forwardRowsGreedy(int[] tokens, int pos0, int[] outputRows) {
        Objects.requireNonNull(tokens, "tokens");
        return causal(tokens, pos0, outputRows, GpuPrefillDriver.Output.ARGMAX).ids();
    }

    /**
     * Evaluates precomputed embedding rows causally at positions {@code pos0 ..} on the contiguous memory, storing
     * them like {@link #prefill}; each row replaces the token-table lookup ({@code GpuEmbeddingStage} class notes).
     *
     * @throws IllegalArgumentException      when a row is null or another width than {@code n_embd} or the model
     *                                       input width
     * @throws UnsupportedOperationException when the model cannot take embedding rows
     *                                       ({@link #embeddingRowsRefusal()}) or has no LM head
     */
    @Override
    public float[] prefillEmbeddings(float[][] rows, int pos0) {
        Objects.requireNonNull(rows, "rows");
        ensureOpen();
        if (rows.length == 0) {
            throw new IllegalArgumentException("no embedding rows to evaluate");
        }
        String refusal = embeddingRowsRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        checkRowWidths(rows);
        GpuMemory memory = requireMemory("prefillEmbeddings");
        align(memory, pos0, rows.length);
        return driver.runRows(rows, pos0, new int[]{rows.length - 1}, GpuPrefillDriver.Output.LOGITS, tap).rows()[0];
    }

    /** Null unless a block of the model reads the step's token ids ({@link EmbeddingRowSupport#refusal}). */
    @Override
    public String embeddingRowsRefusal() {
        return EmbeddingRowSupport.refusal(ctx.spec());
    }

    /**
     * Why the single-pass tree step cannot run over this model's recurrent layers, or null when it can: the memory
     * is paged (the tree verify's masked step needs the paged prefill kernel), the schedule is linear (a looped or
     * hierarchical schedule runs one physical layer's mixer over several trunk slots, whose per-layer captures and
     * state slots would collide), and every recurrent layer's mixer implements {@link GpuRecurrentTreeMixer} and
     * names no refusal (the gated DeltaNet, KDA and Mamba-family mixers do; a Mamba-1 or PLaMo-2 mixer serves its
     * kind).
     */
    @Override
    public String recurrentTreeRefusal() {
        if (ctx.hasMemory() && ctx.memory().pagedSequences() == null) {
            return "a tree verify batch needs the paged prefill kernel: run with neuralfree.kv.paged=true";
        }
        LayerSchedule schedule = ctx.spec().schedule();
        if (schedule != null && schedule.kind() != LayerSchedule.Kind.LINEAR) {
            return "the recurrent tree step runs a linear layer schedule, the model runs a " + schedule.kind()
                    + " schedule";
        }
        for (int i = 0; i < executor.layerCount(); i++) {
            LayerSpec l = executor.spec(i);
            if (l.recurrent() == null) {
                continue;
            }
            if (!(executor.mixer(i) instanceof GpuRecurrentTreeMixer)) {
                return "layer " + l.il() + " runs the " + l.mixerKind()
                        + " mixer, whose single-pass tree step is not covered";
            }
            String refusal = ((GpuRecurrentTreeMixer) executor.mixer(i)).treeStepRefusal();
            if (refusal != null) {
                return "layer " + l.il() + " (" + l.mixerKind() + ") refuses the tree step: " + refusal;
            }
        }
        return null;
    }

    @Override
    public RecurrentTreeRound openRecurrentTreeRound(int rows, int realRows) {
        ensureOpen();
        String refusal = recurrentTreeRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        if (treeRound != null) {
            throw new IllegalStateException("a recurrent tree round is already open");
        }
        int count = 0;
        int maxWidth = 0;
        long maxConv = 0;
        long maxScan = 0;
        for (int i = 0; i < executor.layerCount(); i++) {
            RecurrentSpec r = executor.spec(i).recurrent();
            if (r == null) {
                continue;
            }
            count++;
            maxWidth = Math.max(maxWidth, ((GpuRecurrentTreeMixer) executor.mixer(i)).treeRowWidth());
            maxConv = Math.max(maxConv, r.convStateFloats());
            maxScan = Math.max(maxScan, r.stateFloats());
        }
        int[] ils = new int[count];
        int[] capBase = new int[count];
        int[] widths = new int[count];
        long cap = 0;
        int n = 0;
        for (int i = 0; i < executor.layerCount(); i++) {
            LayerSpec l = executor.spec(i);
            if (l.recurrent() == null) {
                continue;
            }
            int width = ((GpuRecurrentTreeMixer) executor.mixer(i)).treeRowWidth();
            ils[n] = l.il();
            capBase[n] = (int) cap;
            widths[n] = width;
            // the layer's capture slice holds one mixer-input row per verify row
            cap += Math.multiplyExact((long) rows, width);
            n++;
        }
        treeRound = new GpuRecurrentTreeRound(this, ctx.ops(), rows, realRows, maxWidth, maxConv, maxScan, cap,
                ils, capBase, widths);
        return treeRound;
    }

    /** The replay walk of a committing round: every recurrent layer decodes the rows against its committed state. */
    void replayRecurrentTree(GpuRecurrentTreeRound round, int[] replayRows) {
        ScratchSlot out = round.replayOutput();
        for (int i = 0; i < executor.layerCount(); i++) {
            LayerSpec l = executor.spec(i);
            GpuMixer m = executor.mixer(i);
            if (l.recurrent() == null || !(m instanceof GpuRecurrentTreeMixer)) {
                continue;
            }
            for (int r : replayRows) {
                m.decode(0, round.replayInput(l.il(), r), out);
            }
        }
    }

    /** Forgets the closing round; the committed state stays live for the steps that follow. */
    void closeTreeRound(GpuRecurrentTreeRound round) {
        if (treeRound == round) {
            treeRound = null;
        }
    }

    /** Every non-null row holds {@code n_embd} or the model input width; null rows are refused when not allowed. */
    private void checkRowWidths(float[][] rows) {
        for (int i = 0; i < rows.length; i++) {
            if (rows[i] == null) {
                throw new IllegalArgumentException("embedding row " + i + " is null");
            }
            checkRowWidth(rows[i], i);
        }
    }

    private void checkRowWidth(float[] row, int i) {
        if (!embedding.acceptsRowWidth(row.length)) {
            throw new IllegalArgumentException("embedding row " + i + " holds " + row.length + " values, the model"
                    + " input is " + embedding.inputWidth() + " wide (n_embd " + embedding.nEmbd() + ")");
        }
    }

    // ================================================================== batched calls

    /** Sequences the paged memory holds ({@code PagedSequences.maxSequences()}); 1 on the contiguous form. */
    @Override
    public int maxSequences() {
        ensureOpen();
        if (!ctx.hasMemory()) {
            return 1;
        }
        return ctx.memory() instanceof DeviceModelMemory && ((DeviceModelMemory) ctx.memory()).pagedSequences() != null
                ? ((DeviceModelMemory) ctx.memory()).pagedSequences().maxSequences() : 1;
    }

    /**
     * Null when this backend can run {@link #forwardBatch}: a memory with paged sequences whose attention mixers
     * run the paged kernels. Otherwise the reason by name; mixers without a batched step refuse at the call too.
     */
    @Override
    public String batchRefusal() {
        if (!ctx.hasMemory()) {
            return "multi-sequence batches need a model with paged memory";
        }
        return ctx.memory() instanceof DeviceModelMemory && ((DeviceModelMemory) ctx.memory()).pagedSequences() != null
                ? null : "multi-sequence batches need paged GPU memory (neuralfree.parallel > 1 or"
                + " neuralfree.kv.paged=true)";
    }

    /**
     * Evaluates the batch over the paged memory: the rows split into multi-sequence ubatches
     * ({@code split_equal}), each ubatch reserving its pages all-or-nothing, writing its token and embedding rows,
     * running the trunk through the page tables and reading the output rows' logits in batch order.
     *
     * @throws UnsupportedOperationException carrying {@link #batchRefusal()} or the refusing reason (a row in an
     *                                       adapter slot that holds no adapter, embedding rows on a model that
     *                                       cannot take them,
     *                                       a {@link OutputKind#ROUTING} row, a mixer without a batched step)
     * @throws IllegalArgumentException      when an embedding row's width or the RoPE section count does not fit
     *                                       the model
     */
    @Override
    public float[][] forwardBatch(ForwardBatch batch) {
        return maskedForwardBatch(batch, GpuPrefillDriver.Output.LOGITS);
    }

    /**
     * {@link #forwardBatch} reading the output rows' final hidden states (after the output norm, the input of the
     * pooling and classifier heads) instead of their logits. A model planned without memory runs the whole batch as
     * one memoryless step ({@link GpuPrefillDriver#runMemorylessBatch}).
     */
    @Override
    public float[][] hiddenBatch(ForwardBatch batch) {
        return maskedForwardBatch(batch, GpuPrefillDriver.Output.HIDDEN);
    }

    /**
     * One batched call: a step whose tree mask runs over a recurrent target must run inside the open recurrent tree
     * round (the executor's mixer blocks route the recurrent layers' steps through it); without the round the step
     * would advance the committed state per row and leave it after the wrong tokens, so it fails loud. A model with
     * no memory or no recurrent state runs plain, mask or not.
     */
    private float[][] maskedForwardBatch(ForwardBatch batch, GpuPrefillDriver.Output output) {
        if (!ctx.hasMemory() || batch.treeMask() == null || !ctx.memory().hasRecurrentState()) {
            return runBatch(batch, output);
        }
        if (treeRound == null) {
            throw new IllegalStateException("the step carries a tree mask over a model with recurrent state, but"
                    + " no recurrent tree round is open: open one with openRecurrentTreeRound before the masked"
                    + " forward");
        }
        driver.recurrentTreeRound(treeRound);
        try {
            return runBatch(batch, output);
        } finally {
            driver.recurrentTreeRound(null);
        }
    }

    private float[][] runBatch(ForwardBatch batch, GpuPrefillDriver.Output output) {
        if (!ctx.hasMemory()) {
            return driver.runMemorylessBatch(batch, output, tap).rows();
        }
        return batched(batch, output).rows();
    }

    /**
     * Whether the single-pass tree verify's acceptance walk runs on the device ({@code LLM_TreeAccept}): the
     * model runs over paged memory and the backend provides the kernel.
     */
    @Override
    public boolean treeAcceptOnDevice() {
        ensureOpen();
        if (!ctx.hasMemory() || ctx.memory().pagedSequences() == null || !head.hasLmHead()) {
            return false;
        }
        return ctx.ops().prewarm(GpuLogitsHead.TREE_ACCEPT,
                new long[]{Math.min(ctx.ops().local(), LlmKernelConstants.LLM_DIMX)});
    }

    /**
     * {@link #forwardBatch} of the one masked single-pass verify batch with its acceptance walked on the device;
     * a step over a model with recurrent state runs inside the open recurrent tree round exactly as
     * {@link #forwardBatch} does.
     */
    @Override
    public TreeAccept treeAcceptForward(ForwardBatch batch, DraftTree proposal) {
        Objects.requireNonNull(batch, "batch");
        Objects.requireNonNull(proposal, "proposal");
        ensureOpen();
        if (!ctx.hasMemory() || batch.treeMask() == null || !ctx.memory().hasRecurrentState()) {
            return driver.runBatchTreeAccept(batch, batch.ropePositions(), proposal, tap);
        }
        if (treeRound == null) {
            throw new IllegalStateException("the step carries a tree mask over a model with recurrent state, but"
                    + " no recurrent tree round is open: open one with openRecurrentTreeRound before the masked"
                    + " forward");
        }
        driver.recurrentTreeRound(treeRound);
        try {
            return driver.runBatchTreeAccept(batch, batch.ropePositions(), proposal, tap);
        } finally {
            driver.recurrentTreeRound(null);
        }
    }

    /** Whether the head's device row top-k readout runs (its {@code LLM_Row_TopK} kernel). */
    @Override
    public boolean batchTopKOnDevice() {
        ensureOpen();
        return head.rowTopKAvailable();
    }

    /**
     * Top candidates of every output row on the device; only the candidates are read back. The width shrinks to the
     * vocabulary when {@code k} exceeds it, as the host readback does.
     */
    @Override
    public RowTopK forwardBatchTopK(ForwardBatch batch, int k) {
        return forwardBatchTopK(batch, k, null);
    }

    /** Whether the head's prefilter kernel runs on this backend ({@code LLM_LogitsPrefilter}). */
    @Override
    public boolean batchPrefilterOnDevice() {
        ensureOpen();
        return head.prefilterAvailable();
    }

    /**
     * Top candidates of every output row on the device after the per-row prefilter (the logit bias, penalties and
     * DRY); only the candidates are read back.
     */
    @Override
    public RowTopK forwardBatchTopK(ForwardBatch batch, int k, LogitPrefilter prefilter) {
        Objects.requireNonNull(batch, "batch");
        if (k < 1) {
            throw new IllegalArgumentException("top-k readback of " + k + " candidates");
        }
        ensureOpen();
        int width = Math.min(k, head.vocab());
        GpuPrefillDriver.Result r = batched(batch, GpuPrefillDriver.Output.TOPK, width, prefilter);
        return RowTopK.packed(width, batch.outputRows().length, r.topIds(), r.topLogits());
    }

    private GpuPrefillDriver.Result batched(ForwardBatch batch, GpuPrefillDriver.Output output) {
        return batched(batch, output, 0, null);
    }

    private GpuPrefillDriver.Result batched(ForwardBatch batch, GpuPrefillDriver.Output output, int topK) {
        return batched(batch, output, topK, null);
    }

    private GpuPrefillDriver.Result batched(ForwardBatch batch, GpuPrefillDriver.Output output, int topK,
                                            LogitPrefilter prefilter) {
        Objects.requireNonNull(batch, "batch");
        ensureOpen();
        String refusal = batchRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        refusal = batch.adapterSlotRefusal(getClass().getSimpleName(), ctx.projectionDeltas()::holdsSlot);
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        float[][] embd = batch.embeddings();
        if (embd != null) {
            refusal = embeddingRowsRefusal();
            if (refusal != null) {
                throw new UnsupportedOperationException(refusal);
            }
            for (int i = 0; i < embd.length; i++) {
                if (embd[i] != null) {
                    checkRowWidth(embd[i], i);
                }
            }
        }
        int[][] ropeSections = batch.ropePositions();
        RopeSectionSupport.check(ctx.spec().hparams().ropeType(), ropeSections);
        for (int i = 0; i < batch.nTokens(); i++) {
            if (batch.produces(i, OutputKind.ROUTING)) {
                throw new UnsupportedOperationException("row " + i + " requests " + OutputKind.ROUTING
                        + "; the batched GPU path reads logits (forwardBatch), hidden rows (hiddenBatch) or top"
                        + " candidates (forwardBatchTopK)");
            }
        }
        return output == GpuPrefillDriver.Output.TOPK
                ? driver.runBatchTopK(batch, ropeSections, topK, tap, prefilter)
                : driver.runBatch(batch, ropeSections, output, tap);
    }

    private GpuPrefillDriver.Result causal(int[] tokens, int pos0, int[] rows, GpuPrefillDriver.Output output) {
        ensureOpen();
        GpuMemory memory = requireMemory("causal evaluation");
        align(memory, pos0, tokens.length);
        return driver.run(tokens, null, pos0, rows, output, true, tap);
    }

    /** The causal call of a top-k readback: per requested row its {@code k} best candidates in rank order. */
    private GpuPrefillDriver.Result causalTopK(int[] tokens, int pos0, int[] rows, int k) {
        return causalTopK(tokens, pos0, rows, k, null);
    }

    /** {@link #causalTopK(int[], int, int[], int)} with the rows' pre-top-k edits applied first. */
    private GpuPrefillDriver.Result causalTopK(int[] tokens, int pos0, int[] rows, int k, LogitPrefilter prefilter) {
        ensureOpen();
        GpuMemory memory = requireMemory("causal evaluation");
        align(memory, pos0, tokens.length);
        return driver.runTopK(tokens, pos0, rows, k, tap, prefilter);
    }

    /** Whether the head's device row top-k readout runs (its {@code LLM_Row_TopK} kernel). */
    @Override
    public boolean topKOnDevice() {
        ensureOpen();
        return head.rowTopKAvailable();
    }

    /** Whether the head's prefilter kernel runs on this backend ({@code LLM_LogitsPrefilter}). */
    @Override
    public boolean topKPrefilterOnDevice() {
        ensureOpen();
        return head.prefilterAvailable();
    }

    /** Top candidates of every requested row on the device; only the candidates are read back. */
    @Override
    public RowTopK forwardRowsTopK(int[] tokens, int pos0, int k, int[] outputRows) {
        return forwardRowsTopK(tokens, pos0, k, outputRows, null);
    }

    /**
     * Top candidates of every requested row on the device after the per-row prefilter (the logit bias, penalties
     * and DRY); only the candidates are read back.
     */
    @Override
    public RowTopK forwardRowsTopK(int[] tokens, int pos0, int k, int[] outputRows, LogitPrefilter prefilter) {
        Objects.requireNonNull(tokens, "tokens");
        if (k < 1) {
            throw new IllegalArgumentException("top-k readback of " + k + " candidates");
        }
        int[] rows = outputRows != null ? outputRows : identity(tokens.length);
        GpuPrefillDriver.Result r = causalTopK(tokens, pos0, rows, k, prefilter);
        return RowTopK.packed(k, rows.length, r.topIds(), r.topLogits());
    }

    /** Top candidates of the decoded row on the device; only the candidates are read back. */
    @Override
    public RowTopK decodeTopKOnDevice(int token, int pos, int k) {
        return decodeTopKOnDevice(token, pos, k, null);
    }

    /** Top candidates of the decoded row on the device after the row's prefilter. */
    @Override
    public RowTopK decodeTopKOnDevice(int token, int pos, int k, LogitPrefilter prefilter) {
        if (k < 1) {
            throw new IllegalArgumentException("top-k readback of " + k + " candidates");
        }
        GpuPrefillDriver.Result r = causalTopK(new int[]{token}, pos, new int[]{0}, k, prefilter);
        return RowTopK.packed(k, 1, r.topIds(), r.topLogits());
    }

    private static int[] identity(int n) {
        int[] rows = new int[n];
        for (int i = 0; i < n; i++) {
            rows[i] = i;
        }
        return rows;
    }

    /**
     * Rolls the memory back to {@code pos0} when the call re-evaluates stored positions. Checks the context length
     * first, so a call that cannot fit leaves the memory untouched.
     */
    private void align(GpuMemory memory, int pos0, int nTokens) {
        if (pos0 < 0) {
            throw new IllegalArgumentException("position " + pos0 + " is negative");
        }
        int nCtx = memory.plan().nCtx();
        long end = (long) pos0 + nTokens;
        if (end > nCtx) {
            throw new ContextFullException((int) Math.min(Integer.MAX_VALUE, end), nCtx);
        }
        int past = memory.sequence().nPast();
        if (pos0 > past) {
            throw new IllegalArgumentException("position " + pos0 + " leaves a gap after the " + past
                    + " stored positions");
        }
        if (pos0 < past) {
            int drop = past - pos0;
            if (!memory.canRollback(drop)) {
                throw new IllegalStateException("the call starts at " + pos0 + " but the memory holds " + past
                        + " positions and cannot drop the last " + drop);
            }
            rollbackKeepingReplay(memory, drop);
        }
    }

    /**
     * Rolls the memory back and keeps the decode replay when the rollback was host bookkeeping only and the
     * speculative round's replay is on ({@link #CHAIN_REPLAY_PROPERTY}): an uncompressed plan's rollback drops
     * positions from the step inputs and truncates the sequence, which the captured trunk picks up through its
     * per-replay arguments, and the replay's continuity guard is re-armed at the rewind's position. A
     * compressed plan restored its streams' planes on the device, so its trunk captures again.
     */
    private void rollbackKeepingReplay(GpuMemory memory, int drop) {
        int nextPos = memory.sequence().nPast() - drop;
        memory.rollback(drop);
        if (memory.plan().hasCompressed() || !chainReplay) {
            driver.invalidateReplay();
        } else {
            driver.rearmDecodeReplay(nextPos);
        }
    }

    // ================================================================== memoryless calls

    /**
     * @throws UnsupportedOperationException for a model with memory (its attention is causal)
     * @throws IllegalArgumentException      for more tokens than one ubatch
     */
    @Override
    public float[][] forwardAll(int[] tokens, int[] outputRows) {
        ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        if (ctx.hasMemory()) {
            throw new UnsupportedOperationException(ctx.spec().arch().ggufName() + " is planned with memory; "
                    + "non-causal evaluation needs a model without memory (masked diffusion, encoders)");
        }
        return driver.run(tokens, null, 0, outputRows, GpuPrefillDriver.Output.LOGITS, false, tap).rows();
    }

    /**
     * @throws IllegalStateException    when a model with memory holds positions (they would be overwritten)
     * @throws IllegalArgumentException for a model without memory and more tokens than one ubatch
     */
    @Override
    public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
        ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        if (!ctx.hasMemory()) {
            return driver.run(tokens, tokenTypes, 0, null, GpuPrefillDriver.Output.HIDDEN, false, tap).rows();
        }
        GpuMemory memory = ctx.memory();
        int past = memory.sequence().nPast();
        if (past != 0) {
            throw new IllegalStateException("hidden rows of a model with memory need an empty memory, it holds "
                    + past + " positions; reset() first");
        }
        try {
            return driver.run(tokens, tokenTypes, 0, null, GpuPrefillDriver.Output.HIDDEN, true, tap).rows();
        } finally {
            memory.reset();
            driver.invalidateReplay();
        }
    }

    // ================================================================== memory

    @Override
    public void reset() {
        ensureOpen();
        if (ctx.hasMemory()) {
            ctx.memory().reset();
        }
        driver.distrustReplay();
    }

    @Override
    public boolean canRollback(int nTokens) {
        ensureOpen();
        if (!ctx.hasMemory()) {
            if (nTokens < 0) {
                throw new IllegalArgumentException("rollback count " + nTokens + " is negative");
            }
            return nTokens == 0;
        }
        return ctx.memory().canRollback(nTokens);
    }

    @Override
    public void rollback(int nTokens) {
        ensureOpen();
        if (!ctx.hasMemory()) {
            if (nTokens != 0) {
                throw new IllegalStateException("a model without memory holds no positions to roll back");
            }
            return;
        }
        int nextPos = ctx.memory().sequence().nPast() - nTokens;
        ctx.memory().rollback(nTokens);
        if (ctx.memory().plan().hasCompressed() || !chainReplay) {
            // a compressed plan's rollback restored the compressed streams' planes on the device; the next
            // decode captures again. Without the speculative round's replay (CHAIN_REPLAY_PROPERTY, off on
            // this backend) every rollback drops the capture, as before.
            driver.invalidateReplay();
        } else {
            // an uncompressed plan's rollback is host bookkeeping only; the kept capture serves the next
            // round's steps from the rewound position on
            driver.rearmDecodeReplay(nextPos);
        }
    }

    /**
     * Moves the stored rows ({@link DeviceModelMemory#shiftPositions}) and lets every KV mixer re-rotate its keys.
     * Models and cache mixers that cannot re-rotate their keys ({@link ModelSpec#contextShiftRefusal()},
     * {@link GpuLayerExecutor#shiftRefusal}) are refused before anything moves; the memory itself refuses recurrent
     * state and sparse side caches before moving rows. A mixer failure after the rows moved leaves the
     * cache inconsistent, so the memory is reset before the failure propagates (an unsupported shift keeps its
     * {@link UnsupportedOperationException} type).
     *
     * @throws UnsupportedOperationException for models whose positions cannot shift (step35, several positions per
     *                                       token, llama-kv-cache.cpp:1213-1222), recurrent or sparse memory, and a
     *                                       model without memory
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        ensureOpen();
        String refusal = contextShiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        GpuMemory memory = requireMemory("context shift");
        ((DeviceModelMemory) memory).shiftPositions(nKeep, nDiscard);
        driver.invalidateReplay();
        try {
            executor.shiftContext(nKeep, nDiscard);
        } catch (RuntimeException e) {
            try {
                memory.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            String arch = ctx.spec().arch().ggufName();
            String msg = arch + ": context shift failed after the cache rows moved; the memory was reset";
            if (e instanceof UnsupportedOperationException) {
                throw new UnsupportedOperationException(msg, e);
            }
            throw new IllegalStateException(msg, e);
        }
    }

    /**
     * Why {@link #shiftContext} would refuse, or null when it can run: a model llama.cpp can shift
     * ({@link ModelSpec#contextShiftRefusal()}), a device-owned memory, and mixers whose cache geometry the shift
     * kernels address ({@link GpuLayerExecutor#shiftRefusal}). The tensor-parallel backend asks it before it moves
     * any per-rank cache.
     */
    public String contextShiftRefusal() {
        ensureOpen();
        String arch = ctx.spec().arch().ggufName();
        String refusal = ctx.spec().contextShiftRefusal();
        if (refusal != null) {
            return arch + ": context shift is not supported (" + refusal + ")";
        }
        if (!ctx.hasMemory() || !(ctx.memory() instanceof DeviceModelMemory)) {
            return arch + ": context shift needs a device-owned memory, got "
                    + (ctx.hasMemory() ? ctx.memory().getClass().getSimpleName() : "none");
        }
        refusal = executor.shiftRefusal();
        return refusal != null ? arch + ": context shift is not supported (" + refusal + ")" : null;
    }

    /**
     * Why the per-sequence context shift over a paged memory ({@link #shiftSequence}) would refuse, or null when
     * it can run: a model llama.cpp can shift, a paged device memory whose sequence view shifts (a DeepSeek-V4
     * compressor refuses by name), and mixers whose cache geometry the shift kernels address. The engine asks
     * once, at construction.
     */
    @Override
    public String shiftSequenceRefusal() {
        ensureOpen();
        String refusal = ctx.spec().contextShiftRefusal();
        if (refusal != null) {
            return refusal;
        }
        if (!(ctx.hasMemory() && ctx.memory() instanceof DeviceModelMemory)) {
            return "the per-sequence shift needs a device-owned memory";
        }
        DeviceModelMemory device = (DeviceModelMemory) ctx.memory();
        if (device.pagedSequences() == null) {
            return "the per-sequence shift needs paged memory (neuralfree.parallel > 1 or"
                    + " neuralfree.kv.paged=true)";
        }
        refusal = device.sequences().shiftRefusal();
        if (refusal != null) {
            return refusal;
        }
        return executor.shiftRefusal();
    }

    /**
     * The per-sequence context shift over the paged memory (llama.cpp server's slot shift): the sequence view
     * drops the positions and moves the later ones, and the memory runs the move's page copies at once. The mixers
     * then re-rotate the moved keys by the move's per-row deltas, while the rows still sit at the cells the deltas
     * name, and {@link DeviceModelMemory#moveShiftedRows} carries the rotated keys to their new cells before the
     * next step. Refused by {@link #shiftSequenceRefusal} before anything moves; a failure after the bookkeeping
     * leaves the sequence's rows to its slot's retirement (the engine retires the sequence alone — llama.cpp's
     * server refuses a shift it cannot serve the same way).
     *
     * @throws UnsupportedOperationException carrying {@link #shiftSequenceRefusal()} or the sequence view's
     *                                       refusal (a DeepSeek-V4 compressor beyond its rollback snapshots)
     * @throws IllegalArgumentException      for {@code nKeep < 0}, {@code nDiscard <= 0} or a sequence outside the
     *                                       memory
     */
    @Override
    public void shiftSequence(int seq, int nKeep, int nDiscard) {
        ensureOpen();
        String refusal = shiftSequenceRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("context shift unsupported: " + refusal);
        }
        DeviceModelMemory device = (DeviceModelMemory) requireMemory("shiftSequence");
        device.shiftSequence(seq, nKeep, nDiscard);
        try {
            rotateShiftedKeys(nKeep, nDiscard);
        } finally {
            device.moveShiftedRows();
        }
    }

    /**
     * Re-rotates every mixer's stored keys by the per-row deltas the memory's last shift installed, while the rows
     * still sit at the cells those deltas name. A tensor split calls it after it applied the model device's shift to
     * every rank cache, so the split mixers re-rotate their rank's keys by the same deltas; the caller then moves the
     * rows ({@link DeviceModelMemory#moveShiftedRows}).
     *
     * @throws UnsupportedOperationException when a mixer cannot re-rotate its cache
     */
    public void rotateShiftedKeys(int nKeep, int nDiscard) {
        ensureOpen();
        driver.invalidateReplay();
        executor.shiftContext(nKeep, nDiscard);
    }

    /** Why the position edits would refuse: the same requirements as a per-sequence context shift. */
    @Override
    public String sequenceEditRefusal() {
        ensureOpen();
        String refusal = shiftSequenceRefusal();
        if (refusal != null) {
            return refusal;
        }
        return ((DeviceModelMemory) ctx.memory()).sequences().divRefusal();
    }

    /**
     * The {@code seq_add} over the paged memory ({@code DeviceModelMemory#addSequence}): the sequence view moves
     * the positions, the memory runs the edit's page copies, and the mixers re-rotate the moved keys by the
     * installed per-row deltas before {@link DeviceModelMemory#moveShiftedRows} carries them to their new cells.
     */
    @Override
    public void addSequence(int seq, int p0, int p1, int delta) {
        ensureOpen();
        String refusal = sequenceEditRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("position shift unsupported: " + refusal);
        }
        applySequenceEdit(device -> device.addSequence(seq, p0, p1, delta));
    }

    /**
     * The {@code seq_div} over the paged memory ({@code DeviceModelMemory#divSequence}): every moved key
     * re-rotates by its own shift, the rows keep their cells.
     */
    @Override
    public void divSequence(int seq, int p0, int p1, int d) {
        ensureOpen();
        String refusal = sequenceEditRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("position division unsupported: " + refusal);
        }
        applySequenceEdit(device -> device.divSequence(seq, p0, p1, d));
    }

    /**
     * The KV chunk reuse over the paged memory ({@code DeviceModelMemory#reuseSequence}): the re-append's page
     * copies run on the device, then the re-appended keys re-rotate by the run's uniform delta.
     */
    @Override
    public void reuseChunk(int seq, int[][] srcPages, int srcRow0, int pos0, int rows, int delta,
                           int[] tokens, int embeddingToken) {
        ensureOpen();
        String refusal = sequenceEditRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("chunk reuse unsupported: " + refusal);
        }
        applySequenceEdit(device -> device.reuseSequence(seq, srcPages, srcRow0, pos0, rows, delta, tokens,
                embeddingToken));
    }

    /** Runs one sequence edit on the device, re-rotates the keys, then moves the rows; always moves. */
    private void applySequenceEdit(java.util.function.Consumer<DeviceModelMemory> edit) {
        DeviceModelMemory device = (DeviceModelMemory) requireMemory("sequence edit");
        edit.accept(device);
        try {
            driver.invalidateReplay();
            executor.applyShiftDeltas();
        } finally {
            device.moveShiftedRows();
        }
    }

    /**
     * Runs the queued compaction of the tree verify's commit now: the memory carries its pending page copies and
     * row moves on the device and drains the queue, so the commit is complete when the call returns. The row moves
     * dispatch directly between captures and the step replay's trunk sees the same dispatch sequence afterwards,
     * so a single-ubatch tree verify step's capture survives the compaction ({@code GpuPrefillDriver.runBatch}).
     */
    @Override
    public void carryRowMoves() {
        ensureOpen();
        ModelMemory m = ctx.hasMemory() ? ctx.memory() : null;
        if (m instanceof DeviceModelMemory) {
            ((DeviceModelMemory) m).carryRowMoves();
        }
    }

    /** The context's memory, or null for a model without memory. */
    @Override
    public ModelMemory memory() {
        return ctx.hasMemory() ? ctx.memory() : null;
    }

    /** The context this backend runs: its queue, scratch, weights and memory. */
    public GpuForwardContext context() {
        return ctx;
    }

    /**
     * Installs the per-step mirror the tensor-parallel group uses to keep its per-rank caches in step with this
     * backend's main memory, or null to remove it.
     */
    public void setStepMirror(GpuPrefillDriver.StepMirror mirror) {
        driver.stepMirror(mirror);
    }

    /**
     * Keeps decode steps out of a captured graph, for a split whose blocks enqueue or synchronize on peer queues
     * between the main queue's dispatches.
     */
    public void disableDecodeReplay() {
        driver.disableDecodeReplay();
    }

    @Override
    public void setHiddenTap(HiddenTap tap) {
        Objects.requireNonNull(tap, "tap");
        ensureOpen();
        this.tap = tap;
        driver.invalidateReplay();
    }

    private GpuMemory requireMemory(String what) {
        if (!ctx.hasMemory()) {
            throw new IllegalStateException(ctx.spec().arch().ggufName() + " is planned without memory; " + what
                    + " needs one (use forwardAll or hiddenRows)");
        }
        return ctx.memory();
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("GPU forward backend is closed");
        }
    }

    // ================================================================== lifecycle

    /** Waits for the queue, then closes the bound blocks and head preprocessors. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException first = null;
        try {
            driver.closeReplay();
        } catch (RuntimeException e) {
            first = e;
        }
        try {
            ctx.queue().finish();
        } catch (RuntimeException e) {
            first = Closeables.chain(first, e);
        }
        first = Closeables.close(executor, first);
        first = Closeables.close(head, first);
        if (first != null) {
            throw first;
        }
    }

    /** The decode-step replay's state, for diagnostics: a captured graph's dispatches or the warmup progress. */
    public String decodeReplayStats() {
        return driver.decodeReplayStats();
    }

    /** The one-token decode replay's counters, for tests. */
    GpuDecodeReplay.Counters decodeReplayCounters() {
        return driver.decodeReplayCounters();
    }

    /** The LLM kernels this backend runs, for tests. */
    LlmKernels kernels() {
        return ctx.kernels();
    }

    /** The kernel names of the one-token decode replay's captured graph, for tests; empty while none is held. */
    Set<String> decodeReplayKernels() {
        return driver.decodeReplayKernels();
    }

    @Override
    public String toString() {
        return "GpuForwardBackend{" + ctx.spec().arch().ggufName() + ", " + executor + ", "
                + (ctx.hasMemory() ? "memory" : "no memory") + '}';
    }
}
