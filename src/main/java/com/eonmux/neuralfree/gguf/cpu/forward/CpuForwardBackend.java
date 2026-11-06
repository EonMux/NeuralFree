package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuBlockRegistry;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuRecurrentTreeMixer;
import com.eonmux.neuralfree.gguf.cpu.head.CpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.cpu.head.CpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.cpu.head.CpuLogitsHead;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EmbeddingRowSupport;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.RecurrentTreeRound;
import com.eonmux.neuralfree.gguf.engine.UBatch;
import com.eonmux.neuralfree.gguf.engine.lora.AdapterRows;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.BatchSplitter;
import com.eonmux.neuralfree.gguf.engine.RopeSectionSupport;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * {@link ForwardBackend} on the CPU, the twin of {@code GpuForwardBackend}: the embedding stage, one
 * {@link CpuLayerExecutor} per trunk slot (looped and hierarchical schedules alias physical weights through
 * {@link LayerSpec#weightLayer()}), and the logits head. The causal calls run the held sequence; {@code forwardBatch}
 * runs cross-sequence ubatches over the paged memory.
 *
 * <p>Per step of at most {@code ubatch} tokens (the scratch's, capped by the memory plan's; split by
 * {@code MemoryPlan.stepTokens} so the trailing tokens a compressor rollback may drop stay in the last step): the
 * embedding stage writes the residual stream and the step inputs ({@code TOKEN_IDS}, {@code POS}), the memory marks
 * the step's rows ({@link CpuMemory#beginStep}), the executors run in slot order (a single token against memory takes
 * the mixers' decode path), the tokens are appended to the memory's sequence, and the head computes the requested
 * rows of that step only.</p>
 *
 * <p>Memory, as the GPU twin: causal calls ({@link #prefill}, {@link #decode}, {@link #forwardRows}) write the
 * context's memory and continue it at its current length; a call starting earlier rolls the tail back first when
 * the memory allows it. Every input, the context length and the rollback are checked before any step runs, so a
 * rejected call leaves the memory unchanged. Models planned without memory (encoders, diffusion, classifiers) serve
 * {@link #forwardAll} and {@link #hiddenRows} in one step of at most one ubatch; {@link #hiddenRows} on a model with
 * memory runs causally on an empty memory and leaves it empty.</p>
 *
 * <p>Hierarchical (HRM) schedules keep {@code zH} (the embedding) and {@code zL} ({@code hrm.z_l_init}); each pass
 * reads {@code zH + zL}, ends with a weightless RMS norm and writes the stream the schedule names; the output is
 * {@code zH} (models/hrm-text.cpp:168-205, build_stack 92-167).</p>
 *
 * <p>The hidden tap sees the residual stream entering each slot (tap {@code L}) and the final stream before the head
 * (tap {@code nLayer}); each wanted tap point receives one {@link HiddenTap#onHidden} per call with every row of the
 * call, in tap order, after the call's last step.</p>
 *
 * <p>The backend owns its executors, head preprocessors and the blocks they bound; the context's weights,
 * scratch and memory belong to whoever built the context. Weights are read in place from their mapping, so
 * there is no separate layer prefetch. One thread at a time.</p>
 */
public final class CpuForwardBackend implements ForwardBackend {

    private static final String TEMP_ROWS = "CpuForwardBackend.rows";
    private static final String TEMP_OUTPUTS = "CpuForwardBackend.outputs";
    private static final String TEMP_ZH = "CpuForwardBackend.zH";
    private static final String TEMP_ZL = "CpuForwardBackend.zL";
    private static final String TEMP_REPEAT_INPUT = "CpuForwardBackend.repeatInput";
    private static final String TEMP_REPEAT_STATE = "CpuForwardBackend.repeatState";

    private final ModelSpec spec;
    private final CpuScratch scratch;
    private final CpuMemory memory;
    private final int ubatch;
    private final int finalWidth;
    private final CpuEmbeddingStage embedding;
    private final CpuLogitsHead head;
    private final List<CpuLayerExecutor> layers;
    private final int[] rowIdx;
    private final int[] outputIdx;
    private final int[] singleToken = new int[1];
    private final float[] zH;
    private final float[] zL;
    private final float[] zLInit;
    /** Repeat schedule: the stream entering the repeated block, and the state carried between its passes. */
    private final float[] repeatInput;
    private final float[] repeatState;
    private final int repeatWidth;
    private final boolean repeatCombine;
    /** Whether a paged ubatch may mix rows of several adapter slots ({@link AdapterRows#mixable}). */
    private final boolean mixesAdapterSlots;
    /** The id an embedding row records in the token record ({@link EmbeddingRowSupport#recordToken}). */
    private final int embeddingRowToken;
    /** The adapter slots of the running paged ubatch's rows, or null when they all run in slot 0. */
    private int[] ubatchSlots;
    /** The open recurrent tree round, or null when no tree verify runs over a recurrent target. */
    private CpuRecurrentTreeRound treeRound;
    private CpuForwardContext ctx;
    private HiddenTap tap;
    private float[] lastLogits;
    private boolean closed;

    /**
     * Builds and binds every block of the trunk and the head.
     *
     * @param ctx               the model context; its memory, when present, is the one this backend writes
     * @param registry          the CPU blocks (the catalog's factories, including {@link BuiltInCpuExtras})
     * @param headPreprocessors candidate head preprocessors; those that apply are bound and owned by the head
     * @throws UnsupportedOperationException when a trunk layer's mixer, FFN or extra has no CPU block
     * @throws IllegalArgumentException      when the spec lacks an embedding stage or head, or the stages disagree
     */
    public CpuForwardBackend(CpuForwardContext ctx, CpuBlockRegistry registry,
                             List<CpuHeadPreprocessor> headPreprocessors) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(registry, "registry");
        Objects.requireNonNull(headPreprocessors, "headPreprocessors");
        this.spec = ctx.spec();
        this.embeddingRowToken = EmbeddingRowSupport.recordToken(spec);
        this.scratch = ctx.scratch();
        this.memory = ctx.hasMemory() ? ctx.memory() : null;
        this.ubatch = memory != null ? Math.min(scratch.ubatch(), memory.plan().ubatch()) : scratch.ubatch();
        if (ubatch < 1) {
            throw new IllegalArgumentException("memory plan ubatch " + ubatch + " is below 1");
        }
        if (spec.schedule().slots() != spec.nLayer()) {
            throw new IllegalArgumentException(spec.arch().ggufName() + ": " + spec.nLayer()
                    + " trunk layers but the schedule has " + spec.schedule().slots() + " slots");
        }
        this.tap = ctx.hiddenTap();
        this.finalWidth = spec.layer(spec.nLayer() - 1).residual().streamWidth();
        this.rowIdx = scratch.intTemp(TEMP_ROWS, scratch.ubatch());
        this.outputIdx = scratch.intTemp(TEMP_OUTPUTS, scratch.ubatch());
        if (spec.schedule().kind() == LayerSchedule.Kind.HRM) {
            int nEmbd = spec.hparams().nEmbd();
            this.zLInit = ctx.weights().floats(TensorNames.HRM_Z_L_INIT);
            if (zLInit.length != nEmbd) {
                throw new IllegalArgumentException(TensorNames.HRM_Z_L_INIT + " has " + zLInit.length
                        + " values, expected n_embd " + nEmbd);
            }
            this.zH = scratch.temp(TEMP_ZH, Math.multiplyExact(scratch.ubatch(), nEmbd));
            this.zL = scratch.temp(TEMP_ZL, Math.multiplyExact(scratch.ubatch(), nEmbd));
        } else {
            this.zLInit = null;
            this.zH = null;
            this.zL = null;
        }
        LayerSchedule schedule = spec.schedule();
        this.repeatCombine = schedule.kind() == LayerSchedule.Kind.REPEAT
                && schedule.loopCombine() == LayerSchedule.LoopCombine.JUSTNORM;
        if (repeatCombine) {
            this.repeatWidth = spec.layer(0).residual().streamWidth();
            int len = Math.multiplyExact(scratch.ubatch(), repeatWidth);
            this.repeatInput = scratch.temp(TEMP_REPEAT_INPUT, len);
            this.repeatState = scratch.temp(TEMP_REPEAT_STATE, len);
        } else {
            this.repeatWidth = 0;
            this.repeatInput = null;
            this.repeatState = null;
        }
        this.embedding = new CpuEmbeddingStage(ctx);
        this.layers = bindTrunk(registry, ctx, spec);
        this.mixesAdapterSlots = AdapterRows.mixable(spec) && (memory == null || !memory.plan().hasRecurrentState());
        CpuLogitsHead h;
        try {
            h = new CpuLogitsHead(ctx, headPreprocessors);
        } catch (RuntimeException e) {
            closeInto(layers, e);
            throw e;
        }
        this.head = h;
    }

    private static List<CpuLayerExecutor> bindTrunk(CpuBlockRegistry registry, CpuForwardContext ctx,
                                                    ModelSpec spec) {
        List<CpuLayerExecutor> list = new ArrayList<>(spec.nLayer());
        try {
            for (LayerSpec l : spec.layers()) {
                list.add(new CpuLayerExecutor(registry, ctx, l));
            }
        } catch (RuntimeException e) {
            closeInto(list, e);
            throw e;
        }
        return Collections.unmodifiableList(list);
    }

    private static void closeInto(List<CpuLayerExecutor> executors, RuntimeException sink) {
        for (CpuLayerExecutor e : executors) {
            try {
                e.close();
            } catch (RuntimeException failure) {
                sink.addSuppressed(failure);
            }
        }
    }

    // ------------------------------------------------------------------ causal calls

    @Override
    public float[] prefill(int[] tokens, int pos0) {
        checkOpen();
        CpuMemory mem = requireMemory("prefill");
        prepareCausal(mem, tokens, pos0, true);
        TapRows taps = TapRows.open(tap, tokens.length, spec.nLayer());
        float[] logits = null;
        for (int off = 0; off < tokens.length; ) {
            int n = mem.plan().stepTokens(tokens.length - off, ubatch);
            step(tokens, null, off, n, pos0 + off, mem, taps);
            off += n;
            if (off == tokens.length) {
                logits = lastRowLogits(n);
            }
        }
        deliver(taps);
        return logits;
    }

    /**
     * Evaluates precomputed embedding rows at positions {@code pos0 ..} (causal), storing them in memory like
     * {@link #prefill}: each row replaces the token-table lookup and every other input-stage step runs as for
     * tokens (see {@code CpuEmbeddingStage#embedRows}).
     *
     * @return logits of the last row
     * @throws IllegalArgumentException      when a row is null or another width than the model's
     *                                       {@code n_embd} or input width, or a position is out of range
     * @throws UnsupportedOperationException when the model has no LM head or cannot take embedding rows
     *                                       ({@link #embeddingRowsRefusal()})
     */
    @Override
    public float[] prefillEmbeddings(float[][] rows, int pos0) {
        checkOpen();
        CpuMemory mem = requireMemory("prefillEmbeddings");
        String refusal = embeddingRowsRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        prepareRows(mem, rows, pos0);
        TapRows taps = TapRows.open(tap, rows.length, spec.nLayer());
        float[] logits = null;
        for (int off = 0; off < rows.length; ) {
            int n = mem.plan().stepTokens(rows.length - off, ubatch);
            stepRows(rows, off, n, pos0 + off, mem, taps);
            off += n;
            if (off == rows.length) {
                logits = lastRowLogits(n);
            }
        }
        deliver(taps);
        return logits;
    }

    @Override
    public float[] decode(int token, int pos) {
        checkOpen();
        CpuMemory mem = requireMemory("decode");
        singleToken[0] = token;
        prepareCausal(mem, singleToken, pos, true);
        TapRows taps = TapRows.open(tap, 1, spec.nLayer());
        step(singleToken, null, 0, 1, pos, mem, taps);
        float[] logits = lastRowLogits(1);
        deliver(taps);
        return logits;
    }

    /** {@link #decode} with the greedy choice taken on the scratch logits; no logits row is copied out. */
    @Override
    public int decodeGreedyOnDevice(int token, int pos) {
        checkOpen();
        CpuMemory mem = requireMemory("decode");
        singleToken[0] = token;
        prepareCausal(mem, singleToken, pos, true);
        TapRows taps = TapRows.open(tap, 1, spec.nLayer());
        step(singleToken, null, 0, 1, pos, mem, taps);
        rowIdx[0] = 0;
        head.hidden(scratch.slot(CpuScratch.X), finalWidth, rowIdx, 1);
        int id;
        if (head.certifiesArgmax()) {
            id = head.argmaxCertified(1)[0];
        } else {
            head.logits(1);
            id = head.argmax(0);
        }
        deliver(taps);
        return id;
    }

    /**
     * {@link #forwardRows} with greedy selection per requested row; when the head certifies argmax
     * ({@link CpuLogitsHead#certifiesArgmax()}), the ids come from the certified lane and no logits row
     * is materialized, otherwise the default host selection over the full rows.
     */
    @Override
    public int[] forwardRowsGreedy(int[] tokens, int pos0, int[] outputRows) {
        if (!head.certifiesArgmax()) {
            return ForwardBackend.super.forwardRowsGreedy(tokens, pos0, outputRows);
        }
        checkOpen();
        CpuMemory mem = requireMemory("forwardRowsGreedy");
        prepareCausal(mem, tokens, pos0, true);
        int[] rows = rowsOrAll(outputRows, tokens.length);
        TapRows taps = TapRows.open(tap, tokens.length, spec.nLayer());
        int[] ids = new int[rows.length];
        for (int off = 0; off < tokens.length; ) {
            int n = mem.plan().stepTokens(tokens.length - off, ubatch);
            step(tokens, null, off, n, pos0 + off, mem, taps);
            emitIds(rows, off, n, ids);
            off += n;
        }
        deliver(taps);
        return ids;
    }

    @Override
    public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
        checkOpen();
        CpuMemory mem = requireMemory("forwardRows");
        prepareCausal(mem, tokens, pos0, true);
        int[] rows = rowsOrAll(outputRows, tokens.length);
        TapRows taps = TapRows.open(tap, tokens.length, spec.nLayer());
        float[][] out = new float[rows.length][];
        for (int off = 0; off < tokens.length; ) {
            int n = mem.plan().stepTokens(tokens.length - off, ubatch);
            step(tokens, null, off, n, pos0 + off, mem, taps);
            emitRows(rows, off, n, out, true);
            off += n;
        }
        deliver(taps);
        return out;
    }

    /**
     * The heap backend runs batches: a multi-sequence batch over the paged memory, a single-sequence batch over the
     * plain contiguous memory too (the tree flow's masked verify step and chain steps).
     */
    @Override
    public String batchRefusal() {
        return null;
    }

    /**
     * Evaluates the batch over the paged memory: the rows are split into ubatches by the memory's form (llama.cpp
     * {@code split_equal} over a recurrent or hybrid memory — every ubatch takes as many tokens of every sequence
     * as fit, one segment per sequence — and {@code split_seq} over a KV-only memory or an every-rows-output batch),
     * each ubatch reserves its pages all-or-nothing, runs the trunk through the page tables
     * ({@code PagedSequences.prepare} -> the layer steps -> {@code PagedSequences.commit}), and the
     * requested rows' logits come back in batch order. The tokens of a committed step land in each segment
     * sequence's token record through {@code PagedSequences.appendTokens}.
     *
     * <p>Embedding batches (media rows) run the same steps with the embedding stage writing the caller's rows:
     * the rows take their explicit positions (a media chunk's rows can share one position) and, when the batch
     * carries them, its explicit RoPE sections ({@code ForwardBatch.ropePositions}). A non-causal batch
     * ({@code ForwardBatch.nonCausal()}) flags its ubatches so the attention runs without the causal mask, as
     * llama.cpp decodes a media chunk with {@code llama_set_causal_attn(false)}; the flag covers the ubatch only.
     * Positions and sections of token batches flow to the rope and mask exactly as embedding batches' do. A batch
     * may mix token and embedding rows; each ubatch writes every row by its kind.
     *
     * <p>Rows of several adapter slots ({@code ForwardBatch.adapterSlots}, the per-request adapters a
     * {@code LoraRegistry} opened) share the batch: each ubatch sets the slot of its rows in the projection deltas'
     * {@link AdapterRows} table, so every product adds the deltas of each row's slot. A model whose blocks run rows
     * of one product in several pieces ({@link AdapterRows#mixable} false) splits the batch into slot-uniform
     * ubatches.</p>
     *
     * @throws UnsupportedOperationException when the memory is not paged ({@code PagedSequences} null), a row runs
     *                                       in an adapter slot that holds no adapter, or the batch carries embedding
     *                                       rows the model cannot take ({@link #embeddingRowsRefusal()})
     * @throws IllegalArgumentException      when the batch's embedding width or rope section count does not fit the
     *                                       model
     * @throws MemoryFullException           (as {@link ContextFullException}) when a ubatch cannot reserve its pages
     */
    @Override
    public float[][] forwardBatch(ForwardBatch batch) {
        checkOpen();
        if (memory == null) {
            return runMemorylessBatch(batch, true);
        }
        return runPaged(batch, true);
    }

    /**
     * {@link #forwardBatch} reading the output rows' final hidden states (after the output norm) instead of their
     * logits: the pooling, classifier and embedding heads over a batched step.
     */
    @Override
    public float[][] hiddenBatch(ForwardBatch batch) {
        checkOpen();
        if (memory == null) {
            return runMemorylessBatch(batch, false);
        }
        return runPaged(batch, false);
    }

    /**
     * The whole {@code batch} as one memoryless step (a model planned without memory: an encoder or a decision
     * graph that keeps no cache, so every row attends every row): per-row tokens and embedding rows at the batch's
     * positions, one output row per requested row. The batch must hold one sequence and at most one ubatch.
     */
    private float[][] runMemorylessBatch(ForwardBatch batch, boolean logits) {
        Objects.requireNonNull(batch, "batch");
        if (logits && !head.hasLmHead()) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " has no LM head; only hidden rows"
                    + " can be read");
        }
        int n = batch.nTokens();
        if (n > scratch.ubatch()) {
            throw new IllegalArgumentException("the memoryless batch evaluates all " + n + " rows in one step,"
                    + " more than the ubatch " + scratch.ubatch() + "; raise neuralfree.ubatch");
        }
        int[] seqIds = batch.seqIds();
        for (int t = 0; t < seqIds.length; t++) {
            if (seqIds[t] != 0) {
                throw new UnsupportedOperationException("a memoryless batch holds one sequence, row " + t
                        + " runs in sequence " + seqIds[t] + ": " + batchRefusal());
            }
        }
        float[][] embd = batch.embeddings();
        checkBatchRows(batch, embd);
        int[] outputs = batch.outputRows();
        TapRows taps = TapRows.open(tap, n, spec.nLayer());
        float[] x = scratch.slot(CpuScratch.X);
        embedding.embedBatched(batch.tokens(), batch.positions(), embd, x);
        runSlots(0, 0, n, x, taps);
        float[][] out = new float[outputs.length][];
        emitRows(outputs, 0, n, out, logits);
        deliver(taps);
        return out;
    }

    /** Null unless a block of the model reads the step's token ids ({@link EmbeddingRowSupport#refusal}). */
    @Override
    public String embeddingRowsRefusal() {
        return EmbeddingRowSupport.refusal(spec);
    }

    /**
     * Why the single-pass tree step cannot run over this model's recurrent layers, or null when every recurrent
     * layer's mixer serves it: the layer's mixer implements {@link CpuRecurrentTreeMixer} and names no refusal
     * (the gated DeltaNet, KDA and Mamba-family mixers do; a Mamba-1 or PLaMo-2 mixer serves its kind).
     */
    @Override
    public String recurrentTreeRefusal() {
        for (CpuLayerExecutor e : layers) {
            if (e.layer().recurrent() == null) {
                continue;
            }
            if (!(e.mixer() instanceof CpuRecurrentTreeMixer)) {
                return "layer " + e.layer().il() + " runs the " + e.layer().mixerKind()
                        + " mixer, whose single-pass tree step is not covered";
            }
            String refusal = ((CpuRecurrentTreeMixer) e.mixer()).treeStepRefusal();
            if (refusal != null) {
                return "layer " + e.layer().il() + " (" + e.layer().mixerKind() + ") refuses the tree step: "
                        + refusal;
            }
        }
        return null;
    }

    @Override
    public RecurrentTreeRound openRecurrentTreeRound(int rows, int realRows) {
        checkOpen();
        String refusal = recurrentTreeRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        if (treeRound != null) {
            throw new IllegalStateException("a recurrent tree round is already open");
        }
        int count = 0;
        int maxWidth = 0;
        int maxConv = 0;
        int maxScan = 0;
        for (CpuLayerExecutor e : layers) {
            RecurrentSpec r = e.layer().recurrent();
            if (r == null) {
                continue;
            }
            count++;
            maxWidth = Math.max(maxWidth, ((CpuRecurrentTreeMixer) e.mixer()).treeRowWidth());
            maxConv = Math.max(maxConv, stateInts(r.convStateFloats(), "conv state of layer " + e.layer().il()));
            maxScan = Math.max(maxScan, stateInts(r.stateFloats(), "scan state of layer " + e.layer().il()));
        }
        int[] ils = new int[count];
        int[] capBase = new int[count];
        int[] widths = new int[count];
        long cap = 0;
        int i = 0;
        for (CpuLayerExecutor e : layers) {
            RecurrentSpec r = e.layer().recurrent();
            if (r == null) {
                continue;
            }
            int width = ((CpuRecurrentTreeMixer) e.mixer()).treeRowWidth();
            ils[i] = e.layer().il();
            capBase[i] = (int) cap;
            widths[i] = width;
            // the layer's capture slice holds one mixer-input row per verify row
            cap += Math.multiplyExact((long) rows, width);
            i++;
        }
        treeRound = new CpuRecurrentTreeRound(this, rows, realRows, maxWidth, maxConv, maxScan, cap, ils, capBase,
                widths);
        return treeRound;
    }

    /** The replay walk of a committing round: every recurrent layer decodes the rows against its committed state. */
    void replayRecurrentTree(CpuRecurrentTreeRound round, int[] replayRows) {
        float[] out = round.replayOutput();
        for (CpuLayerExecutor e : layers) {
            CpuMixer m = e.mixer();
            if (!(m instanceof CpuRecurrentTreeMixer) || e.layer().recurrent() == null) {
                continue;
            }
            int il = e.layer().il();
            for (int r : replayRows) {
                m.decode(0, round.replayInput(il, r), out);
            }
        }
    }

    /** Forgets the closing round; the committed state stays live for the steps that follow. */
    void closeTreeRound(CpuRecurrentTreeRound round) {
        if (treeRound == round) {
            treeRound = null;
        }
    }

    /**
     * The model keeps recurrent state but no round runs over the step's tree mask: the masked forward would
     * advance the committed state per row and leave it after the wrong tokens, so the step fails loud. With a
     * round open, installs the step's normalized mask (padding rows zeroed) into it and the running step.
     */
    private void requireTreeRound(long[] tree) {
        if (tree == null) {
            return;
        }
        if (treeRound == null) {
            if (memory != null && memory.hasRecurrentState()) {
                throw new IllegalStateException("the step carries a tree mask over a model with recurrent state,"
                        + " but no recurrent tree round is open: open one with openRecurrentTreeRound before the"
                        + " masked forward");
            }
            return;
        }
        treeRound.stepTreeMask(tree);
        ctx.stepTreeRound(treeRound);
    }

    private static int stateInts(long floats, String what) {
        if (floats < 0 || floats > Integer.MAX_VALUE) {
            throw new IllegalStateException(what + " of " + floats + " floats does not fit a tree step");
        }
        return (int) floats;
    }

    /**
     * Runs a contiguous single-sequence batch over the non-paged memory: the tree flow's chain steps and masked
     * verify step, which llama.cpp runs through the same ubatch machinery as the paged form. The rows store at
     * their batch positions (a contiguous row holds its position), rope at the batch's explicit rope sections when
     * it carries them (a tree verify's depth positions) and, under the batch's tree mask, see only their ancestors
     * within the step's fresh rows. One step of at most one ubatch.
     */
    private float[][] runContiguousBatch(ForwardBatch batch, boolean logits) {
        if (logits && !head.hasLmHead()) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " has no LM head; only hidden rows"
                    + " can be read");
        }
        int n = batch.nTokens();
        if (n > ubatch) {
            throw new IllegalArgumentException("the contiguous batch evaluates all " + n + " rows in one step,"
                    + " more than the ubatch " + ubatch + "; raise neuralfree.ubatch");
        }
        if (batch.embeddings() != null) {
            throw new UnsupportedOperationException("the contiguous batch runner takes token rows only; the"
                    + " embedding rows of a batched step run over the paged memory");
        }
        if (batch.treeMask() != null && batch.nonCausal()) {
            throw new IllegalArgumentException("a tree mask and a non-causal step are two different mask rules;"
                    + " a tree verify's step runs causal with the mask");
        }
        CpuMemory mem = requireMemory("forwardBatch");
        checkBatchRows(batch, null);
        int[] tokens = batch.tokens();
        int[] pos = batch.positions();
        int[][] sections = batch.ropePositions();
        long[] tree = batch.treeMask();
        int[] outputs = batch.outputRows();
        TapRows taps = TapRows.open(tap, n, spec.nLayer());
        float[] x = scratch.slot(CpuScratch.X);
        try {
            embedding.embedBatched(tokens, pos, sections, x);
            if (tree != null) {
                ctx.stepTreeMask(tree, n);
                requireTreeRound(tree);
            }
            ctx.stepStorePositions(pos);
            mem.beginStep(pos[0], n);
            runSlots(pos[0], 0, n, x, taps);
            mem.sequence().append(tokens, 0, n);
        } finally {
            ctx.stepTreeMask(null, -1);
            ctx.stepTreeRound(null);
            ctx.stepStorePositions(null);
        }
        float[][] out = new float[outputs.length][];
        emitRows(outputs, 0, n, out, logits);
        deliver(taps);
        return out;
    }

    /** The paged batch run behind {@link #forwardBatch} ({@code logits}) and {@link #hiddenBatch}. */
    private float[][] runPaged(ForwardBatch batch, boolean logits) {
        checkOpen();
        Objects.requireNonNull(batch, "batch");
        CpuMemory memInterface = requireMemory("forwardBatch");
        if (!(memInterface instanceof HeapModelMemory)) {
            throw new UnsupportedOperationException("forwardBatch needs heap memory: " + batchRefusal());
        }
        HeapModelMemory mem = (HeapModelMemory) memInterface;
        if (mem.pagedSequences() == null) {
            if (batch.isSingleSequence() && batch.embeddings() == null) {
                // a contiguous single-sequence batch: a tree verify's masked step or a chain step the tree flow
                // runs through the same batch entry
                return runContiguousBatch(batch, logits);
            }
            throw new UnsupportedOperationException("forwardBatch needs paged memory (neuralfree.parallel > 1 or"
                    + " neuralfree.kv.paged=true): " + batchRefusal());
        }
        float[][] embd = batch.embeddings();
        checkBatchRows(batch, embd);
        int n = batch.nTokens();
        TapRows taps = TapRows.open(tap, n, spec.nLayer());
        float[][] out = new float[batch.outputRows().length][];
        com.eonmux.neuralfree.gguf.memory.paged.PagedSequences paged = mem.pagedSequences();
        BatchSplitter.Mode mode = splitMode(mem.plan().hasRecurrentState(), batch.outputRows().length == n);
        BatchSplitter splitter = new BatchSplitter(true);
        boolean slotUniform = batch.adapterSlots() != null && !mixesAdapterSlots;
        try {
            for (UBatch u : splitter.split(batch, ubatch, mode, slotUniform)) {
                float[][] rows = pagedUbatch(batch, embd, u, mem, paged, taps, logits);
                // map each ubatch output row back to its batch order position by identity: the ubatch rows are
                // ordered by sequence then position, which need not ascend in batch order across segments
                int[] localOutputs = u.layout().outputRows();
                int[] ubatchRows = u.batchRows();
                int[] outputs = batch.outputRows();
                for (int j = 0; j < localOutputs.length; j++) {
                    int batchRow = ubatchRows[localOutputs[j]];
                    int i = Arrays.binarySearch(outputs, batchRow);
                    if (i < 0) {
                        throw new IllegalStateException("ubatch output row " + batchRow
                                + " is not a batch output row");
                    }
                    out[i] = rows[j];
                }
            }
        } finally {
            ctx.stepNonCausal(false);
        }
        deliver(taps);
        return out;
    }

    /**
     * Rejects a batch before any ubatch reserves pages: every row runs in slot 0 or a slot that holds an adapter,
     * every embedding row holds n_embd or the model input width on a model that takes embedding rows, and the
     * batch's rope sections match the model's positions per token.
     */
    private void checkBatchRows(ForwardBatch batch, float[][] embd) {
        String slotRefusal = batch.adapterSlotRefusal(getClass().getSimpleName(), ctx.projectionDeltas()::holdsSlot);
        if (slotRefusal != null) {
            throw new UnsupportedOperationException(slotRefusal);
        }
        if (embd != null) {
            String refusal = embeddingRowsRefusal();
            if (refusal != null) {
                throw new UnsupportedOperationException(refusal);
            }
            int width = embedding.inputWidth();
            for (int i = 0; i < embd.length; i++) {
                if (embd[i] != null && !embedding.acceptsRowWidth(embd[i].length)) {
                    throw new IllegalArgumentException("embedding row " + i + " holds " + embd[i].length
                            + " values, the model input is " + width + " wide (n_embd " + spec.embedding().nEmbd()
                            + ")");
                }
            }
        }
        RopeSectionSupport.check(spec.hparams().ropeType(), batch);
    }

    /**
     * How a paged batch splits into ubatches, by the memory form as llama.cpp splits it: a KV-only memory mixes
     * the sequences in each ubatch (split_simple, llama-kv-cache.cpp:761), so concurrent decodes share one trunk
     * pass; a recurrent or hybrid memory takes as many tokens of every sequence as fit, one state update per
     * sequence per ubatch (split_equal, llama-memory-recurrent.cpp:455), or one sequence per ubatch when every
     * row is an output (split_seq).
     */
    static BatchSplitter.Mode splitMode(boolean recurrentState, boolean allOutputs) {
        if (!recurrentState) {
            return BatchSplitter.Mode.SIMPLE;
        }
        return allOutputs ? BatchSplitter.Mode.SEQ : BatchSplitter.Mode.EQUAL;
    }

    /**
     * One paged ubatch: reserve the step, run the pages' copies on the host (the K/V bytes) — the step's own after
     * its reservation, and any still-owed ones from engine-side mutations (an adopt cut, a fork copy) before it —
     * commit the reservation (the page tables and position mirrors finalize, the copied rows' positions report to
     * the device mirror), begin the step, run the trunk over the page tables, trim the sliding windows to the next
     * queries ({@code PagedSequences.finish}) and end the step. Commit runs before the trunk so the trunk's queries
     * see the copied rows' positions; the copies' source references release after commit has registered them. The hidden tap stores each of the step's rows at its call row
     * ({@code u.batchRows()}). The ubatch's rows take their explicit batch positions (a media chunk's rows share
     * the chunk's base position), its media rows are written by the embedding stage, and a non-causal batch flags
     * the step for the mixers.
     */
    private float[][] pagedUbatch(ForwardBatch batch, float[][] embd, UBatch u, HeapModelMemory mem,
                                  com.eonmux.neuralfree.gguf.memory.paged.PagedSequences paged, TapRows taps,
                                  boolean logits) {
        int[] tokens = u.tokens();
        PagedStep step = mem.openPagedStep(u.layout(), u.positions());
        boolean nonCausal = batch.nonCausal();
        long[] treeBits = batch.treeMask();
        long[] ubatchTree = treeBits == null ? null : gather(treeBits, u.batchRows());
        AdapterRows adapterRows = ctx.projectionDeltas().rows();
        ubatchSlots = AdapterRows.ofRows(batch.adapterSlots(), u.batchRows());
        adapterRows.set(ubatchSlots);
        try {
            float[] x = scratch.slot(CpuScratch.X);
            embedUbatch(batch, embd, u, tokens, x);
            if (nonCausal) {
                ctx.stepNonCausal(true);
            }
            if (ubatchTree != null) {
                ctx.stepTreeMask(ubatchTree, u.batchRows().length);
                requireTreeRound(ubatchTree);
            }
            runSlotsBatched(u.layout(), x, taps, u.batchRows());
            appendRows(paged, step, tokens, u.positions(), embeddingRowToken);
            paged.finish(step);
            float[][] rows = new float[u.layout().outputRows().length][];
            emitRows(u.layout().outputRows(), 0, tokens.length, rows, logits);
            return rows;
        } finally {
            if (nonCausal) {
                ctx.stepNonCausal(false);
            }
            if (ubatchTree != null) {
                ctx.stepTreeMask(null, -1);
                ctx.stepTreeRound(null);
            }
            adapterRows.clear();
            ubatchSlots = null;
            mem.endPagedStep();
        }
    }

    /** The {@code rows}' entries named by {@code at}, in {@code at}'s order. */
    private static long[] gather(long[] rows, int[] at) {
        long[] out = new long[at.length];
        for (int i = 0; i < at.length; i++) {
            out[i] = rows[at[i]];
        }
        return out;
    }

    /**
     * Extends each of the step's sequences' token record with its segment's rows (ubatch row order): their ids
     * ({@code embeddingToken} for an embedding row) and stored positions.
     */
    private static void appendRows(com.eonmux.neuralfree.gguf.memory.paged.PagedSequences paged, PagedStep step,
                                   int[] tokens, int[] positions, int embeddingToken) {
        for (int s = 0; s < step.nSeg(); s++) {
            paged.appendRows(step.segSeq(s), tokens, positions, step.segStart(s), step.segLen(s), embeddingToken);
        }
    }

    /**
     * Writes the ubatch's rows into the residual stream: media rows through the embedding stage's media path (with
     * the batch's explicit rope sections when it carries them), token rows through the batched lookup (with the
     * sections too, filling the four-section position slot the mixers read), and a ubatch holding both through the
     * mixed path, each row by its kind.
     *
     * @param embd the batch's embedding rows ({@code ForwardBatch.embeddings()}), or null for a token batch
     */
    private void embedUbatch(ForwardBatch batch, float[][] embd, UBatch u, int[] tokens, float[] x) {
        int[][] sections = sliceSections(batch.ropePositions(), u);
        if (embd == null) {
            embedding.embedBatched(tokens, u.positions(), sections, x);
            return;
        }
        // the ubatch's rows run in sequence-position order, so the batch's embedding rows are gathered in the
        // ubatch's row order before the stage writes them
        int n = u.nTokens();
        float[][] ubatchRows = new float[n][];
        int embeddingRows = 0;
        for (int r = 0; r < n; r++) {
            ubatchRows[r] = embd[u.batchRow(r)];
            embeddingRows += ubatchRows[r] != null ? 1 : 0;
        }
        if (embeddingRows == 0) {
            embedding.embedBatched(tokens, u.positions(), sections, x);
        } else if (embeddingRows == n) {
            embedding.embedMediaRows(ubatchRows, 0, n, u.positions(), sections, x);
        } else {
            embedding.embedMixedRows(tokens, ubatchRows, u.positions(), sections, x);
        }
    }

    /** The batch's rope sections restricted to the ubatch's rows, or null when the batch carries none. */
    private static int[][] sliceSections(int[][] sections, UBatch u) {
        if (sections == null) {
            return null;
        }
        int[] rows = u.batchRows();
        int[][] out = new int[sections.length][];
        for (int s = 0; s < sections.length; s++) {
            out[s] = new int[rows.length];
            for (int r = 0; r < rows.length; r++) {
                out[s][r] = sections[s][rows[r]];
            }
        }
        return out;
    }

    /**
     * Runs every slot over the batched layout: the embedding stage has filled the token and position slots for the
     * whole ubatch, every layer steps over the layout's segments through the page tables of
     * {@link com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory#currentPagedStep}. {@code callRows} maps each ubatch row
     * to its call row, where the hidden tap stores it.
     */
    private void runSlotsBatched(com.eonmux.neuralfree.gguf.memory.StepLayout layout, float[] x, TapRows taps,
                                 int[] callRows) {
        for (int il = 0; il < layers.size(); il++) {
            CpuLayerExecutor e = layers.get(il);
            repeatBeforeSlot(spec.schedule(), il, layout.nTokens(), x);
            if (taps != null && taps.wants(il)) {
                taps.capture(il, x, callRows, layout.nTokens(), e.streamWidth());
            }
            e.forward(layout, x);
            repeatAfterSlot(spec.schedule(), il, layout.nTokens(), x);
        }
        if (taps != null && taps.wants(layers.size())) {
            taps.capture(layers.size(), x, callRows, layout.nTokens(), finalWidth);
        }
    }

    // ------------------------------------------------------------------ memory-less calls

    /**
     * @throws UnsupportedOperationException for a model with memory (its attention is causal) or without an LM head
     * @throws IllegalArgumentException      for more tokens than one ubatch
     */
    @Override
    public float[][] forwardAll(int[] tokens, int[] outputRows) {
        checkOpen();
        if (memory != null) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " is planned with memory; non-causal"
                    + " evaluation needs a model without memory (masked diffusion, encoders)");
        }
        prepareSingleStep(tokens, null, true, "forwardAll");
        int[] rows = rowsOrAll(outputRows, tokens.length);
        TapRows taps = TapRows.open(tap, tokens.length, spec.nLayer());
        step(tokens, null, 0, tokens.length, 0, null, taps);
        float[][] out = new float[rows.length][];
        emitRows(rows, 0, tokens.length, out, true);
        deliver(taps);
        return out;
    }

    /**
     * @throws IllegalStateException    when a model with memory holds positions (they would be overwritten)
     * @throws IllegalArgumentException for a model without memory and more tokens than one ubatch
     */
    @Override
    public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
        checkOpen();
        if (memory == null) {
            prepareSingleStep(tokens, tokenTypes, false, "hiddenRows");
            TapRows taps = TapRows.open(tap, tokens.length, spec.nLayer());
            step(tokens, tokenTypes, 0, tokens.length, 0, null, taps);
            float[][] out = new float[tokens.length][];
            emitRows(rowsOrAll(null, tokens.length), 0, tokens.length, out, false);
            deliver(taps);
            return out;
        }
        int past = memory.sequence().nPast();
        if (past != 0) {
            throw new IllegalStateException("hidden rows of a model with memory need an empty memory, it holds "
                    + past + " positions; reset() first");
        }
        checkInputs(tokens, tokenTypes, 0, false);
        checkContext(memory, 0, tokens.length);
        int[] rows = rowsOrAll(null, tokens.length);
        TapRows taps = TapRows.open(tap, tokens.length, spec.nLayer());
        float[][] out = new float[tokens.length][];
        try {
            for (int off = 0; off < tokens.length; ) {
                int n = memory.plan().stepTokens(tokens.length - off, ubatch);
                step(tokens, tokenTypes, off, n, off, memory, taps);
                emitRows(rows, off, n, out, false);
                off += n;
            }
        } finally {
            memory.reset();
        }
        deliver(taps);
        return out;
    }

    // ------------------------------------------------------------------ memory

    /** Runs the queued compaction of the heap memory now: its pending page copies, then its pending row moves. */
    @Override
    public void carryRowMoves() {
        checkOpen();
        if (memory instanceof com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory) {
            ((com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory) memory).carryRowMoves();
        }
    }

    @Override
    public void reset() {
        checkOpen();
        if (memory != null) {
            memory.reset();
        }
    }

    @Override
    public boolean canRollback(int nTokens) {
        checkOpen();
        if (memory == null) {
            if (nTokens < 0) {
                throw new IllegalArgumentException("rollback count " + nTokens + " is negative");
            }
            return nTokens == 0;
        }
        return memory.canRollback(nTokens);
    }

    @Override
    public void rollback(int nTokens) {
        checkOpen();
        if (memory == null) {
            if (nTokens != 0) {
                throw new IllegalStateException("a model without memory holds no positions to roll back");
            }
            return;
        }
        memory.rollback(nTokens);
    }

    /**
     * Context shift through {@link HeapModelMemory#shiftPositions} and then every trunk layer that owns a KV cache.
     * Refused before any row moves, as llama.cpp {@code get_can_shift} (llama-kv-cache.cpp:1213-1222), for step35
     * (per-layer RoPE dimensions) and multi-section RoPE ({@code n_pos_per_embd > 1})
     * ({@link ModelSpec#contextShiftRefusal()}), and for a cache mixer that cannot re-rotate its keys
     * ({@link CpuLayerExecutor#shiftRefusal}); refused by the memory for recurrent state and sparse side caches. When
     * a layer still fails after the rows moved, the memory is reset so no stale rotation is ever attended, and the
     * failure is rethrown.
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        checkOpen();
        CpuMemory mem = requireMemory("shiftContext");
        String arch = spec.arch().ggufName();
        String refusal = spec.contextShiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("context shift unsupported for " + arch + ": " + refusal);
        }
        if (!(mem instanceof HeapModelMemory)) {
            throw new UnsupportedOperationException("context shift needs HeapModelMemory, got "
                    + mem.getClass().getSimpleName());
        }
        for (CpuLayerExecutor e : layers) {
            refusal = e.shiftRefusal();
            if (refusal != null) {
                throw new UnsupportedOperationException("context shift unsupported for " + arch + ": " + refusal);
            }
        }
        ((HeapModelMemory) mem).shiftPositions(nKeep, nDiscard);
        try {
            for (CpuLayerExecutor e : layers) {
                e.shiftContext(nKeep, nDiscard);
            }
        } catch (RuntimeException e) {
            try {
                mem.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            String msg = "context shift of " + arch + " failed after the cache rows moved; the memory was reset";
            if (e instanceof UnsupportedOperationException) {
                throw new UnsupportedOperationException(msg, e);
            }
            throw new IllegalStateException(msg, e);
        }
    }

    /** Why {@link #shiftSequence} would refuse, or null: a paged heap memory, a shiftable model and mixers. */
    @Override
    public String shiftSequenceRefusal() {
        checkOpen();
        CpuMemory mem = memory;
        String refusal = spec.contextShiftRefusal();
        if (refusal != null) {
            return refusal;
        }
        if (!(mem instanceof HeapModelMemory) || ((HeapModelMemory) mem).pagedSequences() == null) {
            return "the per-sequence shift needs paged heap memory";
        }
        for (CpuLayerExecutor e : layers) {
            refusal = e.shiftRefusal();
            if (refusal != null) {
                return refusal;
            }
        }
        return null;
    }

    /**
     * The per-sequence context shift over the paged memory ({@code HeapModelMemory#shiftSequence}): the moved
     * keys re-rotate at once, by the move's per-row deltas and while the rows still sit at the cells the shift
     * reports name — the compaction's row moves then carry the rotated keys to their new cells at the next step.
     * Refused by {@link #shiftSequenceRefusal} before any row moves.
     */
    @Override
    public void shiftSequence(int seq, int nKeep, int nDiscard) {
        String refusal = shiftSequenceRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("context shift unsupported: " + refusal);
        }
        HeapModelMemory heap = (HeapModelMemory) requireMemory("shiftSequence");
        heap.shiftSequence(seq, nKeep, nDiscard);
        heap.runPagedEditCopies();
        if (!heap.applyPagedShiftDeltas()) {
            return;
        }
        heap.clearPagedShift();
        for (CpuLayerExecutor e : layers) {
            e.applyShiftDeltas();
        }
    }

    /** Why the position edits would refuse: the paged heap memory, a shiftable model and mixers. */
    @Override
    public String sequenceEditRefusal() {
        return shiftSequenceRefusal();
    }

    /**
     * The {@code seq_add} over the paged memory ({@code HeapModelMemory#addSequence}): the moved keys re-rotate
     * by the edit's per-row deltas at once, while the rows still sit at the cells the edit reports name; a
     * negative move's compaction then carries the rotated keys to their new cells at the next step.
     */
    @Override
    public void addSequence(int seq, int p0, int p1, int delta) {
        String refusal = sequenceEditRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("position shift unsupported: " + refusal);
        }
        applySequenceEdit(heap -> heap.addSequence(seq, p0, p1, delta));
    }

    /** The {@code seq_div} over the paged memory: every row re-rotates by its own shift. */
    @Override
    public void divSequence(int seq, int p0, int p1, int d) {
        String refusal = sequenceEditRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("position division unsupported: " + refusal);
        }
        applySequenceEdit(heap -> heap.divSequence(seq, p0, p1, d));
    }

    /**
     * The KV chunk reuse over the paged memory ({@code HeapModelMemory#reuseSequence}): the re-appended keys
     * re-rotate by the run's uniform delta at once, after the re-append's page copies ran, while the rows still
     * sit at the cells the edit reports name.
     */
    @Override
    public void reuseChunk(int seq, int[][] srcPages, int srcRow0, int pos0, int rows, int delta,
                           int[] tokens, int embeddingToken) {
        String refusal = sequenceEditRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("chunk reuse unsupported: " + refusal);
        }
        applySequenceEdit(heap -> heap.reuseSequence(seq, srcPages, srcRow0, pos0, rows, delta, tokens,
                embeddingToken));
    }

    /** Applies one sequence edit on the heap memory and re-rotates the layers' moved keys by its deltas. */
    private void applySequenceEdit(java.util.function.Consumer<HeapModelMemory> edit) {
        HeapModelMemory heap = (HeapModelMemory) requireMemory("sequence edit");
        edit.accept(heap);
        heap.runPagedEditCopies();
        if (!heap.applyPagedShiftDeltas()) {
            return;
        }
        heap.clearPagedShift();
        for (CpuLayerExecutor e : layers) {
            e.applyShiftDeltas();
        }
    }

    /** The context's memory, or null for a model without memory. */
    @Override
    public ModelMemory memory() {
        checkOpen();
        return memory;
    }

    /** Sequences the paged memory holds ({@code PagedSequences.maxSequences()}); 1 on the contiguous form. */
    @Override
    public int maxSequences() {
        checkOpen();
        return memory != null && memory.pagedSequences() != null ? memory.pagedSequences().maxSequences() : 1;
    }

    /** Installs the tap for every later call; the context is rebuilt with it ({@link CpuForwardContext#withHiddenTap}). */
    @Override
    public void setHiddenTap(HiddenTap tap) {
        Objects.requireNonNull(tap, "tap");
        checkOpen();
        this.tap = tap;
        this.ctx = ctx.withHiddenTap(tap);
    }

    /** Closes the executors and the head. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = new IllegalStateException("closing the CPU forward backend failed");
        closeInto(layers, failure);
        try {
            head.close();
        } catch (RuntimeException e) {
            failure.addSuppressed(e);
        }
        if (failure.getSuppressed().length > 0) {
            throw failure;
        }
    }

    // ------------------------------------------------------------------ steps

    /**
     * One step of {@code n} tokens: the embedding, the memory's step rows when {@code mem} is given, every slot,
     * sequence bookkeeping.
     */
    private void step(int[] tokens, int[] tokenTypes, int off, int n, int pos0, CpuMemory mem, TapRows taps) {
        float[] x = scratch.slot(CpuScratch.X);
        embedding.embed(tokens, tokenTypes, off, n, pos0, x);
        if (mem != null) {
            mem.beginStep(pos0, n);
        }
        runSlots(pos0, off, n, x, taps);
        if (mem != null) {
            mem.sequence().append(tokens, off, n);
        }
    }

    /**
     * One step of {@code n} embedding rows: the row embedding, the memory's step rows, every slot, sequence
     * bookkeeping (the rows record {@link #embeddingRowToken}).
     */
    private void stepRows(float[][] rows, int off, int n, int pos0, CpuMemory mem, TapRows taps) {
        float[] x = scratch.slot(CpuScratch.X);
        embedding.embedRows(rows, off, n, pos0, x);
        mem.beginStep(pos0, n);
        runSlots(pos0, off, n, x, taps);
        mem.sequence().appendEmbeddings(n, embeddingRowToken);
    }

    private void runSlots(int pos0, int off, int n, float[] x, TapRows taps) {
        LayerSchedule schedule = spec.schedule();
        boolean hrm = schedule.kind() == LayerSchedule.Kind.HRM;
        int hrmLen = hrm ? n * zLInit.length : 0;
        if (hrm) {
            System.arraycopy(x, 0, zH, 0, hrmLen);
            for (int r = 0; r < n; r++) {
                System.arraycopy(zLInit, 0, zL, r * zLInit.length, zLInit.length);
            }
        }
        for (int il = 0; il < layers.size(); il++) {
            CpuLayerExecutor e = layers.get(il);
            if (hrm && schedule.passStart(il)) {
                CpuOps.add(zH, 0, zL, 0, x, 0, hrmLen);
            }
            repeatBeforeSlot(schedule, il, n, x);
            if (taps != null && taps.wants(il)) {
                taps.capture(il, x, off, n, e.streamWidth());
            }
            e.forward(pos0, n, x);
            repeatAfterSlot(schedule, il, n, x);
            if (hrm && schedule.passEnd(il)) {
                CpuOps.norm(NormKind.RMS, spec.hparams().normRmsEps(), x, 0, x, 0, n, zLInit.length, null, null);
                float[] dst = schedule.stream(il) == LayerSchedule.Stream.HIGH ? zH : zL;
                System.arraycopy(x, 0, dst, 0, hrmLen);
            }
        }
        if (hrm) {
            System.arraycopy(zH, 0, x, 0, hrmLen);
        }
        if (taps != null && taps.wants(layers.size())) {
            taps.capture(layers.size(), x, off, n, finalWidth);
        }
    }

    /**
     * Repeat schedules with {@link LayerSchedule.LoopCombine#JUSTNORM} before slot {@code il}: the stream entering
     * the repeated block is kept, the state is initialized from it at the first pass, and each pass starts as
     * {@code justNorm(state + input)} (Dory's {@code combined = justnorm(state + input_h)}).
     */
    private void repeatBeforeSlot(LayerSchedule schedule, int il, int n, float[] x) {
        if (!repeatCombine) {
            return;
        }
        int len = n * repeatWidth;
        if (il == 0 && schedule.nInputLayers() == 0) {
            System.arraycopy(x, 0, repeatInput, 0, len);
        }
        if (il == schedule.recurrentStart()) {
            System.arraycopy(repeatInput, 0, repeatState, 0, len);
        }
        if (schedule.loopStart(il)) {
            CpuOps.add(repeatState, 0, repeatInput, 0, x, 0, len);
            CpuOps.norm(NormKind.L2_GDN, schedule.loopEps(), x, 0, x, 0, n, repeatWidth, null, null);
        }
    }

    /**
     * Repeat schedules with {@link LayerSchedule.LoopCombine#JUSTNORM} after slot {@code il}: a pass ends by
     * writing {@code state = justNorm(x)}; the input stream is captured after the last input layer, and the first
     * output layer starts from the carried state.
     */
    private void repeatAfterSlot(LayerSchedule schedule, int il, int n, float[] x) {
        if (!repeatCombine) {
            return;
        }
        int len = n * repeatWidth;
        if (schedule.loopEnd(il)) {
            CpuOps.norm(NormKind.L2_GDN, schedule.loopEps(), x, 0, x, 0, n, repeatWidth, null, null);
            System.arraycopy(x, 0, repeatState, 0, len);
        }
        if (il == schedule.nInputLayers() - 1) {
            System.arraycopy(x, 0, repeatInput, 0, len);
        }
        if (il + 1 == schedule.recurrentEnd() && schedule.nOutputLayers() > 0) {
            System.arraycopy(repeatState, 0, x, 0, len);
        }
    }

    /** Logits of the last row of the step just run, copied to the reused result row. */
    private float[] lastRowLogits(int n) {
        rowIdx[0] = n - 1;
        head.hidden(scratch.slot(CpuScratch.X), finalWidth, rowIdx, 1);
        float[] logits = head.logits(1);
        int vocab = head.vocabRows();
        if (lastLogits == null || lastLogits.length != vocab) {
            lastLogits = new float[vocab];
        }
        System.arraycopy(logits, 0, lastLogits, 0, vocab);
        return lastLogits;
    }

    /**
     * Writes the results of the requested rows that fall in the step {@code [off, off + n)}: logits rows, or hidden
     * rows after the output norm, batched through the head.
     */
    private void emitRows(int[] rows, int off, int n, float[][] out, boolean logits) {
        int max = head.maxRows();
        int count = 0;
        for (int i = 0; i < rows.length; i++) {
            int r = rows[i];
            if (r < off || r >= off + n) {
                continue;
            }
            rowIdx[count] = r - off;
            outputIdx[count] = i;
            count++;
            if (count == max) {
                flush(count, out, logits);
                count = 0;
            }
        }
        if (count > 0) {
            flush(count, out, logits);
        }
    }

    private void flush(int count, float[][] out, boolean logits) {
        AdapterRows adapterRows = ctx.projectionDeltas().rows();
        if (ubatchSlots != null) {
            // the head's products run over the output rows: their slots, in the head's row order
            int[] headSlots = new int[count];
            for (int k = 0; k < count; k++) {
                headSlots[k] = ubatchSlots[rowIdx[k]];
            }
            adapterRows.set(headSlots);
        }
        float[] hidden;
        float[] src;
        int width;
        try {
            hidden = head.hidden(scratch.slot(CpuScratch.X), finalWidth, rowIdx, count);
            width = head.hiddenWidth();
            src = hidden;
            if (logits) {
                src = head.logits(count);
                width = head.vocabRows();
            }
        } finally {
            if (ubatchSlots != null) {
                adapterRows.set(ubatchSlots);
            }
        }
        for (int k = 0; k < count; k++) {
            out[outputIdx[k]] = Arrays.copyOfRange(src, k * width, (k + 1) * width);
        }
    }

    /** The {@link #emitRows} of {@link #forwardRowsGreedy}: the certified ids of the step's requested rows. */
    private void emitIds(int[] rows, int off, int n, int[] ids) {
        int max = head.maxRows();
        int count = 0;
        for (int i = 0; i < rows.length; i++) {
            int r = rows[i];
            if (r < off || r >= off + n) {
                continue;
            }
            rowIdx[count] = r - off;
            outputIdx[count] = i;
            count++;
            if (count == max) {
                flushIds(count, ids);
                count = 0;
            }
        }
        if (count > 0) {
            flushIds(count, ids);
        }
    }

    private void flushIds(int count, int[] ids) {
        AdapterRows adapterRows = ctx.projectionDeltas().rows();
        if (ubatchSlots != null) {
            int[] headSlots = new int[count];
            for (int k = 0; k < count; k++) {
                headSlots[k] = ubatchSlots[rowIdx[k]];
            }
            adapterRows.set(headSlots);
        }
        try {
            head.hidden(scratch.slot(CpuScratch.X), finalWidth, rowIdx, count);
            int[] certified = head.argmaxCertified(count);
            for (int k = 0; k < count; k++) {
                ids[outputIdx[k]] = certified[k];
            }
        } finally {
            if (ubatchSlots != null) {
                adapterRows.set(ubatchSlots);
            }
        }
    }

    private void deliver(TapRows taps) {
        if (taps != null) {
            taps.deliver();
        }
    }

    // ------------------------------------------------------------------ validation

    private CpuMemory requireMemory(String op) {
        if (memory == null) {
            throw new IllegalStateException(spec.arch().ggufName() + " is planned without memory; " + op
                    + " needs one (use forwardAll or hiddenRows)");
        }
        return memory;
    }

    /**
     * Checks a memory-writing call before any work: its inputs, the LM head, the context length, then aligns the
     * memory to {@code pos0} (rolling back the tail when the call re-evaluates stored positions).
     */
    private void prepareCausal(CpuMemory mem, int[] tokens, int pos0, boolean logits) {
        checkInputs(tokens, null, pos0, logits);
        alignMemory(mem, pos0, tokens.length);
    }

    /** {@link #prepareCausal} for a prefill of embedding rows. */
    private void prepareRows(CpuMemory mem, float[][] rows, int pos0) {
        checkRowInputs(rows, pos0);
        alignMemory(mem, pos0, rows.length);
    }

    /** Checks the context length and aligns the memory to {@code pos0} for a call of {@code count} rows. */
    private void alignMemory(CpuMemory mem, int pos0, int count) {
        checkContext(mem, pos0, count);
        int past = mem.sequence().nPast();
        if (pos0 > past) {
            throw new IllegalArgumentException("position " + pos0 + " leaves a gap after the " + past
                    + " stored positions");
        }
        if (pos0 < past) {
            int drop = past - pos0;
            if (!mem.canRollback(drop)) {
                throw new IllegalStateException("the call starts at " + pos0 + " but the memory holds " + past
                        + " positions and cannot drop the last " + drop);
            }
            mem.rollback(drop);
        }
    }

    /** Input checks of an embedding-row call: the rows and the LM head (the positions the stage checks). */
    private void checkRowInputs(float[][] rows, int pos0) {
        Objects.requireNonNull(rows, "rows");
        if (rows.length == 0) {
            throw new IllegalArgumentException("no embedding rows to evaluate");
        }
        if (!head.hasLmHead()) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " has no LM head; only hidden rows"
                    + " can be read");
        }
        embedding.checkRowsInput(rows, pos0);
    }

    /** Checks a memory-less call: its inputs, the LM head when logits are read, and the one-step limit. */
    private void prepareSingleStep(int[] tokens, int[] tokenTypes, boolean logits, String op) {
        checkInputs(tokens, tokenTypes, 0, logits);
        if (tokens.length > scratch.ubatch()) {
            throw new IllegalArgumentException(op + " evaluates all " + tokens.length + " tokens in one step, more"
                    + " than the ubatch " + scratch.ubatch() + "; raise neuralfree.ubatch");
        }
    }

    private void checkInputs(int[] tokens, int[] tokenTypes, int pos0, boolean logits) {
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("no tokens to evaluate");
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("position " + pos0 + " is negative");
        }
        if (logits && !head.hasLmHead()) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " has no LM head; only hidden rows"
                    + " can be read");
        }
        embedding.checkInputs(tokens, tokenTypes, pos0);
    }

    private static void checkContext(CpuMemory mem, int pos0, int n) {
        int nCtx = mem.sequence().nCtx();
        long end = (long) pos0 + n;
        if (end > nCtx) {
            throw new ContextFullException((int) Math.min(end, Integer.MAX_VALUE), nCtx);
        }
    }

    private static int[] rowsOrAll(int[] outputRows, int nTokens) {
        if (outputRows == null) {
            int[] all = new int[nTokens];
            for (int i = 0; i < nTokens; i++) {
                all[i] = i;
            }
            return all;
        }
        for (int r : outputRows) {
            if (r < 0 || r >= nTokens) {
                throw new IllegalArgumentException("output row " + r + " outside [0, " + nTokens + ")");
            }
        }
        return outputRows;
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("CPU forward backend is closed");
        }
    }

    @Override
    public String toString() {
        return "CpuForwardBackend{" + spec.arch().ggufName() + ", slots=" + spec.nLayer() + ", ubatch=" + ubatch
                + (memory != null ? ", memory" : "") + '}';
    }

    // ------------------------------------------------------------------ hidden tap

    /**
     * The rows of the wanted tap points over one call, gathered step by step and delivered once per tap point after
     * the call ({@link HiddenTap}: {@code nRows} is every token of the call, in call order; a batched step stores
     * each of its rows at the call row the ubatch's {@code batchRows} names).
     */
    private static final class TapRows {
        private final HiddenTap tap;
        private final int total;
        private final float[][] rows;
        private final int[] widths;

        private TapRows(HiddenTap tap, int total, int points) {
            this.tap = tap;
            this.total = total;
            this.rows = new float[points][];
            this.widths = new int[points];
        }

        /** Collector of one call over tap points {@code [0, nLayer]}; null when the tap wants none of them. */
        static TapRows open(HiddenTap tap, int total, int nLayer) {
            for (int p = 0; p <= nLayer; p++) {
                if (tap.wants(p)) {
                    return new TapRows(tap, total, nLayer + 1);
                }
            }
            return null;
        }

        boolean wants(int point) {
            return tap.wants(point);
        }

        /** Copies the step's {@code n} rows, which are rows {@code [off, off + n)} of the call. */
        void capture(int point, float[] x, int off, int n, int width) {
            ensure(point, width);
            System.arraycopy(x, 0, rows[point], off * width, n * width);
        }

        /**
         * Copies the step's {@code n} rows (row {@code r} of {@code x} is row {@code callRows[r]} of the call, the
         * ubatch's batch row mapping).
         */
        void capture(int point, float[] x, int[] callRows, int n, int width) {
            ensure(point, width);
            for (int r = 0; r < n; r++) {
                System.arraycopy(x, r * width, rows[point], callRows[r] * width, width);
            }
        }

        /** The call-wide row buffer of {@code point}: every step of the call writes its own rows into it. */
        private void ensure(int point, int width) {
            if (rows[point] == null) {
                rows[point] = new float[Math.multiplyExact(total, width)];
                widths[point] = width;
            } else if (widths[point] != width) {
                throw new IllegalStateException("tap point " + point + " changed width from " + widths[point]
                        + " to " + width + " within a call");
            }
        }

        void deliver() {
            for (int p = 0; p < rows.length; p++) {
                if (rows[p] != null) {
                    tap.onHidden(p, rows[p], total, widths[p]);
                }
            }
        }
    }
}
