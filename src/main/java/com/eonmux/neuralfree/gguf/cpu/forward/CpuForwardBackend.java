package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuBlockRegistry;
import com.eonmux.neuralfree.gguf.cpu.head.CpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.cpu.head.CpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.cpu.head.CpuLogitsHead;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * {@link ForwardBackend} of one sequence on the CPU, the twin of {@code GpuForwardBackend}: the embedding stage, one
 * {@link CpuLayerExecutor} per trunk slot (looped and hierarchical schedules alias physical weights through
 * {@link LayerSpec#weightLayer()}), and the logits head.
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
        this.embedding = new CpuEmbeddingStage(ctx);
        this.layers = bindTrunk(registry, ctx, spec);
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
        head.logits(1);
        int id = head.argmax(0);
        deliver(taps);
        return id;
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

    /** The context's memory, or null for a model without memory. */
    @Override
    public ModelMemory memory() {
        checkOpen();
        return memory;
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
            if (taps != null && taps.wants(il)) {
                taps.capture(il, x, off, n, e.streamWidth());
            }
            e.forward(pos0, n, x);
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
        float[] hidden = head.hidden(scratch.slot(CpuScratch.X), finalWidth, rowIdx, count);
        int width = head.hiddenWidth();
        float[] src = hidden;
        if (logits) {
            src = head.logits(count);
            width = head.vocabRows();
        }
        for (int k = 0; k < count; k++) {
            out[outputIdx[k]] = Arrays.copyOfRange(src, k * width, (k + 1) * width);
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
        checkContext(mem, pos0, tokens.length);
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
     * the call ({@link HiddenTap}: {@code nRows} is every token of the call).
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
            if (rows[point] == null) {
                rows[point] = new float[Math.multiplyExact(total, width)];
                widths[point] = width;
            } else if (widths[point] != width) {
                throw new IllegalStateException("tap point " + point + " changed width from " + widths[point]
                        + " to " + width + " within a call");
            }
            System.arraycopy(x, 0, rows[point], off * width, n * width);
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
