package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
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
 * causally on an empty memory and leaves it empty.</p>
 *
 * <p>The context's queue, scratch, weights, memory and placeholders belong to whoever built the context; this
 * backend closes only the blocks and head preprocessors it bound. Not thread-safe.</p>
 */
public final class GpuForwardBackend implements ForwardBackend {

    private final GpuForwardContext ctx;
    private final GpuEmbeddingStage embedding;
    private final GpuLayerExecutor executor;
    private final GpuLogitsHead head;
    private final GpuPrefillDriver driver;
    private HiddenTap tap;
    private boolean closed;

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
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(registry, "registry");
        Objects.requireNonNull(preprocessors, "preprocessors");
        ModelSpec spec = ctx.spec();
        registry.validate(spec);
        ctx.kernels().prewarm(kernels(spec));
        this.embedding = new GpuEmbeddingStage(ctx);
        GpuLayerExecutor exec = GpuLayerExecutor.forTrunk(ctx, registry);
        GpuLogitsHead h;
        try {
            h = new GpuLogitsHead(ctx, preprocessors);
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
        this.tap = ctx.hiddenTap();
    }

    /** Kernels the executor, the head and the built-in extras launch, plus attention when a layer attends. */
    static List<String> kernels(ModelSpec spec) {
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

    private GpuPrefillDriver.Result causal(int[] tokens, int pos0, int[] rows, GpuPrefillDriver.Output output) {
        ensureOpen();
        GpuMemory memory = requireMemory("causal evaluation");
        align(memory, pos0, tokens.length);
        return driver.run(tokens, null, pos0, rows, output, true, tap);
    }

    /**
     * Rolls the memory back to {@code pos0} when the call re-evaluates stored positions. Checks the context length
     * first, so a call that cannot fit leaves the memory untouched.
     */
    private static void align(GpuMemory memory, int pos0, int nTokens) {
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
            memory.rollback(drop);
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
        }
    }

    // ================================================================== memory

    @Override
    public void reset() {
        ensureOpen();
        if (ctx.hasMemory()) {
            ctx.memory().reset();
        }
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
        ctx.memory().rollback(nTokens);
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
        ModelSpec spec = ctx.spec();
        String arch = spec.arch().ggufName();
        String refusal = spec.contextShiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(arch + ": context shift is not supported (" + refusal + ")");
        }
        GpuMemory memory = requireMemory("context shift");
        if (!(memory instanceof DeviceModelMemory)) {
            throw new UnsupportedOperationException(arch + ": context shift needs a device-owned memory, got "
                    + memory.getClass().getSimpleName());
        }
        refusal = executor.shiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(arch + ": context shift is not supported (" + refusal + ")");
        }
        ((DeviceModelMemory) memory).shiftPositions(nKeep, nDiscard);
        try {
            executor.shiftContext(nKeep, nDiscard);
        } catch (RuntimeException e) {
            try {
                memory.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            String msg = arch + ": context shift failed after the cache rows moved; the memory was reset";
            if (e instanceof UnsupportedOperationException) {
                throw new UnsupportedOperationException(msg, e);
            }
            throw new IllegalStateException(msg, e);
        }
    }

    /** The context's memory, or null for a model without memory. */
    @Override
    public ModelMemory memory() {
        return ctx.hasMemory() ? ctx.memory() : null;
    }

    @Override
    public void setHiddenTap(HiddenTap tap) {
        Objects.requireNonNull(tap, "tap");
        ensureOpen();
        this.tap = tap;
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
            ctx.queue().finish();
        } catch (RuntimeException e) {
            first = e;
        }
        first = Closeables.close(executor, first);
        first = Closeables.close(head, first);
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuForwardBackend{" + ctx.spec().arch().ggufName() + ", " + executor + ", "
                + (ctx.hasMemory() ? "memory" : "no memory") + '}';
    }
}
