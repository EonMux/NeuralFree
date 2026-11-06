package com.eonmux.neuralfree.gguf.hybrid;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuBlockRegistry;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuLayerExecutor;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.head.CpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuLayerExecutor;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.head.GpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.ops.moe.MoeBlocks;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.utils.Closeables;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The {@link ForwardBackend} of a hybrid CPU+GPU load ({@link LayerOffload}): the host layers run through
 * {@link CpuLayerExecutor}s over the heap memory, the residual stream crosses to the device once per step, the
 * device layers run through a {@link GpuLayerExecutor} over the device memory, and the logits head computes on the
 * device. Per ubatch step of {@code n} tokens at position {@code p} (split by the shared plan's
 * {@code MemoryPlan.stepTokens} like every backend): the device step inputs and position buffer, the host embedding
 * stage, the host executors, the residual upload, the device executor, then the requested rows' logits, hidden rows
 * or greedy ids as device reads; the sequence records of both sides append the step. One host synchronization per
 * call, after the last step, except that a layer whose experts run on the host (llama.cpp {@code --cpu-moe}) also
 * synchronizes at its FFN.
 *
 * <p>Sequence operations drive both sides through {@link HybridMemory}: a rollback or snapshot succeeds only when
 * each side allows it and then changes both. The context's queues, weights, scratch and memory belong to whoever
 * built the contexts; this backend closes only the blocks it bound. Not thread-safe.</p>
 */
public final class HybridForwardBackend implements ForwardBackend {

    private static final Logger LOGGER = LogManager.getLogger(HybridForwardBackend.class);

    private final ModelSpec spec;
    private final LayerOffload split;
    private final GpuForwardContext gpuCtx;
    private final CpuForwardContext cpuCtx;
    private final GpuLayerExecutor deviceExecutor;
    private final List<CpuLayerExecutor> hostExecutors;
    private final GpuLogitsHead head;
    private final HybridMemory memory;
    private final ComputeQueue queue;
    private final ScratchSlot xSlot;
    private final ScratchSlot posSlot;
    private final CpuEmbeddingStage embedding;
    private final int ubatch;
    private final int boundaryWidth;
    private final int[] posInts;
    private final ByteBuffer posBytes;
    private HiddenTap tap;
    private boolean closed;

    private HybridForwardBackend(ModelSpec spec, LayerOffload split, GpuForwardContext gpuCtx,
                                 CpuForwardContext cpuCtx, GpuLayerExecutor deviceExecutor,
                                 List<CpuLayerExecutor> hostExecutors, GpuLogitsHead head, HybridMemory memory) {
        this.spec = spec;
        this.split = split;
        this.gpuCtx = gpuCtx;
        this.cpuCtx = cpuCtx;
        this.deviceExecutor = deviceExecutor;
        this.hostExecutors = hostExecutors;
        this.head = head;
        this.memory = memory;
        this.embedding = new CpuEmbeddingStage(cpuCtx);
        this.queue = gpuCtx.queue();
        GpuScratch scratch = gpuCtx.scratch();
        this.ubatch = Math.min(scratch.ubatch(), cpuCtx.scratch().ubatch());
        this.xSlot = scratch.slot(ScratchSlots.X);
        this.posSlot = scratch.slot(ScratchSlots.POS).require((long) MropeTextPositions.SECTIONS * ubatch);
        this.boundaryWidth = split.nGpuLayers() > 0
                ? spec.layer(split.firstGpuLayer()).residual().streamWidth()
                : spec.layer(spec.nLayer() - 1).residual().streamWidth();
        this.posInts = new int[MropeTextPositions.SECTIONS * ubatch];
        this.posBytes = ByteBuffer.allocate(posInts.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        this.tap = gpuCtx.hiddenTap();
    }

    /**
     * A LongCat-Flash layer is two physical blocks that exchange the shortcut MoE output through one scratch temporary,
     * so the host/device boundary may not fall between them.
     *
     * @throws UnsupportedOperationException when the boundary separates the blocks of one layer
     */
    private static void checkLongcatPair(ModelSpec spec, LayerOffload split) {
        if (spec.arch() != com.eonmux.neuralfree.gguf.arch.Arch.LONGCAT_FLASH
                && spec.arch() != com.eonmux.neuralfree.gguf.arch.Arch.LONGCAT_FLASH_NGRAM) {
            return;
        }
        int first = split.firstGpuLayer();
        if (first != 0 && first != spec.nLayer() && first % 2 != 0) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + ": the hybrid split moves layer " + first
                    + " to the GPU, separating the two blocks of one LongCat-Flash layer; the second block reads the"
                    + " first block's shortcut MoE output from its own device's scratch, so the pair must stay"
                    + " together");
        }
    }

    /**
     * Binds the host executors of the split's host layers, the device executor of its device layers and the logits
     * head.
     *
     * @param gpuRegistry        the device blocks, with the host-expert FFN decorator applied
     * @param headPreprocessors  candidate head preprocessors; those that apply are bound and owned by the head
     * @param cpuRegistry        the host blocks
     * @throws UnsupportedOperationException when a registry lacks a block a side needs
     */
    public static HybridForwardBackend bind(GpuForwardContext gpuCtx, GpuBlockRegistry gpuRegistry,
                                            List<GpuHeadPreprocessor> headPreprocessors, CpuForwardContext cpuCtx,
                                            CpuBlockRegistry cpuRegistry, LayerOffload split, HybridMemory memory) {
        Objects.requireNonNull(gpuCtx, "gpuCtx");
        Objects.requireNonNull(gpuRegistry, "gpuRegistry");
        Objects.requireNonNull(cpuCtx, "cpuCtx");
        Objects.requireNonNull(cpuRegistry, "cpuRegistry");
        Objects.requireNonNull(split, "split");
        Objects.requireNonNull(memory, "memory");
        ModelSpec spec = gpuCtx.spec();
        checkLongcatPair(spec, split);
        gpuRegistry.validate(spec);
        GpuLayerExecutor deviceExecutor = split.nGpuLayers() > 0
                ? GpuLayerExecutor.forStack(gpuCtx, gpuRegistry, deviceLayers(spec, split)) : null;
        List<CpuLayerExecutor> hostExecutors = Collections.emptyList();
        GpuLogitsHead head = null;
        try {
            hostExecutors = bindHost(cpuRegistry, cpuCtx, spec, split);
            head = new GpuLogitsHead(gpuCtx, headPreprocessors);
        } catch (RuntimeException e) {
            Closeables.close(deviceExecutor, Closeables.closeAll(hostExecutors, e));
            throw e;
        }
        LOGGER.info("{}: hybrid forward pass, layers [0, {}) on the host and [{}, {}) on the device{}, ubatch {}",
                spec.arch().ggufName(), split.firstGpuLayer(), split.firstGpuLayer(), spec.nLayer(),
                split.hostExpertOverrides().isEmpty() ? "" : ", experts of the first "
                        + split.hostExpertOverrides().size() + " layer(s) on the host",
                Math.min(gpuCtx.scratch().ubatch(), cpuCtx.scratch().ubatch()));
        return new HybridForwardBackend(spec, split, gpuCtx, cpuCtx, deviceExecutor, hostExecutors, head, memory);
    }

    private static List<LayerSpec> deviceLayers(ModelSpec spec, LayerOffload split) {
        return spec.layers().subList(split.firstGpuLayer(), spec.nLayer());
    }

    private static List<CpuLayerExecutor> bindHost(CpuBlockRegistry registry, CpuForwardContext ctx, ModelSpec spec,
                                                   LayerOffload split) {
        if (!split.active()) {
            return Collections.emptyList();
        }
        List<CpuLayerExecutor> list = new ArrayList<>(split.firstGpuLayer());
        try {
            for (int il = 0; il < split.firstGpuLayer(); il++) {
                list.add(new CpuLayerExecutor(registry, ctx, spec.layer(il)));
            }
        } catch (RuntimeException e) {
            Closeables.closeAll(list, e);
            throw e;
        }
        return Collections.unmodifiableList(list);
    }

    /**
     * The FFN block of a device layer whose experts run on the host ({@link LayerOffload#hostExperts}): the CPU MoE
     * block of the layer over the host context, driven by a device-to-host copy of the FFN input and a host-to-device
     * copy of its output. The block synchronizes the queue at every forward, as the host reads the input rows.
     *
     * @throws UnsupportedOperationException when no CPU block serves the layer's MoE kind
     */
    public static GpuFfnBlock hostExpertFfn(CpuForwardContext cpuCtx, LayerSpec layer) {
        return new HostExpertFfn(cpuCtx, layer);
    }

    private static CpuFfnBlock cpuMoeBlock(LayerSpec layer, CpuForwardContext ctx) {
        FfnKind kind = layer.ffnKind();
        for (CpuFfnFactory f : MoeBlocks.cpuFfns()) {
            if (f.kind() == kind) {
                CpuFfnBlock block = f.create(layer);
                block.bind(ctx, layer);
                return block;
            }
        }
        throw new UnsupportedOperationException("no CPU MoE block for " + kind + " (layer " + layer.il() + ")");
    }

    /** The host MoE block of one device layer, fed over the bus. */
    private static final class HostExpertFfn implements GpuFfnBlock {

        private final int nEmbd;
        private final CpuFfnBlock host;
        private final float[] in;
        private final float[] out;
        private ComputeQueue queue;

        HostExpertFfn(CpuForwardContext cpuCtx, LayerSpec layer) {
            this.nEmbd = cpuCtx.spec().hparams().nEmbd();
            this.host = cpuMoeBlock(layer, cpuCtx);
            long rows = cpuCtx.scratch().ubatch() * nEmbd;
            this.in = new float[(int) rows];
            this.out = new float[(int) rows];
        }

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            this.queue = ctx.queue();
        }

        @Override
        public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
            if (queue == null) {
                throw new IllegalStateException("the host expert FFN of layer " + out.name() + " was never bound to"
                        + " a device context");
            }
            if ((long) nTokens * nEmbd > this.in.length) {
                throw new IllegalStateException("host expert FFN of " + nTokens + " rows of " + nEmbd
                        + " exceeds its scratch of " + this.in.length + " floats");
            }
            long bytes = (long) nTokens * nEmbd * Float.BYTES;
            GpuEvent read = queue.enqueueReadBuffer(in.buffer(), (long) in.offset() * Float.BYTES, bytes, this.in);
            if (read != null) {
                read.close();
            }
            queue.finish();
            host.forward(nTokens, this.in, this.out);
            GpuEvent write = queue.enqueueWriteBuffer(out.buffer(), (long) out.offset() * Float.BYTES, bytes, this.out);
            if (write != null) {
                write.close();
            }
        }

        @Override
        public boolean capturableDecode() {
            return false;
        }

        @Override
        public void close() {
            host.close();
        }
    }

    // ================================================================== causal calls

    @Override
    public float[] prefill(int[] tokens, int pos0) {
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("prefill without tokens");
        }
        float[][] rows = causal(tokens, pos0, new int[]{tokens.length - 1}, Output.LOGITS).rows();
        return rows[0];
    }

    @Override
    public float[] decode(int token, int pos) {
        return causal(new int[]{token}, pos, new int[]{0}, Output.LOGITS).rows()[0];
    }

    /** Greedy selection on the device; only the id is read back. */
    @Override
    public int decodeGreedyOnDevice(int token, int pos) {
        return causal(new int[]{token}, pos, new int[]{0}, Output.ARGMAX).ids()[0];
    }

    @Override
    public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
        Objects.requireNonNull(tokens, "tokens");
        return causal(tokens, pos0, outputRows, Output.LOGITS).rows();
    }

    /** Greedy selection of every requested row by the head; only the ids are read back. */
    @Override
    public int[] forwardRowsGreedy(int[] tokens, int pos0, int[] outputRows) {
        Objects.requireNonNull(tokens, "tokens");
        return causal(tokens, pos0, outputRows, Output.ARGMAX).ids();
    }

    private Result causal(int[] tokens, int pos0, int[] rows, Output output) {
        ensureOpen();
        align(pos0, tokens.length);
        return run(tokens, null, pos0, rows, output, true);
    }

    /**
     * Rolls the memory back to {@code pos0} when the call re-evaluates stored positions, on both sides. Checks the
     * context length first, so a call that cannot fit leaves the memory untouched.
     */
    private void align(int pos0, int nTokens) {
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
        throw new UnsupportedOperationException(spec.arch().ggufName() + " is planned with memory; non-causal"
                + " evaluation needs a model without memory (masked diffusion, encoders)");
    }

    /**
     * @throws IllegalStateException when the memory holds positions (they would be overwritten)
     * @throws IllegalArgumentException for more tokens than one ubatch
     */
    @Override
    public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
        ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        checkSingleStep(tokens, tokenTypes);
        int past = memory.sequence().nPast();
        if (past != 0) {
            throw new IllegalStateException("hidden rows of a model with memory need an empty memory, it holds "
                    + past + " positions; reset() first");
        }
        try {
            return run(tokens, tokenTypes, 0, null, Output.HIDDEN, true).rows();
        } finally {
            memory.reset();
        }
    }

    // ================================================================== memory

    @Override
    public void reset() {
        ensureOpen();
        memory.reset();
    }

    @Override
    public boolean canRollback(int nTokens) {
        ensureOpen();
        return memory.canRollback(nTokens);
    }

    @Override
    public void rollback(int nTokens) {
        ensureOpen();
        memory.rollback(nTokens);
    }

    /**
     * Moves the stored rows of both sides and lets every KV mixer re-rotate its keys. Models and cache mixers that
     * cannot re-rotate their keys are refused before anything moves; a mixer failure after the rows moved resets the
     * memory before the failure propagates.
     *
     * @throws UnsupportedOperationException for models whose positions cannot shift, and a mixer refusal of either
     *                                       side
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        ensureOpen();
        String arch = spec.arch().ggufName();
        String refusal = spec.contextShiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(arch + ": context shift is not supported (" + refusal + ")");
        }
        refusal = deviceExecutor != null ? deviceExecutor.shiftRefusal() : null;
        if (refusal == null) {
            for (CpuLayerExecutor e : hostExecutors) {
                refusal = e.shiftRefusal();
                if (refusal != null) {
                    break;
                }
            }
        }
        if (refusal != null) {
            throw new UnsupportedOperationException(arch + ": context shift is not supported (" + refusal + ")");
        }
        memory.host().shiftPositions(nKeep, nDiscard);
        try {
            for (CpuLayerExecutor e : hostExecutors) {
                e.shiftContext(nKeep, nDiscard);
            }
            memory.device().shiftPositions(nKeep, nDiscard);
            if (deviceExecutor != null) {
                deviceExecutor.shiftContext(nKeep, nDiscard);
            }
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

    @Override
    public ModelMemory memory() {
        ensureOpen();
        return memory;
    }

    @Override
    public void setHiddenTap(HiddenTap tap) {
        Objects.requireNonNull(tap, "tap");
        ensureOpen();
        this.tap = tap;
    }

    /** Waits for the queue, then closes the bound blocks of both sides. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException first = null;
        try {
            queue.finish();
        } catch (RuntimeException e) {
            first = e;
        }
        first = Closeables.close(deviceExecutor, first);
        first = Closeables.close(head, first);
        first = Closeables.closeAll(hostExecutors, first);
        if (first != null) {
            throw first;
        }
    }

    // ================================================================== steps

    private void checkSingleStep(int[] tokens, int[] tokenTypes) {
        embedding.checkInputs(tokens, tokenTypes, 0);
        if (tokens.length > ubatch) {
            throw new IllegalArgumentException(tokens.length + " tokens exceed the ubatch of " + ubatch + "; a call"
                    + " without memory runs in one step, raise neuralfree.ubatch");
        }
    }

    private Result run(int[] tokens, int[] types, int pos0, int[] outputRows, Output output, boolean useMemory) {
        int total = tokens.length;
        if (total == 0) {
            throw new IllegalArgumentException("forward call without tokens");
        }
        if (types != null && types.length != total) {
            throw new IllegalArgumentException(types.length + " token types for " + total + " tokens");
        }
        int[] rows = outputRows != null ? outputRows.clone() : identity(total);
        if (rows.length == 0) {
            throw new IllegalArgumentException("forward call without output rows");
        }
        for (int r : rows) {
            if (r < 0 || r >= total) {
                throw new IllegalArgumentException("output row " + r + " is outside the " + total + " tokens");
            }
        }
        if (output != Output.HIDDEN && !head.hasLmHead()) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " has no LM head; only hidden rows"
                    + " can be read");
        }
        checkSpan(pos0, total, useMemory);
        embedding.checkInputs(tokens, types, pos0);
        Calls calls = new Calls(rows.length, output);
        TapReads taps = new TapReads(tap, total);
        boolean finished = false;
        try {
            int chunk = useMemory ? Math.min(ubatch, memory.plan().ubatch()) : ubatch;
            for (int c0 = 0; c0 < total; ) {
                int n = useMemory ? memory.plan().stepTokens(total - c0, chunk) : Math.min(chunk, total - c0);
                step(tokens, types, pos0 + c0, c0, n, rows, output, useMemory, calls, taps);
                if (useMemory) {
                    memory.host().sequence().append(tokens, c0, n);
                    memory.device().sequence().append(tokens, c0, n);
                }
                c0 += n;
            }
            finished = true;
            queue.finish();
            taps.deliver();
            return calls.result();
        } catch (RuntimeException e) {
            if (!finished) {
                try {
                    queue.finish();
                } catch (RuntimeException drain) {
                    e.addSuppressed(drain);
                }
            }
            throw e;
        } finally {
            calls.closeEvents();
            taps.closeEvents();
        }
    }

    private void checkSpan(int pos0, int total, boolean useMemory) {
        if (!useMemory) {
            if (total > ubatch) {
                throw new IllegalArgumentException(total + " tokens exceed the ubatch of " + ubatch + "; a call"
                        + " without memory runs in one step, raise neuralfree.ubatch");
            }
            if (pos0 != 0) {
                throw new IllegalArgumentException("a call without memory starts at position 0, got " + pos0);
            }
            return;
        }
        if (pos0 != memory.sequence().nPast()) {
            throw new IllegalStateException("the memory holds " + memory.sequence().nPast()
                    + " positions, the call starts at " + pos0);
        }
    }

    /**
     * One step of {@code n} tokens at position {@code p}: the device step inputs and positions, the host embedding
     * stage and executors, the residual upload, the device executor, the head reads of the step's requested rows.
     */
    private void step(int[] tokens, int[] types, int p, int c0, int n, int[] rows, Output output, boolean useMemory,
                      Calls calls, TapReads taps) {
        ExecutionObserver obs = gpuCtx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStepStart(p, n);
        }
        if (useMemory) {
            memory.device().stepInputs().upload(p, n);
            memory.host().beginStep(p, n);
        }
        writePositions(p, n);
        taps.chunk(c0);
        float[] x = cpuCtx.scratch().slot(CpuScratch.X);
        embedding.embed(tokens, types, c0, n, p, x);
        runHost(p, n, x, taps);
        uploadResidual(x, n);
        runDevice(p, n, useMemory, taps);
        readOutputs(p, c0, n, rows, output, calls);
        if (observed) {
            obs.onStepEnd(p, n);
        }
    }

    /** The host executors over the residual stream, with the tap points of the host boundary. */
    private void runHost(int p, int n, float[] x, TapReads taps) {
        int hostLayers = split.firstGpuLayer();
        for (int il = 0; il < hostExecutors.size(); il++) {
            CpuLayerExecutor e = hostExecutors.get(il);
            int width = e.streamWidth();
            if (tap.wants(il)) {
                taps.captureHost(il, x, n, width);
            }
            e.forward(p, n, x);
        }
        if (split.active()) {
            if (tap.wants(hostLayers)) {
                taps.captureHost(hostLayers, x, n, boundaryWidth);
            }
        } else if (tap.wants(spec.nLayer())) {
            taps.captureHost(spec.nLayer(), x, n, boundaryWidth);
        }
    }

    /** Uploads the first {@code n * boundaryWidth} floats of the host residual stream to the device {@code X} slot. */
    private void uploadResidual(float[] x, int n) {
        long floats = (long) n * boundaryWidth;
        GpuEvent ev = queue.enqueueWriteBuffer(xSlot.buffer(), (long) xSlot.offset() * Float.BYTES,
                floats * Float.BYTES, Arrays.copyOf(x, (int) floats));
        if (ev != null) {
            ev.close();
        }
    }

    /** The device executor over the uploaded residual stream, with the tap points of the device layers. */
    private void runDevice(int p, int n, boolean useMemory, TapReads taps) {
        if (deviceExecutor == null) {
            return;
        }
        GpuLayerExecutor.StepMode mode = useMemory && n == 1
                ? GpuLayerExecutor.StepMode.DECODE : GpuLayerExecutor.StepMode.PREFILL;
        deviceExecutor.run(p, n, mode, taps);
    }

    /** Enqueues the head reads of the requested rows that fall in this step. */
    private void readOutputs(int p, int c0, int n, int[] rows, Output output, Calls calls) {
        int k = 0;
        int[] local = new int[rows.length];
        int[] targets = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            int r = rows[i];
            if (r >= c0 && r < c0 + n) {
                local[k] = r - c0;
                targets[k] = i;
                k++;
            }
        }
        for (int done = 0; done < k; ) {
            int count = Math.min(n, k - done);
            ScratchSlot hidden = head.hidden(n, Arrays.copyOfRange(local, done, done + count), count);
            calls.read(output, hidden, count, Arrays.copyOfRange(targets, done, done + count));
            done += count;
        }
    }

    /** {@code pos[t + k*n]} for the step into the {@code POS} slot (one write). */
    private void writePositions(int p, int n) {
        RopePositions.fillStep(gpuCtx.spec().hparams().mropeTextPositions(), p, n, posInts);
        int count = MropeTextPositions.SECTIONS * n;
        posBytes.clear();
        for (int i = 0; i < count; i++) {
            posBytes.putInt(posInts[i]);
        }
        GpuEvent ev = queue.enqueueWriteBuffer(posSlot.buffer(), (long) posSlot.offset() * Float.BYTES,
                (long) count * Integer.BYTES, posBytes.array());
        if (ev != null) {
            ev.close();
        }
    }

    private static int[] identity(int n) {
        int[] r = new int[n];
        for (int i = 0; i < n; i++) {
            r[i] = i;
        }
        return r;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("hybrid forward backend is closed");
        }
    }

    @Override
    public String toString() {
        return "HybridForwardBackend{" + spec.arch().ggufName() + ", host layers [0, " + split.firstGpuLayer()
                + "), device layers [" + split.firstGpuLayer() + ", " + spec.nLayer() + "), ubatch=" + ubatch + '}';
    }

    /** What a call reads back per requested row. */
    private enum Output {
        LOGITS,
        HIDDEN,
        ARGMAX
    }

    /** Rows read back by a call, in request order. */
    private static final class Result {
        private final float[][] rows;
        private final int[] ids;

        Result(float[][] rows, int[] ids) {
            this.rows = rows;
            this.ids = ids;
        }

        float[][] rows() {
            return rows;
        }

        int[] ids() {
            return ids;
        }
    }

    /** Head reads of one call and their placement in the result. */
    private final class Calls {
        private final float[][] rows;
        private final int[] ids;
        private final List<float[]> floatParts = new ArrayList<>();
        private final List<byte[]> idParts = new ArrayList<>();
        private final List<int[]> targets = new ArrayList<>();
        private final List<GpuEvent> events = new ArrayList<>();
        private final Output output;

        Calls(int count, Output output) {
            this.output = output;
            this.rows = output == Output.ARGMAX ? null : new float[count][];
            this.ids = output == Output.ARGMAX ? new int[count] : null;
        }

        void read(Output out, ScratchSlot hidden, int count, int[] target) {
            ScratchSlot src;
            if (out == Output.HIDDEN) {
                src = hidden;
            } else {
                ScratchSlot logits = head.logits(count, hidden);
                src = out == Output.ARGMAX ? head.argmax(count) : logits;
            }
            int width = out == Output.HIDDEN ? head.hiddenWidth() : out == Output.LOGITS ? head.vocab() : 1;
            long bytes = (long) count * width * Float.BYTES;
            long srcOff = (long) src.offset() * Float.BYTES;
            if (out == Output.ARGMAX) {
                byte[] dst = new byte[(int) bytes];
                events.add(queue.enqueueReadBuffer(src.buffer(), srcOff, bytes, dst));
                idParts.add(dst);
                floatParts.add(null);
            } else {
                float[] dst = new float[(int) bytes / Float.BYTES];
                events.add(queue.enqueueReadBuffer(src.buffer(), srcOff, bytes, dst));
                floatParts.add(dst);
                idParts.add(null);
            }
            targets.add(target);
        }

        Result result() {
            int width = output == Output.HIDDEN ? head.hiddenWidth() : output == Output.LOGITS ? head.vocab() : 1;
            for (int part = 0; part < targets.size(); part++) {
                int[] target = targets.get(part);
                if (output == Output.ARGMAX) {
                    ByteBuffer bb = ByteBuffer.wrap(idParts.get(part)).order(ByteOrder.LITTLE_ENDIAN);
                    for (int j = 0; j < target.length; j++) {
                        ids[target[j]] = bb.getInt(j * Integer.BYTES);
                    }
                    continue;
                }
                float[] data = floatParts.get(part);
                if (target.length == 1) {
                    rows[target[0]] = data;
                    continue;
                }
                for (int j = 0; j < target.length; j++) {
                    rows[target[j]] = Arrays.copyOfRange(data, j * width, (j + 1) * width);
                }
            }
            return new Result(rows, ids);
        }

        void closeEvents() {
            for (GpuEvent e : events) {
                if (e != null) {
                    e.close();
                }
            }
            events.clear();
        }
    }

    /**
     * The tap points of one call over both sides: points below the first device layer capture from the host
     * residual stream, the rest as device reads, delivered once per point after the queue finished.
     */
    private final class TapReads implements GpuLayerExecutor.TapSink {
        private final HiddenTap tap;
        private final int total;
        private final List<Integer> points = new ArrayList<>();
        private final List<Integer> offsets = new ArrayList<>();
        private final List<Integer> counts = new ArrayList<>();
        private final List<Integer> widths = new ArrayList<>();
        private final List<float[]> parts = new ArrayList<>();
        private final List<GpuEvent> events = new ArrayList<>();
        private int chunkStart;

        TapReads(HiddenTap tap, int total) {
            this.tap = tap;
            this.total = total;
        }

        void chunk(int c0) {
            this.chunkStart = c0;
        }

        /**
         * Host capture of one step of {@code n} rows, which are rows {@code [chunkStart, chunkStart + n)} of the
         * call; the rows are copied, as the host layers keep updating the stream in place.
         */
        void captureHost(int point, float[] x, int n, int width) {
            add(point, Arrays.copyOf(x, Math.multiplyExact(n, width)), n, width);
        }

        @Override
        public boolean wants(int localPoint) {
            return tap.wants(split.firstGpuLayer() + localPoint);
        }

        @Override
        public void capture(int localPoint, ScratchSlot x, int nTokens, int width) {
            int point = split.firstGpuLayer() + localPoint;
            float[] dst = new float[Math.multiplyExact(nTokens, width)];
            events.add(queue.enqueueReadBuffer(x.buffer(), (long) x.offset() * Float.BYTES,
                    (long) dst.length * Float.BYTES, dst));
            add(point, dst, nTokens, width);
        }

        private void add(int point, float[] data, int n, int width) {
            parts.add(data);
            points.add(point);
            offsets.add(chunkStart);
            counts.add(n);
            widths.add(width);
        }

        void deliver() {
            List<Integer> seen = new ArrayList<>();
            for (int i = 0; i < points.size(); i++) {
                int point = points.get(i);
                if (seen.contains(point)) {
                    continue;
                }
                seen.add(point);
                int width = widths.get(i);
                float[] all = new float[Math.multiplyExact(total, width)];
                for (int j = 0; j < points.size(); j++) {
                    if (points.get(j) != point) {
                        continue;
                    }
                    if (widths.get(j) != width) {
                        throw new IllegalStateException("tap point " + point + " changed width from " + width
                                + " to " + widths.get(j) + " within a call");
                    }
                    System.arraycopy(parts.get(j), 0, all, offsets.get(j) * width, counts.get(j) * width);
                }
                tap.onHidden(point, all, total, width);
            }
        }

        void closeEvents() {
            for (GpuEvent e : events) {
                if (e != null) {
                    e.close();
                }
            }
            events.clear();
        }
    }
}
