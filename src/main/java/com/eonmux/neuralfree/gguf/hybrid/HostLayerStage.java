package com.eonmux.neuralfree.gguf.hybrid;

import com.eonmux.neuralfree.gguf.cpu.block.CpuBlockRegistry;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuLayerExecutor;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.head.CpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.forward.SplitHostStage;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.utils.Closeables;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The host part of a multi-GPU layer split whose first trunk layers stay on the host ({@link LayerOffload} over
 * {@code --split-mode layer}): the CPU embedding stage and one {@link CpuLayerExecutor} per host layer over the heap
 * memory of those layers, the same host side {@link HybridForwardBackend} runs before its one device. The context, its
 * weights and the memory belong to whoever built them; {@link #close()} closes only the executors. Not thread-safe.
 */
public final class HostLayerStage implements SplitHostStage {

    private static final Logger LOGGER = LogManager.getLogger(HostLayerStage.class);

    private final CpuForwardContext ctx;
    private final HeapModelMemory memory;
    private final CpuEmbeddingStage embedding;
    private final List<CpuLayerExecutor> executors;
    private final int outputWidth;
    private boolean closed;

    private HostLayerStage(CpuForwardContext ctx, HeapModelMemory memory, List<CpuLayerExecutor> executors,
                           int outputWidth) {
        this.ctx = ctx;
        this.memory = memory;
        this.embedding = new CpuEmbeddingStage(ctx);
        this.executors = executors;
        this.outputWidth = outputWidth;
    }

    /**
     * Binds the executors of trunk layers {@code [0, layers)} over {@code ctx}, whose memory holds those layers.
     *
     * @param registry the host blocks
     * @param memory   the heap memory of the host layers, the memory {@code ctx} binds against
     * @param layers   the host layers; at least one, and at least one layer is left for the devices
     * @throws IllegalArgumentException      when {@code layers} leaves nothing on either side
     * @throws UnsupportedOperationException when the registry lacks a block a host layer needs
     */
    public static HostLayerStage bind(CpuForwardContext ctx, CpuBlockRegistry registry, HeapModelMemory memory,
                                      int layers) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(registry, "registry");
        Objects.requireNonNull(memory, "memory");
        ModelSpec spec = ctx.spec();
        if (layers < 1 || layers >= spec.nLayer()) {
            throw new IllegalArgumentException("the host part of a split runs between 1 and " + (spec.nLayer() - 1)
                    + " of the " + spec.nLayer() + " trunk layers, got " + layers);
        }
        List<CpuLayerExecutor> list = new ArrayList<>(layers);
        try {
            for (int il = 0; il < layers; il++) {
                list.add(new CpuLayerExecutor(registry, ctx, spec.layer(il)));
            }
        } catch (RuntimeException e) {
            Closeables.closeAll(list, e);
            throw e;
        }
        LOGGER.info("{}: layers [0, {}) of the split run on the host, ubatch {}", spec.arch().ggufName(), layers,
                ctx.scratch().ubatch());
        return new HostLayerStage(ctx, memory, Collections.unmodifiableList(list),
                spec.layer(layers).residual().streamWidth());
    }

    @Override
    public int layers() {
        return executors.size();
    }

    @Override
    public int outputWidth() {
        return outputWidth;
    }

    @Override
    public int ubatch() {
        return ctx.scratch().ubatch();
    }

    @Override
    public ModelMemory memory() {
        return memory;
    }

    @Override
    public void beginStep(int p, int n) {
        ensureOpen();
        memory.beginStep(p, n);
    }

    /** The embedding, then every host layer over the stream in place; the stream is the host scratch {@code X}. */
    @Override
    public float[] forward(int[] tokens, int[] types, int off, int n, int p, Taps taps) {
        ensureOpen();
        Objects.requireNonNull(taps, "taps");
        float[] x = ctx.scratch().slot(CpuScratch.X);
        embedding.embed(tokens, types, off, n, p, x);
        for (int il = 0; il < executors.size(); il++) {
            CpuLayerExecutor e = executors.get(il);
            if (taps.wants(il)) {
                taps.capture(il, x, n, e.streamWidth());
            }
            e.forward(p, n, x);
        }
        return x;
    }

    @Override
    public void checkInputs(int[] tokens, int[] types, int pos0) {
        ensureOpen();
        embedding.checkInputs(tokens, types, pos0);
    }

    @Override
    public String shiftRefusal() {
        ensureOpen();
        for (CpuLayerExecutor e : executors) {
            String refusal = e.shiftRefusal();
            if (refusal != null) {
                return refusal;
            }
        }
        return null;
    }

    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        ensureOpen();
        memory.shiftPositions(nKeep, nDiscard);
        for (CpuLayerExecutor e : executors) {
            e.shiftContext(nKeep, nDiscard);
        }
    }

    /** Closes the host executors. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException first = Closeables.closeAll(executors, null);
        if (first != null) {
            throw first;
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the host part of the split is closed");
        }
    }

    @Override
    public String toString() {
        return "HostLayerStage{layers [0, " + executors.size() + "), ubatch " + ctx.scratch().ubatch() + '}';
    }
}
